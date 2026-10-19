"""extract_paraboloid_cst_odb.py -- ODB readout for abaqus_paraboloid_cst.py (run under Abaqus python
from inside a job dir; the queue calls it):

    abaqus python ..\\..\\..\\..\\extract_paraboloid_cst_odb.py --odb cst_prs_vares.odb --out .

Last frame of the last step:
    coord          final node coordinates (label k+1 -> row k)
    S              per-element membrane stress S11, S22, S12 (section-point mean; local-1 = warp)
    SF             per-element section forces SF1, SF2, SF3 (N/mm)
    wrinkle_state  SDV6 section-point mean;  wrinkle_max  its max (0 taut, 1 wrinkled, 2 slack)
    failure_factor SDV4 section-point mean
History over ALL steps (total time):
    hist_t, hist_ke    ALLKE
    hist_rf3           sum of RF3 over the RIM nodes (negative: the rim pulls the dish down)
    hist_allie, hist_allvd, hist_allwk, hist_allae, hist_etotal   energy balance
    rf             nodal reactions RF1-3 at the last frame (zeros if not output)
    frame_t, frame_coord   every field frame of the last step (float32 COORD) -- creep check
Bulk data blocks throughout (the 12.5 mm model has 361k elements).  Writes <job>_final.npz.
"""
from __future__ import print_function

import argparse
import os

import numpy as np

from odbAccess import openOdb


def elem_mean(field, inst, comps, how="mean"):
    comps = list(comps)
    ncomp = len(comps)
    labels, data = [], []
    for b in field.getSubset(region=inst).bulkDataBlocks:
        labels.append(np.asarray(b.elementLabels, int))
        d = np.asarray(b.data, float)
        data.append(d.reshape(len(d), -1)[:, comps])
    lab = np.concatenate(labels)
    dat = np.concatenate(data)
    n = int(lab.max())
    if how == "max":
        out = np.full((n, ncomp), -np.inf)
        np.maximum.at(out, lab - 1, dat)
    else:
        out = np.zeros((n, ncomp))
        np.add.at(out, lab - 1, dat)
        cnt = np.bincount(lab - 1, minlength=n).astype(float)
        out /= np.maximum(cnt, 1.0)[:, None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odb", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    odb = openOdb(a.odb, readOnly=True)
    names = list(odb.steps.keys())
    step = odb.steps[names[-1]]
    frame = step.frames[-1]
    inst = odb.rootAssembly.instances[list(odb.rootAssembly.instances.keys())[0]]
    print("step %s frame t=%.3f" % (step.name, frame.frameValue))

    lab, xyz = [], []
    for b in frame.fieldOutputs["COORD"].getSubset(region=inst).bulkDataBlocks:
        lab.append(np.asarray(b.nodeLabels, int))
        xyz.append(np.asarray(b.data, float))
    lab, xyz = np.concatenate(lab), np.concatenate(xyz)
    coord = np.zeros((int(lab.max()), 3))
    coord[lab - 1] = xyz

    fo = frame.fieldOutputs
    S = elem_mean(fo["S"], inst, (0, 1, 3))          # shell S order: S11, S22, S33(=0), S12
    SF = np.zeros_like(S)                            # membrane forces N11, N22, N12 by LABEL
    SFQ = np.zeros((len(S), 2))
    if "SF" in fo.keys():
        lbl = list(fo["SF"].componentLabels)
        print("SF components:", lbl)
        SF = elem_mean(fo["SF"], inst, [lbl.index(k) for k in ("SF1", "SF2", "SF3")])
        SFQ = elem_mean(fo["SF"], inst, [lbl.index(k) for k in ("SF4", "SF5")])   # transverse shear
    # the element's own local axes (rows = local 1, 2, 3 in global), from the stress output
    axes = np.full((len(S), 3, 3), np.nan)
    for b in fo["S"].getSubset(region=inst).bulkDataBlocks:
        lc = b.localCoordSystem
        if lc is None:
            continue
        lc = np.asarray(lc, float)
        el = np.asarray(b.elementLabels, int) - 1
        if lc.ndim == 2 and lc.shape[1] == 4:        # quaternions (x, y, z, w)
            qx, qy, qz, qw = lc.T
            R = np.stack([
                np.stack([1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)], 1),
                np.stack([2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)], 1),
                np.stack([2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)], 1)], 1)
            axes[el] = R                             # rows = local axes (checked: row 3 = normal)
        else:
            axes[el] = lc.reshape(len(el), 3, 3)
    print("local axes: %d of %d elements, raw shape %s" % (int(np.isfinite(axes[:, 0, 0]).sum()),
                                                         len(S), str(lc.shape)))
    wr = elem_mean(fo["SDV6"], inst, (0,))[:, 0]
    wmax = elem_mean(fo["SDV6"], inst, (0,), "max")[:, 0]
    ff = elem_mean(fo["SDV4"], inst, (0,))[:, 0]
    rf = np.zeros_like(coord)                        # nodal reactions RF1-3, last frame
    if "RF" in fo.keys():
        for b in fo["RF"].getSubset(region=inst).bulkDataBlocks:
            rf[np.asarray(b.nodeLabels, int) - 1] = np.asarray(b.data, float)[:, :3]
    rf_last = rf

    t_all, ke_all, rf_all = [], [], []
    en = dict((k, []) for k in ("ALLIE", "ALLVD", "ALLWK", "ALLAE", "ETOTAL"))
    t0 = 0.0
    for nm in names:
        st = odb.steps[nm]
        asm = [r for k, r in st.historyRegions.items() if k.startswith("Assembly")][0]
        ke = np.asarray(asm.historyOutputs["ALLKE"].data, float)
        for k in en:
            en[k].append(np.asarray(asm.historyOutputs[k].data, float)[:, 1]
                         if k in asm.historyOutputs.keys() else np.full(len(ke), np.nan))
        rf = None
        for k, r in st.historyRegions.items():
            if k.startswith("Node") and "RF3" in r.historyOutputs.keys():
                d = np.asarray(r.historyOutputs["RF3"].data, float)
                rf = d.copy() if rf is None else rf + np.c_[np.zeros(len(d)), d[:, 1]]
        t_all.append(t0 + ke[:, 0])
        ke_all.append(ke[:, 1])
        rf_all.append(np.interp(ke[:, 0], rf[:, 0], rf[:, 1]) if rf is not None
                      else np.full(len(ke), np.nan))
        t0 += st.timePeriod

    job = os.path.splitext(os.path.basename(a.odb))[0]
    out = os.path.join(a.out, job + "_final.npz")
    hist = dict(("hist_" + k.lower(), np.concatenate(v)) for k, v in en.items())
    # every field frame of the last step (the Hold): node coordinates, to see what still moves
    ft, fc = [], []
    for fr in step.frames:
        lab, xyz = [], []
        for b in fr.fieldOutputs["COORD"].getSubset(region=inst).bulkDataBlocks:
            lab.append(np.asarray(b.nodeLabels, int))
            xyz.append(np.asarray(b.data, float))
        lab, xyz = np.concatenate(lab), np.concatenate(xyz)
        c = np.zeros_like(coord, dtype=np.float32)
        c[lab - 1] = xyz
        ft.append(float(fr.frameValue))
        fc.append(c)
    hist["frame_t"] = np.asarray(ft)
    hist["frame_coord"] = np.asarray(fc)
    np.savez(out, coord=coord, S=S, SF=SF, SFQ=SFQ, axes=axes, rf=rf_last, wrinkle_state=wr, wrinkle_max=wmax,
             failure_factor=ff, step=str(step.name), time=float(frame.frameValue),
             hist_t=np.concatenate(t_all), hist_ke=np.concatenate(ke_all),
             hist_rf3=np.concatenate(rf_all), **hist)
    print("wrote %s  (%d nodes, %d elements, %d history points)"
          % (out, len(coord), len(S), sum(len(x) for x in t_all)))
    odb.close()


if __name__ == "__main__":
    main()
