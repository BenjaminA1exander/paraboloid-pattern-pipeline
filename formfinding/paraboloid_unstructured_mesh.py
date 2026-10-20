r"""paraboloid_unstructured_mesh.py -- unstructured (gmsh) seed mesh for
ParaboloidDish_unstructured_JAXFDM.py, the unstructured rung of the paraboloid mesh study.

Same seed geometry as the earlier structured-mesh solver (every node ON the target
paraboloid z = DEPTH(1 - r^2/R^2), rim circle r = R pinned at z = 0, one node exactly at the apex),
but the membrane is a Frontal-Delaunay triangulation of near-uniform 3-D size h instead of the
polar n_theta x n_rings net:

  * no apex fan -- the apex is an ordinary interior node of valence ~6 (embedded point, so the
    W_APEX term and the apex health gates of the solver still have a node to act on);
  * the disc is meshed in the paraboloid's CONFORMAL radial coordinate rho(r), with
    d(rho)/rho = ds/r, so an isotropic triangle in the plane is an isotropic triangle on the
    surface; the planar size field is h / lambda with lambda = r/rho the conformal factor.
    (Meshing in plain r would stretch every triangle meridionally by sqrt(1+z'^2), 1.41 at the rim.)
  * no symmetry is imposed: the mesh breaks the C_n symmetry of the polar net, which is the point.

The node count, not h, is the controlled quantity (MESHSTUDY ladder matches the structured rungs'
node counts); h is found by a few secant steps on N(h) ~ h^-2.

Run in the BASE env (gmsh 4.15):
    python paraboloid_unstructured_mesh.py --n 1153 --out mesh.npz
"""
import argparse
import os

import numpy as np

R_FORM = float(os.environ.get("JAXFDM_R", "3.0"))
DEPTH_FORM = float(os.environ.get("JAXFDM_DEPTH", "1.5"))


def paraboloid_z(r, R=R_FORM, D=DEPTH_FORM):
    return D * (1.0 - (np.asarray(r) / R) ** 2)


def meridian_tables(R=R_FORM, D=DEPTH_FORM, n=20001):
    """r grid with arc length s(r) from the apex and the conformal radius rho(r), rho(R) = R.
    ln(rho/r) = -int_r^R g(t) dt,  g(t) = (sqrt(1 + k^2 t^2) - 1)/t,  k = 2D/R^2 (regular at 0)."""
    k = 2.0 * D / R ** 2
    r = np.linspace(0.0, R, n)
    ds = np.sqrt(1.0 + (k * r) ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(r))])
    g = np.where(r > 0, (ds - 1.0) / np.where(r > 0, r, 1.0), 0.0)
    G_from0 = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(r))])
    G = G_from0[-1] - G_from0                      # int_r^R g
    lam = np.exp(G)                                # conformal factor r/rho (finite at r=0)
    rho = r / lam
    return r, s, rho, lam


def surface_area(R=R_FORM, D=DEPTH_FORM):
    r, s, _, _ = meridian_tables(R, D)
    k = 2.0 * D / R ** 2
    return float(np.trapz(2.0 * np.pi * r * np.sqrt(1.0 + (k * r) ** 2), r))


def gmsh_mesh(h, R=R_FORM, D=DEPTH_FORM, algo=6, smooth=True):
    """Mesh the conformal disc at 3-D size h. Returns (xyz, tris, rim, apex)."""
    import gmsh
    r_t, _, rho_t, lam_t = meridian_tables(R, D)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("paraboloid_conformal_disc")
        disc = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R, R)
        centre = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.embed(0, [centre], 2, disc)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", algo)          # 6 = frontal-Delaunay

        def size_cb(dim, tag, x, y, z, lc):
            rho = np.hypot(x, y)
            return float(h / np.interp(rho, rho_t, lam_t))

        gmsh.model.mesh.setSizeCallback(size_cb)
        gmsh.model.mesh.generate(2)
        if smooth:
            gmsh.model.mesh.optimize("Laplace2D")
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        P = np.asarray(ncoords).reshape(-1, 3)[:, :2]
        tag2i = {int(t): i for i, t in enumerate(ntags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tris = None
        for et, en in zip(etypes, enodes):
            if et == 2:
                tris = np.asarray([tag2i[int(t)] for t in en], int).reshape(-1, 3)
        bnd = set()
        for _, ctag in gmsh.model.getEntities(1):
            bt, _, _ = gmsh.model.mesh.getNodes(1, ctag, includeBoundary=True)
            bnd.update(tag2i[int(t)] for t in bt)
    finally:
        gmsh.finalize()

    # drop unreferenced nodes (gmsh can keep construction points as orphans)
    used = np.unique(tris)
    remap = -np.ones(P.shape[0], int)
    remap[used] = np.arange(used.size)
    P, tris = P[used], remap[tris]
    rim = np.asarray(sorted(remap[i] for i in bnd if remap[i] >= 0), int)

    # conformal plane -> surface: rho -> r (monotone), z on the paraboloid
    rho = np.hypot(P[:, 0], P[:, 1])
    r = np.interp(rho, rho_t, r_t)
    scale = np.where(rho > 0, r / np.where(rho > 0, rho, 1.0), 1.0)
    xyz = np.column_stack([P[:, 0] * scale, P[:, 1] * scale, paraboloid_z(r, R, D)])
    xyz[rim, :2] *= R / np.hypot(xyz[rim, 0], xyz[rim, 1])[:, None]   # exact rim circle
    xyz[rim, 2] = 0.0
    apex = int(np.argmin(rho))
    assert rho[apex] < 1e-12, "apex node missing (embedded centre point not meshed)"
    xyz[apex] = [0.0, 0.0, D]
    return xyz, tris, rim, apex


def quality(xyz, tris):
    """(min angle deg, mean shape, max shape, h_eff, n_inverted-in-plane) -- shape as the solvers
    use it: (la^2+lb^2+lc^2)/(4 sqrt3 A), 1 = equilateral."""
    a, b, c = xyz[tris[:, 0]], xyz[tris[:, 1]], xyz[tris[:, 2]]
    la = np.linalg.norm(b - c, axis=1)
    lb = np.linalg.norm(c - a, axis=1)
    lc = np.linalg.norm(a - b, axis=1)
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    shape = (la ** 2 + lb ** 2 + lc ** 2) / (4.0 * np.sqrt(3.0) * A)
    cosA = np.clip((lb ** 2 + lc ** 2 - la ** 2) / (2 * lb * lc), -1, 1)
    cosB = np.clip((la ** 2 + lc ** 2 - lb ** 2) / (2 * la * lc), -1, 1)
    angs = np.degrees(np.arccos(np.stack([cosA, cosB], 1)))
    angs = np.column_stack([angs, 180.0 - angs.sum(1)])
    axy = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    n_inv = int(min((axy > 0).sum(), (axy < 0).sum()))
    edges = np.unique(np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), 1), axis=0)
    h_eff = float(np.sqrt(4.0 * A.sum() / (np.sqrt(3.0) * edges.shape[0])))
    return float(angs.min()), float(shape.mean()), float(shape.max()), h_eff, n_inv, edges.shape[0]


def mesh_for_nodes(n_target, tol=0.03, max_it=12, **kw):
    """Secant on log N vs log h (N ~ h^-2) until |N/n_target - 1| <= tol."""
    A = surface_area()
    h = np.sqrt(2.0 * A / (np.sqrt(3.0) * n_target))
    best = None
    for _ in range(max_it):
        xyz, tris, rim, apex = gmsh_mesh(h, **kw)
        n = xyz.shape[0]
        if best is None or abs(n / n_target - 1) < abs(best[0].shape[0] / n_target - 1):
            best = (xyz, tris, rim, apex, h)
        if abs(n / n_target - 1) <= tol:
            break
        h *= np.sqrt(n / n_target)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="target node count")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.03)
    args = ap.parse_args()
    xyz, tris, rim, apex, h = mesh_for_nodes(args.n, tol=args.tol)
    amin, smean, smax, h_eff, n_inv, n_e = quality(xyz, tris)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, vertices=xyz, tris=tris, rim=rim, apex=apex, h_nominal=h, h_eff=h_eff,
             n_target=args.n, R=R_FORM, DEPTH=DEPTH_FORM)
    print(f"[mesh] target {args.n} -> {xyz.shape[0]} nodes, {n_e} edges, {tris.shape[0]} tris, "
          f"{rim.size} rim; h={h:.4f} h_eff={h_eff:.4f} (form units); min angle {amin:.1f} deg, "
          f"shape mean {smean:.3f} max {smax:.3f}; inverted {n_inv}; apex node {apex}")
    print(f"[mesh] saved {args.out}")
