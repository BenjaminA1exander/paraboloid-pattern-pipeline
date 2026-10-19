r"""
jaxfdm_scatter_patch.py  --  matrix-free (gather/scatter) connectivity for jax_fdm 0.10.0.

Importing this module monkeypatches jax_fdm so its DENSE (E x N), (E x F) and (F x N)
connectivity matvecs on the follower / face-load path become matrix-free GATHER/SCATTER
ops that read `structure.edges_indexed` / `structure.faces_indexed` (+ a lazily cached
edge->faces index array). This removes the ~1.9 GB dense-connectivity allocation that
blows up the follower `value_and_grad` COMPILE at ~5k+ nodes (the AirfoilWing OOM), while
producing BIT-IDENTICAL equilibria and gradients -- validated end-to-end in
`GatherScatter_rung3_JAXFDM.py` (Delta ~ 1e-16 on xyz, loss and gradient).

WHY: jax_fdm stores the connectivity dense because `BCOO.from_scipy_sparse` crashes under
autodiff with "float() argument ... not 'Zero'" (see the comments in
jax_fdm/.../graphs.py and meshes.py). Gather/scatter sidesteps that bug entirely; its
backward is a plain gather, so there is no symbolic-'Zero' path.

REVERSIBLE: edits NOTHING on disk in the installed jax_fdm. To disable, remove the
`import jaxfdm_scatter_patch` line from the caller, OR set env JAXFDM_NO_SCATTER_PATCH=1
before import, OR call jaxfdm_scatter_patch.uninstall(). Restarting Python fully restores
stock behaviour.

Patched sites (all consume the dense `structure.connectivity`; all on the differentiated path):
  loads.edges_tributary_faces_load           (E x N endpoints, E x F edge->faces, F x N centroids)
  loads.nodes_tributary_edges_load           (0.5 * |C|.T @ edges_load)
  models.EquilibriumModel.equilibrium_state  (C @ xyz ; C.T @ (q*vectors))

Self-test:  python jaxfdm_scatter_patch.py    (builds a small follower solve, checks the loss)
"""
import os
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax.experimental.sparse import BCOO

import jax_fdm.equilibrium.loads as _loads
import jax_fdm.equilibrium.models as _models
from jax_fdm.equilibrium.states import EquilibriumState
from jax_fdm.equilibrium.loads import edge_tributary_face_area

# Stock functions kept so uninstall() can restore them exactly.
_STOCK = {
    "edges_tributary_faces_load": _loads.edges_tributary_faces_load,
    "nodes_tributary_edges_load": _loads.nodes_tributary_edges_load,
    "equilibrium_state": _models.EquilibriumModel.equilibrium_state,
}

_EF_CACHE = {}


def _edges_faces_indexed(structure):
    """(E,2) int: up to 2 face indices per edge, -1 padded. Cached per structure (by id).
    Derived once on the host from the dense connectivity_edges_faces, reproducing exactly
    jax_fdm's `flatnonzero(c_edge_faces, size=2, fill_value=-1)`. Edges with no adjacent
    face (e.g. drop stitches) -> (-1, -1) -> zero face load, as in stock."""
    key = id(structure)
    E = int(np.asarray(structure.edges_indexed).shape[0])
    cached = _EF_CACHE.get(key)
    if cached is not None and cached.shape[0] == E:
        return cached
    cef = structure.connectivity_edges_faces
    ef = np.full((E, 2), -1, dtype=np.int64)
    if isinstance(cef, BCOO):                                     # compact build: (row, col) sorted, unique
        idx = np.asarray(cef.indices)
        rows, cols = idx[:, 0], idx[:, 1]
    else:
        rows, cols = np.nonzero(np.asarray(cef))                  # (E, F) 0/1 dense
    slot = [0] * E
    for r, c in zip(rows.tolist(), cols.tolist()):
        if slot[r] < 2:
            ef[r, slot[r]] = c
            slot[r] += 1
    # Cache a plain NUMPY array (host). Never cache a jnp array built inside a trace --
    # that leaks a tracer into module globals. The caller wraps it with jnp.asarray.
    _EF_CACHE[key] = ef
    return ef


def _edges_tributary_faces_load(xyz, faces_load, structure):
    """Matrix-free replacement for loads.edges_tributary_faces_load (no dense E x N / E x F / F x N)."""
    ei = structure.edges_indexed                                  # (E,2) endpoints
    ef = jnp.asarray(_edges_faces_indexed(structure))             # (E,2) faces per edge (host->device)
    fv = structure.faces_indexed                                  # (F,maxval), -1 pad
    # face centroids by gather-mean, replacing  c_faces_vertices @ xyz
    mask = fv >= 0
    pts = xyz[jnp.where(mask, fv, 0)]
    pts = jnp.where(mask[..., None], pts, 0.0)
    centroids = jnp.sum(pts, axis=1) / jnp.sum(mask, axis=1, keepdims=True)

    def per_edge(edge_nodes, edge_faces):
        line = xyz[edge_nodes, :]                                 # gather 2 endpoints
        floads = faces_load[edge_faces, :]                        # edge_faces has -1 fill
        cents = centroids[edge_faces, :]
        areas = vmap(edge_tributary_face_area, in_axes=(None, 0))(line, cents)
        areas = jnp.where(edge_faces >= 0, areas.ravel(), 0.0)
        return areas @ floads

    return vmap(per_edge, in_axes=(0, 0))(ei, ef)


def _nodes_tributary_edges_load(edges_load, structure):
    """Matrix-free replacement for loads.nodes_tributary_edges_load : 0.5 * |C|.T @ edges_load."""
    ei = structure.edges_indexed
    u, v = ei[:, 0], ei[:, 1]
    z = jnp.zeros((int(structure.num_nodes), edges_load.shape[1]))
    return 0.5 * z.at[u].add(edges_load).at[v].add(edges_load)    # UNSIGNED (note the abs in stock)


def _equilibrium_state(self, q, xyz, loads_nodes, structure):
    """Matrix-free replacement for EquilibriumModel.equilibrium_state : C @ xyz and C.T @ (q*vectors)."""
    ei = structure.edges_indexed
    a, b = ei[:, 0], ei[:, 1]
    vectors = xyz[a] - xyz[b]
    lengths = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    f = q[:, None] * vectors
    scat = jnp.zeros_like(xyz).at[a].add(f).at[b].add(-f)
    residuals = loads_nodes - scat
    forces = jnp.reshape(q, (-1, 1)) * lengths
    return EquilibriumState(xyz=xyz, residuals=residuals, lengths=lengths,
                            forces=forces, loads=loads_nodes, vectors=vectors)


_INSTALLED = False


def install():
    """Activate the gather/scatter patch (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _loads.edges_tributary_faces_load = _edges_tributary_faces_load
    _loads.nodes_tributary_edges_load = _nodes_tributary_edges_load
    _models.EquilibriumModel.equilibrium_state = _equilibrium_state
    _INSTALLED = True


def uninstall():
    """Restore stock jax_fdm behaviour."""
    global _INSTALLED
    _loads.edges_tributary_faces_load = _STOCK["edges_tributary_faces_load"]
    _loads.nodes_tributary_edges_load = _STOCK["nodes_tributary_edges_load"]
    _models.EquilibriumModel.equilibrium_state = _STOCK["equilibrium_state"]
    _INSTALLED = False


if os.environ.get("JAXFDM_NO_SCATTER_PATCH") != "1":
    install()


# ---------------------------------------------------------------------------
# JAXFDM_SCATTER_COMPACT=1 (off by default): build the structure WITHOUT jax_fdm's dense tables.
# Even with the matvecs patched above, MeshSparse/GraphSparse still ALLOCATE four dense float64
# tables at construction: E x N connectivity, N x N adjacency, F x N face matrix, E x F edge-face
# (~32 GB at 18440 nodes; the u18440 paraboloid OOM). With the flag they are built as BCOO sparse
# matrices holding the same entries, and jax_fdm's O(E*F) Python edge->face search is replaced by a
# hash map. Every reader of these tables on our path is patched above (and _edges_faces_indexed
# reads the BCOO indices); any unpatched reader gets a sparse matrix and fails loudly, never silently.
# ---------------------------------------------------------------------------
COMPACT = os.environ.get("JAXFDM_SCATTER_COMPACT", "0") == "1"
if COMPACT and _INSTALLED:
    import scipy.sparse as _sps
    from compas.numerical import connectivity_matrix as _conn_matrix
    from jax_fdm.equilibrium.structures.graphs import adjacency_matrix as _adj_matrix
    from compas.numerical import face_matrix as _face_matrix
    from jax_fdm.equilibrium.structures.meshes import Mesh as _JMesh, MeshSparse as _JMeshSparse
    from jax_fdm.equilibrium.structures.graphs import GraphSparse as _JGraphSparse

    def _edges_faces_hash(self):
        """Same (edge -> faces) incidence as Mesh._edges_faces, by a half-edge hash map (O(E+F))."""
        lookup = {}
        for findex, face in enumerate(np.asarray(self.faces)):
            loop = [int(v) for v in face if v >= 0]
            for u, v in zip(loop, loop[1:] + loop[:1]):
                lookup.setdefault(frozenset((u, v)), []).append(findex)
        return tuple(tuple(lookup.get(frozenset((int(u), int(v))), ())) for u, v in np.asarray(self.edges))

    def _connectivity_sparse(self):
        return BCOO.from_scipy_sparse(_conn_matrix(np.asarray(self.edges_indexed), "csr").astype(float))

    def _adjacency_sparse(self):
        return BCOO.from_scipy_sparse(_adj_matrix(np.asarray(self.edges_indexed), "csr").astype(float))

    def _faces_vertices_sparse(self):
        return BCOO.from_scipy_sparse(_face_matrix(np.asarray(self.faces_indexed), "csr", normalize=True))

    def _edges_faces_sparse(self):
        ef = self._edges_faces()
        rows = [e for e, fs in enumerate(ef) for f in sorted(set(fs))]
        cols = [f for e, fs in enumerate(ef) for f in sorted(set(fs))]
        C = _sps.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ef), int(self.faces.shape[0])))
        C.sort_indices()
        return BCOO.from_scipy_sparse(C)

    def _connectivity_scipy_np(self):
        # stock passes the JAX edges array to compas, which then indexes it element by element (minutes)
        return _conn_matrix(np.asarray(self.edges_indexed), "csc")

    def _sparse_diag_indices_np(csc):
        # stock: one jnp.where per row; same positions (row order, data order) in one numpy pass
        idx, ptr = np.asarray(csc.indices), np.asarray(csc.indptr)
        rows = np.repeat(np.arange(csc.shape[0]), np.diff(ptr))
        pos = np.flatnonzero(idx == rows)
        return jnp.asarray(pos, dtype=jnp.result_type(jnp.int64, jnp.asarray(ptr).dtype))

    from jax_fdm.equilibrium.structures.structures import EquilibriumStructureSparse as _JStructSparse
    _JGraphSparse.connectivity_scipy = property(_connectivity_scipy_np)
    _JStructSparse._get_sparse_diag_indices = staticmethod(_sparse_diag_indices_np)
    _JMesh._edges_faces = _edges_faces_hash
    _JGraphSparse._connectivity_matrix = _connectivity_sparse
    _JGraphSparse._adjacency_matrix = _adjacency_sparse
    _JMeshSparse._connectivity_faces_matrix = _faces_vertices_sparse
    _JMeshSparse._connectivity_edges_faces_matrix = _edges_faces_sparse
    print("[scatter-patch] JAXFDM_SCATTER_COMPACT: structure tables built sparse (no dense E x N, N x N, "
          "F x N, E x F)", flush=True)


# ---------------------------------------------------------------------------
# Self-test: small follower solve; the loss must match the stock reference.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)
    from jax_fdm.datastructures import FDMesh
    from jax_fdm.equilibrium import EquilibriumModelSparse, EquilibriumMeshStructureSparse
    from jax_fdm.equilibrium.states import EquilibriumParametersState

    nx = ny = 6
    def vid(i, j):
        return i * ny + j
    verts = [[float(i), float(j), 0.0] for i in range(nx) for j in range(ny)]
    faces = [[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
             for i in range(nx - 1) for j in range(ny - 1)]
    mesh = FDMesh.from_vertices_and_faces(verts, faces)
    for kk in mesh.vertices():
        i, j = divmod(kk, ny)
        if i in (0, nx - 1) or j in (0, ny - 1):
            mesh.vertex_support(kk)
    for e in mesh.edges():
        mesh.edge_forcedensity(e, 2.0)
    mesh.faces_loads([0.0, 0.0, 0.5])
    st = EquilibriumMeshStructureSparse.from_mesh(mesh)
    pr = EquilibriumParametersState.from_datastructure(mesh)
    md = EquilibriumModelSparse(tmax=40, eta=1e-8, is_load_local=False, implicit_diff=True)
    q0 = jnp.asarray(np.asarray(pr.q, dtype=float))

    def loss(q):
        xyz = md(pr._replace(q=q), st).xyz
        return jnp.sum(xyz[:, 2] ** 2)

    val = float(jax.jit(loss)(q0))
    REF = 1.6065218834      # stock value from GatherScatter_rung3_JAXFDM.py
    print(f"[self-test] installed={_INSTALLED}  patched loss = {val:.10f}  "
          f"(stock ref {REF:.10f})  {'OK' if abs(val - REF) < 1e-6 else '*** MISMATCH ***'}")
