"""``extract_facets`` over a **real** ``.mpco`` written by OpenSees.

The unit coverage in ``test_plot_facets_solver_named.py`` hand-builds the
solver-flavoured ``ElementTypeInfo``. This file closes the loop with no
hand-built anything: mesh in apeGmsh, run the mesh through openseespy
with the MPCO recorder attached, read the file back through
:func:`read_fem_from_mpco`, and require the extracted hull to be
**identical** to the one taken from the native FEMData.

That equality is the assertion that matters. It does not depend on
knowing what the MPCO writer decided to call the element class, nor on
which mid-side-node convention the element uses — if the extractor's
face table disagreed with the recorded connectivity, interior faces
would not cancel and the hull would differ (or stop being closed).

What the real files revealed, and why the alias fallback made this bug
look narrower than it is: ``_auto_alias`` matches Gmsh *shape words*, so
``FourNodeTetrahedron`` → ``'tet4'`` and ``TenNodeTetrahedron`` →
``'tet10'``, but ``stdBrick`` is recorded as class ``Brick`` → ``'brick'``
and ``LadrunoBrick20`` → ``'ladrunobrick20'``. Under the old name
allowlist the tets came through by pure luck of nomenclature while every
brick — the reported symptom — was dropped.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import h5py

from apeGmsh.mesh._femdata_mpco_io import read_fem_from_mpco
from apeGmsh.results.plot._facets import extract_facets

ops = pytest.importorskip("openseespy.opensees", reason="openseespy required")

pytestmark = [pytest.mark.live, pytest.mark.slow]


# ---------------------------------------------------------------------
# Capability probes — this suite needs the MPCO recorder compiled in
# ---------------------------------------------------------------------

def _has_mpco_recorder(tmp_path: Path) -> bool:
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    try:
        ops.recorder("mpco", str(tmp_path / "_probe"), "-N", "displacement")
        return True
    except Exception:
        return False
    finally:
        ops.wipe()


def _has_element(cls: str, npe: int) -> bool:
    """Is ``cls`` linked into this build? (geometry-independent probe)."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, 1.0e3, 0.25, 0.0)
    for i in range(1, npe + 1):
        ops.node(i, float(i), 0.0, 0.0)
    try:
        ops.element(cls, 1, *range(1, npe + 1), 1)
        return True
    except Exception:
        # A live element that merely rejects this degenerate geometry
        # still registered; only an unknown class raises here.
        return False
    finally:
        ops.wipe()


# ---------------------------------------------------------------------
# Mesh + run helpers
# ---------------------------------------------------------------------

# Gmsh hex20 mid-edge order -> the serendipity (C3D20 / 20NodeBrick)
# order the fork's LadrunoBrick20 expects. Corners (0-7) are shared by
# both conventions; only the 12 mid-edge slots are permuted. Fed to
# OpenSees so the analysis runs at all — the extractor itself is
# invariant to it (see test_mid_edge_order_does_not_move_the_hull).
_GMSH_TO_SERENDIPITY = [
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,
]


def _mesh(*, order: int, hexes: bool):
    """Mesh a 1x1x2 box; return (node ids, coords, elem ids, conn, type)."""
    import apeGmsh

    g = apeGmsh.apeGmsh(model_name="facets_mpco", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 2, label="b")
        g.physical.add_volume("b", name="Body")
        if hexes:
            g.mesh.structured.set_transfinite_box("b", n=3)
        else:
            g.mesh.sizing.set_global_size(1.4)
        g.mesh.generation.generate(dim=3)
        if order > 1:
            g.mesh.generation.set_order(order, bubble=False)
        fem = g.mesh.queries.get_fem_data(dim=3)
        grp = list(fem.elements)[0]
        return (
            np.asarray(fem.nodes.ids),
            np.asarray(fem.nodes.coords),
            np.asarray(grp.ids),
            np.asarray(grp.connectivity),
            grp.element_type,
        )
    finally:
        g.end()


def _native_fem(ids, xyz, eids, conn, et):
    from apeGmsh.mesh._element_types import ElementGroup
    from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import (
        ElementComposite, FEMData, MeshInfo, NodeComposite,
    )
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=ids, node_coords=xyz,
                            physical=pg, labels=LabelSet({})),
        elements=ElementComposite(
            groups={et.code: ElementGroup(et, eids, conn)},
            physical=pg, labels=LabelSet({})),
        info=MeshInfo(n_nodes=len(ids), n_elems=len(eids),
                      bandwidth=0, types=[et]),
    )


def _run_mpco(ids, xyz, eids, conn, cls: str, path: Path) -> int:
    """One linear static step with the MPCO recorder attached."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    for nid, p in zip(ids, xyz):
        ops.node(int(nid), float(p[0]), float(p[1]), float(p[2]))
    zmax = float(xyz[:, 2].max())
    top = []
    for nid, p in zip(ids, xyz):
        if abs(p[2]) < 1e-9:
            ops.fix(int(nid), 1, 1, 1)
        elif abs(p[2] - zmax) < 1e-9:
            top.append(int(nid))
    ops.nDMaterial("ElasticIsotropic", 1, 2.0e8, 0.25, 0.0)
    for eid, row in zip(eids, conn):
        ops.element(cls, int(eid), *[int(x) for x in row], 1)
    ops.recorder("mpco", str(path), "-N", "displacement")
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid in top:
        ops.load(nid, 0.0, 0.0, -1.0e3)
    ops.system("UmfPack")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    rc = ops.analyze(1)
    ops.wipe()          # MPCO flushes on close
    return rc


def _read_mpco(path: Path):
    with h5py.File(path, "r") as f:
        stages = sorted(k for k in f if k.startswith("MODEL_STAGE"))
        return read_fem_from_mpco(f[stages[-1]]["MODEL"])


def _hull(tris) -> set:
    """Facets as unordered node sets — winding-independent identity."""
    return {frozenset(int(n) for n in t) for t in tris}


def _is_closed(tris) -> bool:
    """Closed manifold: every undirected edge shared by exactly 2 tris."""
    edges: Counter = Counter()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[(min(int(a), int(b)), max(int(a), int(b)))] += 1
    return bool(edges) and all(n == 2 for n in edges.values())


# ---------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------

# (label, mesh order, structured hexes, OpenSees class, npe, permute?)
_CASES = [
    ("tet4",  1, False, "FourNodeTetrahedron", 4,  False),
    ("tet10", 2, False, "TenNodeTetrahedron",  10, False),
    ("hex8",  1, True,  "stdBrick",            8,  False),
    ("hex20", 2, True,  "LadrunoBrick20",      20, True),
]


@pytest.mark.parametrize(
    "label, order, hexes, cls, npe, permute",
    _CASES, ids=[c[0] for c in _CASES],
)
def test_mpco_round_trip_hull_matches_native(
    label, order, hexes, cls, npe, permute, tmp_path: Path,
) -> None:
    if not _has_mpco_recorder(tmp_path):
        pytest.skip("this openseespy build has no MPCO recorder")
    if not _has_element(cls, npe):
        pytest.skip(f"{cls} is not in this OpenSees build")

    ids, xyz, eids, conn, et = _mesh(order=order, hexes=hexes)
    assert et.npe == npe, f"expected {npe}-node mesh, got {et.npe}"

    tris_native, _ = extract_facets(_native_fem(ids, xyz, eids, conn, et))
    assert tris_native.size, "native mesh produced no facets"
    assert _is_closed(tris_native), "native hull is not closed"

    emit_conn = conn[:, _GMSH_TO_SERENDIPITY] if permute else conn
    path = tmp_path / f"{label}.mpco"
    rc = _run_mpco(ids, xyz, eids, emit_conn, cls, path)
    assert rc == 0, f"{cls} analysis failed (rc={rc})"
    assert path.exists(), "MPCO recorder wrote no file"

    fem = _read_mpco(path)
    grp = list(fem.elements)[0]
    # The reader synthesizes its own name and dim from the class name —
    # whatever it lands on, the facets must not depend on it.
    assert grp.element_type.dim == 3
    assert grp.element_type.npe == npe
    assert len(grp) == len(eids)

    tris_mpco, _ = extract_facets(fem)
    assert _is_closed(tris_mpco), f"{label}: mpco hull is not closed"
    assert _hull(tris_mpco) == _hull(tris_native), (
        f"{label}: hull read back from {cls} differs from the native "
        f"{et.name} hull ({len(tris_mpco)} vs {len(tris_native)} tris)"
    )


# ---------------------------------------------------------------------
# Why the corner subset is safe across codes
# ---------------------------------------------------------------------

@pytest.mark.parametrize("order, hexes, n_corner", [
    (2, False, 4),      # tet10
    (2, True,  8),      # hex20
], ids=["tet10", "hex20"])
def test_mid_edge_order_does_not_move_the_hull(
    order, hexes, n_corner,
) -> None:
    """Gmsh, Abaqus and OpenSees agree that corners come first and
    disagree about the mid-edge sequence. Since the face tables only
    ever index corners, permuting the mid-side columns must be a no-op —
    which is what makes the corner-subset rendering portable."""
    ids, xyz, eids, conn, et = _mesh(order=order, hexes=hexes)
    base = _hull(extract_facets(_native_fem(ids, xyz, eids, conn, et))[0])
    assert base

    rng = np.random.default_rng(20240724)
    n_mid = conn.shape[1] - n_corner
    for _ in range(5):
        perm = np.concatenate([
            np.arange(n_corner), n_corner + rng.permutation(n_mid),
        ])
        shuffled = _native_fem(ids, xyz, eids, conn[:, perm], et)
        assert _hull(extract_facets(shuffled)[0]) == base
