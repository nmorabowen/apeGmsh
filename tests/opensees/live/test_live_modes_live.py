"""#2342 gap (a) live oracles: ``capture_live_modes`` on a plain openseespy
deck (no bridge) plus its FEMData, on the module the bridge resolves.

CI's ``live-stock`` lane runs this on pip openseespy; a run beside the
fork checks the same file there. Each deck is tagged by FEM id:

* a 2-D, ndf 3 cantilever of two ``elasticBeamColumn`` elements;
* a 3-D, ndf 6 L-frame (a column and a beam, lumped rotary inertia);
* a ``stdBrick`` cube (ndf 3) whose top corner drives a 3-D column
  (ndf 6) through ``equalDOF`` on the translations.

Oracles, read back from the ``.ladruno`` through ``Results.from_ladruno``:

* every written eigenvalue is the one ``ops.eigen`` returned in the
  capture, bit for bit (a spy records it), and ``frequency_hz`` /
  ``period_s`` follow from it at rtol 1e-12; the eigenvalues also match
  an independent solve, numpy's generalised eigenproblem of the assembled
  2-D cantilever and ``-fullGenLapack`` on the live domain otherwise;
* every node's every written slot equals the live ``nodeEigenvector`` of
  that node and mode at rtol 1e-12 (raw, unscaled); a rotation is written
  where a beam touches the node, and a brick node has none.

Rotations at nodes other classes touch (#2407 / #2414): a beam node tied
into a brick by ``ASDEmbeddedNodeElement -rot``, a cantilever with a
``mixedBeamColumn``, and one cantilever per rotation-only beam class of the
census (``_ROTATION_TOKEN_FOR_LIVE_CLASS``) each read the live rotations,
with no warning; a class the tables do not know at a beam node refuses by
node, slot and class (ruling R6 = (A)).

Refusals on the live domain: a deck node off its FEM coordinates, a FEM
node the deck never built, a non-positive eigenvalue (a negative
spring), and a second default solve that stock openseespy refuses, each
naming its cause and writing nothing.
"""
from __future__ import annotations

import ast
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite
from apeGmsh.results import Results
from apeGmsh.results.live import capture_live_modes

pytestmark = pytest.mark.live

_RTOL = dict(rtol=1e-12, atol=0.0)
_AXES = ("x", "y", "z")
E, A, I_Z = 2.0e11, 1.0e-2, 8.0e-6
G, J, I_Y = 8.0e10, 1.2e-5, 6.0e-6
M, JM = 50.0, 0.4


@pytest.fixture
def opspy() -> Any:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    yield ops
    ops.wipe()


class Spy:
    """The live module, recording what ``eigen`` returned."""

    def __init__(self, ops: Any) -> None:
        self._ops = ops
        self.eigen_returned: "list[float] | None" = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ops, name)

    def eigen(self, *args: Any) -> Any:
        out = self._ops.eigen(*args)
        self.eigen_returned = [float(v) for v in out]
        return out


def _fem(coords: "np.ndarray", groups: "dict[int, tuple[str, int, list[list[int]], list[int]]]",
         name: str) -> FEMData:
    xyz = np.zeros((coords.shape[0], 3))
    xyz[:, :coords.shape[1]] = coords
    infos, built = [], {}
    for code, (gname, dim, conn, ids) in groups.items():
        info = make_type_info(code=code, gmsh_name=gname, dim=dim, order=1,
                              npe=len(conn[0]), count=len(conn))
        infos.append(info)
        built[code] = ElementGroup(element_type=info,
                                   ids=np.asarray(ids, dtype=np.int64),
                                   connectivity=np.asarray(conn, dtype=np.int64))
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, len(xyz) + 1, dtype=np.int64),
                            node_coords=xyz, physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups=built, physical=pg, labels=LabelSet({})),
        info=MeshInfo(n_nodes=len(xyz), n_elems=sum(len(g[2]) for g in groups.values()),
                      bandwidth=len(xyz), types=infos),
        model_name=name,
    )


def _capture(fem: FEMData, ops: Any, n: int, tmp: Path) -> "tuple[Path, Path, Spy]":
    archive = tmp / "model.h5"
    fem.to_h5(str(archive))
    spy = Spy(ops)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = capture_live_modes(archive, n, tmp / "modes.ladruno", ops=spy)
    return p, archive, spy


def _check_file(p: Path, archive: Path, spy: Spy, ops: Any, n: int,
                rotation_nodes: "set[int]", ndm: int) -> "list[float]":
    """Assert the file against the live domain; return its eigenvalues."""
    lam_live = spy.eigen_returned
    assert lam_live is not None and len(lam_live) == n
    with Results.from_ladruno(p, model_h5=archive) as res:
        modes = sorted(res.modes, key=lambda m: m.mode_index)
        assert [m.mode_index for m in modes] == list(range(1, n + 1))
        trans = [f"displacement_{a}" for a in _AXES[:ndm]]
        rots = ["rotation_z"] if ndm == 2 else [f"rotation_{a}" for a in _AXES]
        for m, lam in zip(modes, lam_live):
            assert m.eigenvalue == lam                         # bit for bit
            omega = math.sqrt(lam)
            np.testing.assert_allclose(m.frequency_hz, omega / (2 * math.pi), **_RTOL)
            np.testing.assert_allclose(m.period_s, 2 * math.pi / omega, **_RTOL)
            assert set(m.nodes.available_components()) == set(trans + rots)
            for c in trans + rots:
                s = m.nodes.get(component=c)
                ids = [int(i) for i in s.node_ids]
                got = np.asarray(s.values, dtype=np.float64)[0]
                rot = c.startswith("rotation")
                dof = (_AXES.index(c[-1]) if not rot
                       else (2 if ndm == 2 else 3 + _AXES.index(c[-1])))
                want = np.array([
                    (0.0 if rot and nid not in rotation_nodes
                     else float(ops.nodeEigenvector(nid, m.mode_index)[dof]))
                    for nid in ids])
                assert sorted(ids) == ids and len(ids) == len(set(ids))
                np.testing.assert_allclose(got, want, **_RTOL, err_msg=f"{c} mode {m.mode_index}")
    return lam_live


def _full_lapack(ops: Any, n: int) -> "np.ndarray":
    return np.asarray(ops.eigen("-fullGenLapack", n), dtype=np.float64)


# ---------------------------------------------------------------------------
# 2-D, ndf 3: a cantilever, against numpy's generalised eigenproblem
# ---------------------------------------------------------------------------


def _beam2d_k(L: float) -> "np.ndarray":
    a, b = E * A / L, E * I_Z / L ** 3
    return np.array([
        [a, 0, 0, -a, 0, 0],
        [0, 12 * b, 6 * b * L, 0, -12 * b, 6 * b * L],
        [0, 6 * b * L, 4 * b * L ** 2, 0, -6 * b * L, 2 * b * L ** 2],
        [-a, 0, 0, a, 0, 0],
        [0, -12 * b, -6 * b * L, 0, 12 * b, -6 * b * L],
        [0, 6 * b * L, 2 * b * L ** 2, 0, -6 * b * L, 4 * b * L ** 2],
    ])


def test_2d_frame_modes_equal_the_live_vectors(opspy: Any, tmp_path: Path) -> None:
    L = 1.5
    xy = np.array([[0.0, 0.0], [L, 0.0], [2 * L, 0.0]])
    fem = _fem(xy, {1: ("Line 2", 1, [[1, 2], [2, 3]], [1, 2])}, "live_modes_2d")
    ops = opspy
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for n, (x, y) in enumerate(xy, start=1):
        ops.node(n, float(x), float(y))
    ops.fix(1, 1, 1, 1)
    ops.geomTransf("Linear", 1)
    for eid, (i, j) in ((1, (1, 2)), (2, (2, 3))):
        ops.element("elasticBeamColumn", eid, i, j, A, E, I_Z, 1)
    for n in (2, 3):
        ops.mass(n, M, M, JM)

    p, archive, spy = _capture(fem, ops, 3, tmp_path)
    lam = _check_file(p, archive, spy, ops, 3, rotation_nodes={1, 2, 3}, ndm=2)

    # Independent: numpy on the assembled stiffness and lumped mass.
    K = np.zeros((9, 9))
    for i, j in ((0, 1), (1, 2)):
        dofs = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        K[np.ix_(dofs, dofs)] += _beam2d_k(L)
    free = list(range(3, 9))
    Mm = np.diag([M, M, JM, M, M, JM])
    w = np.sort(np.linalg.eigvals(np.linalg.solve(Mm, K[np.ix_(free, free)])).real)
    np.testing.assert_allclose(lam, w[:3], rtol=1e-9)


# ---------------------------------------------------------------------------
# 3-D, ndf 6: an L-frame
# ---------------------------------------------------------------------------


def test_3d_frame_modes_equal_the_live_vectors(opspy: Any, tmp_path: Path) -> None:
    xyz = np.array([[0, 0, 0], [0, 0, 1.5], [0, 0, 3.0], [1.5, 0, 3.0], [3.0, 0, 3.0]],
                   dtype=float)
    conn = [[1, 2], [2, 3], [3, 4], [4, 5]]
    fem = _fem(xyz, {1: ("Line 2", 1, conn, [1, 2, 3, 4])}, "live_modes_3d")
    ops = opspy
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for n, c in enumerate(xyz, start=1):
        ops.node(n, *[float(v) for v in c])
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)     # the column, along z
    ops.geomTransf("Linear", 2, 0.0, 0.0, 1.0)     # the beam, along x
    for eid, (i, j) in enumerate(conn, start=1):
        ops.element("elasticBeamColumn", eid, i, j, A, E, G, J, I_Y, I_Z,
                    1 if eid <= 2 else 2)
    for n in range(2, 6):
        ops.mass(n, M, M, M, JM, JM, JM)

    p, archive, spy = _capture(fem, ops, 4, tmp_path)
    lam = _check_file(p, archive, spy, ops, 4, rotation_nodes=set(range(1, 6)), ndm=3)
    np.testing.assert_allclose(lam, _full_lapack(ops, 4), rtol=1e-9)


# ---------------------------------------------------------------------------
# A solid beside a frame: ndf 3 and ndf 6 in one mode
# ---------------------------------------------------------------------------


def _solid_and_frame(ops: Any) -> FEMData:
    cube = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    col = [[3, 0, 1], [3, 0, 2], [3, 0, 3]]
    xyz = np.array(cube + col, dtype=float)
    fem = _fem(xyz, {5: ("Hexahedron 8", 3, [list(range(1, 9))], [1]),
                     1: ("Line 2", 1, [[9, 10], [10, 11]], [2, 3])}, "live_modes_mixed")
    # ``mass`` reads the builder's ndf, so each part has its own builder.
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    for n in range(1, 9):
        ops.node(n, *[float(v) for v in xyz[n - 1]])
    for n in (1, 2, 3, 4):
        ops.fix(n, 1, 1, 1)
    for n in range(5, 9):
        ops.mass(n, M, M, M)
    ops.nDMaterial("ElasticIsotropic", 1, 2.0e8, 0.25)
    ops.element("stdBrick", 1, *range(1, 9), 1)
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for n in (9, 10, 11):
        ops.node(n, *[float(v) for v in xyz[n - 1]])
    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
    for eid, (i, j) in ((2, (9, 10)), (3, (10, 11))):
        ops.element("elasticBeamColumn", eid, i, j, A, E, G, J, I_Y, I_Z, 1)
    ops.fix(9, 0, 0, 0, 1, 1, 1)
    ops.equalDOF(5, 9, 1, 2, 3)        # the cube's top corner drives the column
    for n in (10, 11):
        ops.mass(n, M, M, M, JM, JM, JM)
    ops.constraints("Transformation")
    return fem


def test_solid_and_frame_modes_equal_the_live_vectors(opspy: Any, tmp_path: Path) -> None:
    ops = opspy
    fem = _solid_and_frame(ops)
    p, archive, spy = _capture(fem, ops, 4, tmp_path)
    lam = _check_file(p, archive, spy, ops, 4, rotation_nodes={9, 10, 11}, ndm=3)
    # The brick nodes move in these modes (the coupling is live).
    v = np.array([ops.nodeEigenvector(n, 1) for n in range(5, 9)])
    assert np.abs(v).max() > 0.0
    np.testing.assert_allclose(lam, _full_lapack(ops, 4), rtol=1e-9)


# ---------------------------------------------------------------------------
# #2407 / #2414: rotations at nodes a raw deck's other classes touch
# ---------------------------------------------------------------------------


def _rotations_move(ops: Any, nodes: "tuple[int, ...]", n: int, ndm: int) -> None:
    """The live rotations at ``nodes`` are not all zero, so a file that
    wrote 0.0 there could not pass :func:`_check_file`."""
    first = 2 if ndm == 2 else 3
    v = np.array([ops.nodeEigenvector(nid, k)[first:]
                  for nid in nodes for k in range(1, n + 1)])
    assert np.abs(v).max() > 0.0


def test_an_embedded_beam_node_reads_its_live_rotations(opspy: Any, tmp_path: Path) -> None:
    """#2407: a brick plus a 3-D ``elasticBeamColumn``, with beam node 9
    tied into the brick by ``ASDEmbeddedNodeElement ... -rot``. Node 9's
    rotations are the live ``nodeEigenvector`` (they were 0.0, with a
    ``RotationSlotUnresolvedWarning``), as nodes 10 and 11 are."""
    ops = opspy
    cube = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    col = [[0.2, 0.2, 0.2], [0.2, 0.2, 1.5], [0.2, 0.2, 2.5]]
    xyz = np.array(cube + col, dtype=float)
    fem = _fem(xyz, {5: ("Hexahedron 8", 3, [list(range(1, 9))], [1]),
                     1: ("Line 2", 1, [[9, 10], [10, 11]], [2, 3])}, "live_modes_embedded")
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    for n in range(1, 9):
        ops.node(n, *[float(v) for v in xyz[n - 1]])
    for n in (1, 2, 3, 4):
        ops.fix(n, 1, 1, 1)
    for n in range(5, 9):
        ops.mass(n, M, M, M)
    ops.nDMaterial("ElasticIsotropic", 1, 2.0e8, 0.25)
    ops.element("stdBrick", 1, *range(1, 9), 1)
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for n in (9, 10, 11):
        ops.node(n, *[float(v) for v in xyz[n - 1]])
    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
    for eid, (i, j) in ((2, (9, 10)), (3, (10, 11))):
        ops.element("elasticBeamColumn", eid, i, j, A, E, G, J, I_Y, I_Z, 1)
    # Node 9 lies in the tetrahedron of brick corners 1, 2, 4, 5.
    ops.element("ASDEmbeddedNodeElement", 4, 9, 1, 2, 4, 5, "-rot")
    for n in (10, 11):
        ops.mass(n, M, M, M, JM, JM, JM)
    assert ops.eleType(4) == "ASDEmbeddedNodeElement"
    p, archive, spy = _capture(fem, ops, 3, tmp_path)
    _check_file(p, archive, spy, ops, 3, rotation_nodes={9, 10, 11}, ndm=3)
    _rotations_move(ops, (9,), 3, ndm=3)


def test_a_mixed_beam_cantilever_reads_its_live_rotations(opspy: Any, tmp_path: Path) -> None:
    """#2414: a stock 3-D cantilever, ``elasticBeamColumn`` 1-2 plus
    ``mixedBeamColumn`` 2-3 (``MixedBeamColumn3d``). Nodes 2 and 3 read
    the live rotations in every mode (they were 0.0)."""
    ops = opspy
    xyz = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]], dtype=float)
    fem = _fem(xyz, {1: ("Line 2", 1, [[1, 2], [2, 3]], [1, 2])}, "live_modes_mixed_beam")
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for n, c in enumerate(xyz, start=1):
        ops.node(n, *[float(v) for v in c])
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    ops.section("Elastic", 1, E, A, I_Z, I_Y, G, J)
    ops.beamIntegration("Lobatto", 1, 1, 3)
    ops.element("elasticBeamColumn", 1, 1, 2, A, E, G, J, I_Y, I_Z, 1)
    ops.element("mixedBeamColumn", 2, 2, 3, 1, 1)
    for n in (2, 3):
        ops.mass(n, M, M, M, JM, JM, JM)
    assert ops.eleType(2) == "MixedBeamColumn3d"
    p, archive, spy = _capture(fem, ops, 6, tmp_path)
    _check_file(p, archive, spy, ops, 6, rotation_nodes={1, 2, 3}, ndm=3)
    _rotations_move(ops, (2, 3), 6, ndm=3)


#: Each rotation-only class of ``_ROTATION_TOKEN_FOR_LIVE_CLASS``: its
#: ``ndm`` and the stock ``element`` command (after the tag and nodes)
#: that makes it, over section 1, integration 1 and transformation 1.
ROTATION_ONLY: "dict[str, tuple[int, tuple[Any, ...]]]" = {
    "MixedBeamColumn2d": (2, ("mixedBeamColumn", 1, 1)),
    "MixedBeamColumn3d": (3, ("mixedBeamColumn", 1, 1)),
    "MixedBeamColumnAsym3d": (3, ("mixedBeamColumnAsym", 1, 1)),
    "ForceBeamColumnCBDI3d": (3, ("forceBeamColumnCBDI", 1, 1)),
    "ElasticForceBeamColumn2d": (2, ("elasticForceBeamColumn", 1, 1)),
    "ElasticForceBeamColumn3d": (3, ("elasticForceBeamColumn", 1, 1)),
    "ForceBeamColumn2dThermal": (2, ("forceBeamColumnThermal", 1, 1)),
    "ForceBeamColumn3dThermal": (3, ("forceBeamColumnThermal", 1, 1)),
    "DispBeamColumnNL2d": (2, ("dispBeamColumnNL", 1, 1)),
    "DispBeamColumnNL3d": (3, ("dispBeamColumnNL", 1, 1)),
    "DispBeamColumnAsym3d": (3, ("dispBeamColumnAsym", 1, 1)),
    "DispBeamColumn2dThermal": (2, ("dispBeamColumnThermal", 1, 1)),
    "DispBeamColumn3dThermal": (3, ("dispBeamColumnThermal", 1, 1)),
    "TimoshenkoBeamColumn2d": (2, ("timoshenkoBeamColumn", 1, 1)),
    "TimoshenkoBeamColumn3d": (3, ("timoshenkoBeamColumn", 1, 1)),
    # K11, K33, K44 = 4, 4, 2 about both axes: the unmodified rows.
    "ModElasticBeam3d": (3, ("ModElasticBeam3d", A, E, G, J, I_Y, I_Z,
                             4.0, 4.0, 2.0, 4.0, 4.0, 2.0, 1)),
}


def test_the_rotation_only_oracles_cover_the_table() -> None:
    from apeGmsh.opensees._recorder_translate import _ROTATION_TOKEN_FOR_LIVE_CLASS

    assert set(ROTATION_ONLY) == set(_ROTATION_TOKEN_FOR_LIVE_CLASS)


@pytest.mark.parametrize("cls", list(ROTATION_ONLY))
def test_a_rotation_only_beam_class_reads_its_live_rotations(
    cls: str, opspy: Any, tmp_path: Path,
) -> None:
    """#2414 census: a two-element stock cantilever of ``cls`` along x is
    that ``eleType`` with the FEM's node order, and every node reads the
    live rotations. Dropping the class's row makes the capture refuse."""
    ndm, command = ROTATION_ONLY[cls]
    ops = opspy
    xyz = np.array([[0.0] * ndm, [1.5] + [0.0] * (ndm - 1), [3.0] + [0.0] * (ndm - 1)])
    fem = _fem(xyz, {1: ("Line 2", 1, [[1, 2], [2, 3]], [1, 2])}, f"live_modes_{cls}")
    ops.model("basic", "-ndm", ndm, "-ndf", 3 if ndm == 2 else 6)
    for n, c in enumerate(xyz, start=1):
        ops.node(n, *[float(v) for v in c])
    if ndm == 2:
        ops.fix(1, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        ops.section("Elastic", 1, E, A, I_Z)
    else:
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
        ops.section("Elastic", 1, E, A, I_Z, I_Y, G, J)
    ops.beamIntegration("Lobatto", 1, 1, 3)
    for eid in (1, 2):
        ops.element(command[0], eid, eid, eid + 1, *command[1:])
    for n in (2, 3):
        ops.mass(n, *((M, M, JM) if ndm == 2 else (M, M, M, JM, JM, JM)))
    assert [ops.eleType(e) for e in (1, 2)] == [cls, cls]
    assert [list(ops.eleNodes(e)) for e in (1, 2)] == [[1, 2], [2, 3]]
    n_modes = 2 if ndm == 2 else 4
    p, archive, spy = _capture(fem, ops, n_modes, tmp_path)
    _check_file(p, archive, spy, ops, n_modes, rotation_nodes={1, 2, 3}, ndm=ndm)
    _rotations_move(ops, (2, 3), n_modes, ndm=ndm)


def test_an_unknown_class_at_a_beam_node_refuses_by_name(opspy: Any, tmp_path: Path) -> None:
    """Ruling R6 = (A): a class the tables do not know
    (``ElastomericBearingPlasticity2d``) touching beam node 3 leaves its
    rotation slot unsettled, so the capture refuses by node, slot and
    class and writes nothing, where it wrote 0.0 over the live rotation."""
    ops = opspy
    xy = np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]])
    _cantilever_2d(ops, xy)
    ops.node(4, 3.0, -0.2)                # the bearing's far end, not FEM
    ops.fix(4, 1, 1, 1)
    ops.uniaxialMaterial("Elastic", 1, 1.0e6)
    ops.element("elastomericBearingPlasticity", 3, 3, 4, 1.0e6, 1.0e3, 0.05,
                0.0, 1.0, "-P", 1, "-Mz", 1)
    assert ops.eleType(3) == "ElastomericBearingPlasticity2d"
    out = tmp_path / "modes.ladruno"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=(
                r"capture_live_modes: node 3's rotation slots \['rotation_z'\] "
                r"cannot be settled: a ElastomericBearingPlasticity2d touches "
                r"it.*\(1 node\(s\): \[3\]\)\. Nothing was written")):
            capture_live_modes(_line_fem(xy), 2, out, ops=ops)
    assert not out.exists()
    assert abs(ops.nodeEigenvector(3, 1)[2]) > 0.0     # a real rotation there


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _cantilever_2d(ops: Any, xy: "np.ndarray") -> None:
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for n, (x, y) in enumerate(xy, start=1):
        ops.node(n, float(x), float(y))
    ops.fix(1, 1, 1, 1)
    ops.geomTransf("Linear", 1)
    for eid in range(1, len(xy)):
        ops.element("elasticBeamColumn", eid, eid, eid + 1, A, E, I_Z, 1)
    for n in range(2, len(xy) + 1):
        ops.mass(n, M, M, JM)


def _line_fem(xy: "np.ndarray") -> FEMData:
    conn = [[i, i + 1] for i in range(1, len(xy))]
    return _fem(xy, {1: ("Line 2", 1, conn, list(range(1, len(xy))))}, "live_modes_refuse")


def test_a_deck_off_the_fem_coordinates_refuses(opspy: Any, tmp_path: Path) -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    moved = xy.copy()
    moved[2] = [2.0, 0.8]
    _cantilever_2d(opspy, moved)
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ValueError, match=r"capture_live_modes: .*node 3: live \[2\.0, 0\.8\]"):
        capture_live_modes(_line_fem(xy), 2, out, ops=opspy)
    assert not out.exists()


def test_a_fem_node_the_deck_never_built_refuses(opspy: Any, tmp_path: Path) -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    _cantilever_2d(opspy, xy[:3])
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ValueError, match=r"capture_live_modes: nodes \[4\] are not in the live domain"):
        capture_live_modes(_line_fem(xy), 2, out, ops=opspy)
    assert not out.exists()


def test_a_non_positive_eigenvalue_refuses(opspy: Any, tmp_path: Path) -> None:
    """A negative spring on one free node: its eigenvalue is ``k / m < 0``."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    ops = opspy
    _cantilever_2d(ops, xy[:4])
    ops.node(5, 5.0, 0.0)
    ops.node(6, 5.0, 0.0)
    ops.fix(6, 1, 1, 1)
    ops.fix(5, 0, 1, 1)
    ops.uniaxialMaterial("Elastic", 1, -10.0)
    ops.element("zeroLength", 10, 6, 5, "-mat", 1, "-dir", 1)
    ops.mass(5, M, M, JM)
    # A general solver: the default SPD factorisation fails on an
    # indefinite stiffness and stock eigen then returns garbage values.
    ops.system("BandGeneral")
    # Node 6 (the spring's fixed end, at node 5's position) is not FEM.
    fem = _fem(xy, {1: ("Line 2", 1, [[1, 2], [2, 3], [3, 4]], [1, 2, 3])},
               "live_modes_neg")
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ValueError, match=r"capture_live_modes: mode \d+ has a non-positive or non-finite eigenvalue"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert not out.exists()


def test_a_second_default_solve_refuses_with_the_remedy(opspy: Any, tmp_path: Path) -> None:
    """Stock openseespy 3.7.1: a second default ``eigen`` in a domain with
    no analysis fails ("no EigenSOE has been set"), so the capture refuses,
    names ``ops.wipeAnalysis()``, and that remedy works. The fork solves
    twice without it, so there the capture succeeds at once."""
    from apeGmsh.opensees.emitter.live import get_backend_info

    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    ops = opspy
    _cantilever_2d(ops, xy)
    ops.eigen(2)                                 # the user's own solve
    out = tmp_path / "modes.ladruno"
    if get_backend_info().kind == "stock":
        with pytest.raises(ValueError, match=r"ops.eigen\(2\) failed .*wipeAnalysis") as err:
            capture_live_modes(_line_fem(xy), 2, out, ops=ops)
        assert err.value.__cause__ is not None and not out.exists()
        ops.wipeAnalysis()
    capture_live_modes(_line_fem(xy), 2, out, ops=ops)
    assert out.exists()


def _springs(ops: Any) -> FEMData:
    """Three grounded springs, k = 100, 200 and -10 on unit masses: the
    eigenvalues nearest zero are -10 and 100. The anchors (nodes 11-13,
    at the free nodes' positions) are not FEM nodes."""
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    for i, k in enumerate((100.0, 200.0, -10.0)):
        ops.node(i + 1, float(i), 0.0)
        ops.node(i + 11, float(i), 0.0)
        ops.fix(i + 11, 1, 1)
        ops.fix(i + 1, 0, 1)
        ops.uniaxialMaterial("Elastic", i + 1, k)
        ops.element("zeroLength", i + 1, i + 11, i + 1, "-mat", i + 1, "-dir", 1)
        ops.mass(i + 1, 1.0, 1.0)
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    return _fem(xy, {15: ("Point", 0, [[1], [2], [3]], [1, 2, 3])}, "live_modes_springs")


#: The remedy calls a refusal may name, run in the order it names them.
_REMEDY = re.compile(r"ops\.(wipeAnalysis|system)\(([^)]*)\)")


def _follow(ops: Any, message: str) -> None:
    calls = _REMEDY.findall(message)
    assert calls, message
    for name, args in calls:
        getattr(ops, name)(*ast.literal_eval(f"({args},)") if args else ())


def test_the_documented_remedy_keeps_the_bandgeneral_guard(opspy: Any, tmp_path: Path) -> None:
    """#2365: on stock openseespy a second eigen solve refuses, and the
    remedy its message names, run as written, must keep the
    ``system BandGeneral`` guard: the next solve is the true [-10, 100]
    (``wipeAnalysis()`` alone gives a plausible [1, 1]), and the capture
    then refuses the -10 by name. The fork solves twice, so there the
    capture refuses the -10 at once."""
    from apeGmsh.opensees.emitter.live import get_backend_info

    ops = opspy
    fem = _springs(ops)
    ops.system("BandGeneral")
    np.testing.assert_allclose(ops.eigen(2), [-10.0, 100.0], rtol=1e-12)
    out = tmp_path / "modes.ladruno"
    if get_backend_info().kind == "stock":
        with pytest.raises(ValueError, match=r"ops.eigen\(2\) failed") as err:
            capture_live_modes(fem, 2, out, ops=ops)
        remedy = str(err.value)
        _follow(ops, remedy)
        np.testing.assert_allclose(ops.eigen(2), [-10.0, 100.0], rtol=1e-12)
        _follow(ops, remedy)
    with pytest.raises(ValueError, match=r"mode 1 has a non-positive or non-finite eigenvalue \(-10\)"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert not out.exists()


def test_the_fullgen_solver_guards_the_capture(opspy: Any, tmp_path: Path) -> None:
    """#2366 F1: the default stock solver gives a plausible [1, 1] for the
    3-spring deck; ``solver="-fullGenLapack"`` makes the capture's own
    solve see the -10 and refuse it by name, writing nothing."""
    ops = opspy
    fem = _springs(ops)
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ValueError, match=r"mode 1 has a non-positive or non-finite eigenvalue \(-10\)"):
        capture_live_modes(fem, 2, out, ops=ops, solver="-fullGenLapack")
    assert not out.exists()


def test_an_analysis_with_the_user_s_system_solves_twice(opspy: Any, tmp_path: Path) -> None:
    """#2366 F2: with ``ops.analysis(...)`` defined on a non-default system
    (UmfPack) before the first eigen, a second solve works on stock and
    stays right: [-10, 100] before and after the capture, which refuses
    the -10 by name."""
    ops = opspy
    fem = _springs(ops)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-8, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    np.testing.assert_allclose(ops.eigen(2), [-10.0, 100.0], rtol=1e-12)
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ValueError, match=r"mode 1 has a non-positive or non-finite eigenvalue \(-10\)"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert not out.exists()
    np.testing.assert_allclose(ops.eigen(2), [-10.0, 100.0], rtol=1e-12)
