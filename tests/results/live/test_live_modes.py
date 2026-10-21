"""#2342 gap (a): ``capture_live_modes`` over a fake live domain.

The fake answers the openseespy queries the capture makes (``getNDM``,
``getNodeTags``, ``getNDF``, ``nodeCoord``, ``getEleTags``, ``eleType``,
``eleNodes``, ``eigen``, ``nodeEigenvector``) from tables the test writes,
so the expected file is known entry by entry:

* every written shape is the raw ``nodeEigenvector`` of its node and mode
  (no scaling), and every eigenvalue is the solve's, bit for bit;
* a rotation slot holds the rotation only where a beam touches the node;
  a 3-D solid node writes 0.0 there; a node under an unknown class
  refuses by name where its slot is non-zero in a mode, and writes 0.0
  with one ``RotationSlotUnresolvedWarning`` where it is 0.0 in every mode;
* an ``ASDEmbeddedNodeElement`` that cannot carry a pressure (a node of
  its with ndf other than ``ndm + 1``) leaves a beam node's rotation
  written; an embedded node whose rotation slot no element settles, or
  whose element may carry a pressure, refuses by name (#2407);
* each refusal (wrong coordinates, a missing node, a non-positive or
  non-finite eigenvalue, a failed solve, a short solve, a vector whose
  length is not the node's ndf,
  ``n_modes`` < 1, the archive as the output path) names its cause and
  writes nothing.

The stock-openseespy oracle is ``tests/opensees/live/test_live_modes_live.py``.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite
from apeGmsh.opensees.analysis.eigen import RotationSlotUnresolvedWarning
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.results import Results
from apeGmsh.results.live import capture_live_modes

_AXES = ("x", "y", "z")


def _fem(coords: "np.ndarray", groups: "dict[int, tuple[str, int, list[list[int]], list[int]]]",
         name: str = "live_modes") -> FEMData:
    """Node ``k + 1`` at ``coords[k]``; ``groups`` maps a Gmsh type code to
    ``(name, dim, connectivity, ids)``."""
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


class FakeOps:
    """A live domain from tables: ``coords[n]``, ``ndf[n]``, elements
    ``{tag: (eleType, nodes)}``, eigenvalues and ``vectors[(n, k)]``."""

    def __init__(self, ndm: int, coords: "dict[int, list[float]]",
                 ndf: "dict[int, int]", elements: "dict[int, tuple[str, list[int]]]",
                 eigenvalues: "list[float]", seed: int = 0) -> None:
        self.ndm, self.coords, self.ndf = ndm, coords, ndf
        self.elements, self.eigenvalues = elements, eigenvalues
        rng = np.random.default_rng(seed)
        self.vectors = {(n, k): rng.standard_normal(ndf[n]) * 10.0 ** (k - 3)
                        for n in coords for k in range(1, len(eigenvalues) + 1)}
        self.solved: "list[int]" = []
        #: The arguments of each ``eigen`` call.
        self.calls: "list[tuple[Any, ...]]" = []

    def getNDM(self) -> list[int]:
        return [self.ndm]

    def getNodeTags(self) -> list[int]:
        return list(self.coords)

    def getNDF(self, n: int) -> list[int]:
        return [self.ndf[n]]

    def nodeCoord(self, n: int) -> list[float]:
        return list(self.coords[n])

    def getEleTags(self) -> list[int]:
        return list(self.elements)

    def eleType(self, tag: int) -> str:
        return self.elements[tag][0]

    def eleNodes(self, tag: int) -> list[int]:
        return list(self.elements[tag][1])

    def eigen(self, *args: Any) -> list[float]:
        self.calls.append(args)
        n = int(args[-1])
        self.solved.append(n)
        return list(self.eigenvalues[:n])

    def nodeEigenvector(self, n: int, k: int) -> list[float]:
        return [float(v) for v in self.vectors[(n, k)]]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _frame_2d() -> "tuple[FEMData, FakeOps]":
    """A 2-D, ndf 3 cantilever of two ElasticBeam2d elements."""
    xy = np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]])
    fem = _fem(xy, {1: ("Line 2", 1, [[1, 2], [2, 3]], [1, 2])})
    ops = FakeOps(2, {n + 1: list(xy[n]) for n in range(3)},
                  {1: 3, 2: 3, 3: 3},
                  {1: ("ElasticBeam2d", [1, 2]), 2: ("ElasticBeam2d", [2, 3])},
                  [4.0, 25.0, 300.0])
    return fem, ops


def _solid_and_frame(beam_class: str = "ElasticBeam3d") -> "tuple[FEMData, FakeOps]":
    """A Brick (ndf 3) beside a 3-D beam column (ndf 6), nodes 9-11."""
    cube = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    col = [[3, 0, 1], [3, 0, 2], [3, 0, 3]]
    xyz = np.array(cube + col, dtype=float)
    fem = _fem(xyz, {5: ("Hexahedron 8", 3, [list(range(1, 9))], [1]),
                     1: ("Line 2", 1, [[9, 10], [10, 11]], [2, 3])})
    ndf = {n: 3 for n in range(1, 9)} | {9: 6, 10: 6, 11: 6}
    ops = FakeOps(3, {n + 1: list(xyz[n]) for n in range(11)}, ndf,
                  {1: ("Brick", list(range(1, 9))),
                   2: (beam_class, [9, 10]), 3: (beam_class, [10, 11])},
                  [9.0, 16.0, 81.0, 400.0], seed=1)
    return fem, ops


def _model(fem: FEMData, tmp: Path) -> OpenSeesModel:
    archive = tmp / "read_model.h5"
    if not archive.exists():
        fem.to_h5(str(archive))
    return OpenSeesModel.from_h5(archive)


def _read(p: Path, fem: FEMData) -> "dict[int, dict[str, dict[int, float]]]":
    out: dict[int, dict[str, dict[int, float]]] = {}
    with Results.from_native(p, model=_model(fem, p.parent)) as res:
        for m in res.modes:
            comps: dict[str, dict[int, float]] = {}
            for c in m.nodes.available_components():
                s = m.nodes.get(component=c)
                comps[c] = dict(zip((int(i) for i in s.node_ids),
                                    (float(v) for v in np.asarray(s.values)[0])))
            out[int(m.mode_index)] = comps
    return out


# ---------------------------------------------------------------------------
# The written file
# ---------------------------------------------------------------------------


def test_2d_frame_writes_the_raw_vectors(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = capture_live_modes(fem, 3, tmp_path / "modes.h5", ops=ops)
    assert ops.solved == [3]
    got = _read(p, fem)
    assert sorted(got) == [1, 2, 3]
    for k, comps in got.items():
        assert set(comps) == {"displacement_x", "displacement_y", "rotation_z"}
        for d, c in enumerate(("displacement_x", "displacement_y", "rotation_z")):
            assert comps[c] == {n: float(ops.vectors[(n, k)][d]) for n in (1, 2, 3)}
    with Results.from_native(p, model=_model(fem, tmp_path)) as res:
        modes = sorted(res.modes, key=lambda m: m.mode_index)
        for m, lam in zip(modes, ops.eigenvalues):
            assert m.eigenvalue == lam
            assert m.frequency_hz == math.sqrt(lam) / (2.0 * math.pi)
            assert m.period_s == 2.0 * math.pi / math.sqrt(lam)
    # The one-call route for a bare FEMData opens the same modes.
    with Results.from_fem(fem, p, kind="native", cache_root=tmp_path / "c") as res:
        assert [m.eigenvalue for m in sorted(res.modes, key=lambda m: m.mode_index)] \
            == ops.eigenvalues


def test_solid_and_frame_rotations_only_on_beam_nodes(tmp_path: Path) -> None:
    fem, ops = _solid_and_frame()
    archive = tmp_path / "model.h5"           # the path form of the input
    fem.to_h5(str(archive))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = capture_live_modes(str(archive), 4, tmp_path / "modes.h5", ops=ops)
    got = _read(p, fem)
    for k, comps in got.items():
        assert set(comps) == {f"displacement_{a}" for a in _AXES} | {
            f"rotation_{a}" for a in _AXES}
        for n in range(1, 12):
            vec = ops.vectors[(n, k)]
            for d, a in enumerate(_AXES):
                assert comps[f"displacement_{a}"][n] == float(vec[d])
                want = float(vec[3 + d]) if n >= 9 else 0.0
                assert comps[f"rotation_{a}"][n] == want


def test_an_unknown_class_over_live_rotations_refuses(tmp_path: Path) -> None:
    """The beam nodes' rotation slots are non-zero, and the class touching
    them is not in the live table, so a 0.0 could replace a real rotation:
    the capture refuses, naming the class (#2407)."""
    fem, ops = _solid_and_frame(beam_class="SomeUnknownBeam")
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=(
            r"capture_live_modes: the rotation slots of 3 node\(s\) "
            r"\(\[9, 10, 11\]\).*for node 9, a SomeUnknownBeam touches it.*"
            r"Its live value there reaches .*SomeUnknownBeam is not in the "
            r"live class table.*Nothing was written")):
        capture_live_modes(fem, 2, out, ops=ops)
    assert not out.exists()


@pytest.mark.parametrize("cls", ["FourNodeQuadUP", "BBarFourNodeQuadUP",
                                 "SSPquadUP", "NineFourNodeQuadUP", "LadrunoUP"])
def test_a_stock_up_class_slot_is_a_pressure_not_a_rotation(
        tmp_path: Path, cls: str) -> None:
    """2-D, ndf 3 under a stock u-p class: the third DOF is a pore
    pressure (non-zero here), so no rotation is written, nothing refuses
    and nothing warns."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    fem = _fem(xy, {3: ("Quadrilateral 4", 2, [[1, 2, 3, 4]], [1])})
    ops = FakeOps(2, {n + 1: list(xy[n]) for n in range(4)},
                  {n: 3 for n in range(1, 5)}, {1: (cls, [1, 2, 3, 4])},
                  [4.0, 25.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = capture_live_modes(fem, 2, tmp_path / "modes.h5", ops=ops)
    got = _read(p, fem)
    for k, comps in got.items():
        assert set(comps) == {"displacement_x", "displacement_y"}
        assert any(ops.vectors[(n, k)][2] != 0.0 for n in range(1, 5))


@pytest.mark.parametrize("case", ["mode_2_only", "tiny"])
def test_an_unknown_class_refuses_on_any_non_zero_rotation(
        tmp_path: Path, case: str) -> None:
    """The check reads every captured mode and has no tolerance: a
    rotation that is non-zero only in mode 2, or that is 1e-9 of the
    translations, still refuses."""
    fem, ops = _solid_and_frame(beam_class="SomeUnknownBeam")
    for (n, k), v in ops.vectors.items():
        if n >= 9:
            scale = float(np.abs(v[:3]).max())
            v[3:] = (0.0 if case == "mode_2_only" and k != 2
                     else 1e-9 * scale if case == "tiny" else v[3:])
    assert all(ops.vectors[(9, k)][3:].any() == (case == "tiny" or k == 2)
               for k in (1, 2, 3))
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=(
            r"rotation slots of 3 node\(s\) \(\[9, 10, 11\]\).*SomeUnknownBeam "
            r"is not in the live class table")):
        capture_live_modes(fem, 3, out, ops=ops)
    assert not out.exists()


def test_an_unknown_class_over_zero_rotations_writes_zero_with_one_warning(
        tmp_path: Path) -> None:
    """Where the unsettled slots are 0.0 in every mode, 0.0 is their live
    value: they are written with one warning."""
    fem, ops = _solid_and_frame(beam_class="SomeUnknownBeam")
    for (n, k), v in ops.vectors.items():
        if n >= 9:
            v[3:] = 0.0
    with pytest.warns(RotationSlotUnresolvedWarning, match=r"capture_live_modes") as rec:
        p = capture_live_modes(fem, 2, tmp_path / "modes.h5", ops=ops)
    assert len(rec) == 1
    got = _read(p, fem)
    # No node holds a rotation, so no rotation component is written.
    assert set(got[1]) == {f"displacement_{a}" for a in _AXES}


# ---------------------------------------------------------------------------
# ASDEmbeddedNodeElement (#2407)
# ---------------------------------------------------------------------------


def _embedded(extra: "dict[int, tuple[str, list[int]]]") -> "tuple[FEMData, FakeOps]":
    """:func:`_solid_and_frame` with the column's base node 9 embedded in
    the brick's tet (5, 6, 8, 1), as the bridge emits ``embedded``, plus
    ``extra`` elements."""
    fem, ops = _solid_and_frame()
    ops.elements[4] = ("ASDEmbeddedNodeElement", [9, 5, 6, 8, 1])
    ops.elements.update(extra)
    return fem, ops


def test_an_embedded_beam_node_writes_its_rotation(tmp_path: Path) -> None:
    """The base node's ndf is 6 and the brick's nodes' ndf 3, so the
    element cannot carry a pressure (that needs every node at ndf 4): the
    beam's rotation is written, with no warning, not 0.0."""
    fem, ops = _embedded({})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = capture_live_modes(fem, 4, tmp_path / "modes.h5", ops=ops)
    got = _read(p, fem)
    for k, comps in got.items():
        for n in range(1, 12):
            vec = ops.vectors[(n, k)]
            for d, a in enumerate(_AXES):
                want = float(vec[3 + d]) if n >= 9 else 0.0
                assert comps[f"rotation_{a}"][n] == want
        assert all(comps[f"rotation_{a}"][9] != 0.0 for a in _AXES)


def test_an_embedded_node_no_beam_touches_refuses(tmp_path: Path) -> None:
    """Node 11 (ndf 6) is embedded, and the beam that touched it is gone:
    no element settles its rotation slots, and the element may tie them
    to the host's rotation (``-rot``), so the capture refuses by name
    rather than write 0.0."""
    fem, ops = _embedded({5: ("ASDEmbeddedNodeElement", [11, 5, 6, 8, 1])})
    del ops.elements[3]
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=(
            r"capture_live_modes: the rotation slots of 1 node\(s\) "
            r"\(\[11\]\).*for node 11, no beam-column or shell.*An "
            r"ASDEmbeddedNodeElement touches it and may tie the slot to the "
            r"host's rotation \(-rot\).*Nothing was written")):
        capture_live_modes(fem, 2, out, ops=ops)
    assert not out.exists()


def test_an_embedded_node_that_may_carry_a_pressure_refuses(tmp_path: Path) -> None:
    """2-D, every node of the element at ndf 3 = ndm + 1: the element may
    tie the third DOF to a pore pressure (``-p``), so the beam node's slot
    is not settled and the capture refuses by name."""
    fem, ops = _frame_2d()
    ops.coords[4] = [0.75, 0.5]
    ops.ndf[4] = 3
    ops.vectors |= {(4, k): np.ones(3) for k in (1, 2, 3)}
    ops.elements[3] = ("ASDEmbeddedNodeElement", [2, 1, 3, 4])
    fem = _fem(np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0], [0.75, 0.5]]),
               {1: ("Line 2", 1, [[1, 2], [2, 3]], [1, 2])})
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=(
            r"capture_live_modes: the rotation slots of 4 node\(s\) "
            r"\(\[1, 2, 3, 4\]\).*for node 1, a ASDEmbeddedNodeElement touches "
            r"it.*Every node of that ASDEmbeddedNodeElement has ndf ndm \+ 1")):
        capture_live_modes(fem, 2, out, ops=ops)
    assert not out.exists()


def test_a_2d_embedded_beam_node_over_ndf_2_writes_its_rotation(tmp_path: Path) -> None:
    """2-D: the retained nodes have ndf 2, so no pressure is possible and
    the beam node's ``rotation_z`` is written."""
    fem, ops = _frame_2d()
    for n, xy in ((4, [0.0, 1.0]), (5, [3.0, 1.0])):
        ops.coords[n], ops.ndf[n] = xy, 2
        ops.vectors |= {(n, k): np.ones(2) for k in (1, 2, 3)}
    ops.elements[3] = ("ASDEmbeddedNodeElement", [2, 1, 4, 5])
    fem = _fem(np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0], [0.0, 1.0], [3.0, 1.0]]),
               {1: ("Line 2", 1, [[1, 2], [2, 3]], [1, 2])})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = capture_live_modes(fem, 3, tmp_path / "modes.h5", ops=ops)
    got = _read(p, fem)
    for k, comps in got.items():
        for n in (1, 2, 3):
            assert comps["rotation_z"][n] == float(ops.vectors[(n, k)][2])
        assert comps["rotation_z"][2] != 0.0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_wrong_coordinates_refuse_by_node(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    ops.coords[2] = [1.5, 1.0]
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=r"capture_live_modes: .*node 2: live"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert ops.solved == [] and not out.exists()


def test_a_missing_node_refuses_by_id(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    del ops.coords[3]
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=r"capture_live_modes: nodes \[3\] are not in the live domain"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert ops.solved == [] and not out.exists()


@pytest.mark.parametrize("lam", [0.0, -2.5, float("nan"), float("inf")])
def test_a_non_positive_eigenvalue_refuses_by_mode(tmp_path: Path, lam: float) -> None:
    fem, ops = _frame_2d()
    ops.eigenvalues = [4.0, lam, 300.0]
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=r"capture_live_modes: mode 2 has a non-positive or non-finite eigenvalue"):
        capture_live_modes(fem, 3, out, ops=ops)
    assert not out.exists()


class _SolverError(Exception):
    """Stands in for the module's ``OpenSeesError``."""


def test_a_failed_solve_refuses_with_the_cause(tmp_path: Path) -> None:
    fem, ops = _frame_2d()

    def fail(n: int) -> list[float]:
        raise _SolverError("See stderr output")

    ops.eigen = fail  # type: ignore[method-assign,assignment]
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=r"ops.eigen\(2\) failed .*wipeAnalysis") as err:
        capture_live_modes(fem, 2, out, ops=ops)
    assert isinstance(err.value.__cause__, _SolverError)
    assert not out.exists()
    # The remedy re-declares the system wipeAnalysis() drops (#2365), in
    # the words notebook 17 uses.
    assert ('call ops.wipeAnalysis(), then ops.system("BandGeneral") again, '
            "before capture_live_modes") in str(err.value)
    # ... then the way to keep one's own system (#2366 F2).
    assert ("; or, to keep your own system, define ops.analysis(...) with it "
            "before the first eigen.") in str(err.value)


@pytest.mark.parametrize("solver", ["-genBandArpack", "-fullGenLapack"])
def test_the_solver_flag_is_forwarded(tmp_path: Path, solver: str) -> None:
    fem, ops = _frame_2d()
    capture_live_modes(fem, 3, tmp_path / "modes.h5", ops=ops, solver=solver)
    assert ops.calls == [(solver, 3)]


def test_no_solver_calls_the_default(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    capture_live_modes(fem, 3, tmp_path / "modes.h5", ops=ops)
    assert ops.calls == [(3,)]


@pytest.mark.parametrize("solver", ["-bogus", "fullGenLapack", "-symmBandLapack", "", "-standard"])
def test_an_unknown_solver_flag_refuses_by_name(tmp_path: Path, solver: str) -> None:
    """openseespy ignores an unknown flag and solves with its default, so
    the capture must refuse it rather than pass it on."""
    fem, ops = _frame_2d()
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=rf"solver={solver!r} is not an eigen solver flag"):
        capture_live_modes(fem, 2, out, ops=ops, solver=solver)
    assert ops.calls == [] and not out.exists()


def test_the_docstring_guards_the_capture_s_own_solve() -> None:
    """A check outside the capture does not guard its solve (#2366 F1): the
    docstring names the in-capture guards and suggests no standalone
    ``-fullGenLapack`` solve."""
    doc = " ".join((capture_live_modes.__doc__ or "").split())
    assert "does not guard the capture's own solve" in doc
    assert 'pass ``solver="-fullGenLapack"``' in doc
    assert 'declare ``ops.system("BandGeneral")`` before capturing' in doc
    assert 'ops.eigen("-fullGenLapack"' not in doc


def test_a_short_solve_refuses(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=r"ops.eigen\(5\) returned 3 eigenvalue"):
        capture_live_modes(fem, 5, out, ops=ops)
    assert not out.exists()


def test_a_vector_longer_than_the_live_ndf_refuses(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    ops.vectors[(2, 1)] = np.zeros(6)
    out = tmp_path / "modes.h5"
    with pytest.raises(ValueError, match=r"node 2's eigenvector has 6 entries but its live ndf is 3"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert not out.exists()


@pytest.mark.parametrize("n", [0, -1, True, 2.0])
def test_n_modes_must_be_a_positive_integer(tmp_path: Path, n: Any) -> None:
    fem, ops = _frame_2d()
    with pytest.raises(ValueError, match=r"n_modes must be a positive integer"):
        capture_live_modes(fem, n, tmp_path / "modes.h5", ops=ops)
    assert ops.solved == []


def test_the_archive_is_not_the_output(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    archive = tmp_path / "model.h5"
    fem.to_h5(str(archive))
    before = archive.read_bytes()
    with pytest.raises(ValueError, match=r"is the model archive itself"):
        capture_live_modes(archive, 2, archive, ops=ops)
    assert archive.read_bytes() == before and ops.solved == []


def test_a_missing_archive_refuses(tmp_path: Path) -> None:
    _, ops = _frame_2d()
    with pytest.raises(FileNotFoundError, match=r"no model archive"):
        capture_live_modes(tmp_path / "absent.h5", 2, tmp_path / "modes.h5", ops=ops)


def test_a_domain_with_no_model_refuses(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    ops.ndm = 0
    with pytest.raises(ValueError, match=r"declares no ndm"):
        capture_live_modes(fem, 2, tmp_path / "modes.h5", ops=ops)
