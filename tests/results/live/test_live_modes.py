"""#2342 gap (a): ``capture_live_modes`` over a fake live domain.

The fake answers the openseespy queries the capture makes (``getNDM``,
``getNodeTags``, ``getNDF``, ``nodeCoord``, ``getEleTags``, ``eleType``,
``eleNodes``, ``eigen``, ``nodeEigenvector``) from tables the test writes,
so the expected file is known entry by entry. The file is a ``.ladruno``
(ADR 0131 D10, W1b #2768), read back through ``Results.from_ladruno``:

* every written shape is the raw ``nodeEigenvector`` of its node and mode
  (no scaling), and every eigenvalue is the solve's, bit for bit, with
  ``f = √λ/2π`` and ``T = 2π/√λ`` (the native route's values);
* a rotation slot holds the rotation only where a beam touches the node;
  a 3-D solid node writes 0.0 there, and a node under an unknown class
  writes 0.0 with one ``RotationSlotUnresolvedWarning``;
* the stamp: a FEMData (or its ``fem.to_h5`` archive) gives ``fem_hash``
  alone, byte-equal to the archive's (ADR 0133 D1); a refused stamp
  leaves no file and no staging directory;
* each refusal (wrong coordinates, a missing node, a non-positive or
  non-finite eigenvalue, a failed solve, a short solve, a vector whose
  length is not the node's ndf,
  ``n_modes`` < 1, the archive as the output path, an ``.h5`` path) names
  its cause and writes nothing.

The stock-openseespy oracle is ``tests/opensees/live/test_live_modes_live.py``.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite
from apeGmsh.opensees.analysis.eigen import RotationSlotUnresolvedWarning
from apeGmsh.results import Results
from apeGmsh.results.live import capture_live_modes
from apeGmsh.results.writers import ProvenanceStampRefused

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


def _model(fem: FEMData, tmp: Path) -> Path:
    archive = tmp / "read_model.h5"
    if not archive.exists():
        fem.to_h5(str(archive))
    return archive


def _read(p: Path, fem: FEMData) -> "dict[int, dict[str, dict[int, float]]]":
    out: dict[int, dict[str, dict[int, float]]] = {}
    with Results.from_ladruno(p, model_h5=_model(fem, p.parent)) as res:
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
        p = capture_live_modes(fem, 3, tmp_path / "modes.ladruno", ops=ops)
    assert ops.solved == [3]
    got = _read(p, fem)
    assert sorted(got) == [1, 2, 3]
    for k, comps in got.items():
        assert set(comps) == {"displacement_x", "displacement_y", "rotation_z"}
        for d, c in enumerate(("displacement_x", "displacement_y", "rotation_z")):
            assert comps[c] == {n: float(ops.vectors[(n, k)][d]) for n in (1, 2, 3)}
    with Results.from_ladruno(p, model_h5=_model(fem, tmp_path)) as res:
        modes = sorted(res.modes, key=lambda m: m.mode_index)
        assert len(modes) == 3
        for m, lam in zip(modes, ops.eigenvalues):
            assert m.eigenvalue == lam
            assert m.frequency_hz == math.sqrt(lam) / (2.0 * math.pi)
            assert m.period_s == 2.0 * math.pi / math.sqrt(lam)
    # The FEMData itself binds the same modes, with no archive.
    with Results.from_ladruno(p, fem=fem) as res:
        assert [m.eigenvalue for m in sorted(res.modes, key=lambda m: m.mode_index)] \
            == ops.eigenvalues


def test_solid_and_frame_rotations_only_on_beam_nodes(tmp_path: Path) -> None:
    fem, ops = _solid_and_frame()
    archive = tmp_path / "model.h5"           # the path form of the input
    fem.to_h5(str(archive))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = capture_live_modes(str(archive), 4, tmp_path / "modes.ladruno", ops=ops)
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


def test_an_unknown_class_writes_zero_rotations_with_one_warning(tmp_path: Path) -> None:
    fem, ops = _solid_and_frame(beam_class="SomeUnknownBeam")
    with pytest.warns(RotationSlotUnresolvedWarning, match=r"capture_live_modes") as rec:
        p = capture_live_modes(fem, 2, tmp_path / "modes.ladruno", ops=ops)
    assert len(rec) == 1
    got = _read(p, fem)
    # No node holds a rotation: the recorder layout writes the rotational
    # group anyway, all 0.0, as the native route wrote no rotation.
    for comps in got.values():
        for a in _AXES:
            assert set(comps[f"rotation_{a}"].values()) == {0.0}
            assert len(comps[f"displacement_{a}"]) == 11


# ---------------------------------------------------------------------------
# The stamp (ADR 0133): fem_hash alone for a FEMData, both for a bridge
# ---------------------------------------------------------------------------


def _stamp(p: Path) -> "dict[str, str]":
    with h5py.File(p, "r") as f:
        return {k: str(v) for k, v in f["INFO/PROVENANCE"].attrs.items()}


def _lineage(archive: Path) -> "dict[str, str]":
    with h5py.File(archive, "r") as f:
        return {k: str(v) for k, v in f["meta/lineage"].attrs.items()
                if k in ("model_hash", "fem_hash")}


@pytest.mark.parametrize("form", ["femdata", "archive"])
def test_a_femdata_capture_is_stamped_by_fem_hash_alone(tmp_path: Path, form: str) -> None:
    """The #2342 route: no bridge, so no model_hash; the stamp carries the
    archive's fem_hash byte for byte, the one ``fem.to_h5`` writes, so the
    file pairs with the user's own archive of the mesh."""
    fem, ops = _frame_2d()
    archive = tmp_path / "model.h5"
    fem.to_h5(str(archive))
    want = _lineage(archive)
    assert set(want) == {"fem_hash"}, "precondition: a FEMData archive has no model_hash"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = capture_live_modes(fem if form == "femdata" else archive, 3,
                               tmp_path / "modes.ladruno", ops=ops)
    stamp = _stamp(p)
    assert "model_hash" not in stamp
    assert stamp["fem_hash"] == want["fem_hash"]
    assert stamp["writer"].startswith("apeGmsh live_modes->ladruno ")
    assert stamp["unsupported"] == '["elements"]'


def _truss_bridge(path: Path) -> Path:
    """A bridge-written archive (``ops.h5``) of two bars on nodes 1-2-3."""
    from apeGmsh.opensees import apeSees

    node_ids = np.array([1, 2, 3], dtype=np.int64)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2)
    group = ElementGroup(element_type=info, ids=np.array([2, 3], dtype=np.int64),
                         connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64))
    pg = PhysicalGroupSet({(1, 1): {"name": "Bars", "node_ids": node_ids,
                                    "node_coords": coords, "element_ids": group.ids}})
    fem = FEMData(
        nodes=NodeComposite(node_ids=node_ids, node_coords=coords, physical=pg,
                            labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg, labels=LabelSet({})),
        info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=2, types=[info]),
    )
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    ops.element.Truss(pg="Bars", A=1.0,
                      material=ops.uniaxialMaterial.ElasticMaterial(E=1000.0))
    ops.h5(str(path))
    return path


def test_a_bridge_archive_capture_carries_both_hashes(tmp_path: Path) -> None:
    archive = _truss_bridge(tmp_path / "bridge.h5")
    want = _lineage(archive)
    assert set(want) == {"model_hash", "fem_hash"}
    ops = FakeOps(2, {1: [0.0, 0.0], 2: [1.0, 0.0], 3: [2.0, 0.0]},
                  {1: 2, 2: 2, 3: 2},
                  {2: ("Truss", [1, 2]), 3: ("Truss", [2, 3])}, [4.0, 9.0])
    p = capture_live_modes(archive, 2, tmp_path / "modes.ladruno", ops=ops)
    stamp = _stamp(p)
    assert {k: stamp[k] for k in want} == want
    with Results.from_ladruno(p, model_h5=archive) as res:
        assert [m.eigenvalue for m in sorted(res.modes, key=lambda m: m.mode_index)] \
            == [4.0, 9.0]


def test_a_refused_stamp_leaves_no_file_behind(tmp_path: Path) -> None:
    """An archive the stamp refuses (no /meta/lineage): nothing new at the
    path, no staging directory, and a file already there is untouched."""
    fem, ops = _frame_2d()
    archive = tmp_path / "model.h5"
    fem.to_h5(str(archive))
    with h5py.File(archive, "r+") as f:
        del f["meta/lineage"]
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ProvenanceStampRefused, match=r"no /meta/lineage"):
        capture_live_modes(archive, 2, out, ops=ops)
    assert sorted(q.name for q in tmp_path.iterdir()) == ["model.h5"]

    out.write_bytes(b"an earlier file")
    with pytest.raises(ProvenanceStampRefused):
        capture_live_modes(archive, 2, out, ops=ops)
    assert out.read_bytes() == b"an earlier file"
    assert sorted(q.name for q in tmp_path.iterdir()) == ["model.h5", "modes.ladruno"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["modes.h5", "MODES.H5"])
def test_an_h5_path_refuses_before_the_solve(tmp_path: Path, name: str) -> None:
    fem, ops = _frame_2d()
    out = tmp_path / name
    with pytest.raises(ValueError, match=r"capture_live_modes: path .* has an \.h5 suffix"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert ops.solved == [] and not out.exists()


def test_wrong_coordinates_refuse_by_node(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    ops.coords[2] = [1.5, 1.0]
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ValueError, match=r"capture_live_modes: .*node 2: live"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert ops.solved == [] and not out.exists()


def test_a_missing_node_refuses_by_id(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    del ops.coords[3]
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ValueError, match=r"capture_live_modes: nodes \[3\] are not in the live domain"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert ops.solved == [] and not out.exists()


@pytest.mark.parametrize("lam", [0.0, -2.5, float("nan"), float("inf")])
def test_a_non_positive_eigenvalue_refuses_by_mode(tmp_path: Path, lam: float) -> None:
    fem, ops = _frame_2d()
    ops.eigenvalues = [4.0, lam, 300.0]
    out = tmp_path / "modes.ladruno"
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
    out = tmp_path / "modes.ladruno"
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
    capture_live_modes(fem, 3, tmp_path / "modes.ladruno", ops=ops, solver=solver)
    assert ops.calls == [(solver, 3)]


def test_no_solver_calls_the_default(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    capture_live_modes(fem, 3, tmp_path / "modes.ladruno", ops=ops)
    assert ops.calls == [(3,)]


@pytest.mark.parametrize("solver", ["-bogus", "fullGenLapack", "-symmBandLapack", "", "-standard"])
def test_an_unknown_solver_flag_refuses_by_name(tmp_path: Path, solver: str) -> None:
    """openseespy ignores an unknown flag and solves with its default, so
    the capture must refuse it rather than pass it on."""
    fem, ops = _frame_2d()
    out = tmp_path / "modes.ladruno"
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
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ValueError, match=r"ops.eigen\(5\) returned 3 eigenvalue"):
        capture_live_modes(fem, 5, out, ops=ops)
    assert not out.exists()


def test_a_vector_longer_than_the_live_ndf_refuses(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    ops.vectors[(2, 1)] = np.zeros(6)
    out = tmp_path / "modes.ladruno"
    with pytest.raises(ValueError, match=r"node 2's eigenvector has 6 entries but its live ndf is 3"):
        capture_live_modes(fem, 2, out, ops=ops)
    assert not out.exists()


@pytest.mark.parametrize("n", [0, -1, True, 2.0])
def test_n_modes_must_be_a_positive_integer(tmp_path: Path, n: Any) -> None:
    fem, ops = _frame_2d()
    with pytest.raises(ValueError, match=r"n_modes must be a positive integer"):
        capture_live_modes(fem, n, tmp_path / "modes.ladruno", ops=ops)
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
        capture_live_modes(tmp_path / "absent.h5", 2, tmp_path / "modes.ladruno", ops=ops)


def test_a_domain_with_no_model_refuses(tmp_path: Path) -> None:
    fem, ops = _frame_2d()
    ops.ndm = 0
    with pytest.raises(ValueError, match=r"declares no ndm"):
        capture_live_modes(fem, 2, tmp_path / "modes.ladruno", ops=ops)
