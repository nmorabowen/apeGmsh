"""``spec.emit_default`` over a FEMData archive reads its DOF facts from
the live domain (#2251, ruling (b) on #2191).

A FEMData archive (``fem.to_h5``) declares no ndm and archives no per-node
ndf or element classes, so the classic-recorder route reads them from the
live domain: ``getNDM()``, ``getNDF(node)`` for each recorded node, and
``eleType`` / ``eleNodes`` for the rotation roles. Nothing is inferred
from the FEMData.

The fake build here has no MPCO recorder and answers those queries from a
table. On ``remove`` it writes each node recorder's file from the formula
:func:`_value`, so the read is checked against a closed form. The real
domain is the live-stock case in
``tests/opensees/live/test_femdata_archive_live.py``.

The oracles:

* the ``-dof`` list follows the live ndm and per-node ndf: a planar frame
  node's ``rotation_z`` is DOF 3 (DOF 6 on a 3-D frame node), whatever the
  FEMData's coordinates say;
* a FEMData archive and the bridge's archive of the same deck send the
  same recorder commands, and the bridge archive never asks the live
  domain;
* each refusal names what it refuses.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.results import Results
from apeGmsh.results.live import DefaultRouteWarning
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.test_results_live_default_format import truss_bridge, truss_fem

_STEPS = 3
_DISP3 = ("displacement_x", "displacement_y", "displacement_z")
_ROT3 = ("rotation_x", "rotation_y", "rotation_z")


def _value(step: int, node: int, dof: int) -> float:
    """What the fake build records at ``step`` for ``node``'s DOF ``dof``."""
    return step * (10.0 * node + dof)


class LiveOps:
    """A pip build: no MPCO, a live domain answered from tables.

    ``ndm``; ``ndf`` ``{node: ndf}``; ``elements`` ``{tag: (class, nodes)}``;
    ``coords`` ``{node: xyz}`` (default: the frame FEMData's first ``ndm``,
    where a deck built from it places them). Each live query is counted in
    ``asked``.
    """

    def __init__(self, ndm: int, ndf: "dict[int, int]",
                 elements: "dict[int, tuple[str, tuple[int, ...]]]",
                 coords: "dict[int, tuple[float, ...]] | None" = None) -> None:
        self.ndm, self.ndf, self.elements = ndm, ndf, elements
        self.coords = ({n: tuple(_FRAME_XYZ[n][:ndm]) for n in ndf
                        if n in _FRAME_XYZ} if coords is None else coords)
        self.asked: list[str] = []
        self.recorder_calls: list[tuple] = []
        self._live: dict[int, tuple] = {}

    # -- the live domain ------------------------------------------------
    def getNDM(self) -> list:
        self.asked.append("getNDM")
        return [self.ndm]

    def getNodeTags(self) -> list:
        self.asked.append("getNodeTags")
        return list(self.ndf)

    def getNDF(self, node: int) -> list:
        self.asked.append(f"getNDF:{node}")
        return [self.ndf[node]]

    def nodeCoord(self, node: int) -> list:
        self.asked.append(f"nodeCoord:{node}")
        return list(self.coords[node])

    def getEleTags(self) -> list:
        self.asked.append("getEleTags")
        return list(self.elements)

    def eleType(self, tag: int) -> str:
        self.asked.append("eleType")
        return self.elements[tag][0]

    def eleNodes(self, tag: int) -> list:
        self.asked.append("eleNodes")
        return list(self.elements[tag][1])

    # -- recorders ------------------------------------------------------
    def recorder(self, *args: Any) -> int:
        self.recorder_calls.append(args)
        if args[0] == "mpco":
            raise RuntimeError("recorder mpco: unknown recorder type")
        tag = len(self.recorder_calls)
        self._live[tag] = args
        return tag

    def remove(self, what: str, tag: int) -> None:
        assert what == "recorder"
        args = list(self._live.pop(tag))
        path = Path(args[args.index("-file") + 1])
        i, j = args.index("-node"), args.index("-dof")
        nodes = [int(n) for n in args[i + 1:j]]
        k = j + 1
        dofs: list[int] = []
        while isinstance(args[k], int):
            dofs.append(args[k])
            k += 1
        rows = [[float(s)] + [_value(s, n, d) for n in nodes for d in dofs]
                for s in range(1, _STEPS + 1)]
        np.savetxt(path, np.asarray(rows))


class StrictOps(LiveOps):
    """A build whose live domain must not be asked: the bridge archive's
    facts are its own."""

    def _never(self, *_: Any) -> list:
        raise AssertionError("the bridge archive asked the live domain")

    getNDM = getNDF = getNodeTags = nodeCoord = _never  # type: ignore[assignment]
    getEleTags = eleType = eleNodes = _never  # type: ignore[assignment]


#: The frame FEMData's node coordinates.
_FRAME_XYZ = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0),
              3: (0.0, 1.0, 1.0), 4: (1.0, 1.0, 1.0)}


def _frame_fem() -> FEMData:
    """Nodes 1-4 at 3-D coordinates, one line element 3-4: the FEMData
    says nothing about ndm or ndf."""
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    coords = np.array([_FRAME_XYZ[n] for n in (1, 2, 3, 4)])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=1)
    group = ElementGroup(element_type=info, ids=np.array([1], dtype=np.int64),
                         connectivity=np.array([[3, 4]], dtype=np.int64))
    pg = {(1, 1): {"name": "Beam", "node_ids": node_ids[2:],
                   "node_coords": coords[2:], "element_ids": group.ids}}
    return FEMData(
        nodes=NodeComposite(node_ids=node_ids, node_coords=coords,
                            physical=PhysicalGroupSet(pg), labels=LabelSet({})),
        elements=ElementComposite(groups={1: group},
                                  physical=PhysicalGroupSet(pg),
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=4, n_elems=1, bandwidth=2, types=[info]),
        model_name="femdata_archive_frame",
    )


@pytest.fixture(scope="module")
def frame(tmp_path_factory: pytest.TempPathFactory) -> "tuple[FEMData, Path]":
    fem = _frame_fem()
    path = tmp_path_factory.mktemp("femdata_archive") / "fem.h5"
    fem.to_h5(str(path))
    return fem, path


def _spec(fem: FEMData, *records: "tuple[str, str, tuple, list]") -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=tuple(
            ResolvedRecorderRecord(
                category=cat, name=name, components=comps, dt=None,
                n_steps=None,
                **({"node_ids": np.array(ids)} if cat == "nodes"
                   else {"element_ids": np.array(ids)}))
            for cat, name, comps, ids in records),
    )


def _run(spec: ResolvedRecorderSpec, path: Path, model_h5: Path,
         ops: LiveOps) -> "tuple[Any, list]":
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_default(path, model_h5=model_h5, ops=ops) as run:
            pass
    return run, caught


def _node_calls(ops: LiveOps) -> "list[tuple]":
    return [c for c in ops.recorder_calls if c[0] == "Node"]


def _dofs(call: tuple) -> "list[int]":
    args = list(call)
    out, k = [], args.index("-dof") + 1
    while isinstance(args[k], int):
        out.append(args[k])
        k += 1
    return out


#: A 3-D domain: nodes 1-2 carry ndf 3 (a tet's), nodes 3-4 ndf 6 joined
#: by a 3-D elastic beam.
_MIXED_3D = dict(ndm=3, ndf={1: 3, 2: 3, 3: 6, 4: 6},
                 elements={7: ("FourNodeTetrahedron", (1, 2, 1, 2)),
                           8: ("ElasticBeam3d", (3, 4))})


# ---------------------------------------------------------------------------
# The DOF facts come from the live domain
# ---------------------------------------------------------------------------


def test_mixed_ndf_records_and_reads_each_node_at_its_own_dofs(
    frame, tmp_path,
) -> None:
    fem, model_h5 = frame
    ops = LiveOps(**_MIXED_3D)  # type: ignore[arg-type]
    spec = _spec(fem, ("nodes", "all", _DISP3, [1, 2, 3, 4]),
                 ("nodes", "rot", _ROT3, [3, 4]))
    run, caught = _run(spec, tmp_path / "run", model_h5, ops)
    assert run.route == "recorders"
    assert [w.category for w in caught] == [DefaultRouteWarning]
    assert [_dofs(c) for c in _node_calls(ops)] == [[1, 2, 3], [4, 5, 6]]
    assert {a for a in ops.asked if a.startswith("getNDF")} == {
        f"getNDF:{n}" for n in (1, 2, 3, 4)}          # the recorded nodes only
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        for comps, ids, dofs in ((_DISP3, [1, 2, 3, 4], (1, 2, 3)),
                                 (_ROT3, [3, 4], (4, 5, 6))):
            for comp, dof in zip(comps, dofs):
                slab = res.nodes.get(ids=np.array(ids), component=comp)
                want = np.array([[_value(s, int(n), dof)
                                  for n in slab.node_ids]
                                 for s in range(1, _STEPS + 1)])
                np.testing.assert_array_equal(np.asarray(slab.values), want)
    finally:
        res.close()


def test_a_planar_frame_rotation_is_dof_3_from_the_live_ndm(frame, tmp_path) -> None:
    """The FEMData's coordinates are 3-D; the live domain is planar. A 3-D
    reading would send ``rotation_z`` to DOF 6, which OpenSees answers
    with 0."""
    fem, model_h5 = frame
    ops = LiveOps(ndm=2, ndf={3: 3, 4: 3},
                  elements={8: ("ElasticBeam2d", (3, 4))})
    spec = _spec(fem, ("nodes", "tip", ("displacement_y", "rotation_z"), [3, 4]))
    run, _ = _run(spec, tmp_path / "run", model_h5, ops)
    assert [_dofs(c) for c in _node_calls(ops)] == [[2, 3]]
    res = run.results(cache_root=tmp_path / "cache")
    try:
        slab = res.nodes.get(ids=np.array([4]), component="rotation_z")
        np.testing.assert_array_equal(
            np.asarray(slab.values)[:, 0],
            [_value(s, 4, 3) for s in range(1, _STEPS + 1)])
    finally:
        res.close()


def test_the_read_embeds_no_opensees_zone(frame, tmp_path) -> None:
    fem, model_h5 = frame
    ops = LiveOps(**_MIXED_3D)  # type: ignore[arg-type]
    run, _ = _run(_spec(fem, ("nodes", "all", _DISP3, [1, 2])),
                  tmp_path / "run", model_h5, ops)
    res = run.results(cache_root=tmp_path / "cache")
    try:
        import h5py
        (cached,) = [f for f in (tmp_path / "cache").glob("*.h5")
                     if not f.name.endswith(".model.h5")]
        with h5py.File(str(cached), "r") as f:
            assert "opensees" not in f and "model" in f
    finally:
        res.close()


# ---------------------------------------------------------------------------
# Refusals, by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndf, elements, ids, match", [
    ({1: 3, 3: 6}, {8: ("ElasticBeam3d", (3, 1))}, [1, 3],
     r"nodes \[1\] carry no 'rx' DOF in the live domain"),
    ({3: 6, 4: 6}, {8: ("Truss", (3, 4))}, [3],
     r"'rx' has no established DOF at node 3 \(no beam-column or shell"),
    ({3: 6, 4: 6}, {8: ("ZeroLengthContact3D", (3, 4))}, [3],
     r"node 3 \(a ZeroLengthContact3D touches it"),
], ids=["layout-lacks-it", "truss-only", "unknown-class"])
def test_a_rotation_the_live_domain_cannot_settle_refuses_by_name(
    frame, tmp_path, ndf, elements, ids, match,
) -> None:
    fem, model_h5 = frame
    ops = LiveOps(ndm=3, ndf=ndf, elements=elements)
    spec = _spec(fem, ("nodes", "rot", ("rotation_x",), ids))
    with pytest.raises(ValueError, match=match):
        _run(spec, tmp_path / "run", model_h5, ops)
    assert _node_calls(ops) == []


def test_a_node_the_live_domain_lacks_refuses(frame, tmp_path) -> None:
    fem, model_h5 = frame
    ops = LiveOps(ndm=3, ndf={1: 3}, elements={})
    with pytest.raises(ValueError, match=r"nodes \[2\] are not in the live domain"):
        _run(_spec(fem, ("nodes", "all", _DISP3, [1, 2])),
             tmp_path / "run", model_h5, ops)


def test_element_records_refuse_by_name(frame, tmp_path) -> None:
    fem, model_h5 = frame
    ops = LiveOps(**_MIXED_3D)  # type: ignore[arg-type]
    spec = _spec(fem, ("nodes", "all", _DISP3, [1, 2]),
                 ("elements", "beam", ("axial_force",), [1]))
    with pytest.raises(ValueError, match=r"records \['beam'\] name elements by "
                                         r"FEM id.*FEMData archive"):
        _run(spec, tmp_path / "run", model_h5, ops)
    assert _node_calls(ops) == []


def test_no_live_domain_keeps_the_archive_refusal(frame, tmp_path) -> None:
    """``getNDM()`` answers 0 before any ``ops.model``: nothing to read, so
    the archive's own refusal stands, word for word, ahead of the element
    refusal."""
    fem, model_h5 = frame
    ops = LiveOps(ndm=0, ndf={}, elements={})
    with pytest.raises(ValueError, match=r"model= declares no ndm \(ndm=0\)"):
        _run(_spec(fem, ("nodes", "all", _DISP3, [1, 2]),
                   ("elements", "beam", ("axial_force",), [1])),
             tmp_path / "run", model_h5, ops)


@pytest.mark.parametrize("coords, match", [
    # A planar deck that maps the frame's (y, z) to (x, y): node 3 at
    # (0, 1, 1) placed at (1, 1) would record its z motion as
    # displacement_y.
    ({3: (1.0, 1.0), 4: (1.0, 1.0)},
     r"not the FEMData's first 2 at node 3: live \[1\.0, 1\.0\], "
     r"FEM \[0\.0, 1\.0\]"),
    # A deck built from another FEMData whose node tags overlap.
    ({3: (5.0, 1.0), 4: (6.0, 1.0)},
     r"node 3: live \[5\.0, 1\.0\].*node 4: live \[6\.0, 1\.0\]"),
], ids=["axis-swap", "foreign-femdata"])
def test_a_live_node_that_is_not_the_fem_node_refuses_by_id(
    frame, tmp_path, coords, match,
) -> None:
    fem, model_h5 = frame
    ops = LiveOps(ndm=2, ndf={3: 3, 4: 3},
                  elements={8: ("ElasticBeam2d", (3, 4))}, coords=coords)
    with pytest.raises(ValueError, match=match):
        _run(_spec(fem, ("nodes", "tip", ("displacement_x", "displacement_y"),
                         [3, 4])), tmp_path / "run", model_h5, ops)
    assert _node_calls(ops) == []


def test_coordinates_within_the_tolerance_record(frame, tmp_path) -> None:
    """The tolerance is ``1e-9`` of the largest coordinate: a last-bit
    difference still records."""
    fem, model_h5 = frame
    ops = LiveOps(ndm=2, ndf={3: 3, 4: 3},
                  elements={8: ("ElasticBeam2d", (3, 4))},
                  coords={3: (0.0, 1.0 + 1e-12), 4: (1.0 - 1e-12, 1.0)})
    _run(_spec(fem, ("nodes", "tip", ("displacement_x",), [3, 4])),
         tmp_path / "run", model_h5, ops)
    assert [_dofs(c) for c in _node_calls(ops)] == [[1]]


def test_an_offline_read_of_a_femdata_archive_keeps_refusing(frame, tmp_path) -> None:
    fem, model_h5 = frame
    spec = _spec(fem, ("nodes", "all", _DISP3, [1, 2]))
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match=r"model\.h5 has no declared ndm"):
        Results.from_recorders(spec, tmp_path / "out", fem=fem,
                               model=OpenSeesModel.from_h5(model_h5),
                               cache_root=tmp_path / "cache")


# ---------------------------------------------------------------------------
# A bridge archive is unchanged
# ---------------------------------------------------------------------------


def test_a_bridge_archive_never_asks_the_live_domain_and_matches(tmp_path) -> None:
    """The truss deck through its bridge archive and through its FEMData
    archive: the same recorder commands, and only the FEMData archive
    reads the live domain."""
    fem = truss_fem()
    bridge_h5, fem_h5 = tmp_path / "bridge.h5", tmp_path / "fem.h5"
    truss_bridge(fem).h5(str(bridge_h5))
    fem.to_h5(str(fem_h5))
    spec = _spec(fem, ("nodes", "tip",
                       ("displacement_x", "displacement_y", "reaction_force_x"),
                       [1, 2]))
    table: dict[str, Any] = dict(ndm=2, ndf={1: 2, 2: 2},
                                 elements={1: ("Truss", (1, 2))},
                                 coords={1: (0.0, 0.0), 2: (2.0, 0.0)})
    strict, live = StrictOps(**table), LiveOps(**table)
    _run(spec, tmp_path / "a" / "run", bridge_h5, strict)
    _run(spec, tmp_path / "b" / "run", fem_h5, live)

    def shape(ops: LiveOps) -> "list[tuple]":
        return [tuple(Path(a).name if i == 2 else a for i, a in enumerate(c))
                for c in _node_calls(ops)]

    assert shape(strict) == shape(live) and shape(live)
    assert "getNDM" in live.asked
