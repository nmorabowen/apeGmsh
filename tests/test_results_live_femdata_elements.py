"""``spec.emit_default`` over a FEMData archive records ELEMENT results
from a live pairing (#2269).

A FEMData archive pairs no FEM element id with an OpenSees tag, and
identity is never assumed. The classic-recorder route pairs each recorded
FEM element with the live element of its own id only when the live domain
shows it: ``eleNodes`` equals the FEM connectivity in order, the element's
nodes pass the node identity check, and ``eleType`` has a bridge token
(``_TOKEN_FOR_LIVE_CLASS``). The read carries that pairing through
``RecorderTranscoder(pairing=)``.

The fake build here (``LiveOps`` of the #2251 tests) has no MPCO recorder
and answers the live queries from a table; on ``remove`` it writes each
element recorder's file from the formula :func:`_value`, so the read is
checked against a closed form. The real domain is the live-stock case in
``tests/opensees/live/test_femdata_elements_live.py``.

The oracles:

* a paired Gauss record reads the closed form under its FEM ids, and the
  file is stamped as FEM-indexed;
* each unproved id refuses by id, an unmapped class by name, and nothing
  reaches the domain then;
* the pairing's type token gates an elastic beam's ``localForce``
  stations as the bridge model's does.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
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
from apeGmsh.opensees._internal.typed_records import ElementRecord
from apeGmsh.results.live import DefaultRouteWarning
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.transcoders import RecorderTranscoder
from tests.test_results_live_femdata_archive import _STEPS, LiveOps

_STRESS = ("stress_xx", "stress_yy", "stress_zz",
           "stress_xy", "stress_yz", "stress_xz")
_XYZ = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0),
        4: (0.0, 0.0, 1.0), 5: (1.0, 1.0, 1.0)}
_CONN = {1: (1, 2, 3, 4), 2: (2, 3, 4, 5)}


def _value(step: int, eid: int, k: int) -> float:
    """What the fake build records at ``step`` for element ``eid``'s
    column ``k``."""
    return step * (100.0 * eid + k + 1)


class ElementOps(LiveOps):
    """``LiveOps`` that also writes each ``Element`` recorder's file:
    ``columns`` per element (six: a tet's one Gauss point)."""

    columns = 6

    def remove(self, what: str, tag: int) -> None:
        args = list(self._live[tag])
        if args[0] != "Element":
            return super().remove(what, tag)
        del self._live[tag]
        path = Path(args[args.index("-file") + 1])
        k = args.index("-ele") + 1
        eles: list[int] = []
        while isinstance(args[k], int):
            eles.append(args[k])
            k += 1
        rows = [[float(s)] + [_value(s, e, c) for e in eles
                              for c in range(self.columns)]
                for s in range(1, _STEPS + 1)]
        np.savetxt(path, np.asarray(rows))


def _tets_fem() -> FEMData:
    node_ids = np.arange(1, 6, dtype=np.int64)
    coords = np.array([_XYZ[n] for n in node_ids.tolist()])
    tet = make_type_info(code=4, gmsh_name="Tetrahedron 4", dim=3, order=1,
                         npe=4, count=2)
    group = ElementGroup(element_type=tet, ids=np.array([1, 2], dtype=np.int64),
                         connectivity=np.array([_CONN[1], _CONN[2]],
                                               dtype=np.int64))
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=node_ids, node_coords=coords,
                            physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={4: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=5, n_elems=2, bandwidth=4, types=[tet]),
        model_name="femdata_elements_tets",
    )


@pytest.fixture(scope="module")
def tets(tmp_path_factory: pytest.TempPathFactory) -> "tuple[FEMData, Path]":
    fem = _tets_fem()
    path = tmp_path_factory.mktemp("femdata_elements") / "fem.h5"
    fem.to_h5(str(path))
    return fem, path


def _ops(elements: "dict[int, tuple[str, tuple[int, ...]]] | None" = None,
         coords: "dict[int, tuple[float, ...]] | None" = None) -> ElementOps:
    return ElementOps(
        ndm=3, ndf={n: 3 for n in _XYZ},
        elements=({e: ("FourNodeTetrahedron", c) for e, c in _CONN.items()}
                  if elements is None else elements),
        coords=dict(_XYZ) if coords is None else coords)


def _spec(fem: FEMData, *records: ResolvedRecorderRecord) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=records)


def _gauss(ids: "list[int]", cls: "str | None" = "FourNodeTetrahedron",
           category: str = "gauss") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category=category, name="sigma", components=_STRESS, dt=None,
        n_steps=None, element_ids=np.array(ids), element_class_name=cls)


def _run(spec: ResolvedRecorderSpec, path: Path, model_h5: Path,
         ops: LiveOps) -> Any:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_default(path, model_h5=model_h5, ops=ops) as run:
            pass
    assert [w.category for w in caught] == [DefaultRouteWarning], [
        str(w.message) for w in caught]
    return run


def _element_calls(ops: LiveOps) -> "list[tuple]":
    return [c for c in ops.recorder_calls if c[0] == "Element"]


# ---------------------------------------------------------------------------
# A proved pairing records and reads
# ---------------------------------------------------------------------------


def test_a_paired_gauss_record_reads_under_its_fem_ids(tets, tmp_path) -> None:
    fem, model_h5 = tets
    ops = _ops()
    run = _run(_spec(fem, _gauss([2, 1])), tmp_path / "run", model_h5, ops)
    (call,) = _element_calls(ops)
    k = call.index("-ele") + 1
    assert list(call[k:k + 2]) == [2, 1]               # the FEM ids, proved
    assert ops.asked.count("eleNodes") == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        for c, comp in enumerate(_STRESS):
            slab = res.elements.gauss.get(component=comp, ids=np.array([1, 2]))
            index = [int(i) for i in np.asarray(slab.element_index)]
            want = np.array([[_value(s, e, c) for e in index]
                             for s in range(1, _STEPS + 1)])
            assert sorted(index) == [1, 2]
            np.testing.assert_array_equal(np.asarray(slab.values), want)
    finally:
        res.close()
    (cached,) = [f for f in (tmp_path / "cache").glob("*.h5")
                 if not f.name.endswith(".model.h5")]
    with h5py.File(str(cached), "r") as f:
        assert f.attrs["element_index_space"] == "fem"
        assert "opensees" not in f


# ---------------------------------------------------------------------------
# Refusals: by id, by name, before any recorder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elements, coords, match", [
    ({1: ("FourNodeTetrahedron", _CONN[1])}, None,
     r"FEM element 2 has nodes \[2, 3, 4, 5\], no live element has that tag"),
    ({1: ("FourNodeTetrahedron", _CONN[1]),
      2: ("FourNodeTetrahedron", (2, 4, 3, 5))}, None,
     r"FEM element 2 has nodes \[2, 3, 4, 5\], the live element 2 has "
     r"\[2, 4, 3, 5\]"),
    (None, {**_XYZ, 5: (2.0, 2.0, 2.0)},
     r"node 5: live \[2\.0, 2\.0, 2\.0\], FEM \[1\.0, 1\.0, 1\.0\]"),
    ({1: ("FourNodeTetrahedron", _CONN[1]), 2: ("SSPtetra", _CONN[2])}, None,
     r"live element classes with no bridge element token \('SSPtetra': "
     r"FEM elements \[2\]\)"),
], ids=["no-live-element", "reordered-nodes", "foreign-node", "unmapped-class"])
def test_an_unproved_element_refuses_and_records_nothing(
    tets, tmp_path, elements, coords, match,
) -> None:
    fem, model_h5 = tets
    ops = _ops(elements, coords)
    with pytest.raises(ValueError, match=match):
        _run(_spec(fem, _gauss([1, 2], cls=None)), tmp_path / "run",
             model_h5, ops)
    assert ops.recorder_calls[1:] == []                # only the MPCO probe


def _twin_fem() -> FEMData:
    """Tets 1 and 3 on the same ordered nodes, tet 2 apart."""
    fem = _tets_fem()
    group = next(iter(fem.elements))
    tet = group.element_type
    twin = ElementGroup(element_type=tet,
                        ids=np.array([1, 2, 3], dtype=np.int64),
                        connectivity=np.array([_CONN[1], _CONN[2], _CONN[1]],
                                              dtype=np.int64))
    pg = PhysicalGroupSet({})
    return FEMData(nodes=fem.nodes,
                   elements=ElementComposite(groups={4: twin}, physical=pg,
                                             labels=LabelSet({})),
                   info=MeshInfo(n_nodes=5, n_elems=3, bandwidth=4, types=[tet]),
                   model_name="femdata_elements_twins")


@pytest.mark.parametrize("ids, first, other", [([1], 1, 3), ([2, 3], 3, 1)],
                         ids=["first", "second"])
def test_elements_sharing_ordered_nodes_refuse_naming_both(
    tmp_path, ids, first, other,
) -> None:
    """eleNodes cannot tell tets 1 and 3 apart: a deck that swapped their
    tags (and materials) would record each under the other's id."""
    fem = _twin_fem()
    model_h5 = tmp_path / "fem.h5"
    fem.to_h5(str(model_h5))
    ops = _ops({1: ("FourNodeTetrahedron", _CONN[1]),
                2: ("FourNodeTetrahedron", _CONN[2]),
                3: ("FourNodeTetrahedron", _CONN[1])})
    with pytest.raises(ValueError, match=(
            rf"FEM element {first} shares nodes \[1, 2, 3, 4\] with "
            rf"\[{other}\].*eleNodes cannot tell those elements apart")):
        _run(_spec(fem, _gauss(ids)), tmp_path / "run", model_h5, ops)
    assert ops.recorder_calls[1:] == [] and "eleNodes" not in ops.asked


def test_a_lone_element_beside_twins_records(tmp_path) -> None:
    fem = _twin_fem()
    model_h5 = tmp_path / "fem.h5"
    fem.to_h5(str(model_h5))
    ops = _ops({1: ("FourNodeTetrahedron", _CONN[1]),
                2: ("FourNodeTetrahedron", _CONN[2]),
                3: ("FourNodeTetrahedron", _CONN[1])})
    _run(_spec(fem, _gauss([2])), tmp_path / "run", model_h5, ops)
    assert len(_element_calls(ops)) == 1


@pytest.mark.parametrize("cls, match", [
    (None, r"elements record 'nf' names no element class"),
    ("FourNodeTetrahedron",
     r"no nodal-force layout \[\('FourNodeTetrahedron', 'global_force'\)\]"),
], ids=["no-class", "uncatalogued-class"])
def test_an_undecodable_nodal_force_record_refuses_before_the_run(
    tets, tmp_path, cls, match,
) -> None:
    fem, model_h5 = tets
    ops = _ops()
    rec = ResolvedRecorderRecord(
        category="elements", name="nf",
        components=("nodal_resisting_force_x",), dt=None, n_steps=None,
        element_ids=np.array([1]), element_class_name=cls)
    with pytest.raises(ValueError, match=match):
        _run(_spec(fem, rec), tmp_path / "run", model_h5, ops)
    assert _element_calls(ops) == []


def test_a_record_class_that_is_not_the_live_class_refuses(tets, tmp_path) -> None:
    fem, model_h5 = tets
    ops = _ops()
    with pytest.raises(ValueError, match=r"record 'sigma' names element class "
                                         r"'TenNodeTetrahedron', but the live "
                                         r"domain has FEM elements 1 as "
                                         r"'FourNodeTetrahedron'"):
        _run(_spec(fem, _gauss([1, 2], cls="TenNodeTetrahedron")),
             tmp_path / "run", model_h5, ops)
    assert _element_calls(ops) == []


def test_an_id_the_femdata_lacks_refuses_by_id(tets, tmp_path) -> None:
    fem, model_h5 = tets
    with pytest.raises(ValueError, match=r"FEM elements \[9\] are not in the "
                                         r"FEMData"):
        _run(_spec(fem, _gauss([1, 9])), tmp_path / "run", model_h5, _ops())


@pytest.mark.parametrize("category, comps, cls", [
    ("fibers", ("fiber_stress",), "ForceBeamColumn3d"),
    ("gauss", ("damage_tension", "damage_compression"), "Brick"),
], ids=["fibers", "per-material-gauss"])
def test_a_record_that_needs_the_bridge_archive_refuses_by_name(
    tets, tmp_path, category, comps, cls,
) -> None:
    """A fibres record takes each element's ``beamIntegration`` from the
    model archive, a per-Gauss-point material record its pairing: a
    FEMData archive holds neither."""
    fem, model_h5 = tets
    ops = _ops()
    rec = ResolvedRecorderRecord(
        category=category, name="need", components=comps, dt=None,
        n_steps=None, element_ids=np.array([1]), element_class_name=cls)
    with pytest.raises(ValueError, match=r"records \['need'\] read each "
                                         r"element's integration rule"):
        _run(_spec(fem, rec), tmp_path / "run", model_h5, ops)
    assert "eleNodes" not in ops.asked and ops.recorder_calls[1:] == []


# ---------------------------------------------------------------------------
# The transcoder carries a pairing in hand
# ---------------------------------------------------------------------------


class _Pairing:
    """A pairing in hand: the archive's lineage and typed element rows."""

    def __init__(self, fem: FEMData, tokens: "dict[int, str]") -> None:
        self.lineage = type("L", (), {"fem_hash": fem.snapshot_id,
                                      "model_hash": None})()
        self._rows = tuple(ElementRecord(type_token=t, tag=e, args=(),
                                         connectivity=(), fem_eid=e)
                           for e, t in tokens.items())

    def elements(self) -> "tuple[ElementRecord, ...]":
        return self._rows


def test_the_pairing_type_token_gates_elastic_local_force(tets, tmp_path) -> None:
    """An elastic-beam ``localForce`` record over an element the pairing
    types as a force beam refuses by FEM id, as over the bridge's model."""
    fem, _ = tets
    rec = ResolvedRecorderRecord(
        category="line_stations", name="beam",
        components=("axial_force", "shear_y", "bending_moment_z"), dt=None,
        n_steps=None, element_ids=np.array([1]),
        element_class_name="ElasticBeam2d")
    spec = _spec(fem, rec)
    tr = RecorderTranscoder(spec, tmp_path, tmp_path / "r.h5", fem,
                            pairing=_Pairing(fem, {1: "forceBeamColumn"}))
    with pytest.raises(ValueError, match=r"FEM element\(s\) \[1\] are "
                                         r"\['forceBeamColumn'\] in model="):
        tr.run()


def test_a_pairing_and_a_model_archive_are_exclusive(tets, tmp_path) -> None:
    fem, model_h5 = tets
    with pytest.raises(TypeError, match=r"not both"):
        RecorderTranscoder(_spec(fem), tmp_path, tmp_path / "r.h5", fem,
                           model_h5_src=model_h5, pairing=_Pairing(fem, {}))


# ---------------------------------------------------------------------------
# A record MPCO cannot write takes the classic recorders (#2290 D)
# ---------------------------------------------------------------------------


class MpcoTrussOps(ElementOps):
    """A build that has an MPCO recorder (its probe would return a tag),
    recording one axial-force column per truss."""

    columns = 1

    def recorder(self, *args: Any) -> int:
        if args[0] != "mpco":
            return super().recorder(*args)
        self.recorder_calls.append(args)
        return len(self.recorder_calls)


def test_a_truss_axial_force_record_skips_mpco_for_the_classic_recorders(
    tmp_path,
) -> None:
    """MPCO writes the Truss ``axialForce`` bucket with no ``GAUSS_IDS``, so
    ``mpco_ops_args`` refuses the record. The route takes the classic
    recorders without probing MPCO, even on a build that has it, and the
    read gives the file's axial force at the truss's one Gauss point."""
    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=2)
    group = ElementGroup(element_type=line, ids=np.array([1, 2], dtype=np.int64),
                         connectivity=np.array([[1, 2], [2, 5]], dtype=np.int64))
    pg = PhysicalGroupSet({})
    fem = FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 6, dtype=np.int64),
                            node_coords=np.array([_XYZ[n] for n in range(1, 6)]),
                            physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=5, n_elems=2, bandwidth=2, types=[line]),
        model_name="femdata_elements_trusses")
    model_h5 = tmp_path / "fem.h5"
    fem.to_h5(str(model_h5))
    ops = MpcoTrussOps(ndm=3, ndf={n: 3 for n in _XYZ},
                       elements={1: ("Truss", (1, 2)), 2: ("Truss", (2, 5))},
                       coords=dict(_XYZ))
    rec = ResolvedRecorderRecord(
        category="gauss", name="axial", components=("axial_force",), dt=None,
        n_steps=None, element_ids=np.array([2, 1]), element_class_name="Truss")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _spec(fem, rec).emit_default(tmp_path / "run", model_h5=model_h5,
                                          ops=ops) as run:
            pass
    assert run.route == "recorders"
    assert [c for c in ops.recorder_calls if c[0] == "mpco"] == []
    (w,) = caught
    assert w.category is DefaultRouteWarning
    assert "a record cannot go through MPCO (Record 'axial'" in str(w.message)
    assert not (tmp_path / "run.mpco").exists()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        slab = res.elements.gauss.get(component="axial_force",
                                      ids=np.array([1, 2]))
        index = [int(i) for i in np.asarray(slab.element_index)]
        assert sorted(index) == [1, 2]                 # one Gauss point each
        np.testing.assert_array_equal(np.asarray(slab.natural_coords), [[0.0]] * 2)
        np.testing.assert_array_equal(
            np.asarray(slab.values),
            [[_value(s, e, 0) for e in index] for s in range(1, _STEPS + 1)])
    finally:
        res.close()
