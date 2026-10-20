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
  stations as the bridge model's does;
* a per-Gauss-point material record reads through the pairing, file
  ``gp`` at catalog point ``gp - 1``; a fibres record still refuses by
  name (#2290).
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


def test_a_fibres_record_refuses_by_name(tets, tmp_path) -> None:
    """A fibres record takes each element's ``beamIntegration`` from the
    model archive, which a FEMData archive does not hold."""
    fem, model_h5 = tets
    ops = _ops()
    rec = ResolvedRecorderRecord(
        category="fibers", name="need", components=("fiber_stress",), dt=None,
        n_steps=None, element_ids=np.array([1]),
        element_class_name="ForceBeamColumn3d")
    with pytest.raises(ValueError, match=r"records \['need'\] read each "
                                         r"element's integration rule"):
        _run(_spec(fem, rec), tmp_path / "run", model_h5, ops)
    assert "eleNodes" not in ops.asked and ops.recorder_calls[1:] == []


# ---------------------------------------------------------------------------
# A per-Gauss-point material record reads through the pairing (#2290 A)
# ---------------------------------------------------------------------------

_HEX_XYZ = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (1.0, 1.0, 0.0),
            4: (0.0, 1.0, 0.0), 5: (0.0, 0.0, 1.0), 6: (1.0, 0.0, 1.0),
            7: (1.0, 1.0, 1.0), 8: (0.0, 1.0, 1.0)}
_DAMAGE = ("damage_tension", "damage_compression")


class DamageOps(ElementOps):
    """``ElementOps`` whose ``material <gp> damage`` files hold two columns
    per element, valued by their point: column ``k`` of point ``gp`` is
    ``_value(step, e, 10 * gp + k)``."""

    def remove(self, what: str, tag: int) -> None:
        args = list(self._live[tag])
        if args[0] != "Element" or "material" not in args:
            return super().remove(what, tag)
        del self._live[tag]
        gp = int(args[args.index("material") + 1])
        k = args.index("-ele") + 1
        eles: list[int] = []
        while isinstance(args[k], int):
            eles.append(args[k])
            k += 1
        rows = [[float(s)] + [_value(s, e, 10 * gp + c) for e in eles
                              for c in range(2)]
                for s in range(1, _STEPS + 1)]
        np.savetxt(Path(args[args.index("-file") + 1]), np.asarray(rows))


def _hex_fem(tmp: Path) -> "tuple[FEMData, Path]":
    hexa = make_type_info(code=5, gmsh_name="Hexahedron 8", dim=3, order=1,
                          npe=8, count=1)
    group = ElementGroup(element_type=hexa, ids=np.array([7], dtype=np.int64),
                         connectivity=np.array([list(_HEX_XYZ)], dtype=np.int64))
    pg = PhysicalGroupSet({})
    fem = FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 9, dtype=np.int64),
                            node_coords=np.array(list(_HEX_XYZ.values())),
                            physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={5: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=8, n_elems=1, bandwidth=8, types=[hexa]),
        model_name="femdata_elements_hex")
    path = tmp / "fem.h5"
    fem.to_h5(str(path))
    return fem, path


def _damage_rec(cls: str = "Brick") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="gauss", name="dmg", components=_DAMAGE, dt=None,
        n_steps=None, element_ids=np.array([7]), element_class_name=cls)


def _damage_ops() -> DamageOps:
    return DamageOps(ndm=3, ndf={n: 3 for n in _HEX_XYZ},
                     elements={7: ("Brick", tuple(_HEX_XYZ))},
                     coords=dict(_HEX_XYZ))


def test_a_per_material_damage_record_reads_through_the_pairing(tmp_path) -> None:
    """Eight ``material <gp> damage`` files of a live ``Brick`` read under
    its FEM id, file ``gp`` at the catalog's point ``gp - 1`` (Brick.cpp's
    order, ``k`` fastest)."""
    fem, model_h5 = _hex_fem(tmp_path)
    ops = _damage_ops()
    run = _run(_spec(fem, _damage_rec()), tmp_path / "run", model_h5, ops)
    calls = _element_calls(ops)
    assert [c[c.index("material") + 1] for c in calls] == [
        str(g) for g in range(1, 9)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        s = 1.0 / np.sqrt(3.0)
        for k, comp in enumerate(_DAMAGE):
            slab = res.elements.gauss.get(component=comp, ids=np.array([7]))
            assert [int(i) for i in np.asarray(slab.element_index)] == [7] * 8
            np.testing.assert_allclose(
                np.asarray(slab.natural_coords),
                [((-s, s)[g // 4], (-s, s)[(g // 2) % 2], (-s, s)[g % 2])
                 for g in range(8)], atol=1e-15)
            np.testing.assert_array_equal(
                np.asarray(slab.values),
                [[_value(st, 7, 10 * gp + k) for gp in range(1, 9)]
                 for st in range(1, _STEPS + 1)], err_msg=comp)
    finally:
        res.close()


def test_a_per_material_record_of_another_class_refuses(tmp_path) -> None:
    """The layout is the record's class's, so the live class gates it."""
    fem, model_h5 = _hex_fem(tmp_path)
    ops = _damage_ops()
    with pytest.raises(ValueError, match=r"record 'dmg' names element class "
                                         r"'LadrunoBrick', but the live "
                                         r"domain has FEM elements 7 as "
                                         r"'Brick'"):
        _run(_spec(fem, _damage_rec("LadrunoBrick")), tmp_path / "run",
             model_h5, ops)
    assert _element_calls(ops) == []


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


def test_a_fibres_record_over_a_pairing_refuses_by_name(tets, tmp_path) -> None:
    """A pairing carries no ``beamIntegration``, so a fibres record handed
    to the transcoder with one refuses by name, not on a missing method."""
    fem, _ = tets
    rec = ResolvedRecorderRecord(
        category="fibers", name="fib", components=("fiber_stress",), dt=None,
        n_steps=None, element_ids=np.array([1]),
        element_class_name="ForceBeamColumn3d")
    tr = RecorderTranscoder(_spec(fem, rec), tmp_path, tmp_path / "r.h5", fem,
                            pairing=_Pairing(fem, {1: "forceBeamColumn"}))
    with pytest.raises(ValueError, match=r"Record 'fib' \(category=fibers\): "
                                         r"its stations are counted from each "
                                         r"element's beamIntegration"):
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


def _truss_fem(tmp: Path) -> "tuple[FEMData, Path, MpcoTrussOps]":
    """Trusses 1 (nodes 1-2) and 2 (nodes 2-5), live as ``Truss``."""
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
    model_h5 = tmp / "fem.h5"
    fem.to_h5(str(model_h5))
    ops = MpcoTrussOps(ndm=3, ndf={n: 3 for n in _XYZ},
                       elements={1: ("Truss", (1, 2)), 2: ("Truss", (2, 5))},
                       coords=dict(_XYZ))
    return fem, model_h5, ops


def _truss_rec(components: "tuple[str, ...]" = ("axial_force",),
               name: str = "axial", category: str = "gauss",
               cls: str = "Truss") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category=category, name=name, components=components, dt=None,
        n_steps=None, element_ids=np.array([2, 1]), element_class_name=cls)


def test_a_truss_axial_force_record_skips_mpco_for_the_classic_recorders(
    tmp_path,
) -> None:
    """MPCO writes the Truss ``axialForce`` bucket with no ``GAUSS_IDS``, so
    the route takes the classic recorders without probing MPCO, even on a
    build that has it, naming the rerouted record. The read gives the
    file's axial force at the truss's one Gauss point."""
    fem, model_h5, ops = _truss_fem(tmp_path)
    rec = _truss_rec()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _spec(fem, rec).emit_default(tmp_path / "run", model_h5=model_h5,
                                          ops=ops) as run:
            pass
    assert run.route == "recorders"
    assert [c for c in ops.recorder_calls if c[0] == "mpco"] == []
    (w,) = caught
    assert w.category is DefaultRouteWarning
    assert str(w.message).startswith(
        "emit_default: MPCO cannot write the Truss axial force (records "
        "{'axial': ['axial_force']} record its axialForce bucket with no "
        "GAUSS_IDS, which the MPCO reader refuses); recording classic "
        "recorder files in ")
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


_NO_TOKEN = r"component 'displacement_x' has no MPCO -E token"
_DEFORMATION = (r"component 'axial_strain' cannot go through MPCO: the MPCO "
                r"reader does not decode its section.deformation bucket")


@pytest.mark.parametrize("records, match", [
    ((_truss_rec(("displacement_x",)),), _NO_TOKEN),
    ((_truss_rec(("axial_force", "displacement_x")),), _NO_TOKEN),
    ((_truss_rec(), _truss_rec(("displacement_x",), name="u")), _NO_TOKEN),
    ((_truss_rec(("axial_strain",), name="ls", category="line_stations",
                 cls="ForceBeamColumn3d"),), _DEFORMATION),
    ((_truss_rec(), _truss_rec(("axial_strain",), name="ls",
                               category="line_stations",
                               cls="ForceBeamColumn3d")), _DEFORMATION),
], ids=["no-token", "no-token-beside-axial", "no-token-record-beside-axial",
        "section-deformation", "section-deformation-beside-axial"])
def test_only_the_truss_axial_force_leaves_mpco(tmp_path, records, match) -> None:
    """Every other MPCO refusal stands at ``__enter__``, with or without a
    Truss axial force beside it: a gauss component with no ``-E`` token
    and a line-stations section deformation would read back empty on the
    classic recorders too."""
    fem, model_h5, ops = _truss_fem(tmp_path)
    with pytest.raises(ValueError, match=match):
        _run(_spec(fem, *records), tmp_path / "run", model_h5, ops)
    assert ops.recorder_calls == [] and "eleNodes" not in ops.asked


_TRUSS_AXIAL = (r"component 'axial_force' \(the Truss axial force\) cannot "
                r"go through MPCO")


@pytest.mark.parametrize("cls", ["ElasticBeam3d", None],
                         ids=["named", "class-less"])
def test_an_elastic_beam_axial_force_keeps_the_mpco_refusal(tmp_path, cls) -> None:
    """Only the Truss family's ``axial_force`` leaves MPCO. An elastic
    beam's, named or found by its live ``eleType``, has no Gauss
    ``axialForce`` the classic read decodes either: refused by name at
    ``__enter__``, as on main, not read empty after the run."""
    fem, model_h5, ops = _truss_fem(tmp_path)
    ops.elements = {1: ("ElasticBeam3d", (1, 2)), 2: ("ElasticBeam3d", (2, 5))}
    with pytest.raises(ValueError, match=_TRUSS_AXIAL):
        _run(_spec(fem, _truss_rec(cls=cls)), tmp_path / "run", model_h5, ops)
    assert ops.recorder_calls == [] and "eleNodes" not in ops.asked


def test_a_class_less_truss_axial_force_leaves_mpco_by_its_live_class(
    tmp_path,
) -> None:
    fem, model_h5, ops = _truss_fem(tmp_path)
    run = _run(_spec(fem, _truss_rec(cls=None)), tmp_path / "run", model_h5,
               ops)
    assert run.route == "recorders"
    assert [c for c in ops.recorder_calls if c[0] == "mpco"] == []
    assert "eleType" in ops.asked


def test_the_reroute_raises_no_mpco_line_stations_warning(tmp_path) -> None:
    """A class-less line-stations record beside the Truss axial force: the
    MPCO emit would warn that elastic beams may read empty, but the run
    never makes an MPCO recorder, so nothing warns but the route."""
    fem, model_h5, ops = _truss_fem(tmp_path)
    ls = _truss_rec(("axial_force",), name="ls", category="line_stations",
                    cls=None)
    run = _run(_spec(fem, _truss_rec(), ls), tmp_path / "run", model_h5, ops)
    assert run.route == "recorders"
