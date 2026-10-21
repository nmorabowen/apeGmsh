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
  ``gp`` at catalog point ``gp - 1`` (#2290 A);
* a fibres record reads through the section locations and fibre counts
  the live domain gave at ``__enter__``, each fibre table at its section;
  missing or inconsistent facts refuse by name (#2290 B); stations a
  long-hinge rule writes out of order read when they are the live
  sections, and a near miss or a swapped pair refuses by id (#2370);
* an ``elements`` record with no ``nodal_resisting_*`` component refuses
  at ``__enter__``.

"""
from __future__ import annotations

import dataclasses
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
from apeGmsh.results.spec._emit import emit_logical
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


def _twin_ops(twin: "tuple[str, tuple[int, ...]] | None") -> ElementOps:
    """Tets 1 and 2 live; FEM tet 3 (tet 1's twin) live as ``twin``."""
    live = {1: ("FourNodeTetrahedron", _CONN[1]),
            2: ("FourNodeTetrahedron", _CONN[2])}
    if twin is not None:
        live[3] = twin
    return _ops(live)


def test_a_twin_of_another_live_class_records_under_its_named_class(
    tmp_path,
) -> None:
    """The emit route passes each element's record classes to the pairing:
    tet 1's record names ``FourNodeTetrahedron`` and its twin 3 is live as
    another class, so 1 is told apart by class and reads under its id."""
    fem = _twin_fem()
    model_h5 = tmp_path / "fem.h5"
    fem.to_h5(str(model_h5))
    ops = _twin_ops(("TenNodeTetrahedron", _CONN[1]))
    run = _run(_spec(fem, _gauss([1])), tmp_path / "run", model_h5, ops)
    assert len(_element_calls(ops)) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        slab = res.elements.gauss.get(component="stress_xx")
        assert np.asarray(slab.element_index).tolist() == [1]
        np.testing.assert_allclose(np.asarray(slab.values)[:, 0],
                                   [_value(s, 1, 0) for s in range(1, _STEPS + 1)],
                                   rtol=1e-15, atol=0)
    finally:
        res.close()


@pytest.mark.parametrize("twin, records", [
    (("FourNodeTetrahedron", _CONN[1]), lambda: (_gauss([1]),)),
    (("TenNodeTetrahedron", _CONN[1]), lambda: (_gauss([1], cls=None),)),
    (("TenNodeTetrahedron", _CONN[1]),
     lambda: (_gauss([1]), dataclasses.replace(_gauss([1], cls=None),
                                               name="bare"))),
    (None, lambda: (_gauss([1]),)),
], ids=["equal-class", "no-class", "one-record-names-none", "twin-not-live"])
def test_twins_the_record_classes_cannot_tell_apart_refuse(
    tmp_path, twin, records,
) -> None:
    fem = _twin_fem()
    model_h5 = tmp_path / "fem.h5"
    fem.to_h5(str(model_h5))
    ops = _twin_ops(twin)
    with pytest.raises(ValueError, match=(
            r"FEM element 1 shares nodes \[1, 2, 3, 4\] with \[3\].*eleNodes "
            r"cannot tell those elements apart.* name the record's element "
            r"class where each twin is a live element of another class")):
        _run(_spec(fem, *records()), tmp_path / "run", model_h5, ops)
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


def test_an_elements_record_without_nodal_forces_refuses_before_the_run(
    tets, tmp_path,
) -> None:
    """An ``elements`` record is read as nodal forces only: one with no
    ``nodal_resisting_*`` component refuses by name at ``__enter__``, not
    after the run (#2290)."""
    fem, model_h5 = tets
    ops = _ops()
    rec = ResolvedRecorderRecord(
        category="elements", name="nf", components=("spring_force",),
        dt=None, n_steps=None, element_ids=np.array([1]),
        element_class_name="FourNodeTetrahedron")
    with pytest.raises(ValueError, match=r"elements record 'nf' has no "
                                         r"nodal_resisting_\* component "
                                         r"\(\['spring_force'\]\)"):
        _run(_spec(fem, rec), tmp_path / "run", model_h5, ops)
    assert _element_calls(ops) == []


# ---------------------------------------------------------------------------
# A fibres record reads through the live section facts (#2290 B)
# ---------------------------------------------------------------------------

#: Two 3-D beams on x: element 1 on nodes 1-2 (L = 2), element 2 on 2-3
#: (L = 3).
_BEAM_XYZ = {1: (0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0), 3: (5.0, 0.0, 0.0)}
_BEAM_CONN = {1: (1, 2), 2: (2, 3)}
_BEAM_L = {1: 2.0, 2: 3.0}
#: A 3-point Lobatto rule: the section locations as fractions of L.
_LOBATTO_3 = (0.0, 0.5, 1.0)
#: Each section's fibres: (y, z, area, matTag).
_FIBRES = ((-0.05, 0.0, 0.01, 7.0), (0.05, 0.0, 0.01, 7.0))


def _fib(step: int, eid: int, k: int, j: int) -> float:
    """The fake build's fibre ``j`` stress of element ``eid``'s section
    ``k`` (1-based) at ``step``; its strain is this over 1e4."""
    return step * (1000.0 * eid + 100.0 * k + 10.0 * j + 1.0)


class FiberOps(ElementOps):
    """``ElementOps`` whose beams answer ``sectionLocation`` and
    ``section k fiberData2`` from tables, and whose ``fiberData2`` and
    ``integrationPoints`` recorders write those facts' files.

    ``locations`` ``{tag: section locations}``; ``fibres`` ``{tag: fibre
    count per section}``; ``written`` and ``file_fibres`` override what
    its recorders write (the ``integrationPoints`` locations, the
    ``fiberData2`` fibre counts)."""

    def __init__(self, *, locations: "dict[int, tuple[float, ...] | None]",
                 fibres: "dict[int, tuple[int, ...]]",
                 written: "dict[int, tuple[float, ...]] | None" = None,
                 file_fibres: "dict[int, tuple[int, ...]] | None" = None,
                 cls: str = "ForceBeamColumn3d") -> None:
        super().__init__(ndm=3, ndf={n: 6 for n in _BEAM_XYZ},
                         elements={e: (cls, c) for e, c in _BEAM_CONN.items()},
                         coords=dict(_BEAM_XYZ))
        self.locations, self.fibres = locations, fibres
        self.written = dict(locations) if written is None else written
        self.file_fibres = fibres if file_fibres is None else file_fibres

    def sectionLocation(self, tag: int) -> "list[float] | None":
        self.asked.append(f"sectionLocation:{tag}")
        loc = self.locations[tag]
        return None if loc is None else list(loc)

    def eleResponse(self, tag: int, *args: Any) -> list:
        self.asked.append(f"eleResponse:{tag}:{args}")
        assert args[0] == "section" and args[2] == "fiberData2", args
        k = int(args[1])
        n = self.fibres[tag][k - 1] if k <= len(self.fibres[tag]) else 0
        return [0.0] * (6 * n)

    def remove(self, what: str, tag: int) -> None:
        args = list(self._live[tag])
        if args[0] != "Element" or not ({"fiberData2", "integrationPoints"}
                                        & set(args)):
            return super().remove(what, tag)
        del self._live[tag]
        path = Path(args[args.index("-file") + 1])
        k = args.index("-ele") + 1
        eles: list[int] = []
        while isinstance(args[k], int):
            eles.append(args[k])
            k += 1
        rows = []
        for s in range(1, _STEPS + 1):
            row = [float(s)]
            for e in eles:
                if "integrationPoints" in args:
                    row += list(self.written[e])
                    continue
                for sec, n in enumerate(self.file_fibres[e], start=1):
                    for j in range(n):
                        y, z, a, m = _FIBRES[j % len(_FIBRES)]
                        stress = _fib(s, e, sec, j)
                        row += [y, z, a, m, stress, stress / 1e4]
            rows.append(row)
        np.savetxt(path, np.asarray(rows), fmt="%.17g")


def _beam_fem(tmp: Path) -> "tuple[FEMData, Path]":
    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=2)
    group = ElementGroup(element_type=line, ids=np.array([1, 2], dtype=np.int64),
                         connectivity=np.array([_BEAM_CONN[1], _BEAM_CONN[2]],
                                               dtype=np.int64))
    pg = PhysicalGroupSet({})
    fem = FEMData(
        nodes=NodeComposite(node_ids=np.array([1, 2, 3], dtype=np.int64),
                            node_coords=np.array([_BEAM_XYZ[n] for n in (1, 2, 3)]),
                            physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=2, types=[line]),
        model_name="femdata_elements_fibre_beams")
    model_h5 = tmp / "fem.h5"
    fem.to_h5(str(model_h5))
    return fem, model_h5


def _lobatto(**kw: Any) -> FiberOps:
    loc = {e: tuple(f * L for f in _LOBATTO_3) for e, L in _BEAM_L.items()}
    return FiberOps(locations=kw.pop("locations", loc),
                    fibres=kw.pop("fibres", {1: (2, 2, 2), 2: (2, 2, 2)}), **kw)


def _fibres_rec(ids: "list[int]" = [2, 1]) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="fibers", name="fib", components=("fiber_stress", "fiber_strain"),
        dt=None, n_steps=None, element_ids=np.array(ids),
        element_class_name="ForceBeamColumn3d")


def test_a_fibres_record_reads_through_the_live_section_facts(tmp_path) -> None:
    """The section locations and fibre counts are read from the live domain
    at ``__enter__``; the read places each fibre table at its section, and
    gives the closed form under the FEM ids, station by station."""
    fem, model_h5 = _beam_fem(tmp_path)
    ops = _lobatto()
    run = _run(_spec(fem, _fibres_rec()), tmp_path / "run", model_h5, ops)
    asked = [a for a in ops.asked if a.startswith(("sectionLocation",
                                                   "eleResponse"))]
    assert asked[:4] == ["sectionLocation:2"] + [
        f"eleResponse:2:('section', {k}, 'fiberData2')" for k in (1, 2, 3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        for c, comp in enumerate(("fiber_stress", "fiber_strain")):
            slab = res.elements.fibers.get(component=comp)
            eids = np.asarray(slab.element_index)
            gps = np.asarray(slab.gp_index)
            np.testing.assert_array_equal(eids, np.repeat([2, 1], 6))
            np.testing.assert_array_equal(gps, np.tile(np.repeat([0, 1, 2], 2), 2))
            np.testing.assert_allclose(slab.station_natural_coord,
                                       np.tile(np.repeat([-1.0, 0.0, 1.0], 2), 2),
                                       rtol=0, atol=1e-15)
            np.testing.assert_array_equal(slab.y, np.tile([-0.05, 0.05], 6))
            np.testing.assert_array_equal(slab.material_tag, 7)
            j = np.tile([0, 1], 6)
            want = [[_fib(s, int(e), int(g) + 1, int(jj)) / (1e4 if c else 1.0)
                     for e, g, jj in zip(eids, gps, j)]
                    for s in range(1, _STEPS + 1)]
            np.testing.assert_allclose(np.asarray(slab.values), want,
                                       rtol=1e-15, atol=0)
    finally:
        res.close()


@pytest.mark.parametrize("ops, match", [
    (lambda: _lobatto(locations={1: (0.0, 1.0, 2.0), 2: None}),
     r"fibres record 'fib': FEM element 2 \('ForceBeamColumn3d'\) has no "
     r"sections \(sectionLocation answered None\)"),
    (lambda: _lobatto(fibres={1: (2, 2, 2), 2: (2, 0, 2)}),
     r"FEM element 2 answers fiberData2 of 0 values at section 2, not whole "
     r"fibres of 6"),
], ids=["no-sections", "no-fibre-section"])
def test_missing_live_section_facts_refuse_before_the_run(
    tmp_path, ops, match,
) -> None:
    fem, model_h5 = _beam_fem(tmp_path)
    live = ops()
    with pytest.raises(ValueError, match=match):
        _run(_spec(fem, _fibres_rec()), tmp_path / "run", model_h5, live)
    assert _element_calls(live) == []


def _read_refuses(tmp: Path, ops: FiberOps, match: str) -> None:
    fem, model_h5 = _beam_fem(tmp)
    run = _run(_spec(fem, _fibres_rec()), tmp / "run", model_h5, ops)
    with pytest.raises(ValueError, match=match):
        run.results(cache_root=tmp / "cache")


def test_unequal_live_station_counts_refuse_by_count(tmp_path) -> None:
    ops = _lobatto(locations={1: (0.0, 1.0, 2.0), 2: (0.0, 0.75, 2.25, 3.0)},
                   fibres={1: (2, 2, 2), 2: (2, 2, 2, 2)})
    _read_refuses(tmp_path, ops,
                  r"different integration-point counts in the live domain "
                  r"\(3 points: FEM elements \[1\]; 4 points: FEM elements \[2\]\)")


def test_unequal_live_fibre_counts_refuse_naming_the_sections(tmp_path) -> None:
    """Equal column totals hide unequal tables (2 + 4 + 2 beside 4 + 2 +
    2); the live counts name the sections."""
    ops = _lobatto(fibres={1: (2, 4, 2), 2: (4, 2, 2)})
    _read_refuses(tmp_path, ops,
                  r"its sections carry different fibre counts in the live "
                  r"domain \(2 fibres: sections \['2:2', '2:3', '1:1', '1:3'\]; "
                  r"4 fibres: sections \['2:1', '1:2'\]")


def test_file_stations_that_are_not_the_live_sections_refuse(tmp_path) -> None:
    """A file whose stations are not the sections the live domain gave
    (here element 2's file at Lobatto points of another length) would
    place its fibres at the wrong section: refused by FEM id."""
    loc = {e: tuple(f * L for f in _LOBATTO_3) for e, L in _BEAM_L.items()}
    ops = _lobatto(written={1: loc[1], 2: (0.0, 1.0, 3.0)})
    _read_refuses(tmp_path, ops,
                  r"FEM element 2 has stations \[-1\.0, -0\.333333, 1\.0\] in "
                  r"the file, but its live sections are at \[-1\.0, 0\.0, 1\.0\]")


#: Section locations as fractions of L, out of order as a long-hinge
#: HingeRadau writes them (lp_i + lp_j >= L/4).
_HINGE_FRAC = (0.0, 0.1, 0.3, 0.2, 0.9, 1.0)


def _hinge_ops(**kw: Any) -> FiberOps:
    return _lobatto(locations={e: tuple(f * L for f in _HINGE_FRAC)
                               for e, L in _BEAM_L.items()},
                    fibres={1: (2,) * 6, 2: (2,) * 6}, **kw)


def test_out_of_order_live_sections_read_as_written(tmp_path) -> None:
    """Live sections out of order (a long-hinge HingeRadau rule writes them
    so) are the file's stations: the live locations place each one, as
    they do a line-stations record (#2332), so each fibre table reads at
    the section that wrote it, under its station, with the closed form
    (#2370). The bridge route, which has no live locations, still refuses
    them with the HingeRadau message
    (``tests/test_results_out_fiber_material.py``)."""
    fem, model_h5 = _beam_fem(tmp_path)
    run = _run(_spec(fem, _fibres_rec()), tmp_path / "run", model_h5,
               _hinge_ops())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        for c, comp in enumerate(("fiber_stress", "fiber_strain")):
            slab = res.elements.fibers.get(component=comp)
            eids = np.asarray(slab.element_index)
            gps = np.asarray(slab.gp_index)
            np.testing.assert_array_equal(eids, np.repeat([2, 1], 12))
            np.testing.assert_array_equal(
                gps, np.tile(np.repeat(np.arange(6), 2), 2))
            np.testing.assert_allclose(
                slab.station_natural_coord,
                np.tile(np.repeat(2.0 * np.array(_HINGE_FRAC) - 1.0, 2), 2),
                rtol=0, atol=1e-12)
            j = np.tile([0, 1], 12)
            want = [[_fib(s, int(e), int(g) + 1, int(jj)) / (1e4 if c else 1.0)
                     for e, g, jj in zip(eids, gps, j)]
                    for s in range(1, _STEPS + 1)]
            np.testing.assert_allclose(np.asarray(slab.values), want,
                                       rtol=1e-15, atol=0)
    finally:
        res.close()


#: The file stations of each element when they are its live sections.
_LOBATTO_AT = {e: tuple(f * L for f in _LOBATTO_3) for e, L in _BEAM_L.items()}


@pytest.mark.parametrize("ops, eid, shown, live", [
    # Element 2's middle station 1e-3 off in natural coordinates (L = 3):
    # ten times the -precision 6 slack, so the file is another rule's.
    (lambda: _lobatto(written={1: _LOBATTO_AT[1],
                               2: (0.0, 1.5 + 1.5e-3, 3.0)}),
     2, r"\[-1\.0, 0\.001, 1\.0\]", r"\[-1\.0, 0\.0, 1\.0\]"),
    # The right stations, its last two swapped: only their order is wrong,
    # and the live order is what places each section.
    (lambda: _lobatto(written={1: _LOBATTO_AT[1], 2: (0.0, 3.0, 1.5)}),
     2, r"\[-1\.0, 1\.0, 0\.0\]", r"\[-1\.0, 0\.0, 1\.0\]"),
    # The same two defects on element 1 (L = 2), the record's second
    # element: every element is checked, not the first alone.
    (lambda: _lobatto(written={1: (0.0, 1.0 + 1.0e-3, 2.0),
                               2: _LOBATTO_AT[2]}),
     1, r"\[-1\.0, 0\.001, 1\.0\]", r"\[-1\.0, 0\.0, 1\.0\]"),
    (lambda: _lobatto(written={1: (0.0, 2.0, 1.0), 2: _LOBATTO_AT[2]}),
     1, r"\[-1\.0, 1\.0, 0\.0\]", r"\[-1\.0, 0\.0, 1\.0\]"),
    # An end station 1e-3 off, inside the range check's [-1, 1]: the ends
    # are compared too (element 2's first, element 1's last).
    (lambda: _lobatto(written={1: _LOBATTO_AT[1],
                               2: (1.5e-3, 1.5, 3.0)}),
     2, r"\[-0\.999, 0\.0, 1\.0\]", r"\[-1\.0, 0\.0, 1\.0\]"),
    (lambda: _lobatto(written={1: (0.0, 1.0, 2.0 - 1.0e-3),
                               2: _LOBATTO_AT[2]}),
     1, r"\[-1\.0, 0\.0, 0\.999\]", r"\[-1\.0, 0\.0, 1\.0\]"),
    # Live sections out of order, the file's sorted: the order check would
    # pass it, and only the live locations show its middle pair swapped.
    (lambda: _hinge_ops(written={
        1: tuple(f * 2.0 for f in _HINGE_FRAC),
        2: tuple(f * 3.0 for f in sorted(_HINGE_FRAC))}),
     2, r"\[-1\.0, -0\.8, -0\.6, -0\.4, 0\.8, 1\.0\]",
     r"\[-1\.0, -0\.8, -0\.4, -0\.6, 0\.8, 1\.0\]"),
], ids=["near-miss", "swapped-pair", "near-miss-second-element",
        "swapped-pair-second-element", "first-station-off",
        "last-station-off-second-element", "hinge-sorted"])
def test_fibre_stations_near_the_live_sections_still_refuse_by_id(
    tmp_path, ops, eid, shown, live,
) -> None:
    """Stations close to the live sections, but not them, refuse naming the
    FEM element: the file's and the live locations side by side. The record
    lists element 2 before element 1."""
    _read_refuses(tmp_path, ops(),
                  rf"FEM element {eid} has stations {shown} in the file, but "
                  rf"its live sections are at {live} \(natural coordinates\)\. "
                  rf"The file does not match the live domain\.")


def test_a_file_fibre_count_that_is_not_the_live_one_refuses(tmp_path) -> None:
    ops = _lobatto(file_fibres={1: (3, 3, 3), 2: (3, 3, 3)})
    _read_refuses(tmp_path, ops,
                  r"3 fibres per section in the file, but the live domain "
                  r"gave each section 2 before the run")


# ---------------------------------------------------------------------------
# A line-stations record is sliced by the live section counts (#2328)
# ---------------------------------------------------------------------------

#: Element 1 (L = 2) with 2 sections, element 2 (L = 3) with 4: 6 stations
#: split 3 + 3 by an equal slice, and both blocks then increase within
#: [-1, 1] (-0.8, -0.4, 0 and 0, 1/3, 2/3), so the order check alone reads
#: element 2's first station as element 1's third.
_MIXED = {1: (0.2, 0.6), 2: (1.0, 1.5, 2.0, 2.5)}


def _stations_rec(ids: "list[int]" = [1, 2]) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="line_stations", name="ls",
        components=("axial_force", "bending_moment_z"), dt=None, n_steps=None,
        element_ids=np.array(ids), element_class_name="ForceBeamColumn2d")


def _stations_ops(locations: "dict[int, tuple[float, ...] | None]") -> FiberOps:
    return FiberOps(locations=locations, fibres={1: (), 2: ()},
                    cls="ForceBeamColumn2d")


def test_line_stations_of_unequal_live_counts_refuse_by_count(tmp_path) -> None:
    """The section locations are read at ``__enter__``, and the read
    refuses the record by its counts before the columns are divided. With
    no counts, the equal slice passes the order check and reads element
    2's first station as element 1's (the stock reproducer of #2328)."""
    fem, model_h5 = _beam_fem(tmp_path)
    ops = _stations_ops(dict(_MIXED))
    run = _run(_spec(fem, _stations_rec()), tmp_path / "run", model_h5, ops)
    assert [a for a in ops.asked if a.startswith("sectionLocation")] == [
        "sectionLocation:1", "sectionLocation:2"]
    with pytest.raises(ValueError,
                       match=r"Record 'ls' \(line_stations\): its elements have "
                             r"different integration-point counts in the live "
                             r"domain \(2 points: FEM elements \[1\]; 4 points: "
                             r"FEM elements \[2\]\)"):
        run.results(cache_root=tmp_path / "cache")


def test_line_stations_of_equal_live_counts_read_each_element(tmp_path) -> None:
    fem, model_h5 = _beam_fem(tmp_path)
    loc = {e: tuple(f * L for f in _LOBATTO_3) for e, L in _BEAM_L.items()}
    run = _run(_spec(fem, _stations_rec()), tmp_path / "run", model_h5,
               _stations_ops(loc))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        slab = res.elements.line_stations.get(component="axial_force")
        np.testing.assert_array_equal(slab.element_index, np.repeat([1, 2], 3))
        np.testing.assert_allclose(slab.station_natural_coord,
                                   np.tile([-1.0, 0.0, 1.0], 2),
                                   rtol=0, atol=1e-15)
    finally:
        res.close()


def test_a_line_stations_element_with_no_sections_refuses_before_the_run(
    tmp_path,
) -> None:
    fem, model_h5 = _beam_fem(tmp_path)
    ops = _stations_ops({1: _MIXED[1], 2: None})
    with pytest.raises(ValueError,
                       match=r"line-stations record 'ls': FEM element 2 "
                             r"\('ForceBeamColumn2d'\) has no sections "
                             r"\(sectionLocation answered None\)"):
        _run(_spec(fem, _stations_rec()), tmp_path / "run", model_h5, ops)
    assert _element_calls(ops) == []


def test_line_stations_out_of_order_read_when_they_are_the_live_sections(
    tmp_path,
) -> None:
    """The live section locations prove the slice and its order, as a
    bridge model does: stations a legal rule writes out of order read as
    written, with no "Pass model=" refusal a FEMData user could not follow."""
    fem, model_h5 = _beam_fem(tmp_path)
    loc = {e: tuple(f * L for f in _HINGE_FRAC) for e, L in _BEAM_L.items()}
    ops = _stations_ops(loc)
    ops.columns = 2 * len(_HINGE_FRAC)        # (P, Mz) at each section
    run = _run(_spec(fem, _stations_rec()), tmp_path / "run", model_h5, ops)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        slab = res.elements.line_stations.get(component="axial_force")
        np.testing.assert_array_equal(slab.element_index, np.repeat([1, 2], 6))
        np.testing.assert_allclose(slab.station_natural_coord,
                                   np.tile(2.0 * np.array(_HINGE_FRAC) - 1.0, 2),
                                   rtol=0, atol=1e-12)
    finally:
        res.close()


def test_line_stations_that_are_not_the_live_sections_refuse_by_id(
    tmp_path,
) -> None:
    """A file whose stations are not the sections the live domain gave
    (element 2's at Lobatto points of another length) refuses by FEM id."""
    fem, model_h5 = _beam_fem(tmp_path)
    loc = {e: tuple(f * L for f in _LOBATTO_3) for e, L in _BEAM_L.items()}
    ops = FiberOps(locations=loc, fibres={1: (), 2: ()},
                   written={1: loc[1], 2: (0.0, 1.0, 3.0)},
                   cls="ForceBeamColumn2d")
    run = _run(_spec(fem, _stations_rec()), tmp_path / "run", model_h5, ops)
    with pytest.raises(ValueError, match=(
            r"FEM element 2 of record 'ls' \(line_stations\) has stations "
            r"\[-1\.0, -0\.333333, 1\.0\] in the file, but its live sections "
            r"are at \[-1\.0, 0\.0, 1\.0\]")):
        run.results(cache_root=tmp_path / "cache")


@pytest.mark.parametrize("written, shown", [
    # Element 2's middle station 1e-3 off in natural coordinates (L = 3):
    # ten times the -precision 6 slack, so the file is another rule's.
    ((0.0, 1.5 + 1.5e-3, 3.0), r"\[-1\.0, 0\.001, 1\.0\]"),
    # The right stations, its last two swapped: only their order is wrong,
    # and the live order is what places each section.
    ((0.0, 3.0, 1.5), r"\[-1\.0, 1\.0, 0\.0\]"),
], ids=["near-miss", "swapped-pair"])
def test_line_stations_near_the_live_sections_still_refuse_by_id(
    tmp_path, written, shown,
) -> None:
    fem, model_h5 = _beam_fem(tmp_path)
    loc = {e: tuple(f * L for f in _LOBATTO_3) for e, L in _BEAM_L.items()}
    ops = FiberOps(locations=loc, fibres={1: (), 2: ()},
                   written={1: loc[1], 2: written}, cls="ForceBeamColumn2d")
    run = _run(_spec(fem, _stations_rec()), tmp_path / "run", model_h5, ops)
    with pytest.raises(ValueError, match=(
            rf"FEM element 2 of record 'ls' \(line_stations\) has stations "
            rf"{shown} in the file, but its live sections are at "
            rf"\[-1\.0, 0\.0, 1\.0\]")):
        run.results(cache_root=tmp_path / "cache")


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
    """A pairing that is neither the bridge's model nor the live facts
    carries no section facts, so a fibres record handed to the transcoder
    with one refuses by name, not on a missing method."""
    fem, _ = tets
    rec = ResolvedRecorderRecord(
        category="fibers", name="fib", components=("fiber_stress",), dt=None,
        n_steps=None, element_ids=np.array([1]),
        element_class_name="ForceBeamColumn3d")
    tr = RecorderTranscoder(_spec(fem, rec), tmp_path, tmp_path / "r.h5", fem,
                            pairing=_Pairing(fem, {1: "forceBeamColumn"}))
    with pytest.raises(ValueError, match=r"Record 'fib' \(category=fibers\): "
                                         r"its stations are counted .* a "
                                         r"_Pairing carries neither"):
        tr.run()


def test_a_line_stations_record_over_a_pairing_refuses_as_fibres_do(
    tets, tmp_path,
) -> None:
    """The same condition as the fibres case above, on the line-stations
    read: one exception class (``ValueError``) and one message for both."""
    fem, _ = tets
    rec = ResolvedRecorderRecord(
        category="line_stations", name="ls",
        components=("axial_force", "bending_moment_z"), dt=None, n_steps=None,
        element_ids=np.array([1]), element_class_name="ForceBeamColumn3d")
    sec_lr, gpx_lr = emit_logical(rec, output_dir=str(tmp_path))
    np.savetxt(gpx_lr.file_path, [[1.0, 0.0, 1.0]])
    np.savetxt(sec_lr.file_path, [[1.0, 1.0, 2.0, 3.0, 4.0]])
    tr = RecorderTranscoder(_spec(fem, rec), tmp_path, tmp_path / "r.h5", fem,
                            stage_name="s", stage_kind="static",
                            pairing=_Pairing(fem, {1: "forceBeamColumn"}))
    with pytest.raises(ValueError, match=r"Record 'ls' \(line_stations\): its "
                                         r"stations are counted .* a "
                                         r"_Pairing carries neither"):
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


_DEFORMATION = (r"component 'axial_strain' cannot go through MPCO: the MPCO "
                r"reader does not decode its section.deformation bucket")


@pytest.mark.parametrize("records, match", [
    ((_truss_rec(("axial_strain",), name="ls", category="line_stations",
                 cls="ForceBeamColumn3d"),), _DEFORMATION),
    ((_truss_rec(), _truss_rec(("axial_strain",), name="ls",
                               category="line_stations",
                               cls="ForceBeamColumn3d")), _DEFORMATION),
], ids=["section-deformation", "section-deformation-beside-axial"])
def test_only_the_truss_axial_force_leaves_mpco(tmp_path, records, match) -> None:
    """Every other MPCO refusal stands at ``__enter__``, with or without a
    Truss axial force beside it: a line-stations section deformation would
    read back empty on the classic recorders too."""
    fem, model_h5, ops = _truss_fem(tmp_path)
    with pytest.raises(ValueError, match=match):
        _run(_spec(fem, *records), tmp_path / "run", model_h5, ops)
    assert ops.recorder_calls == [] and "eleNodes" not in ops.asked


@pytest.mark.parametrize("components", [
    ("displacement_x",), ("axial_force", "displacement_x"),
], ids=["no-token", "no-token-beside-axial"])
def test_a_gauss_component_with_no_token_never_reaches_mpco(components) -> None:
    """Every gauss component but the Truss axial force has an MPCO ``-E``
    token; one with none belongs to another category, so the record refuses
    it by name when built, before any route is chosen (#2344)."""
    with pytest.raises(ValueError, match=r"'displacement_x' belongs to "
                                         r"category 'nodes', not 'gauss'"):
        _truss_rec(components, name="u")


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


def _bridge_trusses(tmp: Path) -> Path:
    """Trusses of FEM ids 7 (nodes 1-2) and 9 (nodes 2-3) on the bridge,
    archived as ``ops.h5``: the bridge tags them in sequence, not 7 and 9."""
    from apeGmsh.opensees import apeSees

    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=2)
    ids = np.array([7, 9], dtype=np.int64)
    group = ElementGroup(element_type=line, ids=ids,
                         connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64))
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    pg = PhysicalGroupSet({(1, 1): {"name": "Bar", "node_ids": node_ids,
                                    "node_coords": coords, "element_ids": ids}})
    fem = FEMData(
        nodes=NodeComposite(node_ids=node_ids, node_coords=coords, physical=pg,
                            labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=2, types=[line]),
        model_name="femdata_elements_bridge_trusses")
    bridge = apeSees(fem, default_orientation=None)
    bridge.model(ndm=2, ndf=2)
    mat = bridge.uniaxialMaterial.ElasticMaterial(E=1.0)
    bridge.element.Truss(pg="Bar", A=1.0, material=mat)
    model_h5 = tmp / "ops.h5"
    bridge.h5(str(model_h5))
    return model_h5


def test_a_class_less_truss_axial_force_over_the_bridge_archive_reads_its_tags(
    tmp_path,
) -> None:
    """Over the bridge's archive the live classes are read at the archive's
    tags (``element_tag_map``), not at the FEM ids: FEM elements 7 and 9
    are live elements 2 and 3. Reading them at 7 and 9 would find no live
    element and keep the axial force on MPCO, which refuses it (#2298's
    deferred gap)."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from apeGmsh.results.spec._emit import element_tag_map

    model_h5 = _bridge_trusses(tmp_path)
    model = OpenSeesModel.from_h5(model_h5)
    tags = element_tag_map(model, fem_snapshot_id=model.fem.snapshot_id)
    assert tags == {7: 2, 9: 3}
    ops = MpcoTrussOps(ndm=2, ndf={1: 2, 2: 2, 3: 2},
                       elements={2: ("Truss", (1, 2)), 3: ("Truss", (2, 3))},
                       coords={1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0)})
    rec = ResolvedRecorderRecord(
        category="gauss", name="axial", components=("axial_force",), dt=None,
        n_steps=None, element_ids=np.array([9, 7]), element_class_name=None)
    spec = ResolvedRecorderSpec(fem_snapshot_id=model.fem.snapshot_id,
                                records=(rec,))
    run = _run(spec, tmp_path / "run", model_h5, ops)
    assert run.route == "recorders"
    assert [c for c in ops.recorder_calls if c[0] == "mpco"] == []
    (call,) = _element_calls(ops)
    assert call[call.index("-ele") + 1:call.index("-ele") + 3] == (3, 2)
    assert ops.asked.count("eleType") == 2


def test_the_reroute_raises_no_mpco_line_stations_warning(tmp_path) -> None:
    """A class-less line-stations record beside the Truss axial force: the
    MPCO emit would warn that elastic beams may read empty, but the run
    never makes an MPCO recorder, so nothing warns but the route."""
    fem, model_h5, ops = _truss_fem(tmp_path)
    ls = _truss_rec(("axial_force",), name="ls", category="line_stations",
                    cls=None)
    run = _run(_spec(fem, _truss_rec(), ls), tmp_path / "run", model_h5, ops)
    assert run.route == "recorders"
