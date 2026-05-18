"""S3c follow-on — ``.select()`` on the five element sub-composites.

``docs/plans/selection-unification.md`` §9 item 2.  S3c landed
``ResultChain`` + ``results.nodes/elements.select()``; the five
element sub-composites
(``gauss`` / ``fibers`` / ``layers`` / ``line_stations`` / ``springs``)
were a consciously-deferred follow-on because ``fibers.get`` /
``layers.get`` carry **extra** terminal kwargs
(``gp_indices=`` / ``layer_indices=``) the uniform terminal did not
forward.

This file locks the follow-on:

* every sub-composite exposes ``.select(...)`` returning the **same**
  reused ``ResultChain`` (point family, element level) — no new
  ``SelectionChain`` subclass (``test_selection_idiom`` stays green),
  the shared ``engine_for`` per-composite singleton so base
  ``SelectionChain._compatible`` set-algebra identity holds;
* ``select(...).get(...)`` is **parity-equal** to the existing
  ``results.elements.<sub>.get(ids=<equiv>, component=...)`` —
  id-seeded, pg-seeded, *and* after a spatial daisy-chain
  (``.in_box`` on element centroids) — for all five;
* the chain terminal forwards each sub-composite's extra kwargs
  (``gp_indices=`` for ``fibers``; ``gp_indices=`` /
  ``layer_indices=`` for ``layers``), not just the uniform
  ``component=`` / ``time=`` / ``stage=``;
* the host's own ``.get`` signature stays the single source of truth:
  an extra kwarg the host does not accept fails **loud** (never
  silently dropped by the generic chain);
* resolution is delegated to the existing
  ``_combine_candidates`` → ``_resolve_element_ids`` (locked
  resolution contract, not re-implemented) — proven with a spy;
* ``results.elements.select`` keeps its own (byte-unchanged) impl;
  the five sub-composites *inherit* the shared
  ``_ElementGeometryMixin.select``.

No ``openseespy`` dependency (curated no-openseespy CI gate): the four
native-format sub-composites use a synthetic native HDF5 + a
``SimpleNamespace`` mock FEM (the exact pattern
``tests/test_result_chain.py`` uses).  Springs are an MPCO-only result
(the native writer/reader do not carry them), so the spring row uses a
synthetic ``.mpco`` (the exact pattern
``tests/test_results_mpco_spring_mock.py`` uses) with a bound mock FEM
for centroid spatial filtering.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results._composites import (
    ElementResultsComposite,
    FibersResultsComposite,
    GaussResultsComposite,
    LayersResultsComposite,
    LineStationsResultsComposite,
    SpringsResultsComposite,
    _ElementGeometryMixin,
)
from apeGmsh._kernel.chain import SelectionChain
from apeGmsh.results._result_chain import ResultChain
# selection-unification-v2 P2-I (§6.1 STOP-2(c)): the sub-composite
# ``.select()`` hooks now return the v2 terminal ``MeshSelection``
# (legacy ``ResultChain`` left defined-but-unwired; P3 deletes it).
# The host-return-type assertions below are BEHAVIOURAL (they then
# exercise level / set-algebra / the slab read + extra-kwarg parity +
# the R5 fail-loud invariant, which must stay green) → assert the new
# type.  ``type(sel) is ...`` was a "no NEW subclass" pin — that
# invariant still holds, now against the canonical ``MeshSelection``
# terminal (it has no subclasses), so it asserts the new type rather
# than xfail.  The results terminal read is the verbatim rename
# ``.get`` → ``.values`` on ``MeshSelection``.
from apeGmsh.mesh._mesh_selection import MeshSelection
from apeGmsh.results._slabs import (
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    SpringSlab,
)
from apeGmsh.results.writers import NativeWriter


# =====================================================================
# Synthetic native HDF5 (gauss/fibers/layers/line_stations) + mock FEM
# =====================================================================

def _mock_fem():
    """8 nodes, two quad elements 10 & 20.

    Element 10 = nodes 1-4 (unit square at origin, centroid
    (0.5,0.5,0.0)); element 20 = nodes 5-8 (unit square translated to
    (5,5,0), centroid (5.5,5.5,0.0)).  A box around the origin keeps
    only element 10 — the centroid spatial filter has a non-trivial
    effect, so ``select().in_box(...).get()`` proves the daisy-chain
    narrowed *before* the terminal read.
    """
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [5.0, 5.0, 0.0], [6.0, 5.0, 0.0],
        [6.0, 6.0, 0.0], [5.0, 6.0, 0.0],
    ], dtype=np.float64)
    node_ids = np.arange(1, 9, dtype=np.int64)

    def _resolve(*, element_type=None):
        return (
            np.array([10, 20], dtype=np.int64),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64),
        )

    nodes_ns = SimpleNamespace(
        ids=node_ids,
        coords=coords,
        physical=SimpleNamespace(
            node_ids=lambda n: np.array([], dtype=np.int64),
        ),
        labels=SimpleNamespace(
            node_ids=lambda n: np.array([], dtype=np.int64),
        ),
    )
    elements_ns = SimpleNamespace(
        ids=np.array([10, 20], dtype=np.int64),
        types=[SimpleNamespace(name="quad4")],
        resolve=_resolve,
        physical=SimpleNamespace(element_ids=lambda n: {
            "Near": np.array([10], dtype=np.int64),
            "Both": np.array([10, 20], dtype=np.int64),
        }[n]),
        labels=SimpleNamespace(
            element_ids=lambda n: np.array([], dtype=np.int64),
        ),
    )
    return SimpleNamespace(
        snapshot_id="testhash", nodes=nodes_ns, elements=elements_ns,
    )


def _make_native(tmp_path: Path):
    """One static stage (T=2) with all four native sub-composite groups
    written for elements 10 & 20."""
    path = tmp_path / "subcomposites.h5"
    time = np.array([0.0, 1.0])
    elem_idx = np.array([10, 20], dtype=np.int64)

    # gauss: 1 GP/elem -> (T=2, E=2, nGP=1)
    nat = np.array([[0.0, 0.0]], dtype=np.float64)
    sxx = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float64)

    # fibers: flat rows, 2 GP/elem x 1 fiber -> n=4
    f_e = np.array([10, 10, 20, 20], dtype=np.int64)
    f_g = np.array([0, 1, 0, 1], dtype=np.int64)
    f_y = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    f_z = np.zeros(4, dtype=np.float64)
    f_a = np.full(4, 0.25, dtype=np.float64)
    f_m = np.array([100, 100, 200, 200], dtype=np.int64)
    f_sig = np.arange(2 * 4, dtype=np.float64).reshape(2, 4)

    # layers: 2 GP x 2 layer per elem -> n=8
    l_e = np.array([10] * 4 + [20] * 4, dtype=np.int64)
    l_g = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    l_l = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    l_sg = np.zeros(8, dtype=np.int64)
    l_t = np.full(8, 0.1, dtype=np.float64)
    l_q = np.tile([1.0, 0.0, 0.0, 0.0], (8, 1))
    l_sig = np.arange(2 * 8, dtype=np.float64).reshape(2, 8)

    # line stations: 3 stations -> (T=2, E=2, nst=3)
    snc = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    lf = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3)

    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="static", kind="static", time=time)
        w.write_nodes(
            sid, "partition_0",
            node_ids=np.array([1], dtype=np.int64),
            components={"displacement_x": np.array([[0.0], [1.0]])},
        )
        w.write_gauss_group(
            sid, "partition_0", "g0",
            class_tag=4, int_rule=1,
            element_index=elem_idx, natural_coords=nat,
            components={"stress_xx": sxx},
        )
        w.write_fibers_group(
            sid, "partition_0", "fib0",
            section_tag=1, section_class="FiberSection",
            element_index=f_e, gp_index=f_g,
            y=f_y, z=f_z, area=f_a, material_tag=f_m,
            components={"fiber_stress": f_sig},
        )
        w.write_layers_group(
            sid, "partition_0", "lay0",
            element_index=l_e, gp_index=l_g, layer_index=l_l,
            sub_gp_index=l_sg, thickness=l_t, local_axes_quaternion=l_q,
            components={"layer_stress": l_sig},
        )
        w.write_line_stations_group(
            sid, "partition_0", "ls0",
            class_tag=3, int_rule=1,
            element_index=elem_idx, station_natural_coord=snc,
            components={"sectionForce": lf},
        )
        w.end_stage()

    return Results.from_native(path, fem=_mock_fem())


# Per sub-composite: (attr, slab type, component, extra terminal kwargs)
_NATIVE_CASES = [
    ("gauss", GaussSlab, "stress_xx", {}),
    ("fibers", FiberSlab, "fiber_stress", {"gp_indices": [0]}),
    ("layers", LayerSlab, "layer_stress",
     {"gp_indices": [0], "layer_indices": [1]}),
    ("line_stations", LineStationSlab, "sectionForce", {}),
]


def _sids(seq):
    """Sorted ids from a chain or a plain id sequence.

    selection-unification-v2 P2-I (§6.1 STOP-2(b)): the sub-composite
    ``.select()`` now returns ``MeshSelection`` whose ``__iter__``
    yields ``(id, payload)`` pairs.  Set-algebra / identity is defined
    on the ``_items`` atoms (unchanged by the pair-view), so a chain
    is read via ``_items``; a plain id array iterates as bare ids."""
    if isinstance(seq, SelectionChain):
        return sorted(int(a) for a in seq._items)
    return sorted(int(x) for x in seq)


# =====================================================================
# 1. Every sub-composite has the shared inherited .select()
# =====================================================================

def test_select_is_shared_inherited_not_a_new_chain():
    # Defined once on the shared mixin; the five sub-composites inherit
    # it; results.elements keeps its own byte-unchanged override.
    assert "select" in _ElementGeometryMixin.__dict__
    assert "select" in ElementResultsComposite.__dict__       # own (S3c)
    for cls in (GaussResultsComposite, FibersResultsComposite,
                LayersResultsComposite, LineStationsResultsComposite,
                SpringsResultsComposite):
        assert "select" not in cls.__dict__                   # inherited
        assert cls.select is _ElementGeometryMixin.select


@pytest.mark.parametrize("attr", [c[0] for c in _NATIVE_CASES])
def test_select_returns_reused_resultchain_element_level(tmp_path, attr):
    r = _make_native(tmp_path)
    sub = getattr(r.elements, attr)
    sel = sub.select(ids=[10, 20])
    # P2-I: was ResultChain.  Still ONE canonical terminal (the
    # "no new subclass" invariant holds against MeshSelection, which
    # itself has no subclasses).
    assert isinstance(sel, MeshSelection)        # the reused terminal
    assert type(sel) is MeshSelection            # no new subclass
    assert sel.FAMILY == "point"
    assert sel._level == "element"
    assert _sids(sel) == [10, 20]
    # no selector -> every domain element (fem.elements.ids)
    assert _sids(sub.select()) == [10, 20]
    # pg seed delegates to the existing resolver
    assert _sids(sub.select(pg="Near")) == [10]


# =====================================================================
# 2. select(...).get(...) parity with get(ids=<equiv>, ...) — id seed,
#    pg seed, and after a centroid spatial daisy-chain.  Extra terminal
#    kwargs (gp_indices=/layer_indices=) forwarded where applicable.
# =====================================================================

@pytest.mark.parametrize(
    "attr,slab_t,comp,extra", _NATIVE_CASES,
    ids=[c[0] for c in _NATIVE_CASES],
)
def test_native_subcomposite_select_get_parity(
    tmp_path, attr, slab_t, comp, extra,
):
    r = _make_native(tmp_path)
    sub = getattr(r.elements, attr)
    baseline = sub.get(ids=[10], component=comp, **extra)
    assert isinstance(baseline, slab_t)

    def _assert_parity(slab):
        assert isinstance(slab, slab_t)
        assert type(slab) is slab_t
        np.testing.assert_array_equal(
            slab.element_index, baseline.element_index,
        )
        np.testing.assert_array_equal(slab.values, baseline.values)
        np.testing.assert_array_equal(slab.time, baseline.time)

    # id-seeded   (P2-I: chain terminal .get → .values)
    _assert_parity(
        sub.select(ids=[10]).values(component=comp, **extra)
    )
    # pg-seeded (delegates to _resolve_element_ids)
    _assert_parity(
        sub.select(pg="Near").values(component=comp, **extra)
    )
    # spatial daisy-chain: box around the origin keeps only element 10's
    # centroid (element 20 centroid is at (5.5,5.5,0)) -> same final ids
    _assert_parity(
        sub.select()
           .in_box((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
           .values(component=comp, **extra)
    )
    # ... and chained further with on_plane (z=0 keeps it)
    _assert_parity(
        sub.select(pg="Both")
           .in_box((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
           .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
           .values(component=comp, **extra)
    )


def test_extra_kwargs_actually_filter(tmp_path):
    """The forwarded gp_indices=/layer_indices= are not inert: a
    different value yields a different slab (proves the chain forwards
    them to the host reader, not drops them)."""
    r = _make_native(tmp_path)
    f0 = r.elements.fibers.select(ids=[10]).values(  # P2-I: .get→.values
        component="fiber_stress", gp_indices=[0],
    )
    f1 = r.elements.fibers.select(ids=[10]).values(  # P2-I: .get→.values
        component="fiber_stress", gp_indices=[1],
    )
    assert _sids(np.unique(f0.gp_index)) == [0]
    assert _sids(np.unique(f1.gp_index)) == [1]
    assert not np.array_equal(f0.values, f1.values)

    lyr = r.elements.layers.select(ids=[10]).values(  # P2-I: .get→.values
        component="layer_stress", gp_indices=[0], layer_indices=[1],
    )
    assert set(np.unique(lyr.gp_index)) == {0}
    assert set(np.unique(lyr.layer_index)) == {1}
    # parity with the direct call carrying the same extra kwargs
    direct = r.elements.layers.get(
        ids=[10], component="layer_stress",
        gp_indices=[0], layer_indices=[1],
    )
    np.testing.assert_array_equal(lyr.values, direct.values)


def test_unknown_extra_kwarg_fails_loud(tmp_path):
    """The generic chain never names a sub-composite kwarg; the host's
    own .get signature is the single source of truth.  gauss.get has no
    gp_indices= -> a loud TypeError, not a silent drop."""
    r = _make_native(tmp_path)
    # P2-I: chain terminal .get → .values; the R5 invariant is
    # unchanged — .values forwards **extra opaquely, so the host's own
    # .get signature is still the single source of truth and an
    # unknown kwarg fails loud THERE (never silently dropped here).
    with pytest.raises(TypeError):
        r.elements.gauss.select(ids=[10]).values(
            component="stress_xx", gp_indices=[0],
        )
    with pytest.raises(TypeError):
        r.elements.line_stations.select(ids=[10]).values(
            component="sectionForce", layer_indices=[0],
        )


def test_select_delegates_to_combine_candidates(tmp_path, monkeypatch):
    """select() on a sub-composite calls the EXISTING
    _combine_candidates (-> _resolve_element_ids), not a re-implemented
    resolver — the locked resolution contract is preserved by reuse."""
    r = _make_native(tmp_path)
    seen = {}
    real_combine = GaussResultsComposite._combine_candidates
    real_resolve = GaussResultsComposite._resolve_element_ids

    def _spy_combine(self, *, pg, label, selection, ids, element_type):
        seen["combine"] = True
        return real_combine(
            self, pg=pg, label=label, selection=selection, ids=ids,
            element_type=element_type,
        )

    def _spy_resolve(self, *, pg, label, selection, ids):
        seen["resolve"] = True
        return real_resolve(
            self, pg=pg, label=label, selection=selection, ids=ids,
        )

    monkeypatch.setattr(
        GaussResultsComposite, "_combine_candidates", _spy_combine,
    )
    monkeypatch.setattr(
        GaussResultsComposite, "_resolve_element_ids", _spy_resolve,
    )
    sel = r.elements.gauss.select(pg="Near")
    assert seen.get("combine") is True
    assert seen.get("resolve") is True
    assert _sids(sel) == [10]


def test_set_algebra_same_subcomposite_engine_singleton(tmp_path):
    """engine_for is a per-composite singleton, so two selections from
    the SAME sub-composite share one engine and compose; a node
    selection (different host) is loud."""
    r = _make_native(tmp_path)
    a = r.elements.gauss.select(ids=[10])
    b = r.elements.gauss.select(ids=[20])
    assert _sids(a | b) == [10, 20]
    assert _sids(a & r.elements.gauss.select(ids=[10, 20])) == [10]
    # cross-host (gauss vs nodes) -> different engine adapters -> loud
    with pytest.raises(TypeError, match="different engines"):
        a | r.nodes.select(ids=[1])


# =====================================================================
# 3. Springs — MPCO-only (no native writer/reader); synthetic .mpco +
#    bound mock FEM so the centroid spatial filter has coordinates.
# =====================================================================

def _make_spring_mpco(path: Path, *, element_ids, n_springs, force_values):
    """Minimal ZeroLength MPCO (pattern from
    tests/test_results_mpco_spring_mock.py::_make_spring_mpco)."""
    f = h5py.File(str(path), "w")
    stage = f.create_group("MODEL_STAGE[1]")
    model = stage.create_group("MODEL")
    nodes = model.create_group("NODES")
    nodes.create_dataset(
        "ID", data=np.array(element_ids, dtype=np.int32).reshape(-1, 1),
    )
    nodes.create_dataset("COORDINATES", data=np.zeros((len(element_ids), 3)))
    model.create_group("ELEMENTS")
    model.create_group("SECTION_ASSIGNMENTS")

    results = stage.create_group("RESULTS")
    n_steps = force_values.shape[0]
    time_arr = np.linspace(0.0, 1.0, n_steps)
    on_nodes = results.create_group("ON_NODES")
    disp = on_nodes.create_group("DISPLACEMENT")
    disp.create_dataset(
        "ID", data=np.array(element_ids, dtype=np.int32).reshape(-1, 1),
    )
    data_grp_nd = disp.create_group("DATA")
    for i in range(n_steps):
        ds = data_grp_nd.create_dataset(
            f"STEP_{i}",
            data=np.zeros((len(element_ids), 3), dtype=np.float64),
        )
        ds.attrs["TIME"] = time_arr[i]

    on_elem = results.create_group("ON_ELEMENTS")
    bracket = "19-ZeroLength[1:0:0]"
    tok = on_elem.require_group("basicForce")
    bkt = tok.create_group(bracket)
    meta = bkt.create_group("META")
    meta.create_dataset("MULTIPLICITY", data=np.array([[1]], dtype=np.int32))
    meta.create_dataset("GAUSS_IDS", data=np.array([[-1]], dtype=np.int32))
    meta.create_dataset(
        "NUM_COMPONENTS", data=np.array([[n_springs]], dtype=np.int32),
    )
    comp_str = "0." + ",".join(f"F_{i}" for i in range(n_springs))
    meta.create_dataset("COMPONENTS", data=np.array([comp_str.encode()]))
    bkt.attrs["NUM_COLUMNS"] = n_springs
    bkt.create_dataset(
        "ID", data=np.array(element_ids, dtype=np.int32).reshape(-1, 1),
    )
    dg = bkt.create_group("DATA")
    for i in range(n_steps):
        dg.create_dataset(f"STEP_{i}", data=force_values[i].astype(np.float64))
    f.close()
    return path


def _spring_mock_fem():
    """3 elements 10/20/30; element 10's centroid at the origin, 20 & 30
    far — so a box around the origin keeps only element 10."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [5.0, 6.0, 0.0],
        [9.0, 9.0, 0.0], [9.0, 9.0, 0.0], [9.0, 9.0, 0.0], [9.0, 9.0, 0.0],
    ], dtype=np.float64)

    def _resolve(*, element_type=None):
        return (
            np.array([10, 20, 30], dtype=np.int64),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                     dtype=np.int64),
        )

    nodes_ns = SimpleNamespace(
        ids=np.arange(1, 13, dtype=np.int64),
        coords=coords,
        physical=SimpleNamespace(
            node_ids=lambda n: np.array([], dtype=np.int64),
        ),
        labels=SimpleNamespace(
            node_ids=lambda n: np.array([], dtype=np.int64),
        ),
    )
    elements_ns = SimpleNamespace(
        ids=np.array([10, 20, 30], dtype=np.int64),
        types=[SimpleNamespace(name="quad4")],
        resolve=_resolve,
        physical=SimpleNamespace(
            element_ids=lambda n: np.array([10], dtype=np.int64),
        ),
        labels=SimpleNamespace(
            element_ids=lambda n: np.array([], dtype=np.int64),
        ),
    )
    return SimpleNamespace(
        snapshot_id="springhash", nodes=nodes_ns, elements=elements_ns,
    )


def test_springs_select_get_parity(tmp_path):
    """Springs `.select()` parity (the S3f contract under test).

    Scope note: a `Results.from_mpco` reader synthesises its FEM from
    the MPCO `MODEL/ELEMENTS` group, which for ZeroLength springs is
    empty — so on a *reader-synthesised* FEM the centroid spatial verbs
    (`select().in_box(...)`, and equally the **legacy** `springs.in_box`)
    yield an empty slab. That silent-empty-on-empty-element-set
    behaviour is a **pre-existing limitation of MPCO FEM synthesis +
    the element-geometry helpers**, identical for the legacy path; it
    is NOT introduced by S3f and is out of scope here. To isolate the
    thing S3f actually changes — that `select(...).get(...)` is
    id/value parity-equal to `springs.get(ids=<equiv>, ...)`,
    including after a spatial daisy-chain — we bind a mock FEM that
    carries spring element geometry (the bridge that re-attaches a
    real session's FEMData would supply the same).
    """
    rng = np.random.default_rng(11)
    vals = rng.standard_normal((2, 3, 2))           # (steps, elems, springs)
    path = _make_spring_mpco(
        tmp_path / "s.mpco",
        element_ids=[10, 20, 30], n_springs=2, force_values=vals,
    )
    r = Results.from_mpco(str(path), fem=_spring_mock_fem())
    try:
        comp = "spring_force_0"
        sub = r.elements.springs

        sel = sub.select(ids=[10, 20])
        # P2-I: was ResultChain (still one canonical terminal — the
        # "no new subclass" invariant holds against MeshSelection).
        assert isinstance(sel, MeshSelection)
        assert type(sel) is MeshSelection
        assert sel._level == "element"
        assert _sids(sel) == [10, 20]
        assert _sids(sub.select()) == [10, 20, 30]

        base_multi = sub.get(ids=[10, 20], component=comp)  # composite get
        assert isinstance(base_multi, SpringSlab)
        got_multi = sel.values(component=comp)  # P2-I: chain .get→.values
        np.testing.assert_array_equal(
            got_multi.element_index, base_multi.element_index,
        )
        np.testing.assert_array_equal(got_multi.values, base_multi.values)

        # spatial daisy-chain: box around origin keeps only element 10
        base_one = sub.get(ids=[10], component=comp)  # composite get
        spatial = (sub.select()
                      .in_box((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
                      .values(component=comp))  # P2-I: chain .get→.values
        np.testing.assert_array_equal(
            spatial.element_index, base_one.element_index,
        )
        np.testing.assert_array_equal(spatial.values, base_one.values)
        assert _sids(spatial.element_index) == [10]
    finally:
        r._reader.close()
