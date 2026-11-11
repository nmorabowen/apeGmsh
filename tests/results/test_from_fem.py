"""``Results.from_fem`` — bind a bare :class:`FEMData` snapshot to a
results file without going through the ``apeSees`` bridge.

Covers the reported gap: a physical-group model where
``fem = g.mesh.queries.get_fem_data()`` drove a hand-written OpenSees
deck previously had no first-class route to ``Results`` / ``viewer``.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

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
from apeGmsh.results import Results
from apeGmsh.results._bind import ModelFemMismatchWarning
from apeGmsh.results.Results import _resolve_results_kind
from apeGmsh.results.writers import NativeWriter
from tests.test_results_mpco_model_h5_binding import (
    _plate_fem,
    _write_model_h5,
    _write_mpco,
)

_TRUSS = (
    Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
    / "truss2d.ladruno"
)


# ---------------------------------------------------------------------
# kind resolution (pure)
# ---------------------------------------------------------------------


class TestResolveKind:
    def test_auto_detects_suffix(self) -> None:
        assert _resolve_results_kind("auto", "run.mpco") == "mpco"
        assert _resolve_results_kind("auto", "run.ladruno") == "ladruno"

    def test_explicit_passthrough(self) -> None:
        assert _resolve_results_kind("native", "run.h5") == "native"

    def test_auto_list_input_peeks_first(self) -> None:
        assert _resolve_results_kind(
            "auto", ["a.part-0.mpco", "a.part-1.mpco"]) == "mpco"

    def test_ambiguous_native_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot auto-detect"):
            _resolve_results_kind("auto", "run.h5")

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="not one of"):
            _resolve_results_kind("bogus", "run.mpco")


# ---------------------------------------------------------------------
# composed-fem guard (adversarial F1 — silent element mislabel)
# ---------------------------------------------------------------------


def test_composed_fem_refused(tmp_path: Path) -> None:
    """A composed fem carries no fem_eid<->ops-tag map in a neutral-only
    model.h5 — refuse rather than silently mislabel element results."""
    composed = SimpleNamespace(composed_from=("module",), snapshot_id="x")
    with pytest.raises(ValueError, match="composed"):
        Results.from_fem(composed, tmp_path / "run.mpco")


# ---------------------------------------------------------------------
# error messages point at the new route
# ---------------------------------------------------------------------


def test_from_mpco_missing_model_h5_names_from_fem() -> None:
    with pytest.raises(TypeError, match="from_fem"):
        Results.from_mpco("nonexistent.mpco")


def test_from_native_missing_model_names_from_fem() -> None:
    with pytest.raises(TypeError, match="from_fem"):
        Results.from_native("nonexistent.h5")


# ---------------------------------------------------------------------
# native round-trip — end-to-end plumbing (cache write + bind + read)
# ---------------------------------------------------------------------


def _native_results_no_model(g, tmp_path: Path):
    """A native results file with NO embedded /opensees zone — the
    bare-fem scenario from_fem targets."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    results = tmp_path / "grav.h5"
    with NativeWriter(results) as w:
        w.open(fem=fem)  # no model_h5_src => no bridge model in the file
        sid = w.begin_stage(name="grav", kind="static",
                            time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0",
            node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
            components={
                "displacement_x": np.zeros((1, len(fem.nodes.ids))),
            },
        )
        w.end_stage()
    return results, fem


def test_from_fem_native_roundtrip(g, tmp_path: Path) -> None:
    results, fem = _native_results_no_model(g, tmp_path)

    r = Results.from_fem(fem, results, kind="native", cache_root=tmp_path)

    # bound to the passed snapshot
    assert r.fem is not None
    assert r.fem.snapshot_id == fem.snapshot_id
    # a neutral-only model was materialised and bound
    assert r.model is not None and r.model.fem is not None
    # the cache file landed under <cache_root>/from_fem/
    cached = tmp_path / "from_fem" / f"{fem.snapshot_id}.model.h5"
    assert cached.exists()
    # nodal results read back through the bound reader
    slab = r.nodes.get(component="displacement_x")
    assert slab.values.shape[1] == len(fem.nodes.ids)


def test_from_fem_reuses_cached_model(g, tmp_path: Path) -> None:
    """A second from_fem for the same fem reuses the cached model.h5
    (keyed by snapshot_id) rather than rewriting it."""
    results, fem = _native_results_no_model(g, tmp_path)
    cached = tmp_path / "from_fem" / f"{fem.snapshot_id}.model.h5"

    Results.from_fem(fem, results, kind="native", cache_root=tmp_path)
    assert cached.exists()
    mtime = cached.stat().st_mtime_ns

    Results.from_fem(fem, results, kind="native", cache_root=tmp_path)
    assert cached.stat().st_mtime_ns == mtime  # not rewritten


# ---------------------------------------------------------------------
# #1408 — fem= is checked against the capture's elements
# ---------------------------------------------------------------------
#
# ``from_fem`` writes ``fem.to_h5``, an archive with no ops<->fem element
# tag map, so the capture's element tags are read as fem element ids.
# A deck that numbered its elements otherwise used to bind every
# physical group silently, and ``pg=`` then answered with the wrong
# elements (#1393's b0 probe: 58 of them).  The oracle is the ids
# themselves: the capture's tag 7 is not the fem's element 1.


def _truss_fem(eids: "tuple[int, int]" = (1, 2)) -> FEMData:
    """The ``truss2d.ladruno`` fixture's mesh: nodes 1-3 on x, two lines.

    The file's elements are tags 1 (nodes 1-2) and 2 (nodes 2-3).
    ``eids`` numbers them in the fem; PG ``Right`` is the second.
    """
    ids = np.array([1, 2, 3], dtype=np.int64)
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1,
                          npe=2, count=2)
    lines = ElementGroup(element_type=info,
                         ids=np.asarray(eids, dtype=np.int64),
                         connectivity=np.array([[1, 2], [2, 3]],
                                               dtype=np.int64))
    pgs = PhysicalGroupSet({(1, 7): {
        "name": "Right", "node_ids": ids[1:], "node_coords": xyz[1:],
        "element_ids": np.array([eids[1]], dtype=np.int64)}})
    return FEMData(
        nodes=NodeComposite(node_ids=ids, node_coords=xyz, physical=pgs,
                            labels=LabelSet({})),
        elements=ElementComposite(groups={1: lines}, physical=pgs,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=1, types=[info]),
        model_name="truss2d_from_fem",
    )


def _nodes_only(fem: FEMData) -> FEMData:
    """``fem``'s nodes and node groups, with no elements."""
    return FEMData(
        nodes=fem.nodes,
        elements=ElementComposite(groups={}, physical=PhysicalGroupSet({}),
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=len(fem.nodes.ids), n_elems=0, bandwidth=0,
                      types=[]),
    )


def _mismatch(caught: list) -> "list[str]":
    return [str(w.message) for w in caught
            if issubclass(w.category, ModelFemMismatchWarning)]


def test_from_fem_mpco_renumbered_elements_warn_and_bind_no_groups(
    tmp_path: Path,
) -> None:
    """The fem's element 1 is the capture's tag 7: warn, bind no group."""
    fem = _plate_fem()
    mpco = _write_mpco(tmp_path / "r.mpco", n_stages=1,
                       elements=[(7, [1, 2, 3, 4])])
    with pytest.warns(
        ModelFemMismatchWarning,
        match=r"fem= does not match.*1 renumbered \(e\.g\. 7 is fem element 1\)",
    ):
        r = Results.from_fem(fem, mpco, cache_root=tmp_path)
    with r:
        assert r.fem is not fem
        assert [int(e) for e in r.fem.elements.ids] == [7]
        assert r.fem.nodes.physical.names() == []
        assert r.fem.elements.physical.names() == []
        with pytest.raises(KeyError, match="No group named"):
            r.stage("stage_0").elements.gauss.get(
                pg="Plate", component="stress_xx",
            )


def test_from_fem_mpco_matching_elements_bind_silently(tmp_path: Path) -> None:
    """The capture's tag 1 on nodes 1-4 is the fem's element 1."""
    fem = _plate_fem()
    mpco = _write_mpco(tmp_path / "r.mpco", n_stages=1,
                       elements=[(1, [1, 2, 3, 4])])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelFemMismatchWarning)
        r = Results.from_fem(fem, mpco, cache_root=tmp_path)
    with r:
        assert r.fem is fem
        assert sorted(r.fem.nodes.physical.names()) == [
            "Left", "Plate", "Right",
        ]
        assert [int(e) for e in
                r.fem.elements.physical.element_ids("Plate")] == [1]


def test_explicit_fem_beside_a_broker_archive_is_checked(
    tmp_path: Path,
) -> None:
    """``from_mpco(fem=, model_h5=<fem.to_h5>)``: the same check, same outcome."""
    fem = _plate_fem()
    broker = tmp_path / "broker.h5"
    fem.to_h5(str(broker))
    mpco = _write_mpco(tmp_path / "r.mpco", n_stages=1,
                       elements=[(7, [1, 2, 3, 4])])
    with pytest.warns(ModelFemMismatchWarning, match=r"fem= does not match"):
        r = Results.from_mpco(mpco, model_h5=broker, fem=fem)
    with r:
        assert r.fem is not fem
        assert r.fem.nodes.physical.names() == []


def test_explicit_fem_beside_a_bridge_archive_keeps_priority(
    tmp_path: Path,
) -> None:
    """An archive with an element tag map relabels ops tags: ``fem=`` binds.

    The bridge archive pairs ops tags with fem element ids, so the
    capture's tag is not read as a fem id and there is nothing to check.
    """
    model_h5, fem = _write_model_h5(tmp_path / "model.h5")
    mpco = _write_mpco(tmp_path / "r.mpco", n_stages=2,
                       elements=[(7, [1, 2, 3, 4])])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelFemMismatchWarning)
        r = Results.from_mpco(mpco, model_h5=model_h5, fem=fem)
    with r:
        assert r.fem is fem


def test_from_fem_ladruno_renumbered_elements_warn_and_bind_no_groups(
    tmp_path: Path,
) -> None:
    """The ``.ladruno`` route (``from_ladruno``) runs the same check."""
    path = tmp_path / "truss2d.ladruno"
    shutil.copyfile(_TRUSS, path)
    fem = _truss_fem(eids=(13, 14))
    with pytest.warns(
        ModelFemMismatchWarning,
        match=(r"fem= does not match.*2 renumbered "
               r"\(e\.g\. 1 is fem element 13, 2 is fem element 14\)"),
    ):
        r = Results.from_fem(fem, path, cache_root=tmp_path)
    with r:
        assert r.fem is not fem
        assert sorted(int(e) for e in r.fem.elements.ids) == [1, 2]
        assert r.fem.elements.physical.names() == []
        with pytest.raises(KeyError, match="Right"):
            r.stage("stage_0").elements.get(pg="Right",
                                            component="basicForce")


def test_from_fem_ladruno_matching_elements_bind_silently(
    tmp_path: Path,
) -> None:
    """PG ``Right`` is element 2, and its read returns element 2."""
    path = tmp_path / "truss2d.ladruno"
    shutil.copyfile(_TRUSS, path)
    fem = _truss_fem()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelFemMismatchWarning)
        r = Results.from_fem(fem, path, cache_root=tmp_path)
    with r:
        assert r.fem is fem
        slab = r.stage("stage_0").elements.get(pg="Right",
                                               component="basicForce")
        assert [int(e) for e in slab.element_ids] == [2]


def _strip_conn(n: int) -> np.ndarray:
    """A row of ``n`` unit quads: bottom nodes ``1..n+1``, top ``n+2..2n+2``."""
    return np.array([[k, k + 1, n + 2 + k, n + 1 + k]
                     for k in range(1, n + 1)], dtype=np.int64)


def _strip_nodes(n: int) -> "tuple[np.ndarray, np.ndarray]":
    ids = np.arange(1, 2 * n + 3, dtype=np.int64)
    xyz = np.array([[float(i), 0.0, 0.0] for i in range(n + 1)]
                   + [[float(i), 1.0, 0.0] for i in range(n + 1)])
    return ids, xyz


def _strip_fem(eids: "list[int]") -> FEMData:
    """The strip of ``len(eids)`` quads numbered ``eids``; PG ``Strip``."""
    n = len(eids)
    ids, xyz = _strip_nodes(n)
    info = make_type_info(code=3, gmsh_name="Quadrangle 4", dim=2, order=1,
                          npe=4, count=n)
    quads = ElementGroup(element_type=info,
                         ids=np.asarray(eids, dtype=np.int64),
                         connectivity=_strip_conn(n))
    pgs = PhysicalGroupSet({(2, 1): {
        "name": "Strip", "node_ids": ids, "node_coords": xyz,
        "element_ids": np.asarray(eids, dtype=np.int64)}})
    return FEMData(
        nodes=NodeComposite(node_ids=ids, node_coords=xyz, physical=pgs,
                            labels=LabelSet({})),
        elements=ElementComposite(groups={3: quads}, physical=pgs,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=ids.size, n_elems=n, bandwidth=n + 2,
                      types=[info]),
    )


def _strip_mpco(
    tmp_path: Path, n: int, parts: "list[list[tuple[int, int]]]",
) -> Path:
    """One ``.mpco`` per part; a part is ``(ops_tag, element k)`` rows.

    Element ``k`` (1-based) is the ``k``-th quad of the strip; each part
    holds the nodes its elements use.  Returns the first part.
    """
    ids, xyz = _strip_nodes(n)
    conn = _strip_conn(n)
    paths = []
    for p, rows in enumerate(parts):
        nodes = np.unique(np.concatenate([conn[k - 1] for _, k in rows]))
        name = f"r.part-{p}.mpco" if len(parts) > 1 else "r.mpco"
        paths.append(_write_mpco(
            tmp_path / name, n_stages=1, node_ids=nodes,
            coords=xyz[np.searchsorted(ids, nodes)],
            elements=[(tag, [int(c) for c in conn[k - 1]]) for tag, k in rows],
        ))
    return paths[0]


_MISMATCHES = {
    # #1393's b0 at full size: fem 13..81 against the bridge's tags 2..70.
    "dense-renumbering": (list(range(13, 82)),
                          [[(t, t - 1) for t in range(2, 71)]],
                          r"11 renumbered \(e\.g\. 2 is fem element 13.*"
                          r"58 on other nodes"),
    "partial-id-overlap": ([1, 2, 3, 4],
                           [[(k + 2, k) for k in range(1, 5)]],
                           r"2 renumbered.*2 on other nodes"),
    "permuted-connectivity": ([1, 2, 3, 4],
                              [[(2, 1), (1, 2), (3, 3), (4, 4)]],
                              r"2 on other nodes than the fem element"),
    "one-renumbered-partition": ([1, 2, 3, 4],
                                 [[(1, 1), (2, 2)], [(13, 3), (14, 4)]],
                                 r"2 renumbered \(e\.g\. 13 is fem element 3"),
}


@pytest.mark.parametrize("case", sorted(_MISMATCHES))
def test_from_fem_element_mismatch_warns_and_binds_no_groups(
    tmp_path: Path, case: str,
) -> None:
    """Every pairing that would read the wrong elements by tag warns."""
    eids, parts, match = _MISMATCHES[case]
    fem = _strip_fem(eids)
    path = _strip_mpco(tmp_path, len(eids), parts)
    with pytest.warns(ModelFemMismatchWarning, match=match):
        r = Results.from_fem(fem, path, cache_root=tmp_path)
    with r:
        assert r.fem is not fem
        assert r.fem.elements.physical.names() == []


def test_an_extra_deck_element_is_not_a_mismatch(tmp_path: Path) -> None:
    """A deck element under a new tag (a spring, a link) is extra.

    MPCO ``MODEL/ELEMENTS`` lists the whole domain, so the capture holds
    the fem's element 1 under its own tag plus the deck's tag 99.  The
    PG reads are right: silent, the fem binds.
    """
    fem = _plate_fem()
    mpco = _write_mpco(tmp_path / "r.mpco", n_stages=1,
                       elements=[(1, [1, 2, 3, 4]), (99, [1, 2, 3, 4])])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelFemMismatchWarning)
        r = Results.from_fem(fem, mpco, cache_root=tmp_path)
    with r:
        assert r.fem is fem
        assert [int(e) for e in
                r.fem.elements.physical.element_ids("Plate")] == [1]


def test_a_matching_strip_binds_silently(tmp_path: Path) -> None:
    """The strip helpers pair when the tags are the fem's ids: silent.

    The control for the mismatch cases above: what warns there is the
    numbering, not the fixture.
    """
    fem = _strip_fem([1, 2, 3, 4])
    path = _strip_mpco(tmp_path, 4, [[(1, 1), (2, 2), (3, 3), (4, 4)]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelFemMismatchWarning)
        r = Results.from_fem(fem, path, cache_root=tmp_path)
    with r:
        assert r.fem is fem


@pytest.mark.parametrize("route", ["mpco", "ladruno"])
def test_a_fem_without_elements_binds_silently(
    tmp_path: Path, route: str,
) -> None:
    """No fem elements: no element group to mislabel, nothing to check."""
    if route == "mpco":
        fem = _nodes_only(_plate_fem())
        path = _write_mpco(tmp_path / "r.mpco", n_stages=1,
                           elements=[(7, [1, 2, 3, 4])])
    else:
        fem = _nodes_only(_truss_fem())
        path = tmp_path / "truss2d.ladruno"
        shutil.copyfile(_TRUSS, path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = Results.from_fem(fem, path, cache_root=tmp_path)
    with r:
        assert _mismatch(caught) == []
        assert r.fem is fem


def test_a_file_without_element_records_binds_silently(
    tmp_path: Path,
) -> None:
    """No capture elements: nothing is read by element tag."""
    fem = _plate_fem()
    mpco = _write_mpco(tmp_path / "r.mpco", n_stages=1, elements=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = Results.from_fem(fem, mpco, cache_root=tmp_path)
    with r:
        assert _mismatch(caught) == []
        assert r.fem is fem
        assert sorted(r.fem.nodes.physical.names()) == [
            "Left", "Plate", "Right",
        ]
