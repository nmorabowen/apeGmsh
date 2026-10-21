"""#2329 — the fork's default route reads by ``pg=`` and ``label=``.

``spec.emit_default`` on the fork writes a ``.ladruno`` and opens it with
``Results.from_ladruno(model_h5=...)``. The file's ``MODEL`` group has no
physical groups or labels; the FEMData archived in ``model_h5`` does, and
the route must bind it, as ``from_mpco`` does.

The oracle is the same model and load run again through the stock route
(the fork module with its fork signal hidden writes a ``.mpco``, read with
``from_mpco``): every ``pg=`` and ``label=`` read of the fork run equals
the MPCO read at ``rtol=1e-12``, and the tip equals ``P L / E A``. One
build cannot write both formats in one run, so the model runs twice.

A raw deck over a FEMData-only archive (``fem.to_h5``, no bridge, no
``ndm``) is read by group too; its oracle is the live domain's
displacement at every step, and the tip's closed form.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
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
from apeGmsh.results import Results
from apeGmsh.results.live import DefaultRouteWarning
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.test_results_live_default_format import (
    TRUSS_A,
    TRUSS_E,
    TRUSS_L,
    TRUSS_P,
    TRUSS_STEPS,
    TRUSS_TIP,
    truss_bridge,
)

pytestmark = pytest.mark.ladruno_fork

_LABEL = "truss.tip"
_DISP = ("displacement_x", "displacement_y")


class StockView:
    """The fork module with its fork signal hidden: ``kind == "stock"``
    (as in ``test_default_format_ladruno_live.py``)."""

    _HIDDEN = frozenset({"ladrunoBuild", "ladrunoSchema"})

    def __init__(self, ops: Any) -> None:
        self._ops = ops

    def __getattr__(self, name: str) -> Any:
        if name in StockView._HIDDEN:
            raise AttributeError(name)
        return getattr(self._ops, name)


def _fem() -> FEMData:
    """The two-node truss with its physical groups and one node label."""
    node_ids = np.array([1, 2], dtype=np.int64)
    coords = np.array([[0.0, 0.0, 0.0], [TRUSS_L, 0.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=1)
    group = ElementGroup(element_type=info, ids=np.array([1], dtype=np.int64),
                         connectivity=np.array([[1, 2]], dtype=np.int64))
    pg = PhysicalGroupSet({
        (1, 1): {"name": "Bar", "node_ids": node_ids, "node_coords": coords,
                 "element_ids": group.ids},
        (0, 2): {"name": "Base", "node_ids": node_ids[:1],
                 "node_coords": coords[:1]},
        (0, 3): {"name": "Tip", "node_ids": node_ids[1:],
                 "node_coords": coords[1:]},
    })
    labels = LabelSet({(0, 10): {"name": _LABEL, "node_ids": node_ids[1:],
                                 "node_coords": coords[1:]}})
    return FEMData(
        nodes=NodeComposite(node_ids=node_ids, node_coords=coords,
                            physical=pg, labels=labels),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=labels),
        info=MeshInfo(n_nodes=2, n_elems=1, bandwidth=2, types=[info]),
        model_name="default_route_groups_truss",
    )


def _spec(fem: FEMData) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="all", components=_DISP,
            dt=None, n_steps=None, node_ids=np.array([1, 2]),
        ),),
    )


def _run(tmp_path: Path, *, stock: bool) -> "tuple[Any, list, float]":
    """One run of the truss on the default route, fork or stock."""
    from apeGmsh.opensees._target import backend_info_of
    from apeGmsh.opensees.emitter.live import get_ops

    opspy = get_ops()
    assert backend_info_of(opspy).kind == "fork"
    ops = StockView(opspy) if stock else opspy
    fem = _fem()
    bridge = truss_bridge(fem)
    tmp_path.mkdir()
    model = tmp_path / "model.h5"
    bridge.h5(str(model))
    bridge.run()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _spec(fem).emit_default(tmp_path / "run", model_h5=model,
                                         ops=ops) as run:
                for _ in range(TRUSS_STEPS):
                    assert opspy.analyze(1) == 0
        tip = float(opspy.nodeDisp(2, 1))
    finally:
        opspy.wipe()
    return run, caught, tip


def _read(res: Results, component: str, **sel: str) -> "tuple[np.ndarray, np.ndarray]":
    s = res.stage(res.stages[0].id).nodes.get(component=component, **sel)
    return np.asarray(s.node_ids), np.asarray(s.values)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Any]":
    tmp = tmp_path_factory.mktemp("default_route_groups")
    fork, fork_caught, fork_tip = _run(tmp / "fork", stock=False)
    stock, stock_caught, stock_tip = _run(tmp / "stock", stock=True)
    return {"fork": fork, "fork_caught": fork_caught, "fork_tip": fork_tip,
            "stock": stock, "stock_caught": stock_caught}


def test_the_two_routes_ran(runs) -> None:
    assert runs["fork"].route == "fork" and runs["fork_caught"] == []
    assert runs["fork"].path.suffix == ".ladruno" and runs["fork"].path.exists()
    assert runs["stock"].route == "mpco"
    assert [w.category for w in runs["stock_caught"]] == [DefaultRouteWarning]
    np.testing.assert_allclose(runs["fork_tip"], TRUSS_TIP, rtol=1e-12)


@pytest.mark.parametrize(
    "sel, nodes",
    [({"pg": "Tip"}, [2]), ({"pg": "Bar"}, [1, 2]), ({"label": _LABEL}, [2])],
    ids=["pg-Tip", "pg-Bar", "label"],
)
def test_group_reads_equal_the_mpco_route(runs, sel, nodes) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = runs["fork"].results()
        ref = runs["stock"].results()
    try:
        for comp in _DISP:
            ids, vals = _read(res, comp, **sel)
            ref_ids, ref_vals = _read(ref, comp, **sel)
            np.testing.assert_array_equal(ids, nodes)
            np.testing.assert_array_equal(ids, ref_ids)
            assert vals.shape == (TRUSS_STEPS, len(nodes))
            np.testing.assert_allclose(vals, ref_vals, rtol=1e-12, atol=0.0)
        ids, ux = _read(res, "displacement_x", **sel)
        tip_col = list(ids).index(2)
        np.testing.assert_allclose(ux[-1, tip_col], TRUSS_TIP, rtol=1e-12)
    finally:
        res.close()
        ref.close()


# ---------------------------------------------------------------------------
# A raw deck over a FEMData-only archive (fem.to_h5): no bridge, no ndm
# ---------------------------------------------------------------------------


def _raw_deck(ops: Any) -> None:
    """The truss as a hand-built deck, tagged by FEM id (no bridge)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, TRUSS_L, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, TRUSS_E)
    ops.element("Truss", 1, 1, 2, TRUSS_A, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, TRUSS_P, 0.0)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0 / TRUSS_STEPS)
    ops.analysis("Static")


@pytest.mark.parametrize(
    "sel, nodes",
    [({"pg": "Tip"}, [2]), ({"pg": "Bar"}, [1, 2]), ({"label": _LABEL}, [2])],
    ids=["pg-Tip", "pg-Bar", "label"],
)
def test_a_raw_deck_over_a_femdata_archive_reads_by_group(
    tmp_path, sel, nodes,
) -> None:
    """``emit_default(model_h5=fem.to_h5(...))`` on the fork: the archive
    records no ndm, the file's dimension (2) drives the check, and each
    group read equals the live domain's displacements at every step."""
    from apeGmsh.opensees._target import backend_info_of
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    assert backend_info_of(ops).kind == "fork"
    fem = _fem()
    archive = tmp_path / "fem.h5"
    fem.to_h5(str(archive))
    _raw_deck(ops)
    live = np.zeros((TRUSS_STEPS, 2, 2))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _spec(fem).emit_default(tmp_path / "run", model_h5=archive,
                                         ops=ops) as run:
                for k in range(TRUSS_STEPS):
                    assert ops.analyze(1) == 0
                    for j, n in enumerate((1, 2)):
                        live[k, :, j] = [ops.nodeDisp(n, d) for d in (1, 2)]
    finally:
        ops.wipe()
    assert run.route == "fork" and caught == []
    # A FEMData archive carries no model_hash, so the fork file cannot be
    # stamped: that is the one warning (no ModelFemMismatchWarning).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run.results()
    assert [w.category for w in caught] == [DefaultRouteWarning], [
        str(w.message) for w in caught]
    assert "unstamped" in str(caught[0].message)
    try:
        for d, comp in enumerate(_DISP):
            ids, vals = _read(res, comp, **sel)
            np.testing.assert_array_equal(ids, nodes)
            cols = [n - 1 for n in nodes]
            np.testing.assert_allclose(vals, live[:, d, cols], rtol=1e-12,
                                       atol=0.0)
        ids, ux = _read(res, "displacement_x", **sel)
        np.testing.assert_allclose(ux[-1, list(ids).index(2)], TRUSS_TIP,
                                   rtol=1e-12)
    finally:
        res.close()
