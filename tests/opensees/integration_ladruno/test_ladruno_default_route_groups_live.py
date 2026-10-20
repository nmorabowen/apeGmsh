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
    TRUSS_L,
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
