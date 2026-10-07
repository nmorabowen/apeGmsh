"""An element whose connectivity repeats a node is refused before any write (#1536).

The ``"duplicate node tags"`` refusal lived only in the dead
``emit_element_spec`` that K1-3d S6 (#1546) deleted, so the live element
fan-out wrote ``element Truss 3 3 3 ...`` into the deck. Nothing upstream
refuses such a cell: the scope probe built a real :class:`FEMData` whose
line block repeats a node and it passed construction, ``select()``,
``apeSees.build()`` and every emitter. The refusal now lives in the tag
plan (:func:`apeGmsh.opensees._internal.tag_plan.check_distinct_nodes`),
which every emit path makes before it writes an element.

The oracle is the refusal itself: the first emit raises
:class:`BridgeError` naming the element class, PG, planned tag, FEM id
and the repeated node; ``ops.tcl`` / ``ops.py`` / ``ops.h5`` leave no
file on disk; a healthy model still emits its elements; and with the
check reverted every refusal test fails (the PR body says so).
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
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

_COORDS = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
], dtype=np.float64)

_LINE = dict(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2)
_TRI = dict(code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3)
_QUAD = dict(code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4)


def _fem(*blocks: tuple[dict[str, Any], list[list[int]]], pg: str = "Cells") -> FEMData:
    """A real FEMData: four nodes and the given ``(type, connectivity)`` blocks under one PG."""
    node_ids = np.arange(1, len(_COORDS) + 1, dtype=np.int64)
    groups = {}
    types = []
    all_eids = []
    next_eid = 1
    for kind, conn in blocks:
        conn_arr = np.asarray(conn, dtype=np.int64)
        eids = np.arange(next_eid, next_eid + len(conn_arr), dtype=np.int64)
        next_eid += len(conn_arr)
        info = make_type_info(count=len(conn_arr), **kind)
        groups[int(kind["code"])] = ElementGroup(
            element_type=info, ids=eids, connectivity=conn_arr)
        types.append(info)
        all_eids.append(eids)
    eids_all = np.concatenate(all_eids)
    pgs = {(max(k["dim"] for k, _ in blocks), 100): {
        "name": pg, "node_ids": node_ids, "node_coords": _COORDS,
        "element_ids": eids_all,
    }}
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=_COORDS,
        physical=PhysicalGroupSet(pgs), labels=LabelSet({}))
    elements = ElementComposite(
        groups=groups, physical=PhysicalGroupSet(pgs), labels=LabelSet({}))
    info = MeshInfo(
        n_nodes=len(node_ids), n_elems=len(eids_all), bandwidth=1, types=types)
    return FEMData(nodes=nodes, elements=elements, info=info)


def _truss(fem: FEMData) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6)
    ops.element.Truss(pg="Cells", A=0.01, material=mat)
    return ops


def _quad(fem: FEMData) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1.0e6, nu=0.3)
    ops.element.FourNodeQuad(pg="Cells", thickness=0.1, material=mat)
    return ops


def _emit(ops: apeSees, emitter: Any) -> Any:
    bm = ops.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm.emit(emitter)
    return emitter


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


# The element spec itself is a registered primitive and takes element tag
# 1, so the fan-out's first planned tag is 2 (the probe's deck read
# ``element Truss 2 1 2 ...`` then ``element Truss 3 3 3 ...``).


def test_truss_with_a_repeated_node_is_refused_naming_tag_type_and_node() -> None:
    ops = _truss(_fem((_LINE, [[1, 2], [3, 3]])))
    # The build succeeds: the refusal is the plan's, made on the first emit.
    ops.build()
    with pytest.raises(BridgeError) as info:
        _emit(ops, RecordingEmitter())
    msg = str(info.value)
    assert msg.startswith("Truss(pg='Cells'): element 3 (FEM element 2) repeats node 3")
    assert "(3, 3)" in msg
    assert "must be distinct" in msg


def test_collapsed_quad_names_the_one_repeated_node() -> None:
    ops = _quad(_fem((_QUAD, [[1, 2, 3, 4], [1, 2, 3, 3]])))
    with pytest.raises(
        BridgeError,
        match=r"^FourNodeQuad\(pg='Cells'\): element 3 \(FEM element 2\) "
              r"repeats node 3 in its connectivity \(1, 2, 3, 3\)",
    ):
        _emit(ops, RecordingEmitter())


def test_a_row_repeating_two_nodes_names_both() -> None:
    ops = _quad(_fem((_QUAD, [[1, 1, 2, 2]])))
    with pytest.raises(BridgeError, match=r"element 2 \(FEM element 1\) repeats nodes 1, 2 "):
        _emit(ops, RecordingEmitter())


def test_mixed_npe_fanout_is_checked_row_by_row() -> None:
    # A PG holding lines and a triangle fans out as an object-dtype
    # connectivity; the repeated node sits in the second block.
    ops = _truss(_fem((_LINE, [[1, 2]]), (_TRI, [[2, 3, 2]])))
    with pytest.raises(
        BridgeError,
        match=r"Truss\(pg='Cells'\): element 3 \(FEM element 2\) repeats node 2 "
              r"in its connectivity \(2, 3, 2\)",
    ):
        _emit(ops, RecordingEmitter())


# ---------------------------------------------------------------------------
# Nothing is written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["tcl", "py", "h5"])
def test_a_refused_build_writes_no_deck_or_archive(route: str, tmp_path: Path) -> None:
    ops = _truss(_fem((_LINE, [[1, 2], [3, 3]])))
    ext = {"tcl": "tcl", "py": "py", "h5": "h5"}[route]
    path = tmp_path / f"model.{ext}"
    with pytest.raises(BridgeError, match=r"repeats node 3"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(ops, route)(str(path))
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_tcl_stream_mode_leaves_nothing_behind(tmp_path: Path) -> None:
    ops = _truss(_fem((_LINE, [[1, 2], [3, 3]])))
    path = tmp_path / "model.tcl"
    with pytest.raises(BridgeError, match=r"repeats node 3"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(path), stream=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------


def test_distinct_connectivity_still_emits_every_element() -> None:
    em = _emit(_truss(_fem((_LINE, [[1, 2], [3, 4]]))), TclEmitter())
    assert [ln for ln in em.lines() if ln.startswith("element ")] == [
        "element Truss 2 1 2 0.01 1",
        "element Truss 3 3 4 0.01 1",
    ]
