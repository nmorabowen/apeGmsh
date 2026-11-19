"""ADR 0128 D3, INV-6: the still manifest's numbers against Python's ``Results``.

``apeGmshViewer/src/spec/manifest.ts`` states, for every still, the range and
the extrema of the field shown (value, id and position), its no-data count and
the files read. An agent reports these numbers without looking at the picture,
so this file answers the same questions from the other side of the ADR 0112 D4
wall: it reads the committed fixtures through ``Results`` (or, for the
Gauss-to-node carries, through ``extrapolate_gauss_slab_*``), takes the
positions from ``FEMData`` and the hashes from ``h5py``, and owns
``apeGmshViewer/test/spec/manifest.oracle.json``.
``apeGmshViewer/test/spec/manifest.test.ts`` compares the app's entries with
that file. This test recomputes it and compares, so a regenerated fixture or a
change in ``Results`` fails here; run the file as a script to rewrite it::

    python tests/viewers/test_agent_manifest_oracle.py

The rules restated from the manifest's header, and not imported from the app:
a tie takes the lowest id; ``recorded`` covers every value the file holds at
the step; ``range_history`` is their extremes over the stage's steps; ``drawn`` keeps the visible primitives, and a solid draws only its
faces seen once; an element's position is the mean of its nodes.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import OpenSeesModel
from apeGmsh.results import Results
from apeGmsh.results._gauss_extrapolation import (
    extrapolate_gauss_slab_per_element,
    extrapolate_gauss_slab_to_nodes,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "apeGmshViewer" / "fixtures"
ORACLE = ROOT / "apeGmshViewer" / "test" / "spec" / "manifest.oracle.json"

REL_TOL = 1e-12

#: the sets the cases read: the model, its results, and whether the model
#: carries the bridge's /opensees zone (else it is FEM-only).
SETS: dict[str, dict[str, Any]] = {
    "contour_staged": {"opensees": True},
    "contour_nodata": {"opensees": True},
    "elem_solid": {"opensees": False},
    "elem_mixed": {"opensees": True},
    "extrap_solid": {"opensees": False},
}

#: ``stage`` by name, ``step`` as an index; ``gauss`` is the spec's mode.
#: ``erect`` has every element active; ``base`` has not activated ``Top`` yet
#: (``active``). ``isolate`` is the shot's own choice of groups.
CASES: list[dict[str, Any]] = [
    {"name": "nodal_axis", "set": "contour_staged", "stage": "erect", "step": 1, "field": "displacement", "component": "displacement_z"},
    {"name": "nodal_axis_first_step", "set": "contour_staged", "stage": "erect", "step": 0, "field": "displacement", "component": "displacement_x"},
    {"name": "nodal_magnitude", "set": "contour_staged", "stage": "erect", "step": 1, "field": "displacement", "component": "magnitude"},
    {"name": "nodal_isolated", "set": "contour_staged", "stage": "erect", "step": 1, "field": "displacement", "component": "displacement_x", "isolate": ["Col"]},
    {"name": "nodal_stage_inactive", "set": "contour_staged", "stage": "base", "step": 2, "field": "displacement", "component": "displacement_y", "active": ["Col", "Prop"]},
    {"name": "nodal_no_data", "set": "contour_nodata", "stage": "erect", "step": 1, "field": "displacement", "component": "displacement_x"},
    {"name": "element_mean", "set": "elem_solid", "stage": "load", "step": 1, "field": "stress", "component": "stress_xx", "gauss": "cell"},
    {"name": "element_cell_no_data", "set": "elem_mixed", "stage": "load", "step": 2, "field": "stress", "component": "stress_yy", "gauss": "cell"},
    {"name": "gauss_node_averaged", "set": "extrap_solid", "stage": "load", "step": 2, "field": "stress", "component": "stress_xx", "gauss": "node"},
    {"name": "gauss_node_discrete", "set": "extrap_solid", "stage": "load", "step": 2, "field": "stress", "component": "stress_xx", "gauss": "node_discrete"},
]

TET_FACES = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]
HEX_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def _results(name: str, fem: FEMData) -> Results:
    model = FIXTURES / f"{name}.h5"
    res = FIXTURES / f"{name}.results.h5"
    if SETS[name]["opensees"]:
        return Results.from_native(res, model=OpenSeesModel.from_h5(str(model)))
    return Results.from_fem(fem, res, kind="native")


def _files(name: str) -> dict[str, Any]:
    """The D3 file rows, from the file system and h5py."""
    out: dict[str, Any] = {}
    for role, path in (("model", FIXTURES / f"{name}.h5"), ("results", FIXTURES / f"{name}.results.h5")):
        with h5py.File(path, "r") as f:
            lin = f["meta/lineage"]
            row: dict[str, Any] = {"file": path.name, "size": path.stat().st_size}
            for key in ("model_hash", "fem_hash") + (("results_hash",) if role == "results" else ()):
                row[key] = _read_stamp(lin, key)
        out[role] = row
    return out


def _read_stamp(group: h5py.Group, key: str) -> str | None:
    if key in group.attrs:
        v = group.attrs[key]
    elif key in group:
        v = group[key][()]
    else:
        return None
    return v.decode() if isinstance(v, bytes) else str(v)


class _Drawn:
    """What the app draws of a model with every element visible (or those of ``isolate``)."""

    def __init__(self, fem: FEMData, isolate: list[str] | None) -> None:
        keep: set[int] | None = None
        if isolate:
            keep = {int(i) for pg in isolate for i in fem.elements.physical.element_ids(pg)}
        faces: dict[tuple[int, ...], list[tuple[int, list[int]]]] = {}
        nodes: set[int] = set()
        elements: set[int] = set()
        for grp in fem.elements:
            ids = np.asarray(grp.ids, dtype=np.int64)
            conn = np.asarray(grp.connectivity, dtype=np.int64)
            dim = grp.element_type.dim
            for eid, row in zip(ids.tolist(), conn.tolist()):
                if dim == 3:
                    table = TET_FACES if len(row) == 4 else HEX_FACES
                    for face in table:
                        verts = [row[i] for i in face]
                        faces.setdefault(tuple(sorted(verts)), []).append((eid, verts))
                elif keep is None or eid in keep:
                    elements.add(eid)
                    nodes.update(row)
        for seen in faces.values():
            if len(seen) != 1:
                continue
            eid, verts = seen[0]
            if keep is None or eid in keep:
                elements.add(eid)
                nodes.update(verts)
        self.nodes = nodes
        self.elements = elements


def _point(value: float, node: int | None, element: int | None, position: np.ndarray) -> dict[str, Any]:
    return {"value": float(value), "node": node, "element": element, "position": [float(x) for x in position]}


def _extrema(values: np.ndarray, rank: np.ndarray, describe) -> dict[str, Any] | None:
    """``values`` (one per candidate), ``rank`` its (element, node) tie order, lowest first; NaN is no data."""
    ok = np.flatnonzero(np.isfinite(values))
    if ok.size == 0:
        return None
    order = ok[np.lexsort((rank[ok, 1], rank[ok, 0]))]
    return {
        "min": describe(int(order[np.argmin(values[order])])),
        "max": describe(int(order[np.argmax(values[order])])),
    }


def _nodal_case(case: dict[str, Any], r: Results, fem: FEMData, drawn: _Drawn) -> dict[str, Any]:
    st = r.stage(case["stage"])
    k = case["step"]
    if case["component"] == "magnitude":
        slabs = [st.nodes.get(component=f"displacement_{a}", time=[k]) for a in "xyz"]
        ids = np.asarray(slabs[0].node_ids, dtype=np.int64)
        v = slabs[0].values[0] ** 2 + slabs[1].values[0] ** 2 + slabs[2].values[0] ** 2
        values = np.sqrt(v)
    else:
        slab = st.nodes.get(component=case["component"], time=[k])
        ids = np.asarray(slab.node_ids, dtype=np.int64)
        values = np.asarray(slab.values[0], dtype=np.float64)
    return _node_values(ids, values, fem, drawn)


def _node_values(ids: np.ndarray, values: np.ndarray, fem: FEMData, drawn: _Drawn) -> dict[str, Any]:
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    where = {int(i): n for n, i in enumerate(np.asarray(fem.nodes.ids, dtype=np.int64).tolist())}
    rank = np.stack([np.zeros_like(ids), ids], axis=1)
    describe = lambda i: _point(values[i], int(ids[i]), None, coords[where[int(ids[i])]])  # noqa: E731
    visible = np.array([int(i) in drawn.nodes for i in ids.tolist()])
    finite_at = {int(i) for i, x in zip(ids.tolist(), values.tolist()) if math.isfinite(x)}
    return {
        "recorded": _extrema(values, rank, describe),
        "drawn": _extrema(np.where(visible, values, np.nan), rank, describe),
        "no_data": {"kind": "nodes", "count": len(drawn.nodes - finite_at), "of": len(drawn.nodes)},
    }


def _centroid(fem: FEMData, element: int) -> np.ndarray:
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    where = {int(i): n for n, i in enumerate(np.asarray(fem.nodes.ids, dtype=np.int64).tolist())}
    for grp in fem.elements:
        hit = np.flatnonzero(np.asarray(grp.ids, dtype=np.int64) == element)
        if hit.size:
            row = np.asarray(grp.connectivity, dtype=np.int64)[hit[0]]
            return coords[[where[int(n)] for n in row]].mean(axis=0)
    raise KeyError(element)


def _element_case(case: dict[str, Any], r: Results, fem: FEMData, drawn: _Drawn) -> dict[str, Any]:
    slab = r.stage(case["stage"]).elements.gauss.get(component=case["component"], time=[case["step"]])
    mode = case["gauss"]
    if mode == "node":
        ids, nodal = extrapolate_gauss_slab_to_nodes(slab, fem)
        return _node_values(np.asarray(ids, dtype=np.int64), np.asarray(nodal[0], dtype=np.float64), fem, drawn)
    if mode == "node_discrete":
        per = extrapolate_gauss_slab_per_element(slab, fem)
        coords = np.asarray(fem.nodes.coords, dtype=np.float64)
        where = {int(i): n for n, i in enumerate(np.asarray(fem.nodes.ids, dtype=np.int64).tolist())}
        owner, node, values = [], [], []
        for eid, nodes, vals in zip(per.element_ids.tolist(), per.corner_node_ids, per.values):
            for n, v in zip(np.asarray(nodes).tolist(), np.asarray(vals)[0].tolist()):
                owner.append(eid)
                node.append(n)
                values.append(v)
        owner_a, node_a, values_a = np.array(owner), np.array(node), np.array(values, dtype=np.float64)
        rank = np.stack([owner_a, node_a], axis=1)
        describe = lambda i: _point(values_a[i], int(node_a[i]), int(owner_a[i]), coords[where[int(node_a[i])]])  # noqa: E731
        visible = np.array([e in drawn.elements and n in drawn.nodes for e, n in zip(owner, node)])
        with_value = {e for e, v in zip(owner, values) if math.isfinite(v)}
        return {
            "recorded": _extrema(values_a, rank, describe),
            "drawn": _extrema(np.where(visible, values_a, np.nan), rank, describe),
            "no_data": {"kind": "cells", "count": len(drawn.elements - with_value), "of": len(drawn.elements)},
        }
    # Flat: each element's one value, or the mean of its Gauss points.
    uniq, inv = np.unique(np.asarray(slab.element_index, dtype=np.int64), return_inverse=True)
    values = np.bincount(inv, weights=np.asarray(slab.values[0], dtype=np.float64)) / np.bincount(inv)
    rank = np.stack([uniq, np.zeros_like(uniq)], axis=1)
    describe = lambda i: _point(values[i], None, int(uniq[i]), _centroid(fem, int(uniq[i])))  # noqa: E731
    visible = np.array([int(e) in drawn.elements for e in uniq.tolist()])
    with_value = {int(e) for e, v in zip(uniq.tolist(), values.tolist()) if math.isfinite(v)}
    return {
        "recorded": _extrema(values, rank, describe),
        "drawn": _extrema(np.where(visible, values, np.nan), rank, describe),
        "no_data": {"kind": "cells", "count": len(drawn.elements - with_value), "of": len(drawn.elements)},
    }


def compute() -> dict[str, Any]:
    """The oracle, from the fixtures. ``Results.from_fem`` caches a model under the cwd, so this runs in a temporary one."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            fems = {name: FEMData.from_h5(str(FIXTURES / f"{name}.h5")) for name in SETS}
            results = {name: _results(name, fems[name]) for name in SETS}
            out: dict[str, Any] = {"sets": {name: _files(name) for name in SETS}, "cases": []}
            for case in CASES:
                name = case["set"]
                fem = fems[name]
                r = results[name]
                drawn = _Drawn(fem, case.get("isolate") or case.get("active"))
                reader = _element_case if "gauss" in case else _nodal_case
                body = reader(case, r, fem, drawn)
                stage = next(s for s in r.stages if s.name == case["stage"])
                # The whole-history range: the recorded extremes over every step of the stage.
                ends = [reader({**case, "step": k}, r, fem, drawn)["recorded"] for k in range(stage.n_steps)]
                body["range_history"] = {
                    "min": min(e["min"]["value"] for e in ends if e),
                    "max": max(e["max"]["value"] for e in ends if e),
                }
                with h5py.File(FIXTURES / f"{name}.results.h5", "r") as f:
                    t = float(np.asarray(f[f"stages/{stage.id}/time"][...], dtype=np.float64)[case["step"]])
                out["cases"].append({**case, "time": t, **body})
            return out
        finally:
            os.chdir(cwd)


def _assert_same(got: Any, want: Any, path: str = "") -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            _assert_same(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{path}: length"
        for i, (a, b) in enumerate(zip(got, want)):
            _assert_same(a, b, f"{path}[{i}]")
    elif isinstance(want, float):
        assert math.isclose(got, want, rel_tol=REL_TOL, abs_tol=REL_TOL), f"{path}: {got} != {want}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def test_manifest_oracle_is_current() -> None:
    assert ORACLE.exists(), f"{ORACLE} is missing: run this file as a script to write it"
    _assert_same(compute(), json.loads(ORACLE.read_text(encoding="utf-8")))


def test_cases_cover_every_kind_of_holder() -> None:
    """The oracle must not go vacuous: a tie-free extremum per holder kind, and a real no-data count."""
    by_name = {c["name"]: c for c in compute()["cases"]}
    assert by_name["nodal_axis"]["recorded"]["max"]["node"] is not None
    assert by_name["element_mean"]["recorded"]["max"]["element"] is not None
    point = by_name["gauss_node_discrete"]["recorded"]["max"]
    assert point["node"] is not None and point["element"] is not None
    assert by_name["nodal_no_data"]["no_data"]["count"] > 0
    assert by_name["element_cell_no_data"]["no_data"]["count"] > 0
    # Isolating a group, or a group not yet active, leaves fewer drawn nodes than the whole model shows.
    assert by_name["nodal_isolated"]["no_data"]["of"] < by_name["nodal_axis"]["no_data"]["of"]
    assert by_name["nodal_stage_inactive"]["no_data"]["of"] < by_name["nodal_axis"]["no_data"]["of"]


if __name__ == "__main__":
    ORACLE.write_text(json.dumps(compute(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {ORACLE}")
