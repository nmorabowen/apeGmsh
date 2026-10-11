"""X2-e — numeric oracle: the DATA the old path draws (#1684).

Maintainer ruling 2026-10-11 (#1663/#1665): X2 accepts the session
path's *look* and re-baselines the image goldens from it. What must not
change is the data drawn. This module pins, from the OLD path
(``ResultsDirector`` stills, and the Qt ``ResultsViewer`` for
``export_animation`` in ``test_x2_numeric_oracle_anim.py``), the numbers
in the scene at the moment each frame is shot, and commits them under
``tests/viewers/golden_numeric/`` (``oracle.npz`` + ``manifest.json``).
The oracle outlives X3 (the director's deletion).

The seam (path-agnostic, no ``src/`` hook)
------------------------------------------
Every still / frame path ends in ``pyvista``'s
``BasePlotter.screenshot``: ``render_results`` and ``render_still``
(session) through ``viewers/_stills._screenshot``, and both animation
loops through ``plotter.screenshot(return_img=True)``. The figure in
``render_pack`` / ``assess`` ends in matplotlib's ``Figure.savefig``.
:func:`capture_drawn` wraps those two calls for the duration of a
``with`` block and, just before each shot, reads the scene with
:func:`extract_drawn`:

* ``points``: the union of the points of every *visible* dataset actor
  that is not a glyph (substrate, wireframe, outline, contour), merged
  within ``X_ATOL`` and sorted; for a deformed view these are the
  deformed coordinates actually drawn;
* ``contour``: the one actor that maps point scalars to colour: its
  points, the scalar array and the colour range it maps with (``clim``:
  the mapper's scalar range, or its LUT's when the mapper defers);
* ``arrows``: per glyph instance (``vtkGlyph3D`` output), the anchor
  (centroid of the instance's tail cross-section, i.e. the node the
  arrow starts at), ``GlyphVector`` (the vector glyphed) and ``_size``
  (its drawn length);
* for a matplotlib figure, the ``(x, y)`` data of every line.

Nothing here knows which path built the scene, so X2-b / X2-d point the
SAME tests at the session path by changing ``Results`` internals only:
neither this file nor the ``.npz`` is edited. A path that stops ending
in ``BasePlotter.screenshot`` captures nothing and the oracle fails
closed (frame-count / missing-record assertions), it never passes
vacuously.

Comparison
----------
Points are matched as sets (each golden point to its nearest drawn
point, the matching required to be one-to-one), because two paths may
order the same nodes differently. Tolerances are float noise, not visual
tolerance: VTK stores points and scalars as float32 (eps 1.2e-7) and the
fixture's values are O(1) (coordinates <= 4.1, displacements <= 0.6,
reactions <= 3.6), so float32 round-off is <= 5e-7. ``X_ATOL = 1e-5``
(coordinates), ``V_ATOL = 1e-6`` + ``RTOL = 1e-5`` (scalars, clim,
vectors). The smallest regression the revert-proof makes (a 1% deform
scale, a 1-step shift) moves values by >= 2e-3, two orders above.

The goldens are *also* checked against physics, without rendering:
each recorded frame must equal ``reference + scale * u(step)`` for the
step and scale the manifest names (``_predicted_points``), and a
recorded contour must equal the recorded field at that step. So a
golden is not merely self-consistent: it is the fixture's data.

Regenerate ONLY from the old path, only if the drawn data is meant to
change: ``APEGMSH_REGEN_NUMERIC=1 pytest <this file>`` and
``... test_x2_numeric_oracle_anim.py`` (qt). The two writers merge into
the same ``oracle.npz``.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

from apeGmsh.results import Results
from tests.viewers import test_x2_golden_stills as G

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_numeric"
NPZ_PATH = GOLDEN_DIR / "oracle.npz"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
REGEN = bool(os.environ.get("APEGMSH_REGEN_NUMERIC"))

X_ATOL = 1e-5
V_ATOL = 1e-6
RTOL = 1e-5

_WINDOW = tuple(G._MANIFEST["window"])
_CAMERA = G._MANIFEST["camera"]


# =====================================================================
# The seam: read the drawn data out of whatever scene a path built
# =====================================================================


def _merge_points(pts: np.ndarray, tol: float = X_ATOL) -> np.ndarray:
    """Distinct rows of ``pts`` (merged within ``tol``), lexsorted."""
    kept: list[np.ndarray] = []
    for p in np.asarray(pts, dtype=np.float64).reshape(-1, 3):
        if not any(np.max(np.abs(p - q)) <= tol for q in kept):
            kept.append(p)
    if not kept:
        return np.zeros((0, 3))
    out = np.array(kept)
    return out[np.lexsort((out[:, 2], out[:, 1], out[:, 0]))]


def _glyph_instances(ds: Any) -> dict[str, np.ndarray]:
    """Anchor / vector / size per instance of a ``vtkGlyph3D`` output.

    Instances are contiguous blocks of equal size; the block size is the
    gcd of the runs of constant ``(GlyphVector, _size)``. The anchor is
    the centroid of the instance's points at its minimum axial
    coordinate (an arrow's tail disc is centred on the node). Anything
    else (ragged blocks, a non-constant block) raises: fail closed.
    """
    pts = np.array(ds.points, dtype=np.float64)
    vec = np.array(ds.point_data["GlyphVector"], dtype=np.float64)
    size = (
        np.array(ds.point_data["_size"], dtype=np.float64).reshape(-1)
        if "_size" in ds.point_data else np.ones(len(pts))
    )
    key = np.column_stack([vec, size])
    change = np.flatnonzero(np.any(np.diff(key, axis=0) != 0, axis=1)) + 1
    bounds = np.concatenate([[0], change, [len(pts)]])
    block = 0
    for run in np.diff(bounds):
        block = math.gcd(block, int(run))
    if block == 0 or len(pts) % block:
        raise AssertionError(f"glyph output of {len(pts)} points is ragged")
    anchors, vectors, sizes = [], [], []
    for start in range(0, len(pts), block):
        sl = slice(start, start + block)
        if np.ptp(key[sl], axis=0).max() != 0:
            raise AssertionError("glyph instance with non-constant data")
        v = vec[start]
        n = float(np.linalg.norm(v))
        p = pts[sl]
        if n == 0.0:
            anchors.append(p.mean(axis=0))
        else:
            axial = p @ (v / n)
            span = float(np.ptp(axial)) or 1.0
            tail = axial <= axial.min() + 1e-6 * span
            anchors.append(p[tail].mean(axis=0))
        vectors.append(v)
        sizes.append(size[start])
    return {
        "origins": np.array(anchors),
        "vectors": np.array(vectors),
        "sizes": np.array(sizes),
    }


def _mapped_range(mapper: Any) -> np.ndarray:
    if mapper.GetUseLookupTableScalarRange():
        return np.array(mapper.GetLookupTable().GetRange(), dtype=np.float64)
    return np.array(mapper.GetScalarRange(), dtype=np.float64)


def extract_drawn(plotter: Any) -> dict[str, Any]:
    """The drawn data of a pyvista plotter's scene (see module doc).

    Every array is COPIED (``np.array``): a VTK array can share memory
    with a diagram buffer that the next step overwrites in place (the
    contour's ``_scalar_values``), so a view would make every captured
    frame read as the last step drawn.
    """
    mesh_pts: list[np.ndarray] = []
    contours: list[dict[str, Any]] = []
    glyphs: list[dict[str, np.ndarray]] = []
    for renderer in plotter.renderers:
        for name, actor in renderer.actors.items():
            if not actor.GetVisibility() or actor.IsA("vtkActor2D"):
                continue  # hidden, or 2-D overlay (scalar bar, text)
            mapper = getattr(actor, "mapper", None)
            ds = getattr(mapper, "dataset", None) if mapper is not None else None
            if ds is None:
                raise AssertionError(
                    f"visible 3-D prop {name!r} ({actor.GetClassName()}) "
                    "has no dataset the oracle can read (fail closed)",
                )
            if ds.n_points == 0:
                continue
            if "GlyphVector" in ds.point_data:
                glyphs.append(_glyph_instances(ds))
                continue
            pts = np.array(ds.points, dtype=np.float64)
            mesh_pts.append(pts)
            if not mapper.GetScalarVisibility():
                continue
            name = mapper.GetArrayName() or ds.active_scalars_name
            if name in ds.point_data:
                contours.append({
                    "name": str(name),
                    "points": pts,
                    "scalars": np.array(
                        ds.point_data[name], dtype=np.float64),
                    "clim": _mapped_range(mapper),
                })
            elif name:
                raise AssertionError(
                    f"colour-mapped actor with non-point scalars {name!r}: "
                    "the oracle reads nodal contours only (fail closed)",
                )
    if len(contours) > 1:
        raise AssertionError(
            f"{len(contours)} colour-mapped actors in one frame; the oracle "
            "expects at most one contour (fail closed)",
        )
    arrows = None
    if glyphs:
        arrows = {k: np.concatenate([g[k] for g in glyphs]) for k in glyphs[0]}
    return {
        "kind": "vtk",
        "points": _merge_points(np.concatenate(mesh_pts))
        if mesh_pts else np.zeros((0, 3)),
        "contour": contours[0] if contours else None,
        "arrows": arrows,
    }


def _figure_lines(fig: Any) -> dict[str, Any]:
    lines = [
        (np.array(ln.get_xdata(), dtype=np.float64),
         np.array(ln.get_ydata(), dtype=np.float64))
        for ax in fig.axes for ln in ax.lines
    ]
    return {"kind": "mpl", "lines": lines}


def _shot_name(target: Any) -> str | None:
    if target is None or isinstance(target, bool):
        return None
    name = Path(str(target)).name
    return name.replace(".partial", "")


@contextlib.contextmanager
def capture_drawn() -> Iterator[list[dict[str, Any]]]:
    """Record the drawn data of every shot taken inside the block.

    Yields a list that fills with one record per
    ``BasePlotter.screenshot`` (``extract_drawn``) or ``Figure.savefig``
    (line data), in call order, each with ``"name"``: the file name it
    was shot to (``.partial`` stripped), or ``None`` for an in-memory
    frame.
    """
    from matplotlib.figure import Figure
    from pyvista.plotting.plotter import BasePlotter

    records: list[dict[str, Any]] = []
    shot = BasePlotter.screenshot
    save = Figure.savefig

    def _screenshot(self: Any, filename: Any = None, *a: Any, **k: Any) -> Any:
        rec = extract_drawn(self)
        rec["name"] = _shot_name(filename)
        records.append(rec)
        return shot(self, filename, *a, **k)

    def _savefig(self: Any, fname: Any, *a: Any, **k: Any) -> Any:
        rec = _figure_lines(self)
        rec["name"] = _shot_name(fname)
        records.append(rec)
        return save(self, fname, *a, **k)

    BasePlotter.screenshot = _screenshot  # type: ignore[method-assign]
    Figure.savefig = _savefig  # type: ignore[method-assign]
    try:
        yield records
    finally:
        BasePlotter.screenshot = shot  # type: ignore[method-assign]
        Figure.savefig = save  # type: ignore[method-assign]


# =====================================================================
# Golden store: flat npz keys "<record>|<field>"
# =====================================================================


def flatten(key: str, rec: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if rec["kind"] == "mpl":
        for i, (x, y) in enumerate(rec["lines"]):
            out[f"{key}|line{i}_x"] = x
            out[f"{key}|line{i}_y"] = y
        return out
    out[f"{key}|points"] = rec["points"]
    if rec["contour"] is not None:
        c = rec["contour"]
        order = np.lexsort(c["points"].T[::-1])
        out[f"{key}|contour_points"] = c["points"][order]
        out[f"{key}|contour_scalars"] = c["scalars"][order]
        out[f"{key}|clim"] = c["clim"]
    if rec["arrows"] is not None:
        a = rec["arrows"]
        order = np.lexsort(a["origins"].T[::-1])
        for f in ("origins", "vectors", "sizes"):
            out[f"{key}|arrow_{f}"] = a[f][order]
    return out


def load_golden() -> dict[str, np.ndarray]:
    with np.load(NPZ_PATH) as z:
        return {k: z[k] for k in z.files}


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text("utf-8"))


def write_golden(entries: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    """Merge ``entries`` (npz) and ``meta`` (manifest sections) in place."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    store = load_golden() if NPZ_PATH.exists() else {}
    prefixes = {k.split("|")[0] for k in entries}
    store = {k: v for k, v in store.items() if k.split("|")[0] not in prefixes}
    store.update(entries)
    np.savez_compressed(NPZ_PATH, **dict(sorted(store.items())))
    man = load_manifest() if MANIFEST_PATH.exists() else {}
    man.update(meta)
    MANIFEST_PATH.write_text(
        json.dumps(man, indent=2, sort_keys=True) + "\n", "utf-8",
    )


# =====================================================================
# Comparison
# =====================================================================


def _match(actual: np.ndarray, golden: np.ndarray, what: str) -> np.ndarray:
    """Index into ``actual`` for each golden point; one-to-one, <= X_ATOL."""
    if actual.shape != golden.shape:
        raise AssertionError(
            f"{what}: drawn {actual.shape[0]} points, golden {golden.shape[0]}")
    if golden.size == 0:
        return np.zeros(0, dtype=np.int64)
    d = np.abs(golden[:, None, :] - actual[None, :, :]).max(axis=2)
    idx = d.argmin(axis=1)
    worst = float(d[np.arange(len(golden)), idx].max())
    if worst > X_ATOL or len(set(idx.tolist())) != len(idx):
        raise AssertionError(
            f"{what}: drawn points differ from the golden "
            f"(worst {worst:.3g} > X_ATOL {X_ATOL:g})")
    return idx


def _close(a: np.ndarray, b: np.ndarray, what: str) -> None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.allclose(a, b, rtol=RTOL, atol=V_ATOL):
        worst = float(np.abs(a - b).max()) if a.shape == b.shape else math.inf
        raise AssertionError(
            f"{what}: values differ from the golden (worst {worst:.3g})")


def assert_record(rec: dict[str, Any], key: str, gold: dict[str, np.ndarray]) -> None:
    """Fail unless ``rec`` draws the data recorded under ``key``."""
    exp = {k.split("|", 1)[1]: v for k, v in gold.items()
           if k.split("|", 1)[0] == key}
    if not exp:
        raise AssertionError(f"no golden record {key!r}")
    if rec["kind"] == "mpl":
        got = flatten(key, rec)
        got = {k.split("|", 1)[1]: v for k, v in got.items()}
        if sorted(got) != sorted(exp):
            raise AssertionError(f"{key}: lines {sorted(got)} != {sorted(exp)}")
        for f in exp:
            _close(got[f], exp[f], f"{key} {f}")
        return
    _match(rec["points"], exp["points"], f"{key} points")
    c = rec["contour"]
    if ("contour_scalars" in exp) != (c is not None):
        raise AssertionError(
            f"{key}: contour {'missing' if c is None else 'unexpected'}")
    if c is not None:
        idx = _match(c["points"], exp["contour_points"], f"{key} contour points")
        _close(c["scalars"][idx], exp["contour_scalars"], f"{key} contour scalars")
        _close(c["clim"], exp["clim"], f"{key} clim")
    a = rec["arrows"]
    if ("arrow_origins" in exp) != (a is not None):
        raise AssertionError(
            f"{key}: reaction arrows {'missing' if a is None else 'unexpected'}")
    if a is not None:
        idx = _match(a["origins"], exp["arrow_origins"], f"{key} arrow origins")
        _close(a["vectors"][idx], exp["arrow_vectors"], f"{key} arrow vectors")
        # The drawn length is |vector| x an auto-fit factor; the factor
        # is look (it follows the model size), the proportions are data.
        s, e = a["sizes"][idx], exp["arrow_sizes"]
        _close(s / max(s.max(), 1e-300), e / max(e.max(), 1e-300),
               f"{key} arrow lengths (relative)")


# =====================================================================
# Physics check of a golden, without rendering
# =====================================================================


def fixture_fields(r: Results) -> dict[str, Any]:
    """Reference coords and per-step nodal fields of the plate fixture."""
    stage = r.stages[-1].id
    rs = r.stage(stage)
    coords = np.asarray(r.fem.nodes.coords, dtype=np.float64)
    out: dict[str, Any] = {"coords": coords}
    for comp in ("displacement_x", "displacement_y",
                 "reaction_force_x", "reaction_force_y"):
        slab = rs.nodes.get(component=comp)
        ids = np.asarray(slab.node_ids, dtype=np.int64)
        order = np.argsort(ids)
        out[comp] = np.asarray(slab.values, dtype=np.float64)[:, order]
    out["node_ids"] = np.sort(np.asarray(r.fem.nodes.ids, dtype=np.int64))
    fem_order = np.argsort(np.asarray(r.fem.nodes.ids, dtype=np.int64))
    out["coords"] = coords[fem_order]
    return out


def predicted_points(f: dict[str, Any], step: int, scale: float) -> np.ndarray:
    u = np.zeros_like(f["coords"])
    u[:, 0] = f["displacement_x"][step]
    u[:, 1] = f["displacement_y"][step]
    return f["coords"] + scale * u


def auto_deform_scale(f: dict[str, Any], step: int) -> float:
    """render.py's documented auto-fit: 0.12 x diagonal / max |u|."""
    u = np.column_stack([f["displacement_x"][step], f["displacement_y"][step]])
    diag = float(np.linalg.norm(np.ptp(f["coords"], axis=0)))
    return 0.12 * diag / float(np.abs(u).max())


def check_golden_physics(
    gold: dict[str, np.ndarray], key: str, f: dict[str, Any],
    *, step: int, scale: float, contour: str | None,
) -> None:
    """The golden record ``key`` is the fixture at ``(step, scale)``."""
    pts = gold[f"{key}|points"]
    _match(pts, _merge_points(predicted_points(f, step, scale)),
           f"{key} golden vs reference + {scale:g} u(step {step})")
    if contour is None:
        assert f"{key}|contour_scalars" not in gold, key
        return
    cp = gold[f"{key}|contour_points"]
    pred = predicted_points(f, step, scale)
    idx = _match(pred, cp, f"{key} contour points")
    _close(gold[f"{key}|contour_scalars"], f[contour][step][idx],
           f"{key} golden contour vs {contour}[step {step}]")


# =====================================================================
# Fixture (the X2-a plate, shared)
# =====================================================================


@pytest.fixture(scope="module")
def plate_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return G._write_plate_results(tmp_path_factory.mktemp("x2_numeric"))


@pytest.fixture
def plate(plate_h5: Path, monkeypatch: pytest.MonkeyPatch):
    from apeGmsh.viewers.ui.theme import PALETTES, THEME
    from tests.conftest import _open_model_from_h5

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    monkeypatch.setattr(THEME, "_current", PALETTES[G._MANIFEST["theme"]])
    with Results.from_native(
        plate_h5, model=_open_model_from_h5(plate_h5),
    ) as r:
        yield r


def _require_shots(
    records: list[dict[str, Any]], expected: int, written: Any,
) -> None:
    """Fail closed when a call wrote output the seam did not see.

    ``written`` is what the public call returned. Only a call that wrote
    nothing (``None`` / empty: no GL) may skip, and not on a GL lane.
    Output written with fewer captured shots means the path bypassed
    ``BasePlotter.screenshot``: that is a failure, never a skip.
    """
    if len(records) >= expected:
        return
    if written:
        raise AssertionError(
            f"the call wrote {written!r} but the seam captured "
            f"{len(records)} shot(s), expected {expected}: the path no "
            "longer shoots through BasePlotter.screenshot (fail closed)",
        )
    if os.environ.get("APEGMSH_EXPECT_GL"):
        raise AssertionError(f"{len(records)} shots, expected {expected}")
    pytest.skip("no offscreen GL context")


def _by_name(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        name = rec["name"]
        assert name is not None and name not in out, (
            f"unnamed or repeated shot {name!r}")
        out[name] = rec
    return out


# =====================================================================
# 1. results.render, per view
# =====================================================================

_RENDER = G._MANIFEST["render"]


def _render_view(plate: Results, tmp_path: Path, which: str, **over: Any):
    """``results.render`` of the golden view ``which`` (``over`` mutates)."""
    spec = _RENDER[which]
    kw = dict(view=which, component=spec["component"], step=spec["step"],
              camera=_CAMERA, window_size=_WINDOW)
    kw.update(over)
    with capture_drawn() as recs:
        written = plate.render(tmp_path / f"{which}.png", **kw)
    _require_shots(recs, 1, written)
    assert len(recs) == 1, f"render shot {len(recs)} frames"
    return recs[0]


@pytest.mark.parametrize("view", sorted(_RENDER))
def test_render_draws_golden_data(plate, tmp_path: Path, view: str) -> None:
    rec = _render_view(plate, tmp_path, view)
    key = f"render.{view}"
    if REGEN:
        f = fixture_fields(plate)
        scale = auto_deform_scale(f, 4) if view == "deformed" else 0.0
        write_golden(flatten(key, rec), {key: {
            "step": 4, "deform_scale": scale,
            "contour": None if rec["contour"] is None
            else rec["contour"]["name"],
            "arrows": 0 if rec["arrows"] is None
            else int(len(rec["arrows"]["origins"])),
        }})
        return
    assert_record(rec, key, load_golden())


# =====================================================================
# 2. results.render_pack, per file; 3. assess(figures=True), same data
# =====================================================================


def _pack_records(fn: Any) -> tuple[list[str], dict[str, dict[str, Any]]]:
    with capture_drawn() as recs:
        names = fn()
    _require_shots(recs, 1, names or None)
    return names, _by_name(recs)


def _check_pack(names: list[str], recs: dict[str, dict[str, Any]], key: str) -> None:
    man = load_manifest()[key]
    assert names == man["files"], names
    assert sorted(recs) == sorted(man["files"]), sorted(recs)
    gold = load_golden()
    for name in man["files"]:
        assert_record(recs[name], f"pack.{name}", gold)


def test_render_pack_draws_golden_data(plate, tmp_path: Path) -> None:
    names, recs = _pack_records(lambda: [
        p.name for p in plate.render_pack(
            tmp_path / "pack", camera=_CAMERA, window_size=_WINDOW)])
    if REGEN:
        entries: dict[str, np.ndarray] = {}
        for name, rec in recs.items():
            entries.update(flatten(f"pack.{name}", rec))
        f = fixture_fields(plate)
        write_golden(entries, {"render_pack": {
            "files": names,
            "deform_scale": {"deformed.png": auto_deform_scale(f, 4)},
            "contour": {n: r["contour"]["name"] for n, r in recs.items()
                        if r["kind"] == "vtk" and r["contour"] is not None},
        }})
        return
    _check_pack(names, recs, "render_pack")


def test_assess_figures_draw_render_pack_data(plate, tmp_path: Path) -> None:
    names, recs = _pack_records(lambda: [
        p.name for p in plate.assess(
            figures=True, out_dir=tmp_path / "assess").figures])
    if REGEN:
        pytest.skip("regenerating: assess reuses the render_pack records")
    _check_pack(names, recs, "render_pack")


# =====================================================================
# 4. The goldens are the fixture's data (no rendering)
# =====================================================================


def test_goldens_are_the_fixture_data(plate) -> None:
    if REGEN:
        pytest.skip("regenerating")
    gold, man, f = load_golden(), load_manifest(), fixture_fields(plate)
    # render: step 4 (-1), auto scale for 'deformed', contour as named.
    for view in _RENDER:
        m = man[f"render.{view}"]
        check_golden_physics(
            gold, f"render.{view}", f, step=m["step"],
            scale=m["deform_scale"], contour=m["contour"],
        )
    assert man["render.deformed"]["deform_scale"] > 0
    # reactions: one arrow per supported node, the recorded force.
    a_o = gold["render.reactions|arrow_origins"]
    a_v = gold["render.reactions|arrow_vectors"]
    rx, ry = f["reaction_force_x"][4], f["reaction_force_y"][4]
    loaded = np.flatnonzero((rx != 0) | (ry != 0))
    assert len(a_o) == len(loaded) == man["render.reactions"]["arrows"]
    idx = _match(f["coords"][loaded], a_o, "arrow origins vs supports")
    _close(a_v[:, :2], np.column_stack([rx, ry])[loaded][idx],
           "arrow vectors vs reaction_force_xy[step 4]")
    # pack: mesh undeformed, contour/deformed on the default component.
    pk = man["render_pack"]
    for name in pk["files"]:
        if not name.endswith(".png") or name == "history.png":
            continue
        check_golden_physics(
            gold, f"pack.{name}", f, step=4,
            scale=pk["deform_scale"].get(name, 0.0),
            contour=pk["contour"].get(name),
        )
    # history: one line, time vs the component at its extrema node.
    hx, hy = gold["pack.history.png|line0_x"], gold["pack.history.png|line0_y"]
    assert len(hx) == len(hy) == G._MANIFEST["fixture"]["n_steps"]
    comp = pk["contour"]["contour.png"]
    col = int(np.argmax(np.abs(f[comp][-1])))
    _close(hy, f[comp][:, col], f"history vs {comp} at the extrema node")


# =====================================================================
# 5. Revert-proof: the oracle must reject wrong data (kept as tests)
# =====================================================================


def _rejects(rec: dict[str, Any], key: str, gold: dict[str, np.ndarray]) -> str:
    with pytest.raises(AssertionError) as info:
        assert_record(rec, key, gold)
    return str(info.value)


@pytest.mark.parametrize("label, view, over", [
    ("wrong step", "contour", dict(step=3)),
    ("wrong step (deformed)", "deformed", dict(step=2)),
    ("wrong deform scale", "deformed", dict(deform=("displacement", 1.0))),
    ("deform off", "deformed", dict(view="contour")),
    ("wrong contour component", "contour", dict(component="displacement_x")),
    ("reactions at the wrong step", "reactions", dict(step=1)),
])
def test_oracle_rejects_wrong_render(
    plate, tmp_path: Path, label: str, view: str, over: dict,
) -> None:
    if REGEN:
        pytest.skip("regenerating")
    rec = _render_view(plate, tmp_path, view, **over)
    _rejects(rec, f"render.{view}", load_golden())


def test_oracle_rejects_tampered_data(plate, tmp_path: Path) -> None:
    """Data-level mutants the public knobs cannot make."""
    if REGEN:
        pytest.skip("regenerating")
    gold = load_golden()
    contour = _render_view(plate, tmp_path, "contour")
    bad = {**contour, "contour": {**contour["contour"],
                                  "clim": contour["contour"]["clim"] * 1.01}}
    assert "clim" in _rejects(bad, "render.contour", gold)
    reac = _render_view(plate, tmp_path, "reactions")
    a = reac["arrows"]
    missing = {**reac, "arrows": {k: v[1:] for k, v in a.items()}}
    assert "arrow origins" in _rejects(missing, "render.reactions", gold)
    none = {**reac, "arrows": None}
    assert "arrows missing" in _rejects(none, "render.reactions", gold)


# =====================================================================
# 6. The seam fails closed (review of #1690)
# =====================================================================


def test_seam_rejects_an_unreadable_3d_prop() -> None:
    """A visible 3-D actor the oracle cannot read raises, never skips."""
    import pyvista as pv
    import vtk

    try:
        plotter = pv.Plotter(off_screen=True)
    except (RuntimeError, OSError):  # pragma: no cover
        pytest.skip("no offscreen render context")
    try:
        plotter.add_mesh(pv.Plane())
        rec = extract_drawn(plotter)  # control: readable scene passes
        assert rec["points"].shape[0] > 0
        src = vtk.vtkSphereSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        raw = vtk.vtkActor()
        raw.SetMapper(mapper)
        plotter.add_actor(raw)
        with pytest.raises(AssertionError, match="fail closed"):
            extract_drawn(plotter)
        raw.SetVisibility(False)
        extract_drawn(plotter)  # hidden: nothing drawn, nothing to read
        text = vtk.vtkTextActor()
        text.SetInput("2-D overlay")
        plotter.add_actor(text)
        extract_drawn(plotter)
    finally:
        plotter.close()


def test_a_written_still_without_a_captured_shot_fails(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path that writes its still around ``BasePlotter.screenshot``
    must FAIL the oracle, not skip it (even off a GL lane)."""
    from apeGmsh.viewers import render as render_mod

    def _bypass(plotter: Any, path: Any) -> Path:
        out = Path(path)
        out.write_bytes(b"not a shot")
        return out

    monkeypatch.delenv("APEGMSH_EXPECT_GL", raising=False)
    monkeypatch.setattr(render_mod, "_screenshot", _bypass)
    for call in (
        lambda: _render_view(plate, tmp_path, "contour"),
        lambda: _require_shots([], 1, (tmp_path / "x.png",)),
    ):
        # A skip is an outcome too (BaseException): it must not escape.
        try:
            call()
        except AssertionError as exc:
            assert "fail closed" in str(exc), exc
        except BaseException as exc:  # noqa: BLE001
            pytest.fail(f"a bypassed shot did not fail closed: {exc!r}")
        else:
            pytest.fail("a bypassed shot was accepted")
