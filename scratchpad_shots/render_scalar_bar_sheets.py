"""ADR 0090 evidence — scalar-bar A/B variants, rendered in-process.

Boots the results viewer on the 3D column demo (contour + deform, last
step — the ADR 0089 SHEET_3 state), then mutates the live legend through
the LegendController where the controller has the knob (fmt, ticks,
orientation, font scale) and through the vtkScalarBarActor where it does
not yet (text colour, font family, title text, background chip, banded
ramp, NaN swatch). Zero src changes; the persisted theme is snapshotted
and restored.

Usage:
    python scratchpad_shots/render_scalar_bar_sheets.py <column_demo.h5>
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE))          # tests.conftest helper

OUT = WORKTREE / "scratchpad_shots"
PANELS = OUT / "panels"
PANELS.mkdir(exist_ok=True)

DEMO_H5 = Path(sys.argv[1])

import numpy as np                                     # noqa: E402
from PIL import Image, ImageDraw                       # noqa: E402

from apeGmsh.results import Results                    # noqa: E402
from apeGmsh.viewers.results_viewer import ResultsViewer  # noqa: E402
from tests.conftest import _open_model_from_h5         # noqa: E402

results = Results.from_native(DEMO_H5, model=_open_model_from_h5(DEMO_H5))
viewer = ResultsViewer(results, restore_session=False, save_session=False)
viewer.show(run_loop=False, window_size=(1600, 900))

from qtpy import QtWidgets                             # noqa: E402

app = QtWidgets.QApplication.instance()

from apeGmsh.viewers.ui.theme import THEME, PALETTES   # noqa: E402
from apeGmsh.viewers.core._legend import key_str       # noqa: E402

THEME_AT_START = THEME.current.name


def settle(ms=400):
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


settle(900)
viewer._on_empty_state_add_contour()
viewer._on_empty_state_enable_deform()
settle()
director = viewer.director
director.set_step(director.stages()[0].n_steps - 1)
settle()

legends = viewer._legends()
backend = legends._backend
entry = legends.entries()[0]
KEY = entry.key
BKEY = key_str(KEY)
print("legend:", KEY, "fmt", entry.fmt, "n_labels", entry.n_labels)


def bar_tuple():
    return backend._scalar_bars[BKEY]


def rebuild():
    """Force a fresh bar actor under the current controller state."""
    backend.remove_scalar_bar(BKEY)
    legends._applied.pop(BKEY, None)
    legends.refresh()
    settle(200)
    if BKEY not in backend._scalar_bars:   # pyvista registry hiccup
        legends._applied.pop(BKEY, None)
        legends.refresh()
        settle(300)
    return bar_tuple()


# Pristine LUT (pyvista-built, continuous) — reinstated after banding.
_mapper0 = entry.handle.actor.GetMapper()
ORIG_LUT = _mapper0.GetLookupTable()
ORIG_USE_LUT_RANGE = _mapper0.GetUseLookupTableScalarRange()


# ---------------------------------------------------------------------
# Backdrop chip — the ADR 0090 proposal prototyped as its own 2D actor
# pair (fill + hairline), drawn at background display location so it
# sits under the bar. Covers the WHOLE legend box including the
# horizontal title band, which vtkScalarBarActor's own DrawBackground
# cannot (the band is outside the actor).
# ---------------------------------------------------------------------
_backdrops = []


def clear_backdrops():
    ren = viewer.plotter.renderer
    for a in _backdrops:
        try:
            ren.RemoveActor2D(a)
        except Exception:
            pass
    _backdrops.clear()


def add_backdrop(e, pad_px=8, fill=None, alpha=0.85, border=None):
    import vtk
    p = THEME.current
    fill = fill or p.mantle
    border = border or p.surface0
    vw, vh = backend.viewport_size()
    px, py = pad_px / vw, pad_px / vh
    x, y = e.anchor
    w, h = e.extent
    x0, y0, x1, y1 = x - px, y - py, x + w + px, y + h + py
    pts = vtk.vtkPoints()
    for a, b in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        pts.InsertNextPoint(a, b, 0.0)
    quad = vtk.vtkCellArray()
    quad.InsertNextCell(4)
    for i in range(4):
        quad.InsertCellPoint(i)
    ring = vtk.vtkCellArray()
    ring.InsertNextCell(5)
    for i in (0, 1, 2, 3, 0):
        ring.InsertCellPoint(i)
    pd_fill = vtk.vtkPolyData()
    pd_fill.SetPoints(pts)
    pd_fill.SetPolys(quad)
    pd_ring = vtk.vtkPolyData()
    pd_ring.SetPoints(pts)
    pd_ring.SetLines(ring)
    ren = viewer.plotter.renderer
    for pd, color, op in ((pd_fill, fill, alpha), (pd_ring, border, 1.0)):
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()
        m = vtk.vtkPolyDataMapper2D()
        m.SetInputData(pd)
        m.SetTransformCoordinate(coord)
        a2 = vtk.vtkActor2D()
        a2.SetMapper(m)
        a2.GetProperty().SetColor(*hex_rgb(color))
        a2.GetProperty().SetOpacity(op)
        a2.GetProperty().SetLineWidth(1.0)
        a2.GetProperty().SetDisplayLocationToBackground()
        ren.AddActor2D(a2)
        _backdrops.append(a2)


# ---------------------------------------------------------------------
# Pokes — the proposed treatments, applied to the live actor
# ---------------------------------------------------------------------

def poke(
    *,
    text_color=None,      # palette hex → labels + title
    mono_labels=False,
    title=None,           # replacement display title
    chip=False,           # mantle backdrop + hairline frame
    chip_alpha=0.85,
    band=None,            # int → banded LUT on field + bar
    nan_swatch=False,
    bold_title=None,
):
    p = THEME.current
    _t, bar, title_actor = bar_tuple()
    if text_color is not None:
        rgb = hex_rgb(text_color)
        bar.GetLabelTextProperty().SetColor(*rgb)
        bar.GetTitleTextProperty().SetColor(*rgb)
        if title_actor is not None:
            title_actor.GetTextProperty().SetColor(*rgb)
    if mono_labels:
        bar.GetLabelTextProperty().SetFontFamilyToCourier()
    if bold_title is not None:
        bar.GetTitleTextProperty().SetBold(bool(bold_title))
        if title_actor is not None:
            title_actor.GetTextProperty().SetBold(bool(bold_title))
    if title is not None:
        if title_actor is not None:
            title_actor.SetInput(title)
        else:
            bar.SetTitle(title)
    if chip:
        bar.SetDrawBackground(1)
        bg = bar.GetBackgroundProperty()
        bg.SetColor(*hex_rgb(p.mantle))
        bg.SetOpacity(float(chip_alpha))
        bar.SetDrawFrame(1)
        fr = bar.GetFrameProperty()
        fr.SetColor(*hex_rgb(p.surface0))
        fr.SetOpacity(1.0)
        fr.SetLineWidth(1)
    if band:
        import vtk
        from matplotlib import colormaps
        src = bar.GetLookupTable()
        rng = src.GetRange()
        colors = colormaps["viridis"](np.linspace(0, 1, band))
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(band)
        for i, c in enumerate(colors):
            lut.SetTableValue(i, *c)
        lut.SetRange(*rng)
        lut.Build()
        mapper = entry.handle.actor.GetMapper()
        mapper.SetLookupTable(lut)
        mapper.SetUseLookupTableScalarRange(True)
        bar.SetLookupTable(lut)
        bar.SetMaximumNumberOfColors(band)
        if nan_swatch:
            lut.SetNanColor(0.55, 0.55, 0.55, 1.0)
    if nan_swatch:
        try:
            bar.GetLookupTable().SetNanColor(0.55, 0.55, 0.55, 1.0)
            bar.SetDrawNanAnnotation(1)
            bar.SetNanAnnotation("no data")
        except Exception as exc:
            print("nan swatch failed:", exc)
    viewer.plotter.render()
    settle(120)


def shot(name, crop="legend", pad=26):
    """Screenshot; crop='legend' crops the controller box + margin."""
    img = viewer.plotter.screenshot(return_img=True)
    H, W = img.shape[:2]
    im = Image.fromarray(img)
    if crop == "legend":
        boxes = [(e.anchor, e.extent) for e in legends.entries() if e.visible]
        xs0 = min(a[0] for a, _ in boxes)
        ys0 = min(a[1] for a, _ in boxes)
        xs1 = max(a[0] + e[0] for a, e in boxes)
        ys1 = max(a[1] + e[1] for a, e in boxes)
        left = max(0, int(xs0 * W) - pad)
        right = min(W, int(xs1 * W) + pad)
        top = max(0, int((1 - ys1) * H) - pad)
        bottom = min(H, int((1 - ys0) * H) + pad)
        im = im.crop((left, top, right, bottom))
    path = PANELS / f"{name}.png"
    im.save(path)
    print("saved", path.name, im.size)
    return path


def restore_continuous():
    """Undo the banded-LUT poke by re-attaching the pristine mapper LUT."""
    mapper = entry.handle.actor.GetMapper()
    mapper.SetLookupTable(ORIG_LUT)
    mapper.SetUseLookupTableScalarRange(ORIG_USE_LUT_RANGE)
    viewer.plotter.render()
    settle(100)
    rebuild()


# =====================================================================
# Group T — typography + ticks (mocha)
# =====================================================================
THEME.set_theme("catppuccin_mocha")
settle(800)
p = THEME.current

rebuild()
shot("T0_current")

rebuild()
poke(text_color=p.text)
shot("T1_text_role")

rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z")
shot("T2_mono_title")

legends._entries[KEY].n_labels = 7
rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z")
shot("T3_seven_ticks")
legends._entries[KEY].n_labels = 5

legends.set_fmt(KEY, "%.2e")
rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z")
shot("T4_sci_fmt")
legends.set_fmt(KEY, "%.3g")

legends.set_font_scale(KEY, 1.25)
rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z")
shot("T5_scale_125")
legends.set_font_scale(KEY, 1.0)

legends.set_fmt(KEY, "%#.3g")
rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z")
shot("T6_trailing_zeros")
legends.set_fmt(KEY, "%.3g")

# =====================================================================
# Group C — container chip (mocha + paper)
# =====================================================================
rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z")
shot("C0_mocha_naked")

rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z", chip=True)
shot("C1_mocha_chip")

rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z", chip=True,
     chip_alpha=0.92)
shot("C1b_mocha_chip92")

rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z")
add_backdrop(legends._entries[KEY], pad_px=8)
viewer.plotter.render()
settle(120)
shot("V2_chip_backdrop", pad=34)
clear_backdrops()

THEME.set_theme("paper")
settle(900)
pp = THEME.current

rebuild()
shot("C2_paper_current")

rebuild()
poke(text_color=pp.text, mono_labels=True, title="Displacement Z", chip=True)
shot("C3_paper_chip")

rebuild()
poke(text_color=pp.text, mono_labels=True, title="Displacement Z")
add_backdrop(legends._entries[KEY], pad_px=8, border=pp.surface2)
viewer.plotter.render()
settle(120)
shot("C3b_paper_backdrop_s2", pad=34)
clear_backdrops()

full = viewer.plotter.screenshot(return_img=True)
Image.fromarray(full).save(PANELS / "C2b_paper_context.png")

THEME.set_theme("catppuccin_mocha")
settle(900)
p = THEME.current

# =====================================================================
# Group R — ramp presentation (mocha)
# =====================================================================
rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z", chip=True)
shot("R0_continuous")

rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z", chip=True,
     band=10)
shot("R1_banded10", crop=None)
shot("R1_banded10_crop")

# NaN / out-of-range swatches on top of the still-banded bar — no
# rebuild in between (a rebuild while the mapper holds the synthetic
# banded LUT trips pyvista's registry).
poke(nan_swatch=True)
shot("R2b_nan_only")
try:
    _t, bar, _ta = bar_tuple()
    lut = bar.GetLookupTable()
    lut.SetUseAboveRangeColor(1)
    lut.SetAboveRangeColor(0.85, 0.30, 0.30, 1.0)
    lut.SetUseBelowRangeColor(1)
    lut.SetBelowRangeColor(0.30, 0.30, 0.85, 1.0)
    bar.SetDrawAboveRangeSwatch(1)
    bar.SetDrawBelowRangeSwatch(1)
    viewer.plotter.render()
    settle(120)
except Exception as exc:
    print("range swatches failed:", exc)
shot("R2_banded_nan")

restore_continuous()

# =====================================================================
# Group H — horizontal (mocha)
# =====================================================================
legends.set_vertical(KEY, False)
settle(200)
rebuild()
shot("H0_horizontal_current")

rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z", chip=True)
shot("H1_horizontal_proposed")

rebuild()
poke(text_color=p.text, mono_labels=True, title="Displacement Z")
add_backdrop(legends._entries[KEY], pad_px=8)
viewer.plotter.render()
settle(120)
shot("H2_horizontal_backdrop", pad=34)
clear_backdrops()

# Same, at the PickReadoutHUD glass alpha — over a bright field the
# 0.85 fill lets the substrate glow through the title band.
add_backdrop(legends._entries[KEY], pad_px=8, alpha=0.92)
viewer.plotter.render()
settle(120)
shot("H3_horizontal_backdrop92", pad=34)
clear_backdrops()

legends.set_vertical(KEY, True)
settle(200)

# =====================================================================
# Group M — multi-legend (mocha)
# =====================================================================
def add_second_contour(component):
    from apeGmsh.viewers.diagrams._base import DiagramSpec
    from apeGmsh.viewers.diagrams._dispatch import DIAGRAM_ATTACHED
    from apeGmsh.viewers.diagrams._kinds import kind_def
    from apeGmsh.viewers.diagrams._selectors import normalize

    kdef = kind_def("contour")
    spec = DiagramSpec(
        kind="contour",
        selector=normalize(component=component),
        style=kdef.make_default_style(component),
        stage_id=director.stage_id,
    )
    diagram = kdef.diagram_class(spec, director.results)
    geoms = director.geometries
    geom = geoms.active or geoms.geometries[0]
    comp = geom.compositions.active
    geom.compositions.add_layer(comp.id, diagram)
    director.registry.add(diagram)
    director.dispatcher.fire(DIAGRAM_ATTACHED, layer=diagram)
    return diagram


second = None
try:
    second = add_second_contour("displacement_x")
    settle(400)
    for k in list(legends._applied):
        backend.remove_scalar_bar(k)
        legends._applied.pop(k, None)
    legends.refresh()
    settle(200)
    shot("M0_two_legends_current", pad=40)

    for e in legends.entries():
        bk = key_str(e.key)
        t = backend._scalar_bars.get(bk)
        if t is None:
            continue
        _t, bar, ta = t
        rgb = hex_rgb(p.text)
        bar.GetLabelTextProperty().SetColor(*rgb)
        bar.GetTitleTextProperty().SetColor(*rgb)
        bar.GetLabelTextProperty().SetFontFamilyToCourier()
        nice = {
            "displacement_z": "Displacement Z",
            "displacement_x": "Displacement X",
            "displacement_y": "Displacement Y",
        }.get(e.component, e.component)
        bar.SetTitle(nice)
        bar.SetDrawBackground(1)
        bg = bar.GetBackgroundProperty()
        bg.SetColor(*hex_rgb(p.mantle))
        bg.SetOpacity(0.85)
        bar.SetDrawFrame(1)
        fr = bar.GetFrameProperty()
        fr.SetColor(*hex_rgb(p.surface0))
        fr.SetOpacity(1.0)
    viewer.plotter.render()
    settle(150)
    shot("M1_two_legends_proposed", pad=40)
except Exception as exc:
    print("multi-legend group failed:", exc)
finally:
    if second is not None:
        try:
            director.registry.remove(second)
            settle(300)
        except Exception as exc:
            print("second-contour teardown failed:", exc)

# =====================================================================
# Group F — proposed, full context, three canonical palettes
# =====================================================================
for name in ("catppuccin_mocha", "neutral_studio", "paper"):
    THEME.set_theme(name)
    settle(900)
    pt = THEME.current
    rebuild()
    poke(text_color=pt.text, mono_labels=True, title="Displacement Z")
    border = pt.surface2 if name == "paper" else pt.surface0
    add_backdrop(legends._entries[KEY], pad_px=8, border=border)
    viewer.plotter.render()
    settle(150)
    img = viewer.plotter.screenshot(return_img=True)
    Image.fromarray(img).save(PANELS / f"F_{name}_proposed_context.png")
    clear_backdrops()
    print("saved F context", name)

# ---------------------------------------------------------------------
THEME.set_theme(THEME_AT_START)
settle(400)
viewer.close()
print("panels done; theme restored to", THEME_AT_START)
