# Viewers

Qt/PyVista viewers: session-embedded authoring viewers plus the
results viewer for post-processing.

## ModelViewer — `g.model.viewer()`

::: apeGmsh.viewers.model_viewer.ModelViewer

## MeshViewer — `g.mesh.viewer()`

::: apeGmsh.viewers.mesh_viewer.MeshViewer

## ResultsViewer — `Results.viewer()`

!!! warning "`blocking=True` is the default and crashes the Jupyter kernel"
    `results.viewer()` defaults to `blocking=True`, which drives the
    blocking VTK+Qt event loop and **kills the ipykernel** in a
    notebook. In Jupyter use `results.show_web()` (kernel-safe trame)
    or `results.viewer(blocking=False)` (runs in a subprocess; the
    kernel keeps going).

::: apeGmsh.viewers.results_viewer.ResultsViewer

### The colour scale

Every contour, glyph, fibre and layer diagram that colours by an array
publishes a colour scale. Scales belong to the *view*, not to the
diagram: two diagrams showing the same component share one scale, and
the same component on two concurrent geometries gets one each.

Scales dock down the right-hand edge and stack. Drag one anywhere you
like; drag the corner opposite its anchor to resize it. Because a
scale's box is computed from the text it has to hold — the title, the
tick labels under the current number format — resizing scales the text
and the box follows, so a scale can never end up too small to read.
Drop one back against the edge it came from, or double-click it, and it
rejoins the stack.

Right-click a scale for the rest: hide it, flip it between vertical and
horizontal, reset its size, change the number format, rescale it to the
current step's data range, or open the colour-map editor on the diagram
behind it. The same knobs live in the diagram's settings card when you
want to set them precisely or from a preset.

Whatever you do to a scale survives everything else — changing the
colour map, stepping through time, re-fitting the range, hiding and
showing it again. Placement is saved with the viewer session, so a
scale you moved comes back where you left it; scales still docked are
laid out fresh, which is what you want when you reopen on a different
screen. `Preferences → Labels → Colour-scale text size` sets the
default size for scales you have not resized by hand.

## Web viewers — `results.show_web()` / `results.serve_web()`

Kernel-safe trame/PyVista viewers for notebooks and the browser.
These require the `[viewer]` extra (install with `[viewer]` — or `[all]` — in the pip line from [Install](../index.md#install)).

`results.show_web(*, stage=None, show=True, controls=True,
render_mode="client")` renders an inline trame view inside Jupyter,
with ipywidgets step-slider and layer toggles when `controls=True`. It
returns a `WebViewer`.

`results.serve_web(*, stage=None, render_mode="client", port=None,
open_browser=True, title="apeGmsh")` launches a standalone vuetify3 web
app (blocks until interrupted) — for use outside a notebook.

`render_mode` is a friendly alias: `"client"` (default — WebGL in the
browser, fast camera), `"server"` (kernel-side, image-streamed, for
very large models), or `"hybrid"` (toolbar toggle). Any other value
raises `ValueError`.

```python
from apeGmsh import Results

# Zero-setup sample (real apeSees-emitted model + synthetic pushover)
results = Results.demo()

wv = results.show_web()        # inline trame view + ipywidgets controls
wv.set_step(3)                 # programmatic scrub + re-render

# results.serve_web(port=8080)   # standalone app, outside a notebook
```

`Results.demo(**kwargs)` (and the underlying
`make_demo_results(*, length=10.0, n_elements=8, n_steps=6,
tip_drift=2.0, path=None)`) build a ready-to-view sample with no
`.mpco`/`model.h5` needed — handy for trying the web viewers cold.

::: apeGmsh.viewers.web_viewer.WebViewer

## Geometric transform viewer

::: apeGmsh.viewers.geom_transf_viewer.GeomTransfViewer

## Preferences

### `settings()`

::: apeGmsh.viewers.settings

### `theme_editor()`

::: apeGmsh.viewers.theme_editor
