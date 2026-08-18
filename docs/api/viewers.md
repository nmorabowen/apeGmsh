# Viewers

Qt/PyVista viewers: session-embedded authoring viewers plus the
results session window for post-processing.

## ModelViewer — `g.model.viewer()`

::: apeGmsh.viewers.model_viewer.ModelViewer

## MeshViewer — `g.mesh.viewer()`

::: apeGmsh.viewers.mesh_viewer.MeshViewer

## Results session window — `Results.viewer()`

`results.viewer()` opens the post-solve window. Since ADR 0098 it is
sugar for `results.session().show()`: the document is a
**`ResultsSession`** — tiled mesh and plot panes, each pane carrying
result *slots* — and the window is one client that projects it. A
script can build or read the same object with no window at all, and
what the window saves is that object.

```python
results.viewer()               # the human one-liner (blocking)

s = results.session()          # the same document, no window
s.render("pane.png")           # a still, no Qt
s.show()                       # the same window, from Python
```

!!! warning "Explicit `blocking=True` still crashes the Jupyter kernel"
    `results.viewer()` defaults to `blocking=None` (auto): `True`
    (in-process Qt) in scripts / the CLI, `False` (subprocess) inside
    a Jupyter kernel. An in-memory Results in a notebook cannot spawn
    a subprocess and falls back to `results.show_web()`. An **explicit**
    `blocking=True` still drives the VTK+Qt event loop in-process and
    **kills the ipykernel**.

The blocking call returns the `ResultsSession` once the window closes —
still live, so you can query it, render stills off it, or snapshot it.
`blocking=False` returns the `subprocess.Popen` handle instead, and the
notebook in-memory fallback returns the `WebViewer`: three different
things on purpose, because a session in *this* process is not what a
child window is showing.

### Saving and restoring

`save_session=True` (the default) writes the session —  panes, slots,
pose, time link, selection — to `<results>.viewer-session.json` when
the window closes. `restore_session` decides what happens to a file
already there: `True` restores it silently, `False` ignores it, and
`"prompt"` (the default) asks first, listing what the file contains and
anything about it that no longer fits these results.

The window always opens. A session file that cannot be read never
blocks it and is never destroyed by it: the reason is printed, the
default picture boots, and auto-save is switched off for that window so
the file is still there afterwards. That last part matters because the
save target *is* the file that was refused.

Sessions written by the retired Geometry / Composition / Diagram viewer
are not restorable. The first upgraded open says so in one line and
renames the old file to `<results>.viewer-session.json.legacy` —
overwriting neither it nor an aside already sitting there.

<!-- The Qt names in ``apeGmsh.viewers.session`` are exported lazily
     (PEP 562) so ``realize`` / ``render`` keep working with no qtpy, so
     mkdocstrings has to collect the window from its own module. -->
::: apeGmsh.viewers.session._window.SessionWindow

### Colour scales

A scale is caused by a slot, and belongs to the pane. Filling a
colour-mapped slot creates its scale; clearing the slot destroys it;
two slots showing the same quantity share one scale. Hiding a scale
never hides the picture it names, and each pane carries the scales of
its own slots — so a quad view of four contours is four panes with one
scale each.

Deform is not a slot: a warped mesh with every slot empty draws **no**
scale at all. The scale names the slot's quantity, so warping by
displacement while contouring stress gives a scale that says stress.

Number format, orientation and font scale come from the slot's style;
`Preferences → Labels → Colour-scale text size` sets the default size.

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
