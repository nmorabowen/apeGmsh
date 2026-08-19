# Results session

The **document** behind post-solve presentation: `ResultsSession` —
tiled mesh and plot panes, one time link, one selection. `Results`
answers *what did the solver write?*; the session answers *what is on
screen, at which time, with which pictures, with which pick.*

Since ADR 0098 this is what `results.viewer()` opens. The window
([Viewers](viewers.md)) is one **client** that projects the document;
`render()` is another that draws a still with no Qt at all, and the
snapshot is the document written to disk. VTK actors, docks and
widgets are never truth — which is why a script can build, read and
render exactly what a human arranged in the window.

```python
from apeGmsh import Results
from apeGmsh.results.session import Contour, Deform, Instant

results = Results.demo()

s = results.session()                 # the document — one empty mesh view
view = s.panes[0]
view.contour = Contour("displacement_x")
view.deform = Deform("displacement", scale=5.0)
s.time = Instant("stage_0", 5)

s.render("tip.png")                   # a still, no Qt
```

`results.viewer()` is sugar for `results.session()` + `show()`, and
returns the session once the window closes — still live, so you can
query it, render more stills off it, or snapshot it.

## The session — `results.session()`

`results.session()` boots the default picture: **one** empty mesh view
(grey analysis mesh, mesh + outlines on, no slots, no legends,
unscoped, undeformed) bound to that broker. Section cuts persisted
under `/opensees/cuts/` come back on that view as clips; a cut that
cannot be translated honestly into a plane is skipped with one
`[session]` line rather than silently widening what disappears.

A session with no `results=` is valid IR — the snapshot round-trips it
— but `realize()` and `render()` refuse loudly, because there is
nothing to read numbers from.

::: apeGmsh.results.session.ResultsSession

## Panes

A pane is a `MeshView` or a `PlotView`. Both carry a stable `id`
(`mesh-1`, `plot-2`, …) — that id is how `render()`, the snapshot, the
outline and the MCP verb address a pane, so a stale one raises
`KeyError` rather than guessing.

A plot is a **pane**, not a dock hanging off a contour: several curves
on one chart are one `PlotView`, and its series are live queries
against `Results` evaluated at the cursor. `plot.kind` is fixed at
creation (`history`, `path` or `xy`); `plot.series` and `plot.cursor`
are assignable, and `plot.name` is a label.

`session.remove_pane(pane_id)` drops a pane and detaches it, so a
handle you still hold stops ticking the session.

### `MeshView`

::: apeGmsh.results.session.MeshView

### `PlotView`

::: apeGmsh.results.session.PlotView

## The slot catalog

A mesh view carries **slots**, from a catalog of exactly seven
categories:

| Slot | Record | Carries | Colour-mapped |
| --- | --- | --- | --- |
| `contour` | `Contour` | quantity + `averaged` / `unaveraged` | yes |
| `vector` | `Vector` | quantity (nodal vectors *and* Gauss principal families) | yes |
| `gauss` | `Gauss` | quantity at integration points | yes |
| `line` | `Line` | member-diagram component (N / V / M / torsion) | no |
| `sand` | `Sand` | quantity | yes |
| `loads` | `Loads` | pattern (or `None` for the default) | no |
| `reactions` | `Reactions` | — | no |

Each category name in that first column is an **assignable property of
the mesh view** — `view.contour`, `view.vector`, `view.gauss`,
`view.line`, `view.sand`, `view.loads`, `view.reactions` — reading the
occupant or `None`. (They are built by a factory, so they do not appear
in the generated member list below; the table is their reference.)

The catalog is **closed**. Different categories stack; one category
holds at most one occupant per view, and **filling an occupied slot
replaces the occupant** — there is no "add a second contour". Assign
`None` to clear, and a record of the wrong category refuses loudly.

```python
view.contour = Contour("displacement_x")
view.contour = Contour("displacement_x", averaging="unaveraged")
view.slots
# {'contour': Contour(quantity='displacement_x', averaging='unaveraged')}

view.contour = None                   # clears
view.contour = Deform("displacement")
# TypeError: The 'contour' slot takes a Contour record or None; got Deform.
```

Fibers, shell layers and isochrones do not open an eighth slot: a new
category is an ADR 0098 amendment, not a subclass.

::: apeGmsh.results.session._slots
    options:
      members:
        - Slot
        - Contour
        - Vector
        - Gauss
        - Line
        - Sand
        - Loads
        - Reactions
        - SLOT_CATALOG
        - COLOUR_MAPPED
        - slot_field

## Deform is a pose, not a slot

`view.deform` warps the mesh. It is deliberately **outside** the
catalog, and the reason shows up in the legend: a pose is not a
picture of a quantity, so a warped mesh with every slot empty draws no
colour scale at all. Warping by displacement while contouring stress
gives one scale, and it says *stress*.

`Deform(mode=…)` makes it a **mode pose**: the view then has no
`(stage, step)` at all and is frozen under the session time link.

::: apeGmsh.results.session.Deform

## The legend law

Legends are **derived**, never stored: one scale per distinct field
over the view's occupied colour-mapped slots. Two slots showing the
same quantity therefore share one scale, clearing a slot destroys its
scale, and hiding a scale is chrome — it hides the scale, not the
picture, and does not clear the slot.

```python
from apeGmsh.results.session import Gauss

view.deform = Deform("displacement", scale=5.0)
view.legends()                        # () — a pose causes no scale

view.contour = Contour("stress_xx")
view.gauss = Gauss("stress_xx")
[lg.field for lg in view.legends()]   # ['stress_xx'] — ONE scale
view.legends()[0].categories          # ('contour', 'gauss')

view.set_legend_hidden("stress_xx")   # chrome; the picture stays
view.set_legend_hidden("displacement_x")
# ValueError: No legend for field 'displacement_x' on this view — legends
# exist only for occupied colour-mapped slots (have: ['stress_xx']).
```

Nothing above reads a number: `legends()` is a function of the slots,
so it answers on any view, bound to results or not.

::: apeGmsh.results.session.Legend

## Scope, style and clips

`scope` picks the ONE cell set every layer of the view is a function
of — one composition axis (`physical_groups`, `materials` or
`element_types`) plus checked names, never a boolean `concrete AND
hex`. `style` is the four independent buttons over cells that are
already on. Clips are the view's own section planes, added and edited
through the view (the plane id is identity and cannot be reassigned).

```python
from dataclasses import replace
from apeGmsh.results.session import Scope

view.scope = Scope("physical_groups", ("Cols",))
view.style = replace(view.style, nodes=True)

clip = view.add_clip((0, 0, 1), offset=1.0, name="mid-height")
view.set_clip(clip.plane_id, offset=2.0, flipped=True)
view.remove_clip(clip.plane_id)
```

A scope name the model does not carry is a `ValueError` at realize
time, and it names what the model *does* have.

`add_clip` takes the plane `normal` (normalised for you) plus keyword
`offset`, `name`, `active`, `flipped` and `gizmo_visible`, and returns
the minted `ViewClip`. `remove_clip(plane_id)` drops one, raising
`KeyError` on an unknown id.

::: apeGmsh.results.session.Scope

::: apeGmsh.results.session.MeshStyle

::: apeGmsh.results.session.ViewClip

## Time

An instant is `(stage, step)` — a *name*, so the step is a concrete
recorded index and negative aliases like `-1` are rejected. While the
link is on, the session instant is the instant of every pane; turn it
off and each pane keeps its own. `effective_instant()` is that law in
one place, mode poses included.

```python
s.time_linked                         # True
s.time = Instant("stage_0", 3)
s.effective_instant(view)             # Instant(stage='stage_0', step=3)

view.deform = Deform("displacement", mode=1)
s.effective_instant(view)             # None — a mode pose has no instant
```

::: apeGmsh.results.session.Instant

## Selection and plots

The session holds **one** selection set, nodes XOR Gauss: a write of
the other kind replaces the whole set rather than mixing the two. A
mesh view's `pick_target` only *aims* clicks in that view; it neither
owns nor clears the set.

"Select → new plot" copies the membership into concrete sources at
creation — the plot does not track later edits to the selection.

```python
from apeGmsh.results.session import PlotSeries, PlotSource

plot = s.add_plot(
    "history", series=[PlotSeries(PlotSource.node(9), "displacement_x")],
)

s.selection.set_nodes([5, 9])
s.selection.kind                      # 'nodes'
tip = s.add_plot_from_selection("displacement_x")
len(tip.series)                       # 2

realized = s.realize(pane=tip)        # arrays, no backend needed
realized.series[0].label              # 'node 5 · displacement_x'
```

::: apeGmsh.results.session.SessionSelection

::: apeGmsh.results.session.PlotSource

::: apeGmsh.results.session.PlotSeries

## Snapshot — the document on disk

`session.snapshot()` is the whole document as a JSON-safe dict: panes,
slots, pose, the time link and every pane's own instant, the one
selection set. Nothing derived (legends are a function of the slots)
and nothing about a window — which is the point: an agent can redraw
what a human arranged, with no Qt anywhere.

`save_snapshot()` writes it atomically to
`<results>.viewer-session.json` beside the results file — the same file
the window auto-saves on close.

```python
from apeGmsh.results.session import load_snapshot

view.contour = Contour("displacement_x")
path = s.save_snapshot()              # <results>.viewer-session.json

restored = load_snapshot(path, results=results)
restored.session.panes[0].contour     # Contour(quantity='displacement_x', ...)
restored.notices                      # () — nothing degraded
```

`load_snapshot` returns a `RestoredSession` carrying `notices`:
degradation is not refusal, so a snapshot naming a stage these results
no longer have still restores its panes and slots, drops the instant,
and says so. A file from the retired v13 viewer is not restorable — it
is renamed aside to `…​.viewer-session.json.legacy` and never
overwritten.

::: apeGmsh.results.session._snapshot
    options:
      members:
        - snapshot
        - save_snapshot
        - load_snapshot
        - restore_snapshot
        - default_snapshot_path
        - legacy_shape
        - rename_legacy_aside
        - RestoredSession
        - SnapshotError
        - LegacySessionFile

## See also

- [Render a deformed shape or contour](../how-to/render-a-still.md) — the scripted path, end to end.
- [Viewers](viewers.md) — the Qt window that projects this document.
- [Results](results.md) — the broker the session reads from.
