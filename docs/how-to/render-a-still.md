# Render a deformed shape or contour

**Goal.** Turn a solved `Results` into a picture from a script — no
window, no notebook, no Qt event loop — so it can run in CI, in a
report pipeline, or under an agent.

The interactive door is `results.viewer()`. What it opens is a
**`ResultsSession`**: the document that says what is on screen, at
which time, with which pictures. `results.session()` hands you that
same document with no window attached, and `session.render()` draws a
still off it.

## The whole recipe

```python
from apeGmsh import Results
from apeGmsh.results.session import Contour, Deform, Instant

results = Results.demo()                    # your own Results works the same

s = results.session()                       # the document — no window
view = s.panes[0]                           # the mesh view it booted with

view.contour = Contour("displacement_x")    # fill the contour slot
view.deform = Deform("displacement", scale=5.0)   # set the pose
s.time = Instant("stage_0", 5)              # pick the instant

png = s.render("tip.png", camera="xy", window_size=(1600, 900))
```

`render()` returns the `Path` it wrote, or `None` if it skipped (see
[In CI](#in-ci) below). `camera=` is `iso` / `xy` / `xz` / `yz`, and
defaults to `xy` for a planar model and `iso` otherwise.

`Results.demo()` is a real cantilever pushover with no files to supply
— handy for trying this cold. Swap in `Results.from_native(...)` /
`from_mpco(...)` and the rest of the recipe is unchanged.

## Name a real stage and step

`Instant` is a *name*, not an index trick: the stage id must exist and
the step is a concrete recorded index (`-1` is rejected, because one
instant with two names poisons the time link and the snapshot). Read
both off the broker:

```python
for stage in results.stages:
    print(stage.id, stage.kind, stage.n_steps)
# stage_0 static 6

stage = results.stages[-1]
s.time = Instant(stage.id, stage.n_steps - 1)      # the last recorded step
```

## Swap the picture

The picture lives in the view's **slots** — a closed catalog of seven
(`contour`, `vector`, `gauss`, `line`, `sand`, `loads`, `reactions`).
Different categories stack; filling an occupied one **replaces** its
occupant, and assigning `None` clears it.

```python
view.contour = Contour("displacement_z", averaging="unaveraged")
view.contour = Contour("displacement_x")   # replaces — never a 2nd contour
view.contour = None                        # clears
```

`deform` is deliberately not one of the seven: it is a *pose*, so it
warps the mesh and causes no colour scale. The scale you see is
named by the slot, so warping by displacement while contouring stress
gives one scale and it says stress.

See [Results session](../api/results-session.md) for the whole catalog
and the legend law.

## Narrow it down

Scope picks the one cell set the whole view is a function of — one
composition axis plus checked names. Clips are the view's own section
planes.

```python
from apeGmsh.results.session import Scope

view.scope = Scope("physical_groups", ("Cols",))     # or materials / element_types
view.add_clip((0, 0, 1), offset=5.0, name="mid-height")

s.render("scoped.png")
```

A name the model does not carry raises `ValueError` at render, and the
message lists the names it does have.

## More than one pane

With one pane `render()` needs no address. Add a second and every call
takes a pane id — the same stable id the snapshot and the window's
outline use.

```python
second = s.add_view(name="undeformed")
[p.id for p in s.panes]                    # ['mesh-1', 'mesh-2']

s.render("deformed.png", pane=view.id)
s.render("plain.png", pane=second.id)
```

!!! note "Plot panes are arrays, not stills"
    `session.render()` writes stills of **mesh** panes; asking it for a
    plot pane raises `NotImplementedError`. Use
    `session.realize(pane=...)` to get the resolved arrays and draw
    them however you like:

    ```python
    from apeGmsh.results.session import PlotSeries, PlotSource

    plot = s.add_plot(
        "history", series=[PlotSeries(PlotSource.node(9), "displacement_x")],
    )
    realized = s.realize(pane=plot.id)
    realized.series[0].time      # array([0. , 0.4, 0.8, 1.2, 1.6, 2. ])
    realized.series[0].values
    ```

    For a ready-made chart, `results.plot.history(...)` is the
    matplotlib route.

## Hand the same picture to the window

The session is what the window saves, so a scripted arrangement can be
opened by a human — and vice versa.

```python
s.save_snapshot()          # -> <results>.viewer-session.json
```

`results.viewer()` picks that file up on the next open (it asks first
by default), and returns the session it projected once the window
closes.

## In CI

`APEGMSH_SKIP_VIEWER=1` makes `render()` print `[skip viewer]`, write
no file, and return `None` — so the same script runs headless without a
GL context, and a `None` return is the signal that no picture was
produced.

```python
png = s.render("tip.png", pane=view.id)
if png is None:
    ...        # skipped: no GL, or APEGMSH_SKIP_VIEWER is set
```

---

*Next: [Results session](../api/results-session.md) — the full API of
the document you just drove.*
