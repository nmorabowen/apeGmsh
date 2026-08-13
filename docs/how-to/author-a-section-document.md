# Author a section document (and open the builder)

Build a cross-section as a **saveable, re-editable artifact** instead of a
script that only runs once: a `SectionDocument` is versioned JSON that
describes one section completely, in either the continuum lane (shapes +
booleans + mesh → `SectionProperties`) or the fiber lane (patches + layers +
points + RC templates → a fiber recipe). Open it in the Qt builder, tweak a
bar diameter, re-analyze, hand it to a frame model — the round-trip section
design actually is.

The headless API below is the whole API. `launch_builder()` is an *editor*
for it: every GUI action calls one of these methods, so nothing in the window
is unreachable from a script.

## Recipe — an RC column (fiber lane)

```python
from apeGmsh.sections import SectionDocument

doc = SectionDocument.new(name="col500", kind="fiber", units="N, mm")

# 1. Materials. Dual-role: the continuum params (E, nu) feed the analyzer,
#    the uniaxial spec feeds the OpenSees handoff. Fiber documents need the
#    second; give the type token ops.uniaxialMaterial.<Type> takes.
doc.set_material("core", uniaxial=("Concrete01", {
    "fpc": -34.0, "epsc0": -0.004, "fpcu": -24.0, "epsU": -0.014,
}))
doc.set_material("cover", uniaxial=("Concrete01", {
    "fpc": -28.0, "epsc0": -0.002, "fpcu": -1e-9, "epsU": -0.006,
}))
doc.set_material("bars", uniaxial=("Steel01", {
    "fy": 420.0, "E": 200e3, "b": 0.01,
}))

# 2. An RC template — stored AS PARAMETERS and re-expanded on every build,
#    so editing `cover` moves every bar. core_split=True partitions the
#    concrete exactly into a confined core + a cover shell (you assign the
#    materials; the template never invents confinement parameters).
doc.add_template(
    "rc_rect_column",
    materials={"core": "core", "cover": "cover", "bars": "bars"},
    b=500.0, h=500.0, cover=40.0,       # cover is to the BAR CENTRE
    bars_x=4, bars_y=4,                 # per face, corners included
    bar_area=491.0, core_split=True,
)
doc.set_GJ(1.0e13)
doc.save("col500.section.json")

# 3. Build — deterministic expansion, no session, no bridge objects.
recipe = doc.build()
recipe.areas_by_material()          # exact ΣA per material — the sum check
```

`rc_circ_column(d, cover, n_bars, bar_area, core_split=)` and
`rc_beam(b, h, cover, top_bars, bottom_bars, bar_area, ...)` are the other
two templates. Anything a template does not cover, add literally with
`add_patch_rect` / `add_patch_circ` / `add_layer_straight` / `add_point`.

## Recipe — a composite face (continuum lane)

```python
doc = SectionDocument.new(name="src600")            # continuum is the default
doc.set_material("concrete", E=25e3, nu=0.2, fy=None)
doc.set_material("steel", E=200e3, nu=0.3, fy=345.0)

doc.add_shape("rect_face", id="concrete", b=600.0, h=600.0)
doc.add_shape("W_face", id="steel", bf=300.0, tf=20.0, h=360.0, tw=12.0)

# THE composite primitive: carve the inner region out of the outer one and
# fragment the pair conformally, in one step. The double-cover trap (cutting
# twice, or leaving duplicate edges) is not expressible through it.
doc.add_embed("concrete", "steel")

doc.set_mesh(lc=20.0, order=2)
doc.set_disconnected("raise")
sec = doc.build()                       # -> SectionProperties
sec.geometric().EIxx_c
```

Raw `add_cut(target, tool, remove_tool=)` and `add_fragment_pair(a, b)`
remain available for non-overlapping compositions and for punching holes
with a sacrificial tool shape.

Discrete rebar rides on top of a continuum face as an **overlay** —
`add_bar(material=, x=, y=, area=)` or `add_bar_line(n=, start=, end=, ...)`
— resolved through the fiber lowering at handoff. Concrete area is not
deducted at bar locations (standard fiber-section practice).

## Moment–curvature

```python
from apeGmsh.sections import moment_curvature

mc = moment_curvature(
    doc, axis="z", kappa_max=8e-5, n_steps=60, axial=-1500e3,
)
mc.EI0        # initial secant stiffness = the exact fiber sum ΣE·A·y²
mc.M_max      # largest |M| reached, carrying its sign
mc.complete   # False when a step failed before κ max
mc.curvature, mc.moment                # parallel tuples, starting at (0, 0)
```

`axis="z"` is DOF 6 — the strong axis, whose elastic slope is the analyzer's
`EIxx_c`; `axis="y"` is DOF 5 / `EIyy_c`. `kappa_max` is **signed**, so a
negative target walks the other quadrant (which is how an asymmetric
section's two directions get compared). `axial` follows the OpenSees sign
convention — **compression is negative** — and is applied first and held
constant through the push.

!!! warning "It wipes the OpenSees domain"
    `moment_curvature` builds a two-node `zeroLengthSection` harness in the
    **process-global** OpenSees domain, so it calls `wipe()` first. Do not
    call it while a live analysis is open. For the same reason it runs
    synchronously on the calling thread and must never be moved onto a
    worker: openseespy is a C++ runtime with no reentrancy.

## Hand off to a frame model

```python
# fiber lane — the document IS the section
col = doc.to_section(ops, name="col500")

# continuum lane — elastic lowering (no bars)
girder = ops.section.ComputedSection(analysis=doc.build())
# continuum lane WITH a bars overlay — the Gauss-fiber lowering + rebar
girder = doc.to_section(ops)
```

`handoff_snippet(doc, path="col500.section.json")` renders exactly those
lines as paste-ready text (the builder's **Copy handoff** button); the
continuum form re-opens the document by path and lowers it late, so numbers
are never hand-copied out of the GUI.

`doc.export_script("col500_build.py")` is the other direction: the plain
apeGmsh script the document is equivalent to. It is **one-way** by design —
round-trip editing is the JSON document's job — but it is the escape hatch
if you ever want out of the format.

## Open the builder

```python
from apeGmsh.sections import launch_builder

launch_builder()                          # blank continuum document
launch_builder("col500.section.json")     # edit an existing one
launch_builder(doc, blocking=False)       # notebooks: %gui qt first
```

* **Canvas** (left) — outlines pre-mesh, patch/layer/bar glyphs in the fiber
  lane. The freehand polygon tool has AutoCAD drafting aids: **F7** grid
  snap, **F9** object snap (endpoint / midpoint / centre / quadrant /
  intersection), **F8** ortho, and typed exact input accepting
  `length<angle`, `dx,dy`, and absolute `x,y`.
* **Palette** (right) — per-lane forms for shapes, booleans, bars, mesh
  preferences, fiber items, and the materials table. With apeSteel
  installed, a catalog box prefills the `W_face` form from any AISC v16 or
  EN designation (in **millimetres**, apeSteel's base — and its `h` is the
  *clear web height*, not the catalog depth `d`).
* **Properties dock** — tick **Live** to build + analyze after each edit on
  a worker thread (never the UI thread), and **Run M–κ** for the curve
  above.
* **Toolbar** — Open / Save / Undo (Ctrl+Z) / Redo (Ctrl+Y) / Copy handoff.

Both optional dependencies fail soft: without apeSteel the catalog box is
absent, without an OpenSees backend the M–κ button is greyed with guidance,
and neither is ever required headlessly.

## Gotchas

* **`cover` is to the bar centre**, not to the stirrup face — adjust for the
  tie diameter and bar radius yourself.
* **`bars_x` / `bars_y` count corners.** `bars_x=4, bars_y=4` is 12 bars,
  not 16: the top and bottom rows are straight layers of 4, and each side
  contributes `bars_y - 2` interior points.
* **`W_face`'s `h` is the clear web height** (total depth is `h + 2·tf`).
  The catalog picker maps this for you; typing a catalog `d` by hand builds
  a section two flange thicknesses too deep.
* **`units` is a label.** apeGmsh is unit-agnostic; the field documents your
  choice, it never converts.
* **Version window.** A document is read by loaders at its own minor version
  and the next one (`SECTION_DOC_VERSION`); an older apeGmsh refuses a newer
  document loudly rather than dropping the keys it does not know.
* **`build()` needs `set_mesh(lc=...)`** on a continuum document — it fails
  loud before opening any session rather than picking a size for you.

---

*See also: [Compute section properties](section-properties.md) ·
[Sections](../concepts/sections.md) ·
[Fiber sections & moment–curvature](../examples/fiber-moment-curvature.md)*
