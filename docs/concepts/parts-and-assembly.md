# Parts & assembly

This page explains how a multi-part model becomes one connected whole — how a
Part is authored as a reusable template, how placing it creates instances whose
names stay addressable, and what fragmenting (or fusing) the assembly actually
does to the model you'll mesh.

[The session](session.md) already fixed the hierarchy: a Part is a template,
an instance is a placement, and the session is the one true model. What follows
is the machinery under that sentence — where the geometry lives, how the names
travel, and where the topology gets stitched.

## Authoring a template

Inside a Part you have the full geometry API — primitives, booleans,
transforms — and the two things worth doing deliberately are *naming* and
*annotating*. Names first: any `label=` you attach at geometry time is
recorded, and it will resurface on every instance you later place (how, in a
moment). Annotation second: `part.properties` is a free-form dict — material,
section, whatever metadata belongs to the template — and it rides along into
each instance placed from it.

```python
from apeGmsh import Part

column = Part("column")
column.begin()
column.model.geometry.add_box(0, 0, 0,  0.5, 0.5, 3.0, label="shaft")
column.properties["material"] = "concrete"
column.save()          # → column.step
column.end()
```

The file on disk is the contract between template and assembly, and the format
is STEP for a reason: STEP preserves the full parametric OCC geometry — exact
surfaces, topology, tolerances — not a tessellation. That's what lets the
assembly treat imported geometry as if it had been drawn there: boolean
operations still work, structured-mesh constraints can be applied after the
fact, and the mesh can be regenerated with any settings without ever reopening
the Part. (IGES is accepted too, for legacy exchange; STEP is the default and
the better choice.) Alongside the STEP file, `save()` writes a small JSON
sidecar carrying each label's geometric fingerprint — that sidecar is how your
names survive a round trip through a format that has no idea what an apeGmsh
label is.

You own the file when you name it — `save("column.step")` is a CAD artifact
you can version-control and hand to someone else. Skip `save()` and the Part
auto-persists to a tempfile on close, which is exactly enough to flow into
`g.parts.add(...)` in the same script.

## Placement: one template, many instances

`g.parts.add(part, ...)` imports the template's file into the session, applies
the placement transform, and registers the result as an **instance** under a
label:

```python
import math

g.parts.add(column, label="col_1", translate=(0, 0, 0))
g.parts.add(column, label="col_2", translate=(6, 0, 0))
g.parts.add(column, label="col_3",
            translate=(3, 0, 0),
            rotate=(math.pi/4, 0, 0, 1))   # 45° about Z
```

Rotation is `(angle_rad, ax, ay, az)` about the origin, or
`(angle_rad, ax, ay, az, cx, cy, cz)` about a custom center, and it is applied
*before* the translation — rotate the template in place, then carry it to its
position. If you omit `label`, one is generated from the part name and a
running counter; in practice, name every instance — the label is how you'll
address it for the rest of the model.

External CAD takes the same door: `g.parts.import_step("slab.step",
label="slab", ...)` imports a file from any tool with the same placement
arguments, plus optional healing and deduplication for geometry that arrives
less than clean (see [Import a STEP file](../how-to/import-step.md)). And
geometry that is already *in* the session can be adopted into the registry
rather than imported: wrap inline builds in `with g.parts.part("name"):` (as
the session page showed), or claim loaded entities after the fact with
`g.parts.from_model("name")` or `g.parts.register("name", dimtags)`. However
an instance came to be, it ends up the same kind of record — a label, a set of
entities, a bounding box — and that uniformity is what the next two sections
build on.

Placement isn't necessarily final, either. Every instance carries an
`inst.edit` composite (and every Part a `part.edit`) with `translate`,
`rotate`, `mirror`, `copy`, `pattern_linear`, `pattern_polar`, and `align_to`
— so a column placed once can be arrayed into a grid, and a template can be
mirrored into its opposite-hand twin, without touching the original geometry.

## How instances stay addressable

Place the same template twice and you have a naming problem: both copies were
authored with a face labeled `shaft`. apeGmsh resolves it by **prefixing** —
every label a Part carried is re-created on import as
`"{instance_label}.{label}"`, rebound to the placed geometry through the
sidecar's fingerprints (raw tags don't survive an import; geometric anchors,
transformed by the same placement, do). The instance also gets an umbrella
label equal to its own name, covering everything it owns. So after the
placements above, `"col_1"` is all of the first column, `"col_1.shaft"` is its
labeled box, and `"col_2.shaft"` is the same face on the other instance —
individually addressable, from one template.

The instance object makes those strings discoverable rather than memorized:

```python
inst = g.parts.add(column, label="col_1", translate=(0, 0, 0))
inst.labels.shaft                            # -> "col_1.shaft"
```

A typo raises immediately with the list of labels the instance actually has.
These prefixed names are ordinary labels — Tier 1, not yet solver-facing — so
the usual rule from [the mental model](mental-model.md) applies: promote what
the solver needs, when you need it:

```python
g.labels.promote_to_physical("col_1")
g.labels.promote_to_physical("col_2")
```

From here the instances are just names like any others — loads, constraints,
and selections target `"col_1"` exactly as they'd target a label you created
by hand.

## Fragment or fuse

Placed instances occupy space; they don't yet share topology. As
[Meshing](meshing.md) warned, two solids that merely touch get two independent
meshes — coincident nodes, no connection, a silently broken load path. The
assembly-level fix is one call:

```python
g.parts.fragment_all()
```

Fragmentation is a boolean operation that splits every shape at every
intersection, so each contact interface becomes a single shared surface in the
topology and the mesher produces one set of nodes on it. Three things about
the apeGmsh version are worth knowing. First, it fragments *every* dimension
present at once — a 2D shell wall standing on a 3D foundation becomes
conformal against the volume's face, not silently dropped (pass `dim=` to
restrict it). Second, it is registry-aware: OCC renumbers entities during the
operation, and `fragment_all()` rewrites every instance's entity records in
place, so labels, physical groups, and instance addressing all keep working on
the post-fragment geometry. Third, it warns about entities *not* tracked by
any instance — those participate in the boolean but can't be remapped, so
adopt strays with `from_model()` or `register()` before fragmenting. When only
one interface needs stitching, `g.parts.fragment_pair("beam_top", "col_1")`
does the same thing between two named instances and leaves the rest alone.

Fragmenting keeps the parts *distinct* — separate bodies that share nodes at
their interfaces, each still addressable by its own label. Sometimes that's
wrong: two placed volumes are really one continuous body, and the boundary
between them is an artifact of how you modeled it. Then you **fuse**:

```python
g.parts.fuse_group(["col_1", "col_2"], label="pier")
```

`fuse_group` dissolves the internal interfaces entirely and replaces the
listed instances with one new one — one body, one label, no interface to mesh.
The decision rule: fragment when the parts must remain distinct (different
materials or sections meeting at an interface), fuse when the split was never
physical to begin with. And when the two sides *can't* share a mesh — a coarse
solid against a fine one, a beam inside a volume — you do neither: leave the
meshes non-matching and declare a tie, as covered in
[Tie non-matching meshes](../how-to/tie-meshes.md).

## Composing finished models

Everything above happens *before* the mesh: Parts contribute geometry, the
session stitches it, and one mesh covers the whole. apeGmsh has a second,
later seam with a deliberately similar shape — `g.compose` merges a
previously *saved* model (`model.h5`) into a host session as a module:
already meshed, with its own loads, masses, and constraints resolved. Where a
Part is a geometry template, a module is a finished sub-model; where
instances get prefixed labels, composed modules get prefixed physical groups
(the host keeps its bare names), and the interfaces between modules are
coupled by constraints rather than fragmented — by compose time there is no
geometry left to fragment. The `Assembly` builder (imported from
`apeGmsh.assembly`, deliberately not top-level — the session *is* the
assembly) is a declarative front over the same machinery: declare saved
models as parts, declare the interface couples, and materialize the graph
into one composed session.

Concept-level, that's the whole map: Parts compose geometry before meshing
inside one session; `g.compose` composes finished models across sessions. The
mechanics live in [Compose saved modules](../how-to/compose-modules.md), and
the in-session workflow — Parts, placement, fragmentation, all the way to a
solved frame — is walked end to end in
[Multipart assembly](../examples/multipart-assembly.md).

---

*Next: [Sections](sections.md).*
