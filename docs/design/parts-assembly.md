# Parts & assembly internals

This page explains the machinery behind the Part/Instance split — why it
exists, how names survive the trip through a STEP file, and how the registry
keeps its bookkeeping honest through OCC booleans — so you can predict what
`g.parts` will do and know where to change it.
[Parts & assembly](../concepts/parts-and-assembly.md) covers the same surface
from the user's side; this page is about what's underneath.

## One live session, therefore a file

Everything here follows from a single constraint: Gmsh keeps one
process-global OCC kernel. You can switch named models within a session, but
every boolean, transform, and import mutates the same singleton, so a Part
and an assembly cannot coexist in memory as independent geometries. Something
has to be serialized, and apeGmsh serializes the Part: its canonical form is
a STEP file on disk (plus a JSON sidecar for labels), not a Python object.
The alternative — juggling multiple Gmsh models inside one session — would
be faster, but a Part could then never outlive the session, and the whole
point of a template is that you author it once and reuse it next week. The
cost of the file-based route is that a Part must open and close a full Gmsh
session of its own, and that the label pipeline has to survive a round trip
through a format that has no idea what an apeGmsh label is. The rest of this
page is the price being paid, deliberately.

The implementation lives in four modules under `src/apeGmsh/core/`:
`Part.py` (the isolated geometry session), `_parts_registry.py`
(`PartsRegistry` and the `Instance` record — the assembly side),
`_parts_fragmentation.py` (the fragment/fuse mixin), and `_part_anchors.py`
(sidecar IO and rebinding). There is no `Assembly` class — the `apeGmsh`
session *is* the assembly, and `g.parts` is what makes it behave like one.

## The Part session

`Part` extends the same `_SessionBase` as `apeGmsh` but declares a
deliberately restricted composite set: `model`, `labels`, `physical`,
`inspect`, `plot`, and `edit` — no `mesh`, no solver composites. That
allow-list (`Part._COMPOSITES`) is the enforcement of the invariant that a
Part is geometry only; adding a mesh composite there would quietly break the
Part-as-CAD-artifact model, which is why contributor rule number one is
don't.

The lifecycle is built around auto-persist. If the user exits the `with`
block without calling `save()`, `end()` writes the geometry to a STEP file
in a fresh temp directory (`apeGmsh_part_{name}_*/{name}.step`) before
finalizing Gmsh — the ordering matters, because writing needs a live kernel,
and it must happen even when the user's build code raised (the export
attempt runs first, the user's exception still propagates). Auto-persist
failures are themselves caught and warned rather than raised: they must
never mask the build error underneath.

File deletion is governed by one bit, `_owns_file`. It is true only when the
library wrote the file into a temp directory it created; it is never true
after an explicit `save()`, no matter what happened before. Cleanup runs
through a `weakref.finalize` that closes over the temp-dir path — not over
`self`, which would pin the Part alive — so the tempfile disappears on
garbage collection or on an explicit `cleanup()`. Calling `save()` after an
earlier auto-persist reclaims the temp directory first and then hands
ownership of the new file to the caller. Re-entering a Part (`begin()` on an
object that owns a stale tempfile) cleans up before the new session starts.
The rule all of this serves: the library never deletes a file the user
named. For a CAD tool that is the worst-case bug class, and every new code
path that might remove a file must gate on `_owns_file`.

## The sidecar

STEP cannot carry apeGmsh label strings — Gmsh writes through OCC's plain
STEP writer, which doesn't expose XDE naming, and third-party readers drop
it anyway — so labels travel in a JSON sidecar written next to the CAD file
(`{name}.step.apegmsh.json`). Inside a Part, any geometry call with
`label=` auto-creates a Tier 1 label PG (the Part sets
`_auto_pg_from_label = True` precisely so labels exist to be exported). At
save time, `collect_anchors` walks those label PGs and records one anchor
per entity: the label name, the entity's dimension, its center of mass in
Part-local coordinates, and its bounding box. Only Tier 1 labels are
captured — solver-facing PGs are the user's business in the assembly and
don't travel through files. No named entities, no sidecar.

At import time the anchors are re-anchored geometrically. `_import_cad`
applies the same rotate-then-translate placement to each stored COM (and to
the eight corners of each stored bbox) that it applied to the geometry, then
matches transformed anchors against imported entities of the same dimension.
Matching is greedy and exclusive: every (anchor, candidate) pair within
tolerance is scored by COM distance with bbox distance as the tiebreaker,
the pairs are sorted best-first, and each entity can be claimed once. The
tolerance is scale-aware — a fraction (1e-4) of the imported geometry's
bounding-box diagonal — so the same code works in millimetres and metres.
The bbox tiebreaker is what disambiguates symmetric Parts, where two faces
can be equidistant from a stored COM.

One subtlety: imports default to `highest_dim_only=True`, but a sidecar can
carry anchors at any dimension (a face label on a solid). Rebinding
therefore matches against *all* entities in the model, re-enumerated per
dim, not just the dimtags `importShapes` returned — otherwise every
sub-dimension anchor would silently miss.

Failure policy is uniform across the whole pipeline: sidecar write
failures, unreadable sidecars, unmatched anchors, and label-PG creation
failures all warn and continue. A broken sidecar degrades to "labels don't
round-trip", which is recoverable; a failed import or a failed CAD export is
not, so nothing on the label path is allowed to raise.

## The registry

`Instance` is a small record dataclass: label, source part name, file path,
an `{dim: [tags]}` entity map, the placement transforms, user properties,
a bounding box, and the list of prefixed label names created for it. Two
fields deserve attention. `entities` is mutable *by design* — fragmentation
rewrites it in place (next section), because user code holds references to
Instance objects and replacing the record would silently detach them.
`labels` is the `_InstanceLabels` helper: a slotted wrapper whose
`__getattr__` prepends `"{instance_label}."` and checks the result against
`label_names`, so `inst.labels.web` returns the string `"col.web"`, a typo
raises `AttributeError` listing what actually exists, and `__dir__` feeds
the stripped names to IDE autocomplete. The user never types a raw prefixed
string — which also means a typo can never silently resolve to an empty
selection.

Every way of creating an instance — the `part()` context manager (a
before/after diff of `gmsh.model.getEntities` per dim), `register()` (by
explicit dimtags, or by adopting an existing label or physical group, with
an ownership check that rejects entities already claimed by another part),
`from_model()` (adopt everything untracked), `add()` (the canonical
Part-import path), and `import_step()` (third-party CAD, with optional
healing and deduplication) — funnels through one private method,
`_register_instance`. That funnel is where every instance gets its
`inst.edit` composite wired up, and where the chain-phase guard fires:
parts registration is build-phase only, because the chain-phase broker
carries its own immutable parts snapshot (ADR 0038). The parametric box
builders (`add_plane_wave_box`, the DRM boxes — ADR 0054, ADR 0066) build
directly in the live session with no STEP round-trip, but their instances
go through the same funnel.

`add()` and `import_step()` share `_import_cad`, which does the work in
order: import the file, deduplicate the returned dimtags (OCC reports
shared sub-entities once per parent), apply rotate then translate to the
*highest-dim entities only* — OCC propagates transforms through
sub-topology, and transforming every dim double-transforms the sub-shapes —
then rebind the sidecar, create one prefixed label PG per matched anchor,
and finally create the umbrella label (the instance's own name, covering
its top-dim entities). Prefixing is what makes two instances of one Part
disjoint by construction: the registry rejects duplicate instance labels on
every entry path, and every sidecar label lands as a fresh
`{instance}.{name}` PG. The registry never creates Tier 2 physical groups —
promotion is the user's explicit act, and any new feature that "helps" by
creating solver-facing PGs is bleeding internal naming into solver output.

`delete()` is the deliberately asymmetric operation: it drops the registry
record but leaves the geometry in the session, where it becomes untracked —
visible in the viewer and, importantly, a warning source at fragment time.

## Fragmentation bookkeeping

The three boolean operations live in `_PartsFragmentationMixin` — the one
sanctioned mixin in the codebase, justified because fragment/fuse is a
closed three-method surface that shares every registry field. Each wraps
its OCC call in `pg_preserved()`, the snapshot-then-remap context that keeps
labels and physical groups alive through OCC's renumbering; what this
section adds is the *registry-side* remap that runs inside the same block.

OCC booleans return a `result_map`: for each input dimtag, the list of
output dimtags that replaced it. `_remap_from_result` turns that into a
per-dim `old_tag → [new_tags]` table — matching inputs only to outputs at
the *same* dimension, so a surface entry can't absorb a new volume — and
flattens every instance's `entities[dim]` through it in place. One old tag
can expand to several new tags when the fragment split it; tags no instance
owns are ignored. Fuse has a wrinkle: OCC often returns empty result maps
for fused inputs even though a merged entity exists, so with
`absorbed_into_result=True` an input with an empty map remaps to the
surviving same-dim result entities rather than being silently emptied.

`fragment_all` auto-detects every dimension present (a 2D shell standing on
a 3D solid joins the same OCC call as the volume, so it becomes conformal
instead of being dropped) and warns — with a dedicated
`UnregisteredPartEntityWarning` — about entities no instance tracks: they
participate in the boolean but cannot be remapped, so the user sees the
drift coming instead of discovering it later. `fragment_pair` is the local
variant: only the two named instances' entities enter the OCC call, mixed
dimensions included, and their records are remapped through the same
machinery. `fuse_group` removes the input instances from the registry
entirely and registers one new instance over the fused result (named after
the first input unless overridden) — the one operation where instances die.
Both fragment variants also reap stale per-entity metadata for inputs OCC
consumed, so a later pre-mesh validation doesn't trip over keys that no
longer exist.

## After the mesh: bbox partition

The sidecar's anchors are spent once `parts.add()` returns, and the entity
maps are geometry-time bookkeeping — neither survives into solver land.
What carries instance identity across the mesh boundary is spatial:
`build_node_map` partitions mesh nodes by each instance's axis-aligned
bounding box, vectorized in numpy with a tolerance of 1e-6 times the box
span to absorb OCC float jitter without letting neighbours bleed into each
other. `build_face_map` then assigns a surface element to an instance iff
all its nodes belong to it. `FEMData.from_gmsh` snapshots the node map into
the broker, which is why `fem.nodes.get(target="col_A")` still works after
the Gmsh session is gone. The umbrella and prefixed labels take the other
route — they are ordinary Tier 1 names, promoted and persisted like any
label. Bbox containment is deliberately crude: it is exact for the
disjoint-parts case it serves, and anything finer belongs to labels, which
are exact by construction.

---

*Next: [Results internals](results.md).*
