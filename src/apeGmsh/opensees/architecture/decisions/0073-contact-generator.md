# ADR 0073 — `g.constraints.contact(...)` → fork `contactSurface` + `contact`

**Status:** Accepted (2026-06-25). The face-to-face contact generator (#722)
+ the adversarial-review hardening that gated its merge. Rides the
`emitter.contact_surface(...)` / `emitter.contact(...)` Protocol methods
(added with the contact stack) — paired with the existing
`emitter.constraints("LadrunoContact")` handler emit.

## Context

The Ladruno fork ships a node-to-segment / mortar contact subsystem as a
pair of commands plus a constraint handler
(`SRC/interpreter/OpenSeesOutputCommands.cpp`):

```
contactSurface tag (-slave | -master nps | -slave-segments nps) nodeTag…
contact tag masterTag slaveTag [kn kt mu | auto] [-outward ox oy oz]
        [-mortar -epsN auto|v -mu -epsT -cohesion -tauMax -augTol -maxAug
         -ngp -tie]
constraints LadrunoContact
```

A contact interaction is two named meshed faces. The **master** is always a
faceted surface (`-master nps`, flat corner connectivity with stride
`nps ∈ {3,4}`); the **slave** is either a node set (NTS, `-slave`) or a
faceted surface (mortar, `-slave-segments nps`). The runtime needs the
`LadrunoContact` constraint handler to inject the contact FE adapters into
the assembly.

`g.constraints.contact(master, slave, formulation="nts"|"mortar", …)`
declares the interaction at the geometry level; after meshing it resolves
to a `ContactRecord` and emits the two `contactSurface` defs + the `contact`
verb (+ the handler auto-emit). Fork-only: the commands are unavailable on
stock openseespy and bite at run time only — deck emission works on any
build.

## Decision

**API.** One method on the constraints composite:

```python
g.constraints.contact(master, slave, *, formulation="nts",
                      kn=None, kt=None, mu=None,                 # NTS
                      eps_n=None, eps_t=None, cohesion=None,     # mortar
                      tau_max=None, aug_tol=None, max_aug=None,
                      ngp=None, tie=False, outward=None,
                      master_entities=None, slave_entities=None, name=None)
```

`nts` takes `kn`/`kt`/`mu` (penalty + Coulomb friction; `kn="auto"`
allowed); `mortar` takes `eps_n`/`eps_t` (ALM penalty, `"auto"` allowed) +
the friction-cone / Uzawa flags + `tie` (permanent mesh-tie bond). The two
parameter families are mutually exclusive (validated at construction).

**Resolve → emit.** `ConstraintsComposite.resolve_contacts` pulls the master
faceted surface (+ slave node set / faceted surface) from the live Gmsh
session into a `ContactRecord`; `build.emit_contacts` emits two
`contactSurface` defs (own tag namespace) + the `contact` verb (own
namespace), and `_maybe_auto_emit_constraint_handler` emits
`constraints("LadrunoContact")` whenever any contact is present. The two
grammar builders (`element/contact.py::contact_surface_args`/`contact_args`)
are the single source of truth for the token streams.

The following correctness decisions were forced by an adversarial review
against the fork parser/kernel (branch `ladruno`) and gate the merge:

* **Outward normal is never auto-derived.** The kernel computes a correct
  PER-FACET normal from each facet's connectivity
  (`LadrunoContactProjection.h::normalOriented`) and uses a supplied
  `-outward` only as a single GLOBAL sign reference
  (`LadrunoContactHandler.cpp` `orientDir`). On a curved / closed /
  solid-part master a single global outward silently SKIPS facets
  ~perpendicular to it (Gate H2 fail-safe) and INVERTS facets opposed to it
  (→ inward, wrong contact). When `-outward` is absent the kernel uses a
  correct per-pair `slave − segment-centroid` sign reference. So
  `resolve_contacts` passes `outward` ONLY when the user set it explicitly;
  the prior single-normal auto-derivation (`_outward_for`) is **deleted**. The
  one case that needs an explicit `outward=` is an initially-COINCIDENT
  (zero-gap) flat contact: there the per-pair reference is in-plane and the
  kernel's gate H2 refuses the ambiguous pair, so the user pins the sign
  (matches the fork's "use -outward for just-penetrated starts"). Documented
  on `ContactDef.outward`. A `tie=True` mortar mesh-tie is ALWAYS that
  coincident-flat case, so `ContactDef` fail-louds at construction when
  `tie=True` and `outward` is unset (without it the tie silently binds
  nothing). Range validation accepts the fork's documented zero sentinels
  (`kn`/`eps_n=0` inert path, `tau_max=0` no-cap) but rejects negative and
  non-finite penalties and non-integer `ngp`/`max_aug`.
* **`nts` bare numeric `kn` + `-outward` emits the full `kn kt mu` triple**
  (padding `kt=mu=0.0`). The fork's numeric `kn`-slot reader sizes its
  double read as `m = (remaining >= 3) ? 3 : 1` counting ALL trailing
  tokens (flags included), so a bare numeric `kn` directly followed by
  `-outward` made it read `-outward` as the second double and abort the
  `contact` command. The `auto` and no-`kn` paths peek-and-unread the flag
  safely, so the padding is gated on a numeric `kn` preceding `-outward`.
* **Higher-order surfaces are dropped to corner facets.** `_collect_surface_faces`
  returns full-order connectivity (tri6→6, quad8→8, quad9→9);
  `resolve_contacts` reduces master/slave faces to corner facets (gmsh
  orders corners first) and the grammar builder enforces `nps ∈ {3,4}`. The
  fork contact subsystem understands only 3-node / 4-node facets.
* **Serial-only ⇒ fail-loud under partitioned (MPI) emit.** The fork contact
  subsystem is not parallel; the partitioned emit path never calls
  `emit_contacts`. `_emit_partitioned` raises `BridgeError` when contacts
  (or g.embed ties) are present rather than silently dropping them (mirrors
  the reinforce-ties / rebar-elements guards).
* **Handler exclusivity is fail-loud.** Only one constraint handler can be
  active. Contact forces `LadrunoContact`, which (a) cannot enforce an
  `enforce="equation"` tie (EQ_Constraint) and (b) is Plain-style for MP
  constraints (fork P1a) — equalDOF / rigidLink / rigidDiaphragm /
  couplings are NOT enforced. A model combining contact with either an
  equation tie or any MP constraint raises `BridgeError` rather than
  emitting a silently-wrong deck.
* **Range validation** on `ContactDef` (`kn>0`, `kt`/`mu`/`cohesion>=0`,
  `eps_n`/`eps_t`/`tau_max`/`aug_tol`/`max_aug`/`ngp>0`, non-zero
  `outward`) — the fork parser does not enforce these for the plain contact
  path.

**Fork gating.** `contactSurface`/`contact` are gated in the live emitter
(fail loud on stock openseespy); deck emission (`.tcl()`/`.py()`) works on
any build.

## Scope / deferred

* **Native H5 persistence is deferred.** The H5 emitter no-ops contact (and
  g.embed) records and raises a one-time `H5FeatureDeferredWarning`
  (renamed from the misnomer `H5ReinforceDeviationWarning`; back-compat
  alias kept). A contact model must be emitted to Tcl / openseespy (or run
  live) for a complete deck. No neutral schema columns for contact yet.
* **Explicit-only / solver-coupled flags deferred:** `-soft`, `-visc`,
  `-consistanttan`, `-geomtan`, the edge-edge lane (`-edgeedge` + `-edge*`),
  and the rigid-plane `contactPlane` command.
* **Curved higher-order embed hosts.** g.embed linearises hosts to corner
  sub-elements; a genuinely curved host is detected (mid-side node outside
  the corner bounding box) and warned-once + documented (corner
  linearisation may mislocate nodes) rather than supported.
* **`g.constraints.mortar()`** (the old Lagrange-multiplier surface-tie
  stub) remains `NotImplementedError`; its message now points to the
  fork-backed `contact(formulation="mortar", tie=True)` for a real
  segment-to-segment mortar bond. Whether `mortar()` should *delegate* to
  the fork contact tie (a breaking return-type / semantics change — Lagrange
  tie vs ALM penalty contact-tie) is an open follow-up.

## Consequences

* `g.constraints` gains a fork-backed contact generator (NTS penalty +
  mortar/ALM, friction, mesh-tie) with emit verified against the fork
  parser. Contact + MP / equation / partitioned models fail loud rather
  than emit silently-wrong decks. No `Emitter` Protocol change beyond the
  contact methods already added with the stack. H5 round-trip of contact is
  a documented open item.
