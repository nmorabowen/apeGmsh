# P4 scope — cross-dim gravity + the LOAD-1 bridge-emit half

Scoping note for [ADR 0050](../src/apeGmsh/opensees/architecture/decisions/0050-dimension-indexed-loads-and-displacements.md)
P4. Grounded in a read of the apeSees bridge as it stands on branch
`guppi/apesees-todo-list`. Everything below cites file:line evidence so
the plan rests on the *actual* emit paths, not the ADR's assumptions.

## 1. What the bridge actually emits today (verified)

The apeSees bridge (`opensees/apesees.py`) auto-consumes the FEM broker
**partially**. Main emit path (`_Build.emit`, the block at
`apesees.py:873-884`):

| Broker channel | Resolved by | Auto-emitted? | Path |
| --- | --- | :---: | --- |
| `fem.nodes.loads` (NodalLoadRecord) | g.loads point/line/surface/volume/gravity (nodal form) | ✅ | `_emit_broker_loads` (`apesees.py:3299`) → synthesized Plain pattern + `emitter.load` |
| `fem.nodes.constraints` (MP) | g.constraints equal_dof/rigid/… | ✅ | `emit_mp_constraints` + `build.py:1935-1977` |
| `fem.nodes.sp` (SPRecord) | **g.constraints.bc + g.displacements** | ❌ | *no broker-SP emitter exists* |
| `fem.nodes.masses` (MassRecord) | **g.masses** | ❌ | *no broker-mass emitter exists* |
| `fem.elements.loads` (ElementLoadRecord) | g.loads `target_form="element"` (beamUniform / surfacePressure / bodyForce) | ❌ | *no eleLoad emitter exists* |

`_emit_fixes` / `_emit_masses` (`apesees.py:2974-2982`) emit only the
**bridge-level** `self.fix_records` / `mass_records` — directives from
explicit `apeSees.fix(...)` / `apeSees.mass(...)` (`apesees.py:459-461`,
`3748`), **not** the broker channels.

### Correcting LOAD-1's framing

LOAD-1 says *"apeSees silently ignores **all** g.loads declarations."*
That is too broad: **nodal** loads from `g.loads` *do* emit
(`_emit_broker_loads`). The real, verified gaps are narrower and sharper:

- **G1 — element-form loads never emit.** `fem.elements.loads`
  (`eleLoad`-style) has no bridge consumer. This silently drops:
  - `g.loads.line(..., target_form="element")` — the **fixed-end-moment**
    path for beams (ADR 0050 §5 reason 1).
  - any `target_form="element"` surface/volume load.
- **G2 — cross-dim gravity is impossible.** `GravityLoadDef` is
  volume-only (`expected_dim=3`); nodal gravity on a solid works (it
  becomes `fem.nodes.loads`), but a **beam** (ρ·A·g) or **shell** (ρ·t·g)
  self-weight cannot be expressed at all — see §3.
- The `body_force=` workaround (LOAD-2) is the per-**solid**-element
  argument (`element/solid.py:99` etc.): `ops.element('stdBrick', …, bx,
  by, bz)`. It is element-integrated, 3-D only, and bypasses the
  composite's tributary/consistent choice — exactly LOAD-1's complaint,
  and inapplicable to beams/shells.

**Adjacent (out of P4 scope, but same bug class — flag for a sibling
item):** broker `fem.nodes.sp` (G3) and `fem.nodes.masses` (G4) are also
never auto-emitted. So `g.constraints.bc`, `g.displacements`, and
`g.masses` currently reach OpenSees only if the user *also* hand-writes
`apeSees.fix` / `ops.fix` / `apeSees.mass`. This is a real asymmetry with
broker loads/constraints, but it is **not** what P4 set out to fix; it
deserves its own todo (proposed: **BRIDGE-1 — symmetric broker-channel
emission**).

## 2. Where section data lives (for cross-dim gravity)

Cross-dim gravity needs ρ and the cross-section measure. Both exist in
the bridge's typed primitives:

- **ρ (density):** on materials — `material/nd.py:68` (`ElasticIsotropic.rho`),
  and uniaxial/nd variants.
- **Beam area A:** `section/beam.py` (`ElasticSection.A`); fiber sections
  derive A from patches/layers (`section/fiber.py`).
- **Shell thickness t:** `section/plate.py:125` (`ShellLayer.thickness`,
  `ShellSection`).

The missing piece is a **traversal seam**: given an element (or its tag),
walk element → assigned section → A / t, and element/section → material →
ρ. No such "introspect the section measure for this element" accessor
exists yet; it is the load-bearing new infrastructure for P4.

## 3. Per-dimension gravity reality (the design crux)

Self-weight emits **differently per element dimension** — there is no one
OpenSees mechanism:

| Target dim | Element | Self-weight intensity | OpenSees emission |
| --- | --- | --- | --- |
| 3 | solid (brick/quad) | ρ·g per volume | already works **nodal** (`fem.nodes.loads`); OR per-element `body_force=` arg |
| 1 | beam | ρ·A·g per length | `eleLoad -ele $t -type -beamUniform Wy Wz <Wx>` (needs **G1** + A·ρ) |
| 2 | shell | ρ·t·g per area | **no body-force eleLoad for shells** — must lump to **nodal** (ρ·t·A_trib·g) bridge-side, or use `-accel` UniformExcitation w/ element mass |

So "cross-dim gravity" is **not** uniformly "emit element bodyForce." It
is: dim-3 nodal-or-bodyforce, dim-1 beamUniform eleLoad, dim-2 nodal
lumping from section thickness. Each needs the §2 seam to fetch A / t / ρ.

## 4. Proposed phasing (P4a → P4c)

Smaller, independently shippable slices, ordered by dependency:

### P4a — element-load emission seam (closes G1; no gravity yet)
Add a bridge consumer for `fem.elements.loads` — a `_emit_broker_element_loads`
mirroring `_emit_broker_loads` (one Plain pattern per `rec.pattern`,
pattern-scoped). Walk the ElementLoadRecord set, map `load_type` → the
**already-existing** `emitter.eleLoad` primitive (`emitter/base.py:220`,
implemented in h5/tcl/py — no new emitter method needed):
- `beamUniform` → `emitter.eleLoad(ele, "-beamUniform", wy, wz, wx)`
- `surfacePressure` → `emitter.eleLoad(ele, "-surfacePressure", p, …)`
- `bodyForce` → see open-Q 3 (solids take body force as an **element
  arg**, not `eleLoad` — this branch may rewrite the element spec rather
  than emit a load).

So P4a is mostly: the broker walk + `load_type`→eleLoad-args mapping +
hooking `_emit_broker_element_loads` into the emit block at
`apesees.py:873-877` (and the partitioned path at `1718`). The
emitter-protocol primitive is already there.
→ verify: `g.loads.line(target_form="element")` on a beam yields correct
fixed-end moments in a solved model; element-load records reach the deck
(assert on captured deck lines, as `tests/opensees/unit/test_emitter_*`).

### P4b — cross-dim `GravityLoadDef` (mesh-side) (part of G2)
- Widen `GravityLoadDef` to accept dim 1 / 2 / 3 targets (drop the
  volume-only `expected_dim=3`).
- dim 3: unchanged (nodal ρ·V·g, or element bodyForce).
- dim 1 / 2: emit **element** records carrying `{g, density|None}` — the
  bridge expands them in P4c. (Mesh-side cannot compute ρ·A / ρ·t.)
→ verify: dim-1/2 gravity produces ElementLoadRecords (no mesh-side nodal
lumping attempted without a section); dim-3 unchanged.

### P4c — section-introspection seam + per-dim gravity expansion (closes G2)
- Build the element → section (A / t) → material (ρ) traversal (§2).
- Beam gravity → `beamUniform` with Wz = −ρ·A·g (via P4a's eleLoad).
- Shell gravity → bridge-side **nodal** lumping ρ·t·A_trib·g (no shell
  eleLoad).
- `density=None` → read ρ from the assigned material via the same seam
  (ADR 0050 open-Q 4: confirm one seam serves both gravity and general
  element-form emit).
→ verify: a beam self-weight reaction == ρ·A·L·g; a shell self-weight
reaction == ρ·t·Area·g; `density=None` reads the material.

### P5 (already planned) — docs reconciliation
DOC-1 (guide ⇄ skill auto-emit contradiction — now answerable precisely:
nodal loads + constraints auto-emit; SP/masses/element-loads do **not**
yet), LOAD-2 (2-D `body_force` example), DOC-2 (namespace docstrings).

## 5. Decisions (ratified)

1. **Scope of P4 vs BRIDGE-1 — SPLIT.** G3/G4 (broker SP + masses not
   emitted) become a **separate BRIDGE-1 item**, done right after P4. P4a
   still builds the reusable "walk a broker channel → emit per-pattern"
   shape so BRIDGE-1 is cheap. P4 stays focused on element-loads (G1) +
   gravity (G2).
2. **Shell gravity — NODAL lumping** for v1 (ρ·t·A_trib·g bridge-side, no
   `-accel`/mass coupling).
3. **`bodyForce` on solids — REWRITE the element constructor arg**
   (source-verified). `Brick::addLoad` (`OpenSees/src/element/brick/Brick.cpp:653-674`):
   a brick's body-force magnitude comes **only** from constructor args
   `b[0..2]`; `eleLoad -type -BrickSelfWeight` does `appliedB += loadFactor·b`
   and `-selfWeight` does `appliedB += loadFactor·data·b`. So `eleLoad`
   alone cannot carry a magnitude. The `bodyForce` element-record branch
   therefore sets `body_force=(bx,by,bz)` on the element spec
   (`element/solid.py:99` already supports it); for pattern-scoped/ramped
   gravity it *additionally* emits `eleLoad -ele $t -type -selfWeight gx gy gz`
   inside the pattern. **Corollary:** solid gravity already has a working
   **nodal** path (`g.loads.gravity` nodal → `fem.nodes.loads` → emitted,
   with the tributary/consistent choice) — solids are *not* blocked; only
   beams + shells are.
4. **Fiber-section beam area — DEFERRED (not a concern for v1).** P4c
   scopes beam gravity to sections that expose `A` explicitly
   (`ElasticSection`, `section/beam.py`); fiber-section beam self-weight
   (where A = Σ fiber areas) **fails loud** with a clear message and is a
   later add. Removes the implicit-area traversal from the P4 critical
   path.

## 6. Verification strategy (no GPU needed)

All P4 work is deck-level (Tcl/py emit) + solved-reaction checks — no
viewer. Use the `opensees_venv` python, assert on emitted deck lines
(`emitter` capture, as `tests/opensees/unit/test_emitter_*.py` do) and on
solved reactions for the gravity cases (statics, analytic ρ·V·g /
ρ·A·L·g / ρ·t·Area·g).
