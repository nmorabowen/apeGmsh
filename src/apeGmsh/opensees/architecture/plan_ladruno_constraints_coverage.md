# Plan — close the apeGmsh ↔ Ladruno constraints coverage gaps

Scope decided with the user (2026-06-24): expose the remaining Ladruno-fork
constraint capabilities that the bridge does not yet reach. Four clusters,
each landed as its own PR matching the repo cadence (typed primitive →
resolver/build → 5 emitters → H5 schema bump → ADR → tests).

Source of truth for fork parameters: `OpenSees` branch `ladruno`
(`SRC/element/...`, `SRC/interpreter/OpenSeesOutputCommands.cpp`,
`SRC/modelbuilder/tcl/TclModelBuilder.cpp`).

---

## Coverage today (already wired — do not re-do)

equalDOF, rigidLink (beam/bar), rigidDiaphragm, fix/SP (`bc`/`face_sp`),
imposedMotion (load side), ASDEmbeddedNodeElement (K/KP/rot/pressure),
LadrunoKinematicCoupling (RBE2) + full `CouplingControl`,
LadrunoDistributingCoupling (RBE3) + weights + control,
LadrunoEmbeddedNode penalty/AL ties (translational), LadrunoEmbeddedRebar
(`reinforce`), constraint handlers Plain/Penalty/Transformation/Lagrange/
Auto/**LadrunoProjection**/**LadrunoContact**.

---

## Cluster B3 — `equalDOF_Mixed`  ✅ DONE (ADR 0069, schema 2.17.0)

OpenSees: `equalDOF_Mixed RnodeID CnodeID numDOF RDOF1 CDOF1 … RDOFn CDOFn`
(`TclModelBuilder.cpp:3995`). Couples *different* DOF indices on the
retained vs constrained node — apeGmsh has no equivalent (`equal_dof` is
symmetric; `node_to_surface` is mixed-ndf, not arbitrary DOF pairing).

Files:
- `_kernel/records/_kinds.py` — add `EQUAL_DOF_MIXED` + add to `NODE_PAIR_KINDS`.
- `_kernel/records/_constraints.py` — `NodePairRecord` gains `dofs_r`/`dofs_c`
  (or reuse `dofs` for c and add `master_dofs`). Keep existing `dofs` semantics.
- `core/constraints/defs/` + `ConstraintsComposite.equal_dof_mixed(...)`.
- resolver `_constraint_resolver.py` — co-location not required (mixed-DOF
  may be intentional non-coincident); tolerance optional.
- `opensees/emitter/base.py` — **Protocol widening (architecture event):**
  `equalDOF_mixed(self, master, slave, pairs: Sequence[tuple[int,int]])`.
- 5 emitters: tcl / py / live / h5 / recording.
- H5: new record kind round-trip (schema bump, additive minor).
- ADR `00xx-equaldof-mixed.md`.
- Tests: emit (tcl/py), namespace, validation, H5 round-trip.

## Cluster A1 — EmbeddedNode `pressure`/`kp` (+ staged rot/material follow-up)

Now: `EmbeddedNodeControl(CouplingControl)` adding `pressure: bool`,
`kp: float|None` → renders `-pressure [-kp Kp]` on the `penalty_al`
LadrunoEmbeddedNode tie route (`build._emit_penalty_al_tie`). Polymorphic
`emit_flags` so `_coupling_control_flags` needs no change. H5: add
`cpl_pressure`/`cpl_kp` (+ `sr_` mirrors), schema bump.

STAGED (one coupled follow-up, needs resolver + materials):
`-rot`/`-kr`/`-krAlpha` need host gradients `-dNdx` emitted by the
node-to-surface resolver; `-normal`/`-orient`/`-corot`/`-matN`/`-matT1`/
`-matT2` are the material-driven interface and need material-handle→ops-tag
translation. These are interdependent (parser: `-corot` errors without a
material; `-rot` errors without gradients) so they land together, not piecemeal.

## Cluster B2 — rigid body over a node set (`LadrunoRigidBody`, ele tag 33015)

NOT a swap of `g.constraints.rigid_body` — that ties slaves to a *user* master
node; `LadrunoRigidBody` builds its *own internal CoM node* + condensed mass
over a node set (`element LadrunoRigidBody tag ndm {slaveNodes…} [mUser]
[internalNodeTag]`). Expose as a NEW method, e.g.
`g.constraints.rigid_body_set(label, *, mass=None)` (no external master),
emitting the element (rides `emitter.element`, no Protocol change). New
record or a `NodeGroupRecord` flag (`as_element`, internal-CoM). Keep the
existing rigidLink-chain `rigid_body` untouched. Tests + ADR.

## Cluster B1 — LadrunoContact engine (largest; whole subsystem)

Replaces the `mortar()` `NotImplementedError`. Definition commands
(`OpenSeesOutputCommands.cpp:322+`):
- `contactSurface tag (-slave | -master n | -slave-segments n) nodes…`
- `contact tag master slave [kn kt mu]` NTS: `-outward -cell -geomtan -visc -soft auto`
- `contact … -mortar -epsN -augTol -maxAug -ngp` friction `-mu -epsT -cohesion -tauMax -consistentTan`
- mesh-tie `-tie -epsTie`; edge-edge `-edgeedge -edgeKn -edgeBand -edgeMu …`
- `contactPlane tag slaveSurf nx ny nz px py pz kn [-visc -soft]`
- ALM driver: `ladrunoBeginAugment` / `ladrunoEndAugment` wrapping analyze.

Work: new `ContactSurfaceRecord` + `ContactRecord` (resolve node sets from
physical groups at `get_fem_data`), resolver, build.py emit pass, Protocol
widening (`contact_surface`, `contact`, `contact_plane`, augment bracketing)
across 5 emitters, H5 zone + schema bump, viewer overlay (surfaces/pairs),
ADR(s), broad tests. The `LadrunoContact` handler is already exposed — this
fills the definition side it consumes.

Sequence: B3 → A1(pressure/kp) → B2 → B1; staged rot/material interface after B1.
