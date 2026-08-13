# ADR 0093 — `g.constraints.interface()`: oriented coincident-pair zeroLength interface

**Status:** Proposed (2026-08-12) — driven by the Cerro Lindo SSI program
(`Informe No3/Project/ADR/ADR-0005-ssi-squeezing-interaction-model.md` D1.3 /
D4 / D8.0): a steel arch (`dispBeamColumn` wire) node-for-node coincident with
a tunnel-face continuum boundary needs a **unilateral** (compression-only,
separation allowed) and **strength-capped** (elastic–perfectly-plastic slip at
τ_b × A_trib) interface. With a bilateral bond the model cannot reproduce
demand saturation — ground converges, arch demand grows without bound — and
the field record (metre-class convergence coexisting with damaged-but-standing
arches) falsifies that model class. The verb is therefore mandatory physics
for that campaign, not a convenience. Three scope decisions were taken at
sign-off of the plan: **declarative per-area laws** (not opensees primitives
in the verb), **full partitioned support in v1**, and **2D line masters first**
(3D surface masters refused loudly, deferred).

**Adversarially reviewed same day** (top-tier probe against the fork source
and this repo). Three findings refuted the draft and are folded in below:
INV-1's phantom parenthetical described the *rejected* mirror design and
would have produced the exact tension-only silent interface the ADR exists
to kill (corrected — the phantom stands on the **slave** side); INV-5's
"no tie to break" was false against `build_element_partition_owner`'s
documented first-seen tiebreak, because `_fem_extract` replicates
boundary-entity elements across partitions (scoped — domain elements only,
plus a single-owner assertion); and `ElasticPP` takes a yield **strain**,
not `Fy`, so D1's translation law and S5's gate were rewritten. Six further
gap findings (epp_gap signs, phantom-tag predicate registration, ghost ndf
parity, handler wording, S8 greenfield scope, single-rank equalDOF
justification) are folded into D4/D5/INV-1/INV-5/S8. The sign composition
itself was verified against `ZeroLength.cpp:1991-2009` + `ENTMaterial.cpp:112-118`:
strain = x̂·(u_j − u_i), ENT carries compression only — INV-1 stands.

## Context

The constraints roster has no verb that emits zeroLength spring pairs between
coincident node sets with geometry-derived orientation:

- `contact` / `contact_plane` (ADR 0073/0092) bring the fork's NTS/mortar
  search-and-pairing machinery and large-slide semantics, force the exclusive
  `LadrunoContact` handler (`opensees/apesees.py:5311`), and carry none of the
  per-pair uniaxial-law semantics this interface needs.
- `tie` / `tied_contact` / `embedded` are bonded: no unilaterality, no slip
  cap. `embedded` is a bilateral penalty translation tie with no strength.
- `ops.element.ZeroLength(nodes=(i, j))` (ADR 0049) is the bridge-level
  one-pair form. It has no pairing, no geometry-derived orientation, no
  tributary bookkeeping, shares one material across a fan, and is refused
  under partitioned emit (`apesees.py:2332`).

The Cerro Lindo workaround is post-broker hand-injection: read the paired node
sets from `FEMData`, emit one `zeroLength` per pair via apeSees primitives.
The painful part is orientation — the local axes must follow the face
geometry (the normal swings 90° from wall to crown along the arc), so the
script re-derives per-pair normals apeGmsh already knows from the geometry
kernel. Duplicated knowledge, and a per-model source of silent sign errors:
an ENT with a flipped normal is compression-only in the *wrong direction*,
and the model still converges.

What already exists, and is exactly what the verb needs:

- **Coincident pairing.** `ConstraintResolver._match_node_pairs`
  (`_kernel/resolvers/_constraint_resolver/_resolver.py:216`) — KD-tree
  nearest-within-tolerance, deterministic order, fails loud on many-to-one.
  Used today by `equal_dof` / `penalty`.
- **The additive side-list lane.** `ContactRecord` / `EmbedTieRecord` /
  `ReinforceTieRecord` are records that emit *elements/verbs*, not
  MP constraints; they bypass the `_DISPATCH` MP pipeline, live on their own
  `ElementComposite` lists, and get their own `emit_*` pass
  (`opensees/_internal/build.py:4583`, call sites `apesees.py:1477-1489`).
- **A typed `ZeroLength` primitive with `-orient`.**
  `opensees/element/zero_length.py:148` already plumbs the 6-tuple
  `(x1,x2,x3, yp1,yp2,yp3)` local-axis form and per-dof `mat_dirs`.
- **Tributary models.** The `0.5 × edge_length` accumulation
  (`_kernel/resolvers/_load_resolver.py:412`, in `resolve_line_tributary`)
  and the constraint-side
  `_tributary_areas` (`_resolver.py:752`, RBE3 `weighting="area"`).
- **Outward-normal derivation.** `LoadsComposite._curve_inplane_frame` /
  `_edge_normal_q` (`core/LoadsComposite.py:1287-1371`) solve the signed
  in-plane edge normal for 2D; `_face_outward_normals`
  (`core/LoadsComposite.py:910`) solves physical-outward via adjacent-element
  centroids for 3D.
- **A working per-spring results channel.** The MPCO `springs` topology reads
  per-zeroLength `basicForce` / `deformation` as
  `spring_force_<n>` / `spring_deformation_<n>`
  (`opensees/_response_catalog.py:2202`; composite
  `results.elements.springs`).

Two facts constrain the design more than anything else:

1. **The fork refuses mixed-ndf zeroLength.** `ZeroLength::setDomain`
   (fork `SRC/element/zeroLength/ZeroLength.cpp:611-673`) errors on
   `dofNd1 != dofNd2`. The requester's primary case — ndf=3 `dispBeamColumn`
   node against ndf=2 continuum node — cannot be connected directly.
2. **Co-located pairs break node-tally partition ownership.** ADR 0092's
   amended INV-1 refuses the all-shared tie case
   (`_kernel/resolvers/_contact_ownership.py:137-145`) because a replicated
   boundary node carries no locality information. A zeroLength pair has
   exactly two nodes at the *same coordinates*; on a partition cut hugging the
   interface both are replicated onto both ranks, so the degenerate case is
   not a corner case here — it is the likely case.

## Decision

Add `g.constraints.interface(master_label, slave_label, *, normal=,
tangential=, thickness=, tolerance=1e-6, name=)` as a new **additive
side-list lane** (the contact pattern): `InterfaceDef` list on the composite,
own resolver, `InterfaceRecord` list landing on a new
`fem.elements.interfaces` slot. One `zeroLength` per coincident pair, local
axes from the master face geometry, per-pair tributary-scaled materials.

- **D1 — declarative per-area laws, translated only at emit.** The verb takes
  frozen `_kernel` dataclasses — `NormalLaw(kind="ent"|"epp_gap"|"elastic",
  k_per_area, ...)` and `TangentialLaw(kind="epp"|"elastic", k_per_area,
  tau_b)` — with **flat scalar params only**. They are stored on the record
  (h5-serializable) and translated to typed material primitives in
  `build.py` at emit, scaled per pair. The translation is per-kind and must
  respect each material's actual argument convention (*corrected by the
  review* — the draft wrote `Fy = tau_b × A_trib` as if `ElasticPP` took a
  force; it takes a yield **strain**, `uniaxialMaterial ElasticPP $tag $E
  $epsyP`, fork `ElasticPPMaterial.cpp:52-76`):
  `ent → ENT(E = k_per_area × A_trib)`;
  `epp → ElasticPP(E = k_per_area × A_trib, epsyP = tau_b / k_per_area)` —
  A_trib cancels in the strain, and the physical yield force
  `τ_b × A_trib` is the emergent product `E × epsyP`;
  `epp_gap → ElasticPPGap(E = k_per_area × A_trib, Fy = −τ_b_n × A_trib,
  gap ≤ 0)` — this one *does* take a force, and its signs are governed by
  INV-1. Units close: `k_per_area [F/L³] × A_trib [L²] = [F/L]`, a spring
  stiffness against the zeroLength's relative-displacement "strain".
  This is the first `g.*` verb carrying material data,
  and it preserves the layering rule: `core`/`_kernel` import nothing from
  `apeGmsh.opensees` (today `grep UniaxialMaterial` over `core/`, `_kernel/`,
  `mesh/` returns zero hits — that stays true). A new law kind is a neutral
  schema bump, by design.
- **D2 — orientation is per-pair and geometry-derived; never a face-average
  frame.** 2D line master: per-node outward normal = normalized average of
  the node's adjacent boundary-edge normals, sign fixed outward via the
  adjacent 2D elements' centroids (the kernel-side port of
  `_curve_inplane_frame` / `_edge_normal_q` + the centroid sign-fix of
  `_face_outward_normals`; the resolver may not import from `core`). Corner
  nodes: average then renormalize; **fail loud** if the averaged normal
  opposes either contributing edge normal (a reentrant corner has no single
  outward). 3D surface masters raise `NotImplementedError` naming this ADR.
- **D3 — tributary bookkeeping is the library's.** Per-pair
  `A_trib = ℓ_trib × thickness`, where `ℓ_trib` is the `0.5 × edge_length`
  accumulation over the master polyline (ported to the constraint resolver as
  the sibling of `_tributary_areas`). `thickness` is an explicit kwarg in 2D
  — the verb refuses to guess an out-of-plane thickness.
- **D4 — mixed ndf via a phantom bridge, resolved at resolve time.** Forced
  by `ZeroLength::setDomain`. Per mixed pair the resolver mints a phantom
  node at the pair coordinates with **ndf = the lower side** (via
  `_next_phantom_tag`, `_resolver.py:165`), nests
  `equalDOF(retained=slave_beam_node, constrained=phantom, dofs=[1,2])`, and
  the zeroLength connects `master_continuum ↔ phantom` at matching ndf. The
  beam rotation DOF is never touched (the requested hinge semantics), and the
  phantom has no free DOF — both of its dofs are constrained by the equalDOF
  (a zeroLength endpoint alone would save nothing: ENT's tangent is exactly
  zero on the open side, `ENTMaterial.cpp:123-126`), so explicit integrators
  see no massless orphan. Equal-ndf pairs connect directly, no phantom.
  Precedent for phantom + nested-record compose rewrite:
  `NodeToSurfaceRecord` (`_kernel/records/_constraints.py:620`). Two
  mechanisms the review pinned down: (a) `_next_phantom_tag`
  (`_resolver.py:165`) only mints the integer tag — the phantom's ndf is
  set by the explicit `ndf=` on its `node` emit, and the existing
  `_emit_phantom_nodes` helper **hardcodes 6** (`build.py:4794`), so the
  interface emit needs a per-record phantom ndf, not that helper as-is;
  (b) interface phantoms must be unioned into the phantom-tag predicate
  (`_gather_phantom_nodes`-fed set, `build.py:4389-4392`, consumed by the
  H5 emitter's node classification and by `_plan_rank_constraints`
  `:7366-7376`) before any emission, or they are misclassified as foreign
  nodes. The G2 pair check (`build.py:716-731`) passes by construction —
  both zeroLength endpoints share an ndf, and the equalDOF references
  dofs 1–2 which exist on both sides. Note `infer_node_ndf`
  (`build.py:367`) deliberately skips the zeroLength family (skip at
  `:437`), so interface elements contribute no ndf — both real endpoints
  must get theirs from the structural elements they belong to, which the
  resolver asserts.
- **D5 — emission is a new `emit_interfaces()` pass, handler-independent.**
  Pattern `emit_contacts` (`build.py:4583`), call site alongside it
  (`apesees.py:~1486`). Per record: phantom `node` + nested equalDOF if
  mixed; two per-pair scaled materials (`tags.allocate("uniaxialMaterial")`);
  one `zeroLength` with `mat_dirs` on local dofs 1 (normal) and 2
  (tangential) and `orient=` from the record. Plain `zeroLength` + uniaxials
  is deliberate: it needs **no exclusive constraint handler** (unlike
  contact's `LadrunoContact`), so interfaces coexist with equation ties and
  every MP constraint — with the precision the review demanded: a mixed-ndf
  pair *does* add an ordinary identity equalDOF, so the primary-case deck is
  handler-dependent in the ordinary way (Plain merges identity equalDOFs,
  Transformation and Penalty handle them; none excludes the rest of the
  model). And it is the one zeroLength family the MPCO results catalog
  fully validates (`_response_catalog.py:1508-1512` — the
  `ZeroLengthContact*` variants are written as `UnknownMovableObject` and
  silently invisible to discovery).
- **D6 — per-pair recorder channels ride the existing springs topology.**
  Normal force / gap = `spring_force_0` / `spring_deformation_0`; tangential
  force / slip = `spring_force_1` / `spring_deformation_1`, via an MPCO
  recorder and `results.elements.springs.get(...)`. The native `.out`/
  DomainCapture springs path is a stub today and stays out of scope. The
  record's per-pair orient additionally gives
  `viewers/diagrams/_spring_force.py` the direction data it documents itself
  as lacking — wiring that is a named follow-up, not v1.

### Invariants

- **INV-1 — sign convention.** zeroLength `iNode = master` (**always the
  real continuum node**), `jNode = slave` (the real slave node, or the
  phantom standing on the **slave** side per D4 — *corrected by the review*:
  the draft's parenthetical described the rejected continuum-side-phantom
  mirror, and following it in the mixed-ndf primary case flips `u_j − u_i`
  into a tension-only interface that still converges); local-x =
  **outward normal of the master face**. ZeroLength deformation is
  `u_j − u_i` projected on local-x (verified: `setTran1d` +
  `computeCurrentStrain1d`, fork `ZeroLength.cpp:1991-2009` — the two
  negations cancel), so separation ⇒ positive elongation ⇒ ENT carries zero
  force (`ENTMaterial.cpp:112-118`), and closure ⇒ compression. The sign
  requirement is **per law kind**, not ENT-only: under this convention an
  `epp_gap` normal law must emit `Fy < 0` *and* `gap ≤ 0`
  (`EPPGapMaterial::setTrialStrain` branches on `sign(fy)`,
  `EPPGapMaterial.cpp:147-168`, and a mismatched pair only *warns*,
  `:109-110`) — the emit-time translation owns these signs, never the user.
  Both the resolver (node order in the record) and the emitter honor the
  convention jointly — a flip in either place makes the interface
  tension-only while still converging, which is the silent error class this
  verb exists to kill. One acceptance test loads each sign **on a curved
  master** (wall-to-crown normal swing), where a single-frame implementation
  cannot pass both, and runs for **both** `ent` and `epp_gap` normal laws.
- **INV-2 — per-pair frames survive compose.** Every record row carries its
  own orient 6-tuple; `g.compose` rotates orient vectors and
  rotates+translates phantom coordinates (the `_transform_contact_geometry`
  extension, `mesh/_compose.py:924`). A translation-only compose test would
  green a broken rotation — the compose gate includes a 90° rotation case.
- **INV-3 — tributary closure.** Σ A_trib over pairs = master face length ×
  thickness, to float-sum tolerance (O(n·ε) relative — a sum of n floats
  against an independently computed length cannot be bit-exact), asserted at
  resolve time and re-asserted in the acceptance battery. A node whose tributary share is zero is a
  resolve-time error (the `_tributary_areas` fail-loud pattern,
  `_resolver.py:798`).
- **INV-4 — layering.** Laws are declarative `_kernel` data; the only place
  they meet OpenSees material classes is the emit-time translation table in
  `build.py`. Missing primitives (`ElasticPP`, `ElasticPPGap`) are added to
  `opensees/material/uniaxial.py` as ordinary typed primitives.
- **INV-5 — partitioned: one owner rank per pair, chosen element-side.** An
  interface pair is an atomic unit — element + its two materials + phantom +
  nested equalDOF emit together inside **exactly one** rank's block. The
  owner is the rank owning the **master node's backing continuum element**.
  Node-tally ownership (ADR 0092 INV-1) is wrong here *by construction*, not
  merely fragile: the pair's nodes are co-located, so a cut hugging the
  interface replicates both onto both ranks — the exact all-shared tie
  `_majority_owner` refuses (`_contact_ownership.py:137-145`). ADR 0092
  already names element-side ownership "the real fix rather than a better
  heuristic" (0092 INV-1, first amendment); this record family can have it
  from birth because the resolver walks master faces and can stamp each
  record with its backing element id, which `ContactRecord` cannot.

  *Scoped by the review* — the draft claimed "no heuristic, no tie to
  break", and that is false as stated: `build_element_partition_owner`
  (`build.py:5946`) documents a **first-seen tiebreak for duplicated
  elements** (`:5964-5967`), and elements genuinely are duplicated —
  `_extract_partitions` (`mesh/_fem_extract.py:355-371`) adds a
  multi-partition entity's elements to *every* partition holding it, and
  the multi-partition entities at a cut are exactly the lower-dimensional
  interface entities: in 2D, the boundary **curve** elements a naive
  "walk the master faces" stamp would pick. Two requirements make the pick
  exact rather than heuristic:

  1. The stamped backing element is always a **highest-dimension (domain)
     continuum element** adjacent to the master node — never a boundary
     facet/curve element. (Top-dimension elements are not replicated;
     no ghost-cell path exists in `mesh/`.)
  2. Owner resolution **asserts** the stamped element appears in exactly
     one partition's element set, and fails loud otherwise — the pick is
     exact *under this asserted precondition*, and explicit when it fails,
     never silently first-seen.

  The atomic unit's nested equalDOF is emitted on the owner rank **only**,
  which deviates from the ADR 0027 MP replication rule for a reason worth
  stating: `NodePairRecord`s replicate on every touching rank because both
  endpoints are real nodes with real elements on their home ranks; here the
  constrained node is a **phantom that exists on exactly one rank** — the
  owner mints it — so there is nothing to replicate, and the retained beam
  node's DOFs are shared by tag through the ghost mechanism. The
  slave/beam node, if foreign, is ghosted per ADR 0027/0092 INV-7: `node`
  + replayed SP stream, **never mass, never elements, never loads** — and
  the ghost declaration must carry the **same explicit ndf** as the home
  rank's declaration (mixed-ndf models are exactly where per-node ndf is
  live; a ghost defaulting to the envelope ndf would silently disagree on
  shared-DOF counts). Element and material tags are pre-allocated in flat
  iteration order before the rank fan-out (the ADR 0027 tag-determinism
  rule, `allocate_element_tags` pattern, `build.py:6022`), so 1-rank and
  N-rank decks are byte-comparable. `uncuttable_elements=`
  (`mesh/_mesh_partitioning.py:289`) remains an optional optimization to
  reduce ghosting — never a correctness requirement.

  *Amended during S8 (adversarial review).* Two scope corrections from
  the shipped implementation:

  1. **"Byte-comparable" is conditional as a flat↔partitioned claim.**
     The interface pre-pass runs right after `allocate_element_tags`,
     but the MP passes that also mint elements (rigid-body /
     kinematic-coupling / ASDEmbeddedNodeElement) allocate **inside the
     per-rank loop**, while the flat path allocates them *before*
     `emit_interfaces` — so a model combining interfaces with an
     element-minting MP record has interface tags that drift between
     the flat and partitioned decks (measured: zeroLength 21-24
     partitioned vs 25-28 flat with a coexisting
     `g.constraints.embedded`). Cross-rank determinism, exactly-once
     emission and global tag uniqueness hold unconditionally; full flat
     identity for combined models would require pre-allocating the MP
     element tags too — a named follow-up, not S8. Pinned by the
     mixed-model regression in
     `tests/opensees/integration/test_interface_partitioned_emit.py`.
  2. **A pattern-borne `sp` on a ghosted slave is refused at plan
     time.** The ghost replay stream carries the model-level `fix` tier
     only; pattern `sp` lines fan out on a node's native ranks and are
     never mirrored onto a ghost, so a prescribed displacement on a
     ghosted interface slave would leave the owner rank's copy
     unconstrained — the ADR 0027 INV-2 measured singular-matrix case.
     Folding pattern `sp` into the ghost stream is a named follow-up;
     until then the emit refuses loudly, advising
     `uncuttable_elements=`, `ops.fix`, or serial emit.
- **INV-6 — staged: claimable by name, emitted on the equilibrated ground.**
  `g.constraints.interface(..., name="RockLinerInterface")` +
  `s.interface(name="RockLinerInterface")` inside a stage claims the records
  into the stage pool; the base pass skips claimed records; the stage block
  emits the full unit (phantom, equalDOF, materials, zeroLengths) after the
  stage's activated topology — the liner-install pattern. Side-lists have
  **no** stage-claim path today (`_claim_constraints_by_name`,
  `apesees.py:9763`, walks only `fem.{nodes,elements}.constraints`);
  the claim registry for interfaces is new, thin, and deliberately not
  generalized to the other side-lists in v1. Element-minting inside a stage
  block has precedent (`_emit_rigid_body_elements` via
  `emit_stage_mp_constraints`, `build.py:6830-6879`). Staged + partitioned
  compose: the owner rank's stage block carries the unit, ghost SP replay
  crosses the stage boundary per ADR 0027 INV-2 — this combo is the
  campaign's actual scenario and gets its own slice and gate.

## Implementation plan (slices, each PR-able)

Review tiers per the program's propose/implement/probe cadence: *mechanical*
slices are implemented by a lighter agent and probed by a stronger one;
the two places a plausible-wrong answer survives review — sign (S4/S5) and
ownership (S8/S9) — are probed (or implemented) at the top tier.

1. **S1 — this ADR** (docs-only PR; adversarial review before any code).
2. **S2 — uniaxial primitives `ElasticPP`, `ElasticPPGap`** + ns methods +
   byte-golden emit tests (`ElasticMaterial` / `ENT` are the templates,
   `uniaxial.py:603/640`; both target materials exist in the fork under
   exactly those command names, `SRC/runtime/commands/modeling/uniaxial.hpp:229/:431`
   — and note `ElasticPP`'s argument is a yield *strain*, per D1).
   Independent; needed before S5. *(implement: light / probe: mid)*
3. **S3 — record skeleton**: `NormalLaw`/`TangentialLaw` + `InterfaceRecord`
   (fields: master_node, slave_node, backing_element, orient 6-tuple,
   a_trib, laws, phantom fields, nested equalDOF records, name;
   `tag_rewrite_spec` covering every tag + nested stream), the
   `fem.elements.interfaces` slot, factory carry, and a **loud h5 guard** —
   `g.save()` with unpersisted interface records raises until S6 lands, so
   nothing silently drops (the lesson of `7c7883b4`/`746b513c`, where
   compose silently dropped contact and embed-tie records). Gates:
   `tests/test_record_schema_parity.py`, the guard test. *(light / mid)*
4. **S4 — def + verb + resolver**: `_match_node_pairs` pairing (fail loud on
   unmatched slave nodes and many-to-one), per-node outward normals with the
   D2 sign rule, tributary lengths, backing-element stamping, mixed-ndf
   phantom minting, INV-3 closure assertion. Kernel tests include the
   curved-master sign case and a reentrant-corner refusal. *(mid /
   **top** — sign correctness is the #1 risk)*
5. **S5 — `emit_interfaces()` serial**: golden Tcl + openseespy for
   equal-ndf and mixed-ndf models; per-pair material scaling asserted
   against what is actually in the deck (*corrected by the review* — the
   draft's "`Fy ∝ A_trib`" gate is unassertable for `epp`, whose deck line
   carries a strain): `E ∝ A_trib` with `epsyP` constant across pairs for
   `epp`, `|Fy| ∝ A_trib` for `epp_gap`, `E ∝ A_trib` for `ent`; ordering
   phantom → equalDOF → materials → zeroLength; temporary named partitioned
   refusal until S8 (*that refusal is lifted — S8 shipped 2026-08-12*).
   *(mid / top)*
6. **S6 — h5 round-trip + compose**: payload dtype
   (`mesh/_record_h5.py`, pattern `contact_payload_dtype:447`),
   encode/write/read/decode registered in `write_neutral_zone` and the read
   assembly, `NEUTRAL_SCHEMA_VERSION` 2.28.0 → 2.29.0 + changelog paragraph;
   `_RewrittenBundle` field, rewrite site, merge carry, verifier cover-set
   entry (`_compose.py:3162`), orient rotation + phantom transform per
   INV-2. **Two known sharp edges, found in design review:** (a) the phantom
   high-water mark must be recomputed post-merge — two merged models'
   phantom ranges overlap, and `phantom_node` must ride the node rewrite
   like any node tag; (b) the compose gate includes a rotation case. Also:
   `backing_element` is an *element* tag and must ride the element-offset
   rewrite, not the node rewrite. *(mid / top)*
7. **S7 — staged claim**: `s.interface(name=)`, thin claim registry for the
   interfaces side-list, stage-block emit reusing the `emit_interfaces`
   body, tag-continuity test across base/stage. *(mid / mid, independent
   probe)*
8. **S8 — partitioned emit** (INV-5): per-rank interface plan keyed on
   backing-element owner, ghost declarations (with explicit ndf parity) +
   SP replay, up-front tag allocation, 1-vs-N byte-identity via
   `tests/opensees/_helpers/partition_diff.py`, and a fixture with the cut
   hugging the interface (the all-shared node case) proving the owner is
   still exact. Scope honestly: no side-list has a partitioned emit path
   today — contacts are hard-refused at `apesees.py:2310-2319` and
   ADR 0092's S4 has not shipped — so S8 builds the **first** partitioned
   side-list emit in the codebase, not a port of an existing one. *(**top** implements / two independent mid-tier refuters —
   the ADR 0092 lesson: the duplicate-emission failure converges to a
   plausible wrong answer that equilibrium checks cannot catch)*

   *Landed 2026-08-12; survived both refuters (no correctness
   refutation; the suite mutation-tested — forced every-rank emission
   fails 4/8 integration gates).* Shape: `_plan_rank_interfaces` +
   `allocate_interface_tags` (`build.py`), `_emit_interfaces_partitioned`
   at step 7c-bis of the per-rank loop (`apesees.py`), the shared
   `_emit_interface_record` core consuming pre-allocated tag triples on
   every path (flat decks verified byte-identical to pre-S8). Refuter
   fixes folded in: the flat↔partitioned tag-identity conditional and
   the pattern-`sp`-on-ghost refusal (both recorded as the INV-5 S8
   amendment above), plus a shared flat/partitioned guard refusing an
   UNCLAIMED interface whose endpoint is a stage-bound node (the base
   pass would reference a node that only exists inside the stage block;
   the fix is the S7 claim, `s.interface(name=)`).
9. **S9 — partitioned + staged combo** (the campaign scenario): interface
   claimed into a stage under MPI; extends `_emit_stages_partitioned`
   (`apesees.py:2771`); ghost SP replay across the stage boundary. *(top /
   mid)*

   *Landed 2026-08-12 (probe pending).* Shape: the S9 refusal is
   retired; the base per-rank plan takes the UNCLAIMED subset and
   `_plan_stage_interfaces_partitioned` builds one INV-5 owner/ghost
   plan per stage (same single-owner + master-native assertions, same
   pattern-sp-on-ghost refusal, plus a named refusal for claiming
   into a stage EARLIER than the one owning an endpoint's topology —
   under MP that deck would run with the interface pushing on a
   floating ghost). Claimed rows join the S8 tag pre-pass in the
   sequence the serial-staged deck consumes the allocator (unclaimed
   flat order, then per-stage claim order), so serial-staged and
   partitioned-staged decks are tag-comparable under the same S8
   conditional. The unit emits inside the owner rank's stage bracket
   after the stage MP constraints (the flat `emit_stage_interfaces`
   position), reusing `_emit_interfaces_partitioned` with the
   stage-inclusive ghost SP stream and the rank's live ghost registry
   — ghosts replay stage-bound `s.fix` across the stage boundary and
   later stages mirror their SP deltas onto held ghosts (ADR 0027
   INV-2, both directions). Measured (the interface numeric twin,
   `tests/opensees/subprocess/test_interface_partitioned_numeric_twin.py`,
   serial vs 2-rank OpenSeesMP): worst relative delta 3.6e-15 across
   the S8 base push/pull/slip case and the staged liner install; the
   staged `zeroLength` is born STRAIN-FREE on the equilibrated ground
   (install-stage liner motion at solver zero; post-install
   increments carry the compression ordering in increment space) —
   the INV-6 install-on-equilibrated-ground semantics, now measured.
10. **S10 — acceptance battery** (the requester's own checks): bonded limit
    (stiff bilateral laws) reproduces `tie` on the same mesh; unilateral
    zero-tension with free separation, both signs, curved master, for both
    `ent` and `epp_gap` normal laws (INV-1's per-law sign rule); slip
    saturation Σ tangential = τ_b × L × t; INV-3 closure to machine
    precision; h5 round-trip → emit byte-identity; compose invariance;
    1-vs-N rank identity; MPCO springs channels readable per pair. *(mid /
    light runs+verifies)*
11. **S11 — docs + skill**: `docs/api/constraints.md` taxonomy row + tier
    section + mkdocstrings block; a placing paragraph in
    `docs/concepts/constraints.md`; CHANGELOG below the anchor (union-merge
    rules); `skills/apegmsh/` re-sync. *(prose at top tier per the docs
    voice rule; mechanics light)*

## Alternatives rejected

- **The `_DISPATCH` MP-constraint lane.** The verb emits elements, not MP
  constraints; routing it through `_split_constraints` would land records in
  the wrong buckets and drag the MP replication rules over something that
  must be single-rank (INV-5).
- **Opensees typed primitives as the law argument**
  (`normal=ops.uniaxialMaterial.ENT(...)`). Maximum flexibility, but it
  breaks the `core`↔`opensees` layering, is not h5-round-trippable inside a
  record payload, and per-pair tributary scaling of an arbitrary material is
  not well-defined — the library cannot know which parameters scale with
  area. The declarative law set is small and grows by schema bump.
- **Direct mixed-ndf zeroLength.** Refused by the engine
  (`ZeroLength.cpp:611-673`); not a choice.
- **Phantom on the continuum side at ndf=3** (the mirror of D4). Leaves the
  phantom's rotation DOF with zero stiffness — a singular DOF needing an SP
  fix, and a second sharp edge for explicit runs. The lower-ndf phantom has
  no leftover DOF at all.
- **`zeroLengthContact*` / `zeroLengthInterface2D` elements.** Would bundle
  the unilateral normal internally, but STKO's MPCO recorder writes them as
  `UnknownMovableObject` — invisible to `Results` discovery
  (`_response_catalog.py:1508-1512`) — and they bring their own
  contact-style parameter surface. Plain `zeroLength` + uniaxials is fully
  validated in the catalog and keeps the recorder story free.
- **Node-tally partition ownership** (reusing `resolve_contact_ownership`).
  Degenerate for co-located pairs by construction; see INV-5.
- **A single face-average orientation frame.** Wrong on curved masters — the
  motivating model's normal swings 90° from wall to crown. Contact's own
  documentation of why a global `outward` is refused on curved masters
  (`ConstraintsComposite.py:474-484`) is the same argument.
- **Material dedup across pairs with equal A_trib.** A float-equality trap
  on meshes that are almost-but-not-quite uniform; per-pair materials cost
  only deck lines. Revisit only if deck size measurably hurts (ADR 0061
  budgets).
- **EPP-with-huge-Fy as the bonded limit instead of `elastic` law kinds.**
  Conditioning trap; the bonded-limit acceptance check uses honest elastic
  laws.

## Consequences

- The Cerro Lindo hand-injection script retires; orientation knowledge
  lives once, in the geometry kernel, with the sign asserted by tests
  rather than per-model inspection.
- A new record family means the full lifecycle bill: h5 group + schema
  bump, compose rewrite + verifier cover-set, staged claim plumbing,
  partitioned plan. The slices pay it incrementally with a loud guard (S3)
  covering the persistence gap until S6.
- First `g.*` verb carrying material data. The declarative-law seam is
  deliberately narrow (kind + scalars); pressure to widen it (arbitrary
  hysteretic laws, rate dependence) is pressure to move the call to the
  `ops` layer instead — that boundary should hold.
- Interfaces are handler-independent, so unlike contact they compose with
  equation ties and MP constraints under every handler — one fewer
  exclusivity rule for staged SSI models.
- Per-pair materials add ~2N `uniaxialMaterial` lines and N `element` lines
  per interface; bounded by interface size, not model size (the ADR 0092
  ghost-budget arithmetic applies).
- The springs recorder story is MPCO-only in v1; a native-path `springs`
  category is a named follow-up if the campaign's recorder tooling needs
  it.

## Sign-off questions — settled 2026-08-12

All three settled as proposed at sign-off; S2 (primitives) and S3 (record
skeleton) started the same day.

1. **Law surface v1 — SETTLED: yes.** `NormalLaw` kinds
   `ent | epp_gap | elastic`, `TangentialLaw` kinds `epp | elastic`. Enough
   for the campaign (ENT normal + EPP tangential) and the acceptance battery
   (elastic bonded limit). Additional kinds are schema bumps, not redesigns.
2. **`thickness` semantics — SETTLED: explicit-only.** Explicit kwarg in 2D
   with no default; the verb refuses to guess an out-of-plane thickness.
   Reading a thickness from the master PG's element properties was
   considered and dropped; revisit only if explicit-only proves noisy in
   practice.
3. **Backing-element stamping — SETTLED: deterministic tie-break.** INV-5
   stores one backing **domain** continuum element id per pair at resolve
   time (never a boundary facet/curve element, per the review's scoping —
   those are exactly the entities Gmsh replicates across partitions at a
   cut). A master boundary node generally sits between two adjacent domain
   elements; the resolver picks the one whose centroid the pair's normal
   points away from, ties broken by lowest element tag. When the node's two
   adjacent domain elements land on *different* ranks, the deterministic
   pick decides the owner — accepted, since both choices are correct under
   ghosting (they differ only in which rank hosts the unit), and the pick
   is stamped at resolve time so 1-vs-N byte-identity still holds for a
   given resolved model. `uncuttable_elements=` stays available as an
   optimization for users who want the interface's backing elements
   co-located, never a requirement.
