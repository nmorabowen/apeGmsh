# ADR 0093 — `g.constraints.interface()`: oriented coincident-pair zeroLength interface (summary)

**Status:** Proposed (2026-08-12) — driven by the Cerro Lindo SSI program
(`Informe No3/Project/ADR/ADR-0005-ssi-squeezing-interaction-model.md` D1.3 / D4 / D8.0): […]

## Decision

Add `g.constraints.interface(master_label, slave_label, *, normal=, tangential=, thickness=,
tolerance=1e-6, name=)` as an additive side-list lane (the contact pattern): an `InterfaceDef`
list on the composite, its own resolver, and `InterfaceRecord`s on `fem.elements.interfaces`.
One `zeroLength` per coincident pair, local axes from the master face, per-pair tributary-scaled
materials. The physics it exists for is a unilateral, strength-capped interface.

- **D1: declarative per-area laws, translated only at emit.** Frozen `_kernel` dataclasses
  `NormalLaw(kind="ent"|"epp_gap"|"elastic", k_per_area, ...)` and
  `TangentialLaw(kind="epp"|"elastic", k_per_area, tau_b)`, flat scalars only, stored on the
  record and translated in `build.py`: `ent → ENT(E = k·A)`; `epp → ElasticPP(E = k·A,
  epsyP = tau_b / k)` (a yield strain); `epp_gap → ElasticPPGap(E = k·A, Fy = −τ_b_n·A,
  gap ≤ 0)`. A new law kind is a neutral schema bump.
- **D2: orientation is per-pair and geometry-derived, never a face-average frame.** 2D line
  master: per-node outward normal averaged from adjacent edges, sign fixed by element
  centroids; a reentrant corner fails loud. (3D surface masters were deferred here; see below.)
- **D3:** 2D `A_trib = ℓ_trib × thickness`; `thickness` is explicit and never guessed.
- **D4: mixed ndf via a phantom bridge** (2D). Per mixed pair a phantom node with the lower
  ndf, nested `equalDOF(retained=slave_beam_node, constrained=phantom, dofs=[1,2])`, and the
  zeroLength connects `master_continuum ↔ phantom`. Phantoms join the phantom-tag predicate.
- **D5: a new, handler-independent `emit_interfaces()` pass.** Two per-pair materials and one
  `zeroLength` with `mat_dirs` on local dofs 1 (normal) and 2 (tangential) and `orient=`; no
  exclusive constraint handler, so interfaces coexist with equation ties and MP constraints.
- **D6:** per-pair channels ride the springs topology (`spring_force_0/1`,
  `spring_deformation_0/1`, MPCO, `results.elements.springs.get(...)`).

Invariants:
- **INV-1 sign convention.** `iNode = master` (always the real continuum node), `jNode = slave`
  (or the slave-side phantom); local-x = outward master normal; separation is positive
  elongation, so ENT carries zero force. `epp_gap` must emit `Fy < 0` and `gap ≤ 0`; the
  emit-time translation owns the signs. A flip anywhere gives a silent tension-only interface.
- **INV-2** per-pair frames survive compose. **INV-3** tributary closure (Σ A_trib = master
  length × thickness, asserted). **INV-4** laws meet material classes only in `build.py`.
- **INV-5 partitioned:** one owner rank per pair, chosen from the master node's backing
  **domain** element (never a boundary facet/curve element), asserted to be in exactly one
  partition. The unit (element, materials, phantom, equalDOF) emits on the owner only; a foreign
  slave is ghosted with the same explicit ndf, no mass, elements or loads.
- **INV-6 staged:** `s.interface(name=...)` claims records; the stage block emits the unit.

## Amendments

- 2026-08-12: adversarial review folded in (slave-side phantom in INV-1, INV-5 scoped to domain
  elements, `ElasticPP` takes a strain). Sign-off questions settled the same day.
- 2026-08-12, during S8: flat↔partitioned byte identity is conditional when an element-minting
  MP record coexists; pattern `sp` on a ghosted slave is refused at plan time. S8/S9 landed.
- 2026-08-13, during S10: the bonded-limit reference is `equal_dof`, not `tie`.
- 2026-09-07/08, 3D surface masters (TIMs A10, `internal_docs/plan_interface_3d.md`, fork
  #808 / ADR 96): `surface_frames()` per-node frames and facet-area `A_trib`; `thickness`
  refused in 3D; no phantom in 3D (`_ACCEPTED_3D_NDF_PAIRS`); `orient` is six or nine floats
  (h5 `orient_t2`, neutral 2.32.0); emit `-dir 1 2 3` with uncoupled tangential sliders
  (square slip locus); the orient triad is asserted right-handed. S4 (3D) verified 2D↔3D twins.

Full text: [../0093-zerolength-interface-constraint.md](../0093-zerolength-interface-constraint.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
