# ADR 0046 — Shell-on-solid node-sharing guard + the separate-node idiom

**Status:** Accepted (2026-05-30). Extends
[ADR 0032](0032-explicit-only-per-node-ndf.md) /
[ADR 0033](0033-s2-emit-wiring-per-node-ndf.md) (per-node `ndf`) and
[ADR 0022](0022-mp-constraint-emission-fanout.md) (MP-constraint
emission). Unblocks the docs example E7 ("shell-on-solid").

## Context

Standing a 2-D shell wall on a 3-D solid footing is a common modelling
need. The *intuitive* construction — build the wall surface, fragment
it onto the footing volume so the interface is conformal, declare the
wall `ndf=6` via `g.node_ndf`, and clamp the shell-edge rotations — is
**silently wrong**.

`g.model.boolean.fragment(...)` makes the wall-base line a boundary of
the footing's top face, so the meshes are conformal: the wall-base mesh
nodes are the SAME nodes the footing tets reference. `g.node_ndf` then
stamps those shared nodes `ndf=6` (shell-owned). At solve time each
footing tetrahedron declares `numDOF = 12` (4 nodes × 3), but one of
its nodes now carries 6 DOFs. OpenSees `FE_Element::setID`
(`SRC/analysis/fe_ele/FE_Element.cpp`) sizes the element's
equation-map array `myID` to `numDOF` and copies each connected node's
DOF ids into it; when the cumulative count (3 + 3 + 3 + 6 = 15)
overflows `numDOF = 12` it logs

```
WARNING FE_Element::setID() - numDOF and number of dof at the DOF_Groups
```

and returns `-3`, leaving the tet's stiffness mapped to the WRONG
global equations. The structure still solves and deflects plausibly —
the wall even matches `PH³/3EI` — but global equilibrium is violated:
in the reproducer only **988 N of an applied 2000 N** reached the
supports. There is no constraint handler or rotation clamp that
repairs this; the assembly itself is corrupt. **A node cannot be
shared between an `ndf=3` solid element and an `ndf=6` shell element.**

## Decision

### 1. Fail-loud build-time guard

`apeGmsh.opensees._internal.build.validate_node_ndf_element_compat(fem,
elements)` runs once at the top of `BuiltModel.emit` (before any
element is emitted), so it covers every **fresh-build** emit target —
flat / split / partitioned, and any of `ops.tcl` / `ops.py` / `ops.h5`
/ `ops.run`. It walks each element's connectivity, accumulating per node
the intersection of the `ndf_ok` sets
(`apeGmsh.opensees._element_capabilities.element_class_ndf_ok`) of every
element touching it. The moment a node's intersection goes empty — a
node genuinely shared by two elements with **disjoint** `ndf_ok` (shell
`{6}` ∩ solid `{3}` = ∅) — it raises `BridgeError` naming the node, the
two responsible element types, and the fix. A cheap pre-pass skips the
O(E) connectivity walk entirely when no two element families present are
disjoint (the common single-family case — all solids, all shells,
solids + beams).

The check is deliberately conservative:

* It keys off **element connectivity**, not the `g.node_ndf`
  declaration. A shell-solid shared node is unassemblable for *every*
  ndf the node could be given (ndf=6 overflows the solid's `numDOF`;
  ndf=3 underflows the shell's), so firing on a disjoint-`ndf_ok`
  shared node is never a false positive on a model OpenSees would have
  accepted.
* Element types it cannot classify resolve to `ndf_ok = None` and are
  skipped (silent false negative, never a spurious raise). Two known
  gaps, both acceptable: (a) **multi-ndf** elements — beams/trusses
  carry `{3,6}` / `{2,3,6}` and intersect both solids and shells, so a
  3-D beam fragment-merged into a solid is *not* flagged here (a rarer,
  distinct mistake); (b) the **H5 re-emit** path
  (`OpenSeesModel.from_h5(...).build(...)`) replays a stored record
  graph and does not re-run this guard — a model.h5 that already
  encodes the trap is re-emitted unguarded. Fresh builds (the path that
  *creates* the trap) are always validated.

### 2. The correct idiom — separate coincident nodes + `equalDOF`

The interface must have **two coincident nodes**: a solid node
(`ndf=3`) and a shell node (`ndf=6`) at the same location, tied on the
translational DOFs. This is the construction the partitioned S5 test
(`test_emit_partitioned_mixed_ndf_shell_on_solid.py`) already encodes —
`equalDOF(solid_node, shell_node, 1, 2, 3)`.

For a single line of interface nodes the shell-edge out-of-plane
rotation is a mechanism, so clamp the shell-base rotations
(`ops.fix(nodes=wall_base, dofs=(0, 0, 0, 1, 1, 1))`); the footing then
carries the shear via the translational tie while the moment is reacted
by the clamp. With this idiom the full load transmits:
`Σ reactions == Σ applied` to numerical precision
(`tests/opensees/integration/test_shell_on_solid.py::test_shell_on_solid_correct_idiom_satisfies_equilibrium`).

In apeGmsh, declare the tie with `g.constraints.equal_dof` (conformal /
coincident nodes) or `g.constraints.tie` (non-matching meshes —
shape-function interpolation, emitted as `ASDEmbeddedNodeElement`).
Because `equal_dof` / `tie` need coincident-but-distinct nodes, the two
bodies must **not** be fragment-merged at the interface (use the Part
registry without `fragment_all`, or two separate session bodies).

### 3. node_ndf round-trips through `ops.domain_capture`

`ops.domain_capture(spec, path=...)` now forwards the live bridge
(`DomainCapture(..., bridge=self)`) so the capture materialises a
sidecar `model.h5` and composes its `/opensees/` zone (and the bridge
`ndf` envelope) into the Composed file via
`NativeWriter.write_opensees_from`. Without this the capture file
carried only `/model` + `/stages`, the broker `/model/meta` had no
bridge `ndf`, and `OpenSeesModel.from_h5(path, fem_root="/model")` read
`ndf=0` — `validate_envelope_covers_broker_ndf` then rejected any
`g.node_ndf` model on read.

The sidecar is written via `self.h5(...)`, which raises
`NotImplementedError` for staged (`ops.stage(...)`) and initial-stress
builds (their H5 archival is deferred). So the bridge is forwarded
**only** for non-staged, non-initial-stress builds; staged capture
keeps the pre-Composed behaviour (no `/opensees/` zone, no ndf
round-trip) and continues to work, rather than regressing the
staged-SSI capture workflow. The ndf round-trip for staged models
lands when H5 staged-archival does.

## Consequences

* The silent half-load trap is now a loud `BridgeError` at build.
* The correct shell-on-solid idiom is documented + regression-locked
  (`test_shell_on_solid.py`).
* `g.node_ndf` models round-trip through `Results.from_native`.

## Related

- `apeGmsh.opensees._internal.build.validate_node_ndf_element_compat`
- `apeGmsh.opensees._element_capabilities.element_class_ndf_ok`
- [ADR 0033](0033-s2-emit-wiring-per-node-ndf.md) — per-node ndf emit.
- [ADR 0022](0022-mp-constraint-emission-fanout.md) — `equalDOF` /
  `ASDEmbeddedNodeElement` emission for the interface tie.
