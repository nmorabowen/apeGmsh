# ADR 0037 — Higher-order lines: broker-side split via mesh editing

**Status:** Accepted.  No OpenSees changes; no H5 schema bump.
Bridge surfaces (h5 emitter, opensees zone schema) are untouched.
This ADR sits next to ADR 0022 (the MP-constraint emit path that the
deferred `constrain` mode will eventually ride) and ADR 0036 (which
established the upstream-OpenSees limitation this ADR works around
on the broker side).

## Context

Gmsh's mesh-order setting is global: when a 2nd-order continuum part
of a model is meshed (quadratic shell, tet10, quad9, hex20, etc.),
**every** line entity in the same Gmsh model — including line
entities the user intended as frame elements — is meshed at order 2
too, producing 3-node Line3 elements (Gmsh type code 8) instead of
the 2-node Line2 (type code 1).

OpenSees beam-columns are strictly 2-node 1st-order.  The bridge's
`_check_two_nodes` guard at
[`element/beam_column.py:109`](../../../element/beam_column.py)
raises `ValueError` whenever a frame PG hands the typed primitive a
3-node connectivity tuple.  Today this is a hard stop for any model
that mixes 2nd-order continuum with frame elements (shell-on-frame,
embedded rebar in quadratic concrete, frame braces on a quadratic
plate, …).

The user-facing symptom is unambiguous; the fix is not obvious.
Three places the rewrite could live:

1. **Bridge emission time** (inside `emit_element_spec`) — the bridge
   detects Line3 connectivity, mints the mid-node as a real FE node,
   and emits two sub-elements per Line3.  Cost: one-to-many tag
   bookkeeping leaks into the H5 schema (`/opensees/element_meta`),
   the `tag_recorder` dict shape, the `geomTransf` per-element
   fan-out, and every downstream consumer that maps FEM eids to
   ops_tags.
2. **Broker / mesh editing** — a new verb on `g.mesh.editing` walks
   the named PG's Line3 elements and rewrites them in-place as
   Line2 pairs.  The FEMData snapshot sees Line2 only; the bridge
   never knows higher-order lines existed.  Cost: a single mesh
   mutation; downstream consumers see consistent topology.
3. **A new OpenSees C++ element** — natively 3-node beam-column.
   Cost: months of upstream work, classTags / FEM_ObjectBroker /
   sendSelf-recvSelf wiring for parallel runs, validation surface,
   then bridge integration.

The broker placement (#2) is the **only** option that keeps the
bridge ignorant of higher-order lines and adds no schema state.
ADR 0036's "Future work" track owns option #3.

A naive `policy="constrain"` route was tempting — keep the mid-node
in the OpenSees domain, tie it kinematically to (i, j) via a
2-master/1-slave linear-interp constraint, emit one Line2 between
(i, j).  This **looks** structurally clean (the FEMData snapshot
loses no nodes, no IP duplication, the user's calibrated nonlinear
column hinge locations are preserved) but is blocked by an OpenSees
primitive constraint: `ASDEmbeddedNodeElement` accepts exactly 3 or
4 retained nodes ([ADR 0036, line 13](0036-embedded-host-decomposition.md)),
so a Line2 master pair (2 nodes) cannot be expressed.  `equalDOF` is
unweighted; `rigidLink` is single-master; `rigidDiaphragm` is rigid,
not linear-interp.  No OpenSees primitive today expresses
`u_mid = 0.5*u_i + 0.5*u_j` for an arbitrary DOF set.  The
`constrain` policy is therefore **reserved but raises
`NotImplementedError`** this round, pending upstream OpenSees work
on the same future track as ADR 0036.

## Decision

Add a single new verb on `g.mesh.editing`:

```python
g.mesh.editing.split_higher_order_lines(
    physical_group: str | Iterable[str],
    *,
    policy: Literal["forbid", "split", "constrain"],
    dim: int = 1,
) -> _Editing
```

Three policies, with `policy` required (no default — destructive mesh
mutation should never happen by accident):

| Policy        | Behaviour |
|---------------|-----------|
| `"forbid"`    | Walk every PG's Line3 elements; raise `RuntimeError` if any are present, naming the PG and the count.  Use as a build-time invariant lock when a PG must remain 1st-order. |
| `"split"`     | For each Line3 `(i, j, mid)` on the named PG(s), remove the Line3 element via `gmsh.model.mesh.removeElements`, then add two Line2 elements `(i, mid)` and `(mid, j)` to the **same** dim=1 entity via `gmsh.model.mesh.addElements` with type=1.  The mid-side node — formerly a Gmsh side-node carrying no FE DOFs — now becomes an endpoint of two Line2 elements and acquires DOFs in the OpenSees domain.  No new gmsh nodes are minted: the node ID is preserved end to end. |
| `"constrain"` | RESERVED but NOT IMPLEMENTED.  Raises `NotImplementedError` with a message pointing at ADR 0036 future track.  The kwarg ships now so a future PR can land the policy without an API break. |

`dim` is fixed to 1 this round; `dim != 1` raises
`NotImplementedError`.  Line4 (Gmsh type 26, cubic edges from order-3
continuum meshes) is the next thing this verb generalises to; the
parameter is in place so that generalisation is purely additive.

### Sequencing

`split_higher_order_lines` must be called:
- **After** `g.mesh.generation.generate(...)` — it operates on the
  live mesh, not the geometry.
- **Before** `g.mesh.queries.get_fem_data(...)` — the FEMData
  snapshot must see the rewritten topology.
- **Before** `g.mesh.partitioning.partition(...)` — the split is
  global; running it post-partition would have to coordinate across
  ranks, which is unsupported.
- **Never** inside a staged-analysis block (`s.activate(...)` /
  `s.embedded(...)` from ADR 0034) — the mesh edit is global and
  must complete before any per-stage emission walks the topology.

### Three invariants

- **INV-1 — Bridge does not emit differently on higher-order lines.**
  Nothing in `opensees/_internal/` or `opensees/emitter/` changes
  the emitted OpenSees commands based on Gmsh element order.  The H5
  schema does not bump (no new `/opensees/element_meta` columns, no
  `tag_recorder` shape change, no transform fan-out coupling).  The
  bridge's existing `_check_two_nodes` guard remains the hard floor;
  it gains a sharpened 3-node branch (and a parallel 4-node branch
  for the deferred Line4 case) that points the user at this broker
  verb — that's a *friendlier loud-fail message*, not bridge-side
  awareness of higher-order topology.  The bridge still refuses to
  emit a beam with three nodes; it just tells the user where to go.
- **INV-2 — Topology consistency.**  After
  `split_higher_order_lines(policy="split")`, the FEMData snapshot
  carries Line2 only on the named PG(s).  Every downstream
  consumer (viewer, model_data, expand_pg_to_elements, etc.) sees
  one consistent topology — there is no parallel "macro-origin"
  state to maintain.
- **INV-3 — No new H5 schema state.**  The verb produces no
  persisted artefacts.  The decision to split is a one-shot mesh
  mutation; the only record of it is the rewritten element table
  in the gmsh live mesh and (via `get_fem_data`) the FEMData
  snapshot.

### Concentrated-plasticity trap (documented, not enforced)

`policy="split"` on a PG that will later host a
`forceBeamColumn` / `dispBeamColumn` with a concentrated-plasticity
integration rule (`HingeRadau`, `HingeRadauTwo`, `HingeMidpoint`,
`HingeEndpoint` — verified set per `opensees/integration.py:55-67`)
places the calibrated end-region hinges in the wrong places: both
sub-elements inherit the parent's hinge rule, so end-hinges appear at
i, mid (from sub-0's j-end), mid (from sub-1's i-end), and j — four
hinges instead of two, in the wrong positions.  Documented in the
verb's docstring; runtime detection at bridge emit time is deferred.
Users who need the calibrated hinge locations preserved should wait
for `policy="constrain"` (gated on upstream OpenSees work) or
re-mesh the frame PG at order 1 in a separate model.

## Alternatives rejected

**Bridge emission-time fan-out** (option #1 above).  Rejected per
the critique trail that drove ADR 0037: every downstream consumer
of `expand_pg_to_elements`, every FEM-eid-to-ops-tag map, every
`/opensees/element_meta` reader would need to learn the one-to-many
mapping.  Cost is paid forever; broker-side cost is paid once.

**Add a 3-node beam-column to OpenSees** (option #3 above).  Real
but underweighted only for shear-deformable (Timoshenko) cases where
the 3-node element is well-studied.  Force-based / fiber-section
3-node beams are non-standard in OpenSees and would require a year
of upstream work — sits on the same future track as ADR 0036's
HostProjector abstraction.

**Bundle the recorder fan-out fix with this ADR.**  An earlier draft
bundled the `RecorderDeclaration` element fan-out bug fix
(`fem_eid_to_ops_tag` translation in `_emit_element_level_record`)
into this work because the macro emission-time approach would have
needed the same fix.  After the broker decision, the recorder bug is
orthogonal — it fires whenever any Element primitive consumes an
allocator slot in `_register`, regardless of higher-order lines.
Shipped as its own preceding PR with a dedicated regression test.

## Consequences

### Positive

- One-line user-facing fix for the 2nd-order-continuum + frame
  workflow that has been hard-stop blocked by `_check_two_nodes`.
- No H5 schema bump, no bridge surface changes, no viewer changes.
- The `policy` and `dim` kwargs reserve room for the future
  `constrain` mode and `line4` generalisation without breaking the
  API.
- Reuses the existing `g.mesh.editing` mental model and the
  `crack()` pattern exactly — same dispatch shape, same PG
  resolution, same return-self chaining.

### Negative (acknowledged)

- **Concentrated-plasticity trap.**  See above.  Mitigated by
  documentation; runtime detection deferred.
- **IP doubling under `policy="split"` for distributed-plasticity
  beams.**  Each sub-element gets its own N-IP rule, so 2N IPs span
  the parent length.  Exact for prismatic elastic; honest but not
  numerically equivalent to a single 5-IP element under softening
  / cyclic degradation.  Documented in the docstring.
- **No constrain mode this round.**  Users who specifically need
  to preserve mid-node kinematics without IP duplication have to
  either accept the split caveat or wait for the upstream-OpenSees
  follow-up.

### Neutral

- Mesh editing operates on the live gmsh state, so calling
  `split_higher_order_lines` invalidates any cached FEMData
  snapshot built before the call.  Mirrors the established
  behaviour of `crack()` and `remove_duplicate_nodes()`.

## Future work (deferred, not blocking)

- **`policy="constrain"`** — gated on a new OpenSees primitive that
  expresses 1-slave/N-master linear interpolation for an arbitrary
  DOF set (`MP_Constraint`-based with custom constraint matrix, or
  a new `ASDEmbeddedNodeElement` variant accepting 2 retained
  nodes).  Tracked alongside ADR 0036's HostProjector RFC.
- **`dim=2` / line4 / cubic edges** — purely additive once the
  spike confirms `gmsh.model.mesh.addElements(elementTypes=[26], …)`
  on a mixed-order-per-entity entity works the same way Line3 →
  Line2 did.
- **Bridge-side concentrated-plasticity detection** — auto-tag the
  split-produced sub-elements (e.g. via a `g.mesh_selection` set
  named `__split_origin_lines_<PG>__`) and refuse to emit a
  HingeRadau-family integration rule on them.  Only worth doing
  if the documented trap bites a user in practice.

## References

- [`g.mesh.editing.split_higher_order_lines`](../../../mesh/_mesh_editing.py)
- [`_check_two_nodes` — sharpened error message](../../element/beam_column.py)
- ADR 0022 — MP-constraint emission fan-out
  ([`0022-mp-constraint-emission-fanout.md`](0022-mp-constraint-emission-fanout.md))
- ADR 0036 — Embedded-host decomposition (the upstream-OpenSees
  limitation that blocks the constrain mode)
  ([`0036-embedded-host-decomposition.md`](0036-embedded-host-decomposition.md))
- Upstream C++ — [`ASDEmbeddedNodeElement.cpp`](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp)
  (line 201 — the 3-or-4 retained-node constraint)
