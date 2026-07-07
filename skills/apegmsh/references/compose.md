# Model composition (compose v1, Phase 3)
<!-- skill-freshness: verified against apeGmsh main@8eeda7a3 (2026-07-06) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

Compose stitches independently-built, *saved* `model.h5` modules into one larger
FEM by tag-offsetting + namespacing each module's entities — no re-meshing, no
re-running geometry. It is the durable, cross-session way to build big assemblies
out of small reusable parts. ADR 0038 is the source-of-truth contract; ADRs 0036
(embedded-host decomposition) and 0041 (chain-phase routing) cover interface
bridging.

Mental model:
- Build a part once, `g.save("part.h5")` (neutral zone only — see `fem-broker.md`).
- Reload a host in **chain phase** with `apeGmsh.from_h5(...)` (NO gmsh — see below).
- `g.compose("part.h5", label="...")` grafts the part in, tag-offset + namespaced.
- Bridge module interfaces with chain-phase constraints (`tied_contact`, `embedded`, …).
- `g.save("assembly.h5")` the result, or `apeSees(g._fem).tcl(...)` to emit a deck.

---

## g.compose — the session facade

```python
def compose(self, source, *, label, translate=(0,0,0), rotate=None,
            anchor=None, partition_rank=None, properties=None,
            max_compose_depth=None, compose_size_per_module=None) -> ComposedModule
```
`src/apeGmsh/_core.py:306` → `Compose.compose` `src/apeGmsh/mesh/_compose.py:1525`.

```python
from apeGmsh import apeGmsh
# verified: tests/test_compose_end_to_end.py::test_from_h5_session_compose_workflow

# Reload a saved host module in chain phase (no gmsh build):
g = apeGmsh.from_h5("host.h5")
g.compose("bolt.h5", label="bolt", translate=(10.0, 0.0, 0.0))
g.compose("bolt.h5", label="bolt2", anchor="mount_pad")  # anchor sugar
g.save("assembly.h5")
```

| arg | meaning |
|---|---|
| `source` | path to a saved neutral-zone `model.h5` (a part, or another composed assembly) |
| `label` | **required**, namespace prefix for the module's PGs/labels. Strict — see rules below |
| `translate` | `(dx,dy,dz)` rigid shift applied to the module's node coords |
| `rotate` | `(angle, ax, ay, az)` quaternion-style 4-tuple, or `None` |
| `anchor` | PG-name *sugar* that resolves to a translate; **mutually exclusive with non-zero `translate`** |
| `partition_rank` | pin the module onto a given MPI rank (cross-partition emit, ADR 0027) |
| `properties` | forward-compat dict, stored on the `ComposeRecord` |
| `max_compose_depth` | per-call override of the nested-compose cap (default 3) |
| `compose_size_per_module` | advisory tag-reservation FLOOR per module (see capacity gotcha) |

Returns a `ComposedModule` handle (frozen dataclass; `src/apeGmsh/mesh/_compose.py:1394`)
with working `.label / .source_path / .translate / .rotate / .partition_rank`
properties. **`ComposedModule.pgs()/.labels()/.record_counts()` are STUBBED** —
they raise `NotImplementedError`; use `g.compose_inspect(path)` for inventory.

### label rules (fail loud — `ComposeLabelError`)
`label` must be non-empty, contain **no `.`** (the namespace separator), **no `/`**
(the depth-boundary separator), **no whitespace**, and **not start/end with `_`**
(reserved-prefix convention). `src/apeGmsh/mesh/_compose.py:55`.

```python
# verified: tests/test_compose_facade.py::test_compose_label_dotted_raises
g.compose("m.h5", label="a.b")   # ComposeLabelError
g.compose("m.h5", label="a/b")   # ComposeLabelError (test_compose_label_slash_raises)
```

### anchor vs translate
```python
# verified: tests/test_compose_facade.py::test_compose_anchor_with_nonzero_translate_raises
g.compose("m.h5", label="x", anchor="pad", translate=(1,0,0))  # ComposeAnchorError
```

---

## Namespacing — `'{label}.{pg}'`

The first model in the chain (the `from_h5` host) keeps its PG/label names **bare**.
Every composed module's physical groups and labels are prefixed with its label and
a dot: a part PG `top` composed under `label="bolt"` becomes `bolt.top`. When you
query the assembled FEM or wire interface constraints, you reference the **namespaced
name** for composed modules and the **bare name** for the host.

```python
g = apeGmsh.from_h5("host.h5")     # host PGs stay bare: "face", "base", ...
g.compose("bolt.h5", label="bolt") # bolt's PGs become "bolt.top", "bolt.shank", ...
g.constraints.tied_contact(master_label="face", slave_label="bolt.top")
```

The `pattern` field of load-pattern records is intentionally **NOT** namespaced
(verified: `tests/test_compose_end_to_end.py::test_compose_pattern_field_not_namespaced`).

---

## Inspect / list / tree

```python
# verified: tests/test_compose_facade.py::test_compose_inspect_returns_metadata_for_uncomposed_source
info = g.compose_inspect("bolt.h5")   # metadata-only read; does NOT merge/mutate
# keys: fem_hash, neutral_schema_version, tag_span_max, pg_inventory,
#       label_inventory, record_counts, composed_from, compose_tree, properties
info["neutral_schema_version"]   # e.g. "2.13.0" (current NEUTRAL_SCHEMA_VERSION)
info["pg_inventory"]             # sorted tuple of PG names (node+element sides, deduped)
info["composed_from"]            # () for an uncomposed source; tuple[ComposeRecord] otherwise

# verified: tests/test_compose_facade.py::test_compose_list_populated_from_h5_round_trip
g.compose_list()    # tuple[ComposedModule, ...] — modules composed into this session
g.compose_tree()    # tuple[ComposeTreeNode, ...] — nested-compose hierarchy
```
`compose_inspect` is `src/apeGmsh/_core.py:322`; `compose_list` `:330`; `compose_tree`
`:337`. `ComposeTreeNode(label, record, children)` is `src/apeGmsh/mesh/_compose.py:346`.

---

## FEMData.compose — the pure primitive

`g.compose` is sugar over the FEMData transform. The canonical primitive is a
pure functional transform on a broker (no live session):

```python
# verified: tests/test_compose_end_to_end.py::test_single_module_compose_round_trip
from apeGmsh import FEMData
host = FEMData.from_h5("host.h5")
merged = host.compose("bolt.h5", label="bolt", translate=(10, 0, 0))  # -> new FEMData
```
`src/apeGmsh/mesh/FEMData.py:1831`. Signature mirrors `g.compose` (incl.
`max_compose_depth`). **Drift hazard:** calling `FEMData.compose` directly decouples
the result from any live gmsh — if you then mutate the session mesh/PGs, the composed
records are dropped. `g.compose` handles replay via `session._compose_bundles`; the
bare primitive does not. Prefer `g.compose` in a session.

---

## apeGmsh.from_h5 — chain-phase reassembly (no gmsh)

```python
@classmethod
def from_h5(cls, path, *, model_name=None, verbose=False) -> apeGmsh
```
`src/apeGmsh/_core.py:142`. Rebuilds a session in **chain phase**: the FEM is loaded
straight from the neutral zone — **there is NO gmsh state**.

```python
# verified: tests/test_v1_1_a_2_tied_contact_chain_phase.py::TestTiedContactChainPhase::test_chain_phase_session_path
g = apeGmsh.from_h5("host.h5")
# WORKS in chain phase:
g.compose(...) / g.compose_inspect(...) / g.compose_list() / g.compose_tree()
g.save(...)
g.constraints.tied_contact(master_label="face", slave_label="bolt.top")  # interface bridging
# FAILS in chain phase (no gmsh):
g.model.geometry.add_box(...)      # raises
g.mesh.generation.generate(...)    # raises
```

`FEMData.from_h5` (the broker reload) is a *different* classmethod — it returns a
bare `FEMData`, not a session. See `fem-broker.md`.

### chain-phase routing of interface constraints (ADR 0041)
In chain phase, interface-bridging defs route through `try_chain_phase_route`
(`src/apeGmsh/_kernel/resolvers/_chain_phase_router.py:74`) onto the FEM and are
**NOT** gated by `ChainPhaseError`: `tied_contact`, `embedded`, `equal_dof`,
`rigid_link`, `rigid_diaphragm`, plus loads + masses. Use `master_label=` /
`slave_label=` (bare for host, `'{label}.{pg}'` for composed modules).

```python
# verified: tests/test_v1_1_a_2_embedded_chain_phase.py::TestEmbeddedChainPhase::test_chain_phase_session_path
g.constraints.embedded(host_label="soil", embedded_label="pile.shaft")
```

Two sharp edges:
- `try_chain_phase_route` **swallows KeyError/TypeError → returns False**: a port name
  not yet defined silently drops the def (back-compat with deferred-name resolution).
  A declared interface that resolves nothing is therefore a *silent* no-op here — verify
  it landed by checking `len(list(g._fem.elements.constraints))` grew.
- `tied_contact` needs dim=2 element groups. If the broker has none, the ADR 0041
  Decision-5 path raises `ValueError("re-extract with dim=None")` — a **hard** error
  that propagates (not swallowed). Re-extract the source with `dim=None` before saving.

---

## Nested compose + depth cap

Composing a source that is itself a composed assembly nests provenance. The cap is
`MAX_COMPOSE_DEPTH = 3` (`DEFAULT_MAX_COMPOSE_DEPTH`, `src/apeGmsh/mesh/_compose.py:144`),
overridable per call via `max_compose_depth=`. Exceeding it raises
`ComposeDepthExceededError`. The namespace separator alternates by depth
(depth 1 = `.`, depth 2 = `/`, …) so depth boundaries stay parseable; storage is a
flat graft (every ancestor surfaces as its own top-level `ComposeRecord`, and
`compose_tree()` re-derives the hierarchy).

---

## Error hierarchy + warnings

Facade errors (catch with `except ComposeError`), `src/apeGmsh/mesh/_compose.py:51`:

| error | when |
|---|---|
| `ComposeError` | base for all facade compose errors |
| `ComposeLabelError` (also `ValueError`) | bad `label=` (dot/slash/whitespace/`_`-edge/empty) |
| `ComposeAnchorError` (also `ValueError`) | `anchor=` + non-zero `translate=` |
| `ComposeCapacityError` (also `ValueError`) | `compose_size_per_module=N` < source tag span |
| `ComposeDepthExceededError` | nested depth > `max_compose_depth` |
| `ComposeNamespaceCollisionError` (also `ValueError`) | post-rewrite PG-name collision |

```python
# verified: tests/test_compose_facade.py::test_exception_hierarchy
from apeGmsh.mesh._compose import (
    ComposeError, ComposeLabelError, ComposeAnchorError,
    ComposeCapacityError, ComposeDepthExceededError, ComposeNamespaceCollisionError,
)
```

Warnings (filter independently):
- `ComposeInterfaceSizeWarning` — interface-class constraint count > `WARN_INTERFACE_SIZE`
  (50 000). Canonical class at `apeGmsh.core._compose_errors`
  (`src/apeGmsh/core/_compose_errors.py:69`). Silence:
  `warnings.simplefilter("ignore", ComposeInterfaceSizeWarning)`.
- `ComposeFilterWarning` — filtered record kinds (stages / time-series / load-patterns
  are dropped from a composed module). `src/apeGmsh/mesh/_compose.py:114`.
  Verified: `tests/test_compose_end_to_end.py::test_compose_filter_warning_for_stages`.

`compose_size_per_module` is a FLOOR/advisory (actual = `max(auto_size, value)`);
supplying a value SMALLER than the source span raises `ComposeCapacityError`.

> Note — `ComposeCapacityError` and `ComposeDepthExceededError` exist in **both**
> `apeGmsh.mesh._compose` (facade) and `apeGmsh.core._compose_errors` (verifier). The
> facade `ComposeDepthExceededError` subclasses the core one, so
> `except ComposeError` and `except` the core class both catch it.

---

## Viewing composed models — Module color mode

The viewer colors by source module via a **string-keyed** mode, NOT a `ColorMode`
enum member. `set_mode` accepts `'Module'`, `'Module: Root'`, `'Module: Leaf'`
(`src/apeGmsh/viewers/ui/mesh_tabs.py:122`). Any reference to `ColorMode.MODULE` as an
enum is wrong.

---

## Declarative `Assembly` + `couple` (shipped, sub-path import)

For spatially coupling several saved `model.h5` modules, a declarative builder
**shipped in v2.0.0** (PR #433, ADR 0043 slice 1.4). It is imported from a
**sub-path** — `from apeGmsh.assembly import Assembly` — *not* top-level:
`apeGmsh.Assembly` is deliberately guarded against (`test_library_contracts.py`),
so the top-level mental model "the session IS the assembly" still holds. `Assembly`
is a thin wrapper that *produces* a composed session.

```python
# verified: tests/test_assembly_compose_pipeline.py
from apeGmsh.assembly import Assembly, AssemblyError

g = (
    Assembly("frame")
    .add("col", "col.h5")                                  # first add = HOST (PGs stay bare)
    .add("beam", "beam.h5", translate=(0.0, 3.0, 0.0))     # composed under label "beam"
    .couple("col", "beam", kind="equal_dof",
            ports=("top", "end"), dofs=[1, 2, 3])          # bare per-part PG names
    .materialize()                                         # -> composed apeGmsh session
)
g.save("frame.h5")        # or apeSees(g._fem).tcl(...) / .py(...)
```

- **`Assembly(name)`** → `.add(label, source, *, translate=(0,0,0), rotate=None, anchor=None)`
  → `.couple(part_a, part_b, *, kind, ports, dofs=None, tolerance=None, **options)`
  → `.materialize() -> apeGmsh`.
- **First `add()` is the HOST** — its PGs stay un-namespaced; every later part is
  composed under its label, so its PGs become `"{label}.{pg}"`. `couple` always names
  **bare** per-part PG names (`ports=("top", "end")`); `materialize` resolves host→bare,
  composed→`"{label}.{pg}"`.
- **`kind` ∈ `{equal_dof, tied_contact}`** — `embedded` / `rigid_link` are deferred
  (they need host-volume geometry the bare-PG port model can't express).
- **Fail-loud `AssemblyError`** if no parts were added, a couple names an unknown part,
  or a couple resolves to **zero** new constraint records (a port that tied nothing).
- It's a thin wrapper over `apeGmsh.from_h5(host)` + `g.compose(rest, label=...)` +
  `g.constraints.<kind>(...)` (chain-phase-routed, ADR 0041). Reach the composed snapshot
  via `g._fem`. `Assembly.emit` / `Assembly.graph` are the next slice (not yet shipped).
