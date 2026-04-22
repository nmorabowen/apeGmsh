# Implementation Plan: `PartsRegistry.fuse_group()`

## Motivation

When building complex geometry from simpler primitives (I-beam from three boxes,
L-wall from two rectangles, a foundation from multiple blocks), the user creates
several inline parts that represent **one physical body**. After fusing, internal
surfaces vanish and the result is a single clean volume to mesh.

`g.model.boolean.fuse()` already exists, but it operates on raw dimtags and knows nothing
about the PartsRegistry. The user has to manually remove old Instances and create
a new one — error-prone and easy to forget. `fuse_group()` wraps this into one
call that keeps instance bookkeeping consistent.


## API

```python
def fuse_group(
    self,
    labels: list[str],
    *,
    label: str | None = None,
    dim: int | None = None,
    properties: dict[str, Any] | None = None,
) -> Instance:
```

**Parameters:**

- `labels` — list of existing instance labels to fuse (minimum 2).
- `label` — name for the resulting instance. If `None`, uses the first label
  in the list (the "survivor").
- `dim` — target dimension. Auto-detects highest common dimension if `None`.
- `properties` — metadata for the new instance. If `None`, inherits from the
  first label in the list.

**Returns:** the new `Instance`.

**Raises:**
- `ValueError` if fewer than 2 labels, or any label doesn't exist.
- `RuntimeError` if no common dimension across the listed instances.


## Usage

```python
# Build an I-beam from three boxes inline
with g.parts.part("web"):
    g.model.geometry.add_box(0, 0, 0,  0.01, 0.3, 5.0)
with g.parts.part("flange_bot"):
    g.model.geometry.add_box(-0.1, -0.005, 0,  0.2, 0.005, 5.0)
with g.parts.part("flange_top"):
    g.model.geometry.add_box(-0.1, 0.295, 0,  0.2, 0.005, 5.0)

g.parts.fuse_group(["web", "flange_bot", "flange_top"], label="i_beam")

# Now "i_beam" is one instance, internal faces are gone
# "web", "flange_bot", "flange_top" no longer exist in the registry
```

Also works with imported Parts:

```python
g.parts.add(flange_part, label="fl_top", translate=(0, 0.295, 0))
g.parts.add(flange_part, label="fl_bot", translate=(0, -0.005, 0))
g.parts.add(web_part,    label="web")

g.parts.fuse_group(["fl_top", "fl_bot", "web"], label="i_beam")
```


## Implementation Steps

### Step 1 — Collect entities from listed instances

Gather all `(dim, tag)` pairs from the instances at the target dimension. First
label becomes `obj`, remaining become `tool` (mirrors `fragment_all` convention).

```python
instances = [self._instances[lbl] for lbl in labels]

if dim is None:
    for d in (3, 2, 1):
        if all(d in inst.entities for inst in instances):
            dim = d
            break
    else:
        raise RuntimeError("No common dimension across listed instances.")

obj_inst = instances[0]
tool_insts = instances[1:]

obj  = [(dim, t) for t in obj_inst.entities.get(dim, [])]
tool = [(dim, t) for t in tool_inst.entities.get(dim, [])
        for tool_inst in tool_insts]
```

### Step 2 — Call OCC fuse

```python
result, result_map = gmsh.model.occ.fuse(
    obj, tool, removeObject=True, removeTool=True,
)
gmsh.model.occ.synchronize()
```

`result` contains the surviving `(dim, tag)` pairs. `result_map` is available but
not needed here — fuse collapses everything into one body, so the mapping is
straightforward.

### Step 3 — Remove old Instances from registry

```python
for lbl in labels:
    del self._instances[lbl]
```

### Step 4 — Create new Instance

```python
new_label = label if label is not None else labels[0]
new_entities = {}
for d, t in result:
    new_entities.setdefault(d, []).append(t)

new_props = properties if properties is not None else dict(obj_inst.properties)

inst = Instance(
    label=new_label,
    part_name=new_label,
    entities=new_entities,
    properties=new_props,
    bbox=self._compute_bbox(result),
)
self._instances[new_label] = inst
return inst
```

### Step 5 — Update Model registry (if needed)

`gmsh.model.occ.fuse()` with `removeObject=True, removeTool=True` already
removes the input entities from the OCC kernel. The Model `_registry` gets
cleaned by the OCC sync. No additional cleanup needed unless we want to log
which tags were consumed — follow the same pattern as `_bool_op` in
`_model_boolean.py` (lines 39–47).


## Edge Cases

**Duplicate labels.** Reject with `ValueError` if `labels` has repeats.

**Label collision.** If `label` matches an existing instance that is *not* in the
fuse list, raise `ValueError` (same guard as other entry points).

**Single-entity result.** Fusing overlapping boxes may produce one volume tag.
Fusing non-overlapping boxes may produce multiple volume tags (OCC fuse unions
the set but doesn't merge disjoint bodies). Both cases are valid — the Instance
just stores whatever tags survive.

**Mixed dimensions.** If one instance has 3D entities and another only has 2D,
the auto-detect picks the highest *common* dimension. If there is no common
dimension, raise. The user can force `dim` to override.

**Properties merge.** We take the first instance's properties by default. We do
*not* attempt to merge properties from all instances — that's ambiguous. The user
can pass `properties=` explicitly.


## Tests

1. **Basic fuse:** three inline boxes → `fuse_group` → verify one Instance with
   correct tags, old labels removed, single volume in Gmsh.
2. **Non-overlapping fuse:** two disjoint boxes → verify both volume tags survive
   in the new Instance (OCC fuse keeps disjoint bodies as separate volumes under
   the union).
3. **Properties inheritance:** verify first instance's properties carry through,
   verify explicit `properties=` overrides.
4. **Label collision:** verify error when `label` matches an existing unrelated
   instance.
5. **Fragment after fuse:** `fuse_group` → `fragment_all` against other parts →
   verify instance tags update correctly.
6. **2D fuse:** two surfaces → `fuse_group(dim=2)` → verify works at dim 2.


## File Changes

| File | Change |
|------|--------|
| `src/pyGmsh/core/_parts_registry.py` | Add `fuse_group()` method |
| `tests/test_parts_registry.py` | Add tests 1–6 above |
| `docs/guide_parts_vs_session.md` | Add section on geometric construction with fuse |
