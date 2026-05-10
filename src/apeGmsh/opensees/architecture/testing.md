# Testing architecture

Multiple agents will work in parallel on different slices of the
`apeGmsh.opensees` package. The test layout below makes incoherence
*loud* — agents that diverge from the contracts produce failing
tests, not silent drift.

## Five test layers

```
tests/opensees/
├── unit/             one file per primitive type; uses RecordingEmitter only
├── contract/         every primitive of a family satisfies its base class
├── integration/      bridge + primitives composing across families
├── parity/           Tcl emitter ≡ Py emitter ≡ Live emitter for the same model
├── live/             real openseespy runs (gated; needs venv)
├── subprocess/       OpenSees binary invocations (gated; needs binary)
└── h5/               H5 schema validation + viewer contract
```

Each layer has a different speed / coverage tradeoff:

| Layer | Boots gmsh? | Boots ops? | Subprocess? | Typical runtime per test |
|---|---|---|---|---|
| unit | no | no | no | < 5 ms |
| contract | no | no | no | < 5 ms |
| integration | yes (mesh fixtures) | no | no | 50 – 500 ms |
| parity | no | optionally | no | 10 – 100 ms |
| live | yes | yes | no | 100 ms – 5 s |
| subprocess | yes | no (uses external) | yes | 1 – 30 s |
| h5 | no | no | no | 5 – 50 ms |

Default `pytest` runs everything but `live` and `subprocess`. Those
two are gated by markers (`@pytest.mark.live`, `@pytest.mark.subprocess`)
and run only when the env vars `OPENSEES_VENV` / `OPENSEES_BIN` are
configured (already in [memory: opensees venv](C:\Users\nmora\.claude\projects\C--Users-nmora-Github-apeGmsh\memory\reference_opensees_venv.md)).

## Layer 1 — Unit (RecordingEmitter)

Every primitive class gets at minimum **one** unit test file that
exercises:

1. **Construction** — `Steel02(fy=420e6, E=200e9, b=0.01)` succeeds.
2. **Validation** — illegal parameters raise (`Concrete02` with
   `fpc > 0` flips signs or raises).
3. **Defaults** — optional kwargs apply correct defaults.
4. **`_emit`** — given a `RecordingEmitter` and a tag, the emitted
   call matches the expected `(method, args, kwargs)` exactly.
5. **`dependencies()`** — returns the correct set (sections return
   their materials; elements return their section + transform).
6. **`__repr__`** — non-trivial, includes the type token.

Template:

```python
# tests/opensees/unit/primitives/test_materials_uniaxial.py
import pytest
from apeGmsh.opensees.material.uniaxial import Steel02, Concrete02
from apeGmsh.opensees.emitter.recording import RecordingEmitter


class TestSteel02:
    def test_construction(self):
        m = Steel02(fy=420e6, E=200e9, b=0.01)
        assert m.fy == 420e6
        assert m.tag is None              # standalone, no bridge

    def test_emit_records_correct_call(self):
        m = Steel02(fy=420e6, E=200e9, b=0.01)
        emitter = RecordingEmitter()
        m._emit(emitter, tag=42)
        assert emitter.calls == [
            ("uniaxialMaterial",
             ("Steel02", 42, 420e6, 200e9, 0.01, 20.0, 0.925, 0.15),
             {})
        ]

    def test_optional_isotropic_hardening_extends_emit(self):
        m = Steel02(fy=420e6, E=200e9, b=0.01,
                    a1=0.1, a2=0.5, a3=0.0, a4=1.0)
        emitter = RecordingEmitter()
        m._emit(emitter, tag=1)
        # Last four args are the isotropic-hardening params
        assert emitter.calls[0][1][-4:] == (0.1, 0.5, 0.0, 1.0)

    def test_dependencies_is_empty_for_leaf(self):
        m = Steel02(fy=420e6, E=200e9, b=0.01)
        assert m.dependencies() == ()
```

Unit tests **never** boot openseespy or gmsh. The `RecordingEmitter`
is the entire universe.

## Layer 2 — Contract

For every base class in `_internal/types.py`, one contract test file
verifies that **every concrete subclass** satisfies the base
contract. Pytest parametrization makes this nearly free.

```python
# tests/opensees/contract/test_material_base_contract.py
import pytest
from apeGmsh.opensees.material.uniaxial import (
    Steel01, Steel02, Concrete01, Concrete02, Hysteretic, ElasticMaterial,
)
from apeGmsh.opensees.material import UniaxialMaterial


ALL_UNIAXIAL = [Steel01, Steel02, Concrete01, Concrete02, Hysteretic,
                ElasticMaterial]


@pytest.mark.parametrize("cls", ALL_UNIAXIAL)
class TestUniaxialMaterialContract:
    def test_inherits_from_base(self, cls):
        assert issubclass(cls, UniaxialMaterial)

    def test_has_emit(self, cls):
        assert hasattr(cls, "_emit")

    def test_has_dependencies(self, cls):
        assert hasattr(cls, "dependencies")

    def test_is_frozen_dataclass(self, cls):
        # Param-set immutability across the typed primitive's lifetime
        from dataclasses import is_dataclass, fields
        assert is_dataclass(cls)
        assert cls.__dataclass_params__.frozen
        assert cls.__dataclass_params__.kw_only

    def test_repr_includes_type_token(self, cls):
        # Construct with minimum valid args via class-level fixture
        instance = _minimal_instance(cls)
        assert cls.__name__ in repr(instance)
```

Whenever an agent adds a new material to `material/uniaxial.py`, they
add it to `ALL_UNIAXIAL` — the contract suite picks it up
automatically and any drift fails immediately.

## Layer 3 — Integration

Bridge + primitives composing across families. Uses real FEM
fixtures (small meshes built once per session) but still
`RecordingEmitter` for emit verification.

```python
# tests/opensees/integration/test_full_model_build.py
def test_three_dof_frame_with_fiber_section(g):
    # Build geometry + mesh
    # ... add points, lines, PGs, generate mesh
    fem = g.mesh.queries.get_fem_data(dim=1)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec   = ops.section.Fiber(patches=[...], fibers=[...], GJ=1e9)
    trans = ops.geomTransf.PDelta(csys=Cartesian())
    ops.element.forceBeamColumn(pg="Cols", section=sec, transf=trans, n_ip=5)
    ops.fix(pg="Base", dofs=(1,)*6)

    bm = ops.build()

    # Verify the build produced the expected primitive ordering and refs
    assert bm.primitives[0] is steel        # materials first
    assert bm.primitives[1] is sec          # then sections
    assert sec in [...]                     # ...
```

Integration tests run with `RecordingEmitter` by default — they
verify *what gets emitted*, not *that openseespy can run it*.

## Layer 4 — Parity

For each model fixture, drive it through three emitters and verify
they are equivalent:

```python
@pytest.mark.parametrize("fixture", ["frame_3d", "arch_csys", "tank_cylindrical"])
def test_emitter_parity(fixture):
    bm = _build(fixture)

    rec = RecordingEmitter();      bm.emit(rec)
    tcl = TclEmitter();            bm.emit(tcl);  tcl_lines = tcl.lines()
    py  = PyEmitter();              bm.emit(py);   py_lines  = py.lines()

    # Same number of "logical commands"
    assert len(rec.calls) == _count_commands(tcl_lines)
    assert len(rec.calls) == _count_commands(py_lines)

    # Each command in tcl has a matching one in py and rec
    for call, tcl_line, py_line in zip(
        rec.calls, _grouped_tcl(tcl_lines), _grouped_py(py_lines)
    ):
        _assert_equivalent(call, tcl_line, py_line)
```

Parity tests catch divergence between the three concrete emitters.
If `TclEmitter` forgets to write `vecxz` for 3-D `geomTransf` but
`PyEmitter` does, parity catches it.

## Layer 5 — Live

Real openseespy run on a small model:

```python
@pytest.mark.live
def test_cantilever_static_displacement():
    # ... build a 1-element cantilever
    bm = ops.build()
    bm.run_live(wipe=True)

    import openseespy.opensees as ops_live
    disp = ops_live.nodeDisp(2, 1)
    assert disp == pytest.approx(expected_disp, rel=1e-3)
```

These are slow, gated by the `live` marker, and use the
`opensees_venv` python (per the memory at
`reference_opensees_venv.md`).

## Layer 6 — Subprocess

Same shape as live, but for the Tcl/py invocation paths:

```python
@pytest.mark.subprocess
def test_tcl_invocation_writes_recorder(tmp_path):
    # ... build + emit Tcl
    bm.to_tcl(tmp_path / "model.tcl", run=True)
    assert (tmp_path / "disp.out").exists()
```

Gated by `OPENSEES_BIN`. Skipped if not set.

## Layer 7 — H5 schema + viewer contract

```python
# tests/opensees/h5/test_h5_schema_v1.py
def test_minimal_fixture_validates_against_schema():
    h5_path = _build_minimal()
    _validate_schema_v1(h5_path)        # schema validator from bridge package

def test_viewer_required_groups_present():
    h5_path = _build_minimal()
    with h5py.File(h5_path) as f:
        assert "/meta" in f
        assert int(f["/meta"].attrs["schema_version"].split(".")[0]) == 1

def test_viewer_optional_groups_handled_when_missing():
    h5_path = _build_incomplete()       # only /meta + /elements
    # Viewer SHOULD treat this as "show mesh, hide enrichment panels"
    # We test the producer side: bridge does not write missing groups
    with h5py.File(h5_path) as f:
        assert "/sections" not in f
```

These are the test fixtures the viewer team will receive. They
verify both the producer (bridge) and the schema invariants the
viewer relies on.

## Conventions

### File and class naming

- Test files: `test_<module>.py` mirroring the source module path.
- Test classes: `class Test<Concept>:`.
- Test methods: `test_<behavior>` — readable as a sentence
  (`test_emit_records_correct_call`, not `test_001_emit`).

### Fixture sharing

All shared fixtures live in `tests/opensees/fixtures/`. Each fixture
is a Python module that exports a builder function:

```python
# tests/opensees/fixtures/frame_3d.py
def build(g):
    """Returns (fem, expected) for a 3-D portal frame."""
    # ...
    return fem, expected_dict
```

`expected_dict` is a JSON-serializable dictionary describing what the
test should observe (number of elements, list of materials, expected
vecxz values). Tests assert against `expected`.

**Rule:** fixture builders are immutable. Once committed, their
output is frozen. New variants are new builder functions.

### Markers

```python
# in pyproject.toml or conftest.py
markers = [
    "live: requires openseespy (use OPENSEES_VENV)",
    "subprocess: requires OPENSEES binary on PATH or OPENSEES_BIN",
    "h5: HDF5-related tests",
    "slow: > 1 second per test",
]
```

CI runs `pytest -m "not live and not subprocess"` by default. A
nightly job runs `pytest -m "live or subprocess"`.

### Determinism

Tests must be deterministic. No clock-based randomness. Tag
allocation is sequential (the bridge's `TagAllocator`); compare
against expected tag sequences in tests.

Where randomness is intrinsic (e.g. ground-motion sampling for a
visualization sanity check), seed it explicitly and document.

## What every PR adds

| Adding a new… | Test file required | Other tests touched |
|---|---|---|
| Material | `unit/primitives/test_materials_*.py` | Add class to `ALL_UNIAXIAL` / `ALL_ND` in contract test |
| Section | `unit/primitives/test_sections_*.py` | Add to `ALL_SECTIONS` |
| Element | `unit/primitives/test_elements_*.py` | Add to `ALL_ELEMENTS` |
| Time series | `unit/primitives/test_time_series.py` | Add to `ALL_TIME_SERIES` |
| Pattern | `unit/primitives/test_patterns.py` | — |
| Recorder | `unit/primitives/test_recorders.py` | — |
| Analysis primitive | `unit/primitives/test_analysis.py` | — |
| Emitter | `unit/test_emitter_<name>.py` + `parity/` | New parity case |
| Recipe | `unit/recipes/test_<name>.py` | — |
| Aggregate (Node, etc.) | `integration/test_<aggregate>.py` | — |

## CI gating

```
required (every PR):
    pytest -m "not live and not subprocess"
    mypy --strict apeGmsh/opensees
    ruff check apeGmsh/opensees

nightly:
    pytest -m "live or subprocess"
    h5 fixture validation against schema
```

Type checking is **non-optional** under P12. New code that fails
`mypy --strict` does not merge.

## What this catches

- **Drift in primitive shape** — contract layer.
- **Drift in emit output** — unit + parity layer.
- **Drift between Tcl, py, live** — parity layer.
- **Real solver compatibility** — live + subprocess layer.
- **H5 schema violations** — h5 layer.
- **Type-system regressions** — mypy in CI.

If an agent in parallel breaks one of the contracts, CI fails on the
contract test, not at integration time. That's the goal.
