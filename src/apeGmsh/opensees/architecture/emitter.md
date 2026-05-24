# Emitter abstraction

The emitter is the single seam between **what to emit** (typed
primitives) and **where it goes** (Tcl text, openseespy script,
live `ops` domain, HDF5 archive). Four concrete emitters share one
Protocol; a fifth captures calls for tests.

## The Protocol — Phase 0 baseline

```python
from typing import Protocol

class Emitter(Protocol):
    # Model
    def model(self, *, ndm: int, ndf: int) -> None: ...
    def node(self, tag: int, *coords: float) -> None: ...
    def fix(self, tag: int, *dofs: int) -> None: ...
    def mass(self, tag: int, *values: float) -> None: ...

    # Constitutive
    def uniaxialMaterial(self, mat_type: str, tag: int,
                         *params: float | str) -> None: ...
    def nDMaterial(self, mat_type: str, tag: int,
                   *params: float | str) -> None: ...
    def section(self, sec_type: str, tag: int,
                *params: float | str) -> None: ...
    def geomTransf(self, t_type: str, tag: int,
                   *vec: float) -> None: ...

    # Sections that take blocks (Fiber)
    def section_open(self, sec_type: str, tag: int,
                     *params: float | str) -> None: ...
    def section_close(self) -> None: ...
    def patch(self, kind: str, *args: int | float) -> None: ...
    def fiber(self, y: float, z: float, area: float, mat_tag: int) -> None: ...
    def layer(self, kind: str, *args: int | float) -> None: ...

    # Topology
    def element(self, ele_type: str, tag: int,
                *args: int | float | str) -> None: ...

    # Time series
    def timeSeries(self, ts_type: str, tag: int,
                   *args: int | float | str) -> None: ...

    # Patterns (Tcl wants a block; py wants a stateful current pattern)
    def pattern_open(self, p_type: str, tag: int,
                     *args: int | float | str) -> None: ...
    def pattern_close(self) -> None: ...
    def load(self, tag: int, *forces: float) -> None: ...
    def eleLoad(self, *args: int | float | str) -> None: ...
    def sp(self, tag: int, dof: int, value: float) -> None: ...

    # Recorders
    def recorder(self, kind: str, *args: int | float | str) -> None: ...

    # Analysis
    def constraints(self, c_type: str, *args: float) -> None: ...
    def numberer(self, n_type: str) -> None: ...
    def system(self, s_type: str, *args: int | float | str) -> None: ...
    def test(self, t_type: str, *args: int | float | str) -> None: ...
    def algorithm(self, a_type: str, *args: int | float | str) -> None: ...
    def integrator(self, i_type: str, *args: int | float | str) -> None: ...
    def analysis(self, a_type: str) -> None: ...
    def analyze(self, *, steps: int, dt: float | None = None) -> int: ...
```

The Protocol uses `*args` / `**kwargs` because OpenSees commands
genuinely take variable-length tail args. **This is allowed by P12
because the boundary is internal** — primitives are typed; emitters
speak OpenSees vocabulary; users never see this surface.

The block above is the locked **Phase 0** shape. The Protocol has
since been widened in a series of architecture events; see
[`emitter/base.py`](../emitter/base.py) for the current canonical
shape and the table below for the ADR-cited additions.

## Protocol widenings since Phase 0

Each row is an **architecture event** — the Protocol's header
documents that widening is intentional and cross-emitter. The
schema column tracks the per-zone `opensees_schema_version` bump
this widening drove (see [ADR 0023](decisions/0023-per-zone-schema-versioning.md)).

| ADR | Phase | Methods added | Schema bump | Notes |
|---|---|---|---|---|
| [0022](decisions/0022-mp-constraint-emission-fanout.md) | 7b | `equalDOF`, `rigidLink`, `rigidDiaphragm`, `embeddedNode`, `mp_constraint_comment`; `node(..., ndf=None)` kwarg | 2.6.0 → 2.7.0 | Closes §3.3 deferral so MP constraints emit into runnable decks. |
| [0024](decisions/0024-emitter-protocol-widen-region.md) | — | `region` | 2.7.0 → 2.8.0 | Drives MPCO recorder filter via `-R $regTag`. |
| [0025](decisions/0025-emitter-protocol-widen-eigen.md) | — | `eigen` (returns `list[float]`) | none | One-shot modal solve; live returns values, Tcl/Py/H5 are no-op-ish. |
| [0027](decisions/0027-cross-partition-mp-constraints.md) P4 | — | `partition_open`, `partition_close`, `parallel_runtime_fallback_numberer`, `parallel_runtime_fallback_system` | 2.9.0 → 2.10.0 | Per-rank emission scoping + runtime conditional for ParallelPlain / Mumps. |
| Phase 9 schema 2.3.0 | 9 | `recorder_declaration_begin`, `recorder_declaration_end` | (results-zone) | Brackets each recorder declaration's fan-out so archival emitters can persist declaration intent alongside the OpenSees command itself. |
| [0028](decisions/0028-initial-stress-via-parameter-ramping.md) | SSI-1 | `addToParameter`, `step_hook_ramp`; `analyze` behaviour change (auto hook-wrap) | none — H5 archival deferred | Materializes the STKO `stressControl` pattern: parameter declarations + per-step ramp proc + per-rank `addToParameter` fan-out. |
| [0029](decisions/0029-staged-analysis-context-manager.md) | SSI-2.A | `stage_open`, `stage_close` | none — H5 archival deferred | Brackets a per-stage analysis block. `stage_close` emits the canonical between-stages cleanup. |
| [0030](decisions/0030-stage-bound-topology-activation.md) | SSI-2.B | `domain_change` | none — runtime state, not model definition | Tells OpenSees to rebuild the renumbered DOF map after a stage's element activation. |

The current canonical Protocol shape lives in
[`emitter/base.py`](../emitter/base.py). The header docstring in
that file is the source of truth for every "architecture event"
above; the table here is the navigable index.

## The four concrete emitters

| Class | File | Job |
|---|---|---|
| `LiveOpsEmitter` | `emitter/live.py` | Calls `ops.X(...)` directly. Only emitter that imports `openseespy.opensees`. |
| `TclEmitter` | `emitter/tcl.py` | Accumulates Tcl strings. `pattern_open` writes `pattern Plain N tsTag {`; `pattern_close` writes `}`. `vecxz` rendered inline as space-separated. |
| `PyEmitter` | `emitter/py.py` | Accumulates `ops.X(...)` strings. `pattern_open` writes `ops.timeSeries(...)` (if needed) then `ops.pattern(...)`; `pattern_close` is a no-op. |
| `H5Emitter` | `emitter/h5.py` | Buffers structured records and writes the `/opensees/...` zone of an HDF5 archive (the bridge enrichment). See [h5-schema.md](h5-schema.md) for the on-disk format; reference reader at `emitter/h5_reader.py`. |
| `RecordingEmitter` | `emitter/recording.py` | Captures every method call as `(name, args, kwargs)`. Test fixture only — never written to disk. |

## Where divergences live

The Protocol invents three pairs of methods (`section_open` /
`section_close`, `pattern_open` / `pattern_close`, `*_open` /
`*_close`) precisely because **Tcl uses curly-brace blocks** and
**py uses stateful current-X**. The Protocol expresses both via the
open/close pair; each emitter handles its own dialect.

| Concern | Tcl | Python (openseespy) |
|---|---|---|
| Fiber section | `section Fiber 1 { patch ...; fiber ... }` | `ops.section('Fiber', 1)` then patch/fiber commands while "current section" = 1 |
| Pattern | `pattern Plain 1 Linear { load ...; eleLoad ... }` | `ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1)` then load commands |
| `vecxz` | `geomTransf Linear 1 0 0 1` | `ops.geomTransf('Linear', 1, 0, 0, 1)` |
| Partition scope (ADR 0027) | `if {[getPID] == K} { ... }` | `if getPID() == K:` indented block |
| Stage banner (SSI-2.A) | `# === Stage: insitu ===` comment line at indent 0 | same Python comment line |
| Stage close (SSI-2.A) | `loadConst -time 0.0; wipeAnalysis; ...` | `ops.loadConst(-time=0.0); ops.wipeAnalysis(); ...` |
| Stress ramp proc (SSI-1) | `proc rock_insitu {} { ... updateParameter ... }` + `lappend _apesees_before_step_hooks rock_insitu` | Python closure captured into the emitter's `_before_step_hooks` list |

Primitives never know which dialect is active.

## Phase SSI-1 — `analyze` hook-wrap behaviour

Each emitter carries a private `_step_hooks_registered` flag. The
first call to `step_hook_ramp(...)` flips it to `True`; from that
point on, **every `analyze(...)` call wraps its analyze invocation
with hook-dispatcher calls between steps**:

| Emitter | Bare `analyze(steps=N, dt=Δt)` | Hook-wrapped (after a `step_hook_ramp`) |
|---|---|---|
| `TclEmitter` | `analyze N Δt` | `for {set _apesees_i 0} {$_apesees_i < N} {incr _apesees_i} { _apesees_call_before_step; analyze 1 Δt; _apesees_call_after_step }` |
| `PyEmitter` | `ops.analyze(N, Δt)` | `for _apesees_i in range(N): _apesees_call_before_step(); ops.analyze(1, Δt); _apesees_call_after_step()` |
| `LiveOpsEmitter` | `ops.analyze(N, Δt)` (in-process) | In-process loop: per step, call each captured `before` closure, `ops.analyze(1, Δt)`, each `after` closure. Breaks on the first non-zero return — matching the contract of the bare `ops.analyze(N)` short-circuit. |
| `H5Emitter` | Records `(N, Δt)` into the analyze attribute. | Same — the analyze record is a runtime artifact, not a model-definition change; no hook archival. |
| `RecordingEmitter` | Captures `("analyze", (), {"steps": N, "dt": Δt})`. | Captures the same tuple — the test fixture is observing the high-level intent, not the loop expansion. |

`stage_close()` resets the flag back to `False` on the Tcl + Py +
Live emitters so the next stage's bare `analyze` re-emits as flat
unless that stage itself registers a new ramp **before** the
analyze. The dispatcher *list* is cleared (`set _apesees_before_step_hooks {}`),
but the previously emitted `proc <name>` definitions persist in the
Tcl / Py namespace; they become unreachable until a future
`lappend` puts them back into the list.

See [staged-analysis.md](staged-analysis.md) §"Hook dispatcher" for
the full per-step emit; see ADR
[0028](decisions/0028-initial-stress-via-parameter-ramping.md) for
the design choice. The Phase SSI-1 work explicitly defers H5
archival of the ramp — the H5 emitter no-ops on `addToParameter` /
`step_hook_ramp` / `analyze` for hook-wrapped traffic.

## Phase SSI-2 — staged emit

`stage_open(name)` / `stage_close()` bracket a per-stage analysis
block. Per [ADR 0029](decisions/0029-staged-analysis-context-manager.md):

- `stage_open(name)` emits a banner `# === Stage: <name> ===` at
  outer indent (Tcl, Py — both as comments; Live no-ops because
  comments aren't a live concept, but **raises
  `NotImplementedError` instead** to refuse staged live execution;
  H5 + Recording capture without side effects).
- `stage_close()` emits the cleanup pair `loadConst -time 0.0` +
  `wipeAnalysis`, plus (if hooks are registered) the dispatcher
  list resets. The proc bodies persist; only the `lappend` lists
  reset.

`domain_change()` (Phase SSI-2.B, [ADR
0030](decisions/0030-stage-bound-topology-activation.md)) emits the
OpenSees `domainChange` command after a stage's topology activation
block, before its analysis chain. Tells OpenSees to rebuild its
renumbered DOF map. Live forwards directly to `ops.domainChange()`;
H5 no-ops (domain renumbering is runtime state, not model
definition).

The H5 emitter's `stage_open` / `stage_close` / `addToParameter` /
`step_hook_ramp` / `domain_change` no-ops would normally produce a
silent-drop H5 round-trip (a staged model written and re-read as a
non-staged flat one). Per #313, the **bridge** guard
`apeSees.h5(path)` raises `NotImplementedError` when the build
carries any stage or any `initial_stress` record, pointing the
user at `ops.tcl(path)` / `ops.py(path)`. The H5-emitter no-ops
themselves stay reachable from direct `H5Emitter` unit tests; the
fail-loud landing is at `apesees.py::h5`.

## Execution modes

The user picks emit target × execution mode at the call site:

```python
ops.tcl("frame.tcl")               # write Tcl, do not run
ops.tcl("frame.tcl", run=True)     # write Tcl, then subprocess `OPENSEES frame.tcl`
ops.py("frame.py")                 # write py, do not run
ops.py("frame.py", run=True)       # write py, then subprocess `python frame.py`
ops.run()                          # use LiveOpsEmitter, in-process
ops.run(wipe=True)                 # default — wipe ops domain first
```

For Tcl invocation, the bridge resolves the OpenSees binary in this
order:

1. `bin=` argument to `ops.tcl(..., run=True, bin=...)` if given
2. `OPENSEES_BIN` environment variable
3. `OpenSees` on `$PATH`
4. Raise with a clear error referencing all three.

## Why we locked the Protocol first

The Protocol shape is **load-bearing** for every primitive's `_emit`
method. We locked the Protocol in Phase 0, then implemented the
concrete emitters one by one. Once `Steel02._emit` calls
`emitter.uniaxialMaterial("Steel02", tag, ...)`, that signature can't
change without rippling through every primitive — so the Protocol is
treated as frozen and any addition is an architecture event.

## Adding a new emit target

One file. Implement the Protocol. No primitive code changes. That's
the test of whether the abstraction is right (P8).

```python
# emitter/json.py — example future emitter
from .base import Emitter

class JsonEmitter:
    def __init__(self) -> None:
        self._records: list[dict] = []

    def uniaxialMaterial(self, mat_type, tag, *params):
        self._records.append({
            "kind": "uniaxialMaterial",
            "type": mat_type,
            "tag": tag,
            "params": list(params),
        })
    # ... 25 more methods, each ~3 lines

    def lines(self) -> list[dict]:
        return list(self._records)
```

If a new emit target needs an addition to the Protocol (e.g. a
hypothetical solver-specific command), that's a real architecture
event — bump the Protocol intentionally and update all emitters.
