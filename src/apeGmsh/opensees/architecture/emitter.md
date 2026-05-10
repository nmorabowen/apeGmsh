# Emitter abstraction

The emitter is the single seam between **what to emit** (typed
primitives) and **where it goes** (Tcl text, openseespy script,
live `ops` domain). Three concrete emitters share one Protocol; a
fourth captures calls for tests.

## The Protocol (frozen interface)

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

## The three concrete emitters

| Class | File | Job |
|---|---|---|
| `LiveOpsEmitter` | `emitter/live.py` | Calls `ops.X(...)` directly. Only emitter that imports `openseespy.opensees`. |
| `TclEmitter` | `emitter/tcl.py` | Accumulates Tcl strings. `pattern_open` writes `pattern Plain N tsTag {`; `pattern_close` writes `}`. `vecxz` rendered inline as space-separated. |
| `PyEmitter` | `emitter/py.py` | Accumulates `ops.X(...)` strings. `pattern_open` writes `ops.timeSeries(...)` (if needed) then `ops.pattern(...)`; `pattern_close` is a no-op. |
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

Primitives never know which dialect is active.

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

## Why split now if implementation is deferred

The Protocol shape is **load-bearing** for every primitive's `_emit`
method. Lock the Protocol now; implement concrete emitters later. Once
`Steel02._emit` calls `emitter.uniaxialMaterial("Steel02", tag, ...)`,
that signature can't change without rippling through every primitive.

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
