# ADR 0008 — Three emit targets via Emitter Protocol

**Status:** Accepted

## Context

The user wants to:

1. Write a Tcl deck and either ship it or invoke OpenSees on it.
2. Write a Python (openseespy) deck and either ship it or invoke
   `python` on it.
3. Run live in-process via `import openseespy.opensees`.

Today's `apeGmsh.solvers.OpenSees.export.tcl()` and `.export.py()`
emit by walking internal dicts and concatenating strings. They are
~225 lines each, near-duplicate, and adding any new emit category
means editing both.

## Decision

Introduce an `Emitter` Protocol that mirrors the openseespy command
surface (~25 methods). Three concrete implementations:

| Class | Job |
|---|---|
| `LiveOpsEmitter` | Calls `ops.X(...)` directly (only emitter that imports `openseespy.opensees`). |
| `TclEmitter` | Accumulates Tcl strings, including `pattern { ... }` block scoping. |
| `PyEmitter` | Accumulates `ops.X(...)` strings. |

Plus a fourth for tests:

| `RecordingEmitter` | Captures every method call as `(name, args, kwargs)`. Test fixture. |

Primitives implement `_emit(emitter: Emitter, tag: int) -> None` and
never touch `ops` directly. The user picks the emitter at execution
time:

```python
ops.tcl("frame.tcl")           # write only
ops.tcl("frame.tcl", run=True) # write, then subprocess OpenSees binary
ops.py("frame.py", run=True)   # write, then subprocess python
ops.run()                      # live via LiveOpsEmitter
```

## Alternatives considered

1. **Keep `tcl()` / `py()` as parallel emit methods.** Rejected —
   permanent duplication, drift between the two, hard to add a
   third target.
2. **Single emitter that branches by target string.** Rejected —
   conditional logic in every command; impossible to type-check
   per-target behavior.
3. **Code generation from a YAML spec.** Rejected — over-engineered
   for ~25 commands; readability wins.

## Consequences

**Positive:**

- One source of truth for emission per primitive.
- Adding a fourth emit target (JSON for diagnostics, ANSYS,
  Code_Aster) is one new file (P8).
- `RecordingEmitter` lets us unit-test primitives without booting
  ops or writing files.
- Live execution is a peer of Tcl emission, not a reimplementation.

**Negative:**

- The Protocol has `*args` / `**kwargs` in its signatures because
  OpenSees commands take variable-length tail args. The boundary
  between typed user code and varargs is at `_emit` (P12 carve-out).
- `pattern_open` / `pattern_close` and `section_open` /
  `section_close` are invented Protocol vocabulary to bridge Tcl
  blocks vs Python's stateful current-X. Documented; necessary.

## Reference

- [emitter.md](../emitter.md)
- [charter.md P2, P8](../charter.md)
