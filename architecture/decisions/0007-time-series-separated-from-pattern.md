# ADR 0007 — Time series live outside `pattern/`

**Status:** Accepted

## Context

OpenSees source organizes time series under `SRC/domain/pattern/`
(`LinearSeries`, `PathSeries`, `ConstantSeries` are siblings of
`LoadPattern.cpp`). Conceptually they are distinct: a TimeSeries is a
function of time; a Pattern aggregates loads/SPs and references one
or more TimeSeries.

## Decision

`time_series/` is a top-level subfolder, separate from `pattern/`:

```
apeGmsh/opensees/
├── pattern/
│     pattern.py    Plain, UniformExcitation, MultiSupport, Earthquake
└── time_series/
      time_series.py    Linear, Constant, Path, Trig, Pulse,
                        ASCE41Protocol, FEMA461Protocol, ATC24Protocol
```

Patterns reference time-series instances:

```python
gm = ops.timeSeries.Path(file="elcentro.txt", dt=0.01, factor=9.81)

with ops.pattern.UniformExcitation(direction=1, series=gm) as p:
    pass
```

## Alternatives considered

1. **Mirror OpenSees: `pattern.time_series.Linear`.** Rejected —
   couples two conceptually distinct things by accident of OpenSees
   folder layout.
2. **Combine into `loading/` or `loads/`.** Rejected — `load` is
   already a verb in our API (`p.load(...)`), and the OpenSees
   community knows "TimeSeries" and "Pattern" as separate concepts.

## Consequences

**Positive:**

- Cyclic loading protocols (`ASCE41Protocol`, `FEMA461Protocol`,
  `ATC24Protocol` — the apeSees-pattern testers) live with their
  conceptual peers in `time_series/`.
- Patterns become a smaller, more focused module.
- Material testers can import time series without touching pattern.

**Negative:**

- Departs from OpenSees source layout. Acceptable — same
  rationale as ADR 0004; type tokens (Linear, Path, ConstantSeries)
  carry the OpenSees fluency, not the folder name.

## Reference

- [layout.md](../layout.md)
- `OpenSees/SRC/domain/pattern/LinearSeries.cpp` (the source location
  we are *not* mirroring)
