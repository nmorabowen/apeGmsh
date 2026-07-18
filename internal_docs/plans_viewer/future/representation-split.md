# Future: Source / Representation Split

**Status:** future · **Estimated cost:** ~1–2 weeks · **Depends on:** plan 04 (active-objects)
**Discussed in:** [`../plotting-comparison.md`](../plotting-comparison.md)

## Goal

Split today's `Diagram` (selector + style + actors all in one class) into two
abstractions: a **Source-like** producer that emits a `vtkDataSet` given `(stage,
step)`, and a **Representation** that owns the actor / mapper / LUT.

## Why

- One Diagram = one rendering today. To show the same field with two colormaps requires
  two Diagrams that re-extract the same data.
- The Geometry / Composition / Diagram hierarchy is ad-hoc state piling up to compensate
  (deformation can't easily attach to multiple representations of the same source).
- ParaView's clean Source → Representation split is the textbook fix.

## Mechanism

- `DataProducer.update(stage, step) -> vtkDataSet` — owns selector + extraction.
- `Representation(producer)` — owns actor, mapper, LUT, visibility, opacity. One producer
  can feed N representations.
- Migrate existing diagrams one at a time. Old `Diagram` API stays as a thin wrapper
  during transition.

## Files

- New: `src/apeGmsh/viewers/diagrams/_producer.py`
- New: `src/apeGmsh/viewers/diagrams/_representation.py`
- Migrate (in order): `_contour.py`, `_vector_glyph.py`, `_line_force.py`, …

## Risks

- Big migration. Touches every diagram class. Land one at a time, not all at once.
- Session JSON schema bump (v5 → v6). Roundtrip both during transition.
- Producer/representation boundary may be awkward for tightly coupled diagrams (fiber
  section, layer stack). Decide whether to leave those as monoliths or split them too.

## Done criteria

- A demo composition: one Producer feeding two Representations of the same field, with
  different colormaps, side by side.
- All existing diagrams migrated. Old `Diagram` shim removed.
- Session JSON v6 documented; v5 loads via migration path.
- No regression in existing diagram tests.

## Out of scope

- Pipeline DAG (filters between producers). One source → N reps is the v1 cut.
- Multi-process / parallel.
