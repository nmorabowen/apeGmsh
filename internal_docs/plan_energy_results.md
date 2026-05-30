# Energy Results (ON_DOMAIN / ON_REGIONS) — Implementation Plan

> [!note] Status
> **NOT STARTED.** This is the apeGmsh half of an energy-balance feature whose
> OpenSees side is designed and merged: MPCO_Ladruno ADR decision **D8**
> (`nmorabowen/OpenSees:Ladruno_implementation/03_mpco_ladruno.md`, merged PR #13).
> Depends on the OpenSees `mpcoLadruno` energy implementation landing first (so there
> is a file to read). Co-version against that recorder's `FORMAT_VERSION`.
> Companions: [[plan_results_viewer]], [[Results_architecture]].

## What

OpenSees' `EnergyBalanceRecorder` / `recorder mpcoLadruno -G energy <regionTags>` emits a
**global + per-region energy time-series** — channels `KE, IE, DW, ULW, RES, ERR%` — into
`RESULTS/ON_DOMAIN/energyBalance` (`[1×6]` per step) and
`RESULTS/ON_REGIONS/energyBalance` (`[nRegions×6]` per step).

apeGmsh has **no global/region result concept today** — its seven `ResultLevel`s are all
*spatial* (nodes/elements/gauss/fibers/layers/line-stations/springs) and every slab is keyed
to a node/element id. A value-per-step-with-no-id has nowhere to live. This plan adds the
domain/region level so `results.domain` / `results.regions` work and an energy-balance chart
renders.

## Why

apeGmsh is the canonical consumer — **we own the viewer** (no STKO constraint). Energy
balance is the trustworthiness instrument for explicit dynamics: a closed residual
(`RES ≈ 0`) is the proof a run stayed physical. Rendering it natively (not via a parsed
`.txt` sidecar) is the payoff of owning the viewer.

## Touch-points (grounded in a 2026-05-31 cross-repo recon)

Mirror the existing slab / protocol / reader / composite quartet so the composite layer
stays backend-agnostic:

- **`readers/_protocol.py`** — add `ResultLevel.DOMAIN` (+ `REGIONS`); `read_domain(stage_id, component, *, time_slice)` and `read_regions(stage_id, component, *, region_names, time_slice)` on the `ResultsReader` Protocol.
- **`_slabs.py`** — `DomainScalarSlab(component, values:(T,), time)` and `RegionScalarSlab(component, values:(T,R), region_names:(R,), time)`. Leading time axis matches every other slab.
- **`_composites.py`** — `DomainResultsComposite` / `RegionResultsComposite`, attached on `Results.__init__`/`_derive`. **Regions key by NAME, not ids — do NOT mix in `_SelectionMixin` / `_ElementGeometryMixin`** (regions are not nodes/elements).
- **`schema/_native.py`** — `/stages/<sid>/domain/<component>` `(T,)` and `/stages/<sid>/regions/{_names, <component>}` `(T,R)`, as **siblings of `partitions/`** (domain/region scalars are partition-independent — a reduced quantity). Add `domain_path()`/`regions_path()` builders.
- **`readers/_native.py`** — implement `read_domain`/`read_regions` (must **not** loop `self.partitions(...)` the way `read_nodes` does) + extend `available_components`.
- **`readers/_mpco.py`** — read `RESULTS/ON_DOMAIN` / `RESULTS/ON_REGIONS` mirroring the `_child(stage_grp, "RESULTS/ON_NODES")` walk; pull per-step `TIME`+value from `DATA/STEP_<k>`. **Note: `.ladruno` rides `from_mpco` gated on `INFO/GENERATOR` — there is NO `from_ladruno` constructor.**
- **`writers/_native.py`** + **`capture/_domain.py`** — `write_domain`/`write_regions`; a `_DomainScalarCapturer` modeled on `_NodesCapturer`, flushed in `end_stage`; a `DomainCaptureSpec.domain()/.regions()` category in `capture/spec.py` so energy is captured live in-process too.
- **`_vocabulary.py`** — an `ENERGY` canonical tuple (`kinetic_energy`, `internal_energy`, `damping_work`, `unbalanced_load_work`, `residual`, `error_percent`) + an `"energy"` shorthand; register in `ALL_CANONICAL` / `is_canonical`. Align tokens with the recorder's `KE/IE/DW/ULW/RES/ERR`.
- **`plot/_plot.py`** — `results.plot.energy_balance(stage=, regions=)`: matplotlib `stackplot` of KE/IE/DW/ULW with RES/ERR% on a twin y-axis (no analog exists — genuinely new). Generalize the node-bound `history()` into a node-free `scalar_history()`. Sugar: `results.energy(stage=)` → the 6-channel bundle + time.

## Scope / phasing

- **v1 (this plan):** reader + composites + native & mpco read paths + `_vocabulary` + matplotlib `energy_balance`. **Matplotlib-only.**
- **Deferred:** a non-spatial **"time-chart" diagram kind** inside the interactive 3-D viewer (`viewers/diagrams/_kind_catalog.py` is entirely spatial; needs a new renderer with none of the `Poly3DCollection`/grid infra). Largest piece; not needed for v1.

## Open questions

- Per-region identity: do energy regions map onto existing physical groups (`pg=`) / FEMData regions, or are they an independent name registry supplied by the recorder? Decides whether region names validate against `fem.elements.physical` or get their own registry in the file.
- Should `RES`/`ERR%` be stored (as the recorder emits) or recomputed at read time from KE/IE/DW/ULW? (Recorder stores all six; reader can re-derive RES as a tamper check.)
