# Implementation plan — ADR 0098 ResultsSession (S0–S6)

Companion to `src/apeGmsh/opensees/architecture/decisions/0098-results-session-presentation.md`
(revised 2026-08-16) and `internal_docs/workshop_0098_results_session_why.md`.
Sized against the codebase 2026-08-17 (five-agent sweep, file:line-verified).
Cadence per slice: implement → adversarial probe → PR (one slice, one PR,
green both test lanes).

## Program shape — 15 PRs, 2 ADR amendments

```
S0 ──► S1-A ──► S1-B ──► S2 ──► [Amendment 1] ──► S3 ──► S4-1 ──► S4-2 ──► S4-3
 │                                                            │
 └────► S5a (schema freeze waits for S4) ──► S5b ──► S5c ◄────┘
                                                       │
[S6c dispositions — owner decision, can run any time] ──► S6a flip ──► S6b sweep ──► S6d docs
```

Hard sequence rules from the sizing:

- **Amendment 1 (pane host) before S3 starts** — splitters are the baseline
  (MDI rejected in workshop); the real decision is N `QtInteractor`s in
  splitters vs one render window with N VTK viewports. **Prototype two
  interactors in splitters on Windows AND under CI Mesa before writing the
  amendment** — multi-interactor driver quirks are the single biggest S3 risk.
- **S5a may start after S0 but must not publish the INV-17 contract (S5b)
  until S4's IR fields are frozen**, or the contract pays a second bump.
- **S6b cannot start before S6a merges** (old suite stays green until the
  flip); a long-lived sweep branch rots against the 155-file tests/viewers tree.
- **S6c (six-kind dispositions) first within S6** — it is an owner decision +
  ADR amendment, zero code risk, and it unblocks the sweep. It has no
  dependency and can be recorded any time.

## Model and effort per slice

Models: Fable 5 ($10/$50 MTok) for design-heavy one-way doors; Opus 5
($5/$25) for hard coding against a settled spec; Sonnet 5 ($3/$15, intro
$2/$10 through 2026-08-31) for pattern-following and mechanical breadth.
Effort: `xhigh` for hard coding/agentic (Claude Code's default), `high` for
design/prose, `medium` for mechanical sweeps. Probes follow the
adversarial-review-per-slice cadence; `/code-review high` where a lighter
pass suffices.

| PR | Slice | Size | Mode | Implement | Probe/review |
|---|---|---|---|---|---|
| 1 | S0 session IR + laws | M | design-heavy | **Fable @ high** | Fable @ xhigh |
| 2 | S1-A realize: mesh pane (contour+deform+substrate+legend law+stills) | L | integration | **Fable @ xhigh** | Fable @ xhigh |
| 3 | S1-B realize: remaining slots + plot series + scope axes | M | integration | **Opus @ xhigh** | /code-review high |
| 4 | S2 Qt pane projection + reconciler + Add-slot loop | L | design-heavy | **Fable @ xhigh** | Fable @ xhigh (+ /code-review ultra on the PR if desired) |
| — | Amendment 1: pane host (prototype first) | — | design | **Fable @ high** | owner workshop |
| 5 | S3 pane host + per-view scope + style buttons | L | integration | **Opus @ xhigh** | /code-review high |
| 6 | S4-1 selection surface + pane pick (S4.0+S4.1) | M | integration | **Opus @ xhigh** | /code-review high |
| 7 | S4-2 outline select-all + PlotView pane (S4.2+S4.3) | M | pattern-following | **Sonnet @ high** | Opus @ xhigh |
| 8 | S4-3 session time link (S4.4) | M | integration | **Opus @ xhigh** | /code-review high |
| 9 | S5a snapshot JSON + legacy gate | M | pattern-following | **Sonnet @ high** | /code-review high |
| 10 | S5b results_pin + INV-17 contract | S | mechanical | **Sonnet @ medium** | /code-review medium |
| 11 | S5c MCP render of a snapshot | M | integration | **Opus @ high** | Opus @ xhigh (mutation-test the smoke) |
| — | S6c disposition amendment (six kinds) | S | design | **Fable @ high** | owner sign-off |
| 12 | S6a the flip (viewer() → session().show()) | M | integration | **Opus @ xhigh** | Fable @ xhigh |
| 13 | S6b retirement sweep (~65 files) | L | mechanical | **Opus @ high** builds the disposition table; deletion batches delegated to Sonnet subagents @ medium | /code-review high |
| 14 | S6d docs + skill rewrite | M | pattern-following | **Opus @ high** (docs voice, 0079 D5 — not a Sonnet job) | /code-review medium + mkdocs --strict |

Rationale for the three Fable implementation slots (S0, S1-A, S2): each one
freezes a contract every later slice rides — the slot/selection/time IR, the
realize() layer-id + attach-ordering contract, and the reconciler diff
protocol. A wrong early decision there costs more than the 2× price delta.
Everything after S2 codes against settled contracts — Opus territory — and
the genuinely mechanical work (S5b, sweep batches) drops to Sonnet at intro
pricing.

## Decisions to make BEFORE the slice starts (from the sizing)

These are the mid-slice churn points the fleet found; settle each in the
slice's first hour (or in this plan) — not mid-PR.

1. **S0 — selection-store binding direction.** `results.session` importing
   `viewers.core.selection` inverts package layering (results is
   viewers-free today). Options: accept the import (SelectionState is
   Qt/VTK-free; gmsh only touched on group flush) or a store Protocol in the
   session package with late binding at show()/realize.
   *Recommendation: accept the direct import — SelectionState is already the
   one store (INV-5) and a Protocol adds a seam nothing else needs; record
   the layering exception in S0's PR.*
2. **S0 — node-target encoding** (one-way door into S5 snapshots):
   `SelectionTarget` with `Substrate.RESULTS_TOPO`, `key=node_id`, `dim=0`
   for results nodes; gauss = `key=element_id, sub=gp_index` (already
   anticipated by the `_targets.py:29` docstring). Record in S0.
3. **S0 — design slot records with the S1 mapping table open** (contour
   averaging → `ContourStyle.averaging`, line component → `LineForceStyle`,
   vector quantity spanning nodal vectors AND gauss principals), or S1
   reopens S0's records. Store the materials/element-type scope axis token
   opaquely in S0; resolution is S1/S3 work.
4. **S1 — scalar-bar suppression without touching the mixin**:
   `ScalarColorSupport._register_scalar_bar` fires inside attach; realize
   must pass a null legend-controller (or strip bars post-attach) so the
   ~80 old test files stay untouched. If no clean injection point exists,
   stop and re-plan — a mixin edit has blast radius across the old suite.
5. **S1 — attach/warp ordering**: one decided sequence for
   warp-before-extract vs sync-after (contour extracts copies of grid
   points), pinned by a parity test against `render_results(view='contour')`.
6. **S2 — the diff protocol** (the 0084 keystone): realize() returns the
   full per-pane layer list with STABLE layer ids; the reconciler diffs by
   layer_id → add/update/remove + scalar-bar sync → one coalesced render.
   Granular session events are NOT the v1 protocol (S0 ships
   subscribe/notify only as a change-tick). If S0 shipped a passive
   dataclass IR, S2 adds the observer surface first — that is design work,
   budget it.
7. **S2 — backend/interactor injection is a hard design requirement**: the
   pane widget takes a backend/interactor factory so headless tests inject
   RecordingBackend. Without it the S2 oracle collapses to the xvfb qt lane.
8. **S2 — QSettings scope**: the new SessionWindow must not share
   `QSettings('apeGmsh','ResultsViewer')` schema v7 with the live old
   window. Separate scope (`'ResultsSession'`) until S6; criterion-11
   retired-objectName guard constrains renames.
9. **S4 — multi-stage Instant track**: how the scrubber traverses stages
   with `Instant=(stage,step)` (director combined-mode's concatenated time
   at `_director.py:1142` is reference, not reuse). Decide before S4-3.
10. **S4-2 — path/xy plot kinds**: history is the oracle; agree up front
    whether path/xy ship real or as stubs, else the slice doubles.
    *Settled (S4-2, 2026-08-18): **stubs that keep refusing**.* Neither
    has an authoring path in v1 and `xy` needs an IR field that does not
    exist — `PlotSeries` is `(source, quantity)`, so a second axis
    source has nowhere to live, and `path` needs an ORDERED source
    (§8's selection is a set; the outline select-all is a set) plus an
    arc-length abscissa. §11's S4 verify line is "Pick → plot; linked
    cursor moves meshes" — history. The refusals now name the missing
    IR / authoring surface instead of a slice number, so they stay true
    whenever the kinds land. `PLOT_KINDS` keeps all three tokens: S5a
    serialises the token, and dropping one would be the schema break.
10b. **S4-2 — label / physical_group aggregation rule**: *settled: one
    curve **per member**, expanded by the resolver, with a loud cap.*
    `add_plot_from_selection` already expands a selection into one
    `PlotSeries` per node/GP (§6: "select → New plot COPIES the
    membership"; "Several series on one chart are one plot view"), so a
    membership source that aggregated instead would make the same
    chart mean two different things depending on how it was authored.
    An aggregate curve additionally needs a reducer token (mean? max?
    sum?) the IR has no field for, and defaulting to one silently is
    the "don't pick silently" trap. The cap lives in the resolver
    (`_plots.MAX_SERIES`) because that is the one place BOTH authoring
    paths funnel through — and outline select-all makes
    `add_plot_from_selection` a hang risk for the first time.
11. **S5a — snapshot filename** (settle with the S6 owner): through S2–S5
    the OLD viewer still saves v13 JSON to `<results>.viewer-session.json`
    on close. *Recommendation: the new session writes
    `<results>.session.json` until S6, and adopts the old path + the
    `.legacy` rename-aside at the flip.* Also: the pin-record key is
    `session_snapshot` (`session` is taken in ledger.schema.json).
12. **S6c — dispositions**: recommendation is **internal survival behind
    the show_web hatch** for all six kinds (mechanically near-free — the
    kind registry auto-populates and `viewer.director.registry.add` stays on
    the WebViewer handle; a slot-widening is ~M per kind and the ADR
    rejected an eighth slot). State the expiry condition: if the web-client
    ADR retires the hatch, the six need a new disposition.
13. **S6a — non-blocking return types**: only the blocking path returns the
    `ResultsSession`; `blocking=False` keeps Popen, notebook path keeps the
    WebViewer handle — document, don't unify. `export_animation` is
    explicitly out of the flip PR's scope (INV-11 hatch, still constructs
    the old viewer offscreen).

## Per-slice notes (what the sizing verified)

**S0 (M).** New package `src/apeGmsh/results/session/` (`_time`, `_slots`,
`_views`, `_selection`, `_session`). Reuse as-is: `scene_ir/_targets.py`
SelectionTarget, `viewers/core/selection.py` SelectionState +
`selection_log.py` OpKind.SET (the last-writer replace primitive). Clip
record copies the `ClipPlane` field shape — never imports the controller.
DiagramSpec/SlabSelector are NOT reused (old vocabulary; SlabSelector lacks
materials/element-type axes). Tests: plain pytest, no Qt/GL, oracles straight
from the ADR (fill-replaces, deform-on-slots-empty → zero legends,
contour(S)+gauss(S) → one scale, linked-instant, XOR last-writer through the
REAL SelectionState — one store proven, not mocked).

**S1 (L, two PRs).** realize() is one-shot — no reconciler here. ~60–70% of
per-picture computation is existing per-kind emit code invoked in place,
director-free (verified constructible: spec carries stage_id;
`_stage_pin_resolver`/`_visual_store` optional None); ~30–40% new
orchestration (`_realize`, `_slot_specs` mapping incl. the vector-slot
vector_glyph/principal_glyph resolver, `_legends`, `_stills` relocated from
`render.py:329-525` with GL-skip verbatim, `_plot_query`). Substrate +
style-button layers are new scene-IR emission (today's grey substrate is a
raw `add_mesh`); INV-MESH-2 generalizes `occludes_substrate`. Primary oracle
RecordingBackend layer assertions (no GL — layer tests run on CI);
stills secondary, skip ≠ pass. `Results.session()` lands here; `viewer()`
untouched.

**S2 (L).** Compose the existing `ResultsWindow` shell — verified ZERO
director coupling, generic mount points — don't build a new window. New:
the reconciler (imitate Dispatcher's batching shape, never touch it), the
MeshPane widget (injectable backend), session outline (panes group only),
mesh-view inspector page (7 slot rows; timebox to contour+deform first, the
other five ride the same row mechanism). The autouse `pump_failures` fixture
makes swallowed reconciler errors fail tests. The default CI lane RUNS
offscreen widget tests (`QT_QPA_PLATFORM=offscreen`; `-m "not qt"` only
excludes real-window tests) — the S2 oracle is in-CI, plus one qt-marked
real-window smoke per file under xvfb. Human loop is acceptance on top.

**S3 (L).** After Amendment 1. Per-pane backend is free
(`PyVistaBackend(plotter)`); scalar bars per-pane for free (per-plotter
registry → INV-LEGEND-5). Scope resolution: physical groups + labels have
named indexes in ViewerData; **materials and element-type axes may need a
data/ PR first — verify early**. Scope realization choice (cell-set rebuild
vs HIDDENCELL ghost mask) affects S1's contract and 18.6M-scale
responsiveness — prefer the ghost-mask idiom. `core/scope_controller.py` is
the 0058 spatial BOX — different concept, do not conflate.

**S4 (3 PRs).** S4.0 selection surface is thin law-layer over the one store
(S; verified: ResultsViewer today never writes SelectionState — this is new
surface, not a port). S4.1 pick: `PyVistaPickBackend` reusable verbatim
(rubber-band included — zero new VTK); rewrite `results_pick.py` slim
(nodes|gauss only); node-of-visible-cells mask is new derived data (cache
the incidence, 600k+ node models). S4.2 select-all: query layer fully
reusable (`results.nodes.get(pg=)`, `elements.gauss.get(pg=)`); watch
SelectionLog cost of one 100k-target SET gesture. S4.3 PlotView: resolver
must equal `results.plot.history` arrays (direct equality oracle);
TimeHistoryPanel is template, rewritten (director-coupled). S4.4 time link:
TimeScrubberDock genuinely adaptable (only the director binding changes);
reentrancy needs the `_suppress_observer` discipline across two widgets +
session; animation playback is the reconciler's first sustained-load test.

*S4-2 shipped (2026-08-18) — what the sizing missed, for the S4-3 author:*

- **"Watch SelectionLog cost" was a live defect, not a watch item.**
  `SelectionState.select_batch` AND `SelectionLog._dedup_extend` both
  deduped with a list scan: 1k targets 0.10 s, 4k 1.66 s, 16k 26.7 s,
  100k ~17 min. Both are set-backed now (100k → 0.16 s). If S4-3's
  playback ever writes the store per frame, the cost is linear.
- **Node membership needs NO results read.** `_scope.resolve_scope` already
  turns axis+names into element ids + node ids off `ViewerData`, and using
  the same resolver for both knobs is what keeps "check Web" and "select
  all of Web" meaning the same elements. Only the Gauss half reads (one
  single-step probe slab).
- **The §8 Gauss address encoding now lives in `_gauss_addr.py`**, shared by
  realize's pick targets, select-all's membership and the plot resolver's
  labels. Three consumers had to agree exactly; a divergence is silent.
- **Plot panes have their own reconciler** (`_chart.PlotReconciler`) —
  `SessionReconciler._resolve_pane` filters `MeshView`, so there was no
  repaint path to widen. Its signature is `((pane_id, kind, series),
  effective_instant)`, split so a cursor move takes a no-read path. **S4-3's
  scrubber rides that cheap half**: moving `session.time` already moves
  every plot's playhead and reads nothing, and the mesh panes already
  re-realize off the same `effective_instant`. What S4-3 adds is the widget
  and the reentrancy discipline, not the propagation.
- **The per-pane instant badge is still empty** (`SessionPaneFrame`'s
  `_time_badge`, built and hidden) — S4-3's, as reserved.
- **The window's bottom widget is still the scrubber placeholder.**

**S5 (3 PRs).** S5a: serialize/restore + legacy gate (v13 shape → notice +
`.legacy` rename, never overwrite an existing `.legacy`) + atomic write
(lift `studio/_paths.py:82` `atomic_write_text` into a shared module —
studio→results import direction forbids importing it from studio). Unknown
slot categories refused loudly — that refusal IS the amended-0094-INV-10
enforcement point. S5b: almost everything reusable (stamp, pin copy per
Amendment-5 lifecycle, contract machinery); CONTRACT_VERSION 1.6.0→1.7.0;
add a live-writer round-trip test so the hand-written schema can't drift.
S5c: CLI door — prefer a `results.session` entry over extending
`viewers/__main__.py` (scheduled to flip); MCP `render(session=)` skips the
_VIEWS token check; INVALID_ARGS when combined with view=; **MCP render
refuses old-schema files, never renames** (the rename belongs to the human
flow); primary oracle is restored-realize == built-realize at the IR layer,
PNG behind a GL skip marker.

**S6 (4 PRs, order S6c → S6a → S6b → S6d).** S6 is de-publication, not
deletion — render.py tokens, web_viewer hatch, `_verbs._yield_setup` keep
importing diagrams privately. S6a gates: mkdocs --strict couples the
`ResultsViewer` export removal with the `docs/api/viewers.md` mkdocstrings
rewrite in ONE PR; `test_skill_docs_drift` D-SIG couples the `viewer()`
signature change with the skill's signature quote; regenerate
`_api_index.json` (drift-gated) and `api-flows/flows.json`. S6b: the honesty
table is the real work (~65 files: ~27 per-kind files S1 re-covers, ~38
director/session/Qt-flow files retiring against S2–S5 equivalents;
section_cut's `test_section_cut_h5_autoload` retires ONLY after
persisted-cuts-boot-as-view-clips exists — port the assertions first if
S0–S5 left a gap). Add the import-guard gate (allowlist = the hatch
modules) and mutation-test it once. Orphan behaviors found mid-sweep
(probe overlay, opacity, threshold, eye-cascade) get case-by-case
keep/port/drop. render_showcase scripts: leave on internal imports +
allowlist (modal_sweep is blocked on mode animation anyway). S6d: rewrite
concentrated in pages that DESCRIBE machinery (~6 files); ~20 caller pages
survive because `viewer()` stays the one-liner; internal_docs get a banner
line only (redirects exist; no bulk archive prose per docs voice). Quote
new signatures sparingly — every D-SIG entry pins a signature.

## Standing risks (watch every slice)

- The ~80 old test files must stay green until S6a merges: nothing under
  `apeGmsh.results.session` imports `viewers.diagrams`; nothing modifies
  `results_viewer.py` / `_dispatch.py` before the flip.
- Both test lanes explicitly on every viewer PR (`pytest -m` override drops
  `not qt` — known footgun). CI is the only Linux oracle for viewer tests;
  Windows/GPU green proves nothing for Mesa.
- Mutation-test new smokes (a green smoke proves nothing until a broken
  build fails it) — mandatory for the S2 reconciler oracle, the S5c
  snapshot oracle, and the S6b import guard.
