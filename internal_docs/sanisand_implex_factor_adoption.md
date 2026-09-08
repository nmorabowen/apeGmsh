---
title: "apeGmsh adoption guide — ADR 92 / P2-9, the control-informed IMPL-EX factor (PR #822)"
project: Ladruno
type: guide
audience: apeGmsh team (bridge `apeSees`, `LadrunoSANISAND` emitter in `material/nd.py`, Ladruno recorder/reader)
status: current as of ladruno `179da6ffb` (merge of PR #822, 2026-09-07)
owner: nmora
related:
  - "[[LadrunoSANISAND_implex_guide]]"
  - "[[92_ladruno_sanisand_implex_adr]]"
  - "[[_adr92_p2_9_r3_results]]"
  - "[[_adr92_p2_9_esmeralda_results]]"
  - "[[86_ladruno_sanisand_apegmsh_emitter_guide]]"
  - "[[ladruno_apegmsh_contract]]"
  - "[[ladruno_apegmsh_adoption_guide_2026-09-07]]"
tags: [apegmsh, adoption, sanisand, implex, manzari-dafalias, recorder]
---

> [!note] Copied verbatim from the Ladruno fork, `Ladruno_implementation/ladruno_apegmsh_adoption_guide_2026-09-08.md` (fork PRs #828 and #833; feature merged in fork PR #822, ladruno `179da6ffb`). The fork copy is the source of truth; edit there and re-copy.

# apeGmsh adoption guide — ADR 92 / P2-9

One fork PR merged 2026-09-07 (`179da6ffb`, PR #822 from `wp/92f-implex-control-f`, closing out
ADR 92 P2-9): a new `-implexFactor` argument on `nDMaterial LadrunoSANISAND`. This guide lists
what changed in the fork, what apeGmsh must do to adopt it, and how to verify the adoption.
Minimum fork build: `ops.ladrunoBuild()` must be a descendant of `179da6ffb`. This is a
material-parser change only — nothing here touches the MP contract, the recorder wire format
(beyond the response tables in part (c)), or any element.

---

## 1. Control-informed extrapolation factor: `-implexFactor fixed|control|controlIter` (ADR 92 P2-9, PR #822)

**Unblocks:** nothing on today's default path — `fixed` is byte-identical to every build before
P2-9 (`LadrunoSANISAND_implex_guide.md` §12, opening paragraph). What it unblocks is the TIMs
campaign's ability to *measure* an alternative extrapolation-factor policy without a fork patch,
and gives the apeGmsh bridge a typed seam for that measurement once `controlIter` clears its
remaining gate (see (d) below).

### (a) The token, its three values, defaults, and the refusal rule

```tcl
nDMaterial LadrunoSANISAND $tag <23 constants> \
    -implex -implexControl $tol $reductionLimit \
    -implexFactor fixed|control|controlIter
```

- **Token:** `-implexFactor` (also accepted lower-cased `-implexfactor`; every other
  `-implex*` token in the parser has the same case-insensitive pair —
  `SRC/material/nD/LadrunoSANISAND.cpp:447`).
- **Values, case-sensitive:** `fixed` (the only gate-passed mode, and the **default** — a deck
  that never emits `-implexFactor` gets it), `control`, `controlIter` (its lower-case alias
  `controliter` is also accepted; `fixed`/`control` have no such alias — `LadrunoSANISAND.cpp:461-466`).
  Any other token is a hard parse-time refusal that echoes the operator and both formulas
  (`LadrunoSANISAND.cpp:467-480`).
- **`fixed`** = today's `f = alpha*dt_{n+1}/dt_n`, the clock ratio, with the existing P2-2
  reversal guard still able to knock it to `0` (`LadrunoSANISAND_implex_guide.md` §12 "The
  operator").
- **`control`** = `f* = clamp((A:B)/(B:B), 0, f_max)` frozen at the **first trial of the step**
  (`f_max` is always the clock ratio — `control` only ever *spends less* of it, never raises it).
  **Measured REFUTED** on the fork's R3 registered arm: see (d).
- **`controlIter`** = the same closed-form `f*`, recomputed at **every** Newton trial from that
  trial's own `d_eps` (`f_max` still stored once per step). **Measured PASSES** the same R3 bars,
  at a wall-time/Newton-churn cost: see (d).
- **Refusal rule:** `control` and `controlIter` are refused at construction — and on every
  `getCopy`/`recvSelf` clone, unconditionally, not gated on `verbose` — unless `-implexControl` is
  also given. The exact guard is `if (opt.factorMode != FACTOR_FIXED && !opt.control)`
  (`LadrunoSANISAND.cpp:1982`), and the message names the reason: *"-implexFactor control REQUIRES
  -implexControl. The control-informed factor f* = clamp((A:B)/(B:B), 0, f_max) is chosen by
  minimising the distance to the COMPANION stress sigma_impl ... which exists at the trial only
  under -implexControl"* (`LadrunoSANISAND.cpp:1983-1989`). Separately, any `-implex*` flag
  (including `-implexFactor`) given without the base `-implex` is refused, on the same list as
  `-implexGuard`/`-implexFlipAbsorb` (`LadrunoSANISAND_implex_guide.md` §12 "Refusals" table).
  From the Python bridge's point of view a refusal at construction is a failed `nDMaterial` call
  (per the 2026-09-07 adoption guide's own framing of fork parser refusals, §3).
- **Ordering:** `-implexFactor` may be given before or after `-implexControl` on the same command
  line — the parser cannot check the cross-flag requirement until after the whole flag list is
  parsed, precisely because `-implexControl` may legally follow it (comment at
  `LadrunoSANISAND.cpp:442-446`). The rule that **does** bind is the existing one: **every
  positional argument must precede every `-flag`** (`86_ladruno_sanisand_apegmsh_emitter_guide.md`,
  "Argument-ordering rule the parser enforces") — `-implexFactor` is just one more flag under that
  rule, nothing new.

### (b) The recommended apeGmsh field

**Prerequisite gap, found while reading `nd.py`:** the `LadrunoSANISAND` dataclass
(`src/apeGmsh/opensees/material/nd.py:735-1018`) currently declares **no** `-implex*` field at
all — not `implex`, not `implex_control`, not any P1/P2 flag. (Grepping `nd.py` for
`implex`/`implexGuards`/`implexDetail`/`implexRefusals`/`implexFactor` returns zero hits.) The
only place `implex`/`implex_control` fields exist in `nd.py` today is on the unrelated
`_LadrunoRCConcreteBase` dataclass (`nd.py:2365-2367`, emitted at `nd.py:2535-2540`) — a
different material family, useful only as the existing style precedent for the field names and
the `tuple[float, float] | None` shape of a `*_control` pair. Adopting P2-9 alone is therefore not
a one-field change: it needs the base `implex` / `implex_control` seam added to `LadrunoSANISAND`
first (or alongside), or `implex_factor` has nothing to validate against.

Recommended fields, added to `LadrunoSANISAND` after the existing `max_substeps: int = 0`
(`nd.py:898`):

```python
implex: bool = False
implex_control: tuple[float, float] | None = None   # (err_tol, reduction_limit); None = off
implex_factor: Literal["fixed", "control", "controlIter"] | None = None
```

`implex_factor=None` **omits** the token entirely — the deck falls through to the fork's own
`fixed` default, so an unset field is byte-identical to a deck built before this field existed
(mirrors how `max_substeps=0` already omits `-maxSubsteps`, `nd.py:1016-1017`).

Validation in `__post_init__`, mirroring the fork's own refusal wording so a Python `ValueError`
names the same reason the C++ parser would have given:

```python
if self.implex_factor is not None and not self.implex:
    raise ValueError(
        f"LadrunoSANISAND: implex_factor={self.implex_factor!r} requires implex=True "
        f"(the fork refuses any -implex* flag without -implex)."
    )
if self.implex_factor in ("control", "controlIter") and self.implex_control is None:
    raise ValueError(
        f"LadrunoSANISAND: implex_factor={self.implex_factor!r} requires implex_control "
        f"(err_tol, reduction_limit) -- the fork's own message: '-implexFactor control "
        f"REQUIRES -implexControl' (LadrunoSANISAND.cpp:1983)."
    )
```

Emission — appended to `_emit` (`nd.py:1005-1017`) **after** the existing `-maxSubsteps` block,
keeping every flag after every positional per the ordering rule in (a):

```python
if self.implex:
    args.append("-implex")
if self.implex_control is not None:
    args += ["-implexControl", *self.implex_control]
if self.implex_factor is not None:
    args += ["-implexFactor", self.implex_factor]
```

Order among `-implex` / `-implexControl` / `-implexFactor` themselves does not matter to the
parser (a), but emitting them in this sequence keeps the deck readable (cause before consequence)
and matches the order the command-line synopsis in
`LadrunoSANISAND_implex_guide.md` §1 lists them.

### (c) What to record and read back

- **`implexDetail[5]`** is now documented as "the `f` **actually used** for the last
  extrapolation" — under `fixed` that is the clock ratio, under either control mode it is `f*`,
  and it reads `0` on a P2-2/P2-6 guarded step regardless of mode
  (`LadrunoSANISAND_implex_guide.md` §6 table, §12 "Reporting"). No apeGmsh code currently reads
  `implexDetail` at all (same empty grep as above), so there is no existing reader whose meaning
  silently shifted — but any reader added later must document that `[5]` is mode-dependent.
- **`implexGuards` grew from 6 to 7 slots**, the new `[6]` = "count of steps where the operator
  'backed off', i.e. `f* < 0.5*f_max`, in EITHER control mode ... not counted when `f_max == 0`"
  (`LadrunoSANISAND_implex_guide.md` §6 table, §12 "Reporting"). Because no apeGmsh code reads
  `implexGuards` today either (same grep), there is **no hardcoded 6-slot reader in `nd.py` to
  migrate** — this is a green-field addition, not a fix. If/when apeGmsh adds an
  `implexGuards` reader, size it 7, not 6.
- **`implexGuards` carries no `ResponseType` component names.** WP-86d named the other six fork
  responses (`psi`, `yieldDistance`, `implexError`, `avgImplexError`, `substeps`, `implexDetail`,
  `implexRefusals` — table in the 2026-09-07 adoption guide's item 2) but left `implexGuards` out
  of scope; the guide's own §6 table marks its name column "(none yet — out of WP-86d's scope)".
  A recorder over `implexGuards` therefore falls back to the generic `C1..C7` column names, unlike
  every other fork material response. If apeGmsh's response canonicaliser (the
  `_KIND_TO_ROOT`/`_CONTINUUM_SCALAR_TOKENS`/`RESPONSE_CATALOG` machinery named in the
  2026-09-07 guide's item 2) is extended to `implexGuards`, it must supply the seven names itself
  (e.g. `implexGuards_floor`, `..._reversalGuard`, `..._holdPreserved`, `..._reversalReset`,
  `..._trialGuard`, `..._holdSkip`, `..._controlBackoff`) rather than expecting `COMP_NAMES` from
  the wire — the fork will not emit them.

### (d) Measured verdicts — CLOSED 2026-09-08

Per the fork's Fork R3 registered arm, the first decisive measurement
(`_adr92_p2_9_r3_results.md`; also summarised in `LadrunoSANISAND_implex_guide.md` §12 "R3
verdict"):

| mode | depth `s/B` (bar `>= 0.076`) | overlay mean \|dev\| (bar `<= 2%` pass / `> 5%` refute) | refusals/converged step | wall time | Newton max-iter stalls | verdict |
|---|---|---|---|---|---|---|
| `fixed` | — (baseline) | — (baseline) | 17.97 | — | — | shipped default |
| `control` | 0.0521 — fails | 11.12% — REFUTED | 2.31 | 141.6 s | 1 | **REFUTED** |
| `controlIter` | 0.1149 — passes | 1.70% — PASSES | 7.39 | 1858.4 s (~13x) | 89 | **PASSES R3 bars** |

The numpy oracle independently found the same mechanism (`GD.4`, cited in
`LadrunoSANISAND_implex_guide.md` §12): `control`'s first-trial freeze is the elastic predictor,
whose companion plastic increment is biased small, so `f*` collapses toward `0` on exactly the
steps where the real history matters — the oracle measured this made the path error **100-450x**
worse than `fixed`; recomputed on the converged `d_eps` (i.e. `controlIter`'s anchor) it was
**1.0-2.1x better**.

R3 was not the final gate, though — the pre-registered decision rule's actual gate is the
Esmeralda dense-refuse arm. TIMs reported it 2026-09-08 (`_adr92_p2_9_esmeralda_results.md`,
engine `179da6ffb`): dense honest wall on `controlIter` (control 0.1/0.01) = **0.01755**, against
a P2-7c fixed-f reference wall of 0.01689 (+3.9 % reach) and a ship bar of `>= 0.0177`. That
clears the refutation bar (`< 0.0169`) but misses ship by 0.85 % — neither branch of the rule's
`if` fires, so the **otherwise branch is decisive: P2-9 does not ship as a default.**
`-implexFactor fixed` remains the fork default (no code change, since `fixed` was already
default); `controlIter` is recorded as a graded guard with its measured gain (refusal churn
42 545 -> 102, failed attempts 248 -> 11, overlay comparable-to-better at +0.28 % mean) against
its measured cost (~2.9x wall time on this deck — **not** the R3 leg's ~13x, which does not
generalise; per committed step it is ~8x the Newton iterations of `fixed`, 2.2 -> 17.2, with
the vanilla implicit twin at 52.4, so `controlIter` sits nearer the implicit twin than to
`fixed` and keeps only ~3x of IMPL-EX's per-step edge where `fixed` keeps ~24x).
`control` stays REFUTED. P2-8's fixed threshold (`-implexGuardKp`, listed, not
built) remains the ADR's documented fallback.

**Guidance: do not default to any control mode — this is now the settled verdict, not an
interim one.** `control` is refuted and should be surfaced, if at all, only behind an explicit
opt-in with a loud warning quoting the R3 numbers above — never silently offered as an
alternative to `fixed`. `controlIter` clears both R3 bars and delivers a real, measured gain
(reach, overlay, refusal-churn collapse) on Esmeralda, but it explicitly **does not ship as
default**: it missed the pre-registered ship bar (0.01755 vs 0.0177) and costs ~2.9x wall time
on a production-scale deck. Treat `implex_factor="controlIter"` in apeGmsh as an opt-in for
deep, dense pushes toward a softening seat where refusal churn (not wall time) is the binding
cost — not a recommended setting for ordinary decks, and not a candidate for becoming the
default in a future PR without a new decision.

### (e) Golden-file note

`fixed` is byte-identical to every LadrunoSANISAND build before P2-9
(`LadrunoSANISAND_implex_guide.md` §12, opening paragraph: "`-implexFactor fixed` is the default
and is byte-identical to every build before P2-9"). Since `implex_factor=None` omits the token and
the fork default is `fixed`, **no apeGmsh golden file needs regenerating** for this feature —
unlike the `IntScheme 45` shift documented in `86_ladruno_sanisand_apegmsh_emitter_guide.md` §3.1,
which did move existing regression files.

Docs: `LadrunoSANISAND_implex_guide.md` §1 (flag table), §6 (responses), §12 (P2-9 in full,
including "When to reach for `controlIter`"); `92_ladruno_sanisand_implex_adr.md` (P2-9 row,
plan/decisions); `_adr92_p2_9_r3_results.md` (the R3 arm measurements);
`_adr92_p2_9_esmeralda_results.md` (the closing dense/loose-refuse arm and the CLOSED verdict);
`SRC/material/nD/LadrunoSANISAND.cpp` (parser, lines 442-481 and 1982-1989).

---

## 2. Checklist for the adoption PR(s) on the apeGmsh side

1. Bridge: require `ops.ladrunoBuild()` descendant of `179da6ffb` before emitting
   `-implexFactor` (the bridge's existing fail-loud build check, per the 2026-09-07 guide's
   item 1).
2. Prerequisite: add `implex: bool = False` and `implex_control: tuple[float, float] | None =
   None` to `LadrunoSANISAND` if not already landed by a parallel PR — `implex_factor` has
   nothing to validate against otherwise.
3. Emitter field: `implex_factor: Literal["fixed", "control", "controlIter"] | None = None` on
   `LadrunoSANISAND`, emitted after `-maxSubsteps` per (b) above.
4. Validation: both `__post_init__` checks in (b) — the `implex` requirement and the
   `implex_control` requirement for `control`/`controlIter` — with messages that cite the fork's
   own refusal text.
5. Unit test: assert the emitted argument list places `-implex`, `-implexControl` and
   `-implexFactor` after every positional and the 5-argument tail, in that order, for a deck that
   sets all three (mirrors `test_tantype_does_not_change_the_converged_answer`-style emission
   tests already in the fork's own suite).
6. Refusal test: `implex_factor="control"` with `implex_control=None` raises before any
   `nDMaterial` call is emitted; same for `implex_factor` set with `implex=False`.
7. Docs slice: a short note in apeGmsh's own material docs pointing at
   `LadrunoSANISAND_implex_guide.md` §12 for the semantics and warning that `control` is
   REFUTED (measured) and `controlIter` is TIMs-campaign-only, not a general recommendation.
8. If a `implexGuards`/`implexDetail` response reader is added at the same time, size
   `implexGuards` at 7 slots from the start (no 6-slot code path ever existed in `nd.py` to
   migrate — see (c)) and supply the seven names in (c) rather than relying on `ResponseType`.

Fork-side tests to mirror (none apeGmsh-specific exist yet for P2-9): the R3 registered-arm gate
described in `_adr92_p2_9_r3_results.md`; the construction-refusal cases described in
`LadrunoSANISAND_implex_guide.md` §12 "Refusals".
