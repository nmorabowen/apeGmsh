# apeGmsh ← Ladruno: adoption guide for the 2026-09-16 fork batch

Four fork work packages merged on 2026-09-16. This guide says, per feature:
**what changed in the fork**, the **exact tokens / verbs / refusal semantics**,
**what apeGmsh must change and where** (with `file:line` into this repo), a
**verification recipe**, and a **"nothing to do"** list.

It is the direct successor of
[`ladruno_adoption_2026-09-15.md`](ladruno_adoption_2026-09-15.md) (apeGmsh PR
[#1147](https://github.com/nmorabowen/apeGmsh/pull/1147)), whose adoption items
shipped as [#1148](https://github.com/nmorabowen/apeGmsh/pull/1148) and
[#1149](https://github.com/nmorabowen/apeGmsh/pull/1149). Same shape, same
place, same reason: every adoption point below is an apeGmsh file.

> ### STATUS — all four fork PRs are **MERGED**
>
> Unlike the 2026-09-15 guide, nothing here is read from an open PR. Every
> token, verb, refusal and line number below was read from `origin/ladruno` at
> the batch tip **`48c0e99bc`**.
>
> | WP | PR | merge commit | what it is |
> |---|---|---|---|
> | WP-105 / F12 | [#844](https://github.com/nmorabowen/OpenSees/pull/844) | `3f4847566` | **docs only** — the `IntScheme 2` verdict: PARTIAL |
> | WP-106 / ADR-93 II.1 | [#842](https://github.com/nmorabowen/OpenSees/pull/842) | `1133279a5` | `LadrunoSANISAND -pRe <kPa>`, an elastic-only stiffness floor |
> | WP-108 | [#845](https://github.com/nmorabowen/OpenSees/pull/845) | `049b295fc` | the "`-maxSubsteps` has NO EFFECT with IntScheme 2" warning was **false**; gone |
> | WP-107 / ADR-75b L3-1 | [#843](https://github.com/nmorabowen/OpenSees/pull/843) | `48c0e99bc` | threaded `Domain::update()` element loop + `ladrunoThreads` |
>
> Merge order on `ladruno` (oldest first): #844 → #842 → #845 → #843. The
> previous batch's tip, `634824e1f`, is the parent of all four.
>
> apeGmsh-side line numbers are against `origin/main` at **`1df2d0dc`** (the
> branch point for this document, i.e. right after #1149).

**Read §4 first if you read nothing else.** #845 fixes a false fork warning —
and apeGmsh does not merely *read* that warning, it **re-implements the same
false claim client-side**, in a `frozenset` at
[`material/nd.py:451`](../src/apeGmsh/opensees/material/nd.py), in two runtime
warnings (`nd.py:1048-1060`, `:1069-1080`), in **two** test files that pin the
bug as correct behaviour, and in two prose docs. That is the only item in this
batch that makes apeGmsh emit a **wrong diagnostic today**. Everything else is
new surface (§1, §2) or guidance (§3).

**SANISAND strip users, in one line.** Nothing in this batch changes a deck you
already emit: `-pRe` defaults to `0` (byte-identical), `ladrunoThreads` defaults
to `1` (byte-identical), #844 ships no code, and #845 only removes a warning
line. What changes is what you are *allowed to say*: `IntScheme 2` is now a
documented material-point option and **not** a BVP integrator, a `max_substeps`
cap on `int_scheme=2` is now honoured rather than inert, and `-pRe` is available
but is **measured worse** on the campaign's own surcharged strip.

---

## 1. WP-106 — `-pRe`, an elastic-only confinement floor on `LadrunoSANISAND` (#842)

### 1.1 What changed in the fork

ADR-93's subject is the free-surface ring beside a footing edge, where `p' → 0`
and SANISAND's pressure-dependent moduli go with it: `G, K ~ sqrt(p/P_atm)`, so
a point at the surface carries almost no stiffness and the substepper seizes
trying to integrate it. The existing `-Presidual` does **not** help: it reaches
~30 plastic-side mean-stress sites (`GetF`, `GetPSI`, `M^b`, `M^d`, `D`, the
`D_factor` sigmoid, the low-`p` integrator guards) and **never**
`GetElasticModuli`. `p_r` floors **strength**; nothing floored **stiffness**.

`-pRe <value>` is the mirror image. One added term, at the three
`GetElasticModuli` overloads and nowhere else in the file:

```cpp
// SRC/material/nD/UWmaterials/ManzariDafalias.cpp:4956, :5020, :5062
double pn = one3 * GetTrace(sigma) + m_PreElastic;   // Ladruno (ADR-93 II.1)
pn = (pn <= m_Pmin) ? m_Pmin : pn;
```

so the effective argument under the moduli is `sqrt(max(p + pRe, p_min)/P_atm)`.
`m_PreElastic` is zeroed in `ManzariDafalias::initialize()`
(`ManzariDafalias.cpp:932`), so vanilla is untouched.

**The tokens.** `-pRe` is canonical;
`-pre` / `-Pre` / `-PRe` / `-Pelastic` / `-pelastic` are accepted synonyms
(`SRC/material/nD/LadrunoSANISAND.cpp:286-288`). One value, in the deck's stress
units. Default `0` = OFF = byte-identical.

**The parser refusals, exactly** (`LadrunoSANISAND.cpp:286-319`):

| deck | outcome |
|---|---|
| `-pRe -1` | **REFUSED** — "`-pRe must be >= 0`" (`:311-318`) |
| `-pRe 1 -pRe 5` | **REFUSED** — "given more than once" (`:295-302`); a synonym counts as the same flag |
| `-pRe 1 -Pelastic 5` | **REFUSED**, same rule |
| `-pRe 50` (i.e. `> 0.1·P_atm`) | accepted **+ WARNING** |
| `-pRe 1 -Pmin 5` (i.e. `pRe <= p_min`) | accepted **+ NOTE** in the echo (`:1275`) |
| `-pRe 1` with default `-Pmin 0.0101` | accepted, silent — the documented pairing |

The repeat-refusal is deliberate and is the one that differs from every other
flag in that parser: the others last-win silently, which the fork calls
survivable for a switch and not for a constitutive constant, because the echo
can print only one value.

**Wiring.** Trailing constructor argument on both full constructors
(`LadrunoSANISAND.h:251`, `:270`), `mPreElasticInput` as the stored request
(`LadrunoSANISAND.h:413`), `applyLadrunoConstants()` as the single writer of the
base seam (`LadrunoSANISAND.cpp:1091`) so `revertToStart → initialize()` cannot
drop it, `getCopy(const char*)` carries it to every Gauss point on both wrapper
branches (`:1523`, `:1548`), fork wire `Vector(34) → Vector(35)` slot 34
(`:1770` send, `:1801` recv), echo at `:1259-1260`, `Print()` at `:4533-4534`.

**Stage 0 is inert, and that is correct — not a defect.** `mElastFlag` is a
static on the base, `0` until `updateMaterialStage … 1`, and in the
`mElastFlag == 0` branch the `sqrt(pn/P_atm)` factor is **dropped entirely**.
The vanilla elastic stage is pressure-**independent**, so `pn` is computed and
unused, and `-Pmin` is inert there for the same reason. The fork measured a
stage-0 leg bit-identical at `pRe` = 0, 1e6 **and** with `-Pmin` moved
`0.0101 → 10.0`. This matters to an emitter: **a gravity/K0 stage emitted with
`-pRe` is byte-identical to one without it**, and a harness that tries to
observe the flag during stage 0 will conclude, wrongly, that it does nothing.

Where it *is* observable at zero strain: the **initial elastic operator** after
`updateMaterialStage … 1` + `revertToStart`, via
`refreshInitialElasticOperator()` (`LadrunoSANISAND.cpp:1141-1156`), which
re-derives `mCe = mCep = mCep_Consistent` at `p = P_atm`. It scales by exactly
`sqrt((P_atm + pRe)/P_atm)` — the fork pins `2.000000000` on all six modes of an
unstrained `stdBrick` at `pRe = 3·P_atm`, to `1e-9` on the ratio. The function
early-returns at `mPreElasticInput == 0.0`, so a default deck re-executes not
one floating-point operation.

**Banner:** `LadrunoSANISAND — -pRe elastic-only stiffness floor (ADR-93 II.1)`
(`Ladruno_scripts/banner_features.txt`, last line at `48c0e99bc`).

### 1.2 Measured effect — and the refutation that decides the default

Three claims, and they are **different claims**; quoting one for another is the
trap this section exists to prevent.

**(a) It adds no strength — a GAUSS-POINT claim.** At `p0 = 1.01` kPa,
`pRe = 1` kPa moves `eta/M^b` by under `1e-5` absolute where `-Presidual 1.01`
moves it **+18.1 %**. The bounding state `eta = M^b` is unmoved. This is *not* a
statement that a footing carries the same load at the same settlement — at
working settlements the floored arm's `q` is 5.2–6.8 % higher.

**(b) It cuts the seizure 324× — where the point actually reaches the floor.**
At `p0 = 0.5` kPa the unfloored arm runs 1570 substeps/step and stalls at step
13; `pRe = 1` kPa takes it to 4.8/step and completes 40/40. **Not monotone:**
`pRe = 0.1` kPa on the same leg failed at step 1.

**(c) On the campaign's own strip footing it is WORSE on every axis.** This is
the F13 BVP leg the ADR-93 request asked for, and it is the reason the default
stays `0`. Plane-strain strip, `B = 2` m, implicit, `--surcharge 7.65` kPa
(PM-01 D20's minimum-embedment pressure), coarse `h0 = 1.0` m:

| arm | `s/B` in ~2000 s | `q` end | median substeps/step | worst point | failed | subdiv |
|---|---|---|---|---|---|---|
| `pRe = 0` | **0.0603** | 1392.94 kPa | **26 348** | **1 387** | 55 | 0/80 |
| `pRe = 1` | 0.0389 | 991.49 kPa | **50 789** | **18 831** | 245 | 16/80 |

Committed load–settlement delta over the common range: **median +5.88 %, max
+7.96 %**. Both arms are **WALL-terminated, so neither `q` is a capacity.**

Why: the live ring under that surcharge is **confined at `p_min = 6.25` kPa**,
not at the floor. `sqrt((6.25 + 1)/6.25) = 1.077` is a 7 % bump on `G` at a point
that had stiffness already. The floor does not know where the ring is; it acts
wherever `p` is comparable to `pRe`.

Full record: `Ladruno_implementation/93_ladruno_sanisand_zero_confinement_adr.md`
§7.4–§7.10 (§7.9 is the BVP leg), and
`Ladruno_implementation/LadrunoSANISAND_implex_guide.md:190-260` (§3.1, "a
stiffness floor, and the three things it is not").

### 1.3 What apeGmsh must adopt

**(a) A `p_re` field on the `LadrunoSANISAND` primitive, with validation
mirroring the parser — in TWO places.**

The primitive is the frozen dataclass
[`src/apeGmsh/opensees/material/nd.py:802`](../src/apeGmsh/opensees/material/nd.py).
The **public** surface is the namespace factory
[`_internal/ns/nd.py:264`](../src/apeGmsh/opensees/_internal/ns/nd.py)
(`def LadrunoSANISAND(self, *, …)`), whose keyword-only signature spells every
field out by hand (`:267-297` — `p_residual` `:290`, `p_min` `:291`,
`honor_tol_r` `:292`, `max_substeps` `:293`) and forwards them at `:325-359`.
**A field added to only one of the two is invisible from `ops.nDMaterial.…` and
reds `tests/opensees/unit/test_ns_wrapper_default_parity.py`**, which exists to
pin exactly that drift. Add it to both, with the same default.

The primitive's fork-constant block is `nd.py:993-996`:

```python
p_residual: float = 0.0
p_min: float | None = None      # None -> resolved to 1.0e-3 * P_atm
honor_tol_r: bool = False
max_substeps: int = 0           # 0 = uncapped = vanilla's behaviour
```

Add `p_re: float = 0.0` there, immediately after `p_residual` — it is the same
kind of thing and the docstring can then contrast the two in place.

Validation belongs in `__post_init__` (`nd.py:1003`), beside the existing
`p_residual` (`:1036-1042`) and `p_min` (`:1043-1047`) checks, and mirrors the
parser exactly:

- `p_re < 0` → **`ValueError`** (parser refuses; `LadrunoSANISAND.cpp:311-318`).
  Use the parser's own sentence: it is a **stiffness** floor, `-Presidual` is the
  strength one.
- `p_re > 0.1 * P_atm` → **`SanisandIntegrationWarning`**. `P_atm` is a field on
  the same dataclass, so apeGmsh can raise this at declaration time rather than
  waiting for the deck to be read — the same reasoning that already resolves
  `p_min=None` client-side at `nd.py:1157-1160`.
- `0 < p_re <= p_min` (with `p_min` resolved as `_emit` resolves it) →
  **`SanisandIntegrationWarning`** carrying the fork's NOTE: as `p → 0` the
  floored argument tends to `max(pRe, p_min)`, so the clamp already dominates at
  the ring **while `G` is still perturbed wherever `p ~ pRe`** — worst of both,
  and invisible from either value alone.
- The fork's repeat-refusal has **no apeGmsh analogue and needs none**: a
  dataclass field cannot be given twice. Say so in the docstring so nobody
  "adopts" it.

**Emission** goes in `_emit` at `nd.py:1130`. Put it **after** `-Presidual`
(`nd.py:1156`) and **before** `-Pmin` (`nd.py:1157-1160`), matching the fork
guide's synopsis ordering
(`LadrunoSANISAND_implex_guide.md:73`:
`-Presidual $pr -pRe $pre -Pmin $pmin -honorTolR $h -maxSubsteps $N`).

**Emit it only when non-zero**, i.e. follow `max_substeps`
(`nd.py:1167-1168`), **not** `-Presidual` / `-Pmin` / `-honorTolR` (which always
emit). The three that always emit do so because their defaults differ from
vanilla's; `p_re`'s default **is** vanilla's, so an unset deck must stay
byte-identical to the one produced before the field existed. This is the same
rule ADR 0103 D4 already wrote down for `max_substeps` and ADR 0107 D4 for
`implex_factor`.

Use the canonical token `"-pRe"`. Do not offer the synonyms — the fork accepts
them for a memo's sake, and a second spelling in apeGmsh is a second thing to
keep in sync.

**(b) Docstring: the three claims, kept apart.** The class docstring at
`nd.py:803-970` already has a precedent for this in the `implex_control`
paragraph (`nd.py:944-956`), which spends six lines saying what a fork
measurement does **not** license. `p_re` needs the same treatment:

1. capacity-neutrality is a **Gauss-point** claim (`eta = M^b` unmoved), not a
   BVP claim;
2. it is **path-changing everywhere** — `L` changes by
   `λE/(Kp + λE)` vs `E/(Kp + E)` with `λ = sqrt((p + pRe)/p)`, `1.41` at
   `p' = 1` kPa — and is neutral only in the limit `b:n → 0`, i.e. at the
   bounding surface;
3. on the fork's own surcharged strip it is **measured worse** (§1.2(c)), so it
   is not a default and not a recommendation.

Also record the **explicit-`dt` consequence** the fork's guide carries
(`LadrunoSANISAND_implex_guide.md:239`): a floored `G` shortens the critical
time step by `1/sqrt((p + pRe)/p)`, up to 41 % at `p' = 1` kPa. Any apeGmsh
explicit-dynamics path that sizes `dt` from a material estimate must budget for
it.

**(c) A documented minimum-build constant.** `SANISAND_PRE_FLOOR_MIN_BUILD =
"1133279a5"` beside the five that already live in `nd.py` —
`:223` `DP_ADR95_MIN_FORK_BUILD`, `:1369` `ASDP_MIN_FORK_BUILD`,
`:1378` `ASDP_CLOSEST_POINT_MIN_BUILD`, `:1392`
`SANISAND_IMPLEX_FACTOR_MIN_BUILD`, `:1405` `ASDP_DILATANT_APEX_MIN_BUILD`
(and `_target.py:113` `TIMS_FORK_BATCH_MIN_BUILD`, `emitter/tcl.py:65`
`TCL_COUPLING_TOKENS_MIN_BUILD` elsewhere) — exported from `nd.py:71`'s
`__all__` and cited from the `p_re` docstring. All of them are
documented-not-enforced; this one is too.

**Documented, not enforced**, and here the ADR 0107 D4 reasoning holds cleanly:
an older parser meets `-pRe` as an unknown flag and **refuses at parse time**
(`LadrunoSANISAND.cpp:783` lists the accepted flag set in its usage line), so
there is no silent-wrong-answer path to guard against. A refused deck is loud.

**(d) H5 round-trip.** `p_re` is a primitive field like `p_residual`; it
round-trips through whatever mechanism carries the rest of the dataclass. It is
**not** a new schema zone and needs no bump *if* the material primitive's fields
are serialised generically — confirm this before writing the code, because
`_kernel/_coupling_control.py`'s `cpl_al_update` needed an explicit
`NEUTRAL_SCHEMA_VERSION` bump in #1149 and the two mechanisms are not the same.

### 1.4 Verification recipe

The fork's gate is `tests/test_ladruno_sanisand_pre_floor.py` (14 passed, 1
skipped; `15 passed in 104.59s` with `--runslow`).

apeGmsh-side:

1. **Byte-identity at the default.** Emit a `LadrunoSANISAND` deck with `p_re`
   unset and with `p_re=0.0` and assert both produce the **same deck text** as
   the pre-change emitter — no `-pRe` token in either. This is the same shape as
   the existing `max_substeps=0` golden and belongs beside it in
   `tests/opensees/unit/primitives/test_materials_nd.py`.
2. **Token position.** Assert the emitted flag tail is
   `-Presidual … -pRe … -Pmin … -honorTolR …` in that order, and that no
   positional follows any flag (a positional after a flag is a **hard parse
   error** in `OPS_LadrunoSANISAND`, by design — `nd.py:1149-1152` already says
   so).
3. **Refusals.** `p_re=-1.0` raises `ValueError`; `p_re=0.5*P_atm` warns;
   `p_re` at or below the resolved `p_min` warns.
3b. **Namespace parity.** `ops.nDMaterial.LadrunoSANISAND(p_re=…)` reaches the
   primitive, and `tests/opensees/unit/test_ns_wrapper_default_parity.py` stays
   green — i.e. the factory default and the dataclass default are the same `0.0`.
   The namespace-reachability pattern is
   `tests/…/test_materials_nd.py:940` / `:954`.
4. **Live (optional, `integration_ladruno`).** On a build ≥ `1133279a5`:
   `updateMaterialStage … 1`, `revertToStart`, then `eigen` on an unstrained
   `stdBrick` — every mode scales by exactly `sqrt((P_atm + p_re)/P_atm)`. Use
   `p_re = 3*P_atm` so the factor is the exact binary `2.0`. This is the *only*
   zero-strain probe that sees the flag; a stage-0 probe is bit-identical by
   construction and would read as a false negative.

### 1.5 Nothing to do

- `-Presidual`, `-Pmin`, `-honorTolR`, `-maxSubsteps`, the whole `-implex*`
  family: unchanged by #842.
- `ManzariDafalias` / `SAniSandMS` (`nd.py:622`, `nd.py:783`): `-pRe` is a
  `LadrunoSANISAND` flag only; the base classes have no parser for it.
- The reader (`results/readers/_ladruno_element_io.py`): `-pRe` adds **no**
  response and **no** bucket. `implexRefusals` stays 6 wide.
- Recorder plumbing, `_response_catalog.py`: nothing.
- Golden files: nothing moves at the default, which is the point of (a)'s
  emit-only-when-non-zero rule.

---

## 2. WP-107 — the threaded `Domain::update()` element loop and `ladrunoThreads` (#843)

### 2.1 What changed in the fork

ADR-75b closed Lane 3 for the cluster path — loop A was 0.26 % of the step at
540 675 DOF under MUMPS, failing the >40 % gate by 42×. WP-107 takes up §13's
own last paragraph: a **desktop-scoped** case, now measured rather than assumed.
On a TIMs-shaped deck (`LadrunoQuad -bbar` + `LadrunoSANISAND`, 6 400 elements,
`system Pardiso`) loop A is **51.23 %** of the step — and on the same element
with `ElasticIsotropic`, **7.80 %**.

**Build.** CMake `option(LADRUNO_OPENMP … OFF)` (`CMakeLists.txt:790`), compile
flag attached **PRIVATE to `OPS_Domain` + `OPS_Utilities` only** (`:818-824`) so
the fork's 7 dormant PFEM `#pragma omp` lines stay dead.
`Ladruno_scripts/build.bat:340-342` passes `-DLADRUNO_OPENMP=ON` explicitly (and
`=OFF` under `LADRUNO_NO_OPENMP`).

**The default stays OFF in CMake, and that is a defect, not a preference.**
Flipping it ON made Zone-A segfault deterministically, twice on the same commit,
in `test_adr30_projection_p0.py::test_massless_dof_is_not_policeable_by_the_soe_layer`
— the zero-mass `system Diagonal` case, **at 1 thread**, where the threaded loop
returns before touching anything. gcc's `-fopenmp` codegen/link turns an already
untrustworthy singular-mass failure path into a hard crash; it does not reproduce
on MSVC. **So: Windows/MSVC via `build.bat` has it; Linux/Esmeralda and CI do
not, and CI does not exercise the threaded loop at all.** Full record:
`BUILDING.md:76-99`, `Ladruno_internal/BUILD_GOTCHAS.md` §15, ADR-75b §14.4.

**Runtime.** One knob, three spellings, default **1**:

| surface | spelling |
|---|---|
| Tcl | `ladrunoThreads <n>` (query with no arg) — `SRC/interpreter/TclWrapper.cpp:1992` |
| Python | `ops.ladrunoThreads(n)` — `SRC/interpreter/PythonWrapper.cpp:3378` |
| env | `LADRUNO_THREADS` — seeded once per process, `SRC/utility/LadrunoThreads.h` |

The verb is `OPS_LadrunoThreads` (`SRC/interpreter/OpenSeesOutputCommands.cpp:1621`),
which returns the stored count. Nothing ever calls `omp_set_num_threads()`; the
count is applied **only** through an explicit `num_threads(n)` clause on
`Domain::update`'s own region, so PFEM's dormant pragmas keep running on the
runtime default.

**At 1 thread the serial path is taken** — not a one-thread parallel region — so
`threads=1` is byte-identical to an `LADRUNO_OPENMP=OFF` build, verified by the
fork.

**Bit-identical by construction, not by tolerance.** `Domain::update()` has no
reduction into the SOE (ADR-75b §2.1); `ok` is an integer `reduction(+:)` and the
reported failing element is the lowest index in *serial* order (a `critical`,
because MSVC is OpenMP 2.0 and has no `reduction(min:)`). 12/12 runs at 1/2/4/8
threads share one full-field md5.

### 2.2 The refusal roster, exactly

`Domain::ladrunoThreadedUpdate()` (`SRC/domain/domain/Domain.cpp:2552`) is
**all-or-nothing**: one un-audited element in the domain and the *whole* loop
runs serial, loudly.

| condition | site | behaviour |
|---|---|---|
| MPI **build** (`OpenSeesSP` / `OpenSeesMP` / `OpenSeesPyMP`) | `Domain.cpp:2586-2593` via `SRC/utility/LadrunoParallelBuild.cpp` | **REFUSED**, warns |
| `PartitionedDomain` / `Subdomain` | `PartitionedDomain.h:116`, `Subdomain.h:97` | **REFUSED**, warns |
| deep profiler armed | `Domain.cpp:2608-2617` | **REFUSED**, warns |
| fewer than 2 elements | `Domain.cpp:2628-2629` | silently serial — **no message** |
| any element answering `ladrunoThreadSafeUpdate() == false` | `Domain.cpp:2637-2646` | **REFUSED**, names the tag and classTag |

**ALLOWED as shipped** (`Element.h:143` defaults to `false`, so this list is the
entire allowlist):

- `LadrunoQuad` std / bbar / ssp under `-geom linear` — `LadrunoQuad.cpp:719`
- `ElasticIsotropicPlaneStrain2D` — `ElasticIsotropicPlaneStrain2D.h:96`, and
  only on the `update()` path (`setTrialStrain` writes only the instance's
  `epsilon`)

**REFUSED, and the important one:** `ManzariDafalias` / `LadrunoSANISAND`, every
`IntScheme` (`LadrunoSANISAND.cpp:1931`). The audit said it was safe — no
function-scope static anywhere on its update call graph — and the first sweep
returned 12/12 bit-identical at 1/2/4/8 threads. Then a harder load path
**segfaulted, 4/4 at 4 threads**, as soon as the plastic branch was exercised in
volume. `#pragma omp critical` around the *entire* `theEle->update()` still
faults, so it is **not a data race between element updates** — something about
running that path on an OpenMP worker thread at all. Root cause **not located**,
so the family is refused. Also refused: `LadrunoQuad -eas` and `-geom finite`,
and everything un-audited.

**Announcements are per-domain, not per-process** (red-team S3, `Domain.cpp`
around `:2486` and `:2683`), keyed on `(element generation, outcome, tag, thread
count)` with the generation bumped by `addElement` / `removeElement` /
`clearAll`. The fork's own comment names apeGmsh as a reason: *"A second model in
the same interpreter — pytest, apeGmsh, any in-process parameter study — must say
what it did too."* A steady state prints nothing, so a Newton iteration is
silent.

### 2.3 Measured speed-up — read this before exposing the knob

Idle box, 3 repeats, `MKL_NUM_THREADS=1`, `system Pardiso`, **elastic** deck,
14 400 elements:

| threads | speed-up | bit-identical |
|---|---|---|
| 1 | 1.00× | yes |
| 2 | **1.03×** | yes |
| 4 | **1.03×** | yes |
| 8 | **0.94× — a REGRESSION** | yes |

Loop A is 7.80 % of that deck's step, so Amdahl caps the whole-step prize at
**1.08×**. The fork explicitly **withdrew** an earlier 1.00/1.09/1.11× table
taken on a contended box. The deck where loop A is 51 % is the SANISAND one —
which is **refused**. So: *the only deck that would pay is the one that cannot
run threaded.* Say that wherever apeGmsh exposes the knob.

### 2.4 What apeGmsh must adopt

**(a) `ops.ladruno_threads(n)` on the live bridge — a run option, not an
analysis-chain primitive.**

It is a **process/runtime** setting, not a declaration: it does not round-trip
H5, it is not part of the model, and it must not appear in a `StageRecord`. The
right home is beside the other fork-only live verbs on `apeSees`
(`apesees.py:9030` `ladruno_projection_tie_force`, `:9134`
`ladruno_contact_force`, `:9163` `ladruno_contact_info`, `:9178`
`ladruno_mortar_penetration`, `:9191` `ladruno_mortar_tie_residual`) forwarding
to a matching method on `LiveOpsEmitter`
(`emitter/live.py:1103-1156`, the same block).

Semantics the forwarder owes:

- **Refuse on a non-fork backend, up front.** The right seam is
  `LiveOpsEmitter._stock_build_gate` (`emitter/live.py:770`, testing
  `hasattr(self._ops, "criticalTimeStep")` at `:789`) — the gate used for
  commands whose effect **cannot be verified after the fact**, which is exactly
  `ladrunoThreads`'s shape. Follow the message-constant convention beside it
  (`live.py:41` `_AUGMENT_FORK_REQUIRED`, `:52` `_PROFILER_FORK_REQUIRED`, `:67`,
  `:80`, `:90`, `:101`, `:114`, `:216`) so the refusal names
  `APEGMSH_OPENSEES_BIN` (`live.py:45`) rather than raising `AttributeError`. Do
  **not** use the post-hoc `_element_fork_gated` path (`live.py:747`): there is
  nothing to read back.
- **Refuse on a build below the floor.** `get_backend_build()`
  (`emitter/live.py:426`) returns `ops.ladrunoBuild()`; a build that predates
  `48c0e99bc` has no verb at all and `getattr(ops, "ladrunoThreads", None)` is
  `None`. Fail loud with the min-build constant (§5), do not silently no-op —
  a silently-serial "threaded" run is exactly how a bench lies, which is the
  anti-goal `LadrunoThreads.h` exists to serve.
- **Refuse `n > 1` where the fork will refuse it**, client-side and *before* the
  run, in the three cases apeGmsh can decide from the declaration:
  1. a partitioned model (apeGmsh already knows — `_emit_flat` and the
     `partition_open`/`partition_close` brackets at `live.py:444-460`);
  2. a model whose element set is not a subset of
     `{LadrunoQuad(std|bbar|ssp) with geom="linear"}` — apeGmsh has the registry
     to decide this (`_element_capabilities.py`, where
     `element_propagates_material_refusal` lives at `:815`);
  3. a model holding any `ManzariDafalias`-family material — walk the material
     graph the way `validate_sanisand_substep_cap`
     (`_internal/build.py:4086`) already does, which is the same transitive walk
     through `PlaneStrain` / `LogStrain` wrappers.
  A warning is **not** enough for (3): the fork's failure mode there is a
  **segfault**, which takes the Python process with it and loses the session.
- **Never set it implicitly.** Default 1, no auto-tuning, no "use all cores". The
  measured curve regresses at 8 threads on the only deck that can run.
- **Do not swallow the announcement.** The fork prints once per domain per
  outcome; apeGmsh's live path must not run with `opserr` redirected in a way
  that hides "running the element loop SERIAL" — otherwise a refused run and a
  threaded run look identical.

**(b) `LADRUNO_THREADS` in the subprocess lanes.** `apesees.py:11450` builds
`child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}` and hands it to
`stream_run` (`_run.py:126`, `Popen` at `:173`, `env=env` at `:181`). Note the
lanes do **not** share an env var: the live lane reads `APEGMSH_OPENSEES_BIN`
(`emitter/live.py:290`), while the subprocess lanes resolve through
`OPENSEES_BIN` (`_target.py:154`) and `OPENSEES_VENV` (`:182`) — a distinction
`docs/concepts/backend-capabilities.md:44`/`:49` currently blurs, and one a
thread-count knob must not repeat. Two things are owed:

1. a way to set `LADRUNO_THREADS` for the child (the env var is read **once per
   process**, so it cannot be changed after the fact);
2. **an explicit refusal to inherit a stray `LADRUNO_THREADS` into an MPI
   lane.** `mpiexec` propagates it to every rank; the fork fences this at the
   binary (`Domain.cpp:2586-2593`), but apeGmsh should not be the thing that set
   it. `run_remote` (`apesees.py:11476`) writes SLURM job scripts — one job-script
   line here is the fork's own worked example of a silently 64-way
   oversubscribed run.

**(c) A new ADR.** This is a new public surface with refusal semantics, a
platform asymmetry and a "do not adopt for the deck you actually care about"
verdict. It does not fit inside 0103 (the SANISAND *deck contract*) or 0104 (the
`Substep` rung). Number it after `0109`; cite ADR-75b §14 and §14.4, and record
D-items for: run-option-not-primitive, client-side pre-refusal including the
SANISAND segfault, default 1, and the Linux-OFF caveat.

**(d) Doctor / capabilities.** `OpenSeesCapabilities` (`_target.py:66-101`)
carries `has_fork` (`:77`, set from `has_profiler` at `:221`), `has_profiler`,
`has_ladruno_up`, `build`; it is reached via `apeSees.capabilities()`
(`apesees.py:8228`). Add `has_threaded_update: bool = False` — but probe it
**behaviourally**, not from the build hash: `ops.ladrunoThreads(2)` on a binary
built without `LADRUNO_OPENMP` warns and stays serial
(`OpenSeesOutputCommands.cpp:1637-1643`), and `ops.ladrunoBuild()` is a
**configure**-time stamp that lags an incremental build (§5). The honest probe is
"the verb exists **and** requesting 2 did not print the WITHOUT-LADRUNO_OPENMP
warning". Note this is the **first** engine capability that is false on a
correctly-built fork binary, so `has_fork and not has_threaded_update` must not
read as "something is wrong".

**(e) The published docs page.** `docs/concepts/backend-capabilities.md:75`
(`## The fork-only surface`) enumerates the fork-only elements (`:80-84`) and
integrators (`:95-98`). A fork-only *verb* with a platform asymmetry belongs
there, and it is the page a user checks before writing `ladruno_threads` into a
script. While editing it, fix the `APEGMSH_OPENSEES_BIN` resolution order at
`:44`/`:49`, which is stated as general and is live-lane only (§2.4(b)).

### 2.5 Verification recipe

The fork's gate is `tests/test_wp107_threaded_update.py`, 18 `zone_a` cases with
a module-level `skipif` driven by a child-process probe, biased to **RUN** on
every ambiguous outcome. Its mutation gate (M3: `LadrunoQuad::shp`/`shpBar`
`thread_local` → `static`) was actually run: the pre-existing 128-test suite
stayed **green** and the new file went **7 failed / 11 passed**.

apeGmsh-side, none of it needing a threaded build:

1. **Refusal roster, unit.** A model with a `ManzariDafalias`-family material and
   `ladruno_threads(4)` raises; the message names the material. Same for
   `LadrunoQuad(geom="finite")`, `-eas`, a partitioned model, and any element
   outside the allowlist.
2. **`threads=1` is a no-op.** Declaring `ladruno_threads(1)` produces the same
   emitted deck and the same live call sequence as not declaring it.
3. **Stock-backend refusal.** With `get_backend_name() == "stock-openseespy"`,
   the verb raises with the `APEGMSH_OPENSEES_BIN` message.
4. **Live (`integration_ladruno`), Windows only.** On a `build.bat` binary ≥
   `48c0e99bc`: a 2×2 `LadrunoQuad` + `ElasticIsotropicPlaneStrain2D` model with
   ≥ 2 elements announces `THREADED on 2 threads` at `n=2` and does **not**
   announce at `n=1`; full nodal field is bit-identical between the two. Mark it
   `skipif` on a non-fork / non-OpenMP build, biased to run, exactly as the fork
   does — an apeGmsh test that skips silently on the office Linux box is a test
   that will never fail.

### 2.6 Nothing to do

- Every emitted deck: `ladrunoThreads` is a **runtime** verb; no Tcl/py deck
  emitter changes, no H5 field, no schema bump, no `StageRecord` field.
- `Ladder` / `Substep` (`analysis/strategy.py:247`, `:534`) and
  `_internal/analyze_rc.py`: #843 adds no return code and changes none. `-4`
  (`analyze_rc.py:42`) and `-33086` keep their #1148 meanings.
- The reader and `Results`: no new response, no bucket.
- `system Pardiso` threads (`ADR 0106`): a **different** knob. The fork's
  measurements were taken at `MKL_NUM_THREADS=1` precisely so the two would not
  multiply; do not fold them into one setting.
- MPI / `run_remote` behaviour: unchanged, because the fork refuses at the
  binary. The only apeGmsh obligation is (b)(2) — do not *set* the env var into
  an MPI lane.

---

## 3. WP-105 — the `IntScheme 2` verdict (#844)

### 3.1 The verdict: PARTIAL

**Documentation only.** No fork source file was touched, no model was run in
#844; the measurements are against build `634824e1f`.

**QUALIFIED where the strain increment is GIVEN** (a companion return, a
prescribed-strain probe, zero free DOF). Against `IntScheme 1`:

| `dEz` | accuracy | cost |
|---|---|---|
| `1e-5` | matches scheme 1 to `1.3e-3` / `2.9e-3` max relative stress deviation (`p0 = 100` / `20` kPa); terminal `eta` within `4.2e-4` / `2.0e-4` | **0.64×** — scheme 2 is *slower* here |
| `1e-4` (the campaign increment) | **3.7–4.3× more accurate** | **4.2–7.6× cheaper** |
| `4.6e-4` | **7–30× more accurate** | **10–13× cheaper** |

And at `p0 = 20` kPa it is **scheme 1**, not scheme 2, that leaves its own
bounding surface (`eta/M^b = 1.056`).

**REFUTED as a load-controlled BVP's primary integrator.** Under a global
Newton at `dEz >= 1e-4` it **stalls in 8 of 8** free-standing drained-triaxial
arms (scheme 1: 1 of 8); loosening the global tolerance `1e-9 → 1e-7` does not
rescue it; failing steps cost **12–134 s** against a 30 ms normal step. On the
real CP1/ADR-95 bearing leg (`x10z8`, `h1.0_e0.6944`, 1200 s each) scheme 2
committed 11 steps to `s/B = 4e-5` against the baseline's 51 steps to
`s/B = 0.019` — **475× shallower for the same wall clock**, `ds` pinned at 25×
the subdivision floor, **100 % of committed steps on the relaxed rung 3**.

**ADR-92 D3's stated rationale corrected, conclusion unchanged.** D3 justified
the scheme-1 default with "58–74 % of scheme 2's calls integrate explicitly at
low `p`". Measured: **0 of 1820** steps on every replayed triaxial path at
`p0 = 100`/`20` kPa, and **0 of 80** on the descent of a `p → p_min` path. The
58–74 % figure reproduces (53 %) only once the point is **pinned at `p_min` with
a zero deviator**. D3's conclusion survives on stronger grounds than D3 gave.

Full tables:
`Ladruno_files/testbed/hypo_bearing/adr92_f12/F12_intscheme2_verdict.md`;
guide text at `LadrunoSANISAND_implex_guide.md:129-147` (§3) and §9
(`:567`).

### 3.2 Two source facts an emitter has to carry

**(a) CPPM can never report failure.** `ManzariDafalias::integrate()` **discards**
`BackwardEuler_CPPM`'s return value, `debugFlag` is compiled off, and the CPPM
ladder always ends in `errFlag = 1` after recursing through up to **512
half-increments**. So a scheme-2 step that cannot return costs 12–134 s and then
reports success. Nothing refuses; nothing is logged. A driver watching return
codes sees a converged run.

**(b) `TanType 2` under scheme 2 is a genuine algorithmic tangent — and any CPPM
fallback silently overwrites it** with `ModifiedEuler`'s chained tangent. So the
tangent a scheme-2 deck runs is not a property of the deck; it depends on whether
the ladder fell back on that step, which nothing reports.

Both are in `LEDGER_quirks.md` as new rows.

### 3.3 What apeGmsh must adopt

**Guidance only — no emitter change.** `int_scheme=2` is already accepted
(`nd.py:1016`) and already emitted in the always-on 5-positional tail
(`nd.py:1140-1147`).

**(a) `int_scheme` docstring, `nd.py:880-906`.** Add a scheme-2 paragraph with
the same shape as the `implex_control` caveat: material-point **yes**, BVP
integrator **no**, and the two source facts from §3.2. The operative sentence is
the fork's: *use it where the increment is given; do not make it the primary
integrator of a load- or displacement-controlled BVP without a cap on the
ladder.*

**(b) ADR 0103, dated amendment.** 0103's own subtitle is "Changes the
`LadrunoSANISAND.tan_type` default `0 → 2`", and **D1** (`:71`) is the
default-`tan_type` decision — which §3.2(b) now qualifies: on `int_scheme=2`,
`tan_type=2` is the algorithmic tangent *except* on a fallback step, silently.
0103 already carries an `Amendment (2026-09-15, fork PR #838)` at `:145`, so
add `Amendment (2026-09-16, fork PRs #844/#845)` after it covering both §3 and
§4. **Do not change D1's decision** — the default stays `2`; the amendment
records what `2` means under scheme 2.

**(c) `internal_docs/guide_ladruno_sanisand_integrator.md`.** §2 ("`tan_type` —
DONE", `:28`) gets the fallback caveat; a new short §3a or a bullet under §3
(`:75`) carries the verdict. This is the working-memory doc 0103's header points
at (`0103-…md:19-21`), so the measurement tables belong here rather than in the
ADR.

Noted while looking: **there is no published `docs/` page covering SANISAND,
`IntScheme`, `TanType` or `-maxSubsteps` at all** — a grep for `sanisand` over
`docs/` returns zero markdown hits, and the only published `implex=True` guidance
(`docs/guides/nonlinear_concrete_solver.md:54`, `:226`) is for
`LadrunoConcrete3D`. Everything in this family lives in `internal_docs/` and in
docstrings. That is a gap, not an adoption item, and it is not created by this
batch — but it is why §3's guidance has to go in the **docstring** (a) as well as
the internal guide (c).

**(d) Do NOT add an `int_scheme=2` warning.** apeGmsh's existing scheme warnings
(`nd.py:1023-1034`, schemes 3/5) fire on schemes with **no error control at
all**. Scheme 2 is error-controlled and is the better operator in its own
regime; warning on it would be wrong in the material-point case, which is the
case the fork *qualified*. The docstring is the right surface.

**(e) The relaxed-rung discipline, again.** The bearing leg spent **100 % of
committed steps on rung 3**. #1149 already put "the relaxed-rung count belongs
next to any ASD-vs-UW figure" in the `Ladder` docstring
(`analysis/strategy.py:534`); extend that sentence to cover any SANISAND
integrator comparison. A scheme comparison quoting `s/B` without the rung
histogram is comparing two different analyses.

### 3.4 Verification recipe

Nothing to execute — this is a docs change. What must be *checked* is that no
apeGmsh text still recommends scheme 2 as a general robustness improvement, and
that the two `not verified` boundaries from #844 are reproduced wherever the
verdict is quoted: it covers **neither** plane strain, `LadrunoUP`,
cyclic/reversal paths, `-implex` + scheme 2, nor parallel; `-implex` was **OFF
in every WP-105 arm**; and no capacity/plateau/limit-point claim is licensed
(the bearing leg never reached a comparable `s/B`, so the 1 % load–settlement
bar could not be evaluated at all).

### 3.5 Nothing to do

- Emitters: none. The tail already always emits (`nd.py:1140-1147`), which is
  exactly why the deck is unambiguous about which scheme it runs.
- `_SCHEMES_REACHING_MODIFIED_EULER`: changed by **§4**, not by §3.
- `tan_type` default: stays `2` (ADR 0103 D1).
- The reader, the `Ladder`, `analyze_rc`: nothing.

---

## 4. WP-108 — the false "`-maxSubsteps` has NO EFFECT with IntScheme 2" warning is gone (#845)

### 4.1 What was wrong

`LadrunoSANISAND::schemeReachesModifiedEuler()` returned `false` for
`mScheme == 2`, so the constructor (`LadrunoSANISAND.cpp:1403-1409`) and
`Print()` (`:4584`, `:4622`) both told the user `-maxSubsteps` and `-honorTolR`
had **NO EFFECT** with `IntScheme 2`.

WP-105 measured that false. `ManzariDafalias::explicit_integrator`'s
`switch(mScheme)` does not enumerate `INT_BackwardEuler`, so whenever
`BackwardEuler_CPPM`'s own recursive-halving ladder falls back to
`explicit_integrator` — on non-convergence or ladder exhaustion — that call hits
`default:` → `ModifiedEuler`, **the same seam `-maxSubsteps` / `-honorTolR`
read**. Measured directly: a `-maxSubsteps 100` cap turned a run that completed
40/40 steps uncapped (up to 1282 substeps) into one that refuses at step 18.

**The fix is one branch** (`LadrunoSANISAND.cpp:1204-1210`), ahead of the
existing `s > 9` catch-all:

```cpp
if (s == INT_LSANISAND_BackwardEuler)
    return true;
```

so the helper now answers `true` for schemes **0, 1 and 2** (plus anything
`> 9` other than 45). All four call sites route through it, so one function
closes all of them. Scheme 1's output is unchanged. The helper's contract, in
the fork's own words, is *"can this deck's cap ever bind"*, not *"does every step
of this scheme call ModifiedEuler"* — conditional, fallback-only routing still
earns `true`.

Fork gate: `tests/test_ladruno_sanisand_intscheme2_maxsubsteps.py`, 3 tests,
with the pre-fix binary as its own negative control (2 failed before, 0 after).

### 4.2 apeGmsh carries its own copy of the same false claim — in five places

This is the item that matters. apeGmsh does **not** scrape the fork's warning
text — there is no `opserr` / stderr parser keyed on it anywhere
(`_run.py`'s only stderr consumer is `_solver_stats.py`, anchored on the Pardiso
labels at `:112-121`). Instead apeGmsh **re-derives the same claim
client-side**, and therefore reproduces the bug independently of which fork build
runs the deck:

1. **`src/apeGmsh/opensees/material/nd.py:451-453`** —
   ```python
   _SCHEMES_REACHING_MODIFIED_EULER: frozenset[int] = frozenset({0, 1}) | frozenset(
       s for s in range(10, 100) if s != 45
   )
   ```
   Scheme **2 is absent**. The comment above it (`:441-450`) explicitly says it
   "Mirrors the fork's own `LadrunoSANISAND::schemeReachesModifiedEuler`" — and
   it did, faithfully, including the defect.

2. **`nd.py:1069-1080`** — the `max_substeps` warning keyed on that set
   (`:1071`): *"`max_substeps=N` has NO EFFECT with `int_scheme=2`"*. **False
   since `049b295fc`, and it was false before that too** — the fork's warning was
   wrong, not merely differently-scoped.

3. **`nd.py:1048-1060`** — the `honor_tol_r` warning, same set (`:1050`), same
   defect. #845 fixes both seams with one branch; apeGmsh must fix both with one
   set.

4. **`tests/opensees/unit/primitives/test_materials_nd.py:1339-1348`** — the
   `honor_tol_r` test **pins the bug as correct**, comment included:
   ```python
   # (2 and 45 skip ModifiedEuler; 0 and 1 reach it.  3/5 would ALSO
   # raise the dead-scheme warning, so they stay out of this test.)
   @pytest.mark.parametrize("scheme", [2, 45])
   def test_honor_tol_r_warns_when_scheme_skips_modified_euler(...)
   ```
   with the silent counterpart parametrized `[0, 1]` at `:1350-1356`.

5. **`tests/opensees/unit/test_sanisand_deck_gates.py:232` (`TestSubstepCapField`)**
   — the **same pair again for `max_substeps`**:
   `test_warns_when_the_scheme_skips_modified_euler` parametrized `[2, 45]` at
   `:240-247`, `test_silent_when_the_scheme_reaches_modified_euler`
   parametrized `[0, 1]` at `:249-257`. Two files, four parametrizations; a fix
   that touches only `test_materials_nd.py` cannot go green.

Prose carrying the same claim:

6. **`internal_docs/guide_ladruno_sanisand_integrator.md:87-90`** — rule 2 of §3:
   *"It is inert on schemes that never reach `ModifiedEuler` … Reuse
   `_SCHEMES_REACHING_MODIFIED_EULER`"*.
7. **ADR 0103 D4** (`:117-127`) — *"It warns, like `honor_tol_r`, on a scheme
   that never reaches `ModifiedEuler`."*

**The consequence, stated plainly: a user who asks apeGmsh for
`LadrunoSANISAND(int_scheme=2, max_substeps=N)` is told the cap will do nothing,
and the cap does something — it is the difference between a run that completes
and one that refuses at step 18.** Worse, the advice the warning gives ("Use
`int_scheme=1`") points at the scheme §3 just measured as the *less* accurate one
per increment.

### 4.3 What apeGmsh must adopt

**(a) Add `2` to `_SCHEMES_REACHING_MODIFIED_EULER` (`nd.py:451-453`)** and
rewrite the comment block above it (`:441-450`) to carry the fork's reasoning:
scheme 2 reaches `ModifiedEuler` **conditionally**, through
`BackwardEuler_CPPM`'s fallback, and the predicate answers "can the cap ever
bind", not "does every step route there". Cite
`LadrunoSANISAND.cpp:1193-1215` and the WP-105 measurement (40/40 uncapped →
refusal at step 18 under `-maxSubsteps 100`).

Both warnings then correct themselves, because both key on the same set.

**(b) While you are in there — the `range(10, 100)` half is dead code.**
`__post_init__` refuses any `int_scheme` outside `(0..9, 45)` at `nd.py:1016-1020`
(`tests/.../test_materials_nd.py:1381-1383` pins `int_scheme=10` → `ValueError`),
so no value in `range(10, 100)` can ever reach the membership test. The fork's
own branch is `s > 9 && s != 45` over `[0, 255]`, which apeGmsh's narrower set
does not match either. Either drop the generator and write
`frozenset({0, 1, 2})`, or keep it and say in one line that it is unreachable
defence mirroring the fork's catch-all. Do not leave it as an unremarked
divergence.

**(c) Flip the tests, do not delete them — in both files.** In
`test_materials_nd.py:1339-1356` and
`tests/opensees/unit/test_sanisand_deck_gates.py:240-257`, scheme `2` moves from
the `…_warns_when_the_scheme_skips_modified_euler` parametrize to the
`…_silent_when_the_scheme_reaches_modified_euler` one; `45` stays warning, `0`
and `1` stay silent. Fix the comment at `test_materials_nd.py:1341` too — *"2 and
45 skip ModifiedEuler"* is the claim, not just the parametrize. Four
parametrizations move; nothing is deleted.

**(d) Fix the two prose sites** — `guide_ladruno_sanisand_integrator.md:87-90`
and ADR 0103 D4 (`:117-127`), the latter as part of the same dated 2026-09-16
amendment §3.3(b) already calls for.

**(e) A minimum-build constant, and this one has teeth.**
`SANISAND_SCHEME2_CAP_MIN_BUILD = "049b295fc"` beside the others (§1.3(c)). The
behaviour it gates is **not** new — the cap always bound on scheme 2; what
changed is only whether the binary *says* it does not. So the constant is purely
documentary: on an older build the deck runs identically and prints one false
warning line. State that, so nobody treats the absent warning as a feature probe.

**(f) Watch for a knock-on in the substep-cap gate.** `validate_sanisand_substep_cap`
(`_internal/build.py:4086`) refuses a capped material on a DISCARD element. That
gate is keyed on the *element*, not the scheme, so (a) does not change it — but
it does mean the `int_scheme=2` + `max_substeps` combination that was previously
"warned about and then gated" is now "gated only". Confirm the gate still fires
on `int_scheme=2`; it should, and a test is cheap.

### 4.4 Verification recipe

1. **The corrected predicate.** `2 in _SCHEMES_REACHING_MODIFIED_EULER`;
   `45 not in`; `3, 5` warn for their own (unchanged) reason.
2. **Both warnings, both directions.** `honor_tol_r=True` and
   `max_substeps=100` are each **silent** on `int_scheme` ∈ {0, 1, 2} and each
   **warn** on 45.
3. **Mutation check.** Reverting (a) must red exactly the four cases in (2) that
   mention scheme 2 — and nothing else. If it reds more, the set is doing work
   somewhere unaudited.
4. **Live (`integration_ladruno`), optional, on a build ≥ `049b295fc`.**
   Reproduce the fork's own control rather than the warning text: a generous /
   uncapped `int_scheme=2` run on a `p → p_min` floor path completes its steps;
   the same deck at `max_substeps=100` fails a step first. That checks the seam,
   not the string.

### 4.5 Nothing to do

- Log parsing: apeGmsh has none keyed on this warning, so nothing to un-key. If
  anyone adds one later, `_solver_stats.py` (`:112-121`) is the precedent for
  anchoring on exact labels and never guessing.
- `max_substeps`'s emit rule (`nd.py:1167-1168`), its `>= 0` refusal
  (`nd.py:1060-1067`), and the element gate (`build.py:4086`): unchanged.
- The `-implex` companion rules: unchanged. `-implex` still refuses every scheme
  but 1 and 2 (`LadrunoSANISAND.cpp:814-833`), and still requires
  `-maxSubsteps > 0` on both.

---

## 5. The fork build / version guard

`ops.ladrunoBuild()` is reached in apeGmsh through
[`emitter/live.py:426`](../src/apeGmsh/opensees/emitter/live.py)
(`get_backend_build()`) and surfaces on
[`_target.py:92`](../src/apeGmsh/opensees/_target.py)
(`OpenSeesCapabilities.build`).

| what | minimum `ladrunoBuild()` | enforce? |
|---|---|---|
| the batch as a whole | `48c0e99bc` | — |
| `-pRe` on `LadrunoSANISAND` (#842) | `1133279a5` | **documented, not enforced** — an older parser refuses the unknown flag loudly |
| the scheme-2 cap warning is gone (#845) | `049b295fc` | documentary only — behaviour is identical on either side |
| `ladrunoThreads` / threaded loop (#843) | `48c0e99bc` | **enforce** — see below |
| the `IntScheme 2` verdict (#844) | none | docs only; measurements are against `634824e1f` |

**`ladrunoThreads` is the one that must be enforced**, and not by the hash.
Three reasons stack:

1. **`ops.ladrunoBuild()` is a CONFIGURE-time stamp and lags.** The fork captures
   the hash in `execute_process` at configure time; an incremental
   `build.bat` does not re-configure. #845's own PR body reports the binary
   stamping `634824e1f` — its **parent** commit. It **understates** what was
   built, every time.
2. **Compiled-in ≠ threaded.** A build at the right hash configured *without*
   `-DLADRUNO_OPENMP=ON` — which is every bare `cmake` build, including CI and
   any Linux/Esmeralda build — has the verb and runs serial, warning once.
3. **Threaded ≠ threading your model.** The allowlist is all-or-nothing and, for
   the SANISAND deck anyone would want it for, refuses.

So the probe is **behavioural, layered**: `getattr(ops, "ladrunoThreads", None)`
is not `None` → request 2 and check the binary did **not** print the
`built WITHOUT LADRUNO_OPENMP` warning → check the domain announced `THREADED`
rather than a refusal. That is the same discipline the 2026-09-15 guide reached
for `len(implexRefusals) == 6` and #1149 used for the Tcl `K_t: 1e+06` probe.

---

## 6. Consolidated adoption checklist

**"Code"** = an apeGmsh source change is required; **"Doc"** = a docstring / ADR
/ guide line; **"None"** = verified nothing to do. Size is a rough estimate of
the change, not of the review.

| # | § | Item | Where | Kind | Prio | Size |
|---|---|---|---|---|---|---|
| 1 | 4.3(a) | Add `2` to `_SCHEMES_REACHING_MODIFIED_EULER`; rewrite the comment with the fallback reasoning | `material/nd.py:451-453` (+ `:441-450`) | **Code** | **P0** | XS |
| 2 | 4.3(c) | Flip scheme `2` from the warns-parametrize to the silent one — **four parametrizations across two files** — and fix the "2 and 45 skip ModifiedEuler" comment | `tests/…/primitives/test_materials_nd.py:1339-1356`; `tests/opensees/unit/test_sanisand_deck_gates.py:240-257` | **Code** | **P0** | S |
| 3 | 4.3(b) | Resolve the dead `range(10, 100)` half of the set (drop it, or document it as unreachable mirror of the fork's `s > 9`) | `nd.py:451-453`, `:1016-1020` | **Code** | **P0** | XS |
| 4 | 4.3(d) | Correct the two prose sites that assert scheme 2 never reaches `ModifiedEuler` | `internal_docs/guide_ladruno_sanisand_integrator.md:87-90`; ADR `0103…md:117-127` | Doc | **P0** | S |
| 5 | 4.3(f) | Confirm `validate_sanisand_substep_cap` still fires for `int_scheme=2` + a DISCARD host; add the case | `_internal/build.py:4086` + test | **Code** | P1 | XS |
| 6 | 4.3(e) | `SANISAND_SCHEME2_CAP_MIN_BUILD = "049b295fc"`, documentary | `nd.py` (beside `:1378`/`:1392`/`:1405`) | Doc | P1 | XS |
| 7 | 3.3(a) | `int_scheme` docstring: scheme 2 = material-point **yes**, BVP integrator **no**; + CPPM-cannot-fail and the silently-overwritten `TanType 2` | `nd.py:880-906` | Doc | P1 | S |
| 8 | 3.3(b) | ADR 0103 `Amendment (2026-09-16, fork PRs #844/#845)` — qualifies D1's `tan_type=2` under scheme 2; corrects D4's `ModifiedEuler` claim. **D1's decision does not move** | `0103-sanisand-integrator-deck-contract.md` (after `:145`) | Doc | P1 | S |
| 9 | 3.3(c) | Verdict tables + §2 fallback caveat in the working-memory guide | `internal_docs/guide_ladruno_sanisand_integrator.md:28`, `:75` | Doc | P1 | M |
| 10 | 3.3(e) | Extend the relaxed-rung-count sentence to cover SANISAND integrator comparisons | `analysis/strategy.py:534` (`Ladder` docstring) | Doc | P2 | XS |
| 11 | 1.3(a) | `p_re: float = 0.0` on the primitive **and** on the namespace factory (or the field is unreachable and default-parity reds) + `__post_init__` validation mirroring the parser (raise `< 0`; warn `> 0.1·P_atm`; warn `<= p_min`) | `nd.py:993-996`, `:1036-1047`; `_internal/ns/nd.py:264-297`, `:325-359` | **Code** | P1 | S |
| 12 | 1.3(a) | Emit `["-pRe", p_re]` **only when non-zero**, between `-Presidual` and `-Pmin` | `nd.py:1156-1160` | **Code** | P1 | XS |
| 13 | 1.3(b) | `p_re` docstring: the three claims kept apart (Gauss-point neutrality · path-changing everywhere · measured worse on the surcharged strip) + the explicit-`dt` consequence | `nd.py:803-970` | Doc | P1 | S |
| 14 | 1.3(c) | `SANISAND_PRE_FLOOR_MIN_BUILD = "1133279a5"`, exported and cited | `nd.py:71`, `:1392`-ish | Doc | P1 | XS |
| 15 | 1.3(d) | Confirm the new field round-trips H5 generically; bump `NEUTRAL_SCHEMA_VERSION` **only if** it does not | `fem-broker` schema seam | **Code** | P1 | XS–M |
| 16 | 1.4 | Tests: byte-identity at unset **and** `p_re=0.0`; flag position; the three refusals; namespace reachability + default parity | `tests/…/test_materials_nd.py:1226`, `:1293`, `:940`; `tests/opensees/unit/test_ns_wrapper_default_parity.py` | **Code** | P1 | S |
| 17 | 2.4(a) | `ops.ladruno_threads(n)` live forwarder beside the other fork-only verbs, behind `_stock_build_gate` | `apesees.py:9030-9191`; `emitter/live.py:1103-1156`, gate `:770` | **Code** | P2 | M |
| 18 | 2.4(a) | Client-side pre-refusals: non-fork backend · build below floor · partitioned model · element outside the allowlist · **any `ManzariDafalias`-family material (raise — the fork SEGFAULTS)** | same, + `_element_capabilities.py:815`, the `_material_graph` walk at `_internal/build.py:4063`/`:4086` | **Code** | P2 | M |
| 19 | 2.4(b) | `LADRUNO_THREADS` for the subprocess lanes, and an explicit refusal to set it into an MPI / `run_remote` lane | `apesees.py:11450`, `:11476`; `_run.py:126` | **Code** | P2 | S |
| 20 | 2.4(d) | `has_threaded_update` on `OpenSeesCapabilities`, probed **behaviourally** (verb present **and** no WITHOUT-LADRUNO_OPENMP warning); it is the first capability that is legitimately false on a good fork build | `_target.py:66-101`, `:221`; `apesees.py:8228` | **Code** | P2 | S |
| 20b | 2.4(e) | Add the verb to the published fork-only surface; fix the `APEGMSH_OPENSEES_BIN` resolution order there (it is live-lane only) | `docs/concepts/backend-capabilities.md:75`, `:44`/`:49` | Doc | P2 | S |
| 21 | 2.4(c) | New ADR (number after `0109`): run-option-not-primitive · the refusal roster · default 1 · Linux-OFF · "the deck that would pay is the one that is refused" | `architecture/decisions/` | Doc | P2 | M |
| 22 | 2.5 | Tests: refusal roster (unit), `threads=1` no-op, stock-backend refusal, + a Windows-only live bit-identity case with a run-biased `skipif` | `tests/opensees/unit`, `tests/opensees/integration_ladruno` | **Code** | P2 | M |
| 23 | 5 | The layered behavioural build probe for `ladrunoThreads`; do **not** gate on `ladrunoBuild()` alone | wherever (17) lands | Doc | P2 | XS |
| 24 | — | Log/stderr parsing of the `NO EFFECT` warning | — | **None** — apeGmsh has none | — | — |
| 25 | — | Deck emitters (Tcl/py) for `ladrunoThreads`; H5 field; `StageRecord` | `emitter/tcl.py`, `emitter/py.py` | **None** | — | — |
| 26 | — | `analyze_rc` / `Substep` / `Ladder` — #843 adds no return code | `_internal/analyze_rc.py:42`, `analysis/strategy.py:392` | **None** | — | — |
| 27 | — | Reader / `Results` / `_response_catalog` — no new response or bucket in any of the four PRs | `results/readers/_ladruno_element_io.py` | **None** | — | — |
| 28 | — | `ManzariDafalias` / `SAniSandMS` — `-pRe` is `LadrunoSANISAND`-only | `nd.py:622`, `:783` | **None** | — | — |
| 29 | — | `tan_type` default stays `2`; the always-on 5-positional tail stays | `nd.py:993-997`, `:1140-1147` | **None** | — | — |
| 30 | — | `system Pardiso` thread count (ADR 0106) — a different knob, deliberately not folded in | `_solver_stats.py` | **None** | — | — |

**Blocking order.** Items **1–4 are the only ones that fix a wrong answer
apeGmsh gives today** and should land first, as one small PR; item 2 must move
with item 1 or the suite cannot go green. Items 7–9 depend on nothing and can
ride along. Item 11 gates 12/13/16. Item 15 gates 11 if the schema turns out not
to be generic. Items 17–23 are a separate PR with its own ADR (21) and should
not be mixed with the SANISAND work — different surface, different risk, and the
only one of the four features with a platform asymmetry.

---

## 7. Do not adopt / not yet

**`ladrunoThreads` for SANISAND — never, until the fork locates the crash.**
The fork refuses `ManzariDafalias` / `LadrunoSANISAND` at every `IntScheme`
(`LadrunoSANISAND.cpp:1931`) because the update path **segfaults** on an OpenMP
worker thread — 4/4 at 4 threads — with the root cause **unlocated** after four
excluded hypotheses. apeGmsh must refuse it client-side too (item 18), and must
refuse by **raising**, not warning: a segfault takes the interpreter down and
loses the session, which in a live apeGmsh run means the model as well. Do not
add a "force" or "experimental" escape hatch. A located-but-unfixed hazard is
strictly worse than an un-audited one, because the audit manufactures
confidence — and this one's audit was wrong once already.

**`p_re` as a default — no, and not as a recommendation either.** Default `0`.
The only deck in the fork's own campaign that resembles an apeGmsh user's strip
footing measured it **worse on every axis it was meant to improve** (§1.2(c)),
and the ADR's status is "available, default 0, **not adopted**". Expose the
field; let a user who has a `p' → 0` free-surface point ask for it deliberately,
having read §1.2(a)–(c). Do not wire it into any preset, recipe or example, and
do not let a docstring imply it is a robustness knob.

**`int_scheme=2` as an apeGmsh-suggested option — not in the BVP lane.** It is
qualified only where the strain increment is **given**. apeGmsh's decks are
almost all BVPs under a global Newton, which is exactly the regime the fork
refuted (8/8 stalls, 475× shallower on the bearing leg, 100 % of committed steps
on the relaxed rung). Keep it available — it is already accepted — and keep it
out of guidance except in the material-point paragraph. And do **not** add a
warning against it (§3.3(d)): a warning would be wrong in the case that was
qualified.

**The threaded loop on Linux / Esmeralda — not available at all.** `build.bat`
is the only recipe that turns it on, and gcc + `-fopenmp` segfaults the fork on
an unrelated test. Any apeGmsh doc, example or HPC recipe (`apeGmsh.hpc`,
`run_remote`) that mentions `ladrunoThreads` must say so in the same breath, or a
user will put it in a SLURM script.

**Do not fold `ladrunoThreads` into a general "threads" or "parallel" option.**
The fork's whole design rationale for one verb in one file is that MKL solver
threads × assembly threads × MPI ranks oversubscribe and make every bench lie
(`LadrunoThreads.h`). apeGmsh already has a separate Pardiso-threads surface
(ADR 0106). Keep them separate and named.

---

## 8. Open items handed back to the fork

1. **The SANISAND OpenMP segfault is unlocated.** Four hypotheses excluded by
   experiment; `#pragma omp critical` around the whole `theEle->update()` still
   faults, so it is not a race between element updates. The fork's own next step
   is recorded in ADR-75b §14.1: **the box has no `cdb`/`WinDbg`/`procdump` and
   the Release build emits no PDBs** — get the stack first
   (`/Zi` + `/DEBUG`, then `cdb -g -G -c ".ecxr;kb 40;~*kb 20;q"`). One faulting
   frame ends it. Until then apeGmsh's item-18 refusal is load-bearing.
   *No fork PR open for this at `48c0e99bc`.*

2. **gcc + `-fopenmp` crashes the fork on Linux.** Deterministic, twice on the
   same commit, in the zero-mass `system Diagonal` test, at 1 thread. Needs a
   Linux ASAN/gdb work package; the CMake default flips ON when it is fixed
   (BUILDING.md, BUILD_GOTCHAS §15, ADR-75b §14.4). Until then **CI does not
   exercise the threaded loop** — a green Zone-A implies nothing about it, and
   all 18 cases show `SKIPPED`. *No fork PR open for this at `48c0e99bc`.*

3. **F13's fine-surface leg was not run.** #842's BVP arms were coarsened on
   purpose (`h0 = 1.0` m), and the near-surface Gauss point sits at
   **`p' = 6.25` kPa**, not the `p' ≈ 5` kPa the ADR-93 request aimed at, which
   needs `h0 = 0.5` / `0.25`. The fork's own words: *"The sign of the effect is
   unlikely to flip between 6.25 and 5 kPa, but that is an inference, not a
   measurement."* apeGmsh should not quote the §1.2(c) refutation as covering the
   `p' ≈ 5` kPa regime.

4. **`-implex` + `IntScheme 2` was not run — at all.** `-implex` was **OFF in
   every WP-105 arm**, and scheme 2 is one of the only two schemes `-implex`
   qualifies. So the commit-time companion's own regime under scheme 2 —
   precisely the regime §3.1 says scheme 2 is *good* at — is unmeasured. Also
   unrun: plane strain, `LadrunoUP`, cyclic/reversal paths, parallel, and whether
   the phase-(b) refutation survives `-tangentPredictor`, a displacement-
   controlled push, or a per-iterate cap on the CPPM ladder.

5. **The fork's own apeGmsh emitter guide is now stale on one line.**
   `Ladruno_implementation/86_ladruno_sanisand_apegmsh_emitter_guide.md:149`
   still reads *"**Only emit it on IntScheme 0 or 1.**"* for `-maxSubsteps`.
   #845 makes that false — the cap binds on scheme 2 as well, and
   `schemeReachesModifiedEuler()` now says so. #845 updated
   `LadrunoSANISAND_implex_guide.md` §3 but not this file. Worth one line,
   because this is the document apeGmsh emitter work is supposed to read.

6. **`-pRe` is absent from the fork→apeGmsh contract table.**
   `Ladruno_implementation/ladruno_apegmsh_contract.md` has no `-pRe` row;
   the flag is documented only in `LadrunoSANISAND_implex_guide.md:104` and
   `:190-260`. The contract's "Feature → apeGmsh reference table" (`:41`) is
   where an emitter author looks first.

7. **Still open from the previous batch:** fork
   [#841](https://github.com/nmorabowen/OpenSees/pull/841) (WP-104, draft) —
   `wipe` zeroing `LadrunoSANISAND`'s process-wide IMPL-EX diagnostic ledger,
   filed off the finding #1149's F10 test made (`implexRefusals` counters
   surviving `ops.wipe()` across material instances in one process). Until it
   merges, any apeGmsh assertion on an absolute `implexRefusals` value remains
   process-order-dependent; assert **no growth**, not a literal.

---

## 9. What could not be verified here

- **No apeGmsh code was changed and no apeGmsh test was run.** Every apeGmsh
  claim above is read from the source at `origin/main` `1df2d0dc`, not executed.
  In particular, the "`_SCHEMES_REACHING_MODIFIED_EULER`'s `range(10, 100)` half
  is unreachable" claim in §4.3(b) is read from the interaction of `nd.py:451`
  and `nd.py:1016`, not proven by a probe.
- **No fork binary was built or run for this guide.** Every measurement quoted —
  the 324× / 1.93× / +5.88 % `pRe` numbers, the 1.03/1.03/0.94× thread scaling,
  the 51.23 % loop-A share, the 8-of-8 stalls, the 475× bearing leg, the 40/40 →
  step-18 cap control — is the fork's own, quoted from the PR bodies and the
  merged docs at `48c0e99bc`.
- **The venv's own fork build was not checked against the batch tip.** #1149
  reported it at `634824e1f`, which is the **parent** of all four PRs, so nothing
  in this batch has been exercised live from apeGmsh. Every "live" recipe in §§1–4
  is a recipe, not a result.
- **`ops.ladrunoBuild()` is a CONFIGURE-time stamp and understates the binary**
  (§5). #845's own PR body is the worked example: the binary built from the fixed
  source stamps `634824e1f`. Do not use it alone to prove a binary contains any of
  the above.
- **#842's own "not verified" list stands**: the finer-mesh / `p' ≈ 5` kPa BVP
  legs, the full Zone-A sweep (only the nine SANISAND/Manzari files + the new gate
  ran), a real two-rank MP run, and D5a / II.2 (the `D_factor` sigmoid at
  `p_r = 0`).
- **#843's own "not verified" list stands**: full Zone-A not run;
  ThreadSanitizer not run (it is the tool that would find the §2.2 defect);
  the §3 root cause unlocated; the threaded **failure-reporting** path is
  unreachable with a one-material allowlist whose `setTrialStrain*` overloads all
  `return 0`, so it has never executed at any thread count; no hybrid MPI+threads
  measurement is now possible at all; loops B/C not threaded.
- **#844 is a verdict archived from a report, with two runs
  (`ct_p100_s2_n200_t7`, `ct_p20_s1_n40_t7`) that died with no traceback in the
  first sweep** and re-ran clean in isolation — recorded as non-reproducible and
  **not** claimed as findings. Its measurements are against `634824e1f`, not the
  batch tip.
- **#845 did not run the full Zone-A sweep** — only the SANISAND suites, and its
  negative control is the pre-fix binary rather than a source mutation.
- **The "no stderr/log parser keys on the fork's warnings" claim (§4.2)** is from
  a grep over `src/apeGmsh` for the warning text and over `_run.py` /
  `emitter/live.py` for `opserr` / stderr consumers, plus reading
  `_solver_stats.py:112-121` and `_run.py:78` (`_PROGRESS_RE`, apeGmsh's own
  marker). It is a negative result from a search, which is weaker than a positive
  one: re-run the grep before relying on it. The same caveat covers "no test
  matches `threads` as an engine knob".
- **The H5 round-trip question in §1.3(d) is open, not answered.** Whether a new
  `LadrunoSANISAND` field serialises generically or needs a
  `NEUTRAL_SCHEMA_VERSION` bump was not traced to the writer; #1149's
  `cpl_al_update` needed an explicit bump and is the reason to check rather than
  assume. It is item 15 precisely because it is unresolved here.
