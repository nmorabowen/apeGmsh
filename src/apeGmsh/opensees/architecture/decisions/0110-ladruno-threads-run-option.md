# ADR 0110 — `ladruno_threads`, a fork-only run option with a client-side refusal roster

**Status:** Proposed (2026-09-16).

**Sibling of, and deliberately not merged with, [ADR 0106](0106-pardiso-stats-capture.md)**
— that ADR owns the *solver's* thread count (`system Pardiso`, `MKL_NUM_THREADS`).
This one owns the *element state-determination loop's* thread count. They are two
knobs on two different loops and they multiply; §D2 says why they stay apart.
Unrelated to [ADR 0103](0103-sanisand-integrator-deck-contract.md) and
[ADR 0104](0104-substep-run-to-criterion-controller.md), which are deck contracts
for the material this ADR refuses.

**Fork-side sources** (branch `ladruno`, all read for this ADR): fork PR **#843**
(WP-107), on top of the batch floor `48c0e99bc` —
`SRC/utility/LadrunoThreads.h` (the one knob and its "why this file is the only
knob" rationale), `SRC/domain/domain/Domain.cpp`
(`Domain::ladrunoAnnounceAudit`, `Domain::ladrunoThreadedUpdate`),
`SRC/interpreter/OpenSeesOutputCommands.cpp:1621` (`OPS_LadrunoThreads`) and
`:4746` (`OPS_logFile`), `SRC/interpreter/PythonWrapper.cpp:3378` / `:3575`,
`SRC/element/Element.h:143` and `SRC/material/Material.h:72` (both default
`false`), `SRC/element/ladrunoPlane/LadrunoQuad.cpp:719`,
`SRC/material/nD/ElasticIsotropicPlaneStrain2D.h:96`,
`SRC/material/nD/LadrunoSANISAND.cpp:1931`,
`SRC/material/nD/ElasticIsotropicMaterial.cpp:126`,
`SRC/handler/StandardStream.cpp:52`, `BUILDING.md:76-99`, and
`Ladruno_implementation/75b_ladruno_threaded_assembly_adr.md` §13, §14, §14.0,
§14.1, §14.4. apeGmsh working notes:
`internal_docs/ladruno_adoption_2026-09-16.md` §2, §5, §7, §8, rows 17–23.

## Context

Fork PR #843 threads exactly one loop: the element state-determination loop in
`Domain::update()`, "loop A" in ADR-75b §2. It is the one loop in the fork with no
floating-point reduction into the SOE, so a threaded run is bit-identical to a
serial one *by construction*, not to a tolerance — one full-field md5 across 12
runs at 1/2/4/8 threads. Three further fork facts decide everything below.

**It almost never pays.** On an idle box the measured whole-step speed-up is
1.00× / 1.03× / 1.03× / **0.94×** at 1/2/4/8 threads (ADR-75b §14.0); loop A is
7.80 % of that deck's step, so Amdahl caps the prize at 1.08×. The one deck where
loop A is 51.23 % of the step is the `LadrunoQuad -bbar` + `LadrunoSANISAND`
strip — and that deck is refused. The fork's own sentence is the one to carry:
*the only deck that would pay is the one that cannot run threaded.*

**The allowlist is one element and one material, and it is all-or-nothing.**
`Element::ladrunoThreadSafeUpdate()` returns `false` at `Element.h:143` and
`Material::ladrunoThreadSafeUpdate()` returns `false` at `Material.h:72`; a single
un-opted-in element anywhere in the domain sends the *whole* loop serial with a
warning (`Domain.cpp`, the allowlist sweep in `ladrunoThreadedUpdate`).
`LadrunoQuad::ladrunoThreadSafeUpdate()` (`LadrunoQuad.cpp:719`) refuses `EAS` and
`-geom finite`, then **delegates to every material copy it holds**
(`LadrunoQuad.cpp:730`). The only `NDMaterial` in the fork that answers `true` is
`ElasticIsotropicPlaneStrain2D` (`:96`) — the class
`ElasticIsotropicMaterial::getCopy("PlaneStrain")` hands back
(`ElasticIsotropicMaterial.cpp:126`).

**The refusal that matters is a segfault, not a warning.** The
`ManzariDafalias` / `LadrunoSANISAND` family is refused at every `IntScheme`
(`LadrunoSANISAND.cpp:1931`) because its update path **segfaults on an OpenMP
worker thread**, 4/4 at 4 threads, once the plastic branch is exercised in volume.
`#pragma omp critical` around the whole `theEle->update()` still faults, so it is
not a race between element updates; four hypotheses are excluded by experiment
(ADR-75b §14.1) and the root cause is **not located**. That is the one failure in
this feature that apeGmsh cannot survive: a segfault takes the interpreter down,
and in a live apeGmsh run the interpreter is holding the model.

On top of that there is a platform asymmetry. `LADRUNO_OPENMP` defaults **OFF**
in CMake and only `Ladruno_scripts\build.bat` turns it on, because flipping the
default ON made gcc + `-fopenmp` segfault Zone-A deterministically at **1 thread**
in an unrelated zero-mass `system Diagonal` test (`BUILDING.md:76-99`, ADR-75b
§14.4). So: Windows/MSVC has the threaded loop; Linux, Esmeralda and CI have the
verb and run serial. **CI can never exercise this feature.**

## Decision

### D1 — A run option on the live bridge, not a primitive

`ladrunoThreads` is a process/runtime setting. It does not round-trip H5, it is
not part of the model, and it must not reach a deck. So: a new
`apeSees.ladruno_threads(n)` beside the five existing fork-only live verbs —
`ladruno_projection_tie_force` (`apesees.py:9030`), `ladruno_contact_force`
(`:9134`), `ladruno_contact_info` (`:9163`), `ladruno_mortar_penetration`
(`:9178`), `ladruno_mortar_tie_residual` (`:9191`) — forwarding to a matching
method in the `LiveOpsEmitter` block at `emitter/live.py:1103-1156`.

It does **not** follow those five in one respect: they are post-run *queries* and
share `_require_live_for_contact_query` (`apesees.py:9121`), which demands a prior
live `analyze`. `ladruno_threads` is a pre-run *setting* and has no such
requirement — see D3 for what it does require instead.

No `StageRecord` field, no H5 field, no `NEUTRAL_SCHEMA_VERSION` bump, no token in
`emitter/tcl.py` or `emitter/py.py` — the same line ADR 0106 D1 drew for solver
stats, for the same reason: a runtime setting inside a declaration breaks H5
replay.

### D2 — Default 1, never implicit, never auto-tuned, never merged with the MKL knob

The default is 1 and nothing in apeGmsh ever raises it on the user's behalf. No
"use all cores", no core-count heuristic, no preset, no recipe, no example. On
the only deck that can run threaded, 8 threads is *slower than serial*. Every
docstring that exposes this knob carries the fork's sentence verbatim — *the only
deck that would pay is the one that cannot run threaded* — because a user who
reads only the parameter name will otherwise reasonably assume this is free speed.

It is kept strictly separate from the Pardiso/MKL thread count
(`analysis/system.py:176`, `:248-250`; `_internal/ns/analysis.py:292`;
`docs/guides/nonlinear_concrete_solver.md:135`), which ADR 0106 owns. The fork's
rationale for one count in one file behind one verb is that solver threads ×
assembly threads × MPI ranks oversubscribe and make every bench lie
(`LadrunoThreads.h`); #843's measurements were taken at `MKL_NUM_THREADS=1`
precisely so the two would not multiply. A combined `threads=` option would
recreate the exact confusion both ADRs exist to prevent.

### D3 — A client-side pre-refusal roster for `n > 1`, and it raises

`n == 1` is a no-op: it is the default, the fork takes the byte-identical serial
path at 1, and asking for it must **not** require a fork build. Everything below
applies only to `n > 1`.

Five refusals, all raising `RuntimeError` / `ValueError` before any fork process
is touched:

1. **Stock backend.** Through `LiveOpsEmitter._stock_build_gate`
   (`emitter/live.py:770`, testing `hasattr(self._ops, "criticalTimeStep")` at
   `:785`) with a new message constant `_THREADS_FORK_REQUIRED` beside its
   neighbours at `live.py:41-216`, naming `APEGMSH_OPENSEES_BIN` the way
   `_AUGMENT_FORK_REQUIRED` (`:41`) does. This is the right gate and not
   `_element_fork_gated` (`:747`): that one is post-hoc, it creates the element
   and then checks `getEleTags`, and here there is nothing to read back.
2. **Verb absent.** `getattr(self._ops, "ladrunoThreads", None) is None` on a
   build predating `48c0e99bc`. Raise naming the min-build constant; do not
   silently no-op. A silently-serial "threaded" run is exactly how a bench lies.
3. **Partitioned model.** Not via `_in_partition` — that flag is set by
   `partition_open` (`live.py:1313`) and cleared by `partition_close` (`:1343`),
   it exists only *during* a per-rank emission block, and `_stock_build_gate`
   deliberately no-ops while it is set. The declaration-side answer is
   `is_partitioned(self.fem)` (`_internal/build.py:7598`), which
   `apesees.py:1267` already uses to pick the emit path.
4. **Any element outside the allowlist** (D4).
5. **Any `ManzariDafalias`-family material anywhere in the material graph** —
   which D4 folds into the same allowlist sweep, because the correct rule is
   stronger than "not SANISAND".

**(5) raises rather than warns, and there is no escape hatch.** No `force=`, no
`experimental=`, no environment override. The fork's failure mode is a segfault
with the root cause unlocated after a bounded hunt; a located-but-unfixed hazard
is worse than an un-audited one because the audit manufactures confidence, and
this audit was wrong once already — the first sweep returned 12/12 bit-identical
before the harder load path faulted 4/4.

**When the model-dependent checks run: at the call, against the declaration.**
`ladruno_threads(n)` with `n > 1` calls `self.build()` and walks
`[p for p in bm.primitives if isinstance(p, Element)]` — the same expression
`apesees.py:11682` already uses, and the same list
`validate_sanisand_substep_cap` receives at `apesees.py:1348`. This is the simple
correct hook and it does not care whether elements have been emitted live yet,
because the *declaration* is the authority, not the live domain: the fork
re-audits on every `Domain::update()` anyway, so a late `ops.element(...)` would
flip the fork's own verdict regardless of what apeGmsh checked. Deferring the
check into `BuiltModel.emit()` beside the other validators was rejected: those
run on every emit target, and this is a live-lane-only runtime setting that a
`tcl()` deck must not be gated on.

### D4 — The allowlist lives on `_ElemSpec`, and apeGmsh mirrors, never extends

A new tri-state slot on `_ElemSpec` (`_element_capabilities.py:66`),
`thread_safe_update: bool | None = None`, with an accessor
`element_thread_safe_update(class_name)` alongside
`element_propagates_material_refusal` (`:815`). The convention is already in the
file and already load-bearing: `True` measured safe, `False` measured unsafe,
`None` unmeasured. Here, unlike the refusal-propagation gate, **`None` refuses** —
the fork's base class refuses too, so "unmeasured" and "refused" are the same
answer on the engine, and mirroring that is the only way the two stay in step.

Exactly one entry is `True`: `"LadrunoQuad"` (`_element_capabilities.py:520`).

The guide's three-way condition — *"LadrunoQuad std/bbar/ssp under `-geom linear`,
not `-eas`"* — **over-specifies the apeGmsh side.** `LadrunoQuad`
(`element/solid.py:1172`) has `formulation in ("std", "bbar", "ssp")`, raises on
`"eas"` in `__post_init__`, and has no `geom` axis at all (only `LadrunoBrick`
has `-geom`). The element half therefore reduces to "the element is a
`LadrunoQuad`".

The **material** half is where the guide is too weak, and this is the one place
this ADR departs from it. `LadrunoQuad.cpp:730` asks each material copy, and the
only `NDMaterial` answering `true` is `ElasticIsotropicPlaneStrain2D`
(`ElasticIsotropicPlaneStrain2D.h:96`), reached only through
`getCopy("PlaneStrain")` (`ElasticIsotropicMaterial.cpp:126`). Refusing only the
`ManzariDafalias` family would let `J2Plasticity`, `DruckerPrager`, `LadrunoJ2`,
`ASDConcrete3D`, every wrapper in `material/nd.py`, and even `ElasticIsotropic`
itself under `plane_type="PlaneStress"` past the client gate — all of which the
fork silently runs serial. So the client-side material rule is an **allowlist of
one**: the element's `material` must be `ElasticIsotropic`
(`material/nd.py:90`) and its `plane_type` must be `"PlaneStrain"`
(`element/solid.py:1231`).

The walk is `_material_graph(prim)` (`_internal/build.py:4063`), the transitive
`.dependencies()` sweep `validate_sanisand_substep_cap` (`:4086`) already uses —
so a `PlaneStrain(base=LadrunoSANISAND(...))` or `LogStrain(...)` wrapper cannot
hide the real constitutive model one level down. Any graph that is not exactly
one bare `ElasticIsotropic` is refused, and the message names the offending
material class.

**The maintenance rule.** `Element.h:143` and `Material.h:72` are the source of
truth. apeGmsh mirrors the fork's opt-ins and **never adds one of its own**. When
the fork allowlists a second class, this table gains a row in the same PR that
adopts it; until then, a `None` here is a refusal, not an invitation to guess.

### D5 — `has_threaded_update` is behavioural, opt-in, and the observation channel is `ops.logFile`

`OpenSeesCapabilities` (`_target.py:66-101`) gains
`has_threaded_update: bool | None = None`. It must not derive from
`hasattr(ops, "profiler")` the way `has_fork`, `has_profiler` and
`has_ladruno_up` all do today (`probe_live_capabilities()`, `:196-212`), and it
must not derive from `ops.ladrunoBuild()` either — that stamp is captured by CMake
at **configure** time and lags every incremental `build.bat`, understating the
binary each time (guide §5).

**The observability question, answered from the fork source: the return value
does not reveal it.** `OPS_LadrunoThreads`
(`OpenSeesOutputCommands.cpp:1621-1650`) writes the WITHOUT-OpenMP notice to
`opserr` only —

> `WARNING ladrunoThreads: this binary was built WITHOUT LADRUNO_OPENMP, so the
> element loop stays SERIAL no matter what is requested. Rebuild with
> -DLADRUNO_OPENMP=ON (Ladruno_scripts\build.bat does this by default).`

— and then returns `ladruno_getNumThreads()` through `OPS_SetIntOutput`,
**identically on both builds**. `ladruno_setNumThreads` clamps only on `< 1` and
on hardware concurrency (`LadrunoThreads.h`), neither of which depends on
`LADRUNO_OPENMP`. So a Python caller sees `2` either way.

`opserr` is a C++ `OPS_Stream` writing to `std::cerr`; `contextlib.redirect_stderr`
cannot see it. The mechanism that can is the interpreter's own `logFile` verb:
`OPS_logFile` (`OpenSeesOutputCommands.cpp:4746`), bound as `ops.logFile`
(`PythonWrapper.cpp:3575`), calls `opserr.setFile(name, mode, echo)`, and
`StandardStream::setFile` (`SRC/handler/StandardStream.cpp:52`) opens the file and
sets `echoApplication = echo` — default `true`, so the console still receives
everything and the file is a parseable tee. ADR 0106 D3 already source-verified
this exact route and priced it: `opserr.setFile` is **process-global, one-way
(no verb closes it) and OVERWRITE by default**, which is why that ADR deferred the
live lane behind its own opt-in.

So the probe is layered and **explicitly opt-in**, never run by the free
`capabilities()` call at `apesees.py:8228`:

1. `getattr(ops, "ladrunoThreads", None)` is not `None`, else `False`;
2. read the current count with the no-arg query form, `ops.ladrunoThreads()`;
3. `ops.logFile(<probe file>)` — echo left on (D7);
4. `ops.ladrunoThreads(2)`; if it returns `1`, the box clamped to hardware
   concurrency and the probe is **inconclusive** (`None`), not `False` — a
   single-core box never trips the `stored > 1` branch that prints the warning;
5. otherwise `True` iff `built WITHOUT LADRUNO_OPENMP` is absent from the file;
6. restore the count the query in (2) returned.

`None` is therefore "not probed / inconclusive", which is why the field is
tri-state rather than the plain `bool = False` the guide suggests: a default
`False` on an unprobed capability is a claim, and this one costs a durable side
effect to earn.

**This is the first engine capability that is legitimately `False` on a correctly
built fork binary** — every Linux and every CI fork build lands there. Nothing in
apeGmsh may read `has_fork and not has_threaded_update` as a broken install.

### D6 — A narrow `ladruno_threads=` kwarg on the subprocess run paths, and scrub it out of MPI lanes

`LADRUNO_THREADS` is read once per process (`LadrunoThreads.h`), so it must be in
the child's environment at exec. `apesees.py:11450` already builds
`child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}` and hands it to
`stream_run(..., env=child_env)` (`:11457`; `_run.py:126`, `Popen` at `:173`,
`env=env` at `:181`). There is no caller-facing `env=` on `tcl()`
(`apesees.py:10733`), `py()` (`:11370`) or `run_remote` (`:11476`).

**Recommended: a narrow `ladruno_threads: int | None = None` kwarg on `tcl()` and
`py()`, folded into that one dict.** A general `env=` passthrough was considered
and rejected on simplicity grounds: it is a new public surface with an unbounded
contract (merge or replace? who wins against `os.environ`? does it reach
`run_remote`'s SLURM script? what happens when a caller sets `OPENSEES_BIN`
through it?) bought for exactly one knob; a second env knob is the moment to
generalise. Note also that these lanes resolve their binary through
`OPENSEES_BIN` (`_target.py:154`) and `OPENSEES_VENV` (`:182`), **not**
`APEGMSH_OPENSEES_BIN`, which is live-lane only — the two lanes already disagree
about environment and this knob must not deepen that (D8).

**MPI, `run_remote` and partitioned decks: never set it, and SCRUB an inherited
value.** `run_remote` renders its SLURM script in `hpc/_cluster.py:45-95`, where
`sbatch` propagates the submission environment wholesale (`--export=ALL`, noted in
the comment at `:80`) and `mpiexec`/`srun` then hands it to every rank. The fork
fences this at the binary (`ladruno_parallelBuild()` refuses first and widest in
`Domain::ladrunoThreadedUpdate`), but apeGmsh must not be the thing that set it.

Scrub, not refuse: pop `LADRUNO_THREADS` from the child environment for the
`run_remote` / `per_rank` / partitioned lanes and warn once. An inherited value is
almost always a leftover shell export from a desktop run, not a deliberate act on
this submission, and turning that into a hard failure of an unrelated cluster job
is a worse trade than dropping it loudly. Passing `ladruno_threads=` explicitly
*into* one of those lanes is a different matter and **raises** — that one is
deliberate and wrong.

### D7 — Do not swallow the announcement

The fork prints once per domain per outcome, keyed on
`(element generation, outcome, tag, thread count)`
(`Domain::ladrunoAnnounceAudit`), so a steady state is silent and a Newton
iteration prints nothing — but a refusal and a successful threaded run each print
exactly once, and they are the only way to tell the two apart. The fork's own
comment names apeGmsh as the reason that gate is per-domain rather than
per-process: *"A second model in the same interpreter — pytest, apeGmsh, any
in-process parameter study — must say what it did too."*

Concretely: `ladruno_threads` must never call `ops.logFile(..., "-noEcho")`, and
the D5 probe must leave `echo` at its default `true`. The subprocess lanes are
already safe — `stream_run` merges the child's stderr into stdout
(`_run.py:175-176`) and tees every line to the log — which makes the subprocess
lane the *more* honest of the two here.

### D8 — The docs page gains a verbs entry, and loses a wrong sentence

`docs/concepts/backend-capabilities.md:75` (`## The fork-only surface`) lists
fork-only elements and integrators and no verbs at all. It gains a `### Verbs`
section naming the six live fork-only verbs, with `ladruno_threads` carrying its
Windows-only caveat **in the same breath** — a reader who meets the name without
the caveat will put it in a SLURM script. That section is hand-written and must
sit **outside** the `<!-- capability-map:… -->` markers, which
`tests/test_capability_map_drift.py:59` diffs against a registry that has no verb
table to diff against.

While editing: `:44` and `:49` state the `APEGMSH_OPENSEES_BIN` resolution order
as general. It is **live-lane only**. The subprocess lanes resolve through
`OPENSEES_BIN` (`_target.py:154`) and `OPENSEES_VENV` (`:182`), and a
thread-count knob that lands in the subprocess environment is exactly the feature
that would trip over the blur.

## What was measured

**Nothing, from apeGmsh.** Every number below is the fork's own, quoted from
ADR-75b §14 and PR #843, and no apeGmsh code was written or run for this ADR. The
dev venv's engine is `634824e1f`, which predates `48c0e99bc`:
`hasattr(ops, "ladrunoThreads")` is `False` there, so not one line of this
proposal can be exercised live today.

The fork's numbers, with their conditions:

- **Speed-up (ADR-75b §14.0):** idle box, 3 repeats, elastic deck, 14 400
  `LadrunoQuad -bbar`, 12 steps, `system Pardiso`, `MKL_NUM_THREADS=1` —
  1.00× / 1.03× / 1.03× / **0.94×** at 1/2/4/8 threads, one full-field md5 across
  all twelve runs. Loop A is 7.80 % of that step, Amdahl ceiling 1.08×. An
  earlier 1.00/1.09/**1.11**× table, taken on a contended box, is **withdrawn** by
  the fork along with its "lower bounds" caption.
- **Where the time actually is:** loop A is **51.23 %** of the step on the
  `LadrunoQuad -bbar` + `LadrunoSANISAND` 6 400-element deck against **7.80 %** on
  the same element with `ElasticIsotropic`. The 51 % deck is refused.
- **The refusal that raises:** `ManzariDafalias` IntScheme 1 + `LadrunoQuad -bbar`,
  6 400 elements, 4 threads, `ds = 0.02` — **segfault 4/4**, after an earlier
  12/12 bit-identical sweep on a gentler load path.
- **The Linux defect:** gcc + `-fopenmp`, `test_adr30_projection_p0.py::`
  `test_massless_dof_is_not_policeable_by_the_soe_layer`, exit 139, deterministic
  twice on one commit, at **1 thread**, not reproducible on MSVC.

Not verified, carried forward from #843's own list (guide §9): ThreadSanitizer was
**not** run — it is the tool that would find the class of defect the allowlist
exists to contain; the threaded **failure-reporting** path (lowest-serial-index
`critical`, the extra diagnostic line, step-cut parity) has **never executed at
any thread count**, because the sole allowlisted material's `setTrialStrain*`
overloads all `return 0`, so it is answered by reading rather than by experiment;
the SANISAND root cause is unlocated after four excluded hypotheses; the full
Zone-A sweep was not run and CI shows all 18 `zone_a` cases `SKIPPED`; and no
hybrid MPI+threads measurement is possible at all, with loops B/C not threaded.

### Test plan

Mapping guide §2.5's four recipes onto files. Everything in (1)–(3) runs on CI
against a fake `ops` object, because the refusals are decided from the declaration
and the backend name before any fork call:

1. **Refusal roster, unit** — `tests/opensees/unit/test_ladruno_threads_gates.py`.
   `ladruno_threads(4)` raises on: a `LadrunoSANISAND` (bare and wrapped in
   `PlaneStrain` / `LogStrain`, which is what the `_material_graph` walk is for);
   any element that is not `LadrunoQuad`; a `LadrunoQuad` carrying anything but a
   bare `ElasticIsotropic`; a `LadrunoQuad` at `plane_type="PlaneStress"`; a
   partitioned `fem`. Each message names the offending class. CI: **yes.**
   (`LadrunoQuad(formulation="eas")` needs no case here — `element/solid.py`'s
   `__post_init__` already refuses it at construction, which is the D4 reduction.)
2. **`threads=1` is a no-op** — same file. `ladruno_threads(1)` on a *stock*
   backend does not raise, produces no live call, and leaves the emitted deck
   byte-identical. CI: **yes**, and this is the case that pins "the default costs
   nothing".
3. **Stock-backend refusal** — same file, `get_backend_name() == "stock-openseespy"`
   and `n > 1` raises with the `APEGMSH_OPENSEES_BIN` message. CI: **yes.**
4. **Live bit-identity, Windows only** —
   `tests/opensees/integration_ladruno/test_ladruno_threads_live.py`. On a
   `build.bat` binary at or above `48c0e99bc`: a 2×2 `LadrunoQuad` +
   `ElasticIsotropic` (`plane_type="PlaneStrain"`) model with ≥ 2 elements
   announces `THREADED on 2 threads` at `n=2`, announces nothing at `n=1`, and the
   full nodal field is identical between the two. `skipif` **biased to run** on
   every ambiguous outcome, exactly as the fork's own gate is — an apeGmsh test
   that skips silently on the office Linux box is a test that will never fail.
   CI: **no, and it never will be** (D5). The skip reason must name the running
   build.

The D5 probe needs its own case: mutating the probe's marker string must turn it
red, the ADR 0108 D4 discipline.

## Consequences

- No public signature changes to any existing method, so `_api_index.json` does
  not move: `apeSees` gains one method, `OpenSeesCapabilities` one field with a
  default, `_ElemSpec` one optional field whose `None` default preserves every
  existing entry's behaviour.
- **No golden deck moves.** Nothing is emitted — `emitter/tcl.py` and
  `emitter/py.py` are untouched, `_internal/analyze_rc.py` gains no return code
  (#843 adds none), the readers gain no bucket, `NEUTRAL_SCHEMA_VERSION` does not
  bump.
- `capabilities()` is unchanged by default — `has_threaded_update` reads `None`
  until someone asks for the probe, which is the only way to keep that call free
  of `ops.logFile`'s process-global side effect.
- The live lane sets this in-process, the subprocess lanes set it through the
  environment at exec, and the two therefore behave differently by construction.
  D8's docs fix is the mitigation, not a cure.
- Three claims in `internal_docs/ladruno_adoption_2026-09-16.md` §2.4 were checked
  against HEAD and do not hold: the `_stock_build_gate` `hasattr` test is at
  `live.py:785`, not `:789`; the `partition_open`/`partition_close` brackets are at
  `:1313`/`:1343`, not `:444-460`, and are the wrong seam for a pre-refusal anyway
  (D3); and the element condition "std/bbar/ssp under `-geom linear`" has no
  apeGmsh counterpart (D4). The material rule in §2.4(a)(3) is **too weak**, not
  merely imprecise — see D4.

## Open questions for the owner

1. **D6, kwarg shape.** A narrow `ladruno_threads: int | None = None` on `tcl()`
   and `py()` (recommended), or a general `env: dict[str, str] | None = None`
   passthrough on the same two methods? The narrow one is the simplicity-first
   answer for one knob; the general one is the answer if a second environment knob
   is already foreseen.
2. **D6, scrub versus refuse.** For `run_remote` / `per_rank` / partitioned lanes,
   silently-but-loudly **scrub** an inherited `LADRUNO_THREADS` from the child
   environment and warn once (recommended), or **refuse** the run outright? Scrub
   trades a rare surprise ("my export was ignored") against never failing an
   unrelated cluster submission over a stale shell variable.
3. **D3, timing of the model-dependent checks.** Run them at the
   `ladruno_threads(n)` call against `self.build()` (recommended), or defer them
   into `BuiltModel.emit()` beside `validate_sanisand_substep_cap`
   (`apesees.py:1348`)? Deferring reuses an existing validator seam but gates
   every emit target on a live-lane-only setting.
4. **D5, tri-state or plain bool.** `has_threaded_update: bool | None = None`
   (recommended — `None` means "not probed", and probing costs a process-global
   `opserr` redirect) or the guide's `bool = False`? The plain bool is simpler to
   consume and is a claim the free `capabilities()` call cannot honestly make.
5. **Whether to ship this at all yet.** Every D-item above is real work for a
   feature whose measured best case is **1.03×**, whose only paying deck is
   refused by a segfault nobody has located, which cannot run on Linux or in CI,
   and whose largest single artefact is a refusal roster. The honest alternative
   is to **defer entirely**: adopt nothing, leave the ADR at Proposed, and revisit
   when the fork closes guide §8 item 1 (the SANISAND stack) or item 2 (the gcc
   `-fopenmp` crash). The cost of deferring is real but bounded — a user on a
   `build.bat` binary can already set `LADRUNO_THREADS` in the shell and get the
   fork's own refusal warnings, unguarded and after the run has started, which on
   a SANISAND deck means a segfault instead of a Python exception. That is the
   trade: D3's roster is worth shipping only if someone is going to reach for this
   knob before the fork fixes item 1. The owner is better placed than this ADR to
   say whether anyone will.
