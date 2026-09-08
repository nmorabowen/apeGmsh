# ADR 0106 — Capturing `system Pardiso -stats`: a run-side solver-stats record, per stage

**Status:** Proposed (2026-09-07). TIMs slice **A8-parse** — the reading
half of the profiler work whose emitting half shipped as A8 (#1106).
No code in this ADR; it is the plan the slices in §Slices execute.

**Depends on:** fork PR #821, build ≥ `a240b9183`
(`TIMS_FORK_BATCH_MIN_BUILD` in [`_target.py:113`](../../_target.py)).
Below that build the flag prints a different, once-per-pattern format
and nothing here fires.

**Extends:** the `Pardiso(stats=True)` flag already emitted at
[`analysis/system.py:285-299`](../../analysis/system.py), and the
per-stage profiler bracket of TIMs A8
([`_internal/build.py:1570`](../../_internal/build.py),
[`apesees.py:2928`](../../apesees.py)).

Nothing in [`_DEFERRED.md`](../_DEFERRED.md) covers solver statistics,
stderr capture, or run-side measurement records — this is the first
decision in that space.

## Context

`ops.system.Pardiso(stats=True)` has emitted `-stats` since the flag
existed, and after fork PR #821 the fork answers it with a block on
**stderr**, after **every numeric factorisation** — phase 22,
refactorisations included, so once per Newton step under a fresh tangent
and once per stage under a constant one:

```
PARDISO stats: n=<n> nnz(A)=<nnz> matrixType=<mtype> threads=<nthreads>
  factor entries iparm(18)  = <nnz in L+U>
  peak memory KB iparm(15)  = <peak during symbolic>
  perm memory KB iparm(16)  = <permanent>
  fact memory KB iparm(17)  = <numerical factorization + solve>
  factor Mflops  iparm(19)  = <mflops>
```

The labels are exact and pinned by the fork's own
`tests/test_pardiso_stats.py`; a 54-DOF brick gives
`factor entries iparm(18) = 1836`.

These five numbers are the only answer this project has to the question
every large desktop model eventually asks — *does this fit, and how far
from the edge am I?* The three memory lines are the capacity; `iparm(18)`
is the fill the ordering actually produced, which is the number that
predicts the next refinement's cost. On the fork's desktop targets
PARDISO is the whole portfolio: the serial `MumpsSolver` is **never
compiled**, so `system Mumps` on `OpenSees.exe` or the desktop `.pyd`
answers *unknown system type*, and MUMPS statistics exist only on
`OpenSeesMP` rank 0. Desktop capacity measurement is PARDISO measurement.

Today apeGmsh writes the flag and then throws the answer away. There is
exactly one place output is captured — `stream_run`
([`_run.py:104`](../../_run.py)) — and it runs the deck with
`stdout=PIPE, stderr=STDOUT` (`:139-140`), tees every line verbatim to
`<deck>.log` (`:148-151`), and parses precisely two things out of the
stream: apeGmsh's own `APEGMSH_PROGRESS` marker (`_PROGRESS_RE`, `:56`)
and a warning tally (`_WARN_RE`, `:59`). The blocks land in the log and
stop there. `stream_run` **returns `None`**, and so do
`apeSees.tcl(run=True)` and `apeSees.py(run=True)`
([`apesees.py:10467`](../../apesees.py), `:10975`) — there is no run
result object anywhere in the bridge to hang a measurement on. That is
the shape of the gap: not "we parse it badly", but "there is nowhere for
a parsed number to go".

The stream also has **no stage boundaries**. A staged run emits
`# === Stage: <name> ===` as a Tcl *comment* ([`tcl.py:1340`](../../emitter/tcl.py))
— invisible at runtime — and the stage name reaches the runtime stream
only inside the fail-loud banner of a *failed* increment
([`tcl.py:894`](../../emitter/tcl.py)), which by construction never
prints on a healthy run. A flat merged stream of factorisation blocks
with no way to say which stage paid for them is a worse artifact than it
sounds: gravity, consolidation and push have different tangents,
different constraint sets and different fill, and the whole-run maximum
tells you only that *something* was expensive.

The A8 profiler slice is deliberately no precedent for where the answer
goes. `ProfileRecord` is a frozen, emission-side declaration; the fork
writes its own HDF5 and apeGmsh ships no reader for it
([`profiler.py:66`](../../../profiler.py) forwards to the fork's
out-of-tree viewer). H5 archival refuses a stage carrying one outright
([`h5.py:3428-3440`](../../emitter/h5.py)) on the correct grounds that
runtime telemetry has no model-definition store. Everything below keeps
that line: a measurement is not a declaration, and it never enters an
archive.

## Decision

### D1 — Parsed stats are a run-side record, produced by one pure parser with two entry points

A new module `opensees/_solver_stats.py` owns three frozen `slots`
dataclasses and one function:

```python
SolverStatsBlock   # one factorisation: n, nnz_a, matrix_type, threads,
                   # factor_entries, peak_memory_kb, perm_memory_kb,
                   # fact_memory_kb, mflops
StageSolverStats   # one stage's reduction (D6)
RunSolverStats     # stages: tuple[StageSolverStats, ...]  +  the run-level
                   # bucket + factorisations + malformed_blocks
parse_solver_stats(lines: Iterable[str]) -> RunSolverStats
```

The parser is pure, stdlib-only, and knows nothing about subprocesses.
`stream_run` feeds it the stream line by line as it tees, and returns
the finished `RunSolverStats`; `apeSees.tcl()` / `apeSees.py()` hand it
back to the caller (`RunSolverStats | None`, `None` when `-stats` was
never requested). The same function also takes a **log path**, so a run
that died can be re-parsed off the `<deck>.log` that was tee'd before
the `RuntimeError` was raised — and the interesting factorisation is
very often the one that killed the run. One parser, one code path, two
ways in; no exception-attribute smuggling to make the failure path work.

Changing `apeSees.tcl` / `apeSees.py` from `-> None` to
`-> RunSolverStats | None` is additive for every existing caller, but it
**stales `studio/_api_index.json`** and two CI lanes with it — the
slice that lands it rebuilds the index in the same PR.

**Rejected — mutate a build-side record.** Hanging the numbers on
`StageRecord` or `ProfileRecord` would make a *build* carry the result
of a *run*: the same `BuiltModel` emitted twice would differ, H5 replay
would have to decide what to do with a measurement, and the `h5.py:3428`
refusal would need a second, weaker sibling. `ProfileRecord` is frozen
for exactly this reason. It also walks straight into the stub trap —
`tests/opensees/h5` builds stage records as `SimpleNamespace`, so every
new `StageRecord` field is a landmine (`getattr(rec, "profile", None)`
at `h5.py:3428` is the scar). **No new `StageRecord` field is added by
this ADR.**

**Rejected as the primary artifact — a `<deck>.solver_stats.json`
sidecar.** It is the right shape for the consumers that cannot receive a
return value (a batch HPC job, the studio habitat), and it is named in
the runway for exactly them. As *the* answer it is worse: it makes the
in-process caller round-trip through the filesystem to read a number the
same process just watched go by.

### D2 — Stage attribution rides one new marker line, gated on the deck actually asking for stats

The tcl and py emitters gain a runtime stage marker beside the existing
`APEGMSH_PROGRESS` one, emitted from `stage_open` / `stage_close`:

```tcl
puts "APEGMSH_STAGE open <name>"    ; flush stdout
puts "APEGMSH_STAGE close <name>"   ; flush stdout
```

```python
print("APEGMSH_STAGE open <name>", flush=True)
```

Name last on the line so a stage name containing spaces survives
(`re.compile(r"APEGMSH_STAGE (open|close) (.+)$")`); quoting is
normalised the way `tcl.py:913` already normalises a strategy name.
Both markers, not just `open` — a `close` marker is two lines per stage
and it keeps a post-stage factorisation (an `eigen`, a
`stage_close`-adjacent solve) out of the stage that happened to precede
it.

**The gate is the deck's own stats request, not the `progress` flag.**
`progress=` defaults to `True` ([`apesees.py:10280`](../../apesees.py)),
so hanging the marker on it would move the bytes of essentially every
deck this project has ever emitted. Instead a build-level predicate —
`deck_requests_solver_stats(...)`, resolving the flat/staged/partitioned
system declarations the way
[`validate_ladruno_up_solver`](../../_internal/build.py) already resolves
them — sets `emitter._emit_stage_markers`. A deck with no
`Pardiso(stats=True)` anywhere emits byte-identically to today (INV-1).
The same predicate answers D4's "was a block expected?".

**Rejected — key on the fork's `profiler report <stage>.h5` echo.** It
ties stats attribution to a *different* opt-in feature: a stage would
have to declare `s.profile(...)` to get its solver numbers labelled, and
the echo's exact text is the fork's to change.

**Rejected — no attribution, whole-run maximum only.** It is the cheap
option and it answers the wrong question; see §Context.

**Honest limit.** `stderr=STDOUT` merges two pipes at the OS level, so an
opserr block and a stdout marker are ordered only approximately at the
merge point, and a block can be interleaved mid-line under load. This is
survivable because of what D6 reduces to: a single block misattributed
at a stage boundary shifts a maximum between two adjacent stages; it
cannot corrupt a value. A mid-line interleave produces a malformed block,
which D4 counts rather than believes.

### D3 — Subprocess lanes only; the live lane is deferred, with its mechanism recorded

The tcl and py subprocess lanes go through `stream_run`, which already
owns the pipe, the tee and the line loop. That is the whole
implementation surface, and it is where this ships.

The in-process live lane ([`emitter/live.py`](../../emitter/live.py))
captures nothing: opserr goes to the C-level stderr fd. **It is cheaper
than an `os.dup2` dance, and still not free.** Source-verified on the
fork: the interpreter exposes `logFile` in both wrappers
(`SRC/interpreter/PythonWrapper.cpp:3567`,
`SRC/interpreter/TclWrapper.cpp:2152` → `OPS_logFile()` at
`SRC/interpreter/OpenSeesOutputCommands.cpp:4683`), and it calls
`opserr.setFile(filename, mode, echo)` — so `ops.logFile(path)` routes
opserr into a file that `parse_solver_stats(path)` can read afterwards,
with `echo` defaulting to `true` so the console is not silenced
(`StandardStream::setFile`, `SRC/handler/StandardStream.cpp:52-75`).

It is deferred anyway, on two properties of that same source. The
redirect is **process-global and one-way**: `setFile` closes any open
file and opens the new one, and no interpreter verb turns file logging
back off — the only "undo" is pointing it at another path, and
`OVERWRITE` is the default mode, so a careless second call truncates the
first file. apeGmsh silently redirecting a user's diagnostics into a file
of its choosing, for the rest of their session, is a side effect that
needs its own opt-in knob and its own conversation. The value is
identical in both lanes and the subprocess lane already has a log file,
so shipping there first costs nothing and settles the record shape the
live knob will reuse.

### D4 — The parser recognises exactly the #821 block, counts what it cannot parse, and warns when nothing appears

Two patterns, anchored on the fork's exact labels:

```python
_HEAD = re.compile(
    r"PARDISO stats:\s+n=(\d+)\s+nnz\(A\)=(\d+)\s+"
    r"matrixType=(-?\d+)\s+threads=(\d+)"
)
_FIELD = re.compile(
    r"^\s+(factor entries iparm\(18\)|peak memory KB iparm\(15\)|"
    r"perm memory KB iparm\(16\)|fact memory KB iparm\(17\)|"
    r"factor Mflops\s+iparm\(19\))\s*=\s*(-?[\d.eE+-]+)\s*$"
)
```

Whitespace runs are tolerated (the fork aligns the `=` column); the label
tokens are not. A header opens a block; a block is complete when all five
fields have been seen; anything else — a header interrupted by a second
header, a field before any header, a missing field, a non-numeric value —
increments `malformed_blocks` and produces no `SolverStatsBlock`.

**The parser never raises.** It is reading a stream that carries every
other thing the solver prints, and killing a healthy 4-hour solve over a
mangled telemetry line would be the wrong trade in every direction. The
mutation check in §Slices asserts the *counter*, which is a louder signal
than an exception nobody catches.

**The old once-per-pattern format is ignored, deliberately.** Its lines
match neither pattern, so they are invisible. apeGmsh does not attempt to
parse it: we do not have its text pinned by a test, and guessing at a
format to extract capacity numbers from is how a silently wrong "your
model fits" gets printed.

**A requested block that never appears warns once.** When
`deck_requests_solver_stats(...)` was true and `factorisations == 0`
after the run, `stream_run` emits one `SolverStatsWarning`:

> `system Pardiso -stats` was emitted but no `PARDISO stats:` block
> appeared in the run output. Fork builds before `a240b9183`
> (`TIMS_FORK_BATCH_MIN_BUILD`) print the older once-per-pattern format,
> which apeGmsh does not parse. Solver statistics are unavailable for
> this run.

That one sentence covers both live causes — an old fork, and a flag that
did not take — and names the constant a reader can act on.

### D5 — An explicit `Mumps` on a serial deck is refused at build time

New `validate_serial_mumps(...)` in
[`_internal/build.py`](../../_internal/build.py), sitting immediately
after `validate_ladruno_up_solver` (line 3579) and reusing its
`enforce` / `staged` / `flat_systems` / `stage_systems` seam verbatim —
that function already solves "which system will this deck actually
solve on", including the `wipeAnalysis`-per-stage re-default, and
re-deriving it would be the second copy.

It raises a `BridgeError`:

> `system Mumps` declared on a serial deck (`len(fem.partitions) <= 1`):
> the Ladruno fork's desktop targets do not compile the serial
> `MumpsSolver`, so `system Mumps` answers *unknown system type* on
> `OpenSees.exe` and on the desktop openseespy build — leaving the run
> on whatever SOE was already in place (ProfileSPD by default) rather
> than stopping. Declare `ops.system.Pardiso()` for a threaded desktop
> solve, or partition the mesh (`g.mesh.partitioning`) and run under
> `OpenSeesMP`.

The rationale is not tidiness, it is the silent-wrong-answer class this
project keeps legislating against: a rejected `system` command does not
abort the deck (`OpenSees.exe` exits 0 on a Tcl error), so the model
solves to convergence on a solver the user did not choose. Refusing at
build time turns a wrong answer into a sentence.

Scope is tight, and each exclusion is an existing decision:

- **explicit declarations only.** ADR 0027 INV-5's auto-emitted
  `Mumps`/`UmfPack` pair ([`apesees.py:7885-7935`](../../apesees.py),
  runtime `catch` at [`tcl.py:1634-1651`](../../emitter/tcl.py)) fires
  only on partitioned decks and is already runtime-conditional;
- **partitioned decks are untouched** — that is what Mumps is for;
- **ADR 0077 INV-8** ([`apesees.py:10830-10840`](../../apesees.py))
  *requires* Mumps for parallel ARPACK and is upstream of this gate;
- there is no `MumpsParallel` Python type to refuse beside it — apeGmsh's
  typed systems expose a single `Mumps` (the name is OpenSees' own
  `MumpsParallelSolver`); corrected on S4, PR #1118.

This is the one decision here that is separable from stats capture. It
is in scope because it is the same fact — *the fork's desktop portfolio
is PARDISO, full stop* — that makes the rest of this ADR worth building,
and it ships as its own revertable slice.

**Rejected — warn instead of raise.** A warning is right when the user's
choice is merely suspect (the `_maybe_auto_emit_parallel_system` mirror
warns because a serial solver on a partitioned deck is *slow*, not
*wrong*). Here the declared solver does not exist and something else
answers in its place; there is no reading under which the deck does what
it says.

### D6 — Per-stage reduction: max on the four capacity numbers, last-seen on the four identity ones

`StageSolverStats` carries, per stage:

| field | reduction |
|---|---|
| `factorisations` | count of complete blocks in the stage |
| `max_factor_entries` | max — `iparm(18)`, nnz in L+U |
| `max_peak_memory_kb` | max — `iparm(15)` |
| `max_perm_memory_kb` | max — `iparm(16)` |
| `max_fact_memory_kb` | max — `iparm(17)` |
| `max_mflops` | max — `iparm(19)` |
| `n`, `nnz_a`, `matrix_type`, `threads` | last seen |

Capacity takes the maximum because the question is *did it fit*, and a
stage that refactorises per Newton step can have its worst factorisation
anywhere in the middle: a contact set closing, a constraint handler
promoting, a stiffening tangent all move fill and memory up and then
back down. Reporting the last block would under-report exactly the case
the measurement exists for. Under a constant tangent the stage
factorises once, so max and last coincide and the choice costs nothing.
`iparm(18)` is grouped with the memory lines against the fork's own
framing because it *is* a capacity number — it is the fill that
`iparm(17)` is mostly made of, and it is the term that predicts the next
mesh refinement.

The identity fields take the last value seen. `n`, `nnz(A)`,
`matrixType` and `threads` describe the system that was handed to the
solver, not the cost of solving it; when they change mid-stage the last
one is the shape the stage finished on. They are not maxima and are not
presented as such.

The five maxima are **independent** — they need not come from one
factorisation, and the record does not claim they do. Stated plainly
because the alternative (carry the single worst block, whole) answers
"what did the worst factorisation look like" instead of "what did this
stage demand", and the second question is the one that sizes a machine.

`RunSolverStats` holds the per-stage tuple plus a run-level bucket for
blocks that fall outside any `APEGMSH_STAGE` window — a flat
(unstaged) deck puts everything there, which is the correct and only
answer for a deck that has no stages.

## Invariants

What a green test has to prove:

- **INV-1 — A deck that does not ask for stats is untouched.** No
  `Pardiso(stats=True)` anywhere ⇒ the emitted deck is byte-identical to
  today in every emitter, no `APEGMSH_STAGE` line exists, and
  `stream_run` applies no stats pattern to any line (the parse is not
  merely cheap, it does not run).
- **INV-2 — The log stays verbatim.** Parsing never consumes, reorders,
  rewrites or drops a line from `<deck>.log`. The tee is byte-identical
  to the child's output whether or not stats are on.
- **INV-3 — One block, one factorisation.** A complete block increments
  `factorisations` exactly once; a corrupted block increments
  `malformed_blocks` and never `factorisations`, and yields no
  `SolverStatsBlock`.
- **INV-4 — Attribution is total.** Every complete block lands in
  exactly one bucket: a named stage, or the run-level bucket. No block
  is counted twice and none is discarded for having no home.
- **INV-5 — No measurement reaches a declaration.** `StageRecord` and
  `ProfileRecord` gain no field, `build('h5')` bytes do not move, and
  the `SimpleNamespace` stage-record stubs in `tests/opensees/h5`
  continue to work unmodified.
- **INV-6 — A missing block warns exactly once**, names
  `TIMS_FORK_BATCH_MIN_BUILD`, and does not fail the run.
- **INV-7 — A failed run's numbers survive.** `parse_solver_stats` over
  the tee'd log of a run that raised returns the same record the
  streaming parse had accumulated up to the failure.
- **INV-8 — D5's gate is narrow.** An explicit serial `Mumps`
  raises; a partitioned `Mumps`, the ADR 0027
  auto-emitted runtime fallback, and the ADR 0077 parallel-ARPACK path
  are all unaffected.

## Slices

**S1 — the parser, alone.** `opensees/_solver_stats.py`: the three
dataclasses, `parse_solver_stats`, the two patterns, the D6 reduction.
No wiring. Tests (`tests/opensees/unit/test_solver_stats_parse.py`):
the fork's 54-DOF block parses to `factor_entries == 1836`; a
multi-block stream reduces per D6; a stream with zero blocks returns an
empty record; blocks bracketed by `APEGMSH_STAGE` lines attribute
correctly, including a stage name with spaces; blocks outside any window
land in the run-level bucket.

*Mutation check, in the same file* — six mutants of a known-good block,
each asserting `malformed_blocks == 1` **and** `factorisations == 0`:
a label typo (`iparm(18)` → `iparm(8)`), a dropped `nnz(A)=` field in
the header, a non-numeric value, a truncated block missing the Mflops
line, two headers with one field set between them, and one line of the
old once-per-pattern format. A parser that cannot fail these is not
measuring anything.

**S2 — the stage marker and its gate.** `deck_requests_solver_stats` in
`build.py`; `_emit_stage_markers` on the tcl and py emitters; markers in
`stage_open` / `stage_close`. The live emitter is untouched. Tests: the
existing staged golden decks are byte-identical (INV-1, both emitters);
a stats-declaring staged deck emits exactly one open/close pair per
stage; the marker survives a stage name with spaces and quotes; a
partitioned staged deck emits the marker inside each rank guard without
disturbing the guard's own bytes.

**S3 — wiring and the return value.** `stream_run(...,
expect_solver_stats=...) -> RunSolverStats | None`; `apeSees.tcl` /
`apeSees.py` pass the predicate and return the record; the INV-6
warning; `studio/_api_index.json` rebuilt in the same PR. Tests
(`tests/opensees/subprocess/test_solver_stats_capture.py`): a fake child
process printing a canned interleaved log (the ADR 0095 S12a fixture
idiom) yields the expected per-stage record; the tee matches the child's
output byte for byte (INV-2); a run declaring stats with no block warns
once naming the constant (INV-6); a child that exits non-zero still
leaves a log `parse_solver_stats` can read (INV-7).

**S4 — D5's serial-`Mumps` refusal.** `validate_serial_mumps` plus its
call sites, mirroring `validate_ladruno_up_solver`'s. Tests: serial
explicit `Mumps` raises with the sentence;
partitioned `Mumps` is accepted; the ADR 0027 auto-emitted fallback is
accepted; a staged deck where only one stage declares `Mumps` raises
naming that stage; an eigen-only / H5 emit (`enforce=False`) is skipped.
Independently revertable.

**S5 — live acceptance (`ladruno_fork` marker).** A 54-DOF brick with
`Pardiso(stats=True)`, run through `apeSees.tcl(run=True)` against the
real binary: `factor_entries == 1836` for the first factorisation,
`factorisations` equal to the number of steps that refactorised, and the
per-stage split matching a two-stage deck. Auto-skipped without
`APEGMSH_OPENSEES_BIN`, per `tests/conftest.py`.

## Runway — what is deferred and named

1. **The live lane** (D3): an explicit knob over `ops.logFile`, with the
   process-global one-way redirect made visible to the caller rather
   than imposed.
2. **MUMPS statistics on `OpenSeesMP` rank 0** — a different block, a
   different lane (`hpc/_job.py:155` tails, not `stream_run`), and no
   consumer has asked yet.
3. **A `<deck>.solver_stats.json` sidecar** for consumers that cannot
   take a return value (batch HPC, the studio habitat). Derived from the
   D1 record; not a second source of truth.
4. **Feeding the peak to a controller** — ADR 0104's `Substep` reads
   solver signals, and "the last factorisation's fill grew 40 %" is one.
   Speculative until a push actually wants it.
5. **H5 archival of the record: not deferred, refused.** A measurement
   is not a model definition (ADR 0011 / 0014, and the `h5.py:3428`
   profile refusal is the same call). If an archived run record is ever
   wanted it is a results-side artifact, not an emit-side one.
6. **Parsing the pre-#821 once-per-pattern format** — no, unless a fork
   build we must support cannot be upgraded, and then only against a
   pinned fixture.

## Cross-references

- [ADR 0027](0027-cross-partition-mp-constraints.md) INV-5 — the auto-emitted
  `Mumps`/`UmfPack` runtime fallback D5 must not disturb.
- [ADR 0074](0074-ladruno-up-porous-element.md) D4 and
  `validate_ladruno_up_solver` ([`build.py:3579`](../../_internal/build.py))
  — the flat/staged/partitioned system-resolution seam D5 reuses, and
  the precedent for a solver gate with no escape hatch.
- [ADR 0077](0077-parallel-modal-analysis.md) INV-8 — parallel ARPACK requires
  Mumps; upstream of D5.
- [ADR 0104](0104-substep-run-to-criterion-controller.md) — the
  controller that is the eventual consumer of a live solver signal.
- [ADR 0095](0095-apegmsh-studio.md) Amendment 3 / S12a — the
  `stream_run` fake-stream test fixture S3 reuses, and the
  progress-sidecar precedent for parsing a solve stream.
- Fork: PR #821, `tests/test_pardiso_stats.py`,
  `SRC/interpreter/OpenSeesOutputCommands.cpp:4683` (`OPS_logFile`),
  `SRC/handler/StandardStream.cpp:52` (`setFile` echo semantics).
