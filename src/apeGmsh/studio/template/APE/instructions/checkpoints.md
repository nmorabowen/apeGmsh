# Checkpoints — git as the model's accepted history

One rule carries this file: **`main` holds checkpoints only.** A
checkpoint is a commit where the model source, its accepted cases
(`run.json`), and the stage report agree. Anything on `main` is safe
to resume from blind; everything in between lives on a branch.

Results never enter git — the habitat `.gitignore` already excludes
them (no LFS, no force-adds; do not fight it). Git records *that a run
happened and from which source*: each case's `run.json` is tracked and
carries `model_sha` + `git_dirty` (see `models/README.md`).

An un-versioned habitat (`init --no-git`, or no git on that machine)
can adopt this any time: `git init -b main`, one commit of everything,
continue from here.

## The loop

1. **Branch per question** — `work/<model>/<slug>`, one question or
   lineage step per branch. Never one branch per run.
2. **Commit, then run.** Small commits on a work branch are cheap and
   private; land the source change *before* launching a case so the
   `model_sha` in `run.json` names the code that actually ran. A dirty
   run is not rejected — `git_dirty: true` follows the case into the
   stage report as a disclosure.
3. **Merge is acceptance.** When the stage-accepted bar is met
   (`reporting.md`: oracle line, EDP table, ≥1 still/curve, case
   links) and the branch diff has had a review pass:

   ```bash
   git switch main
   git merge --no-ff work/<model>/<slug>
   git tag -a checkpoint/<model>/<slug> -m "<stage>: <oracle line>"
   ```

   The merge message carries stage + oracle line + case links.
   `--no-ff` keeps the branch one readable unit, so
   `git log --first-parent main` reads as the chain of accepted
   answers.
4. **Resume from a tag.** `git tag -l 'checkpoint/*'` lists resumable
   states; a new question branches from `main`, or from an older
   checkpoint when forking a lineage. Dead ends never merge: record
   the lesson in `postmortem/`, delete the branch.

## Branches are the question axis — nothing else

| You want | Use |
|----------|-----|
| What the model *used to be* | A `checkpoint/*` tag (history) |
| Run the same model differently, today (elastic vs nonlinear) | Case drivers in `models/<id>/src/` — all at tip |
| Two idealizations coexisting (shell global vs solid detail) | Two model ids under `models/` |
| A question you are answering now | A `work/` branch — the only thing branches are for |

The anti-pattern is the long-lived variant branch — an "elastic
branch" kept as a default while `main` goes nonlinear. The two
diverge, every sync is a merge headache, and geometry fixes land on
one side only. Keep live variants as drivers at tip.

## Process files commit to `main`

"Checkpoints only" governs the **model surface** — `models/**` and
`reports/**` reach `main` through work branches and checkpoint merges.
Everything that is habitat *process* — `postmortem/**`,
`APE/memory/**`, the backlog, playbook edits, `scripts/` — commits
**directly to `main`**: closing a session must not require a merge
ceremony, and a postmortem is a record, not a reviewable model claim.
A process commit that happens to ride an active work branch is
harmless (it reaches `main` at the merge); salvage a dying branch's
postmortem onto `main` before deleting the branch.

## With a forge (optional, per habitat)

Same contract, one substitution: when the habitat has a remote
(GitHub private, self-hosted), the **PR is the review surface** for
step 3 — push the branch, the PR body carries the same stage / oracle
/ case links, and merging the PR *is* the checkpoint. Nothing else
changes; an air-gapped habitat skips this layer and loses only the
inline review comments. Nothing in studio or these scripts ever calls
a forge.

## Who does what

The agent drives the mechanics — branching, per-session commits, the
review pass, drafting the merge message. **Merging is the human's
acceptance call**, same as merging a PR. No script writes to git for
you: `start_session` prints `branch @ sha (dirty: N)`, `finish_session`
prints the uncommitted count, and both stop there.
