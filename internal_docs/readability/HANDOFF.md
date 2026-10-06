# Readability workshop handoff (2026-10-04)

Goal: an STE-like standard for agent-written apeGmsh **model scripts** (not library code). Readable as an engineering procedure, with no loss of modelling power.

## Done
- 4 baseline samples: `samples/01`–`04_*.py`.
- 3-loop multi-model workshop: `workshop/`. Read REPORT.md, contract_v0..v3.md, api_gaps.md, and loop1-3/.
  - Contract vs control readability is about +2.4, or about +1.2 when each model is matched with itself. Sonnet gains; Haiku sits at fidelity 1 in both arms. No power lost.
  - Most unreadable code traces to API gaps (read-back by label, one element per member, supports with label=, stage ceremony).
- Issues #1321–#1328 filed from the samples (J2Plasticity never yields when sig0 >= 1e8 is upstream OpenSees).

## In flight when the session ended (check before redoing)
- Fix PRs (agent worktrees `.claude/worktrees/agent-*`): #1322/#1323/#1326/#1328 (docs + numpy int), #1321/#1327 (warnings), #1324/#1325 (from_mpco stages/PGs). Check with `gh pr list --search "1321 OR 1324 OR 1322"`.
- A scoping agent filing issues for the 8 silent hazards in REPORT.md §5 (promote_to_physical replaces the group, detached diaphragm master, Path series across stages, empty in_box, np.float64 in deck, ndm mismatch, false body-force warning, duplicate corner fix). Check with `gh issue list --search "promote_to_physical"`.

## Silent hazards: scoped and filed, NOT fixed yet (fix in the next session)
#1332 promote_to_physical drops a second group (halved a load) · #1333 detached diaphragm master is singular ·
#1334 Path series across stages gives zero load · #1335 empty in_box → silent to_physical, and "Available PGs: []" bug ·
#1336 np.float64 repr kills decks · #1337 ndm=2 silently drops a nonzero z · #1338 false WarnBodyForceDoubleCount ·
hazard 8 is already #1259 (add the staged refusal message there). The h1-h8 repro scripts lived in the workshop session scratchpad and were not kept; each issue states its reproduction.
Priority: #1336, #1332, #1337 (wrong answers or dead decks), then #1333, #1334, #1338, #1335.

## Fix PRs: the user said "merge them". The fixers were STOPPED before opening PRs (2026-10-04)
- #1321/#1327: worktree `.claude/worktrees/agent-a73d809d2bb341713`, branch `fix/silent-traps-j2-recombine`,
  commit 825d2fb6, clean, not pushed. The suite was being checked: 20 failures, all in `tests/opensees/integration_ladruno`
  (stale fork build, probably not caused by this branch; confirm against origin/main), then push and open the PR.
- #1322/#1323/#1326/#1328: worktree `agent-aa8349bfa8009bf25`, branch `fix/doc-drift-1322-1323-1326-1328`,
  13 uncommitted files. It was waiting on the suite: run sync_skill --check plus the suite, then commit, push and open the PR.
- #1324/#1325: no worktree found, so the work is probably lost. Redo it from the issues.
 Land each with `python scripts/land_pr.py <N>` once it is green; semantic PRs get the cross-family review first.

## Next (the user has not chosen yet; recommended order)
1. Reference scripts in the skill (truss L1 c, frame L2 e, footing L3 b), plus a contract cut to about 10 rules.
2. API gaps from api_gaps.md.
3. An advisory model-script lint (machine-checkable rules are listed in REPORT.md §6).
4. A validation loop: an unseen task, all models in both arms, a v0 arm.
