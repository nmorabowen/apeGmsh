# Handoff — PR merge sequence (B main-green + A contact-stack landing)

Written 2026-06-25. Successor to `handoff_contact_stack_and_main_red.md`. All
engineering + verification is **done**; this doc is the **merge/landing
runbook** + the two open decisions. Nothing here has been pushed/merged — the
merge mechanics are the maintainer's call (branch protection / topology).

## Snapshot

- `main` @ `26c8953c` (#737) — **RED on BOTH axes**: suite (12 stale
  assertions) + static-gates (15 mypy errors, baseline 0; pre-existing since
  #726/#727).
- **PR #738** `fix/main-red-stale-assertions → main` — fixes the 12 suite
  assertions. CI: **lock-tests ✅**, suite ❌ (only the known #730
  `test_live_emitter_restores_real_ops_after_close` order-flake), static-gates
  ❌ (the mypy — fixed by #740, not #738).
- **PR #740** `fix/main-red-mypy → main` — fixes the 15 mypy errors. CI:
  **static-gates ✅, lock-tests ✅**, suite ❌ (the 12 — fixed by #738, not
  #740).
- **`fix/contact-merge`** (local, NOT pushed) — the contact/embed stack. 10
  commits over `main` (see `git log origin/main..fix/contact-merge`):
  `75fbc63f` g.embed, `552fa911` contact gen, `14f0f8ea` merge main,
  `50dae7b7` post-merge fixups, `1a66c38f` prior handoff, `2cd81ee2` review
  round 1, `a4ca95b0` ADR 0072, `533ca191` review round 2, `50a4767a`
  curved-detector fix, `ce12b1c4` review round 3 (workflow).
- Stale contact PRs: **#722** `feat/contact-generator → feat/g-embed-generator`
  (non-main base!), **#723** `feat/contact-extensions → feat/contact-generator`
  (stacked). #721 CLOSED unmerged (its g.embed lives in `fix/contact-merge`).
  **#703** (Linux viewer) is unrelated — leave it.

## The interlock (why neither B PR is green alone)

Both #738 and #740 branch off a **doubly-red** `main`, and each fixes only ONE
axis — so each PR's CI is red on the axis the *other* fixes. They touch
**disjoint files** (#738 = `tests/*`, #740 = `src/*`), so they merge cleanly in
any order. `main` only goes green after BOTH land. The repo already merged
past red static-gates for #726/#727, so admin-merge-past-one-red-axis is an
established pattern here.

## Part 1 — make `main` green (merge #740 then #738)

Order is interchangeable (disjoint files); this order makes the **second** PR
fully green so only the first is an admin-merge-past-red.

1. **Merge #740 first** (squash). Its only red is the suite-12, which #738
   fixes; static-gates + lock-tests are green. After merge, `main` =
   #737 + mypy-clean (`src/apeGmsh/opensees` mypy 0); suite still red (12).
   ```
   gh pr merge 740 --squash --delete-branch
   ```
2. **Rebase #738 onto the new `main`, re-push** so its CI re-runs against a
   mypy-clean base:
   ```
   git fetch origin
   git checkout fix/main-red-stale-assertions
   git rebase origin/main
   git push --force-with-lease
   ```
   #738 should now be **fully green** except possibly the #730 order-flake in
   the suite job (documented, transient). If it fires, re-run just that job:
   `gh run rerun --job <suite-job-id>` (or admin-merge — the flake is known and
   unrelated). Then:
   ```
   gh pr merge 738 --squash --delete-branch
   ```
3. `main` is now **green on both axes**. Confirm: `gh run list --branch main
   --limit 1`.

(If branch protection allows admin override, you can instead just squash-merge
both in either order and let the post-merge `main` CI confirm green — the
disjoint files guarantee no conflict.)

## Part 2 — land the contact stack (after `main` is green)

`fix/contact-merge` currently **inherits** the pre-B `main` (it merged main at
`14f0f8ea`, before the B fixes existed), so it still carries the 12 stale
assertions + 15 mypy errors. It must pick up the B fixes before it can be
green. Its OWN contact/embed changes are fully verified (3 review rounds; ruff
0-new, mypy 0-new, all contact/embed/H5 tests pass).

### Recommended: one fresh PR → `main`, supersede #722/#723

Simplest — avoids untangling #722's non-main base + #723's stacking, and gives
one clean review of the whole stack:

```
git checkout fix/contact-merge
git merge origin/main           # pull in B's fixes (disjoint from contact work)
# resolve nothing expected (B = tests/* + material/live src; contact = its own)
git push -u origin fix/contact-merge
gh pr create --base main --head fix/contact-merge \
  --title "feat(opensees): g.constraints.contact + g.embed (NTS/mortar contact + node-to-host embed)" \
  --body "<summarize: #722+#723 content, ADR 0072, 3 adversarial-review rounds>"
# then close the stale stacked PRs as superseded:
gh pr close 722 --comment "Superseded by the consolidated contact-stack PR (rebased onto main, adversarial-review-hardened, ADR 0072)."
gh pr close 723 --comment "Superseded — its -soft/-visc/-consistanttan/-geomtan extensions are deferred in ADR 0072; reopen when implemented."
```

NOTE on #723: it adds `-soft`/`-visc`/`-consistanttan`/`-geomtan`, which ADR
0072 explicitly **defers** (core-first scope). `fix/contact-merge` does NOT
include them. Closing #723 as superseded means those flags are dropped for now
— reopen/re-port when that work is scheduled. If you want them in this landing,
say so and they need their own review pass (they were never adversarially
reviewed here).

### Alternative: retarget #722 + rebase #723 (preserves the stacked PRs)

Per the original handoff. More fiddly: retarget #722 base
`feat/g-embed-generator → main`, force-push `fix/contact-merge`'s content onto
`feat/contact-generator`, then rebase #723 onto it. Only worth it if you want
to keep the two-PR review structure and land #723's extensions now.

### After landing

- Verify the full suite on the merged `main` (contact tests are non-live /
  emit-only; the 8 fork-live tests need the `opensees_env` fork build).
- The `g.constraints.mortar()` `NotImplementedError` now points users at
  `contact(formulation="mortar", tie=True, outward=...)` — see decision 2.

## Open decisions (need your call)

1. **#723 extensions** (`-soft`/`-visc`/`-consistanttan`/`-geomtan`): drop for
   now (close #723, deferred per ADR 0072) vs port + review into this landing.
2. **`mortar()` delegation**: leave the `NotImplementedError` pointing at the
   fork-backed `contact(formulation="mortar", tie=True)` (current), vs actually
   **delegate** `g.constraints.mortar()` to it — a breaking return-type +
   semantics change (Lagrange-multiplier tie vs ALM penalty contact-tie), so it
   needs its own decision + review, not a silent swap.

## What's verified (so you can trust the landing)

- Contact stack survived an ad-hoc 3-agent review **and** a 30-agent
  adversarial-review **workflow** (6 dimensions, double-verified findings).
  12 findings → **9 unanimous-real, all fixed** (rounds 1–3 on
  `fix/contact-merge`); 2 correctly refuted.
- Highlights fixed: silent-drop under partitioned emit; NTS bare-`kn`+`-outward`
  parser abort; higher-order nps stride; per-facet outward (no harmful
  auto-derive); contact+equation / contact+MP fail-loud (penalty correctly
  excluded); `tie=True` needs explicit outward (else gate-H2 zero-force);
  NaN/inf penalty rejection; quad9 curved-host false positive; H5
  deferred-contact handler consistency; range validation (numpy/bool/auto/
  finite/integer/zero-sentinels).
- Static gates on `fix/contact-merge`: ruff +0, mypy +0 over its base. Once
  rebased on the green `main`, both gates are clean (the +15 mypy is B's #740).
- ADR 0072 documents the API + every design decision.
