### FIXED — `build_schema_corpus.py --base HEAD` anchors every non-current era on `main` (#1365, slice #1404)

The schema-corpus builder took the era history from `--base`, so a bump PR
run with `--base HEAD` recorded the outgoing minor's era as the bump
commit's first parent *on the branch*, and a multi-commit bump wrote that
minor's file with a branch-only writer; a squash merge then orphaned the
SHA `MANIFEST.json` names. The history and every non-current era now come
from `git merge-base HEAD origin/main`, so their SHAs are first-parent
commits of `main` that no squash can orphan; `--base` names only the
commit that writes the current minor (`HEAD` in a bump PR, the merge-base
by default), and the builder refuses a `--base` that does not stamp the
current minor instead of recording it as a gap. The manifest's `base` is
the merge-base. `tests/opensees/h5/test_schema_corpus.py` now holds every
non-current era SHA (and its bump commits) to `origin/main` ancestry, and
skips with a reason in a shallow clone rather than passing on nothing; a
scratch-repo test replays the multi-commit bump PR of #1365 and fails on
the pre-fix resolution.
