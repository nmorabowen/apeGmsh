### ADDED — `scripts/verify_move.py`: proof that a move changed no def or class body (program slice C2.2, #1256)

`python scripts/verify_move.py --base origin/main --head WORKTREE <paths>` compares the multiset of
normalized `ast.dump` over every module-level and class-member `def` / `async def` / `class`, keyed by
qualified name minus the module path, so a def moved between files matches. Docstrings count; imports
and module-level assignments are an informational delta. Exit 0 prints `verify_move: OK — N defs,
identical multiset`; exit 1 lists the added, removed and changed defs with an `ast.dump` diff.
`--json` gives machine-readable output. Stdlib only; self-test in `tests/test_verify_move.py`.
