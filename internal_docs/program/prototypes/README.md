# Panel prototypes

These scripts are **unreviewed prototypes** written by the 2026-09-28 expert panel; the papers are in #1192. They are kept only as starting material for the program links that turn them into production code.

**This folder expires on 2026-11-30.** Delete each file once its link lands. X4's retirement ledger tracks the date.

**Rules for using them:**
- Nothing imports these files and nothing collects them as tests.
- Run them with `C:\Users\nmora\venv\opensees_venv\Scripts\python.exe` from the worktree root. They read the AST and never import apeGmsh.

| File | From | What it does | Productionized by |
|---|---|---|---|
| `nav.py` | P6 | AST-only code navigation: `map`, `at`, `where`, `refs`, `up`, `deps`, `h5`, `impl`, `family`, `pkg`. On the panel's 5 questions it took 8 calls and ~2.8k tokens, against 27+ calls and ~23k tokens with Grep/Read. | **Shipped** (N1.1, #1223): `scripts/nav.py`; the prototype is deleted |
| `docpaths.py` | P6 | Checks that backticked paths in the agent-facing docs resolve. | N1 → the `doc-path` quirk rule |
| `contract_cov.py` | P6 | Walks the `ALL_*` contract lists. Today `ALL_ND` covers 8 of 22, and 54 of 181 concrete primitives are unlisted. | C1 → the family-completeness gate |
| `legacy.py` | P7 | Reachability of the legacy results-viewer modules from the live roots (44 modules, ~23.9k LOC). | X2 / X3 |
| `clones.py` | P7 | Clone detection over normalized 8-line windows. | X1 / likely-tier folds |
| `closures.py` | P1 | Inventory of the closures in `ModelViewer.show`. | S3 (on touch) |
| `prstats2.py` | P8 | PR statistics: fix share with a control group, and index-touch share. | T, the weekly KPIs (`prog-auditor`) |
