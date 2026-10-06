### ADDED — quirk lint: `stale-patch-target`, `comment-provenance`, `raw-meta-ndm`; `compose-streams` re-keyed by package glob (program slice C3.2, #1429, closes #1405)

`scripts/check_quirks.py` gains three rules, each with positive and negative self-tests in
`tests/test_check_quirks.py`:

- `stale-patch-target`: a `mock.patch("a.b.c")` or `monkeypatch.setattr("a.b.c", ...)` string
  target in `tests/` that does not resolve statically in `src/` (module file or package, then a
  top-level def, class, assignment, import re-export, or class member; a star import, a module
  `__getattr__` or an inheriting class stays silent). Never imports apeGmsh. Main has no stale
  targets (67 string targets read).
- `comment-provenance`: comments *added* in `src/` since the merge base that carry an issue or PR
  number, "shipped in", "as of <date>" or "previously"/"used to". It runs with
  `--base REF`; `static-gates` now checks out full history and passes the PR's base branch.
  Existing comments are never flagged.
- `raw-meta-ndm` (lesson #1405): `meta["ndm"]` or `meta.get("ndm", ...)` on a meta/attrs receiver
  outside `h5_reader.read_spatial_ndm` and its helpers. It flags the three raw reads at `fabfa042`
  (`model_data.py:486`, `results/capture/_domain.py:652`, `emitter/h5_reader.py:514`) and is silent
  on main.

`compose-streams` now matches `src/apeGmsh/mesh/_compose*` and `_femdata_h5*` by glob, so it still
fires after `_compose.py` becomes a package or the h5 I/O is split.
