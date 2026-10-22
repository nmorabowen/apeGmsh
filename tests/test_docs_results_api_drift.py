"""Results-API docs drift lane — the calls a results page shows must exist.

On 2026-09-28 ``docs/how-to/choose-results-strategy.md``, the page that
tells a reader how to record a run, prescribed ``spec.capture(...)``,
``ops.tcl(..., recorders=spec)`` and ``ops.tcl(..., recorders=spec,
mpco=True)``. None of the three existed: ``capture`` left the recorder
spec when Phase 9 deleted the fluent ``Recorders`` helper, and
``apeSees.tcl`` takes neither ``recorders=`` nor ``mpco=``.
``results-mpco.md`` repeated them. A reader copying any of those cells hit
an ``AttributeError`` or a ``TypeError``, and nothing failed.

This lane resolves the Python that the curated results pages show against
the live code, in the source-scanning style of ``test_skill_docs_drift.py``:

* every ``apeGmsh`` import in a python fence must import;
* every attribute chain rooted at a name the page binds (an import, a
  constructor call, a call whose return annotation names a class, a
  ``with ... as`` target, a page-defined helper that returns one) must
  exist on the live class. Class attributes, dataclass fields, class
  annotations, ``self.<name> = ...`` in any method, and a session's
  ``_COMPOSITES`` table all count;
* every keyword passed to a resolved callable must be one its live
  signature accepts, unless it takes ``**kwargs``;
* inline code spans outside the fences (the decision grid is a table)
  that parse as a Python expression get the same checks, against the names
  the page's fences bound, plus the two receivers prose uses by convention
  (``ops``, ``spec``) when the fences never bind them.

Whatever the resolver cannot type (third-party modules, unannotated or
union returns, names a page never binds) is skipped, not guessed. The lane
can miss drift, but it never reports drift that is not there. The controls
at the bottom replay the incident and must fail. The resolver itself lives
in ``tests/_doc_resolver.py``, shared with the skill fence gate.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._doc_resolver import _scan

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

# The results-API pages whose Python is resolved. Each must resolve at
# least ``_MIN_CHECKED`` names, so a page whose fences stop binding what
# they use cannot pass by checking nothing.
_PAGES = (
    "how-to/choose-results-strategy.md",
    "how-to/results-mpco.md",
    "how-to/read-results.md",
    "concepts/results.md",
    "examples/results-strategies.md",
)
_MIN_CHECKED = 3


# ── Corpus check ────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", _PAGES)
def test_results_page_api_resolves(rel: str) -> None:
    path = DOCS / rel
    assert path.is_file(), f"{rel} not found — update _PAGES if it moved."
    page = _scan(path.read_text(encoding="utf-8"), rel)
    assert len(page.checked) >= _MIN_CHECKED, (
        f"{rel}: only {sorted(page.checked)} resolved — its fences no longer "
        "bind the names they use, so the lane is guarding (almost) nothing."
    )
    if page.problems:
        raise AssertionError(
            "Results-API docs drift — these calls do not exist on the live "
            "code (fix the PAGE; the code is the truth):\n"
            + "\n".join(page.problems)
        )


# ── Positive controls — the resolver must actually resolve ──────────

# The grid's cells as choose-results-strategy.md shipped them at 07f757e0
# (cell prose trimmed): its fences bound neither `spec` nor `ops`.
_INCIDENT_GRID = """\
| **Run: in-process** (notebook) | **`spec.capture(...)`** — the default. \
| **`spec.emit_recorders("out/")`** — classic recorders. \
| **`spec.emit_mpco("run.mpco")`** — MPCO recorder in-process. |
| **Run: export** (cluster / external) | — \
| **`ops.tcl(..., recorders=spec)`** / **`ops.py(...)`** — emit a deck. \
| **`ops.tcl(..., recorders=spec, mpco=True)`** — one `recorder mpco` line. |
"""

# The same phantoms as results-mpco.md shipped them: bound in a fence.
_INCIDENT_FENCE = """\
```python
from apeGmsh.opensees import apeSees
from apeGmsh.results.spec import ResolvedRecorderSpec

ops_bridge = apeSees(fem)
spec = ResolvedRecorderSpec(fem_snapshot_id="0")
ops_bridge.tcl("model.tcl", recorders=spec, mpco=True)
```

Use native domain capture (`spec.capture(...)`) instead.
"""


def test_control_incident_grid_is_flagged() -> None:
    problems = "\n".join(_scan(_INCIDENT_GRID, "incident.md").problems)
    for expected in (
        "`ResolvedRecorderSpec` has no attribute `capture`",
        "`apeSees.tcl()` takes no keyword `recorders=`",
        "`apeSees.tcl()` takes no keyword `mpco=`",
    ):
        assert expected in problems, f"{expected!r} not flagged in:\n{problems}"
    # The cells that were right must stay quiet.
    assert "emit_recorders" not in problems and "emit_mpco" not in problems
    assert "apeSees.py" not in problems


def test_control_incident_fence_is_flagged() -> None:
    problems = "\n".join(_scan(_INCIDENT_FENCE, "incident.md").problems)
    for expected in (
        "`ResolvedRecorderSpec` has no attribute `capture`",
        "`apeSees.tcl()` takes no keyword `recorders=`",
        "`apeSees.tcl()` takes no keyword `mpco=`",
    ):
        assert expected in problems, f"{expected!r} not flagged in:\n{problems}"


def test_control_live_chains_resolve() -> None:
    page = _scan(textwrap.dedent("""\
        ```python
        from apeGmsh.opensees import apeSees
        from apeGmsh.results.spec import ResolvedRecorderSpec

        def build():
            ops = apeSees(fem)
            return ops

        ops = build()
        ops.recorder.MPCO(file="run.mpco", nodal_responses=("displacement",))
        ops.tcl("model.tcl", analyze_steps=10)
        spec = ResolvedRecorderSpec(fem_snapshot_id="x", records=())
        with spec.emit_recorders("out/", ops=ops) as live:
            live.begin_stage("gravity", kind="static")
        ```
        """), "control.md")
    assert page.problems == [], page.problems
    # Across a page-defined helper, an attribute set in ``__init__``
    # (``self.recorder``), an annotated return (``-> LiveRecorders``) and
    # a ``with ... as`` target.
    assert {
        "apeSees.recorder", "_RecorderNS.MPCO", "apeSees.tcl",
        "ResolvedRecorderSpec.emit_recorders", "LiveRecorders.begin_stage",
    } <= page.checked, sorted(page.checked)


def test_control_unknown_names_are_skipped_not_guessed() -> None:
    page = _scan(
        "```python\nimport numpy as np\nnp.not_a_numpy_name()\n"
        "fem.whatever.at_all(x=1)\n```\n\n`opspy.recorder('mpco', ...)`\n",
        "skip.md",
    )
    assert page.problems == [] and page.checked == set()
