"""Emit-cost REGRESSION gate for the ADR 0038 10k x 4 cell.

Distinct from :mod:`test_cross_rank_constraint_cost`, which is a
*feasibility* gate: its ADR 0038 thresholds ask "is partitioned emit
viable at SSI scale at all?" and sit far above the working point
(``deck_emit_sec < 5.0`` against a ~0.4 s reality). A 3x regression
lands at ~1.3 s and sails through. That is exactly what happened —
ADR 0065 v2 B2+B3 (#775) made partitioned emit 2-3x slower and the
nightly benchmark reported PASS for 124 commits, because it also
asserts nothing at all.

This module closes both holes. It measures the gate cell and compares
against a COMMITTED baseline, so the question becomes "did this change
make emit slower than we agreed it was?" rather than "is emit
catastrophic yet?".

Two properties make it CI-safe:

* **Parse-normalised.** ``deck_parse_py_sec`` is a bare
  :func:`compile` over the emitted Python deck. No apeGmsh code can
  affect it except by changing how many lines it emits — which
  ``deck_lines`` pins separately. It therefore measures the runner,
  and ``emit / parse`` divides the runner out. This ratio is what
  identified the original regression: it went 0.34 -> 1.10 while the
  absolute numbers were muddied by an unrelated slow machine.
* **Generous tolerance.** The ratio still drifts with memory pressure
  (emit allocates, ``compile`` mostly does not), measured up to ~45%
  under heavy background load on a dev box. :data:`RATIO_TOLERANCE`
  sits well above that drift and well below the 2-3x class of
  regression this exists to catch. It is a tripwire, not a
  microbenchmark — tightening it buys flakiness, not sensitivity.

``deck_lines`` is asserted **exactly**. It has zero run-to-run
variance, and any change to it invalidates the ratio comparison
(more lines legitimately means more emit AND more parse), so a
deliberate deck-shape change must re-baseline rather than silently
shift the denominator.

Re-baselining
-------------
When a change legitimately moves these numbers, regenerate with::

    pytest tests/benchmarks/test_emit_regression_gate.py --emit-gate-write-baseline

and commit ``emit_gate_baseline.json`` in the same PR as the change
that moved it, so the diff shows the cost being accepted.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median
from typing import cast

import pytest

psutil = pytest.importorskip("psutil")

import os  # noqa: E402

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

from apeGmsh.opensees import apeSees  # noqa: E402

from tests.benchmarks.test_cross_rank_constraint_cost import (  # noqa: E402
    GATE_DECK_LINES,
    GATE_EMIT_SEC,
    GATE_INTERFACE,
    GATE_PARSE_SEC,
    GATE_RANKS,
    _build_embedded_fem,
)

BASELINE_PATH = Path(__file__).with_name("emit_gate_baseline.json")

#: Multiple of the baseline ``emit / parse`` ratio that still passes.
#:
#: Calibrated against the real thing rather than guessed. Replaying this
#: gate on the pre-fix tree (``c87a8936``, before #876) measures
#: **2.32x** baseline on tet and **1.99x** on hex — hex is the tighter
#: side because its larger deck makes ``parse`` bigger, shrinking the
#: ratio for the same emit cost. A tolerance of 2.0 therefore catches
#: tet but lets hex through by 0.5%, so the ceiling has to sit below
#: 1.99. At 1.7 both cells trip with margin (tet 27%, hex 15%) while
#: steady-state idle runs sit at ~60% of the ceiling, against a
#: measured run-to-run spread of ~1.5%.
#:
#: Do not tighten this toward the observed spread. Emit allocates and
#: ``compile`` mostly does not, so the ratio drifts upward under memory
#: pressure — up to ~1.6x on a dev box with 10 of 24 cores busy. A
#: dedicated CI runner should stay far below that, but the headroom is
#: deliberate: a gate that cries wolf gets ignored, which is the same
#: failure mode as having no gate.
RATIO_TOLERANCE = 1.7

ELEMENT_KINDS = ("tet_host_line_embed", "hex_host_line_embed")


#: Timed repeats per cell. The FEM build dominates setup cost and is
#: hoisted out of the loop, so extra emits are nearly free — and the
#: median shrugs off a transient spike on a shared CI runner, which a
#: single shot would report as a regression.
REPS = 3


def _measure(kind: str, tmp: Path) -> dict:
    """Emit + parse the gate cell ``REPS`` times; return the medians.

    Mirrors ``test_cross_rank_constraint_cost`` step for step so the
    two report the same quantities: Tcl emit is the timed region,
    ``deck_lines`` counts the Tcl deck, and parse cost is
    :func:`compile` over the separately-emitted Python deck.
    """
    from apeGmsh.opensees.material.nd import ElasticIsotropic
    from apeGmsh.opensees.material.uniaxial import ElasticMaterial

    fem, host_pg, rebar_pg = _build_embedded_fem(
        GATE_INTERFACE, GATE_RANKS, kind,
    )
    nd_mat = ElasticIsotropic(E=2.0e10, nu=0.2)
    ux_mat = ElasticMaterial(E=2.0e11)

    def _setup(ops: object) -> None:
        ops.model(ndm=3, ndf=3)  # type: ignore[attr-defined]
        ops.register(nd_mat)  # type: ignore[attr-defined]
        ops.register(ux_mat)  # type: ignore[attr-defined]
        if kind == "tet_host_line_embed":
            ops.element.FourNodeTetrahedron(  # type: ignore[attr-defined]
                pg=host_pg, material=nd_mat,
            )
        else:
            ops.element.stdBrick(  # type: ignore[attr-defined]
                pg=host_pg, material=nd_mat,
            )
        ops.element.Truss(  # type: ignore[attr-defined]
            pg=rebar_pg, A=1e-4, material=ux_mat,
        )

    emits: list[float] = []
    parses: list[float] = []
    deck_lines = 0

    for i in range(REPS):
        tcl_path = tmp / f"gate_{kind}_{i}.tcl"
        ops_tcl = apeSees(cast("object", fem))
        _setup(ops_tcl)
        t0 = time.perf_counter()
        ops_tcl.tcl(str(tcl_path))
        emits.append(time.perf_counter() - t0)

        tcl_text = tcl_path.read_text()
        deck_lines = tcl_text.count("\n") + (
            0 if tcl_text.endswith("\n") else 1
        )

        py_path = tmp / f"gate_{kind}_{i}.py"
        ops_py = apeSees(cast("object", fem))
        _setup(ops_py)
        ops_py.py(str(py_path))
        py_src = py_path.read_text()

        t1 = time.perf_counter()
        compile(py_src, str(py_path), "exec")
        parses.append(time.perf_counter() - t1)

    emit = median(emits)
    parse = median(parses)
    return {
        "deck_lines": deck_lines,
        "emit_sec": emit,
        "parse_sec": parse,
        "ratio": emit / parse,
    }


@pytest.fixture(scope="module")
def measured(tmp_path_factory, request) -> dict:
    """Measure both gate cells once, after a warmup pass.

    The warmup matters. ``test_cross_rank_constraint_cost`` runs all 24
    cells in ONE process, so by the time the gate cell fires the
    allocator arenas and import caches are hot; a cold single-cell run
    measures 2-3x higher and is not comparable to that table. The 1k
    cell costs well under a second and puts this module in the same
    regime.
    """
    tmp = tmp_path_factory.mktemp("emit_gate")
    for kind in ELEMENT_KINDS:
        _build_embedded_fem(1_000, GATE_RANKS, kind)
    out = {kind: _measure(kind, tmp) for kind in ELEMENT_KINDS}

    for kind, m in out.items():
        print(
            f"\n[emit-gate] {kind} lines={m['deck_lines']} "
            f"emit={m['emit_sec']:.3f}s parse={m['parse_sec']:.3f}s "
            f"ratio={m['ratio']:.3f}",
        )

    if request.config.getoption("--emit-gate-write-baseline"):
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    kind: {
                        "deck_lines": m["deck_lines"],
                        "ratio": round(m["ratio"], 4),
                    }
                    for kind, m in out.items()
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n[emit-gate] wrote baseline -> {BASELINE_PATH}")
    return out


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():  # pragma: no cover - repo always ships it
        pytest.fail(
            f"missing baseline {BASELINE_PATH.name}; regenerate with "
            "--emit-gate-write-baseline (see module docstring)",
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.mark.bench
@pytest.mark.parametrize("kind", ELEMENT_KINDS)
def test_deck_lines_match_baseline_exactly(kind: str, measured: dict) -> None:
    """Deck shape is pinned — zero variance, so this is a hard equality.

    A changed line count is not necessarily a bug, but it does mean the
    ratio check below is comparing against the wrong denominator, so it
    must be re-baselined deliberately.
    """
    base = _load_baseline()[kind]
    got = measured[kind]["deck_lines"]
    assert got == base["deck_lines"], (
        f"{kind}: deck_lines {got} != baseline {base['deck_lines']}. "
        "The emitted deck changed shape. If intended, re-run with "
        "--emit-gate-write-baseline and commit the new baseline in this PR."
    )


@pytest.mark.bench
@pytest.mark.parametrize("kind", ELEMENT_KINDS)
def test_emit_cost_has_not_regressed(kind: str, measured: dict) -> None:
    """The actual regression gate: emit cost relative to parse cost."""
    base = _load_baseline()[kind]
    got = measured[kind]
    ceiling = base["ratio"] * RATIO_TOLERANCE
    assert got["ratio"] <= ceiling, (
        f"{kind}: emit/parse ratio {got['ratio']:.3f} exceeds "
        f"{ceiling:.3f} (baseline {base['ratio']:.3f} x {RATIO_TOLERANCE}). "
        f"Raw: emit={got['emit_sec']:.3f}s parse={got['parse_sec']:.3f}s. "
        "Parse is a bare compile() over the deck, so it tracks the "
        "runner, not apeGmsh — a ratio jump with deck_lines unchanged "
        "means emit-side work got more expensive. See ADR 0065 / #876 "
        "for the scalar-lookup-in-a-loop shape this was built to catch."
    )


@pytest.mark.bench
def test_adr_0038_feasibility_thresholds_still_hold(measured: dict) -> None:
    """The original ADR 0038 scope gate, now actually asserted.

    Loose by design — it answers "is partitioned emit viable at SSI
    scale", not "did we regress". Kept because it is the number ADR
    0038 committed to; the ratio test above is what catches drift.
    """
    got = measured["tet_host_line_embed"]
    assert got["emit_sec"] < GATE_EMIT_SEC
    assert got["parse_sec"] < GATE_PARSE_SEC
    assert got["deck_lines"] < GATE_DECK_LINES
