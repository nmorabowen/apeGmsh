"""Fork-only — F10's self-weight IMPL-EX diagnosis (fork PR #837), the
"bucket is readable and zero on an uncapped run" check.

The adoption guide (``internal_docs/ladruno_adoption_2026-09-15.md`` §4)
is explicit that F10 is a DIAGNOSIS, not a feature: no new token, no
default moved, no response change. There is no self-weight SANISAND
campaign harness anywhere in this repo today (checked
``benchmarks/``, ``scripts/``, ``examples/``, ``tests/``,
``internal_docs/`` for ``self-weight`` / ``selfweight`` / ``self_weight``,
``implex_control``, per-leg CSV writers, and a ``campaign``/``leg``
runner — the only hits are the unrelated body-force double-count guard,
the TIMs strip-footing discriminator gate, and the plain "self-weight"
prose in the loads/masses guides). This file is therefore the harness
the guide asks for, cut down to the concrete claim it wants proved: on
a bare ``-implex`` push (``implex=True, implex_control=None``, no
``-implexControl``) with NO ``max_substeps`` cap, the **companion**
slot of ``material.implexRefusals`` (canonical name
``implex_refusals_companion``, slot 3 of 6 — see
``_ladruno_element_io.py``'s widened-4-to-6 table) stays ``0`` across
every recorded row. The guide's leg N measured this on a strip-footing
BVP; this is the same shape on the sibling F7 test's 1-brick fixture.

Two things this test deliberately does NOT claim (guide §4.2 items 2
and 3, echoed on ``LadrunoSANISAND.implex_control``'s docstring):
a companion count of 0 here is a TERMINATION result on ONE small,
elastic-dominated deck, not a capacity, and it says nothing about
whether the control is unnecessary in general — the guide's own
low-confinement corner sits at 1.27x this deck's minimum ``p'``.

Deck shape: the same 1 m^3 ``LadrunoBrick`` / isotropic-confinement /
``loadConst`` / stage-1 flip fixture as
``test_ladruno_sanisand_commit_refusal_live.py``, but the SANISAND is
built ``implex=True, implex_control=None`` and the deviatoric leg is a
GENTLE push (one tenth the sibling's load) over several small
``LoadControl`` increments rather than one aggressive step, so the run
completes rather than aborting -- the sibling's own full-strength push
drives this small deck's mean stress toward the ``p_min`` floor within
a handful of increments regardless of the substep cap (MEASURED live,
not in the guide's prose; the guide's leg N runs a real strip-footing
mesh, not a 1-brick material driver). ``max_substeps`` cannot
literally be ``0`` either: the fork's parser hard-refuses ``-implex``
on ``IntScheme 1`` with ``-maxSubsteps 0`` (measured live -- "REQUIRES
-maxSubsteps > 0", ADR 92 D3 / ADR-90 GATE U -- the companion runs at
commitState with no global Newton left to react, so it must be able to
FAIL rather than force-accept). "Uncapped" for this deck means a cap
generous enough never to bind (``5000``, against a push this small);
the companion bucket only ever climbs when a cap actually binds, so a
companion count that never grows past its pre-push baseline is the
expected proof it never did, not a tautology from an absent flag.

Why "never grows past baseline" and not a literal ``== 0``: MEASURED
live, running this deck in the same process right after either sibling
F7 test (which deliberately drives a companion into its "REFUSING
every further update" latch) leaves that count as a non-zero STATIC
baseline this fresh material inherits, even across a fresh
``LiveOpsEmitter(wipe=True)`` and a different material tag -- a
process-level engine quirk, not a reader or fixture bug (ruled out by
directly probing ``eleResponse`` outside the recorder/reader path).
Comparing the last confinement row to the final row isolates exactly
what THIS leg can affect and reduces to the guide's literal ``== 0``
on an uncontaminated (solo, or first-in-session) run.

Concurrent-writer guard (guide §4.2 item 5): this single test writes
one throwaway ``.ladruno`` file under pytest's own ``tmp_path``, so it
needs no guard here, but any REAL per-leg CSV/``.ladruno`` harness
built on this shape needs a lock file or a refuse-if-exists guard --
the fork tore a results CSV by launching the same leg twice.

Skips: the root conftest auto-skips ``ladruno_fork`` off the live
backend; the sibling module's ``_require_commit_refusal_support`` probe
is reused verbatim (same six-wide check) since a build that predates
the SIX-wide response also predates a labelled ``companion`` slot.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

from .test_ladruno_sanisand_commit_refusal_live import (
    _GORINI,
    _node_coords,
    _plane,
    _require_commit_refusal_support,
    _single_hex_fem,
)

pytestmark = pytest.mark.ladruno_fork

_WANTED_NAMES = (
    "implex_refusals_total", "implex_refusals_sign_change",
    "implex_refusals_control", "implex_refusals_companion",
    "implex_refusals_commit_latched", "implex_refusals_latched",
)


def test_selfweight_bare_implex_uncapped_companion_stays_zero(tmp_path) -> None:
    """Bare ``-implex`` (no ``-implexControl``), uncapped: the companion
    bucket is readable off a real ``.ladruno`` file and never grows
    past its pre-push baseline (0 on a clean run -- see module
    docstring for why a literal ``== 0`` is not process-order-safe).
    """
    _require_commit_refusal_support()

    fem = _single_hex_fem("f10_selfweight")
    coords = _node_coords(fem)
    bottom = _plane(fem, 2, 0.0)
    top = _plane(fem, 2, 1.0)
    origin = next(n for n, c in coords.items() if c == (0.0, 0.0, 0.0))
    on_x = next(n for n, c in coords.items() if c == (1.0, 0.0, 0.0))

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    # A throwaway, unattached material to push the SANISAND off tag 1
    # -- MEASURED live: a LadrunoSANISAND that LATCHES ("REFUSING every
    # further update") on a given tag, as the sibling F7 tests both
    # deliberately do, leaves that tag's refusal state behind even
    # across ``ops.wipe()`` and a freshly constructed material object,
    # when this test runs in the same process after either F7 test
    # (e.g. the combined verification run). Tags are allocated at
    # CONSTRUCTION order regardless of whether a material is ever
    # attached to an element, so this alone is enough to land the real
    # material on a tag the sibling module never touches.
    ops.nDMaterial.ElasticIsotropic(E=1000.0, nu=0.3)
    # MEASURED live (not stated in the adoption guide's prose): the
    # fork's parser hard-refuses ``-implex`` on ``IntScheme 1`` with
    # ``-maxSubsteps 0`` ("REQUIRES -maxSubsteps > 0") -- the companion
    # runs at commitState with no global Newton left to react, so it
    # must be ABLE to fail rather than force-accept at dT_min (ADR 92
    # D3 / ADR-90 GATE U). "Uncapped" for this deck therefore means a
    # cap generous enough never to bind, not ``max_substeps=0`` --
    # the assertion below is exactly the proof that it never binds.
    sand = ops.nDMaterial.LadrunoSANISAND(
        **_GORINI, int_scheme=1,
        implex=True, implex_control=None, max_substeps=5000,
    )
    ops.element.LadrunoBrick(pg="soil", material=sand)
    ops.fix(nodes=[origin], dofs=(1, 1, 1))
    ops.fix(nodes=[on_x], dofs=(0, 1, 1))
    ops.fix(
        nodes=[n for n in bottom if n not in (origin, on_x)], dofs=(0, 0, 1),
    )

    p_c = 100.0
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid, (x, y, z) in coords.items():
            fx = (p_c / 4.0) if x == 0.0 else (-p_c / 4.0)
            fy = (p_c / 4.0) if y == 0.0 else (-p_c / 4.0)
            fz = (-p_c / 4.0) if z == 1.0 else 0.0
            p.load(node=nid, forces=(fx, fy, fz))

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormUnbalance(tol=1.0e-4 * p_c, max_iter=200)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()

    path = str(tmp_path / "f10_selfweight.ladruno")
    ops.recorder.Ladruno(
        file=path, elem_responses=("stress", "material.implexRefusals"),
    )

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    mtag = ops.tag_for(sand)

    for i in range(10):
        rc = emitter.analyze(steps=1)
        assert rc == 0, f"isotropic confinement step {i} failed, rc={rc}"

    # ADR 0057 Section 2's loadConst contract -- freeze the confinement
    # load and reset pseudo time before the deviatoric leg (same as the
    # sibling F7 fixture).
    emitter.ops.loadConst("-time", 0.0)
    emitter.update_material_stage(mtag, 1)

    emitter.ops.timeSeries("Linear", 2)
    emitter.ops.pattern("Plain", 2, 2)
    for nid in top:
        emitter.ops.load(nid, 20.0, 0.0, -5.0)
    emitter.ops.integrator("LoadControl", 0.01)

    # A GENTLE deviatoric push, not F7's deliberately aggressive one
    # (F7 wants a starved cap to bite on the very first attempted
    # step; F10 wants a self-weight-like leg that actually reaches a
    # target). The sibling fixture's full-strength push (200, 0, -50)
    # drives p toward the p_min floor within a handful of increments
    # regardless of the substep cap -- MEASURED live, not in the
    # guide's prose (the guide's leg N runs a real strip-footing mesh,
    # not this 1-brick material driver). One tenth the load keeps the
    # stress path away from that floor over the whole leg.
    for i in range(20):
        rc = emitter.ops.analyze(1)
        assert rc == 0, f"deviatoric step {i} failed, rc={rc}"

    emitter.ops.remove("recorders")
    r = Results.from_ladruno(path)
    available = r.elements.gauss.available_components()
    missing = [c for c in _WANTED_NAMES if c not in available]
    assert not missing, f"{missing} absent from {sorted(available)}"

    companion = np.asarray(
        r.elements.gauss.get(component="implex_refusals_companion").values
    )
    # ``implex_refusals_companion`` is a STICKY, non-decreasing counter
    # per Gauss point (module docstring / _ladruno_element_io.py).
    # MEASURED live (not in the guide's prose): running this deck in
    # the SAME process right after either sibling F7 test -- which
    # deliberately drives a companion into its "REFUSING every further
    # update" latch on a starved cap -- leaves that count in a static
    # baseline this fresh material inherits even across a fresh
    # ``LiveOpsEmitter(wipe=True)`` and a different material tag; a
    # solo run of this module (or the recipe's own isolated process)
    # starts that baseline at 0, matching the guide's leg N ledger
    # ("0/0/0/0") exactly. The claim this test proves is scoped to
    # what THIS leg can affect: the bucket does not GROW across the 10
    # isotropic-confinement rows + this deck's bare-``-implex`` push,
    # i.e. row 9 (the last confinement commit, before the stage-1
    # flip) and every later row agree. On a solo run row 9 is 0, so
    # this reduces to exactly the guide's literal assertion.
    baseline = companion[9]
    assert np.all(companion == baseline), (
        "implex_refusals_companion must not GROW across the bare "
        "-implex leg (guide Section 4.2 item 1) -- confinement-row "
        f"baseline {baseline}, full history {companion}"
    )
    if np.all(baseline == 0.0):
        print("F10: companion baseline 0 (solo/uncontaminated run) -- "
              "matches the guide's leg N ledger literally.")
    else:
        print(
            "F10: companion baseline "
            f"{baseline} inherited from an earlier -implex material in "
            "this process (see comment above) -- still proves no GROWTH."
        )

    # Report the other five slots too (paste into the slice report --
    # this is a termination result, not a capacity, per the guide).
    slots = {
        name: np.asarray(
            r.elements.gauss.get(component=name).values
        )
        for name in _WANTED_NAMES
    }
    print("F10 uncapped implexRefusals slots (last row):", {
        name: values[-1] if len(values) else None
        for name, values in slots.items()
    })
