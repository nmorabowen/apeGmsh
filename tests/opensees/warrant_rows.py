"""Warrant rows: ADR 0121 D3, invariants 9 and 10.

A member that behaves differently on stock OpenSees and on the Ladruno fork
is *Warranted* when its warrant row is green: its fork test **passed** in
``live-fork``, and its stock test passed in ``live-stock``, where the stock
test either pins a refusal (``refused``: a ``pytest.raises`` before
``analyze`` whose error names the fork) or checks the stock answer against
an oracle (``honest``). Every other member is Inventory. Nobody fills in a
status: a row only says which two tests decide it, and the lanes decide.

Each live lane sets ``APEGMSH_WARRANT_LANE`` (``stock`` in the ``live-stock``
job of ``tests.yml``, ``fork`` in ``live-fork.yml``). With it set, the root
``tests/conftest.py`` applies the rule of D3 and invariant 10:

* a row test of this lane that skips (or xfails) **fails the lane**: its
  report turns into a failure that names the row;
* a row test of this lane that the run did not select at all (a dropped
  marker, a renamed function) is a usage error before any test runs;
* ruling 3: a fork-lane row test marked ``ladruno_mkl`` that skips with the
  ``ladruno_mkl:`` reason stays a skip. It counts for the lane's green, as
  every ``ladruno_mkl`` skip does, and never for the member: the terminal
  summary reports that side as not green, so the member stays Inventory.
* post-pin (#2498, T's option (b) of 2026-10-23): a fork-lane row test
  that skips because :func:`require_fork_features` found the live build
  reporting the absence of a feature the row names in ``post_pin`` stays a
  skip, on the same terms as ruling 3: it counts for the lane, never for
  the member. The skip lifts itself once ``FORK_PIN`` moves to a build that
  has the feature.

Post-pin features. A test that needs a fork feature newer than ``FORK_PIN``
calls :func:`require_fork_features` with names from :data:`POST_PIN_FEATURES`.
Each feature has a one-line probe declaration and the line the fork prints
when it lacks the feature. The test skips **only** on that evidence: when
the probe declaration is accepted, fails for any other reason, or prints
nothing this process can read, the test runs, and fails loudly if it is
wrong. A skip therefore never hides a failure on a build that has the
feature (``tests/opensees/unit/test_live_option_rejection_message.py``).

``tests/opensees/unit/test_warrant_rows.py`` locks the registry (node ids
exist, markers fit the lanes, a refused row's stock test holds a
``pytest.raises`` that names the fork, roster rows cite the row's tests)
and invariant 9 over the fork members (``GRANDFATHERED_FORK_MEMBERS``).
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

#: The environment switch each live lane sets.
LANE_ENV = "APEGMSH_WARRANT_LANE"
LANES = ("stock", "fork")
STOCK_STATES = ("refused", "honest")

#: The reason prefix ``tests/opensees/integration_ladruno/conftest.py``
#: gives a ``ladruno_mkl`` skip (Pardiso missing from the build).
MKL_SKIP_PREFIX = "ladruno_mkl:"

#: The reason prefix :func:`require_fork_features` gives a post-pin skip.
POST_PIN_SKIP_PREFIX = "post_pin:"
_POST_PIN_LACKS = f"{POST_PIN_SKIP_PREFIX} the live build lacks "
_POST_PIN_SEP = " + "
_POST_PIN_END = " -- "


@dataclass(frozen=True)
class WarrantRow:
    """One member, the two tests that decide its warrant, and its state."""

    member: str
    stock_state: str
    stock_test: str
    fork_test: str
    roster: str | None = None
    #: Names from :data:`POST_PIN_FEATURES` that the fork test needs and
    #: ``FORK_PIN`` may predate; a post-pin skip naming only these stays a
    #: skip in the fork lane (module docstring).
    post_pin: tuple[str, ...] = ()

    def test_for(self, lane: str) -> str:
        return self.stock_test if lane == "stock" else self.fork_test


_TET = "tests/opensees/live/test_tet10_volume_live.py::"
_MEMBRANE = "tests/opensees/live/test_ladruno_membrane_live.py::"
_FORCE_BEAM = "tests/opensees/live/test_ladruno_force_beam_column_live.py::"

ROWS: tuple[WarrantRow, ...] = (
    # Refused exemplar. Stock TenNodeTetrahedron reads 1/6 of the true
    # stiffness (roster D1); the live emitter refuses it on stock at emit
    # time, before analyze, with an error that names the fork; and
    # the fork meets the closed form K = E*A/H.
    WarrantRow(
        member="TenNodeTetrahedron",
        stock_state="refused",
        stock_test=_TET + "test_live_emitter_refuses_tet10_on_stock",
        fork_test=_TET + "test_tet10_block_matches_closed_form_on_a_fixed_fork",
        roster="D1",
    ),
    # Honest exemplar. FourNodeTetrahedron meets the same closed form on
    # both engines.
    WarrantRow(
        member="FourNodeTetrahedron",
        stock_state="honest",
        stock_test=_TET + "test_tet4_block_matches_closed_form",
        fork_test=_TET + "test_tet4_block_matches_closed_form_on_the_fork",
    ),
    WarrantRow(
        member="LadrunoMembrane",
        stock_state="refused",
        stock_test=_MEMBRANE + "test_live_emitter_refuses_ladruno_membrane_on_stock",
        fork_test=_MEMBRANE + "test_ladruno_membrane_distributes_pressure_to_supports_on_fork",
        roster="D10",
    ),
    WarrantRow(
        member="LadrunoForceBeamColumn",
        stock_state="refused",
        stock_test=_FORCE_BEAM + "test_live_emitter_refuses_ladruno_force_beam_column_on_stock",
        fork_test=_FORCE_BEAM + "test_ladruno_force_beam_column_cantilever_matches_closed_form_on_fork",
        roster="D11",
        # Absent at ladruno-v20261017 ("element type ... is unknown"); #2498.
        post_pin=("LadrunoForceBeamColumn",),
    ),
)

#: The fork members that predate the freeze (ADR 0121 D4, ruling 5): the
#: live emitter's ``_FORK_ONLY_ELEMENTS`` and ``_FORK_ONLY_INTEGRATORS``,
#: and the material classes whose docstring carries the fork command header
#: or a 33xxx tag (``test_warrant_rows.py::is_fork_material``); 42 at the
#: freeze point. They stay Inventory with no deadline (X4). Invariant 9: any
#: other fork member ships with a ``refused`` warrant row. Never add a name
#: here; remove one when the member goes.
GRANDFATHERED_FORK_MEMBERS = frozenset({
    # fork material classes
    "LadrunoBondSlip", "LadrunoCohesiveHinge", "LadrunoCohesiveHingeBiaxial",
    "LadrunoConcrete3D", "LadrunoJ2", "LadrunoJ2Finite", "LadrunoRCConcrete",
    "LadrunoRCFiniteStrain", "LadrunoRebarBuckling", "LadrunoSANISAND",
    "LadrunoUniaxialJ2", "LogStrain", "LogStrain2D", "InitDefGrad",
    "StagedStrain",
    # _FORK_ONLY_ELEMENTS
    "BezierTri6", "BezierTet10", "LadrunoEmbeddedRebar", "LadrunoBrick",
    "LadrunoBrick20", "LadrunoLST", "LadrunoQuad", "LadrunoCST", "LadrunoUP",
    "LadrunoKinematicCoupling", "LadrunoDistributingCoupling",
    "LadrunoEmbeddedNode", "LadrunoRigidBody", "LadrunoDispBeamColumn",
    "LadrunoIMKBeam",
    # _FORK_ONLY_INTEGRATORS
    "ExplicitBathe", "ExplicitBatheLNVD", "CentralDifferenceLadruno",
    "LadrunoArcLength", "LadrunoDynamicRelaxation", "LadrunoIndirectControl",
    "LadrunoLoadControl", "LadrunoHHT", "LadrunoGeneralizedAlpha",
    "CentralDifferenceSMS", "ExplicitBatheSMS", "ExplicitBatheLNVDSMS",
})

#: ``len(GRANDFATHERED_FORK_MEMBERS)`` at the freeze point (PR-2 #1691, 1d515405);
#: the X4 handoff. The set may shrink, never grow past it.
FREEZE_POINT_FORK_MEMBERS = 42


def lane_from_env(environ: Mapping[str, str]) -> str | None:
    """The lane named by ``APEGMSH_WARRANT_LANE``, or ``None`` when unset.

    An unknown value raises: a typo must not switch the rule off.
    """
    value = environ.get(LANE_ENV, "").strip()
    if not value:
        return None
    if value not in LANES:
        raise ValueError(
            f"{LANE_ENV}={value!r} is not a warrant lane; use one of {LANES}"
        )
    return value


def lane_tests(lane: str) -> dict[str, WarrantRow]:
    """Node id of each row test that runs in ``lane``, to its row."""
    return {row.test_for(lane): row for row in ROWS}


def skip_failure(
    lane: str, row: WarrantRow, reason: str, markers: frozenset[str],
    *, xfail: bool = False,
) -> str | None:
    """Why a skip of ``row``'s ``lane`` test fails the lane, or ``None``.

    ``None`` only for ruling 3, a ``ladruno_mkl`` test in the fork lane that
    skipped because the build lacks Pardiso, and for a fork-lane post-pin
    skip whose missing features are all in ``row.post_pin``. Either skip
    counts for the lane and never for the member (:func:`side_green`).
    """
    if (
        not xfail
        and lane == "fork"
        and "ladruno_mkl" in markers
        and reason.startswith(MKL_SKIP_PREFIX)
    ):
        return None
    if not xfail and lane == "fork" and row.post_pin:
        missing = post_pin_missing(reason)
        if missing and set(missing) <= set(row.post_pin):
            return None
    what = "xfailed" if xfail else "skipped"
    return (
        f"warrant row {row.member!r}: its {lane} test {what} in its own "
        f"lane ({reason}). ADR 0121 D3: a warrant test that skips in its "
        f"lane fails the lane."
    )


def side_green(outcome: str) -> bool:
    """A member's side is green only when its own test passed."""
    return outcome == "passed"


# ---------------------------------------------------------------------------
# Post-pin features (#2498)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostPinFeature:
    """A fork feature newer than some ``FORK_PIN``, and how to probe it.

    ``declare`` issues the one-line probe declaration on a freshly wiped
    domain; ``absent`` matches the line the fork prints when it lacks the
    feature. Only that line is evidence of absence. ``predates`` is the
    ``FORK_PIN`` sha the gate was justified against: once the pin moves,
    ``test_every_gate_predates_the_current_fork_pin`` fails until the bump
    re-justifies (re-stamps) or removes the gate, so no gate outlives its
    pin unexamined.
    """

    name: str
    declare: Callable[[Any], None]
    absent: "re.Pattern[str]"
    predates: str


def _two_nodes_2d(ops: Any) -> None:
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.geomTransf("Linear", 1)


def _imk_with(*option: Any) -> Callable[[Any], None]:
    def declare(ops: Any) -> None:
        _two_nodes_2d(ops)
        ops.uniaxialMaterial("Elastic", 1, 1.0e9)
        ops.element("LadrunoIMKBeam", 1, 1, 2, 0.04, 2.0e11, 1.6e-4, 1,
                    "-matZ", 1, *option)
    return declare


def _cohesive_hinge_implex(ops: Any) -> None:
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.uniaxialMaterial("LadrunoCohesiveHinge", 1, 45.0e3, 250.0,
                         "-linear", "-implex")


def _force_beam_column(ops: Any) -> None:
    _two_nodes_2d(ops)
    ops.section("Elastic", 1, 30e9, 0.09, 6.75e-4)
    ops.beamIntegration("Lobatto", 1, 1, 3)
    ops.element("LadrunoForceBeamColumn", 1, 1, 2, 1, 1)


#: ``FORK_PIN`` ``sha`` of ladruno-v20261017, the build these gates predate.
_PIN_20261017 = "100eb9ae11a92bbcf623a44181a39304a697dcbb"

#: Every post-pin feature a test may require, by name. The patterns are the
#: lines ladruno-v20261017 printed on the live-fork lane (run 39773542334).
POST_PIN_FEATURES: dict[str, PostPinFeature] = {
    f.name: f for f in (
        PostPinFeature(
            "LadrunoIMKBeam -shear", _imk_with("-shear", 8.0e10, 0.03),
            re.compile(r"LadrunoIMKBeam.*unknown option '-shear'"),
            predates=_PIN_20261017,
        ),
        PostPinFeature(
            "LadrunoIMKBeam -shearHinge", _imk_with("-shearHinge", 1),
            re.compile(r"LadrunoIMKBeam.*unknown option '-shearHinge'"),
            predates=_PIN_20261017,
        ),
        PostPinFeature(
            "LadrunoCohesiveHinge -implex", _cohesive_hinge_implex,
            re.compile(r"LadrunoCohesiveHinge.*unknown option '-implex'"),
            predates=_PIN_20261017,
        ),
        PostPinFeature(
            "LadrunoForceBeamColumn", _force_beam_column,
            re.compile(r"element type LadrunoForceBeamColumn is unknown"),
            predates=_PIN_20261017,
        ),
    )
}


def fork_pin_sha(text: str) -> str:
    """The ``sha`` of a ``FORK_PIN`` file's ``text``, read as the live-fork
    lane reads it (``grep -E '^sha=' FORK_PIN | cut -d= -f2-``): the rest of
    the one line that starts with ``sha=``. None, or several, raises."""
    shas = [line.split("=", 1)[1] for line in text.splitlines()
            if line.startswith("sha=")]
    if len(shas) != 1:
        raise ValueError(f"FORK_PIN must hold exactly one sha= line, found {len(shas)}")
    return shas[0]


def stale_gates(pin_sha: str) -> list[str]:
    """Each post-pin gate whose ``predates`` is not ``pin_sha``, described."""
    return [
        f"{f.name!r} predates {f.predates[:12]}, FORK_PIN is now {pin_sha[:12]}"
        for f in POST_PIN_FEATURES.values() if f.predates != pin_sha
    ]


@contextlib.contextmanager
def _captured_output() -> Iterator[list[str]]:
    """Collect what the engine prints, through ``sys.stdout`` /
    ``sys.stderr`` and through file descriptors 1 and 2; the text lands in
    the yielded list on exit. A stream this process cannot see (a ``.pyd``
    with its own C runtime on Windows) reads as nothing, never as absence.
    """
    box: list[str] = []
    py_out, py_err = io.StringIO(), io.StringIO()
    saved_streams = sys.stdout, sys.stderr
    with tempfile.TemporaryFile() as raw:
        for stream in saved_streams:
            stream.flush()
        saved_fds = os.dup(1), os.dup(2)
        os.dup2(raw.fileno(), 1)
        os.dup2(raw.fileno(), 2)
        sys.stdout, sys.stderr = py_out, py_err
        try:
            yield box
        finally:
            sys.stdout, sys.stderr = saved_streams
            os.dup2(saved_fds[0], 1)
            os.dup2(saved_fds[1], 2)
            os.close(saved_fds[0])
            os.close(saved_fds[1])
            raw.seek(0)
            box.append(raw.read().decode("utf-8", errors="replace"))
            box.append(py_out.getvalue())
            box.append(py_err.getvalue())


def probe_absence(ops: Any, name: str) -> str | None:
    """The line by which ``ops`` reports lacking feature ``name``, or ``None``.

    ``None`` means "not shown absent": the declaration was accepted, or it
    failed for another reason, or the engine printed nothing this process
    can read. The caller then runs its test. An unknown ``name`` raises.
    """
    if name not in POST_PIN_FEATURES:
        raise KeyError(
            f"{name!r} is not a post-pin feature; known: "
            f"{sorted(POST_PIN_FEATURES)}"
        )
    feature = POST_PIN_FEATURES[name]
    ops.wipe()
    with _captured_output() as text:
        try:
            feature.declare(ops)
        except Exception:  # noqa: BLE001 - the printed line decides, not this
            pass
        finally:
            ops.wipe()
    hit = feature.absent.search("\n".join(text))
    return hit.group(0) if hit else None


#: One probe per feature per process: the imported build cannot change.
_ABSENCE_CACHE: dict[str, "str | None"] = {}


def require_fork_features(*names: str) -> None:
    """Skip the calling test, naming each feature the live build reports
    missing; otherwise return and let the test run (#2498)."""
    if not names:
        raise ValueError("require_fork_features needs at least one feature")
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    evidence: dict[str, str] = {}
    for name in names:
        if name not in _ABSENCE_CACHE:
            _ABSENCE_CACHE[name] = probe_absence(ops, name)
        line = _ABSENCE_CACHE[name]
        if line is not None:
            evidence[name] = line
    if evidence:
        import pytest

        pytest.skip(post_pin_reason(evidence))


def post_pin_reason(evidence: Mapping[str, str]) -> str:
    """The skip reason for the features a build reported missing."""
    return (
        _POST_PIN_LACKS + _POST_PIN_SEP.join(evidence) + _POST_PIN_END
        + "; ".join(f"{k}: {v!r}" for k, v in evidence.items())
        + ". FORK_PIN predates them (#2498); the test runs once it moves."
    )


def post_pin_missing(reason: str) -> tuple[str, ...]:
    """The registered features a post-pin skip ``reason`` names; ``()`` when
    ``reason`` is no post-pin skip or names an unregistered feature."""
    if not reason.startswith(_POST_PIN_LACKS) or _POST_PIN_END not in reason:
        return ()
    head = reason[len(_POST_PIN_LACKS):].split(_POST_PIN_END, 1)[0]
    names = tuple(head.split(_POST_PIN_SEP))
    if not all(n in POST_PIN_FEATURES for n in names):
        return ()
    return names
