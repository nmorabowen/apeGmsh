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

``tests/opensees/unit/test_warrant_rows.py`` locks the registry (node ids
exist, markers fit the lanes, a refused row's stock test holds a
``pytest.raises`` that names the fork, roster rows cite the row's tests)
and invariant 9 over the fork members (``GRANDFATHERED_FORK_MEMBERS``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

#: The environment switch each live lane sets.
LANE_ENV = "APEGMSH_WARRANT_LANE"
LANES = ("stock", "fork")
STOCK_STATES = ("refused", "honest")

#: The reason prefix ``tests/opensees/integration_ladruno/conftest.py``
#: gives a ``ladruno_mkl`` skip (Pardiso missing from the build).
MKL_SKIP_PREFIX = "ladruno_mkl:"


@dataclass(frozen=True)
class WarrantRow:
    """One member, the two tests that decide its warrant, and its state."""

    member: str
    stock_state: str
    stock_test: str
    fork_test: str
    roster: str | None = None

    def test_for(self, lane: str) -> str:
        return self.stock_test if lane == "stock" else self.fork_test


_TET = "tests/opensees/live/test_tet10_volume_live.py::"

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
)

#: The fork members that predate the freeze (ADR 0121 D4, ruling 5): the
#: live emitter's ``_FORK_ONLY_ELEMENTS`` and ``_FORK_ONLY_INTEGRATORS``,
#: and the ``Ladruno*`` material classes of ``material/uniaxial.py`` and
#: ``material/nd.py``; 38 at the freeze point. They stay Inventory with no
#: deadline (X4). Invariant 9: any other fork member ships with a
#: ``refused`` warrant row. Never add a name here; remove one when the
#: member goes.
GRANDFATHERED_FORK_MEMBERS = frozenset({
    # Ladruno* material classes
    "LadrunoBondSlip", "LadrunoCohesiveHinge", "LadrunoCohesiveHingeBiaxial",
    "LadrunoConcrete3D", "LadrunoJ2", "LadrunoJ2Finite", "LadrunoRCConcrete",
    "LadrunoRCFiniteStrain", "LadrunoRebarBuckling", "LadrunoSANISAND",
    "LadrunoUniaxialJ2",
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
FREEZE_POINT_FORK_MEMBERS = 38


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

    ``None`` only for ruling 3: a ``ladruno_mkl`` test in the fork lane that
    skipped because the build lacks Pardiso. That skip counts for the lane
    and never for the member (:func:`side_green`).
    """
    if (
        not xfail
        and lane == "fork"
        and "ladruno_mkl" in markers
        and reason.startswith(MKL_SKIP_PREFIX)
    ):
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
