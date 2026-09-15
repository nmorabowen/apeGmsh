"""``probe_ladruno_branch`` — the ADR 0108 capability probe.

An engine older than fork ``61b3efa04`` parses a ``DruckerPrager`` deck
identically to a repaired one and answers
``eleResponse(e, "material", gp, "ladrunoBranch")`` with an EMPTY list —
no exception. That silence is the only cheap signal that the run's
tension-cutoff answers cannot be believed, so the probe turns it into a
boolean, and refuses any third width rather than letting a changed
response quietly re-label the eight by-position reader columns.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._element_capabilities import (
    LADRUNO_BRANCH_ELEMENT_CLASSES,
    LADRUNO_BRANCH_WIDTH,
    probe_ladruno_branch,
)


class _RecordingOps:
    """Minimal ``ops`` stand-in returning ``n`` floats and logging the call."""

    def __init__(self, n: int) -> None:
        self._n = n
        self.calls: list[tuple] = []

    def eleResponse(self, *args):
        self.calls.append(args)
        return [0.0] * self._n


def test_width_is_the_documented_eight() -> None:
    assert LADRUNO_BRANCH_WIDTH == 8


def test_empty_reply_means_a_pre_fix_engine() -> None:
    assert probe_ladruno_branch(_RecordingOps(0), 7) is False


def test_eight_floats_mean_a_post_fix_engine() -> None:
    assert probe_ladruno_branch(_RecordingOps(8), 7) is True


@pytest.mark.parametrize("n", [1, 5, 7, 9, 36])
def test_any_other_width_is_refused(n: int) -> None:
    """A third width means the fork changed the response, and every
    by-position name in the ``.ladruno`` reader is now wrong."""
    with pytest.raises(ValueError, match="ladrunoBranch"):
        probe_ladruno_branch(_RecordingOps(n), 7)


def test_it_asks_the_material_not_the_element() -> None:
    """The token is a MATERIAL response — the bare element-level
    spelling records and answers nothing on the fork."""
    ops = _RecordingOps(8)
    probe_ladruno_branch(ops, 42, gauss_point=3)
    assert ops.calls == [(42, "material", "3", "ladrunoBranch")]


def test_gauss_point_is_one_based_by_default() -> None:
    ops = _RecordingOps(8)
    probe_ladruno_branch(ops, 1)
    assert ops.calls[0][2] == "1"


def test_the_forwarding_element_classes_are_the_forks_four() -> None:
    """The classes whose ``setResponse`` forwards ``material <gp> <tok>``
    to the NDMaterial (fork ADR-95). Aiming the probe at anything else
    answers ``[]`` for reasons that have nothing to do with the build."""
    assert LADRUNO_BRANCH_ELEMENT_CLASSES == frozenset({
        "LadrunoBrick", "LadrunoBrick20", "BezierTet10",
        "TenNodeTetrahedron",
    })
