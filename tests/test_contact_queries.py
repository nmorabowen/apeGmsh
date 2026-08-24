"""Fork ADR-85 adoption S6 (thin) — live contact queries.

The ``LadrunoContact`` subsystem is a constraint handler plus a contact
domain, **not** an ``Element``, so it has no element response and therefore
**no recorder channel**: contact data is live-query-only. That is why this
lane is four thin bridge wrappers rather than a ``Results`` reader — there
is nothing on disk for a reader to read.

Four fork commands are exposed:

* ``ladrunoContactForce(node)``   -> NTS-only scalar normal-force magnitude
* ``ladrunoContactInfo()``        -> (contacts, commits, reverts, mortar)
* ``ladrunoMortarPenetration()``  -> mortar ALM convergence measure (a LENGTH)
* ``ladrunoMortarTieResidual()``  -> mortar-tie ALM convergence measure

The unit + fork-gate legs run anywhere. The live leg needs a Ladruno build
carrying the 2-D contact lane and is skipped otherwise (the build installed
in ``opensees_venv`` predates it).
"""
from __future__ import annotations

from typing import cast

import pytest


# --------------------------------------------------------------------------
# Fork-gate — the live emitter refuses stock openseespy by name
# (no openseespy needed: a stub stands in for the module)
# --------------------------------------------------------------------------

class _StockOps:
    """Stand-in for stock openseespy: none of the four queries exist."""


class _ForkOps:
    """Stand-in for a fork build, with the four queries."""

    def ladrunoContactForce(self, tag: int) -> float:      # noqa: N802
        return 1234.5

    def ladrunoContactInfo(self):                          # noqa: N802
        return [2, 7, 1, 1]

    def ladrunoMortarPenetration(self) -> float:           # noqa: N802
        return 5.0e-7

    def ladrunoMortarTieResidual(self) -> float:           # noqa: N802
        return 2.5e-9


class _MortarOnlyOps(_ForkOps):
    """A pure-mortar model. Measured on fork ``b17e8bd82``: the NTS counter
    reads 0 while the engine is very much alive."""

    def ladrunoContactInfo(self):                          # noqa: N802
        return [0, 20, 0, 1]


def _emitter_with(ops_stub: object):
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    em = LiveOpsEmitter.__new__(LiveOpsEmitter)
    em._ops = cast("object", ops_stub)      # type: ignore[attr-defined]
    return em


@pytest.mark.parametrize(
    "verb",
    ["ladruno_contact_force", "ladruno_contact_info",
     "ladruno_mortar_penetration", "ladruno_mortar_tie_residual"],
)
def test_every_query_refuses_stock_by_name(verb):
    """A bare AttributeError would send the reader hunting in apeGmsh; the
    gate has to name the fork instead."""
    em = _emitter_with(_StockOps())
    fn = getattr(em, verb)
    with pytest.raises(RuntimeError, match="require the Ladruno fork build"):
        fn(1) if verb == "ladruno_contact_force" else fn()


def test_queries_pass_through_on_a_fork_build():
    em = _emitter_with(_ForkOps())
    assert em.ladruno_contact_force(7) == pytest.approx(1234.5)
    assert em.ladruno_mortar_penetration() == pytest.approx(5.0e-7)
    assert em.ladruno_mortar_tie_residual() == pytest.approx(2.5e-9)


def test_contact_info_is_named_not_positional():
    """Four bare ints in a row are trivially transposable; the whole point of
    the NamedTuple is that ``n_mortar_contacts`` cannot be read as a commit
    count by accident."""
    info = _emitter_with(_ForkOps()).ladruno_contact_info()
    assert (info.n_contacts, info.n_commits,
            info.n_reverts, info.n_mortar_contacts) == (2, 7, 1, 1)
    assert info.n_contacts == 2


def test_a_mortar_only_model_is_not_mistaken_for_no_engine():
    """The counters are disjoint LANES, not a total and a subset. Measured on
    fork ``b17e8bd82``, a pure-mortar model reports ``n_contacts=0``, so
    reading that as "no contact engine" misclassifies every mortar-only
    model — and every other contact query returns a bare ``0.0`` that only
    this can disambiguate."""
    info = _emitter_with(_MortarOnlyOps()).ladruno_contact_info()
    assert info.n_contacts == 0            # the NTS lane is empty...
    assert info.n_mortar_contacts == 1     # ...but the engine is live
    assert info.total_contacts == 1


def test_total_contacts_is_zero_only_with_no_engine():
    from apeGmsh.opensees.emitter.live import ContactInfo

    assert ContactInfo(0, 0, 0, 0).total_contacts == 0
    assert ContactInfo(3, 9, 0, 2).total_contacts == 5


# --------------------------------------------------------------------------
# Facade — fails loud BEFORE any live analysis, and says why there is no
# recorded alternative (no openseespy needed: never reaches the emitter)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "verb",
    ["ladruno_contact_force", "ladruno_contact_info",
     "ladruno_mortar_penetration", "ladruno_mortar_tie_residual"],
)
def test_facade_requires_a_live_analysis(verb):
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees._internal.build import BridgeError
    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    bridge = apeSees(cast("object", make_two_node_beam()))
    fn = getattr(bridge, verb)
    with pytest.raises(BridgeError, match="no live analysis has run"):
        fn(1) if verb == "ladruno_contact_force" else fn()


def test_facade_error_states_there_is_no_recorded_route():
    """The obvious next question after "call analyze first" is "or can I just
    record it?" — the answer is no, and the message has to say so."""
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees._internal.build import BridgeError
    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    bridge = apeSees(cast("object", make_two_node_beam()))
    with pytest.raises(BridgeError) as exc:
        bridge.ladruno_contact_force(1)
    msg = str(exc.value)
    assert "no recorder channel" in msg
    assert "live-query-only" in msg
