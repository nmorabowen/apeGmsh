"""Live contact-query procedures (Ladruno fork).

Moved verbatim from ``apesees.apeSees`` (ADR 0114 S1-b); the mixin holds no
state and no ``__init__``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .._internal.build import (
    BridgeError,
)

if TYPE_CHECKING:
    from ..emitter.live import ContactInfo, LiveOpsEmitter

from ._host import _ProcedureHost


class _ContactQueryMixin(_ProcedureHost):
    def _require_live_for_contact_query(self, verb: str) -> "LiveOpsEmitter":
        """The shared no-live-analysis guard for the contact queries."""
        if self._live_emitter is None:
            raise BridgeError(
                f"apeSees.{verb}: no live analysis has run. Call analyze(...) "
                "first (the live path) — the contact queries read the CURRENT "
                "state of the contact engine, so they only mean anything after "
                "a step. There is no recorded alternative: the LadrunoContact "
                "subsystem has no recorder channel, so contact data is "
                "live-query-only (fork ADR-85 adoption S6)."
            )
        return self._live_emitter

    def ladruno_contact_force(self, node: int) -> float:
        """Total normal contact-force **magnitude** on an NTS slave node.

        The sum over that node's active master-segment pairs of
        ``tn = kn·<−gap>₊`` (fork ``ladrunoContactForce``, ADR-39 B3). Works in
        2-D and 3-D. Requires a prior **live** :meth:`analyze`; fork-only.

        Four limits, none of which the return value can tell you about — read
        them before using this number:

        * **NTS lane only.** It is fed exclusively from the segment / end-cap
          branch, so a **mortar** or **rigid-plane** slave always reads
          ``0.0``. Those lanes have no force query at all; recover their forces
          from reactions or the penalty-depth identity instead.
        * **A magnitude, not a vector.** Near a corner or the 2-D D4 end-cap
          the pair normal is not axis-aligned, so this does **not** equal any
          single global force component. The fork's own guide says so.
        * **Zero is ambiguous.** ``0.0`` means "not in contact" *and* "no
          contact engine in this domain". Call :meth:`ladruno_contact_info`
          and check ``total_contacts`` to tell them apart — **not**
          ``n_contacts``, which counts the NTS lane only and reads ``0`` on a
          perfectly live mortar-only model.
        * **A released 3-D pair reports its last-active force forever** — a
          known, deferred fork defect (reproduced at ``f_query = 1000.0``
          against ``f_true = 0.0``). The 2-D lane carries the fix.
        """
        return self._require_live_for_contact_query(
            "ladruno_contact_force").ladruno_contact_force(node)

    def ladruno_contact_info(self) -> "ContactInfo":
        """Engine counters — ``(n_contacts, n_commits, n_reverts,
        n_mortar_contacts)`` (fork ``ladrunoContactInfo``).

        Mostly useful as the disambiguator for the other three queries: they
        all return ``0.0`` both for "nothing happening here" and for "no
        contact engine at all". Use ``info.total_contacts``, the sum of the two
        lane counters — ``n_contacts`` and ``n_mortar_contacts`` are **disjoint
        lanes**, not a total and a subset, so a mortar-only model reports
        ``n_contacts == 0`` with a live engine (measured on fork
        ``b17e8bd82``). Requires a prior live :meth:`analyze`; fork-only.
        """
        return self._require_live_for_contact_query(
            "ladruno_contact_info").ladruno_contact_info()

    def ladruno_mortar_penetration(self) -> float:
        """Max KKT-active normal penetration over all mortar slave nodes
        (fork ``ladrunoMortarPenetration``, ADR-41 C2.2).

        A **length**, not a force — dimension-blind, and unaffected by the
        mortar ``thickness=``. It is the mortar lane's ALM convergence measure:
        the quantity a held-load augmentation loop watches to decide it has
        augmented enough. ``0.0`` with no mortar contact. Requires a prior live
        :meth:`analyze`; fork-only.
        """
        return self._require_live_for_contact_query(
            "ladruno_mortar_penetration").ladruno_mortar_penetration()

    def ladruno_mortar_tie_residual(self) -> float:
        """Max weighted relative-displacement bond residual over all mortar
        **tie** slave nodes (fork ``ladrunoMortarTieResidual``, ADR-41 C4).

        The tie's ALM convergence measure, the counterpart of
        :meth:`ladruno_mortar_penetration` for ``tie=True``. ``0.0`` with no
        tie declared. Requires a prior live :meth:`analyze`; fork-only.
        """
        return self._require_live_for_contact_query(
            "ladruno_mortar_tie_residual").ladruno_mortar_tie_residual()
