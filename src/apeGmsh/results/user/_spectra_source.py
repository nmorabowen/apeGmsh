"""The ``/spectra`` cases as the evaluator's spectrum source (ADR 0130 D5, A6, A7).

:class:`UserSpectra` adapts the cases :meth:`UserFile.spectra` reads to the
``SpectraProvider`` seam (:mod:`._states`) through which a combination reads its
``spectrum`` and ``reference`` terms. ``Results.combinations()`` passes one for
the set's ``.user.h5`` unless it is given another.

**Which result and column a quantity reads.** A quantity is ``(level,
component)``, as ``UserCombinations.evaluate`` names it:

=================  ==============================================================
``level``          read from the case
=================  ==============================================================
``nodes``          the one ``ON_NODES`` column whose recorder spelling the
                   readers translate to ``component`` (``DISPLACEMENT`` and ``Uy``
                   are ``displacement_y``), labelled ``component``, so that it joins
                   a stage term's column
``domain``         the ``ON_DOMAIN`` result labelled ``component`` (an extractor's
                   name, or ``base_shear``), every column, labelled as the case
                   stores them (``base_shear``: ``1..ndm``, the directions), on one
                   row
``elements``,      refused: a spectra 1.0.0 case holds no element result
``gauss``,
``line_stations``
=================  ==============================================================

A quantity that no result of the case holds, or that two hold, refuses
``unresolved``. ``UserCombinations.evaluate`` takes the recorded levels only, so
a domain quantity is read through this provider directly for now.

**Refusals.** A missing ``.user.h5`` or case is ``unresolved``; a case or a
``/spectra`` zone the reader refused is ``refused``. The zone is stale when the
model hashes differ (A6), and a stale zone's terms refuse ``stale`` unless the
provider was made with ``allow_stale=True``. The file is read once, at the first
spectrum term, and the provider keeps what it read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ...opensees._internal.schema_version import SchemaVersionError
from ..readers._mpco_translation import canonical_node_component
from ._descriptor import StateAddress
from ._errors import UserFileError
from ._evaluate import Case, CombinationRefused, Field
from ._reader import UserFile
from ._spectra import CaseQuantity, SpectraIndex, SpectrumCase
from ._staleness import staleness
from ._states import LEVELS, domain_keys, node_keys

__all__ = ["UserSpectra"]

#: Every level a quantity may name here: the recorded ones, and the domain.
QUANTITY_LEVELS = (*LEVELS, "domain")


def _label(address: StateAddress) -> str:
    """The name a refusal of one term carries, as ``read_state`` names a lone state."""
    return f"<{address.kind} {address.source}>"


class UserSpectra:
    """The ``/spectra`` of one ``.user.h5``, read per quantity (module docstring).

    ``user_h5`` is ``None`` for a set with no ``.user.h5``: every term then
    refuses ``unresolved``. ``model_h5`` is the set's model archive, which A6
    compares the zone's hashes with; without one the zone is stale.
    """

    def __init__(
        self,
        user_h5: "Optional[str | Path]",
        *,
        model_h5: "Optional[str | Path]" = None,
        allow_stale: bool = False,
    ) -> None:
        self.user_h5 = None if user_h5 is None else Path(user_h5)
        self.model_h5 = None if model_h5 is None else Path(model_h5)
        self.allow_stale = bool(allow_stale)
        self._index: Optional[SpectraIndex] = None
        self._stale: Optional[str] = None

    # ---- the file, read once -------------------------------------------

    def _load(self, who: str) -> SpectraIndex:
        if self._index is None:
            if self.user_h5 is None:
                raise CombinationRefused(
                    who, "unresolved", "this set has no .user.h5, so it holds no /spectra case")
            try:
                user = UserFile(self.user_h5)
                index = user.spectra()
            except (UserFileError, SchemaVersionError) as exc:
                raise CombinationRefused(who, "refused", f"/spectra cannot be read: {exc}") from exc
            if "spectra" in user.zones:
                zone = staleness(user, model_h5=self.model_h5).zones["spectra"]
                self._stale = zone.reason if zone.stale else None
            self._index = index
        if self._stale is not None and not self.allow_stale:
            raise CombinationRefused(
                who, "stale", f"/spectra is stale ({self._stale}); make the provider with "
                "allow_stale=True to read it (ADR 0130 A6)")
        return self._index

    def _case(self, name: str, who: str) -> SpectrumCase:
        index = self._load(who)
        try:
            return index.by_name(name)
        except KeyError as exc:
            reason = "refused" if any(r.name == name for r in index.refused) else "unresolved"
            raise CombinationRefused(who, reason, str(exc.args[0])) from exc

    # ---- the quantity --------------------------------------------------

    @staticmethod
    def _locate(
        case: SpectrumCase, quantity: tuple[str, str], who: str,
    ) -> tuple[CaseQuantity, CaseQuantity, Optional[int]]:
        """``(per-mode result, its reference, column)``; the column is ``None`` for
        a domain result, which is read whole."""
        level, component = quantity
        if level not in QUANTITY_LEVELS:
            raise ValueError(f"level {level!r} is not one of {QUANTITY_LEVELS}")
        if level == "domain":
            hits = [(q, None) for q in case.results
                    if q.target == "domain" and q.label == component]
        elif level == "nodes":
            hits = [(q, col) for q in case.results if q.target == "nodes"
                    for col, label in enumerate(q.components)
                    if canonical_node_component(q.label, label) == component]
        else:
            raise CombinationRefused(
                who, "unresolved", f"case {case.name!r} holds no {level} result: a "
                "spectra 1.0.0 case holds ON_NODES and ON_DOMAIN only")
        if len(hits) != 1:
            held = [f"{q.target}/{q.label}({','.join(q.components)})" for q in case.results]
            what = "no" if not hits else f"{len(hits)} results for"
            raise CombinationRefused(
                who, "unresolved", f"case {case.name!r} holds {what} {level} quantity "
                f"{component!r}; it holds {held}")
        result, column = hits[0]
        reference = next(q for q in case.reference
                         if (q.target, q.label) == (result.target, result.label))
        return result, reference, column

    @staticmethod
    def _field(q: CaseQuantity, row: int, column: Optional[int], component: str) -> Field:
        data = np.asarray(q.data, dtype="f8")[row]  # (n, c)
        if column is None:
            return Field(domain_keys(), q.components, data)
        return Field(node_keys(q.ids), (component,), data[:, column:column + 1])

    # ---- the SpectraProvider seam ---------------------------------------

    def field(self, address: StateAddress, quantity: tuple[str, str]) -> Field:
        """One signed increment (``spectrum``, ``mode = k``) or the reference state."""
        who = _label(address)
        if address.kind not in ("spectrum", "reference"):
            raise CombinationRefused(who, "bad-term", f"a {address.kind} term is not in /spectra")
        if address.kind == "spectrum" and address.mode == 0:
            raise CombinationRefused(
                who, "bad-term", "a whole case is not one state; combine it with a rule")
        case = self._case(address.source, who)
        result, reference, column = self._locate(case, quantity, who)
        if address.kind == "reference":
            return self._field(reference, 0, column, quantity[1])
        rows = np.flatnonzero(np.asarray(case.mode) == address.mode)
        if rows.size != 1:
            raise CombinationRefused(
                who, "unresolved", f"case {case.name!r} has no mode {address.mode}: its modes "
                f"are {np.asarray(case.mode).tolist()}")
        return self._field(result, int(rows[0]), column, quantity[1])

    def case(self, name: str, quantity: tuple[str, str]) -> Case:
        """The whole case: per mode, the signed increment of the quantity."""
        who = f"<spectrum {name}>"
        found = self._case(name, who)
        result, _, column = self._locate(found, quantity, who)
        return Case(
            mode=found.mode, lam=found.lam, sa_period=found.sa_period, damping=found.damping,
            increments=tuple(self._field(result, i, column, quantity[1])
                             for i in range(np.asarray(found.mode).size)),
        )
