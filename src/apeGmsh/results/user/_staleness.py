"""The staleness table of ADR 0130 A6, read from the file and its set.

| Zone | Stale when |
|---|---|
| ``/meta``, ``/combinations`` | never; a combination with a stale source is listed |
| ``/modes`` | the model hashes differ |
| ``/spectra``, ``/entities`` | the model hashes differ |
| ``/results`` | by ``@basis``: ``results_hash``, or the model hashes |

"Model hashes" compares whichever of ``fem_hash`` and ``model_hash`` both sides
have. What cannot be compared is stale: nothing is drawn as current on no proof.

The per-row ``lambda`` rule of ``/modes`` is not here: the zone has no writer and
no registered version until slice VU6, which owns the row check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from ...opensees._internal.schema_version import SchemaVersionError
from ._errors import UserFileError, ZoneRefused
from ._fingerprint import results_fingerprint
from ._reader import UserFile
from ._writer import model_hashes

__all__ = ["Staleness", "ZoneState", "staleness"]

_MODEL_ZONES = ("modes", "spectra", "entities")
#: A term's data comes from these zones. A ``stage`` or ``mode`` term reads the
#: ``.ladruno`` itself, which is current by definition, so it cannot be stale.
_TERM_ZONE = {"spectrum": "spectra", "reference": "spectra"}


@dataclass(frozen=True)
class ZoneState:
    """Whether one zone or entry is stale, and why."""

    stale: bool
    reason: str = ""


@dataclass(frozen=True)
class Staleness:
    """``zones`` by root name, ``results`` by entry name, stale combinations."""

    zones: Mapping[str, ZoneState] = field(default_factory=dict)
    results: Mapping[str, ZoneState] = field(default_factory=dict)
    combinations: tuple[str, ...] = ()


def _model_state(user: UserFile, model_h5: "Optional[str | Path]") -> ZoneState:
    if model_h5 is None:
        return ZoneState(True, "no model file to compare with")
    try:
        fem_hash, model_hash = model_hashes(model_h5)
    except UserFileError as exc:
        return ZoneState(True, f"the model hashes cannot be read: {exc}")
    if fem_hash != user.fem_hash:
        return ZoneState(True, f"fem_hash differs: {user.fem_hash} here, {fem_hash} in the model")
    if model_hash is not None and user.model_hash is not None and model_hash != user.model_hash:
        return ZoneState(True, f"model_hash differs: {user.model_hash} here, {model_hash} in the model")
    return ZoneState(False)


def _results_state(user: UserFile, results_file: "Optional[str | Path]") -> ZoneState:
    if user.results_hash is None:
        return ZoneState(True, "this file has no results_hash")
    if results_file is None:
        return ZoneState(True, "no results file to compare with")
    try:
        current = results_fingerprint(results_file)
    except UserFileError as exc:
        return ZoneState(True, f"the results file cannot be fingerprinted: {exc}")
    if current != user.results_hash:
        return ZoneState(True, f"results_hash differs: {user.results_hash} here, {current} on disk")
    return ZoneState(False)


def staleness(
    user: "UserFile | str | Path",
    *,
    model_h5: "Optional[str | Path]" = None,
    results_file: "Optional[str | Path]" = None,
) -> Staleness:
    """Apply A6 to ``user`` against the set's model and results files.

    A zone that cannot be read at all is reported stale with the refusal as its
    reason; it never raises here, because this is the listing that says so.
    """
    uf = user if isinstance(user, UserFile) else UserFile(user)
    by_model = _model_state(uf, model_h5)
    by_results = _results_state(uf, results_file)

    zones: dict[str, ZoneState] = {"meta": ZoneState(False)}
    for root in _MODEL_ZONES:
        if root in uf.zones:
            zones[root] = by_model

    entries: dict[str, ZoneState] = {}
    try:
        for entry in uf.results().entries:
            entries[entry.name] = by_results if entry.basis == "results" else by_model
        if "results" in uf.zones:
            zones["results"] = ZoneState(
                any(s.stale for s in entries.values()), "by @basis, per entry")
    except (ZoneRefused, SchemaVersionError) as exc:
        zones["results"] = ZoneState(True, f"refused: {exc}")

    stale_rows: tuple[str, ...] = ()
    try:
        table = uf.combinations(model_h5=model_h5)
        if "combinations" in uf.zones:
            zones["combinations"] = ZoneState(False)
        stale_rows = _stale_combinations(table.rows, zones)
    except (ZoneRefused, SchemaVersionError, OSError) as exc:
        zones["combinations"] = ZoneState(True, f"refused: {exc}")
    return Staleness(zones=zones, results=entries, combinations=stale_rows)


def _stale_combinations(rows, zones: Mapping[str, ZoneState]) -> tuple[str, ...]:
    """Rows with a stale source, directly or through another combination."""
    stale: set[str] = set()
    for row in rows:
        for term in row.terms:
            zone = _TERM_ZONE.get(term.kind)
            if zone is not None and zone in zones and zones[zone].stale:
                stale.add(row.name)
    changed = True
    while changed:
        changed = False
        for row in rows:
            if row.name not in stale and any(
                t.kind == "combination" and t.source in stale for t in row.terms
            ):
                stale.add(row.name)
                changed = True
    return tuple(r.name for r in rows if r.name in stale)
