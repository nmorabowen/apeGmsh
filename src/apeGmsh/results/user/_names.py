"""The name rule of ADR 0130 A2.

A name is data in ``@name`` or a table column, never an HDF5 link, so real
names such as ``Delta_after (L/480)`` need no escaping. A name is non-empty
NFC UTF-8 with no control characters and no leading or trailing whitespace,
and compares byte-equal.
"""
from __future__ import annotations

from typing import Any

from ...opensees._internal._combination_rules import name_problem

__all__ = ["name_problem", "text_or_empty"]


def text_or_empty(raw: Any) -> str:
    """A stored string as ``str``, or ``""`` when its bytes are not UTF-8.

    ``""`` is refused downstream as an empty name, an unknown rule or a bad
    term, so an undecodable cell refuses its row and not the zone.
    """
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    if isinstance(raw, str):
        try:
            raw.encode("utf-8")
        except UnicodeEncodeError:  # h5py hands undecodable attrs back as surrogates
            return ""
        return raw
    return ""
