"""apeGmsh._kernel._label_prefix — pure label-PG prefix predicates.

Relocated **verbatim** from ``apeGmsh.core.Labels`` (lines 68, 71-87)
as part of selection-unification-v2 P1-K (HT4): ``core/Labels.py`` is
NOT wholesale-relocatable (it eagerly ``import gmsh`` and hosts a
gmsh-driven class), but these four prefix helpers are pure ``str``
logic with zero dependencies.

The internal ``_label:`` prefix distinguishes label PGs (geometry-time
bookkeeping, invisible to the solver) from user PGs (solver-facing).

``apeGmsh.core.Labels`` keeps these names importable via a downward
re-export (Option-i — keeps the byte-unchanged contract/pin tests,
which import ``add_prefix`` from ``apeGmsh.core.Labels``, working).
"""

from __future__ import annotations

# The internal prefix that distinguishes label PGs from user PGs.
LABEL_PREFIX = "_label:"


def is_label_pg(name: str) -> bool:
    """Return True if *name* is an internal label PG name."""
    return name.startswith(LABEL_PREFIX)


def strip_prefix(name: str) -> str:
    """Strip the ``_label:`` prefix from a label PG name."""
    if name.startswith(LABEL_PREFIX):
        return name[len(LABEL_PREFIX):]
    return name


def add_prefix(name: str) -> str:
    """Add the ``_label:`` prefix to a bare label name."""
    if name.startswith(LABEL_PREFIX):
        return name
    return LABEL_PREFIX + name
