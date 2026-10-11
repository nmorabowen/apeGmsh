"""errors: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations

from typing import Sequence

from ..tag_allocator import TagLawError


class BridgeError(RuntimeError):
    """Build-pipeline error — a primitive's dependency is unregistered,
    a PG is missing, or a fan-out cannot proceed for a structural
    reason. Distinct from :class:`ValueError` (caller error during
    primitive construction)."""


class TagPlanMiss(BridgeError, TagLawError):
    """An emit asked the tag plan for an owner it does not hold.

    The message names the tag family and the owner. A miss means the plan
    was not made for this emit (another model's, or one that dropped the
    owner): the emit reads tags and never mints one, so there is no other
    tag to write (ADR 0114 D4, amended).
    """


def plan_owner(record: object) -> str:
    """How a plan miss names its owner: the record's type and, when it
    carries one, its user-facing ``name``."""
    name = record.name if hasattr(record, "name") else None
    kind = type(record).__name__
    return f"{kind} {name!r}" if isinstance(name, str) and name else kind


def _coverage_miss(
    family: str, planned: "Sequence[object]", records: "Sequence[object]",
) -> str:
    """The first record ``records`` holds that ``planned`` lacks (or the
    first ``planned`` holds that ``records`` lacks), named for a miss."""
    have = {id(r) for r in planned}
    want = {id(r) for r in records}
    lost = [r for r in records if id(r) not in have]
    if lost:
        return (f"the {family} plan holds no {plan_owner(lost[0])} that the "
                "emit of its FEM writes")
    extra = [r for r in planned if id(r) not in want]
    if extra:
        return (f"the {family} plan holds {plan_owner(extra[0])}, which the "
                "emit of its FEM does not write")
    return f"the {family} plan holds a record twice"
