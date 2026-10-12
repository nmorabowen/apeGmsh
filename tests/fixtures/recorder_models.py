"""Stand-in ``model=`` for recorder-emit tests (#1711).

Element-level recorder emits translate FEM ids to OpenSees tags through
the built model's ``(fem_eid, tag)`` pairing and refuse without one.
These stand-ins carry only that pairing, as real ``ElementRecord`` rows.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from apeGmsh.opensees._internal.typed_records import ElementRecord


class TagMapModel:
    """An ``OpenSeesModel`` stand-in exposing ``elements()`` only."""

    def __init__(self, fem_to_ops: Mapping[int, int]) -> None:
        self._records = tuple(
            ElementRecord(type_token="stub", tag=int(tag), args=(),
                          connectivity=(), fem_eid=int(fem))
            for fem, tag in fem_to_ops.items()
        )

    def elements(self) -> tuple[ElementRecord, ...]:
        return self._records


def identity_model(ids: Iterable[int]) -> TagMapModel:
    """Every FEM id is its own OpenSees tag (an ``element_tags="fem"``
    build, or a hand-written deck that numbers elements by FEM id)."""
    return TagMapModel({int(i): int(i) for i in ids})
