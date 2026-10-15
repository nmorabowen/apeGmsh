"""Stand-in ``model=`` for recorder-emit tests (#1711, #2018).

Element-level recorder emits translate FEM ids to OpenSees tags through
the built model's ``(fem_eid, tag)`` pairing, and node-level ones map
each slot to the DOF the node carries it at through the model's ``ndm``
and per-node ndf. Both refuse without a model. These stand-ins carry
only that: real ``ElementRecord`` rows, an ``ndm`` and a ``nodes_ndf``.
"""
from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Iterable, Mapping

from apeGmsh.opensees._internal.typed_records import ElementRecord


class TagMapModel:
    """An ``OpenSeesModel`` stand-in: ``elements()``, ``ndm``,
    ``nodes_ndf`` and the ``lineage.fem_hash`` an emit checks against the
    spec's ``fem_snapshot_id``."""

    def __init__(
        self,
        fem_to_ops: Mapping[int, int],
        fem_hash: str,
        *,
        ndm: int = 3,
        nodes_ndf: Mapping[int, int] | None = None,
    ) -> None:
        self.lineage = SimpleNamespace(fem_hash=fem_hash)
        self.ndm = int(ndm)
        self.nodes_ndf = MappingProxyType(
            {int(n): int(d) for n, d in (nodes_ndf or {}).items()})
        self._records = tuple(
            ElementRecord(type_token="stub", tag=int(tag), args=(),
                          connectivity=(), fem_eid=int(fem))
            for fem, tag in fem_to_ops.items()
        )

    def elements(self) -> tuple[ElementRecord, ...]:
        return self._records


def identity_model(
    ids: Iterable[int], fem_hash: str, *,
    ndm: int = 3, nodes_ndf: Mapping[int, int] | None = None,
) -> TagMapModel:
    """Every FEM id is its own OpenSees tag (an ``element_tags="fem"``
    build, or a hand-written deck that numbers elements by FEM id)."""
    return TagMapModel({int(i): int(i) for i in ids}, fem_hash,
                       ndm=ndm, nodes_ndf=nodes_ndf)


def node_model(
    nodes: Iterable[int], fem_hash: str, *, ndf: int = 6, ndm: int = 3,
    elements: Mapping[int, int] | None = None,
) -> TagMapModel:
    """Every node in ``nodes`` at one ``(ndm, ndf)``: by default the 3-D
    frame layout, where a slot's DOF is its index in
    ``(x, y, z, rx, ry, rz)``."""
    return TagMapModel(dict(elements or {}), fem_hash, ndm=ndm,
                       nodes_ndf={int(n): int(ndf) for n in nodes})
