"""Model archives the workspace tests judge sets against (ADR 0132 D3).

A plain module, imported as ``tests.results.user._workspace_helpers``; the folder
has no ``__init__.py`` (see ``_user_helpers``).

:func:`frame` is the worked case of ``architecture/h5-schema.md`` ("The
verdict"): a frame whose base nodes 1 to 3 sit at z = 0 and roof nodes 31 to 33
at z = 9, with a ground node 103 on node 3's position (12, 0, 0) and two
zero-length springs 501 and 502 on the pair 3-103.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import h5py
import numpy as np

FEM_HASH = "f" * 32
MODEL_HASH = "a" * 32

#: Column x positions and storey heights of the frame.
XS = (0.0, 6.0, 12.0)
ZS = (0.0, 3.0, 6.0, 9.0)


def frame_nodes() -> dict[int, tuple[float, float, float]]:
    """Node ``10 * level + column``: 1-3 at z = 0, ..., 31-33 at z = 9; ground 103."""
    nodes = {10 * lvl + col + 1: (x, 0.0, z) for lvl, z in enumerate(ZS) for col, x in enumerate(XS)}
    nodes[103] = (12.0, 0.0, 0.0)
    return nodes


def frame_columns() -> list[tuple[int, int]]:
    """The nine column segments, as node pairs, in FEM order (FEM ids 1 to 9)."""
    return [(10 * lvl + col + 1, 10 * (lvl + 1) + col + 1) for lvl in range(3) for col in range(3)]


def write_model(
    path: Path,
    *,
    nodes: Mapping[int, tuple[float, float, float]],
    fem_hash: Optional[str] = FEM_HASH,
    model_hash: Optional[str] = MODEL_HASH,
    lines: Optional[list[tuple[int, int]]] = None,
    beam_tags: Optional[list[int]] = None,
    springs: Optional[list[int]] = None,
    spring_pair: tuple[int, int] = (3, 103),
    opensees: bool = True,
) -> Path:
    """A model archive holding what the verdict reads, laid out as ``ops.h5`` writes it.

    ``lines`` go to ``/elements/line2`` with FEM ids 1..n; ``beam_tags[i]`` is the
    OpenSees tag of line ``i``; ``springs`` are minted zero-length tags on
    ``spring_pair`` (``fem_eids = -1``, the pair in ``inline_connectivity``).
    """
    lines = frame_columns() if lines is None else lines
    beam_tags = list(range(201, 201 + len(lines))) if beam_tags is None else beam_tags
    springs = [501, 502] if springs is None else springs
    with h5py.File(path, "w") as f:
        lineage = f.create_group("meta/lineage")
        if fem_hash is not None:
            lineage.attrs["fem_hash"] = fem_hash
        if model_hash is not None:
            lineage.attrs["model_hash"] = model_hash
        ids = np.array(sorted(nodes), dtype="i8")
        f.create_dataset("nodes/ids", data=ids)
        f.create_dataset("nodes/coords", data=np.array([nodes[i] for i in ids], dtype="f8"))
        line = f.create_group("elements/line2")
        line.create_dataset("ids", data=np.arange(1, len(lines) + 1, dtype="i8"))
        line.create_dataset("connectivity", data=np.array(lines, dtype="i8").reshape(-1, 2))
        if opensees:
            meta = f.create_group("opensees/element_meta")
            beam = meta.create_group("elasticBeamColumn")
            beam.create_dataset("ids", data=np.array(beam_tags, dtype="i8"))
            beam.create_dataset("fem_eids", data=np.arange(1, len(lines) + 1, dtype="i8"))
            if springs:
                zl = meta.create_group("zeroLength")
                zl.create_dataset("ids", data=np.array(springs, dtype="i8"))
                zl.create_dataset("fem_eids", data=np.full(len(springs), -1, dtype="i8"))
                inline = zl.create_dataset("inline_connectivity", (len(springs),),
                                           dtype=h5py.vlen_dtype(np.dtype("i8")))
                for i in range(len(springs)):
                    inline[i] = np.array(spring_pair, dtype="i8")
    return path


def frame(path: Path, **kw) -> Path:
    """The worked-case frame; ``nodes=`` overrides positions, the rest as :func:`write_model`."""
    nodes = frame_nodes()
    nodes.update(kw.pop("nodes", {}))
    return write_model(path, nodes=nodes, **kw)


def column_centroid(k: int) -> tuple[float, float, float]:
    a, b = frame_columns()[k]
    n = frame_nodes()
    return tuple((np.array(n[a]) + np.array(n[b])) / 2.0)  # type: ignore[return-value]
