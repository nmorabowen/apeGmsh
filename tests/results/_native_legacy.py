"""A legacy native ``results.h5`` with several partitions in one stage.

``NativeWriter`` refuses a second partition per stage (ADR 0123 D5), but
files written before that refusal, or by a third-party caller of the old
API, can hold several, and the reader keeps opening them (D4, the #2360
rule). The tests of that read build such files here: the writer makes
the envelope (root attributes and an empty stage), then every partition
is added with raw h5py, exactly as an older writer laid it out
(``architecture/h5-schema.md``, "Partitions").

The partitions land after the writer closed the file, so the file's
``/meta/lineage`` ``results_hash`` does not cover them. The reader does
not check that hash at open, and no test of this builder relies on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from apeGmsh.results.schema import _native
from apeGmsh.results.writers import NativeWriter

#: ``{partition_id: (node_ids, {component: (T, N) values})}``.
Partitions = Mapping[str, "tuple[Sequence[int], Mapping[str, object]]"]


def write_multi_partition(
    path: Path, partitions: Partitions, *, time: np.ndarray,
    kind: str = _native.KIND_STATIC,
) -> str:
    """Write one stage holding ``partitions`` (nodes only) to ``path``.

    Returns the stage id. Each component must be ``(T, N)``, with ``T``
    the length of ``time`` and ``N`` the partition's node count.
    """
    time = np.asarray(time, dtype=np.float64)
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind=kind, time=time)
        w.end_stage()
    with h5py.File(path, "r+") as f:
        parts = f[_native.partitions_path(sid)]
        for pid, (ids, comps) in partitions.items():
            node_ids = np.asarray(ids, dtype=np.int64)
            nodes = parts.create_group(pid).create_group(_native.GROUP_NODES)
            nodes.create_dataset(_native.DSET_IDS, data=node_ids)
            for name, values in comps.items():
                arr = np.asarray(values, dtype=np.float64)
                if arr.shape != (time.size, node_ids.size):
                    raise ValueError(
                        f"{pid}/{name}: shape {arr.shape}, expected "
                        f"({time.size}, {node_ids.size})."
                    )
                nodes.create_dataset(name, data=arr)
    return sid
