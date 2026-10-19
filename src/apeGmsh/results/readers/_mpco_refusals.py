"""Refusals shared by the MPCO element reads (#2173).

MPCO writes each element once, to exactly one bucket of a token. A read
that stitches buckets would return an element twice when a corrupt file
holds its id in two buckets, or twice in one bucket's ``ID``. The reads
refuse that with :class:`MPCOElementInTwoBuckets`.

A leaf module: it imports only numpy, so ``_mpco.py`` and every
``_mpco_*_io.py`` module can import it without a cycle.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy import ndarray


class MPCOElementInTwoBuckets(ValueError):
    """An element id an MPCO read returns sits in two buckets of one
    token, or twice in one bucket. The file is corrupt."""


def refuse_element_in_two_buckets(
    ids_by_path: Mapping[str, ndarray],
    *,
    file: str,
) -> None:
    """Raise :class:`MPCOElementInTwoBuckets` when an id repeats.

    ``ids_by_path`` maps each bucket's HDF5 path to the element ids the
    read returns from it. An id that appears twice, in two buckets or in
    one, refuses. The error names the smallest such id, the paths of the
    first two buckets that hold it (one path for a repeat inside one
    bucket), and ``file``.
    """
    paths = list(ids_by_path)
    parts = [np.asarray(ids_by_path[p], dtype=np.int64).ravel()
             for p in paths]
    total = sum(p.size for p in parts)
    if total < 2:
        return
    all_ids = np.concatenate(parts)
    plain = np.sort(all_ids)
    if not (plain[1:] == plain[:-1]).any():
        return
    # Only a corrupt file gets here: find which buckets hold the id.
    owners = np.repeat(np.arange(len(parts)), [p.size for p in parts])
    order = np.argsort(all_ids, kind="stable")
    sorted_ids = all_ids[order]
    repeat = np.flatnonzero(sorted_ids[1:] == sorted_ids[:-1])
    if repeat.size == 0:
        return
    j = int(repeat[0])
    eid = int(sorted_ids[j])
    first = paths[int(owners[order[j]])]
    second = paths[int(owners[order[j + 1]])]
    where = (
        f"twice in the MPCO bucket {first}" if first == second
        else f"in two MPCO buckets: {first} and {second}"
    )
    n_ids = np.unique(sorted_ids[repeat]).size
    more = (
        f" {n_ids - 1} more element id(s) the read returns repeat too."
        if n_ids > 1 else ""
    )
    raise MPCOElementInTwoBuckets(
        f"{file}: element {eid} appears {where}. MPCO writes each element "
        f"once, to exactly one bucket, so the file is corrupt; the read "
        f"refuses rather than return element {eid} twice.{more}"
    )
