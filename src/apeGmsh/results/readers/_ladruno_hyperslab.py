"""Hyperslab reads of a ``.ladruno`` ``DATA[T × nIds × nComp]`` dataset.

The recorder writes every result as one chunked ``DATA`` dataset. Since
fork WP-164 the chunks are about 1 MiB and a large dataset tiles the id
axis, so one entity's history (``DATA[:, k, :]``) touches a few chunks.
Reading it with ``DATA[...]`` first would still pull the whole
``T × nIds × nComp`` array into memory. :func:`read_hyperslab` reads only
the requested time steps, rows and columns.

h5py takes at most one index list per selection, and that list must be
strictly increasing. So the row axis is the only one that may go to h5py
as a list; time and columns go as their bounding range, and the exact
selection (any order, repeats allowed) is taken in numpy afterwards.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    import h5py

# A row list at least this dense within its span is read as the bounding
# range: one contiguous read beats h5py's per-element point selection.
_DENSE_ROWS = 0.5

_Key = Union[slice, ndarray]


def _bounding(idx: ndarray) -> "tuple[slice, Optional[ndarray]]":
    """``(slice over [min, max], positions within it)``; ``None`` = all of it."""
    lo, hi = int(idx.min()), int(idx.max())
    post = idx - lo
    if post.size == hi - lo + 1 and np.array_equal(post, np.arange(post.size)):
        return slice(lo, hi + 1), None
    return slice(lo, hi + 1), post


def _row_key(rows: ndarray) -> "tuple[_Key, Optional[ndarray]]":
    """h5py key for the row axis, and the numpy re-selection to apply."""
    uniq, inverse = np.unique(rows, return_inverse=True)
    span = int(uniq[-1]) - int(uniq[0]) + 1
    if uniq.size / span >= _DENSE_ROWS:
        key, post = _bounding(uniq)
        if post is None and uniq.size == rows.size:
            return key, None
        base = np.arange(uniq.size) if post is None else post
        return key, base[inverse]
    if uniq.size == rows.size:
        return uniq, None  # already strictly increasing: rows == uniq
    return uniq, inverse


def _read_block(ds: Any, key: tuple) -> ndarray:
    """The one place that touches the dataset (tests spy on it)."""
    return np.asarray(ds[key], dtype=np.float64)


def read_hyperslab(
    ds: "h5py.Dataset",
    t_idx: ndarray,
    rows: "Optional[ndarray]" = None,
    cols: "Optional[ndarray]" = None,
) -> ndarray:
    """``DATA[t_idx][:, rows][:, :, cols]`` without reading the rest.

    ``t_idx``, ``rows`` and ``cols`` are integer index arrays in any
    order (repeats allowed); ``None`` for ``rows`` or ``cols`` means all
    of that axis. Returns ``(len(t_idx), n_rows, n_cols)`` float64,
    equal to the same selection taken from a full read.
    """
    _, R_all, C_all = (int(n) for n in ds.shape)
    t = np.asarray(t_idx, dtype=np.int64).ravel()
    r = None if rows is None else np.asarray(rows, dtype=np.int64).ravel()
    c = None if cols is None else np.asarray(cols, dtype=np.int64).ravel()
    out_shape = (
        t.size,
        R_all if r is None else r.size,
        C_all if c is None else c.size,
    )
    if 0 in out_shape:
        return np.empty(out_shape, dtype=np.float64)

    t_key, t_post = _bounding(t)
    r_key: _Key
    r_post: Optional[ndarray]
    if r is None:
        r_key, r_post = slice(None), None
    else:
        r_key, r_post = _row_key(r)
    c_key: slice
    c_post: Optional[ndarray]
    if c is None:
        c_key, c_post = slice(None), None
    else:
        c_key, c_post = _bounding(c)

    block = _read_block(ds, (t_key, r_key, c_key))
    if t_post is not None:
        block = block[t_post]
    if r_post is not None:
        block = block[:, r_post]
    if c_post is not None:
        block = block[:, :, c_post]
    return block


__all__ = ["read_hyperslab"]
