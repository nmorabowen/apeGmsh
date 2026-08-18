"""The §8 Gauss address encoding, in ONE place.

ADR 0098 §8's Gauss target is ``(element_id, gp_index)`` where
``gp_index`` counts integration points **within that one element**, in
the slab's own row order. Three consumers of this slice have to agree
on it exactly:

* :mod:`._realize` turns a Gauss slab into the pick targets a click
  resolves against, and into the highlight drawn over them;
* :mod:`._select_all` turns a physical group / element type / element
  into the Gauss membership the outline writes into the selection;
* :mod:`._plots` turns a Gauss source back into one column of a slab.

They came from one module until S4-2 needed the second and third. A
divergence between any two would be silent and wrong in the worst way
— the outline would select points the pane highlights elsewhere, or a
plot would draw a different integration point than the one clicked —
so the encoding lives here rather than being restated per consumer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results


def probe_component(results: "Results", stage_id: str) -> Optional[str]:
    """The component read purely for its integration-point ADDRESSES.

    Any recorded Gauss component answers the same ``(element_index,
    natural_coords)`` pair, so the first alphabetically is read as the
    address probe and its values are discarded. ``None`` when the
    stage records no Gauss composite — there are no integration points
    to address, which is a disabled control, never an error.
    """
    try:
        recorded = results.inspect.components(stage=stage_id).get("gauss", ())
    except Exception:
        return None
    tokens = sorted(str(t) for t in recorded)
    return tokens[0] if tokens else None


def gp_index_within_element(element_index: "np.ndarray") -> "np.ndarray":
    """Each slab row's index WITHIN its element, in slab row order.

    The slab carries no such column (unlike ``FiberSlab``), so it is
    derived: a STABLE sort groups each element's rows while preserving
    their slab order, and the position inside the group is the index.
    That makes it the same ordering a reader resolves with
    ``where(element_index == element_id)[gp_index]``.
    """
    element_index = np.asarray(element_index, dtype=np.int64)
    n = int(element_index.size)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(element_index, kind="stable")
    grouped = element_index[order]
    starts = np.flatnonzero(
        np.r_[True, grouped[1:] != grouped[:-1]]
    ).astype(np.int64)
    counts = np.diff(np.r_[starts, np.int64(n)])
    within = np.arange(n, dtype=np.int64) - np.repeat(starts, counts)
    out = np.empty(n, dtype=np.int64)
    out[order] = within
    return out


__all__ = ["gp_index_within_element", "probe_component"]
