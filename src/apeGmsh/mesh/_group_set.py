"""
_group_set — Named group sets for physical groups and labels.
==============================================================

Provides ``NamedGroupSet`` (shared base) and its two concrete tiers:

* ``PhysicalGroupSet`` — solver-facing physical groups (Tier 2)
* ``LabelSet``         — geometry-time internal labels (Tier 1)

Both store a ``{(dim, tag): info_dict}`` structure and expose
name-first, direct-array access with ``object`` dtype coercion
applied once at init time (not per-call).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    import pandas as pd


# =====================================================================
# Helper
# =====================================================================

def _to_object(arr: ndarray) -> ndarray:
    """Cast to object dtype so iteration yields Python ``int``."""
    return np.asarray(arr).astype(object)


# =====================================================================
# NamedGroupSet — shared base
# =====================================================================

class NamedGroupSet:
    """Base class for :class:`PhysicalGroupSet` and :class:`LabelSet`.

    Stores ``{(dim, tag): info_dict}`` where each ``info_dict`` has at
    minimum ``'name'``, ``'node_ids'``, ``'node_coords'``, and
    optionally ``'element_ids'``, ``'connectivity'`` for dim >= 1.

    Provides name-first, direct-array access.  ``object`` dtype is
    applied once during ``__init__`` (not per-call).

    Parameters
    ----------
    groups : dict
        ``{(dim, pg_tag): {'name', 'node_ids', 'node_coords',
        'element_ids'?, 'connectivity'?}}``
    """

    def __init__(self, groups: dict[tuple[int, int], dict]) -> None:
        # Apply dtype coercion once at construction time
        self._groups: dict[tuple[int, int], dict] = {}
        for key, info in groups.items():
            coerced = dict(info)
            coerced['node_ids'] = _to_object(info['node_ids'])
            coerced['node_coords'] = np.asarray(
                info['node_coords'], dtype=np.float64)
            if 'element_ids' in info:
                coerced['element_ids'] = _to_object(info['element_ids'])
            # Per-type groups (new extraction format)
            if 'groups' in info:
                coerced['groups'] = info['groups']
            # Legacy flat connectivity (keep if present)
            if 'connectivity' in info:
                coerced['connectivity'] = _to_object(info['connectivity'])
            self._groups[key] = coerced

        self._name_index: dict[str, tuple[int, int]] | None = None

    # ── Internal resolution ──────────────────────────────────

    def _build_name_index(self) -> dict[str, tuple[int, int]]:
        if self._name_index is None:
            import logging
            _log = logging.getLogger(__name__)
            idx: dict[str, tuple[int, int]] = {}
            for (d, t) in sorted(self._groups.keys()):
                name = self._groups[(d, t)].get('name', '')
                if not name:
                    continue
                if name in idx:
                    existing = idx[name]
                    _log.warning(
                        "Duplicate group name %r: (dim=%d, tag=%d) "
                        "shadows (dim=%d, tag=%d). Use (dim, tag) "
                        "tuple for explicit access.",
                        name, d, t, existing[0], existing[1])
                    continue
                idx[name] = (d, t)
            self._name_index = idx
        return self._name_index

    def _resolve(self, target) -> dict:
        """Resolve *target* (str name, int tag, or (dim, tag) tuple)
        to the internal info dict.  Raises KeyError on miss."""
        if isinstance(target, str):
            idx = self._build_name_index()
            key = idx.get(target)
            if key is None:
                raise KeyError(
                    f"No group named {target!r}. "
                    f"Available: {self.names()}")
            return self._groups[key]
        if isinstance(target, tuple):
            info = self._groups.get(target)
            if info is None:
                raise KeyError(
                    f"No group {target}. Available: {self.get_all()}")
            return info
        # int tag — search across dims (lowest first)
        for (d, t) in sorted(self._groups.keys()):
            if t == int(target):
                return self._groups[(d, t)]
        raise KeyError(
            f"No group with tag {target}. Available: {self.get_all()}")

    # ── Name-first, direct-array access ──────────────────────

    def node_ids(self, target) -> ndarray:
        """Node IDs for a group.

        Parameters
        ----------
        target : str, int, or (dim, tag)
            Group name, tag, or ``(dim, tag)`` tuple.

        Returns
        -------
        ndarray(N,) — object dtype (yields Python ``int``).
        """
        return self._resolve(target)['node_ids']

    def node_coords(self, target) -> ndarray:
        """Node coordinates for a group.

        Returns
        -------
        ndarray(N, 3) — float64.
        """
        return self._resolve(target)['node_coords']

    def element_ids(self, target) -> ndarray:
        """Element IDs for a group (dim >= 1 only).

        Returns
        -------
        ndarray(E,) — object dtype.

        Raises
        ------
        ValueError
            If the group has no element data (dim=0 groups).
        """
        info = self._resolve(target)
        eids = info.get('element_ids')
        if eids is None:
            name = info.get('name', str(target))
            raise ValueError(
                f"Group '{name}' has no element data "
                f"(element data is only available for dim >= 1).")
        return eids

    def connectivity(self, target) -> ndarray:
        """Element connectivity for a group (dim >= 1 only).

        Returns
        -------
        ndarray(E, npe) — object dtype.

        If the group has mixed element types, rows with fewer nodes
        are padded with ``-1``.
        """
        info = self._resolve(target)
        conn = info.get('connectivity')
        if conn is not None:
            return conn
        # Build from per-type groups if available
        grps = info.get('groups')
        if grps:
            blocks = [g['conn'] for g in grps.values()
                      if g['conn'].size > 0]
            if blocks:
                max_npe = max(b.shape[1] for b in blocks)
                padded = []
                for b in blocks:
                    if b.shape[1] < max_npe:
                        pad = np.full(
                            (b.shape[0], max_npe - b.shape[1]),
                            -1, dtype=b.dtype)
                        padded.append(np.hstack([b, pad]))
                    else:
                        padded.append(b)
                return _to_object(np.vstack(padded))
        name = info.get('name', str(target))
        raise ValueError(
            f"Group '{name}' has no element data "
            f"(element data is only available for dim >= 1).")

    # ── Queries ──────────────────────────────────────────────

    def names(self, dim: int = -1) -> list[str]:
        """Return all group names, optionally filtered by dimension."""
        result = []
        for (d, _), info in sorted(self._groups.items()):
            name = info.get('name', '')
            if name and (dim == -1 or d == dim):
                result.append(name)
        return result

    def get_all(self, dim: int = -1) -> list[tuple[int, int]]:
        """Return all groups as ``(dim, tag)`` pairs."""
        if dim == -1:
            return sorted(self._groups.keys())
        return sorted(k for k in self._groups if k[0] == dim)

    def get_name(self, dim: int, tag: int) -> str:
        """Return the name of a group, or ``""`` if unnamed."""
        info = self._groups.get((dim, tag))
        if info is None:
            raise KeyError(
                f"No group (dim={dim}, tag={tag}). "
                f"Available: {self.get_all()}")
        return info.get('name', '')

    def get_tag(self, dim: int, name: str) -> int | None:
        """Look up the tag of a named group.  Returns None if not found."""
        for (d, pg_tag), info in self._groups.items():
            if d == dim and info.get('name', '') == name:
                return pg_tag
        return None

    # ── Dict-like access ─────────────────────────────────────

    def __contains__(self, name: str) -> bool:
        """Check if a group name exists."""
        idx = self._build_name_index()
        return name in idx

    def __getitem__(self, name: str) -> dict:
        """Get the info dict for a group by name.

        Raises KeyError if not found.
        """
        return self._resolve(name)

    # ── Display ──────────────────────────────────────────────

    def summary(self) -> "pd.DataFrame":
        """DataFrame describing every group.

        Returns
        -------
        pd.DataFrame indexed by ``(dim, pg_tag)`` with columns:
        ``name``, ``n_nodes``, ``n_elems``.
        """
        import pandas as pd

        rows: list[dict] = []
        for (dim, pg_tag), info in sorted(self._groups.items()):
            elem_ids = info.get('element_ids')
            rows.append({
                'dim':     dim,
                'pg_tag':  pg_tag,
                'name':    info.get('name', ''),
                'n_nodes': len(info['node_ids']),
                'n_elems': len(elem_ids) if elem_ids is not None else 0,
            })

        if not rows:
            return pd.DataFrame(
                columns=['dim', 'pg_tag', 'name', 'n_nodes', 'n_elems'])

        return (pd.DataFrame(rows)
                .set_index(['dim', 'pg_tag'])
                .sort_index())

    # ── Dunder ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._groups)

    def __bool__(self) -> bool:
        return bool(self._groups)

    def __iter__(self):
        return iter(sorted(self._groups.keys()))


# =====================================================================
# Concrete tiers
# =====================================================================

class PhysicalGroupSet(NamedGroupSet):
    """Snapshot of solver-facing physical groups.

    Accessed via ``fem.nodes.physical`` / ``fem.elements.physical``
    (shared reference) and indirectly via ``fem.nodes.get(pg="Base")``.
    """

    def __repr__(self) -> str:
        if not self._groups:
            return "PhysicalGroupSet(empty)"
        return f"PhysicalGroupSet({len(self._groups)} groups)"


class LabelSet(NamedGroupSet):
    """Snapshot of geometry-time labels (Tier 1).

    Accessed via ``fem.nodes.labels`` / ``fem.elements.labels``
    (shared reference) and indirectly via
    ``fem.nodes.get(label="col.web")``.
    """

    def __repr__(self) -> str:
        if not self._groups:
            return "LabelSet(empty)"
        return f"LabelSet({len(self._groups)} labels)"
