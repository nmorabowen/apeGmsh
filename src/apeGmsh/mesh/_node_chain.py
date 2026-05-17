"""``NodeChain`` — the point-family chainable over FEM nodes.

Imports only the leaf ``apeGmsh._chain`` and numpy.  It does **not**
import ``apeGmsh.core`` (keeps the eager/deferred import polarity
intact — see ``docs/plans/selection-unification.md`` §3).  The engine
is duck-typed: any object exposing ``.ids`` (ndarray of node ids) and
``.coords`` ((N,3) float64) works (``NodeComposite`` satisfies this).
"""

from __future__ import annotations

import numpy as np

from .._chain import SelectionChain


class NodeChain(SelectionChain):
    """Daisy-chainable node selection (point family)."""

    FAMILY = "point"

    # ── coordinate access ───────────────────────────────────
    def _row_map(self) -> dict:
        cache = getattr(self._engine, "_apegmsh_chain_idrow", None)
        if cache is None:
            ids = np.asarray(self._engine.ids)
            cache = {int(n): i for i, n in enumerate(ids)}
            setattr(self._engine, "_apegmsh_chain_idrow", cache)
        return cache

    def _coords_of(self, atoms: tuple) -> np.ndarray:
        coords = np.asarray(self._engine.coords, dtype=np.float64)
        rm = self._row_map()
        if not atoms:
            return np.empty((0, 3), dtype=np.float64)
        rows = [rm[int(a)] for a in atoms]
        return coords[rows]

    # ── point-family spatial hooks (numpy kernel) ───────────
    def _spatial_box(self, atoms, lo, hi, *, inclusive: bool) -> tuple:
        if not atoms:
            return ()
        c = self._coords_of(atoms)
        lo = np.asarray(lo, dtype=np.float64).reshape(3)
        hi = np.asarray(hi, dtype=np.float64).reshape(3)
        if inclusive:                       # closed [lo, hi]
            mask = np.all((c >= lo) & (c <= hi), axis=1)
        else:                               # half-open [lo, hi)  (canonical)
            mask = np.all((c >= lo) & (c < hi), axis=1)
        return tuple(a for a, k in zip(atoms, mask) if k)

    def _spatial_sphere(self, atoms, center, radius: float) -> tuple:
        r = float(radius)
        if r < 0:
            raise ValueError(f"radius must be non-negative, got {r}.")
        if not atoms:
            return ()
        c = self._coords_of(atoms)
        ctr = np.asarray(center, dtype=np.float64).reshape(3)
        mask = np.linalg.norm(c - ctr, axis=1) <= r          # closed ball
        return tuple(a for a, k in zip(atoms, mask) if k)

    def _spatial_plane(self, atoms, point, normal, tol: float) -> tuple:
        t = float(tol)
        if t < 0:
            raise ValueError(f"tolerance must be non-negative, got {t}.")
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        nn = np.linalg.norm(n)
        if nn == 0:
            raise ValueError("normal vector has zero length.")
        if not atoms:
            return ()
        c = self._coords_of(atoms)
        p = np.asarray(point, dtype=np.float64).reshape(3)
        dist = np.abs((c - p) @ (n / nn))
        return tuple(a for a, k in zip(atoms, dist <= t) if k)

    # ── terminal ────────────────────────────────────────────
    def _materialize(self) -> np.ndarray:
        """Spike terminal: the selected node-id array.

        S3 will return the existing ``NodeResult`` here; for the S0a
        import-safety spike a plain id array is sufficient.
        """
        return np.asarray(self._items, dtype=np.int64)
