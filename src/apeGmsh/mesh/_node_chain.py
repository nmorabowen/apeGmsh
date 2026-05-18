"""``NodeChain`` — the point-family chainable over FEM nodes.

Imports only the leaf ``apeGmsh._chain`` and numpy.  It does **not**
import ``apeGmsh.core`` (keeps the eager/deferred import polarity
intact — see ``docs/plans/selection-unification.md`` §3).  The engine
is duck-typed: any object exposing ``.ids`` (ndarray of node ids) and
``.coords`` ((N,3) float64) works (``NodeComposite`` satisfies this).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._chain import SelectionChain

if TYPE_CHECKING:
    # Type-only: the runtime import stays deferred inside _materialize
    # (TYPE_CHECKING is False at runtime, so the FEMData <-> _node_chain
    # load cycle the docstring describes is not reopened).
    from .FEMData import NodeResult


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
    def result(self) -> "NodeResult":
        return self._materialize()

    def _materialize(self) -> "NodeResult":
        """The selected nodes as the **existing** ``NodeResult``.

        Identical type/shape to what ``fem.nodes.get(...)`` returns
        today (object-dtype ids + ``(N, 3)`` float64 coords), built from
        the chain's selected node ids and their coordinates (reusing
        :meth:`_coords_of`, which maps atoms -> engine coord rows).

        ``NodeResult`` is imported **deferred** (inside this method, not
        at module top): ``mesh/FEMData.py`` defines ``NodeResult`` *and*
        ``NodeComposite`` (which imports this module via a function-body
        deferred import), so a module-level import here would close a
        load-time ``_node_chain`` <-> ``FEMData`` cycle.  This module
        therefore keeps importing only the package-root leaf
        ``apeGmsh._chain`` + numpy at load time (see
        ``tests/test_import_dag_polarity.py``).
        """
        from .FEMData import NodeResult  # deferred — avoids load cycle

        atoms = self._items
        ids = np.asarray(atoms, dtype=np.int64)
        coords = self._coords_of(atoms)
        return NodeResult(ids, coords)
