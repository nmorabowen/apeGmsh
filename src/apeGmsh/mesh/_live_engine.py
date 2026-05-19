"""``_LiveMeshEngine`` + ``engine_for`` — the per-(ms, level, dim)
live-mesh selection engine adapter.

Relocated **verbatim** (selection-unification v2 P3-R / §6.3 §3) from
the now-deleted ``mesh/_mesh_selection_chain.py``.  The
``MeshSelectionChain`` class itself is removed with that module; this
engine adapter survives because ``MeshSelectionSet`` builds it via
:func:`engine_for`.  Pure leaf — ``typing`` only (no ``numpy`` /
``_kernel`` / ``SelectionChain``) → adds **no** new
``tests/test_import_dag_polarity.py`` BASELINE triple (the 4 chain
triples are removed −4/+0 in the same P3-R commit).
"""

from __future__ import annotations

from typing import Any

#: The two levels a live-mesh selection's atoms can live at.
VALID_LEVELS = ("node", "element")


class _LiveMeshEngine:
    """Opaque back-reference a live-mesh selection is bound to.

    Holds only what the chain needs and a level discriminator:

    * ``ms`` — the spawning
      :class:`~apeGmsh.mesh.MeshSelectionSet.MeshSelectionSet`.  The
      chain reaches the **live mesh exclusively through** its existing
      ``_get_mesh_nodes`` / ``_get_mesh_elements`` methods (the same
      path ``add_nodes`` / ``add_elements`` use); the chain never
      touches ``gmsh`` itself.
    * ``level`` — ``"node"`` or ``"element"``.
    * ``dim``  — element dimension (1/2/3) when ``level == "element"``;
      ``0`` (unused) for the node level.

    Identity (not value) is what the base
    :meth:`SelectionChain._compatible` compares, so set-algebra is loud
    across two differently-bound live-mesh selections — same contract
    as the mesh / results chains.
    """

    # The two ``_apegmsh_lm_*`` slots are lazily-populated per-engine
    # coordinate / centroid caches (mirrors NodeChain / ElementChain /
    # ResultChain engine-side memoisation).
    __slots__ = (
        "ms", "level", "dim",
        "_apegmsh_lm_node_idrow", "_apegmsh_lm_elem_centroid",
    )

    def __init__(self, ms: Any, level: str, dim: int) -> None:
        if level not in VALID_LEVELS:
            raise ValueError(
                f"MeshSelectionChain level={level!r} invalid; expected "
                f"one of {VALID_LEVELS}."
            )
        self.ms = ms
        self.level = level
        self.dim = dim
        self._apegmsh_lm_node_idrow = None
        self._apegmsh_lm_elem_centroid = None


#: Attribute the per-``MeshSelectionSet`` engine-adapter map is
#: memoised under (mirrors ``ResultChain``'s per-composite adapter
#: cache, but keyed by ``(level, dim)`` because one
#: ``MeshSelectionSet`` spans both levels and every element ``dim``).
_ENGINE_CACHE_ATTR = "_apegmsh_mesh_selection_chain_engines"


def engine_for(ms: Any, level: str, dim: int) -> _LiveMeshEngine:
    """Return the **stable per-(ms, level, dim)** engine adapter.

    The base :meth:`SelectionChain._compatible` gates set-algebra by
    engine *identity* (``self._engine is other._engine``).  A live-mesh
    selection cannot use the ``MeshSelectionSet`` itself as the engine
    (the chain needs a level / dim discriminator the composite does not
    carry), so an adapter is built once per ``(level, dim)`` and
    memoised on the ``MeshSelectionSet``.  Consequences, matching the
    locked contract used by the other chains:

    * two node selections from the same ``g.mesh_selection`` share one
      adapter → ``select(ids=a) | select(ids=b)`` composes;
    * a node selection and an element selection (different ``level``)
      get different adapters → cross-level set-algebra is loud;
    * element selections at different ``dim`` get different adapters →
      combining a 2-D and a 3-D element selection is loud;
    * two different sessions have different ``MeshSelectionSet``
      objects → different adapters → cross-session set-algebra is loud.
    """
    cache = getattr(ms, _ENGINE_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(ms, _ENGINE_CACHE_ATTR, cache)
    key = (level, dim)
    eng = cache.get(key)
    if eng is None:
        eng = _LiveMeshEngine(ms, level, dim)
        cache[key] = eng
    return eng
