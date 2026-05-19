"""``_ResultChainEngine`` + ``engine_for`` — the per-composite results
selection engine adapter.

Relocated **verbatim** (selection-unification v2 P3-R / §6.3 §3) from
the now-deleted ``results/_result_chain.py``.  The ``ResultChain``
class itself is removed with that module; this engine adapter survives
because the retained ``MeshSelection`` results path and
``results/_composites.py`` build it via :func:`engine_for`.  Pure leaf
— ``typing`` only (no ``numpy`` / ``_kernel`` / ``SelectionChain``) →
adds **no** new ``tests/test_import_dag_polarity.py`` BASELINE triple
(the 4 chain triples are removed −4/+0 in the same P3-R commit).
"""

from __future__ import annotations

from typing import Any

#: The two levels a results selection's atoms can live at.
VALID_LEVELS = ("node", "element")


class _ResultChainEngine:
    """Opaque back-reference a results selection is bound to.

    Minimal level discriminator + the two things the chain needs:

    * ``results`` — the bound :class:`~apeGmsh.results.Results` (used
      only to reach ``results._fem`` for coordinates / centroids,
      duck-typed exactly as the existing results spatial helpers do);
    * ``host`` — the spawning composite (``NodeResultsComposite`` /
      ``ElementResultsComposite``), whose **existing** ``.get(...)`` the
      terminal delegates to (so the slab read reuses the existing
      reader path verbatim — automatic parity);
    * ``level`` — ``"node"`` or ``"element"``.

    Identity (not value) is what the base
    :meth:`SelectionChain._compatible` compares, so set-algebra is loud
    across two differently-bound results selections — same contract as
    the mesh chains.
    """

    # The two ``_apegmsh_rc_*`` slots are lazily-populated per-engine
    # coordinate/centroid caches (mirrors NodeChain/ElementChain's
    # engine-side memoisation, which write onto their composite engine).
    __slots__ = (
        "results", "host", "level",
        "_apegmsh_rc_node_idrow", "_apegmsh_rc_elem_centroid",
    )

    def __init__(self, results: Any, host: Any, level: str) -> None:
        if level not in VALID_LEVELS:
            raise ValueError(
                f"ResultChain level={level!r} invalid; expected one of "
                f"{VALID_LEVELS}."
            )
        self.results = results
        self.host = host
        self.level = level
        self._apegmsh_rc_node_idrow = None
        self._apegmsh_rc_elem_centroid = None


#: Attribute the per-composite engine adapter is memoised under.
_ENGINE_CACHE_ATTR = "_apegmsh_result_chain_engine"


def engine_for(results: Any, host: Any, level: str) -> _ResultChainEngine:
    """Return the **stable per-composite** engine adapter for ``host``.

    The base :meth:`SelectionChain._compatible` gates set-algebra by
    engine *identity* (``self._engine is other._engine``), exactly as it
    does for the mesh chains — whose engine *is* the composite (a stable
    singleton on the FEMData).  A results selection cannot use the
    composite itself as the engine (the chain needs a level
    discriminator the composite does not carry), so the adapter is built
    once per composite and memoised on it.  Consequences, matching the
    locked contract:

    * two selections from the *same* ``results.<level>`` share one
      adapter → ``select(ids=a) | select(ids=b)`` composes;
    * a node selection and an element selection come from *different*
      host composites → different adapters → cross-level set-algebra is
      loud;
    * two different ``Results`` have different composites → different
      adapters → cross-results set-algebra is loud.

    ``Results._derive`` builds fresh composites, so a derived
    ``Results`` gets its own adapter (cross-derive is loud too — the
    user must pair selections from one ``Results``).
    """
    cached = getattr(host, _ENGINE_CACHE_ATTR, None)
    if cached is not None and cached.level == level:
        # ``results`` can change underneath a re-derived composite that
        # somehow reuses the host object; keep the back-ref fresh while
        # preserving identity for the set-algebra contract.
        cached.results = results
        return cached
    eng = _ResultChainEngine(results, host, level)
    setattr(host, _ENGINE_CACHE_ATTR, eng)
    return eng
