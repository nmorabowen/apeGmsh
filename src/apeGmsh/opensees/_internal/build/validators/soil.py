"""soil: refuse a multi-yield soil solved with ``algorithm Linear``.

``PressureIndependMultiYield`` and both ``PressureDependMultiYield``
classes compute no stress in ``setTrialStrain``: ``getStress()`` computes
and caches it, and ``commitState`` commits the cache
(``PressureIndependMultiYield.cpp:326-353``, ``:493-511``, ``:558-570``;
the same pattern at ``PressureDependMultiYield.cpp:783/901`` and
``PressureDependMultiYield02.cpp:833/934``). ``Linear`` forms the residual
once, before its solve, and the analysis commits right after the update
(``Linear.cpp:114-150``, ``DirectIntegrationAnalysis.cpp:229-259``), so
every step commits the previous state's stress. Stress and strain drift
apart for the rest of the run: a 1-D column shaken in the elastic stage
answered about 2 % of the Newton surface acceleration, under every
constraint handler, solver and Rayleigh variant. Newton-type algorithms
form the residual again after the update and are correct. The bridge
refuses the pairing rather than emit a deck that runs to a wrong answer.

Scope: the gate runs on every solving emit (Tcl, py, live). It does not
run on an archival ``ops.h5`` write, so a replay of an archive declared
with Linear is not gated; and it scans every element of the model, so a
Linear stage that runs before the stage activating the soil elements is
refused too (split the model, or use Newton there).
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..errors import BridgeError


def _soil_classes(specs: "Iterable[object]") -> set[str]:
    """The multi-yield soil classes the element specs reach, directly or
    behind a wrapper (the dependency closure)."""
    from ....material.nd_soil import MULTI_YIELD_SOIL_CLASSES

    found: set[str] = set()
    seen: set[int] = set()
    frontier: list[object] = list(specs)
    while frontier:
        prim = frontier.pop()
        if id(prim) in seen:
            continue
        seen.add(id(prim))
        if type(prim).__name__ in MULTI_YIELD_SOIL_CLASSES:
            found.add(type(prim).__name__)
        deps = getattr(prim, "dependencies", None)
        if callable(deps):
            frontier.extend(deps())
    return found


def refuse_multi_yield_soil_under_linear(
    specs: "Iterable[object]",
    *,
    staged: bool,
    flat_algorithms: "Sequence[object]",
    stage_chain: "Sequence[tuple[str, Sequence[object], frozenset[int]]]",
) -> None:
    """Raise when an ``algorithm Linear`` (a chain's own or a strategy
    rung) can step a domain holding a multi-yield soil material."""
    from ....analysis.algorithm import Linear

    if staged:
        where = [name for name, algos, _ids in stage_chain
                 if any(isinstance(a, Linear) for a in algos)]
        where_text = f"stage(s) {', '.join(repr(w) for w in where)}"
    else:
        where = (["flat"] if any(isinstance(a, Linear)
                                 for a in flat_algorithms) else [])
        where_text = "the analysis chain"
    if not where:
        return
    soils = _soil_classes(specs)
    if not soils:
        return
    raise BridgeError(
        f"algorithm Linear in {where_text} with {', '.join(sorted(soils))}: "
        f"these materials compute their stress only in getStress(), and a "
        f"Linear step commits before forming the residual again, so it "
        f"commits a stale stress and the response collapses (about 2 % of "
        f"the Newton answer on a shaken column, whatever the constraint "
        f"handler). Use ops.algorithm.Newton(), ModifiedNewton (initial "
        f"tangent: a cheap elastic-stage run) or KrylovNewton."
    )


def refuse_up_soil_dimension(fem: Any, element_plan: "Iterable[Any]") -> None:
    """Raise when a ``LadrunoUP`` region carries a bare multi-yield soil
    whose ``nd`` is not the region's dimension.

    The u-p element takes its dimension from its cells at build, so the
    construction-time check of the other solid elements cannot see it;
    a mismatch makes OpenSees ``exit(-1)`` at the first strain update.
    The region is 2-D when all its nodes share one z coordinate.
    """
    import numpy as np

    from ....material.nd_soil import MULTI_YIELD_SOIL_CLASSES

    coords: "dict[int, Any] | None" = None
    for spec, rows in element_plan:
        if type(spec).__name__ != "LadrunoUP":
            continue
        mat = getattr(spec, "material", None)
        name = type(mat).__name__
        nd = getattr(mat, "nd", None)
        if name not in MULTI_YIELD_SOIL_CLASSES or nd is None:
            continue
        conn = np.asarray(getattr(rows, "conn", ()), dtype=np.int64)
        if conn.size == 0:
            continue
        if coords is None:
            nodes = fem.nodes
            coords = {int(n): c for n, c in zip(nodes.ids, nodes.coords)}
        z = np.array([float(coords[int(n)][2]) for n in np.unique(conn[conn >= 0])])
        dim = 2 if np.ptp(z) <= 1e-12 * max(1.0, float(np.abs(z).max())) else 3
        if int(nd) != dim:
            raise BridgeError(
                f"LadrunoUP on pg {getattr(spec, 'pg', '?')!r}: {name}(nd={nd}) "
                f"in a {dim}-D region; OpenSees exits the process on that "
                f"mismatch. Declare the material with nd={dim}."
            )
