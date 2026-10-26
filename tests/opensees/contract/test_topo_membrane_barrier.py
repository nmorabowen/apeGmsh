"""topological_order: every LadrunoMembrane after every other element, in O(M + E).

The membrane rule used to be one edge from each membrane to each element (O(M x E), 82 s per call
on a 20-storey building); it is now one barrier node. These tests pin the order the old encoding
gave and the cost.
"""
from __future__ import annotations

import random
import time

from apeGmsh.opensees._internal.build.topo import topological_order
from apeGmsh.opensees._internal.types import Element, Primitive


class _Leaf(Primitive):
    def __init__(self, name):
        self.name = name

    def _emit(self, emitter, tag):  # pragma: no cover - never emitted here
        pass

    def dependencies(self):
        return ()


class _Elem(Element):
    def __init__(self, name, deps=()):
        self.name, self._deps = name, tuple(deps)

    def _emit(self, emitter, tag):  # pragma: no cover
        pass

    def dependencies(self):
        return self._deps


class LadrunoMembrane(_Elem):
    """Same class name as the real one: the rule keys on the type name."""


def _old_order(primitives):
    """The pre-barrier algorithm (edge from every membrane to every element), for comparison."""
    from apeGmsh.opensees._internal.build.topo import _collect_reachable

    seen = {}
    for p in primitives:
        _collect_reachable(p, seen)
    elems = [i for i, p in seen.items() if isinstance(p, Element) and type(p).__name__ != "LadrunoMembrane"]
    deps = {}
    for i, p in seen.items():
        d = [id(x) for x in p.dependencies()]
        if type(p).__name__ == "LadrunoMembrane":
            d += [e for e in elems if e not in d]
        deps[i] = d
    indeg = {i: sum(1 for x in deps[i] if x in seen) for i in seen}
    kids = {i: [] for i in seen}
    for c, ps in deps.items():
        for p in ps:
            if p in kids:
                kids[p].append(c)
    q, out = [i for i in seen if indeg[i] == 0], []
    while q:
        n = q.pop(0)
        out.append(seen[n])
        for c in kids[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return tuple(out)


def _model(n_elem, n_mem, seed=0):
    rnd = random.Random(seed)
    mats = [_Leaf(f"m{k}") for k in range(5)]
    secs = [_Elem(f"s{k}", [rnd.choice(mats)]) for k in range(5)]  # sections stand in as composites
    elems = [_Elem(f"e{k}", [rnd.choice(secs)]) for k in range(n_elem)]
    mems = [LadrunoMembrane(f"M{k}", [rnd.choice(elems)] if k % 3 == 0 else []) for k in range(n_mem)]
    prims = elems + mems + mats
    rnd.shuffle(prims)
    return prims


def test_same_order_as_the_edge_encoding():
    prims = _model(120, 60)
    assert [p.name for p in topological_order(prims)] == [p.name for p in _old_order(prims)]


def test_membranes_after_every_element():
    out = topological_order(_model(200, 80, seed=3))
    last_elem = max(i for i, p in enumerate(out) if isinstance(p, Element) and type(p).__name__ != "LadrunoMembrane")
    first_mem = min(i for i, p in enumerate(out) if type(p).__name__ == "LadrunoMembrane")
    assert first_mem > last_elem


def test_no_membranes_and_no_elements():
    leaves = [_Leaf("a"), _Leaf("b")]
    assert [p.name for p in topological_order(leaves)] == ["a", "b"]
    mems = [LadrunoMembrane("M0"), LadrunoMembrane("M1")]
    assert [p.name for p in topological_order(mems)] == ["M0", "M1"]


def test_linear_cost_at_building_size():
    """2,000 membranes x 5,000 elements: the edge encoding took minutes; the barrier, well under 1 s."""
    prims = _model(5000, 2000, seed=7)
    t0 = time.perf_counter()
    out = topological_order(prims)
    assert time.perf_counter() - t0 < 2.0
    assert len(out) == len({id(p) for p in out})
