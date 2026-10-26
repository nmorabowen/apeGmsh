"""topological_order: every LadrunoMembrane after every other element, in O(M + E).

The membrane rule used to be one edge from each membrane to each element (O(M x E), 82 s per call
on a 20-storey building). It is now one extra in-degree per membrane, released inline by the last
non-membrane element. Element tags are allocated in this order (``element_tags="sequential"``), so
the order must be the old one EXACTLY, not merely a valid topological order. The oracle is a frozen
copy of main's algorithm before the change (``_main_topological_order`` below); the differential
tests compare against it on seeded random DAGs, cyclic graphs and a real-primitive model.
"""
from __future__ import annotations

import random
import time

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.topo import topological_order
from apeGmsh.opensees._internal.types import Element, Primitive


class _Leaf(Primitive):
    def __init__(self, name, deps=()):
        self.name, self._deps = name, list(deps)

    def _emit(self, emitter, tag):  # pragma: no cover - never emitted here
        pass

    def dependencies(self):
        return tuple(self._deps)


class _Elem(Element):
    def __init__(self, name, deps=()):
        self.name, self._deps = name, list(deps)

    def _emit(self, emitter, tag):  # pragma: no cover
        pass

    def dependencies(self):
        return tuple(self._deps)


class LadrunoMembrane(_Elem):
    """Same class name as the real one: the rule keys on the type name."""


# ---------------------------------------------------------------------------
# Oracle: main's topological_order before the O(M + E) change (origin/main af48f7ca1), frozen.
# Do not "fix" or speed this up: it is the definition of the order the tags were planned in.
# ---------------------------------------------------------------------------

def _main_collect_reachable(p, seen):
    pid = id(p)
    if pid in seen:
        return
    seen[pid] = p
    for d in p.dependencies():
        _main_collect_reachable(d, seen)


def _main_topological_order(primitives):
    seen = {}
    for p in primitives:
        _main_collect_reachable(p, seen)

    non_membrane_elem_ids = [
        i for i, p in seen.items()
        if isinstance(p, Element) and type(p).__name__ != "LadrunoMembrane"
    ]
    deps = {}
    for i, p in seen.items():
        p_deps = [id(d) for d in p.dependencies()]
        if type(p).__name__ == "LadrunoMembrane":
            for eid in non_membrane_elem_ids:
                if eid not in p_deps:
                    p_deps.append(eid)
        deps[i] = p_deps
    in_degree = {i: 0 for i in seen}
    for i, ds in deps.items():
        for d in ds:
            if d in in_degree:
                in_degree[i] += 1

    children_of = {i: [] for i in seen}
    for child_id, parent_ids in deps.items():
        for parent_id in parent_ids:
            if parent_id in children_of:
                children_of[parent_id].append(child_id)

    queue = [i for i in seen if in_degree[i] == 0]
    out = []
    while queue:
        node = queue.pop(0)
        out.append(seen[node])
        for child in children_of[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(out) != len(seen):
        raise BridgeError(
            "topological_order: cycle detected in primitive dependency "
            f"graph. Reached {len(out)} of {len(seen)} primitives."
        )
    return tuple(out)


def _verdict(fn, prims):
    """The order as names, or the cycle message: both must match the oracle."""
    try:
        return [p.name for p in fn(prims)]
    except BridgeError as exc:
        return f"BridgeError: {exc}"


def _random_graph(rnd, *, cyclic):
    """Leaves, elements and membranes with random dependencies.

    Any kind may depend on any other (a leaf on an element stands in for a recorder or pattern
    that reads element tags), so a membrane's own dependency is often dequeued AFTER the last
    non-membrane element: the case a queued barrier node got wrong.
    """
    n = rnd.randint(2, 12)
    kinds = {"L": _Leaf, "E": _Elem, "M": LadrunoMembrane}
    nodes = []
    for k in range(n):
        kind = rnd.choice("LEM")
        nodes.append(kinds[kind](f"{kind}{k}"))
    perm = list(range(n))
    rnd.shuffle(perm)
    for a in range(n):
        for b in range(n if cyclic else a):
            if a != b and rnd.random() < 0.2:
                nodes[perm[a]]._deps.append(nodes[perm[b]])
    prims = nodes[:]
    rnd.shuffle(prims)
    return prims


def _membrane_dep_after_last_element(order):
    """True when some membrane depends on a primitive placed after the last non-membrane element."""
    at = {id(p): k for k, p in enumerate(order)}
    elems = [k for k, p in enumerate(order) if isinstance(p, Element) and type(p).__name__ != "LadrunoMembrane"]
    if not elems:
        return False
    last = max(elems)
    return any(
        at[id(d)] > last
        for p in order if type(p).__name__ == "LadrunoMembrane"
        for d in p.dependencies()
    )


def test_differential_random_dags_match_main():
    rnd = random.Random(20261026)
    late_dep = 0
    for _ in range(3000):
        prims = _random_graph(rnd, cyclic=False)
        expected = _verdict(_main_topological_order, prims)
        assert _verdict(topological_order, prims) == expected, [p.name for p in prims]
        # An element that depends on a membrane is a cycle under the membrane rule, even here.
        if not isinstance(expected, str):
            late_dep += _membrane_dep_after_last_element(_main_topological_order(prims))
    # The sample must exercise the case the queued barrier got wrong, or it proves nothing.
    assert late_dep > 100


def test_differential_cyclic_graphs_same_verdict_and_message():
    rnd = random.Random(1026)
    cycles = 0
    for _ in range(2000):
        prims = _random_graph(rnd, cyclic=True)
        expected = _verdict(_main_topological_order, prims)
        assert _verdict(topological_order, prims) == expected, [p.name for p in prims]
        cycles += isinstance(expected, str)
    assert cycles > 200  # both verdicts are exercised


def test_zero_non_membrane_elements():
    """No element to wait for: membranes keep their plain input order among the leaves."""
    m0, m1 = LadrunoMembrane("M0"), LadrunoMembrane("M1")
    leaf_late = _Leaf("late", [m0])
    for prims in ([m0, m1], [leaf_late, m1, _Leaf("a")], [m1, _Leaf("b", [m1]), m0]):
        assert _verdict(topological_order, prims) == _verdict(_main_topological_order, prims)
    assert [p.name for p in topological_order([m0, m1])] == ["M0", "M1"]
    leaves = [_Leaf("a"), _Leaf("b")]
    assert [p.name for p in topological_order(leaves)] == ["a", "b"]


def test_real_primitives_keep_main_membrane_order():
    """The review's repro: a membrane whose section is a dependency of the last element.

    main emits walls, floor_b, floor_a (floor_b is released with walls' children; floor_a waits for
    its PlateFiber). A queued barrier gave walls, floor_a, floor_b, swapping their tag blocks.
    """
    from apeGmsh.opensees.element.shell import LadrunoMembrane as RealMembrane
    from apeGmsh.opensees.element.shell import ShellMITC4
    from apeGmsh.opensees.material.nd import ElasticIsotropic, PlateFiber
    from apeGmsh.opensees.section.plate import ElasticMembranePlateSection

    wall_sec = ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.3)
    iso = ElasticIsotropic(E=30e9, nu=0.2)
    slab = PlateFiber(material=iso)
    walls = ShellMITC4(pg="walls", section=wall_sec)
    floor_a = RealMembrane(pg="floor_a", mat=slab, thick=0.2)
    floor_b = RealMembrane(pg="floor_b", section=wall_sec)
    prims = [wall_sec, iso, slab, walls, floor_a, floor_b]

    got = topological_order(prims)
    assert [id(p) for p in got] == [id(p) for p in _main_topological_order(prims)]
    assert [p.pg for p in got if isinstance(p, Element)] == ["walls", "floor_b", "floor_a"]


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
    assert [p.name for p in topological_order(prims)] == [p.name for p in _main_topological_order(prims)]


def test_membranes_after_every_element():
    out = topological_order(_model(200, 80, seed=3))
    last_elem = max(i for i, p in enumerate(out) if isinstance(p, Element) and type(p).__name__ != "LadrunoMembrane")
    first_mem = min(i for i, p in enumerate(out) if type(p).__name__ == "LadrunoMembrane")
    assert first_mem > last_elem


def test_linear_cost_at_building_size():
    """2,000 membranes x 5,000 elements: the edge encoding took minutes; this, well under 1 s."""
    prims = _model(5000, 2000, seed=7)
    t0 = time.perf_counter()
    out = topological_order(prims)
    assert time.perf_counter() - t0 < 2.0
    assert len(out) == len({id(p) for p in out})
