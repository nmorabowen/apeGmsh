"""Partitioned emit of embedded reinforcement (g.rebar embedded + auto bars).

Every LadrunoEmbeddedRebar tie and every bar CorotTruss must land on exactly
one rank, and every node an element references on a rank must be declared on
that rank (owned or ghost).
"""
from __future__ import annotations

import re
from collections import Counter

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Cage
from apeGmsh.opensees import apeSees


def _rank_blocks(deck: str) -> "dict[int, str]":
    blocks: "dict[int, list[str]]" = {}
    rank, depth = None, 0
    for line in deck.splitlines():
        m = re.match(r"if \{\[getPID\] == (\d+)\} \{", line)
        if m and depth == 0:
            rank, depth = int(m.group(1)), 1
            continue
        if rank is not None:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                rank, depth = None, 0
                continue
            blocks.setdefault(rank, []).append(line)
    return {r: "\n".join(v) for r, v in blocks.items()}


def _build(tmp_path, nparts):
    with apeGmsh(model_name=f"reinforce_part_{nparts}") as g:
        g.model.geometry.add_box(0, 0, 0, 1000, 200, 200, label="beam")
        g.physical.add_volume(g.model.select(dim=3).tags(), name="Concrete")
        g.physical.add_surface(
            g.model.select(dim=2).in_box((-1, -1, -1), (1, 201, 201)).tags(),
            name="Fixed")
        bars = [g.rebar.bar([(20, y, z), (980, y, z)], db=12, material="steel",
                            name=f"b{i}")
                for i, (y, z) in enumerate(((40, 40), (160, 40), (40, 160), (160, 160)))]
        g.rebar.place(Cage(bars=tuple(bars)), into="Concrete",
                      coupling="embedded", perfect=1.0e8, emit_elements=True)
        g.mesh.recipe.structured("beam", n=(11, 3, 3), generate=False)
        g.mesh.sizing.set_global_size(50.0)
        g.mesh.generation.generate(dim=3)
        if nparts > 1:
            g.mesh.partitioning.partition(nparts)
        fem = g.mesh.queries.get_fem_data(dim=None)

        ops = apeSees(fem, _artifacts=False)  # partitioned snapshot: no automatic model.h5
        ops.model(ndm=3, ndf=3)
        conc = ops.nDMaterial.ElasticIsotropic(E=30000.0, nu=0.2)
        ops.uniaxialMaterial.Steel02(fy=420.0, E=200000.0, b=0.01, name="steel")
        ops.element.stdBrick(pg="Concrete", material=conc)
        ops.fix(pg="Fixed", dofs=(1, 1, 1))
        path = tmp_path / f"deck_{nparts}.tcl"
        ops.tcl(str(path))
    return path.read_text()


def _count(text: str) -> Counter:
    return Counter(re.findall(r"^\s*element (\S+) ", text, re.M))


def test_partitioned_reinforcement_routed_once_with_declared_nodes(tmp_path):
    flat = _count(_build(tmp_path, 1))
    deck = _build(tmp_path, 2)
    blocks = _rank_blocks(deck)
    assert sorted(blocks) == [0, 1]

    total = Counter()
    for text in blocks.values():
        declared = {int(n) for n in re.findall(r"^\s*node (\d+) ", text, re.M)}
        dup = [n for n, c in Counter(re.findall(r"^\s*node (\d+) ", text, re.M)).items() if c > 1]
        assert not dup
        for m in re.finditer(r"^\s*element (\S+) \d+ (.*)$", text, re.M):
            kind, rest = m.group(1), m.group(2).split()
            if kind == "LadrunoEmbeddedRebar":
                nodes = [int(rest[0])] + [int(x) for x in rest[2:2 + int(rest[1])]]
            elif kind == "CorotTruss":
                nodes = [int(rest[0]), int(rest[1])]
            elif kind == "stdBrick":
                nodes = [int(x) for x in rest[:8]]
            else:
                continue
            assert all(n in declared for n in nodes), (kind, nodes)
        total.update(_count(text))

    for kind in ("stdBrick", "LadrunoEmbeddedRebar", "CorotTruss"):
        assert total[kind] == flat[kind] > 0, kind
    tags = re.findall(r"^\s*element \S+ (\d+) ", deck, re.M)
    assert len(tags) == len(set(tags))
