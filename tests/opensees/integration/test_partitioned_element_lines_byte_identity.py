"""Partitioned element-emit fidelity (ADR 0065 Tier 2, element term).

The compact per-rank element streaming (row-index buckets into numpy
connectivity + box-at-emit, replacing the resident ``element_plan`` tuples +
``plan_by_rank`` reference-lists + 54M boxed conn ints) MUST NOT change the
emitted elements: every continuum element is emitted exactly once, with its
exact connectivity in the exact node order, and the emit is deterministic.

This checks those invariants against the partitioned FEM's own connectivity,
which is platform-independent — unlike a committed golden of literal node tags,
which is METIS-partition- and platform-specific (gmsh ``partition()`` renumbers
the mesh, and Linux CI numbers nodes differently from a Windows dev box, so a
static golden is red on one platform or the other). Per-rank block structure and
exclusive ownership are covered by
``test_emit_partitioned_element_fan_out_tcl.py``; the unpartitioned baseline by
``test_emit_unpartitioned_byte_identical_to_today.py``.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.material.nd import ElasticIsotropic


def _partitioned_fem(nparts: int):
    with apeGmsh(model_name="g", verbose=False) as g:
        skin = g.parts.add_plane_wave_box(
            x=(600.0, 9), y=(600.0, 9), z=[(200.0, 3), (400.0, 5)])
        g.mesh.generation.generate(dim=3)
        g.mesh.partitioning.partition(nparts)
        fem = g.mesh.queries.get_fem_data()
    return fem, skin.soil_pgs


def _emit_element_lines(fem, soil_pgs) -> list[str]:
    ops = apeSees(fem, _artifacts=False)  # partitioned snapshot: no automatic model.h5
    ops.model(ndm=3, ndf=3)
    for pg in soil_pgs:
        ops.element.stdBrick(pg=pg, material=ops.register(
            ElasticIsotropic(E=3e10, nu=0.25, rho=2000.0)))
    fd, path = tempfile.mkstemp(suffix=".tcl")
    os.close(fd)
    try:
        ops.tcl(path)
        with open(path, encoding="utf-8") as f:
            # strip the per-rank ``if {[getPID]==K} {`` indent.
            return [ln.strip() for ln in f
                    if ln.strip().startswith("element ")]
    finally:
        os.remove(path)


def _emitted_connectivities(lines: list[str]) -> list[tuple[int, ...]]:
    # ``element stdBrick <tag> <n1> ... <nN> <matTag>`` → the connectivity span.
    return sorted(tuple(int(x) for x in ln.split()[3:-1]) for ln in lines)


def _fem_connectivities(fem, soil_pgs) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for pg in soil_pgs:
        for grp in fem.elements.select(pg=pg).groups():
            for i in range(len(grp.ids)):
                out.append(tuple(int(x) for x in grp.connectivity[i]))
    return sorted(out)


@pytest.mark.parametrize("nparts", [2, 4])
def test_partitioned_emit_reproduces_fem_connectivity(nparts):
    """Every continuum element is emitted exactly once across ranks, with its
    exact connectivity in the exact node order the FEM holds."""
    fem, soil_pgs = _partitioned_fem(nparts)
    lines = _emit_element_lines(fem, soil_pgs)

    emit_conns = _emitted_connectivities(lines)
    fem_conns = _fem_connectivities(fem, soil_pgs)

    assert len(lines) == len(fem_conns), (
        f"np{nparts}: emitted {len(lines)} element lines vs "
        f"{len(fem_conns)} FEM continuum elements"
    )
    assert emit_conns == fem_conns
    tags = [int(ln.split()[2]) for ln in lines]
    assert len(set(tags)) == len(tags), "an element tag was emitted twice"


@pytest.mark.parametrize("nparts", [2, 4])
def test_partitioned_element_emit_is_deterministic(nparts):
    """Re-emitting the same partitioned FEM yields byte-identical element lines
    (no set/dict iteration order leaking into tag numbering or per-rank order)."""
    fem, soil_pgs = _partitioned_fem(nparts)
    assert _emit_element_lines(fem, soil_pgs) == _emit_element_lines(fem, soil_pgs)
