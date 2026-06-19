"""Partitioned element-emit byte-identity golden (ADR 0065 Tier 2, element term).

The compact per-rank element streaming (row-index buckets into numpy
connectivity + box-at-emit, replacing the resident `element_plan` tuples +
`plan_by_rank` reference-lists + 54M boxed conn ints) MUST NOT change the
emitted deck. This pins the full ordered partitioned element-line list
(captured from pre-refactor source) for np=2 and np=4 multi-PG models, the
partitioned counterpart to `test_emit_unpartitioned_byte_identical_to_today`.

If this drifts, the refactor changed tag numbering, per-rank emission order,
or node-tag rendering — not an acceptable "improvement".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.material.nd import ElasticIsotropic

_GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden"


def _partitioned_element_lines(nparts: int) -> list[str]:
    with apeGmsh(model_name="g", verbose=False) as g:
        skin = g.parts.add_plane_wave_box(
            x=(600.0, 9), y=(600.0, 9), z=[(200.0, 3), (400.0, 5)])
        g.mesh.generation.generate(dim=3)
        g.mesh.partitioning.partition(nparts)
        fem = g.mesh.queries.get_fem_data()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    for pg in skin.soil_pgs:
        ops.element.stdBrick(pg=pg, material=ops.register(
            ElasticIsotropic(E=3e10, nu=0.25, rho=2000.0)))
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".tcl")
    os.close(fd)
    try:
        ops.tcl(path)
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f
                    if ln.strip().startswith("element ")]
    finally:
        os.remove(path)


@pytest.mark.parametrize("nparts", [2, 4])
def test_partitioned_element_lines_match_golden(nparts):
    golden_file = _GOLDEN / f"partitioned_element_lines_np{nparts}.txt"
    golden = golden_file.read_text(encoding="utf-8").splitlines()
    actual = _partitioned_element_lines(nparts)
    assert len(actual) == len(golden), (
        f"np{nparts}: {len(actual)} element lines vs golden {len(golden)}"
    )
    assert actual == golden
