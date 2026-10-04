"""#1335 (2): the "PG not found" BridgeError lists the real groups.

``_available_pg_names`` filtered ``PhysicalGroupSet._groups`` keys for
``str``, but those keys are ``(dim, tag)`` tuples, so both "not found in
FEM snapshot" errors (node and element fan-out) always printed
``Available PGs: []``.

Oracle: the snapshot's own ``fem.nodes.physical.names()`` (``['Soil']``
for this model), which the error must reproduce exactly.
"""
from __future__ import annotations

import ast
import re

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    expand_pg_to_elements,
    expand_pg_to_nodes,
)


@pytest.fixture
def fem():
    with apeGmsh(model_name="pg_not_found", verbose=False) as g:
        g.model.geometry.add_rectangle(-5.0, -5.0, 0, 10.0, 5.0,
                                       label="soil")
        g.physical.add_surface("soil", name="Soil")
        g.mesh.generation.generate(dim=2)
        yield g.mesh.queries.get_fem_data(dim=2)


def _listed(msg: str) -> list[str]:
    m = re.search(r"Available (?:element )?PGs: (\[.*?\])", msg)
    assert m is not None, msg
    return ast.literal_eval(m.group(1))


def test_oracle_snapshot_names(fem):
    assert fem.nodes.physical.names() == ["Soil"]


def test_node_fanout_error_lists_real_pgs(fem):
    with pytest.raises(BridgeError, match="'Bottom' not found") as exc:
        expand_pg_to_nodes(fem, "Bottom")
    assert _listed(str(exc.value)) == fem.nodes.physical.names()


def test_element_fanout_error_lists_real_pgs(fem):
    with pytest.raises(BridgeError, match="'Bottom' not found") as exc:
        expand_pg_to_elements(fem, "Bottom")
    assert _listed(str(exc.value)) == fem.elements.physical.names()


def test_issue_repro_emit_error_lists_real_pgs(fem, tmp_path):
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3)
    ops.element.FourNodeQuad(pg="Soil", thickness=1.0, material=mat,
                             plane_type="PlaneStrain")
    ops.fix(pg="Bottom", dofs=(1, 1))
    with pytest.raises(BridgeError, match="'Bottom' not found") as exc:
        ops.py(str(tmp_path / "deck.py"))
    assert _listed(str(exc.value)) == ["Soil"]
