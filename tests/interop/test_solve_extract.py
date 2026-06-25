"""solve_and_extract — joint-keyed displacements + reactions for one case.

The apeGmsh half of the ADR 0009 solve cross-check. Validated here with
statics invariants (global equilibrium: total support reaction balances the
applied load) on self-contained models, so it needs no live ETABS.
"""
from __future__ import annotations

import pytest

from apeGmsh.interop import StructuralModel, solve_and_extract

# A 4x4 m slab on four pinned corners under a uniform downward pressure.
_SLAB = {
    "schema_version": "0.1",
    "units": {"length": "m", "force": "kN"},
    "nodes": [
        {"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "2", "x": 4.0, "y": 0.0, "z": 0.0},
        {"id": "3", "x": 4.0, "y": 4.0, "z": 0.0},
        {"id": "4", "x": 0.0, "y": 4.0, "z": 0.0},
    ],
    "frames": [],
    "areas": [{"id": "S1", "nodes": ["1", "2", "3", "4"], "section": "SLAB", "kind": "slab"}],
    "sections": [{"name": "SLAB", "kind": "shell", "material": "C", "thickness": 0.30}],
    "materials": [{"name": "C", "E": 2.5e7, "nu": 0.2}],
    "restraints": [{"node": n, "dofs": [1, 1, 1, 1, 1, 1]} for n in ("1", "2", "3", "4")],
    "loads": {"Dead": {"area": [{"area": "S1", "direction": "Z", "value": -5.0}]}},
}


def test_solve_and_extract_equilibrium():
    pytest.importorskip("openseespy")
    model = StructuralModel.from_dict(_SLAB)
    res = solve_and_extract(model, case="Dead", global_size=1.0)

    assert res.converged
    assert res.case == "Dead"
    # Every input joint gets a displacement; the four supports get reactions.
    assert set(res.displacements) == {"1", "2", "3", "4"}
    assert set(res.reactions) == {"1", "2", "3", "4"}
    # Each reading is a 6-vector.
    assert all(len(v) == 6 for v in res.displacements.values())

    # Global equilibrium: total vertical reaction balances the applied pressure
    # (q=5 kN/m^2 over 4x4 m = 80 kN down -> +80 kN of support reaction).
    rz = sum(v[2] for v in res.reactions.values())
    assert rz == pytest.approx(80.0, rel=1e-6)
    # No spurious horizontal resultant under gravity.
    rx = sum(v[0] for v in res.reactions.values())
    ry = sum(v[1] for v in res.reactions.values())
    assert rx == pytest.approx(0.0, abs=1e-6)
    assert ry == pytest.approx(0.0, abs=1e-6)


def test_solve_and_extract_defaults_first_case_and_validates_case():
    pytest.importorskip("openseespy")
    model = StructuralModel.from_dict(_SLAB)
    # case=None -> first (only) pattern.
    assert solve_and_extract(model).case == "Dead"
    with pytest.raises(ValueError, match="not in load patterns"):
        solve_and_extract(model, case="Nope")
