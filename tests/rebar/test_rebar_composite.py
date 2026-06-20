"""P1 — g.rebar composite + conformal embed coupling (ADR 0066 §6.1).

These need a live gmsh session (unlike the pure-data P0 tests).
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Bar, Cage, Stirrup


def _node_set(dim: int) -> set[int]:
    """All node tags referenced by elements of the given dimension."""
    _types, _tags, node_tags = gmsh.model.mesh.getElements(dim)
    s: set[int] = set()
    for arr in node_tags:
        s.update(int(x) for x in arr)
    return s


def test_rebar_is_a_registered_composite():
    with apeGmsh(model_name="rebar_reg") as g:
        assert hasattr(g, "rebar")
        # spec emitters return frozen L1 specs
        b = g.rebar.bar([(0, 0, 0), (0, 0, 1)], db="#8", material="rebar")
        assert isinstance(b, Bar)
        s = g.rebar.stirrup_rect(0.5, 0.5, 0.04, db=0.012, material="rebar")
        assert isinstance(s, Stirrup)


def test_conformal_place_shares_nodes_with_host():
    with apeGmsh(model_name="rebar_conformal") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="ConcreteVol")
        # NB: a bar whose endpoints sit exactly on the host boundary faces
        # trips a tetgen PLC error; an interior bar proves the conformal
        # mechanism. (Boundary-touching full-height bars need endpoint-into-
        # face embedding — a P1 robustness follow-up, see ADR §6.1.)
        bar = g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)],
                          db="#8", material="rebar", name="L1")
        placement = g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                                  coupling="conformal")
        assert placement.coupling == "conformal"
        assert len(placement.members) == 1
        assert placement.members[0].pg == "rebar.L1"

        g.mesh.sizing.set_global_size(0.2)
        g.mesh.generation.generate(dim=3)

        line_nodes = _node_set(1)
        vol_nodes = _node_set(3)
        assert line_nodes, "no 1-D (bar) mesh nodes were produced"
        assert vol_nodes, "no 3-D (host) mesh nodes were produced"
        # conformal coupling ⇒ every bar node is also a host-volume node
        assert line_nodes <= vol_nodes


def test_place_rejects_unknown_coupling_and_true_arc():
    with apeGmsh(model_name="rebar_guard") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="V")
        bar = g.rebar.bar([(0.5, 0.5, 0.0), (0.5, 0.5, 1.0)], db="#8",
                          material="rebar")
        cage = Cage(bars=(bar,))
        with pytest.raises(ValueError):
            g.rebar.place(cage, into="V", coupling="bogus")
        with pytest.raises(NotImplementedError):
            g.rebar.place(cage, into="V", coupling="embedded")
        with pytest.raises(NotImplementedError):
            g.rebar.place(cage, into="V", coupling="conformal", true_arc=True)


def test_place_after_snapshot_is_chain_phase_guarded():
    with apeGmsh(model_name="rebar_chainphase") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="V")
        g.mesh.sizing.set_global_size(0.6)
        g.mesh.generation.generate(dim=3)
        _ = g.mesh.queries.get_fem_data()          # canonical snapshot sets _fem
        bar = g.rebar.bar([(0.5, 0.5, 0.0), (0.5, 0.5, 1.0)], db="#8",
                          material="rebar")
        with pytest.raises(Exception):
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="conformal")
