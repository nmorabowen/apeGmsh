"""P1 — g.rebar composite + conformal embed coupling (ADR 0066 §6.1).

These need a live gmsh session (unlike the pure-data P0 tests).
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Bar, Cage, Stirrup
from apeGmsh.rebar.detailing import ACI318


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
                          db=0.0254, material="rebar", name="L1")
        placement = g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                                  coupling="conformal")
        assert placement.coupling == "conformal"
        assert len(placement.members) == 1
        assert placement.members[0].pg == "rebar0.L1"
        assert placement.members[0].diameter == pytest.approx(0.0254)  # #8

        g.mesh.sizing.set_global_size(0.2)
        g.mesh.generation.generate(dim=3)

        line_nodes = _node_set(1)
        vol_nodes = _node_set(3)
        assert line_nodes, "no 1-D (bar) mesh nodes were produced"
        assert vol_nodes, "no 3-D (host) mesh nodes were produced"
        # conformal coupling ⇒ every bar node is also a host-volume node
        assert line_nodes <= vol_nodes


def test_place_rejects_unknown_coupling():
    with apeGmsh(model_name="rebar_guard") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="V")
        bar = g.rebar.bar([(0.5, 0.5, 0.0), (0.5, 0.5, 1.0)], db=0.02,
                          material="rebar")
        with pytest.raises(ValueError):
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="bogus")


def test_embedded_place_forwards_to_reinforce():
    with apeGmsh(model_name="rebar_embedded") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0)
        g.physical.add_volume([vol], name="ConcreteVol")   # embedded needs a PG host
        bar = g.rebar.bar([(0.25, 0.25, 0.1), (0.25, 0.25, 1.9)],
                          db=0.025, material="rebar", name="L1")
        placement = g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                                  coupling="embedded", perfect=1.0e8)
        assert placement.coupling == "embedded"
        # forwarded to the shipped g.reinforce binding composite
        defs = g.reinforce.reinforce_defs
        assert len(defs) == 1
        assert defs[0].slave_label == "rebar0.L1"
        assert defs[0].master_label == "ConcreteVol"
        assert defs[0].perfect == 1.0e8
        assert defs[0].bar_diameter == pytest.approx(0.025)


def test_embedded_requires_bond_or_perfect():
    with apeGmsh(model_name="rebar_embedded_guard") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="V")
        bar = g.rebar.bar([(0.5, 0.5, 0.1), (0.5, 0.5, 0.9)], db=0.02,
                          material="rebar")
        with pytest.raises(ValueError):
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="embedded")


def test_embedded_designation_db_needs_standard():
    with apeGmsh(model_name="rebar_embedded_std") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.physical.add_volume([vol], name="V")
        bar = g.rebar.bar([(0.5, 0.5, 0.1), (0.5, 0.5, 0.9)], db="#8",
                          material="rebar")
        with pytest.raises(ValueError):           # no standard set
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="embedded",
                          perfect=1.0e8)
        g.rebar.use_standard(ACI318())            # now the designation resolves
        g.rebar.place(Cage(bars=(bar,)), into="V", coupling="embedded",
                      perfect=1.0e8)
        assert g.reinforce.reinforce_defs[-1].bar_diameter == pytest.approx(1.0)


def test_mixed_coupling_splits_conformal_and_embedded():
    with apeGmsh(model_name="rebar_mixed") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0)
        g.physical.add_volume([vol], name="ConcreteVol")
        bar = g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)], db=0.025,
                          material="rebar", name="L1")
        tie = g.rebar.stirrup_rect(0.5, 0.5, 0.04, db=0.012, material="rebar",
                                   z=1.0, name="T1")
        cage = Cage(bars=(bar,), stirrups=(tie,))
        placement = g.rebar.place(
            cage, into="ConcreteVol", coupling="conformal",
            per_member_coupling={"tie": "embedded"}, perfect=1.0e8,
        )
        assert placement.coupling == "mixed"
        # exactly the tie went to g.reinforce; the longitudinal bar is conformal
        defs = g.reinforce.reinforce_defs
        assert len(defs) == 1
        assert defs[0].slave_label == "rebar0.T1"
        bar_member = next(m for m in placement.members if m.role == "longitudinal")
        assert bar_member.coupling == "conformal"


def test_embedded_resolves_at_get_fem_data():
    with apeGmsh(model_name="rebar_embedded_resolve") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0)
        g.physical.add_volume([vol], name="ConcreteVol")   # real PG for resolve
        bar = g.rebar.bar([(0.25, 0.25, 0.2), (0.25, 0.25, 1.8)], db=0.025,
                          material="rebar", name="L1")
        g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                      coupling="embedded", perfect=1.0e8)
        g.mesh.sizing.set_global_size(0.25)
        g.mesh.generation.generate(dim=3)
        g.mesh.queries.get_fem_data()             # triggers g.reinforce.resolve
        assert g.reinforce.reinforce_records, "embedded bar did not resolve"


def test_duplicate_member_name_raises():
    with apeGmsh(model_name="rebar_dup") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="V")
        b1 = g.rebar.bar([(0.2, 0.2, 0.1), (0.2, 0.2, 0.9)], db=0.02,
                         material="rebar", name="X")
        b2 = g.rebar.bar([(0.8, 0.8, 0.1), (0.8, 0.8, 0.9)], db=0.02,
                         material="rebar", name="X")     # duplicate name
        with pytest.raises(ValueError):
            g.rebar.place(Cage(bars=(b1, b2)), into="V", coupling="conformal")


def test_independent_placements_get_unique_pg_bases():
    with apeGmsh(model_name="rebar_seq") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 1, 1, 2)
        g.physical.add_volume([vol], name="V")
        for _ in range(2):
            bar = g.rebar.bar([(0.5, 0.5, 0.2), (0.5, 0.5, 1.8)], db=0.02,
                              material="rebar")          # unnamed -> auto-name
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="embedded",
                          perfect=1.0e8)
        labels = [d.slave_label for d in g.reinforce.reinforce_defs]
        assert labels == ["rebar0.longitudinal_0", "rebar1.longitudinal_0"]
        assert len(set(labels)) == 2                      # no PG merge / double-tie


def test_conformal_after_generate_raises():
    with apeGmsh(model_name="rebar_late") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="V")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)                 # mesh exists, no snapshot
        bar = g.rebar.bar([(0.5, 0.5, 0.2), (0.5, 0.5, 0.8)], db=0.02,
                          material="rebar")
        with pytest.raises(RuntimeError):                 # silent-no-op guard
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="conformal")


def test_embedded_both_bond_and_perfect_raises():
    with apeGmsh(model_name="rebar_xor") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.physical.add_volume([vol], name="V")
        bar = g.rebar.bar([(0.5, 0.5, 0.1), (0.5, 0.5, 0.9)], db=0.02,
                          material="rebar")
        with pytest.raises(ValueError):                   # exactly-one XOR
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="embedded",
                          bond="bondlaw", perfect=1.0e8)


def test_beam_on_curved_bar_is_gated():
    with apeGmsh(model_name="rebar_beam") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="V")
        bar = g.rebar.bar([(0.2, 0.2, 0.1), (0.2, 0.2, 0.5), (0.2, 0.5, 0.9)],
                          db=0.02, material="rebar", element="beam")
        with pytest.raises(NotImplementedError):          # ADR §7 gate
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="conformal")


def test_embedded_host_must_be_physical_group():
    with apeGmsh(model_name="rebar_label_host") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="V")   # label, not PG
        bar = g.rebar.bar([(0.5, 0.5, 0.1), (0.5, 0.5, 0.9)], db=0.02,
                          material="rebar")
        with pytest.raises(ValueError):
            g.rebar.place(Cage(bars=(bar,)), into="V", coupling="embedded",
                          perfect=1.0e8)


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
