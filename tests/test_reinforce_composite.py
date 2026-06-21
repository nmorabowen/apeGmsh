"""R2b — g.reinforce composite, end-to-end on a real non-matching mesh.

Builds a concrete tet4 block with a pre-meshed rebar line PG threaded
through its interior (non-matching: the rebar nodes are an independent
1-D entity, not shared with the host), then exercises the full
declare → get_fem_data → apeSees emit path. No fork build required —
emission produces deck text on any OpenSees.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees


def _build_rebar_in_tet(g, *, x0=0.5, y0=0.5, z_lo=0.2, z_hi=0.8, size=0.4):
    """Concrete unit-box tet host + an interior vertical rebar line.

    Returns nothing; leaves ``concrete`` (dim-3 PG) and ``rebar``
    (dim-1 PG) declared on ``g`` with a generated 3-D mesh.
    """
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    p0 = gmsh.model.occ.addPoint(x0, y0, z_lo)
    p1 = gmsh.model.occ.addPoint(x0, y0, z_hi)
    ln = gmsh.model.occ.addLine(p0, p1)
    g.model.sync()
    g.mesh.sizing.set_global_size(size)
    g.mesh.generation.generate(3)
    g.physical.add(3, [box], name="concrete")
    g.physical.add(1, [ln], name="rebar")
    return box, ln


def _emit_tcl(fem, *, ndm=3, ndf=3, declare=None):
    """Emit ``fem`` to a Tcl string. ``declare(ops)`` may register bridge
    primitives (e.g. a named bond material) before emit."""
    ops = apeSees(fem)
    ops.model(ndm=ndm, ndf=ndf)
    if declare is not None:
        declare(ops)
    path = os.path.join(tempfile.gettempdir(), "apegmsh_reinforce_test.tcl")
    ops.tcl(path)
    with open(path) as fh:
        return fh.read()


def test_perfect_bond_resolves_and_emits():
    with apeGmsh(model_name="reinforce_perfect", verbose=False) as g:
        _build_rebar_in_tet(g)
        g.reinforce(host="concrete", bars="rebar",
                    perfect=1.0e12, bar_diameter=0.025)
        fem = g.mesh.queries.get_fem_data(dim=3)

        ties = fem.elements.reinforce_ties
        assert len(ties) >= 2, "expected one tie per interior rebar node"
        for t in ties:
            assert t.in_bounds, "interior rebar nodes must map inside a host"
            assert abs(float(np.sum(t.weights)) - 1.0) < 1e-9
            assert len(t.host_nodes) == len(t.weights) == 4  # tet4 host
            assert t.perfect == 1.0e12 and t.bond is None
            # vertical rebar → axis along +z (sign is arbitrary)
            assert abs(abs(float(t.direction[2])) - 1.0) < 1e-6

        txt = _emit_tcl(fem)
        lines = [l for l in txt.splitlines() if "LadrunoEmbeddedRebar" in l]
        assert len(lines) == len(ties)
        assert all("-shape" in l and "-dir" in l and "-perfect" in l
                   for l in lines)


def test_bond_law_name_resolves_to_tag():
    """A LadrunoBondSlip declared by name on the bridge resolves to its
    tag at emit; the tie carries -bond <tag> -bondScale."""
    with apeGmsh(model_name="reinforce_bond", verbose=False) as g:
        _build_rebar_in_tet(g)
        g.reinforce(host="concrete", bars="rebar",
                    bond="bond1", bar_diameter=0.02)
        fem = g.mesh.queries.get_fem_data(dim=3)
        ties = fem.elements.reinforce_ties
        assert ties and all(t.bond == "bond1" for t in ties)
        assert all(t.bond_scale is not None and t.bond_scale > 0
                   for t in ties)

        def declare(ops):
            ops.uniaxialMaterial.LadrunoBondSlip(
                tau_max=10.0, s1=0.1, s2=0.5, s3=1.0,
                tau_f=2.0, alpha=0.4, name="bond1")

        txt = _emit_tcl(fem, declare=declare)
        lines = [l for l in txt.splitlines() if "LadrunoEmbeddedRebar" in l]
        assert lines and all("-bond" in l and "-bondScale" in l
                             for l in lines)
        assert all("-perfect" not in l for l in lines)


def test_bondscale_is_pi_d_ltrib():
    """bondScale = π·d·L_trib — an interior node of a uniformly split bar
    has L_trib = one segment length, an endpoint half that."""
    with apeGmsh(model_name="reinforce_bondscale", verbose=False) as g:
        _build_rebar_in_tet(g, z_lo=0.0, z_hi=1.0, size=0.5)
        d = 0.02
        g.reinforce(host="concrete", bars="rebar",
                    bond="b", bar_diameter=d)
        fem = g.mesh.queries.get_fem_data(dim=3)
        scales = sorted(float(t.bond_scale) for t in fem.elements.reinforce_ties)
        # endpoints get the smallest L_trib; all are positive multiples of π·d.
        assert all(s > 0 for s in scales)
        # the smallest (endpoint) ~ half the largest (interior) for a
        # uniform split.
        assert scales[0] == pytest.approx(0.5 * scales[-1], rel=0.25)


def test_to_h5_persists_ties_without_warning():
    """Writing a reinforced model to the neutral model.h5 now PERSISTS the
    ties (ADR 0067 P5.1, neutral schema 2.15.0) — no deferral warning, and
    they round-trip back into the broker. (Was: warned + dropped.)"""
    with apeGmsh(model_name="reinforce_h5", verbose=False) as g:
        _build_rebar_in_tet(g)
        g.reinforce(host="concrete", bars="rebar",
                    perfect=1.0e12, bar_diameter=0.025)
        fem = g.mesh.queries.get_fem_data(dim=3)
        n_ties = len(fem.elements.reinforce_ties)
        assert n_ties >= 2
        path = os.path.join(tempfile.gettempdir(), "apegmsh_reinforce.h5")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fem.to_h5(path)
        assert not [x for x in w if "deferred" in str(x.message).lower()
                    or "not persisted" in str(x.message).lower()]
        from apeGmsh.mesh._femdata_h5_io import read_fem_h5
        assert len(read_fem_h5(path).elements.reinforce_ties) == n_ties


def test_out_of_bounds_rebar_node_fails_loud():
    """A rebar node outside every host element raises (fail-loud default
    OOB policy), not a silent extrapolated tie."""
    with apeGmsh(model_name="reinforce_oob", verbose=False) as g:
        # Rebar segment runs OUTSIDE the box (z from 1.5 to 2.5).
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        p0 = gmsh.model.occ.addPoint(0.5, 0.5, 1.5)
        p1 = gmsh.model.occ.addPoint(0.5, 0.5, 2.5)
        ln = gmsh.model.occ.addLine(p0, p1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="concrete")
        g.physical.add(1, [ln], name="rebar")
        g.reinforce(host="concrete", bars="rebar", perfect=1.0e12)
        with pytest.raises(Exception):
            g.mesh.queries.get_fem_data(dim=3)


def test_snap_projects_and_warns():
    """``snap=True`` projects a stray rebar node onto the nearest host
    and warns, instead of raising."""
    with apeGmsh(model_name="reinforce_snap", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        # Slightly above the top face so it falls just outside the host.
        p0 = gmsh.model.occ.addPoint(0.5, 0.5, 1.05)
        p1 = gmsh.model.occ.addPoint(0.5, 0.5, 1.20)
        ln = gmsh.model.occ.addLine(p0, p1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="concrete")
        g.physical.add(1, [ln], name="rebar")
        g.reinforce(host="concrete", bars="rebar",
                    perfect=1.0e12, snap=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fem = g.mesh.queries.get_fem_data(dim=3)
        assert fem.elements.reinforce_ties  # produced, not raised
        assert any("snap" in str(x.message).lower()
                   or "out" in str(x.message).lower() for x in w)
