"""B1a — ``g.rebar.place(emit_elements=True)`` auto-emits the bar's own
structural element (``CorotTruss``) — ADR 0067 P5.2 / B1.

The cage emits geometry + coupling regardless; ``emit_elements=True`` adds
the bar's axial element (distinct from the ``LadrunoEmbeddedRebar`` coupling,
which carries no axial stiffness). These need a live gmsh session + a bridge
emit.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Cage
from apeGmsh.opensees import apeSees


def _conformal_bar(g, *, element="truss"):
    """A single interior bar in a box host. (Interior endpoints avoid the
    boundary-PLC tetgen trip.) Returns the L1 :class:`Bar` spec."""
    g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="ConcreteVol")
    return g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)],
                       db=0.0254, material="rebar", element=element,
                       name="L1")


def _emit_tcl(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01, name="rebar")
    path = os.path.join(tempfile.gettempdir(), "apegmsh_rebar_emit.tcl")
    ops.tcl(path)
    with open(path) as fh:
        return fh.read()


def test_emit_elements_off_is_default_no_structural_element():
    """Default ``emit_elements=False`` → no rebar_elements record, and the
    deck carries no auto-emitted CorotTruss (today's behavior)."""
    with apeGmsh(model_name="rebar_emit_off") as g:
        bar = _conformal_bar(g)
        g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                      coupling="conformal")
        g.mesh.sizing.set_global_size(0.2)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        assert not fem.elements.rebar_elements
        txt = _emit_tcl(fem)
        assert "CorotTruss" not in txt


def test_emit_elements_truss_emits_corottruss_per_cell():
    """``emit_elements=True`` on a truss bar → one rebar_elements record and
    one ``CorotTruss`` per bar line cell, referencing the bar material."""
    with apeGmsh(model_name="rebar_emit_truss") as g:
        bar = _conformal_bar(g)
        g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                      coupling="conformal", emit_elements=True)
        g.mesh.sizing.set_global_size(0.2)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)

        recs = fem.elements.rebar_elements
        assert len(recs) == 1
        rec = recs[0]
        assert rec.element == "truss"
        assert rec.material == "rebar"
        assert rec.pg == "rebar0.L1"
        # area = π d²/4 for #8 (db = 0.0254)
        assert rec.area == pytest.approx(3.141592653589793 * 0.0254 ** 2 / 4)

        # connectivity resolved from the live mesh (dim-1 cells)
        n_cells = len(rec.connectivity)
        assert n_cells >= 1
        # one CorotTruss per line cell of the bar
        txt = _emit_tcl(fem)
        ct = [ln for ln in txt.splitlines()
              if ln.strip().startswith("element CorotTruss")]
        assert len(ct) == n_cells


def test_emit_elements_unregistered_material_fails_loud():
    """A bar whose material name isn't registered on the bridge fails loud
    at emit (no dangling tag)."""
    with apeGmsh(model_name="rebar_emit_badmat") as g:
        bar = _conformal_bar(g)
        g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                      coupling="conformal", emit_elements=True)
        g.mesh.sizing.set_global_size(0.2)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        # NB: no material named "rebar" declared.
        path = os.path.join(tempfile.gettempdir(), "apegmsh_rebar_badmat.tcl")
        with pytest.raises(ValueError, match="rebar"):
            ops.tcl(path)


def test_emit_elements_beam_not_yet_wired():
    """A straight ``element='beam'`` bar with emit_elements=True raises at
    emit (beam auto-emit is B1b)."""
    with apeGmsh(model_name="rebar_emit_beam") as g:
        bar = _conformal_bar(g, element="beam")
        g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                      coupling="conformal", emit_elements=True)
        g.mesh.sizing.set_global_size(0.2)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01, name="rebar")
        path = os.path.join(tempfile.gettempdir(), "apegmsh_rebar_beam.tcl")
        with pytest.raises(NotImplementedError, match="beam"):
            ops.tcl(path)
