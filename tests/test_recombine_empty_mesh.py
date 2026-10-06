"""#1327: ``g.mesh.structured.recombine()`` before ``generate()`` is a no-op.

``recombine()`` merges the triangles of an existing mesh, so called before
``generate()`` it does nothing and the surface comes out all-triangle. A
quad-only element on that PG used to fail at emit with a bare
"ShellMITC4: expected 4 node tags, got 3." that said nothing about
recombination.

Two contracts, both verified with ``-W error::...RecombineEmptyMeshWarning``:

* ``recombine()`` warns ``RecombineEmptyMeshWarning`` iff the model holds
  no 2-D elements (it stays silent on a generated 2-D mesh).
* A quad-only element on a triangle PG raises ``BridgeError`` before the
  fan-out, naming the PG and ``set_recombine``.
"""
from __future__ import annotations

import warnings

import gmsh
import pytest

from apeGmsh.mesh._mesh_structured import RecombineEmptyMeshWarning
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    check_quad_element_on_triangles,
)
from apeGmsh.opensees.element.shell import ShellMITC3, ShellMITC4
from apeGmsh.opensees.element.solid import Tri31


def _slab(g, *, pre_generate_recombine: bool = False,
          set_recombine: bool = False) -> None:
    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="sq")
    g.physical.add_surface("sq", name="Slab")
    g.mesh.sizing.set_global_size(0.25)
    if set_recombine:
        g.mesh.structured.set_recombine("Slab")
    if pre_generate_recombine:
        g.mesh.structured.recombine()
    g.mesh.generation.generate(dim=2)


def _shell_deck(fem, element: str, tmp_path) -> None:
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2, rho=0)
    getattr(ops.element, element)(pg="Slab", section=sec)
    ops.tcl(str(tmp_path / "slab.tcl"))


# ---------------------------------------------------------------------------
# recombine() on an empty mesh
# ---------------------------------------------------------------------------

def test_recombine_before_generate_warns(g):
    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="sq")
    with pytest.warns(RecombineEmptyMeshWarning, match="set_recombine"):
        g.mesh.structured.recombine()


def test_recombine_after_1d_only_mesh_warns(g):
    """No 2-D elements at all (generate(1)) is still nothing to merge."""
    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="sq")
    g.mesh.generation.generate(dim=1)
    with pytest.warns(RecombineEmptyMeshWarning):
        g.mesh.structured.recombine()


def test_recombine_after_generate_is_silent_and_merges(g):
    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="sq")
    g.mesh.sizing.set_global_size(0.25)
    g.mesh.generation.generate(dim=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RecombineEmptyMeshWarning)
        g.mesh.structured.recombine()
    etypes, _, _ = gmsh.model.mesh.getElements(dim=2, tag=-1)
    assert 3 in [int(t) for t in etypes]          # quads exist now


# ---------------------------------------------------------------------------
# Emit: quad-only element on triangle cells names the cause
# ---------------------------------------------------------------------------

def test_issue_repro_quad_shell_on_triangles_names_pg_and_fix(g, tmp_path):
    with pytest.warns(RecombineEmptyMeshWarning):
        _slab(g, pre_generate_recombine=True)
    fem = g.mesh.queries.get_fem_data(dim=2)
    assert {t.name for t in fem.elements.types} == {"tri3"}
    with pytest.raises(BridgeError) as exc:
        _shell_deck(fem, "ShellMITC4", tmp_path)
    msg = str(exc.value)
    assert "ShellMITC4(pg='Slab')" in msg
    assert "3-node triangles" in msg
    assert "set_recombine('Slab')" in msg
    assert "ShellMITC3" in msg


def test_set_recombine_gives_quads_and_emits(g, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RecombineEmptyMeshWarning)
        _slab(g, set_recombine=True)
    fem = g.mesh.queries.get_fem_data(dim=2)
    assert {t.name for t in fem.elements.types} == {"quad4"}
    _shell_deck(fem, "ShellMITC4", tmp_path)
    assert "ShellMITC4" in (tmp_path / "slab.tcl").read_text()


def test_triangle_shell_on_triangles_still_emits(g, tmp_path):
    _slab(g)
    fem = g.mesh.queries.get_fem_data(dim=2)
    _shell_deck(fem, "ShellMITC3", tmp_path)
    assert "ShellMITC3" in (tmp_path / "slab.tcl").read_text()


# ---------------------------------------------------------------------------
# The check itself, on the list form the partitioned fan-out passes
# ---------------------------------------------------------------------------

def _sec():
    from apeGmsh.opensees.section.plate import ElasticMembranePlateSection
    return ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2, rho=0.0)


def test_check_raises_on_tri6_for_quad_element():
    spec = ShellMITC4(pg="Slab", section=_sec())
    with pytest.raises(BridgeError, match="6-node triangles"):
        check_quad_element_on_triangles(spec, [(1, (1, 2, 3, 4, 5, 6))])


def test_check_passes_quads_and_ignores_non_quad_classes():
    check_quad_element_on_triangles(
        ShellMITC4(pg="Slab", section=_sec()), [(1, (1, 2, 3, 4))])
    check_quad_element_on_triangles(
        ShellMITC3(pg="Slab", section=_sec()), [(1, (1, 2, 3))])
    from apeGmsh.opensees.material.nd import ElasticIsotropic
    check_quad_element_on_triangles(
        Tri31(pg="Slab", material=ElasticIsotropic(E=1.0, nu=0.2),
              thickness=1.0),
        [(1, (1, 2, 3))])
