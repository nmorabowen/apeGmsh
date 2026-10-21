"""``cpp_class_name_for_pgs`` names the one element class a PG set carries.

The in-process capture spec (deleted in #2191) used it to fill
``element_class_name``: :func:`_identify_layout` needs a single class to
separate two catalog entries sharing a flat column width. ``LadrunoLST``
and ``BezierTri6`` are BOTH 3 Gauss points x 4 components under
``stress_plane_strain`` (and both 3 x 3 under ``stress``), with different
Gauss orderings — so without the hint such a record raises
``Ambiguous catalog match``.

The helper walks the bridge's typed ``Element`` primitives, never the
legacy ``g.opensees`` ``_elem_assignments`` attribute (removed in Phase 8).
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._element_capabilities import cpp_class_name_for_pgs

WITH_ZZ = ("stress_xx", "stress_yy", "stress_xy", "stress_zz")


def _model(element: str, *, pg: str = "Plate", second_pg: bool = False):
    """A 2-D plane-strain plate meshed for ``element``."""
    with apeGmsh(model_name="hint", verbose=False) as g:
        rect = g.model.geometry.add_rectangle(0, 0, 0, 4, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        if element == "LadrunoQuad":
            g.mesh.structured.set_recombine(rect)
        g.mesh.generation.generate(2)
        if element == "LadrunoLST":
            g.mesh.generation.set_order(2, bubble=False)
        g.physical.add(2, [rect], name=pg)
        fem = g.mesh.queries.get_fem_data(dim=2)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.LadrunoJ2(K=1.333e8, G=8.0e7, sig0=1.2e5)
    getattr(ops.element, element)(
        pg=pg, material=mat, thickness=0.1, plane_type="PlaneStrain",
    )
    return fem, ops


def test_legacy_attribute_really_is_gone() -> None:
    """Pins WHY the old lookup was dead, so a revert cannot pass silently.

    Asserted on an INSTANCE, not the class: ``_primitives`` is absent on
    ``apeSees`` the class yet present on every instance, so a class-level
    ``hasattr`` proves nothing either way.
    """
    _fem, ops = _model("LadrunoLST")
    assert not hasattr(ops, "_elem_assignments")
    assert getattr(ops, "_primitives", None), "the new source must exist"


@pytest.mark.parametrize(
    "element", ["LadrunoLST", "LadrunoCST", "LadrunoQuad"],
)
def test_helper_names_the_bridge_class(element: str) -> None:
    """A PG drawn with one class answers that class."""
    _fem, ops = _model(element)
    assert cpp_class_name_for_pgs(ops, ("Plate",)) == element


# =====================================================================
# The helper's own contract
# =====================================================================

def test_helper_returns_none_when_pgs_span_two_classes() -> None:
    """Two classes cannot disambiguate one layout — answer ``None``.

    The transcoder needs exactly one class; guessing either would be
    worse than admitting the record is mixed.
    """
    with apeGmsh(model_name="mixed", verbose=False) as g:
        left = g.model.geometry.add_rectangle(0, 0, 0, 2, 1)
        right = g.model.geometry.add_rectangle(2, 0, 0, 2, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(2)
        g.physical.add(2, [left], name="L")
        g.physical.add(2, [right], name="R")
        fem = g.mesh.queries.get_fem_data(dim=2)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.LadrunoJ2(K=1.333e8, G=8.0e7, sig0=1.2e5)
    ops.element.LadrunoCST(
        pg="L", material=mat, thickness=0.1, plane_type="PlaneStrain",
    )
    ops.element.Tri31(
        pg="R", material=mat, thickness=0.1, plane_type="PlaneStrain",
    )
    assert cpp_class_name_for_pgs(ops, ("L",)) == "LadrunoCST"
    assert cpp_class_name_for_pgs(ops, ("R",)) == "Tri31"
    assert cpp_class_name_for_pgs(ops, ("L", "R")) is None


def test_helper_is_defensive_about_its_inputs() -> None:
    _fem, ops = _model("LadrunoLST")
    assert cpp_class_name_for_pgs(None, ("Plate",)) is None
    assert cpp_class_name_for_pgs(ops, ()) is None
    assert cpp_class_name_for_pgs(ops, None) is None
    assert cpp_class_name_for_pgs(ops, ("NoSuchPG",)) is None


# =====================================================================
# What the hint actually buys — the ambiguity it resolves
# =====================================================================

def test_hint_resolves_the_width_12_plane_strain_ambiguity() -> None:
    """The resolved hint is exactly what ``_identify_layout`` needs.

    12 columns under ``stress_plane_strain`` is genuinely shared by
    ``LadrunoLST`` and ``BezierTri6`` (both 3 GP x 4 comp). Unhinted it
    must fail loud; hinted with what the bridge resolved it must pick
    the 3-GP, 4-component layout.
    """
    from apeGmsh.results.transcoders._recorder import _identify_layout

    _fem, ops = _model("LadrunoLST")
    hint = cpp_class_name_for_pgs(ops, ("Plate",))
    assert hint == "LadrunoLST"

    with pytest.raises(ValueError, match="Ambiguous"):
        _identify_layout("stress_plane_strain", 12)

    layout, cls, _rule = _identify_layout(
        "stress_plane_strain", 12, class_hint=hint,
    )
    assert cls == "LadrunoLST"
    assert layout.n_gauss_points == 3
    assert layout.component_layout == WITH_ZZ
