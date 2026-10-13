"""Parity pin for the rebar public surface (X4-e fold, #1809).

The 62 symbols of the step-1 parity table: 22 module-level names (the 16
``apeGmsh.rebar`` exports, 3 ``rebar._geometry`` helpers, 3 composite-module
names reached through both the new home and the ``core.RebarComposite``
facade) and 40 members (27 detailing, 13 on ``g.rebar``). Each member's
parameter names are frozen, so deleting a method or renaming a keyword fails
here. Plus smoke tests for the three that had none: ``g.rebar.stirrup``,
``g.rebar.validate_pre_mesh`` and ``BarCatalog.to_inches``.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

import apeGmsh.rebar as rebar
from apeGmsh import apeGmsh

PACKAGE_ALL = frozenset({
    "Bar", "BarBuilder", "BarLayout", "Cage", "Hook", "Path", "Stirrup",
    "TieLayout", "METADATA", "Vec3",
    "ACI318", "ACI318_seismic", "BarCatalog", "DetailingError",
    "DetailingStandard", "Raw",
})
GEOMETRY = ("bend_plane", "hook_primitives", "outward_tangent")
COMPOSITE_NAMES = ("RebarComposite", "RebarMember", "RebarPlacement")
COMPOSITE_PATHS = ("apeGmsh.rebar._composite", "apeGmsh.core.RebarComposite")

# (class, member) -> parameter names after self; None = a property.
DETAILING = {
    ("BarCatalog", "unit_per_inch"): None,
    ("BarCatalog", "unit_per_mm"): None,
    ("BarCatalog", "bar_diameter"): ("designation",),
    ("BarCatalog", "bar_area"): ("designation",),
    ("BarCatalog", "to_inches"): ("db_model",),
    ("DetailingStandard", "bar_diameter"): ("designation",),
    ("DetailingStandard", "bar_area"): ("designation",),
    ("DetailingStandard", "min_bend_diameter"): ("db", "kind"),
    ("DetailingStandard", "hook_tail"): ("angle", "db", "kind"),
    ("DetailingStandard", "default_corner_radius"): ("db", "kind"),
    ("DetailingStandard", "resolve_length"): ("spec", "db"),
    ("DetailingStandard", "resolve_hook"): ("hook", "db", "kind"),
    ("DetailingStandard", "make_hook"): ("kind", "db", "angle"),
    ("Raw", "min_bend_diameter"): ("db", "kind"),
    ("Raw", "hook_tail"): ("angle", "db", "kind"),
    ("Raw", "default_corner_radius"): ("db", "kind"),
    ("Raw", "resolve_hook"): ("hook", "db", "kind"),
    ("Raw", "make_hook"): ("kind", "db", "angle"),
    ("ACI318", "min_bend_diameter"): ("db", "kind"),
    ("ACI318", "default_corner_radius"): ("db", "kind"),
    ("ACI318", "hook_tail"): ("angle", "db", "kind"),
    ("ACI318", "resolve_hook"): ("hook", "db", "kind"),
    ("ACI318", "make_hook"): ("kind", "db", "angle", "turn", "true_arc", "name"),
    ("ACI318_seismic", "confinement_length"): ("member_depth", "clear_span"),
    ("ACI318_seismic", "confinement_spacing"): ("min_member_dim", "db_long", "hx"),
    ("ACI318_seismic", "beam_confinement_length"): ("member_depth",),
    ("ACI318_seismic", "beam_confinement_spacing"): ("eff_depth", "db_long"),
}

# g.rebar methods -> parameter names after self (12), plus ``placements``.
COMPOSITE = {
    "use_standard": ("standard",),
    "bar": ("points", "db", "material", "role", "element", "start_hook",
            "end_hook", "corner_radius", "curve", "arc_center", "name"),
    "stirrup": ("points", "db", "material", "closure_hook", "role",
                "corner_radius", "curve", "arc_center", "name"),
    "stirrup_rect": ("bx", "by", "cover", "db", "material", "kw"),
    "column": ("section", "height", "cover", "longitudinal", "ties", "base_z",
               "origin", "standard", "top_hook", "bottom_hook", "end_cover",
               "crossties", "confinement_style"),
    "beam": ("section", "length", "cover", "top", "bottom", "stirrups",
             "base_x", "origin", "standard", "end_cover", "crossties",
             "confinement_style"),
    "circular_column": ("diameter", "height", "cover", "n_bars", "bar_db",
                        "bar_material", "ties", "base_z", "origin", "standard",
                        "top_hook", "bottom_hook", "end_cover", "spiral",
                        "n_segments", "bundle", "bundle_pattern", "true_arc"),
    "wall": ("length", "thickness", "height", "cover", "vertical_db",
             "vertical_spacing", "horizontal_db", "horizontal_spacing",
             "curtains", "material", "vertical_material", "horizontal_material",
             "crossties", "crosstie_db", "crosstie_spacing", "base_z", "origin",
             "standard", "end_cover"),
    "bundle": ("points", "n", "db", "material", "toward", "pattern", "spacing",
               "role", "element", "start_hook", "end_hook", "name"),
    "place": ("cage", "into", "coupling", "per_member_coupling", "bond",
              "perfect", "kt", "kt_alpha", "enforce", "bipenalty", "dtcr",
              "tolerance", "snap", "host_dim", "true_arc",
              "on_conformal_infeasible", "twin_tail", "emit_elements", "name"),
    "resolve": (),
    "validate_pre_mesh": (),
}


def _params(func) -> tuple[str, ...]:
    return tuple(p for p in inspect.signature(func).parameters if p != "self")


def test_symbol_count_is_62() -> None:
    module_level = len(PACKAGE_ALL) + len(GEOMETRY) + len(COMPOSITE_NAMES)
    members = len(DETAILING) + len(COMPOSITE) + 1          # + placements
    assert (module_level, members) == (22, 40)


def test_package_exports() -> None:
    assert set(rebar.__all__) == PACKAGE_ALL
    for name in PACKAGE_ALL:
        assert getattr(rebar, name) is not None


def test_geometry_helpers() -> None:
    geo = importlib.import_module("apeGmsh.rebar._geometry")
    for name in GEOMETRY:
        assert callable(getattr(geo, name))


@pytest.mark.parametrize("path", COMPOSITE_PATHS)
def test_composite_names_from_both_paths(path: str) -> None:
    mod = importlib.import_module(path)
    home = importlib.import_module("apeGmsh.rebar._composite")
    for name in COMPOSITE_NAMES:
        assert getattr(mod, name) is getattr(home, name)


@pytest.mark.parametrize(("cls", "member"), sorted(DETAILING))
def test_detailing_member(cls: str, member: str) -> None:
    attr = inspect.getattr_static(getattr(rebar, cls), member)
    want = DETAILING[(cls, member)]
    if want is None:
        assert isinstance(attr, property)
    else:
        assert _params(attr) == want


@pytest.mark.parametrize("method", sorted(COMPOSITE))
def test_composite_method(method: str) -> None:
    from apeGmsh.rebar._composite import RebarComposite
    assert _params(getattr(RebarComposite, method)) == COMPOSITE[method]


def test_session_rebar_is_the_moved_class() -> None:
    from apeGmsh.core.RebarComposite import RebarComposite as Facade
    from apeGmsh.rebar._composite import RebarComposite
    with apeGmsh(model_name="rebar_surface") as g:
        assert type(g.rebar) is RebarComposite is Facade
        assert g.rebar.placements == []


# ---- smoke tests for the previously untested public members -------------

def test_stirrup_smoke() -> None:
    pts = [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.2, 0.3, 0.0),
           (0.0, 0.3, 0.0), (0.0, 0.0, 0.0)]
    with apeGmsh(model_name="rebar_stirrup") as g:
        st = g.rebar.stirrup(pts, db=0.01, material="steel", name="s1")
        explicit = g.rebar.stirrup(pts, db=0.01, material="steel",
                                   closure_hook=rebar.Hook.standard_90())
    assert isinstance(st, rebar.Stirrup)
    assert st.path.points == tuple(pts)
    assert (st.db, st.material, st.role, st.name) == (0.01, "steel", "tie", "s1")
    assert st.closure_hook == rebar.Hook.seismic_135()      # the default closure
    assert explicit.closure_hook == rebar.Hook.standard_90()


def test_validate_pre_mesh_smoke() -> None:
    with apeGmsh(model_name="rebar_validate") as g:
        assert g.rebar.validate_pre_mesh() is None


@pytest.mark.parametrize(("catalog", "db_model", "inches"), [
    (rebar.BarCatalog(), 1.0, 1.0),
    (rebar.BarCatalog(unit_length=0.0254), 0.0254, 1.0),
    (rebar.BarCatalog(unit_length=1.0, base="metric"), 25.4, 1.0),
])
def test_to_inches_smoke(catalog, db_model: float, inches: float) -> None:
    assert catalog.to_inches(db_model) == pytest.approx(inches)
    assert catalog.to_inches(catalog.bar_diameter("#8")) == pytest.approx(1.0)
