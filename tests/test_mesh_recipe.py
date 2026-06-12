"""
Tests for ``g.mesh.recipe`` — one-call unstructured / structured
meshing recipes (ADR 0059).

Covers: the whole-model one-liners (including the zero-argument bbox
heuristic), the targeted compose-then-generate flow with field-based
region sizing, the ``generate=None`` auto-by-scope rule, the
``fallback`` modes of ``structured`` (unstructured / warn / strict),
the mixed quad/tri interface guard (``check()`` standalone and
auto-run at recipe generate; raw ``generate()`` untouched), folding a
user-authored background field into the recipe's Min combiner, and the
closed-curve warn-skip fix in ``set_transfinite``'s cascade.
"""
from __future__ import annotations

import warnings

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh import MeshRecipeError


# =====================================================================
# Helpers
# =====================================================================


def _n_elements(dim: int, tag: int = -1) -> int:
    _, tags, _ = gmsh.model.mesh.getElements(dim, tag)
    return sum(len(t) for t in tags)


def _element_types(dim: int, tag: int = -1) -> set[int]:
    types, _, _ = gmsh.model.mesh.getElements(dim, tag)
    return {int(t) for t in types}


def _fragmented_box_pair(g, la: str = "a", lb: str = "b") -> None:
    """Two unit boxes sharing one face, conformally fragmented."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label=la)
    g.model.geometry.add_box(1, 0, 0, 1, 1, 1, label=lb)
    g.model.boolean.fragment(la, lb)


# =====================================================================
# Whole-model unstructured
# =====================================================================


class TestUnstructuredWholeModel:
    def test_one_liner_generates(self):
        with apeGmsh(model_name="recipe_u1") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.unstructured(max_size=0.3)
            assert _n_elements(3) > 0

    def test_disables_cad_point_size_sources(self):
        with apeGmsh(model_name="recipe_u2") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.unstructured(max_size=0.3)
            assert gmsh.option.getNumber("Mesh.MeshSizeFromPoints") == 0.0
            assert gmsh.option.getNumber("Mesh.MeshSizeFromCurvature") == 0.0

    def test_sets_global_band(self):
        with apeGmsh(model_name="recipe_u3") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.unstructured(max_size=0.4, min_size=0.1)
            assert gmsh.option.getNumber("Mesh.MeshSizeMax") == 0.4
            assert gmsh.option.getNumber("Mesh.MeshSizeMin") == 0.1

    def test_zero_argument_heuristic(self):
        """No sizes at all — bbox-diagonal default still yields a mesh."""
        with apeGmsh(model_name="recipe_u4") as g:
            g.model.geometry.add_box(0, 0, 0, 10, 10, 10, label="b")
            g.mesh.recipe.unstructured()
            assert _n_elements(3) > 0
            # diag = 10*sqrt(3) ≈ 17.3 → max_size ≈ 0.87
            assert gmsh.option.getNumber("Mesh.MeshSizeMax") == pytest.approx(
                10.0 * 3.0 ** 0.5 / 20.0
            )

    def test_empty_model_fails_loud(self):
        with apeGmsh(model_name="recipe_u5") as g:
            with pytest.raises(MeshRecipeError, match="bounding box is empty"):
                g.mesh.recipe.unstructured()

    def test_bad_band_rejected(self):
        with apeGmsh(model_name="recipe_u6") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            with pytest.raises(ValueError, match="max_size"):
                g.mesh.recipe.unstructured(max_size=-1.0)
            with pytest.raises(ValueError, match="min_size"):
                g.mesh.recipe.unstructured(max_size=0.3, min_size=0.5)

    def test_2d_model_generates_dim_2(self):
        with apeGmsh(model_name="recipe_u7") as g:
            g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="r")
            g.mesh.recipe.unstructured(max_size=0.2)
            assert _n_elements(2) > 0


# =====================================================================
# Targeted unstructured — field-based region sizing
# =====================================================================


class TestUnstructuredTargeted:
    def test_targeted_does_not_generate(self):
        with apeGmsh(model_name="recipe_t1") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.unstructured("b", max_size=0.3)
            assert _n_elements(3) == 0

    def test_targeted_generate_true_overrides(self):
        with apeGmsh(model_name="recipe_t2") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.unstructured("b", max_size=0.3, generate=True)
            assert _n_elements(3) > 0

    def test_whole_model_generate_false_overrides(self):
        with apeGmsh(model_name="recipe_t3") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.unstructured(max_size=0.3, generate=False)
            assert _n_elements(3) == 0

    def test_disjoint_regions_get_their_sizes(self):
        """Two disjoint boxes — no shared interface, crisp contrast."""
        with apeGmsh(model_name="recipe_t4") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="fine")
            g.model.geometry.add_box(3, 0, 0, 1, 1, 1, label="coarse")
            g.mesh.recipe.unstructured("fine", max_size=0.12)
            g.mesh.recipe.unstructured("coarse", max_size=0.5)
            g.mesh.generation.generate(dim=3)
            t_fine = g.labels.entities("fine", dim=3)[0]
            t_coarse = g.labels.entities("coarse", dim=3)[0]
            n_fine = _n_elements(3, t_fine)
            n_coarse = _n_elements(3, t_coarse)
            # equal volumes: (0.5/0.12)^3 ≈ 72x nominal; demand 5x with
            # generous slack for mesher variance.
            assert n_fine > 5 * n_coarse, (n_fine, n_coarse)

    def test_conformal_regions_compose(self):
        """Shared face — sizes still differ (interface transition allowed)."""
        with apeGmsh(model_name="recipe_t5") as g:
            _fragmented_box_pair(g, "fine", "coarse")
            g.mesh.recipe.unstructured("fine", max_size=0.12)
            g.mesh.recipe.unstructured("coarse", max_size=0.5)
            g.mesh.generation.generate(dim=3)
            t_fine = g.labels.entities("fine", dim=3)[0]
            t_coarse = g.labels.entities("coarse", dim=3)[0]
            assert _n_elements(3, t_fine) > 2 * _n_elements(3, t_coarse)

    def test_region_sizing_uses_fields_not_point_lc(self):
        with apeGmsh(model_name="recipe_t6") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.unstructured("b", max_size=0.3)
            kinds = [d.get('kind') for d in g.mesh._directives]
            assert 'recipe_region_field' in kinds
            assert 'set_size' not in kinds  # no point-lc path

    def test_unknown_target_fails_loud(self):
        with apeGmsh(model_name="recipe_t7") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            with pytest.raises(Exception, match="nope"):
                g.mesh.recipe.unstructured("nope", max_size=0.3)

    def test_user_background_field_is_folded(self):
        """A pre-existing user background field joins the Min combiner."""
        with apeGmsh(model_name="recipe_t8") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            user_f = g.mesh.field.math_eval("0.25")
            g.mesh.field.set_background(user_f)
            g.mesh.recipe.unstructured("b", max_size=0.1)
            recipe = g.mesh.recipe
            assert user_f in recipe._folded_external_tags
            assert recipe._min_field_tag is not None
            # the recipe Min field is now the registered background
            assert g.mesh._background_field_tag == recipe._min_field_tag


# =====================================================================
# Structured recipe
# =====================================================================


class TestStructuredRecipe:
    def test_whole_model_hex_box(self):
        with apeGmsh(model_name="recipe_s1") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.structured(n=4)
            assert _element_types(3) == {5}        # hex8 only
            assert _n_elements(3) == 27            # 3^3

    def test_recombine_false_gives_simplices(self):
        with apeGmsh(model_name="recipe_s2") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.structured(n=4, recombine=False)
            assert 5 not in _element_types(3)
            assert _n_elements(3) > 0

    def test_targeted_structured_composes(self):
        with apeGmsh(model_name="recipe_s3") as g:
            _fragmented_box_pair(g)
            g.mesh.recipe.structured("a", n=4)
            assert _n_elements(3) == 0             # declared, not generated
            g.mesh.recipe.structured("b", n=4, generate=True)
            assert _element_types(3) == {5}
            assert _n_elements(3) == 54            # 27 + 27

    def test_2d_whole_model_targets_surfaces(self):
        """No volumes — whole-model structured falls through to surfaces."""
        with apeGmsh(model_name="recipe_s4") as g:
            g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="r")
            g.mesh.recipe.structured(n=4)
            assert _element_types(2) == {3}        # quad4 only
            assert _n_elements(2) == 9

    def test_size_form(self):
        with apeGmsh(model_name="recipe_s5") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.structured(size=0.5)
            assert _element_types(3) == {5}
            assert _n_elements(3) == 8             # 2^3

    def test_bad_fallback_rejected(self):
        with apeGmsh(model_name="recipe_s6") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            with pytest.raises(ValueError, match="fallback"):
                g.mesh.recipe.structured(n=4, fallback="explode")


class TestStructuredFallback:
    """A cylinder is not hex-decomposable (closed rim curves)."""

    def test_default_fallback_still_meshes_everything(self):
        with apeGmsh(model_name="recipe_f1") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
            g.model.geometry.add_cylinder(3, 0, 0, 0, 0, 1, 0.5, label="cyl")
            with pytest.warns(UserWarning, match="set_transfinite"):
                g.mesh.recipe.structured(size=0.25, recombine=False)
            t_cyl = g.labels.entities("cyl", dim=3)[0]
            t_box = g.labels.entities("box", dim=3)[0]
            assert _n_elements(3, t_cyl) > 0       # fallback meshed it
            assert _n_elements(3, t_box) > 0

    def test_fallback_emits_region_field_at_equivalent_size(self):
        with apeGmsh(model_name="recipe_f2") as g:
            g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 1, 0.5, label="cyl")
            with pytest.warns(UserWarning):
                g.mesh.recipe.structured(size=0.25, generate=False)
            fields = [
                d for d in g.mesh._directives
                if d.get('kind') == 'recipe_region_field'
            ]
            assert len(fields) == 1
            assert fields[0]['size'] == pytest.approx(0.25)

    def test_fallback_n_form_derives_size(self):
        with apeGmsh(model_name="recipe_f3") as g:
            g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 1, 0.5, label="cyl")
            with pytest.warns(UserWarning):
                g.mesh.recipe.structured(n=5, generate=False)
            fields = [
                d for d in g.mesh._directives
                if d.get('kind') == 'recipe_region_field'
            ]
            assert len(fields) == 1
            assert fields[0]['size'] > 0.0

    def test_strict_raises(self):
        with apeGmsh(model_name="recipe_f4") as g:
            g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 1, 0.5, label="cyl")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(MeshRecipeError, match="strict"):
                    g.mesh.recipe.structured(size=0.25, fallback="strict")

    def test_warn_mode_applies_no_fallback_field(self):
        with apeGmsh(model_name="recipe_f5") as g:
            g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 1, 0.5, label="cyl")
            with pytest.warns(UserWarning):
                g.mesh.recipe.structured(
                    size=0.25, fallback="warn", generate=False,
                )
            fields = [
                d for d in g.mesh._directives
                if d.get('kind') == 'recipe_region_field'
            ]
            assert fields == []

    def test_closed_curve_volume_warn_skips_not_crashes(self):
        """The pre-ADR-0058 behavior was an uncaught ValueError."""
        with apeGmsh(model_name="recipe_f6") as g:
            g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 1, 0.5, label="cyl")
            with pytest.warns(UserWarning, match="chord direction"):
                g.mesh.structured.set_transfinite(size=0.25)
            assert g.mesh.structured._last_skipped != []


# =====================================================================
# Mixed-interface guard
# =====================================================================


class TestMixedInterfaceGuard:
    def test_check_raises_on_hex_tet_interface(self):
        with apeGmsh(model_name="recipe_g1") as g:
            _fragmented_box_pair(g)
            g.mesh.recipe.structured("a", n=4)
            g.mesh.recipe.unstructured("b", max_size=0.4)
            with pytest.raises(MeshRecipeError, match="recombine=False"):
                g.mesh.recipe.check()

    def test_guard_auto_runs_on_recipe_generate(self):
        with apeGmsh(model_name="recipe_g2") as g:
            _fragmented_box_pair(g)
            g.mesh.recipe.structured("a", n=4)
            with pytest.raises(MeshRecipeError):
                g.mesh.recipe.unstructured("b", max_size=0.4, generate=True)

    def test_recombine_false_passes_guard_and_meshes(self):
        with apeGmsh(model_name="recipe_g3") as g:
            _fragmented_box_pair(g)
            g.mesh.recipe.structured("a", n=4, recombine=False)
            g.mesh.recipe.unstructured("b", max_size=0.4)
            g.mesh.recipe.check()                  # no raise
            g.mesh.generation.generate(dim=3)
            assert _n_elements(3) > 0

    def test_all_structured_passes_guard(self):
        with apeGmsh(model_name="recipe_g4") as g:
            _fragmented_box_pair(g)
            g.mesh.recipe.structured("a", n=4)
            g.mesh.recipe.structured("b", n=4)
            g.mesh.recipe.check()                  # no raise

    def test_raw_generate_is_untouched(self):
        """The guard never auto-wires into g.mesh.generation.generate()."""
        with apeGmsh(model_name="recipe_g5") as g:
            _fragmented_box_pair(g)
            g.mesh.recipe.structured("a", n=4)
            g.mesh.recipe.unstructured("b", max_size=0.4)
            # raw generate path: gmsh fails or degrades on its own terms,
            # but apeGmsh must not raise MeshRecipeError here.
            try:
                g.mesh.generation.generate(dim=3)
            except MeshRecipeError:
                pytest.fail("guard leaked into the raw generate() path")
            except Exception:
                pass  # gmsh-level failure is expected and acceptable

    def test_disjoint_structured_and_unstructured_pass(self):
        with apeGmsh(model_name="recipe_g6") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="a")
            g.model.geometry.add_box(3, 0, 0, 1, 1, 1, label="b")
            g.mesh.recipe.structured("a", n=4)
            g.mesh.recipe.unstructured("b", max_size=0.4)
            g.mesh.recipe.check()                  # no shared face → fine
            g.mesh.generation.generate(dim=3)
            assert _n_elements(3) > 0
