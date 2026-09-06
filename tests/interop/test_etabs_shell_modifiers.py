"""ETABS area property modifiers -> ``LadrunoShellModifier`` (Ladruno ADR 91).

Before this path existed, an imported ETABS building silently lost its
cracked-section stiffness: a wall assigned ``f11 = f22 = f12 = 0.35``
per ACI 318-25 6.6.3.1.1 was built at gross stiffness, roughly 3x too
stiff in plane, with no warning and a model that converges to
confidently wrong drifts and periods.

Two failure modes here are silent rather than loud, so both are pinned:

* bucketing areas on the section name alone, which hands every wall in
  a group whichever cracking happened to come first;
* the ``mass`` modifier, which this path lumps on the apeGmsh side and
  so would ignore if it were only put on the emitted section.
"""
from __future__ import annotations

import copy

import pytest

from apeGmsh import apeGmsh
from apeGmsh.interop import (
    AreaModifiers,
    StructuralModel,
    build_opensees,
    import_structural_model,
)

from tests.interop.test_etabs_import import _BOX


CRACKED = {"f11": 0.35, "f22": 0.35, "f12": 0.35}


def _model_with(area_mods: dict) -> StructuralModel:
    d = copy.deepcopy(_BOX)
    for area in d["areas"]:
        if area["id"] in area_mods:
            area["modifiers"] = area_mods[area["id"]]
    return StructuralModel.from_dict(d)


def _import(model: StructuralModel, size: float = 2.0):
    sess = apeGmsh(model_name="test_shell_modifiers", verbose=False)
    sess.begin()
    try:
        result = import_structural_model(sess, model)
        sess.mesh.sizing.set_global_size(size)
        sess.mesh.generation.generate(dim=2)
        sess.mesh.partitioning.renumber(base=1)
        fem = sess.mesh.queries.get_fem_data(dim=None)
    finally:
        sess.end()
    return result, fem


def _modifier_line(text: str) -> str:
    return next(
        ln for ln in text.splitlines()
        if ln.startswith("section LadrunoShellModifier")
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestAreaModifiersSchema:
    def test_absent_modifiers_parse_as_none(self) -> None:
        model = StructuralModel.from_dict(_BOX)
        assert all(a.modifiers is None for a in model.areas)

    def test_modifiers_parse(self) -> None:
        model = _model_with({"W1": CRACKED})
        w1 = next(a for a in model.areas if a.id == "W1")
        assert w1.modifiers == AreaModifiers(f11=0.35, f22=0.35, f12=0.35)
        assert w1.modifiers.m11 == 1.0  # unspecified -> gross

    def test_unknown_modifier_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown modifier"):
            AreaModifiers.from_dict({"f11": 0.35, "f33": 0.5})

    def test_negative_modifier_is_refused(self) -> None:
        with pytest.raises(ValueError, match="f11 must be >= 0.0"):
            AreaModifiers(f11=-0.1)

    def test_is_identity(self) -> None:
        assert AreaModifiers().is_identity
        assert not AreaModifiers(f11=0.35).is_identity

    def test_stiffness_excludes_weight(self) -> None:
        stiffness = AreaModifiers(f11=0.35, weight=1.0).stiffness
        assert "weight" not in stiffness
        assert len(stiffness) == 9
        assert stiffness["f11"] == 0.35


# ---------------------------------------------------------------------------
# The headline: modifiers must not be silently dropped
# ---------------------------------------------------------------------------

class TestModifiersReachTheDeck:
    def test_cracked_wall_emits_a_modifier_section(self, tmp_path) -> None:
        model = _model_with({"W1": CRACKED})
        result, fem = _import(model)
        ops = build_opensees(fem, model, result, ndm=3, ndf=6)
        tcl = tmp_path / "out.tcl"
        ops.tcl(str(tcl))
        text = tcl.read_text()

        assert "section LadrunoShellModifier" in text
        line = _modifier_line(text)
        for flag in ("-f11 0.35", "-f22 0.35", "-f12 0.35"):
            assert flag in line
        # Untouched flags stay off the line.
        assert "-m11" not in line and "-mass" not in line

    def test_etabs_path_never_warns_about_a_nonlinear_inner(
        self, tmp_path, recwarn,
    ) -> None:
        # build_opensees always wraps an ElasticMembranePlateSection, so
        # the nonlinear-inner warning must never fire here -- one per
        # wall group would be pure noise on a real building.
        from apeGmsh.opensees.section import (
            ShellModifierNonlinearInnerWarning,
        )

        model = _model_with({"W1": CRACKED})
        result, fem = _import(model)
        build_opensees(fem, model, result, ndm=3, ndf=6)
        assert [
            w for w in recwarn
            if issubclass(w.category, ShellModifierNonlinearInnerWarning)
        ] == []

    def test_uncracked_model_emits_no_wrapper(self, tmp_path) -> None:
        model = StructuralModel.from_dict(_BOX)
        result, fem = _import(model)
        ops = build_opensees(fem, model, result, ndm=3, ndf=6)
        tcl = tmp_path / "out.tcl"
        ops.tcl(str(tcl))
        assert "LadrunoShellModifier" not in tcl.read_text()

    def test_all_ones_modifiers_are_treated_as_gross(self, tmp_path) -> None:
        model = _model_with({"W1": {"f11": 1.0, "m22": 1.0}})
        result, fem = _import(model)
        assert all(ag.modifiers is None for ag in result.area_groups)
        ops = build_opensees(fem, model, result, ndm=3, ndf=6)
        tcl = tmp_path / "out.tcl"
        ops.tcl(str(tcl))
        assert "LadrunoShellModifier" not in tcl.read_text()


# ---------------------------------------------------------------------------
# The trap: bucketing on the section name alone
# ---------------------------------------------------------------------------

class TestBucketingIncludesModifiers:
    @staticmethod
    def _one_section_box() -> dict:
        d = copy.deepcopy(_BOX)
        for area in d["areas"]:
            area["section"] = "SLAB"
        return d

    def test_same_section_different_modifiers_split_into_two_groups(self) -> None:
        d = self._one_section_box()
        for area in d["areas"]:
            if area["id"] == "W1":
                area["modifiers"] = CRACKED
        result, _fem = _import(StructuralModel.from_dict(d))

        slab_groups = [ag for ag in result.area_groups if ag.section == "SLAB"]
        assert len(slab_groups) == 2, "modifiers must split the bucket"
        assert len({ag.pg for ag in slab_groups}) == 2, "PGs must be distinct"
        mods = {ag.pg: ag.modifiers for ag in slab_groups}
        assert sum(m is None for m in mods.values()) == 1
        cracked = next(m for m in mods.values() if m is not None)
        assert cracked.f11 == 0.35

    def test_single_variant_keeps_the_bare_section_name(self) -> None:
        # Backward compatibility: nothing changes for models without
        # modifiers, and a uniformly-cracked section keeps its name too.
        model = _model_with({"W1": CRACKED})
        result, _fem = _import(model)
        assert {ag.pg for ag in result.area_groups} == {"SLAB", "WALL"}

    def test_suffixes_are_stable_across_runs(self) -> None:
        d = self._one_section_box()
        d["areas"][0]["modifiers"] = {"f11": 0.7}
        d["areas"][1]["modifiers"] = {"f11": 0.35}
        first = {
            ag.pg: ag.modifiers
            for ag in _import(StructuralModel.from_dict(d))[0].area_groups
        }
        second = {
            ag.pg: ag.modifiers
            for ag in _import(StructuralModel.from_dict(d))[0].area_groups
        }
        assert first == second
        # Ordered by modifier value, so the harder-cracked one sorts first.
        assert first["SLAB__m1"].f11 == 0.35
        assert first["SLAB__m2"].f11 == 0.7


# ---------------------------------------------------------------------------
# The refusal: weight has no representation
# ---------------------------------------------------------------------------

class TestWeightModifierRefused:
    def test_weight_modifier_raises_with_an_actionable_message(self) -> None:
        model = _model_with({"W1": {"weight": 0.5}})
        sess = apeGmsh(model_name="test_weight_refusal", verbose=False)
        sess.begin()
        try:
            with pytest.raises(ValueError, match="weight modifier"):
                import_structural_model(sess, model)
        finally:
            sess.end()

    def test_unit_weight_modifier_is_accepted(self) -> None:
        model = _model_with({"W1": dict(CRACKED, weight=1.0)})
        result, _fem = _import(model)
        assert any(ag.modifiers is not None for ag in result.area_groups)


# ---------------------------------------------------------------------------
# The mass modifier: applied where this path actually creates mass
# ---------------------------------------------------------------------------

def _total_nodal_mass(model: StructuralModel, tmp_path, tag: str) -> float:
    result, fem = _import(model)
    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    tcl = tmp_path / f"{tag}.tcl"
    ops.tcl(str(tcl))
    return sum(
        float(ln.split()[2])
        for ln in tcl.read_text().splitlines()
        if ln.startswith("mass ")
    )


class TestMassModifier:
    def test_mass_modifier_scales_the_lumped_areal_density(self, tmp_path) -> None:
        # This path lumps shell mass on the apeGmsh side (g.masses.surface),
        # so a -mass flag on the emitted section alone would be silently
        # inert here -- the lumped total must actually move.
        gross = _total_nodal_mass(
            StructuralModel.from_dict(_BOX), tmp_path, "gross",
        )
        halved = _total_nodal_mass(
            _model_with({"W1": {"mass": 0.5}}), tmp_path, "halved",
        )
        # Wall W1 is 4 m x 3 m, rho = 2.4, t = 0.25 -> 7.2 gross; halving
        # its mass modifier must remove exactly half of that.
        assert gross - halved == pytest.approx(3.6, rel=1e-9)

    def test_mass_modifier_is_recorded_on_the_group(self) -> None:
        result, _fem = _import(_model_with({"W1": {"mass": 0.5}}))
        wall = next(ag for ag in result.area_groups if ag.section == "WALL")
        assert wall.modifiers is not None and wall.modifiers.mass == 0.5

    def test_mass_modifier_also_reaches_the_section(self, tmp_path) -> None:
        model = _model_with({"W1": {"mass": 0.5}})
        result, fem = _import(model)
        ops = build_opensees(fem, model, result, ndm=3, ndf=6)
        tcl = tmp_path / "out.tcl"
        ops.tcl(str(tcl))
        assert "-mass 0.5" in _modifier_line(tcl.read_text())
