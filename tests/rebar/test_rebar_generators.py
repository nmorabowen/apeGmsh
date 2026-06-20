"""P4 — standardized column/beam generators + fluent BarBuilder (ADR 0066 §8)."""
from __future__ import annotations

import pytest

import warnings

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import (
    Bar, BarBuilder, BarLayout, Hook, TieLayout,
)
from apeGmsh.rebar.detailing import ACI318, ACI318_seismic, BarCatalog

# A bar catalogue whose model unit is the metre (1 in = 0.0254 m) so the
# absolute ACI floors (18 in l_o, 3 in hook) scale into a metres model.
_M = BarCatalog(unit_length=0.0254, base="imperial")


def test_column_perimeter_bars_and_densified_ties():
    with apeGmsh(model_name="gen_col") as g:
        cage = g.rebar.column(
            section=("rect", 0.5, 0.5), height=3.0, cover=0.04,
            longitudinal=BarLayout(n_x=3, n_y=3, db=0.025),
            ties=TieLayout(db=0.01, spacing=0.2, hinge_spacing=0.1,
                           hinge_length=0.5))
        # 3×3 perimeter = 2·3 + 2·3 − 4 = 8 longitudinal bars, all vertical,
        # inset by cover (cross-ties carry role="crosstie" — see below)
        longitudinal = [b for b in cage.bars if b.role == "longitudinal"]
        assert len(longitudinal) == 8
        for b in longitudinal:
            p0, p1 = b.path.points
            assert p0[2] == pytest.approx(0.04) and p1[2] == pytest.approx(2.96)
            assert p0[0] == p1[0] and p0[1] == p1[1]
        # ties densified in the end hinge zones → more than the uniform count
        zs = sorted(s.path.points[0][2] for s in cage.stirrups)
        assert len(zs) > round(3.0 / 0.2)                 # > uniform
        gaps = [b - a for a, b in zip(zs, zs[1:])]
        assert min(gaps) == pytest.approx(0.1, abs=1e-6)  # hinge_spacing present
        assert max(gaps) == pytest.approx(0.2, abs=1e-6)  # regular spacing present


def test_column_crossties_support_intermediate_bars():
    with apeGmsh(model_name="gen_col_ct") as g:
        cage = g.rebar.column(
            section=("rect", 0.6, 0.6), height=3.0, cover=0.04,
            longitudinal=BarLayout(n_x=3, n_y=4, db=0.025),
            ties=TieLayout(db=0.01, spacing=0.2))
        cts = [b for b in cage.bars if b.role == "crosstie"]
        levels = len(cage.stirrups)
        z_levels = [s.path.points[0][2] for s in cage.stirrups]
        # interior bars per face pair: (n_x-2) y-legs + (n_y-2) x-legs = 1+2=3
        per_level = (3 - 2) + (4 - 2)
        assert per_level == 3
        assert len(cts) == per_level * levels
        for b in cts:
            p0, p1 = b.path.points
            assert p0[2] == p1[2]                       # horizontal (one z level)
            assert any(p0[2] == pytest.approx(zl) for zl in z_levels)  # tie level
            # spans the section in exactly one in-plane direction
            assert (p0[0] != p1[0]) ^ (p0[1] != p1[1])
            # hooked both ends, one 135° + one 90°
            assert b.start_hook is not None and b.end_hook is not None
            assert {b.start_hook.angle, b.end_hook.angle} == {90.0, 135.0}


def test_column_crossties_alternate_end_for_end():
    with apeGmsh(model_name="gen_col_alt") as g:
        cage = g.rebar.column(
            section=("rect", 0.6, 0.6), height=3.0, cover=0.04,
            longitudinal=BarLayout(n_x=3, n_y=3, db=0.025),
            ties=TieLayout(db=0.01, spacing=0.3))
        # the single y-leg (interior x) across successive levels alternates
        # which end carries the 135° seismic hook (ACI 318 §18.7.5.2)
        y_legs = sorted(
            (b for b in cage.bars
             if b.role == "crosstie" and b.path.points[0][0] == b.path.points[1][0]),
            key=lambda b: b.path.points[0][2])
        starts = [b.start_hook.angle for b in y_legs]
        assert len(starts) >= 2
        assert all(a != b for a, b in zip(starts, starts[1:]))   # strictly flip


def test_column_crossties_opt_out():
    with apeGmsh(model_name="gen_col_noct") as g:
        with pytest.warns(UserWarning, match="laterally supported"):
            cage = g.rebar.column(
                section=("rect", 0.6, 0.6), height=2.0, cover=0.04,
                longitudinal=BarLayout(n_x=3, n_y=3, db=0.025),
                ties=TieLayout(db=0.01, spacing=0.25), crossties=False)
        assert not [b for b in cage.bars if b.role == "crosstie"]


def test_column_crossties_place_embedded_with_standard():
    with apeGmsh(model_name="gen_col_ct_place") as g:
        g.rebar.use_standard(ACI318_seismic())
        vol = g.model.geometry.add_box(0, 0, 0, 0.6, 0.6, 3.0)
        g.physical.add_volume([vol], name="Col")
        cage = g.rebar.column(
            section=("rect", 0.6, 0.6), height=3.0, cover=0.05,
            longitudinal=BarLayout(n_x=3, n_y=3, db=0.025),
            ties=TieLayout(db=0.01, spacing=0.6))
        g.rebar.place(cage, into="Col", coupling="embedded", perfect=1.0e8)
        # every member — longitudinal bars + hoops + cross-ties — embeds
        assert len(g.reinforce.reinforce_defs) == len(cage.bars) + len(cage.stirrups)
        # the seismic standard resolved the cross-tie hook tails (no drop)
        assert any(b.role == "crosstie" for b in cage.bars)


def test_beam_crossties_vertical_legs_for_aligned_bars():
    with apeGmsh(model_name="gen_beam_ct") as g:
        cage = g.rebar.beam(
            section=("rect", 0.4, 0.6), length=4.0, cover=0.04,
            top=BarLayout(n_x=4, db=0.02), bottom=BarLayout(n_x=4, db=0.02),
            stirrups=TieLayout(db=0.01, spacing=0.5))
        cts = [b for b in cage.bars if b.role == "crosstie"]
        stations = len(cage.stirrups)
        # 4 bars per face → 2 interior pairs → 2 legs per station
        assert len(cts) == 2 * stations
        for b in cts:
            p0, p1 = b.path.points
            assert p0[0] == p1[0]                       # vertical leg (constant x)
            assert p1[2] > p0[2]                        # bottom bar → top bar
            assert {b.start_hook.angle, b.end_hook.angle} == {90.0, 135.0}


def test_beam_crossties_warn_on_count_mismatch():
    with apeGmsh(model_name="gen_beam_ct_mm") as g:
        with pytest.warns(UserWarning, match="counts differ"):
            cage = g.rebar.beam(
                section=("rect", 0.4, 0.6), length=4.0, cover=0.04,
                top=BarLayout(n_x=2, db=0.02), bottom=BarLayout(n_x=4, db=0.02),
                stirrups=TieLayout(db=0.01, spacing=0.5))
        # top has no interior bar → no aligned pair → no legs
        assert not [b for b in cage.bars if b.role == "crosstie"]


def test_column_overlapping_hoops_tile_the_core():
    with apeGmsh(model_name="gen_col_oh") as g:
        cage = g.rebar.column(
            section=("rect", 0.6, 0.6), height=3.0, cover=0.04,
            longitudinal=BarLayout(n_x=3, n_y=3, db=0.025),
            ties=TieLayout(db=0.01, spacing=0.3),
            confinement_style="overlapping_hoops")
        levels = len({round(s.path.points[0][2], 9) for s in cage.stirrups})
        # no straight cross-tie legs in this style
        assert not [b for b in cage.bars if b.role == "crosstie"]
        # perimeter hoop + (n_x-1)*(n_y-1)=4 cell-hoops per level
        per_level = 1 + (3 - 1) * (3 - 1)
        assert len(cage.stirrups) == per_level * levels

        # cell-hoops span a single sub-cell (~¼ section); the perimeter hoop
        # spans nearly the full section — separate them by x-extent
        def _xext(s):
            xv = [p[0] for p in s.path.points]
            return max(xv) - min(xv)
        cells = [s for s in cage.stirrups if _xext(s) < 0.4]
        assert len(cells) == (3 - 1) * (3 - 1) * levels
        for s in cells:
            pts = s.path.points
            assert pts[0] == pts[-1]                      # closed ring
            assert len({(round(p[0], 9), round(p[1], 9)) for p in pts}) == 4


def test_column_overlapping_hoops_places_embedded():
    with apeGmsh(model_name="gen_col_oh_place") as g:
        g.rebar.use_standard(ACI318_seismic(_M))
        vol = g.model.geometry.add_box(0, 0, 0, 0.6, 0.6, 3.0)
        g.physical.add_volume([vol], name="Col")
        cage = g.rebar.column(
            section=("rect", 0.6, 0.6), height=3.0, cover=0.05,
            longitudinal=BarLayout(n_x=3, n_y=3, db=0.025),
            ties=TieLayout(db=0.01, spacing=0.6, hinge_spacing=0.6,
                           hinge_length=0.6),
            confinement_style="overlapping_hoops")
        g.rebar.place(cage, into="Col", coupling="embedded", perfect=1.0e8)
        assert len(g.reinforce.reinforce_defs) == len(cage.bars) + len(cage.stirrups)


def test_column_confinement_style_validated():
    with apeGmsh(model_name="gen_col_oh_bad") as g:
        with pytest.raises(ValueError, match="confinement_style"):
            g.rebar.column(
                section=("rect", 0.6, 0.6), height=3.0, cover=0.04,
                longitudinal=BarLayout(n_x=3, n_y=3, db=0.025),
                ties=TieLayout(db=0.01, spacing=0.3),
                confinement_style="spiral")


def test_column_seismic_confinement_auto_derived():
    std = ACI318_seismic(_M)
    with apeGmsh(model_name="gen_col_conf") as g:
        g.rebar.use_standard(std)
        with pytest.warns(UserWarning, match="confinement zone auto-derived"):
            cage = g.rebar.column(
                section=("rect", 0.6, 0.6), height=3.0, cover=0.05,
                longitudinal=BarLayout(n_x=3, n_y=3, db=0.025),
                ties=TieLayout(db=0.01, spacing=0.3))     # no hinge params
        # expected l_o / s_o straight from the standard (same geometry inputs)
        inset = 0.05 + 0.01 + 0.025 / 2.0
        hx = (0.6 - 2 * inset) / (3 - 1)
        s_o = std.confinement_spacing(min_member_dim=0.6, db_long=0.025, hx=hx)
        zs = sorted(s.path.points[0][2] for s in cage.stirrups)
        gaps = [round(b - a, 9) for a, b in zip(zs, zs[1:])]
        # the dense end-zone spacing s_o and the regular middle spacing both
        # appear (plus small partial gaps at the zone boundaries)
        assert any(g == pytest.approx(s_o, abs=1e-6) for g in gaps)   # dense = s_o
        assert any(g == pytest.approx(0.3, abs=1e-6) for g in gaps)   # ties.spacing
        assert min(gaps) < 0.3                                # genuinely densified


def test_column_seismic_confinement_explicit_overrides():
    with apeGmsh(model_name="gen_col_conf_ovr") as g:
        g.rebar.use_standard(ACI318_seismic(_M))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cage = g.rebar.column(
                section=("rect", 0.6, 0.6), height=3.0, cover=0.05,
                longitudinal=BarLayout(n_x=2, n_y=2, db=0.025),
                ties=TieLayout(db=0.01, spacing=0.3, hinge_spacing=0.08,
                               hinge_length=0.5))
        assert not any("auto-derived" in str(w.message) for w in rec)
        zs = sorted(s.path.points[0][2] for s in cage.stirrups)
        gaps = [round(b - a, 9) for a, b in zip(zs, zs[1:])]
        assert any(g == pytest.approx(0.08, abs=1e-6) for g in gaps)   # honoured
        assert any(g == pytest.approx(0.3, abs=1e-6) for g in gaps)


def test_column_non_seismic_stays_uniform():
    with apeGmsh(model_name="gen_col_nonseis") as g:
        g.rebar.use_standard(ACI318(_M))                     # non-seismic
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cage = g.rebar.column(
                section=("rect", 0.6, 0.6), height=3.0, cover=0.05,
                longitudinal=BarLayout(n_x=2, n_y=2, db=0.025),
                ties=TieLayout(db=0.01, spacing=0.3))        # no hinge params
        assert not any("auto-derived" in str(w.message) for w in rec)
        zs = sorted(s.path.points[0][2] for s in cage.stirrups)
        gaps = [round(b - a, 9) for a, b in zip(zs, zs[1:])]
        assert len(set(gaps)) == 1                            # uniform, no zone


def test_beam_seismic_confinement_auto_derived():
    std = ACI318_seismic(_M)
    with apeGmsh(model_name="gen_beam_conf") as g:
        g.rebar.use_standard(std)
        with pytest.warns(UserWarning, match="hoop confinement zone auto-derived"):
            cage = g.rebar.beam(
                section=("rect", 0.4, 0.6), length=5.0, cover=0.05,
                top=BarLayout(n_x=2, db=0.025), bottom=BarLayout(n_x=2, db=0.025),
                stirrups=TieLayout(db=0.01, spacing=0.25))   # no hinge params
        eff_depth = 0.6 - 0.05 - 0.01 - 0.025 / 2.0
        s_h = std.beam_confinement_spacing(eff_depth=eff_depth, db_long=0.025)
        xs = sorted(s.path.points[0][0] for s in cage.stirrups)   # x-stations
        gaps = [round(b - a, 9) for a, b in zip(xs, xs[1:])]
        assert any(g == pytest.approx(s_h, abs=1e-6) for g in gaps)   # dense = s_h
        assert any(g == pytest.approx(0.25, abs=1e-6) for g in gaps)  # regular
        assert min(gaps) < 0.25                               # genuinely densified


def test_beam_non_seismic_stays_uniform():
    with apeGmsh(model_name="gen_beam_nonseis") as g:
        g.rebar.use_standard(ACI318(_M))                     # non-seismic
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cage = g.rebar.beam(
                section=("rect", 0.4, 0.6), length=5.0, cover=0.05,
                top=BarLayout(n_x=2, db=0.025), bottom=BarLayout(n_x=2, db=0.025),
                stirrups=TieLayout(db=0.01, spacing=0.25))
        assert not any("auto-derived" in str(w.message) for w in rec)
        xs = sorted(s.path.points[0][0] for s in cage.stirrups)
        gaps = [round(b - a, 9) for a, b in zip(xs, xs[1:])]
        assert len(set(gaps)) == 1                            # uniform, no zone


def test_column_uniform_ties_when_no_hinge():
    with apeGmsh(model_name="gen_col_uni") as g:
        cage = g.rebar.column(
            section=("rect", 0.4, 0.4), height=2.0, cover=0.04,
            longitudinal=BarLayout(n_x=2, n_y=2, db=0.02),
            ties=TieLayout(db=0.01, spacing=0.25))
        assert len(cage.bars) == 4                         # 4 corner bars
        zs = sorted(s.path.points[0][2] for s in cage.stirrups)
        gaps = [round(b - a, 9) for a, b in zip(zs, zs[1:])]
        assert len(set(gaps)) == 1                          # all uniform


def test_beam_top_bottom_bars_and_yz_stirrups():
    with apeGmsh(model_name="gen_beam") as g:
        cage = g.rebar.beam(
            section=("rect", 0.3, 0.5), length=4.0, cover=0.04,
            top=BarLayout(n_x=2, db=0.02), bottom=BarLayout(n_x=3, db=0.02),
            stirrups=TieLayout(db=0.01, spacing=0.2))
        assert len(cage.bars) == 5                         # 2 top + 3 bottom
        tops = [b for b in cage.bars if b.role == "top"]
        bots = [b for b in cage.bars if b.role == "bottom"]
        assert len(tops) == 2 and len(bots) == 3
        # top bars sit higher (z) than bottom bars; both run along x
        assert tops[0].path.points[0][2] > bots[0].path.points[0][2]
        for b in cage.bars:
            p0, p1 = b.path.points
            assert p0[0] == pytest.approx(0.04) and p1[0] == pytest.approx(3.96)
        # stirrups are rings in the y-z plane at x-stations (constant x)
        s0 = cage.stirrups[0]
        xs = {round(p[0], 9) for p in s0.path.points}
        assert len(xs) == 1                                # constant x → y-z ring


def test_fluent_bar_builder_equivalent_to_l1():
    with apeGmsh(model_name="gen_fluent") as g:
        built = (g.rebar.bar(db=0.025, material="rebar")
                 .through([(0, 0, 0), (0, 0, 3.0)])
                 .hook_end(Hook.standard_90())
                 .as_("L1"))
        assert isinstance(built, Bar)
        assert built.name == "L1"
        assert built.end_hook is not None and built.end_hook.angle == 90.0
        assert built.db == 0.025
        # an abandoned builder is inert (no Bar, nothing emitted)
        b = g.rebar.bar(db=0.02, material="rebar")
        assert isinstance(b, BarBuilder)


def test_builder_requires_points():
    with apeGmsh(model_name="gen_fluent2") as g:
        with pytest.raises(ValueError):
            g.rebar.bar(db=0.02, material="rebar").build()


def test_fluent_path_rejects_misplaced_kwargs():
    with apeGmsh(model_name="gen_fluent3") as g:
        # on the builder path, hooks/name must use the chain, not bar() kwargs
        with pytest.raises(ValueError):
            g.rebar.bar(db=0.02, material="rebar", end_hook=Hook.standard_90())


def test_column_conformal_meshes_end_to_end():
    with apeGmsh(model_name="gen_col_conf") as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3.0, label="Col")
        cage = g.rebar.column(
            section=("rect", 0.5, 0.5), height=3.0, cover=0.05,
            longitudinal=BarLayout(n_x=2, n_y=2, db=0.025),
            ties=TieLayout(db=0.01, spacing=1.0))
        g.rebar.place(cage, into="Col", coupling="conformal")
        g.mesh.sizing.set_global_size(0.3)
        g.mesh.generation.generate(dim=3)         # interior cage → no boundary PLC
        fem = g.mesh.queries.get_fem_data()
        assert fem.info.n_nodes > 0


def test_tie_levels_stay_inside_member_on_hinge_overlap():
    with apeGmsh(model_name="gen_hinge_ovl") as g:
        cage = g.rebar.column(
            section=("rect", 0.4, 0.4), height=0.6, cover=0.04,
            longitudinal=BarLayout(n_x=2, n_y=2, db=0.02),
            ties=TieLayout(db=0.01, spacing=0.2, hinge_spacing=0.1,
                           hinge_length=0.5))        # 2·0.5 > span → confined
        z0, z1 = 0.04, 0.6 - 0.04
        for s in cage.stirrups:
            z = s.path.points[0][2]
            assert z0 - 1e-9 <= z <= z1 + 1e-9         # never outside member


def test_column_validation_guards():
    with apeGmsh(model_name="gen_guard") as g:
        with pytest.raises(ValueError):               # n<2 → no rectangular perimeter
            g.rebar.column(section=("rect", 0.4, 0.4), height=2.0, cover=0.04,
                           longitudinal=BarLayout(n_x=1, n_y=3, db=0.02),
                           ties=TieLayout(db=0.01, spacing=0.2))
        with pytest.raises(ValueError):               # height ≤ 0
            g.rebar.column(section=("rect", 0.4, 0.4), height=0.0, cover=0.04,
                           longitudinal=BarLayout(n_x=2, n_y=2, db=0.02),
                           ties=TieLayout(db=0.01, spacing=0.2))
        with pytest.raises(ValueError):               # cover+tie+db/2 too large
            g.rebar.column(section=("rect", 0.1, 0.1), height=2.0, cover=0.03,
                           longitudinal=BarLayout(n_x=2, n_y=2, db=0.05),
                           ties=TieLayout(db=0.01, spacing=0.2))


def test_layout_specs_validate():
    with pytest.raises(ValueError):
        BarLayout(n_x=0)
    with pytest.raises(ValueError):
        TieLayout(db=0.01, spacing=0)
    with pytest.raises(ValueError):
        TieLayout(db=0.01, spacing=0.2, hinge_spacing=0.1)   # hinge_length missing


def test_column_cage_places_embedded_end_to_end():
    with apeGmsh(model_name="gen_col_place") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3.0)
        g.physical.add_volume([vol], name="Col")
        cage = g.rebar.column(
            section=("rect", 0.5, 0.5), height=3.0, cover=0.05,
            longitudinal=BarLayout(n_x=2, n_y=2, db=0.025),
            ties=TieLayout(db=0.01, spacing=0.5))
        g.rebar.place(cage, into="Col", coupling="embedded", perfect=1.0e8)
        # one embedded tie per cage member (4 bars + tie rings)
        n_members = len(cage.bars) + len(cage.stirrups)
        assert len(g.reinforce.reinforce_defs) == n_members
