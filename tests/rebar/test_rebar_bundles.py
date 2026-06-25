"""Bundled longitudinal bars (ACI 318-19 §25.6) — `BarLayout(bundle=)` on the
column/beam/circular generators + the hand-authoring `g.rebar.bundle()`."""
from __future__ import annotations

import math

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Bar, BarLayout, TieLayout


# ── L1 validation ────────────────────────────────────────────────────

def test_bundle_layout_validation():
    BarLayout(n_x=2, n_y=2, db="#8", bundle=2)              # ok
    BarLayout(n_x=2, n_y=2, db="#8", bundle=3, bundle_pattern="triangle")
    BarLayout(n_x=2, n_y=2, db="#14", bundle=2)             # #14 allows 2
    with pytest.raises(ValueError):
        BarLayout(n_x=2, bundle=5)                          # > 4 bars
    with pytest.raises(ValueError):
        BarLayout(n_x=2, bundle=0)                          # < 1
    with pytest.raises(ValueError):
        BarLayout(n_x=2, bundle=2, bundle_pattern="triangle")   # count mismatch
    with pytest.raises(ValueError):
        BarLayout(n_x=2, bundle=2, bundle_pattern="bogus")      # unknown pattern
    with pytest.raises(ValueError):
        BarLayout(n_x=2, bundle=3, db="#14")               # #14 capped at 2
    with pytest.raises(ValueError):
        BarLayout(n_x=2, bundle=4, db="#18")               # #18 capped at 2


# ── column ───────────────────────────────────────────────────────────

def test_column_bundle_doubles_every_position():
    with apeGmsh(model_name="bun_col") as g:
        single = g.rebar.column(
            section=("rect", 0.5, 0.5), height=3.0, cover=0.04,
            longitudinal=BarLayout(n_x=2, n_y=2, db=0.025),
            ties=TieLayout(db=0.01, spacing=1.0))
        bundled = g.rebar.column(
            section=("rect", 0.5, 0.5), height=3.0, cover=0.04,
            longitudinal=BarLayout(n_x=2, n_y=2, db=0.025, bundle=2),
            ties=TieLayout(db=0.01, spacing=1.0))
    n1 = len([b for b in single.bars if b.role == "longitudinal"])
    n2 = len([b for b in bundled.bars if b.role == "longitudinal"])
    assert n1 == 4 and n2 == 8                              # 4 corners → bundles of 2


def test_column_face_bundle_preserves_cover_and_spreads_tangentially():
    db = 0.025
    with apeGmsh(model_name="bun_col_face") as g:
        cage = g.rebar.column(
            section=("rect", 0.6, 0.6), height=3.0, cover=0.04,
            longitudinal=BarLayout(n_x=3, n_y=3, db=db, bundle=2),
            ties=TieLayout(db=0.01, spacing=1.0))
    inset = 0.04 + 0.01 + db / 2.0
    # the mid bar on the bottom face sits at (x=0.3, y=inset): its 2-bar
    # bundle stays on that cover line (constant y) and spreads ±db/2 in x
    face = [b for b in cage.bars
            if b.role == "longitudinal"
            and b.path.points[0][1] == pytest.approx(inset, abs=1e-9)
            and abs(b.path.points[0][0] - 0.3) < 0.1]
    assert len(face) == 2
    ys = {round(b.path.points[0][1], 9) for b in face}
    assert ys == {round(inset, 9)}                          # cover preserved
    xs = sorted(b.path.points[0][0] for b in face)
    assert xs[1] - xs[0] == pytest.approx(db, abs=1e-9)     # centre-to-centre = db
    assert sum(xs) / 2.0 == pytest.approx(0.3, abs=1e-9)    # centred on nominal


def test_column_triangle_bundle_stacks_one_bar_inward():
    db = 0.025
    with apeGmsh(model_name="bun_col_tri") as g:
        cage = g.rebar.column(
            section=("rect", 0.6, 0.6), height=3.0, cover=0.04,
            longitudinal=BarLayout(n_x=3, n_y=3, db=db, bundle=3),
            ties=TieLayout(db=0.01, spacing=1.0))
    inset = 0.04 + 0.01 + db / 2.0
    bottom_mid = [b for b in cage.bars
                  if b.role == "longitudinal"
                  and abs(b.path.points[0][0] - 0.3) < 0.1
                  and b.path.points[0][1] < 0.3]              # bottom-half cluster
    assert len(bottom_mid) == 3
    ys = sorted(b.path.points[0][1] for b in bottom_mid)
    # two outer bars on the cover line, one apex stacked a diameter inward
    assert ys[0] == pytest.approx(inset, abs=1e-9)
    assert ys[1] == pytest.approx(inset, abs=1e-9)
    assert ys[2] == pytest.approx(inset + db, abs=1e-9)


def test_column_bundle_cover_leak_is_bounded():
    db = 0.025
    with apeGmsh(model_name="bun_col_cover") as g:
        cage = g.rebar.column(
            section=("rect", 0.6, 0.6), height=3.0, cover=0.04,
            longitudinal=BarLayout(n_x=3, n_y=3, db=db, bundle=4),
            ties=TieLayout(db=0.01, spacing=1.0))
    inset = 0.04 + 0.01 + db / 2.0
    # face bars keep full cover along the inward normal; a corner bundle
    # spreads tangentially so one bar leans toward a face — but never by more
    # than the half-spread projected onto the face (≤ √2/2 · db), and never
    # outside the section. (For strict corner cover, inset for √n·db.)
    leak = math.sqrt(2.0) / 2.0 * (db / 2.0) * 2.0          # = √2/2 · db
    for b in (x for x in cage.bars if x.role == "longitudinal"):
        x, y, _ = b.path.points[0]
        clear = min(x, 0.6 - x, y, 0.6 - y)
        assert clear > 0.0
        assert clear >= inset - leak - 1e-9


def test_column_bundle_count_in_validation():
    with apeGmsh(model_name="bun_col_bad") as g:
        with pytest.raises(ValueError, match="cross the centre"):
            g.rebar.column(
                section=("rect", 0.09, 0.09), height=2.0, cover=0.01,
                longitudinal=BarLayout(n_x=2, n_y=2, db=0.02, bundle=3),
                ties=TieLayout(db=0.006, spacing=0.2, db_value=0.006))


def test_column_bundle_places_embedded_as_independent_members():
    with apeGmsh(model_name="bun_col_place") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3.0)
        g.physical.add_volume([vol], name="Col")
        cage = g.rebar.column(
            section=("rect", 0.5, 0.5), height=3.0, cover=0.05,
            longitudinal=BarLayout(n_x=2, n_y=2, db=0.025, bundle=2),
            ties=TieLayout(db=0.01, spacing=0.6))
        g.rebar.place(cage, into="Col", coupling="embedded", perfect=1.0e8)
        # one embedded tie per member — every bundled bar is its own member
        assert len(g.reinforce.reinforce_defs) == len(cage.bars) + len(cage.stirrups)


# ── circular column ──────────────────────────────────────────────────

def test_circular_bundle_expands_each_position():
    with apeGmsh(model_name="bun_circ") as g:
        cage = g.rebar.circular_column(
            diameter=0.6, height=3.0, cover=0.05, n_bars=6, bar_db=0.025,
            ties=TieLayout(db=0.01, spacing=0.5), bundle=2)
    longit = [b for b in cage.bars if b.role != "spiral"]
    assert len(longit) == 12                                # 6 positions × 2
    r_long = 0.3 - 0.05 - 0.01 - 0.025 / 2.0
    # both bars of a 2-bundle stay ~on the bar circle (tangential spread)
    for b in longit:
        x, y, _ = b.path.points[0]
        assert math.hypot(x, y) == pytest.approx(r_long, abs=0.025)


def test_circular_triangle_bundle_has_inner_bar():
    db = 0.025
    with apeGmsh(model_name="bun_circ_tri") as g:
        cage = g.rebar.circular_column(
            diameter=0.8, height=3.0, cover=0.05, n_bars=6, bar_db=db,
            ties=TieLayout(db=0.01, spacing=0.5), bundle=3)
    r_long = 0.4 - 0.05 - 0.01 - db / 2.0
    radii = sorted(math.hypot(*b.path.points[0][:2]) for b in cage.bars
                   if b.role != "spiral")
    assert min(radii) == pytest.approx(r_long - db, abs=1e-6)   # apex stacked in
    assert max(radii) == pytest.approx(
        math.hypot(r_long, db / 2.0), abs=1e-6)                 # outer pair


def test_circular_bundle_too_deep_raises():
    with apeGmsh(model_name="bun_circ_bad") as g:
        with pytest.raises(ValueError, match="cross the centre"):
            g.rebar.circular_column(
                diameter=0.1, height=2.0, cover=0.01, n_bars=4, bar_db=0.025,
                ties=TieLayout(db=0.005, spacing=0.3, db_value=0.005),
                bundle=3)


# ── beam ─────────────────────────────────────────────────────────────

def test_beam_bundle_expands_top_and_bottom():
    with apeGmsh(model_name="bun_beam") as g:
        cage = g.rebar.beam(
            section=("rect", 0.4, 0.6), length=4.0, cover=0.04,
            top=BarLayout(n_x=2, db=0.02, bundle=2),
            bottom=BarLayout(n_x=2, db=0.02, bundle=2),
            stirrups=TieLayout(db=0.01, spacing=0.5))
    assert len([b for b in cage.bars if b.role == "top"]) == 4     # 2 × 2
    assert len([b for b in cage.bars if b.role == "bottom"]) == 4


def test_beam_mid_top_bundle_keeps_cover_depth():
    db = 0.02
    with apeGmsh(model_name="bun_beam_mid") as g:
        cage = g.rebar.beam(
            section=("rect", 0.4, 0.6), length=4.0, cover=0.04,
            top=BarLayout(n_x=3, db=db, bundle=2),
            bottom=BarLayout(n_x=3, db=db, bundle=2),
            stirrups=TieLayout(db=0.01, spacing=1.0), crossties=False)
    z_top = 0.6 - 0.04 - 0.01 - db / 2.0
    # the mid top bar (centred in width) leans straight down → its 2-bundle
    # stays at the top cover depth and spreads in y
    mid = [b for b in cage.bars if b.role == "top"
           and abs(b.path.points[0][1] - 0.2) < 0.05]
    assert len(mid) == 2
    assert all(b.path.points[0][2] == pytest.approx(z_top, abs=1e-9) for b in mid)


def test_beam_bundle_too_deep_raises():
    with apeGmsh(model_name="bun_beam_bad") as g:
        with pytest.raises(ValueError, match="cross mid-depth"):
            g.rebar.beam(
                section=("rect", 0.4, 0.09), length=4.0, cover=0.01,
                top=BarLayout(n_x=2, db=0.02, bundle=3),
                bottom=BarLayout(n_x=2, db=0.02),
                stirrups=TieLayout(db=0.006, spacing=0.5, db_value=0.006))


# ── hand authoring ───────────────────────────────────────────────────

def test_hand_bundle_returns_independent_named_bars():
    with apeGmsh(model_name="bun_hand") as g:
        bars = g.rebar.bundle(
            [(0, 0, 0), (0, 0, 3.0)], n=3, db=0.025, material="rebar",
            toward=(1.0, 1.0, 0.0), name="L")
    assert len(bars) == 3 and all(isinstance(b, Bar) for b in bars)
    assert [b.name for b in bars] == ["L_b0", "L_b1", "L_b2"]
    starts = {b.path.points[0] for b in bars}
    assert len(starts) == 3                                  # three distinct lines


def test_hand_bundle_spacing_override():
    with apeGmsh(model_name="bun_hand_sp") as g:
        bars = g.rebar.bundle(
            [(0, 0, 0), (0, 0, 3.0)], n=2, db=0.02, material="rebar",
            toward=(0, 1.0, 0), spacing=0.05)
    # 2-bar line spreads ±spacing/2 tangentially about the anchor
    xs = sorted(b.path.points[0][0] for b in bars)
    assert xs[1] - xs[0] == pytest.approx(0.05, abs=1e-9)


def test_hand_bundle_in_cage_places():
    from apeGmsh._kernel.defs.rebar import Cage
    with apeGmsh(model_name="bun_hand_place") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 0.4, 0.4, 3.0)
        g.physical.add_volume([vol], name="Col")
        bars = g.rebar.bundle(
            [(0.1, 0.1, 0.1), (0.1, 0.1, 2.9)], n=2, db=0.025,
            material="rebar", toward=(0.2, 0.2, 0.1))
        cage = Cage(bars=bars)
        g.rebar.place(cage, into="Col", coupling="embedded", perfect=1.0e8)
        assert len(g.reinforce.reinforce_defs) == 2


def test_hand_bundle_validation():
    with apeGmsh(model_name="bun_hand_bad") as g:
        with pytest.raises(ValueError):
            g.rebar.bundle([(0, 0, 0), (0, 0, 1)], n=5, db=0.02,
                           material="rebar", toward=(1, 0, 0))
        with pytest.raises(ValueError):
            g.rebar.bundle([(0, 0, 0)], n=2, db=0.02, material="rebar",
                           toward=(1, 0, 0))                # need ≥ 2 points
