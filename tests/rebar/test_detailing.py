"""P0 — detailing standards + bar catalogue (ADR 0066 §4). Off-session."""
from __future__ import annotations

import math

import pytest

from apeGmsh._kernel.defs.rebar import Hook
from apeGmsh.rebar.detailing import (
    ACI318, ACI318_seismic, BarCatalog, DetailingError, DetailingStandard, Raw,
)


# ── BarCatalog ───────────────────────────────────────────────────────

def test_catalog_imperial_inches_default():
    c = BarCatalog()                       # base imperial, unit_length=1.0 (inches)
    assert c.bar_diameter("#8") == 1.0
    assert c.bar_area("#8") == pytest.approx(0.79)
    assert c.bar_diameter("#3") == 0.375


def test_catalog_unit_length_scales_to_mm():
    c = BarCatalog(unit_length=25.4, base="imperial")    # model in mm
    assert c.bar_diameter("#8") == pytest.approx(25.4)
    assert c.unit_per_mm == pytest.approx(1.0)


def test_catalog_metric_designation_any_base():
    c = BarCatalog()                       # imperial base, inches
    # 20 mm expressed in inches
    assert c.bar_diameter("20mm") == pytest.approx(20.0 / 25.4)
    cm = BarCatalog(unit_length=1.0, base="metric")      # model in mm
    assert cm.bar_diameter("20mm") == pytest.approx(20.0)
    assert cm.bar_area("20mm") == pytest.approx(math.pi * 20.0 ** 2 / 4.0)


def test_catalog_raw_float_passthrough():
    c = BarCatalog()
    assert c.bar_diameter(0.625) == 0.625
    assert c.bar_area(0.625) == pytest.approx(math.pi * 0.625 ** 2 / 4.0)


@pytest.mark.parametrize("bad", ["#99", "20cm", "rebar", 0.0, -1])
def test_catalog_bad_designation_raises(bad):
    with pytest.raises(DetailingError):
        BarCatalog().bar_diameter(bad)


def test_catalog_bad_base_or_unit_raises():
    with pytest.raises(DetailingError):
        BarCatalog(base="furlong")
    with pytest.raises(DetailingError):
        BarCatalog(unit_length=0)


# ── ACI318 bend diameters (Table 25.3.1 / 25.3.2) ────────────────────

def test_aci_min_bend_primary_buckets():
    s = ACI318()                            # inches
    assert s.min_bend_diameter(1.000) == pytest.approx(6.0)      # #8  -> 6db
    assert s.min_bend_diameter(1.128) == pytest.approx(8.0 * 1.128)  # #9 -> 8db
    assert s.min_bend_diameter(1.693) == pytest.approx(10.0 * 1.693)  # #14 -> 10db


def test_aci_min_bend_stirrup_buckets():
    s = ACI318()
    assert s.min_bend_diameter(0.375, kind="stirrup_tie") == pytest.approx(4.0 * 0.375)
    assert s.min_bend_diameter(0.750, kind="stirrup_tie") == pytest.approx(6.0 * 0.750)


def test_aci_bend_bucket_is_unit_safe():
    # same physical #8 bar in mm must give the same multiple (6db), not a
    # different bucket from the raw magnitude
    s_in = ACI318(BarCatalog())                              # inches
    s_mm = ACI318(BarCatalog(unit_length=25.4))              # mm
    assert s_in.min_bend_diameter(1.000) == pytest.approx(6.0)
    assert s_mm.min_bend_diameter(25.4) == pytest.approx(6.0 * 25.4)


def test_aci_default_corner_radius_is_half_inside_bend():
    s = ACI318()
    assert s.default_corner_radius(1.0) == pytest.approx(3.0)    # 6db/2


# ── ACI318 hook tails (§25.3.1 / §25.3.2) ────────────────────────────

def test_aci_hook_tail_primary():
    s = ACI318()
    assert s.hook_tail(90, 1.0) == pytest.approx(12.0)           # 12db
    assert s.hook_tail(180, 1.0) == pytest.approx(4.0)           # max(4db, 2.5in)
    # small bar: the 2.5in floor governs the 180° hook
    assert s.hook_tail(180, 0.5) == pytest.approx(2.5)


def test_aci_hook_tail_stirrup_is_bar_size_dependent():
    # Table 25.3.2: #3-#5 -> 6db, #6-#8 -> 12db (BOTH 90 and 135 rows)
    s = ACI318()
    assert s.hook_tail(90, 0.5, kind="stirrup_tie") == pytest.approx(3.0)    # #4 -> 6db
    assert s.hook_tail(135, 0.5, kind="stirrup_tie") == pytest.approx(3.0)   # #4 -> 6db
    assert s.hook_tail(90, 1.0, kind="stirrup_tie") == pytest.approx(12.0)   # #8 -> 12db
    assert s.hook_tail(135, 1.0, kind="stirrup_tie") == pytest.approx(12.0)  # #8 -> 12db


def test_aci_seismic_hook_is_flat_6db_not_the_size_split():
    # §25.3.4 seismic hook = flat 6db (+3in floor), NOT the Table 25.3.2
    # 12db that a #8 NON-seismic tie gets
    s = ACI318()
    assert s.hook_tail(135, 1.0, kind="seismic_hoop") == pytest.approx(6.0)
    assert s.hook_tail(135, 1.0, kind="stirrup_tie") == pytest.approx(12.0)


def test_aci_primary_135_has_no_standard_hook():
    # Table 25.3.1 defines only 90 and 180 standard development hooks
    with pytest.raises(DetailingError):
        ACI318().hook_tail(135, 1.0, kind="primary")


def test_aci_hook_tail_unknown_angle_raises():
    with pytest.raises(DetailingError):
        ACI318().hook_tail(45, 1.0)
    with pytest.raises(DetailingError):
        ACI318().min_bend_diameter(1.0, kind="bogus")


def test_make_hook_constructs_resolved_standard_hook():
    s = ACI318()
    h = s.make_hook("primary", 1.0, angle=90)
    assert h.angle == 90.0 and h.tail == pytest.approx(12.0)
    assert h.bend_radius == pytest.approx(3.5)
    with pytest.raises(DetailingError):
        Raw().make_hook("primary", 1.0, angle=90)


# ── ACI318_seismic 135° floor ────────────────────────────────────────

def test_seismic_135_tail_floor():
    s = ACI318_seismic()
    # small bar: 6db = 2.25in < 3in floor -> 3in governs
    assert s.hook_tail(135, 0.375, kind="seismic_hoop") == pytest.approx(3.0)
    # large bar: 6db governs
    assert s.hook_tail(135, 1.0, kind="seismic_hoop") == pytest.approx(6.0)


def test_seismic_floor_scales_with_units():
    s = ACI318_seismic(BarCatalog(unit_length=25.4))            # mm
    # 3in floor in mm = 76.2; 6db for a small bar (db=9.525mm) = 57.15 < 76.2
    assert s.hook_tail(135, 9.525, kind="seismic_hoop") == pytest.approx(76.2)


# ── Raw escape hatch ─────────────────────────────────────────────────

def test_raw_delegates_diameter_but_blocks_code_rules():
    r = Raw()
    assert r.bar_diameter("#8") == 1.0          # catalogue still works
    for call in (lambda: r.min_bend_diameter(1.0),
                 lambda: r.hook_tail(90, 1.0),
                 lambda: r.default_corner_radius(1.0)):
        with pytest.raises(DetailingError):
            call()


def test_raw_resolves_fully_explicit_hook_only():
    r = Raw()
    explicit = Hook(angle=90, tail=12.0, bend_radius=3.5)
    out = r.resolve_hook(explicit, db=1.0)
    assert out.tail == 12.0 and out.bend_radius == 3.5
    with pytest.raises(DetailingError):
        r.resolve_hook(Hook(angle=90, tail="12db"), db=1.0)   # bend_radius=None


# ── resolve_hook (bind-time numericisation) ──────────────────────────

def test_aci_resolve_hook_fills_numbers():
    s = ACI318()
    out = s.resolve_hook(Hook.standard_90(), db=1.0)            # #8-ish
    assert out.tail == pytest.approx(12.0)                      # 12db
    # centerline radius = inside (6db/2=3.0) + db/2 (0.5) = 3.5
    assert out.bend_radius == pytest.approx(3.5)
    assert isinstance(out.tail, float) and isinstance(out.bend_radius, float)


def test_aci_resolve_hook_applies_seismic_floor():
    s = ACI318_seismic()
    # explicit "6db" still picks up the 3in seismic floor on a small bar
    out = s.resolve_hook(Hook.seismic_135(), db=0.375, kind="seismic_hoop")
    assert out.tail == pytest.approx(3.0)


# ── Seismic confinement zone (ACI 318 §18.7.5) ───────────────────────

def test_confinement_length_governing_term():
    s = ACI318_seismic()                                       # inches model
    # member depth governs: max(24, 120/6=20, 18) = 24
    assert s.confinement_length(member_depth=24.0, clear_span=120.0) == 24.0
    # clear-span/6 governs: max(12, 120/6=20, 18) = 20
    assert s.confinement_length(member_depth=12.0, clear_span=120.0) == 20.0
    # 18 in floor governs: max(12, 60/6=10, 18) = 18
    assert s.confinement_length(member_depth=12.0, clear_span=60.0) == 18.0


def test_confinement_spacing_so_equation_and_caps():
    s = ACI318_seismic()                                       # inches model
    # tight bar spacing → s_o eqn clamps high → 6db / b_min/4 cap governs
    # min(24/4=6, 6*1.0=6, clamp(4+(14-6)/3=6.67→6)) = 6.0
    assert s.confinement_spacing(min_member_dim=24.0, db_long=1.0,
                                 hx=6.0) == pytest.approx(6.0)
    # wider bar spacing → the 4+(14-hx)/3 equation governs (4.667 in)
    assert s.confinement_spacing(min_member_dim=24.0, db_long=1.0,
                                 hx=12.0) == pytest.approx(4.0 + (14.0 - 12.0) / 3.0)
    # h_x capped at 14 in → s_o floors at 4 in even for very wide spacing
    assert s.confinement_spacing(min_member_dim=24.0, db_long=1.0,
                                 hx=40.0) == pytest.approx(4.0)


def test_confinement_spacing_unit_safe_mm():
    s = ACI318_seismic(BarCatalog(unit_length=25.4, base="imperial"))  # mm model
    # same #8 bar (25.4 mm), b_min=600 mm, hx=305 mm (~12 in) → s_o eqn governs
    got = s.confinement_spacing(min_member_dim=600.0, db_long=25.4, hx=305.0)
    so_in = 4.0 + (14.0 - 305.0 / 25.4) / 3.0
    assert got == pytest.approx(min(600.0 / 4.0, 6.0 * 25.4, so_in * 25.4))


def test_confinement_rejects_nonpositive():
    s = ACI318_seismic()
    with pytest.raises(DetailingError):
        s.confinement_length(member_depth=0.0, clear_span=120.0)
    with pytest.raises(DetailingError):
        s.confinement_spacing(min_member_dim=24.0, db_long=1.0, hx=0.0)


def test_confinement_absent_on_non_seismic():
    assert not hasattr(ACI318(), "confinement_length")
    assert not hasattr(Raw(), "confinement_length")


# ── Protocol conformance ─────────────────────────────────────────────

def test_standards_satisfy_protocol():
    assert isinstance(ACI318(), DetailingStandard)
    assert isinstance(ACI318_seismic(), DetailingStandard)
    assert isinstance(Raw(), DetailingStandard)
