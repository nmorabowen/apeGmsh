"""P0 — L1 reinforcement-cage spec objects (ADR 0066 §3). Off-session."""
from __future__ import annotations

import pytest

from apeGmsh._kernel.defs.rebar import (
    CAGE_SCHEMA, CAGE_SCHEMA_VERSION, METADATA, Bar, Cage, Hook, Path, Stirrup,
)


# ── Hook ─────────────────────────────────────────────────────────────

def test_hook_factories_set_aci_angles_and_defer_tail():
    # factories carry only the bend ANGLE; tail/bend_radius are filled by
    # the DetailingStandard at bind time (the ACI multiples live there)
    assert Hook.standard_90().angle == 90.0
    assert Hook.standard_90().tail is None
    assert Hook.standard_135().angle == 135.0
    assert Hook.standard_180().angle == 180.0
    assert Hook.seismic_135().angle == 135.0      # alias of standard_135
    assert Hook.seismic_135().tail is None
    assert Hook.seismic_135().bend_radius is None


def test_hook_allows_none_tail_and_explicit_override():
    assert Hook(angle=90).tail is None
    assert Hook(angle=90, tail="12db").tail == "12db"
    assert Hook(angle=90, tail=5.0).tail == 5.0


@pytest.mark.parametrize("bad", [
    dict(angle=200, tail="6db"),          # angle > 180
    dict(angle=0, tail="6db"),            # angle <= 0
    dict(angle=float("nan"), tail=1.0),   # non-finite angle
    dict(angle=90, tail="6mm"),           # not a "<k>db" token nor number
    dict(angle=90, tail=-5.0),            # non-positive length
    dict(angle=90, tail="0db"),           # zero-multiple token (was accepted)
    dict(angle=90, tail=float("nan")),    # non-finite length
    dict(angle=90, bend_radius=float("inf"), tail=1.0),   # non-finite radius
    dict(angle=90, tail="6db", turn=(0, 0, 0)),     # zero turn vector
    dict(angle=90, tail="6db", turn="sideways"),    # unknown turn token
])
def test_hook_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        Hook(**bad)


def test_hook_turn_vector_is_float_normalised():
    assert Hook(angle=90, turn=(1, 0, 0)).turn == (1.0, 0.0, 0.0)


def test_hook_turn_accepts_token_and_vector():
    assert Hook(angle=90, tail=1.0, turn="centroid").turn == "centroid"
    assert Hook(angle=90, tail=1.0, turn=(1, 0, 0)).turn == (1, 0, 0)


# ── Path ─────────────────────────────────────────────────────────────

def test_path_defaults_to_metadata_bend_and_floats_points():
    p = Path([(0, 0, 0), (0, 0, 3)])
    assert p.is_metadata_bend
    assert p.corner_radius == METADATA
    assert all(isinstance(c, float) for c in p.points[0])


@pytest.mark.parametrize("bad", [
    [(0, 0, 0)],                          # < 2 points
    [(0, 0), (1, 1)],                     # not 3D
    [(0, 0, 0), (0, 0, 0)],               # zero-length (coincident) segment
    [(0, 0, 0), (1, 0, 0), (1, 0, 0)],    # duplicate consecutive interior vertex
    [(0, 0, 0), (float("inf"), 0, 0)],    # non-finite coordinate
])
def test_path_rejects_bad_points(bad):
    with pytest.raises(ValueError):
        Path(bad)


def test_path_allows_closed_loop_first_equals_last():
    # non-consecutive first==last (a closed tie) is fine
    p = Path([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0)])
    assert p.points[0] == p.points[-1]


def test_path_numeric_corner_radius_is_not_metadata():
    p = Path([(0, 0, 0), (1, 0, 0)], corner_radius="2db")
    assert not p.is_metadata_bend


# ── Bar / Stirrup ────────────────────────────────────────────────────

def test_bar_defaults_and_validation():
    b = Bar(path=Path([(0, 0, 0), (0, 0, 1)]), db="#8", material="rebar")
    assert b.element == "truss" and b.role == "longitudinal"
    with pytest.raises(ValueError):
        Bar(path=Path([(0, 0, 0), (0, 0, 1)]), db="#8", material="rebar",
            element="shell")
    with pytest.raises(ValueError):
        Bar(path=Path([(0, 0, 0), (0, 0, 1)]), db=-1, material="rebar")
    with pytest.raises(ValueError):
        Bar(path=Path([(0, 0, 0), (0, 0, 1)]), db="#8", material="")


def test_stirrup_rect_insets_to_centerline_and_defaults_seismic_hook():
    s = Stirrup.rect(0.5, 0.5, 0.04, db=0.012, material="rebar", z=0.1)
    # cover 0.04 + db/2 0.006 = 0.046 inset; closes back to the first corner
    assert s.path.points[0] == (0.046, 0.046, 0.1)
    assert len(s.path.points) == 5
    assert s.path.points[0] == s.path.points[-1]
    assert s.closure_hook.angle == 135.0


def test_stirrup_rect_degenerate_section_raises():
    with pytest.raises(ValueError):
        Stirrup.rect(0.1, 0.1, 0.04, db=0.04, material="rebar")


@pytest.mark.parametrize("kw", [
    dict(bx=float("nan"), by=0.5, cover=0.04, db=0.012),   # nan section
    dict(bx=float("inf"), by=float("inf"), cover=0.04, db=0.012),
    dict(bx=-0.5, by=0.5, cover=0.04, db=0.012),           # negative section
    dict(bx=0.5, by=0.5, cover=0.04, db=float("nan")),     # nan db
])
def test_stirrup_rect_rejects_non_finite_or_nonpositive(kw):
    with pytest.raises(ValueError):
        Stirrup.rect(material="rebar", **kw)


def test_stirrup_rect_designation_db_needs_db_value():
    with pytest.raises(ValueError):
        Stirrup.rect(0.5, 0.5, 0.04, db="#4", material="rebar")
    s = Stirrup.rect(0.5, 0.5, 0.04, db="#4", material="rebar", db_value=0.0127)
    assert s.db == "#4"


# ── Cage + serialization round-trip ──────────────────────────────────

def _sample_cage() -> Cage:
    bar = Bar(path=Path([(0, 0, 0), (0, 0, 3.0)]), db="#8", material="rebar",
              end_hook=Hook.standard_90(), name="L1")
    st = Stirrup.rect(0.5, 0.5, 0.04, db=0.012, material="rebar", z=0.1)
    return Cage(bars=(bar,), stirrups=(st,))


def test_cage_empty_raises():
    with pytest.raises(ValueError):
        Cage()


def test_cage_round_trip_is_stable():
    cage = _sample_cage()
    d = cage.to_dict()
    cage2 = Cage.from_dict(d)
    assert cage2.to_dict() == d
    assert cage2.bars[0].end_hook.angle == 90.0
    assert cage2.bars[0].end_hook.tail is None       # deferred to the standard
    assert cage2.stirrups[0].closure_hook.angle == 135.0
    assert cage2.bars[0].name == "L1"


def test_cage_to_dict_is_schema_tagged_and_omits_standard():
    cage = _sample_cage()
    d = cage.to_dict()
    assert set(d) == {"__schema__", "version", "bars", "stirrups"}
    assert d["__schema__"] == CAGE_SCHEMA
    assert d["version"] == CAGE_SCHEMA_VERSION


def test_cage_from_dict_rejects_unknown_version():
    d = _sample_cage().to_dict()
    d["version"] = CAGE_SCHEMA_VERSION + 99
    with pytest.raises(ValueError):
        Cage.from_dict(d)


def test_cage_from_dict_accepts_untagged_legacy_dict():
    # a dict without __schema__/version still loads (defaults to current)
    cage = _sample_cage()
    legacy = {"bars": [b.to_dict() for b in cage.bars],
              "stirrups": [s.to_dict() for s in cage.stirrups]}
    assert Cage.from_dict(legacy).to_dict() == cage.to_dict()
