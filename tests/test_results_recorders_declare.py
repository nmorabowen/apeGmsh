"""Phase 4 — declarative API of the standalone Recorders composite.

Pure unit tests against ``Recorders()`` directly — no session, no
gmsh, no FEMData. The class is designed to be testable in isolation.
"""
from __future__ import annotations

import pytest

from apeGmsh.solvers.Recorders import Recorders


# =====================================================================
# Standalone construction
# =====================================================================

def test_standalone_construction() -> None:
    r = Recorders()
    assert len(r) == 0


def test_clear() -> None:
    r = Recorders()
    r.nodes(pg="Top", components=["displacement"])
    r.gauss(pg="Body", components=["stress"])
    assert len(r) == 2
    r.clear()
    assert len(r) == 0


def test_chaining() -> None:
    r = Recorders()
    out = (r
           .nodes(pg="Top", components=["displacement"])
           .gauss(pg="Body", components=["stress"])
           .modal(n_modes=5))
    assert out is r
    assert len(r) == 3


# =====================================================================
# Per-category declarations
# =====================================================================

def test_nodes_declaration() -> None:
    r = Recorders()
    r.nodes(pg="Top", components=["displacement", "velocity"], dt=0.01)
    rec = list(r)[0]
    assert rec.category == "nodes"
    assert rec.components == ("displacement", "velocity")
    assert rec.pg == ("Top",)
    assert rec.dt == 0.01
    assert rec.n_steps is None


def test_gauss_declaration() -> None:
    r = Recorders()
    r.gauss(pg="Body", components=["stress"], n_steps=5)
    rec = list(r)[0]
    assert rec.category == "gauss"
    assert rec.n_steps == 5


def test_modal_declaration() -> None:
    r = Recorders()
    r.modal(n_modes=10)
    rec = list(r)[0]
    assert rec.category == "modal"
    assert rec.n_modes == 10
    assert rec.components == ()


def test_all_categories_present() -> None:
    r = Recorders()
    r.nodes(pg="P", components=["displacement"])
    r.elements(pg="P", components=["nodal_resisting_force_x"])
    r.line_stations(pg="P", components=["axial_force"])
    r.gauss(pg="P", components=["stress"])
    r.fibers(pg="P", components=["fiber_stress"])
    r.layers(pg="P", components=["stress"])
    r.modal(n_modes=3)
    cats = [rec.category for rec in r]
    assert cats == [
        "nodes", "elements", "line_stations",
        "gauss", "fibers", "layers", "modal",
    ]


# =====================================================================
# Auto-naming
# =====================================================================

def test_auto_names_unique() -> None:
    r = Recorders()
    r.nodes(pg="A", components=["displacement"])
    r.nodes(pg="B", components=["velocity"])
    r.gauss(pg="C", components=["stress"])
    names = [rec.name for rec in r]
    assert names == ["nodes_0", "nodes_1", "gauss_2"]


def test_explicit_name_preserved() -> None:
    r = Recorders()
    r.nodes(pg="Top", components=["displacement"], name="top_disp")
    assert list(r)[0].name == "top_disp"


# =====================================================================
# Component handling
# =====================================================================

def test_single_string_component() -> None:
    r = Recorders()
    r.nodes(pg="Top", components="displacement")
    assert list(r)[0].components == ("displacement",)


def test_list_components() -> None:
    r = Recorders()
    r.nodes(pg="Top", components=["displacement", "velocity"])
    assert list(r)[0].components == ("displacement", "velocity")


def test_empty_components_raises() -> None:
    r = Recorders()
    with pytest.raises(ValueError, match="At least one component"):
        r.nodes(pg="Top", components=[])


# =====================================================================
# Cadence validation
# =====================================================================

def test_dt_and_n_steps_mutually_exclusive() -> None:
    r = Recorders()
    with pytest.raises(ValueError, match="at most one"):
        r.nodes(pg="Top", components=["displacement"], dt=0.01, n_steps=5)


def test_negative_dt_raises() -> None:
    r = Recorders()
    with pytest.raises(ValueError, match="positive"):
        r.nodes(pg="Top", components=["displacement"], dt=-0.1)


def test_zero_n_steps_raises() -> None:
    r = Recorders()
    with pytest.raises(ValueError, match="positive"):
        r.nodes(pg="Top", components=["displacement"], n_steps=0)


def test_modal_n_modes_validation() -> None:
    r = Recorders()
    with pytest.raises(ValueError, match="positive int"):
        r.modal(n_modes=0)
    with pytest.raises(ValueError, match="positive int"):
        r.modal(n_modes=-1)


# =====================================================================
# Selector validation
# =====================================================================

def test_ids_with_named_selector_raises() -> None:
    r = Recorders()
    with pytest.raises(ValueError, match="not multiple"):
        r.nodes(ids=[1, 2, 3], pg="Top", components=["displacement"])
    with pytest.raises(ValueError, match="not multiple"):
        r.nodes(ids=[1], selection="x", components=["displacement"])


def test_named_selectors_can_combine() -> None:
    """Per Phase 2 backfill: pg/label/selection union together."""
    r = Recorders()
    r.nodes(
        pg="Top", selection="aux",
        components=["displacement"],
    )
    rec = list(r)[0]
    assert rec.pg == ("Top",)
    assert rec.selection == ("aux",)


def test_no_selector_means_all() -> None:
    """Omitting all selectors records the whole topology level."""
    r = Recorders()
    r.nodes(components=["displacement"])
    rec = list(r)[0]
    assert rec.pg == () and rec.label == () and rec.selection == ()
    assert rec.ids is None
