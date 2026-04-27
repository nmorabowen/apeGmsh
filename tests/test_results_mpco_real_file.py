"""Phase 3 — read a real STKO MPCO file end-to-end through the composite API.

Fixture: ``tests/fixtures/results/elasticFrame.mpco`` is a copy of
``STKO_to_python/stko_results_examples/elasticFrame/elasticFrame_mesh_results/results.mpco``
— a 12-node 11-element elastic beam frame with 10 transient steps
(2 stages, after a domain change).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results

_FIXTURE = Path(__file__).parent / "fixtures" / "results" / "elasticFrame.mpco"


@pytest.fixture
def mpco_path() -> Path:
    if not _FIXTURE.exists():
        pytest.skip(f"MPCO fixture not present at {_FIXTURE}")
    return _FIXTURE


# =====================================================================
# Stage discovery
# =====================================================================

def test_stages_discovered(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        stages = r.stages
        # The fixture has 2 model stages (one per domain change)
        assert len(stages) >= 1
        assert all(s.kind == "transient" for s in stages)


def test_stage_names_match_mpco(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        names = [s.name for s in r.stages]
        # MPCO group names are MODEL_STAGE[<stamp>]
        assert any(n.startswith("MODEL_STAGE[") for n in names)


def test_time_vector(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        s = r.stages[0]
        t = r._reader.time_vector(s.id)
        assert t.size == s.n_steps
        assert t[0] > 0   # MPCO records starts at the first analysis step


# =====================================================================
# Available components surfacing
# =====================================================================

def test_canonical_components_surface(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        s0 = r.stage(r.stages[0].id)
        comps = set(s0.nodes.available_components())
        # The fixture records all the standard nodal results.
        for expected in [
            "displacement_x", "displacement_y", "displacement_z",
            "rotation_x", "rotation_y", "rotation_z",
            "velocity_x", "acceleration_x",
            "reaction_force_x", "reaction_moment_x",
        ]:
            assert expected in comps, f"missing {expected}"


# =====================================================================
# Slab reads
# =====================================================================

def test_displacement_full_read(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        s0 = r.stage(r.stages[0].id)
        slab = s0.nodes.get(component="displacement_x")
        # Fixture: 12 nodes, 10 steps
        assert slab.values.shape == (10, 12)
        assert slab.node_ids.size == 12
        assert slab.time.size == 10


def test_rotation_read_distinct_from_displacement(mpco_path: Path) -> None:
    """Rotation has its own group in MPCO and shouldn't alias displacement."""
    with Results.from_mpco(mpco_path) as r:
        s0 = r.stage(r.stages[0].id)
        d = s0.nodes.get(component="displacement_x").values
        rot = s0.nodes.get(component="rotation_x").values
        # Different physical quantities → different numerical values.
        assert not np.allclose(d, rot)


def test_filter_by_node_ids(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        s0 = r.stage(r.stages[0].id)
        all_slab = s0.nodes.get(component="displacement_x")
        first_three = all_slab.node_ids[:3]
        sub = s0.nodes.get(component="displacement_x", ids=first_three)
        np.testing.assert_array_equal(sub.node_ids, first_three)
        np.testing.assert_allclose(sub.values, all_slab.values[:, :3])


def test_time_slice(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        s0 = r.stage(r.stages[0].id)
        s = s0.nodes.get(component="displacement_x", time=0)
        assert s.values.shape == (1, 12)


# =====================================================================
# Partial FEMData synthesis
# =====================================================================

def test_partial_fem_synthesized(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        fem = r.fem
        assert fem is not None
        # 12 nodes
        assert len(fem.nodes.ids) == 12
        # 11 elastic beam elements
        n_elems = sum(len(g) for g in fem.elements)
        assert n_elems == 11


def test_partial_fem_has_no_labels(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        fem = r.fem
        # MPCO doesn't carry apeGmsh labels — the LabelSet is empty.
        assert fem.nodes.labels.names() == []
        assert fem.elements.labels.names() == []


def test_partial_fem_snapshot_id_is_stable(mpco_path: Path) -> None:
    """Computing snapshot_id twice on the same MPCO yields the same hash."""
    with Results.from_mpco(mpco_path) as r1:
        h1 = r1.fem.snapshot_id
    with Results.from_mpco(mpco_path) as r2:
        h2 = r2.fem.snapshot_id
    assert h1 == h2


# =====================================================================
# Element-level reads on this fixture
# =====================================================================
#
# elasticFrame.mpco is an elastic beam frame — it does NOT record
# continuum stress or fiber data. The empty-slab returns below are the
# correct behaviour, not a stub. End-to-end gauss decoding of a real
# MPCO file is exercised by ``test_results_mpco_element_real.py``.

def test_gauss_empty_when_fixture_has_no_stress(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        s0 = r.stage(r.stages[0].id)
        slab = s0.elements.gauss.get(component="stress_xx")
        assert slab.values.shape[1] == 0


def test_fibers_empty_when_fixture_has_no_fibers(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        s0 = r.stage(r.stages[0].id)
        slab = s0.elements.fibers.get(component="fiber_stress")
        assert slab.values.shape[1] == 0
