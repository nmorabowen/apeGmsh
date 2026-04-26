"""Phase 3 — partial FEMData synthesis from MPCO MODEL/."""
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


def test_nodes_have_3d_coords_even_if_2d_source(mpco_path: Path) -> None:
    """All apeGmsh FEMData carry 3-column coords, even for 2D MPCO files."""
    with Results.from_mpco(mpco_path) as r:
        coords = np.asarray(r.fem.nodes.coords)
        assert coords.shape[1] == 3


def test_element_groups_per_class(mpco_path: Path) -> None:
    with Results.from_mpco(mpco_path) as r:
        type_names = [g.type_name for g in r.fem.elements]
        # The fixture has just one element class (ElasticBeam3d)
        assert len(type_names) == 1
        # Type name carries the OpenSees class name (lowercased alias)
        assert "elasticbeam3d" in type_names[0].lower()


def test_synthetic_codes_are_negative(mpco_path: Path) -> None:
    """MPCO-derived element codes use ``-class_tag`` to avoid Gmsh collision."""
    with Results.from_mpco(mpco_path) as r:
        for group in r.fem.elements:
            assert group.element_type.code < 0


def test_pg_queries_without_regions_return_empty_or_raise(mpco_path: Path) -> None:
    """The fixture has no MPCO Regions → no PGs; pg= queries fail clearly."""
    with Results.from_mpco(mpco_path) as r:
        s0 = r.stage(r.stages[0].id)
        # No PGs registered → unknown-name lookup raises.
        with pytest.raises((KeyError, ValueError)):
            s0.nodes.get(pg="NonExistent", component="displacement_x")


def test_id_query_works_without_pg(mpco_path: Path) -> None:
    """ID-based queries work without PG/label support."""
    with Results.from_mpco(mpco_path) as r:
        s0 = r.stage(r.stages[0].id)
        all_ids = np.asarray(r.fem.nodes.ids, dtype=np.int64)
        slab = s0.nodes.get(component="displacement_x", ids=all_ids[:5])
        assert slab.node_ids.size == 5


def test_snapshot_id_does_not_match_native(g, mpco_path: Path, tmp_path: Path) -> None:
    """An MPCO-derived fem has a different snapshot_id than a native fem.

    This is by design — element type codes differ (negated class_tag vs
    Gmsh codes). bind() between the two would correctly refuse.
    """
    with Results.from_mpco(mpco_path) as r_mpco:
        mpco_id = r_mpco.fem.snapshot_id

    # Build a native fem (different mesh — but illustrates the point).
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    native_fem = g.mesh.queries.get_fem_data(dim=3)

    # Different by content, of course; the broader point is that even if
    # the geometry were identical, MPCO synthesis uses negated class
    # tags so the hashes would still differ. This documents the design
    # contract for users.
    assert native_fem.snapshot_id != mpco_id
