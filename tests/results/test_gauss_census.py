"""``results.elements.gauss.tension_census`` / ``corner_census`` (ADR 0108).

Two derived fields the fork's ADR-95 DruckerPrager campaign used to find
where a frictional collapse deck gets into trouble:

* **tension** — Gauss points with ``mean_stress >= 0``, built from the
  ordinary ``stress`` response so it works on any material;
* **corner** — Gauss points with ``dp_branch == 3``, read from the
  by-position ``material.ladrunoBranch`` columns, and therefore specific
  to the UW ``DruckerPrager`` return map.

Driven off a hand-written native results file so the assertions are
about the census arithmetic and the selectors, not about an engine.  The
engine half is the ``ladruno_fork`` live gate,
``tests/opensees/integration_ladruno/test_ladruno_dp_branch_live.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import GaussCensus, Results
from apeGmsh.results.writers import NativeWriter

# Two elements x two Gauss points = four points, laid out so that every
# census below has a known, non-trivial answer:
#
#   slot   element  mean stress   dp_branch
#   0      e0       -3.0          1  cone
#   1      e0       +1.0          2  cutoff
#   2      e1        0.0          3  corner
#   3      e1       -5.0          0  elastic
#
# tension (mean_stress >= 0) -> slots 1 and 2 (the 0.0 is INCLUDED)
# corner  (dp_branch == 3)   -> slot 2 only
_MEAN = np.array([-3.0, 1.0, 0.0, -5.0])
_BRANCH = np.array([1.0, 2.0, 3.0, 0.0])
_ELEMENTS = np.array([101, 102], dtype=np.int64)
_NATURAL = np.array([[-0.5, 0.0, 0.0], [+0.5, 0.0, 0.0]])


def _hydrostatic_components() -> "dict[str, np.ndarray]":
    """Six stress columns whose ``I1/3`` is exactly ``_MEAN``.

    Hydrostatic on purpose: ``mean_stress`` is the only derived scalar
    under test, and a pure pressure state makes the expected value
    readable straight off ``_MEAN``.
    """
    diag = _MEAN.reshape(1, 2, 2)          # (T=1, E=2, GP=2)
    zero = np.zeros_like(diag)
    return {
        "stress_xx": diag, "stress_yy": diag, "stress_zz": diag,
        "stress_xy": zero, "stress_yz": zero, "stress_xz": zero,
    }


@pytest.fixture
def census_results(g, tmp_path: Path):
    """A native results file carrying the four Gauss points above."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    path = tmp_path / "census.h5"
    components = _hydrostatic_components()
    components["dp_branch"] = _BRANCH.reshape(1, 2, 2)
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="push", kind="static", time=np.array([2.5]))
        w.write_gauss_group(
            sid, "partition_0", group_id="solids",
            class_tag=33002, int_rule=3,
            element_index=_ELEMENTS,
            natural_coords=_NATURAL,
            components=components,
        )
        w.end_stage()
    return Results.from_fem(fem, path, kind="native", cache_root=tmp_path)


class TestTensionCensus:
    def test_counts_and_locates_the_non_compressive_points(
        self, census_results,
    ) -> None:
        c = census_results.elements.gauss.tension_census()

        assert isinstance(c, GaussCensus)
        assert c.component == "mean_stress"
        assert (c.count, c.examined) == (2, 4)
        np.testing.assert_allclose(np.sort(c.values), [0.0, 1.0])
        # Slot 1 is element 101, slot 2 is element 102.
        np.testing.assert_array_equal(np.sort(c.element_index), [101, 102])
        assert c.natural_coords.shape == (2, 3)
        assert c.time == pytest.approx(2.5)

    def test_zero_mean_stress_counts_as_tension(self, census_results) -> None:
        """``>= 0``, not ``> 0`` — a point returned exactly to the apex
        sits at I1 = T and is the whole point of the census."""
        c = census_results.elements.gauss.tension_census()
        assert 0.0 in set(c.values.tolist())

    def test_selectors_narrow_the_population(self, census_results) -> None:
        c = census_results.elements.gauss.tension_census(ids=[101])
        assert (c.count, c.examined) == (1, 2)
        np.testing.assert_array_equal(c.element_index, [101])

    def test_needs_no_fork_response(self, census_results) -> None:
        """Built from ``stress`` alone — nothing DruckerPrager-specific."""
        c = census_results.elements.gauss.tension_census(ids=[102])
        assert c.count == 1


class TestCornerCensus:
    def test_counts_and_locates_branch_three(self, census_results) -> None:
        c = census_results.elements.gauss.corner_census()

        assert c.component == "dp_branch"
        assert (c.count, c.examined) == (1, 4)
        np.testing.assert_allclose(c.values, [3.0])
        np.testing.assert_array_equal(c.element_index, [102])
        assert c.time == pytest.approx(2.5)

    def test_does_not_count_the_cutoff_branch(self, census_results) -> None:
        """Branch 2 (tension cutoff) is NOT a corner — the two are
        different active sets and the campaign counted them apart."""
        c = census_results.elements.gauss.corner_census()
        assert 2.0 not in set(c.values.tolist())

    def test_empty_when_no_point_is_at_the_corner(
        self, census_results,
    ) -> None:
        c = census_results.elements.gauss.corner_census(ids=[101])
        assert c.count == 0
        assert c.element_index.size == 0
        assert c.natural_coords.shape[0] == 0


def test_corner_census_without_the_response_says_so(
    g, tmp_path: Path,
) -> None:
    """A pre-``61b3efa04`` engine records no ``ladrunoBranch`` bucket, so
    the file has no ``dp_branch``. Name it rather than return an empty
    census, which would read as "no corner points"."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    path = tmp_path / "no_branch.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(name="push", kind="static", time=np.array([1.0]))
        w.write_gauss_group(
            sid, "partition_0", group_id="solids",
            class_tag=33002, int_rule=3,
            element_index=_ELEMENTS,
            natural_coords=_NATURAL,
            components=_hydrostatic_components(),
        )
        w.end_stage()
    r = Results.from_fem(fem, path, kind="native", cache_root=tmp_path)

    assert "dp_branch" not in r.elements.gauss.available_components()
    with pytest.raises((ValueError, KeyError), match="dp_branch"):
        r.elements.gauss.corner_census()


def test_census_global_coords_matches_the_slab(census_results) -> None:
    """``GaussCensus.global_coords`` is the same reconstruction
    ``GaussSlab.global_coords`` does, restricted to the matching rows."""
    gauss = census_results.elements.gauss
    slab = gauss.get(component="dp_branch")
    c = gauss.corner_census()

    fem = census_results.fem
    all_xyz = slab.global_coords(fem)
    keep = np.flatnonzero(np.asarray(slab.values)[-1] == 3.0)
    np.testing.assert_allclose(c.global_coords(fem), all_xyz[keep])
