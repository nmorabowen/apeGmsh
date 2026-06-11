"""``results.plot`` — gauss contours (averaged / discrete) + fiber cloud.

Static-plot parity with the interactive viewer:

* ``plot.contour(topology="gauss", averaging=...)`` mirrors the
  ``ContourStyle`` vocabulary. With one GP per element the element value
  is recovered exactly at every corner, so the *discrete* path must
  paint each boundary facet with its own element's value verbatim,
  while the *averaged* path smooths across shared nodes.
* ``plot.fibers`` places the dot cloud at the true station ξ + (y, z)
  section offsets in the beam frame (recorder-first).

GPU-free (matplotlib Agg); assertions are on artist data, not pixels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
FIBERBEAM = FIXTURES / "fiberbeam.ladruno"


# =====================================================================
# Fixture — meshed cube, one GP per element, stress_xx = eid * 10 + t
# =====================================================================

@pytest.fixture
def gauss_results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    elem_ids = np.concatenate([
        np.asarray(group.ids, dtype=np.int64) for group in fem.elements
    ])
    n_elem = elem_ids.size
    n_steps = 2
    sxx = np.zeros((n_steps, n_elem, 1), dtype=np.float64)
    for t in range(n_steps):
        sxx[t, :, 0] = elem_ids * 10.0 + t
    nat = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    path = tmp_path / "gauss_contour.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=elem_ids, natural_coords=nat,
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _painted_face_values(ax) -> np.ndarray:
    arrays = [
        np.asarray(coll.get_array(), dtype=np.float64)
        for coll in ax.collections
        if coll.get_array() is not None and np.asarray(coll.get_array()).size
    ]
    assert arrays, "no scalar-mapped collection was drawn"
    return np.concatenate(arrays)


# =====================================================================
# contour topology="gauss"
# =====================================================================

def test_gauss_discrete_paints_own_element_values(gauss_results) -> None:
    r = gauss_results
    ax = r.plot.contour(
        "stress_xx", topology="gauss", averaging="discrete", step=0,
    )
    # Per-facet values must be EXACTLY the owning element's value
    # (1 GP → corners carry the GP value; the facet mean is that value).
    vals = _painted_face_values(ax)
    tri_owner, _ = r.plot._facet_owners()
    np.testing.assert_allclose(
        vals[: tri_owner.size], tri_owner * 10.0, atol=1e-9,
    )
    assert "discrete" in ax.get_title()


def test_gauss_averaged_smooths_across_elements(gauss_results) -> None:
    r = gauss_results
    ax_avg = r.plot.contour(
        "stress_xx", topology="gauss", averaging="averaged", step=0,
    )
    ax_disc = r.plot.contour(
        "stress_xx", topology="gauss", averaging="discrete", step=0,
    )
    avg = _painted_face_values(ax_avg)
    disc = _painted_face_values(ax_disc)
    assert avg.shape == disc.shape
    # Smoothing must actually change something (adjacent elements carry
    # different values in this fixture) ...
    assert not np.allclose(avg, disc)
    # ... but averaged values are convex combinations of element values,
    # so they stay inside the discrete range.
    finite = avg[np.isfinite(avg)]
    assert finite.min() >= disc[np.isfinite(disc)].min() - 1e-9
    assert finite.max() <= disc[np.isfinite(disc)].max() + 1e-9


def test_gauss_step_indexing(gauss_results) -> None:
    r = gauss_results
    ax0 = r.plot.contour(
        "stress_xx", topology="gauss", averaging="discrete", step=0,
    )
    ax1 = r.plot.contour(
        "stress_xx", topology="gauss", averaging="discrete", step=1,
    )
    np.testing.assert_allclose(
        _painted_face_values(ax1) - _painted_face_values(ax0), 1.0,
        atol=1e-9,
    )


def test_contour_validates_topology_and_averaging(gauss_results) -> None:
    r = gauss_results
    with pytest.raises(ValueError, match="topology must be"):
        r.plot.contour("stress_xx", topology="cells")
    with pytest.raises(ValueError, match="averaging must be"):
        r.plot.contour("stress_xx", topology="gauss", averaging="mean")


# =====================================================================
# plot.fibers
# =====================================================================

def test_fibers_cloud_at_true_section_positions() -> None:
    # fiberbeam: 1 beam along +X (identity recorder frame), 3 Lobatto
    # stations (ξ = -1, 0, +1 → x = 0, 0.5, 1), 4 fibers per station at
    # section y = ±0.025 (a 2-D fiber section: z is identically 0) —
    # 12 dots total.
    r = Results.from_ladruno(FIBERBEAM)
    ax = r.plot.fibers("fiber_stress", step=0)
    assert type(ax).__name__ == "Axes3D"

    cloud = [
        coll for coll in ax.collections
        if coll.get_array() is not None
        and np.asarray(coll.get_array()).size == 12
    ]
    assert len(cloud) == 1
    xs, ys, zs = (np.asarray(a, dtype=np.float64) for a in cloud[0]._offsets3d)
    np.testing.assert_allclose(np.unique(np.round(xs, 9)), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(np.unique(np.round(ys, 9)), [-0.025, 0.025])
    np.testing.assert_allclose(zs, 0.0, atol=1e-12)

    # Colors are the slab values at step 0, order-aligned.
    slab = r.elements.fibers.get(component="fiber_stress", time=0)
    np.testing.assert_allclose(
        np.asarray(cloud[0].get_array(), dtype=np.float64),
        np.asarray(slab.values[0], dtype=np.float64),
    )


def test_fibers_selector_and_gp_filter() -> None:
    r = Results.from_ladruno(FIBERBEAM)
    ax = r.plot.fibers("fiber_stress", gp_indices=[1], with_mesh=False)
    cloud = [
        coll for coll in ax.collections
        if coll.get_array() is not None
        and np.asarray(coll.get_array()).size
    ]
    assert len(cloud) == 1
    # Only the middle station's 4 fibers remain, at x = 0.5.
    xs = np.asarray(cloud[0]._offsets3d[0], dtype=np.float64)
    assert xs.size == 4
    np.testing.assert_allclose(xs, 0.5)
