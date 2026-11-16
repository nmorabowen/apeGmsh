"""Crack maps: the ``.ladruno`` crack buckets and ``results.plot.cracks``.

ASDConcrete3D records ``material.crackPattern`` (9 columns per Gauss
point: three crack vectors ``C1x..C3z``, each along a crack normal and as
long as that crack's opening) and ``material.crackWidth`` (``cw``, the
largest opening). The reader used to drop both with a
``GaussColumnDroppedWarning``; these lock their canonical names
(``crack_c{k}_{axis}``, ``crack_width``) and the matplotlib crack map.

Fixture: ``crack_bricks.ladruno`` — three stdBrick elements of
ASDConcrete3D pulled in x until the third cracks (see
``tests/fixtures/ladruno/_generate_fixtures.py``). Fork-free to read.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import h5py  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from apeGmsh.results import Results  # noqa: E402
from apeGmsh.results.plot._cracks import (  # noqa: E402
    CRACK_VECTOR_COMPONENTS,
    crack_segments,
    default_half_length,
    parse_plane,
)
from apeGmsh.results.readers._ladruno_element_io import (  # noqa: E402
    GaussColumnDroppedWarning,
    GaussColumnNameMismatchWarning,
    material_bucket_canonicals,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
CRACKS = FIXTURES / "crack_bricks.ladruno"
QUAD = FIXTURES / "quad2d.ladruno"


def _raw_bucket(token: str) -> np.ndarray:
    """The bucket's DATA as the file stores it: (T, elements, GP*width)."""
    with h5py.File(CRACKS, "r") as f:
        grp = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"][token]
        return np.asarray(grp[next(iter(grp))]["DATA"][...])


def _crack_lines(ax) -> LineCollection:
    found = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(found) == 1
    return found[0]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def test_crack_buckets_are_canonical_gauss_components() -> None:
    r = Results.from_ladruno(CRACKS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GaussColumnDroppedWarning)
        warnings.simplefilter("error", GaussColumnNameMismatchWarning)
        comps = set(r.elements.gauss.available_components())
    assert set(CRACK_VECTOR_COMPONENTS) | {"crack_width"} <= comps


def test_crack_vectors_read_point_major_from_an_element_major_file() -> None:
    # The file stores element-major rows (element, then GP x 9 columns);
    # a GaussSlab is point-major (every element's GP 0, then GP 1, ...).
    raw = _raw_bucket("material.crackPattern")           # (T, 3, 8*9)
    n_t, n_e, width = raw.shape
    n_gp = width // 9
    expected = raw.reshape(n_t, n_e, n_gp, 9).transpose(0, 2, 1, 3)
    expected = expected.reshape(n_t, n_gp * n_e, 9)
    r = Results.from_ladruno(CRACKS)
    for col, comp in enumerate(CRACK_VECTOR_COMPONENTS):
        slab = r.elements.gauss.get(component=comp)
        np.testing.assert_array_equal(slab.values, expected[:, :, col])
    np.testing.assert_array_equal(
        slab.element_index, np.tile([1, 2, 3], n_gp),
    )


def test_crack_width_is_the_largest_crack_vector() -> None:
    r = Results.from_ladruno(CRACKS)
    vecs = np.stack([
        r.elements.gauss.get(component=c).values
        for c in CRACK_VECTOR_COMPONENTS
    ], axis=-1)
    largest = np.linalg.norm(
        vecs.reshape(*vecs.shape[:2], 3, 3), axis=-1,
    ).max(axis=-1)
    cw = r.elements.gauss.get(component="crack_width").values
    assert cw.max() > 0.0                    # the fixture really cracked
    np.testing.assert_allclose(cw, largest, rtol=1e-6, atol=1e-12)


def test_crack_tokens_are_keyed_by_token_not_label() -> None:
    assert material_bucket_canonicals("material.crackWidth") == (
        "crack_width",
    )
    assert material_bucket_canonicals("material.cw") == ("crack_width",)
    assert material_bucket_canonicals("material.CrackPattern") == (
        CRACK_VECTOR_COMPONENTS
    )
    # crushPattern writes the same C1x..C3z labels for the compression
    # side; it must not be read as cracks.
    assert material_bucket_canonicals("material.crushPattern") is None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_parse_plane() -> None:
    assert parse_plane("xz") == (0, 2)
    assert parse_plane("ZY") == (2, 1)
    for bad in ("xx", "x", "xw", "xyz"):
        with pytest.raises(ValueError, match="two distinct axes"):
            parse_plane(bad)


def test_crack_segment_runs_across_the_projected_normal() -> None:
    points = np.array([[1.0, 5.0, 2.0]])
    vectors = np.zeros((1, 3, 3))
    vectors[0, 0] = [0.3, 0.0, 0.4]        # opening 0.5, normal in x-z
    vectors[0, 1] = [0.0, 0.2, 0.0]        # normal along y: no trace on x-z
    vectors[0, 2] = [0.01, 0.0, 0.0]       # below the threshold
    segs, openings = crack_segments(
        points, vectors, plane=(0, 2), min_opening=0.05, half_length=1.0,
    )
    assert segs.shape == (1, 2, 2)
    np.testing.assert_allclose(openings, [0.5])
    np.testing.assert_allclose(segs.mean(axis=1)[0], [1.0, 2.0])
    direction = segs[0, 1] - segs[0, 0]
    np.testing.assert_allclose(np.linalg.norm(direction), 2.0)
    assert abs(direction @ np.array([0.3, 0.4])) < 1e-12


def test_default_half_length_is_under_half_the_grid_pitch() -> None:
    xs, zs = np.meshgrid(np.arange(10.0) * 2.0, np.arange(4.0) * 2.0)
    grid = np.column_stack([xs.ravel(), zs.ravel()])
    stacked = np.vstack([grid, grid])      # two layers along the view axis
    hl = default_half_length(stacked)
    assert 0.5 < hl < 1.0                  # pitch 2: under half of it


# ---------------------------------------------------------------------------
# results.plot.cracks
# ---------------------------------------------------------------------------

def test_cracks_side_view_draws_every_open_crack() -> None:
    r = Results.from_ladruno(CRACKS)
    ax = r.plot.cracks(plane="xz")
    lines = _crack_lines(ax)
    segs = np.asarray(lines.get_segments())
    # Only the third brick (x in [2, 3]) cracked: 8 GPs x 3 crack planes.
    assert segs.shape == (24, 2, 2)
    assert segs[:, :, 0].min() > 2.0
    openings = np.asarray(lines.get_array())
    assert openings.max() == pytest.approx(
        r.elements.gauss.get(component="crack_width").values[-1].max(),
    )
    # The widest cracks open along the pull (normal x): vertical traces.
    main = segs[openings > 1e-3]
    assert main.shape[0] == 8
    np.testing.assert_allclose(main[:, 0, 0], main[:, 1, 0])
    # Outline: one projected polygon per brick.
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) == 1 and len(polys[0].get_paths()) == 3
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "z"


def test_cracks_threshold_and_selection() -> None:
    r = Results.from_ladruno(CRACKS)
    ax = r.plot.cracks(min_opening=1e-3)
    assert len(_crack_lines(ax).get_segments()) == 8
    # An uncracked selection draws the outline and no crack lines.
    ax = r.plot.cracks(ids=[1, 2], outline=True)
    assert not any(isinstance(c, LineCollection) for c in ax.collections)
    assert "0 drawn" in ax.get_title()


def test_cracks_end_view_skips_cracks_seen_face_on() -> None:
    # On the y-z view the main cracks (normal x) lie in the view plane
    # and have no trace; only the two inclined planes are drawn.
    r = Results.from_ladruno(CRACKS)
    ax = r.plot.cracks(plane="yz")
    assert len(_crack_lines(ax).get_segments()) == 16


def test_cracks_refuses_a_result_without_crack_vectors() -> None:
    r = Results.from_ladruno(QUAD)
    with pytest.raises(RuntimeError, match="material.crackPattern"):
        r.plot.cracks()
