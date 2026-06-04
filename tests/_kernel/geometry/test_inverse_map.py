"""Battery for the guarded isoparametric inverse map (ADR 20 §6 Gate 1).

Proves the map (a) recovers ξ for random interior points of straight-sided
hex/tet/quad/tri hosts to ~1e-12, (b) gives a partition of unity, and —
critically — (c) that non-convergence / out-of-bounds is *caught*, not
silently embedded.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.geometry._inverse_map import (
    InverseMapWarning,
    inverse_map_single,
    locate_point,
)


# ---------------------------------------------------------------------------
# Reference host geometries (straight-sided)
# ---------------------------------------------------------------------------

# unit tet
TET4 = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=float)

# axis-aligned unit hex, gmsh/VTK corner order
HEX8 = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=float)

# planar quad in z=0
QUAD4 = np.array([
    [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
], dtype=float)

TRI3 = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0],
], dtype=float)


def _N_hex8(xi):
    s = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    return 0.125 * (1 + s[:, 0]*xi[0]) * (1 + s[:, 1]*xi[1]) * (1 + s[:, 2]*xi[2])


class TestRoundTrip:
    @pytest.mark.parametrize("seed", range(8))
    def test_hex8_recovers_xi(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        xi_true = rng.uniform(-0.9, 0.9, size=3)
        p = _N_hex8(xi_true) @ HEX8
        xi, w, excess, conv = inverse_map_single(p, HEX8, "hex8")
        assert conv
        assert np.allclose(xi, xi_true, atol=1e-10)
        assert excess == pytest.approx(0.0, abs=1e-9)
        assert w.sum() == pytest.approx(1.0)
        # weights reconstruct the point
        assert np.allclose(w @ HEX8, p, atol=1e-10)

    @pytest.mark.parametrize("seed", range(8))
    def test_tet4_recovers_xi(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # random interior barycentric
        b = rng.uniform(0.05, 1.0, size=4)
        b /= b.sum()
        p = b @ TET4
        xi, w, excess, conv = inverse_map_single(p, TET4, "tet4")
        assert conv
        assert np.allclose(w, b, atol=1e-12)
        assert excess == pytest.approx(0.0, abs=1e-12)
        assert w.sum() == pytest.approx(1.0)

    def test_quad4_recovers_xi(self) -> None:
        xi, w, excess, conv = inverse_map_single(
            np.array([1.0, 0.5, 0.0]), QUAD4, "quad4"
        )
        assert conv
        # centre of the 2x1 quad -> (0, 0)
        assert np.allclose(xi, [0.0, 0.0], atol=1e-10)
        assert np.allclose(w, [0.25, 0.25, 0.25, 0.25])

    def test_tri3_area_coords(self) -> None:
        xi, w, excess, conv = inverse_map_single(
            np.array([1/3, 1/3, 0.0]), TRI3, "tri3"
        )
        assert conv
        assert np.allclose(w, [1/3, 1/3, 1/3])
        assert excess == pytest.approx(0.0, abs=1e-12)


class TestBounds:
    def test_hex_corner_is_in_bounds(self) -> None:
        # the corner node 7 (1,1,1) -> xi=(1,1,1), excess 0
        _, w, excess, conv = inverse_map_single(HEX8[6], HEX8, "hex8")
        assert conv and excess == pytest.approx(0.0, abs=1e-9)

    def test_point_outside_hex_has_positive_excess(self) -> None:
        p = np.array([1.5, 0.5, 0.5])      # ξ=2 in x → excess 1
        xi, w, excess, conv = inverse_map_single(p, HEX8, "hex8")
        assert conv
        assert excess == pytest.approx(1.0, rel=1e-6)

    def test_degenerate_tet_flagged_not_converged(self) -> None:
        flat = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.0],  # coplanar
        ], dtype=float)
        _, _, excess, conv = inverse_map_single(
            np.array([0.25, 0.25, 0.0]), flat, "tet4"
        )
        assert not conv
        assert excess == float("inf")


class TestLocate:
    def _two_hex_stack(self):
        # two unit hexes stacked in z: [0,1] and [1,2]
        lower = HEX8.copy()
        upper = HEX8.copy()
        upper[:, 2] += 1.0
        return [lower, upper], ["hex8", "hex8"]

    def test_locate_picks_correct_host(self) -> None:
        coords, kinds = self._two_hex_stack()
        res = locate_point(np.array([0.5, 0.5, 1.5]), coords, kinds)
        assert res.host_index == 1          # the upper hex
        assert res.in_bounds
        assert np.allclose(res.xi, [0.0, 0.0, 0.0], atol=1e-9)

    def test_locate_inside_lower(self) -> None:
        coords, kinds = self._two_hex_stack()
        res = locate_point(np.array([0.5, 0.5, 0.25]), coords, kinds)
        assert res.host_index == 0
        assert res.in_bounds

    def test_outside_all_hosts_raises_by_default(self) -> None:
        coords, kinds = self._two_hex_stack()
        with pytest.raises(ValueError, match="lies outside every host"):
            locate_point(np.array([5.0, 5.0, 5.0]), coords, kinds, label="bars")

    def test_outside_all_hosts_snaps_with_warning(self) -> None:
        coords, kinds = self._two_hex_stack()
        with pytest.warns(InverseMapWarning, match="SNAPPED"):
            res = locate_point(
                np.array([0.5, 0.5, 2.5]), coords, kinds, snap=True,
            )
        assert not res.in_bounds
        assert res.host_index == 1          # nearest host (upper)

    def test_near_edge_point_accepted(self) -> None:
        coords, kinds = self._two_hex_stack()
        # just inside the shared face z=1, in the lower hex
        res = locate_point(np.array([0.5, 0.5, 0.999999]), coords, kinds)
        assert res.in_bounds

    def test_label_threads_into_error(self) -> None:
        coords, kinds = self._two_hex_stack()
        with pytest.raises(ValueError, match="reinforcement 'col-rebar'"):
            locate_point(
                np.array([9.0, 9.0, 9.0]), coords, kinds, label="col-rebar",
            )


class TestGuards:
    def test_unsupported_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported host kind"):
            inverse_map_single(np.zeros(3), HEX8, "hex20")

    def test_wrong_node_count_raises(self) -> None:
        with pytest.raises(ValueError, match="needs 8 node coords"):
            inverse_map_single(np.zeros(3), TET4, "hex8")

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            locate_point(np.zeros(3), [HEX8], ["hex8", "hex8"])

    def test_empty_hosts_raises(self) -> None:
        with pytest.raises(ValueError, match="no host elements"):
            locate_point(np.zeros(3), [], [])
