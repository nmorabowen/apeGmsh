"""Analytical oracle tests for Slice 3 — Exact Piecewise-Cubic Beam Span Diagrams
(``src/apeGmsh/results/_beam_diagrams.py``).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.results import (
    BeamSpanDiagram,
    BeamSpanSegmentLoad,
    ElementSlab,
    LineStationSlab,
    parse_ele_load_beam_uniform,
    parse_membrane_subsegment_loads,
    reconstruct_beam_element_diagram,
    reconstruct_beam_span_diagrams,
)

# ===========================================================================
# Helpers: Exact Hermite Fixed-End Forces for Partial Trapezoidal Load
# ===========================================================================


def _hermite_fixed_end_forces_1d(
    L: float, a: float, b: float, wa: float, wb: float,
) -> tuple[float, float, float, float]:
    """Exact fixed-end reactions ``(V_i, M_i, V_j, M_j)`` for a load ``w(x)`` on ``[a, b]``.

    Uses exact 4-point Gauss-Legendre quadrature (exact for polynomials up to
    degree 7; integrand ``w(x) * N(x)`` is degree 4) on ``[a, b]``.
    Here ``w(x)`` is positive in ``+y`` (so downward load has ``w < 0`` and
    produces ``V_i > 0, M_i > 0, V_j > 0, M_j < 0``).
    """
    pts, wts = np.polynomial.legendre.leggauss(4)
    xm = 0.5 * (a + b)
    xr = 0.5 * (b - a)
    c = b - a
    vi = 0.0
    mi = 0.0
    vj = 0.0
    mj = 0.0
    for gp, gw in zip(pts, wts):
        x = xm + xr * float(gp)
        s = x / L
        w_x = wa + (wb - wa) * (x - a) / c
        n_v1 = 1.0 - 3.0 * (s ** 2) + 2.0 * (s ** 3)
        n_t1 = L * (s - 2.0 * (s ** 2) + (s ** 3))
        n_v2 = 3.0 * (s ** 2) - 2.0 * (s ** 3)
        n_t2 = L * (-(s ** 2) + (s ** 3))
        dx = xr * float(gw)
        vi -= w_x * n_v1 * dx
        mi -= w_x * n_t1 * dx
        vj -= w_x * n_v2 * dx
        mj -= w_x * n_t2 * dx
    return vi, mi, vj, mj


def _eval_piecewise_load(
    x: np.ndarray, L: float, loads: list[BeamSpanSegmentLoad], attr_a: str, attr_b: str,
) -> np.ndarray:
    """Evaluate total distributed load w(x) at interior points x."""
    w = np.zeros_like(x, dtype=np.float64)
    for seg in loads:
        a = seg.aL * L
        b = seg.bL * L
        c = b - a
        if c <= 1e-14:
            continue
        wa = float(getattr(seg, attr_a))
        wb = float(getattr(seg, attr_b))
        mask = (x > a) & (x < b)
        w[mask] += wa + (wb - wa) * (x[mask] - a) / c
    return w


# ===========================================================================
# Oracle 1: Partial Trapezoidal Simply Supported & Fixed-Fixed 2D/3D Beams
# ===========================================================================


class TestOracle1PartialTrapezoidalBeams:
    def test_simply_supported_2d_partial_trapezoid_oracle(self) -> None:
        """L = 6 m beam with partial trapezoid on [1.5, 4.5] (aL=0.25, bL=0.75)
        from wya = -10 kN/m to wyb = -30 kN/m.
        """
        L = 6.0
        seg = BeamSpanSegmentLoad(aL=0.25, bL=0.75, wya=-10.0, wyb=-30.0)

        # Analytical resultant: F_total = -60 kN acting at x_R = 3.25 m
        # Reactions: Vy_i = +27.5 kN, Vy_j = +32.5 kN, Mz_i = 0, Mz_j = 0
        local_force_2d = [0.0, 27.5, 0.0, 0.0, 32.5, 0.0]

        diag = reconstruct_beam_element_diagram(
            element_id=1, length=L, local_force=local_force_2d, loads=[seg], n_sub=9,
        )
        assert isinstance(diag, BeamSpanDiagram)
        assert diag.element_id == 1
        assert diag.length == pytest.approx(L)

        # 1. Exact breakpoints x = 1.5 m and x = 4.5 m are in diag.x
        assert np.min(np.abs(diag.x - 1.5)) < 1e-14
        assert np.min(np.abs(diag.x - 4.5)) < 1e-14

        # 2. Exact shear root x* = sqrt(42)/2 m is in diag.x and |Vy(x*)| < 1e-12
        x_star_exact = 0.5 * math.sqrt(42.0)
        idx_star = int(np.argmin(np.abs(diag.x - x_star_exact)))
        assert abs(diag.x[idx_star] - x_star_exact) < 1e-12
        assert abs(diag.Vy[idx_star]) < 1e-12

        # 3. Closed-form analytical peak moment Mz(x*) matches to < 1e-12 relative error
        xi_star = x_star_exact - 1.5
        mz_star_exact = (
            27.5 * x_star_exact
            - 5.0 * (xi_star ** 2)
            - (10.0 / 9.0) * (xi_star ** 3)
        )
        rel_err = abs(diag.Mz[idx_star] - mz_star_exact) / abs(mz_star_exact)
        assert rel_err < 1e-12
        assert np.max(diag.Mz) == pytest.approx(mz_star_exact, rel=1e-12)

        # Whereas 2-station linear interpolation between x=0 and x=L gives 0 everywhere!
        linear_interp_mid = 0.5 * (diag.Mz[0] + diag.Mz[-1])
        assert abs(linear_interp_mid) < 1e-12
        assert diag.Mz[idx_star] > 50.0  # ~ 59.358 kN*m

        # 4. Endpoint consistency at x = 0 and x = L
        assert diag.Vy[0] == pytest.approx(27.5, abs=1e-12)
        assert diag.Vy[-1] == pytest.approx(-32.5, abs=1e-12)
        assert diag.Mz[0] == pytest.approx(0.0, abs=1e-12)
        assert diag.Mz[-1] == pytest.approx(0.0, abs=1e-12)

        # 5. Differential relations dVy/dx = wy(x) and dMz/dx = Vy(x) inside each sub-interval
        dx = np.diff(diag.x)
        x_mid = 0.5 * (diag.x[:-1] + diag.x[1:])
        wy_mid = _eval_piecewise_load(x_mid, L, [seg], "wya", "wyb")
        dvy_dx = np.diff(diag.Vy) / dx
        assert np.max(np.abs(dvy_dx - wy_mid)) < 1e-12

        # Exact fundamental theorem of calculus on quadratic Vy(x):
        # (Mz[i+1] - Mz[i]) / dx[i] == 0.5*(Vy[i] + Vy[i+1]) - (1/12)*k*(dx[i]^2)
        k_mid = np.where((x_mid > 1.5) & (x_mid < 4.5), (-30.0 - (-10.0)) / 3.0, 0.0)
        vy_avg_exact = 0.5 * (diag.Vy[:-1] + diag.Vy[1:]) - (1.0 / 12.0) * k_mid * (dx ** 2)
        dmz_dx = np.diff(diag.Mz) / dx
        assert np.max(np.abs(dmz_dx - vy_avg_exact)) < 1e-12

    def test_fixed_fixed_3d_partial_trapezoid_oracle(self) -> None:
        """3D fixed-fixed beam (L = 6 m) with partial trapezoidal loads in y, z, and x."""
        L = 6.0
        a, b = 1.5, 4.5
        wya, wyb = -10.0, -30.0
        wza, wzb = 15.0, -25.0
        wxa, wxb = 4.0, 8.0

        seg = BeamSpanSegmentLoad(
            aL=a / L, bL=b / L,
            wya=wya, wyb=wyb,
            wza=wza, wzb=wzb,
            wxa=wxa, wxb=wxb,
        )

        # Exact Hermite fixed-end forces
        vy_i, mz_i, vy_j, mz_j = _hermite_fixed_end_forces_1d(L, a, b, wya, wyb)
        vz_i, my_i_raw, vz_j, my_j_raw = _hermite_fixed_end_forces_1d(L, a, b, wza, wzb)
        # In 3D OpenSees right-handed coordinates (e_x x e_z = -e_y), My fixed-end moments
        # have opposite sign to Mz for the same +z load direction:
        my_i, my_j = -my_i_raw, -my_j_raw

        # Axial fixed-end forces for linear wx on [a, b]
        # Total axial force Fx = 0.5*(wxa + wxb)*(b - a) = 18 kN
        # Wait: at x=L, N(L) = -N_i - Fx_total = N_j => N_i + N_j = -Fx_total
        fx_total = 0.5 * (wxa + wxb) * (b - a)
        n_i = -0.5 * fx_total
        n_j = -0.5 * fx_total
        t_i = 3.5
        t_j = -3.5

        local_force_3d = [
            n_i, vy_i, vz_i, t_i, my_i, mz_i,
            n_j, vy_j, vz_j, t_j, my_j, mz_j,
        ]

        diag = reconstruct_beam_element_diagram(
            element_id=7, length=L, local_force=local_force_3d, loads=[seg], n_sub=10,
        )

        # Endpoint checks at x = 0 and x = L
        assert diag.N[0] == pytest.approx(-n_i, abs=1e-12)
        assert diag.N[-1] == pytest.approx(n_j, abs=1e-12)
        assert diag.Vy[0] == pytest.approx(vy_i, abs=1e-12)
        assert diag.Vy[-1] == pytest.approx(-vy_j, abs=1e-12)
        assert diag.Vz[0] == pytest.approx(-vz_i, abs=1e-12)
        assert diag.Vz[-1] == pytest.approx(vz_j, abs=1e-12)
        assert np.allclose(diag.T, -t_i, atol=1e-12)
        assert diag.Mz[0] == pytest.approx(-mz_i, abs=1e-12)
        assert diag.Mz[-1] == pytest.approx(mz_j, abs=1e-12)
        assert diag.My[0] == pytest.approx(-my_i, abs=1e-12)
        assert diag.My[-1] == pytest.approx(my_j, abs=1e-12)

        # Differential relations dVz/dx = -wz(x) and dMy/dx = +Vz(x) inside each sub-interval
        dx = np.diff(diag.x)
        x_mid = 0.5 * (diag.x[:-1] + diag.x[1:])
        wz_mid = _eval_piecewise_load(x_mid, L, [seg], "wza", "wzb")
        dvz_dx = np.diff(diag.Vz) / dx
        assert np.max(np.abs(dvz_dx - (-wz_mid))) < 1e-12
        kz_mid = np.where((x_mid > a) & (x_mid < b), (wzb - wza) / (b - a), 0.0)
        vz_avg_exact = 0.5 * (diag.Vz[:-1] + diag.Vz[1:]) + (1.0 / 12.0) * kz_mid * (dx ** 2)
        dmy_dx = np.diff(diag.My) / dx
        assert np.max(np.abs(dmy_dx - vz_avg_exact)) < 1e-12

        # Interior shear root for Vy(x*) = 0
        # Inside [1.5, 4.5], Vy(1.5 + xi) = vy_i - 10*xi - (10/3)*xi^2 = 0
        xi_y_star = (-10.0 + math.sqrt(100.0 + 4.0 * (10.0 / 3.0) * vy_i)) / (2.0 * (10.0 / 3.0))
        x_y_star = a + xi_y_star
        idx_y = int(np.argmin(np.abs(diag.x - x_y_star)))
        assert abs(diag.x[idx_y] - x_y_star) < 1e-12
        assert abs(diag.Vy[idx_y]) < 1e-12

        mz_star_exact = (
            -mz_i
            + vy_i * x_y_star
            + 0.5 * wya * (xi_y_star ** 2)
            + (1.0 / 6.0) * ((wyb - wya) / (b - a)) * (xi_y_star ** 3)
        )
        assert abs(diag.Mz[idx_y] - mz_star_exact) / abs(mz_star_exact) < 1e-12

        # Interior shear root for Vz(x*) = 0
        # Inside [1.5, 4.5], Vz(1.5 + xi) = -vz_i - 15*xi + (20/3)*xi^2 = 0
        xi_z_star = (15.0 + math.sqrt(225.0 + 4.0 * (20.0 / 3.0) * vz_i)) / (2.0 * (20.0 / 3.0))
        x_z_star = a + xi_z_star
        idx_z = int(np.argmin(np.abs(diag.x - x_z_star)))
        assert abs(diag.x[idx_z] - x_z_star) < 1e-12
        assert abs(diag.Vz[idx_z]) < 1e-12

        my_star_exact = (
            -my_i
            - vz_i * x_z_star
            - 0.5 * wza * (xi_z_star ** 2)
            - (1.0 / 6.0) * ((wzb - wza) / (b - a)) * (xi_z_star ** 3)
        )
        assert abs(diag.My[idx_z] - my_star_exact) / abs(my_star_exact) < 1e-12


# ===========================================================================
# Oracle 2: LadrunoMembrane 9-column subSegmentLoads + Multi-Segment Superposition
# ===========================================================================


class TestOracle2MembraneSubSegmentLoads:
    def test_yield_line_45_degree_slab_multi_segment_superposition(self) -> None:
        """6 m x 4 m two-way slab under q = 12 kPa (w_max = -24 kN/m):
        - Long beam (tag 101, L = 6 m): [0, 1/3] rising 0 -> -24, [1/3, 2/3] constant -24,
          [2/3, 1.0] falling -24 -> 0.
        - Short beam (tag 102, L = 4 m): [0, 0.5] rising 0 -> -24, [0.5, 1.0] falling -24 -> 0.
        """
        raw_subsegment_loads = [
            # Long beam 101 (3 segments)
            101.0, 0.0, 1.0 / 3.0, 0.0, -24.0, 0.0, 0.0, 0.0, 0.0,
            101.0, 1.0 / 3.0, 2.0 / 3.0, -24.0, -24.0, 0.0, 0.0, 0.0, 0.0,
            101.0, 2.0 / 3.0, 1.0, -24.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Short beam 102 (2 segments)
            102.0, 0.0, 0.5, 0.0, -24.0, 0.0, 0.0, 0.0, 0.0,
            102.0, 0.5, 1.0, -24.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]

        parsed = parse_membrane_subsegment_loads(raw_subsegment_loads)
        assert set(parsed.keys()) == {101, 102}
        assert len(parsed[101]) == 3
        assert len(parsed[102]) == 2

        # Long beam (L = 6 m): total load = 24 + 48 + 24 = 96 kN -> Vy_i = Vy_j = 48 kN
        diag_long = reconstruct_beam_element_diagram(
            element_id=101,
            length=6.0,
            local_force=[0.0, 48.0, 0.0, 0.0, 0.0, 0.0, 0.0, 48.0, 0.0, 0.0, 0.0, 0.0],
            loads=parsed[101],
            n_sub=8,
        )
        # Shear conservation: Vy(0) - Vy(L) = 96 kN
        assert (diag_long.Vy[0] - diag_long.Vy[-1]) == pytest.approx(96.0, abs=1e-12)
        assert diag_long.Vy[-1] == pytest.approx(-48.0, abs=1e-12)
        assert diag_long.Mz[-1] == pytest.approx(0.0, abs=1e-12)

        # Exact midspan x = 3.0 m is a station (where Vy = 0)
        idx_mid_long = int(np.argmin(np.abs(diag_long.x - 3.0)))
        assert abs(diag_long.x[idx_mid_long] - 3.0) < 1e-12
        assert abs(diag_long.Vy[idx_mid_long]) < 1e-12
        # Analytical M_max(3.0) = 48*3 - 24*(3 - 4/3) - 24*(3 - 2.5) = 144 - 40 - 12 = 92.0 kN*m
        assert diag_long.Mz[idx_mid_long] == pytest.approx(92.0, rel=1e-12)

        # Short beam (L = 4 m): total load = 24 + 24 = 48 kN -> Vy_i = Vy_j = 24 kN
        diag_short = reconstruct_beam_element_diagram(
            element_id=102,
            length=4.0,
            local_force=[0.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0],
            loads=parsed[102],
            n_sub=8,
        )
        assert (diag_short.Vy[0] - diag_short.Vy[-1]) == pytest.approx(48.0, abs=1e-12)
        idx_mid_short = int(np.argmin(np.abs(diag_short.x - 2.0)))
        assert abs(diag_short.x[idx_mid_short] - 2.0) < 1e-12
        assert abs(diag_short.Vy[idx_mid_short]) < 1e-12
        # Analytical M_max(2.0) = 24*2 - 24*(2 - 4/3) = 32.0 kN*m
        assert diag_short.Mz[idx_mid_short] == pytest.approx(32.0, rel=1e-12)


# ===========================================================================
# Oracle 3: parse_ele_load_beam_uniform & OpenSeesModel + Results Integration
# ===========================================================================


class _StubElementsComposite:
    def __init__(
        self,
        slabs: dict[str, ElementSlab],
        line_station_slabs: dict[str, LineStationSlab] | None = None,
    ) -> None:
        self._slabs = slabs
        self.line_stations = _StubLineStationsComposite(line_station_slabs or {})

    def available_components(self) -> list[str]:
        return list(self._slabs.keys())

    def get(self, *, component: str, time: Any = None, **kwargs: Any) -> ElementSlab:
        return self._slabs[component]


class _StubLineStationsComposite:
    def __init__(self, slabs: dict[str, LineStationSlab]) -> None:
        self._slabs = slabs

    def available_components(self) -> list[str]:
        return list(self._slabs.keys())

    def get(self, *, component: str, time: Any = None, **kwargs: Any) -> LineStationSlab:
        return self._slabs[component]


class _StubResults:
    def __init__(
        self,
        *,
        fem: FEMData | None,
        model: OpenSeesModel | None,
        slabs: dict[str, ElementSlab],
        line_station_slabs: dict[str, LineStationSlab] | None = None,
    ) -> None:
        self.fem = fem
        self.model = model
        self.elements = _StubElementsComposite(slabs, line_station_slabs)


def _single_beam_fem(length: float = 6.0, fem_eid: int = 10) -> FEMData:
    coords = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=np.float64)
    node_ids = np.array([1, 2], dtype=np.int64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([fem_eid], dtype=np.int64),
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet({
            (0, 1): {
                "name": "Ends",
                "node_ids": node_ids,
                "node_coords": coords,
            },
        }),
        labels=LabelSet({
            (0, 1): {
                "name": "EndI",
                "node_ids": node_ids[:1],
                "node_coords": coords[:1],
            },
            (0, 2): {
                "name": "EndJ",
                "node_ids": node_ids[1:],
                "node_coords": coords[1:],
            },
        }),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({
            (1, 1): {
                "name": "Beams",
                "node_ids": node_ids,
                "node_coords": coords,
                "element_types": [1],
                "element_ids": np.array([fem_eid], dtype=np.int64),
                "connectivity": np.array([[1, 2]], dtype=np.int64),
            },
        }),
        labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=2, n_elems=1, bandwidth=1, types=[line_info])
    return FEMData(nodes=nodes, elements=elements, info=info)


class TestOracle3EleLoadParsingAndModelIntegration:
    def test_parse_ele_load_beam_uniform_all_2d_and_3d_arities(self) -> None:
        # 2D: 2 floats (wy, wx)
        res = parse_ele_load_beam_uniform(
            ("-ele", 1, 2, "-type", "-beamUniform", -12.0, 3.0), ndm=2, scale=2.0,
        )
        assert res is not None
        tags, seg = res
        assert tags == (1, 2)
        assert seg == BeamSpanSegmentLoad(
            aL=0.0, bL=1.0, wya=-24.0, wyb=-24.0, wza=0.0, wzb=0.0, wxa=6.0, wxb=6.0,
        )

        # 2D: 4 floats (wy, wx, aL, bL)
        res = parse_ele_load_beam_uniform(
            ("-ele", 5, "-type", "-beamUniform", -10.0, 1.0, 0.2, 0.8), ndm=2,
        )
        assert res is not None
        tags, seg = res
        assert tags == (5,)
        assert seg == BeamSpanSegmentLoad(
            aL=0.2, bL=0.8, wya=-10.0, wyb=-10.0, wza=0.0, wzb=0.0, wxa=1.0, wxb=1.0,
        )

        # 2D: 6 floats (wya, wxa, aL, bL, wyb, wxb)
        res = parse_ele_load_beam_uniform(
            ("-ele", 8, "-type", "-beamUniform", -10.0, 1.0, 0.25, 0.75, -30.0, 5.0), ndm=2,
        )
        assert res is not None
        tags, seg = res
        assert tags == (8,)
        assert seg == BeamSpanSegmentLoad(
            aL=0.25, bL=0.75, wya=-10.0, wyb=-30.0, wza=0.0, wzb=0.0, wxa=1.0, wxb=5.0,
        )

        # 3D: 2 floats (wy, wz)
        res = parse_ele_load_beam_uniform(
            ("-ele", 3, "-type", "-beamUniform", -15.0, 4.0), ndm=3,
        )
        assert res is not None
        tags, seg = res
        assert tags == (3,)
        assert seg == BeamSpanSegmentLoad(
            aL=0.0, bL=1.0, wya=-15.0, wyb=-15.0, wza=4.0, wzb=4.0, wxa=0.0, wxb=0.0,
        )

        # 3D: 3 floats (wy, wz, wx)
        res = parse_ele_load_beam_uniform(
            ("-ele", 3, "-type", "-beamUniform", -15.0, 4.0, -2.0), ndm=3,
        )
        assert res is not None
        tags, seg = res
        assert tags == (3,)
        assert seg == BeamSpanSegmentLoad(
            aL=0.0, bL=1.0, wya=-15.0, wyb=-15.0, wza=4.0, wzb=4.0, wxa=-2.0, wxb=-2.0,
        )

        # 3D: 5 floats (wy, wz, wx, aL, bL)
        res = parse_ele_load_beam_uniform(
            ("-range", 10, 12, "-type", "-beamUniform", -15.0, 4.0, -2.0, 0.1, 0.9), ndm=3,
        )
        assert res is not None
        tags, seg = res
        assert tags == (10, 11, 12)
        assert seg == BeamSpanSegmentLoad(
            aL=0.1, bL=0.9, wya=-15.0, wyb=-15.0, wza=4.0, wzb=4.0, wxa=-2.0, wxb=-2.0,
        )

        # 3D: 8 floats (wya, wza, wxa, aL, bL, wyb, wzb, wxb)
        res = parse_ele_load_beam_uniform(
            ("-ele", 9, "-type", "-beamUniform", -10.0, 2.0, 1.0, 0.25, 0.75, -30.0, 6.0, 3.0),
            ndm=3,
        )
        assert res is not None
        tags, seg = res
        assert tags == (9,)
        assert seg == BeamSpanSegmentLoad(
            aL=0.25, bL=0.75, wya=-10.0, wyb=-30.0, wza=2.0, wzb=6.0, wxa=1.0, wxb=3.0,
        )

        # Non-beamUniform returns None
        assert parse_ele_load_beam_uniform(("-ele", 1, "-type", "-surfaceLoad"), ndm=3) is None
        assert parse_ele_load_beam_uniform(("-ele", 1, "-type", "-selfWeight", 0.0, -1.0, 0.0), ndm=3) is None

    def test_reconstruct_beam_span_diagrams_with_opensees_model_and_results(
        self, tmp_path: Path,
    ) -> None:
        """End-to-end integration: build an apeSees model with Plain.beam_uniform,
        archive to model.h5, load OpenSeesModel, and reconstruct beam span diagrams.
        """
        fem = _single_beam_fem(length=6.0, fem_eid=10)
        ops = apeSees(cast("Any", fem))
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        ops.element.elasticBeamColumn(
            pg="Beams", A=0.01, E=200e9, G=80e9, J=1e-4, Iy=1e-4, Iz=1e-4, transf=transf,
        )
        ops.fix(pg="Ends", dofs=(1, 1, 1, 1, 0, 0))
        ts = ops.timeSeries.Linear()
        with ops.pattern.Plain(series=ts) as p:
            p.beam_uniform(pg="Beams", wy=(-10.0, -30.0), span=(0.25, 0.75))

        h5_path = tmp_path / "beam_trap.h5"
        ops.h5(str(h5_path))
        om = OpenSeesModel.from_h5(str(h5_path))

        # Verify OpenSeesModel has the eleLoad in om.patterns()
        pats = om.patterns()
        assert len(pats) == 1
        assert len(pats[0].ele_loads) == 1

        # Create Results stub with fem_eid = 10 (testing fem_eid <-> ops_tag=1 pairing!)
        lf_vec = np.array([[[0.0, 27.5, 0.0, 0.0, 0.0, 0.0, 0.0, 32.5, 0.0, 0.0, 0.0, 0.0]]])
        slab = ElementSlab(
            component="localForce",
            values=lf_vec,
            element_ids=np.array([10], dtype=np.int64),
            time=np.array([1.0]),
        )
        results = _StubResults(fem=fem, model=om, slabs={"localForce": slab})

        diagrams = reconstruct_beam_span_diagrams(
            cast("Any", results), n_sub=8,
        )
        assert 10 in diagrams
        diag = diagrams[10]
        x_star_exact = 0.5 * math.sqrt(42.0)
        xi_star = x_star_exact - 1.5
        mz_star_exact = 27.5 * x_star_exact - 5.0 * (xi_star ** 2) - (10.0 / 9.0) * (xi_star ** 3)
        assert np.max(diag.Mz) == pytest.approx(mz_star_exact, rel=1e-12)
        assert diag.Mz[-1] == pytest.approx(0.0, abs=1e-12)

    def test_reconstruct_beam_span_diagrams_canonical_nodal_and_line_stations(self) -> None:
        """Verify reconstruct_beam_span_diagrams with canonical nodal_resisting_*_local_*
        components and with LineStationSlab fallback + raw subsegment_loads.
        """
        fem = _single_beam_fem(length=4.0, fem_eid=1)
        # Short beam from Oracle 2: L = 4 m, rising [0, 0.5] 0 -> -24, falling [0.5, 1.0] -24 -> 0
        raw_sub = [
            1.0, 0.0, 0.5, 0.0, -24.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.5, 1.0, -24.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        eids = np.array([1], dtype=np.int64)
        t_arr = np.array([1.0])
        slabs_2d = {
            "nodal_resisting_force_local_x": ElementSlab(
                component="nodal_resisting_force_local_x",
                values=np.array([[[0.0, 0.0]]]),
                element_ids=eids,
                time=t_arr,
            ),
            "nodal_resisting_force_local_y": ElementSlab(
                component="nodal_resisting_force_local_y",
                values=np.array([[[24.0, 24.0]]]),
                element_ids=eids,
                time=t_arr,
            ),
            "nodal_resisting_moment_local_z": ElementSlab(
                component="nodal_resisting_moment_local_z",
                values=np.array([[[0.0, 0.0]]]),
                element_ids=eids,
                time=t_arr,
            ),
        }
        res_nodal = _StubResults(fem=fem, model=None, slabs=slabs_2d)
        diags_nodal = reconstruct_beam_span_diagrams(
            cast("Any", res_nodal), subsegment_loads=raw_sub, n_sub=6,
        )
        assert diags_nodal[1].Mz.max() == pytest.approx(32.0, rel=1e-12)

        # Also test LineStationSlab fallback
        ls_slabs = {
            "axial_force": LineStationSlab(
                component="axial_force",
                values=np.array([[0.0, 0.0]]),
                element_index=np.array([1, 1], dtype=np.int64),
                station_natural_coord=np.array([-1.0, 1.0]),
                time=t_arr,
            ),
            "shear_y": LineStationSlab(
                component="shear_y",
                values=np.array([[24.0, -24.0]]),
                element_index=np.array([1, 1], dtype=np.int64),
                station_natural_coord=np.array([-1.0, 1.0]),
                time=t_arr,
            ),
            "bending_moment_z": LineStationSlab(
                component="bending_moment_z",
                values=np.array([[0.0, 0.0]]),
                element_index=np.array([1, 1], dtype=np.int64),
                station_natural_coord=np.array([-1.0, 1.0]),
                time=t_arr,
            ),
        }
        res_ls = _StubResults(fem=fem, model=None, slabs={}, line_station_slabs=ls_slabs)
        diags_ls = reconstruct_beam_span_diagrams(
            cast("Any", res_ls), subsegment_loads=parse_membrane_subsegment_loads(raw_sub), n_sub=6,
        )
        assert diags_ls[1].Mz.max() == pytest.approx(32.0, rel=1e-12)

    def test_3d_line_stations_vz_sign_and_multistep_selection(self) -> None:
        """Regression test for 3D LineStationSlab shear_z sign, station ordering,
        and non-final step selection (step=0 vs step=1).
        """
        fem = _single_beam_fem(length=4.0, fem_eid=1)
        # Step 0 has load w_z = -12 kN/m uniform on L=4m -> Vz_I = +24, Vz_J = +24,
        # so catalog section-force Vz(0) = -24, Vz(L) = +24,
        # My(0) = 0, My(L) = 0, My(2.0) = -24.0 kN*m (since dMy/dx = +Vz(x)).
        # Step 1 has double the load (w_z = -24 kN/m -> Vz(0) = -48, Vz(L) = +48).
        # Also reverse station_natural_coord = [+1.0, -1.0] to verify sorting!
        t_arr = np.array([0.5, 1.0])
        eidx = np.array([1, 1], dtype=np.int64)
        xi_rev = np.array([1.0, -1.0])
        ls_slabs_3d = {
            "axial_force": LineStationSlab(
                component="axial_force",
                values=np.array([[10.0, 10.0], [20.0, 20.0]]),
                element_index=eidx,
                station_natural_coord=xi_rev,
                time=t_arr,
            ),
            "shear_y": LineStationSlab(
                component="shear_y",
                values=np.array([[0.0, 0.0], [0.0, 0.0]]),
                element_index=eidx,
                station_natural_coord=xi_rev,
                time=t_arr,
            ),
            "shear_z": LineStationSlab(
                component="shear_z",
                # At xi=+1 (first col): Vz(L) = +24 (step 0), +48 (step 1)
                # At xi=-1 (second col): Vz(0) = -24 (step 0), -48 (step 1)
                values=np.array([[24.0, -24.0], [48.0, -48.0]]),
                element_index=eidx,
                station_natural_coord=xi_rev,
                time=t_arr,
            ),
            "torsion": LineStationSlab(
                component="torsion",
                values=np.array([[5.0, 5.0], [10.0, 10.0]]),
                element_index=eidx,
                station_natural_coord=xi_rev,
                time=t_arr,
            ),
            "bending_moment_y": LineStationSlab(
                component="bending_moment_y",
                values=np.array([[0.0, 0.0], [0.0, 0.0]]),
                element_index=eidx,
                station_natural_coord=xi_rev,
                time=t_arr,
            ),
            "bending_moment_z": LineStationSlab(
                component="bending_moment_z",
                values=np.array([[0.0, 0.0], [0.0, 0.0]]),
                element_index=eidx,
                station_natural_coord=xi_rev,
                time=t_arr,
            ),
        }
        res_3d = _StubResults(fem=fem, model=None, slabs={}, line_station_slabs=ls_slabs_3d)
        seg_step0 = {1: [BeamSpanSegmentLoad(aL=0.0, bL=1.0, wya=0.0, wyb=0.0, wza=-12.0, wzb=-12.0)]}
        diags_s0 = reconstruct_beam_span_diagrams(
            cast("Any", res_3d), subsegment_loads=seg_step0, step=0, n_sub=4,
        )
        d0 = diags_s0[1]
        assert d0.N[0] == pytest.approx(10.0, abs=1e-12)
        assert d0.T[0] == pytest.approx(5.0, abs=1e-12)
        assert d0.Vz[0] == pytest.approx(-24.0, abs=1e-12)
        assert d0.Vz[-1] == pytest.approx(24.0, abs=1e-12)
        assert d0.My[0] == pytest.approx(0.0, abs=1e-12)
        assert d0.My[-1] == pytest.approx(0.0, abs=1e-12)
        idx_mid = int(np.argmin(np.abs(d0.x - 2.0)))
        assert d0.Vz[idx_mid] == pytest.approx(0.0, abs=1e-12)
        assert d0.My[idx_mid] == pytest.approx(-24.0, rel=1e-12)

    def test_catalog_station_signs_and_line_stations_roundtrip_parity(self) -> None:
        """Verify that reconstruct_beam_element_diagram endpoints and the
        LineStationSlab fallback in reconstruct_beam_span_diagrams agree with
        _LOCAL_FORCE_END_I_SIGN / local_force_station_signs across all 6 3D
        components (N, Vy, Vz, T, My, Mz), including a 3D cantilever tip load
        and a 3D partial trapezoidal load.
        """
        from apeGmsh.opensees._response_catalog import (
            _LOCAL_FORCE_END_I_SIGN,
            local_force_station_signs,
        )

        assert _LOCAL_FORCE_END_I_SIGN["shear_z"] == -1.0
        assert _LOCAL_FORCE_END_I_SIGN["bending_moment_y"] == -1.0

        comps_3d = (
            "axial_force",
            "shear_y",
            "shear_z",
            "torsion",
            "bending_moment_y",
            "bending_moment_z",
        )
        signs = local_force_station_signs(comps_3d)  # shape (2, 6)

        # 1. 3D cantilever (L = 4 m) with tip load (Fx, Fy, Fz, Mx) = (1, 2, 3, 0.4) at node j:
        # localForce = [N_i, Vy_i, Vz_i, T_i, My_i, Mz_i, N_j, Vy_j, Vz_j, T_j, My_j, Mz_j]
        L_cant = 4.0
        lf_cant = np.array(
            [-1.0, -2.0, -3.0, -0.4, 12.0, -8.0, 1.0, 2.0, 3.0, 0.4, 0.0, 0.0],
            dtype=np.float64,
        )
        diag_cant = reconstruct_beam_element_diagram(1, L_cant, lf_cant, n_sub=4)
        expected_i = signs[0] * lf_cant[:6]
        expected_j = signs[1] * lf_cant[6:]
        got_i = np.array(
            [
                diag_cant.N[0],
                diag_cant.Vy[0],
                diag_cant.Vz[0],
                diag_cant.T[0],
                diag_cant.My[0],
                diag_cant.Mz[0],
            ]
        )
        got_j = np.array(
            [
                diag_cant.N[-1],
                diag_cant.Vy[-1],
                diag_cant.Vz[-1],
                diag_cant.T[-1],
                diag_cant.My[-1],
                diag_cant.Mz[-1],
            ]
        )
        np.testing.assert_allclose(got_i, expected_i, atol=1e-12)
        np.testing.assert_allclose(got_j, expected_j, atol=1e-12)
        # Closed-form section forces along cantilever: Vz(x) = +3.0, My(x) = -3.0*(4 - x) -> dMy/dx = +Vz
        np.testing.assert_allclose(diag_cant.Vz, 3.0, atol=1e-12)
        np.testing.assert_allclose(diag_cant.My, -3.0 * (L_cant - diag_cant.x), atol=1e-12)
        np.testing.assert_allclose(
            np.diff(diag_cant.My) / np.diff(diag_cant.x), diag_cant.Vz[:-1], atol=1e-12,
        )

        # 2. 3D span with distributed load: localForce path and catalog-signed LineStationSlab
        # path must reconstruct the exact same N, Vy, Vz, T, My, Mz arrays.
        L = 4.0
        fem = _single_beam_fem(length=L, fem_eid=1)
        seg = BeamSpanSegmentLoad(
            aL=0.0, bL=1.0, wya=-6.0, wyb=-6.0, wza=-12.0, wzb=-12.0, wxa=4.0, wxb=4.0,
        )
        lf_ss = np.array(
            [-16.0, 12.0, 24.0, -5.0, 0.0, 0.0, 0.0, 12.0, 24.0, 5.0, 0.0, 0.0],
            dtype=np.float64,
        )
        t_arr = np.array([1.0])
        eids = np.array([1], dtype=np.int64)
        res_lf = _StubResults(
            fem=fem,
            model=None,
            slabs={
                "localForce": ElementSlab(
                    component="localForce",
                    values=lf_ss.reshape(1, 1, 12),
                    element_ids=eids,
                    time=t_arr,
                ),
            },
        )
        ls_slabs = {
            comp: LineStationSlab(
                component=comp,
                values=np.array([[signs[0, k] * lf_ss[k], signs[1, k] * lf_ss[6 + k]]]),
                element_index=np.array([1, 1], dtype=np.int64),
                station_natural_coord=np.array([-1.0, 1.0]),
                time=t_arr,
            )
            for k, comp in enumerate(comps_3d)
        }
        res_ls = _StubResults(fem=fem, model=None, slabs={}, line_station_slabs=ls_slabs)

        d_lf = reconstruct_beam_span_diagrams(
            cast("Any", res_lf), subsegment_loads={1: [seg]}, n_sub=4,
        )[1]
        d_ls = reconstruct_beam_span_diagrams(
            cast("Any", res_ls), subsegment_loads={1: [seg]}, n_sub=4,
        )[1]
        for attr in ("x", "N", "Vy", "Vz", "T", "My", "Mz"):
            np.testing.assert_allclose(
                getattr(d_ls, attr), getattr(d_lf, attr), atol=1e-12, err_msg=attr,
            )

    @pytest.mark.live
    def test_live_opensees_3d_beam_uniform_wz_section_force_oracle(self) -> None:
        """Live OpenSees oracle: 3D forceBeamColumn (5-point Lobatto, Aggregator
        with Vy and Vz) under 3D -beamUniform (wy=-6, wz=-12, wx=4).
        Reconstructed N, Vy, Vz, T, Mz, My must match OpenSees 'section force'
        at all 5 Gauss-Lobatto stations for both simply supported and cantilever BCs.
        """
        pytest.importorskip("openseespy.opensees")
        from apeGmsh.opensees.emitter.live import LiveOpsEmitter

        L = 4.0
        wy, wz, wx = -6.0, -12.0, 4.0
        xi_lobatto_5 = np.array([-1.0, -math.sqrt(3.0 / 7.0), 0.0, math.sqrt(3.0 / 7.0), 1.0])
        x_lobatto_5 = 0.5 * (xi_lobatto_5 + 1.0) * L
        a1, a3 = float(x_lobatto_5[1] / L), float(x_lobatto_5[3] / L)
        segs = [
            BeamSpanSegmentLoad(aL=0.0, bL=a1, wya=wy, wyb=wy, wza=wz, wzb=wz, wxa=wx, wxb=wx),
            BeamSpanSegmentLoad(aL=a1, bL=a3, wya=wy, wyb=wy, wza=wz, wzb=wz, wxa=wx, wxb=wx),
            BeamSpanSegmentLoad(aL=a3, bL=1.0, wya=wy, wyb=wy, wza=wz, wzb=wz, wxa=wx, wxb=wx),
        ]

        for bc in ("ss", "cantilever"):
            fem = _single_beam_fem(length=L, fem_eid=10)
            ops = apeSees(cast("Any", fem))
            ops.model(ndm=3, ndf=6)
            transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
            shear_mat = ops.uniaxialMaterial.ElasticMaterial(E=1e9)
            base_sec = ops.section.Elastic(
                E=200e9, A=0.01, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
            )
            agg_sec = ops.section.Aggregator(
                materials_by_dof={"Vy": shear_mat, "Vz": shear_mat},
                base_section=base_sec,
            )
            integ = ops.beamIntegration.Lobatto(section=agg_sec, n_ip=5)
            ops.element.forceBeamColumn(pg="Beams", transf=transf, integration=integ)
            if bc == "ss":
                ops.fix(label="EndI", dofs=(1, 1, 1, 1, 0, 0))
                ops.fix(label="EndJ", dofs=(0, 1, 1, 0, 0, 0))
            else:
                ops.fix(label="EndI", dofs=(1, 1, 1, 1, 1, 1))
            with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
                p.beam_uniform(pg="Beams", wy=wy, wz=wz, wx=wx)
            ops.constraints.Plain()
            ops.numberer.Plain()
            ops.system.UmfPack()
            ops.test.NormUnbalance(tol=1e-12, max_iter=10)
            ops.algorithm.Newton()
            ops.integrator.LoadControl(dlam=1.0)
            ops.analysis.Static()

            emitter = LiveOpsEmitter(wipe=True)
            ops.build().emit(emitter)
            assert emitter.analyze(steps=1) == 0

            # Tag of the forceBeamColumn element
            ele_tags = emitter.ops.getEleTags()
            assert len(ele_tags) == 1
            etag = int(ele_tags[0])
            lf = np.asarray(emitter.ops.eleResponse(etag, "localForce"), dtype=np.float64)
            sec_forces = np.array(
                [
                    emitter.ops.eleResponse(etag, "section", str(s), "force")
                    for s in range(1, 6)
                ],
                dtype=np.float64,
            )
            # Aggregator section force order: [P, Mz, My, T, Vy, Vz]
            diag = reconstruct_beam_element_diagram(10, L, lf, loads=segs, n_sub=4)
            np.testing.assert_allclose(
                np.interp(x_lobatto_5, diag.x, diag.N), sec_forces[:, 0], atol=1e-9, err_msg=f"{bc} N",
            )
            np.testing.assert_allclose(
                np.interp(x_lobatto_5, diag.x, diag.Mz), sec_forces[:, 1], atol=1e-9, err_msg=f"{bc} Mz",
            )
            np.testing.assert_allclose(
                np.interp(x_lobatto_5, diag.x, diag.My), sec_forces[:, 2], atol=1e-9, err_msg=f"{bc} My",
            )
            np.testing.assert_allclose(
                np.interp(x_lobatto_5, diag.x, diag.T), sec_forces[:, 3], atol=1e-9, err_msg=f"{bc} T",
            )
            np.testing.assert_allclose(
                np.interp(x_lobatto_5, diag.x, diag.Vy), sec_forces[:, 4], atol=1e-9, err_msg=f"{bc} Vy",
            )
            np.testing.assert_allclose(
                np.interp(x_lobatto_5, diag.x, diag.Vz), sec_forces[:, 5], atol=1e-9, err_msg=f"{bc} Vz",
            )

    def test_fem_eid_and_ops_tag_collision_isolation(self, tmp_path: Path) -> None:
        """Regression test: when fem_eids = [1, 2] (with lengths 4.0 m and 6.0 m)
        and ops_tags = [2, 3] (shifted by geomTransf tag 1), a load on fem_eid=1
        (ops_tag=2) must NOT leak onto fem_eid=2 (ops_tag=3), nor overwrite its length.
        """
        coords = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64,
        )
        node_ids = np.array([1, 2, 3], dtype=np.int64)
        line_info = make_type_info(
            code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2,
        )
        line_group = ElementGroup(
            element_type=line_info,
            ids=np.array([1, 2], dtype=np.int64),
            connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64),
        )
        nodes = NodeComposite(
            node_ids=node_ids,
            node_coords=coords,
            physical=PhysicalGroupSet({
                (0, 1): {"name": "Supports", "node_ids": node_ids, "node_coords": coords},
            }),
            labels=LabelSet({}),
        )
        elements = ElementComposite(
            groups={1: line_group},
            physical=PhysicalGroupSet({
                (1, 1): {
                    "name": "AllBeams",
                    "node_ids": node_ids,
                    "node_coords": coords,
                    "element_types": [1],
                    "element_ids": np.array([1, 2], dtype=np.int64),
                    "connectivity": np.array([[1, 2], [2, 3]], dtype=np.int64),
                },
            }),
            labels=LabelSet({
                (1, 1): {
                    "name": "Span1Only",
                    "node_ids": node_ids[:2],
                    "node_coords": coords[:2],
                    "element_types": [1],
                    "element_ids": np.array([1], dtype=np.int64),
                    "connectivity": np.array([[1, 2]], dtype=np.int64),
                },
            }),
        )
        fem = FEMData(
            nodes=nodes,
            elements=elements,
            info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=1, types=[line_info]),
        )
        ops = apeSees(cast("Any", fem))
        ops.model(ndm=2, ndf=3)
        transf = ops.geomTransf.Linear()  # gets tag 1 -> beams get ops_tags 2 and 3!
        ops.element.elasticBeamColumn(
            pg="AllBeams", A=0.01, E=200e9, Iz=1e-4, transf=transf,
        )
        ops.fix(pg="Supports", dofs=(1, 1, 0))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            # Load ONLY fem_eid=1 (which has ops_tag=2!) with wy = -12 kN/m on L = 4 m
            p.beam_uniform(label="Span1Only", wy=-12.0)

        h5_path = tmp_path / "collision.h5"
        ops.h5(str(h5_path))
        om = OpenSeesModel.from_h5(str(h5_path))

        # Span 1 (fem_eid=1, L=4m): loaded with wy=-12 -> Vy_i = 24, Vy_j = 24, M_max = 24 kN*m
        # Span 2 (fem_eid=2, L=6m): UNLOADED -> Vy_i = 0, Vy_j = 0, M_z(x) == 0 everywhere
        lf_vals = np.array([
            [
                [0.0, 24.0, 0.0, 0.0, 24.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ])
        slab = ElementSlab(
            component="localForce",
            values=lf_vals,
            element_ids=np.array([1, 2], dtype=np.int64),
            time=np.array([1.0]),
        )
        results = _StubResults(fem=fem, model=om, slabs={"localForce": slab})
        diags = reconstruct_beam_span_diagrams(cast("Any", results), n_sub=4)

        assert diags[1].length == pytest.approx(4.0)
        assert diags[2].length == pytest.approx(6.0)
        assert np.max(diags[1].Mz) == pytest.approx(24.0, rel=1e-12)
        assert np.max(np.abs(diags[2].Vy)) == pytest.approx(0.0, abs=1e-12)
        assert np.max(np.abs(diags[2].Mz)) == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# Edge Cases & Validation
# ===========================================================================


class TestBeamDiagramsEdgeCases:
    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="length must be finite and > 0"):
            reconstruct_beam_element_diagram(1, 0.0, [0.0] * 6)
        with pytest.raises(ValueError, match="length must be finite and > 0"):
            reconstruct_beam_element_diagram(1, -2.0, [0.0] * 6)
        with pytest.raises(ValueError, match="n_sub must be >= 1"):
            reconstruct_beam_element_diagram(1, 5.0, [0.0] * 6, n_sub=0)
        with pytest.raises(ValueError, match="length 6 \\(2D\\) or 12 \\(3D\\)"):
            reconstruct_beam_element_diagram(1, 5.0, [0.0] * 8)
        with pytest.raises(ValueError, match="0.0 <= aL <= bL <= 1.0"):
            reconstruct_beam_element_diagram(
                1, 5.0, [0.0] * 6, loads=[BeamSpanSegmentLoad(aL=0.8, bL=0.2, wya=-10.0, wyb=-10.0)],
            )
        with pytest.raises(ValueError, match="0.0 <= aL <= bL <= 1.0"):
            reconstruct_beam_element_diagram(
                1, 5.0, [0.0] * 6, loads=[BeamSpanSegmentLoad(aL=-0.1, bL=0.5, wya=-10.0, wyb=-10.0)],
            )
        with pytest.raises(ValueError, match="0.0 <= aL <= bL <= 1.0"):
            reconstruct_beam_element_diagram(
                1, 5.0, [0.0] * 6, loads=[BeamSpanSegmentLoad(aL=0.2, bL=1.2, wya=-10.0, wyb=-10.0)],
            )
        with pytest.raises(ValueError, match="divisible by 9"):
            parse_membrane_subsegment_loads([1.0, 2.0, 3.0])
        assert parse_membrane_subsegment_loads([]) == {}
        with pytest.raises(ValueError, match="ndm must be 2 or 3"):
            parse_ele_load_beam_uniform(("-ele", 1, "-type", "-beamUniform", -1.0, 0.0), ndm=1)
        with pytest.raises(ValueError, match="expected 2, 4, or 6"):
            parse_ele_load_beam_uniform(("-ele", 1, "-type", "-beamUniform", -1.0, 0.0, 0.5), ndm=2)
        with pytest.raises(ValueError, match="expected 2, 3, 5, or 8"):
            parse_ele_load_beam_uniform(("-ele", 1, "-type", "-beamUniform", -1.0, 0.0, 0.5, 0.8), ndm=3)
        with pytest.raises(ValueError, match="0.0 <= aL < bL <= 1.0"):
            parse_ele_load_beam_uniform(
                ("-ele", 1, "-type", "-beamUniform", -1.0, 0.0, 0.8, 0.2), ndm=2,
            )

