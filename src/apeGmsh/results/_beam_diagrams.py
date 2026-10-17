"""Exact piecewise-cubic beam internal force diagram reconstruction.

Reconstructs axial force :math:`N(x)`, transverse shears :math:`V_y(x), V_z(x)`,
torsion :math:`T(x)`, and bending moments :math:`M_z(x), M_y(x)` along 2D and 3D
frame elements from OpenSees ``localForce`` end vectors and active partial
trapezoidal span loads (``eleLoad -type -beamUniform`` and ``LadrunoMembrane``
``"subSegmentLoads"`` responses).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from ..opensees.opensees_model import OpenSeesModel
    from .Results import Results

__all__ = [
    "BeamSpanSegmentLoad",
    "BeamSpanDiagram",
    "parse_membrane_subsegment_loads",
    "parse_ele_load_beam_uniform",
    "reconstruct_beam_element_diagram",
    "reconstruct_beam_span_diagrams",
]

_BEAM_NODAL_2D_LOCAL: tuple[str, ...] = (
    "nodal_resisting_force_local_x",
    "nodal_resisting_force_local_y",
    "nodal_resisting_moment_local_z",
)
_BEAM_NODAL_3D_LOCAL: tuple[str, ...] = (
    "nodal_resisting_force_local_x",
    "nodal_resisting_force_local_y",
    "nodal_resisting_force_local_z",
    "nodal_resisting_moment_local_x",
    "nodal_resisting_moment_local_y",
    "nodal_resisting_moment_local_z",
)


@dataclass(frozen=True)
class BeamSpanSegmentLoad:
    """One partial trapezoidal distributed load segment on ``[aL, bL]``.

    All load intensities (``wya``, ``wyb``, ``wza``, ``wzb``, ``wxa``, ``wxb``)
    are expressed in the beam element's local coordinate system ``(x, y, z)``.
    """

    aL: float
    bL: float
    wya: float
    wyb: float
    wza: float = 0.0
    wzb: float = 0.0
    wxa: float = 0.0
    wxb: float = 0.0


@dataclass(frozen=True)
class BeamSpanDiagram:
    """Exact piecewise-cubic internal force diagrams along one beam element.

    Sign conventions match OpenSees ``section force`` along the element's
    positive ``+x`` face:

    * ``N(x)``: axial force (tension positive; :math:`N(0) = -N_I`,
      :math:`N(L) = +N_J`, :math:`dN/dx = -w_x(x)`).
    * ``Vy(x)``: local-y shear (:math:`V_y(0) = +V_{y,I}`,
      :math:`V_y(L) = -V_{y,J}`, :math:`dV_y/dx = w_y(x)`).
    * ``Vz(x)``: local-z shear (:math:`V_z(0) = +V_{z,I}`,
      :math:`V_z(L) = -V_{z,J}`, :math:`dV_z/dx = w_z(x)`).
    * ``T(x)``: torsion (:math:`T(0) = -T_I`).
    * ``Mz(x)``: bending moment about local z associated with :math:`V_y`
      (:math:`M_z(0) = -M_{z,I}`, :math:`M_z(L) = +M_{z,J}`,
      :math:`dM_z/dx = V_y(x)`).
    * ``My(x)``: bending moment about local y associated with :math:`V_z`
      (:math:`M_y(0) = -M_{y,I}`, :math:`M_y(L) = +M_{y,J}`,
      :math:`dM_y/dx = -V_z(x)`).
    """

    element_id: int
    length: float
    xi: ndarray
    x: ndarray
    N: ndarray
    Vy: ndarray
    Vz: ndarray
    T: ndarray
    Mz: ndarray
    My: ndarray


def _integrate_trapezoid_array(
    x: ndarray, a: float, b: float, wa: float, wb: float,
) -> tuple[ndarray, ndarray]:
    """Return ``(dF(x), dM(x))`` at stations ``x`` for a partial trapezoid on ``[a, b]``."""
    c = b - a
    dF = np.zeros_like(x, dtype=np.float64)
    dM = np.zeros_like(x, dtype=np.float64)
    if c <= 1e-14 or (abs(wa) + abs(wb) <= 1e-15):
        return dF, dM

    k = (wb - wa) / c
    inside = (x > a) & (x < b)
    xi = x[inside] - a
    dF[inside] = wa * xi + 0.5 * k * (xi ** 2)
    dM[inside] = 0.5 * wa * (xi ** 2) + (1.0 / 6.0) * k * (xi ** 3)

    past = x >= b
    f_seg = 0.5 * (wa + wb) * c
    ma_seg = (c ** 2) * (wa + 2.0 * wb) / 6.0
    dF[past] = f_seg
    dM[past] = f_seg * (x[past] - a) - ma_seg
    return dF, dM


def _dedup_sorted_points(pts: Sequence[float], length: float) -> list[float]:
    """Sort ``pts`` in ``[0, length]`` and merge points closer than ``1e-12 * length``."""
    tol = 1e-12 * max(1.0, length)
    clipped = sorted(min(max(float(p), 0.0), length) for p in pts)
    out: list[float] = [0.0]
    for p in clipped:
        if p - out[-1] > tol:
            out.append(p)
    if length - out[-1] <= tol:
        out[-1] = length
    else:
        out.append(length)
    return out


def _quadratic_roots_on_interval(
    v0: float, w0: float, slope: float, h: float,
) -> list[float]:
    """Find all isolated real roots ``tau in (0, h)`` of ``v0 + w0*tau + 0.5*slope*tau^2 = 0``."""
    if h <= 0.0:
        return []
    tol_tau = 1e-11 * h
    scale = max(abs(v0), abs(w0) * h, 0.5 * abs(slope) * (h ** 2), 1.0)
    a_coef = 0.5 * slope
    b_coef = w0
    c_coef = v0

    candidates: list[float] = []
    if abs(a_coef) * (h ** 2) <= 1e-14 * scale:
        # Linear shear V(tau) = v0 + w0 * tau
        if abs(b_coef) * h > 1e-14 * scale:
            candidates.append(-c_coef / b_coef)
    else:
        disc = b_coef * b_coef - 4.0 * a_coef * c_coef
        if disc >= -1e-15 * (scale ** 2):
            sqrt_disc = math.sqrt(max(0.0, disc))
            if b_coef >= 0.0:
                q = -0.5 * (b_coef + sqrt_disc)
            else:
                q = -0.5 * (b_coef - sqrt_disc)
            if abs(a_coef) > 0.0:
                candidates.append(q / a_coef)
            if abs(q) > 0.0:
                candidates.append(c_coef / q)

    roots: list[float] = []
    for tau in candidates:
        if not (tol_tau < tau < h - tol_tau):
            continue
        # One Newton-Raphson polish step for full IEEE-754 double precision
        val = c_coef + tau * (b_coef + a_coef * tau)
        deriv = b_coef + 2.0 * a_coef * tau
        if abs(deriv) * h > 1e-14 * scale:
            tau_polished = tau - val / deriv
            if tol_tau < tau_polished < h - tol_tau:
                tau = tau_polished
        if all(abs(tau - r) > tol_tau for r in roots):
            roots.append(tau)
    return sorted(roots)


def reconstruct_beam_element_diagram(
    element_id: int,
    length: float,
    local_force: Sequence[float] | ndarray,
    loads: Sequence[BeamSpanSegmentLoad] = (),
    *,
    n_sub: int = 9,
) -> BeamSpanDiagram:
    """Reconstruct exact piecewise-cubic 2D/3D internal force diagrams on one beam.

    Parameters
    ----------
    element_id
        Element identifier (FEM element ID or OpenSees element tag).
    length
        Physical element length :math:`L > 0`.
    local_force
        OpenSees ``localForce`` end-node vector of length 6 (2D:
        ``[N_i, Vy_i, Mz_i, N_j, Vy_j, Mz_j]``) or 12 (3D:
        ``[N_i, Vy_i, Vz_i, T_i, My_i, Mz_i, N_j, Vy_j, Vz_j, T_j, My_j, Mz_j]``).
    loads
        Active partial trapezoidal distributed loads along the element.
    n_sub
        Number of uniform subdivisions inside each critical sub-interval
        (must be ``>= 1``).
    """
    L = float(length)
    if not math.isfinite(L) or L <= 0.0:
        raise ValueError(
            f"reconstruct_beam_element_diagram: length must be finite and > 0, got {length!r}."
        )
    if int(n_sub) < 1:
        raise ValueError(
            f"reconstruct_beam_element_diagram: n_sub must be >= 1, got {n_sub!r}."
        )
    n_sub_int = int(n_sub)

    lf = np.asarray(local_force, dtype=np.float64).reshape(-1)
    if lf.size == 6:
        N_I = float(lf[0])
        Vy_I = float(lf[1])
        Mz_I = float(lf[2])
        Vz_I = 0.0
        T_I = 0.0
        My_I = 0.0
    elif lf.size == 12:
        N_I = float(lf[0])
        Vy_I = float(lf[1])
        Vz_I = float(lf[2])
        T_I = float(lf[3])
        My_I = float(lf[4])
        Mz_I = float(lf[5])
    else:
        raise ValueError(
            "reconstruct_beam_element_diagram: local_force must have length 6 (2D) "
            f"or 12 (3D), got {lf.size}."
        )

    # Validate and filter active segments with non-zero length
    active_loads: list[tuple[float, float, BeamSpanSegmentLoad]] = []
    raw_breakpoints: list[float] = [0.0, L]
    for seg in loads:
        a_norm = float(seg.aL)
        b_norm = float(seg.bL)
        if (
            not math.isfinite(a_norm)
            or not math.isfinite(b_norm)
            or a_norm < -1e-9
            or b_norm > 1.0 + 1e-9
            or b_norm < a_norm - 1e-9
        ):
            raise ValueError(
                "reconstruct_beam_element_diagram: segment [aL, bL] must satisfy "
                f"0.0 <= aL <= bL <= 1.0, got aL={seg.aL!r}, bL={seg.bL!r}."
            )
        a = min(max(a_norm, 0.0), 1.0) * L
        b = min(max(b_norm, 0.0), 1.0) * L
        if b - a > 1e-14 * L:
            active_loads.append((a, b, seg))
            raw_breakpoints.append(a)
            raw_breakpoints.append(b)

    crit_points = _dedup_sorted_points(raw_breakpoints, L)

    # Evaluate shear at initial critical points to locate interior extrema (Vy=0, Vz=0)
    crit_arr = np.asarray(crit_points, dtype=np.float64)
    vy_crit = np.full_like(crit_arr, Vy_I)
    vz_crit = np.full_like(crit_arr, Vz_I)
    for a, b, seg in active_loads:
        df_y, _ = _integrate_trapezoid_array(crit_arr, a, b, float(seg.wya), float(seg.wyb))
        df_z, _ = _integrate_trapezoid_array(crit_arr, a, b, float(seg.wza), float(seg.wzb))
        vy_crit += df_y
        vz_crit += df_z

    tol_seg = 1e-12 * L
    extra_roots: list[float] = []
    for m in range(len(crit_points) - 1):
        xm = crit_points[m]
        xp = crit_points[m + 1]
        h = xp - xm
        if h <= tol_seg:
            continue
        wy0 = 0.0
        sy = 0.0
        wz0 = 0.0
        sz = 0.0
        for a, b, seg in active_loads:
            if a <= xm + tol_seg and b >= xp - tol_seg:
                c = b - a
                ky = (float(seg.wyb) - float(seg.wya)) / c
                kz = (float(seg.wzb) - float(seg.wza)) / c
                wy0 += float(seg.wya) + ky * (xm - a)
                sy += ky
                wz0 += float(seg.wza) + kz * (xm - a)
                sz += kz
        for tau in _quadratic_roots_on_interval(float(vy_crit[m]), wy0, sy, h):
            extra_roots.append(xm + tau)
        for tau in _quadratic_roots_on_interval(float(vz_crit[m]), wz0, sz, h):
            extra_roots.append(xm + tau)

    if extra_roots:
        crit_points = _dedup_sorted_points([*crit_points, *extra_roots], L)

    # Subdivide each sub-interval into n_sub uniform steps while preserving exact critical points
    station_chunks: list[ndarray] = []
    n_intervals = len(crit_points) - 1
    for idx in range(n_intervals):
        c0 = crit_points[idx]
        c1 = crit_points[idx + 1]
        pts = np.linspace(c0, c1, n_sub_int + 1, dtype=np.float64)
        pts[0] = c0
        pts[-1] = c1
        if idx < n_intervals - 1:
            station_chunks.append(pts[:-1])
        else:
            station_chunks.append(pts)

    x = np.concatenate(station_chunks) if station_chunks else np.array([0.0, L], dtype=np.float64)
    xi = x / L
    xi[0] = 0.0
    xi[-1] = 1.0

    N = np.full_like(x, -N_I)
    Vy = np.full_like(x, Vy_I)
    Vz = np.full_like(x, Vz_I)
    T = np.full_like(x, -T_I)
    Mz = -Mz_I + Vy_I * x
    My = -My_I - Vz_I * x

    for a, b, seg in active_loads:
        dFx, _ = _integrate_trapezoid_array(x, a, b, float(seg.wxa), float(seg.wxb))
        dFy, dMy_z = _integrate_trapezoid_array(x, a, b, float(seg.wya), float(seg.wyb))
        dFz, dMz_y = _integrate_trapezoid_array(x, a, b, float(seg.wza), float(seg.wzb))

        N -= dFx
        Vy += dFy
        Vz += dFz
        Mz += dMy_z
        My -= dMz_y

    return BeamSpanDiagram(
        element_id=int(element_id),
        length=L,
        xi=xi,
        x=x,
        N=N,
        Vy=Vy,
        Vz=Vz,
        T=T,
        Mz=Mz,
        My=My,
    )


def parse_membrane_subsegment_loads(
    raw: Sequence[float] | ndarray,
) -> dict[int, list[BeamSpanSegmentLoad]]:
    """Parse a 9-column ``LadrunoMembrane`` ``"subSegmentLoads"`` response vector.

    Each 9-value row is ``[frameTag, aL, bL, wya, wyb, wza, wzb, wxa, wxb]``.
    Returns a dictionary mapping ``int(frameTag)`` to its list of
    :class:`BeamSpanSegmentLoad` instances.
    """
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {}
    if arr.size % 9 != 0:
        raise ValueError(
            "parse_membrane_subsegment_loads: expected flat length divisible by 9 "
            f"(got {arr.size})."
        )
    n_sub = arr.size // 9
    by_beam: dict[int, list[BeamSpanSegmentLoad]] = {}
    for s in range(n_sub):
        base = 9 * s
        frame_tag = int(round(float(arr[base + 0])))
        seg = BeamSpanSegmentLoad(
            aL=float(arr[base + 1]),
            bL=float(arr[base + 2]),
            wya=float(arr[base + 3]),
            wyb=float(arr[base + 4]),
            wza=float(arr[base + 5]),
            wzb=float(arr[base + 6]),
            wxa=float(arr[base + 7]),
            wxb=float(arr[base + 8]),
        )
        by_beam.setdefault(frame_tag, []).append(seg)
    return by_beam


def parse_ele_load_beam_uniform(
    args: Sequence[int | float | str],
    ndm: int = 3,
    scale: float = 1.0,
) -> tuple[tuple[int, ...], BeamSpanSegmentLoad] | None:
    """Parse an OpenSees ``eleLoad`` argument sequence for ``-beamUniform``.

    Supports ``-ele *tags`` and ``-range start end`` / ``-eleRange start end``,
    followed by ``-type -beamUniform *vals`` with:

    * ``ndm == 2``: 2 floats ``(wy, wx)``, 4 floats ``(wy, wx, aL, bL)``, or
      6 floats ``(wya, wxa, aL, bL, wyb, wxb)``.
    * ``ndm == 3``: 2 floats ``(wy, wz)``, 3 floats ``(wy, wz, wx)``,
      5 floats ``(wy, wz, wx, aL, bL)``, or
      8 floats ``(wya, wza, wxa, aL, bL, wyb, wzb, wxb)``.

    Returns ``None`` if the ``eleLoad`` is not a ``-beamUniform`` directive
    (e.g. ``-surfaceLoad`` or ``-selfWeight``).
    """
    if ndm not in (2, 3):
        raise ValueError(f"parse_ele_load_beam_uniform: ndm must be 2 or 3, got {ndm!r}.")
    tokens = list(args)
    if not tokens:
        return None

    try:
        type_idx = tokens.index("-type")
    except ValueError:
        return None

    if type_idx + 1 >= len(tokens):
        raise ValueError(f"parse_ele_load_beam_uniform: missing token after '-type' in {args!r}.")

    load_type = str(tokens[type_idx + 1])
    if load_type != "-beamUniform":
        return None

    # Parse target element tags before '-type'
    head = tokens[:type_idx]
    ele_tags: list[int] = []
    i = 0
    while i < len(head):
        tok = head[i]
        if tok == "-ele":
            i += 1
            while i < len(head) and not (isinstance(head[i], str) and str(head[i]).startswith("-")):
                ele_tags.append(int(float(head[i])))
                i += 1
        elif tok in ("-range", "-eleRange"):
            if i + 2 >= len(head):
                raise ValueError(
                    f"parse_ele_load_beam_uniform: {tok} requires start and end tags in {args!r}."
                )
            start_tag = int(float(head[i + 1]))
            end_tag = int(float(head[i + 2]))
            ele_tags.extend(range(start_tag, end_tag + 1))
            i += 3
        else:
            ele_tags.append(int(float(tok)))
            i += 1

    if not ele_tags:
        raise ValueError(
            f"parse_ele_load_beam_uniform: no target element tags found in {args!r}."
        )

    vals = [float(v) for v in tokens[type_idx + 2 :]]
    s = float(scale)

    if ndm == 2:
        if len(vals) == 2:
            wy, wx = vals
            seg = BeamSpanSegmentLoad(
                aL=0.0, bL=1.0,
                wya=wy * s, wyb=wy * s,
                wza=0.0, wzb=0.0,
                wxa=wx * s, wxb=wx * s,
            )
        elif len(vals) == 4:
            wy, wx, a_l, b_l = vals
            seg = BeamSpanSegmentLoad(
                aL=a_l, bL=b_l,
                wya=wy * s, wyb=wy * s,
                wza=0.0, wzb=0.0,
                wxa=wx * s, wxb=wx * s,
            )
        elif len(vals) == 6:
            wya, wxa, a_l, b_l, wyb, wxb = vals
            seg = BeamSpanSegmentLoad(
                aL=a_l, bL=b_l,
                wya=wya * s, wyb=wyb * s,
                wza=0.0, wzb=0.0,
                wxa=wxa * s, wxb=wxb * s,
            )
        else:
            raise ValueError(
                "parse_ele_load_beam_uniform: expected 2, 4, or 6 float values for "
                f"2D -beamUniform, got {len(vals)} in {args!r}."
            )
    else:
        if len(vals) == 2:
            wy, wz = vals
            seg = BeamSpanSegmentLoad(
                aL=0.0, bL=1.0,
                wya=wy * s, wyb=wy * s,
                wza=wz * s, wzb=wz * s,
                wxa=0.0, wxb=0.0,
            )
        elif len(vals) == 3:
            wy, wz, wx = vals
            seg = BeamSpanSegmentLoad(
                aL=0.0, bL=1.0,
                wya=wy * s, wyb=wy * s,
                wza=wz * s, wzb=wz * s,
                wxa=wx * s, wxb=wx * s,
            )
        elif len(vals) == 5:
            wy, wz, wx, a_l, b_l = vals
            seg = BeamSpanSegmentLoad(
                aL=a_l, bL=b_l,
                wya=wy * s, wyb=wy * s,
                wza=wz * s, wzb=wz * s,
                wxa=wx * s, wxb=wx * s,
            )
        elif len(vals) == 8:
            wya, wza, wxa, a_l, b_l, wyb, wzb, wxb = vals
            seg = BeamSpanSegmentLoad(
                aL=a_l, bL=b_l,
                wya=wya * s, wyb=wyb * s,
                wza=wza * s, wzb=wzb * s,
                wxa=wxa * s, wxb=wxb * s,
            )
        else:
            raise ValueError(
                "parse_ele_load_beam_uniform: expected 2, 3, 5, or 8 float values for "
                f"3D -beamUniform, got {len(vals)} in {args!r}."
            )

    if not (0.0 - 1e-9 <= seg.aL < seg.bL <= 1.0 + 1e-9):
        raise ValueError(
            "parse_ele_load_beam_uniform: span [aL, bL] must satisfy "
            f"0.0 <= aL < bL <= 1.0, got aL={seg.aL!r}, bL={seg.bL!r}."
        )

    return tuple(ele_tags), seg


def _select_time_row(arr: ndarray, step: int) -> ndarray:
    """Select time row ``step`` when ``arr`` has multiple steps, else row 0."""
    if arr.shape[0] == 1:
        return arr[0]
    return arr[step]


def _extract_beam_local_forces(
    results: "Results", step: int,
) -> dict[int, ndarray]:
    """Extract per-element 6- or 12-component ``localForce`` vectors at ``step``."""
    out: dict[int, ndarray] = {}

    # 1. Check results.elements
    elems = getattr(results, "elements", None)
    if elems is not None:
        try:
            avail = set(elems.available_components())
        except Exception:
            avail = set()

        if "localForce" in avail:
            slab = elems.get(component="localForce", time=step)
            vals = np.asarray(slab.values, dtype=np.float64)
            if vals.ndim == 2:
                vals = vals[np.newaxis, :, :]
            if vals.size > 0 and vals.shape[2] in (6, 12):
                step_vals = _select_time_row(vals, step)
                for idx, eid in enumerate(np.asarray(slab.element_ids, dtype=np.int64)):
                    out[int(eid)] = step_vals[idx].copy()
                if out:
                    return out

        if set(_BEAM_NODAL_3D_LOCAL).issubset(avail):
            slabs_3d = [elems.get(component=c, time=step) for c in _BEAM_NODAL_3D_LOCAL]
            eids = np.asarray(slabs_3d[0].element_ids, dtype=np.int64)
            if eids.size > 0:
                comp_arrays = [
                    _select_time_row(np.asarray(s.values, dtype=np.float64), step)
                    for s in slabs_3d
                ]
                for idx, eid in enumerate(eids):
                    vec = np.array(
                        [
                            comp_arrays[0][idx, 0],
                            comp_arrays[1][idx, 0],
                            comp_arrays[2][idx, 0],
                            comp_arrays[3][idx, 0],
                            comp_arrays[4][idx, 0],
                            comp_arrays[5][idx, 0],
                            comp_arrays[0][idx, 1],
                            comp_arrays[1][idx, 1],
                            comp_arrays[2][idx, 1],
                            comp_arrays[3][idx, 1],
                            comp_arrays[4][idx, 1],
                            comp_arrays[5][idx, 1],
                        ],
                        dtype=np.float64,
                    )
                    out[int(eid)] = vec
                return out

        if set(_BEAM_NODAL_2D_LOCAL).issubset(avail):
            slabs_2d = [elems.get(component=c, time=step) for c in _BEAM_NODAL_2D_LOCAL]
            eids = np.asarray(slabs_2d[0].element_ids, dtype=np.int64)
            if eids.size > 0:
                comp_arrays = [
                    _select_time_row(np.asarray(s.values, dtype=np.float64), step)
                    for s in slabs_2d
                ]
                for idx, eid in enumerate(eids):
                    vec = np.array(
                        [
                            comp_arrays[0][idx, 0],
                            comp_arrays[1][idx, 0],
                            comp_arrays[2][idx, 0],
                            comp_arrays[0][idx, 1],
                            comp_arrays[1][idx, 1],
                            comp_arrays[2][idx, 1],
                        ],
                        dtype=np.float64,
                    )
                    out[int(eid)] = vec
                return out

        # 2. Fallback: check results.elements.line_stations
        line_st = getattr(elems, "line_stations", None)
        if line_st is not None:
            try:
                ls_avail = set(line_st.available_components())
            except Exception:
                ls_avail = set()
            req_3d = {
                "axial_force", "shear_y", "shear_z",
                "torsion", "bending_moment_y", "bending_moment_z",
            }
            req_2d = {"axial_force", "shear_y", "bending_moment_z"}
            if req_3d.issubset(ls_avail):
                s_n = line_st.get(component="axial_force", time=step)
                s_vy = line_st.get(component="shear_y", time=step)
                s_vz = line_st.get(component="shear_z", time=step)
                s_t = line_st.get(component="torsion", time=step)
                s_my = line_st.get(component="bending_moment_y", time=step)
                s_mz = line_st.get(component="bending_moment_z", time=step)
                v_n = _select_time_row(np.asarray(s_n.values, dtype=np.float64), step)
                v_vy = _select_time_row(np.asarray(s_vy.values, dtype=np.float64), step)
                v_vz = _select_time_row(np.asarray(s_vz.values, dtype=np.float64), step)
                v_t = _select_time_row(np.asarray(s_t.values, dtype=np.float64), step)
                v_my = _select_time_row(np.asarray(s_my.values, dtype=np.float64), step)
                v_mz = _select_time_row(np.asarray(s_mz.values, dtype=np.float64), step)
                eidx = np.asarray(s_n.element_index, dtype=np.int64)
                xi_nat = np.asarray(s_n.station_natural_coord, dtype=np.float64)
                for eid in np.unique(eidx):
                    mask = np.where(eidx == eid)[0]
                    order = np.argsort(xi_nat[mask])
                    i0, i1 = int(mask[order[0]]), int(mask[order[-1]])
                    out[int(eid)] = np.array(
                        [
                            -float(v_n[i0]),
                            float(v_vy[i0]),
                            float(v_vz[i0]),
                            -float(v_t[i0]),
                            -float(v_my[i0]),
                            -float(v_mz[i0]),
                            float(v_n[i1]),
                            -float(v_vy[i1]),
                            -float(v_vz[i1]),
                            float(v_t[i1]),
                            float(v_my[i1]),
                            float(v_mz[i1]),
                        ],
                        dtype=np.float64,
                    )
                if out:
                    return out
            elif req_2d.issubset(ls_avail):
                s_n = line_st.get(component="axial_force", time=step)
                s_vy = line_st.get(component="shear_y", time=step)
                s_mz = line_st.get(component="bending_moment_z", time=step)
                v_n = _select_time_row(np.asarray(s_n.values, dtype=np.float64), step)
                v_vy = _select_time_row(np.asarray(s_vy.values, dtype=np.float64), step)
                v_mz = _select_time_row(np.asarray(s_mz.values, dtype=np.float64), step)
                eidx = np.asarray(s_n.element_index, dtype=np.int64)
                xi_nat = np.asarray(s_n.station_natural_coord, dtype=np.float64)
                for eid in np.unique(eidx):
                    mask = np.where(eidx == eid)[0]
                    order = np.argsort(xi_nat[mask])
                    i0, i1 = int(mask[order[0]]), int(mask[order[-1]])
                    out[int(eid)] = np.array(
                        [
                            -float(v_n[i0]),
                            float(v_vy[i0]),
                            -float(v_mz[i0]),
                            float(v_n[i1]),
                            -float(v_vy[i1]),
                            float(v_mz[i1]),
                        ],
                        dtype=np.float64,
                    )
                if out:
                    return out

    return out


def _call_or_attr(obj: Any, name: str, default: Any = ()) -> Any:
    """Return ``getattr(obj, name)()`` when callable, else the attribute value."""
    val = getattr(obj, name, default)
    return val() if callable(val) else val


def _resolve_element_lengths_and_tag_maps(
    results: "Results",
    model: "OpenSeesModel | None",
) -> tuple[dict[int, float], dict[int, float], dict[int, int], dict[int, int]]:
    """Return ``(fem_lengths, ops_lengths, fem_to_ops, ops_to_fem)`` for line elements."""
    fem = getattr(results, "fem", None)
    if fem is None and model is not None:
        fem = getattr(model, "fem", None)

    node_coords: dict[int, ndarray] = {}
    fem_conn: dict[int, tuple[int, int]] = {}
    if fem is not None:
        n_ids = np.asarray(fem.nodes.ids, dtype=np.int64).reshape(-1)
        n_xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
        for idx, nid in enumerate(n_ids):
            node_coords[int(nid)] = n_xyz[idx]
        elem_groups: Any = getattr(fem.elements, "groups", None)
        if elem_groups is None:
            elem_groups = getattr(fem.elements, "_groups", None)
        group_iter = elem_groups.values() if isinstance(elem_groups, Mapping) else fem.elements
        for grp in group_iter:
            g_ids = np.asarray(grp.ids, dtype=np.int64).reshape(-1)
            g_conn = np.asarray(grp.connectivity, dtype=np.int64)
            if g_conn.ndim == 2 and g_conn.shape[1] >= 2:
                for idx, eid in enumerate(g_ids):
                    fem_conn[int(eid)] = (int(g_conn[idx, 0]), int(g_conn[idx, 1]))

    if model is not None and not node_coords:
        for nrec in _call_or_attr(model, "nodes", ()):
            node_coords[int(nrec.tag)] = np.asarray(nrec.coords, dtype=np.float64)

    fem_lengths: dict[int, float] = {}
    for feid, (n_i, n_j) in fem_conn.items():
        if n_i in node_coords and n_j in node_coords:
            fem_lengths[feid] = float(np.linalg.norm(node_coords[n_j] - node_coords[n_i]))

    ops_lengths: dict[int, float] = {}
    fem_to_ops: dict[int, int] = {}
    ops_to_fem: dict[int, int] = {}
    if model is not None:
        for rec in _call_or_attr(model, "elements", ()):
            ops_tag = int(rec.tag)
            fem_eid = int(rec.fem_eid)
            if fem_eid >= 0:
                fem_to_ops[fem_eid] = ops_tag
                ops_to_fem[ops_tag] = fem_eid
                if fem_eid in fem_lengths:
                    ops_lengths[ops_tag] = fem_lengths[fem_eid]
            if len(rec.connectivity) >= 2:
                ni, nj = int(rec.connectivity[0]), int(rec.connectivity[1])
                if ni in node_coords and nj in node_coords:
                    elen = float(np.linalg.norm(node_coords[nj] - node_coords[ni]))
                    ops_lengths[ops_tag] = elen
                    if fem_eid >= 0 and fem_eid not in fem_lengths:
                        fem_lengths[fem_eid] = elen

    return fem_lengths, ops_lengths, fem_to_ops, ops_to_fem


def reconstruct_beam_span_diagrams(
    results: "Results",
    *,
    model: "OpenSeesModel | None" = None,
    subsegment_loads: Mapping[int, Sequence[BeamSpanSegmentLoad]] | Sequence[float] | ndarray | None = None,
    step: int = -1,
    n_sub: int = 9,
) -> dict[int, BeamSpanDiagram]:
    """Reconstruct exact piecewise-cubic span diagrams for all beam elements in ``results``.

    Parameters
    ----------
    results
        The :class:`~apeGmsh.results.Results` container providing beam
        ``localForce`` end vectors.
    model
        Optional :class:`~apeGmsh.opensees.OpenSeesModel` providing element
        connectivity and ``PatternRecord.ele_loads``. Defaults to
        ``results.model`` when bound.
    subsegment_loads
        Optional ``LadrunoMembrane`` sub-segment loads, passed either as a
        pre-parsed mapping ``{frame_tag: [BeamSpanSegmentLoad, ...]}`` or as
        a raw 9-column ``"subSegmentLoads"`` sequence.
    step
        Time step index to evaluate (default ``-1``, the final step).
    n_sub
        Number of uniform subdivisions inside each critical sub-interval.
    """
    resolved_model = model
    if resolved_model is None:
        try:
            resolved_model = results.model
        except Exception:
            resolved_model = None

    ndm = 3
    if resolved_model is not None and getattr(resolved_model, "ndm", None) is not None:
        ndm = int(resolved_model.ndm)  # type: ignore[arg-type]

    local_forces = _extract_beam_local_forces(results, step=step)
    if not local_forces:
        return {}

    fem_lengths, ops_lengths, fem_to_ops, ops_to_fem = _resolve_element_lengths_and_tag_maps(
        results, resolved_model,
    )

    # Determine whether local_forces keys live in FEM element ID space or OpenSees tag space.
    # Results slabs translate element IDs to fem_eid whenever FEMData/model pairing exists.
    use_fem_id_space = bool(fem_lengths) and all(eid in fem_lengths for eid in local_forces)
    lengths_by_eid = fem_lengths if use_fem_id_space else {**fem_lengths, **ops_lengths}

    def _to_result_eid(ops_tag: int) -> int:
        if use_fem_id_space and ops_tag in ops_to_fem:
            return ops_to_fem[ops_tag]
        return ops_tag

    loads_by_eid: dict[int, list[BeamSpanSegmentLoad]] = {}
    if resolved_model is not None:
        patterns: list[Any] = list(_call_or_attr(resolved_model, "patterns", ()))
        for stg in _call_or_attr(resolved_model, "stages", ()):
            patterns.extend(_call_or_attr(stg, "patterns", ()))
        for pat in patterns:
            for erec in getattr(pat, "ele_loads", ()):
                parsed = parse_ele_load_beam_uniform(erec.args, ndm=ndm, scale=1.0)
                if parsed is not None:
                    ele_tags, seg_load = parsed
                    for t in ele_tags:
                        target_eid = _to_result_eid(int(t))
                        loads_by_eid.setdefault(target_eid, []).append(seg_load)

    if subsegment_loads is not None:
        if isinstance(subsegment_loads, Mapping):
            # Pre-parsed mapping may be keyed by result eid (fem_eid) or by OpenSees frameTag
            keys_match_results = all(int(k) in local_forces for k in subsegment_loads)
            for f_tag, seg_list in subsegment_loads.items():
                key_int = int(f_tag)
                target_eid = key_int if keys_match_results else _to_result_eid(key_int)
                loads_by_eid.setdefault(target_eid, []).extend(seg_list)
        else:
            parsed_sub = parse_membrane_subsegment_loads(subsegment_loads)
            for f_tag, seg_list in parsed_sub.items():
                target_eid = _to_result_eid(int(f_tag))
                loads_by_eid.setdefault(target_eid, []).extend(seg_list)

    diagrams: dict[int, BeamSpanDiagram] = {}
    for eid, lf_vec in local_forces.items():
        if eid not in lengths_by_eid:
            continue
        elem_loads = loads_by_eid.get(eid, ())
        diagrams[eid] = reconstruct_beam_element_diagram(
            element_id=eid,
            length=lengths_by_eid[eid],
            local_force=lf_vec,
            loads=elem_loads,
            n_sub=n_sub,
        )

    return diagrams
