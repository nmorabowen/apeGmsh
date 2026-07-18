"""Tests — ADR 0078 follow-up: stress recovery on ``disconnected="sum"``.

The distribution policy under test: ``N``/``Mxx``/``Myy`` use the
**global** plane-sections composite state (common centroid, Steiner
terms); ``Mzz`` distributes to parts ∝ ``GJᵢ/ΣGJ``; ``Vx``/``Vy`` ∝ the
part flexural-rigidity shares (``EIyyᵢ`` / ``EIxxᵢ``).  Gates: global +
per-part equilibrium (Σ per-part actions = the applied action), the
two-rectangle exactness oracle (per-part fields == a standalone section
under the distributed loads, node-matched), the two-disk closed-form
torsion oracle, the Steiner discriminator for stacked parts, and the
per-region ``get(pg=)`` contract.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — never open a window in tests

import numpy as np
import pytest

from apeGmsh.sections import SectionMaterial, SectionProperties
from apeGmsh.sections._fe import block_quadrature


def _mesh(g, *, lc: float, order: int = 2):
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim=2)
    if order > 1:
        g.mesh.generation.set_order(order)
    return g.mesh.queries.get_fem_data(dim=2)


def _rect(g, b, h, *, x0=0.0, y0=0.0, pg=None):
    tag = g.model.geometry.add_rectangle(x0, y0, 0.0, b, h)
    if pg:
        g.physical.add(2, [tag], name=pg)
    return tag


def _integrate(sec, st, *, part: int | None = None):
    """Quadrature integrals of the recovered (flat) nodal fields about
    the global elastic centroid, optionally restricted to one connected
    part: returns ``(∫σ dA, ∫σ·ȳ dA, ∫σ·x̄ dA, ∫τ_zx dA, ∫τ_zy dA,
    ∫(x̄·τ_zy − ȳ·τ_zx) dA)``.  Single-material fixtures only (the flat
    view is exact away from material interfaces)."""
    snap = sec._snapshot
    geo = sec.geometric()
    sig = np.nan_to_num(st.sigma_zz)
    tzx = np.nan_to_num(st.tau_zx)
    tzy = np.nan_to_num(st.tau_zy)
    out = np.zeros(6)
    for b in snap.blocks:
        keep = np.ones(len(b.conn), dtype=bool)
        if part is not None:
            keep = snap.node_component[b.conn[:, 0]] == part
        if not keep.any():
            continue
        conn = b.conn[keep]
        q = block_quadrature(
            type(b)(
                code=b.code, type_name=b.type_name,
                n_corners=b.n_corners, eids=b.eids[keep],
                conn=conn, mat_idx=b.mat_idx[keep],
            ),
            snap.coords,
            centroid=(geo.cx, geo.cy),
        )
        s_ip = np.einsum("ia,ea->ei", q.N, sig[conn])
        zx_ip = np.einsum("ia,ea->ei", q.N, tzx[conn])
        zy_ip = np.einsum("ia,ea->ei", q.N, tzy[conn])
        w = q.wdetj
        out += [
            float((w * s_ip).sum()),
            float((w * s_ip * q.y).sum()),
            float((w * s_ip * q.x).sum()),
            float((w * zx_ip).sum()),
            float((w * zy_ip).sum()),
            float((w * (q.x * zy_ip - q.y * zx_ip)).sum()),
        ]
    return out


# ─────────────────────────────────────────────────────────────────────
# equilibrium: Σ per-part actions = applied action (asymmetric parts)
# ─────────────────────────────────────────────────────────────────────

def _asymmetric_twin(g, *, lc=0.07):
    """Two dissimilar rectangles (different A, I, GJ; same centroid
    height so Mxx has no Steiner ambiguity in the per-part checks)."""
    _rect(g, 1.0, 1.0)
    _rect(g, 0.6, 1.4, x0=3.0, y0=-0.2)   # centroid height 0.5, same as A
    fem = _mesh(g, lc=lc)
    return SectionProperties(fem, name="twin", disconnected="sum")


def test_sum_sigma_equilibrium(g):
    sec = _asymmetric_twin(g)
    N, Mxx, Myy = 100.0, 50.0, 20.0
    st = sec.stress(N=N, Mxx=Mxx, Myy=Myy)
    iN, iMxx, iMyy, iVx, iVy, iT = _integrate(sec, st)
    assert iN == pytest.approx(N, rel=1e-9)
    assert iMxx == pytest.approx(Mxx, rel=1e-9)
    assert iMyy == pytest.approx(Myy, rel=1e-9)
    # plane-sections σ produces no shear
    assert abs(iVx) < 1e-9 * N and abs(iVy) < 1e-9 * N


def test_sum_torsion_equilibrium_and_gj_shares(g):
    sec = _asymmetric_twin(g)
    Mzz = 20.0
    st = sec.stress(Mzz=Mzz)
    *_, iVx, iVy, iT = _integrate(sec, st)
    assert iT == pytest.approx(Mzz, rel=5e-3)
    # pure torsion: zero force resultant
    assert abs(iVx) < 1e-3 * Mzz and abs(iVy) < 1e-3 * Mzz
    # per-part torque = Mzz·GJᵢ/ΣGJ
    sols = sec._warp_solutions
    GJ = sum(s.GJ for s in sols)
    for c, s in enumerate(sols):
        *_, iT_c = _integrate(sec, st, part=c)
        assert iT_c == pytest.approx(Mzz * s.GJ / GJ, rel=5e-3)


def test_sum_shear_equilibrium_and_flexural_shares(g):
    sec = _asymmetric_twin(g)
    Vx, Vy = 10.0, 30.0
    st = sec.stress(Vx=Vx, Vy=Vy)
    iN, iMxx, iMyy, iVx, iVy, _ = _integrate(sec, st)
    assert iVx == pytest.approx(Vx, rel=5e-3)
    assert iVy == pytest.approx(Vy, rel=5e-3)
    # per-part resultants follow the flexural-rigidity shares
    sols = sec._warp_solutions
    EIxx = sum(s.EIxx for s in sols)
    EIyy = sum(s.EIyy for s in sols)
    for c, s in enumerate(sols):
        *_, iVx_c, iVy_c, _t = _integrate(sec, st, part=c)
        assert iVx_c == pytest.approx(Vx * s.EIyy / EIyy, rel=5e-3)
        assert iVy_c == pytest.approx(Vy * s.EIxx / EIxx, rel=5e-3)


# ─────────────────────────────────────────────────────────────────────
# two-rectangle exactness oracle: per-part fields == a standalone
# section under the distributed loads (node-matched)
# ─────────────────────────────────────────────────────────────────────

def _own_session(model_name, build):
    """Run ``build(g)`` in a private session; the analyzer snapshots at
    construction, so it survives ``end()``."""
    from apeGmsh import apeGmsh

    s = apeGmsh(model_name=model_name, verbose=False)
    s.begin()
    try:
        return build(s)
    finally:
        s.end()


def test_two_rect_exactness_vs_standalone():
    """Identical twin rectangles under N/Mxx/Vx/Vy/Mzz: the left part's
    recovered fields equal a standalone single-rectangle analyzer loaded
    with the distributed actions (Myy excluded — its global Steiner
    state is deliberately NOT a per-part statement)."""
    lc = 0.1

    def build_twin(s):
        _rect(s, 1.0, 2.0)
        _rect(s, 1.0, 2.0, x0=3.0)
        fem = _mesh(s, lc=lc)
        return SectionProperties(fem, name="twin", disconnected="sum")

    def build_single(s):
        _rect(s, 1.0, 2.0)
        fem = _mesh(s, lc=lc)
        return SectionProperties(fem, name="single")

    twin = _own_session("twin_oracle", build_twin)
    single = _own_session("single_oracle", build_single)

    twin.warping()
    sols = twin._warp_solutions
    GJ = sum(s.GJ for s in sols)
    EIxx = sum(s.EIxx for s in sols)
    EIyy = sum(s.EIyy for s in sols)
    # identify the left part (centroid at x=0.5)
    left = min(range(len(sols)), key=lambda c: sols[c].cx)
    sL = sols[left]

    N, Mxx, Vx, Vy, Mzz = 100.0, 40.0, 10.0, 30.0, 20.0
    st_t = twin.stress(N=N, Mxx=Mxx, Vx=Vx, Vy=Vy, Mzz=Mzz)
    st_s = single.stress(
        N=N * sL.EA / twin.geometric().EA,
        Mxx=Mxx * sL.EIxx / EIxx,
        Vx=Vx * sL.EIyy / EIyy,
        Vy=Vy * sL.EIxx / EIxx,
        Mzz=Mzz * sL.GJ / GJ,
    )

    # node matching: the standalone face occupies the left part's spot,
    # so coordinates must correspond exactly (deterministic meshing)
    tc = twin._snapshot.coords
    scz = single._snapshot.coords
    rows_L = np.flatnonzero(twin._snapshot.node_component == left)
    key_t = np.round(tc[rows_L], 9)
    key_s = np.round(scz, 9)
    order_t = np.lexsort((key_t[:, 1], key_t[:, 0]))
    order_s = np.lexsort((key_s[:, 1], key_s[:, 0]))
    assert len(rows_L) == len(scz)
    assert np.allclose(key_t[order_t], key_s[order_s], atol=1e-9)

    it = rows_L[order_t]
    is_ = order_s
    for comp in ("sigma_zz", "tau_zx", "tau_zy", "von_mises"):
        vt = getattr(st_t, comp)[it]
        vs = getattr(st_s, comp)[is_]
        scale = max(np.nanmax(np.abs(vs)), 1e-30)
        assert np.nanmax(np.abs(vt - vs)) < 1e-9 * scale, comp


def test_two_disk_torsion_closed_form(g):
    """Two disks of different radius under Mzz: boundary τ on each disk
    matches the closed form for its GJ share — τᵢ = Mzzᵢ·rᵢ/Jᵢ with
    Mzzᵢ = Mzz·rᵢ⁴/(r₁⁴+r₂⁴)."""
    r1, r2 = 1.0, 0.6
    geo = g.model.geometry
    for cx, r in ((0.0, r1), (4.0, r2)):
        c = geo.add_circle(cx, 0.0, 0.0, r)
        loop = geo.add_curve_loop([c])
        geo.add_plane_surface([loop])
    fem = _mesh(g, lc=0.06)
    sec = SectionProperties(fem, name="disks", disconnected="sum")
    Mzz = 20.0
    st = sec.stress(Mzz=Mzz)
    coords = sec._snapshot.coords
    share1 = r1**4 / (r1**4 + r2**4)
    for cx, r, share in ((0.0, r1, share1), (4.0, r2, 1.0 - share1)):
        rr = np.hypot(coords[:, 0] - cx, coords[:, 1])
        boundary = (rr > 0.995 * r) & (rr < 1.005 * r)
        J = np.pi * r**4 / 2
        tau_exact = (Mzz * share) * r / J
        assert np.nanmedian(st.tau[boundary]) == pytest.approx(
            tau_exact, rel=5e-3
        )


# ─────────────────────────────────────────────────────────────────────
# Steiner discriminator: σ is the GLOBAL plane-sections state
# ─────────────────────────────────────────────────────────────────────

def test_sum_sigma_uses_global_steiner_state(g):
    """Two stacked strips with a gap under Mxx: σ = M·ȳ/I_comp with the
    common-centroid Steiner I — NOT a per-part M distribution (which
    would flip sign inside each part about its own centroid)."""
    b, h, gap = 1.0, 0.5, 1.0
    _rect(g, b, h)                    # y ∈ [0, 0.5]
    _rect(g, b, h, y0=h + gap)        # y ∈ [1.5, 2.0]
    fem = _mesh(g, lc=0.07)
    sec = SectionProperties(fem, name="stack", disconnected="sum")
    M = 50.0
    st = sec.stress(Mxx=M)
    cy = 1.0                          # symmetry
    I_own = b * h**3 / 12
    d = 0.75                          # part-centroid offset from cy
    I_comp = 2 * (I_own + b * h * d**2)
    coords = sec._snapshot.coords
    for y_level in (0.0, 0.5, 1.5, 2.0):
        nodes = np.isclose(coords[:, 1], y_level)
        sig_exact = M * (y_level - cy) / I_comp
        assert st.sigma_zz[nodes] == pytest.approx(sig_exact, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────
# per-region access + plots on a "sum" section
# ─────────────────────────────────────────────────────────────────────

def test_get_pg_contract_on_sum(g):
    E = 200.0
    _rect(g, 1.0, 1.0, pg="left")
    _rect(g, 1.0, 1.0, x0=3.0, pg="right")
    fem = _mesh(g, lc=0.15)
    sec = SectionProperties(
        fem,
        materials={
            "left": SectionMaterial(E=E, nu=0.3),
            "right": SectionMaterial(E=E, nu=0.3),
        },
        name="twin_pg",
        disconnected="sum",
    )
    N = 42.0
    st = sec.stress(N=N)
    sig_left = st.get("sigma_zz", pg="left")
    coords = sec._snapshot.coords
    on_left = coords[:, 0] < 1.5
    # exact per-region values, NaN outside
    assert np.nanmax(np.abs(sig_left[on_left] - N / 2.0)) < 1e-9 * N
    assert np.all(np.isnan(sig_left[~on_left]))
    with pytest.raises(KeyError, match="unknown material region"):
        st.get("sigma_zz", pg="ghost")


def test_sum_plots_headless(g):
    _rect(g, 1.0, 1.0)
    _rect(g, 1.0, 1.0, x0=3.0)
    fem = _mesh(g, lc=0.2)
    sec = SectionProperties(fem, name="twin", disconnected="sum")
    ax = sec.stress(Vy=1.0, Mzz=1.0).plot("tau")
    assert ax is not None
    # shear_flow overlay now rides the per-part unit fields
    ax2 = sec.plot_warping(shear_flow=True)
    assert "shear flow" in ax2.get_title()
    import matplotlib.pyplot as plt

    plt.close("all")
