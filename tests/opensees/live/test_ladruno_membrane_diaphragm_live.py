"""Live oracles for ``LadrunoMembrane`` on fork build ``c83ee812e`` (fork WP-204).

A square panel with a node every ``_H`` on its boundary is stretched by a
prescribed displacement on two opposite edges and is free to contract.

* ``f11 = f22 = f12 = f`` scales the whole plane-stress modulus, so every
  reaction scales by ``f`` exactly. Under the uniform stretch the e1
  uniaxial modulus is ``f11 * E`` whatever ``f22`` (and e2 is ``f22 * E``),
  so ``f11 = 0.5`` halves the x reaction and leaves the y one, to 1e-9;
  ``f12`` (shear) leaves both.
* The uniform stretch is a linear field, which a condensed CST mesh must
  reproduce exactly: ``R = E t eps S`` and ``u = eps x``, ``v = -nu eps y``
  at every boundary node, at ``nu = 0`` and ``0.2``. This needs fork build
  ``b6d3e3d2`` (fork WP-208); ``c83ee812e`` read 0.72 to 0.74.
* A centre node in ``diaphragm_nodes=`` gets the same in-plane stiffness as
  the same node in ``support_nodes=`` (equal displacements under a load on it,
  to 1e-9), but no gravity share: its vertical reaction under a
  ``surface_load`` is 0, against a nonzero one as a support.
* The five force responses answer ``eleResponse`` with the documented sizes.
  The mean ``N11`` equals the edge reaction over the edge length, the virtual
  work of any equilibrated field. ``diaphragmForces`` balance, and an element
  recorder writes ``meanMembraneForce``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import live
from tests.opensees.fixtures.fem_stub import FEMStub, _ElementsStub, _NodesStub

pytestmark = [pytest.mark.live, pytest.mark.ladruno_fork]

_A, _B, _E, _NU, _T, _F = 4.0, 2.0, 30e6, 0.2, 0.25, 100.0
_S, _H, _EPS = 2.0, 0.5, 1e-4


def _has_c83_membrane() -> bool:
    """The bound build parses ``-f11`` (refuses ``-f11 0``) on LadrunoMembrane."""
    ops = live.get_ops()
    live.LiveOpsEmitter(wipe=True)
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for i, (x, y) in enumerate(((0, 0), (1, 0), (1, 1)), start=1):
        ops.node(i, float(x), float(y), 0.0)
    try:
        ops.element(
            "LadrunoMembrane", 1, "-nodes", 1, 2, 3, "-stiff", 1.0e6, 0.2,
            0.1, "-f11", 0.0,
        )
        tags = ops.getEleTags()
        refused = 1 not in ([tags] if isinstance(tags, int) else (tags or []))
    except Exception:
        refused = True
    finally:
        live.LiveOpsEmitter(wipe=True)
    return refused


@pytest.fixture(autouse=True)
def _require_c83() -> None:
    if not _has_c83_membrane():
        pytest.skip("fork build predates c83ee812e (no LadrunoMembrane -f11)")


def _fem(centre: bool) -> FEMStub:
    ids = [1, 2, 3, 4]
    coords = [(0.0, 0.0, 0.0), (_A, 0.0, 0.0), (_A, _B, 0.0), (0.0, _B, 0.0)]
    pgs: dict[str, list[int]] = {"Centre": []}
    if centre:
        ids.append(5)
        coords.append((0.5 * _A, 0.5 * _B, 0.0))
        pgs["Centre"] = [5]
    return FEMStub(
        nodes=_NodesStub(ids=ids, coords=coords, node_pgs=pgs),
        elements=_ElementsStub(elem_pgs={}),
    )


def _ring() -> "tuple[FEMStub, list[int], np.ndarray]":
    """A square ``_S x _S`` panel whose boundary carries a node every ``_H``."""
    n = int(round(_S / _H))
    pts: list[tuple[float, float]] = []
    pts += [(i * _H, 0.0) for i in range(n)]
    pts += [(_S, j * _H) for j in range(n)]
    pts += [(_S - i * _H, _S) for i in range(n)]
    pts += [(0.0, _S - j * _H) for j in range(n)]
    ids = list(range(1, len(pts) + 1))
    xy = np.asarray(pts, dtype=float)
    fem = FEMStub(
        nodes=_NodesStub(
            ids=ids, coords=[(x, y, 0.0) for x, y in pts], node_pgs={},
        ),
        elements=_ElementsStub(elem_pgs={}),
    )
    return fem, ids, xy


def _run(ops: apeSees) -> Any:
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    emitter = live.LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    return emitter.ops


def _stretch(
    direction: int, *, out: "Path | None" = None, nu: float = _NU,
    **mods: float,
) -> "dict[str, Any]":
    """Stretch the panel by a uniform strain ``_EPS`` along x (1) or y (2).

    The two edges normal to ``direction`` are displacement-controlled
    (0 and ``_EPS * _S``); the panel is free to contract laterally, held by
    one node. The exact field is linear, which the condensed CST mesh
    reproduces, so the edge reaction is ``f * E * t * _EPS * _S``.
    """
    fem, ids, xy = _ring()
    ops = apeSees(cast(Any, fem))
    ops.model(ndm=3, ndf=6)
    ops.element.LadrunoMembrane(
        nodes=tuple(ids), stiff=(_E, nu, _T), mesh_size=_H, **mods,
    )
    axis = direction - 1
    lo = [n for n, p in zip(ids, xy) if abs(p[axis]) < 1e-12]
    hi = [n for n, p in zip(ids, xy) if abs(p[axis] - _S) < 1e-12]
    anchor = ids[0]  # the (0, 0) corner holds the lateral rigid motion
    for n in ids:
        d = [0, 0, 1, 1, 1, 1]
        if n in lo or n in hi:
            d[axis] = 1
        if n == anchor:
            d[1 - axis] = 1
        ops.fix(nodes=[n], dofs=tuple(d))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for n in hi:
            p.sp(node=n, dof=direction, value=_EPS * _S)
    if out is not None:
        # The membrane's OpenSees tag is the build's, not the spec's: read it
        # off a dry Tcl emit.
        deck = out.with_suffix(".tcl")
        ops.tcl(str(deck))
        line = next(
            ln for ln in deck.read_text().splitlines()
            if ln.startswith("element LadrunoMembrane ")
        )
        ops.recorder.Element(
            file=str(out), response=("meanMembraneForce",),
            elements=(int(line.split()[2]),),
        )
    lv = _run(ops)
    try:
        lv.reactions()
        res: dict[str, Any] = {
            "R": abs(sum(float(lv.nodeReaction(n, direction)) for n in hi)),
            "hi": [ids.index(n) for n in hi],
            "xy": xy,
            "u": np.asarray(
                [[lv.nodeDisp(n, 1), lv.nodeDisp(n, 2)] for n in ids],
                dtype=float,
            ),
        }
        tag = int(np.atleast_1d(lv.getEleTags())[0])
        for tok in ("meanMembraneForce", "membraneForces", "diaphragmForces",
                    "diaphragmVertices", "diaphragmTriangles"):
            res[tok] = np.asarray(lv.eleResponse(tag, tok), dtype=float)
        if out is not None:
            lv.remove("recorders")
    finally:
        live.LiveOpsEmitter(wipe=True)
    return res


def _ratio(direction: int, **mods: float) -> float:
    return _stretch(direction, **mods)["R"] / (_E * _T * _EPS * _S)


def test_uniform_modifier_scales_the_stiffness_exactly() -> None:
    """``f11 = f22 = f12 = f`` scales the whole modulus, so every reaction."""
    for d in (1, 2):
        base = _ratio(d)
        assert _ratio(d, f11=0.5, f22=0.5, f12=0.5) == pytest.approx(
            0.5 * base, rel=1e-9)


@pytest.mark.parametrize("nu", [0.0, 0.2])
def test_f11_and_f22_scale_their_own_axis_only(nu: float) -> None:
    """Under a uniform stretch the e1 uniaxial modulus is
    ``D'11 - D'12**2 / D'22 = f11 * E`` whatever ``f22``, and the e2 one is
    ``f22 * E``: the hand ratio is exact."""
    for direction, mods, want in (
        (1, {"f11": 0.5}, 0.5), (2, {"f11": 0.5}, 1.0),
        (2, {"f22": 0.5}, 0.5), (1, {"f22": 0.5}, 1.0),
        (1, {"f11": 0.5, "f22": 0.25}, 0.5),
        (1, {"f12": 0.3}, 1.0),
    ):
        got = _stretch(direction, nu=nu, **mods)["R"] / (_E * _T * _EPS * _S)
        assert got == pytest.approx(want, rel=1e-9), (direction, mods)


@pytest.mark.parametrize("nu", [0.0, 0.2])
def test_uniform_stretch_is_exact(nu: float) -> None:
    """A displacement-controlled uniform stretch is a linear field, which a
    CST mesh condensed onto its boundary must reproduce exactly: the edge
    reaction is ``E t eps S`` and every boundary node sits on
    ``u = eps x``, ``v = -nu eps y`` (fork WP-208 keeps the slab edge straight
    between loop nodes; build ``c83ee812e`` read 0.72 to 0.74)."""
    res = _stretch(1, nu=nu)
    assert res["R"] / (_E * _T * _EPS * _S) == pytest.approx(1.0, rel=1e-9)
    x, y = res["xy"][:, 0], res["xy"][:, 1]
    scale = _EPS * _S
    np.testing.assert_allclose(res["u"][:, 0], _EPS * x, atol=1e-9 * scale)
    np.testing.assert_allclose(res["u"][:, 1], -nu * _EPS * y, atol=1e-9 * scale)


def _centre_panel(role: str, *, load: str) -> "tuple[np.ndarray, float]":
    """Centre-node in-plane displacement under a centre load, or its gravity
    reaction under a surface load; the node is a support or a diaphragm node."""
    ops = apeSees(cast(Any, _fem(centre=True)))
    ops.model(ndm=3, ndf=6)
    kw: dict[str, Any] = (
        {"support_nodes": "Centre"} if role == "support"
        else {"diaphragm_nodes": "Centre"}
    )
    slab = ops.element.LadrunoMembrane(
        nodes=(1, 2, 3, 4), stiff=(_E, _NU, _T),
        pressure=-10.0 if load == "gravity" else None,
        method="tributary", **kw,
    )
    for n in (1, 2, 3, 4):
        ops.fix(nodes=[n], dofs=(1, 1, 1, 1, 1, 1))
    if load == "gravity":
        ops.fix(nodes=[5], dofs=(0, 0, 1, 1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.surface_load(ele=slab)
    else:
        ops.fix(nodes=[5], dofs=(0, 0, 1, 1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=5, forces=(_F, 0.5 * _F, 0.0, 0.0, 0.0, 0.0))
    lv = _run(ops)
    try:
        lv.reactions()
        u = np.asarray([lv.nodeDisp(5, 1), lv.nodeDisp(5, 2)], dtype=float)
        rz = float(lv.nodeReaction(5, 3))
    finally:
        live.LiveOpsEmitter(wipe=True)
    return u, rz


def test_diaphragm_node_has_support_stiffness_and_no_gravity_share() -> None:
    u_supp, _ = _centre_panel("support", load="lateral")
    u_diaph, _ = _centre_panel("diaphragm", load="lateral")
    assert float(np.abs(u_diaph).max()) > 0.0
    np.testing.assert_allclose(u_diaph, u_supp, rtol=1e-9)
    _, rz_supp = _centre_panel("support", load="gravity")
    _, rz_diaph = _centre_panel("diaphragm", load="gravity")
    assert abs(rz_supp) > 1.0  # the support takes its tributary share
    assert rz_diaph == pytest.approx(0.0, abs=1e-9)


def test_force_responses_and_recorder(tmp_path: Path) -> None:
    out = tmp_path / "mean.out"
    res = _stretch(1, out=out, f11=0.5)
    r = res["R"]
    tri = res["diaphragmTriangles"]
    n_tri = tri.size // 3
    n_vert = res["diaphragmVertices"].size // 3
    assert tri.size == 3 * n_tri and n_tri >= 2
    assert int(tri.min()) >= 0 and int(tri.max()) < n_vert
    forces = res["membraneForces"].reshape(-1, 3)
    assert forces.shape == (n_tri, 3)
    # The field is uniform: N11 = f11 E t eps in every triangle, N22 = N12 = 0.
    n11 = 0.5 * _E * _T * _EPS
    np.testing.assert_allclose(forces[:, 0], n11, rtol=1e-9)
    np.testing.assert_allclose(forces[:, 1:], 0.0, atol=1e-9 * n11)
    # Any equilibrated field: the area integral of N11 is the virtual work of
    # the edge reactions, sum(f_x * x) = R * S, so the mean N11 is R / S.
    mean = res["meanMembraneForce"]
    assert mean.shape == (3,)
    assert mean[0] == pytest.approx(r / _S, rel=1e-9)
    assert mean[1] == pytest.approx(0.0, abs=1e-9 * abs(mean[0]))
    # The slab's in-plane force on its nodes balances, and the x = S edge
    # carries R.
    dforce = res["diaphragmForces"].reshape(-1, 3)
    assert dforce.shape[0] == len(_ring()[1])
    np.testing.assert_allclose(dforce.sum(axis=0), 0.0, atol=1e-9 * abs(r))
    assert abs(dforce[res["hi"], 0].sum()) == pytest.approx(abs(r), rel=1e-9)
    rec = np.atleast_2d(np.loadtxt(out))
    np.testing.assert_allclose(rec[-1], mean, rtol=1e-5, atol=1e-6 * abs(mean[0]))
