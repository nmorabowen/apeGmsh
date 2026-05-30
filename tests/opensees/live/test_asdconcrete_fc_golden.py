"""Golden-parity test: apeGmsh's owned ASDConcrete backbone == OpenSees ``-fc``.

ADR 0044 has apeGmsh generate the ASDConcrete backbone in Python (a port of
the native ``-fc`` law builder) and emit it as explicit ``-Te/-Ts/...`` points.
This test closes the loop: it builds the *binary's* ``-fc`` material, reads the
generated hardening law back through an element, and asserts it equals
``ASDConcrete3D.from_fc(...)`` to machine precision.

The element is a unit cube so its characteristic length is ``1.0``; pinning
``lch_ref = 1.0`` makes ``lch_scale == 1`` so the binary does **no** rescale and
``eleResponse`` returns the raw ``-fc``-generated curve (the generator output),
not a regularized one. (Building the same material with a mismatched ``lch_ref``
is how the per-element rescale in ADR 0044 Fact 1 is observed.)

Gated by the ``live`` marker — only runs when ``openseespy`` is importable.
"""
from __future__ import annotations

import math

import pytest

openseespy = pytest.importorskip("openseespy.opensees")

# Deferred until openseespy is confirmed present.
from apeGmsh.opensees.material.nd import ASDConcrete3D  # noqa: E402


_E, _FC, _LCH = 30_000.0, 30.0, 1.0   # N/mm; lch_ref == unit-cube lch -> no rescale


def _binary_fc_backbone(ops) -> dict[str, list[float]]:
    """Build the native ``-fc`` material on a unit brick and read it back.

    The binary exposes total strain (``Te``/``Ce``), nominal stress
    (``Ts``/``Cs``) and effective stress (``Tq``/``Cq``) — but **not** damage
    directly; damage is recovered as ``d = 1 - s/q``.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    cube = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for i, (x, y, z) in enumerate(cube, 1):
        ops.node(i, float(x), float(y), float(z))
    ops.nDMaterial("ASDConcrete3D", 1, _E, 0.2,
                   "-fc", _FC, "-autoRegularization", _LCH)
    ops.element("stdBrick", 1, *range(1, 9), 1)
    return {r: list(ops.eleResponse(1, "material", "1", r))
            for r in ("Te", "Ts", "Tq", "Ce", "Cs", "Cq")}


def _assert_close(name: str, got: list[float], exp: tuple[float, ...]) -> None:
    assert len(got) == len(exp), f"{name}: length {len(got)} != {len(exp)}"
    for i, (g, e) in enumerate(zip(got, exp)):
        assert math.isclose(g, e, rel_tol=1e-9, abs_tol=1e-12), (
            f"{name}[{i}]: binary -fc {g!r} != apeGmsh {e!r}"
        )


@pytest.mark.live
def test_from_fc_matches_native_fc_curve() -> None:
    binary = _binary_fc_backbone(openseespy)

    # apeGmsh's owned generator, same physical inputs, same reference length.
    m = ASDConcrete3D.from_fc(E=_E, v=0.2, fc=_FC, lch_ref=_LCH)

    # Strain + stress backbones must match the binary's -fc curve exactly.
    _assert_close("Te", binary["Te"], m.Te)
    _assert_close("Ts", binary["Ts"], m.Ts)
    _assert_close("Ce", binary["Ce"], m.Ce)
    _assert_close("Cs", binary["Cs"], m.Cs)

    # Damage is not a readable response; recover it as d = 1 - s/q and compare.
    for side, s, q, d_exp in (("T", binary["Ts"], binary["Tq"], m.Td),
                              ("C", binary["Cs"], binary["Cq"], m.Cd)):
        d_bin = [1.0 - si / qi if qi > 1e-12 else 0.0 for si, qi in zip(s, q)]
        for i, (db, de) in enumerate(zip(d_bin, d_exp)):
            assert math.isclose(db, de, rel_tol=1e-7, abs_tol=1e-9), (
                f"{side}d[{i}]: binary {db!r} != apeGmsh {de!r}"
            )


@pytest.mark.live
def test_per_element_rescale_stretches_softening() -> None:
    # ADR 0044 Fact 1: with lch_ref > element lch, the binary regularizes the
    # softening branch upward (gnew = g_input * lch_ref/lch). A unit cube
    # (lch=1) with lch_ref=50 stretches the post-peak strain ~50x while the
    # pre-peak points are unchanged.
    ops = openseespy
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    cube = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for i, (x, y, z) in enumerate(cube, 1):
        ops.node(i, float(x), float(y), float(z))
    ops.nDMaterial("ASDConcrete3D", 1, _E, 0.2,
                   "-fc", _FC, "-autoRegularization", 50.0)
    ops.element("stdBrick", 1, *range(1, 9), 1)
    Te = list(ops.eleResponse(1, "material", "1", "Te"))

    unreg = ASDConcrete3D.from_fc(E=_E, v=0.2, fc=_FC, lch_ref=1.0)
    assert math.isclose(Te[2], unreg.Te[2], rel_tol=1e-9)        # pre-peak: same
    assert Te[3] > 10.0 * unreg.Te[3]                            # post-peak: stretched
