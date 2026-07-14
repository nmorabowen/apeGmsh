"""B3 live happy-path — the fork build accepts ``BezierTri6`` and the run
gate passes without a false positive.

On a stock (non-fork) build the gate would *correctly* raise — so this
test probes the build and skips cleanly there. The fork-build success path
(element created, ``getEleTags`` reflects it, the gate caches ``verified``)
is asserted only on a real Ladruno build.
"""
from __future__ import annotations

import pytest

# Gate 1: openseespy present (collect-time skip otherwise).
openseespy = pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402


def _fork_has_bezier_tri6() -> bool:
    """True if the live build knows ``BezierTri6`` (a fork-only element)."""
    ops = openseespy
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    for i, (x, y) in enumerate(
        [(0, 0), (2, 0), (0, 2), (1, 0), (1, 1), (0, 1)], start=1,
    ):
        ops.node(i, float(x), float(y))
    ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
    try:
        ops.element("BezierTri6", 1, 1, 2, 3, 4, 5, 6, 1.0, "PlaneStress", 1)
        return 1 in (ops.getEleTags() or [])
    except Exception:
        return False
    finally:
        ops.wipe()


pytestmark = pytest.mark.skipif(
    not _fork_has_bezier_tri6(),
    reason="live OpenSees build is not the Ladruno fork (no BezierTri6)",
)


def test_run_gate_passes_on_fork_build() -> None:
    e = LiveOpsEmitter()  # wipes
    e.model(ndm=2, ndf=2)
    for i, (x, y) in enumerate(
        [(0, 0), (2, 0), (0, 2), (1, 0), (1, 1), (0, 1)], start=1,
    ):
        e.node(i, float(x), float(y))
    e.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
    # The gate runs here: on the fork build the element builds, so it must
    # NOT raise and must mark itself verified.
    e.element("BezierTri6", 1, 1, 2, 3, 4, 5, 6, 1.0, "PlaneStress", 1)
    assert "BezierTri6" in e._fork_verified_types
    assert 1 in (e._ops.getEleTags() or [])
