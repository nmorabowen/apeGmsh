"""Live: a tet4 embedded tie through :class:`LiveOpsEmitter` binds all
four corners (#1621, program slice B4-a).

One ``FourNodeTetrahedron`` (stock element) with three corners fixed and
a point load on the apex; one node embedded at barycentric coordinates
``(0.1, 0.2, 0.3, 0.4)`` tied with ``embeddedNode`` to the four corners.
The tie has no stiffness of its own beyond the penalty, so the host
responds as if alone and the closed form for the embedded node is the
tet4 interpolation ``u = 0.4 * u_apex`` (the other three weights multiply
zero). With the 4th corner dropped, as the int-argument emitters did
(``OPS_GetString`` + ``std::stoi`` in ``ASDEmbeddedNodeElement.cpp``),
the node is tied to the fixed triangle and reads ``u = 0``.

Runs on stock openseespy and on the fork alike: ``ASDEmbeddedNodeElement``
and ``FourNodeTetrahedron`` are upstream elements. Measured on
main (fork 79e06236): ``u_emb = [0.0, 0.0, 0.0]`` against
``u_apex = [0.15, 0, 0]``; with the fix ``u_emb = [0.06, 0, 0]``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees.emitter.live import (  # noqa: E402
    LiveOpsEmitter, get_ops,
)

_CORNERS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
]
_WEIGHTS = (0.1, 0.2, 0.3, 0.4)
_P = 10.0


@pytest.mark.live
def test_tet4_embedded_node_follows_the_four_corner_interpolation() -> None:
    emitter = LiveOpsEmitter(wipe=True)
    ops = get_ops()  # the module the emitter drives: same domain
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    for tag, xyz in enumerate(_CORNERS, 1):
        ops.node(tag, *xyz)
    point = tuple(
        sum(w * c[k] for w, c in zip(_WEIGHTS, _CORNERS)) for k in range(3)
    )
    ops.node(5, *point)
    ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
    ops.element("FourNodeTetrahedron", 1, 1, 2, 3, 4, 1)
    for tag in (1, 2, 3):
        ops.fix(tag, 1, 1, 1)

    # The call under test: 4 retained nodes, the tet4 host.
    emitter.embeddedNode(2, 5, 1, 2, 3, 4, stiffness=1.0e12)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(4, _P, 0.0, 0.0)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    assert ops.analyze(1) == 0

    u_apex = ops.nodeDisp(4)
    u_emb = ops.nodeDisp(5)
    assert u_apex[0] > 0.0
    expected = [_WEIGHTS[3] * u for u in u_apex]
    # The penalty tie (K=1e12 against E=1e3) satisfies the interpolation
    # to ~1e-7; the 3-corner tie reads 0 here, a 100 % miss.
    assert u_emb == pytest.approx(expected, rel=1e-5, abs=1e-9), (
        f"embedded node {u_emb} vs tet4 interpolation {expected} "
        f"(apex {u_apex}); zero means the 4th corner was dropped"
    )
