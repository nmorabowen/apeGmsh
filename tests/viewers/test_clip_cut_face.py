"""Cut-face seams that outlive the legacy cap renderer (ADR 0083 S3).

The cap renderer (``core/_clip_cut_face.py``) and its panel checkbox
were deleted with the legacy ``ResultsViewer`` (X3-d, #1734). What
stays is pinned here: the contour's own ``cut_face_source`` gate, the
``clip_cut_face`` session flag, and the controller's restore.
"""
from __future__ import annotations

import numpy as np

from apeGmsh.viewers.core._clip_planes import ClipPlaneSetController
from apeGmsh.viewers.scene_ir import (
    CellBlocks,
    ColorSpec,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarField,
)



# =====================================================================
# A solid whose field lives entirely in its interior
# =====================================================================

def _solid_layer(n: int = 7, *, layer_id: str = "solid") -> MeshLayer:
    """A hex box on ``[-1, 1]^3`` carrying a bubble field.

    ``cos(pi x / 2) cos(pi y / 2) cos(pi z / 2)``: exactly zero on every
    one of the six faces, 1.0 at the centre. The exact shape the ADR
    measured the empty box on — from outside there is nothing to see,
    which is the whole reason section planes exist.
    """
    xs = np.linspace(-1.0, 1.0, n)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def idx(i, j, k):
        return (i * n + j) * n + k

    hexes = np.asarray([
        [
            idx(i, j, k), idx(i + 1, j, k),
            idx(i + 1, j + 1, k), idx(i, j + 1, k),
            idx(i, j, k + 1), idx(i + 1, j, k + 1),
            idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1),
        ]
        for i in range(n - 1) for j in range(n - 1) for k in range(n - 1)
    ])
    bubble = (
        np.cos(np.pi * pts[:, 0] / 2.0)
        * np.cos(np.pi * pts[:, 1] / 2.0)
        * np.cos(np.pi * pts[:, 2] / 2.0)
    )
    return MeshLayer(
        layer_id=layer_id,
        points=PointSet(pts),
        cells=CellBlocks({"hexahedron": hexes}),
        fields=(ScalarField("v", bubble, "point"),),
        color=ColorSpec(
            mode="by_array", array_name="v",
            lut=LutSpec(name="jet", vmin=0.0, vmax=1.0),
        ),
    )




def test_a_hidden_contour_offers_no_cut_face_source():
    """``cut_face_source`` is the contour's own gate — a cap must never
    outlive the diagram it mirrors."""
    from apeGmsh.viewers.diagrams._contour import ContourDiagram

    diagram = ContourDiagram.__new__(ContourDiagram)
    diagram._handle = object()
    diagram._layer = _solid_layer()
    diagram._visible = True
    assert diagram.cut_face_source()[2] is diagram._layer.color

    diagram._visible = False
    assert diagram.cut_face_source() is None
    diagram._visible = True
    diagram._handle = None
    assert diagram.cut_face_source() is None


# =====================================================================
# F-SESSION — the flag round-trips, and v9 loads clean
# =====================================================================

def test_the_cut_face_flag_round_trips_through_a_session(tmp_path):
    from apeGmsh.viewers.diagrams._session import (
        SESSION_SCHEMA_VERSION,
        deserialize_session,
        load_session,
        save_session,
    )

    # ``clip_cut_face`` arrived in v10; later bumps (v11 added the
    # per-geometry threshold) do not concern this flag. The exact
    # constant is pinned once, in test_session_persistence.py.
    assert SESSION_SCHEMA_VERSION >= 10

    path = save_session(
        specs=[], results_path=tmp_path / "r.h5", fem_snapshot_id=None,
        target_path=tmp_path / "s.json", clip_cut_face=True,
    )
    assert load_session(path).clip_cut_face is True

    # A v9 session predates the field entirely and must load as "off",
    # which is what a fresh viewer starts at.
    legacy = deserialize_session({
        "schema_version": 9,
        "results_path": str(tmp_path / "r.h5"),
        "diagrams": [],
        "clip_planes": [],
        "clip_apply_cuts": True,
        "clip_show_gizmos": True,
    })
    assert legacy.clip_cut_face is False


def test_the_controller_restores_the_flag():
    controller = ClipPlaneSetController(None)
    controller.restore([], cut_face=True)
    assert controller.cut_face is True
    controller.restore([])
    assert controller.cut_face is False, "a legacy restore reads as off"


# =====================================================================
# S3 adversarial review — D1: a cap must not outlive its contour
# =====================================================================

def test_a_contour_hidden_by_its_composition_backs_no_cap():
    """D1: ``is_visible`` is user *intent*, and the composition gate
    restores it after hiding the actor (``_base.apply_effective_visibility``),
    so gating on it drew a cut face for a contour that was not on
    screen — a polygon floating where an invisible contour used to be.
    The gate has to be read through ``is_effectively_visible``."""
    from apeGmsh.viewers.diagrams._contour import ContourDiagram

    d = ContourDiagram.__new__(ContourDiagram)
    d._visible = True
    d._effective_visible = True
    d._actors = []
    d._backend = None
    d._handle = "HANDLE"
    d._layer = type("L", (), {"color": "COLOR"})()
    d._layer_id = lambda: "layer-1"

    assert d.cut_face_source() is not None, "a visible contour backs a cap"

    # Parked in an inactive composition: actor hidden, intent preserved.
    ContourDiagram.apply_effective_visibility(d, False)
    assert d.is_visible is True, "user intent must survive the gate"
    assert d.is_effectively_visible is False
    assert d.cut_face_source() is None, "hidden contour must back no cap"

    # Back in the active composition.
    ContourDiagram.apply_effective_visibility(d, True)
    assert d.cut_face_source() is not None

    # And the user's own hide still works, gate or no gate.
    ContourDiagram.set_visible(d, False)
    assert d.cut_face_source() is None
