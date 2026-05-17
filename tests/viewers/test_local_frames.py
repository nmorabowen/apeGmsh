"""``LocalFrame`` / ``iter_local_frames`` — the reusable per-element
orientation seam (drives the local-axis overlay today, frame section
extrusion later), plus the ``ViewerElements`` vecxz plumbing it reads.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from apeGmsh.viewers.data._elements import (
    ElementLoadView,
    SurfaceConstraintView,
    ViewerElementGroup,
    ViewerElements,
    ViewerElementType,
)
from apeGmsh.viewers.data._nodes import _NamedNodeSelection
from apeGmsh.viewers.diagrams._beam_geometry import (
    LocalFrame,
    iter_local_frames,
)


def _empty_sel() -> _NamedNodeSelection:
    return _NamedNodeSelection({}, raise_on_missing=False, label="x")


def _line_elements(vecxz: dict | None) -> ViewerElements:
    """One line2 element (id 10, nodes 1→2) + one tri (skipped)."""
    line = ViewerElementGroup(
        element_type=ViewerElementType(
            code=1, name="line2", gmsh_name="Line 2",
            dim=1, npe=2, order=1,
        ),
        ids=np.array([10], dtype=np.int64),
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    tri = ViewerElementGroup(
        element_type=ViewerElementType(
            code=2, name="tri3", gmsh_name="Triangle 3",
            dim=2, npe=3, order=1,
        ),
        ids=np.array([20], dtype=np.int64),
        connectivity=np.array([[1, 2, 3]], dtype=np.int64),
    )
    sel = _empty_sel()
    return ViewerElements(
        groups=[line, tri],
        physical=sel, labels=sel, selection=sel,
        loads=ElementLoadView([]),
        constraints=SurfaceConstraintView([]),
        vecxz=vecxz,
    )


_COORDS = {
    1: np.array([0.0, 0.0, 0.0]),
    2: np.array([2.0, 0.0, 0.0]),   # +x axis, length 2, midpoint (1,0,0)
}


def _node_coord(nid: int):
    return _COORDS.get(int(nid))


def test_viewer_elements_vecxz_plumbing() -> None:
    ve = _line_elements({10: np.array([0.0, 1.0, 0.0])})
    assert ve.has_vecxz is True
    np.testing.assert_allclose(ve.vecxz_for(10), [0.0, 1.0, 0.0])
    assert ve.vecxz_for(999) is None          # non-beam / unknown

    ve0 = _line_elements(None)                 # from_fem path
    assert ve0.has_vecxz is False
    assert ve0.vecxz_for(10) is None


def test_iter_local_frames_default_vecxz() -> None:
    """No OpenSees enrichment → default frame (global-Z reference)."""
    view = SimpleNamespace(elements=_line_elements(None))
    frames = list(iter_local_frames(view, _node_coord))

    assert len(frames) == 1                    # the tri (dim=2) is skipped
    f = frames[0]
    assert isinstance(f, LocalFrame)
    assert f.element_id == 10
    np.testing.assert_allclose(f.x, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(f.origin, [1.0, 0.0, 0.0])
    assert abs(f.length - 2.0) < 1e-12
    # default_vecxz for a non-vertical beam is global Z.
    np.testing.assert_allclose(np.abs(f.z), [0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(np.abs(f.y), [0.0, 1.0, 0.0], atol=1e-9)


def test_iter_local_frames_uses_real_vecxz() -> None:
    """The model's real vecxz reaches the frame — the whole point of
    the join.  vecxz=(0,1,0) rotates the triad off the default."""
    view = SimpleNamespace(
        elements=_line_elements({10: np.array([0.0, 1.0, 0.0])}),
    )
    f = list(iter_local_frames(view, _node_coord))[0]
    np.testing.assert_allclose(f.x, [1.0, 0.0, 0.0])
    # z lies in the (x, vecxz) plane, ⟂ x → aligned with vecxz here.
    np.testing.assert_allclose(np.abs(f.z), [0.0, 1.0, 0.0], atol=1e-9)
    # right-handed: y = z × x = (0,1,0) × (1,0,0) = (0,0,-1)
    np.testing.assert_allclose(np.abs(f.y), [0.0, 0.0, 1.0], atol=1e-9)
    # orthonormal triad
    for a in (f.x, f.y, f.z):
        assert abs(np.linalg.norm(a) - 1.0) < 1e-9
    assert abs(np.dot(f.x, f.y)) < 1e-9
    assert abs(np.dot(f.x, f.z)) < 1e-9


def test_iter_local_frames_skips_degenerate() -> None:
    """Missing node coords → element skipped, sweep continues."""
    view = SimpleNamespace(elements=_line_elements(None))
    frames = list(iter_local_frames(view, lambda nid: None))
    assert frames == []
