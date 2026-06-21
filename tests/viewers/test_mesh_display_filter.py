from __future__ import annotations


class _Actor:
    def __init__(self) -> None:
        self.visible = None

    def SetVisibility(self, value) -> None:
        self.visible = bool(value)


class _Registry:
    dims = [2, 3]

    def __init__(self) -> None:
        self.dim_actors = {2: _Actor(), 3: _Actor()}
        self.dim_wire_actors = {}
        self.dim_silhouette_actors = {2: _Actor(), 3: _Actor()}
        self.dim_node_actors = {2: _Actor(), 3: _Actor()}


class _Plotter:
    def __init__(self) -> None:
        self.renders = 0

    def render(self) -> None:
        self.renders += 1


def test_mesh_filter_controls_silhouettes_and_respects_show_nodes() -> None:
    from apeGmsh.viewers.mesh_viewer import MeshViewer

    viewer = MeshViewer.__new__(MeshViewer)
    viewer._registry = _Registry()
    viewer._plotter = _Plotter()
    viewer._explode_ctrl = None
    viewer._show_nodes = False

    viewer._apply_mesh_filter_visibility({3})

    reg = viewer._registry
    assert reg.dim_actors[2].visible is False
    assert reg.dim_silhouette_actors[2].visible is False
    assert reg.dim_node_actors[2].visible is False
    assert reg.dim_actors[3].visible is True
    assert reg.dim_silhouette_actors[3].visible is True
    assert reg.dim_node_actors[3].visible is False
    assert viewer._plotter.renders == 1
