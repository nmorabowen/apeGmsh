"""Studio host: ModelViewer vs MeshViewer by live mesh presence."""
from __future__ import annotations

from pathlib import Path

from apeGmsh.studio._host import open_host, session_has_mesh


class _StubSession:
    def __init__(self) -> None:
        self.name = "stub"
        self.model = object()


def test_session_has_mesh_false_when_get_elements_fails(monkeypatch) -> None:
    import gmsh

    def boom(dim):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(gmsh.model.mesh, "getElements", boom)
    assert session_has_mesh() is False


def test_open_host_mesh_viewer_when_meshed(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []

    class FakeMeshViewer:
        def __init__(self, parent, **kwargs):
            assert kwargs.get("on_selection_changed") is not None
            seen.append("mesh")

        def show(self, *, title=None):
            seen.append(title or "")

    class FakeModelViewer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ModelViewer should not open")

        def show(self, **kwargs):
            pass

    monkeypatch.setattr("apeGmsh.studio._host.session_has_mesh", lambda: True)
    monkeypatch.setattr(
        "apeGmsh.viewers.mesh_viewer.MeshViewer", FakeMeshViewer,
    )
    monkeypatch.setattr(
        "apeGmsh.viewers.model_viewer.ModelViewer", FakeModelViewer,
    )
    open_host(_StubSession(), envelope_path=tmp_path / "sel.json", title="t")
    assert seen[0] == "mesh"
    assert seen[1] == "t"


def test_open_host_model_viewer_when_unmeshed(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []

    class FakeMeshViewer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MeshViewer should not open")

        def show(self, **kwargs):
            pass

    class FakeModelViewer:
        def __init__(self, parent, model, **kwargs):
            assert kwargs.get("on_selection_changed") is not None
            seen.append("model")

        def show(self, *, title=None):
            seen.append(title or "")

    monkeypatch.setattr("apeGmsh.studio._host.session_has_mesh", lambda: False)
    monkeypatch.setattr(
        "apeGmsh.viewers.mesh_viewer.MeshViewer", FakeMeshViewer,
    )
    monkeypatch.setattr(
        "apeGmsh.viewers.model_viewer.ModelViewer", FakeModelViewer,
    )
    open_host(_StubSession(), envelope_path=tmp_path / "sel.json", title="t")
    assert seen[0] == "model"
    assert seen[1] == "t"
