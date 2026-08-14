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


def test_apply_highlight_to_state_uses_select_batch() -> None:
    from apeGmsh.studio._highlight import apply_highlight_to_state

    class FakeSel:
        def __init__(self) -> None:
            self.calls: list = []

        def select_batch(self, dts, *, replace=False):
            self.calls.append((list(dts), replace))

    sel = FakeSel()
    applied = apply_highlight_to_state(
        sel,
        names=["front"],
        resolve=lambda names: [(2, 5)] if "front" in names else [],
    )
    assert applied == [(2, 5)]
    assert sel.calls == [([(2, 5)], True)]


def test_apply_highlight_miss_is_noop() -> None:
    from apeGmsh.studio._highlight import apply_highlight_to_state

    class FakeSel:
        def __init__(self) -> None:
            self.calls: list = []

        def select_batch(self, dts, *, replace=False):
            self.calls.append((list(dts), replace))

    sel = FakeSel()
    applied = apply_highlight_to_state(
        sel, names=["front"], resolve=lambda names: [],
    )
    assert applied == []
    assert sel.calls == []


def test_consume_highlight_ignores_seeded_mtime(tmp_path: Path) -> None:
    from apeGmsh.studio._highlight import (
        consume_highlight_update,
        seed_highlight_mtime,
        write_highlight,
    )

    path = tmp_path / "highlight.json"
    write_highlight(path, names=["front"], dimtags=[(2, 5)])
    seeded = seed_highlight_mtime(path)
    request, mtime = consume_highlight_update(path, seeded)
    assert request is None
    assert mtime == seeded


def test_consume_highlight_applies_after_rewrite(tmp_path: Path) -> None:
    from apeGmsh.studio._highlight import (
        consume_highlight_update,
        seed_highlight_mtime,
        write_highlight,
    )

    path = tmp_path / "highlight.json"
    write_highlight(path, names=["old"], dimtags=[])
    seeded = seed_highlight_mtime(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("old", "front"),
        encoding="utf-8",
    )
    # Windows mtime can stay equal on a same-second rewrite.
    import os
    os.utime(path, (seeded + 1, seeded + 1))
    request, mtime = consume_highlight_update(path, seeded)
    assert request is not None
    assert request["names"] == ["front"]
    assert mtime != seeded
