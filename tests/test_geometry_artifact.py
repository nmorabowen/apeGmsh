"""V2b (#1305): the D1 unconditional session write and the geometry sibling.

Oracles (ADR 0112 D1 / D2a, V0 decisions 4-6 with amendments 2-4):

* A session with ``save_to=None`` leaves ``<dir>/<model_name>.h5`` and
  ``<dir>/<model_name>.geometry.h5`` with **equal** ``/meta/session_id``;
  two runs differ.
* On a box with a cylindrical hole, the tessellated surface area is within
  2 % of the kernel's ``occ.getMass`` (both capture routes).
* ``snapshot_id`` is equal with the geometry write on and off: the sibling
  is a separate file and no hash reads it.
* A never-meshed session writes a ``temp_mesh`` geometry and leaves no
  mesh behind; the temporary mesh is 2-D only.
* A forced int32 overflow raises before any file is written.
* The failure paths warn (``GeometryArtifactWarning`` once) and never raise.
"""
from __future__ import annotations

import dataclasses
import os
import uuid
import warnings
from pathlib import Path

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._core import default_artifact_dir
from apeGmsh.mesh import _geometry_h5_io as gio
from apeGmsh.mesh._geometry_h5_io import (
    GeometryArtifactWarning,
    GeometryInt32Overflow,
    capture_geometry,
    capture_temp_mesh,
    geometry_sibling_path,
    write_geometry_h5,
)
from tests.fixtures.schema import GEOMETRY_CURRENT

SID = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _hole_box(g: apeGmsh) -> float:
    """A 2 x 1 x 1 box with a through cylindrical hole; returns the exact
    surface area from the OCC kernel."""
    g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="block")
    g.physical.add_volume("block", name="body")
    cyl = gmsh.model.occ.addCylinder(1.0, 0.5, -0.1, 0, 0, 1.2, 0.25)
    gmsh.model.occ.cut([(3, 1)], [(3, cyl)])
    gmsh.model.occ.synchronize()
    return float(sum(gmsh.model.occ.getMass(2, t) for _, t in gmsh.model.getEntities(2)))


def _tessellated_area(cap: gio.GeometryCapture) -> float:
    total = 0.0
    vo, to = cap.surfaces_vertex_offsets, cap.surfaces_triangle_offsets
    for s in range(cap.surfaces_entity.shape[0]):
        v = cap.surfaces_vertices[vo[s]:vo[s + 1]]
        t = cap.surfaces_triangles[to[s]:to[s + 1]]
        p = v[t]
        total += 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
        ).sum()
    return total


def _file_area(path: Path) -> float:
    with h5py.File(path, "r") as f:
        geo = f["geometry"]
        cap = gio.GeometryCapture(
            source=geo.attrs["source"], gmsh_version="", curve_samples=0,
            lod_size=0.0, bbox=np.zeros(6), status="ok",
            surfaces_entity=geo["surfaces/entity"][()],
            surfaces_vertex_offsets=geo["surfaces/vertex_offsets"][()],
            surfaces_vertices=geo["surfaces/vertices"][()],
            surfaces_triangle_offsets=geo["surfaces/triangle_offsets"][()],
            surfaces_triangles=geo["surfaces/triangles"][()],
        )
    return _tessellated_area(cap)


def _session_id(path: Path) -> str:
    with h5py.File(path, "r") as f:
        return str(f["meta"].attrs["session_id"])


def _small_box(g: apeGmsh) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.add_volume("body", name="body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(3)


# ---------------------------------------------------------------------------
# D1: the unconditional write and the conventional path
# ---------------------------------------------------------------------------


def test_default_artifact_dir_resolution(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path / "env"))
    assert default_artifact_dir() == tmp_path / "env"
    monkeypatch.delenv("APEGMSH_ARTIFACT_DIR")
    # under pytest ``__main__`` is pytest's own entry point, which has a file
    import sys
    main_file = getattr(sys.modules["__main__"], "__file__", None)
    if main_file:
        assert default_artifact_dir() == Path(main_file).resolve().parent
    else:  # pragma: no cover - a REPL-like host
        assert default_artifact_dir() == Path.cwd()


def test_no_save_to_writes_model_and_sibling_with_equal_session_id(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with apeGmsh(model_name="d1") as g:
            _small_box(g)
            fem = g.mesh.queries.get_fem_data()
            in_session = g._session_id
    model = tmp_path / "d1.h5"
    sibling = tmp_path / "d1.geometry.h5"
    assert model.is_file() and sibling.is_file()
    assert geometry_sibling_path(model) == sibling
    assert _session_id(model) == _session_id(sibling) == in_session == fem.session_id
    with h5py.File(sibling, "r") as f:
        assert f["meta"].attrs["geometry_schema_version"] == GEOMETRY_CURRENT
        assert f["geometry"].attrs["source"] == "mesh"
        assert f["geometry"].attrs["status"] == "ok"
        assert f["geometry"].attrs["curve_samples"] == gio.CURVE_SAMPLES
        rows = list(zip(
            f["geometry/memberships/kind"].asstr()[()],
            f["geometry/memberships/name"].asstr()[()],
            f["geometry/memberships/pg"][()].tolist(),
        ))
    assert ("label", "body", -1) in rows
    assert any(k == "physical_group" and n == "body" and pg > 0 for k, n, pg in rows)


def test_two_runs_mint_different_session_ids(monkeypatch, tmp_path: Path) -> None:
    ids = []
    for i in range(2):
        d = tmp_path / f"run{i}"
        d.mkdir()
        monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(d))
        with apeGmsh(model_name="twice") as g:
            _small_box(g)
        assert _session_id(d / "twice.h5") == _session_id(d / "twice.geometry.h5")
        ids.append(_session_id(d / "twice.h5"))
    assert ids[0] != ids[1]


def test_two_extractions_in_one_session_share_the_id() -> None:
    with apeGmsh(model_name="same") as g:
        _small_box(g)
        a = g.mesh.queries.get_fem_data(dim=3)
        b = g.mesh.queries.get_fem_data(dim=2)
        assert a.session_id == b.session_id == g._session_id


def test_save_to_override_places_sibling_beside_it(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path / "elsewhere"))
    out = tmp_path / "custom" / "my_model.h5"
    out.parent.mkdir()
    with apeGmsh(model_name="m", save_to=out) as g:
        _small_box(g)
    assert out.is_file()
    assert (tmp_path / "custom" / "my_model.geometry.h5").is_file()
    assert not (tmp_path / "elsewhere").exists()


def test_from_h5_session_writes_no_geometry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="src") as g:
        _small_box(g)
    before = sorted(p.name for p in tmp_path.iterdir())
    g2 = apeGmsh.from_h5(tmp_path / "src.h5", model_name="replay")
    g2.end()  # never began: no kernel, nothing written
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_part_session_writes_nothing(monkeypatch, tmp_path: Path) -> None:
    from apeGmsh.core.Part import Part

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with Part("p", auto_persist=False) as part:
        part.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
    assert list(tmp_path.glob("*.h5")) == []


# ---------------------------------------------------------------------------
# D2a: the tessellation oracles
# ---------------------------------------------------------------------------


def test_mesh_capture_area_within_2pct_of_kernel(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="hole") as g:
        exact = _hole_box(g)
        g.mesh.sizing.set_global_size(0.1)
        g.mesh.generation.generate(3)
        cap = g._geometry_capture
        assert cap is not None and cap.source == "mesh" and cap.status == "ok"
        assert abs(_tessellated_area(cap) - exact) / exact < 0.02
        # every surface tessellated, every volume closed by its faces
        assert cap.surfaces_entity.shape[0] == len(gmsh.model.getEntities(2))
        assert cap.volumes_faces.shape[0] == cap.volumes_face_offsets[-1]
    assert abs(_file_area(tmp_path / "hole.geometry.h5") - exact) / exact < 0.02


def test_never_meshed_session_temp_mesh_area_and_no_mesh_left(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="unmeshed") as g:
        exact = _hole_box(g)
        cap = capture_temp_mesh()
        assert cap.source == "temp_mesh" and cap.status == "ok"
        assert abs(_tessellated_area(cap) - exact) / exact < 0.02
        assert cap.lod_size == pytest.approx(
            gio.LOD_FRACTION * np.linalg.norm(cap.bbox[3:] - cap.bbox[:3])
        )
        # the temporary mesh is gone
        assert len(gmsh.model.mesh.getNodes()[0]) == 0
        assert g._geometry_capture is None
    # end() took the same route: model.h5 carries no mesh, the sibling does
    with h5py.File(tmp_path / "unmeshed.geometry.h5", "r") as f:
        assert f["geometry"].attrs["source"] == "temp_mesh"
        assert f["geometry/surfaces/triangles"].shape[0] > 0
    with h5py.File(tmp_path / "unmeshed.h5", "r") as f:
        assert f["nodes/ids"].shape[0] == 0
    assert _session_id(tmp_path / "unmeshed.h5") == _session_id(
        tmp_path / "unmeshed.geometry.h5"
    )


def test_temp_mesh_is_2d_only(monkeypatch) -> None:
    """Amendment 3: the fallback never builds a 3-D mesh."""
    dims: list[int] = []
    real = gmsh.model.mesh.generate

    def spy(dim=3):
        dims.append(dim)
        return real(dim)

    with apeGmsh(model_name="dims") as g:
        _hole_box(g)
        monkeypatch.setattr(gmsh.model.mesh, "generate", spy)
        capture_temp_mesh()
    # one call here, one from end()'s own fallback: every one is 2-D
    assert dims == [2, 2]


def test_generate_dim1_defers_to_temp_mesh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="curves") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
        g.mesh.generation.generate(1)
        assert g._geometry_capture is None
    with h5py.File(tmp_path / "curves.geometry.h5", "r") as f:
        assert f["geometry"].attrs["source"] == "temp_mesh"
        assert f["geometry/surfaces/triangles"].shape[0] > 0


def test_surface_with_embedded_curve_tessellates() -> None:
    """A slab with an embedded column line: its elements use nodes
    classified on the embedded curve, which a per-surface
    ``getNodes(includeBoundary=True)`` does not return (San Ramon 1A lost
    13 slabs this way)."""
    with apeGmsh(model_name="embed") as g:
        gm = g.model.geometry
        gm.add_rectangle(0, 0, 0, 2, 2, label="slab")
        p1 = gm.add_point(0.7, 0.9, 0)
        p2 = gm.add_point(1.3, 1.1, 0)
        line = gm.add_line(p1, p2, label="column")
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.embed(1, [line], 2, 1)
        g.mesh.sizing.set_global_size(0.3)
        g.mesh.generation.generate(2)
        cap = g._geometry_capture
    assert cap is not None and cap.status == "ok"
    exact = 4.0
    assert abs(_tessellated_area(cap) - exact) / exact < 0.02


def test_curves_are_sampled_parametrically() -> None:
    with apeGmsh(model_name="arc") as g:
        gm = g.model.geometry
        a = gm.add_point(1, 0, 0)
        c = gm.add_point(0, 0, 0)
        b = gm.add_point(0, 1, 0)
        gm.add_arc(a, c, b, label="quarter")
        cap = capture_geometry(source="mesh")
    assert cap.curves_entity.shape[0] == 1
    pts = cap.curves_vertices[cap.curves_vertex_offsets[0]:cap.curves_vertex_offsets[1]]
    assert pts.shape == (gio.CURVE_SAMPLES, 3)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0)
    assert cap.points_xyz.shape == (3, 3)


def test_snapshot_id_equal_with_geometry_write_on_and_off(monkeypatch, tmp_path: Path) -> None:
    def run(name: str, off: bool) -> tuple[str, Path]:
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(d))
        if off:
            monkeypatch.setattr(
                gio, "write_geometry_h5",
                lambda *a, **k: (_ for _ in ()).throw(OSError("disabled")),
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with apeGmsh(model_name="snap") as g:
                _small_box(g)
                sid = g.mesh.queries.get_fem_data().snapshot_id
        if off:
            assert [x for x in w if issubclass(x.category, GeometryArtifactWarning)]
        return sid, d

    on_id, on_dir = run("on", off=False)
    off_id, off_dir = run("off", off=True)
    assert on_id == off_id
    assert (on_dir / "snap.geometry.h5").is_file()
    assert not (off_dir / "snap.geometry.h5").exists()
    with h5py.File(on_dir / "snap.h5", "r") as a, h5py.File(off_dir / "snap.h5", "r") as b:
        assert a["meta"].attrs["snapshot_id"] == b["meta"].attrs["snapshot_id"]


# ---------------------------------------------------------------------------
# The int32 policy and the failure paths
# ---------------------------------------------------------------------------


def test_int32_overflow_refuses_before_writing(tmp_path: Path) -> None:
    with apeGmsh(model_name="ovf") as g:
        _small_box(g)
        cap = g._geometry_capture
    assert cap is not None
    offsets = cap.surfaces_triangle_offsets.copy()
    offsets[-1] = 2 ** 31
    bad = dataclasses.replace(cap, surfaces_triangle_offsets=offsets)
    out = tmp_path / "ovf.geometry.h5"
    with pytest.raises(GeometryInt32Overflow, match="int32"):
        write_geometry_h5(out, bad, session_id=SID, model_name="ovf")
    assert not out.exists()
    # a tag of 2^31 - 1 still fits
    tags = cap.entities_tag.copy()
    tags[0] = 2 ** 31 - 1
    write_geometry_h5(out, dataclasses.replace(cap, entities_tag=tags),
                      session_id=SID, model_name="ovf")
    with h5py.File(out, "r") as f:
        assert f["geometry/entities/tag"].dtype == np.int32
        assert int(f["geometry/entities/tag"][0]) == 2 ** 31 - 1


def test_no_int64_in_the_zone(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="dtypes") as g:
        _small_box(g)
    with h5py.File(tmp_path / "dtypes.geometry.h5", "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "iu":
                assert obj.dtype.itemsize <= 4, name
        f["geometry"].visititems(visit)
        assert f["geometry/entities/dim"].dtype == np.int8
        assert f["geometry/surfaces/triangles"].dtype == np.int32


def test_malformed_session_id_is_refused(tmp_path: Path) -> None:
    with apeGmsh(model_name="sid") as g:
        _small_box(g)
        cap = g._geometry_capture
    with pytest.raises(ValueError):
        write_geometry_h5(tmp_path / "x.geometry.h5", cap, session_id="not-a-uuid", model_name="x")


def test_capture_failure_warns_once_and_session_still_ends(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))

    def boom():
        raise RuntimeError("kernel exploded")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="fail") as g:
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            monkeypatch.setattr(gio, "capture_temp_mesh", boom)
    geo_warnings = [x for x in w if issubclass(x.category, GeometryArtifactWarning)]
    assert len(geo_warnings) == 1
    assert "kernel exploded" in str(geo_warnings[0].message)
    assert not gmsh.isInitialized()
    assert (tmp_path / "fail.h5").is_file()           # the model still landed
    assert not (tmp_path / "fail.geometry.h5").exists()


def test_generate_capture_failure_warns_and_end_retries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))

    def boom(**kwargs):
        raise RuntimeError("tessellation failed")

    monkeypatch.setattr(gio, "capture_geometry", boom)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="retry") as g:
            _small_box(g)
            assert g._geometry_capture is None
    msgs = [str(x.message) for x in w if issubclass(x.category, GeometryArtifactWarning)]
    assert any("tessellation failed" in m for m in msgs)


def test_unwritable_artifact_dir_warns_and_finalizes(monkeypatch, tmp_path: Path) -> None:
    blocker = tmp_path / "file_not_dir"
    blocker.write_bytes(b"")
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(blocker / "nested"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="nowrite") as g:
            _small_box(g)
    assert not gmsh.isInitialized()
    assert any("autosave" in str(x.message) for x in w)


def test_suite_artifact_dir_is_pinned_to_tmp() -> None:
    """The conftest fixture keeps the suite from littering the repo."""
    env = os.environ.get("APEGMSH_ARTIFACT_DIR", "")
    assert env and Path(env).is_dir()
    repo = Path(__file__).resolve().parents[1]
    assert repo not in Path(env).resolve().parents and Path(env).resolve() != repo
