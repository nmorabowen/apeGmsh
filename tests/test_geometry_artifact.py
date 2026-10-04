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
import json
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
    """The ``off`` arm disables the *capture* (the ``generate()`` hook and
    the ``end()`` fallback both resolve ``gio.capture_geometry`` at call
    time), not only the write: a capture that touched the mesh (a refine,
    a re-generate) would change the ``on`` snapshot and fail here."""

    def run(name: str, off: bool) -> tuple[str, Path]:
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(d))
        if off:
            def no_capture(**kwargs):
                raise RuntimeError("capture disabled")
            monkeypatch.setattr(gio, "capture_geometry", no_capture)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with apeGmsh(model_name="snap") as g:
                _small_box(g)
                sid = g.mesh.queries.get_fem_data().snapshot_id
        geo_warnings = [x for x in w if issubclass(x.category, GeometryArtifactWarning)]
        assert bool(geo_warnings) is off
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

    def boom(**kwargs):
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


# ---------------------------------------------------------------------------
# Review findings on 97812b18 (#1331): foreign files, internal sessions,
# atomic writes, an existing 2-D mesh, and the user's mesh left untouched
# ---------------------------------------------------------------------------


def _mesh_fingerprint() -> tuple[int, int, int]:
    """Node count, element count and a hash of tags + coords + connectivity."""
    import hashlib

    tags, coords, _ = gmsh.model.mesh.getNodes()
    h = hashlib.sha256()
    h.update(np.asarray(tags, dtype=np.int64).tobytes())
    h.update(np.asarray(coords, dtype=np.float64).tobytes())
    n_elems = 0
    for dim in (1, 2, 3):
        etypes, etags, enodes = gmsh.model.mesh.getElements(dim)
        for et, tg, nd in zip(etypes, etags, enodes):
            n_elems += len(tg)
            h.update(np.asarray([et], dtype=np.int64).tobytes())
            h.update(np.asarray(tg, dtype=np.int64).tobytes())
            h.update(np.asarray(nd, dtype=np.int64).tobytes())
    return len(tags), n_elems, int.from_bytes(h.digest()[:8], "big")


def test_foreign_files_are_never_overwritten(monkeypatch, tmp_path: Path) -> None:
    """Finding 1: a non-apeGmsh ``data.h5`` / ``data.geometry.h5`` beside a
    session named ``data`` is left byte-identical, with one warning each."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    model = tmp_path / "data.h5"
    sibling = tmp_path / "data.geometry.h5"
    model.write_bytes(b"not an hdf5 file, the user's own data")
    with h5py.File(sibling, "w") as f:   # a real HDF5 file, but not ours
        f.create_dataset("x", data=np.arange(3))
    before = (model.read_bytes(), sibling.read_bytes())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="data") as g:
            _small_box(g)
    assert (model.read_bytes(), sibling.read_bytes()) == before
    msgs = [str(x.message) for x in w if "not an apeGmsh artifact" in str(x.message)]
    assert len(msgs) == 2
    assert not list(tmp_path.glob("*.tmp-*"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.geometry.h5", "data.h5"]


def test_our_own_artifacts_are_replaced(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="again") as g:
        _small_box(g)
    first = _session_id(tmp_path / "again.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with apeGmsh(model_name="again") as g:
            _small_box(g)
    second = _session_id(tmp_path / "again.h5")
    assert first != second
    assert _session_id(tmp_path / "again.geometry.h5") == second


def test_overwrite_false_is_honoured_by_the_automatic_write(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="keep") as g:
        _small_box(g)
    model, sibling = tmp_path / "keep.h5", tmp_path / "keep.geometry.h5"
    before = (model.read_bytes(), sibling.read_bytes())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="keep", overwrite=False) as g:
            _small_box(g)
    assert (model.read_bytes(), sibling.read_bytes()) == before
    assert len([x for x in w if "overwrite=False" in str(x.message)]) == 2


def test_internal_sessions_opt_out(monkeypatch, tmp_path: Path) -> None:
    """Finding 2: the private ``_artifacts=False`` flag writes nothing."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="internal", _artifacts=False) as g:
        _small_box(g)
        assert g._geometry_capture is None
    assert list(tmp_path.iterdir()) == []


def test_mesh_document_writes_nothing(monkeypatch, tmp_path: Path) -> None:
    """Finding 2: the section mesh worker (a child interpreter) and the
    in-process ``build_fem`` leave no artifact anywhere."""
    from apeGmsh.sections import SectionDocument
    from apeGmsh.sections._mesh_proc import mesh_document

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    doc = SectionDocument.new(name="plate", kind="continuum")
    doc.set_material("s", E=200e3, nu=0.3, fy=345.0)
    doc.add_shape("rect_face", id="s", b=4.0, h=2.0, material="s")
    doc.set_mesh(lc=0.5)
    fem = mesh_document(doc.to_dict())
    assert fem.info.n_elems > 0
    doc.build()
    assert list(tmp_path.iterdir()) == []
    import apeGmsh as pkg
    pkg_dir = Path(pkg.__file__).resolve().parent
    assert not list(pkg_dir.rglob("plate*.h5"))


def test_failed_model_write_leaves_previous_file_and_no_temp(
    monkeypatch, tmp_path: Path
) -> None:
    """Finding 3: ``model.h5`` is written to a temp beside it and replaced."""
    from apeGmsh.mesh.FEMData import FEMData

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="atomic") as g:
        _small_box(g)
    model = tmp_path / "atomic.h5"
    before = model.read_bytes()
    real_to_h5 = FEMData.to_h5

    def torn(self, path, *a, **k):
        Path(path).write_bytes(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(FEMData, "to_h5", torn)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="atomic") as g:
            _small_box(g)
    monkeypatch.setattr(FEMData, "to_h5", real_to_h5)
    assert model.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp-*"))
    assert any("disk full" in str(x.message) for x in w)


def test_failed_geometry_write_leaves_previous_file_and_no_temp(
    monkeypatch, tmp_path: Path
) -> None:
    """Finding 3: a failure after ``/meta`` is written never leaves a torn
    sibling that still pairs with the model."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="torn") as g:
        _small_box(g)
    sibling = tmp_path / "torn.geometry.h5"
    before = sibling.read_bytes()
    real_payload = gio._write_payload

    def torn_payload(f, c, ints, floats, **kw):
        f.create_group("meta").attrs["session_id"] = kw["session_id"]
        raise OSError("disk full")

    monkeypatch.setattr(gio, "_write_payload", torn_payload)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="torn") as g:
            _small_box(g)
    monkeypatch.setattr(gio, "_write_payload", real_payload)
    assert sibling.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp-*"))
    assert any(issubclass(x.category, GeometryArtifactWarning) for x in w)


def test_existing_2d_mesh_is_captured_as_mesh_without_generate(
    monkeypatch, tmp_path: Path
) -> None:
    """Finding 4: a ``from_msh`` import already has 2-D elements; ``end()``
    captures them as ``source=mesh`` with the display ``lod_size`` and
    never calls ``generate``."""
    msh = tmp_path / "plate.msh"
    with apeGmsh(model_name="writer", _artifacts=False) as g:
        g.model.geometry.add_rectangle(0, 0, 0, 2, 1, label="plate")
        g.mesh.sizing.set_global_size(0.25)
        g.mesh.generation.generate(2)
        gmsh.write(str(msh))

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    calls: list[int] = []
    real = gmsh.model.mesh.generate
    with apeGmsh(model_name="imported") as g:
        g.loader.from_msh(msh, dim=2)
        n_before = len(gmsh.model.mesh.getNodes()[0])
        monkeypatch.setattr(gmsh.model.mesh, "generate",
                            lambda dim=3: (calls.append(dim), real(dim)))
        bbox = np.asarray(gmsh.model.getBoundingBox(-1, -1))
    assert calls == []
    with h5py.File(tmp_path / "imported.geometry.h5", "r") as f:
        geo = f["geometry"]
        assert geo.attrs["source"] == "mesh"
        assert geo.attrs["status"] == "ok"
        assert geo.attrs["lod_size"] == pytest.approx(
            gio.LOD_FRACTION * np.linalg.norm(bbox[3:] - bbox[:3])
        )
        assert geo["surfaces/triangles"].shape[0] > 0
    with h5py.File(tmp_path / "imported.h5", "r") as f:
        assert f["nodes/ids"].shape[0] == n_before


def test_capture_never_mutates_the_users_mesh(monkeypatch, tmp_path: Path) -> None:
    """Finding 5: node and element counts and a hash of tags, coordinates
    and connectivity are identical before and after every capture route,
    and right before ``end()`` finalizes gmsh."""
    from apeGmsh import _session as S

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    seen: dict[str, tuple] = {}
    real_release = S._gmsh_release

    def spy_release():
        seen["at_release"] = _mesh_fingerprint()
        real_release()

    monkeypatch.setattr(S, "_gmsh_release", spy_release)
    with apeGmsh(model_name="untouched") as g:
        _hole_box(g)
        g.mesh.sizing.set_global_size(0.15)
        g.mesh.generation.generate(3)
        fp = _mesh_fingerprint()
        assert fp[1] > 0
        capture_geometry(source="mesh")
        assert _mesh_fingerprint() == fp
        gio.capture_fallback()                 # reuses the 2-D elements
        assert _mesh_fingerprint() == fp
        g._geometry_capture = None             # force end() down the fallback
    assert seen["at_release"] == fp
    with h5py.File(tmp_path / "untouched.geometry.h5", "r") as f:
        assert f["geometry"].attrs["source"] == "mesh"


# ---------------------------------------------------------------------------
# Review round 2 on 89f00932 (#1331): the generic envelope key is not proof
# of ownership, and the three remaining library-internal sessions are pinned
# ---------------------------------------------------------------------------


def test_generic_schema_version_attr_is_not_ours(monkeypatch, tmp_path: Path) -> None:
    """A third-party ``data.h5`` with ``/meta@schema_version="3.1"`` (the
    generic envelope name) and its own datasets is foreign: byte-identical
    after a session named ``data``, one warning; only the free sibling
    target is written."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    foreign = tmp_path / "data.h5"
    with h5py.File(foreign, "w") as f:
        f.create_group("meta").attrs["schema_version"] = "3.1"
        f.create_dataset("my/experiment", data=np.arange(5))
    before = foreign.read_bytes()
    assert not gio.is_apegmsh_artifact(foreign)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="data") as g:
            _small_box(g)
    assert foreign.read_bytes() == before
    assert len([x for x in w if "not an apeGmsh artifact" in str(x.message)]) == 1
    assert (tmp_path / "data.geometry.h5").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.geometry.h5", "data.h5"]


def test_is_apegmsh_artifact_ownership_rules(tmp_path: Path) -> None:
    """Ours: any per-zone key, or (a file from before the per-zone split)
    the envelope key together with ``apeGmsh_version``. Not ours: the
    envelope key alone, ``apeGmsh_version`` alone, no ``/meta``, not HDF5."""
    from apeGmsh.opensees._internal.schema_version import _ZONE_KEY

    def make(name: str, attrs: dict, meta: bool = True) -> Path:
        p = tmp_path / name
        with h5py.File(p, "w") as f:
            if meta:
                f.create_group("meta").attrs.update(attrs)
        return p

    for zone, key in _ZONE_KEY.items():
        assert gio.is_apegmsh_artifact(make(f"{zone}.h5", {key: "1.0.0"}))
    assert gio.is_apegmsh_artifact(
        make("legacy.h5", {"schema_version": "2.4.0", "apeGmsh_version": "1.9.0"})
    )
    assert not gio.is_apegmsh_artifact(make("envelope_only.h5", {"schema_version": "3.1"}))
    assert not gio.is_apegmsh_artifact(make("version_only.h5", {"apeGmsh_version": "2.0.0"}))
    assert not gio.is_apegmsh_artifact(make("no_meta.h5", {}, meta=False))
    raw = tmp_path / "raw.h5"
    raw.write_bytes(b"not hdf5")
    assert not gio.is_apegmsh_artifact(raw)


class _Stop(Exception):
    """Stops a library entry point right after its session is constructed."""


def _spy_session_class(monkeypatch) -> list[dict]:
    """Replace the ``apeGmsh`` class an entry point imports lazily
    (``from apeGmsh import apeGmsh`` inside the function) with a spy that
    records the constructor kwargs and stops the entry point there."""
    import apeGmsh as pkg

    seen: list[dict] = []

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        raise _Stop

    monkeypatch.setattr(pkg, "apeGmsh", spy)
    return seen


_SLAB_SM = {
    "schema_version": "0.1",
    "units": {"length": "m", "force": "kN"},
    "nodes": [
        {"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "2", "x": 4.0, "y": 0.0, "z": 0.0},
        {"id": "3", "x": 4.0, "y": 4.0, "z": 0.0},
        {"id": "4", "x": 0.0, "y": 4.0, "z": 0.0},
    ],
    "frames": [],
    "areas": [{"id": "S1", "nodes": ["1", "2", "3", "4"], "section": "SLAB", "kind": "slab"}],
    "sections": [{"name": "SLAB", "kind": "shell", "material": "C", "thickness": 0.30}],
    "materials": [{"name": "C", "E": 2.5e7, "nu": 0.2}],
    "restraints": [{"node": n, "dofs": [1, 1, 1, 1, 1, 1]} for n in ("1", "2", "3", "4")],
    "loads": {"Dead": {"area": [{"area": "S1", "direction": "Z", "value": -5.0}]}},
}


def test_solve_and_extract_session_writes_no_artifacts(monkeypatch, tmp_path: Path) -> None:
    """``interop.solve.solve_and_extract``: the real session meshes and
    ends (its ``finally``) before ``build_opensees`` runs, which is stubbed
    to stop the solve (it needs openseespy); nothing lands in the artifact
    directory. Without ``_artifacts=False`` ``xcheck.h5`` and its sibling
    would be here."""
    from apeGmsh.interop import StructuralModel
    from apeGmsh.interop import solve as solve_mod

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(solve_mod, "build_opensees", stop)
    with pytest.raises(_Stop):
        solve_mod.solve_and_extract(StructuralModel.from_dict(_SLAB_SM), case="Dead")
    assert list(tmp_path.iterdir()) == []


def test_strut_tie_build_session_writes_no_artifacts(monkeypatch, tmp_path: Path) -> None:
    """``interop.strut_tie._build`` meshes the corbel for real (gmsh only)
    and leaves nothing in the artifact directory."""
    from apeGmsh.interop.strut_tie import _build

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    fixture = Path(__file__).parent / "interop" / "fixtures" / "cook_mitchell_corbel.stm.json"
    model = json.loads(fixture.read_text(encoding="utf-8"))
    fe = _build(model, case=None, mesh_size=60.0, extra_fixed_planes=(), verbose=False)
    assert fe.applied is not None
    assert list(tmp_path.iterdir()) == []


def test_results_demo_session_opts_out(monkeypatch, tmp_path: Path) -> None:
    """``results.demo.make_demo_results`` constructs its session with the
    private flag (spied: the demo never calls ``end()``, so a file-based
    oracle cannot pin it)."""
    from apeGmsh.results.demo import make_demo_results

    seen = _spy_session_class(monkeypatch)
    with pytest.raises(_Stop):
        make_demo_results(path=tmp_path)
    assert [kw.get("_artifacts") for kw in seen] == [False]


def test_artifact_zones_by_root_group(tmp_path: Path) -> None:
    def make(name: str, groups: tuple[str, ...]) -> Path:
        p = tmp_path / name
        with h5py.File(p, "w") as f:
            f.create_group("meta")
            for g_ in groups:
                f.create_group(g_)
        return p

    assert gio.artifact_zones(make("n.h5", ("nodes", "elements"))) == {"neutral"}
    assert gio.artifact_zones(make("b.h5", ("nodes", "opensees"))) == {"neutral", "opensees"}
    assert gio.artifact_zones(make("r.h5", ("nodes", "opensees", "stages"))) == {
        "neutral", "opensees", "results",
    }
    assert gio.artifact_zones(make("g.h5", ("geometry",))) == {"geometry"}
    assert gio.artifact_zones(make("p.h5", ("provenance",))) == {"provenance"}
    assert gio.artifact_zones(make("m.h5", ())) == frozenset()


def test_apesees_file_at_the_model_path_is_not_replaced(monkeypatch, tmp_path: Path) -> None:
    """Finding 3 (maintainer ruling: skip and warn). An ``apeSees(fem).h5``
    written at the session's own model path inside the ``with`` block
    holds ``/opensees``, which the end-of-session neutral write would
    drop: ``end()`` warns, leaves the file byte-identical with its
    ``/opensees`` zone, and still writes the geometry sibling, paired."""
    from apeGmsh.opensees import apeSees

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    target = tmp_path / "bridge.h5"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="bridge") as g:
            _small_box(g)
            fem = g.mesh.queries.get_fem_data()
            ops = apeSees(fem)
            ops.model(ndm=3, ndf=3)
            mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
            ops.element.FourNodeTetrahedron(pg="body", material=mat)
            ops.h5(str(target))
            before = target.read_bytes()
    assert target.read_bytes() == before
    msgs = [str(x.message) for x in w if "would drop" in str(x.message)]
    assert len(msgs) == 1 and "opensees" in msgs[0]
    with h5py.File(target, "r") as f:
        assert "opensees" in f
    sibling = tmp_path / "bridge.geometry.h5"
    assert sibling.is_file()
    assert _session_id(sibling) == _session_id(target)
    assert not list(tmp_path.glob("*.tmp-*"))


def test_rerun_without_a_provenance_table_never_drops_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    """Opus on 15e3b5ef: ``write_fem_h5`` writes ``/provenance`` only when the
    snapshot carries a table, so a re-run whose snapshot has none must not
    replace an earlier ``model.h5`` that has the zone: one warning, the file
    byte-identical with ``/provenance`` intact, the sibling still written."""
    from apeGmsh._internal import provenance as prov

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="prov") as g:
        _small_box(g)
    model = tmp_path / "prov.h5"
    with h5py.File(model, "r") as f:
        assert "provenance" in f
    before = model.read_bytes()
    (tmp_path / "prov.geometry.h5").unlink()

    monkeypatch.setattr(prov, "table_for", lambda session: None)  # no table
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="prov") as g:
            _small_box(g)
            assert g.mesh.queries.get_fem_data().provenance is None
    assert model.read_bytes() == before
    msgs = [str(x.message) for x in w if "would drop" in str(x.message)]
    assert len(msgs) == 1 and "provenance" in msgs[0]
    with h5py.File(model, "r") as f:
        assert "provenance" in f
    assert (tmp_path / "prov.geometry.h5").is_file()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_suite_artifact_dir_is_pinned_to_tmp() -> None:
    """The conftest fixture keeps the suite from littering the repo."""
    env = os.environ.get("APEGMSH_ARTIFACT_DIR", "")
    assert env and Path(env).is_dir()
    repo = Path(__file__).resolve().parents[1]
    assert repo not in Path(env).resolve().parents and Path(env).resolve() != repo
