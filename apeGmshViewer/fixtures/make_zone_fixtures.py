"""Write the ADR 0112 zone fixtures, by hand, to architecture/h5-schema.md.

    python apeGmshViewer/fixtures/make_zone_fixtures.py

run from the repository root with an interpreter that has h5py and numpy.
It writes, beside this file:

* ``zones.h5``: ``shoebuckle.h5`` plus ``/meta/session_id`` and a
  ``/provenance`` zone (1.0.0) whose records point at the lines of
  ``examples/shoebuckle_arch.py`` that declare the beam chain;
* ``zones.geometry.h5``: a ``/geometry`` zone (1.0.0) for a unit cube
  (8 points, 12 straight curves of 32 samples, 6 two-triangle faces with
  outward normals, 1 volume), with the same ``session_id``.

h5py writes them, not the app, so the app's reader is checked against the
dtypes a Python writer produces (int8, int32, float64, variable-length
UTF-8). No apeGmsh import: the zone writers (V2b, V2c) are not on main yet,
and the app never runs apeGmsh (ADR 0112 D4).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SESSION = "0f8e5c1a2b3d4e5f8a9b0c1d2e3f4a5b"
STR = h5py.string_dtype(encoding="utf-8")


def provenance(f: h5py.File) -> None:
    script = REPO / "examples" / "shoebuckle_arch.py"
    # Hash the script with LF line ends (as the repository stores it), so the
    # fixture and its test agree on a CRLF checkout too. A real capture hashes
    # the bytes on disk.
    sha = hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    f["meta"].attrs["provenance_schema_version"] = "1.0.0"
    g = f.create_group("provenance")
    g.attrs["base_dir"] = "/work/apeGmsh"
    files = g.create_group("files")
    files["path"] = np.array(["examples/shoebuckle_arch.py", "/opt/helpers/frames.py"], dtype=object).astype(STR)
    files["sha256"] = np.array([sha, "0" * 64], dtype=object).astype(STR)
    files["kind"] = np.array(["script", "module"], dtype=object).astype(STR)
    # sites: file row, 1-based line, function
    site_rows = [
        (0, 285, "declare_model"),  # 0 uniaxialMaterial
        (0, 293, "declare_model"),  # 1 section
        (0, 295, "declare_model"),  # 2 geomTransf
        (0, 296, "declare_model"),  # 3 beamIntegration
        (0, 297, "declare_model"),  # 4 element
        (0, 451, "<module>"),       # 5 the script's outermost line
        (1, 12, "pin_bases"),       # 6 a helper outside the script
    ]
    sites = g.create_group("sites")
    sites["file"] = np.array([r[0] for r in site_rows], dtype=np.int32)
    sites["line"] = np.array([r[1] for r in site_rows], dtype=np.int32)
    sites["function"] = np.array([r[2] for r in site_rows], dtype=object).astype(STR)
    rec_rows = [
        ("opensees/uniaxialMaterial/#1", 0, 5),
        ("opensees/section/#1", 1, 5),
        ("opensees/geomTransf/#1", 2, 5),
        ("opensees/beamIntegration/#1", 3, 5),
        ("opensees/element/#1", 4, 5),
        ("opensees/fix/#1", 6, 5),
        ("mesh/physical_group/Arch", 5, -1),
    ]
    rec = g.create_group("records")
    rec["path"] = np.array([r[0] for r in rec_rows], dtype=object).astype(STR)
    rec["site"] = np.array([r[1] for r in rec_rows], dtype=np.int32)
    rec["script"] = np.array([r[2] for r in rec_rows], dtype=np.int32)
    rec["seq"] = np.arange(len(rec_rows), dtype=np.int32)


def cube_geometry(path: Path) -> None:
    corners = np.array(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )  # index = x + 2y + 4z
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    # Faces as corner quads, counter-clockwise seen from outside.
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    n_samples = 32

    dims = [0] * 8 + [1] * 12 + [2] * 6 + [3]
    tags = list(range(1, 9)) + list(range(1, 13)) + list(range(1, 7)) + [1]
    K = len(dims)
    ent_bbox = np.zeros((K, 6))
    for k in range(8):
        ent_bbox[k] = [*corners[k], *corners[k]]
    for i, (a, b) in enumerate(edges):
        lo, hi = np.minimum(corners[a], corners[b]), np.maximum(corners[a], corners[b])
        ent_bbox[8 + i] = [*lo, *hi]
    for i, q in enumerate(faces):
        pts = corners[list(q)]
        ent_bbox[20 + i] = [*pts.min(0), *pts.max(0)]
    ent_bbox[26] = [0, 0, 0, 1, 1, 1]

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["geometry_schema_version"] = "1.0.0"
        meta.attrs["session_id"] = SESSION
        g = f.create_group("geometry")
        g.attrs["source"] = "mesh"
        g.attrs["gmsh_version"] = "4.13.1"
        g.attrs["curve_samples"] = np.int32(n_samples)
        g.attrs["lod_size"] = 0.03 * np.sqrt(3.0)
        g.attrs["bbox"] = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        g.attrs["status"] = "ok"

        e = g.create_group("entities")
        e["dim"] = np.array(dims, dtype=np.int8)
        e["tag"] = np.array(tags, dtype=np.int32)
        e["bbox"] = ent_bbox
        e["ok"] = np.ones(K, dtype=np.int8)

        p = g.create_group("points")
        p["entity"] = np.arange(8, dtype=np.int32)
        p["xyz"] = corners

        c = g.create_group("curves")
        c["entity"] = np.arange(8, 20, dtype=np.int32)
        t = np.linspace(0.0, 1.0, n_samples)[:, None]
        c["vertices"] = np.concatenate([corners[a] + t * (corners[b] - corners[a]) for a, b in edges])
        c["vertex_offsets"] = (np.arange(13) * n_samples).astype(np.int32)

        s = g.create_group("surfaces")
        s["entity"] = np.arange(20, 26, dtype=np.int32)
        s["vertices"] = np.concatenate([corners[list(q)] for q in faces])
        s["vertex_offsets"] = (np.arange(7) * 4).astype(np.int32)
        s["triangles"] = np.tile(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), (6, 1))
        s["triangle_offsets"] = (np.arange(7) * 2).astype(np.int32)

        v = g.create_group("volumes")
        v["entity"] = np.array([26], dtype=np.int32)
        v["faces"] = np.arange(6, dtype=np.int32)
        v["face_offsets"] = np.array([0, 6], dtype=np.int32)

        m = g.create_group("memberships")
        rows = [(3, 1, "label", "Box", -1), (2, 1, "physical_group", "Bottom", 5), (2, 2, "physical_group", "Top", 6)]
        m["dim"] = np.array([r[0] for r in rows], dtype=np.int8)
        m["tag"] = np.array([r[1] for r in rows], dtype=np.int32)
        m["kind"] = np.array([r[2] for r in rows], dtype=object).astype(STR)
        m["name"] = np.array([r[3] for r in rows], dtype=object).astype(STR)
        m["pg"] = np.array([r[4] for r in rows], dtype=np.int32)


def main() -> None:
    model = HERE / "zones.h5"
    shutil.copyfile(HERE / "shoebuckle.h5", model)
    with h5py.File(model, "a") as f:
        f["meta"].attrs["session_id"] = SESSION
        provenance(f)
    cube_geometry(HERE / "zones.geometry.h5")
    for p in (model, HERE / "zones.geometry.h5"):
        print(f"wrote {p.relative_to(REPO)} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
