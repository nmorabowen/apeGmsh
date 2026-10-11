"""Write the ADR 0112 zone fixtures, by hand, to architecture/h5-schema.md.

    python apeGmshViewer/fixtures/make_zone_fixtures.py

run from the repository root with an interpreter that has h5py and numpy.
It writes, beside this file:

* ``zones.h5``: ``shoebuckle.h5`` plus ``/meta/session_id``, an
  ``/opensees/names`` table naming the material ``Steel`` and the section
  ``W_section``, and a hand-written ``/provenance`` zone (1.0.0) whose
  records point at the lines of ``examples/shoebuckle_arch.py`` that
  declare the beam chain, in place of the writer's own 1.1.0 zone;
* ``zones.geometry.h5``: a ``/geometry`` zone (1.0.0) for a unit cube
  (8 points, 12 straight curves of 32 samples, 6 two-triangle faces with
  outward normals, 1 volume), with the same ``session_id``;
* ``definitions.h5``: ``shoebuckle.h5`` as a staged opensees 2.26.0 file
  with ``/opensees/decls``, ``/opensees/decl_params``, three time series and
  two stages (see ``definitions`` below).

h5py writes them, not the app, so the app's reader is checked against the
dtypes a Python writer produces (int8, int32, float64, variable-length
UTF-8). No apeGmsh import: the zone writers (V2b, V2c) are not on main yet,
and the app never runs apeGmsh (ADR 0112 D4).
"""

from __future__ import annotations

import hashlib
import json
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
    # The material and the section carry user names (/opensees/names, below),
    # so their records are keyed by name; the rest are unnamed (#k).
    rec_rows = [
        ("opensees/uniaxialMaterial/Steel", 0, 5),
        ("opensees/section/W_section", 1, 5),
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


def definitions(path: Path) -> None:
    """``shoebuckle.h5`` as a staged opensees 2.26.0 file, by hand.

    The model-definition zones the app reads (h5-schema.md): ``/opensees/decls``
    (2.25.0) and ``/opensees/decl_params`` (2.26.0) for every declaration of
    the file, three time series, a global pattern and recorder, and two stages
    (``gravity``: static, a pattern, a fix and a recorder; ``push``: transient,
    a pattern and a ``remove_sp``), in place of the global ``/opensees/analysis``
    (a staged file carries none). No ``/sequence`` zone (V3a is not on main).
    The shapes copy the two real 2.26.0 archives V3 was scoped on (the K&R
    three-point bend and the ``two_column_frame`` staged golden).
    """
    shutil.copyfile(HERE / "shoebuckle.h5", path)
    with h5py.File(path, "a") as f:
        f["meta"].attrs["opensees_schema_version"] = "2.26.0"
        ops = f["opensees"]
        del ops["analysis"]
        n_fix = ops["bcs/fix"].shape[0]
        n_elem = ops["element_meta/dispBeamColumn/ids"].shape[0]
        first_elem = int(ops["element_meta/dispBeamColumn/ids"][:].min())

        # Two more series (tags 2, 3) for the stage patterns.
        for tag in (2, 3):
            ts = ops["time_series"].create_group(f"Linear_{tag}")
            ts.attrs["type"] = "Linear"
            ts.attrs["tag"] = np.int64(tag)
            ts.attrs["params"] = np.array([], dtype=np.float64)

        def pattern(parent: h5py.Group, tag: int, emit_index: int) -> None:
            p = parent.create_group(f"Plain_{tag}")
            p.attrs["type"] = "Plain"
            p.attrs["tag"] = np.int64(tag)
            p.attrs["params"] = np.array([1.0])
            p.attrs["series_ref"] = f"/opensees/time_series/Linear_{tag}"
            p.attrs["emit_index"] = np.int64(emit_index)
            loads = np.zeros(1, dtype=[("target_kind", STR), ("target", STR), ("forces", "<f8", (3,))])
            loads[0] = ("pg", "Arch", (0.0, -1.0, 0.0))
            p["loads"] = loads

        def analysis(g: h5py.Group, transient: bool) -> None:
            a = g.create_group("analysis")
            for k, v in {
                "handler": "Plain", "numberer": "RCM", "system": "UmfPack",
                "test": "NormDispIncr", "algorithm": "Newton",
                "integrator": "Newmark" if transient else "LoadControl",
                "analysis": "Transient" if transient else "Static",
            }.items():
                a.attrs[k] = v
            a.attrs["test_args"] = np.array([1e-6, 10.0, 0.0, 2.0])
            a.attrs["integrator_args"] = np.array([0.5, 0.25] if transient else [0.1])

        stages = ops.create_group("stages")
        stages.attrs["n_stages"] = np.int64(2)
        s0 = stages.create_group("stage_000")
        s0.attrs["name"] = "gravity"
        s0.attrs["analyze_steps"] = np.int64(10)
        s0.attrs["domain_change"] = np.int64(1)
        analysis(s0, transient=False)
        pattern(s0.create_group("patterns"), 2, 1)
        fix = np.zeros(1, dtype=[("target_kind", STR), ("target", STR), ("dofs", "<i8", (3,))])
        fix[0] = ("node", "62", (0, 1, 0))
        s0.create_group("bcs")["fix"] = fix
        rec = s0.create_group("recorders").create_group("Node_1")
        rec.attrs["kind"] = "typed"
        rec.attrs["type"] = "Node"
        rec.attrs["file"] = "gravity.out"
        rec.attrs["params_str"] = np.array(["-file", "gravity.out", "-node", "", "-dof", "", "disp"], dtype=object).astype(STR)
        rec.attrs["params"] = np.array([np.nan, np.nan, np.nan, 62.0, np.nan, 2.0, np.nan])
        s1 = stages.create_group("stage_001")
        s1.attrs["name"] = "push"
        s1.attrs["analyze_steps"] = np.int64(5)
        s1.attrs["analyze_dt"] = 0.01
        s1.attrs["set_time"] = 0.0
        s1.attrs["domain_change"] = np.int64(0)
        analysis(s1, transient=True)
        pattern(s1.create_group("patterns"), 3, 1)
        s1["remove_sp"] = np.array([[62, 2]], dtype=np.int64)

        # /opensees/decls: key, family, name, synth; the tag runs and the
        # tagless rows; one decl_params row per owner (the region has two).
        M, S, T, BI = (
            "opensees/uniaxialMaterial/#1", "opensees/section/#1",
            "opensees/geomTransf/#1", "opensees/beamIntegration/#1",
        )
        rows = [
            # key, family, name, synth, [(type, params, params_names, transf, integration, section)]
            (M, "uniaxialMaterial", "", 0, [("ASDSteel1D", {"E": 2e11, "fy": 3.75e8, "fu": 4.8e8, "eu": 0.2, "fracture": True}, "", [], [], [])]),
            (S, "section", "", 0, [("Fiber", {"GJ": None, "patches": [{"$struct": "RectPatch", "fields": {"material": {"$decl": M}, "ny": 8, "nz": 1}}]}, "", [], [], [])]),
            (T, "geomTransf", "", 0, [("Linear", {"orientation": {"$opaque": "AlongBeam"}, "vecxz": None}, "", [], [], [])]),
            (BI, "beamIntegration", "", 0, [("Legendre", {"section": {"$decl": S}, "n": 2}, '["section", "n"]', [], [], [S])]),
            ("opensees/element/#1", "element", "", 0, [("dispBeamColumn", {"pg": "Arch", "transf": {"$decl": T}, "integration": {"$decl": BI}}, "", [T], [BI], [])]),
            ("opensees/timeSeries/#1", "timeSeries", "", 0, [("Linear", {"factor": 1.0}, "", [], [], [])]),
            ("opensees/pattern/#1", "pattern", "", 0, [("Plain", {"series": {"$decl": "opensees/timeSeries/#1"}}, "", [], [], [])]),
            ("opensees/recorder/#1", "recorder", "", 0, [("MPCO", {"file": "shoebuckle.mpco"}, "", [], [], [])]),
            ("opensees/fix/supports", "fix", "supports", 0, [("FixRecord", {"pg": None, "nodes": [1, 124], "dofs": [1, 1, 0]}, "", [], [], [])]),
            ("opensees/timeSeries/ramp", "timeSeries", "ramp", 0, [("Linear", {"factor": 1.0}, "", [], [], [])]),
            ("opensees/pattern/gravity", "pattern", "gravity", 0, [("Plain", {"series": {"$decl": "opensees/timeSeries/ramp"}}, "", [], [], [])]),
            ("opensees/recorder/crown", "recorder", "crown", 0, [("Node", {"file": "gravity.out", "nodes": [62], "dofs": [2], "response": "disp"}, "", [], [], [])]),
            ("opensees/timeSeries/#2", "timeSeries", "", 0, [("Linear", {"factor": 1.0}, "", [], [], [])]),
            ("opensees/pattern/push", "pattern", "push", 0, [("Plain", {"series": {"$decl": "opensees/timeSeries/#2"}}, "", [], [], [])]),
            ("opensees/region/arch", "region", "arch", 0, [
                ("RegionAssignmentRecord", {"name": "arch", "elements": [2, 3]}, "", [], [], []),
                ("RegionAssignmentRecord", {"name": "arch", "elements": [4]}, "", [], [], []),
            ]),
            ("opensees/timeSeries/support:gravity/hold", "timeSeries", "", 1, [("Constant", {"factor": 1.0}, "", [], [], [])]),
        ]
        d = ops.create_group("decls")
        d["key"] = np.array([r[0] for r in rows], dtype=object).astype(STR)
        d["family"] = np.array([r[1] for r in rows], dtype=object).astype(STR)
        d["name"] = np.array([r[2] for r in rows], dtype=object).astype(STR)
        d["synth"] = np.array([r[3] for r in rows], dtype=np.int8)
        runs = [
            ("uniaxialMaterial", 1, 1, 0), ("section", 1, 1, 1), ("geomTransf", 1, 1, 2),
            ("beamIntegration", 1, 1, 3), ("element", first_elem, n_elem, 4),
            ("timeSeries", 1, 1, 5), ("pattern", 1, 1, 6), ("timeSeries", 2, 1, 9),
            ("pattern", 2, 1, 10), ("timeSeries", 3, 1, 12), ("pattern", 3, 1, 13),
        ]
        tags = d.create_group("tags")
        tags["kind"] = np.array([r[0] for r in runs], dtype=object).astype(STR)
        tags["first"] = np.array([r[1] for r in runs], dtype=np.int64)
        tags["count"] = np.array([r[2] for r in runs], dtype=np.int64)
        tags["decl"] = np.array([r[3] for r in runs], dtype=np.int64)
        tr = d.create_group("rows")
        tr["bcs/fix"] = np.full(n_fix, 8, dtype=np.int64)
        tr["recorders"] = np.array([7], dtype=np.int64)
        tr["stages/stage_000/bcs/fix"] = np.array([8], dtype=np.int64)
        tr["stages/stage_000/recorders"] = np.array([11], dtype=np.int64)

        owners = [(i, o) for i, r in enumerate(rows) for o in r[4]]
        dp = ops.create_group("decl_params")
        dp["decl"] = np.array([i for i, _ in owners], dtype=np.int64)
        dp["type"] = np.array([o[0] for _, o in owners], dtype=object).astype(STR)
        dp["params"] = np.array([json.dumps(o[1]) for _, o in owners], dtype=object).astype(STR)
        dp["params_names"] = np.array([o[2] for _, o in owners], dtype=object).astype(STR)
        for col, j in (("transf_ref", 3), ("integration_ref", 4), ("section_ref", 5)):
            g = dp.create_group(col)
            counts = [len(o[j]) for _, o in owners]
            g["first"] = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
            g["count"] = np.array(counts, dtype=np.int64)
            g["key"] = np.array([k for _, o in owners for k in o[j]], dtype=object).astype(STR)


def main() -> None:
    model = HERE / "zones.h5"
    shutil.copyfile(HERE / "shoebuckle.h5", model)
    with h5py.File(model, "a") as f:
        f["meta"].attrs["session_id"] = SESSION
        # main's writer stamps its own /provenance (1.1.0, #1378) into
        # shoebuckle.h5; this fixture carries the hand-written 1.0.0 zone
        # instead, so drop the writer's before writing it.
        if "provenance" in f:
            del f["provenance"]
            del f["meta"].attrs["provenance_schema_version"]
        # Bridge-side names (opensees 2.13+): name, kind (the family), tag.
        names = f["opensees"].create_group("names")
        names["name"] = np.array(["Steel", "W_section"], dtype=object).astype(STR)
        names["kind"] = np.array(["uniaxialMaterial", "section"], dtype=object).astype(STR)
        names["tag"] = np.array([1, 1], dtype=np.int64)
        provenance(f)
    cube_geometry(HERE / "zones.geometry.h5")
    definitions(HERE / "definitions.h5")
    for p in (model, HERE / "zones.geometry.h5", HERE / "definitions.h5"):
        print(f"wrote {p.relative_to(REPO)} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
