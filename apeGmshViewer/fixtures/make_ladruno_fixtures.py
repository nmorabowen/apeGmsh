"""Write the ``.ladruno`` fixtures of ``reader/ladruno.ts`` (ADR 0122 slice 4, #2126).

    python apeGmshViewer/fixtures/make_ladruno_fixtures.py          # the h5py ones
    python apeGmshViewer/fixtures/make_ladruno_fixtures.py --modes  # + the fork run

run from the repository root. The first form needs h5py and numpy only; the
second also needs a Ladruno fork build of OpenSees behind ``openseespy`` (the
banner lists "Ladruno — modular HDF5 .ladruno recorder"). It writes, beside
this file:

* ``ladruno_truss2d.ladruno``: a byte copy of the fork-written
  ``tests/fixtures/ladruno/truss2d.ladruno`` (fork build ``605affeb``,
  FORMAT_VERSION 1), unstamped: ``INFO/PROVENANCE`` is the empty group the
  recorder creates. Two trusses (EA = 1000, L = 1) in a line, a load of 10 at
  node 3 under LoadControl 0.25 for 4 steps, so
  ``Ux(node j, t) = 10 (j - 1) t / 1000`` and ``Uy = 0``. It records
  ``DISPLACEMENT`` and an element ``basicForce`` (which the reader lists as
  unread).
* ``ladruno_truss2d.stamped.ladruno``: the same file with the six lowercase
  ``INFO/PROVENANCE`` attributes of the fork schema §7b.1 written by hand,
  each a scalar variable-length UTF-8 string (``STAMP`` below).
* ``ladruno_truss2d.badstamp.ladruno``: the same file with a stamp whose
  values are not all the §7b.1 encoding (``BAD_STAMP`` below): a fixed-length
  ``model_hash``, a variable-length ASCII ``fem_hash``, a ``stage_names`` that
  is not a JSON array of strings, an integer ``unsupported``, and a
  ``writer`` that is variable-length UTF-8 but a shape-(1,) array, not a
  scalar. The reader reports each one, and pairs nothing.
* ``ladruno_modes.ladruno`` (``--modes``): a fork run. Two trusses (EA = 1000,
  L = 1) along x, node 1 fixed, y and the rotation fixed everywhere, a mass
  of 1 in x at nodes 2 and 3; ``eigen -fullGenLapack 2`` then ``record``, with
  ``-N displacement modesOfVibration modesOfVibrationRotational``. The two
  free DOFs give the closed form ``lambda = (k / m) (3 -/+ sqrt 5) / 2`` with
  ``k = 1000``, ``m = 1``, and mode shapes ``Ux3 / Ux2 = 2 - lambda / k``
  (the golden ratio for mode 1). The fork promotes the stage to
  ``KIND=eigen``; it holds one recorded step of ``DISPLACEMENT`` and the two
  modes under ``MODES_OF_VIBRATION(U)`` and ``(R)`` (the rotation column is
  fixed, so it is all zeros). Committed from fork build ``79e0623670``, the
  build the opensees venv imported on 2026-10-16. It predates the repository's
  ``FORK_PIN`` (``c744d6e70``) and writes no ``INFO/FORMAT_MINOR_VERSION``, so
  the file is FORMAT_MINOR_VERSION 0 (schema §2a: its ``KIND`` is the
  recorder-wide label, here the eigen promotion). Regenerate it on a pin
  build to get minor 1; the closed forms do not change.

No apeGmsh import: the app never runs apeGmsh (ADR 0112 D4), and these files
are what the fork writes, not what apeGmsh would make of them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "tests" / "fixtures" / "ladruno" / "truss2d.ladruno"
VLEN_UTF8 = h5py.string_dtype(encoding="utf-8")

#: The hand-written §7b.1 stamp: the six lowercase attributes, as strings.
STAMP: dict[str, str] = {
    "model_hash": "0123456789abcdef0123456789abcdef",
    "fem_hash": "fedcba9876543210fedcba9876543210",
    "stage_names": json.dumps(["push"], separators=(",", ":")),
    "element_index_space": "fem",
    "unsupported": "[]",
    "writer": "apeGmsh fixture (make_ladruno_fixtures.py)",
}


def truss2d() -> None:
    shutil.copyfile(SOURCE, HERE / "ladruno_truss2d.ladruno")


def stamped() -> None:
    out = HERE / "ladruno_truss2d.stamped.ladruno"
    shutil.copyfile(SOURCE, out)
    with h5py.File(out, "r+") as f:
        prov = f["INFO/PROVENANCE"]
        for name, value in STAMP.items():
            prov.attrs.create(name, data=value, dtype=VLEN_UTF8)


def badstamp() -> None:
    out = HERE / "ladruno_truss2d.badstamp.ladruno"
    shutil.copyfile(SOURCE, out)
    with h5py.File(out, "r+") as f:
        prov = f["INFO/PROVENANCE"]
        # Fixed-length bytes: the value reads as the right text, the encoding is wrong.
        prov.attrs.create("model_hash", data=np.bytes_(STAMP["model_hash"]))
        prov.attrs.create("fem_hash", data=STAMP["fem_hash"], dtype=h5py.string_dtype(encoding="ascii"))
        prov.attrs.create("stage_names", data='{"push": 1}', dtype=VLEN_UTF8)
        prov.attrs.create("unsupported", data=np.int32(0))
        prov.attrs.create("element_index_space", data="fem", dtype=VLEN_UTF8)
        # The right encoding at the wrong shape: a (1,) array, not a scalar.
        prov.attrs.create("writer", data=[STAMP["writer"]], dtype=VLEN_UTF8)


_MODES_RUN = r"""
import sys
import openseespy.opensees as ops

path = sys.argv[1]
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 3)
for i in range(3):
    ops.node(i + 1, float(i), 0.0)
ops.fix(1, 1, 1, 1)
ops.fix(2, 0, 1, 1)
ops.fix(3, 0, 1, 1)
ops.uniaxialMaterial("Elastic", 1, 1000.0)
ops.element("truss", 1, 1, 2, 1.0, 1)
ops.element("truss", 2, 2, 3, 1.0, 1)
ops.mass(2, 1.0, 0.0, 0.0)
ops.mass(3, 1.0, 0.0, 0.0)
ops.recorder("ladruno", path, "-N", "displacement", "modesOfVibration", "modesOfVibrationRotational")
print("lambda", ops.eigen("-fullGenLapack", 2))
ops.record()
ops.wipe()
"""


def modes() -> None:
    out = HERE / "ladruno_modes.ladruno"
    if out.exists():
        out.unlink()
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    subprocess.run([sys.executable, "-c", _MODES_RUN, str(out)], check=True, env=env)
    with h5py.File(out, "r") as f:
        on = f["MODEL_STAGE[1]/RESULTS/ON_NODES"]
        assert {"DISPLACEMENT", "MODES_OF_VIBRATION(U)", "MODES_OF_VIBRATION(R)"} <= set(on), list(on)


def frame3d() -> None:
    """Write ``ladruno_frame3d.h5`` and ``ladruno_frame3d.ladruno`` (paired 3-D 2-bay/2-story frame with floor slabs)."""
    model_out = HERE / "ladruno_frame3d.h5"
    ladruno_out = HERE / "ladruno_frame3d.ladruno"

    node_ids = np.arange(1, 16, dtype=np.int64)
    node_coords = np.array(
        [
            # Front frame y = 0.0: Story 1 (nodes 1..6), Story 2 (nodes 7..9)
            [0.0, 0.0, 0.0],  # 1
            [0.0, 0.0, 3.0],  # 2
            [4.0, 0.0, 3.0],  # 3
            [4.0, 0.0, 0.0],  # 4
            [8.0, 0.0, 0.0],  # 5
            [8.0, 0.0, 3.0],  # 6
            [0.0, 0.0, 6.0],  # 7
            [4.0, 0.0, 6.0],  # 8
            [8.0, 0.0, 6.0],  # 9
            # Rear slab corners y = 4.0: Story 1 (nodes 10..12), Story 2 (nodes 13..15)
            [0.0, 4.0, 3.0],  # 10
            [4.0, 4.0, 3.0],  # 11
            [8.0, 4.0, 3.0],  # 12
            [0.0, 4.0, 6.0],  # 13
            [4.0, 4.0, 6.0],  # 14
            [8.0, 4.0, 6.0],  # 15
        ],
        dtype=np.float64,
    )
    beam_ids = np.arange(1, 11, dtype=np.int64)
    beam_conn = np.array(
        [
            [1, 2],  # 1: Story 1 left col
            [2, 3],  # 2: Story 1 bay 1 girder
            [4, 3],  # 3: Story 1 mid col
            [3, 6],  # 4: Story 1 bay 2 girder (colinear with 2)
            [5, 6],  # 5: Story 1 right col
            [2, 7],  # 6: Story 2 left col (colinear with 1)
            [3, 8],  # 7: Story 2 mid col (colinear with 3)
            [6, 9],  # 8: Story 2 right col (colinear with 5)
            [7, 8],  # 9: Story 2 bay 1 girder
            [8, 9],  # 10: Story 2 bay 2 girder (colinear with 9)
        ],
        dtype=np.int64,
    )
    slab_ids = np.array([11, 12, 13, 14], dtype=np.int64)
    slab_conn = np.array(
        [
            [2, 3, 11, 10],  # 11: Story 1 bay 1 slab
            [3, 6, 12, 11],  # 12: Story 1 bay 2 slab
            [7, 8, 14, 13],  # 13: Story 2 bay 1 slab
            [8, 9, 15, 14],  # 14: Story 2 bay 2 slab
        ],
        dtype=np.int64,
    )

    with h5py.File(model_out, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["neutral_schema_version"] = "2.35.0"
        meta.attrs["opensees_schema_version"] = "2.24.0"
        meta.attrs["ndm"] = np.int64(3)
        meta.attrs["ndf"] = np.int64(6)
        lin = meta.create_group("lineage")
        lin.attrs["model_hash"] = STAMP["model_hash"]
        lin.attrs["fem_hash"] = STAMP["fem_hash"]
        lin.attrs["results_hash"] = "abcdef0123456789abcdef0123456789"

        nodes = f.create_group("nodes")
        nodes["ids"] = node_ids
        nodes["coords"] = node_coords

        elems = f.create_group("elements")
        line2 = elems.create_group("line2")
        line2.attrs["code"] = np.int64(1)
        line2.attrs["dim"] = np.int64(1)
        line2.attrs["npe"] = np.int64(2)
        line2["ids"] = beam_ids
        line2["connectivity"] = beam_conn

        quad4 = elems.create_group("quad4")
        quad4.attrs["code"] = np.int64(3)
        quad4.attrs["dim"] = np.int64(2)
        quad4.attrs["npe"] = np.int64(4)
        quad4["ids"] = slab_ids
        quad4["connectivity"] = slab_conn

        pg_side = f.create_group("physical_groups").create_group("element_side")
        cols_pg = pg_side.create_group("Columns")
        cols_pg.attrs["dim"] = np.int64(1)
        cols_pg.attrs["tag"] = np.int64(1)
        cols_pg["element_ids"] = np.array([1, 3, 5, 6, 7, 8], dtype=np.int64)
        girder_pg = pg_side.create_group("Girder")
        girder_pg.attrs["dim"] = np.int64(1)
        girder_pg.attrs["tag"] = np.int64(2)
        girder_pg["element_ids"] = np.array([2, 4, 9, 10], dtype=np.int64)
        slabs_pg = pg_side.create_group("Slabs")
        slabs_pg.attrs["dim"] = np.int64(2)
        slabs_pg.attrs["tag"] = np.int64(3)
        slabs_pg["element_ids"] = slab_ids

        ops = f.create_group("opensees")
        tr = ops.create_group("transforms").create_group("Linear_1")
        tr.attrs["type"] = "Linear"
        tr.attrs["tag"] = np.int64(1)
        tr.attrs["params"] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tr["per_element_vecxz"] = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        tr["per_element_emitted_tag"] = np.array([1], dtype=np.int64)

        em_grp = ops.create_group("element_meta")
        em = em_grp.create_group("elasticBeamColumn")
        em["ids"] = beam_ids
        em["fem_eids"] = beam_ids
        em["args"] = np.tile(
            np.array([[0.01, 2.0e11, 8.0e10, 1.0e-4, 1.0e-4, 2.0e-4, 1.0]], dtype=np.float64),
            (len(beam_ids), 1),
        )
        em_slab = em_grp.create_group("ShellMITC4")
        em_slab["ids"] = slab_ids
        em_slab["fem_eids"] = slab_ids
        em_slab["args"] = np.tile(np.array([[1.0]], dtype=np.float64), (len(slab_ids), 1))

    xi = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    time_arr = np.array([0.33, 0.67, 1.0], dtype=np.float64)
    step_arr = np.array([0, 1, 2], dtype=np.int32)
    with h5py.File(ladruno_out, "w") as f:
        info = f.create_group("INFO")
        info.attrs.create("GENERATOR", data="Ladruno", dtype=VLEN_UTF8)
        info.attrs["FORMAT_VERSION"] = np.int32(1)
        info.attrs["FORMAT_MINOR_VERSION"] = np.int32(1)
        info.attrs["SPATIAL_DIM"] = np.int32(3)
        prov = info.create_group("PROVENANCE")
        frame_stamp = {
            k: v
            for k, v in STAMP.items()
            if k != "element_index_space"
        }
        frame_stamp["stage_names"] = json.dumps(["gravity", "lateral"], separators=(",", ":"))
        for k, v in frame_stamp.items():
            prov.attrs.create(k, data=v, dtype=VLEN_UTF8)

        s_quat = float(np.sqrt(0.5))
        n_nodes = len(node_ids)
        n_beams = len(beam_ids)
        for s_idx, stage_kind in enumerate(["gravity", "lateral"], start=1):
            st = f.create_group(f"MODEL_STAGE[{s_idx}]")
            st.attrs.create("KIND", data="static", dtype=VLEN_UTF8)

            model_grp = st.create_group("MODEL")
            nodes_grp = model_grp.create_group("NODES")
            nodes_grp.create_dataset("COORDINATES", data=node_coords)
            elems_grp = model_grp.create_group("ELEMENTS")
            fbc_model = elems_grp.create_group("20-ForceBeamColumn3d[0]")
            quad_grp = fbc_model.create_group("QUADRATURE")
            quad_grp.create_dataset("GP_PARAM", data=xi.reshape(3, 1))

            la_grp = model_grp.create_group("LOCAL_AXES").create_group("ForceBeamColumn3d")
            la_grp.create_dataset("ID", data=beam_ids.astype(np.int32))
            # 90 deg rotation about X -> local z_rec = [0, 1, 0], matching Linear_1 vecxz
            la_grp.create_dataset(
                "FRAME",
                data=np.tile(np.array([[s_quat, s_quat, 0.0, 0.0]], dtype=np.float64), (n_beams, 1)),
            )

            res = st.create_group("RESULTS")
            on_nodes = res.create_group("ON_NODES")
            disp_grp = on_nodes.create_group("DISPLACEMENT")
            disp_grp.attrs.create("COMPONENTS", data="Ux,Uy,Uz,Rx,Ry,Rz", dtype=VLEN_UTF8)
            disp_grp.create_dataset("ID", data=node_ids.astype(np.int32).reshape(-1, 1))
            disp_grp.create_dataset("TIME", data=time_arr)
            disp_grp.create_dataset("STEP", data=step_arr)
            disp_data = np.zeros((3, n_nodes, 6), dtype=np.float64)
            for step_i, lf in enumerate([0.33, 0.67, 1.0]):
                sway = 0.015 * lf if stage_kind == "lateral" else 0.0
                for n_i, (_, _, z_coord) in enumerate(node_coords):
                    if z_coord > 0.0:
                        story_factor = z_coord / 3.0
                        disp_data[step_i, n_i, 0] = sway * story_factor
                        disp_data[step_i, n_i, 2] = -0.02 * lf * story_factor
            disp_grp.create_dataset("DATA", data=disp_data)

            on_elems = res.create_group("ON_ELEMENTS")
            sf_grp = on_elems.create_group("section.force")
            sf_block = sf_grp.create_group("20-ForceBeamColumn3d[0]")
            sf_block.create_dataset("ID", data=beam_ids.astype(np.int32))
            sf_block.create_dataset("TIME", data=time_arr)
            sf_block.create_dataset("STEP", data=step_arr)
            # 3 steps x 10 elements x (3 stations * 6 components [N, Vy, Vz, T, My, Mz] = 18 cols)
            sf_data = np.zeros((3, n_beams, 18), dtype=np.float64)
            col_indices = {0, 2, 4, 5, 6, 7}  # 0-based indices for elements 1, 3, 5, 6, 7, 8
            for step_i, lf in enumerate([0.33, 0.67, 1.0]):
                for b_i in range(n_beams):
                    if b_i in col_indices:
                        sign = 1.0 if b_i in {0, 5} else -1.0
                        story_scale = 1.0 if b_i < 5 else 0.65
                        for st_i, x_val in enumerate(xi):
                            c0 = st_i * 6
                            sf_data[step_i, b_i, c0 : c0 + 6] = [
                                -30.0 * lf * story_scale,
                                sign * 15.0 * lf * story_scale,
                                sign * 2.0 * lf * story_scale,
                                sign * 1.0 * lf * story_scale,
                                sign * 3.0 * lf * story_scale * x_val,
                                sign * 45.0 * lf * story_scale * x_val,
                            ]
                    else:
                        story_scale = 1.0 if b_i in {1, 3} else 0.75
                        mz_girder = [-30.0 * lf * story_scale, 60.0 * lf * story_scale, -30.0 * lf * story_scale]
                        vy_girder = [30.0 * lf * story_scale, 0.0, -30.0 * lf * story_scale]
                        for st_i, x_val in enumerate(xi):
                            c0 = st_i * 6
                            sf_data[step_i, b_i, c0 : c0 + 6] = [
                                -15.0 * lf * story_scale,
                                vy_girder[st_i],
                                1.5 * lf * story_scale * x_val,
                                0.5 * lf * story_scale,
                                4.0 * lf * story_scale * (1.0 - x_val * x_val),
                                mz_girder[st_i],
                            ]
            sf_block.create_dataset("DATA", data=sf_data)
            sf_map = sf_block.create_group("COLUMN_MAP")
            sf_map.attrs.create(
                "COMP_NAMES",
                data="N,Vy,Vz,T,My,Mz\nN,Vy,Vz,T,My,Mz\nN,Vy,Vz,T,My,Mz",
                dtype=VLEN_UTF8,
            )
            sf_map.create_dataset("LEVELS", data=np.array([2, 2, 2], dtype=np.int32))
            sf_map.create_dataset("GAUSS_ID", data=np.array([0, 1, 2], dtype=np.int32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", action="store_true", help="also run the fork to write ladruno_modes.ladruno")
    args = ap.parse_args()
    truss2d()
    stamped()
    badstamp()
    frame3d()
    if args.modes:
        modes()


if __name__ == "__main__":
    main()
