"""Pin the OpenSees-solved line-diagram oracle fixtures and orientation probe (#2471).

Reads the committed ``apeGmshViewer/fixtures/line_oracle_*.{h5,ladruno,golden.json}``
triples produced by ``tests/fixtures/ladruno/make_line_diagram_oracle.py`` (no
Ladruno fork needed at test time, and no import across the ADR 0112 D4 app
wall). Checks:

1. Every fixture file is present and within the 200 KB size budget.
2. Each ``.ladruno`` is paired to its sibling ``model.h5`` by ``stamp_provenance``
   (``model_hash`` and ``fem_hash`` match byte for byte).
3. The orientation probe holds on every fixture and in an in-memory emit check:
   ``apeGmsh`` / ``apeSees`` never emits a beam whose ``(iNode, jNode)`` is not
   its gmsh line cell's ``(corner_0, corner_1)``; curves drawn against ``+x``
   (``line_oracle_ss_udl`` group ``Rev``) produce cells and elements whose
   ``i -> j`` runs right-to-left.
4. Every station value in ``.golden.json`` agrees with its closed-form free-body
   statics resultant at ``rtol=1e-9``, and the headline analytical values from
   ``#2471`` are pinned explicitly.
5. ``Results.from_ladruno(ladruno, model_h5=model_h5)`` reproduces the golden
   station values, element FEM ids, natural coordinates and ``MODEL/LOCAL_AXES``
   frame quaternions across all six fixtures (including ``line_oracle_ss_udl``
   where ``fem_id != ops_tag``), alongside ``Results.from_ladruno(ladruno)``
   (``r_raw``, keyed by ``ops_tag``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._response_catalog import _LOCAL_FORCE_END_I_SIGN
from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._tag_translation import ElementTagTranslator

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "apeGmshViewer" / "fixtures"
MAX_BYTES = 200 * 1024
STEMS = (
    "line_oracle_ss_udl",
    "line_oracle_cantilever3d",
    "line_oracle_fbc_legendre",
    "line_oracle_fbc_lobatto",
    "line_oracle_column3d",
    "line_oracle_basic_only",
)


def _text(attr: Any) -> str:
    v = np.asarray(attr).reshape(-1)[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def _load_golden(stem: str) -> dict[str, Any]:
    path = FIXTURES / f"{stem}.golden.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("stem", STEMS)
def test_oracle_fixture_files_exist_and_stay_under_size_budget(stem: str) -> None:
    for ext in (".h5", ".ladruno", ".golden.json"):
        path = FIXTURES / f"{stem}{ext}"
        assert path.is_file(), f"missing {path}"
        size = path.stat().st_size
        assert 0 < size < MAX_BYTES, f"{path.name} is {size} bytes (budget < {MAX_BYTES})"


@pytest.mark.parametrize("stem", STEMS)
def test_oracle_provenance_and_lineage_pair_model_and_ladruno(stem: str) -> None:
    golden = _load_golden(stem)
    assert golden["fixture"] == stem
    assert golden["model_h5"] == f"{stem}.h5"
    assert golden["ladruno"] == f"{stem}.ladruno"

    with h5py.File(FIXTURES / golden["model_h5"], "r") as mf:
        lineage = mf["meta/lineage"].attrs
        model_hash = _text(lineage["model_hash"])
        fem_hash = _text(lineage["fem_hash"])

    with h5py.File(FIXTURES / golden["ladruno"], "r") as lf:
        prov = lf["INFO/PROVENANCE"].attrs
        assert sorted(prov) == golden["ladruno_provenance_keys"]
        assert _text(prov["model_hash"]) == model_hash
        assert _text(prov["fem_hash"]) == fem_hash
        assert _text(prov["opensees_build_sha"]) == golden["opensees_build_sha"]
        assert "DISPLACEMENT" in lf["MODEL_STAGE[1]/RESULTS/ON_NODES"]


@pytest.mark.parametrize("stem", STEMS)
def test_element_orientation_probe_matches_cell_corners_on_every_fixture(stem: str) -> None:
    golden = _load_golden(stem)
    with h5py.File(FIXTURES / golden["model_h5"], "r") as mf:
        cells = {
            int(e): [int(c) for c in conn]
            for grp in mf["elements"].values()
            for e, conn in zip(grp["ids"][()], grp["connectivity"][()])
        }
        fem_to_ops = {
            int(e): int(t)
            for tok in mf["opensees/element_meta"]
            for t, e in zip(
                mf["opensees/element_meta"][tok]["ids"][()],
                mf["opensees/element_meta"][tok]["fem_eids"][()],
            )
        }
    with h5py.File(FIXTURES / golden["ladruno"], "r") as lf:
        ladruno_nodes = {
            int(row[0]): [int(c) for c in row[1:]]
            for grp in lf["MODEL_STAGE[1]/MODEL/ELEMENTS"].values()
            for row in grp["CONNECTIVITY"][()]
        }

    assert len(golden["elements"]) == len(cells)
    for el in golden["elements"]:
        eid = el["fem_id"]
        tag = el["ops_tag"]
        assert fem_to_ops[eid] == tag
        assert el["nodes"] == cells[eid]
        assert ladruno_nodes[tag] == cells[eid]


def test_apesees_emits_beam_nodes_in_exact_fem_cell_corner_order(tmp_path: Path) -> None:
    """Direct emitter probe: forward and reversed line cells keep (corner_0, corner_1)."""
    from apeGmsh.opensees import OpenSeesModel

    node_ids = np.array([1, 2, 3], dtype=np.int64)
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2)
    # Cell 10 runs 1 -> 2 (+x); cell 20 runs 3 -> 2 (-x, iNode > jNode).
    conn = np.array([[1, 2], [3, 2]], dtype=np.int64)
    eids = np.array([10, 20], dtype=np.int64)
    group = ElementGroup(element_type=info, ids=eids, connectivity=conn)
    pg = {(1, 1): {"name": "Beams", "node_ids": node_ids, "node_coords": coords, "element_ids": eids}}
    fem = FEMData(
        nodes=NodeComposite(node_ids=node_ids, node_coords=coords,
                            physical=PhysicalGroupSet(pg), labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=PhysicalGroupSet(pg), labels=LabelSet({})),
        info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=2, types=[info]),
    )
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Beams", transf=tr, E=2.0e8, A=0.01, Iz=1.0e-4)

    out_h5 = tmp_path / "probe.h5"
    ops.h5(str(out_h5))
    tcl_lines = [
        ln.split()
        for ln in OpenSeesModel.from_h5(str(out_h5)).build("tcl").splitlines()
        if ln.startswith("element elasticBeamColumn ")
    ]
    # element elasticBeamColumn $eleTag $iNode $jNode ...
    assert [(int(toks[3]), int(toks[4])) for toks in tcl_lines] == [(1, 2), (3, 2)]


@pytest.mark.parametrize("stem", STEMS)
def test_golden_stations_agree_with_closed_form_statics_and_sign_table(stem: str) -> None:
    golden = _load_golden(stem)
    assert golden["sign_convention"]["end_i_sign"] == dict(sorted(_LOCAL_FORCE_END_I_SIGN.items()))
    for el in golden["elements"]:
        for bucket in el["buckets"].values():
            assert bucket["step"] == [0, 1]
            assert bucket["time"] == [0.5, 1.0]
            for comp, st in bucket["stations"].items():
                np.testing.assert_allclose(
                    st["values"], st["closed_form"], rtol=1e-9, atol=1e-9,
                    err_msg=f"{stem} fem_id={el['fem_id']} {comp}",
                )


def test_headline_closed_form_pins() -> None:
    # 1. SS beam under UDL: Mz(L/2) = wL^2/8 = 45 on tension side; Rev runs against +x.
    ss = _load_golden("line_oracle_ss_udl")
    by_id = {el["fem_id"]: el for el in ss["elements"]}
    assert by_id[4]["frame"]["x"] == [1.0, 0.0, 0.0]
    assert by_id[6]["frame"]["x"] == [-1.0, 0.0, 0.0]
    assert by_id[6]["frame"]["y"] == [0.0, -1.0, 0.0]
    assert by_id[4]["coords_i"][0] < by_id[4]["coords_j"][0]
    assert by_id[6]["coords_i"][0] > by_id[6]["coords_j"][0]
    # Midspan (x = 3.0) is node j (xi = +1) of element 5 (Fwd) and element 7 (Rev).
    mz_fwd = by_id[5]["buckets"]["localForce"]["stations"]["bending_moment_z"]["values"][-1]
    mz_rev = by_id[7]["buckets"]["localForce"]["stations"]["bending_moment_z"]["values"][-1]
    np.testing.assert_allclose(mz_fwd, [33.75, 45.0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mz_rev, [-33.75, -45.0], rtol=1e-9, atol=1e-9)

    # 2. 3-D cantilever: My vs Mz plane, Vz = +Fz, T = +Mx.
    ct = _load_golden("line_oracle_cantilever3d")
    root_st = ct["elements"][0]["buckets"]["localForce"]["stations"]
    np.testing.assert_allclose(root_st["axial_force"]["values"][-1], [1.0, 1.0], rtol=1e-9)
    np.testing.assert_allclose(root_st["shear_y"]["values"][-1], [-2.0, -2.0], rtol=1e-9)
    np.testing.assert_allclose(root_st["shear_z"]["values"][-1], [3.0, 3.0], rtol=1e-9)
    np.testing.assert_allclose(root_st["torsion"]["values"][-1], [0.4, 0.4], rtol=1e-9)
    np.testing.assert_allclose(root_st["bending_moment_y"]["values"][-1], [-12.0, -6.0], rtol=1e-9)
    np.testing.assert_allclose(root_st["bending_moment_z"]["values"][-1], [8.0, 4.0], rtol=1e-9)

    # 3. Force-based beam: Legendre vs Lobatto abscissae and the #2480 GP_PARAM finding.
    leg = _load_golden("line_oracle_fbc_legendre")["elements"][0]["buckets"]["section.force"]
    assert leg["gp_param_matches_true_xi"] is False
    np.testing.assert_allclose(
        leg["stations"]["bending_moment_z"]["xi"],
        [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)],
        atol=1e-12,
    )
    np.testing.assert_allclose(leg["gp_param_by_gauss_id"], [-1.0, 0.0, 1.0], atol=1e-12)

    lob = _load_golden("line_oracle_fbc_lobatto")["elements"][0]["buckets"]["section.force"]
    assert lob["gp_param_matches_true_xi"] is True
    np.testing.assert_allclose(
        lob["stations"]["bending_moment_z"]["xi"],
        [-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(lob["gp_param_by_gauss_id"], lob["stations"]["bending_moment_z"]["xi"], atol=1e-12)

    # 4. Vertical 3-D column with vecxz=(0,1,0) (asymmetric R != R.T) and the degenerate vecxz=(0,0,1) refusal probe.
    col = _load_golden("line_oracle_column3d")
    col_el0 = col["elements"][0]
    assert col_el0["frame"] == {"x": [0.0, 0.0, 1.0], "y": [1.0, 0.0, 0.0], "z": [0.0, 1.0, 0.0]}
    col_r = np.array([col_el0["frame"]["x"], col_el0["frame"]["y"], col_el0["frame"]["z"]])
    assert not np.allclose(col_r, col_r.T)
    np.testing.assert_allclose(col_el0["frame_quaternion_wxyz"], [0.5, 0.5, 0.5, 0.5], atol=1e-12)
    col_st = col_el0["buckets"]["localForce"]["stations"]
    np.testing.assert_allclose(col_st["axial_force"]["values"][-1], [-10.0, -10.0], rtol=1e-9)
    np.testing.assert_allclose(col_st["shear_y"]["values"][-1], [-2.0, -2.0], rtol=1e-9)
    np.testing.assert_allclose(col_st["shear_z"]["values"][-1], [3.0, 3.0], rtol=1e-9)
    np.testing.assert_allclose(col_st["torsion"]["values"][-1], [0.5, 0.5], rtol=1e-9)
    np.testing.assert_allclose(col_st["bending_moment_y"]["values"][-1], [-9.0, -4.5], rtol=1e-9)
    np.testing.assert_allclose(col_st["bending_moment_z"]["values"][-1], [6.0, 3.0], rtol=1e-9)
    deg = col["degenerate_vecxz"]
    assert deg["vecxz"] == [0.0, 0.0, 1.0]
    assert deg["opensees_accepts_it"] is False
    assert any("LinearCrdTransf3d" in s for s in deg["opensees_said"])

    # 5. basicForce-only element: [-M_1, +M_2] at xi=-1,+1 and N, T at xi=0.
    bo = _load_golden("line_oracle_basic_only")["elements"][0]["buckets"]["basicForce"]["stations"]
    assert bo["bending_moment_z"]["xi"] == [-1.0, 1.0]
    assert bo["bending_moment_y"]["xi"] == [-1.0, 1.0]
    assert bo["axial_force"]["xi"] == [0.0]
    assert bo["torsion"]["xi"] == [0.0]
    np.testing.assert_allclose(bo["bending_moment_z"]["values"][-1], [8.6, 4.6], rtol=1e-9)
    np.testing.assert_allclose(bo["bending_moment_y"]["values"][-1], [-11.5, -5.5], rtol=1e-9)
    np.testing.assert_allclose(bo["axial_force"]["values"][-1], [1.0], rtol=1e-9)
    np.testing.assert_allclose(bo["torsion"]["values"][-1], [0.4], rtol=1e-9)


@pytest.mark.parametrize("stem", STEMS)
def test_results_from_ladruno_matches_golden_stations_and_frames(stem: str) -> None:
    golden = _load_golden(stem)
    r_raw = Results.from_ladruno(FIXTURES / golden["ladruno"])
    la_raw = r_raw.elements.local_axes()
    assert la_raw.element_ids.tolist() == [el["ops_tag"] for el in golden["elements"]]
    for idx, el in enumerate(golden["elements"]):
        assert el["frame_quaternion_rows"] == "local x, y, z axes (R rows, global components)"
        np.testing.assert_allclose(la_raw.quaternions[idx], el["frame_quaternion_wxyz"], atol=1e-12)
        want_rows = np.array([el["frame"]["x"], el["frame"]["y"], el["frame"]["z"]])
        np.testing.assert_allclose(la_raw.matrices[idx], want_rows, atol=1e-12)
        if stem == "line_oracle_column3d":
            assert not np.allclose(la_raw.matrices[idx], la_raw.matrices[idx].T)

    r = Results.from_ladruno(
        FIXTURES / golden["ladruno"],
        model_h5=FIXTURES / golden["model_h5"],
    )
    la = r.elements.local_axes()
    assert la.element_ids.tolist() == [el["fem_id"] for el in golden["elements"]]
    np.testing.assert_allclose(la.quaternions, la_raw.quaternions, atol=1e-12)
    for el in golden["elements"]:
        la_one = r.elements.local_axes(ids=[el["fem_id"]])
        assert la_one.element_ids.tolist() == [el["fem_id"]]
        np.testing.assert_allclose(la_one.quaternions[0], el["frame_quaternion_wxyz"], atol=1e-12)

    if stem == "line_oracle_ss_udl":
        la_rev = r.elements.local_axes(pg="Rev")
        assert la_rev.element_ids.tolist() == [6, 7]
        np.testing.assert_allclose(
            la_rev.quaternions,
            [el["frame_quaternion_wxyz"] for el in golden["elements"][2:]],
            atol=1e-12,
        )
        slab_rev = r.elements.line_stations.get(component="bending_moment_z", pg="Rev")
        assert slab_rev.element_index.tolist() == [6, 6, 7, 7]
        assert slab_rev.local_axes_quaternion is not None
        np.testing.assert_allclose(
            slab_rev.local_axes_quaternion,
            [el["frame_quaternion_wxyz"] for el in golden["elements"][2:] for _ in range(2)],
            atol=1e-12,
        )

    # Components across the recorded buckets, ranked section.force -> localForce -> basicForce.
    all_comps: list[str] = []
    for resp in golden["responses"]:
        for comp in golden["elements"][0]["buckets"][resp]["stations"]:
            if comp not in all_comps:
                all_comps.append(comp)

    for comp in all_comps:
        slab = r.elements.line_stations.get(component=comp)
        slab_raw = r_raw.elements.line_stations.get(component=comp)
        assert slab.local_axes_quaternion is not None
        assert slab_raw.local_axes_quaternion is not None
        want_eids: list[int] = []
        want_tags: list[int] = []
        want_xi: list[float] = []
        want_vals: list[np.ndarray] = []
        want_quats: list[list[float]] = []
        for el in golden["elements"]:
            # Pick the highest-priority bucket on this element that serves `comp`.
            chosen_resp = next(resp for resp in golden["responses"] if comp in el["buckets"][resp]["stations"])
            bkt = el["buckets"][chosen_resp]
            st = bkt["stations"][comp]
            # LadrunoReader reads QUADRATURE/GP_PARAM for section.force stations.
            xi = (
                bkt["gp_param_by_gauss_id"]
                if chosen_resp == "section.force"
                else st["xi"]
            )
            n_st = len(xi)
            want_eids.extend([el["fem_id"]] * n_st)
            want_tags.extend([el["ops_tag"]] * n_st)
            want_xi.extend(xi)
            want_vals.append(np.asarray(st["values"], dtype=float))
            want_quats.extend([el["frame_quaternion_wxyz"]] * n_st)

        assert slab.element_index.tolist() == want_eids
        assert slab_raw.element_index.tolist() == want_tags
        np.testing.assert_allclose(slab.station_natural_coord, want_xi, atol=1e-12)
        np.testing.assert_allclose(slab.values, np.concatenate(want_vals, axis=1), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(slab.local_axes_quaternion, want_quats, atol=1e-12)
        np.testing.assert_allclose(slab_raw.local_axes_quaternion, want_quats, atol=1e-12)


def _write_four_beam_ladruno(
    dst: Path,
    ops_tags: np.ndarray,
    frames: np.ndarray,
    *,
    part_info: tuple[int, int] | None = None,
) -> Path:
    """Build a 4-beam (or N-beam) .ladruno from committed ladruno_portal3d.ladruno."""
    shutil.copyfile(FIXTURES / "ladruno_portal3d.ladruno", dst)
    n = int(ops_tags.size)
    with h5py.File(dst, "r+") as f:
        if part_info is not None:
            pid, nparts = part_info
            f["INFO"].attrs["PARTITIONED"] = 1
            f["INFO"].attrs["PARTITION_ID"] = pid
            f["INFO"].attrs["NUM_PARTITIONS"] = nparts
        stg = f["MODEL_STAGE[1]"]
        del stg["MODEL/SECTION_ASSIGNMENTS"]
        del stg["RESULTS/ON_ELEMENTS/section.force"]
        for grp in stg["MODEL/ELEMENTS"].values():
            del grp["CONNECTIVITY"], grp["GLOBAL_GP_COORDS"]
            conn = np.column_stack([
                ops_tags.astype(np.int32),
                np.ones((n, 1), dtype=np.int32),
                np.full((n, 1), 2, dtype=np.int32),
            ])
            grp.create_dataset("CONNECTIVITY", data=conn)
        for grp in stg["MODEL/LOCAL_AXES"].values():
            del grp["ID"], grp["FRAME"]
            grp.create_dataset("ID", data=ops_tags.astype(np.int64))
            grp.create_dataset("FRAME", data=np.asarray(frames, dtype=np.float64))
        for bkt in stg["RESULTS/ON_ELEMENTS/localForce"].values():
            if isinstance(bkt, h5py.Group) and "ID" in bkt:
                base_data = bkt["DATA"][:, :1, :]
                del bkt["ID"], bkt["DATA"]
                bkt.create_dataset("ID", data=ops_tags.astype(np.int32).reshape(n, 1))
                bkt.create_dataset("DATA", data=np.repeat(base_data, n, axis=1))
    return dst


def test_ladruno_line_station_quaternions_disjoint_and_offset_tag_maps(tmp_path: Path) -> None:
    """Pin #2485 across offset, disjoint, partial, LOCAL_AXES-only, and partitioned tag maps."""
    from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader

    elem_quats = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
    ], dtype=np.float64)
    want_quats = np.repeat(elem_quats, 2, axis=0)

    # 1. Offset overlapping tags (fem_id = [4, 5, 6, 7] vs ops_tag = [3, 4, 5, 6]):
    # fem_id=5 (ops_tag=4, +x) must NOT receive ops_tag=5's -x quaternion,
    # and fem_id=7 (ops_tag=6) must NOT receive NaN.
    offset_path = _write_four_beam_ladruno(
        tmp_path / "offset_tags.ladruno",
        np.array([3, 4, 5, 6], dtype=np.int64),
        elem_quats,
    )
    offset_f2o = {4: 3, 5: 4, 6: 5, 7: 6}
    offset_o2f = {t: e for e, t in offset_f2o.items()}
    with LadrunoReader(offset_path) as reader:
        reader.attach_tag_map(ElementTagTranslator(offset_f2o, offset_o2f))
        stage_id = reader.stages()[0].id
        slab_off = reader.read_line_stations(stage_id, "bending_moment_z")
        assert slab_off.element_index.tolist() == [4, 4, 5, 5, 6, 6, 7, 7]
        assert slab_off.local_axes_quaternion is not None
        np.testing.assert_allclose(slab_off.local_axes_quaternion, want_quats, atol=1e-12)

    # 2. Disjoint tags (fem_id = [1, 2, 3, 4] vs ops_tag = [101, 102, 103, 104]).
    path = _write_four_beam_ladruno(
        tmp_path / "disjoint_tags.ladruno",
        np.array([101, 102, 103, 104], dtype=np.int64),
        elem_quats,
    )
    fem_to_ops = {1: 101, 2: 102, 3: 103, 4: 104}
    ops_to_fem = {t: e for e, t in fem_to_ops.items()}
    with LadrunoReader(path) as reader:
        reader.attach_tag_map(ElementTagTranslator(fem_to_ops, ops_to_fem))
        stage_id = reader.stages()[0].id
        slab = reader.read_line_stations(stage_id, "bending_moment_z")
        assert slab.element_index.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
        assert slab.local_axes_quaternion is not None
        assert not np.isnan(slab.local_axes_quaternion).any()
        np.testing.assert_allclose(slab.local_axes_quaternion, want_quats, atol=1e-12)

        la = reader.read_local_axes(stage_id)
        assert la.element_ids.tolist() == [1, 2, 3, 4]
        np.testing.assert_allclose(la.quaternions, elem_quats, atol=1e-12)
        la_rev = reader.read_local_axes(stage_id, element_ids=np.array([3, 4]))
        assert la_rev.element_ids.tolist() == [3, 4]
        np.testing.assert_allclose(la_rev.quaternions, elem_quats[2:], atol=1e-12)

    # 3. Unrelated stub translator (does not describe the file): IDs pass through as ops_tag,
    # including when a file records element tags only under MODEL/LOCAL_AXES.
    with LadrunoReader(path) as reader:
        reader.attach_tag_map(ElementTagTranslator({10: 999}, {999: 10}))
        stage_id = reader.stages()[0].id
        slab_stub = reader.read_line_stations(stage_id, "bending_moment_z")
        assert slab_stub.element_index.tolist() == [101, 101, 102, 102, 103, 103, 104, 104]
        assert slab_stub.local_axes_quaternion is not None
        np.testing.assert_allclose(slab_stub.local_axes_quaternion, want_quats, atol=1e-12)
        la_stub = reader.read_local_axes(stage_id)
        assert la_stub.element_ids.tolist() == [101, 102, 103, 104]

    axes_only_path = tmp_path / "axes_only.ladruno"
    shutil.copyfile(path, axes_only_path)
    with h5py.File(axes_only_path, "r+") as f:
        del f["MODEL_STAGE[1]/MODEL/ELEMENTS"]
        del f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
    with LadrunoReader(axes_only_path) as reader:
        reader.attach_tag_map(ElementTagTranslator({10: 101}, {101: 10}))
        la_axes_only = reader.read_local_axes(reader.stages()[0].id)
        assert la_axes_only.element_ids.tolist() == [101, 102, 103, 104]

    # 4. Partial LOCAL_AXES (element 104 / fem_id 4 has no recorded frame):
    # its station rows get NaN quaternions and read_local_axes(element_ids=[1, 4]) pads identity.
    partial_path = tmp_path / "partial_axes.ladruno"
    shutil.copyfile(path, partial_path)
    with h5py.File(partial_path, "r+") as f:
        for grp in f["MODEL_STAGE[1]/MODEL/LOCAL_AXES"].values():
            ids_3 = grp["ID"][:3]
            frames_3 = grp["FRAME"][:3]
            del grp["ID"], grp["FRAME"]
            grp.create_dataset("ID", data=ids_3)
            grp.create_dataset("FRAME", data=frames_3)

    with LadrunoReader(partial_path) as reader:
        reader.attach_tag_map(ElementTagTranslator(fem_to_ops, ops_to_fem))
        stage_id = reader.stages()[0].id
        slab_part = reader.read_line_stations(stage_id, "bending_moment_z")
        assert slab_part.element_index.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
        assert slab_part.local_axes_quaternion is not None
        np.testing.assert_allclose(slab_part.local_axes_quaternion[:6], want_quats[:6], atol=1e-12)
        assert np.isnan(slab_part.local_axes_quaternion[6:]).all()

        la_part = reader.read_local_axes(stage_id)
        assert la_part.element_ids.tolist() == [1, 2, 3]
        la_padded = reader.read_local_axes(stage_id, element_ids=np.array([1, 4]))
        assert la_padded.element_ids.tolist() == [1, 4]
        np.testing.assert_allclose(la_padded.quaternions[0], elem_quats[0], atol=1e-12)
        np.testing.assert_allclose(la_padded.quaternions[1], [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    # 5. Multi-partition .ladruno (LadrunoMultiPartitionReader) with non-identity translator.
    part0 = _write_four_beam_ladruno(
        tmp_path / "mp.part-0.ladruno",
        np.array([101, 102], dtype=np.int64),
        elem_quats[:2],
        part_info=(0, 2),
    )
    part1 = _write_four_beam_ladruno(
        tmp_path / "mp.part-1.ladruno",
        np.array([103, 104], dtype=np.int64),
        elem_quats[2:],
        part_info=(1, 2),
    )
    with LadrunoMultiPartitionReader([part0, part1]) as mp_reader:
        mp_reader.attach_tag_map(ElementTagTranslator(fem_to_ops, ops_to_fem))
        stage_id = mp_reader.stages()[0].id
        slab_mp = mp_reader.read_line_stations(stage_id, "bending_moment_z")
        assert slab_mp.element_index.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
        assert slab_mp.local_axes_quaternion is not None
        np.testing.assert_allclose(slab_mp.local_axes_quaternion, want_quats, atol=1e-12)
