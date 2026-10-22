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
   station values, element FEM ids, natural coordinates and local-axes
   quaternions across all six fixtures.
"""

from __future__ import annotations

import json
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

    # 4. Vertical 3-D column with vecxz=(1,0,0) and the degenerate vecxz=(0,0,1) refusal probe.
    col = _load_golden("line_oracle_column3d")
    col_el0 = col["elements"][0]
    assert col_el0["frame"] == {"x": [0.0, 0.0, 1.0], "y": [0.0, -1.0, 0.0], "z": [1.0, 0.0, -0.0]}
    col_st = col_el0["buckets"]["localForce"]["stations"]
    np.testing.assert_allclose(col_st["axial_force"]["values"][-1], [-10.0, -10.0], rtol=1e-9)
    np.testing.assert_allclose(col_st["shear_y"]["values"][-1], [3.0, 3.0], rtol=1e-9)
    np.testing.assert_allclose(col_st["shear_z"]["values"][-1], [2.0, 2.0], rtol=1e-9)
    np.testing.assert_allclose(col_st["torsion"]["values"][-1], [0.5, 0.5], rtol=1e-9)
    np.testing.assert_allclose(col_st["bending_moment_y"]["values"][-1], [-6.0, -3.0], rtol=1e-9)
    np.testing.assert_allclose(col_st["bending_moment_z"]["values"][-1], [-9.0, -4.5], rtol=1e-9)
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
    la = r_raw.elements.local_axes()
    assert la.element_ids.tolist() == [el["ops_tag"] for el in golden["elements"]]
    for idx, el in enumerate(golden["elements"]):
        np.testing.assert_allclose(la.quaternions[idx], el["frame_quaternion_wxyz"], atol=1e-12)
        want_rows = np.array([el["frame"]["x"], el["frame"]["y"], el["frame"]["z"]])
        np.testing.assert_allclose(la.matrices[idx], want_rows, atol=1e-12)

    r = Results.from_ladruno(
        FIXTURES / golden["ladruno"],
        model_h5=FIXTURES / golden["model_h5"],
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
        np.testing.assert_allclose(slab_raw.local_axes_quaternion, want_quats, atol=1e-12)
