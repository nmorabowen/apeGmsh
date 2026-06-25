"""Phase 2 regression: StructuralModel JSON -> apeGmsh geometry -> beam mesh
-> OpenSees deck. Self-contained (no apeETABS / live ETABS dependency).

See apeETABS ADR 0009 + build-plan W3 Phase 2.
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh.interop import (
    StructuralModel,
    apply_subgrade_springs,
    build_opensees,
    import_structural_model,
)

# Minimal model: one vertical column + one horizontal beam exercises both
# orientation buckets and both PG-naming branches.
_MODEL = {
    "schema_version": "0.1",
    "units": {"length": "m", "force": "kN"},
    "nodes": [
        {"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "2", "x": 0.0, "y": 0.0, "z": 3.0},
        {"id": "3", "x": 4.0, "y": 0.0, "z": 3.0},
    ],
    "frames": [
        {"id": "C1", "i": "1", "j": "2", "section": "COL", "kind": "column"},
        {"id": "B1", "i": "2", "j": "3", "section": "BEAM", "kind": "beam"},
    ],
    "sections": [
        {"name": "COL", "kind": "frame", "material": "M",
         "props": {"A": 0.16, "Iy": 2.1e-3, "Iz": 2.1e-3, "J": 3.6e-3}},
        {"name": "BEAM", "kind": "frame", "material": "M",
         "props": {"A": 0.12, "Iy": 9.0e-4, "Iz": 1.6e-3, "J": 1.8e-3}},
    ],
    "materials": [{"name": "M", "E": 2.5e7, "nu": 0.2}],
    "restraints": [{"node": "1", "dofs": [1, 1, 1, 1, 1, 1]}],
    "loads": {"Live": {"nodal": [{"node": "3", "force_xyz": [0.0, 0.0, -10.0]}]}},
}


# Wall + slab + frame box: exercises shared-edge conformality (the Phase 3
# crux). Wall W1 shares edge 5-6 with slab S1; beams sit on slab edges;
# columns meet slab corners. A conformal mesh has NO coincident duplicate nodes.
_BOX = {
    "schema_version": "0.1",
    "units": {"length": "m", "force": "kN"},
    "nodes": [
        {"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "2", "x": 4.0, "y": 0.0, "z": 0.0},
        {"id": "3", "x": 4.0, "y": 4.0, "z": 0.0},
        {"id": "4", "x": 0.0, "y": 4.0, "z": 0.0},
        {"id": "5", "x": 0.0, "y": 0.0, "z": 3.0},
        {"id": "6", "x": 4.0, "y": 0.0, "z": 3.0},
        {"id": "7", "x": 4.0, "y": 4.0, "z": 3.0},
        {"id": "8", "x": 0.0, "y": 4.0, "z": 3.0},
    ],
    "frames": [
        {"id": "C1", "i": "1", "j": "5", "section": "COL", "kind": "column"},
        {"id": "C2", "i": "2", "j": "6", "section": "COL", "kind": "column"},
        {"id": "C3", "i": "3", "j": "7", "section": "COL", "kind": "column"},
        {"id": "C4", "i": "4", "j": "8", "section": "COL", "kind": "column"},
        {"id": "B1", "i": "6", "j": "7", "section": "BEAM", "kind": "beam"},
        {"id": "B2", "i": "7", "j": "8", "section": "BEAM", "kind": "beam"},
        {"id": "B3", "i": "8", "j": "5", "section": "BEAM", "kind": "beam"},
    ],
    "areas": [
        {"id": "S1", "nodes": ["5", "6", "7", "8"], "section": "SLAB", "kind": "slab"},
        {"id": "W1", "nodes": ["1", "2", "6", "5"], "section": "WALL", "kind": "wall"},
    ],
    "sections": [
        {"name": "COL", "kind": "frame", "material": "C",
         "props": {"A": 0.16, "Iy": 2.1e-3, "Iz": 2.1e-3, "J": 3.6e-3}},
        {"name": "BEAM", "kind": "frame", "material": "C",
         "props": {"A": 0.12, "Iy": 9.0e-4, "Iz": 1.6e-3, "J": 1.8e-3}},
        {"name": "SLAB", "kind": "shell", "material": "C", "thickness": 0.20},
        {"name": "WALL", "kind": "shell", "material": "C", "thickness": 0.25},
    ],
    "materials": [{"name": "C", "E": 2.5e7, "nu": 0.2, "rho": 2.4}],
    "restraints": [{"node": n, "dofs": [1, 1, 1, 1, 1, 1]} for n in ("1", "2", "3", "4")],
    "loads": {"Dead": {"area": [{"area": "S1", "direction": "Z", "value": -5.0}]}},
}

# Frames-only single story WITH a diaphragm (no shell to back it) -> the
# importer must EMIT a rigidDiaphragm. Neither sm.json fixture exercises this.
_FRAME_DIA = {
    "schema_version": "0.1",
    "units": {"length": "m", "force": "kN"},
    "nodes": [
        {"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "2", "x": 4.0, "y": 0.0, "z": 0.0},
        {"id": "3", "x": 4.0, "y": 4.0, "z": 0.0},
        {"id": "4", "x": 0.0, "y": 4.0, "z": 0.0},
        {"id": "5", "x": 0.0, "y": 0.0, "z": 3.0},
        {"id": "6", "x": 4.0, "y": 0.0, "z": 3.0},
        {"id": "7", "x": 4.0, "y": 4.0, "z": 3.0},
        {"id": "8", "x": 0.0, "y": 4.0, "z": 3.0},
    ],
    "frames": [
        {"id": "C1", "i": "1", "j": "5", "section": "COL"},
        {"id": "C2", "i": "2", "j": "6", "section": "COL"},
        {"id": "C3", "i": "3", "j": "7", "section": "COL"},
        {"id": "C4", "i": "4", "j": "8", "section": "COL"},
        {"id": "B1", "i": "5", "j": "6", "section": "BEAM"},
        {"id": "B2", "i": "6", "j": "7", "section": "BEAM"},
        {"id": "B3", "i": "7", "j": "8", "section": "BEAM"},
        {"id": "B4", "i": "8", "j": "5", "section": "BEAM"},
    ],
    "sections": [
        {"name": "COL", "kind": "frame", "material": "C",
         "props": {"A": 0.16, "Iy": 2.1e-3, "Iz": 2.1e-3, "J": 3.6e-3}},
        {"name": "BEAM", "kind": "frame", "material": "C",
         "props": {"A": 0.12, "Iy": 9.0e-4, "Iz": 1.6e-3, "J": 1.8e-3}},
    ],
    "materials": [{"name": "C", "E": 2.5e7, "nu": 0.2, "rho": 2.4}],
    "restraints": [{"node": n, "dofs": [1, 1, 1, 1, 1, 1]} for n in ("1", "2", "3", "4")],
    "diaphragms": [{"name": "D1", "story": "S1", "nodes": ["5", "6", "7", "8"]}],
    "loads": {"Live": {"nodal": [{"node": "5", "force_xyz": [20.0, 0.0, 0.0]}]}},
}


def _build_box_fem(global_size: float = 1.0):
    model = StructuralModel.from_dict(_BOX)
    sess = apeGmsh(model_name="test_etabs_box", verbose=False)
    sess.begin()
    try:
        result = import_structural_model(sess, model)
        sess.mesh.sizing.set_global_size(global_size)
        sess.mesh.generation.generate(dim=2)
        sess.mesh.partitioning.renumber(base=1)
        fem = sess.mesh.queries.get_fem_data(dim=None)
    finally:
        sess.end()
    return model, result, fem


def test_areas_build_conformal_shell_and_beam_mesh():
    import numpy as np

    model, result, fem = _build_box_fem()

    assert {ag.pg for ag in result.area_groups} == {"SLAB", "WALL"}

    # Both element families present.
    line = fem.elements.select(pg="COL").ids
    shell = fem.elements.select(pg="SLAB").ids
    assert len(line) > 0 and len(shell) > 0

    # CONFORMALITY: no two distinct nodes share a location. If the wall and
    # slab meshes were not welded along edge 5-6, coincident duplicates appear.
    coords = np.asarray(fem.nodes.coords, dtype=float).round(6)
    uniq = {tuple(c) for c in coords}
    assert len(uniq) == coords.shape[0], "coincident duplicate nodes -> non-conformal"


def test_shell_beam_deck_solves(tmp_path):
    pytest.importorskip("openseespy")
    import runpy

    import openseespy.opensees as ops_mod

    model, result, fem = _build_box_fem()
    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    py = tmp_path / "box.py"
    ops.py(str(py))

    runpy.run_path(str(py))
    top = next(n for n in ops_mod.getNodeTags()
               if abs(ops_mod.nodeCoord(n, 3) - 3.0) < 1e-6)
    ops_mod.timeSeries("Linear", 99)
    ops_mod.pattern("Plain", 99, 99)
    ops_mod.load(top, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ops_mod.system("BandGeneral")
    ops_mod.numberer("RCM")
    ops_mod.constraints("Transformation")
    ops_mod.integrator("LoadControl", 1.0)
    ops_mod.test("NormDispIncr", 1e-8, 20)
    ops_mod.algorithm("Linear")
    ops_mod.analysis("Static")
    assert ops_mod.analyze(1) == 0


def test_self_mass_and_distributed_load_box(tmp_path):
    pytest.importorskip("openseespy")
    import math
    import runpy

    import openseespy.opensees as ops_mod

    model, result, fem = _build_box_fem()
    assert result.has_masses is True            # rho=2.4 -> shell + line mass
    assert result.load_patterns == ["Dead"]     # area pressure -> tributary nodal
    # The shell-less diaphragm logic doesn't apply here (no diaphragms).
    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    py = tmp_path / "boxm.py"
    ops.py(str(py))
    text = py.read_text()
    assert "ops.mass(" in text                  # self-mass emitted
    assert "ops.load(" in text                  # area pressure -> nodal loads

    runpy.run_path(str(py))
    lam = ops_mod.eigen("-genBandArpack", 1)[0]  # mass matrix must be non-singular
    assert lam > 0 and math.isfinite(2 * math.pi / math.sqrt(lam))


def test_frames_only_diaphragm_emits_and_solves(tmp_path):
    pytest.importorskip("openseespy")
    import runpy

    import openseespy.opensees as ops_mod

    model = StructuralModel.from_dict(_FRAME_DIA)
    sess = apeGmsh(model_name="test_dia", verbose=False)
    sess.begin()
    try:
        result = import_structural_model(sess, model)
        for fg in result.frame_groups:
            sess.mesh.structured.set_transfinite_curve(fg.pg, n_nodes=2)
        sess.mesh.generation.generate(dim=1)
        sess.mesh.partitioning.renumber(dim=1, method="rcm", base=1)
        fem = sess.mesh.queries.get_fem_data(dim=1)
    finally:
        sess.end()

    # No shell backs this diaphragm -> it must be emitted, not skipped.
    assert result.diaphragms[0].shell_backed is False

    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    py = tmp_path / "dia.py"
    ops.py(str(py))
    assert "rigidDiaphragm" in py.read_text()

    runpy.run_path(str(py))
    ops_mod.system("BandGeneral")
    ops_mod.numberer("RCM")
    ops_mod.constraints("Transformation")
    ops_mod.integrator("LoadControl", 1.0)
    ops_mod.test("NormDispIncr", 1e-8, 30)
    ops_mod.algorithm("Linear")
    ops_mod.analysis("Static")
    assert ops_mod.analyze(1) == 0


def _build_fem():
    """Transfinite n_nodes=2 -> exactly one element per member (deterministic)."""
    model = StructuralModel.from_dict(_MODEL)
    sess = apeGmsh(model_name="test_etabs_import", verbose=False)
    sess.begin()
    try:
        result = import_structural_model(sess, model)
        for fg in result.frame_groups:
            sess.mesh.structured.set_transfinite_curve(fg.pg, n_nodes=2)
        sess.mesh.generation.generate(dim=1)
        sess.mesh.partitioning.renumber(dim=1, method="rcm", base=1)
        fem = sess.mesh.queries.get_fem_data(dim=1)
    finally:
        sess.end()
    return model, result, fem


def test_schema_version_guard():
    bad = dict(_MODEL, schema_version="9.9")
    with pytest.raises(ValueError, match="schema_version"):
        StructuralModel.from_dict(bad)


def test_import_builds_groups():
    model, result, fem = _build_fem()

    # 3 corner nodes, 2 members -> 2 line elements at coarse size.
    assert fem.info.n_nodes == 3
    assert fem.info.n_elems == 2

    # One frame group per section, each tagged by orientation.
    pgs = {fg.pg: fg.orient for fg in result.frame_groups}
    assert pgs == {"COL": "v", "BEAM": "h"}

    # Single full-fixity restraint group; one load pattern (Live nodal).
    assert [rg.pg for rg in result.restraint_groups] == ["fix_111111"]
    assert result.load_patterns == ["Live"]


def test_emitted_deck_contents(tmp_path):
    model, result, fem = _build_fem()
    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    tcl = tmp_path / "out.tcl"
    ops.tcl(str(tcl))
    text = tcl.read_text()

    assert text.count("geomTransf Linear") == 2          # one per orientation
    assert text.count("element elasticBeamColumn") == 2   # one per member
    assert text.count("\nfix ") + text.startswith("fix ") >= 1
    # Column uses E=2.5e7 and its own A=0.16 from the section/material.
    assert "0.16 25000000.0" in text


def test_deck_solves(tmp_path):
    pytest.importorskip("openseespy")
    import runpy

    import openseespy.opensees as ops_mod

    model, result, fem = _build_fem()
    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    py = tmp_path / "out.py"
    ops.py(str(py))

    runpy.run_path(str(py))
    ops_mod.system("BandGeneral")
    ops_mod.numberer("RCM")
    ops_mod.constraints("Transformation")
    ops_mod.integrator("LoadControl", 1.0)
    ops_mod.test("NormDispIncr", 1e-8, 10)
    ops_mod.algorithm("Linear")
    ops_mod.analysis("Static")
    assert ops_mod.analyze(1) == 0


# =====================================================================
# Phase 5 — point + area (subgrade / Winkler) springs
# =====================================================================

# A foundation mat on soil: the ONLY support is an area (subgrade) spring with
# all three per-unit-area stiffnesses, so the deck is solvable (vertical bed +
# horizontal/drilling stability) and the settlement is mesh-independent q/k.
_MAT = {
    "schema_version": "0.1",
    "units": {"length": "m", "force": "kN"},
    "nodes": [
        {"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "2", "x": 4.0, "y": 0.0, "z": 0.0},
        {"id": "3", "x": 4.0, "y": 4.0, "z": 0.0},
        {"id": "4", "x": 0.0, "y": 4.0, "z": 0.0},
    ],
    "frames": [],
    "areas": [{"id": "F1", "nodes": ["1", "2", "3", "4"], "section": "SLAB", "kind": "slab"}],
    "sections": [{"name": "SLAB", "kind": "shell", "material": "C", "thickness": 0.30}],
    "materials": [{"name": "C", "E": 2.5e7, "nu": 0.2, "rho": 2.4}],
    "area_springs": [{"area": "F1", "k": [1.0e4, 1.0e4, 1.5e4], "property": "Suelo"}],
    "loads": {"Dead": {"area": [{"area": "F1", "direction": "Z", "value": -5.0}]}},
}

# A column whose base is supported only by a point spring (6 diagonal
# stiffnesses) — a flexible footing idealisation.
_COL_SPRING = {
    "schema_version": "0.1",
    "units": {"length": "m", "force": "kN"},
    "nodes": [
        {"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "2", "x": 0.0, "y": 0.0, "z": 3.0},
    ],
    "frames": [{"id": "C", "i": "1", "j": "2", "section": "COL", "kind": "column"}],
    "sections": [{"name": "COL", "kind": "frame", "material": "M",
                  "props": {"A": 0.16, "Iy": 2.1e-3, "Iz": 2.1e-3, "J": 3.6e-3}}],
    "materials": [{"name": "M", "E": 2.5e7, "nu": 0.2}],
    "springs": [{"node": "1", "k": [1.0e5, 1.0e5, 1.0e6, 5.0e5, 5.0e5, 5.0e4]}],
    "loads": {"H": {"nodal": [{"node": "2", "force_xyz": [10.0, 0.0, 0.0]}]}},
}


def test_parse_springs_and_area_springs():
    model = StructuralModel.from_dict(_COL_SPRING)
    assert model.springs[0].node == "1"
    assert model.springs[0].k == (1.0e5, 1.0e5, 1.0e6, 5.0e5, 5.0e5, 5.0e4)
    mat = StructuralModel.from_dict(_MAT)
    assert mat.area_springs[0].area == "F1"
    assert mat.area_springs[0].k == (1.0e4, 1.0e4, 1.5e4)
    assert mat.area_springs[0].property == "Suelo"


def test_apply_subgrade_springs_noop_without_springs():
    # The plain frame model has no springs -> the step declares none.
    model = StructuralModel.from_dict(_MODEL)
    sess = apeGmsh(model_name="nospring", verbose=False)
    sess.begin()
    try:
        result = import_structural_model(sess, model)
        for fg in result.frame_groups:
            sess.mesh.structured.set_transfinite_curve(fg.pg, n_nodes=2)
        sess.mesh.generation.generate(dim=1)
        sess.mesh.partitioning.renumber(dim=1, method="rcm", base=1)
        assert apply_subgrade_springs(sess, model, result) == 0
    finally:
        sess.end()
    assert result.spring_grounds == []


def _build_mat_fem(global_size=1.0):
    model = StructuralModel.from_dict(_MAT)
    sess = apeGmsh(model_name="mat", verbose=False)
    sess.begin()
    try:
        result = import_structural_model(sess, model)
        sess.mesh.sizing.set_global_size(global_size)
        sess.mesh.generation.generate(dim=2)
        sess.mesh.partitioning.renumber(base=1)
        n = apply_subgrade_springs(sess, model, result)
        fem = sess.mesh.queries.get_fem_data(dim=None)
    finally:
        sess.end()
    return model, result, fem, n


def test_area_winkler_one_ground_per_surface_node():
    model, result, fem, n = _build_mat_fem()
    # One grounded spring per meshed slab node; grounds are decoupled nodes.
    slab_nodes = len(fem.nodes.select(pg="SLAB").ids)
    assert n == slab_nodes > 4                      # interior nodes too, not just corners
    assert len(result.spring_grounds) == n
    assert len(fem.nodes.decoupled_ids) == n


def test_area_subgrade_deck_solves_settlement_is_q_over_k(tmp_path):
    pytest.importorskip("openseespy")
    import runpy

    import openseespy.opensees as ops_mod

    model, result, fem, _ = _build_mat_fem()
    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    py = tmp_path / "mat.py"
    ops.py(str(py))
    text = py.read_text()
    assert "zeroLength" in text                  # grounded springs emitted
    assert "Elastic" in text                     # spring materials emitted

    runpy.run_path(str(py))
    ops_mod.system("UmfPack")
    ops_mod.numberer("RCM")
    ops_mod.constraints("Transformation")
    ops_mod.integrator("LoadControl", 1.0)
    ops_mod.test("NormDispIncr", 1e-10, 30)
    ops_mod.algorithm("Linear")
    ops_mod.analysis("Static")
    assert ops_mod.analyze(1) == 0
    # Uniform pressure q=5 on a bed of k=1.5e4 -> uniform settlement q/k,
    # independent of the mesh (the Winkler-bed correctness check).
    uz = [ops_mod.nodeDisp(t, 3) for t in ops_mod.getNodeTags()]
    assert min(uz) == pytest.approx(-5.0 / 1.5e4, rel=1e-3)


def test_point_spring_deck_solves_and_translates(tmp_path):
    pytest.importorskip("openseespy")
    import runpy

    import openseespy.opensees as ops_mod

    model = StructuralModel.from_dict(_COL_SPRING)
    sess = apeGmsh(model_name="colspring", verbose=False)
    sess.begin()
    try:
        result = import_structural_model(sess, model)
        for fg in result.frame_groups:
            sess.mesh.structured.set_transfinite_curve(fg.pg, n_nodes=2)
        sess.mesh.generation.generate(dim=1)
        sess.mesh.partitioning.renumber(dim=1, method="rcm", base=1)
        n = apply_subgrade_springs(sess, model, result)
        fem = sess.mesh.queries.get_fem_data(dim=1)
    finally:
        sess.end()
    assert n == 1

    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    py = tmp_path / "col.py"
    ops.py(str(py))

    runpy.run_path(str(py))
    ops_mod.system("UmfPack")
    ops_mod.numberer("RCM")
    ops_mod.constraints("Transformation")
    ops_mod.integrator("LoadControl", 1.0)
    ops_mod.test("NormDispIncr", 1e-10, 30)
    ops_mod.algorithm("Linear")
    ops_mod.analysis("Static")
    assert ops_mod.analyze(1) == 0
    # Base node rides the horizontal spring: ux = H / Kux = 10 / 1e5.
    base = next(t for t in ops_mod.getNodeTags()
                if abs(ops_mod.nodeCoord(t, 3)) < 1e-6)
    assert ops_mod.nodeDisp(base, 1) == pytest.approx(10.0 / 1.0e5, rel=1e-6)


# --- area-spring orientation (area local axes, not forced global) -----------

def _area_frame_of(corners):
    """Build a one-area model from corner coords and return (e1, e2, e3=normal)."""
    import numpy as np

    from apeGmsh.interop.etabs_import import _area_frame
    nodes = [{"id": str(i), "x": c[0], "y": c[1], "z": c[2]} for i, c in enumerate(corners)]
    model = StructuralModel.from_dict({
        "schema_version": "0.1", "units": {"length": "m", "force": "kN"},
        "nodes": nodes, "frames": [],
        "areas": [{"id": "A", "nodes": [str(i) for i in range(len(corners))],
                   "section": "S", "kind": "slab"}],
        "sections": [{"name": "S", "kind": "shell", "material": "M", "thickness": 0.2}],
        "materials": [{"name": "M", "E": 2e7, "nu": 0.2}],
    })
    o = _area_frame(model, model.areas[0])
    e1, e2 = np.array(o[:3]), np.array(o[3:])
    return e1, e2, np.cross(e1, e2)


def test_area_frame_normal_and_axes():
    import numpy as np

    # Horizontal slab -> local-3 = +Z, local-1 = +X (current default behaviour).
    e1, _e2, e3 = _area_frame_of([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    assert np.allclose(e3, [0, 0, 1])
    assert np.allclose(e1, [1, 0, 0])
    # Vertical wall in the X-Z plane -> normal along +/-Y.
    _e1, _e2, e3 = _area_frame_of([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])
    assert np.allclose(np.abs(e3), [0, 1, 0])
    # 45deg tilt about X -> normal in the Y-Z plane at 45deg (parallel check).
    c = 1.0
    _e1, _e2, e3 = _area_frame_of([(0, 0, 0), (1, 0, 0), (1, c, c), (0, c, c)])
    assert np.allclose(np.abs(np.cross(e3, [0, -1, 1])), 0, atol=1e-9)
    # local axes are orthonormal.
    assert np.allclose(np.dot(_e1, e3), 0, atol=1e-9)


def test_area_frame_local_axis_deg_rotation():
    import numpy as np
    # A horizontal slab rotated 90deg about its normal: local-1 X -> Y.
    nodes = [{"id": str(i), "x": c[0], "y": c[1], "z": 0.0}
             for i, c in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)])]
    from apeGmsh.interop.etabs_import import _area_frame
    model = StructuralModel.from_dict({
        "schema_version": "0.1", "units": {"length": "m", "force": "kN"},
        "nodes": nodes, "frames": [],
        "areas": [{"id": "A", "nodes": ["0", "1", "2", "3"], "section": "S",
                   "kind": "slab", "local_axis_deg": 90.0}],
        "sections": [{"name": "S", "kind": "shell", "material": "M", "thickness": 0.2}],
        "materials": [{"name": "M", "E": 2e7, "nu": 0.2}],
    })
    e1 = np.array(_area_frame(model, model.areas[0])[:3])
    assert np.allclose(e1, [0, 1, 0], atol=1e-9)


# A square mat tilted 30deg about X, supported only by its subgrade bed.
def _tilted_mat(angle_deg=30.0):
    import math
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    L = 4.0
    return {
        "schema_version": "0.1", "units": {"length": "m", "force": "kN"},
        "nodes": [
            {"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": "2", "x": L, "y": 0.0, "z": 0.0},
            {"id": "3", "x": L, "y": L * c, "z": L * s},
            {"id": "4", "x": 0.0, "y": L * c, "z": L * s},
        ],
        "frames": [],
        "areas": [{"id": "F1", "nodes": ["1", "2", "3", "4"], "section": "SLAB", "kind": "slab"}],
        "sections": [{"name": "SLAB", "kind": "shell", "material": "C", "thickness": 0.30}],
        "materials": [{"name": "C", "E": 2.5e7, "nu": 0.2, "rho": 2.4}],
        "area_springs": [{"area": "F1", "k": [1.0e4, 1.0e4, 1.5e4], "property": "Suelo"}],
        "loads": {"Dead": {"area": [{"area": "F1", "direction": "Z", "value": -5.0}]}},
    }


def test_tilted_mat_oriented_springs_emit_and_solve(tmp_path):
    pytest.importorskip("openseespy")
    import runpy

    import openseespy.opensees as ops_mod

    model = StructuralModel.from_dict(_tilted_mat())
    sess = apeGmsh(model_name="tiltmat", verbose=False)
    sess.begin()
    try:
        result = import_structural_model(sess, model)
        sess.mesh.sizing.set_global_size(1.0)
        sess.mesh.generation.generate(dim=2)
        sess.mesh.partitioning.renumber(base=1)
        n = apply_subgrade_springs(sess, model, result)
        fem = sess.mesh.queries.get_fem_data(dim=None)
    finally:
        sess.end()
    assert n > 4
    # The springs carry an explicit orientation (not the global default).
    assert all(sg.orient is not None for sg in result.spring_grounds)

    ops = build_opensees(fem, model, result, ndm=3, ndf=6)
    py = tmp_path / "tilt.py"
    ops.py(str(py))
    assert "-orient" in py.read_text()                # oriented zeroLength emitted

    runpy.run_path(str(py))
    ops_mod.system("UmfPack")
    ops_mod.numberer("RCM")
    ops_mod.constraints("Transformation")
    ops_mod.integrator("LoadControl", 1.0)
    ops_mod.test("NormDispIncr", 1e-10, 30)
    ops_mod.algorithm("Linear")
    ops_mod.analysis("Static")
    assert ops_mod.analyze(1) == 0                    # oriented bed is non-singular
