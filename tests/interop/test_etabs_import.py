"""Phase 2 regression: StructuralModel JSON -> apeGmsh geometry -> beam mesh
-> OpenSees deck. Self-contained (no apeETABS / live ETABS dependency).

See apeETABS ADR 0009 + build-plan W3 Phase 2.
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh.interop import StructuralModel, build_opensees, import_structural_model

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
