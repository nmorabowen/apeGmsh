"""Phase 5 — Tcl / Python recorder command emission.

Pure unit tests against synthetic resolved specs — no session
required. Snapshot-style assertions verify the emitted command
strings.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.fixtures.recorder_models import identity_model, node_model


def _make_spec(*records: ResolvedRecorderRecord) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id="testhash",
        records=tuple(records),
    )


def _model3d(spec: ResolvedRecorderSpec, *, ndf: int = 6, ndm: int = 3):
    """Every recorded node at one ``(ndm, ndf)``: the 3-D frame by default."""
    nodes = {int(n) for r in spec.records if r.node_ids is not None
             for n in r.node_ids}
    return node_model(nodes, spec.fem_snapshot_id, ndf=ndf, ndm=ndm)


# =====================================================================
# Nodal emission — single ops_type
# =====================================================================

def test_node_displacement_only_3d() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="top_disp",
        components=("displacement_x", "displacement_y", "displacement_z"),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2, 3]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec))
    assert line.startswith("recorder Node ")
    assert "-file top_disp_disp.out" in line
    assert "-time" in line
    assert "-node 1 2 3" in line
    assert "-dof 1 2 3" in line
    assert line.rstrip().endswith("disp  ;# top_disp disp")


def test_node_displacement_with_dt() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec))
    assert "-dT 0.01" in line


def test_node_displacement_xml_format() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec), file_format="xml")
    assert "-xml r_disp.xml" in line
    assert "-file" not in line


def test_node_output_dir_prefix() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec), output_dir="out/")
    assert "-file out/r_disp.out" in line


def test_output_dir_no_trailing_slash() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec), output_dir="results")
    # Emitter inserts the separator
    assert "-file results/r_disp.out" in line


# =====================================================================
# Nodal emission — multiple ops_types per record
# =====================================================================

def test_node_disp_plus_velocity_emits_two_commands() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="kinematics",
        components=("displacement_x", "velocity_x"),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2]),
    ))
    lines = spec.to_tcl_commands(model=_model3d(spec))
    assert len(lines) == 2
    cmds = sorted(lines)
    assert any("kinematics_disp.out" in l and "disp" in l for l in cmds)
    assert any("kinematics_vel.out" in l and "vel" in l for l in cmds)


def test_node_translational_plus_rotational_one_command() -> None:
    """Translation and rotation are both ``disp`` recorder type — one cmd."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="full",
        components=(
            "displacement_x", "displacement_y", "displacement_z",
            "rotation_x", "rotation_y", "rotation_z",
        ),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec))
    assert "-dof 1 2 3 4 5 6" in line
    assert line.rstrip().endswith("disp  ;# full disp")


def test_node_dofs_sorted() -> None:
    """Out-of-order canonical components emit sorted DOFs."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_z", "displacement_x"),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec))
    assert "-dof 1 3" in line


# =====================================================================
# Reaction / unbalance / pressure
# =====================================================================

def test_reaction_force_emits_reaction_recorder() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("reaction_force_x", "reaction_force_y", "reaction_force_z"),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec))
    assert line.rstrip().endswith("reaction  ;# r reaction")


def test_force_emits_unbalance_recorder() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("force_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec))
    assert "unbalance" in line


def test_pore_pressure_emits_pressure_recorder() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("pore_pressure",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=_model3d(spec))
    assert "pressure" in line
    # Default pressure DOF in 3D u-p is 4
    assert "-dof 4" in line


# =====================================================================
# Element-level emission
# =====================================================================

def test_gauss_emits_element_stress_recorder() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="gauss", name="body",
        components=("stress_xx",),
        dt=0.01, n_steps=None,
        element_ids=np.array([10, 20]),
    ))
    [line] = spec.to_tcl_commands(model=identity_model([10, 20], spec.fem_snapshot_id))
    assert line.startswith("recorder Element ")
    assert "-ele 10 20" in line
    assert "-dT 0.01" in line
    # OpenSees solid elements register the keyword "stresses" (plural)
    # in setResponse — see _recorder_emit._ELEMENT_CATEGORY_RESPONSE.
    assert line.rstrip().endswith("stresses  ;# body gauss")


def test_elements_emits_globalForce() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="elements", name="r",
        components=("nodal_resisting_force_x",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    [line] = spec.to_tcl_commands(model=identity_model([1], spec.fem_snapshot_id))
    assert "globalForce" in line


def test_line_stations_emits_section_force_and_gpx_pair() -> None:
    """Phase 11b Step 2c.1: line_stations emits a paired integrationPoints
    recorder so the .out transcoder can reconstruct per-element IP layout."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="line_stations", name="r",
        components=("axial_force",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    ))
    lines = spec.to_tcl_commands(model=identity_model([1], spec.fem_snapshot_id))
    # Two recorders: section.force + paired integrationPoints.
    assert len(lines) == 2
    section_lines = [ln for ln in lines if "section force" in ln]
    gpx_lines = [ln for ln in lines if "integrationPoints" in ln]
    assert len(section_lines) == 1
    assert len(gpx_lines) == 1
    # gpx file path follows the convention: <base>_gpx.out.
    assert "r_line_stations.out" in section_lines[0]
    assert "r_line_stations_gpx.out" in gpx_lines[0]
    # Both target the same element IDs and (here) same cadence.
    assert "-ele 1" in section_lines[0] and "-ele 1" in gpx_lines[0]


def test_line_stations_gpx_path_helper() -> None:
    from apeGmsh.results.spec._emit import line_station_gpx_path
    assert (
        line_station_gpx_path("r_line_stations.out")
        == "r_line_stations_gpx.out"
    )
    assert (
        line_station_gpx_path("out/foo_line_stations.out")
        == "out/foo_line_stations_gpx.out"
    )
    assert (
        line_station_gpx_path("foo.xml")
        == "foo_gpx.xml"
    )


# =====================================================================
# Deferred categories — emit a TODO comment
# =====================================================================

@pytest.mark.parametrize("category", ["fibers", "layers", "modal"])
def test_deferred_category_emits_todo(category: str) -> None:
    rec = ResolvedRecorderRecord(
        category=category, name="r",
        components=("fiber_stress",) if category in ("fibers",) else (
            ("stress_xx",) if category == "layers" else ()
        ),
        dt=None, n_steps=None,
        element_ids=(np.array([1]) if category != "modal" else None),
        n_modes=5 if category == "modal" else None,
    )
    spec = _make_spec(rec)
    [line] = spec.to_tcl_commands()
    if category in ("fibers", "layers"):
        # Phase 11c: fiber/layer recorders are MPCO-only — comment
        # points the user at MPCO instead of generic TODO.
        assert "MPCO-only" in line
        assert "Results.from_mpco" in line
    else:
        assert "TODO Phase 5+" in line
    assert category in line


# =====================================================================
# Cadence warnings
# =====================================================================

def test_n_steps_warning_in_tcl_emit() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=5,
        node_ids=np.array([1]),
    ))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec.to_tcl_commands(model=_model3d(spec))
    assert any("n_steps" in str(w.message) for w in caught)


# =====================================================================
# Python emission
# =====================================================================

def test_python_node_displacement() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([1, 2]),
    ))
    [line] = spec.to_python_commands(model=_model3d(spec))
    assert line.startswith("ops.recorder(")
    assert "'Node'" in line
    assert "'-file'" in line and "'r_disp.out'" in line
    assert "'-time'" in line
    assert "'-dT', 0.01" in line
    assert "'-node', 1, 2" in line
    assert "'-dof', 1" in line
    assert "'disp'" in line


def test_python_element() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="gauss", name="r",
        components=("stress_xx",),
        dt=None, n_steps=None,
        element_ids=np.array([100]),
    ))
    [line] = spec.to_python_commands(model=identity_model([100], spec.fem_snapshot_id))
    assert "'Element'" in line
    assert "'-ele', 100" in line
    assert "'stresses'" in line


# =====================================================================
# Empty record / empty IDs short-circuit
# =====================================================================

def test_empty_node_ids_emits_nothing() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([], dtype=np.int64),
    ))
    assert spec.to_tcl_commands() == []


def test_empty_element_ids_emits_nothing() -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="gauss", name="r",
        components=("stress_xx",),
        dt=None, n_steps=None,
        element_ids=np.array([], dtype=np.int64),
    ))
    assert spec.to_tcl_commands() == []


# =====================================================================
# n_steps= without dt= warns for every category (#1947)
# =====================================================================
#
# A classic recorder has no step-count option, so ``n_steps=5`` with
# no ``dt=`` is emitted every step. Before #1947 only the nodes emitter
# said so; an element record silently recorded every step.

_CADENCE_COMPONENTS = {
    "nodes": ("displacement_x",),
    "elements": ("nodal_resisting_force_x",),
    "gauss": ("stress_xx",),
    "line_stations": ("axial_force",),
}


def _cadence_record(
    category: str, *, dt: float | None, n_steps: int | None,
) -> ResolvedRecorderRecord:
    ids = np.array([1, 2])
    return ResolvedRecorderRecord(
        category=category, name=f"cad_{category}",
        components=_CADENCE_COMPONENTS[category],
        dt=dt, n_steps=n_steps,
        node_ids=ids if category == "nodes" else None,
        element_ids=None if category == "nodes" else ids,
    )


def _cadence_emit(route: str, rec: ResolvedRecorderRecord) -> list[str]:
    spec = _make_spec(rec)
    model = identity_model([1, 2], spec.fem_snapshot_id,
                           nodes_ndf={1: 6, 2: 6})
    if route == "tcl":
        return spec.to_tcl_commands(model=model)
    return spec.to_python_commands(model=model)


@pytest.mark.parametrize("route", ["tcl", "python"])
@pytest.mark.parametrize("category", list(_CADENCE_COMPONENTS))
def test_n_steps_without_dt_warns_once_per_record(
    category: str, route: str,
) -> None:
    rec = _cadence_record(category, dt=None, n_steps=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lines = _cadence_emit(route, rec)
    user = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user) == 1, [str(w.message) for w in caught]
    msg = str(user[0].message)
    assert repr(rec.name) in msg
    assert "n_steps=5" in msg
    assert "dt=" in msg and "-T nsteps" in msg
    assert "Phase 7" not in msg
    # The fix adds a warning and changes no output: the emitted lines
    # equal those of the same record without n_steps=.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = _cadence_emit(
            route, _cadence_record(category, dt=None, n_steps=None),
        )
    assert lines == plain
    assert lines and all("-dT" not in ln for ln in lines)


@pytest.mark.parametrize("route", ["tcl", "python"])
@pytest.mark.parametrize("category", list(_CADENCE_COMPONENTS))
def test_n_steps_with_dt_emits_dT_silently(category: str, route: str) -> None:
    rec = _cadence_record(category, dt=0.01, n_steps=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines = _cadence_emit(route, rec)
    flag = "-dT 0.01" if route == "tcl" else "'-dT', 0.01"
    assert lines and all(flag in ln for ln in lines)


# =====================================================================
# Node DOFs follow the node layout; model= is required (#2018)
# =====================================================================
#
# ``emit_logical`` keys node DOFs by spatial slot (``rotation_z`` is 6);
# the command that reaches OpenSees maps each slot to the DOF the nodes
# carry it at, through ``_node_layout(ndf, ndm)`` of the built model.
# Before #2018 a planar frame's ``rotation_z`` was recorded at ``-dof 6``,
# a DOF the node lacks, and read back as the padded 0.0.

_FRAME_2D = ("displacement_x", "displacement_y", "rotation_z")
_ALL_SIX = ("displacement_x", "displacement_y", "displacement_z",
            "rotation_x", "rotation_y", "rotation_z")


def _node_spec(components: tuple[str, ...], ids: tuple[int, ...] = (1, 2),
               ) -> ResolvedRecorderSpec:
    return _make_spec(ResolvedRecorderRecord(
        category="nodes", name="n", components=components,
        dt=None, n_steps=None, node_ids=np.array(ids),
    ))


def _ndf_model(spec: ResolvedRecorderSpec, ndm: int, nodes_ndf: dict):
    return identity_model([], spec.fem_snapshot_id, ndm=ndm,
                          nodes_ndf=nodes_ndf)


def test_2d_frame_rotation_z_is_dof_3() -> None:
    spec = _node_spec(_FRAME_2D)
    model = _ndf_model(spec, 2, {1: 3, 2: 3})
    [tcl] = spec.to_tcl_commands(model=model)
    [py] = spec.to_python_commands(model=model)
    assert " -dof 1 2 3 -precision" in tcl
    assert "'-dof', 1, 2, 3, '-precision'" in py


def test_2d_rotation_only_is_dof_3() -> None:
    spec = _node_spec(("rotation_z", "reaction_moment_z"))
    lines = spec.to_tcl_commands(model=_ndf_model(spec, 2, {1: 3, 2: 3}))
    assert len(lines) == 2
    assert all(" -dof 3 -precision" in ln for ln in lines)


def test_3d_frame_keeps_dofs_1_to_6() -> None:
    spec = _node_spec(_ALL_SIX)
    [line] = spec.to_tcl_commands(model=_ndf_model(spec, 3, {1: 6, 2: 6}))
    assert " -dof 1 2 3 4 5 6 -precision" in line
    spec = _node_spec(("rotation_z",))
    [line] = spec.to_tcl_commands(model=_ndf_model(spec, 3, {1: 6, 2: 6}))
    assert " -dof 6 -precision" in line


def test_3d_solid_ndf3_records_translations_and_refuses_rotation() -> None:
    spec = _node_spec(("displacement_x", "displacement_z"))
    [line] = spec.to_tcl_commands(model=_ndf_model(spec, 3, {1: 3, 2: 3}))
    assert " -dof 1 3 -precision" in line
    spec = _node_spec(("rotation_z",))
    with pytest.raises(ValueError, match=r"no recorded node carries 'rz'"):
        spec.to_tcl_commands(model=_ndf_model(spec, 3, {1: 3, 2: 3}))


@pytest.mark.parametrize("ndm", [2, 3])
@pytest.mark.parametrize("ndf", [4, 5])
def test_ndf_4_and_5_carry_translations_only(ndm: int, ndf: int) -> None:
    """A u-p node's DOF ndm+1 is pressure, never a rotation."""
    spec = _node_spec(_ALL_SIX[:ndm])
    [line] = spec.to_tcl_commands(
        model=_ndf_model(spec, ndm, {1: ndf, 2: ndf}))
    want = " ".join(str(d) for d in range(1, ndm + 1))
    assert f" -dof {want} -precision" in line
    spec = _node_spec(("rotation_z",))
    with pytest.raises(ValueError, match=r"no recorded node carries 'rz'"):
        spec.to_tcl_commands(model=_ndf_model(spec, ndm, {1: ndf, 2: ndf}))


def test_mixed_2d_quad_and_frame_record_rotation_at_dof_3() -> None:
    """The frame node carries rz at DOF 3; the quad node's padded column
    is masked on read (``mask_absent_rotations``, #1971)."""
    spec = _node_spec(_FRAME_2D)
    [line] = spec.to_tcl_commands(model=_ndf_model(spec, 2, {1: 2, 2: 3}))
    assert " -dof 1 2 3 -precision" in line


def test_mixed_3d_tet_and_shell_keep_dofs_1_to_6() -> None:
    spec = _node_spec(_ALL_SIX)
    [line] = spec.to_tcl_commands(model=_ndf_model(spec, 3, {1: 3, 2: 6}))
    assert " -dof 1 2 3 4 5 6 -precision" in line


def test_nodes_placing_a_slot_at_different_dofs_refuse() -> None:
    spec = _node_spec(("rotation_z",))
    with pytest.raises(ValueError, match=r"different DOFs"):
        spec.to_tcl_commands(model=_ndf_model(spec, 2, {1: 3, 2: 6}))


def test_2d_displacement_z_refuses_instead_of_recording_the_rotation() -> None:
    spec = _node_spec(("displacement_z",))
    with pytest.raises(ValueError, match=r"no recorded node carries 'z'"):
        spec.to_tcl_commands(model=_ndf_model(spec, 2, {1: 3, 2: 3}))


def test_a_translation_some_nodes_lack_refuses() -> None:
    """Nothing masks a padded translation column, so it is not recorded."""
    spec = _node_spec(("displacement_z",))
    with pytest.raises(ValueError, match=r"nodes \[1\] carry no 'z'"):
        spec.to_tcl_commands(model=_ndf_model(spec, 3, {1: 2, 2: 6}))


def test_a_node_the_model_has_no_ndf_for_refuses() -> None:
    spec = _node_spec(("displacement_x",), ids=(1, 99))
    with pytest.raises(ValueError, match=r"nodes \[99\] have no ndf"):
        spec.to_tcl_commands(model=_ndf_model(spec, 3, {1: 6}))


def test_pore_pressure_dof_is_not_a_slot() -> None:
    spec = _node_spec(("pore_pressure",))
    [line] = spec.to_tcl_commands(model=_ndf_model(spec, 3, {1: 4, 2: 4}))
    assert " -dof 4 -precision" in line
    assert line.rstrip().endswith("pressure  ;# n pressure")


@pytest.mark.parametrize("route", ["to_tcl_commands", "to_python_commands"])
@pytest.mark.parametrize("components", [("displacement_x",), _FRAME_2D])
def test_command_routes_refuse_node_records_without_model(
    route: str, components: tuple[str, ...],
) -> None:
    """Even a ux-only record: no slot's DOF is known without the model."""
    spec = _node_spec(components)
    with pytest.raises(ValueError, match=r"Pass model="):
        getattr(spec, route)()


@pytest.mark.parametrize("formatter", ["format_tcl", "format_python",
                                       "to_ops_args"])
def test_each_formatter_refuses_an_unmapped_node_recorder(
    formatter: str,
) -> None:
    from apeGmsh.results.spec import _emit

    [lr] = _emit.emit_logical(_node_spec(_FRAME_2D).records[0])
    assert lr.dofs == (1, 2, 6) and lr.deck_dofs is None   # slots
    with pytest.raises(ValueError, match=r"model="):
        getattr(_emit, formatter)(lr)


def test_ops_ndf_raised_node_records_rotation_at_dof_3(tmp_path) -> None:
    """A real built model: a 2-D truss envelope (ndf 2) and a decoupled
    master raised to ndf 3 by ``ops.ndf``. Its ``rotation_z`` is DOF 3;
    the truss nodes carry none."""
    from apeGmsh import apeGmsh
    from apeGmsh.opensees import OpenSeesModel, apeSees

    g = apeGmsh(model_name="b2018_raised")
    g.begin()
    try:
        a = g.model.geometry.add_point(0.0, 0.0, 0.0)
        b = g.model.geometry.add_point(1.0, 0.0, 0.0)
        g.physical.add(1, [g.model.geometry.add_line(a, b)], name="Bar")
        h = g.decouple_node(coords=(0.5, 1.0, 0.0), label="master")
        g.mesh.generation.generate(dim=1)
        fem = g.mesh.queries.get_fem_data(dim=1)
        ops = apeSees(fem, default_orientation=None)
        ops.model(ndm=2, ndf=2)
        ops.element.Truss(
            pg="Bar", A=1e-3,
            material=ops.uniaxialMaterial.ElasticMaterial(E=1.0))
        ops.ndf(h, ndf=3)
        path = tmp_path / "raised.h5"
        ops.h5(str(path))
        truss = tuple(int(n) for n in fem.nodes.select(pg="Bar").ids)
    finally:
        g.end()
    model = OpenSeesModel.from_h5(str(path))
    master = int(h.tag)
    assert model.nodes_ndf[master] == 3
    assert {model.nodes_ndf[n] for n in truss} == {2}

    def spec_for(components: tuple[str, ...], ids) -> ResolvedRecorderSpec:
        return ResolvedRecorderSpec(
            fem_snapshot_id=model.lineage.fem_hash,
            records=(ResolvedRecorderRecord(
                category="nodes", name="m", components=components,
                dt=None, n_steps=None, node_ids=np.array(ids)),),
        )

    [line] = spec_for(_FRAME_2D, [master]).to_tcl_commands(model=model)
    assert f"-node {master} -dof 1 2 3 -precision" in line
    with pytest.raises(ValueError, match=r"no recorded node carries 'rz'"):
        spec_for(("rotation_z",), truss).to_tcl_commands(model=model)
