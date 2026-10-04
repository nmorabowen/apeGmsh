"""STKO translator, work package C: conditions, definitions, analysis steps
(ADR 0111, rules C1-C14 of ``internal_docs/stko_translator_rules.md``).

The documents are built as :class:`ScdModel` objects in the test: the plan is
pure data over the model, so nothing here needs HDF5. One small mesh serves
every hand-computable rule::

    7 (0,0,3)
    |  column eid 20 (edge 0, element property 2, Elastic section 4)
    1----2----3        eid 21: edge mesh 1-2 (edge 1), carries no element property
    | q10 | q11 |      quads eid 10 (faces 0) and 11 (face 1): 2 x 1 each, area 2
    4----5----6
    8 = master (2, 0.5, 0), links eid 30 -> nodes 1,2,3 and eid 31 -> nodes 4,5,6
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apeGmsh.interop.stko.model import (
    Condition,
    Geometry,
    Interaction,
    Mesh,
    MeshElement,
    ScdModel,
    SubShapeRef,
    SubShapes,
    XObject,
)
from apeGmsh.interop.stko.translate_conditions import (
    REGISTRY,
    build_conditions,
    check_supported,
    declare_session,
    plan_conditions,
)
from apeGmsh.interop.stko.translate_types import (
    ConditionsPlan,
    DiaphragmGroup,
    MeshMap,
    TimeSeriesSpec,
    UnsupportedSTKOTypes,
    quat_matrix,
)

G = 1  # the one geometry

NODES = {
    1: (0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0), 3: (4.0, 0.0, 0.0),
    4: (0.0, 1.0, 0.0), 5: (2.0, 1.0, 0.0), 6: (4.0, 1.0, 0.0),
    7: (0.0, 0.0, 3.0), 8: (2.0, 0.5, 0.0),
}
Q90X = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))   # 90 deg about x


# ── builders ──────────────────────────────────────────────────────────

def xo(oid: int, name: str, xtype: str, refs: tuple[str, ...] = (), **attrs: Any) -> XObject:
    return XObject(id=oid, name=name, type=xtype, attributes=attrs, references=frozenset(refs))


def cond(oid: int, name: str, xtype: str, *, vertices=(), edges=(), faces=(),
         interactions=(), **attrs: Any) -> Condition:
    geometry = {G: SubShapes(vertices=tuple(vertices), edges=tuple(edges), faces=tuple(faces))} \
        if (vertices or edges or faces) else {}
    return Condition(xobject=xo(oid, name, xtype, **attrs), geometry=geometry,
                     interactions=tuple(interactions))


def elastic_section(area: float = 100.0) -> XObject:
    return xo(4, "C70x70", "sections.Elastic", Section={"PROPS": np.array([area, 7.0, 8.0, 9.0])})


def make_scd(*, conditions=(), steps=(), definitions=None, props=None, quats=None,
             edge_pp=(4, 0), face_pp=(3, 3)) -> ScdModel:
    ids = np.array(sorted(NODES), dtype=np.int32)
    xyz = np.array([NODES[i] for i in ids], dtype=float)
    elements = {
        10: MeshElement(303, 201, (1, 2, 5, 4)), 11: MeshElement(303, 201, (2, 3, 6, 5)),
        20: MeshElement(102, 2, (1, 7)), 21: MeshElement(102, 2, (1, 2)),
        30: MeshElement(600, 0, (8, 1, 2, 3)), 31: MeshElement(600, 0, (8, 4, 5, 6)),
    }
    orientation = {10: (0.0, 0.0, 0.0, 1.0), 11: (0.0, 0.0, 0.0, 1.0), 20: (0.0, 0.0, 0.0, 1.0)}
    orientation.update(quats or {})
    mesh = Mesh(
        node_ids=ids, coordinates=xyz, node_flags=np.zeros(len(ids), dtype=np.int32),
        elements=elements,
        domains={(G, "faces"): {0: np.array([10]), 1: np.array([11])},
                 (G, "edges"): {0: np.array([20]), 1: np.array([21])}},
        vertex_nodes={G: np.array([1, 3, 7])}, orientation=orientation,
    )
    geom = Geometry(
        id=G, name="frame", shape_index=1, counts={"vertices": 3, "edges": 2, "faces": 2},
        element_property={"edges": np.array([2, 0]), "faces": np.array([1, 1])},
        physical_property={"edges": np.array(edge_pp), "faces": np.array(face_pp)},
        local_axes={},
    )
    physical = {3: xo(3, "Slab", "sections.ElasticMembranePlateSection"), 4: elastic_section()}
    physical.update(props or {})
    link = Interaction(
        id=1, name="rd", type="NN", masters=(SubShapeRef(G, 1, 0),), slaves=(SubShapeRef(G, 3, 0),),
        elements=(30, 31), element_property=9, physical_property=0,
    )
    return ScdModel(
        path=Path("synthetic.scd"), version=(4, 1, 0), geometries={G: geom}, mesh=mesh,
        selection_sets={}, physical_properties=physical,
        element_properties={1: xo(1, "shell", "shell.ASDShellQ4"),
                            2: xo(2, "beam", "beam_column_elements.elasticBeamColumn")},
        conditions={c.id: c for c in conditions}, interactions={1: link}, local_axes={},
        definitions={d.id: d for d in (definitions if definitions is not None else [LINEAR])},
        analysis_steps={s.id: s for s in sorted(steps, key=lambda s: s.id)},
    )


def analyses_attrs(*, static: bool = True, **over: Any) -> dict[str, Any]:
    a: dict[str, Any] = {
        "Static": static, "numIncr": 2, "duration": 1.0, "duration/transient": 60.0,
        "loadConst": True, "staticIntegrators": "Load Control",
        "transientIntegrators": "Newmark Method", "gamma": 0.5, "beta": 0.25,
        "Adaptive Time Step": False,
        "constraints": "Auto", "Automatic": True, "oom/auto": 3, "-verbose/auto": False,
        "User-Defined": False, "userPenalty/auto": 1e18,
        "numbererType": "Parallel Reverse Cuthill-McKee Numberer",
        "system": "Mumps", "-ICNTL14": True, "ICNTL14 Value": 200, "-matrixType": False,
        "matrixType": "Unsymmetric",
        "testCommand": "Energy Increment Test", "tol/EnergyIncr": 1e-4, "iter/EnergyIncr": 10,
        "pFlag/EnergyIncr": "0",
        "algorithm": "Linear", "use_formTangent/Linear": False, "-factorOnce": True,
    }
    a.update(over)
    return a


def analyses(oid: int, name: str = "analysis", **over: Any) -> XObject:
    return xo(oid, name, "Analyses.AnalysesCommand", **analyses_attrs(**over))


def load_pattern(oid: int, load=(), mass_to_load=(), ts: int = 1, **over: Any) -> XObject:
    return xo(oid, f"lp{oid}", "Patterns.addPattern.loadPattern", tsTag=ts, load=tuple(load),
              massToLoad=tuple(mass_to_load), eleLoad=None, sp=None, genericLoad=None,
              **{"-fact": False, "cFactor": 1.0, **over})


def constraint_pattern(oid: int, sp=(), mp=()) -> XObject:
    return xo(oid, f"cp{oid}", "Patterns.addPattern.constraintPattern", sp=tuple(sp) or None,
              mp=tuple(mp) or None)


LINEAR = xo(1, "lin", "timeSeries.Linear", **{"-factor": False, "cFactor": 1.0})


def path_series(oid: int, name: str = "th", **over: Any) -> XObject:
    a: dict[str, Any] = {"-factor": True, "cFactor": 1000.0, "constant": True, "dt": 0.0025,
                         "list_of_values": (0.0,), "-startTime": False, "tStart": 0.0}
    a.update(over)
    return xo(oid, name, "timeSeries.Path", **a)


def mass_total(plan: ConditionsPlan) -> np.ndarray:
    return np.sum([m[:3] for m in plan.masses.values()], axis=0)


# ── C1: nodal mass ────────────────────────────────────────────────────

def test_face_mass_is_the_consistent_lumping_of_the_areal_mass() -> None:
    m = 3.0   # per area; each 2 x 1 quad has area 2 -> m * 2 / 4 per corner
    scd = make_scd(conditions=[cond(3, "fm", "Mass.FaceMass", faces=(0, 1), Mode="constant",
                                    mass=(m, 2 * m, 4 * m))])
    plan = plan_conditions(scd)
    corner = m * 2 / 4
    assert plan.masses[1] == (corner, 2 * corner, 4 * corner, 0.0, 0.0, 0.0)
    assert plan.masses[2][0] == pytest.approx(2 * corner)     # shared by both quads
    assert plan.masses[5][0] == pytest.approx(2 * corner)
    assert set(plan.masses) == {1, 2, 3, 4, 5, 6}
    assert mass_total(plan)[0] == pytest.approx(m * 4.0)      # area 4


def test_face_mass_on_a_warped_quad_still_integrates_to_its_area() -> None:
    scd = make_scd(conditions=[cond(3, "fm", "Mass.FaceMass", faces=(0,), Mode="constant",
                                    mass=(1.0, 1.0, 1.0))])
    scd.mesh.coordinates[list(scd.mesh.node_ids).index(5)] = (2.0, 1.0, 0.4)
    total = mass_total(plan_conditions(scd))[0]
    p = {n: np.array(NODES[n]) for n in (1, 2, 5, 4)}
    p[5] = np.array((2.0, 1.0, 0.4))
    # bilinear patch area by 2x2 Gauss, the rule STKO itself uses
    area = 0.0
    g = 1 / math.sqrt(3)
    for xi, eta in ((-g, -g), (g, -g), (g, g), (-g, g)):
        dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        P = np.array([p[1], p[2], p[5], p[4]])
        area += np.linalg.norm(np.cross(dxi @ P, deta @ P))
    assert total == pytest.approx(area)


def test_node_mass_sits_on_the_vertex_node_and_masses_add_over_conditions() -> None:
    scd = make_scd(conditions=[
        cond(5, "nm", "Mass.NodeMass", vertices=(2,), Mode="constant", mass=(10.0, 20.0, 30.0)),
        cond(6, "nm2", "Mass.NodeMass", vertices=(2,), Mode="constant", mass=(1.0, 1.0, 1.0)),
    ])
    plan = plan_conditions(scd)
    assert plan.masses == {7: (11.0, 21.0, 31.0, 0.0, 0.0, 0.0)}


def test_every_mass_condition_counts_whether_or_not_a_pattern_uses_it() -> None:
    scd = make_scd(conditions=[cond(3, "fm", "Mass.FaceMass", faces=(0,), Mode="constant",
                                    mass=(1.0, 1.0, 1.0))], steps=[load_pattern(9)])
    assert 1 in plan_conditions(scd).masses


def test_auto_edge_mass_uses_rho_times_the_elastic_section_area() -> None:
    scd = make_scd(conditions=[cond(2, "am", "Mass.AutoEdgeMass", edges=(0,), rho=2.0e-3,
                                    **{"Convert to load": False, "gx": 0.0, "gy": 0.0, "gz": -10.0,
                                       "Type": "load"})])
    plan = plan_conditions(scd)
    per_end = 2.0e-3 * 100.0 * 3.0 / 2
    assert set(plan.masses) == {1, 7}
    for n in (1, 7):
        assert plan.masses[n] == pytest.approx((per_end,) * 3 + (0.0,) * 3)


def _fiber_section(surface: list, punctual: list | None = None) -> XObject:
    fs: dict[str, Any] = {
        "CENTER_AND_AREA": np.array([0.0, 0.0, 1.0e9]),
        "SURFACE_FIBER_GROUPS": {"ITEM_1": {"FIBERS": np.array(surface, float),
                                            "@attrs": {"PHYS_PROP_ID": np.array([7])}}},
    }
    if punctual:
        fs["PUNCTUAL_FIBER_GROUPS"] = {"ITEM_1": {"FIBERS": np.array(punctual, float),
                                                  "@attrs": {"PHYS_PROP_ID": np.array([8])}}}
    return xo(11, "fib", "sections.Fiber", **{"Fiber section": fs})


def test_auto_edge_mass_area_of_a_fiber_section_is_its_surface_fibers_only() -> None:
    fib = _fiber_section([[0, 0, 60.0], [1, 1, 40.0]], punctual=[[0, 0, 5000.0]])   # rebar excluded
    scd = make_scd(
        props={11: fib}, edge_pp=(11, 0),
        conditions=[cond(2, "am", "Mass.AutoEdgeMass", edges=(0,), rho=1.0,
                         **{"Convert to load": False, "gx": 0.0, "gy": 0.0, "gz": 0.0,
                            "Type": "load"})])
    assert plan_conditions(scd).masses[1][0] == pytest.approx(100.0 * 3.0 / 2)


def test_auto_edge_mass_through_a_beam_section_property_finds_the_rectangular_fiber_section() -> None:
    bsp = XObject(id=12, name="bsp", type="special_purpose.BeamSectionProperty",
                  attributes={"secTag/1": 13}, references=frozenset({"secTag/1"}))
    rect = xo(13, "rect", "sections.RectangularFiberSection", Width=10.0, Height=20.0)
    scd = make_scd(
        props={12: bsp, 13: rect}, edge_pp=(12, 0),
        conditions=[cond(2, "am", "Mass.AutoEdgeMass", edges=(0,), rho=1.0,
                         **{"Convert to load": False, "gx": 0.0, "gy": 0.0, "gz": 0.0,
                            "Type": "load"})])
    assert plan_conditions(scd).masses[7][0] == pytest.approx(200.0 * 3.0 / 2)


# ── C2-C5: loads ──────────────────────────────────────────────────────

def test_face_force_lumps_per_element_corner_and_sums_shared_nodes() -> None:
    f = 5.0
    scd = make_scd(
        conditions=[cond(13, "ff", "Loads.Force.FaceForce", faces=(0, 1), Mode="constant",
                         F=(0.0, 0.0, -f), Global=True)],
        steps=[load_pattern(9, load=(13,))])
    p = plan_conditions(scd).patterns[0]
    assert p.loads[1] == pytest.approx((0.0, 0.0, -f * 2 / 4, 0.0, 0.0, 0.0))
    assert p.loads[2][2] == pytest.approx(-f * 2 / 2)
    assert sum(v[2] for v in p.loads.values()) == pytest.approx(-f * 4)
    assert p.conditions == (13,) and p.kind == "Plain" and p.series == 1


def test_face_force_in_the_local_frame_is_rotated_by_the_element_quaternion() -> None:
    scd = make_scd(
        conditions=[cond(13, "ff", "Loads.Force.FaceForce", faces=(0,), Mode="constant",
                         F=(0.0, 0.0, -1.0), Global=False)],
        steps=[load_pattern(9, load=(13,))], quats={10: Q90X})
    p = plan_conditions(scd).patterns[0]
    # Rx(90) maps local -z to global +y; the quad (1,2,5,4) has area 2
    assert p.loads[1][1] == pytest.approx(2 / 4)
    assert p.loads[1][2] == pytest.approx(0.0, abs=1e-12)
    expected = quat_matrix(Q90X) @ np.array([0.0, 0.0, -1.0])
    assert np.allclose(expected, [0.0, 1.0, 0.0])


def test_edge_force_lumps_half_the_load_times_length_per_end() -> None:
    scd = make_scd(
        conditions=[cond(14, "ef", "Loads.Force.EdgeForce", edges=(0,), Mode="constant",
                         F=(1.0, 0.0, -2.0), Global=True)],
        steps=[load_pattern(9, load=(14,))])
    p = plan_conditions(scd).patterns[0]
    assert set(p.loads) == {1, 7}
    for n in (1, 7):
        assert p.loads[n] == pytest.approx((1.5, 0.0, -3.0, 0.0, 0.0, 0.0))


def test_node_force_goes_on_the_vertex_node_whole() -> None:
    scd = make_scd(
        conditions=[cond(15, "nf", "Loads.Force.NodeForce", vertices=(1,), Mode="constant",
                         F=(7.0, 0.0, -8.0))],
        steps=[load_pattern(9, load=(15,))])
    assert plan_conditions(scd).patterns[0].loads == {3: (7.0, 0.0, -8.0, 0.0, 0.0, 0.0)}


def test_mass_to_load_converts_gravity_times_the_edge_mass() -> None:
    am = cond(2, "am", "Mass.AutoEdgeMass", edges=(0,), rho=2.0e-3,
              **{"Convert to load": True, "gx": 0.0, "gy": 0.0, "gz": -9810.0, "Type": "load"})
    nf = cond(15, "nf", "Loads.Force.NodeForce", vertices=(2,), Mode="constant", F=(0.0, 0.0, -1.0))
    scd = make_scd(conditions=[am, nf], steps=[load_pattern(9, load=(15,), mass_to_load=(2,))])
    p = plan_conditions(scd).patterns[0]
    w = 2.0e-3 * 100.0 * 3.0 / 2 * -9810.0
    assert p.loads[1] == pytest.approx((0.0, 0.0, w, 0.0, 0.0, 0.0))
    assert p.loads[7][2] == pytest.approx(w - 1.0)            # the node force adds on vertex 2's node
    assert p.conditions == (15, 2)                            # load first, then massToLoad


def test_mass_to_load_is_inert_when_convert_to_load_is_off() -> None:
    am = cond(2, "am", "Mass.AutoEdgeMass", edges=(0,), rho=1.0,
              **{"Convert to load": False, "gx": 0.0, "gy": 0.0, "gz": -9810.0, "Type": "load"})
    scd = make_scd(conditions=[am], steps=[load_pattern(9, mass_to_load=(2,))])
    assert plan_conditions(scd).patterns[0].loads == {}


def test_zero_loads_are_dropped() -> None:
    scd = make_scd(
        conditions=[cond(15, "nf", "Loads.Force.NodeForce", vertices=(1,), Mode="constant",
                         F=(0.0, 0.0, 0.0))],
        steps=[load_pattern(9, load=(15,))])
    assert plan_conditions(scd).patterns[0].loads == {}


# ── C6: fix ───────────────────────────────────────────────────────────

FIX = dict(Ux=True, Uy=True, Uz=True, Rx=True, Ry=True, **{"Rz/3D": True, "2D": False, "3D": True,
           "U-R (Displacement+Rotation)": True})


def test_fix_collects_every_node_of_the_meshed_entities_and_the_vertices() -> None:
    scd = make_scd(
        conditions=[cond(1, "fix", "Constraints.sp.fix", vertices=(1,), edges=(1,), **FIX)],
        steps=[constraint_pattern(8, sp=(1,))])
    plan = plan_conditions(scd)
    # edge 1 carries mesh element 21 (nodes 1, 2), vertex 1 is node 3
    assert plan.fixes == {(1, 1, 1, 1, 1, 1): (1, 2, 3)}
    assert plan.constraint_conditions == (1,)


def test_fix_masks_group_nodes_and_nodes_stay_ascending() -> None:
    f2 = {**FIX, "Ux": False, "Uy": False, "Rz/3D": False}
    scd = make_scd(
        conditions=[cond(1, "a", "Constraints.sp.fix", vertices=(2,), **FIX),
                    cond(2, "b", "Constraints.sp.fix", vertices=(0, 1), **f2)],
        steps=[constraint_pattern(8, sp=(1, 2))])
    assert plan_conditions(scd).fixes == {
        (0, 0, 1, 1, 1, 0): (1, 3), (1, 1, 1, 1, 1, 1): (7,)}


def test_a_fix_no_pattern_references_is_not_applied_and_is_listed_as_ignored() -> None:
    scd = make_scd(conditions=[cond(1, "fix", "Constraints.sp.fix", vertices=(1,), **FIX)])
    plan = plan_conditions(scd)
    assert plan.fixes == {}
    assert [(i.xobj_meta, i.ids) for i in plan.ignored
            if i.category == "condition"] == [("Constraints.sp.fix", (1,))]


# ── C7: rigid diaphragm ───────────────────────────────────────────────

def test_rigid_diaphragm_pairs_come_from_the_interaction_links() -> None:
    scd = make_scd(
        conditions=[cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3)],
        steps=[constraint_pattern(8, mp=(33,))])
    plan = plan_conditions(scd)
    assert plan.diaphragm_pairs == frozenset((3, 8, s) for s in range(1, 7))


# ── C10: time series ──────────────────────────────────────────────────

def test_time_series_arguments_and_placeholder_flag() -> None:
    ts = [LINEAR, path_series(2, **{"-startTime": True, "tStart": 0.5}), path_series(3, **{"-factor": False})]
    plan = plan_conditions(make_scd(definitions=ts))
    by = {t.stko_id: t for t in plan.time_series}
    assert by[1] == TimeSeriesSpec(1, "Linear")
    assert (by[2].type, by[2].dt, by[2].factor, by[2].start_time, by[2].values, by[2].placeholder) == (
        "Path", 0.0025, 1000.0, 0.5, (0.0,), True)
    assert by[3].factor == 1.0 and by[3].start_time is None


def test_a_supplied_record_replaces_the_placeholder() -> None:
    plan = plan_conditions(make_scd(definitions=[path_series(2)]), records={2: [0.0, 0.1, -0.2]})
    t = plan.time_series[0]
    assert t.values == (0.0, 0.1, -0.2) and t.placeholder is False


def test_records_for_an_unknown_or_linear_series_are_refused() -> None:
    scd = make_scd(definitions=[LINEAR, path_series(2)])
    with pytest.raises(ValueError, match="not definitions"):
        plan_conditions(scd, records={9: [1.0]})
    with pytest.raises(ValueError, match="Linear"):
        plan_conditions(scd, records={1: [1.0]})


def test_a_document_series_with_real_values_is_not_a_placeholder() -> None:
    scd = make_scd(definitions=[path_series(2, list_of_values=(0.0, 1.0, 0.0))])
    t = plan_conditions(scd).time_series[0]
    assert t.values == (0.0, 1.0, 0.0) and not t.placeholder


# ── C11: Rayleigh ─────────────────────────────────────────────────────

def _rayleigh(**over: Any) -> XObject:
    a = {"Input Type": "Automatic", "alphaM/Rayleigh": 0.4, "Betak/Rayleigh": 0.002,
         "alphaM/RayleighUser": 0.9, "Betak/RayleighUser": 0.009,
         "Kcurr/Rayleigh": False, "Kinit/Rayleigh": True, "Kcomm/Rayleigh": False}
    a.update(over)
    return xo(7, "damping", "Misc_commands.rayleigh", **a)


def test_rayleigh_puts_beta_in_the_flagged_slots() -> None:
    r = plan_conditions(make_scd(steps=[_rayleigh()])).rayleigh
    assert (r.alpha_m, r.beta_k, r.beta_k_init, r.beta_k_comm) == (0.4, 0.0, 0.002, 0.0)


def test_rayleigh_manual_input_and_all_three_slots() -> None:
    r = plan_conditions(make_scd(steps=[_rayleigh(**{
        "Input Type": "Manual", "Kcurr/Rayleigh": True, "Kcomm/Rayleigh": True})])).rayleigh
    assert (r.alpha_m, r.beta_k, r.beta_k_init, r.beta_k_comm) == (0.9, 0.009, 0.009, 0.009)


def test_no_rayleigh_step_means_no_damping() -> None:
    assert plan_conditions(make_scd()).rayleigh is None


# ── C12: stages ───────────────────────────────────────────────────────

def _staged_scd(**kw: Any) -> ScdModel:
    nf = cond(15, "nf", "Loads.Force.NodeForce", vertices=(0,), Mode="constant", F=(1.0, 0.0, 0.0))
    return make_scd(
        conditions=[nf, *kw.pop("conditions", [])],
        definitions=[LINEAR, path_series(2)],
        steps=[
            load_pattern(9, load=(15,)), analyses(10, "selfWeight", numIncr=5),
            analyses(14, "dummy"),
            xo(21, "TH_x", "Patterns.addPattern.UniformExcitation", direction="dx", tsTag=2,
               **{"-fact": False, "-vel0": False, "-disp": False, "-vel": False, "-int": False}),
            analyses(24, "TH", static=False, loadConst=False, numIncr=24000, **{
                "algorithm": "Krylov-Newton", "-iterate/KrylovNewton": False,
                "-increment/KrylovNewton": False, "-maxDim/KrylovNewton": True,
                "maxDim/KrylovNewton": 10, "testCommand": "Norm Displacement Increment Test",
                "tol/NormDispIncr": 1e-4, "iter/NormDispIncr": 100, "pFlag/NormDispIncr": "0"}),
            *kw.pop("steps", []),
        ])


def test_each_analyses_command_closes_a_stage_that_owns_the_patterns_before_it() -> None:
    plan = plan_conditions(_staged_scd())
    assert [(s.stko_id, s.name, s.analysis, s.patterns) for s in plan.stages] == [
        (10, "selfWeight", "Static", (9,)), (14, "dummy", "Static", ()),
        (24, "TH", "Transient", (21,))]
    assert [(p.stko_id, p.kind) for p in plan.patterns] == [(9, "Plain"), (21, "UniformExcitation")]
    assert plan.patterns[1].direction == 1 and plan.patterns[1].series == 2


def test_static_stage_data_matches_the_analyses_command() -> None:
    s0 = plan_conditions(_staged_scd()).stages[0]
    assert (s0.n_incr, s0.duration, s0.load_const) == (5, 1.0, True)
    assert s0.integrator == ("LoadControl", (0.2,))
    assert s0.chain["constraints"] == {"type": "Auto", "verbose": False,
                                       "auto_penalty_oom": 3, "user_penalty": None}
    assert s0.chain["numberer"] == "ParallelRCM"
    assert s0.chain["system"] == {"type": "Mumps", "icntl14": 200, "matrix_type": None}
    assert s0.chain["test"] == {"type": "EnergyIncr", "tol": 1e-4, "max_iter": 10, "desired_iter": 10,
                                "print_flag": 0, "n_type": None}
    assert s0.chain["algorithm"] == {"type": "Linear", "factor_once": True}


def test_transient_stage_is_data_with_its_newmark_parameters_and_duration() -> None:
    th = plan_conditions(_staged_scd()).stages[2]
    assert (th.n_incr, th.duration, th.load_const) == (24000, 60.0, False)
    assert th.integrator == ("Newmark", (0.5, 0.25))
    assert th.chain["algorithm"] == {"type": "KrylovNewton", "max_dim": 10}
    assert th.chain["test"]["max_iter"] == 100 and th.chain["test"]["desired_iter"] == 100


def test_adaptive_transient_test_iterations_are_stored_as_stko_writes_them() -> None:
    """STKO writes ``2 x iter`` in the test of an adaptive stage (AnalysesCommand.py:
    ``iter = iter * 2``); iter itself is the driver's desired count. 1B: iter 50 ->
    ``test NormDispIncr 0.0001 100``; 1C/1D: 200 -> 400."""
    scd = _staged_scd()
    th = scd.analysis_steps[24]
    th.attributes.update({"Adaptive Time Step": True, "max factor": 1.0, "min factor": 1e-3,
                          "max factor incr": 1.5, "min factor incr": 1e-3})
    test = plan_conditions(scd).stages[2].chain["test"]
    assert (test["max_iter"], test["desired_iter"]) == (200, 100)


def test_test_print_flag_and_ntype_follow_their_use_flags() -> None:
    """test.py writeTcl_test: pFlag only with use_pFlag, nType only with use_nType."""
    nf = cond(15, "nf", "Loads.Force.NodeForce", vertices=(0,), Mode="constant", F=(1.0, 0.0, 0.0))
    off = make_scd(conditions=[nf], steps=[load_pattern(9, load=(15,)),
                                           analyses(10, **{"pFlag/EnergyIncr": "2"})])
    assert plan_conditions(off).stages[0].chain["test"]["print_flag"] == 0
    on = make_scd(conditions=[nf], steps=[load_pattern(9, load=(15,)), analyses(10, **{
        "pFlag/EnergyIncr": "2", "use_pFlag/EnergyIncr": True})])
    assert plan_conditions(on).stages[0].chain["test"]["print_flag"] == 2
    ntype = make_scd(conditions=[nf], steps=[load_pattern(9, load=(15,)), analyses(10, **{
        "use_nType/EnergyIncr": True, "nType/EnergyIncr": 1})])
    assert ("option", "Analyses.AnalysesCommand:stage") in _items(ntype)


def test_patterns_after_the_last_analysis_are_listed_as_ignored() -> None:
    scd = make_scd(conditions=[cond(15, "nf", "Loads.Force.NodeForce", vertices=(0,),
                                    Mode="constant", F=(1.0, 0.0, 0.0))],
                   definitions=[LINEAR], steps=[analyses(10), load_pattern(11, load=(15,))])
    plan = plan_conditions(scd)
    assert [(i.xobj_meta, i.ids) for i in plan.ignored if "never active" in i.reason] == [
        ("Patterns.addPattern.loadPattern", (11,))]


def test_recorders_regions_monitors_and_custom_commands_are_ignored_not_dropped() -> None:
    scd = make_scd(steps=[
        xo(1, "reg", "Misc_commands.region"), xo(3, "rec", "Recorders.MPCORecorder"),
        xo(5, "eb", "Misc_commands.customCommand", TCLscript="recorder EnergyBalance -time"),
        xo(6, "mon", "Misc_commands.monitor"),
        xo(23, "ie", "Misc_commands.ImplexAutoErrorControlActivate")])
    ig = {(i.xobj_meta): i for i in plan_conditions(scd).ignored}
    assert set(ig) == {"Misc_commands.region", "Recorders.MPCORecorder",
                       "Misc_commands.customCommand", "Misc_commands.monitor",
                       "Misc_commands.ImplexAutoErrorControlActivate"}
    assert ig["Misc_commands.customCommand"].payload == "recorder EnergyBalance -time"


def test_implex_error_control_is_ignored_only_after_the_last_analysis() -> None:
    """MAJOR-4: before an analysis it would change how that analysis runs (STKO's
    ImplexAutoErrorControlActivate.py writes the error-control setup); after the
    last one it controls nothing, like a pattern after the last analysis."""
    after = make_scd(steps=[analyses(10), xo(23, "ie", "Misc_commands.ImplexAutoErrorControlActivate")])
    assert check_supported(after) == []
    (ig,) = [i for i in plan_conditions(after).ignored
             if i.xobj_meta == "Misc_commands.ImplexAutoErrorControlActivate"]
    assert "no analysis follows it" in ig.reason
    before = make_scd(steps=[xo(9, "ie", "Misc_commands.ImplexAutoErrorControlActivate"), analyses(10)])
    assert [(u.category, u.xobj_meta, u.ids) for u in check_supported(before)] == [
        ("option", "Misc_commands.ImplexAutoErrorControlActivate:before analysis", (9,))]
    with pytest.raises(UnsupportedSTKOTypes, match="before analysis"):
        plan_conditions(before)


def test_a_static_stage_after_the_transient_stage_is_refused() -> None:
    scd = make_scd(steps=[analyses(10, "TH", static=False), analyses(11, "after")])
    assert [(u.xobj_meta, u.ids) for u in check_supported(scd)] == [
        ("Analyses.AnalysesCommand:static after transient", (11,))]
    ok = make_scd(steps=[analyses(10, "gravity"), analyses(11, "TH", static=False)])
    assert check_supported(ok) == []


# ── C13: IMPL-EX dTime targets ────────────────────────────────────────

def _implex_props() -> dict[int, XObject]:
    def ref(oid: int, name: str, t: str, key: str, **a: Any) -> XObject:
        return XObject(id=oid, name=name, type=t, attributes=a, references=frozenset({key}))

    return {
        5: xo(5, "UC", "materials.nD.ASDConcrete3D", integration="IMPL-EX", eta=0.0),
        6: ref(6, "plate", "materials.nD.PlateFromPlaneStress", "matTag", matTag=5),
        7: ref(7, "layered", "sections.LayeredShell", "matTag", matTag=(6,), thickness=(1.0,)),
        8: xo(8, "elastic concrete", "materials.nD.ASDConcrete3D", integration="Implicit", eta=0.0),
    }


def test_implex_targets_are_analysis_elements_whose_property_reaches_an_implex_material() -> None:
    scd = make_scd(props=_implex_props(), face_pp=(7, 3), edge_pp=(4, 5))
    # faces 0 (quad 10) reach material 5 through two references; quad 11 is plain elastic;
    # edge 1 carries the IMPL-EX material but no element property: eid 21 is not an analysis element
    assert plan_conditions(scd).implex_dt_targets == (10,)


def test_viscous_material_without_implex_is_a_target_too() -> None:
    props = {5: xo(5, "visc", "materials.nD.ASDConcrete3D", integration="Implicit", eta=0.01)}
    scd = make_scd(props=props, face_pp=(3, 5))
    assert plan_conditions(scd).implex_dt_targets == (11,)


def test_no_implex_material_no_targets() -> None:
    assert plan_conditions(make_scd(props=_implex_props(), face_pp=(3, 3))).implex_dt_targets == ()


# ── summaries ─────────────────────────────────────────────────────────

def test_summaries_describe_every_condition_as_intent() -> None:
    scd = make_scd(
        conditions=[
            cond(3, "fm", "Mass.FaceMass", faces=(0,), Mode="constant", mass=(1.0, 1.0, 1.0)),
            cond(14, "ef", "Loads.Force.EdgeForce", edges=(0,), Mode="constant",
                 F=(0.0, 0.0, -1.0), Global=True),
            cond(1, "fix", "Constraints.sp.fix", vertices=(1,), **FIX),
            cond(23, "lat", "Loads.Force.FaceForce", faces=(0,), Mode="function", F=(0.0, 0.0, 0.0),
                 Fx="1e-3*z**1.5", Fy="0", Fz="0", Global=True),
        ],
        steps=[load_pattern(9, load=(14,)), constraint_pattern(8, sp=(1,))])
    s = {x.stko_id: x for x in plan_conditions(scd).summaries}
    assert (s[3].per, s[3].value, s[3].patterns, s[3].pgs) == ("area", (1.0, 1.0, 1.0), (), {})
    assert (s[14].per, s[14].patterns, s[14].extra["global"]) == ("length", (9,), True)
    assert (s[1].per, s[1].value, s[1].patterns) == ("none", (1.0,) * 6, (8,))
    assert s[23].extra["function"] == ("1e-3*z**1.5", "0", "0")     # kept as intent though unreferenced


# ── check_supported: the loud failures ────────────────────────────────

def _items(scd: ScdModel) -> dict[tuple[str, str], Any]:
    return {(u.category, u.xobj_meta): u for u in check_supported(scd)}


def test_a_clean_document_has_nothing_unsupported() -> None:
    assert check_supported(_staged_scd()) == []


def test_unknown_and_tier_only_types_fail_with_one_error_listing_all_of_them() -> None:
    scd = make_scd(
        conditions=[
            cond(40, "emb", "Constraints.mp.ASDEmbeddedNodeElement"),
            cond(41, "rot", "Mass.NodeRotationalMass", vertices=(0,)),
            cond(42, "drm", "Loads.Generic.H5DRM"),
            cond(1, "fix", "Constraints.sp.fix", vertices=(1,), **FIX),
        ],
        steps=[constraint_pattern(8, sp=(1,), mp=(40,)),
               xo(30, "drm", "Patterns.addPattern.H5DRM"),
               xo(31, "abs", "Misc_commands.ASDAbsorbingBoundaryActivate"),
               xo(32, "ev", "Analyses.eigen"),
               xo(33, "ts", "Patterns.addPattern.loadPattern", tsTag=7, load=None, massToLoad=None,
                  eleLoad=None, sp=None, genericLoad=None, **{"-fact": False})],
        definitions=[xo(7, "const", "timeSeries.Constant")])
    items = _items(scd)
    assert items[("condition", "Constraints.mp.ASDEmbeddedNodeElement")].reason == "tier-only: not in this version"
    assert items[("condition", "Mass.NodeRotationalMass")].reason == "unknown STKO type"
    assert items[("analysis_step", "Patterns.addPattern.H5DRM")].reason == "tier-only: not in this version"
    assert items[("analysis_step", "Misc_commands.ASDAbsorbingBoundaryActivate")].reason == (
        "tier-only: not in this version")
    assert items[("analysis_step", "Analyses.eigen")].reason == "unknown STKO type"
    assert items[("definition", "timeSeries.Constant")].reason == "unknown STKO type"
    # the H5DRM condition is not a mass and no pattern references it: not scanned
    assert ("condition", "Loads.Generic.H5DRM") not in items
    with pytest.raises(UnsupportedSTKOTypes) as exc:
        plan_conditions(scd)
    msg = str(exc.value)
    for needle in ("ASDEmbeddedNodeElement", "NodeRotationalMass", "H5DRM", "Analyses.eigen",
                   "timeSeries.Constant", "ASDAbsorbingBoundaryActivate"):
        assert needle in msg
    assert len(exc.value.items) == len(check_supported(scd))


def test_every_analysis_step_type_is_supported_ignored_or_refused_never_dropped() -> None:
    tier_only = sorted(t for (cat, t), st in REGISTRY.items()
                       if cat == "analysis_step" and st == "tier_only")
    ignored = sorted(t for (cat, t), st in REGISTRY.items()
                     if cat == "analysis_step" and st == "ignored")
    scd = make_scd(steps=[xo(i, f"s{i}", t) for i, t in enumerate(
        [*tier_only, *ignored, "Misc_commands.unheard_of"], 1)])
    refused = {u.xobj_meta for u in check_supported(scd) if u.category == "analysis_step"}
    assert refused == {*tier_only, "Misc_commands.unheard_of"}
    listed = {i.xobj_meta for i in plan_conditions(make_scd(steps=[
        xo(i, f"s{i}", t) for i, t in enumerate(ignored, 1)])).ignored}
    assert listed == set(ignored)


def test_function_mode_is_refused_only_where_a_pattern_reaches_it() -> None:
    lat = dict(Mode="function", F=(0.0, 0.0, 0.0), Fx="1*z", Fy="0", Fz="0", Global=True)
    unref = make_scd(conditions=[cond(23, "lat", "Loads.Force.FaceForce", faces=(0,), **lat)])
    assert check_supported(unref) == []                     # 1C's unreferenced loads_Lateral_X
    ref = make_scd(conditions=[cond(23, "lat", "Loads.Force.FaceForce", faces=(0,), **lat)],
                   steps=[load_pattern(9, load=(23,))])
    assert ("option", "Loads.Force.FaceForce:Mode=function") in _items(ref)


def test_a_mass_in_function_mode_is_refused_even_without_a_pattern() -> None:
    scd = make_scd(conditions=[cond(3, "fm", "Mass.FaceMass", faces=(0,), Mode="function",
                                    mass=(0.0, 0.0, 0.0))])
    assert ("option", "Mass.FaceMass:Mode=function") in _items(scd)


def test_pattern_options_that_are_not_translated() -> None:
    scd = make_scd(
        definitions=[LINEAR, path_series(2, constant=False)],
        steps=[
            xo(9, "lp", "Patterns.addPattern.loadPattern", tsTag=1, load=None, massToLoad=None,
               eleLoad=(5,), sp=(6,), genericLoad=None, **{"-fact": True, "cFactor": 2.0}),
            xo(21, "ue", "Patterns.addPattern.UniformExcitation", direction="dx", tsTag=2,
               **{"-fact": True, "-vel0": True, "-disp": False, "-vel": False, "-int": False})],
    )
    keys = {k for k in _items(scd)}
    assert {("option", "Patterns.addPattern.loadPattern:-fact"),
            ("option", "Patterns.addPattern.loadPattern:eleLoad"),
            ("option", "Patterns.addPattern.loadPattern:sp"),
            ("option", "Patterns.addPattern.UniformExcitation:-fact"),
            ("option", "Patterns.addPattern.UniformExcitation:-vel0"),
            ("option", "timeSeries.Path:non_constant")} <= keys


def test_a_pattern_whose_time_series_does_not_exist_is_refused() -> None:
    scd = make_scd(steps=[load_pattern(9, ts=77)])
    assert ("option", "pattern:tsTag") in _items(scd)


def test_conflicting_fix_masks_and_a_slave_in_two_diaphragms_are_refused() -> None:
    other = {**FIX, "Ux": False}
    scd = make_scd(
        conditions=[cond(1, "a", "Constraints.sp.fix", vertices=(1,), **FIX),
                    cond(2, "b", "Constraints.sp.fix", vertices=(1,), **other)],
        steps=[constraint_pattern(8, sp=(1, 2))])
    assert ("option", "Constraints.sp.fix:conflicting masks") in _items(scd)

    scd2 = make_scd(
        conditions=[cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3)],
        steps=[constraint_pattern(8, mp=(33,))])
    link = scd2.interactions[1]
    object.__setattr__(link, "elements", (30, 31, 32))
    scd2.mesh.elements[32] = MeshElement(600, 0, (7, 1))        # node 1 slave of masters 8 and 7
    assert ("option", "Constraints.mp.rigidDiaphragm:slave in two diaphragms") in _items(scd2)


def test_fix_dimension_and_model_type_options() -> None:
    f = {**FIX, "U-R (Displacement+Rotation)": False}
    scd = make_scd(conditions=[cond(1, "fix", "Constraints.sp.fix", vertices=(1,), **f)],
                   steps=[constraint_pattern(8, sp=(1,))])
    assert ("option", "Constraints.sp.fix:ModelType") in _items(scd)


def test_auto_edge_mass_options() -> None:
    base = {"Convert to load": True, "gx": 0.0, "gy": 0.0, "gz": -1.0}
    eleload = cond(2, "am", "Mass.AutoEdgeMass", edges=(0,), rho=1.0, Type="eleLoad", **base)
    scd = make_scd(conditions=[eleload], steps=[load_pattern(9, mass_to_load=(2,))])
    assert ("option", "Mass.AutoEdgeMass:Type") in _items(scd)
    # the same condition not converted by any pattern is harmless
    assert ("option", "Mass.AutoEdgeMass:Type") not in _items(make_scd(conditions=[eleload]))

    no_area = make_scd(
        conditions=[cond(2, "am", "Mass.AutoEdgeMass", edges=(0,), rho=1.0, Type="load", **base)],
        edge_pp=(3, 0))                                          # an ElasticMembranePlateSection
    assert ("option", "Mass.AutoEdgeMass:section area") in _items(no_area)

    two = make_scd(
        props={13: xo(13, "rect", "sections.RectangularFiberSection", Width=1.0, Height=1.0),
               12: XObject(id=12, name="bsp", type="special_purpose.BeamSectionProperty",
                           attributes={"secTag/1": 13, "other": 4}, references=frozenset({"secTag/1", "other"}))},
        edge_pp=(12, 0),
        conditions=[cond(2, "am", "Mass.AutoEdgeMass", edges=(0,), rho=1.0, Type="load", **base)])
    assert ("option", "Mass.AutoEdgeMass:section closure") in _items(two)


def test_a_mass_to_load_that_is_not_an_auto_edge_mass_is_refused() -> None:
    scd = make_scd(
        conditions=[cond(3, "fm", "Mass.FaceMass", faces=(0,), Mode="constant", mass=(1.0, 1.0, 1.0))],
        steps=[load_pattern(9, mass_to_load=(3,))])
    assert ("option", "Mass.FaceMass:massToLoad") in _items(scd)


def test_mesh_element_type_under_a_load_is_checked() -> None:
    scd = make_scd(
        conditions=[cond(13, "ff", "Loads.Force.FaceForce", faces=(0,), Mode="constant",
                         F=(0.0, 0.0, 1.0), Global=True)],
        steps=[load_pattern(9, load=(13,))])
    scd.mesh.elements[10] = MeshElement(401, 0, (1, 2, 5, 4, 1, 2, 5, 4))
    assert ("option", "Loads.Force.FaceForce:element type") in _items(scd)


def test_stage_options_that_are_not_translated() -> None:
    scd = make_scd(steps=[
        analyses(10, "adaptive", **{"Adaptive Time Step": True}),
        analyses(11, "dc", staticIntegrators="Displacement Control"),
        analyses(12, "keep", loadConst=False),
        analyses(13, "empty", numIncr=0),
        analyses(14, "lm", algorithm="Secant Newton"),
        analyses(15, "tr", static=False, transientIntegrators="Explicit Bathe"),
        analyses(16, "mat", **{"-matrixType": True, "matrixType": "Symmetric General"}),
    ])
    reasons = {u.reason for u in check_supported(scd) if u.category == "option"}
    for needle in ("Adaptive Time Step on a static stage", "staticIntegrators='Displacement Control'",
                   "loadConst=False on a static stage", "duration * numIncr = 0",
                   "algorithm='Secant Newton'", "transientIntegrators='Explicit Bathe'",
                   "Symmetric General"):
        assert any(needle in r for r in reasons), needle


def test_rayleigh_and_constraints_after_the_first_analysis_are_refused_and_a_second_rayleigh_too() -> None:
    nf = cond(15, "nf", "Loads.Force.NodeForce", vertices=(0,), Mode="constant", F=(1.0, 0.0, 0.0))
    scd = make_scd(
        conditions=[nf, cond(1, "fix", "Constraints.sp.fix", vertices=(1,), **FIX)],
        definitions=[LINEAR],
        steps=[analyses(5), _rayleigh(), constraint_pattern(8, sp=(1,))])
    keys = _items(scd)
    assert ("option", "Misc_commands.rayleigh:after analysis") in keys
    assert ("option", "Patterns.addPattern.constraintPattern:after analysis") in keys
    two = make_scd(steps=[_rayleigh(), xo(8, "r2", "Misc_commands.rayleigh", **_rayleigh().attributes)])
    assert ("option", "Misc_commands.rayleigh:multiple") in _items(two)


def test_diaphragm_groups_come_from_referenced_conditions_only() -> None:
    from apeGmsh.interop.stko.translate_conditions import diaphragm_groups
    rd = cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3)
    assert diaphragm_groups(make_scd(conditions=[rd])) == ()
    (d,) = diaphragm_groups(make_scd(conditions=[rd], steps=[constraint_pattern(8, mp=(33,))]))
    assert (d.condition, d.perp_dirn, d.master_node, d.slave_nodes) == (33, 3, 8, (1, 2, 3, 4, 5, 6))
    assert (d.master_pg, d.slave_pg) == ("rd:33:8:master", "rd:33:8:slaves")


def test_diaphragm_groups_the_bridge_cannot_carry_verbatim_are_refused() -> None:
    rd = cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3)
    cp = constraint_pattern(8, mp=(33,))
    # a slave at the master's coordinates: the nearest-node master pick is ambiguous
    at_master = make_scd(conditions=[rd], steps=[cp])
    at_master.mesh.coordinates[list(at_master.mesh.node_ids).index(5)] = NODES[8]
    assert [(u.xobj_meta, u.ids) for u in check_supported(at_master)] == [
        ("Constraints.mp.rigidDiaphragm:slave at the master", (33,))]
    # one node in two groups (here: a second condition on the same links)
    rd2 = cond(34, "rd2", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3)
    two = make_scd(conditions=[rd, rd2], steps=[constraint_pattern(8, mp=(33, 34))])
    keys = {u.xobj_meta for u in check_supported(two)}
    assert keys == {"Constraints.mp.rigidDiaphragm:node in two diaphragms"}


# ── declare_session ───────────────────────────────────────────────────

class _FakeConstraints:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def rigid_diaphragm(self, *a: Any, **k: Any) -> None:
        self.calls.append((a, k))


def _mesh_map(diaphragms=()) -> MeshMap:
    return MeshMap(
        node_ids=tuple(sorted(NODES)), element_groups=(), subshape_entities={},
        selection_set_pgs={}, condition_pgs={}, diaphragms=tuple(diaphragms), carrier_ids=range(100, 100))


def test_declare_session_issues_one_rigid_diaphragm_per_group() -> None:
    scd = make_scd(
        conditions=[cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3)],
        steps=[constraint_pattern(8, mp=(33,))])
    plan = plan_conditions(scd)
    mesh = _mesh_map([DiaphragmGroup(33, 3, 8, (1, 2, 3, 4, 5, 6), "rd:33:8:master", "rd:33:8:slaves")])
    g = SimpleNamespace(constraints=_FakeConstraints())
    declare_session(g, scd, plan, mesh)
    (args, kw), = g.constraints.calls
    assert args == ("rd:33:8:master", "rd:33:8:slaves")
    assert kw["master_point"] == (2.0, 0.5, 0.0) and kw["plane_normal"] == (0.0, 0.0, 1.0)
    assert kw["constrained_dofs"] == [1, 2, 6]
    # every slave is in the master's plane: the tolerance is only the 1e-6 floor
    assert 0 < kw["plane_tolerance"] < 1e-3
    assert kw["name"] == "rd:33:8"


def test_declare_session_tolerance_keeps_an_off_plane_slave() -> None:
    """MAJOR-1: the resolver drops nodes farther than plane_tolerance from the
    master's plane; STKO (and OpenSees' rigidDiaphragm, with a warning) keeps them.
    The tolerance covers the farthest slave, so the record is STKO's verbatim."""
    scd = make_scd(
        conditions=[cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3)],
        steps=[constraint_pattern(8, mp=(33,))])
    scd.mesh.coordinates[list(scd.mesh.node_ids).index(6), 2] = 0.3     # slave 6 off-plane
    plan = plan_conditions(scd)
    mesh = _mesh_map([DiaphragmGroup(33, 3, 8, (1, 2, 3, 4, 5, 6), "m", "s")])
    g = SimpleNamespace(constraints=_FakeConstraints())
    declare_session(g, scd, plan, mesh)
    assert g.constraints.calls[0][1]["plane_tolerance"] > 0.3


def test_declare_session_refuses_a_mesh_that_disagrees_with_the_plan() -> None:
    scd = make_scd(
        conditions=[cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3)],
        steps=[constraint_pattern(8, mp=(33,))])
    plan = plan_conditions(scd)
    g = SimpleNamespace(constraints=_FakeConstraints())
    with pytest.raises(ValueError, match="disagree"):
        declare_session(g, scd, plan, _mesh_map([DiaphragmGroup(33, 3, 8, (1, 2), "m", "s")]))
    assert g.constraints.calls == []


def test_declare_session_without_diaphragms_does_nothing() -> None:
    scd = make_scd()
    g = SimpleNamespace(constraints=_FakeConstraints())
    declare_session(g, scd, plan_conditions(scd), _mesh_map())
    assert g.constraints.calls == []


@pytest.mark.parametrize("perp, normal, dofs", [
    (1, (1.0, 0.0, 0.0), [2, 3, 4]), (2, (0.0, 1.0, 0.0), [1, 3, 5]), (3, (0.0, 0.0, 1.0), [1, 2, 6])])
def test_declare_session_normal_and_dofs_follow_perp_dirn(perp: int, normal: tuple, dofs: list) -> None:
    scd = make_scd(
        conditions=[cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=perp)],
        steps=[constraint_pattern(8, mp=(33,))])
    plan = plan_conditions(scd)
    mesh = _mesh_map([DiaphragmGroup(33, perp, 8, (1, 2, 3, 4, 5, 6), "m", "s")])
    g = SimpleNamespace(constraints=_FakeConstraints())
    declare_session(g, scd, plan, mesh)
    kw = g.constraints.calls[0][1]
    assert kw["plane_normal"] == normal and kw["constrained_dofs"] == dofs


# ── build_conditions on a recording bridge ────────────────────────────

class _Rec:
    """Records every call made on it; the object a call returns records too."""

    def __init__(self, log: list, path: str = "ops") -> None:
        self._log, self._path = log, path

    def __getattr__(self, name: str) -> "_Rec":
        if name.startswith("__"):
            raise AttributeError(name)
        return _Rec(self._log, f"{self._path}.{name}")

    def __call__(self, *a: Any, **k: Any) -> "_Rec":
        self._log.append((self._path, a, k))
        return _Rec(self._log, f"{self._path}()")

    def __enter__(self) -> "_Rec":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _calls(log: list, path: str) -> list[tuple[tuple, dict]]:
    return [(a, k) for p, a, k in log if p == path]


def _plan_for_build() -> ConditionsPlan:
    nf = cond(15, "nf", "Loads.Force.NodeForce", vertices=(0,), Mode="constant", F=(1.0, 0.0, 0.0))
    fm = cond(3, "fm", "Mass.NodeMass", vertices=(1,), Mode="constant", mass=(2.0, 2.0, 2.0))
    fix = cond(1, "fix", "Constraints.sp.fix", vertices=(2,), **FIX)
    scd = make_scd(
        conditions=[nf, fm, fix],
        definitions=[LINEAR, path_series(2, **{"-startTime": True, "tStart": 0.25})],
        steps=[_rayleigh(), constraint_pattern(8, sp=(1,)), load_pattern(9, load=(15,)),
               analyses(10, "gravity", numIncr=4),
               xo(21, "TH", "Patterns.addPattern.UniformExcitation", direction="dz", tsTag=2,
                  **{"-fact": False, "-vel0": False, "-disp": False, "-vel": False, "-int": False}),
               analyses(24, "TH", static=False)])
    return plan_conditions(scd)


def test_build_conditions_declares_fix_mass_rayleigh_and_the_series_patterns_use() -> None:
    log: list = []
    series = build_conditions(_Rec(log), _plan_for_build(), _mesh_map())
    assert _calls(log, "ops.fix") == [((), {"nodes": [7], "dofs": (1, 1, 1, 1, 1, 1)})]
    assert _calls(log, "ops.mass") == [((), {"nodes": [3], "values": (2.0, 2.0, 2.0, 0.0, 0.0, 0.0)})]
    assert _calls(log, "ops.damping.rayleigh") == [((), {
        "alpha_m": 0.4, "beta_k": 0.0, "beta_k_init": 0.002, "beta_k_comm": 0.0})]
    assert _calls(log, "ops.timeSeries.Linear") == [((), {"factor": 1.0})]
    assert _calls(log, "ops.timeSeries.Path") == [((), {
        "values": (0.0,), "dt": 0.0025, "factor": 1000.0, "start_time": 0.25})]
    assert sorted(series) == [1, 2]       # the UniformExcitation's series is created for the TH driver


def test_build_conditions_static_stage_holds_its_pattern_chain_and_run() -> None:
    log: list = []
    build_conditions(_Rec(log), _plan_for_build(), _mesh_map())
    assert _calls(log, "ops.stage") == [(("gravity",), {})]               # the transient stage is data
    assert len(_calls(log, "ops.stage().pattern")) == 1                    # one Plain pattern
    assert _calls(log, "ops.stage().pattern().load") == [((), {"node": 1, "forces": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)})]
    assert _calls(log, "ops.stage().run") == [((), {"n_increments": 4})]
    assert _calls(log, "ops.integrator.LoadControl") == [((), {"dlam": 0.25})]
    assert _calls(log, "ops.analysis.Static") == [((), {})]
    (_, kw), = _calls(log, "ops.stage().analysis")
    assert set(kw) == {"test", "algorithm", "integrator", "constraints", "numberer", "system", "analysis"}
    assert _calls(log, "ops.test.EnergyIncr") == [((), {"tol": 1e-4, "max_iter": 10, "print_flag": 0})]
    assert _calls(log, "ops.algorithm.Linear") == [((), {"tangent": "tangent", "factor_once": True})]
    assert not _calls(log, "ops.pattern.UniformExcitation")


def test_build_conditions_serial_chain_swaps_the_mpi_choices_and_stko_chain_keeps_them() -> None:
    serial: list = []
    build_conditions(_Rec(serial), _plan_for_build(), _mesh_map())
    assert _calls(serial, "ops.numberer.RCM") and not _calls(serial, "ops.numberer.ParallelRCM")
    assert _calls(serial, "ops.system.Pardiso") and not _calls(serial, "ops.system.Mumps")
    stko: list = []
    build_conditions(_Rec(stko), _plan_for_build(), _mesh_map(), chain="stko")
    assert _calls(stko, "ops.numberer.ParallelRCM")
    assert _calls(stko, "ops.system.Mumps") == [((), {"icntl14": 200})]
    for log in (serial, stko):
        assert _calls(log, "ops.constraints.Auto") == [((), {"verbose": False, "auto_penalty_oom": 3.0})]


def test_build_conditions_without_stages_declares_the_plain_patterns_globally() -> None:
    log: list = []
    build_conditions(_Rec(log), _plan_for_build(), _mesh_map(), stages=False)
    assert not _calls(log, "ops.stage")
    assert len(_calls(log, "ops.pattern.Plain")) == 1
    assert _calls(log, "ops.pattern.Plain().load") == [((), {"node": 1, "forces": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)})]
    assert not _calls(log, "ops.stage().analysis")


def test_build_conditions_resets_the_implex_dtime_at_each_static_stage_as_stko_does() -> None:
    """Rule C13 / STKO_DT_UTIL_OnBeforeAnalyze: at increment 1 of every static stage
    STKO sets dTimeCommit, dTimeInitial and dTime of every target element to the
    stage's increment duration / numIncr (1B/1C: 0.5 and 0.2 in different stages,
    so OpenSees' own carried increment would differ). Transient stages are the
    time-history driver's."""
    scd = make_scd(props=_implex_props(), face_pp=(7, 3), definitions=[LINEAR, path_series(2)],
                   steps=[analyses(10, "g1", numIncr=2), analyses(11, "g2", numIncr=5),
                          analyses(12, "TH", static=False)])
    plan = plan_conditions(scd)
    assert plan.implex_dt_targets == (10,)
    log: list = []
    build_conditions(_Rec(log), plan, _mesh_map())
    assert _calls(log, "ops.stage().update_parameter") == [
        ((name, dt), {"elements": (10,)})
        for dt in (0.5, 0.2) for name in ("dTimeCommit", "dTimeInitial", "dTime")]
    off: list = []
    build_conditions(_Rec(off), plan, _mesh_map(), implex_dtime=False)
    assert not _calls(off, "ops.stage().update_parameter")
    none: list = []
    build_conditions(_Rec(none), plan_conditions(make_scd(steps=[analyses(10)])), _mesh_map())
    assert not _calls(none, "ops.stage().update_parameter")


def test_build_conditions_refuses_nodes_the_session_does_not_have() -> None:
    plan = _plan_for_build()
    mesh = MeshMap(node_ids=(1, 2), element_groups=(), subshape_entities={}, selection_set_pgs={},
                   condition_pgs={}, diaphragms=(), carrier_ids=range(0))
    with pytest.raises(ValueError, match="not in the session's mesh"):
        build_conditions(_Rec([]), plan, mesh)


def test_chain_choices_map_to_the_primitives_that_exist() -> None:
    nf = cond(15, "nf", "Loads.Force.NodeForce", vertices=(0,), Mode="constant", F=(1.0, 0.0, 0.0))
    over = {"constraints": "Plain Constraints", "numbererType": "Reverse Cuthill-McKee Numberer",
            "system": "UmfPack SOE", "-lvalueFact": False, "testCommand": "Norm Unbalance Test",
            "tol/NormUnbalance": 1e-3, "iter/NormUnbalance": 25, "pFlag/NormUnbalance": "1",
            "use_pFlag/NormUnbalance": True,
            "algorithm": "Krylov-Newton", "-iterate/KrylovNewton": True, "tangIter/KrylovNewton": "initial",
            "-increment/KrylovNewton": False, "-maxDim/KrylovNewton": True, "maxDim/KrylovNewton": 7}
    scd = make_scd(conditions=[nf], definitions=[LINEAR],
                   steps=[load_pattern(9, load=(15,)), analyses(10, **over)])
    log: list = []
    build_conditions(_Rec(log), plan_conditions(scd), _mesh_map())
    assert _calls(log, "ops.constraints.Plain") and _calls(log, "ops.numberer.RCM")
    assert _calls(log, "ops.system.UmfPack")
    assert _calls(log, "ops.test.NormUnbalance") == [((), {"tol": 1e-3, "max_iter": 25, "print_flag": 1})]
    assert _calls(log, "ops.algorithm.KrylovNewton") == [((), {"iterate": "initial", "increment": None, "max_dim": 7})]


def test_penalty_and_lagrange_constraint_handlers_map_with_their_weights() -> None:
    nf = cond(15, "nf", "Loads.Force.NodeForce", vertices=(0,), Mode="constant", F=(1.0, 0.0, 0.0))
    pen = {"constraints": "Penalty Method", "alphaS/penaltyMethod": 1e12, "alphaM/penaltyMethod": 1e10}
    lag = {"constraints": "Lagrange Multipliers", "Optional lagrangeMultipliers": True,
           "alphaS/LagrangeMultipliers": 2.0, "alphaM/LagrangeMultipliers": 3.0}
    lag_default = {"constraints": "Lagrange Multipliers", "Optional lagrangeMultipliers": False}
    for over, call, kw in ((pen, "ops.constraints.Penalty", {"alpha_sp": 1e12, "alpha_mp": 1e10}),
                           (lag, "ops.constraints.Lagrange", {"alpha_sp": 2.0, "alpha_mp": 3.0}),
                           (lag_default, "ops.constraints.Lagrange", {})):
        scd = make_scd(conditions=[nf], steps=[load_pattern(9, load=(15,)), analyses(10, **over)])
        log: list = []
        build_conditions(_Rec(log), plan_conditions(scd), _mesh_map())
        assert _calls(log, call) == [((), kw)]


# ── a real session and bridge ─────────────────────────────────────────

def _slab_session(g: Any) -> MeshMap:
    """The slab of :func:`make_scd` in a real session, as the mesh module builds it:
    discrete entities carrying STKO's own node and element ids (never ``generate``),
    the diaphragm master and slaves each owned by a carrier point entity with a PG,
    the two quads on a surface entity with the PG ``slab``."""
    import gmsh

    model = gmsh.model
    def coords(ns: list[int]) -> list[float]:
        return [c for n in ns for c in NODES[n]]

    master = model.addDiscreteEntity(0)
    model.mesh.addNodes(0, master, [8], coords([8]))
    model.mesh.addElementsByType(master, 15, [900], [8])
    slaves = model.addDiscreteEntity(0)
    model.mesh.addNodes(0, slaves, [1, 2, 3, 4, 5, 6], coords([1, 2, 3, 4, 5, 6]))
    model.mesh.addElementsByType(slaves, 15, [901, 902, 903, 904, 905, 906], [1, 2, 3, 4, 5, 6])
    slab = model.addDiscreteEntity(2)
    model.mesh.addElementsByType(slab, 3, [10, 11], [1, 2, 5, 4, 2, 3, 6, 5])
    g.physical.add(0, [master], name="rd:33:8:master")
    g.physical.add(0, [slaves], name="rd:33:8:slaves")
    g.physical.add(2, [slab], name="slab")
    return _mesh_map([DiaphragmGroup(33, 3, 8, (1, 2, 3, 4, 5, 6), "rd:33:8:master", "rd:33:8:slaves")])


def _slab_scd() -> ScdModel:
    fix = {**FIX, "Ux": False, "Uy": False, "Rz/3D": False}
    return make_scd(
        conditions=[
            cond(3, "fm", "Mass.FaceMass", faces=(0, 1), Mode="constant", mass=(1.0, 1.0, 1.0)),
            cond(1, "fix", "Constraints.sp.fix", vertices=(1,), **fix),
            cond(13, "ff", "Loads.Force.FaceForce", faces=(0, 1), Mode="constant",
                 F=(0.0, 0.0, -2.0), Global=True),
            cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3),
        ],
        steps=[constraint_pattern(8, sp=(1,), mp=(33,)), load_pattern(9, load=(13,)),
               analyses(10, "gravity", numIncr=3)])


def test_a_real_session_resolves_the_declared_diaphragm_to_the_plans_pairs(g) -> None:
    scd = _slab_scd()
    plan = plan_conditions(scd)
    mesh = _slab_session(g)
    declare_session(g, scd, plan, mesh)
    fem = g.mesh.queries.get_fem_data(dim=None)
    pairs = {(3, int(c.master_node), int(c.slave_node)) for c in fem.nodes.constraints.pairs()}
    assert pairs == set(plan.diaphragm_pairs) == {(3, 8, s) for s in range(1, 7)}


def _carried_session(g: Any, coords: dict[int, tuple[float, float, float]], *, decoy: bool) -> MeshMap:
    """:func:`_slab_session` with explicit coordinates and, with ``decoy``, an
    unrelated node 9 at the master's exact coordinates owned by another entity."""
    import gmsh

    model = gmsh.model

    def xyz(ns: list[int]) -> list[float]:
        return [c for n in ns for c in coords[n]]

    master = model.addDiscreteEntity(0)
    model.mesh.addNodes(0, master, [8], xyz([8]))
    model.mesh.addElementsByType(master, 15, [900], [8])
    slaves = model.addDiscreteEntity(0)
    model.mesh.addNodes(0, slaves, [1, 2, 3, 4, 5, 6], xyz([1, 2, 3, 4, 5, 6]))
    model.mesh.addElementsByType(slaves, 15, [901, 902, 903, 904, 905, 906], [1, 2, 3, 4, 5, 6])
    slab = model.addDiscreteEntity(2)
    model.mesh.addElementsByType(slab, 3, [10, 11], [1, 2, 5, 4, 2, 3, 6, 5])
    if decoy:
        other = model.addDiscreteEntity(0)
        model.mesh.addNodes(0, other, [9], xyz([8]))
        model.mesh.addElementsByType(other, 15, [907], [9])
    g.physical.add(0, [master], name="rd:33:8:master")
    g.physical.add(0, [slaves], name="rd:33:8:slaves")
    g.physical.add(2, [slab], name="slab")
    return _mesh_map([DiaphragmGroup(33, 3, 8, (1, 2, 3, 4, 5, 6), "rd:33:8:master", "rd:33:8:slaves")])


def test_a_real_session_keeps_an_off_plane_slave_and_the_carried_master(g) -> None:
    """MAJOR-1 regression: slave 6 lies 0.3 off the master's plane (the old 1e-6
    tolerance dropped it) and an unrelated node 9 sits exactly on the master (the
    resolver's first nearest-node pick searches the whole model). The resolved
    records are still STKO's pairs, and verify_diaphragms says so."""
    from apeGmsh.interop.stko.translate_conditions import resolved_diaphragm_pairs, verify_diaphragms

    scd = _slab_scd()
    coords = dict(NODES)
    coords[6] = (4.0, 1.0, 0.3)
    scd.mesh.coordinates[list(scd.mesh.node_ids).index(6)] = coords[6]
    plan = plan_conditions(scd)
    mesh = _carried_session(g, coords, decoy=True)
    declare_session(g, scd, plan, mesh)
    fem = g.mesh.queries.get_fem_data(dim=None)
    assert sorted(resolved_diaphragm_pairs(fem)) == sorted(plan.diaphragm_pairs)
    assert {s for _, _, s in plan.diaphragm_pairs} == {1, 2, 3, 4, 5, 6}
    verify_diaphragms(fem, plan)


def test_the_old_tolerance_would_have_dropped_the_off_plane_slave(g) -> None:
    """The failure the fix removes, shown on the same session: the resolver with a
    1e-6 tolerance (the previous default) drops slave 6, and verify_diaphragms
    catches it."""
    from apeGmsh.interop.stko.translate_conditions import verify_diaphragms

    scd = _slab_scd()
    coords = dict(NODES)
    coords[6] = (4.0, 1.0, 0.3)
    scd.mesh.coordinates[list(scd.mesh.node_ids).index(6)] = coords[6]
    plan = plan_conditions(scd)
    _carried_session(g, coords, decoy=False)
    g.constraints.rigid_diaphragm("rd:33:8:master", "rd:33:8:slaves", master_point=NODES[8],
                                  plane_normal=(0.0, 0.0, 1.0), constrained_dofs=[1, 2, 6],
                                  plane_tolerance=1e-6)
    fem = g.mesh.queries.get_fem_data(dim=None)
    with pytest.raises(ValueError, match=r"plan only: \[\(3, 8, 6\)\]"):
        verify_diaphragms(fem, plan)


def test_verify_diaphragms_refuses_records_that_are_not_the_plans() -> None:
    from apeGmsh.interop.stko.translate_conditions import verify_diaphragms

    plan = plan_conditions(make_scd(
        conditions=[cond(33, "rd", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3)],
        steps=[constraint_pattern(8, mp=(33,))]))

    def fem(*records: tuple[int, int, list[int]]) -> Any:
        return SimpleNamespace(nodes=SimpleNamespace(constraints=SimpleNamespace(
            rigid_diaphragms=lambda: iter(records))))

    verify_diaphragms(fem((3, 8, [1, 2, 3, 4, 5, 6])), plan)
    with pytest.raises(ValueError, match="plan only"):
        verify_diaphragms(fem((3, 8, [1, 2, 3, 4, 5])), plan)                 # a dropped slave
    with pytest.raises(ValueError, match="resolved only"):
        verify_diaphragms(fem((3, 5, [1, 2, 3, 4, 6, 8])), plan)              # another master
    with pytest.raises(ValueError, match="resolved twice"):
        verify_diaphragms(fem((3, 8, [1, 2, 3, 4, 5, 6]), (3, 8, [6])), plan)


def test_build_conditions_checks_the_bridges_diaphragms_before_declaring(g) -> None:
    from apeGmsh.opensees import apeSees

    scd = _slab_scd()
    mesh = _slab_session(g)
    declare_session(g, scd, plan_conditions(scd), mesh)
    fem = g.mesh.queries.get_fem_data(dim=None)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    # a plan without the diaphragm: the FEM's record is not the plan's
    no_rd = dataclasses.replace(plan_conditions(scd), diaphragm_pairs=frozenset())
    with pytest.raises(ValueError, match="resolved only"):
        build_conditions(ops, no_rd, MeshMap(
            node_ids=tuple(int(n) for n in fem.nodes.ids), element_groups=(), subshape_entities={},
            selection_set_pgs={}, condition_pgs={}, diaphragms=(), carrier_ids=range(0)))


def test_emitted_tcl_has_the_masses_fix_loads_diaphragm_and_stage(g, tmp_path: Path) -> None:
    import re

    from apeGmsh.opensees import apeSees

    scd = _slab_scd()
    plan = plan_conditions(scd)
    mesh = _slab_session(g)
    declare_session(g, scd, plan, mesh)
    fem = g.mesh.queries.get_fem_data(dim=None)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=3.0e4, nu=0.2, h=0.1, rho=0.0)
    ops.element.ASDShellQ4(pg="slab", section=sec)
    mesh_for_build = MeshMap(
        node_ids=tuple(int(n) for n in fem.nodes.ids), element_groups=(), subshape_entities={},
        selection_set_pgs={}, condition_pgs={}, diaphragms=mesh.diaphragms, carrier_ids=range(0))
    build_conditions(ops, plan, mesh_for_build)
    path = tmp_path / "slab.tcl"
    # The STKO fixture fixes node 3 only; diaphragm master 8 is a lone
    # point, so the bridge warns that its uz, rx, ry are free (#1333).
    # The deck under test is the translator's, unchanged.
    from apeGmsh.opensees import DetachedDiaphragmMasterWarning
    with pytest.warns(DetachedDiaphragmMasterWarning, match="master node 8"):
        ops.tcl(str(path), progress=False)
    text = path.read_text(encoding="utf-8")

    masses = {int(m.group(1)): [float(v) for v in m.group(2).split()]
              for m in re.finditer(r"^mass (\d+) (.*)$", text, re.M)}
    assert set(masses) == set(plan.masses) == {1, 2, 3, 4, 5, 6}
    assert all(v[3:] == [0.0] * 3 for v in masses.values())
    assert sum(v[0] for v in masses.values()) == pytest.approx(4.0)       # unit areal mass x area 4
    assert re.findall(r"^fix (\d+) 0 0 1 1 1 0$", text, re.M) == ["3"]
    pairs = {(int(m.group(1)), int(m.group(2)), int(s)) for m in
             re.finditer(r"^rigidDiaphragm (\d+) (\d+) (.*)$", text, re.M) for s in m.group(3).split()}
    assert pairs == set(plan.diaphragm_pairs)
    assert "# === Stage: gravity ===" in text
    assert text.index("# === Stage: gravity ===") < text.index("pattern Plain")
    loads = [float(m.group(1)) for m in re.finditer(r"^\s*load \d+ \S+ \S+ (\S+)", text, re.M)]
    assert sum(loads) == pytest.approx(-8.0)                              # -2 per area x area 4
    assert re.search(r"^analyze|analyze 1", text, re.M)
