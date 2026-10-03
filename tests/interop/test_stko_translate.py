"""STKO translator, end to end (ADR 0111; rules section 9 of
``internal_docs/stko_translator_rules.md``): ``collect_unsupported``,
``translate_scd`` and ``build_opensees`` on one synthetic document that every
slice supports.

Synthetic document (N, mm, t)::

    4 ---- 5 ---- 6        geometry 1: faces 0, 1 carry quads 10 (1 2 5 4), 11 (2 3 6 5)
    | q10  | q11  |        edge 0 carries column 20 (7 1); edge 1 the edge mesh 21 (1 2), no element
    1 ---- 2 ---- 3        vertices 0, 1, 2 -> nodes 1, 3, 7
    |
    7 (column base, z = -3000)
    8 (geometry 2's vertex): rigid-diaphragm master, links 30..35 -> slaves 1..6

Conditions: FaceMass on both faces, fix on node 7 (all) and on the master
(0 0 1 1 1 0), FaceForce on both faces, rigid diaphragm (perpDirn 3). Steps:
constraint pattern, Rayleigh, one load pattern, one static analysis.

The oracle test at the end runs the four Tier-1 documents when
``APEGMSH_STKO_ORACLES`` points at ``models/stko-rev0-tcl`` (the research
repository's folder; the one environment variable of every STKO oracle test
in ``tests/interop``); CI never has them. It reads STKO's deck files by
name (:data:`STKO_DECK_FILES`), never by glob, so a stray file in an oracle
folder cannot enter the comparison. The full per-category deck parity
lives in that repository (``models/apegmsh/validation/translator_parity.py``).
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import apeGmsh.interop.stko.translate as translate_module
from apeGmsh.interop.stko import (
    TranslateResult,
    UnsupportedSTKOTypes,
    build_opensees,
    collect_unsupported,
    declare_translation,
    translate_scd,
)
from apeGmsh.interop.stko.model import (
    Condition,
    Geometry,
    Interaction,
    Mesh,
    MeshElement,
    ScdModel,
    SelectionSet,
    SubShapeRef,
    SubShapes,
    XObject,
)
from apeGmsh.interop.stko.translate_types import Unsupported

NODES = {
    1: (0.0, 0.0, 0.0), 2: (2000.0, 0.0, 0.0), 3: (4000.0, 0.0, 0.0),
    4: (0.0, 1000.0, 0.0), 5: (2000.0, 1000.0, 0.0), 6: (4000.0, 1000.0, 0.0),
    7: (0.0, 0.0, -3000.0), 8: (2000.0, 500.0, 0.0),
}
IDENTITY = (0.0, 0.0, 0.0, 1.0)
#: Column along +z: local x = z, vecxz (column 2) = -x after a -90 deg turn about y.
Q_COLUMN = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))
FIX_ALL = {"Ux": True, "Uy": True, "Uz": True, "Rx": True, "Ry": True, "Rz/3D": True,
           "2D": False, "3D": True, "U-R (Displacement+Rotation)": True}


def xo(oid: int, name: str, xtype: str, refs: tuple[str, ...] = (), **attrs: Any) -> XObject:
    return XObject(id=oid, name=name, type=xtype, attributes=dict(attrs), references=frozenset(refs))


def shell_ep() -> XObject:
    return xo(1, "Shell", "shell.ASDShellQ4", **{
        "Drilling DOF Type": "Elastic Drilling DOF", "Drilling Stabilization": 0.01,
        "Kinematics": "Linear", "Use EAS": True})


def beam_ep() -> XObject:
    return xo(2, "Column", "beam_column_elements.elasticBeamColumn", **{
        "-alpha": False, "-cMass": False, "-depth": False, "-mass": False, "-releasey": False,
        "-releasez": False, "2D": False, "3D": True, "Dimension": "3D", "massDens": 0.0,
        "transfType": "Linear"})


def slab_section() -> XObject:
    return xo(3, "Slab", "sections.ElasticMembranePlateSection",
              E=25000.0, Ep_mod=1.0, h=200.0, nu=0.2, rho=0.0)


def column_section() -> XObject:
    return xo(4, "C70x70", "sections.Elastic", refs=("Eb material", "Em material", "G material"), **{
        "A_modifier": 1.0, "Iyy_modifier": 1.0, "Izz_modifier": 1.0, "J_modifier": 1.0,
        "Dimension": "3D", "E": 25000.0, "G": 10500.0, "Eb material": 0, "Em material": 0,
        "G material": 0, "Section": {"PROPS": np.array([490000.0, 2.0e10, 2.0e10, 3.4e10])},
        "Shear Deformable": False, "Use Uniaxial Materials": False,
        "Y/section_offset": 0.0, "Z/section_offset": 0.0})


def cond(oid: int, name: str, xtype: str, geometry: dict[int, SubShapes] | None = None,
         interactions: tuple[int, ...] = (), **attrs: Any) -> Condition:
    return Condition(xobject=xo(oid, name, xtype, **attrs), geometry=geometry or {},
                     interactions=interactions)


def analyses(oid: int, name: str) -> XObject:
    return xo(oid, name, "Analyses.AnalysesCommand", **{
        "Static": True, "numIncr": 2, "duration": 1.0, "duration/transient": 60.0,
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
        "algorithm": "Linear", "use_formTangent/Linear": False, "-factorOnce": True})


def make_scd(*, extra_conditions: tuple[Condition, ...] = (), extra_steps: tuple[XObject, ...] = (),
             extra_interactions: tuple[Interaction, ...] = (),
             face_ep: tuple[int, int] = (1, 1)) -> ScdModel:
    ids = np.array(sorted(NODES), dtype=np.int32)
    elements = {
        10: MeshElement(303, 201, (1, 2, 5, 4)), 11: MeshElement(303, 201, (2, 3, 6, 5)),
        20: MeshElement(102, 2, (7, 1)), 21: MeshElement(102, 2, (1, 2)),
        **{30 + i: MeshElement(600, 0, (8, s)) for i, s in enumerate(range(1, 7))},
    }
    mesh = Mesh(
        node_ids=ids, coordinates=np.array([NODES[i] for i in ids], dtype=float),
        node_flags=np.zeros(len(ids), dtype=np.int32), elements=elements,
        domains={(1, "faces"): {0: np.array([10]), 1: np.array([11])},
                 (1, "edges"): {0: np.array([20]), 1: np.array([21])}},
        vertex_nodes={1: np.array([1, 3, 7]), 2: np.array([8])},
        orientation={10: IDENTITY, 11: IDENTITY, 20: Q_COLUMN, 21: IDENTITY},
    )
    slab = Geometry(
        id=1, name="slab", shape_index=1, counts={"vertices": 3, "edges": 2, "faces": 2},
        element_property={"vertices": np.zeros(3, int), "edges": np.array([2, 0]),
                          "faces": np.array(face_ep)},
        physical_property={"vertices": np.zeros(3, int), "edges": np.array([4, 0]),
                           "faces": np.array([3, 3])},
        local_axes={},
    )
    master = Geometry(id=2, name="master", shape_index=2, counts={"vertices": 1},
                      element_property={"vertices": np.zeros(1, int)},
                      physical_property={"vertices": np.zeros(1, int)}, local_axes={})
    link = Interaction(id=1, name="rd", type="NN", masters=(SubShapeRef(2, 1, 0),),
                       slaves=(SubShapeRef(1, 3, 0), SubShapeRef(1, 3, 1)),
                       elements=tuple(range(30, 36)))
    conditions = (
        cond(1, "base", "Constraints.sp.fix", {1: SubShapes(vertices=(2,))}, **FIX_ALL),
        cond(2, "spDiaphragm", "Constraints.sp.fix", {2: SubShapes(vertices=(0,))},
             **{**FIX_ALL, "Ux": False, "Uy": False, "Rz/3D": False}),
        cond(3, "mass_slab", "Mass.FaceMass", {1: SubShapes(faces=(0, 1))},
             Mode="constant", mass=(1.0e-6, 1.0e-6, 1.0e-6)),
        cond(4, "dead", "Loads.Force.FaceForce", {1: SubShapes(faces=(0, 1))},
             Mode="constant", F=(0.0, 0.0, -0.005), Global=True),
        cond(5, "diaphragm", "Constraints.mp.rigidDiaphragm", interactions=(1,), perpDirn=3),
        *extra_conditions,
    )
    steps = (
        xo(6, "constraints", "Patterns.addPattern.constraintPattern", sp=(1, 2), mp=(5,)),
        xo(7, "damping", "Misc_commands.rayleigh", **{
            "Input Type": "Automatic", "alphaM/Rayleigh": 0.4, "Betak/Rayleigh": 0.002,
            "alphaM/RayleighUser": 0.9, "Betak/RayleighUser": 0.009,
            "Kcurr/Rayleigh": False, "Kinit/Rayleigh": True, "Kcomm/Rayleigh": False}),
        xo(9, "gravity_loads", "Patterns.addPattern.loadPattern", tsTag=1, load=(4,),
           massToLoad=(), eleLoad=None, sp=None, genericLoad=None,
           **{"-fact": False, "cFactor": 1.0}),
        analyses(10, "gravity"),
        *extra_steps,
    )
    return ScdModel(
        path=Path("synthetic.scd"), version=(4, 1, 0), geometries={1: slab, 2: master},
        mesh=mesh,
        selection_sets={"Slab": SelectionSet(id=1, name="Slab", items={1: SubShapes(faces=(0, 1))})},
        physical_properties={3: slab_section(), 4: column_section()},
        element_properties={1: shell_ep(), 2: beam_ep()},
        conditions={c.id: c for c in conditions},
        interactions={1: link, **{i.id: i for i in extra_interactions}}, local_axes={},
        definitions={1: xo(1, "lin", "timeSeries.Linear", **{"-factor": False, "cFactor": 1.0})},
        analysis_steps={s.id: s for s in sorted(steps, key=lambda s: s.id)},
    )


@pytest.fixture
def scd() -> ScdModel:
    return make_scd()


# ── collect_unsupported ──────────────────────────────────────────────

def test_the_synthetic_document_is_fully_supported(scd) -> None:
    assert collect_unsupported(scd) == ()


def test_collect_unsupported_lists_offenders_of_every_slice_at_once() -> None:
    bad = make_scd(
        face_ep=(1, 5),
        extra_interactions=(Interaction(id=2, name="embedded", type="NE", masters=(), slaves=()),),
        extra_conditions=(cond(40, "drm", "Loads.Generic.H5DRM"),),
        extra_steps=(xo(20, "drm", "Patterns.addPattern.H5DRM"),),
    )
    bad.element_properties[5] = xo(5, "Spring", "zero_length_elements.zeroLength")
    items = collect_unsupported(bad)
    keys = {(u.category, u.xobj_meta) for u in items}
    assert ("interaction", "NE") in keys                                          # mesh slice
    assert ("element_property", "zero_length_elements.zeroLength") in keys         # props slice
    assert ("analysis_step", "Patterns.addPattern.H5DRM") in keys                 # conditions slice
    assert list(items) == sorted(items, key=lambda u: (u.category, u.xobj_meta, u.reason))
    with pytest.raises(UnsupportedSTKOTypes) as err:
        translate_scd(object(), bad)          # raises before the session is used at all
    assert err.value.items == items
    assert "zero_length_elements.zeroLength" in str(err.value) and "H5DRM" in str(err.value)


def test_collect_unsupported_merges_the_same_offender_reported_by_two_slices(monkeypatch, scd) -> None:
    a = Unsupported("element_property", "x.Type", (3,), ("b",), "tier-only: not in this version")
    b = Unsupported("element_property", "x.Type", (1,), ("a",), "tier-only: not in this version")
    c = Unsupported("condition", "y.Type", (9,), ("c",), "unknown STKO type")
    monkeypatch.setattr(translate_module.translate_mesh, "check_supported", lambda s: [a])
    monkeypatch.setattr(translate_module.translate_props, "check_supported", lambda s: [b, c])
    monkeypatch.setattr(translate_module.translate_conditions, "check_supported", lambda s: [])
    assert collect_unsupported(scd) == (
        Unsupported("condition", "y.Type", (9,), ("c",), "unknown STKO type"),
        Unsupported("element_property", "x.Type", (1, 3), ("a", "b"), "tier-only: not in this version"),
    )


# ── translate_scd ────────────────────────────────────────────────────

def test_translate_scd_puts_stkos_mesh_in_the_session_with_its_ids(g, scd) -> None:
    result = translate_scd(g, scd)
    assert isinstance(result, TranslateResult) and result.scd is scd
    fem = g.mesh.queries.get_fem_data(dim=None)
    assert sorted(int(n) for n in fem.nodes.ids) == list(range(1, 9))
    assert result.mesh.node_ids == tuple(range(1, 9))
    by_dim = {grp.dim: grp for grp in result.mesh.element_groups}
    assert by_dim[2].element_ids == (10, 11) and by_dim[1].element_ids == (20,)
    assert sorted(int(e) for e in fem.elements.select(pg=by_dim[2].pg).ids) == [10, 11]
    pairs = {(3, int(c.master_node), int(c.slave_node)) for c in fem.nodes.constraints.pairs()}
    assert pairs == set(result.plan.diaphragm_pairs) == {(3, 8, s) for s in range(1, 7)}


def test_translate_scd_fills_each_condition_summary_with_its_session_groups(g, scd) -> None:
    result = translate_scd(g, scd)
    summaries = {s.stko_id: s for s in result.plan.summaries}
    for cid, pgs in result.mesh.condition_pgs.items():
        if cid in summaries:
            assert dict(summaries[cid].pgs) == dict(pgs)
    assert summaries[3].pgs                        # the face mass has a session group
    assert result.ignored == result.plan.ignored


def test_translate_scd_reads_a_path(g, scd, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(translate_module, "read_scd", lambda p: seen.append(p) or scd)
    result = translate_scd(g, "model.scd")
    assert seen == ["model.scd"] and result.scd is scd


# ── build_opensees ───────────────────────────────────────────────────

def _deck(g, scd, tmp_path: Path, **kw: Any) -> str:
    result = translate_scd(g, scd)
    fem = g.mesh.queries.get_fem_data(dim=None)
    ops = build_opensees(fem, result, **kw)
    path = tmp_path / "deck.tcl"
    ops.tcl(str(path), progress=False)
    return path.read_text(encoding="utf-8")


def _element_tags(text: str) -> dict[int, tuple[str, tuple[int, ...]]]:
    """Element tag -> (type, first nodes) of every ``element`` line."""
    out = {}
    for m in re.finditer(r"^element (\S+) (\d+) (.*)$", text, re.M):
        nn = 4 if m.group(1) == "ASDShellQ4" else 2
        out[int(m.group(2))] = (m.group(1), tuple(int(v) for v in m.group(3).split()[:nn]))
    return out


def test_build_opensees_emits_the_whole_document(g, scd, tmp_path: Path) -> None:
    text = _deck(g, scd, tmp_path)
    nodes = {int(m.group(1)) for m in re.finditer(r"^node (\d+) ", text, re.M)}
    assert nodes == set(NODES)
    # element_tags="fem" (the default): the deck's element tags are STKO's ids.
    assert _element_tags(text) == {
        10: ("ASDShellQ4", (1, 2, 5, 4)), 11: ("ASDShellQ4", (2, 3, 6, 5)),
        20: ("elasticBeamColumn", (7, 1))}
    shells = re.findall(r"^element ASDShellQ4 \d+ (\d+ \d+ \d+ \d+) (\d+) (.*)$", text, re.M)
    assert sorted(s[0] for s in shells) == ["1 2 5 4", "2 3 6 5"]
    assert all("-drillingStab 0.01" in s[2] and "-local 1.0 0.0 0.0" in s[2] for s in shells)
    assert re.search(r"^section ElasticMembranePlateSection \d+ 25000.0 0.2 200.0 0.0", text, re.M)
    beams = re.findall(r"^element elasticBeamColumn \d+ 7 1 490000.0 25000.0 10500.0 "
                       r"34000000000.0 20000000000.0 20000000000.0 \d+$", text, re.M)
    assert len(beams) == 1
    masses = {int(m.group(1)): float(m.group(2)) for m in
              re.finditer(r"^mass (\d+) (\S+) ", text, re.M)}
    assert sum(masses.values()) == pytest.approx(1.0e-6 * 4000.0 * 1000.0)
    assert sorted(re.findall(r"^fix (\d+) (.*)$", text, re.M)) == [
        ("7", "1 1 1 1 1 1"), ("8", "0 0 1 1 1 0")]
    pairs = {(int(m.group(1)), int(m.group(2)), int(s)) for m in
             re.finditer(r"^rigidDiaphragm (\d+) (\d+) (.*)$", text, re.M) for s in m.group(3).split()}
    assert pairs == {(3, 8, s) for s in range(1, 7)}
    assert re.search(r"^rayleigh 0.4 0.0 0.002 0.0$", text, re.M)
    assert "# === Stage: gravity ===" in text and "loadConst -time 0.0" in text
    loads = [float(m.group(1)) for m in re.finditer(r"^\s*load \d+ \S+ \S+ (\S+)", text, re.M)]
    assert sum(loads) == pytest.approx(-0.005 * 4000.0 * 1000.0)


def test_build_opensees_sequential_tags_are_the_bridges_own(g, scd, tmp_path: Path) -> None:
    tags = _element_tags(_deck(g, scd, tmp_path, element_tags="sequential"))
    first = min(tags)
    assert sorted(tags) == [first, first + 1, first + 2] and set(tags) != {10, 11, 20}
    assert sorted(v for v in tags.values()) == [
        ("ASDShellQ4", (1, 2, 5, 4)), ("ASDShellQ4", (2, 3, 6, 5)),
        ("elasticBeamColumn", (7, 1))]


def test_build_opensees_refuses_a_bridge_without_the_tag_option(g, scd, monkeypatch) -> None:
    result = translate_scd(g, scd)
    fem = g.mesh.queries.get_fem_data(dim=None)
    monkeypatch.setattr(translate_module, "_bridge_takes_element_tags", lambda: False)
    with pytest.raises(TypeError, match="element_tags"):
        build_opensees(fem, result)


def test_build_opensees_without_stages_declares_the_patterns_globally(g, scd, tmp_path: Path) -> None:
    text = _deck(g, scd, tmp_path, stages=False)
    assert "# === Stage" not in text and "analyze" not in text
    assert re.search(r"^pattern Plain \d+ \d+ \{", text, re.M)


def test_build_opensees_refuses_an_unknown_tag_mode(g, scd) -> None:
    result = translate_scd(g, scd)
    fem = g.mesh.queries.get_fem_data(dim=None)
    with pytest.raises(ValueError, match="element_tags"):
        build_opensees(fem, result, element_tags="stko")  # type: ignore[arg-type]


def test_declare_translation_returns_the_primitives_and_series(g, scd) -> None:
    from apeGmsh.opensees import apeSees

    result = translate_scd(g, scd)
    ops = apeSees(g.mesh.queries.get_fem_data(dim=None))
    ops.model(ndm=3, ndf=6)
    props, series = declare_translation(ops, result)
    assert set(props.primitives) == {3}            # the Elastic section is read, not built (P2)
    assert len(props.elements) == 2 and set(series) == {1}


# ── Tier-1 oracles (local only) ──────────────────────────────────────

ORACLES = os.environ.get("APEGMSH_STKO_ORACLES")
#: STKO's own export files, read by name: anything else in an oracle folder
#: (a translated deck, a pickled plan) never enters the comparison.
STKO_DECK_FILES = ("nodes.tcl", "elements.tcl")
TIER1 = {
    "1A": ("Tier_1/1A/1A_TH_000.scd", "Tier_1/1A/input files", 14003, 13704),
    "1B": ("Tier_1/1B/1B_TH_000.scd", "Tier_1/1B/campaign_sta2_rup1", 2547, 2456),
    "1C": ("Tier_1/1C/1C_TH_000.scd", "Tier_1/1C/campaign_sta2_rup1", 3979, 3704),
    "1D": ("Tier_1/1D/1D_TH_000.scd", "Tier_1/1D/campaign_sta2_rup1", 13459, 13160),
}


@pytest.mark.skipif(not ORACLES, reason="APEGMSH_STKO_ORACLES is not set: the San Ramon "
                    "Tier-1 documents and STKO's decks live outside this repository")
@pytest.mark.parametrize("case", sorted(TIER1))
def test_tier1_document_translates_to_stkos_node_set_and_mass(case: str, tmp_path: Path) -> None:
    from apeGmsh import apeGmsh

    scd_rel, deck_rel, n_nodes, n_elements = TIER1[case]
    root = Path(str(ORACLES))
    deck = root / deck_rel
    stko_nodes: dict[int, float] = {}
    stko_elements: dict[int, tuple[int, ...]] = {}
    for name in STKO_DECK_FILES:
        for line in (deck / name).read_text().splitlines():
            t = line.split()
            if t[:1] == ["node"]:
                m = float(t[t.index("-mass") + 1]) if "-mass" in t else 0.0
                stko_nodes[int(t[1])] = max(stko_nodes.get(int(t[1]), 0.0), m)
            elif t[:1] == ["element"]:
                nn = 4 if t[1] == "ASDShellQ4" else 2
                stko_elements[int(t[2])] = tuple(int(v) for v in t[3:3 + nn])
    with apeGmsh(model_name=f"oracle_{case}", verbose=False) as g:
        result = translate_scd(g, root / scd_rel)
        fem = g.mesh.queries.get_fem_data(dim=None)
    ops = build_opensees(fem, result)
    path = tmp_path / f"{case}.tcl"
    ops.tcl(str(path), progress=False)
    text = path.read_text(encoding="utf-8")
    nodes = {int(m.group(1)) for m in re.finditer(r"^node (\d+) ", text, re.M)}
    assert nodes == set(stko_nodes) and len(nodes) == n_nodes
    ours = {tag: conn for tag, (_, conn) in _element_tags(text).items()}
    assert len(ours) == n_elements
    assert ours == stko_elements          # STKO's element ids as tags, same node order
    mass = sum(float(m.group(1)) for m in re.finditer(r"^mass \d+ (\S+) ", text, re.M))
    assert mass == pytest.approx(sum(stko_nodes.values()), rel=1e-9)
