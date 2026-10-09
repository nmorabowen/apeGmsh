"""ADR 0117 D3 (AS4-a): the assembly coupling verbs beyond ``tie``.

``node``, ``equal_dof``, ``rigid_link``, ``rigid_diaphragm``, ``embedded``
and ``couple(kind="kinematic"|"distributing")``. Oracles, each naming the
right answer independently of the code under test:

* records — each verb resolves to the closed-form record set of its ports:
  the 9 co-located interface pairs of two stacked 2x2x2 blocks, a master
  that is the reference node with offsets ``x_i - x_ref``, the 9 nodes of a
  2x2 plate per diaphragm, one embedded node per node of a 1x1x1 cube that
  lies strictly inside the host, and RBE2 / RBE3 rows on the 9 top nodes.
* reference nodes — FEM id ``k`` for the ``k``-th node, decoupled, labelled,
  and the instance windows unchanged by declaring one (ADR 0038 rounding).
* INV-7 — a coupling that resolves zero records, or names a missing group,
  raises naming both ports; a bare unknown port raises listing the
  instances; a refused call records nothing.
* INV-1 — node and coupling names have no ``.`` and share one namespace with
  instance labels and tie names.
* reload — every kind round-trips through ``Assembly.h5`` / ``from_h5``
  (params as JSON, ``n_records``); a row with foreign params is refused on
  write and on read.
* D3 — ``contact`` / ``interface`` / ``embed`` / ``reinforce`` are not
  assembly verbs, and ``couple(kind="contact")`` raises.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from tests.assembly.test_two_instances_one_tie import (
    GRANULE,
    H,
    SIDE,
    block_fem,
    declare_block,
    declare_plate,
    plate_fem,
    write_instance,
)

#: A rotation that stands the xy plate up in the plane y = 0.
STAND_UP = ((1.0, 0.0, 0.0), math.pi / 2)
#: Vertical offset of the second wall: a 2 mm gap, so no node coincides.
WALL_DZ = SIDE + 2.0
#: The diaphragm reference node, between the two walls.
CM = (5.0, 0.0, SIDE + 1.0)
#: The RBE2 / RBE3 reference node, above the block's top centre.
REF = (SIDE / 2, SIDE / 2, H + 5.0)
#: Placement of the 1x1x1 cube embedded in the block: no node on the
#: block's 5 mm grid, every node strictly inside.
CUBE_AT = (1.3, 1.3, 1.3)


def cube_fem(workdir: Path):
    """A single hex8 cube of side 2, PG ``Vol``."""
    with apeGmsh(model_name="cube", save_to=str(workdir / "cube_mesh.h5"),
                 overwrite=True) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, label="c")
        g.physical.add_volume("c", name="Vol")
        g.mesh.recipe.structured(size=2.0, fallback="strict")
        return g.mesh.queries.get_fem_data(dim=None)


def dotted_block_fem(workdir: Path):
    """``block_fem`` whose interface groups have dotted source names:
    ``deck.bot`` (z=0) and ``deck.top`` (z=H). Elements stay on ``Vol``."""
    from tests.assembly.test_two_instances_one_tie import _faces_at_z

    with apeGmsh(model_name="dblock", save_to=str(workdir / "dblock_mesh.h5"),
                 overwrite=True) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, SIDE, SIDE, H, label="v")
        g.physical.add_volume("v", name="Vol")
        g.physical.add_volume("v", name="core.all")
        g.physical.add_surface(_faces_at_z(g, 0.0), name="deck.bot")
        g.physical.add_surface(_faces_at_z(g, H), name="deck.top")
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        return g.mesh.queries.get_fem_data(dim=None)


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("as4a")
    return {
        "block": write_instance(d / "block.h5", block_fem(d), declare_block),
        "dblock": write_instance(d / "dblock.h5", dotted_block_fem(d),
                                 declare_block),
        "plate": write_instance(d / "plate.h5", plate_fem(d), declare_plate),
        "cube": write_instance(d / "cube.h5", cube_fem(d), declare_block),
        "dir": d,
    }


def _stack(files):
    from apeGmsh.assembly import Assembly

    return (Assembly("stack")
            .instance("pier_1", files["block"])
            .instance("pier_2", files["block"], translate=(0.0, 0.0, H)))


def _walls(files):
    from apeGmsh.assembly import Assembly

    return (Assembly("walls")
            .instance("w1", files["plate"], rotate=STAND_UP)
            .instance("w2", files["plate"], rotate=STAND_UP,
                      translate=(0.0, 0.0, WALL_DZ))
            .node("cm", CM))


def _ids(fem, **sel) -> list[int]:
    return sorted(int(i) for i in fem.nodes.select(**sel).ids)


def _coords(fem) -> dict[int, np.ndarray]:
    return dict(zip((int(i) for i in fem.nodes.ids),
                    np.asarray(fem.nodes.coords, dtype=float)))


def _node_records(fem, kind: str) -> list:
    return [r for r in fem.nodes.constraints if r.kind == kind]


def _elem_records(fem, kind: str) -> list:
    return [r for r in fem.elements.constraints if r.kind == kind]


# ---------------------------------------------------------------------------
# Reference nodes
# ---------------------------------------------------------------------------

def test_reference_nodes_are_labelled_decoupled_ids_below_every_window(files):
    from apeGmsh.mesh.FEMData import PROVENANCE_DECOUPLED

    plain = _stack(files).bridge(ndm=3, ndf=3).fem
    fem = (_stack(files).node("a", (1.0, 2.0, 3.0)).node("b", REF)
           .bridge(ndm=3, ndf=3).fem)
    assert _ids(fem, label="a") == [1] and _ids(fem, label="b") == [2]
    c = _coords(fem)
    np.testing.assert_array_equal(c[1], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(c[2], REF)
    prov = dict(zip((int(i) for i in fem.nodes.ids), fem.nodes.provenance))
    assert prov[1] == prov[2] == PROVENANCE_DECOUPLED
    assert {int(i) for i in fem.nodes.decoupled_ids} == {1, 2}
    # Declaring a node moves no instance: the windows start at GRANULE.
    rest = sorted(int(i) for i in fem.nodes.ids if int(i) > 2)
    assert rest == sorted(int(i) for i in plain.nodes.ids)
    assert min(rest) == GRANULE


@pytest.mark.parametrize("name, match", [
    ("a.b", "contains '.'"), ("", "non-empty"), ("pier_1", "already declared"),
    ("_a", "start or end"),
])
def test_node_name_rules(files, name, match):
    from apeGmsh.assembly import AssemblyError

    asm = _stack(files)
    with pytest.raises(AssemblyError, match=match):
        asm.node(name, REF)
    assert asm.nodes == ()


def test_one_namespace_for_instances_nodes_and_ties(files):
    from apeGmsh.assembly import AssemblyError

    asm = _stack(files).node("ref", REF)
    with pytest.raises(AssemblyError, match="already declared"):
        asm.instance("ref", files["block"])
    with pytest.raises(AssemblyError, match="already declared"):
        asm.tie("pier_1.top", "pier_2.bot", name="pier_2")
    with pytest.raises(AssemblyError, match="already declared"):
        asm.equal_dof("pier_1.top", "pier_2.bot", name="ref")
    asm.equal_dof("pier_1.top", "pier_2.bot", name="glue")
    with pytest.raises(AssemblyError, match="already declared"):
        asm.node("glue", REF)
    with pytest.raises(AssemblyError, match="non-finite|finite"):
        asm.node("n2", (0.0, math.nan, 0.0))
    assert [n.name for n in asm.nodes] == ["ref"]
    assert [t.name for t in asm.ties] == ["glue"]


# ---------------------------------------------------------------------------
# Each verb resolves its closed-form record set
# ---------------------------------------------------------------------------

def test_equal_dof_pairs_the_nine_coincident_interface_nodes(files):
    fem = (_stack(files).equal_dof("pier_1.top", "pier_2.bot", dofs=[1, 2, 3])
           .bridge(ndm=3, ndf=3).fem)
    recs = _node_records(fem, "equal_dof")
    c = _coords(fem)
    assert len(recs) == 9
    assert {r.master_node for r in recs} == set(_ids(fem, pg="pier_1.top"))
    assert {r.slave_node for r in recs} == set(_ids(fem, pg="pier_2.bot"))
    for r in recs:
        assert list(r.dofs) == [1, 2, 3]
        np.testing.assert_allclose(c[r.master_node], c[r.slave_node], atol=1e-12)


def test_rigid_link_masters_on_the_reference_node(files):
    fem = (_stack(files).node("ref", REF)
           .rigid_link("ref", "pier_2.top", link_type="rod")
           .bridge(ndm=3, ndf=3).fem)
    recs = _node_records(fem, "rigid_rod")
    c = _coords(fem)
    assert sorted(r.slave_node for r in recs) == _ids(fem, pg="pier_2.top")
    for r in recs:
        assert r.master_node == 1
        np.testing.assert_allclose(r.offset, c[r.slave_node] - np.array(REF),
                                   atol=1e-12)


def test_rigid_diaphragm_gathers_each_wall_on_the_reference_node(files):
    fem = (_walls(files)
           .rigid_diaphragm("cm", "w1.Slab", plane_normal=(0, 1, 0),
                            constrained_dofs=(1, 3, 5))
           .rigid_diaphragm("cm", "w2.Slab", plane_normal=(0, 1, 0),
                            constrained_dofs=(1, 3, 5))
           .bridge(ndm=3, ndf=6).fem)
    recs = _node_records(fem, "rigid_diaphragm")
    assert len(recs) == 2
    assert [r.master_node for r in recs] == [1, 1]
    assert [sorted(r.slave_nodes) for r in recs] == [
        _ids(fem, pg="w1.Slab"), _ids(fem, pg="w2.Slab")]
    for r in recs:
        assert list(r.dofs) == [1, 3, 5]
        np.testing.assert_allclose(r.plane_normal, (0.0, 1.0, 0.0))


def test_rigid_diaphragm_needs_a_master_point_for_an_instance_master(files):
    from apeGmsh.assembly import AssemblyError

    asm = _walls(files)
    with pytest.raises(AssemblyError, match="master_point= is required"):
        asm.rigid_diaphragm("w1.Slab", "w2.Slab", plane_normal=(0, 1, 0))
    asm.rigid_diaphragm("w1.Slab", "w2.Slab", plane_normal=(0, 1, 0),
                        master_point=(0.0, 0.0, 0.0), constrained_dofs=(1, 3, 5))
    fem = asm.bridge(ndm=3, ndf=6).fem
    (rec,) = _node_records(fem, "rigid_diaphragm")
    c = _coords(fem)
    np.testing.assert_allclose(c[rec.master_node], (0.0, 0.0, 0.0), atol=1e-12)
    assert len(rec.slave_nodes) == 17  # 9 + 9 wall nodes, less the master


def test_embedded_ties_every_cube_node_into_the_block(files):
    from apeGmsh.assembly import Assembly

    fem = (Assembly("emb").instance("host", files["block"])
           .instance("bar", files["cube"], translate=CUBE_AT)
           .embedded("host.Vol", "bar.Vol")
           .bridge(ndm=3, ndf=3).fem)
    recs = _elem_records(fem, "embedded")
    host = set(_ids(fem, pg="host.Vol"))
    assert sorted(r.slave_node for r in recs) == _ids(fem, pg="bar.Vol")
    assert len(recs) == 8
    for r in recs:
        assert set(int(m) for m in r.master_nodes) <= host
        assert math.isclose(float(np.sum(r.weights)), 1.0, abs_tol=1e-12)


def test_kinematic_and_distributing_couple_the_top_to_the_reference(files):
    fem = (_stack(files).node("ref", REF)
           .couple("pier_2.top", kind="kinematic", reference="ref", name="rbe2")
           .couple("pier_1.top", kind="distributing", reference="ref",
                   name="rbe3")
           .bridge(ndm=3, ndf=3).fem)
    (rbe2,) = _node_records(fem, "kinematic_coupling")
    assert rbe2.master_node == 1 and rbe2.name == "rbe2"
    assert sorted(rbe2.slave_nodes) == _ids(fem, pg="pier_2.top")
    assert list(rbe2.dofs) == []          # dofs=None: every slave DOF
    (rbe3,) = _elem_records(fem, "distributing")
    assert rbe3.slave_node == 1 and rbe3.name == "rbe3"
    assert sorted(int(m) for m in rbe3.master_nodes) == _ids(fem, pg="pier_1.top")
    assert rbe3.weights is None          # uniform


def test_the_deck_emits_each_coupling(files, tmp_path):
    from apeGmsh.assembly import Assembly

    ops = (_stack(files).node("ref", REF)
           .couple("pier_2.top", kind="kinematic", reference="ref")
           .bridge(ndm=3, ndf=3))
    ops.ndf(1, ndf=6)
    ops.tcl(str(tmp_path / "k.tcl"), flat=True)
    deck = (tmp_path / "k.tcl").read_text(encoding="utf-8").splitlines()
    assert f"node 1 {REF[0]} {REF[1]} {REF[2]} -ndf 6" in deck
    (rbe2,) = [ln for ln in deck if ln.startswith("element LadrunoKinematicCoupling")]
    assert rbe2.split()[3:5] == ["1", "9"]

    # Review F5: RBE3 is ``element LadrunoDistributingCoupling tag R N
    # i1..iN [-w w1..wN]``, independents in sorted order; ``area`` weights
    # are the tributary areas of the 2x2 top face of 5 mm quads: 25/4 at a
    # corner, 2 x 25/4 on an edge, 4 x 25/4 at the centre.
    for weighting in ("uniform", "area"):
        rbe3 = (_stack(files).node("ref", REF)
                .couple("pier_2.top", kind="distributing", reference="ref",
                        weighting=weighting)
                .bridge(ndm=3, ndf=3))
        rbe3.ndf(1, ndf=6)
        rbe3.tcl(str(tmp_path / "d.tcl"), flat=True)
        (line,) = [ln for ln in (tmp_path / "d.tcl").read_text(
            encoding="utf-8").splitlines()
            if ln.startswith("element LadrunoDistributingCoupling")]
        tok = line.split()
        top = _ids(rbe3.fem, pg="pier_2.top")
        assert tok[3:5] == ["1", "9"]
        assert [int(t) for t in tok[5:14]] == top
        if weighting == "uniform":
            assert len(tok) == 14, line
            continue
        assert tok[14] == "-w"
        c = _coords(rbe3.fem)

        def tributary(nid: int) -> float:
            on_edge = sum(abs(v - SIDE / 2) > 1.0 for v in c[nid][:2])
            return {2: 6.25, 1: 12.5, 0: 25.0}[on_edge]
        np.testing.assert_allclose([float(w) for w in tok[15:]],
                                   [tributary(t) for t in top], rtol=1e-12)

    emb = (Assembly("emb").instance("host", files["block"])
           .instance("bar", files["cube"], translate=CUBE_AT)
           .embedded("host.Vol", "bar.Vol").bridge(ndm=3, ndf=3))
    emb.tcl(str(tmp_path / "e.tcl"), flat=True)
    lines = (tmp_path / "e.tcl").read_text(encoding="utf-8").splitlines()
    assert sum(ln.startswith("element ASDEmbeddedNodeElement") for ln in lines) == 8


# ---------------------------------------------------------------------------
# Dotted source names — a port maps through the merge engine's prefix rule
# ---------------------------------------------------------------------------

#: Each verb family on dotted ports, and the records it resolves on the
#: same geometry with plain ports (the oracle: ``block_fem``'s ``top`` /
#: ``bot`` / ``Vol`` are the same node sets as ``deck.top`` / ``deck.bot`` /
#: ``core.all``).
_DOTTED = {
    "tie": (lambda a, p: a.tie(f"pier_1.{p['top']}", f"pier_2.{p['bot']}",
                               enforce="equation", dofs=[1, 2, 3]),
            "elements", "tie", 9),
    "equal_dof": (lambda a, p: a.equal_dof(f"pier_1.{p['top']}",
                                           f"pier_2.{p['bot']}", dofs=[1, 2, 3]),
                  "nodes", "equal_dof", 9),
    "rigid_link": (lambda a, p: a.rigid_link("ref", f"pier_2.{p['top']}",
                                             link_type="rod"),
                   "nodes", "rigid_rod", 9),
    "rigid_diaphragm": (
        lambda a, p: a.rigid_diaphragm("ref", f"pier_2.{p['top']}",
                                       plane_tolerance=6.0),
        "nodes", "rigid_diaphragm", 1),
    "embedded": (lambda a, p: a.embedded(f"pier_1.{p['vol']}", "bar.Vol"),
                 "elements", "embedded", 8),
    "kinematic": (lambda a, p: a.couple(f"pier_2.{p['top']}", kind="kinematic",
                                        reference="ref"),
                  "nodes", "kinematic_coupling", 1),
    "distributing": (lambda a, p: a.couple(f"pier_2.{p['top']}",
                                           kind="distributing", reference="ref"),
                     "elements", "distributing", 1),
}


@pytest.mark.parametrize("family", sorted(_DOTTED))
def test_a_dotted_source_name_resolves_like_a_plain_one(files, family, tmp_path):
    """``"pier_1.deck.top"`` names the group compose stored as
    ``pier_1/deck.top`` (``_prefix_namespaced_name``, ADR 0038)."""
    from apeGmsh.assembly import Assembly

    declare, side, kind, n = _DOTTED[family]
    got = {}
    for src, names in (("block", {"top": "top", "bot": "bot", "vol": "Vol"}),
                       ("dblock", {"top": "deck.top", "bot": "deck.bot",
                                   "vol": "core.all"})):
        asm = (Assembly("d")
               .instance("pier_1", files[src])
               .instance("pier_2", files[src], translate=(0.0, 0.0, H))
               .instance("bar", files["cube"], translate=CUBE_AT)
               .node("ref", REF))
        declare(asm, names)
        ops = asm.bridge(ndm=3, ndf=3)
        fem = ops.fem
        recs = (_node_records if side == "nodes" else _elem_records)(fem, kind)
        got[src] = len(recs)
        if src == "dblock":
            assert "pier_1/deck.top" in fem.nodes.physical
            ops.ndf(1, ndf=6)
            asm.h5(tmp_path / "d.h5")
            back = Assembly.from_h5(tmp_path / "d.h5")
            assert back.ties == asm.ties
            assert back.ties[0].definition == asm.ties[0].definition
    assert got == {"block": n, "dblock": n}


def test_an_assembly_archive_instanced_again_writes_and_reads_back(files, tmp_path):
    """A nested archive: rows of instance ``X`` carry the joined label
    ``X/A``, whose root is ``X``; a port into it, ``"X.A/deck.top"``, maps
    to ``X.A/deck.top`` by the same prefix rule (ADR 0038)."""
    from apeGmsh.assembly import Assembly
    from apeGmsh.assembly._h5 import read_assembly_zone

    # The inner archive carries a reference node but no coupling: an
    # archive whose deck holds a coupling element is refused as an
    # instance source by the rehydrator (MP element rows are not carried).
    inner = Assembly("inner").instance("A", files["dblock"]).node("ref", REF)
    inner.bridge(ndm=3, ndf=3)
    p1 = tmp_path / "inner.h5"
    inner.h5(p1)

    outer = (Assembly("outer").instance("X", p1)
             .instance("Y", p1, translate=(0.0, 0.0, H))
             .equal_dof("X.A/deck.top", "Y.A/deck.bot", dofs=[1, 2, 3],
                        name="glue"))
    ops = outer.bridge(ndm=3, ndf=3)
    fem = ops.fem
    assert "X.A/deck.top" in fem.nodes.physical
    assert len(_node_records(fem, "equal_dof")) == 9
    assert set(fem.nodes.module_label) >= {"X/A", "Y/A"}
    for nid in fem.nodes.decoupled_ids:  # the inner reference nodes
        ops.ndf(int(nid), ndf=6)
    p2 = tmp_path / "outer.h5"
    outer.h5(p2)

    zone = read_assembly_zone(p2)
    rows = {r.label: r for r in zone.instances}
    assert set(rows) == {"X", "Y"}
    for label, r in rows.items():
        owned = [int(i) for i, lbl in zip(fem.nodes.ids, fem.nodes.module_label)
                 if str(lbl).split("/")[0].split(".")[0] == label]
        assert r.fem_id_base <= min(owned)
        assert max(owned) < r.fem_id_base + r.fem_id_span
    back = Assembly.from_h5(p2)
    assert [i.label for i in back.instances] == ["X", "Y"]
    assert back.ties == outer.ties
    assert {t.name: t.n_records for t in zone.ties} == {"glue": 9}


# ---------------------------------------------------------------------------
# INV-7 — zero records and bad ports fail loud
# ---------------------------------------------------------------------------

def _empty_cases(files):
    """(declare, master, slave): a coupling between existing ports that
    resolves zero records."""
    from apeGmsh.assembly import Assembly

    far = (Assembly("far").instance("pier_1", files["block"])
           .instance("pier_2", files["block"], translate=(0.0, 0.0, 5 * H))
           .node("ref", REF))
    return {
        "equal_dof": (lambda: far.equal_dof("pier_1.bot", "pier_2.top"),
                      "pier_1.bot", "pier_2.top"),
        "rigid_link": (lambda: far.rigid_link("ref", "ref"), "ref", "ref"),
        "rigid_diaphragm": (
            lambda: far.rigid_diaphragm("pier_1.top", "pier_2.bot",
                                        master_point=(0.0, 0.0, 100.0),
                                        plane_tolerance=1.0),
            "pier_1.top", "pier_2.bot"),
        "embedded": (lambda: far.embedded("pier_1.Vol", "pier_1.top"),
                     "pier_1.Vol", "pier_1.top"),
    }, far


@pytest.mark.parametrize(
    "kind", ["equal_dof", "rigid_link", "rigid_diaphragm", "embedded"])
def test_a_coupling_resolving_no_record_names_both_ports(files, kind):
    from apeGmsh.assembly import AssemblyError

    cases, asm = _empty_cases(files)
    declare, master, slave = cases[kind]
    declare()
    with pytest.raises(AssemblyError, match="resolved to no record") as info:
        asm.bridge(ndm=3, ndf=3)
    msg = str(info.value)
    assert f"{kind}({master!r}, {slave!r})" in msg


@pytest.mark.parametrize("verb", [
    lambda a: a.equal_dof("pier_1.top", "pier_2.nope"),
    lambda a: a.rigid_link("ref", "pier_2.nope"),
    lambda a: a.rigid_diaphragm("ref", "pier_2.nope"),
    lambda a: a.embedded("pier_1.Vol", "pier_2.nope"),
    lambda a: a.couple("pier_2.nope", kind="kinematic", reference="ref"),
    lambda a: a.couple("pier_2.nope", kind="distributing", reference="ref"),
])
def test_a_missing_group_raises_at_bridge_naming_the_port(files, verb):
    from apeGmsh.assembly import AssemblyError

    asm = _stack(files).node("ref", REF)
    verb(asm)
    with pytest.raises(AssemblyError, match="pier_2.nope"):
        asm.bridge(ndm=3, ndf=3)


@pytest.mark.parametrize("verb, match", [
    (lambda a: a.equal_dof("top", "pier_2.bot"),
     r"names no assembly object.*\['ref'\].*\['pier_1', 'pier_2'\]"),
    (lambda a: a.embedded("ref", "pier_2.bot"),
     r"names no assembly object.*\['pier_1', 'pier_2'\]"),
    (lambda a: a.couple("ref", kind="kinematic", reference="ref"),
     r"names no assembly object"),
    (lambda a: a.couple("pier_2.top", kind="kinematic", reference="pier_1.top"),
     "not a declared reference node"),
    (lambda a: a.couple("pier_2.top", kind="kinematic", reference="nope"),
     "names no assembly object"),
    (lambda a: a.couple("pier_2.top", kind="distributing"), "reference= .* required"),
    (lambda a: a.couple("pier_2.top", kind="distributing", reference="ref",
                        dofs=[1]), "option of kind='kinematic'"),
    (lambda a: a.couple("pier_2.top", kind="kinematic", reference="ref",
                        weighting="area"), "option of kind='distributing'"),
    (lambda a: a.couple("pier_2.top", kind="distributing", reference="ref",
                        weighting="mass"), "weighting='mass'"),
    (lambda a: a.equal_dof("pier_1.top", "pier_2.bot", dofs=[0]), "1..6"),
    (lambda a: a.equal_dof("pier_1.top", "pier_2.bot", tolerance=-1.0), "> 0"),
    (lambda a: a.rigid_link("ref", "pier_2.top", link_type="bar"), "'beam' or 'rod'"),
    (lambda a: a.rigid_diaphragm("ref", "pier_2.top", plane_normal=(0, 0, 0)),
     "is zero"),
    (lambda a: a.embedded("pier_1.Vol", "pier_2.bot", stiffness=-1.0), "> 0"),
])
def test_bad_declarations_raise_and_record_nothing(files, verb, match):
    from apeGmsh.assembly import AssemblyError

    asm = _stack(files).node("ref", REF)
    with pytest.raises(AssemblyError, match=match):
        verb(asm)
    assert asm.ties == ()
    # The refused call left no provenance claim: the same name declares.
    asm.equal_dof("pier_1.top", "pier_2.bot", name="retry")
    assert [t.name for t in asm.ties] == ["retry"]


@pytest.mark.parametrize("case", ["walls_horizontal_plane", "ref_far_above"])
def test_a_diaphragm_whose_only_plane_node_is_its_master_raises(files, case):
    """Review F1: the reference master is always in its own plane, so a
    diaphragm whose slave port has no node there routes one record with
    no slave (``rigidDiaphragm 3 1``). It must raise naming both ports."""
    from apeGmsh.assembly import AssemblyError

    if case == "walls_horizontal_plane":
        # cm sits at z = 11; the walls' nodes at z = 10 and 12 are 1 away.
        asm = _walls(files).rigid_diaphragm(
            "cm", "w1.Slab", plane_normal=(0, 0, 1), constrained_dofs=(1, 2, 6),
            plane_tolerance=0.5)
        ports, ndf = ("cm", "w1.Slab"), 6
    else:
        asm = (_stack(files).node("ref", (0.0, 0.0, 1000.0))
               .rigid_diaphragm("ref", "pier_2.top"))
        ports, ndf = ("ref", "pier_2.top"), 3
    with pytest.raises(AssemblyError, match="resolved to no record") as info:
        asm.bridge(ndm=3, ndf=ndf)
    assert f"rigid_diaphragm({ports[0]!r}, {ports[1]!r})" in str(info.value)


def test_the_v1_couple_refuses_instance_form_options(files):
    """Review F2: v1 forwards unknown keywords to ``g.constraints``; the
    instance-form ``reference=`` / ``weighting=`` must not vanish there."""
    from apeGmsh.assembly import Assembly, AssemblyError

    for kw in ({"reference": "ref"}, {"weighting": "area"}):
        asm = Assembly("v1").add("h", str(files["block"]))
        with pytest.raises(AssemblyError, match="reference= and weighting="):
            asm.couple("h", "h", kind="equal_dof", ports=("top", "bot"), **kw)
        assert asm._couples == []
    asm.couple("h", "h", kind="equal_dof", ports=("top", "bot"))
    assert len(asm._couples) == 1


# ---------------------------------------------------------------------------
# D3 — the unroutable verbs stay out
# ---------------------------------------------------------------------------

def test_contact_interface_embed_and_reinforce_are_not_assembly_verbs(files):
    from apeGmsh.assembly import Assembly, AssemblyError

    for verb in ("contact", "interface", "embed", "reinforce"):
        assert not hasattr(Assembly, verb), verb
    asm = _stack(files).node("ref", REF)
    for kind in ("contact", "interface", "tie", "equal_dof"):
        with pytest.raises(AssemblyError, match="not assembly couplings"):
            asm.couple("pier_1.top", "pier_2.bot", kind=kind, ports=("a", "b"))
    assert asm.ties == ()


# ---------------------------------------------------------------------------
# Reload — every kind persists and round-trips
# ---------------------------------------------------------------------------

def _every_kind(files):
    from apeGmsh.assembly import Assembly

    return (Assembly("all")
            .instance("pier_1", files["block"])
            .instance("pier_2", files["block"], translate=(0.0, 0.0, H))
            .instance("bar", files["cube"], translate=CUBE_AT)
            .node("ref", REF)
            .node("low", (SIDE / 2, SIDE / 2, -5.0))
            .tie("pier_1.top", "pier_2.bot", enforce="equation", dofs=[1, 2, 3],
                 name="t")
            .equal_dof("pier_1.top", "pier_2.bot", dofs=[1, 2], name="eq")
            .rigid_link("low", "pier_1.bot", link_type="rod", name="rl")
            .rigid_diaphragm("ref", "pier_2.top", plane_tolerance=6.0,
                             constrained_dofs=(1, 2), name="rd")
            .embedded("pier_1.Vol", "bar.Vol", tolerance=0.01, name="emb")
            .couple("pier_2.top", kind="kinematic", reference="ref",
                    dofs=[1, 2, 3], name="rbe2")
            .couple("pier_1.bot", kind="distributing", reference="low",
                    weighting="area"))


#: Records each kind of ``_every_kind`` resolves, by closed form.
N_RECORDS = {"ref": 1, "low": 1, "t": 9, "eq": 9, "rl": 9, "rd": 1,
             "emb": 8, "rbe2": 1, "": 1}


def test_every_kind_round_trips_through_the_archive(files, tmp_path):
    from apeGmsh.assembly import Assembly
    from apeGmsh.assembly._h5 import read_assembly_zone

    asm = _every_kind(files)
    ops = asm.bridge(ndm=3, ndf=3)
    ops.ndf(1, ndf=6)
    ops.ndf(2, ndf=6)
    out = tmp_path / "all.h5"
    asm.h5(out)

    zone = read_assembly_zone(out)
    assert [(t.name, t.kind) for t in zone.ties] == [
        ("ref", "node"), ("low", "node"), ("t", "tie"), ("eq", "equal_dof"),
        ("rl", "rigid_link"), ("rd", "rigid_diaphragm"), ("emb", "embedded"),
        ("rbe2", "kinematic_coupling"), ("", "distributing_coupling")]
    assert {t.name: t.n_records for t in zone.ties} == N_RECORDS
    rd = json.loads(next(t.params for t in zone.ties if t.name == "rd"))
    assert rd == {"constrained_dofs": [1, 2], "master_point": list(REF),
                  "plane_normal": [0.0, 0.0, 1.0], "plane_tolerance": 6.0}

    back = Assembly.from_h5(out)
    assert back.nodes == asm.nodes
    assert back.ties == asm.ties
    for a, b in zip(asm.ties, back.ties):
        assert a.definition == b.definition
    # The re-listed couplings resolve as the declared ones did.
    from apeGmsh.mesh import FEMData
    flat = FEMData.from_h5(str(out))
    assert _ids(flat, label="ref") == [1] and _ids(flat, label="low") == [2]


def _tampered(src: Path, dst: Path, edit) -> Path:
    dst.write_bytes(src.read_bytes())
    with h5py.File(str(dst), "r+") as f:
        edit(f)
    return dst


def test_foreign_params_are_refused_on_write_and_on_read(files, tmp_path):
    from apeGmsh.assembly import Assembly, AssemblyError
    from apeGmsh.assembly._assembly import _tie_rows
    from apeGmsh.assembly._h5 import validate_rows

    asm = (_stack(files).node("ref", REF)
           .couple("pier_2.top", kind="kinematic", reference="ref", name="k"))
    ops = asm.bridge(ndm=3, ndf=3)
    ops.ndf(1, ndf=6)
    out = tmp_path / "k.h5"
    asm.h5(out)

    assert asm._bridged is not None
    rows = _tie_rows(asm._bridged)
    bad = [dataclasses.replace(r, params='{"dofs":null,"extra":1}')
           if r.kind == "kinematic_coupling" else r for r in rows]
    with pytest.raises(AssemblyError, match="expected \\['dofs'\\]"):
        validate_rows("stack", _instance_rows_for(asm), bad)

    def edit(f):
        params = f["assembly/ties/params"]
        vals = [p.decode() if isinstance(p, bytes) else p for p in params[()]]
        params[1] = vals[1].replace('"dofs":null', '"dofs":[0]')
    with pytest.raises(AssemblyError, match="1..6"):
        Assembly.from_h5(_tampered(out, tmp_path / "bad.h5", edit))

    # Review F3: a foreign *key* in a coupling row is refused on read.
    def foreign_key(f):
        params = f["assembly/ties/params"]
        vals = [p.decode() if isinstance(p, bytes) else p for p in params[()]]
        assert vals[1] == '{"dofs":null}'
        params[1] = '{"dofs":null,"extra":1}'
    with pytest.raises(AssemblyError, match=r"params carry \['dofs', 'extra'\]"):
        Assembly.from_h5(_tampered(out, tmp_path / "bad3.h5", foreign_key))

    def node_ports(f):
        f["assembly/ties/master"][0] = "pier_1.top"
    with pytest.raises(AssemblyError, match="empty ports"):
        Assembly.from_h5(_tampered(out, tmp_path / "bad2.h5", node_ports))


def _instance_rows_for(asm):
    from apeGmsh.assembly._assembly import _instance_rows

    assert asm._bridged is not None
    return _instance_rows(asm._bridged)


def test_h5_refuses_a_node_declared_after_bridge(files, tmp_path):
    from apeGmsh.assembly import AssemblyError

    asm = _stack(files)
    asm.bridge(ndm=3, ndf=3)
    asm.node("late", REF)
    with pytest.raises(AssemblyError, match="after bridge"):
        asm.h5(tmp_path / "late.h5")
