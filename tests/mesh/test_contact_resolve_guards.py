"""Contact resolve guards (program slice B4-c, #1628): #1262 and #1264.

Both live in ``ConstraintsComposite.resolve_contacts`` and both were found
by the piles-validation ladder on a cylindrical pile skin:

* **#1262** — ``ContactDef`` requires one global ``outward`` for a mortar
  tie, and the fork uses it only as a per-facet SIGN reference. On a
  closed master the pairing is silently wrong (one tie measured 3.65x too
  stiff against a 4-sector split). Resolve now measures the span of the
  master's facet normals, each oriented outward from its volume, and
  refuses a tie when some pair is more than 90° apart, naming the sector
  remedy. The span is the master's alone (the outward's sign does not
  enter), so a slab's top and bottom under one label refuse while a 60°
  sector with an edge-aligned outward passes. On main the closed-cylinder
  case raises nothing.
* **#1264** — ``_collect_node_set`` de-duplicates within one contact only,
  so per-sector NTS contacts on one slave label both claim the seam nodes
  of adjacent slave entities and the fork ADDS their tractions. Resolve
  now keeps every slave node with the first declaration among contacts
  that share the label AND whose masters touch (share a node: sector
  neighbours of one body), and warns :class:`ContactSlaveOverlapWarning`.
  Two piles against one soil label share no master node and keep their
  full sets. On main the two halves below overlap on 12 seam nodes.

The fixtures are real OCC meshes (no fork build needed: the records are
resolved at ``get_fem_data``). The sector helpers key on the OCC surface
type, which is the only robust way to tell a quarter-cylinder's curved
skin from its two flat cut faces.
"""
from __future__ import annotations

import math
import re
import warnings

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.core.ConstraintsComposite import ContactSlaveOverlapWarning

R, H = 0.5, 2.0


def _curved_faces(vol: int) -> list[int]:
    return [abs(t) for d, t in gmsh.model.getBoundary([(3, vol)], oriented=False)
            if d == 2 and gmsh.model.getType(2, abs(t)) == "Cylinder"]


def _radial(surface: int) -> tuple[float, float, float]:
    com = gmsh.model.occ.getCenterOfMass(2, surface)
    n = math.hypot(com[0], com[1])
    return (com[0] / n, com[1] / n, 0.0)


def _pile_in_soil(g, *, sectors: int):
    """A pile skin (``skin``: one closed lateral surface, or ``sectors``
    quarter-cylinder surfaces) inside a soil box with a coincident hole
    (``hole``), meshed independently so the two surfaces carry distinct
    node sets."""
    if sectors == 1:
        piles = [g.model.geometry.add_cylinder(0, 0, 0, 0, 0, H, R)]
    else:
        piles = []
        for k in range(sectors):
            t = g.model.geometry.add_cylinder(
                0, 0, 0, 0, 0, H, R, angle=2 * math.pi / sectors)
            gmsh.model.occ.rotate([(3, t)], 0, 0, 0, 0, 0, 1,
                                  k * 2 * math.pi / sectors)
            piles.append(t)
        g.model.sync()
    soil = g.model.geometry.add_box(-2, -2, 0, 4, 4, H)
    tool = g.model.geometry.add_cylinder(0, 0, 0, 0, 0, H, R)
    soil = g.model.boolean.cut(soil, tool)[0]
    g.model.sync()
    skins = [s for p in piles for s in _curved_faces(p)]
    holes = _curved_faces(soil)
    assert len(skins) == sectors and holes
    g.mesh.sizing.set_global_size(0.4)
    g.mesh.generation.generate(3)
    g.physical.add(3, piles + [soil], name="solid")
    g.physical.add(2, skins, name="skin")
    g.physical.add(2, holes, name="hole")
    return skins, holes


# --------------------------------------------------------------------------
# #1262 — a tie on a non-flat master refuses with the sector remedy
# --------------------------------------------------------------------------

def test_tie_on_closed_cylinder_master_refuses_naming_sectors(tmp_path):
    with apeGmsh(model_name="b4_cyl", verbose=False,
                 save_to=tmp_path / "m.h5") as g:
        _pile_in_soil(g, sectors=1)
        g.constraints.contact("skin", "hole", formulation="mortar",
                              tie=True, outward=(1.0, 0.0, 0.0),
                              name="tie_all")
        before = list(g.constraints.contact_defs)
        with pytest.raises(ValueError) as exc:
            g.mesh.queries.get_fem_data(dim=3)
        msg = str(exc.value)
        assert "'tie_all'" in msg
        assert "span more than 90°" in msg
        # The message states what was measured: the widest pair of a
        # closed skin's outward normals is antipodal.
        assert re.search(r"widest pair of its \d+ facet normals .* is "
                         r"1[78]\d\.\d° apart", msg), msg
        assert "master_entities=/slave_entities=" in msg
        assert "radial outward=" in msg
        # Seam A: a resolve-time raise leaves the session's declarations
        # untouched and appends no half-resolved record.
        assert g.constraints.contact_defs == before
        assert not g.constraints.contact_records
        g.constraints.contact_defs.clear()   # keep the exit autosave quiet


def test_tie_on_four_radial_sectors_resolves(tmp_path):
    """The remedy the refusal names: one tie per 90° sector with a radial
    outward. Each sector's normals span exactly 90° about it, so the
    bound is inclusive. No warning may fire (run with
    ``-W error::UserWarning``)."""
    with apeGmsh(model_name="b4_sec", verbose=False,
                 save_to=tmp_path / "m.h5") as g:
        skins, _ = _pile_in_soil(g, sectors=4)
        for s in skins:
            g.constraints.contact("skin", "hole", formulation="mortar",
                                  tie=True, outward=_radial(s),
                                  master_entities=[(2, s)], name=f"sec{s}")
        fem = g.mesh.queries.get_fem_data(dim=3)
        recs = g.constraints.contact_records
    assert [r.name for r in recs] == [f"sec{s}" for s in skins]
    assert all(r.tie and r.master_faces.shape[0] > 0 for r in recs)
    assert len(fem.elements.contacts) == 4


def test_tie_on_flat_master_is_unchanged(tmp_path):
    """The flat two-box tie of ``test_contact_h5_roundtrip`` still resolves
    with its single global outward, on either sign of it (gmsh facet
    winding is not load-bearing: the fork sign-fixes, and so does the
    check)."""
    for sign in (1.0, -1.0):
        with apeGmsh(model_name="b4_flat", verbose=False,
                     save_to=tmp_path / "m.h5") as g:
            box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            box2 = g.model.geometry.add_box(0, 0, 1, 1, 1, 1)
            g.model.sync()
            top = [abs(t) for d, t in gmsh.model.getBoundary(
                [(3, box1)], oriented=False)
                if d == 2 and gmsh.model.occ.getCenterOfMass(2, abs(t))[2] > 0.99]
            bot = [abs(t) for d, t in gmsh.model.getBoundary(
                [(3, box2)], oriented=False)
                if d == 2 and gmsh.model.occ.getCenterOfMass(2, abs(t))[2] < 1.01]
            g.mesh.sizing.set_global_size(0.5)
            g.mesh.generation.generate(3)
            g.physical.add(3, [box1, box2], name="solid")
            g.physical.add(2, top, name="master")
            g.physical.add(2, bot, name="slave")
            g.constraints.contact("master", "slave", formulation="mortar",
                                  tie=True, outward=(0.0, 0.0, sign))
            fem = g.mesh.queries.get_fem_data(dim=3)
        assert len(fem.elements.contacts) == 1


def test_tie_on_sixty_degree_sector_with_edge_aligned_outward_resolves(tmp_path):
    """The span is a property of the master alone. A 60° sector spans 60°
    whatever the outward: with it aligned to one sector EDGE (the far
    edge's normals sit 60° off it) the tie still resolves, and no warning
    fires (run with ``-W error::UserWarning``)."""
    with apeGmsh(model_name="b4_sec60", verbose=False,
                 save_to=tmp_path / "m.h5") as g:
        pile = g.model.geometry.add_cylinder(0, 0, 0, 0, 0, H, R,
                                             angle=math.pi / 3)
        soil = g.model.geometry.add_box(-2, -2, 0, 4, 4, H)
        soil = g.model.boolean.cut(
            soil, g.model.geometry.add_cylinder(0, 0, 0, 0, 0, H, R))[0]
        g.model.sync()
        skins, holes = _curved_faces(pile), _curved_faces(soil)
        assert len(skins) == 1 and holes
        g.mesh.sizing.set_global_size(0.2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [pile, soil], name="solid")
        g.physical.add(2, skins, name="skin")
        g.physical.add(2, holes, name="hole")
        # The sector sweeps from the +x axis: (1, 0, 0) is its first edge.
        g.constraints.contact("skin", "hole", formulation="mortar",
                              tie=True, outward=(1.0, 0.0, 0.0))
        fem = g.mesh.queries.get_fem_data(dim=3)
    assert len(fem.elements.contacts) == 1


def test_tie_on_slab_top_and_bottom_refuses(tmp_path):
    """Opposed faces under one label: a slab's top and bottom are 180°
    apart once each normal is oriented outward from the slab, so a tie
    with ``outward=(0, 0, 1)`` is refused — an unsigned check would pass
    it."""
    with apeGmsh(model_name="b4_slab", verbose=False,
                 save_to=tmp_path / "m.h5") as g:
        slab = g.model.geometry.add_box(0, 0, 1, 1, 1, 0.2)
        below = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        faces = [(abs(t), gmsh.model.occ.getCenterOfMass(2, abs(t))[2])
                 for d, t in gmsh.model.getBoundary([(3, slab)],
                                                    oriented=False)
                 if d == 2]
        top_bottom = [t for t, z in faces if abs(z - 1.0) < 1e-6
                      or abs(z - 1.2) < 1e-6]
        assert len(top_bottom) == 2
        under = [abs(t) for d, t in gmsh.model.getBoundary(
            [(3, below)], oriented=False)
            if d == 2 and gmsh.model.occ.getCenterOfMass(2, abs(t))[2] > 0.99]
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [slab, below], name="solid")
        g.physical.add(2, top_bottom, name="faces")
        g.physical.add(2, under, name="under")
        g.constraints.contact("faces", "under", formulation="mortar",
                              tie=True, outward=(0.0, 0.0, 1.0),
                              name="two_sided")
        with pytest.raises(ValueError, match=r"'two_sided'.*span more than "
                                             r"90°.*widest pair.*180\.0° apart"):
            g.mesh.queries.get_fem_data(dim=3)
        g.constraints.contact_defs.clear()


def test_non_tie_contact_on_closed_master_is_unchanged(tmp_path):
    """The guard is a TIE guard: a plain NTS contact on the closed skin
    (no outward, the fork's per-facet normal) resolves as before."""
    with apeGmsh(model_name="b4_nts", verbose=False,
                 save_to=tmp_path / "m.h5") as g:
        _pile_in_soil(g, sectors=1)
        g.constraints.contact("skin", "hole", formulation="nts",
                              kn=1e6, kt=1e6, mu=0.3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    assert len(fem.elements.contacts) == 1


# --------------------------------------------------------------------------
# #1264 — per-sector NTS contacts on one slave label get disjoint slave sets
# --------------------------------------------------------------------------

def _split_hole(g):
    """A pile skin and a soil hole split at x=0 into two conformal halves
    (``left`` / ``right`` surfaces under the one label ``hole``), so the
    two vertical seam lines at (0, +-R) belong to both halves."""
    pile = g.model.geometry.add_cylinder(0, 0, 0, 0, 0, H, R)
    a = g.model.geometry.add_box(-2, -2, 0, 2, 4, H)
    b = g.model.geometry.add_box(0, -2, 0, 2, 4, H)
    a = g.model.boolean.cut(a, g.model.geometry.add_cylinder(0, 0, 0, 0, 0, H, R))[0]
    b = g.model.boolean.cut(b, g.model.geometry.add_cylinder(0, 0, 0, 0, 0, H, R))[0]
    vols = g.model.boolean.fragment([a], [b])
    g.model.sync()
    holes = [h for v in vols for h in _curved_faces(v)]
    left = [h for h in holes if gmsh.model.occ.getCenterOfMass(2, h)[0] < 0]
    right = [h for h in holes if h not in left]
    assert left and right
    g.mesh.sizing.set_global_size(0.4)
    g.mesh.generation.generate(3)
    g.physical.add(3, [pile] + list(vols), name="solid")
    g.physical.add(2, _curved_faces(pile), name="skin")
    g.physical.add(2, holes, name="hole")
    label_nodes: set[int] = set()
    for h in holes:
        nt, _, _ = gmsh.model.mesh.getNodes(2, h, includeBoundary=True)
        label_nodes |= {int(t) for t in nt}
    return left, right, label_nodes


def test_sector_contacts_on_one_slave_label_are_disjoint(tmp_path):
    with apeGmsh(model_name="b4_split", verbose=False,
                 save_to=tmp_path / "m.h5") as g:
        left, right, label_nodes = _split_hole(g)
        for nm, ents in (("left", left), ("right", right)):
            g.constraints.contact("skin", "hole", formulation="nts",
                                  kn=1e6, kt=1e6, mu=0.3,
                                  slave_entities=[(2, h) for h in ents],
                                  name=nm)
        with pytest.warns(ContactSlaveOverlapWarning,
                          match=r"'right' shares slave label 'hole' and a "
                                r"touching master with earlier contact\(s\) "
                                r"'left' \(\d+ nodes\)"):
            g.mesh.queries.get_fem_data(dim=3)
        first, second = g.constraints.contact_records
        a, b = set(first.slave_nodes), set(second.slave_nodes)
        # The seam nodes stay with the FIRST declaration.
        assert (first.name, second.name) == ("left", "right")
        assert a & b == set()
        assert a | b == label_nodes
        assert len(a) > len(b)
        # Reload (B1 D2): a broker mutation rebuilds the FEM cache, and the
        # rebuild resolves the same split in the same order.
        g.constraints.bc(pg="solid", dofs=[1, 1, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ContactSlaveOverlapWarning)
            g.mesh.queries.get_fem_data(dim=3)
        again = g.constraints.contact_records
        assert [list(r.slave_nodes) for r in again] == \
            [list(first.slave_nodes), list(second.slave_nodes)]
        g.constraints.contact_defs.clear()   # keep the exit autosave quiet


def test_contacts_on_different_slave_labels_do_not_interact(tmp_path):
    """Two labels over the same halves: nothing is dropped, nothing warns
    (run with ``-W error::UserWarning``)."""
    with apeGmsh(model_name="b4_two_labels", verbose=False,
                 save_to=tmp_path / "m.h5") as g:
        left, right, label_nodes = _split_hole(g)
        g.physical.add(2, left, name="hole_left")
        g.physical.add(2, right, name="hole_right")
        g.constraints.contact("skin", "hole_left", formulation="nts",
                              kn=1e6, kt=1e6, mu=0.3, name="left")
        g.constraints.contact("skin", "hole_right", formulation="nts",
                              kn=1e6, kt=1e6, mu=0.3, name="right")
        g.mesh.queries.get_fem_data(dim=3)
        a, b = (set(r.slave_nodes) for r in g.constraints.contact_records)
    assert a | b == label_nodes
    assert a & b                      # the seam is in both, by design


def test_two_piles_against_one_soil_label_keep_full_slave_sets(tmp_path):
    """Two piles (``skin1``, ``skin2``) against one slave label ``holes``:
    the masters share no node, so these are separate bodies, not sector
    neighbours — each contact keeps the whole label, as on base, and
    nothing warns (run with ``-W error::UserWarning``)."""
    with apeGmsh(model_name="b4_two_piles", verbose=False,
                 save_to=tmp_path / "m.h5") as g:
        p1 = g.model.geometry.add_cylinder(-1, 0, 0, 0, 0, H, R)
        p2 = g.model.geometry.add_cylinder(1, 0, 0, 0, 0, H, R)
        soil = g.model.geometry.add_box(-3, -2, 0, 6, 4, H)
        soil = g.model.boolean.cut(
            soil, [g.model.geometry.add_cylinder(-1, 0, 0, 0, 0, H, R),
                   g.model.geometry.add_cylinder(1, 0, 0, 0, 0, H, R)])[0]
        g.model.sync()
        holes = _curved_faces(soil)
        assert len(holes) == 2
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(3)
        g.physical.add(3, [p1, p2, soil], name="solid")
        g.physical.add(2, _curved_faces(p1), name="skin1")
        g.physical.add(2, _curved_faces(p2), name="skin2")
        g.physical.add(2, holes, name="holes")
        label_nodes: set[int] = set()
        for h in holes:
            nt, _, _ = gmsh.model.mesh.getNodes(2, h, includeBoundary=True)
            label_nodes |= {int(t) for t in nt}
        g.constraints.contact("skin1", "holes", formulation="nts",
                              kn=1e6, kt=1e6, mu=0.3)
        g.constraints.contact("skin2", "holes", formulation="nts",
                              kn=1e6, kt=1e6, mu=0.3)
        g.mesh.queries.get_fem_data(dim=3)
        sets = [set(r.slave_nodes) for r in g.constraints.contact_records]
    assert sets == [label_nodes, label_nodes]


def test_fully_claimed_slave_set_refuses_naming_both(tmp_path):
    """A later contact whose every slave node an earlier one on the same
    label already holds would be an empty, silent no-op: refuse it."""
    with apeGmsh(model_name="b4_empty", verbose=False,
                 save_to=tmp_path / "m.h5") as g:
        _split_hole(g)
        g.constraints.contact("skin", "hole", formulation="nts",
                              kn=1e6, kt=1e6, mu=0.3, name="whole")
        g.constraints.contact("skin", "hole", formulation="nts",
                              kn=1e6, kt=1e6, mu=0.3, name="again")
        with pytest.raises(ValueError, match=r"'again' shares slave label "
                                             r"'hole' and a touching master "
                                             r"with earlier contact\(s\) "
                                             r"'whole'.*would be empty"):
            g.mesh.queries.get_fem_data(dim=3)
        assert not g.constraints.contact_records
        g.constraints.contact_defs.clear()
