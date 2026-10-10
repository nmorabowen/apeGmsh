"""Contact resolve guards (program slice B4-c, #1628): #1262 and #1264.

Both live in ``ConstraintsComposite.resolve_contacts`` and both were found
by the piles-validation ladder on a cylindrical pile skin:

* **#1262** — ``ContactDef`` requires one global ``outward`` for a mortar
  tie, and the fork uses it only as a per-facet SIGN reference. On a
  closed master the pairing is silently wrong (one tie measured 3.65x too
  stiff against a 4-sector split). Resolve now refuses a tie whose master
  facet normals span more than 90° about the declared outward and names
  the sector remedy. On main the closed-cylinder case raises nothing.
* **#1264** — ``_collect_node_set`` de-duplicates within one contact only,
  so per-sector NTS contacts on one slave label both claim the seam nodes
  of adjacent slave entities and the fork ADDS their tractions. Resolve
  now keeps every slave node with the first declaration on that label and
  warns :class:`ContactSlaveOverlapWarning`. On main the two halves below
  overlap on 12 seam nodes.

The fixtures are real OCC meshes (no fork build needed: the records are
resolved at ``get_fem_data``). The sector helpers key on the OCC surface
type, which is the only robust way to tell a quarter-cylinder's curved
skin from its two flat cut faces.
"""
from __future__ import annotations

import math
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
                          match=r"'right' shares slave label 'hole' with "
                                r"earlier contact\(s\) 'left' \(\d+ nodes\)"):
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
                                             r"'hole' with earlier contact"
                                             r"\(s\) 'whole'.*would be empty"):
            g.mesh.queries.get_fem_data(dim=3)
        assert not g.constraints.contact_records
        g.constraints.contact_defs.clear()
