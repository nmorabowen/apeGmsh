"""End-to-end emit of ``g.constraints.interface()`` through ``apeSees``
(ADR 0093 S5).

The unit file (``tests/opensees/unit/test_interface_emit.py``) pins the
record → emitter contract line by line. This one drives the whole path —
verb → resolver → ``fem.elements.interfaces`` → ``emit_interfaces`` — on
two real meshes, and covers the two model-level questions a stub cannot
answer:

* an **equal-ndf** deck (quads on both sides) and a **mixed-ndf** deck
  (2-dof continuum master against a 3-dof beam slave), including the
  handler the mixed case must now get — its phantom-bridge ``equalDOF``
  is a real ``MP_Constraint`` that a Plain-style handler would leave
  unenforced under contact;
* the partitioned path, which since ADR 0093 S8 emits each pair's
  atomic unit inside its owner rank's block (the deep gates live in
  ``test_interface_partitioned_emit.py``; here only the lifted refusal
  is pinned);
* **ADR 0093 S7** — the staged liner-install: ``s.interface(name=)``
  moves the whole per-pair unit into the stage that activates the liner,
  on the ground the first stage equilibrated.
"""
from __future__ import annotations

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError

NORMAL = NormalLaw(kind="ent", k_per_area=1.0e9)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)
THICKNESS = 0.5


def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _two_squares(g, n: int = 2):
    """Left square [0,1]^2 (the continuum master) + right square
    [1,2]^2, un-fragmented, so the two curves at x=1 carry coincident
    but distinct node sets — the topology the verb pairs."""
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    g.model.sync()
    g.mesh.structured.set_transfinite([(2, left), (2, right)], n=n)
    g.mesh.generation.generate(2)
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
    g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")


def _fem(*, slave_ndf=None, partition=0, n=2):
    with apeGmsh(model_name="iface_s5", verbose=False) as g:
        _two_squares(g, n=n)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, slave_ndf=slave_ndf, name="RockLiner")
        if partition:
            g.mesh.partitioning.partition(partition)
        # dim=None keeps the dim-1 line cells, so the beam-slave
        # case really has a 3-dof 'wire' to bridge to.
        return g.mesh.queries.get_fem_data()


def _quad_ops(fem, *pgs):
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in pgs:
        ops.element.FourNodeQuad(
            pg=pg, thickness=THICKNESS, material=mat,
            plane_type="PlaneStrain")
    return ops


def _deck(ops, tmp_path, name="deck.tcl"):
    path = tmp_path / name
    ops.tcl(str(path))
    return [ln.strip() for ln in path.read_text().splitlines()]


# ======================================================================
# Equal-ndf: quads on both sides, no phantom
# ======================================================================
def test_equal_ndf_flat_deck_carries_the_interface_block(tmp_path):
    fem = _fem()
    assert len(fem.elements.interfaces) == 2     # n=2 ⇒ 2 nodes on the face
    lines = _deck(_quad_ops(fem, "rock", "liner"), tmp_path)

    zl = [ln for ln in lines if ln.startswith("element zeroLength ")]
    assert len(zl) == 2
    # Two tributary-scaled uniaxials per pair, and NO phantom node lines.
    assert len([ln for ln in lines if ln.startswith("uniaxialMaterial ENT")]) == 2
    assert len(
        [ln for ln in lines if ln.startswith("uniaxialMaterial ElasticPP ")]
    ) == 2
    assert not [ln for ln in lines if ln.startswith("equalDOF ")]

    # INV-1 read straight off the deck: iNode is a master ("face") node,
    # jNode is a slave ("wire") node, and local-x is +x (the left
    # square's outward normal on its right edge).
    masters = {int(r.master_node) for r in fem.elements.interfaces}
    slaves = {int(r.slave_node) for r in fem.elements.interfaces}
    for ln in zl:
        tok = ln.split()
        assert int(tok[3]) in masters and int(tok[4]) in slaves
        i = tok.index("-orient")
        assert [float(v) for v in tok[i + 1:i + 7]] == [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    # The interface block lands AFTER the mesh elements (flat order), so
    # its element tags continue the shared namespace.
    mesh_tags = [int(ln.split()[2]) for ln in lines
                 if ln.startswith("element quad ")]
    assert min(int(ln.split()[2]) for ln in zl) > max(mesh_tags)


def test_equal_ndf_deck_is_reproducible(tmp_path):
    fem = _fem()
    first = _deck(_quad_ops(fem, "rock", "liner"), tmp_path, "a.tcl")
    second = _deck(_quad_ops(fem, "rock", "liner"), tmp_path, "b.tcl")
    assert first == second


def test_equal_ndf_needs_no_constraint_handler(tmp_path):
    # No MP constraint anywhere ⇒ the auto-emit stays silent (interfaces
    # are handler-independent, ADR 0093 D5).
    lines = _deck(_quad_ops(_fem(), "rock", "liner"), tmp_path)
    assert not [ln for ln in lines if ln.startswith("constraints ")]


# ======================================================================
# Mixed ndf: 2-dof continuum master vs 3-dof beam slave
# ======================================================================
def _mixed_ops(fem):
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeQuad(
        pg="rock", thickness=THICKNESS, material=mat,
        plane_type="PlaneStrain")
    ops.element.elasticBeamColumn(
        pg="wire", transf=ops.geomTransf.Linear(),
        A=0.02, E=200e9, Iz=1.0e-4)
    return ops


def test_mixed_ndf_flat_deck_phantom_equaldof_materials_zerolength(tmp_path):
    fem = _fem(slave_ndf=3)
    recs = fem.elements.interfaces
    assert len(recs) == 2 and all(r.phantom_node is not None for r in recs)
    lines = _deck(_mixed_ops(fem), tmp_path)

    # The phantom node is declared with the record's OWN ndf (2), not the
    # shared helper's hardcoded 6.
    phantom_lines = [
        ln for ln in lines
        if ln.startswith("node ") and ln.endswith("-ndf 2")
    ]
    phantoms = {int(r.phantom_node) for r in recs}
    assert {int(ln.split()[1]) for ln in phantom_lines} == phantoms

    eq = [ln for ln in lines if ln.startswith("equalDOF ")]
    assert len(eq) == 2
    for ln in eq:
        _, retained, constrained, d1, d2 = ln.split()
        assert int(constrained) in phantoms          # phantom is constrained
        assert int(retained) not in phantoms         # the real beam node
        assert (d1, d2) == ("1", "2")

    zl = [ln for ln in lines if ln.startswith("element zeroLength ")]
    assert len(zl) == 2
    for ln in zl:
        tok = ln.split()
        # INV-1: iNode is the real continuum master, jNode the phantom.
        assert int(tok[3]) not in phantoms
        assert int(tok[4]) in phantoms

    # …and the per-pair block is ordered phantom → equalDOF → mats → element.
    for rec in recs:
        i_node = lines.index(f"node {int(rec.phantom_node)} "
                             f"{float(rec.phantom_coords[0])!r} "
                             f"{float(rec.phantom_coords[1])!r} "
                             f"{float(rec.phantom_coords[2])!r} -ndf 2")
        assert lines[i_node - 1] == "# RockLiner"
        assert lines[i_node + 1].startswith("equalDOF ")
        assert lines[i_node + 2].startswith("uniaxialMaterial ENT ")
        assert lines[i_node + 3].startswith("uniaxialMaterial ElasticPP ")
        assert lines[i_node + 4].startswith("element zeroLength ")


def test_mixed_ndf_gets_a_constraint_handler(tmp_path):
    # The phantom bridge IS an ordinary identity equalDOF, so the deck is
    # handler-dependent in the ordinary way — the auto-emit must see it
    # even though it lives on fem.elements.interfaces, not
    # fem.nodes.constraints.
    with pytest.warns(UserWarning):
        lines = _deck(_mixed_ops(_fem(slave_ndf=3)), tmp_path)
    assert "constraints Transformation" in lines


def test_declared_slave_ndf_must_match_the_real_slave(tmp_path):
    # slave_ndf omitted while the slave really is a 3-dof beam node: the
    # deck would carry a mixed-ndf zeroLength the engine refuses.
    ops = _mixed_ops(_fem())
    with pytest.raises(BridgeError, match=r"slave_ndf=3"):
        ops.tcl(str(tmp_path / "bad.tcl"))


def test_declared_phantom_against_a_two_dof_slave_fails_loud(tmp_path):
    # The mirror: slave_ndf=3 declared, but the slave carries quads.
    ops = _quad_ops(_fem(slave_ndf=3), "rock", "liner")
    with pytest.raises(BridgeError, match="phantom bridge"):
        ops.tcl(str(tmp_path / "bad.tcl"))


# ======================================================================
# Partitioned emit (ADR 0093 S8 — the S5 refusal is LIFTED)
# ======================================================================
def test_partitioned_emit_no_longer_refuses_interfaces(tmp_path):
    # S5's blanket refusal is gone: a partitioned interface model emits,
    # and every zeroLength lands in the deck exactly once. The full S8
    # gates (single owner block, ghost ndf parity, 1-vs-N byte
    # identity) live in test_interface_partitioned_emit.py.
    fem = _fem(partition=2, n=4)
    assert len(fem.partitions) == 2
    assert fem.elements.interfaces
    ops = _quad_ops(fem, "rock", "liner")
    path = tmp_path / "deck.tcl"
    ops.tcl(str(path))
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    assert len([ln for ln in lines
                if ln.startswith("element zeroLength ")]) == len(
        fem.elements.interfaces)


# ======================================================================
# Staged liner install (ADR 0093 S7)
# ======================================================================
def _chain(ops):
    return {
        "test":        ops.test.NormDispIncr(tol=1e-6, max_iter=25),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=1.0),
        "constraints": ops.constraints.Transformation(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _staged_ops(fem):
    """Stage 1 loads the bare ground; stage 2 activates the liner PG and
    claims the interface, so the liner is installed on the equilibrated
    ground and carries only post-install increments."""
    ops = _quad_ops(fem, "rock", "liner")
    with ops.stage(name="ground") as s:
        s.analysis(**_chain(ops))
        s.run(n_increments=1, dt=1.0)
    with ops.stage(name="install") as s:
        s.activate(pgs=["liner"])
        s.interface(name="RockLiner")
        s.analysis(**_chain(ops))
        s.run(n_increments=1, dt=1.0)
    return ops


def test_staged_claim_moves_the_whole_unit_into_the_stage(tmp_path):
    fem = _fem()
    lines = _deck(_staged_ops(fem), tmp_path)

    zl = [ln for ln in lines if ln.startswith("element zeroLength ")]
    assert len(zl) == 2                       # once per pair, not twice

    open_ground = lines.index("# === Stage: ground ===")
    open_install = lines.index("# === Stage: install ===")
    first_zl = min(lines.index(ln) for ln in zl)
    # Nothing interface-shaped before the stages; everything inside the
    # stage that activates the liner.
    assert first_zl > open_install > open_ground
    assert not [ln for ln in lines[:open_ground]
                if ln.startswith("uniaxialMaterial ENT ")]
    # …and each zeroLength lands after the liner's own quads and before
    # this stage's domainChange barrier.
    install = lines[open_install:]
    last_quad = max(
        i for i, ln in enumerate(install) if ln.startswith("element quad ")
    )
    barrier = install.index("domainChange")
    for ln in zl:
        assert last_quad < install.index(ln) < barrier


def test_staged_interface_tags_continue_the_base_allocator(tmp_path):
    fem = _fem()
    lines = _deck(_staged_ops(fem), tmp_path)
    ele_tags = [int(ln.split()[2]) for ln in lines if ln.startswith("element ")]
    assert len(ele_tags) == len(set(ele_tags))
    zl_tags = [int(ln.split()[2]) for ln in lines
               if ln.startswith("element zeroLength ")]
    mesh_tags = [int(ln.split()[2]) for ln in lines
                 if ln.startswith("element quad ")]
    assert min(zl_tags) > max(mesh_tags)


def test_staged_deck_is_reproducible(tmp_path):
    first = _deck(_staged_ops(_fem()), tmp_path, "a.tcl")
    second = _deck(_staged_ops(_fem()), tmp_path, "b.tcl")
    assert first == second


def test_flat_escape_hatch_emits_a_partitioned_interface_model(tmp_path):
    # ``flat=True`` is the sanctioned serial escape hatch for the
    # serial-only side lists (see test_contact_partition_fail_loud).
    fem = _fem(partition=2, n=4)
    path = tmp_path / "flat.tcl"
    _quad_ops(fem, "rock", "liner").tcl(str(path), flat=True)
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    assert len([ln for ln in lines
                if ln.startswith("element zeroLength ")]) == 4
    assert "getPID" not in "\n".join(lines)
