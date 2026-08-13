"""ADR 0093 S9 — stage-claimed ``g.constraints.interface()`` under
partitioned (MPI) emit: the campaign's actual scenario.

The Cerro Lindo shape: the ground stage equilibrates the partitioned
rock, the install stage activates the liner PG and claims the interface
by name, service stages apply increments. S7 gave the flat staged claim,
S8 the partitioned base pass; S9 composes them — the claimed atomic
unit (mixed-ndf phantom + nested equalDOF + two tributary-scaled
uniaxials + one zeroLength) emits inside the OWNER rank's bracket of
the CLAIMING stage's block, exactly once, with the S8 ownership rules
(INV-5: owner = the rank holding the stamped backing continuum
element) applied unchanged inside the stage.

What is pinned here, on real gmsh-meshed two-body models:

* **owner-rank stage block, exactly once** — every claimed unit inside
  the claiming stage's block, inside the owner's ``getPID`` bracket,
  once across all ranks and stages; nothing interface-shaped in the
  base pass or earlier stages (the liner is NOT installed at t = 0);
* **serial-staged ↔ partitioned-staged comparability** — the claimed
  unit's lines (tags included) match the flat staged deck's
  byte-for-byte modulo ghost declarations, because the claimed rows
  join the S8 tag pre-pass in the sequence the serial-staged deck
  consumes the allocator (unclaimed flat order, then per-stage claim
  order) — under the same S8-documented conditional (no coexisting
  element-minting MP pass);
* **ghost + stage-fix replay, both directions (ADR 0027 INV-2)** — a
  foreign slave ghost declared in the claiming stage's owner bracket
  replays the owner's SP stream up to AND INCLUDING that stage
  (stage-bound ``s.fix`` on the slave included), and a LATER stage's
  ``s.fix`` on the ghosted slave is mirrored into the holder's
  bracket (the forward half);
* **a claim-only stage still opens the owner bracket** — the unified
  content gate counts the claimed plan, so a stage whose ONLY content
  is ``s.interface(name=)`` cannot silently drop the unit (the
  measured s.equal_dof-alone failure mode), and the ghost declared
  there replays the model-level ``fix`` tier;
* **the pattern-sp-on-ghost refusal still fires for stage patterns** —
  a stage pattern prescribing displacement on a slave the claimed
  plan will ghost refuses at plan time (ADR 0027 INV-2);
* **mixed ndf inside the stage context** — the phantom declares
  ``-ndf 2`` (the record's own), the foreign beam slave ghosts with
  its HOME ``-ndf 3``, the nested equalDOF stays owner-only;
* **the later-stage-node claim refusal** — claiming an interface into
  a stage EARLIER than the one that activates its slave topology
  refuses at plan time (under MP that deck would run with the
  interface pushing on a floating ghost — the ADR 0092
  plausible-wrong-answer class);
* **determinism** — two partitioned staged emits are byte-identical.

Emit-time only — no OpenSees run needed. The measured (runtime)
guarantee is ``tests/opensees/subprocess/
test_interface_partitioned_numeric_twin.py``.
"""
from __future__ import annotations

import re

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError

NORMAL = NormalLaw(kind="ent", k_per_area=1.0e9)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)
THICKNESS = 0.5

_RANK_OPEN_RE = re.compile(r"^\s*if \{\[getPID\] == (\d+)\} \{\s*$")


# ---------------------------------------------------------------------
# Fixtures — the ADR 0093 two-squares model with a controlled cut
# ---------------------------------------------------------------------


def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _two_squares(g, n: int):
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    g.model.sync()
    g.mesh.structured.set_transfinite([(2, left), (2, right)], n=n)
    g.mesh.generation.generate(2)
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
    g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")


def _surface_of(dim: int, tag: int) -> int:
    while dim < 2:
        up, _down = gmsh.model.getAdjacencies(dim, tag)
        dim, tag = dim + 1, int(up[0])
    return tag


def _partition_left_right(g) -> None:
    """The cut hugs the interface from the side: every left-body element
    (boundary entities included) goes to partition 1, the right body to
    partition 2 — so every slave (wire) node is FOREIGN to the owner
    rank and the S9 ghosting engages for real."""
    elem_tags: "list[int]" = []
    parts: "list[int]" = []
    for dim, tag in gmsh.model.getEntities():
        if dim > 2:
            continue
        surf = _surface_of(dim, tag)
        part = 1 if gmsh.model.occ.getCenterOfMass(2, surf)[0] < 1.0 else 2
        _etypes, etags_list, _enodes = gmsh.model.mesh.getElements(
            dim=dim, tag=tag)
        for etags in etags_list:
            for t in etags:
                elem_tags.append(int(t))
                parts.append(part)
    g.mesh.partitioning.partition_explicit(2, elem_tags, parts)


def _fem(*, slave_ndf=None, n=4):
    with apeGmsh(model_name="iface_s9", verbose=False) as g:
        _two_squares(g, n=n)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, slave_ndf=slave_ndf, name="RockLiner")
        _partition_left_right(g)
        return g.mesh.queries.get_fem_data()


def _pchain(ops, handler: str = "Transformation"):
    """A complete parallel-friendly per-stage chain (each stage must
    declare its own under staging)."""
    return {
        "test":        ops.test.NormDispIncr(tol=1e-6, max_iter=25),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=1.0),
        "constraints": getattr(ops.constraints, handler)(),
        "numberer":    ops.numberer.ParallelPlain(),
        "system":      ops.system.Mumps(),
        "analysis":    ops.analysis.Static(),
    }


def _quad_ops(fem):
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in ("rock", "liner"):
        ops.element.FourNodeQuad(
            pg=pg, thickness=THICKNESS, material=mat,
            plane_type="PlaneStrain")
    return ops


def _mixed_ops(fem):
    """2-dof continuum master vs 3-dof beam slave — the campaign's
    primary case — under an ndf=2 envelope."""
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


def _staged(ops, *, activate=("liner",), claim="RockLiner",
            install_fix=None, service=False, service_fix=None):
    """ground → install (activate + claim) [→ service]."""
    with ops.stage(name="ground") as s:
        s.analysis(**_pchain(ops))
        s.run(n_increments=1, dt=1.0)
    with ops.stage(name="install") as s:
        if activate:
            s.activate(pgs=list(activate))
        if claim is not None:
            s.interface(name=claim)
        if install_fix is not None:
            s.fix(pg="wire", dofs=install_fix)
        s.analysis(**_pchain(ops))
        s.run(n_increments=1, dt=1.0)
    if service or service_fix is not None:
        with ops.stage(name="service") as s:
            if service_fix is not None:
                s.fix(pg="wire", dofs=service_fix)
            s.analysis(**_pchain(ops))
            s.run(n_increments=1, dt=1.0)
    return ops


def _deck_text(ops, tmp_path, name="deck.tcl", **kw) -> str:
    path = tmp_path / name
    ops.tcl(str(path), **kw)
    return path.read_text(encoding="utf-8")


def _lines(text: str) -> "list[str]":
    return [ln.strip() for ln in text.splitlines()]


def _stage_block(lines: "list[str]", name: str) -> "list[str]":
    """The lines between this stage's banner and its close."""
    start = lines.index(f"# === Stage: {name} ===")
    rest = lines[start + 1:]
    for i, ln in enumerate(rest):
        if ln.startswith("# === Stage: ") or ln.startswith("loadConst "):
            return rest[:i]
    return rest


def _rank_brackets(lines: "list[str]") -> "dict[int, list[str]]":
    """``{rank: stripped lines}`` across every ``getPID`` bracket in
    ``lines`` (a stage block may hold several per rank)."""
    out: "dict[int, list[str]]" = {}
    rank: "int | None" = None
    depth = 0
    for raw in lines:
        if rank is None:
            m = _RANK_OPEN_RE.match(raw)
            if m:
                rank = int(m.group(1))
                depth = 1
                out.setdefault(rank, [])
            continue
        depth += raw.count("{") - raw.count("}")
        if depth <= 0:
            rank = None
            continue
        if raw:
            out[rank].append(raw)
    return out


def _zl(lines: "list[str]") -> "list[str]":
    return [ln for ln in lines if ln.startswith("element zeroLength ")]


def _iface_portion(lines: "list[str]") -> "list[str]":
    starts = [i for i, ln in enumerate(lines) if ln == "# RockLiner"]
    ends = [i for i, ln in enumerate(lines)
            if ln.startswith("element zeroLength ")]
    assert starts and ends, "no interface unit found"
    return lines[starts[0]:ends[-1] + 1]


def _drop_ghost_lines(
    lines: "list[str]", ghost_ids: "set[int]",
) -> "list[str]":
    out = []
    for ln in lines:
        tok = ln.split()
        if tok and tok[0] in ("node", "fix") and int(tok[1]) in ghost_ids:
            continue
        out.append(ln)
    return out


# =====================================================================
# (1) The claimed unit: owner rank's stage block, exactly once
# =====================================================================
def test_claimed_units_emit_once_inside_owner_rank_stage_block(tmp_path):
    fem = _fem()
    n_pairs = len(fem.elements.interfaces)
    assert n_pairs == 4
    lines = _lines(_deck_text(_staged(_quad_ops(fem)), tmp_path))

    zl_all = _zl(lines)
    assert len(zl_all) == n_pairs                # exactly once, deck-wide

    # Nothing interface-shaped before the install stage: not in the
    # base (pre-stage) pass, not in the ground stage — the liner is
    # NOT installed at t = 0.
    pre_install = lines[:lines.index("# === Stage: install ===")]
    assert not _zl(pre_install)
    assert not [ln for ln in pre_install
                if ln.startswith("uniaxialMaterial ENT ")]

    # Inside the install stage: every unit inside the OWNER (rank 0 —
    # every backing element is a left-square quad) bracket only.
    install = _stage_block(lines, "install")
    brackets = _rank_brackets(install)
    assert _zl(brackets.get(0, [])) == zl_all
    assert not _zl(brackets.get(1, []))
    # …and the activated liner quads emit on THEIR rank (1), not the
    # owner's — the stage's activated topology fans out per element
    # owner while the interface unit follows the backing element.
    assert [ln for ln in brackets[1] if ln.startswith("element quad ")]
    assert not [ln for ln in brackets[0] if ln.startswith("element quad ")]
    # The unit sits before the stage's domainChange barrier.
    assert install.index(zl_all[-1]) < install.index("domainChange")


# =====================================================================
# (2) Serial-staged ↔ partitioned-staged comparability (tags included)
# =====================================================================
def test_serial_staged_vs_partitioned_staged_unit_lines_comparable(
    tmp_path,
):
    fem = _fem()
    slave_ids = {int(r.slave_node) for r in fem.elements.interfaces}

    part = _lines(_deck_text(_staged(_quad_ops(fem)), tmp_path, "part.tcl"))
    flat = _lines(_deck_text(
        _staged(_quad_ops(fem)), tmp_path, "flat.tcl", flat=True))

    part_install = _rank_brackets(_stage_block(part, "install"))[0]
    flat_install = _stage_block(flat, "install")

    part_iface = _drop_ghost_lines(_iface_portion(part_install), slave_ids)
    flat_iface = _iface_portion(flat_install)
    # Byte-for-byte, tags included: the claimed rows join the S8 tag
    # pre-pass in the exact sequence the serial-staged deck consumes
    # the allocator, so the two paths draw the SAME triples (no
    # element-minting MP pass coexists here — the S8-documented
    # conditional).
    assert part_iface == flat_iface


# =====================================================================
# (3) Ghost + stage-bound fix replay — ADR 0027 INV-2, both directions
# =====================================================================
def test_ghost_replays_stage_fix_and_later_stage_fix_mirrors(tmp_path):
    fem = _fem()
    slave_ids = sorted(int(r.slave_node) for r in fem.elements.interfaces)

    # install: s.fix(dof 1) on the wire — the stage-bound tier the
    # ghost declaration must replay ACROSS the stage boundary;
    # service: s.fix(dof 2) on the wire — the forward half: the owner
    # rank already holds the ghosts, so the later stage's delta must
    # mirror into its bracket.
    ops = _staged(
        _quad_ops(fem), install_fix=(1, 0), service_fix=(0, 1),
    )
    lines = _lines(_deck_text(ops, tmp_path))

    install_owner = _rank_brackets(_stage_block(lines, "install"))[0]
    for nid in slave_ids:
        decls = [i for i, ln in enumerate(install_owner)
                 if ln.split()[0] == "node" and int(ln.split()[1]) == nid]
        assert len(decls) == 1, (
            f"ghost {nid} declared {len(decls)} times in the owner's "
            "install bracket"
        )
        # The replayed stream directly after the decl carries the
        # stage-bound s.fix — the stage-inclusive stream, not just the
        # model-level tier.
        assert install_owner[decls[0] + 1] == f"fix {nid} 1 0", (
            f"ghost {nid} did not replay the install stage's s.fix"
        )

    service = _stage_block(lines, "service")
    brackets = _rank_brackets(service)
    for nid in slave_ids:
        # Home rank (1): the native stage-bound fix.
        assert f"fix {nid} 0 1" in brackets[1]
        # Owner rank (0): the mirror onto the held ghost — without it
        # the two ranks silently drift apart from this stage on.
        assert f"fix {nid} 0 1" in brackets[0], (
            f"service-stage s.fix on ghosted slave {nid} was not "
            "mirrored into the holder's bracket (ADR 0027 INV-2 "
            "forward half)"
        )


# =====================================================================
# (4) A claim-only stage still opens the owner bracket + global-tier
#     fix replay
# =====================================================================
def test_claim_only_stage_opens_owner_bracket_and_replays_global_fix(
    tmp_path,
):
    # Liner always-on (no s.activate): the ONLY stage content is the
    # claim. The unified content gate must still open the owner
    # bracket — a silently skipped bracket is the measured
    # s.equal_dof-alone failure mode. The wire nodes are global here,
    # so a model-level ops.fix on them is legal and must ride the
    # ghost replay into the stage block.
    fem = _fem()
    n_pairs = len(fem.elements.interfaces)
    slave_ids = sorted(int(r.slave_node) for r in fem.elements.interfaces)
    ops = _quad_ops(fem)
    ops.fix(pg="wire", dofs=(1, 0))
    lines = _lines(_deck_text(
        _staged(ops, activate=()), tmp_path))

    install_owner = _rank_brackets(_stage_block(lines, "install"))[0]
    assert len(_zl(install_owner)) == n_pairs
    for nid in slave_ids:
        i = next(i for i, ln in enumerate(install_owner)
                 if ln.split()[0] == "node" and int(ln.split()[1]) == nid)
        assert install_owner[i + 1] == f"fix {nid} 1 0"


# =====================================================================
# (5) The pattern-sp refusal still fires for stage patterns
# =====================================================================
def test_stage_pattern_sp_on_ghosted_claimed_slave_refuses(tmp_path):
    fem = _fem()
    ops = _quad_ops(fem)
    with ops.stage(name="ground") as s:
        s.analysis(**_pchain(ops))
        s.run(n_increments=1, dt=1.0)
    with ops.stage(name="install") as s:
        s.activate(pgs=["liner"])
        s.interface(name="RockLiner")
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.sp(pg="wire", dof=1, value=0.01)
        s.analysis(**_pchain(ops))
        s.run(n_increments=1, dt=1.0)
    with pytest.raises(BridgeError) as exc:
        ops.tcl(str(tmp_path / "deck.tcl"))
    msg = str(exc.value)
    assert "ADR 0027 INV-2" in msg
    assert "ghost-declared" in msg and "ADR 0093" in msg
    # The refusal names the offending slave nodes (a stage-owned Plain
    # is registered with the bridge too, so the sweep may surface it
    # under the global "pattern" label — either way it fires at plan
    # time, before a single line is emitted).
    slave_ids = {int(r.slave_node) for r in fem.elements.interfaces}
    assert any(str(nid) in msg for nid in slave_ids)


# =====================================================================
# (6) Mixed ndf inside the stage context
# =====================================================================
def test_mixed_ndf_claimed_unit_phantom_ndf2_ghost_beam_ndf3(tmp_path):
    fem = _fem(slave_ndf=3)
    recs = list(fem.elements.interfaces)
    assert all(r.phantom_node is not None for r in recs)
    slave_ids = {int(r.slave_node) for r in recs}
    phantoms = {int(r.phantom_node) for r in recs}

    ops = _staged(_mixed_ops(fem), activate=("wire",))
    lines = _lines(_deck_text(ops, tmp_path))

    install = _stage_block(lines, "install")
    brackets = _rank_brackets(install)
    owner, home = brackets[0], brackets[1]

    assert len(_zl(owner)) == len(recs)
    assert not _zl(home)
    for nid in sorted(slave_ids):
        ghost = [ln for ln in owner
                 if ln.split()[0] == "node" and int(ln.split()[1]) == nid]
        assert len(ghost) == 1
        # INV-5 ghost ndf parity inside the stage context: the beam
        # slave ghosts with its HOME ndf even under the ndf=2 envelope…
        assert ghost[0].endswith("-ndf 3")
        # …byte-identical to the home rank's native declaration.
        assert ghost[0] in home
    # The phantom declares the record's OWN ndf (2).
    for p in sorted(phantoms):
        assert [ln for ln in owner
                if ln.startswith(f"node {p} ") and ln.endswith("-ndf 2")]
    # The nested equalDOF is owner-only (the phantom exists on exactly
    # one rank — nothing to replicate).
    eq_owner = [ln for ln in owner if ln.startswith("equalDOF ")]
    assert len(eq_owner) == len(recs)
    for ln in eq_owner:
        _, retained, constrained, d1, d2 = ln.split()
        assert int(constrained) in phantoms
        assert int(retained) in slave_ids
        assert (d1, d2) == ("1", "2")
    assert not [ln for ln in home if ln.startswith("equalDOF ")]
    # INV-7: the ghost brings no beam element with it.
    assert not [ln for ln in owner
                if ln.startswith("element elasticBeamColumn ")]
    assert [ln for ln in home
            if ln.startswith("element elasticBeamColumn ")]
    # zeroLength joins master ↔ phantom (INV-1).
    for ln in _zl(owner):
        tok = ln.split()
        assert int(tok[4]) in phantoms


# =====================================================================
# (7) Claiming into a stage earlier than the topology refuses
# =====================================================================
def test_claim_into_stage_before_topology_activation_refuses(tmp_path):
    fem = _fem()
    ops = _quad_ops(fem)
    with ops.stage(name="ground") as s:
        s.analysis(**_pchain(ops))
        s.run(n_increments=1, dt=1.0)
    with ops.stage(name="early_claim") as s:
        s.interface(name="RockLiner")       # ← claims BEFORE the liner
        s.analysis(**_pchain(ops))
        s.run(n_increments=1, dt=1.0)
    with ops.stage(name="late_activate") as s:
        s.activate(pgs=["liner"])
        s.analysis(**_pchain(ops))
        s.run(n_increments=1, dt=1.0)
    with pytest.raises(BridgeError) as exc:
        ops.tcl(str(tmp_path / "deck.tcl"))
    msg = str(exc.value)
    assert "LATER stage" in msg
    assert "'late_activate'" in msg and "'early_claim'" in msg
    assert "ADR 0093 S9" in msg


# =====================================================================
# (8) Determinism
# =====================================================================
def test_two_partitioned_staged_emits_are_byte_identical(tmp_path):
    first = _deck_text(_staged(_quad_ops(_fem())), tmp_path, "a.tcl")
    second = _deck_text(_staged(_quad_ops(_fem())), tmp_path, "b.tcl")
    assert first == second


def test_mixed_ndf_partitioned_staged_emit_is_reproducible(tmp_path):
    first = _deck_text(
        _staged(_mixed_ops(_fem(slave_ndf=3)), activate=("wire",)),
        tmp_path, "a.tcl")
    second = _deck_text(
        _staged(_mixed_ops(_fem(slave_ndf=3)), activate=("wire",)),
        tmp_path, "b.tcl")
    assert first == second
