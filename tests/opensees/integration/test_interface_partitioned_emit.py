"""ADR 0093 S8 — partitioned ``g.constraints.interface()`` emit (INV-5).

An interface pair is an ATOMIC unit — mixed-ndf phantom + nested
equalDOF + two tributary-scaled uniaxials + one zeroLength — that lands
inside EXACTLY ONE rank's block: the rank owning the pair's stamped
backing continuum element. Node-tally ownership (ADR 0092 INV-1) is
wrong here *by construction*: the pair's nodes are co-located, so a cut
hugging the interface replicates both onto both ranks — the all-shared
tie the contact resolver refuses as undecidable. ADR 0092 measured the
failure this design exists to kill: duplicate emission converges to a
plausible WRONG answer (double stiffness, half penetration) while base
reactions still balance, so equilibrium checks cannot catch it.

What is pinned here, on real gmsh-meshed two-body models:

* **one owner block + 1-vs-N byte identity** — the unit's lines inside
  the owner's block are byte-identical to the flat deck's (ghost decl
  prefixes aside, the ADR 0027 INV-1 phrasing), with IDENTICAL material
  and element tags, because the tag pre-pass allocates every record's
  triple in flat side-list order BEFORE the rank fan-out.
  **Flat↔partitioned tag identity is CONDITIONAL** (ADR 0093 INV-5, as
  amended during S8): it holds when no other element-minting MP pass
  coexists. Rigid-body / kinematic-coupling / ASDEmbeddedNodeElement
  tags are minted inside the per-rank 7b pass, after the interface
  pre-pass, while the flat path mints them before ``emit_interfaces``
  — so a combined model's interface tags drift between the two decks
  (measured: 21-24 vs 25-28 with a coexisting ``g.constraints.
  embedded``). What holds UNCONDITIONALLY — and is pinned by the
  mixed-model regression below — is exactly-once emission, global tag
  uniqueness within each deck, and cross-rank determinism;
* **the all-shared-cut degenerate case** — the cut hugs the interface,
  both pair nodes verified replicated onto both ranks, and the owner is
  still exact via the backing element (no tie, no heuristic);
* **foreign slave ghosting** — the owner's block declares a foreign
  slave with its HOME ndf (INV-5 ghost ndf parity: a mixed-ndf beam
  slave ghosts as ``-ndf 3`` even under an ndf=2 envelope) + the
  owner's SP replay, and carries no mass / no element / no load for it
  (ADR 0092 INV-7);
* **the pattern-sp-on-ghost refusal** — a pattern-borne ``sp`` on a
  node the plan would ghost is refused at plan time (ADR 0027 INV-2:
  pattern sp fans out on native ranks only and is never mirrored onto
  a ghost, so the owner's copy would stay unconstrained — the measured
  singular-matrix case);
* **the S9 refusal** — a stage-claimed interface under MPI still
  refuses, naming its own slice, ahead of everything;
* **determinism** — two partitioned emits are byte-identical.

Emit-time only — no OpenSees run needed.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError

from tests.opensees._helpers.partition_diff import (
    assert_partition_blocks_equivalent,
)

NORMAL = NormalLaw(kind="ent", k_per_area=1.0e9)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)
THICKNESS = 0.5

_RANK_OPEN_RE = re.compile(r"^\s*if \{\[getPID\] == (\d+)\} \{\s*$")


def _rank_lines(deck_text: str) -> "dict[int, list[str]]":
    """Split a partitioned (non-staged) Tcl deck into ``{rank: lines}``.

    Only lines inside a ``if {[getPID] == K} { ... }`` bracket are
    returned, ``.strip()``-ed (the bracket indent is emit formatting).
    """
    out: "dict[int, list[str]]" = {}
    rank: "int | None" = None
    depth = 0
    for raw in deck_text.splitlines():
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
        if raw.strip():
            out[rank].append(raw.strip())
    return out


# ---------------------------------------------------------------------
# Fixtures — the ADR 0093 two-squares model, with a CONTROLLED cut
# ---------------------------------------------------------------------


def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _two_squares(g, n: int):
    """Left square [0,1]^2 (the continuum master) + right square
    [1,2]^2, un-fragmented: the two curves at x=1 carry coincident but
    distinct node sets — the topology the verb pairs."""
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
    """Walk upward adjacencies to the surface an entity bounds — the
    two squares are un-fragmented, so every boundary entity (including
    the coincident ones at x=1) belongs to exactly one body."""
    while dim < 2:
        up, _down = gmsh.model.getAdjacencies(dim, tag)
        dim, tag = dim + 1, int(up[0])
    return tag


def _partition_left_right(g) -> None:
    """Split exactly at the interface: every element of the left body —
    including the boundary entities coincident with the cut — goes to
    partition 1, the right body to partition 2. Raw centroids cannot do
    this (the master and slave curves both sit at x=1.0), so lower-dim
    elements follow the body they bound."""
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


def _partition_across(g, threshold: float) -> None:
    """Run the cut ACROSS the interface at ``y = threshold``: every
    element (all dims) goes to partition 1 when its centroid sits below,
    else partition 2 — so the pair AT the threshold has both of its
    co-located nodes on the partition boundary, replicated onto both
    ranks (the all-shared degenerate case)."""
    elem_tags: "list[int]" = []
    parts: "list[int]" = []
    for dim in (2, 1, 0):
        _etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(
            dim=dim, tag=-1)
        for etags, enodes in zip(etags_list, enodes_list):
            if len(etags) == 0:
                continue
            nper = len(enodes) // len(etags)
            for k, tag in enumerate(etags):
                nds = enodes[k * nper:(k + 1) * nper]
                xyz = np.mean(
                    [gmsh.model.mesh.getNode(int(nid))[0] for nid in nds],
                    axis=0,
                )
                elem_tags.append(int(tag))
                parts.append(1 if xyz[1] < threshold else 2)
    g.mesh.partitioning.partition_explicit(2, elem_tags, parts)


def _fem(*, slave_ndf=None, n=4, cut=None, n_parts=None, embed=False):
    """``cut="split"`` puts the whole left body on rank 0 and the right
    on rank 1 (the cut hugs the interface from the side — foreign
    slaves); ``cut=("y", 0.5)`` runs the cut ACROSS the interface (the
    mid pair's nodes land on the partition boundary — the all-shared
    case); ``n_parts=K`` uses the native METIS partitioner instead.
    ``embed=True`` adds a coexisting element-minting MP pass
    (``g.constraints.embedded`` → ASDEmbeddedNodeElement) for the
    tag-identity-caveat regression."""
    with apeGmsh(model_name="iface_s8", verbose=False) as g:
        _two_squares(g, n=n)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, slave_ndf=slave_ndf, name="RockLiner")
        if embed:
            g.constraints.embedded("rock", "wire", name="Bond")
        if cut == "split":
            _partition_left_right(g)
        elif cut is not None:
            _axis, threshold = cut
            _partition_across(g, threshold)
        elif n_parts is not None:
            g.mesh.partitioning.partition(n_parts)
        return g.mesh.queries.get_fem_data()


def _quad_ops(fem, *, fix_wire: "tuple[int, int] | None" = None):
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in ("rock", "liner"):
        ops.element.FourNodeQuad(
            pg=pg, thickness=THICKNESS, material=mat,
            plane_type="PlaneStrain")
    if fix_wire is not None:
        ops.fix(pg="wire", dofs=fix_wire)
    return ops


def _mixed_ops(fem, *, fix_wire: "tuple[int, int, int] | None" = None):
    """2-dof continuum master vs 3-dof beam slave (the campaign's
    primary case) under an ndf=2 envelope — per-node ndf is LIVE."""
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeQuad(
        pg="rock", thickness=THICKNESS, material=mat,
        plane_type="PlaneStrain")
    ops.element.elasticBeamColumn(
        pg="wire", transf=ops.geomTransf.Linear(),
        A=0.02, E=200e9, Iz=1.0e-4)
    if fix_wire is not None:
        ops.fix(pg="wire", dofs=fix_wire)
    return ops


def _deck_text(ops, tmp_path, name="deck.tcl", **kw) -> str:
    path = tmp_path / name
    ops.tcl(str(path), **kw)
    return path.read_text(encoding="utf-8")


def _iface_portion(lines: "list[str]") -> "list[str]":
    """The lines from the first ``# RockLiner`` unit comment through the
    last ``element zeroLength`` — the interface block (plus, in a rank
    block, any interleaved ghost decl lines, which the caller filters)."""
    starts = [i for i, ln in enumerate(lines) if ln == "# RockLiner"]
    ends = [i for i, ln in enumerate(lines)
            if ln.startswith("element zeroLength ")]
    assert starts and ends, "no interface unit found"
    return lines[starts[0]:ends[-1] + 1]


def _drop_ghost_lines(
    lines: "list[str]", ghost_ids: "set[int]",
) -> "list[str]":
    """Remove a foreign slave's ghost ``node`` decl + SP replay lines —
    the one intentional textual divergence from the flat deck."""
    out = []
    for ln in lines:
        tok = ln.split()
        if tok and tok[0] in ("node", "fix") and int(tok[1]) in ghost_ids:
            continue
        out.append(ln)
    return out


def _zl_lines(lines: "list[str]") -> "list[str]":
    return [ln for ln in lines if ln.startswith("element zeroLength ")]


# =====================================================================
# (1) 1-vs-N byte identity — one owner block, identical tags
# =====================================================================
def test_two_rank_owner_block_matches_the_flat_deck_byte_for_byte(tmp_path):
    fem = _fem(cut="split")
    assert len(fem.partitions) == 2
    recs = list(fem.elements.interfaces)
    assert len(recs) == 4                                  # n=4 face nodes

    part_text = _deck_text(_quad_ops(fem), tmp_path, "part.tcl")
    flat_text = _deck_text(_quad_ops(fem), tmp_path, "flat.tcl", flat=True)

    blocks = _rank_lines(part_text)
    owners = [r for r, lines in blocks.items() if _zl_lines(lines)]
    assert owners == [0], (
        f"interface units in rank blocks {owners} — every backing "
        "element is a left-square quad, so rank 0 owns ALL units "
        "(ADR 0093 INV-5)"
    )

    slave_ids = {int(r.slave_node) for r in recs}
    owner_iface = _drop_ghost_lines(
        _iface_portion(blocks[0]), slave_ids,
    )
    flat_iface = _iface_portion(
        [ln.strip() for ln in flat_text.splitlines()],
    )
    # Byte-for-byte — comments, uniaxialMaterial lines, zeroLength
    # lines, tags and all: the up-front tag pre-pass makes the flat and
    # per-rank tag streams identical. (Equal-ndf ghost decls carry no
    # -ndf token, so they are filtered by tag here — the ADR 0027
    # INV-1 helper form of this comparison is the mixed-ndf test
    # below, where every ghost decl carries -ndf and the helper itself
    # does the stripping.)
    assert owner_iface == flat_iface


def test_mixed_ndf_owner_block_matches_flat_modulo_ghost_decls(tmp_path):
    # The ADR 0027 INV-1 phrasing, applied verbatim by the locked
    # helper: the owner block's RAW interface portion and the flat
    # deck's differ only by node-decl-with--ndf lines. Every mixed-ndf
    # ghost decl carries the home ndf (-ndf 3) and every phantom decl
    # its own (-ndf 2), so assert_partition_blocks_equivalent strips
    # them on both sides itself — no pre-filtering, no tautology —
    # and what remains (comments, equalDOF, materials, zeroLength,
    # tags and all) must match byte-for-byte.
    fem = _fem(slave_ndf=3, cut="split")
    part_text = _deck_text(_mixed_ops(fem), tmp_path, "part.tcl")
    flat_text = _deck_text(_mixed_ops(fem), tmp_path, "flat.tcl", flat=True)

    blocks = _rank_lines(part_text)
    owners = [r for r, lines in blocks.items() if _zl_lines(lines)]
    assert owners == [0]
    assert_partition_blocks_equivalent(
        "\n".join(_iface_portion(blocks[0])),
        "\n".join(_iface_portion(
            [ln.strip() for ln in flat_text.splitlines()],
        )),
        label="iface-mixed-2rank",
    )


def test_three_rank_units_each_emit_exactly_once_with_flat_tags(tmp_path):
    fem = _fem(n=6, n_parts=3)
    assert len(fem.partitions) == 3
    n_pairs = len(fem.elements.interfaces)
    assert n_pairs == 6

    part_text = _deck_text(_quad_ops(fem), tmp_path, "part.tcl")
    flat_text = _deck_text(_quad_ops(fem), tmp_path, "flat.tcl", flat=True)
    flat_lines = [ln.strip() for ln in flat_text.splitlines()]
    blocks = _rank_lines(part_text)

    part_zl = [ln for lines in blocks.values() for ln in _zl_lines(lines)]
    flat_zl = _zl_lines(flat_lines)
    assert len(part_zl) == len(flat_zl) == n_pairs
    # Each unit in exactly one block, with the FLAT deck's exact line —
    # same endpoints, same material tags, same element tag, same orient.
    assert sorted(part_zl) == sorted(flat_zl)
    for kind in ("uniaxialMaterial ENT ", "uniaxialMaterial ElasticPP "):
        part_mats = [ln for lines in blocks.values() for ln in lines
                     if ln.startswith(kind)]
        flat_mats = [ln for ln in flat_lines if ln.startswith(kind)]
        assert sorted(part_mats) == sorted(flat_mats)
        assert len(part_mats) == n_pairs


# =====================================================================
# (2) The all-shared-cut degenerate case — owner exact via the element
# =====================================================================
def test_cut_across_the_interface_owner_still_exact(tmp_path):
    # n=3 ⇒ pairs at y ∈ {0, 0.5, 1}; the cut at y=0.5 puts the mid
    # pair's BOTH nodes on the partition boundary — replicated onto
    # both ranks, zero locality information in the node sets (the
    # ADR 0092 undecidable tie). The backing element decides, exactly.
    fem = _fem(n=3, cut=("y", 0.5))
    parts = sorted(fem.partitions, key=lambda p: p.id)
    assert len(parts) == 2

    coords_of = {
        int(nid): tuple(map(float, xyz[:2]))
        for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords)
    }
    mid = [r for r in fem.elements.interfaces
           if abs(coords_of[int(r.master_node)][1] - 0.5) < 1e-9]
    assert len(mid) == 1
    rec = mid[0]

    # Verify the fixture IS the degenerate case: both pair nodes
    # replicated onto both ranks.
    for nid in (int(rec.master_node), int(rec.slave_node)):
        for part in parts:
            assert nid in {int(n) for n in part.node_ids}, (
                f"fixture failure: node {nid} not replicated onto "
                f"partition {part.id} — the cut does not hug the "
                "interface"
            )

    # The exact owner: the ONE partition holding the backing element.
    holding = [
        i for i, part in enumerate(parts)
        if int(rec.backing_element) in {int(e) for e in part.element_ids}
    ]
    assert len(holding) == 1
    owner = holding[0]

    text = _deck_text(_quad_ops(fem), tmp_path)
    blocks = _rank_lines(text)
    master, slave = int(rec.master_node), int(rec.slave_node)
    unit_line = [
        ln for ln in _zl_lines(blocks[owner])
        if int(ln.split()[3]) == master and int(ln.split()[4]) == slave
    ]
    assert len(unit_line) == 1, (
        "the mid pair's zeroLength must emit in the backing element's "
        "rank block"
    )
    other = 1 - owner
    assert not [
        ln for ln in _zl_lines(blocks.get(other, []))
        if int(ln.split()[3]) == master
    ], "the mid pair leaked into the non-owner block (ADR 0093 INV-5)"


# =====================================================================
# (3) Foreign slave ghosting — home ndf + SP replay, geometry only
# =====================================================================
def test_foreign_slave_ghosts_with_sp_replay_and_no_mass_or_elements(
    tmp_path,
):
    fem = _fem(cut="split")
    recs = list(fem.elements.interfaces)
    slave_ids = {int(r.slave_node) for r in recs}
    # The vertical cut puts every slave (wire) node's elements on rank
    # 1 — but the wire CURVE elements sit exactly at x=1 and were
    # assigned to partition 2, so slaves are native to rank 1 only.
    parts = sorted(fem.partitions, key=lambda p: p.id)
    rank0_native = {int(n) for n in parts[0].node_ids}
    foreign = sorted(slave_ids - rank0_native)
    assert foreign, "fixture failure: no foreign slave to ghost"

    ops = _quad_ops(fem, fix_wire=(1, 0))
    # A real nodal load on the slave side, so INV-7's "never loads"
    # half is asserted against a load that actually exists in the deck
    # (it must land on the ghost's PRIMARY rank, never the owner's
    # block).
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="wire", forces=(0.0, -1.0e3))
    text = _deck_text(ops, tmp_path)
    lines = _rank_lines(text)[0]

    for nid in foreign:
        decls = [i for i, ln in enumerate(lines)
                 if ln.split()[0] == "node" and int(ln.split()[1]) == nid]
        assert len(decls) == 1, (
            f"ghost node {nid} declared {len(decls)} times in the "
            "owner block"
        )
        # The owner's SP stream replays IMMEDIATELY after the decl
        # (ADR 0027 INV-2 machinery).
        assert lines[decls[0] + 1] == f"fix {nid} 1 0"
        # INV-7 / ADR 0092: geometry + SP only — never mass, never
        # elements, never loads.
        assert not [ln for ln in lines
                    if ln.startswith(f"mass {nid} ")]
        assert not [ln for ln in lines
                    if ln.startswith(f"load {nid} ")]
    # INV-7, element half: the ghost brings no slave-side quad with it —
    # rank 1's owned quads are absent from rank 0's block.
    rank1_lines = _rank_lines(text)[1]
    r1_quads = {ln for ln in rank1_lines if ln.startswith("element quad ")}
    r0_quads = {ln for ln in lines if ln.startswith("element quad ")}
    assert r1_quads and not (r0_quads & r1_quads)
    # …and the "never loads" assertion above is not vacuous: the wire
    # load exists, on the slave's primary (home) rank.
    assert any(
        ln.startswith(f"load {nid} ") for nid in foreign
        for ln in rank1_lines
    )


def test_mixed_ndf_foreign_beam_slave_ghosts_with_home_ndf(tmp_path):
    # INV-5's ghost ndf parity clause, verbatim: the beam slave's home
    # rank declares it ndf=3 (inferred, envelope is 2); the ghost decl
    # in the owner's block must carry the SAME explicit -ndf 3 — a
    # ghost defaulting to the envelope would silently disagree on
    # shared-DOF counts exactly where per-node ndf is live.
    fem = _fem(slave_ndf=3, cut="split")
    recs = list(fem.elements.interfaces)
    assert all(r.phantom_node is not None for r in recs)
    slave_ids = {int(r.slave_node) for r in recs}
    parts = sorted(fem.partitions, key=lambda p: p.id)
    foreign = sorted(slave_ids - {int(n) for n in parts[0].node_ids})
    assert foreign

    text = _deck_text(_mixed_ops(fem, fix_wire=(1, 0, 0)), tmp_path)
    blocks = _rank_lines(text)
    owner_lines, home_lines = blocks[0], blocks[1]

    for nid in foreign:
        ghost = [ln for ln in owner_lines
                 if ln.split()[0] == "node" and int(ln.split()[1]) == nid]
        assert len(ghost) == 1
        assert ghost[0].endswith("-ndf 3"), (
            f"ghost decl {ghost[0]!r} must carry the HOME ndf "
            "(ADR 0093 INV-5 ghost ndf parity)"
        )
        # Byte parity with the home rank's native declaration.
        assert ghost[0] in home_lines
        # SP replay adjacent, per-node.
        idx = owner_lines.index(ghost[0])
        assert owner_lines[idx + 1] == f"fix {nid} 1 0 0"
        # INV-7: the ghost brings no beam element with it.
    assert not [ln for ln in owner_lines
                if ln.startswith("element elasticBeamColumn ")]

    # The atomic unit sits on the owner rank only: phantom (ndf=2),
    # equalDOF (owner-only — the phantom exists on exactly one rank),
    # zeroLength master↔phantom.
    phantoms = {int(r.phantom_node) for r in recs}
    for p in phantoms:
        assert [ln for ln in owner_lines
                if ln.startswith(f"node {p} ") and ln.endswith("-ndf 2")]
    eq_owner = [ln for ln in owner_lines if ln.startswith("equalDOF ")]
    assert len(eq_owner) == len(recs)
    for ln in eq_owner:
        _, retained, constrained, d1, d2 = ln.split()
        assert int(constrained) in phantoms
        assert int(retained) in slave_ids
        assert (d1, d2) == ("1", "2")
    assert not [ln for ln in blocks[1] if ln.startswith("equalDOF ")], (
        "the nested equalDOF replicated onto the non-owner rank — the "
        "phantom exists on exactly one rank, there is nothing to "
        "replicate (ADR 0093 INV-5)"
    )
    assert len(_zl_lines(owner_lines)) == len(recs)
    assert not _zl_lines(blocks[1])


# =====================================================================
# (3b) Pattern-borne sp on a ghosted slave — refused at plan time
# =====================================================================
def test_pattern_sp_on_ghosted_slave_refuses(tmp_path):
    # ghost_sp_ops replays the model-level `fix` tier only; pattern sp
    # lines fan out on NATIVE ranks and are never mirrored onto a
    # ghost, so the owner rank's copy would stay unconstrained — the
    # ADR 0027 INV-2 measured singular-matrix case. The campaign's
    # own shape: a prescribed displacement on the liner (wire) side.
    fem = _fem(cut="split")
    ops = _quad_ops(fem)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(pg="wire", dof=1, value=0.01)
    with pytest.raises(BridgeError) as exc:
        ops.tcl(str(tmp_path / "deck.tcl"))
    msg = str(exc.value)
    assert "ADR 0027 INV-2" in msg
    assert "ghost-declared" in msg and "ADR 0093" in msg
    assert "uncuttable_elements" in msg          # the advertised fixes
    assert "ops.fix" in msg


def test_pattern_sp_on_node_target_ghosted_slave_refuses(tmp_path):
    # Same refusal through the node= target form, naming the node.
    fem = _fem(cut="split")
    slave = int(list(fem.elements.interfaces)[0].slave_node)
    ops = _quad_ops(fem)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(node=slave, dof=2, value=-0.005)
    with pytest.raises(BridgeError, match=rf"\[{slave}\]"):
        ops.tcl(str(tmp_path / "deck.tcl"))


def test_pattern_sp_on_native_nodes_still_emits(tmp_path):
    # Negative control: an sp on the master ("face") side — native to
    # the owner rank, nothing ghosted — emits normally.
    fem = _fem(cut="split")
    ops = _quad_ops(fem)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(pg="face", dof=1, value=0.01)
    text = _deck_text(ops, tmp_path)
    assert [ln for lines in _rank_lines(text).values()
            for ln in lines if ln.startswith("sp ")]
    assert len(_zl_lines(_rank_lines(text)[0])) == 4


# =====================================================================
# (3c) Coexisting element-minting MP pass — the tag-identity caveat
# =====================================================================
def test_interface_plus_embedded_exactly_once_with_documented_drift(
    tmp_path,
):
    # Flat↔partitioned interface-tag identity is CONDITIONAL (module
    # docstring / ADR 0093 INV-5 amended during S8): ASDEmbedded
    # element tags are minted inside the per-rank 7b pass, AFTER the
    # interface pre-pass, while the flat path mints them before
    # emit_interfaces. What must hold regardless — and is pinned here —
    # is exactly-once emission, global element-tag uniqueness within
    # each deck, and the drift itself (so if allocation order ever
    # unifies, this pin flags the docs for cleanup).
    fem = _fem(cut="split", embed=True)
    recs = list(fem.elements.interfaces)
    assert recs

    part_text = _deck_text(_quad_ops(fem), tmp_path, "part.tcl")
    flat_text = _deck_text(_quad_ops(fem), tmp_path, "flat.tcl", flat=True)
    flat_lines = [ln.strip() for ln in flat_text.splitlines()]
    blocks = _rank_lines(part_text)
    # The coexisting pass really mints elements in both decks.
    assert [ln for ln in flat_lines
            if ln.startswith("element ASDEmbeddedNodeElement ")]
    assert [ln for lines in blocks.values() for ln in lines
            if ln.startswith("element ASDEmbeddedNodeElement ")]

    part_zl = [ln for lines in blocks.values() for ln in _zl_lines(lines)]
    flat_zl = _zl_lines(flat_lines)
    assert len(part_zl) == len(flat_zl) == len(recs)   # exactly once

    def _ele_tags(lines):
        return [int(ln.split()[2]) for ln in lines
                if ln.startswith("element ")]

    part_tags = [t for lines in blocks.values() for t in _ele_tags(lines)]
    assert len(part_tags) == len(set(part_tags))       # globally unique
    assert len(_ele_tags(flat_lines)) == len(set(_ele_tags(flat_lines)))

    # Endpoints and orient match flat pair-for-pair; the TAGS drift —
    # the documented conditional (measured 21-24 vs 25-28 on this
    # model).
    def _sans_tag(ln):
        tok = ln.split()
        return tuple(tok[:2] + tok[3:])

    assert sorted(map(_sans_tag, part_zl)) == sorted(map(_sans_tag, flat_zl))
    assert sorted(int(ln.split()[2]) for ln in part_zl) != \
        sorted(int(ln.split()[2]) for ln in flat_zl), (
            "flat and partitioned interface tags now MATCH with a "
            "coexisting element-minting MP pass — the conditional "
            "documented in ADR 0093 INV-5 (S8 amendment) and this "
            "module's docstring can be retired"
        )
    # Cross-rank determinism is unconditional.
    assert part_text == _deck_text(_quad_ops(fem), tmp_path, "part2.tcl")


# =====================================================================
# (4) Staged + partitioned stays refused, naming S9
# =====================================================================
def test_stage_claimed_interface_under_mpi_still_refuses_naming_s9(
    tmp_path,
):
    fem = _fem(cut="split")
    ops = _quad_ops(fem)
    chain = {
        "test":        ops.test.NormDispIncr(tol=1e-6, max_iter=25),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=1.0),
        "constraints": ops.constraints.Transformation(),
        "numberer":    ops.numberer.ParallelPlain(),
        "system":      ops.system.Mumps(),
        "analysis":    ops.analysis.Static(),
    }
    with ops.stage(name="install") as s:
        s.interface(name="RockLiner")
        s.analysis(**chain)
        s.run(n_increments=1, dt=1.0)
    with pytest.raises(BridgeError) as exc:
        ops.tcl(str(tmp_path / "deck.tcl"))
    msg = str(exc.value)
    assert "ADR 0093 S9" in msg
    # Lifting the S8 side-list refusal must NOT have let the staged
    # claim fall through to a base-pass emit.
    assert "s.interface(name=" in msg


# =====================================================================
# (5) Determinism
# =====================================================================
def test_two_partitioned_emits_are_byte_identical(tmp_path):
    fem = _fem(cut="split")
    first = _deck_text(_quad_ops(fem), tmp_path, "a.tcl")
    second = _deck_text(_quad_ops(fem), tmp_path, "b.tcl")
    assert first == second


def test_mixed_ndf_partitioned_emit_is_reproducible(tmp_path):
    fem = _fem(slave_ndf=3, cut="split")
    first = _deck_text(_mixed_ops(fem), tmp_path, "a.tcl")
    second = _deck_text(_mixed_ops(fem), tmp_path, "b.tcl")
    assert first == second
