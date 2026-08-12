"""ADR 0092 S4 — partitioned contact emit (INV-1/INV-2/INV-7).

A contact interaction under partitioned (MPI) emit lands inside EXACTLY ONE
rank's block — the owner, chosen master-side (INV-1: element-exact where the
mesh's connectivity resolves each master facet to its backing solid; the
node tally otherwise) — preceded by ghost declarations of every interface
node the owner does not natively own (INV-2: ``node tag x y z`` + the
owner's replayed SP stream, the ADR 0027 ghost machinery). A ghost carries
geometry and SP state ONLY — never mass, never elements, never loads
(INV-7: the distributed solver sums shared-DOF contributions, so a massed
ghost double-counts and BOTH ranks agree on the same wrong answer; measured
~20% off, fork ADR-78 P0.e). Emission on one rank is structural, not
checked: two-rank emission converges to a plausible WRONG answer (half the
penetration) with no warning (fork ADR-78 P0.d).

The emitted shape mirrors the fork's P0 harness
(``OpenSees/Ladruno_files/testbed/contact_parallel/mp.tcl``), which matched
its serial twin to 1.4e-14: ghosts as ``node`` + replayed ``fix``, both
``contactSurface`` defs + the ``contact`` verb on the owner rank only.

Refusals (INV-5 — each error names the interaction and says why):
an undecidable owner (every candidate node replicated, counts tied — the
executed S1 counterexample) refuses rather than guessing; the S3 ``soft=``
refusal survives on the partitioned path; staged partitioned contact is
refused (the staged pipeline skips the analysis-chain auto-emit, so
``LadrunoContact`` would never be forced). No ghost-cost bound exists: the
2026-08-11 adversarial review WITHDREW the ghost budget (ADR 0092 §Sign-off
Q2), so none is enforced.

Emit-time only — no fork build needed.
"""
from __future__ import annotations

import re
from typing import cast

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import ContactRecord
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

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


def _contact_surface_node_tags(line: str) -> "set[int]":
    """Node tags referenced by one ``contactSurface`` line.

    ``-master <nps> <flat conn...>`` / ``-slave-segments <nps> <flat...>``
    carry a stride token first; ``-slave <nodes...>`` does not.
    """
    toks = line.split()
    assert toks[0] == "contactSurface"
    kind = toks[2]
    if kind in ("-master", "-slave-segments"):
        return {int(t) for t in toks[4:]}
    assert kind == "-slave"
    return {int(t) for t in toks[3:]}


# ---------------------------------------------------------------------------
# Fixtures — a real 2-rank two-body pounding model (gmsh-meshed)
# ---------------------------------------------------------------------------


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-3) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of vol {volume_tag} at z={z}")


def _two_box_fem(model_name: str, **contact_kw):
    """Two stacked boxes, one contact interaction, partitioned into 2."""
    with apeGmsh(model_name=model_name, verbose=False) as g:
        box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        box2 = g.model.geometry.add_box(0, 0, 1.05, 1, 1, 1)
        g.model.sync()
        master = _face_at_z(box1, 1.0)
        slave = _face_at_z(box2, 1.05)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box1, box2], name="solid")
        g.physical.add(2, [master], name="master")
        g.physical.add(2, [slave], name="slave")
        g.physical.add(2, [_face_at_z(box1, 0.0)], name="base")
        g.constraints.contact("master", "slave", **contact_kw)
        g.mesh.partitioning.partition(2)
        return g.mesh.queries.get_fem_data(dim=3)


def _ops(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeTetrahedron(pg="solid", material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))
    # SP state ON the slave interface — what the ghost replay (d) must
    # carry into the owner's block.
    ops.fix(pg="slave", dofs=(1, 1, 0))
    return ops


@pytest.fixture(scope="module")
def pounding_deck(tmp_path_factory):
    """One emitted 2-rank deck + its FEM, shared by the shape assertions."""
    fem = _two_box_fem("s4_pound", formulation="nts", kn="auto", name="pound")
    assert len(fem.partitions) == 2
    deck = tmp_path_factory.mktemp("s4") / "deck.tcl"
    _ops(fem).tcl(str(deck))
    return fem, deck.read_text(encoding="utf-8")


def _owner_block(deck_text: str) -> "tuple[int, list[str]]":
    """(owner_rank, its block lines) — asserts exactly one owner (a)."""
    blocks = _rank_lines(deck_text)
    owners = [
        rank for rank, lines in blocks.items()
        if any(ln.startswith("contact ") for ln in lines)
    ]
    assert len(owners) == 1, (
        f"contact verb in {len(owners)} rank blocks ({owners}) — "
        "must be exactly one (ADR 0092 INV-1; two-rank emission is the "
        "measured silent-wrong case, fork ADR-78 P0.d)"
    )
    return owners[0], blocks[owners[0]]


def _rank_owned_node_ids(fem, rank: int) -> "set[int]":
    parts = sorted(fem.partitions, key=lambda p: p.id)
    return {int(n) for n in parts[rank].node_ids}


# ---------------------------------------------------------------------------
# (a) + (b) — one owner block; every referenced node declared in it
# ---------------------------------------------------------------------------


def test_contact_verb_in_exactly_one_rank_block(pounding_deck):
    fem, text = pounding_deck
    _, lines = _owner_block(text)                # asserts (a)
    # Both contactSurface defs live in the same (owner) block, and none
    # anywhere else.
    surface_lines = [ln for ln in lines
                     if ln.startswith("contactSurface")]
    assert len(surface_lines) == 2
    all_blocks = _rank_lines(text)
    total = sum(
        1 for block in all_blocks.values() for ln in block
        if ln.startswith("contactSurface")
    )
    assert total == 2, "contactSurface defs leaked outside the owner block"


def test_every_referenced_interface_node_declared_in_owner_block(
    pounding_deck,
):
    fem, text = pounding_deck
    _, lines = _owner_block(text)
    referenced: "set[int]" = set()
    for ln in lines:
        if ln.startswith("contactSurface"):
            referenced |= _contact_surface_node_tags(ln)
    assert referenced
    declared = {int(ln.split()[1]) for ln in lines if ln.startswith("node ")}
    missing = referenced - declared
    assert not missing, (
        f"contactSurface references {len(missing)} node(s) never declared "
        f"(native or ghost) in the owner block: {sorted(missing)[:10]} — "
        "the fork marks unresolvable interface nodes HUGE_VAL and runs a "
        "silently partial interface (ADR 0092 INV-2)"
    )


# ---------------------------------------------------------------------------
# (c) — a ghost carries NO mass, NO element membership, NO loads (INV-7)
# ---------------------------------------------------------------------------


def _ghost_tags(fem, owner_rank: int, lines: "list[str]") -> "set[int]":
    declared = {int(ln.split()[1]) for ln in lines if ln.startswith("node ")}
    return declared - _rank_owned_node_ids(fem, owner_rank)


def test_ghosts_carry_geometry_and_sp_state_only(pounding_deck):
    fem, text = pounding_deck
    owner, lines = _owner_block(text)
    ghosts = _ghost_tags(fem, owner, lines)
    assert ghosts, "fixture must actually straddle ranks to test INV-7"

    for ln in lines:
        toks = ln.split()
        if toks[0] == "mass":
            assert int(toks[1]) not in ghosts, (
                f"ghost node {toks[1]} carries mass — the distributed "
                "solver sums shared-DOF mass, both ranks then agree on "
                "the same wrong answer (fork ADR-78 P0.e / INV-7)"
            )
        elif toks[0] == "load":
            assert int(toks[1]) not in ghosts, (
                f"ghost node {toks[1]} carries a load (INV-7)"
            )
        elif toks[0] == "element":
            # element <type> <tag> <n1..nk> <matTag> — every node arg of
            # the fixture's tets sits in toks[3:7].
            assert toks[1] == "FourNodeTetrahedron"
            elem_nodes = {int(t) for t in toks[3:7]}
            overlap = elem_nodes & ghosts
            assert not overlap, (
                f"element {toks[2]} in the owner block references ghost "
                f"node(s) {sorted(overlap)} — a ghost owns no elements "
                "on the declaring rank (INV-7)"
            )


# ---------------------------------------------------------------------------
# (d) — ghost fix lines replay the owner's SP stream
# ---------------------------------------------------------------------------


def test_ghost_sp_replay_matches_owning_ranks_stream(pounding_deck):
    fem, text = pounding_deck
    owner, lines = _owner_block(text)
    ghosts = _ghost_tags(fem, owner, lines)
    # Every ghost here is a slave-interface node; the model fixes the
    # whole slave surface (1, 1, 0). The ghost's declaration must replay
    # exactly that stream — no more (a ghost more constrained than its
    # owner silently answers stiffer), no less (singular matrix,
    # measured; ADR 0027 INV-2).
    slave_fixed = {
        int(n) for n in fem.nodes.select(pg="slave").ids
    }
    fixes_by_node: "dict[int, list[str]]" = {}
    for ln in lines:
        if ln.startswith("fix "):
            fixes_by_node.setdefault(int(ln.split()[1]), []).append(ln)
    for nid in sorted(ghosts):
        expected = (
            [f"fix {nid} 1 1 0"] if nid in slave_fixed else []
        )
        assert fixes_by_node.get(nid, []) == expected, (
            f"ghost {nid}: replayed SP stream "
            f"{fixes_by_node.get(nid, [])} != owner's {expected}"
        )
    # Adjacency: each ghost's fix follows its node line immediately (the
    # replay renders at declaration, ADR 0027 INV-2).
    for i, ln in enumerate(lines):
        if not ln.startswith("node "):
            continue
        nid = int(ln.split()[1])
        if nid in ghosts and nid in slave_fixed:
            assert lines[i + 1] == f"fix {nid} 1 1 0", (
                f"ghost {nid}: SP replay not adjacent to its node line"
            )


# ---------------------------------------------------------------------------
# (e) — undecidable owner refuses with the NAMED error (INV-1 / INV-5)
# ---------------------------------------------------------------------------


def _tied_stub_fem() -> FEMStub:
    """The executed S1 counterexample shape: every master node replicated
    onto both ranks with tied counts, and no element connectivity exposed
    (the stub's elements composite is not iterable), so the exact
    element-side pick cannot engage and the node tally must refuse."""
    nodes = _NodesStub(
        ids=[10, 11, 12, 13, 20, 21, 30, 31],
        coords=[
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.0, 0.0, 2.0), (1.0, 0.0, 2.0),
        ],
        node_pgs={"Base": [20, 21]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Bars": _ElementGroupView(
                ids=(1, 2),
                connectivity=((20, 21), (30, 31)),
            ),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.set_partitions([
        (0, [10, 11, 12, 13, 20, 21], [1]),
        (1, [10, 11, 12, 13, 30, 31], [2]),
    ])
    fem.elements.contacts = [
        ContactRecord(
            kind="contact",
            formulation="nts",
            master_faces=np.asarray([[10, 11, 12, 13]], dtype=np.int64),
            master_nps=4,
            slave_nodes=[30, 31],
            kn=1.0e6,
            name="tied",
        ),
    ]
    fem.elements.contact_planes = []
    return fem


def test_undecidable_owner_refuses_named(tmp_path):
    fem = _tied_stub_fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1e6)
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    with pytest.raises(BridgeError) as excinfo:
        ops.tcl(str(tmp_path / "deck.tcl"))
    msg = str(excinfo.value)
    assert "g.constraints.contact interaction #1 ('tied')" in msg
    assert "cannot choose an owner rank" in msg       # the resolver's why
    assert "ADR 0092 INV-1" in msg                    # the named contract
    assert "uncuttable_elements" in msg               # the suggested fix
    assert "serial-only" not in msg                   # the blanket is gone


# ---------------------------------------------------------------------------
# (f) — the S3 soft refusal still fires on the partitioned path
# ---------------------------------------------------------------------------


def test_soft_still_refused_on_the_relaxed_partitioned_path(tmp_path):
    fem = _two_box_fem("s4_soft_survives", formulation="nts",
                       kn=1.0e6, soft=True)
    with pytest.raises(BridgeError) as excinfo:
        _ops(fem).tcl(str(tmp_path / "deck.tcl"))
    msg = str(excinfo.value)
    assert "ADR 0092 INV-3" in msg
    assert "ASSEMBLED mass of BOTH contact surfaces" in msg


# ---------------------------------------------------------------------------
# staged partitioned contact — refused, named (S4 scope cut)
# ---------------------------------------------------------------------------


def test_staged_partitioned_contact_refused_named(tmp_path):
    fem = _two_box_fem("s4_staged", formulation="nts", kn=1.0e6)
    ops = _ops(fem)
    with ops.stage(name="s1") as s:
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-6, max_iter=20),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Transformation(),
            numberer=ops.numberer.ParallelPlain(),
            system=ops.system.Mumps(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    with pytest.raises(BridgeError) as excinfo:
        ops.tcl(str(tmp_path / "deck.tcl"))
    msg = str(excinfo.value)
    assert "STAGED partitioned" in msg
    assert "LadrunoContact" in msg


# ---------------------------------------------------------------------------
# serial emit is untouched (byte-identical serial emit is a hard rule)
# ---------------------------------------------------------------------------


def test_partitioned_deck_matches_serial_twin_shape(tmp_path):
    """The S4 deck's contact content equals the serial twin's: identical
    (stripped) `contact` verb line; identical surface node SETS —
    partitioning legitimately reorders node extraction, so the listings
    compare order-insensitively (measured at S3, recorded in the ADR)."""
    def shape(deck_path):
        verbs, surfaces = [], []
        for ln in deck_path.read_text().splitlines():
            ln = ln.strip()
            if ln.startswith("contact "):
                verbs.append(ln)
            elif ln.startswith("contactSurface"):
                surfaces.append(
                    (ln.split()[2], frozenset(_contact_surface_node_tags(ln)))
                )
        return sorted(verbs), sorted(surfaces, key=repr)

    with apeGmsh(model_name="s4_twin_serial", verbose=False) as g:
        box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        box2 = g.model.geometry.add_box(0, 0, 1.05, 1, 1, 1)
        g.model.sync()
        master = _face_at_z(box1, 1.0)
        slave = _face_at_z(box2, 1.05)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box1, box2], name="solid")
        g.physical.add(2, [master], name="master")
        g.physical.add(2, [slave], name="slave")
        g.physical.add(2, [_face_at_z(box1, 0.0)], name="base")
        g.constraints.contact("master", "slave",
                              formulation="nts", kn="auto")
        fem_serial = g.mesh.queries.get_fem_data(dim=3)

    fem_part = _two_box_fem("s4_twin_part", formulation="nts", kn="auto")

    serial_deck = tmp_path / "serial.tcl"
    part_deck = tmp_path / "part.tcl"
    _ops(fem_serial).tcl(str(serial_deck))
    _ops(fem_part).tcl(str(part_deck))

    assert shape(serial_deck) == shape(part_deck)
