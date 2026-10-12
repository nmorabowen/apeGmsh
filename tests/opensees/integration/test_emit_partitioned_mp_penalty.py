"""#1625 — a partitioned MP constraint under a handler that multiplies copies.

The partitioned emit writes a replicable MP record (equalDOF, rigidLink,
rigidDiaphragm, rigid_body chain, node_to_surface) on every rank that holds
one of its nodes (ADR 0027). ``Transformation`` needs that copy per rank.
``Penalty``, ``Lagrange`` and ``Auto`` assemble one penalty or multiplier
per copy, so the partitioned model answered stiffer than the serial one
(eigenvalues 4.9e-5 off on 2 ranks, 7.3e-5 on 4) and nothing said so.
The emit now refuses that combination before it writes anything, keyed
on the handler the deck actually runs under (declared, auto-emitted, or
each stage's), and names the two remedies the numeric oracle verifies:
``Transformation`` and a serial run. That oracle (serial vs 2 and 4 ranks,
run) is ``tests/opensees/subprocess/test_partitioned_mp_penalty_parity.py``.

Oracles, independent of the code under test:

* the rigid diaphragm's slaves are every plate node, which Gmsh splits
  over every partition, so it spans ranks by construction;
* an ``equalDOF`` between two nodes that only one partition holds (read
  off ``fem.partitions``) has one copy, so no handler can multiply it;
* under ``Transformation``, ``Plain`` or no declared handler the deck
  writes one ``rigidDiaphragm`` line per rank, as before.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir
from apeGmsh.opensees.emitter.recording import RecordingEmitter

pytestmark = pytest.mark.filterwarnings("ignore:partitioned run")

LX, LY, NX, NY = 6.0, 4.0, 6, 4
GRID = [(i * LX / NX, j * LY / NY) for i in range(NX + 1) for j in range(NY + 1)]
CM = (LX / 2, LY / 2, 0.0)


def diaphragm_plate(n_parts: int):
    """A shell plate on ground springs, Gmsh-partitioned, plus a decoupled
    master at its centre -> ``(fem, grounds, master)``."""
    with apeGmsh(model_name=f"rd_plate_p{n_parts}", verbose=False) as g:
        g.model.geometry.add_rectangle(0, 0, 0, LX, LY, label="plate")
        g.physical.add_surface("plate", name="Plate")
        grounds = {k: g.decouple_node(coords=(x, y, -1.0), label=f"gnd_{k}")
                   for k, (x, y) in enumerate(GRID)}
        master = g.decouple_node(coords=CM, label="cm")
        surf = gmsh.model.getEntities(2)[0][1]
        for _d, c in gmsh.model.getBoundary([(2, surf)], oriented=False):
            b = gmsh.model.getBoundingBox(1, abs(c))
            gmsh.model.mesh.setTransfiniteCurve(
                abs(c), NX + 1 if abs(b[3] - b[0]) > 1e-3 else NY + 1)
        gmsh.model.mesh.setTransfiniteSurface(surf)
        gmsh.model.mesh.setRecombine(2, surf)
        g.mesh.generation.generate(dim=2)
        if n_parts > 1:
            g.mesh.partitioning.partition(n_parts)
        fem = g.mesh.queries.get_fem_data(dim=2)
    return fem, grounds, master


def plate_node(fem, x: float, y: float) -> int:
    dec = {int(n) for n in fem.nodes.decoupled_ids}
    for n, c in zip(fem.nodes.ids, np.asarray(fem.nodes.coords)):
        if int(n) not in dec and np.allclose(c, (x, y, 0.0), atol=1e-9):
            return int(n)
    raise AssertionError(f"no plate node at {(x, y)}")


def with_diaphragm(fem, master) -> "object":
    slaves = [plate_node(fem, x, y) for (x, y) in GRID]
    offsets = np.asarray([(x, y, 0.0) for (x, y) in GRID]) - np.asarray(CM)
    return fem.with_constraint(NodeGroupRecord(
        kind="rigid_diaphragm", name="floor", master_node=int(master.tag),
        slave_nodes=slaves, dofs=[1, 2, 6], offsets=offsets,
        plane_normal=np.array([0.0, 0.0, 1.0])))


def declare_model(fem, grounds, master):
    """The plate, its springs, the master's out-of-plane fix and its mass."""
    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.4, rho=2400.0)
    ops.element.ASDShellQ4(pg="Plate", section=sec)
    spring = ops.uniaxialMaterial.ElasticMaterial(E=2.0e8)
    dirs = tuple(ZeroLengthMatDir(material=spring, dof=d) for d in (1, 2, 3))
    for k, (x, y) in enumerate(GRID):
        ops.element.ZeroLength(nodes=(grounds[k], plate_node(fem, x, y)), mat_dirs=dirs)
        ops.fix(nodes=[int(grounds[k].tag)], dofs=(1, 1, 1))
        ops.ndf(grounds[k], ndf=3)
    ops.fix(nodes=[int(master.tag)], dofs=(0, 0, 1, 1, 1, 0))
    ops.mass(nodes=[int(master.tag)], values=(5e4, 5e4, 0, 0, 0, 2e5))
    return ops


def handler(ops, name: str):
    """Declare (or not) the handler ``name`` on ``ops``' model-level chain."""
    ns = ops.constraints
    return {
        "Penalty": lambda: ns.Penalty(alpha_sp=1e12, alpha_mp=1e12),
        "Lagrange": lambda: ns.Lagrange(),
        "Auto": lambda: ns.Auto(),
        "AutoUser": lambda: ns.Auto(auto_penalty=False, user_penalty=1e12),
        "Transformation": lambda: ns.Transformation(),
        "Plain": lambda: ns.Plain(),
        "none": lambda: None,
    }[name]()


def chain(ops, name: str, n_parts: int) -> None:
    handler(ops, name)
    if n_parts > 1:
        ops.numberer.ParallelPlain()
        ops.system.Mumps()
    else:
        ops.numberer.RCM()
        ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.Newmark(gamma=0.5, beta=0.25)
    ops.analysis.Transient()


def diaphragm_ops(n_parts: int, name: str, *, serial_chain: bool = False):
    """The diaphragm model; ``serial_chain`` declares the serial
    numberer and system on a partitioned FEM (the ``flat=True`` remedy)."""
    fem, grounds, master = diaphragm_plate(n_parts)
    fem = with_diaphragm(fem, master)
    ops = declare_model(fem, grounds, master)
    chain(ops, name, 1 if serial_chain else n_parts)
    return ops


def emit(ops, path: Path, **kw) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(path), **kw)
    return path.read_text(encoding="utf-8")


def diaphragm_lines_by_rank(text: str) -> "dict[int, int]":
    out: dict[int, int] = {}
    rank = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("if {[getPID] == "):
            rank = int(s.split("==")[1].split("}")[0])
        elif s == "}":
            rank = None
        elif s.startswith("rigidDiaphragm") and rank is not None:
            out[rank] = out.get(rank, 0) + 1
    return out


@pytest.mark.parametrize("n_parts", [2, 4])
@pytest.mark.parametrize("name", ["Penalty", "Lagrange", "Auto", "AutoUser"])
def test_a_diaphragm_spanning_ranks_is_refused_under_a_multiplying_handler(
        tmp_path, n_parts, name):
    ops = diaphragm_ops(n_parts, name)
    deck = tmp_path / "model.tcl"
    deck.write_text("SENTINEL: the user's previous deck\n", encoding="utf-8")
    with pytest.raises(BridgeError) as err:
        emit(ops, deck)
    msg = str(err.value)
    assert "#1625" in msg and "'floor'" in msg
    assert "Transformation" in msg and "flat=True" in msg
    # the deck already at the path is untouched, and no other file appears
    assert deck.read_text(encoding="utf-8") == "SENTINEL: the user's previous deck\n"
    assert [f.name for f in tmp_path.iterdir()] == ["model.tcl"]


def test_the_refusal_comes_before_the_partitioned_emit_writes():
    """Only the ``model`` header precedes the partitioned emit; nothing of
    it (no global primitive, no rank block) is written."""
    rec = RecordingEmitter()
    bm = diaphragm_ops(2, "Penalty").build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BridgeError, match="#1625"):
            bm.emit(rec)
    assert [name for name, _a, _k in rec.calls] == ["model"]


def test_the_auto_emitted_lagrange_of_an_equation_tie_is_refused(tmp_path):
    """No handler declared + an ``enforce="equation"`` tie: the bridge
    auto-emits Lagrange, which multiplies the diaphragm's copies."""
    fem, grounds, master = diaphragm_plate(2)
    a, b = _one_rank_pair(fem)
    fem = with_diaphragm(fem, master)
    fem = fem.with_constraint(InterpolationRecord(
        kind=ConstraintKind.TIE, name="eq", slave_node=b, master_nodes=[a],
        weights=np.array([1.0]), dofs=[3], enforce="equation"))
    ops = declare_model(fem, grounds, master)
    chain(ops, "none", 2)
    with pytest.raises(BridgeError, match="Lagrange") as err:
        emit(ops, tmp_path / "model.tcl")
    assert "#1625" in str(err.value)


def test_the_auto_emitted_projection_of_an_equation_tie_is_named(tmp_path):
    """Under an explicit integrator the equation tie auto-emits
    ``LadrunoProjection``, and the refusal names that handler. Whether
    projection multiplies MP copies is unmeasured, so it is refused."""
    fem, grounds, master = diaphragm_plate(2)
    a, b = _one_rank_pair(fem)
    fem = with_diaphragm(fem, master)
    fem = fem.with_constraint(InterpolationRecord(
        kind=ConstraintKind.TIE, name="eq", slave_node=b, master_nodes=[a],
        weights=np.array([1.0]), dofs=[3], enforce="equation"))
    ops = declare_model(fem, grounds, master)
    ops.integrator.CentralDifferenceLadruno()
    with pytest.raises(BridgeError, match="LadrunoProjection") as err:
        emit(ops, tmp_path / "model.tcl")
    assert "Lagrange" not in str(err.value)


@pytest.mark.parametrize("n_parts", [2, 4])
def test_the_routed_spring_bed_is_not_refused_under_penalty(tmp_path, n_parts):
    """ADR 0120 D1 pins each side node to one rank, so its ``equalDOF`` has
    one copy: Penalty multiplies nothing and the deck is written."""
    from tests.opensees.integration.test_emit_partitioned_node_pair_routing import (
        declare, plate_on_springs,
    )

    fem, sides, grounds = plate_on_springs(n_parts)
    ops, fem = declare(fem, sides, grounds)
    ops.constraints.Penalty(alpha_sp=1e12, alpha_mp=1e12)
    text = emit(ops, tmp_path / "model.tcl")
    assert text.count("equalDOF ") == len(sides)


def _modal_ops(*, analysis: bool):
    fem, grounds, master = diaphragm_plate(2)
    fem = with_diaphragm(fem, master)
    ops = declare_model(fem, grounds, master)
    ops.constraints.Penalty(alpha_sp=1e12, alpha_mp=1e12)
    if analysis:
        ops.numberer.ParallelPlain()
        ops.system.Mumps()
        ops.test.NormDispIncr(tol=1e-12, max_iter=10)
        ops.algorithm.Linear()
        ops.integrator.LoadControl(dlam=1.0)
        ops.analysis.Static()
    return ops


def test_the_arpack_modal_deck_forces_transformation_and_is_not_refused(tmp_path):
    """``modal_deck(solver="arpack")`` forces ``constraints Transformation``
    before ``eigen`` (ADR 0077 INV-10): with no Analysis of the user's, the
    declared Penalty never runs, so the replicated diaphragm is right."""
    deck = tmp_path / "modal.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _modal_ops(analysis=False).modal_deck(str(deck), solver="arpack", num_modes=6)
    text = deck.read_text(encoding="utf-8")
    assert diaphragm_lines_by_rank(text) == {0: 1, 1: 1}
    assert "constraints Transformation" in text


def test_the_arpack_modal_deck_with_a_user_analysis_is_refused(tmp_path):
    """A declared Analysis is built with the declared Penalty, which then
    multiplies the diaphragm's copies: still refused."""
    deck = tmp_path / "modal.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BridgeError, match="#1625"):
            _modal_ops(analysis=True).modal_deck(
                str(deck), solver="arpack", num_modes=6)
    assert not deck.exists()


@pytest.mark.parametrize("name", ["Penalty", "Lagrange", "Auto"])
def test_the_serial_remedy_emits(tmp_path, name):
    """``flat=True``, the refusal's serial remedy, writes the deck."""
    text = emit(diaphragm_ops(2, name, serial_chain=True), tmp_path / "model.tcl",
                flat=True)
    assert "getPID" not in text
    assert text.count("rigidDiaphragm 3 ") == 1


@pytest.mark.parametrize("n_parts", [2, 4])
@pytest.mark.parametrize("name", ["Transformation", "Plain", "none"])
def test_replicating_handlers_keep_one_diaphragm_copy_per_rank(
        tmp_path, n_parts, name):
    text = emit(diaphragm_ops(n_parts, name), tmp_path / "model.tcl")
    assert diaphragm_lines_by_rank(text) == {r: 1 for r in range(n_parts)}


def test_the_serial_deck_is_not_refused(tmp_path):
    text = emit(diaphragm_ops(1, "Penalty"), tmp_path / "model.tcl")
    assert text.count("rigidDiaphragm 3 ") == 1


def _one_rank_pair(fem) -> "tuple[int, int]":
    """Two adjacent plate nodes held by exactly one (the same) partition."""
    held: dict[int, set[int]] = {}
    for idx, p in enumerate(fem.partitions):
        for n in np.asarray(p.node_ids):
            held.setdefault(int(n), set()).add(idx)
    alone = {n for n, r in held.items() if len(r) == 1}
    for (x, y) in GRID:
        a = plate_node(fem, x, y)
        b = plate_node(fem, x + LX / NX, y) if x + LX / NX <= LX else None
        if b is not None and a in alone and b in alone and held[a] == held[b]:
            return a, b
    raise AssertionError("no adjacent pair held by one partition")


def test_a_constraint_held_by_one_rank_is_not_refused(tmp_path):
    fem, grounds, master = diaphragm_plate(2)
    a, b = _one_rank_pair(fem)
    fem = fem.with_constraint(NodePairRecord(
        kind="equal_dof", name="tie", master_node=a, slave_node=b, dofs=[3]))
    ops = declare_model(fem, grounds, master)
    chain(ops, "Penalty", 2)
    text = emit(ops, tmp_path / "model.tcl")
    assert sum(1 for ln in text.splitlines()
               if ln.strip().startswith(f"equalDOF {a} {b} ")) == 1


# ---------------------------------------------------------------------------
# The stage-claimed path (#1620 item 1)
# ---------------------------------------------------------------------------


def _staged_ops(stage_handlers: "tuple[str, str]", *, claim: bool):
    """Two stages; the diaphragm is claimed by stage 2 (``claim``) or global."""
    fem, grounds, master = diaphragm_plate(2)
    fem = with_diaphragm(fem, master)
    ops = declare_model(fem, grounds, master)
    for i, name in enumerate(stage_handlers, start=1):
        with ops.stage(name=f"s{i}") as s:
            if claim and i == 2:
                s.rigid_diaphragm(name="floor")
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
                algorithm=ops.algorithm.Linear(),
                integrator=ops.integrator.LoadControl(dlam=1.0),
                constraints=handler(ops, name),
                numberer=ops.numberer.ParallelPlain(),
                system=ops.system.Mumps(),
                analysis=ops.analysis.Static(),
            )
            s.run(n_increments=1)
    return ops


@pytest.mark.parametrize("claim", [True, False])
@pytest.mark.parametrize("handlers", [
    ("Transformation", "Penalty"),     # the stage that holds it multiplies
    ("Penalty", "Transformation"),     # an earlier stage does (global only)
])
def test_staged_diaphragm_is_refused_under_a_multiplying_stage_handler(
        tmp_path, claim, handlers):
    ops = _staged_ops(handlers, claim=claim)
    deck = tmp_path / "model.tcl"
    if claim and handlers[1] == "Transformation":
        # the claimed record lives from stage 2 on, which is Transformation
        text = emit(ops, deck)
        assert sum(diaphragm_lines_by_rank_any(text).values()) == 2
        return
    with pytest.raises(BridgeError, match="#1625"):
        emit(ops, deck)
    assert not deck.exists()


def test_staged_transformation_keeps_the_copies(tmp_path):
    for claim in (True, False):
        text = emit(_staged_ops(("Transformation", "Transformation"), claim=claim),
                    tmp_path / f"model_{claim}.tcl")
        assert sum(diaphragm_lines_by_rank_any(text).values()) == 2


def diaphragm_lines_by_rank_any(text: str) -> "dict[int, int]":
    """``rigidDiaphragm`` lines per rank bracket, at any nesting depth."""
    out: dict[int, int] = {}
    rank = None
    depth = 0
    for line in text.splitlines():
        s = line.strip()
        if rank is None and s.startswith("if {[getPID] == "):
            rank = int(s.split("==")[1].split("}")[0])
            depth = 1
            continue
        if rank is not None:
            depth += s.count("{") - s.count("}")
            if depth <= 0:
                rank = None
                continue
            if s.startswith("rigidDiaphragm"):
                out[rank] = out.get(rank, 0) + 1
    return out
