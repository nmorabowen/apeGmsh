"""An ``equalDOF`` whose retained node is partly fixed (#2396).

OpenSees' Transformation handler (the bridge default for ``equalDOF``)
resets the constrained node's free tied DOF at every step when the
retained node carries an SP on another tied DOF
(``TransformationDOF_Group::enforceSPs``). The bridge rewrites such a tie
as ``equalDOF(m, s, free DOFs)`` + ``fix(s, fixed DOFs)``, the same
constraint set under every handler.

The model: two unit squares meshed apart, so their shared edge carries
coincident node pairs tied by ``g.constraints.equal_dof``; only square
A's base is fixed vertically, so the bottom pair's retained node is fixed
on DOF 2 and free on DOF 1.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.constraints import PartlyFixedTieWarning


@pytest.fixture(scope="module")
def squares() -> Any:
    g = apeGmsh(model_name="t2396", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A")
        g.model.geometry.add_rectangle(1, 0, 0, 1, 1, label="B")
        g.physical.add_surface("A", name="SA")
        g.physical.add_surface("B", name="SB")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.structured.set_transfinite_automatic()
        g.constraints.equal_dof("SA", "SB", dofs=[1, 2], name="iface")
        g.mesh.generation.generate(dim=2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()
    xyz = {int(n): c for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    masters = {int(r.master_node): int(r.slave_node)
               for r in fem.nodes.constraints}
    base_a = sorted(n for n, c in xyz.items()
                    if abs(c[1]) < 1e-9 and (c[0] < 1 - 1e-9 or n in masters))
    bottom_master = next(n for n in masters if abs(xyz[n][1]) < 1e-9)
    corner = min(base_a, key=lambda n: xyz[n][0])
    top_right = max(xyz, key=lambda n: (xyz[n][1], xyz[n][0]))
    return fem, base_a, bottom_master, masters[bottom_master], corner, top_right


def _ops(squares, *, staged_release: bool = False) -> apeSees:
    fem, base_a, master, _s, corner, top_right = squares
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3)
    ops.element.FourNodeQuad(pg="SA", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="SB", thickness=1.0, material=mat)
    ops.fix(nodes=base_a, dofs=(0, 1))
    ops.fix(nodes=[corner], dofs=(1, 0))
    if not staged_release:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=top_right, forces=(1.0, 0.0))
    else:
        chain = dict(
            test=ops.test.NormDispIncr(tol=1e-10, max_iter=5),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Lagrange(),
            numberer=ops.numberer.RCM(), system=ops.system.UmfPack(),
            analysis=ops.analysis.Static())
        with ops.stage(name="release") as s:
            s.remove_sp(nodes=[master], dofs=(2,))
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.load(node=top_right, forces=(1.0, 0.0))
            s.analysis(**chain)
            s.run(n_increments=1)
    return ops


def _lines(ops: apeSees, path: Path) -> list[str]:
    ops.tcl(str(path))
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip().startswith(("equalDOF", "fix "))]


def test_partly_fixed_master_is_rewritten(squares, tmp_path: Path) -> None:
    _fem, _b, m, s, _c, _t = squares
    lines = _lines(_ops(squares), tmp_path / "m.tcl")
    assert f"equalDOF {m} {s} 1" in lines          # DOF 2 left out ...
    assert f"fix {s} 0 1" in lines                 # ... and fixed on the slave
    others = [ln for ln in lines if ln.startswith("equalDOF")
              and not ln.startswith(f"equalDOF {m} ")]
    assert others and all(ln.endswith(" 1 2") for ln in others)


@pytest.mark.live
def test_transformation_now_matches_lagrange(squares) -> None:
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import get_ops

    u = {}
    for handler in ("Transformation", "Lagrange"):
        ops = _ops(squares)
        getattr(ops.constraints, handler)()
        ops.algorithm.Linear()
        ops.integrator.LoadControl(dlam=1.0)
        ops.numberer.RCM()
        ops.system.UmfPack()
        ops.analysis.Static()
        assert ops.analyze(steps=3) == 0
        osp = get_ops()
        u[handler] = np.array([[osp.nodeDisp(int(n), k) for k in (1, 2)]
                               for n in squares[0].nodes.ids])
    assert np.abs(u["Transformation"]).max() > 0
    assert np.allclose(u["Transformation"], u["Lagrange"],
                       atol=1e-9 * np.abs(u["Lagrange"]).max())


def test_a_staged_release_keeps_the_tie_and_warns(squares, tmp_path) -> None:
    _fem, _b, m, s, _c, _t = squares
    with pytest.warns(PartlyFixedTieWarning, match="#2396"):
        lines = _lines(_ops(squares, staged_release=True), tmp_path / "m.tcl")
    assert f"equalDOF {m} {s} 1 2" in lines


def test_the_archive_writes_the_rewritten_fix(squares, tmp_path) -> None:
    ops = _ops(squares)
    from apeGmsh.opensees._internal.artifact_write import BridgeArtifactWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", BridgeArtifactWarning)
        ops.h5(str(tmp_path / "arch.h5"))


def _staged(squares, verb: str) -> apeSees:
    """One static stage carrying the load and a ``verb`` on the bottom pair."""
    fem, base_a, master, slave, corner, top_right = squares
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3)
    ops.element.FourNodeQuad(pg="SA", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="SB", thickness=1.0, material=mat)
    ops.fix(nodes=base_a, dofs=(0, 1))
    ops.fix(nodes=[corner], dofs=(1, 0))
    chain = dict(
        test=ops.test.NormDispIncr(tol=1e-10, max_iter=5),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.LoadControl(dlam=1.0),
        constraints=ops.constraints.Lagrange(),
        numberer=ops.numberer.RCM(), system=ops.system.UmfPack(),
        analysis=ops.analysis.Static())
    with ops.stage(name="s") as s:
        if verb == "fix_slave":
            s.fix(nodes=[slave], dofs=(0, 1))
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=top_right, forces=(1.0, 0.0))
            if verb == "sp_master":
                p.sp(node=master, dof=1, value=1e-5)
        s.analysis(**chain)
        s.run(n_increments=1)
    return ops


@pytest.mark.parametrize("verb", ["fix_slave", "sp_master"])
def test_a_varying_sp_on_the_pair_keeps_the_tie_and_warns(
        squares, verb, tmp_path: Path) -> None:
    _fem, _b, m, s, _c, _t = squares
    with pytest.warns(PartlyFixedTieWarning, match="#2396"):
        lines = _lines(_staged(squares, verb), tmp_path / "m.tcl")
    assert f"equalDOF {m} {s} 1 2" in lines
    assert lines.count(f"fix {s} 0 1") <= 1      # never a duplicate SP


# -- #2406: the other emit paths ------------------------------------------

class _Rec:
    """Records what the constraint emitters write."""

    def __init__(self) -> None:
        self.lines: list[tuple] = []

    def __getattr__(self, verb: str) -> Any:
        if verb.startswith("_"):
            raise AttributeError(verb)
        return lambda *a, **_k: self.lines.append((verb, *a))


def _fixed(fixed: dict, staged=frozenset(), ndm: int = 2, ndf: int = 2):
    from apeGmsh.opensees._internal.build.constraints import FixedDofs

    return FixedDofs(fixed={n: frozenset(d) for n, d in fixed.items()},
                     staged=frozenset(staged), ndf={}, default_ndf=ndf,
                     owner={n: object() for n in fixed}, decl_cursor=None,
                     ndm=ndm)


def test_equal_dof_mixed_splits_on_the_retained_dof() -> None:
    from apeGmsh._kernel.records._constraints import NodePairRecord
    from apeGmsh.opensees._internal.build.constraints import _emit_equal_dofs

    rec = NodePairRecord(kind="equal_dof_mixed", master_node=1, slave_node=2,
                         dofs=[1, 2], master_dofs=[2, 1])
    em = _Rec()
    _emit_equal_dofs(em, [rec], fixed=_fixed({1: {2}}))
    # retained DOF 2 is held: its pair (2 -> 1) becomes fix(slave, DOF 1)
    assert em.lines == [("equalDOF_mixed", 1, 2, [(1, 2)]), ("fix", 2, 1, 0)]


def test_node_to_surface_pairs_split() -> None:
    from apeGmsh._kernel.records._constraints import (
        NodePairRecord, NodeToSurfaceRecord,
    )
    from apeGmsh.opensees._internal.build.constraints import _emit_equal_dofs

    pair = NodePairRecord(kind="equal_dof", master_node=7, slave_node=3,
                          dofs=[1, 2, 3])
    rec = NodeToSurfaceRecord(kind="node_to_surface", master_node=1,
                              equal_dof_records=[pair])
    em = _Rec()
    _emit_equal_dofs(em, [rec], fixed=_fixed({7: {3}}, ndm=3, ndf=3))
    assert em.lines == [("equalDOF", 7, 3, 1, 2), ("fix", 3, 0, 0, 1)]


@pytest.mark.parametrize("kind, fixed, staged, warns", [
    ("rod", {1: {2}}, (), True),          # 2-D rod maps DOFs 1, 2
    ("rod", {1: {1, 2}}, (), False),      # fully held
    ("rod", {1: {3}}, (), False),         # the rotation is not mapped
    ("beam", {1: {3}}, (), True),         # ... but a beam maps it
    ("rod", {}, ((1, 1),), True),         # a stage varies DOF 1 only
    ("rod", {}, (), False),
    ("rod", {1: {1, 2}}, ((1, 1),), True),    # held, a stage releases DOF 1
    ("rod", {}, ((1, 1), (1, 2)), False),     # every mapped DOF varies
])
def test_rigid_link_warns_on_a_partly_fixed_master(
        kind, fixed, staged, warns) -> None:
    from apeGmsh._kernel.records._constraints import NodePairRecord
    from apeGmsh.opensees._internal.build.constraints import _emit_rigid_links

    rec = NodePairRecord(kind=f"rigid_{kind}", master_node=1, slave_node=2)
    em = _Rec()
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        _emit_rigid_links(em, [rec],
                          fixed=_fixed(fixed, staged, ndm=2, ndf=3))
    hit = [w for w in got if issubclass(w.category, PartlyFixedTieWarning)]
    assert bool(hit) is warns
    assert em.lines == [("rigidLink", {"rod": "bar"}.get(kind, kind), 1, 2)]


def test_a_stage_claimed_tie_keeps_its_form_and_warns(
        squares, tmp_path: Path) -> None:
    """A ``fix`` added mid-analysis is total, an ``equalDOF`` incremental:
    a slave that moved in an earlier stage would snap back (review of
    #2421), so a stage-claimed tie is never rewritten."""
    fem, base_a, m, s, corner, top_right = squares
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3)
    ops.element.FourNodeQuad(pg="SA", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="SB", thickness=1.0, material=mat)
    ops.fix(nodes=base_a, dofs=(0, 1))
    ops.fix(nodes=[corner], dofs=(1, 0))
    with ops.stage(name="tie") as st:
        st.equal_dof(name="iface")
        with st.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=top_right, forces=(1.0, 0.0))
        st.analysis(
            test=ops.test.NormDispIncr(tol=1e-10, max_iter=5),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Transformation(),
            numberer=ops.numberer.RCM(), system=ops.system.UmfPack(),
            analysis=ops.analysis.Static())
        st.run(n_increments=1)
    with pytest.warns(PartlyFixedTieWarning, match="claims the tie"):
        lines = _lines(ops, tmp_path / "m.tcl")
    assert f"equalDOF {m} {s} 1 2" in lines
    assert f"fix {s} 0 1" not in lines


@pytest.mark.parametrize("staged", [False, True])
def test_a_partitioned_emit_rewrites_on_every_rank(staged, tmp_path) -> None:
    g = apeGmsh(model_name="t2406p", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A")
        g.model.geometry.add_rectangle(1, 0, 0, 1, 1, label="B")
        g.physical.add_surface("A", name="SA")
        g.physical.add_surface("B", name="SB")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.structured.set_transfinite_automatic()
        g.constraints.equal_dof("SA", "SB", dofs=[1, 2], name="iface")
        g.mesh.generation.generate(dim=2)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    finally:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)   # no model.h5
            g.end()
    xyz = {int(n): c for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    pairs = {int(r.master_node): int(r.slave_node)
             for r in fem.nodes.constraints}
    m = next(n for n in pairs if abs(xyz[n][1]) < 1e-9)
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3)
    ops.element.FourNodeQuad(pg="SA", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="SB", thickness=1.0, material=mat)
    ops.fix(nodes=[m], dofs=(0, 1))
    ops.fix(nodes=[n for n in xyz if xyz[n][0] < 1e-9], dofs=(1, 1))
    if staged:
        with ops.stage(name="tie") as st:
            st.equal_dof(name="iface")
            st.analysis(
                test=ops.test.NormDispIncr(tol=1e-10, max_iter=5),
                algorithm=ops.algorithm.Newton(),
                integrator=ops.integrator.LoadControl(dlam=1.0),
                constraints=ops.constraints.Transformation(),
                numberer=ops.numberer.ParallelPlain(),
                system=ops.system.Mumps(), analysis=ops.analysis.Static())
            st.run(n_increments=1)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("always", PartlyFixedTieWarning)
        ops.tcl(str(tmp_path / "p.tcl"))
    warned = any(issubclass(w.category, PartlyFixedTieWarning) for w in got)
    text = (tmp_path / "p.tcl").read_text()
    ties = [ln.strip() for ln in text.splitlines()
            if ln.strip().startswith(f"equalDOF {m} ")]
    dofs = "1 2" if staged else "1"
    assert warned is staged
    assert ties and all(t == f"equalDOF {m} {pairs[m]} {dofs}" for t in ties)
    assert text.count(f"fix {pairs[m]} 0 1") == (0 if staged else len(ties))


def test_mixed_pairs_sharing_a_varying_retained_dof_do_not_warn() -> None:
    from apeGmsh._kernel.records._constraints import NodePairRecord
    from apeGmsh.opensees._internal.build.constraints import _emit_equal_dofs

    rec = NodePairRecord(kind="equal_dof_mixed", master_node=1, slave_node=2,
                         dofs=[1, 2], master_dofs=[1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", PartlyFixedTieWarning)
        _emit_equal_dofs(_Rec(), [rec], fixed=_fixed({}, {(1, 1)}))


def test_a_rigid_body_warns_once() -> None:
    from apeGmsh._kernel.records._constraints import NodeGroupRecord
    from apeGmsh.opensees._internal.build.constraints import _emit_rigid_links

    rec = NodeGroupRecord(kind="rigid_body", master_node=1,
                          slave_nodes=list(range(2, 12)))
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        _emit_rigid_links(_Rec(), [rec], fixed=_fixed({1: {2}}, ndf=3))
    assert sum(issubclass(w.category, PartlyFixedTieWarning)
               for w in got) == 1


def test_a_tie_into_a_rigid_link_master_is_a_chain() -> None:
    """equalDOF m -> s with s retaining a rigidLink: rewriting would leave
    s partly fixed under the link, so the tie keeps its form and warns."""
    from apeGmsh._kernel.records._constraints import NodePairRecord
    from apeGmsh.opensees._internal.build.constraints import _emit_equal_dofs

    tie = NodePairRecord(kind="equal_dof", master_node=2, slave_node=3,
                         dofs=[1, 2])
    fd = _fixed({2: {2}})
    from dataclasses import replace

    fd = replace(fd, retained=frozenset({2, 3}), constrained=frozenset({3, 4}))
    em = _Rec()
    with pytest.warns(PartlyFixedTieWarning, match="chain"):
        _emit_equal_dofs(em, [tie], fixed=fd)
    assert em.lines == [("equalDOF", 2, 3, 1, 2)]
