"""Deck replay restores ``equationConstraint`` and per-node ``fix`` widths (#1798).

Two of the three semantic diffs the R1b round-trip oracle (K2a, #1794)
found between ``ops.tcl(flat=True)`` and
``OpenSeesModel.from_h5(ops.h5()).build("tcl")``:

1. **The replay dropped every ``apeSees.equation_constraint`` row.** The
   archive holds each one as an ``EquationConstraintRecord`` row of
   ``/opensees/decl_params`` (K1-7), but ``_replay_into`` never read it, so
   a replayed deck ran without the tie and said nothing. The replay now
   re-emits each row through the bridge's own ``emit_equation_constraints``
   (the same node and DOF checks), and a declaration with no readable row
   refuses instead of dropping.
2. **The replay padded a mixed-ndf node's ``fix`` to the envelope ndf.**
   ``/opensees/bcs/fix`` stores every row at the envelope width (a
   fixed-width compound), so ``fix 1 1 1`` on a 2-DOF quad node under an
   ``ndf=3`` envelope came back as ``fix 1 1 1 0``. OpenSees accepts both
   (Tcl reads ``argc - 2`` flags; openseespy reads the node's own ndf), so
   the answer did not change, but the replayed deck was not the deck the
   bridge wrote. ``from_h5`` now trims each row, global and per stage, to
   its node's ``/opensees/nodes_ndf`` width, and refuses a row whose node
   that map lacks or whose cut-off slots are not zero.

The third item (the partitioned archive's runtime-fallback ``numberer`` /
``system`` lines) is a flat-vs-partitioned mode difference, not a replay
bug; see the PR for #1801.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter.h5 import H5LedgerWarning
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem
from tests.opensees.h5.test_h5_builder_scope_replay import (
    _staged_bridge,
    build_mixed_ndf_bridge,
    build_mixed_ndf_fem,
)
from tests.opensees.h5.test_h5_decl_params import _broad_frame


def _bridge_and_replay(
    ops: apeSees, tmp_path: Path, target: str = "tcl",
) -> tuple[str, str, Path]:
    deck = tmp_path / f"bridge.{target}"
    archive = tmp_path / "model.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the auto-emit / ledger advisories
        if target == "tcl":
            ops.tcl(str(deck), progress=False, flat=True)
        else:
            ops.py(str(deck), progress=False)
        ops.h5(str(archive))
        replay = OpenSeesModel.from_h5(str(archive)).build(target)
    assert isinstance(replay, str)
    return deck.read_text(encoding="utf-8"), replay, archive


def _lines(deck: str, head: str) -> list[str]:
    return [ln.strip() for ln in deck.splitlines() if ln.strip().startswith(head)]


# ---------------------------------------------------------------------------
# Item 1: equationConstraint
# ---------------------------------------------------------------------------


def test_broad_frame_replays_its_equation_constraint(tmp_path: Path) -> None:
    bridge, replay, _ = _bridge_and_replay(_broad_frame(), tmp_path)
    want = _lines(bridge, "equationConstraint")
    assert want == ["equationConstraint 1 1 1.0 2 1 1.0"]
    assert _lines(replay, "equationConstraint") == want
    # The tie needs an EQ-capable handler; the archive's chain carries it.
    assert _lines(replay, "constraints") == ["constraints Lagrange"]


def _two_ties() -> apeSees:
    ops = _tied_column(load=False)
    ops.equation_constraint(constrained=(2, 3), retained=[(2, 1, 2.0), (2, 2, -0.5)])
    return ops


def test_equation_constraints_replay_in_call_order(tmp_path: Path) -> None:
    bridge, replay, _ = _bridge_and_replay(_two_ties(), tmp_path)
    want = _lines(bridge, "equationConstraint")
    assert len(want) == 2
    assert _lines(replay, "equationConstraint") == want


def test_py_target_replays_the_equation_constraint(tmp_path: Path) -> None:
    bridge, replay, _ = _bridge_and_replay(_broad_frame(), tmp_path, "py")
    want = _lines(bridge, "ops.equationConstraint")
    assert len(want) == 1
    assert _lines(replay, "ops.equationConstraint") == want


def _rewrite_eq_row(archive: Path, edit: Callable[[dict[str, Any]], Any]) -> None:
    """Rewrite the equation-constraint row's stored params JSON in place."""
    with h5py.File(str(archive), "a") as f:
        dp = f["opensees/decl_params"]
        types = [t.decode() if isinstance(t, bytes) else str(t) for t in dp["type"][()]]
        (i,) = [k for k, t in enumerate(types) if t == "EquationConstraintRecord"]
        raw = dp["params"][()]
        rows = [r.decode() if isinstance(r, bytes) else str(r) for r in raw]
        params = json.loads(rows[i])
        edit(params)
        rows[i] = json.dumps(params, separators=(",", ":"))
        del dp["params"]
        dp.create_dataset("params", data=np.array(rows, dtype=h5py.string_dtype()))


def test_an_unreadable_equation_constraint_row_refuses(tmp_path: Path) -> None:
    _, _, archive = _bridge_and_replay(_broad_frame(), tmp_path)
    _rewrite_eq_row(archive, lambda p: p.pop("retained"))
    with pytest.raises(MalformedH5Error, match="equation_constraint"):
        OpenSeesModel.from_h5(str(archive)).build("tcl")


def test_a_bad_equation_constraint_row_refuses(tmp_path: Path) -> None:
    # A DOF past the node's ndf: the bridge's own emit check, on replay too.
    _, _, archive = _bridge_and_replay(_broad_frame(), tmp_path)
    _rewrite_eq_row(archive, lambda p: p.__setitem__("cdof", 7))
    with pytest.raises(Exception, match="exceeds that node's ndf"):
        OpenSeesModel.from_h5(str(archive)).build("tcl")


# A live closed form: a cantilever column along z (L = 1) with Iy = Iz,
# loaded by P along x at the tip, with the tie u_x(2) - u_y(2) = 0. Each
# bending direction has the tip stiffness k = 3 E I / L^3, so the tie shares
# P between them: u_x = u_y = P / (2 k). Without the tie, u_y = 0 and
# u_x = P / k.
_E, _I, _P = 200e9, 1e-4, 1.0e4
_K = 3.0 * _E * _I / 1.0**3


def _tied_column(*, load: bool = True) -> apeSees:
    ops = apeSees(cast("Any", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    sec = ops.section.Elastic(E=_E, A=0.01, Iz=_I, Iy=_I, G=80e9, J=1e-4)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ops.fix(nodes=[1], dofs=(1,) * 6)
    ops.equation_constraint(constrained=(2, 1), retained=[(2, 2, -1.0)])
    if load:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=2, forces=(_P, 0.0, 0.0, 0.0, 0.0, 0.0))
        ops.numberer.Plain()
        ops.system.FullGeneral()
        ops.test.NormDispIncr(tol=1e-12, max_iter=10)
        ops.algorithm.Newton()
        ops.integrator.LoadControl(dlam=1.0)
        ops.analysis.Static()
    return ops


_CHILD = """
import sys
src = open(sys.argv[1], encoding="utf-8").read()
ns = {"__name__": "replayed_deck"}
exec(compile(src, sys.argv[1], "exec"), ns)
ops = ns["ops"]
assert ops.analyze(1) == 0
print(repr((ops.nodeDisp(2, 1), ops.nodeDisp(2, 2))))
"""


def _child_has_openseespy() -> bool:
    return importlib.util.find_spec("openseespy") is not None


@pytest.mark.live
@pytest.mark.skipif(not _child_has_openseespy(), reason="openseespy not importable")
def test_replayed_tie_holds_in_opensees(tmp_path: Path) -> None:
    """The replayed openseespy deck enforces the tie (closed form).

    Run in a child interpreter: stock openseespy keeps EQ rows across
    ``wipe()``, so an in-process run would poison every later live test.
    """
    _bridge, replay, _ = _bridge_and_replay(_tied_column(), tmp_path, "py")
    deck = tmp_path / "replayed.py"
    deck.write_text(replay, encoding="utf-8")
    child = tmp_path / "child.py"
    child.write_text(_CHILD, encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(child), str(deck)],
        capture_output=True, text=True, check=False, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    ux, uy = eval(out.stdout.strip().splitlines()[-1])  # noqa: S307 - our own repr
    expected = _P / (2.0 * _K)
    assert ux == pytest.approx(expected, rel=1e-6)
    assert uy == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Item 2: fix width on mixed ndf
# ---------------------------------------------------------------------------

_MIXED: tuple[tuple[str, Callable[[], apeSees]], ...] = (
    ("build_mixed_ndf_bridge", build_mixed_ndf_bridge),
    ("_staged_bridge[Rock]", lambda: _staged_bridge(staged_pg="Rock")),
    ("_staged_bridge[Liner]", lambda: _staged_bridge(staged_pg="Liner")),
)


@pytest.mark.parametrize("build", [b for _, b in _MIXED], ids=[n for n, _ in _MIXED])
def test_mixed_ndf_fix_replays_at_the_node_width(
    build: Callable[[], apeSees], tmp_path: Path,
) -> None:
    bridge, replay, _ = _bridge_and_replay(build(), tmp_path)
    want = collections.Counter(_lines(bridge, "fix "))
    assert want["fix 1 1 1"] == 1 and want["fix 2 1 1"] == 1
    assert collections.Counter(_lines(replay, "fix ")) == want


def test_fixes_read_back_at_each_node_width(tmp_path: Path) -> None:
    _, _, archive = _bridge_and_replay(build_mixed_ndf_bridge(), tmp_path)
    om = OpenSeesModel.from_h5(str(archive))
    widths = {int(r.tag): len(r.dofs) for r in om.fixes()}
    assert widths == {1: 2, 2: 2, 5: 3}


def test_rewrite_keeps_the_fix_store(tmp_path: Path) -> None:
    """The trimmed rows re-pad to the same envelope-wide store on to_h5."""
    _, _, archive = _bridge_and_replay(build_mixed_ndf_bridge(), tmp_path)
    out = tmp_path / "rewritten.h5"
    OpenSeesModel.from_h5(str(archive)).to_h5(str(out))
    with h5py.File(str(archive), "r") as a, h5py.File(str(out), "r") as b:
        ra, rb = a["opensees/bcs/fix"][()], b["opensees/bcs/fix"][()]
        assert ra.dtype == rb.dtype
        assert list(ra["target"]) == list(rb["target"])
        assert np.array_equal(ra["dofs"], rb["dofs"])


def test_a_fixed_node_missing_from_nodes_ndf_refuses(tmp_path: Path) -> None:
    _, _, archive = _bridge_and_replay(build_mixed_ndf_bridge(), tmp_path)
    with h5py.File(str(archive), "a") as f:
        grp = f["opensees/nodes_ndf"]
        tags, ndf = grp["tags"][()], grp["ndf"][()]
        keep = tags != 1
        del grp["tags"], grp["ndf"]
        grp.create_dataset("tags", data=tags[keep])
        grp.create_dataset("ndf", data=ndf[keep])
    with pytest.raises(MalformedH5Error, match="node 1"):
        OpenSeesModel.from_h5(str(archive))


def test_a_set_flag_past_the_node_width_refuses(tmp_path: Path) -> None:
    _, _, archive = _bridge_and_replay(build_mixed_ndf_bridge(), tmp_path)
    with h5py.File(str(archive), "a") as f:
        rows = f["opensees/bcs/fix"][()]
        for r in rows:
            if int(r["target"]) == 1:
                r["dofs"][2] = 1
        # numpy structured rows are views: write the array back.
        f["opensees/bcs/fix"][...] = rows
    with pytest.raises(MalformedH5Error, match="node 1"):
        OpenSeesModel.from_h5(str(archive))


# ---------------------------------------------------------------------------
# Item 1, staged archive and pre-decls archive
# ---------------------------------------------------------------------------


def _staged_tied() -> apeSees:
    """The mixed-ndf model with the quad staged in, plus a tie between the
    two global beam nodes; the stage declares the EQ-capable Lagrange."""
    ops = apeSees(build_mixed_ndf_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    sec = ops.section.Elastic(E=2.0e8, A=0.01, Iz=1e-5)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.dispBeamColumn(
        pg="Liner", transf=ops.geomTransf.Linear(), integration=integ)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="LinerBase", dofs=(1, 1, 1))
    ops.equation_constraint(constrained=(6, 1), retained=[(5, 2, 0.5)])
    ts = ops.timeSeries.Linear()
    with ops.stage(name="dig") as s:
        s.activate(pgs=["Rock"])
        s.fix(pg="Base", dofs=(1, 1))
        with s.pattern(series=ts) as p:
            p.load(pg="LinerBase", forces=(10.0, 0.0, 0.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.5),
            constraints=ops.constraints.Lagrange(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=2)
    return ops


def test_staged_archive_replays_its_equation_constraint(tmp_path: Path) -> None:
    bridge, replay, archive = _bridge_and_replay(_staged_tied(), tmp_path)
    assert OpenSeesModel.from_h5(str(archive)).stages()  # the staged path
    want = _lines(bridge, "equationConstraint")
    assert want == ["equationConstraint 6 1 1.0 5 2 0.5"]
    assert _lines(replay, "equationConstraint") == want
    # Declared in the global prefix, before the stage's domainChange.
    lines = [ln.strip() for ln in replay.splitlines()]
    assert lines.index(want[0]) < lines.index("domainChange")


def test_an_archive_without_decls_warns_that_ties_are_unverified(
    tmp_path: Path,
) -> None:
    """No /opensees/decls (pre-2.25.0): replay loads, and says it cannot see
    the equation constraints, rather than dropping them in silence."""
    _, _, archive = _bridge_and_replay(_broad_frame(), tmp_path)
    with h5py.File(str(archive), "a") as f:
        for name in ("decls", "decl_params"):
            if name in f["opensees"]:
                del f["opensees"][name]
    om = OpenSeesModel.from_h5(str(archive))
    assert om.declarations is None
    with pytest.warns(H5LedgerWarning, match="cannot verify apeSees.equation_constraint"):
        om.build("tcl")
