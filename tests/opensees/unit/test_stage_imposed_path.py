"""``s.imposed_path(node=, ratios=, series=)`` — a prescribed multi-DOF
path on one node inside a stage.

The rotational / staged counterpart to ``ops.imposed_displacement``,
which is translations-only (ux / uy / uz → DOFs 1-3) and global.  Here
``ratios`` is positional over the node's DOFs, so DOFs 4..6 are
reachable, and the pattern is stage-scoped.

Locks the emitted deck text (Tcl + openseespy): six ratios → six ``sp``
lines with the right DOF numbers and values, zeros skipped, and a nodal
load on another DOF of the SAME node in the SAME stage coexisting in the
deck.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.pattern.pattern import Plain

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_beam_fem() -> FEMStub:
    """Two beam elements on nodes 1-3 — ndf 6 in 3D, so DOFs 4..6 exist."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3],
            coords=[
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 2.0),
            ],
            node_pgs={"Base": [1], "Top": [3]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Cols": _ElementGroupView(
                    ids=(1, 2), connectivity=((1, 2), (2, 3)),
                ),
            },
        ),
    )


def _full_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Transformation(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _beam_ops(ndf: int = 6) -> apeSees:
    ops = apeSees(_make_beam_fem(), default_orientation=None)
    ops.model(ndm=3, ndf=ndf)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=2e11, Iz=1e-5, Iy=1e-5, G=8e10, J=2e-5,
    )
    return ops


def _sp_lines(lines: list[str]) -> list[str]:
    return [ln.strip() for ln in lines if ln.strip().startswith("sp ")]


# ===========================================================================
# Builder surface
# ===========================================================================


def test_returns_a_stage_scoped_plain_pattern() -> None:
    ops = _beam_ops()
    with ops.stage(name="slip") as s:
        p = s.imposed_path(
            node=3, ratios=(1.0,), series=ops.timeSeries.Linear(),
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert isinstance(p, Plain)
    assert ops._stage_records[0].pattern_specs == (p,)
    # Claimed for the stage, so the global pattern pass skips it.
    assert id(p) in ops._stage_claimed_pattern_ids


def test_rejects_empty_ratios() -> None:
    ops = _beam_ops()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="at least one entry"):
            s.imposed_path(
                node=3, ratios=(), series=ops.timeSeries.Linear(),
            )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_rejects_all_zero_ratios() -> None:
    """An all-zero path emits nothing — an inert directive, almost
    always a mistake (and a prescribed zero is a fixity, not a path)."""
    ops = _beam_ops()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="every ratio is zero"):
            s.imposed_path(
                node=3, ratios=(0.0, 0.0, 0.0),
                series=ops.timeSeries.Linear(),
            )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_rejects_more_ratios_than_the_model_ndf() -> None:
    ops = _beam_ops(ndf=3)
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="the model's ndf is 3"):
            s.imposed_path(
                node=3, ratios=(1.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                series=ops.timeSeries.Linear(),
            )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_exactly_ndf_ratios_is_allowed() -> None:
    ops = _beam_ops(ndf=6)
    with ops.stage(name="ok") as s:
        s.imposed_path(
            node=3, ratios=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            series=ops.timeSeries.Linear(),
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    ops.build().emit(RecordingEmitter())  # no raise


# ===========================================================================
# Tcl deck text
# ===========================================================================


def test_tcl_six_ratios_emit_six_sp_lines_with_the_right_dofs() -> None:
    ops = _beam_ops()
    with ops.stage(name="slip") as s:
        s.imposed_path(
            node=3,
            ratios=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            series=ops.timeSeries.Linear(),
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    assert _sp_lines(emitter.lines()) == [
        "sp 3 1 0.1",
        "sp 3 2 0.2",
        "sp 3 3 0.3",
        "sp 3 4 0.4",
        "sp 3 5 0.5",
        "sp 3 6 0.6",
    ]


def test_tcl_rotation_only_path_reaches_dof_6() -> None:
    """``ops.imposed_displacement`` cannot express this — it only maps
    ux / uy / uz onto DOFs 1-3."""
    ops = _beam_ops()
    with ops.stage(name="twist") as s:
        s.imposed_path(
            node=3, ratios=(0.0, 0.0, 0.0, 0.0, 0.0, 0.01),
            series=ops.timeSeries.Linear(),
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    assert _sp_lines(emitter.lines()) == ["sp 3 6 0.01"]


def test_tcl_zero_ratios_are_skipped_not_emitted_as_zero() -> None:
    ops = _beam_ops()
    with ops.stage(name="slip") as s:
        s.imposed_path(
            node=3, ratios=(1.0, 0.0, -2.0),
            series=ops.timeSeries.Linear(),
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    assert _sp_lines(emitter.lines()) == ["sp 3 1 1.0", "sp 3 3 -2.0"]


def test_tcl_a_held_load_on_another_dof_of_the_same_node_coexists() -> None:
    """A prescribed SP and a nodal load are different rows in the same
    pattern — driving DOF 6 must not evict a load on DOFs 1-3 of the
    same node in the same stage."""
    ops = _beam_ops()
    with ops.stage(name="slip") as s:
        p = s.imposed_path(
            node=3, ratios=(0.0, 0.0, 0.0, 0.0, 0.0, 0.01),
            series=ops.timeSeries.Linear(),
        )
        with p:
            p.load(node=3, forces=(0.0, 0.0, -5e3, 0.0, 0.0, 0.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = [ln.strip() for ln in emitter.lines()]
    assert "sp 3 6 0.01" in lines
    assert "load 3 0.0 0.0 -5000.0 0.0 0.0 0.0" in lines
    # Both inside the SAME pattern block (one pattern_open in the stage).
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    assert names.count("pattern_open") == 1
    open_i = names.index("pattern_open")
    close_i = names.index("pattern_close")
    inner = names[open_i:close_i]
    assert "sp" in inner and "load" in inner


def test_tcl_a_second_stage_gets_its_own_pattern() -> None:
    ops = _beam_ops()
    for name, ratio in (("a", 0.01), ("b", 0.02)):
        with ops.stage(name=name) as s:
            s.imposed_path(
                node=3, ratios=(0.0, 0.0, 0.0, 0.0, 0.0, ratio),
                series=ops.timeSeries.Linear(),
            )
            s.analysis(**_full_chain(ops))
            s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    assert _sp_lines(emitter.lines()) == ["sp 3 6 0.01", "sp 3 6 0.02"]


# ===========================================================================
# openseespy deck text
# ===========================================================================


def test_py_emits_the_same_sp_rows() -> None:
    ops = _beam_ops()
    with ops.stage(name="slip") as s:
        s.imposed_path(
            node=2, ratios=(0.5, 0.0, 0.0, 0.0, 0.25, 0.0),
            series=ops.timeSeries.Linear(),
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = PyEmitter()
    ops.build().emit(emitter)
    lines = [ln.strip() for ln in emitter.lines()]
    sp = [ln for ln in lines if ln.startswith("ops.sp(")]
    assert sp == ["ops.sp(2, 1, 0.5)", "ops.sp(2, 5, 0.25)"]


# ===========================================================================
# Emit slot
# ===========================================================================


def test_pattern_emits_after_the_chain_and_before_analyze() -> None:
    ops = _beam_ops()
    with ops.stage(name="slip") as s:
        s.imposed_path(
            node=3, ratios=(0.0, 0.0, 0.0, 0.0, 0.0, 0.01),
            series=ops.timeSeries.Linear(),
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    assert (
        names.index("analysis")
        < names.index("pattern_open")
        < names.index("analyze")
    )
