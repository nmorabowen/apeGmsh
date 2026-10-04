"""ADR 0113 slice 1 -- the IMPL-EX time driver (``ops.implex_time``) and
the dTime-trap refusals.

The driver is STKO's ``STKO_DT_UTIL_OnBeforeAnalyze`` as persistent
parameters: ``dTime`` / ``dTimeCommit`` / ``dTimeInitial`` over every
element whose material closure reaches an IMPL-EX (or ``eta > 0``)
ASDConcrete, and each stage's increment written right after its
``analysis`` line.  The San Ramon port (research repo
``models/apegmsh/sanramon/staging.py``) wrote the same driver as deck
patches; its G20 gate found it bitwise identical to STKO's
``setParameter -val -ele`` loop.  ``test_golden_*`` replays both preludes
in a plain Tcl interpreter (``tkinter.Tcl``, no OpenSees) with the
OpenSees commands stubbed to a log, and requires the same command
sequence -- serial and per rank.
"""
from __future__ import annotations

import re

import pytest

from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.analysis.implex import (
    ImplexTime,
    implex_element_specs,
    reads_dtime,
)
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.section.fiber import FiberPoint
from apeGmsh.opensees.section.plate import ShellLayer

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

# ---------------------------------------------------------------------------
# Fixture: an IMPL-EX wall (ASDShellQ4 -> LayeredShell -> PlateFromPlaneStress
# -> ASDConcrete3D), an IMPL-EX fiber column (forceBeamColumn -> Lobatto ->
# Fiber -> ASDConcrete1D) and an elastic slab that is NOT a target.
# ---------------------------------------------------------------------------

WALL = (11, 12)
COLUMN = (21, 22)
SLAB = (31,)


def _fem() -> FEMStub:
    coords = {
        1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (2.0, 0.0, 0.0),
        4: (0.0, 0.0, 1.0), 5: (1.0, 0.0, 1.0), 6: (2.0, 0.0, 1.0),
        7: (3.0, 0.0, 0.0), 8: (3.0, 0.0, 1.0), 9: (3.0, 0.0, 2.0),
        10: (0.0, 1.0, 1.0), 13: (1.0, 1.0, 1.0),
    }
    ids = sorted(coords)
    return FEMStub(
        nodes=_NodesStub(
            ids=ids, coords=[coords[i] for i in ids],
            node_pgs={"Base": [1, 2, 3, 7]},
        ),
        elements=_ElementsStub(elem_pgs={
            "wall": _ElementGroupView(
                ids=WALL, connectivity=((1, 2, 5, 4), (2, 3, 6, 5)),
            ),
            "column": _ElementGroupView(
                ids=COLUMN, connectivity=((7, 8), (8, 9)),
            ),
            "slab": _ElementGroupView(
                ids=SLAB, connectivity=((4, 5, 13, 10),),
            ),
        }),
    )


def _model(fem: FEMStub | None = None, *, implex: bool = True,
           eta: float = 0.0) -> apeSees:
    ops = apeSees(fem or _fem(), default_orientation=None, element_tags="fem")
    ops.model(ndm=3, ndf=6)
    c3 = ops.nDMaterial.ASDConcrete3D(
        E=30000.0, v=0.2, fc=30.0, implex=implex, eta=eta, lch_ref=100.0)
    plate = ops.nDMaterial.PlateFromPlaneStress(material=c3, G_out=12500.0)
    wall = ops.section.LayeredShell(layers=tuple(
        ShellLayer(material=plate, thickness=50.0) for _ in range(3)
    ))
    ops.element.ASDShellQ4(pg="wall", section=wall)
    c1 = ops.uniaxialMaterial.ASDConcrete1D(
        E=30000.0, fc=30.0, implex=implex, lch_ref=100.0)
    fib = ops.section.Fiber(fibers=(
        FiberPoint(material=c1, y=-100.0, z=0.0, area=1.0e4),
        FiberPoint(material=c1, y=100.0, z=0.0, area=1.0e4),
    ), GJ=1.0e10)
    integ = ops.beamIntegration.Lobatto(section=fib, n_ip=3)
    tr = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.forceBeamColumn(pg="column", transf=tr, integration=integ)
    el = ops.nDMaterial.ElasticIsotropic(E=30000.0, nu=0.2, rho=0.0)
    pf = ops.nDMaterial.PlateFiber(material=el)
    slab = ops.section.LayeredShell(layers=tuple(
        ShellLayer(material=pf, thickness=50.0) for _ in range(3)
    ))
    ops.element.ASDShellQ4(pg="slab", section=slab)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    return ops


def _static_chain(ops: apeSees, dlam: float, **lc: object) -> dict[str, object]:
    return {
        "test": ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=dlam, **lc),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.RCM(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def _transient_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test": ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.Newmark(gamma=0.5, beta=0.25),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.RCM(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Transient(),
    }


def _three_stages(ops: apeSees) -> None:
    """1D's shape: gravity 0.1 x 10, hold 0.02 x 50, transient 0.0025."""
    with ops.stage(name="gravity") as s:
        s.analysis(**_static_chain(ops, 0.1))
        s.run(n_increments=10)
    with ops.stage(name="hold") as s:
        s.analysis(**_static_chain(ops, 0.02))
        s.run(n_increments=50)
    with ops.stage(name="transient") as s:
        s.analysis(**_transient_chain(ops))
        s.run(n_increments=4, dt=0.0025)


def _tcl(ops: apeSees) -> list[str]:
    em = TclEmitter()
    ops.build().emit(em)
    return em.lines()


def _proc_and_targets(lines: list[str]) -> tuple[tuple[int, int, int], str]:
    i = lines.index(
        "# apeSees IMPL-EX time driver (ADR 0113): persistent dTime / "
        "dTimeCommit / dTimeInitial parameters"
    )
    tags = tuple(int(lines[i + k].split()[1]) for k in (1, 2, 3))
    return tags, "\n".join(lines[i:])  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Declaration and targets (D1, D2)
# ---------------------------------------------------------------------------


def test_modes() -> None:
    assert ImplexTime().mode == "stko"
    assert ImplexTime().drives
    assert not ImplexTime(mode="off").drives
    with pytest.raises(ValueError, match="reserved"):
        ImplexTime(mode="follow")
    with pytest.raises(ValueError, match="mode must be"):
        ImplexTime(mode="sometimes")  # type: ignore[arg-type]


def test_implex_time_returns_and_replaces_the_declaration() -> None:
    ops = _model()
    assert ops.implex_time() == ImplexTime("stko")
    ops.implex_time(mode="off")
    assert ops.build().implex_time == ImplexTime("off")


def test_targets_come_from_the_material_graph() -> None:
    ops = _model()
    specs = implex_element_specs(
        [p for p in ops.build().primitives if hasattr(p, "pg")]
    )
    assert sorted(s.pg for s in specs) == ["column", "wall"]


def test_eta_alone_makes_a_target_and_neither_makes_none() -> None:
    from apeGmsh.opensees.material.uniaxial import ASDConcrete1D

    assert reads_dtime(ASDConcrete1D.from_fc(E=3e4, fc=30.0, eta=1e-4))
    assert not reads_dtime(ASDConcrete1D.from_fc(E=3e4, fc=30.0))
    ops = _model(implex=False, eta=0.0)
    ops.implex_time()
    _three_stages(ops)
    with pytest.raises(BridgeError, match="no element reaches"):
        _tcl(ops)


# ---------------------------------------------------------------------------
# Emission (D3)
# ---------------------------------------------------------------------------


def test_tcl_prelude_and_proc() -> None:
    ops = _model()
    ops.implex_time()
    _three_stages(ops)
    lines = _tcl(ops)
    (p_dt, p_co, p_in), text = _proc_and_targets(lines)
    i = lines.index(f"parameter {p_dt}")
    assert lines[i:i + 18] == [
        f"parameter {p_dt}",
        f"parameter {p_co}",
        f"parameter {p_in}",
        "proc _apesees_implex_dt {dt first} {",
        "    if {$first} {",
        f"        updateParameter {p_co} $dt",
        f"        updateParameter {p_in} $dt",
        "    }",
        f"    updateParameter {p_dt} $dt",
        "}",
        "foreach _apesees_e [list \\",
        "    11 12 21 22] {",
        f"    addToParameter {p_dt} element $_apesees_e dTime",
        f"    addToParameter {p_co} element $_apesees_e dTimeCommit",
        f"    addToParameter {p_in} element $_apesees_e dTimeInitial",
        "}",
        "# === Stage: gravity ===",
        "domainChange",
    ]
    # the elastic slab is not a target; the elements precede the prelude
    assert "31" not in text.split("# === Stage")[0]
    last_element = max(k for k, s in enumerate(lines) if s.startswith("element "))
    assert last_element < i


def test_tcl_one_call_per_stage_right_after_analysis() -> None:
    ops = _model()
    ops.implex_time()
    _three_stages(ops)
    lines = _tcl(ops)
    calls = [(k, s) for k, s in enumerate(lines) if s.startswith("_apesees_implex_dt ")]
    assert [s for _k, s in calls] == [
        "_apesees_implex_dt 0.1 1",
        "_apesees_implex_dt 0.02 1",
        "_apesees_implex_dt 0.0025 1",
    ]
    for k, _s in calls:
        assert lines[k - 1].startswith("analysis ")


def test_target_list_wraps_at_twenty_ids() -> None:
    n = 45
    coords = [(float(i), 0.0, 0.0) for i in range(n + 1)]
    fem = FEMStub(
        nodes=_NodesStub(ids=list(range(1, n + 2)), coords=coords,
                         node_pgs={"Base": [1]}),
        elements=_ElementsStub(elem_pgs={"column": _ElementGroupView(
            ids=tuple(range(101, 101 + n)),
            connectivity=tuple((i, i + 1) for i in range(1, n + 1)),
        )}),
    )
    ops = apeSees(fem, default_orientation=None, element_tags="fem")
    ops.model(ndm=3, ndf=6)
    c1 = ops.uniaxialMaterial.ASDConcrete1D(E=3e4, fc=30.0, implex=True)
    fib = ops.section.Fiber(
        fibers=(FiberPoint(material=c1, y=0.0, z=0.0, area=1.0),), GJ=1.0)
    ops.element.forceBeamColumn(
        pg="column", transf=ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0)),
        integration=ops.beamIntegration.Lobatto(section=fib, n_ip=3))
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.implex_time()
    with ops.stage(name="g") as s:
        s.analysis(**_static_chain(ops, 0.5))
        s.run(n_increments=2)
    lines = _tcl(ops)
    i = lines.index("foreach _apesees_e [list \\")
    assert lines[i + 1] == "    " + " ".join(map(str, range(101, 121))) + " \\"
    assert lines[i + 2] == "    " + " ".join(map(str, range(121, 141))) + " \\"
    assert lines[i + 3] == "    " + " ".join(map(str, range(141, 146))) + "] {"


def test_py_deck_has_the_same_driver() -> None:
    ops = _model()
    ops.implex_time()
    _three_stages(ops)
    em = PyEmitter()
    ops.build().emit(em)
    lines = em.lines()
    i = lines.index("def _apesees_implex_dt(dt, first):")
    p = [int(lines[i - k].split("(")[1].rstrip(")")) for k in (3, 2, 1)]
    assert lines[i:i + 9] == [
        "def _apesees_implex_dt(dt, first):",
        "    if first:",
        f"        ops.updateParameter({p[1]}, dt)",
        f"        ops.updateParameter({p[2]}, dt)",
        f"    ops.updateParameter({p[0]}, dt)",
        "for _apesees_e in [",
        "    11, 12, 21, 22,",
        "]:",
        f"    ops.addToParameter({p[0]}, 'element', _apesees_e, 'dTime')",
    ]
    assert [s for s in lines if s.startswith("_apesees_implex_dt(")] == [
        "_apesees_implex_dt(0.1, True)",
        "_apesees_implex_dt(0.02, True)",
        "_apesees_implex_dt(0.0025, True)",
    ]
    compile("\n".join(lines), "<deck>", "exec")


def test_recording_order_declare_targets_then_stage_calls() -> None:
    ops = _model()
    ops.implex_time()
    _three_stages(ops)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    d = names.index("implex_time_declare")
    t = names.index("implex_time_targets")
    assert d < t < names.index("stage_open")
    assert rec.calls[t][1][1] == WALL + COLUMN
    ups = [c for c in rec.calls if c[0] == "implex_time_update"]
    assert [(c[1][0], c[2]["first"]) for c in ups] == [
        (0.1, True), (0.02, True), (0.0025, True),
    ]


def test_undeclared_and_off_emit_nothing() -> None:
    for mode in (None, "off"):
        ops = _model()
        if mode is not None:
            ops.implex_time(mode="off")
        _three_stages(ops)
        lines = _tcl(ops)
        assert not any("_apesees_implex_dt" in s for s in lines)
        assert not any(s.startswith("parameter ") for s in lines)


def test_h5_archive_refuses_the_driver(tmp_path) -> None:
    ops = _model()
    ops.implex_time()
    _three_stages(ops)
    with pytest.raises(NotImplementedError, match="ADR 0113"):
        ops.h5(str(tmp_path / "model.h5"))


# ---------------------------------------------------------------------------
# Partitioned decks: each rank attaches only what it owns (D3, D4)
# ---------------------------------------------------------------------------


def _partitioned_fem() -> FEMStub:
    fem = _fem()
    fem.set_partitions([
        (1, [1, 2, 4, 5, 10, 13], [11, 21, 31]),
        (2, [2, 3, 5, 6, 7, 8, 9], [12, 22]),
    ])
    return fem


def test_partitioned_targets_are_per_rank() -> None:
    ops = _model(_partitioned_fem())
    ops.implex_time()
    _three_stages(ops)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    calls = rec.calls
    d = [k for k, c in enumerate(calls) if c[0] == "implex_time_declare"]
    assert len(d) == 1
    owned: dict[int, tuple[int, ...]] = {}
    rank = None
    for name, args, _kw in calls[d[0]:]:
        if name == "partition_open":
            rank = args[0]
        elif name == "partition_close":
            rank = None
        elif name == "implex_time_targets":
            assert rank is not None, "targets must be inside a rank block"
            owned[rank] = args[1]
        elif name == "stage_open":
            break
    assert owned == {0: (11, 21), 1: (12, 22)}
    ups = [c for c in calls if c[0] == "implex_time_update"]
    assert len(ups) == 3


# ---------------------------------------------------------------------------
# Refusals (D3, D4, D6)
# ---------------------------------------------------------------------------


def test_refuses_an_unstaged_model() -> None:
    ops = _model()
    ops.implex_time()
    with pytest.raises(BridgeError, match="staged decks only"):
        _tcl(ops)


def test_refuses_a_variable_increment_stage() -> None:
    ops = _model()
    ops.implex_time()
    with ops.stage(name="g") as s:
        s.analysis(**_static_chain(ops, 0.1, num_iter=5, min_lam=0.01,
                                   max_lam=0.2))
        s.run(n_increments=10)
    with pytest.raises(BridgeError, match="num_iter"):
        _tcl(ops)
    ops = _model()
    ops.implex_time()
    with ops.stage(name="vt") as s:
        chain = _transient_chain(ops)
        chain["analysis"] = ops.analysis.VariableTransient()
        s.analysis(**chain)
        s.run(n_increments=10, dt=0.01)
    with pytest.raises(BridgeError, match="one known number"):
        _tcl(ops)


def test_refuses_a_stage_activated_target() -> None:
    ops = _model()
    ops.implex_time()
    with ops.stage(name="g") as s:
        s.activate(pgs=["wall"])
        s.analysis(**_static_chain(ops, 0.1))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="activates IMPL-EX target"):
        _tcl(ops)


def test_refuses_removing_a_target() -> None:
    ops = _model()
    ops.implex_time()
    with ops.stage(name="g") as s:
        s.analysis(**_static_chain(ops, 0.1))
        s.run(n_increments=1)
    with ops.stage(name="cut") as s:
        s.remove_element(elements=[12])
        s.analysis(**_static_chain(ops, 0.1))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="removes IMPL-EX target"):
        _tcl(ops)


def test_removing_a_non_target_is_fine() -> None:
    ops = _model()
    ops.implex_time()
    with ops.stage(name="g") as s:
        s.analysis(**_static_chain(ops, 0.1))
        s.run(n_increments=1)
    with ops.stage(name="cut") as s:
        s.remove_element(elements=[31])
        s.analysis(**_static_chain(ops, 0.1))
        s.run(n_increments=1)
    assert any(s.startswith("_apesees_implex_dt") for s in _tcl(ops))


def test_refuses_two_owners_of_dtime() -> None:
    ops = _model()
    ops.implex_time()
    with ops.stage(name="g") as s:
        s.update_parameter("dTime", 0.1, pg="wall")
        s.analysis(**_static_chain(ops, 0.1))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="one owner"):
        _tcl(ops)


def test_off_refuses_any_dtime_write() -> None:
    ops = _model()
    ops.implex_time(mode="off")
    with ops.stage(name="g") as s:
        s.update_parameter("dTimeCommit", 0.1, pg="wall")
        s.analysis(**_static_chain(ops, 0.1))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="mode='off'"):
        _tcl(ops)


def _reset_stage(ops: apeSees, name: str, dlam: float, value: float,
                 names: tuple[str, ...] = ("dTimeCommit", "dTimeInitial", "dTime"),
                 ) -> None:
    with ops.stage(name=name) as s:
        for n in names:
            s.update_parameter(n, value, elements=list(WALL + COLUMN))
        s.analysis(**_static_chain(ops, dlam))
        s.run(n_increments=2)


def test_trap_a_stage_without_dtime_after_a_writer() -> None:
    """The translator's rule C13 reset followed by a stage that writes
    nothing: its materials keep the previous stage's dTime."""
    ops = _model()
    _reset_stage(ops, "gravity", 0.1, 0.1)
    with ops.stage(name="transient") as s:
        s.analysis(**_transient_chain(ops))
        s.run(n_increments=4, dt=0.0025)
    with pytest.raises(BridgeError, match="dTime trap.*'transient'"):
        _tcl(ops)


def test_trap_a_dtime_that_is_not_the_increment() -> None:
    ops = _model()
    _reset_stage(ops, "gravity", 0.1, 0.02)
    with pytest.raises(BridgeError, match="writes dTime = 0.02 but steps"):
        _tcl(ops)


def test_trap_a_partial_reset_counts_as_a_writer() -> None:
    ops = _model()
    _reset_stage(ops, "gravity", 0.1, 0.1, names=("dTimeCommit",))
    with pytest.raises(BridgeError, match="does not write dTime"):
        _tcl(ops)


def test_a_reset_in_every_stage_passes() -> None:
    """The translator's C13 shape: every static stage resets all three to
    its own increment -- correct for constant-increment stages."""
    ops = _model()
    with ops.stage(name="plain") as s:
        s.analysis(**_static_chain(ops, 0.5))
        s.run(n_increments=2)
    _reset_stage(ops, "gravity", 0.1, 0.1)
    _reset_stage(ops, "hold", 0.02, 0.02)
    lines = _tcl(ops)
    assert sum(s.startswith("updateParameter ") for s in lines) == 6


# ---------------------------------------------------------------------------
# Golden: the San Ramon deck-patch driver vs ours, replayed in plain Tcl
# ---------------------------------------------------------------------------

# Verbatim output of ``sanramon.staging.prelude_lines(targets, tags)``
# (research repo ``models/apegmsh/sanramon/staging.py``, the driver G20
# proved bitwise identical to STKO's) for ``targets = (11, 12, 21, 22)``,
# ``tags = (910001, 910002, 910003)``.
SANRAMON_PRELUDE = r"""
set sr_implex_all [list \
    11 12 21 22]
set sr_implex_local {}
foreach _sr_e [getEleTags] { set _sr_local($_sr_e) 1 }
foreach _sr_e $sr_implex_all { if {[info exists _sr_local($_sr_e)]} { lappend sr_implex_local $_sr_e } }
array unset _sr_local
parameter 910001
parameter 910002
parameter 910003
foreach _sr_e $sr_implex_local {
    addToParameter 910001 element $_sr_e dTime
    addToParameter 910002 element $_sr_e dTimeCommit
    addToParameter 910003 element $_sr_e dTimeInitial
}
proc sr_implex_dt {dt first} {
    if {$first} {
        updateParameter 910002 $dt
        updateParameter 910003 $dt
    }
    updateParameter 910001 $dt
}
"""

_STUBS = r"""
set ::log {}
proc parameter {args} { lappend ::log [concat parameter $args] }
proc addToParameter {args} { lappend ::log [concat addToParameter $args] }
proc updateParameter {args} { lappend ::log [concat updateParameter $args] }
proc puts {args} {}
"""


def _replay(script: str, *, ele_tags: list[int], pid: int,
            calls: list[tuple[str, float, int]]) -> list[str]:
    tkinter = pytest.importorskip("tkinter")
    try:
        tcl = tkinter.Tcl()
    except Exception as exc:  # pragma: no cover - no Tcl runtime
        pytest.skip(f"no Tcl interpreter: {exc}")
    tcl.eval(_STUBS)
    tcl.eval(f"proc getEleTags {{}} {{ return {{{' '.join(map(str, ele_tags))}}} }}")
    tcl.eval(f"proc getPID {{}} {{ return {pid} }}")
    tcl.eval(script)
    for proc, dt, first in calls:
        tcl.eval(f"{proc} {dt!r} {first}")
    return list(tcl.splitlist(tcl.eval("set ::log")))


def _driver_block(lines: list[str]) -> str:
    """Our prelude: from the driver comment to the first stage banner,
    with the bridge's tags renamed to the San Ramon ones."""
    (p_dt, p_co, p_in), _ = _proc_and_targets(lines)
    i = lines.index(
        "# apeSees IMPL-EX time driver (ADR 0113): persistent dTime / "
        "dTimeCommit / dTimeInitial parameters"
    )
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("# === Stage"))
    rename = {p_dt: 910001, p_co: 910002, p_in: 910003}
    return re.sub(
        r"\b(parameter|addToParameter|updateParameter) (\d+)\b",
        lambda m: f"{m.group(1)} {rename[int(m.group(2))]}",
        "\n".join(lines[i:j]),
    )


def _norm(log: list[str]) -> list[str]:
    return [" ".join(s.split()) for s in log]


def test_golden_serial_replay_equals_the_sanramon_driver() -> None:
    ops = _model()
    ops.implex_time()
    _three_stages(ops)
    ours = _driver_block(_tcl(ops))
    seq = [(0.1, 1), (0.02, 1), (0.0025, 1), (0.00125, 0)]
    theirs_log = _replay(
        SANRAMON_PRELUDE, ele_tags=[11, 12, 21, 22, 31], pid=0,
        calls=[("sr_implex_dt", dt, f) for dt, f in seq])
    ours_log = _replay(
        ours, ele_tags=[11, 12, 21, 22, 31], pid=0,
        calls=[("_apesees_implex_dt", dt, f) for dt, f in seq])
    assert _norm(ours_log) == _norm(theirs_log)
    assert len(theirs_log) == 3 + 12 + 3 * 3 + 1


@pytest.mark.parametrize("pid,local", [(0, [11, 21, 31]), (1, [12, 22])])
def test_golden_per_rank_replay_equals_the_getEleTags_intersection(
    pid: int, local: list[int],
) -> None:
    """Rank K of our partitioned deck attaches exactly what the San Ramon
    prelude attaches when ``getEleTags`` returns rank K's domain."""
    ops = _model(_partitioned_fem())
    ops.implex_time()
    _three_stages(ops)
    ours = _driver_block(_tcl(ops))
    seq = [(0.1, 1), (0.0025, 1)]
    theirs_log = _replay(
        SANRAMON_PRELUDE, ele_tags=local, pid=pid,
        calls=[("sr_implex_dt", dt, f) for dt, f in seq])
    ours_log = _replay(
        ours, ele_tags=local, pid=pid,
        calls=[("_apesees_implex_dt", dt, f) for dt, f in seq])
    assert _norm(ours_log) == _norm(theirs_log)
