"""ADR 0104 — the ``Substep`` rung (ADR 0057 Phase B) contract.

Pins, per the ADR:

1. **Declaration** — the policy validates loud (floor above base, a
   zero wall budget, a half-declared plateau, two Substep rungs in one
   ladder), and a ``Substep``-carrying ladder REFUSES ``to_spec``
   rather than emitting a deck that ignores it.
2. **The controller rescues what a fixed step cannot** — a synthetic
   stiff-then-soft driver whose convergence radius collapses past the
   knee: the fixed-step run fails, the same driver under ``Substep``
   reaches the target.
3. **A spent budget is a FAILURE verdict** — never dressed up as
   success, even though the run committed real steps (#587 floor).
4. **Plateau** — a flattened curve stops the run on the tail-slope
   criterion, before either budget could be spent.
5. **Degenerate inputs** (the adversarial probe, ADR 0104 §Probe) —
   zero budget, target already met, wall budget spent, a driver that
   never converges, and a driver that converges without moving the
   declared control DOF.

Every test drives a **fake** ``SubstepDriver``: the controller is
step-size policy over an ``analyze`` return code, so a live OpenSees
backend would test the backend, not this.  The clock is injected for
the same reason.
"""
from __future__ import annotations

import math

import pytest

from apeGmsh.opensees.analysis.algorithm import KrylovNewton, Newton
from apeGmsh.opensees.analysis.strategy import (
    Ladder,
    OpenSeesPyDriver,
    Substep,
    SubstepResult,
)


# ---------------------------------------------------------------------------
# Fake drivers
# ---------------------------------------------------------------------------


class BilinearDriver:
    """Stiff-then-soft push with a step-size-dependent convergence radius.

    Below ``knee`` the response is stiff (slope ``k0``) and any step
    converges.  Past the knee it softens to ``k1`` and the driver only
    converges for ``ds <= radius`` — the synthetic stand-in for a
    plastic model whose Newton radius collapses at yield.  ``load`` is
    the accumulated force, so ``(disp, load)`` is a genuine curve with
    an initial tangent and a soft tail.
    """

    def __init__(
        self, *, knee: float = 1.0, k0: float = 100.0, k1: float = 1.0,
        radius: float = 0.125,
    ) -> None:
        self.knee, self.k0, self.k1, self.radius = knee, k0, k1, radius
        self.u = 0.0
        self.q = 0.0
        self.attempts = 0

    def analyze(self, ds: float) -> int:
        self.attempts += 1
        if self.u >= self.knee and ds > self.radius:
            return -3
        k = self.k0 if self.u < self.knee else self.k1
        self.u += ds
        self.q += k * ds
        return 0

    def disp(self, node: int, dof: int) -> float:
        return self.u

    def load(self) -> float:
        return self.q


class NeverConvergesDriver:
    """Every increment fails, at any step size."""

    def __init__(self) -> None:
        self.attempts = 0

    def analyze(self, ds: float) -> int:
        self.attempts += 1
        return -3

    def disp(self, node: int, dof: int) -> float:
        return 0.0

    def load(self) -> float:
        return 0.0


class FlatteningDriver:
    """A curve that hardens, then goes flat — a genuine plateau.

    The tangent is ``k0`` until ``knee`` and ``k0 * flat`` after it;
    every increment converges, so nothing but the criterion can stop
    the run before the (deliberately distant) target.
    """

    def __init__(
        self, *, knee: float = 1.0, k0: float = 100.0, flat: float = 0.001,
    ) -> None:
        self.knee, self.k0, self.flat = knee, k0, flat
        self.u = 0.0
        self.q = 0.0

    def analyze(self, ds: float) -> int:
        k = self.k0 if self.u < self.knee else self.k0 * self.flat
        self.u += ds
        self.q += k * ds
        return 0

    def disp(self, node: int, dof: int) -> float:
        return self.u

    def load(self) -> float:
        return self.q


class FrozenDofDriver:
    """Converges every increment but never moves the measured DOF."""

    def analyze(self, ds: float) -> int:
        return 0

    def disp(self, node: int, dof: int) -> float:
        return 0.0

    def load(self) -> float:
        return 0.0


class _Clock:
    """Deterministic monotonic clock: one tick per read."""

    def __init__(self, step: float = 1.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        t = self.t
        self.t += self.step
        return t


def _fixed_step(driver: BilinearDriver, *, ds: float, target: float) -> bool:
    """The un-adapted loop: one step size, first failure ends it."""
    while driver.u < target - 1e-12:
        if driver.analyze(min(ds, target - driver.u)) != 0:
            return False
    return True


# ---------------------------------------------------------------------------
# 1. Declaration
# ---------------------------------------------------------------------------


def _policy(**over: object) -> Substep:
    kw: dict[str, object] = dict(
        node=7, dof=3, target=2.0, ds=0.5, ds_min=1e-3,
    )
    kw.update(over)
    return Substep(**kw)  # type: ignore[arg-type]


def test_floor_above_base_step_is_refused() -> None:
    with pytest.raises(ValueError, match="above the base step"):
        _policy(ds=1e-4, ds_min=1e-2)


def test_cap_below_base_step_is_refused() -> None:
    with pytest.raises(ValueError, match="below the base step"):
        _policy(ds_max=0.1)


def test_cap_defaults_to_the_base_step() -> None:
    # ADR 0057 §2: the step never regrows above the nominal.
    assert _policy().ds_max == 0.5


def test_zero_wall_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="wall_budget must be > 0"):
        _policy(wall_budget=0.0)


def test_half_declared_plateau_is_refused() -> None:
    with pytest.raises(ValueError, match="declared\n?\\s*together"):
        _policy(plateau=0.05)
    with pytest.raises(ValueError, match="declared\n?\\s*together"):
        _policy(plateau_window=0.5)


def test_plateau_epsilon_must_be_a_fraction() -> None:
    with pytest.raises(ValueError, match="fraction in"):
        _policy(plateau=1.5, plateau_window=0.5)


def test_non_positive_magnitudes_are_refused() -> None:
    for field, value in (("target", 0.0), ("ds", -1.0), ("ds_min", 0.0)):
        with pytest.raises(ValueError, match="must be > 0"):
            _policy(**{field: value})


def test_ladder_accepts_one_substep_rung() -> None:
    lad = Ladder(rungs=(Newton(), _policy(), KrylovNewton()), name="push")
    assert lad.substep is _policy() or lad.substep == _policy()


def test_ladder_refuses_two_substep_rungs() -> None:
    with pytest.raises(ValueError, match="at most one Substep"):
        Ladder(rungs=(_policy(), _policy()), name="x")


def test_ladder_without_substep_reports_none() -> None:
    assert Ladder(rungs=(Newton(),), name="x").substep is None


def test_substep_ladder_refuses_to_spec() -> None:
    # Emitting the algorithm rungs alone would ship a deck that
    # silently ignores the declared step policy (ADR 0104 D6).
    lad = Ladder(rungs=(_policy(), KrylovNewton()), name="push")
    with pytest.raises(ValueError, match="not emitted into decks yet"):
        lad.to_spec(base=Newton())


# ---------------------------------------------------------------------------
# 2. The controller rescues what a fixed step cannot
# ---------------------------------------------------------------------------


def test_fixed_step_fails_where_substep_reaches_the_target() -> None:
    # Same driver, same base step: past the knee only ds <= 0.125
    # converges, so the fixed-step loop dies there.
    assert not _fixed_step(BilinearDriver(), ds=0.5, target=2.0)

    driver = BilinearDriver()
    res = _policy(budget=8).drive(driver)

    assert res.verdict == "target"
    assert res.ok
    assert res.subdivisions >= 2          # 0.5 -> 0.25 -> 0.125
    assert driver.u == pytest.approx(2.0)
    assert res.disp == pytest.approx(2.0)


def test_substep_regrows_after_consecutive_good_steps() -> None:
    driver = BilinearDriver(knee=1e9)         # never softens, never fails
    res = _policy(ds=0.5, ds_max=1.0, target=8.0).drive(driver)
    assert res.verdict == "target"
    assert res.ds_final == pytest.approx(1.0)  # grew to the cap and stopped


def test_the_regrow_probe_does_not_spend_the_budget() -> None:
    # ADR 0104 F1: regrow doubles the step every `regrow_after` good
    # steps, so past the knee it re-fails BY CONSTRUCTION.  A run-wide
    # subdivision total is therefore spent by probing rather than by
    # the model failing — the budget is per-increment DEPTH (ADR 0057
    # §2 `max_halvings`), and this run proves the difference: it
    # halves far more than `budget` times in total and still lands.
    driver = BilinearDriver(radius=0.125)
    res = _policy(ds_max=1.0, target=8.0, budget=4).drive(driver)
    assert res.verdict == "target"
    assert res.subdivisions > 4


def test_substep_lands_exactly_on_the_target() -> None:
    # The last step is whatever is left — never an overshoot (the
    # loadConst contract, ADR 0057 §2 evidence point 4).
    driver = BilinearDriver(knee=100.0)      # never softens
    res = _policy(ds=0.3, target=1.0).drive(driver)
    assert res.verdict == "target"
    assert driver.u == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Budget exhaustion is a FAILURE verdict
# ---------------------------------------------------------------------------


def test_budget_exhaustion_is_a_failure_not_a_success() -> None:
    # A run that committed real steps before the knee, then spent its
    # whole budget: the verdict is 'budget' and ok is False.
    driver = BilinearDriver(radius=1e-9)
    res = _policy(target=4.0, ds_min=1e-12, budget=3).drive(driver)

    assert res.verdict == "budget"
    assert not res.ok
    assert res.subdivisions == 4          # budget + 1 == the one over
    assert res.steps > 0                  # it DID commit steps
    assert "budget" in res.reason


def test_floor_is_a_failure_verdict() -> None:
    driver = BilinearDriver(radius=1e-9)
    res = _policy(target=4.0, ds_min=0.05, budget=100).drive(driver)
    assert res.verdict == "floor"
    assert not res.ok
    assert res.ds_final < 0.05


def test_never_converging_driver_spends_the_budget() -> None:
    driver = NeverConvergesDriver()
    res = _policy(ds_min=1e-12, budget=4).drive(driver)
    assert res.verdict == "budget"
    assert not res.ok
    assert res.steps == 0
    assert driver.attempts == 5           # budget + 1 attempts


def test_zero_budget_refuses_the_first_subdivision() -> None:
    driver = NeverConvergesDriver()
    res = _policy(budget=0).drive(driver)
    assert res.verdict == "budget"
    assert not res.ok
    assert driver.attempts == 1


# ---------------------------------------------------------------------------
# 4. The plateau criterion
# ---------------------------------------------------------------------------


def test_plateau_stops_a_flattened_run() -> None:
    driver = FlatteningDriver(knee=1.0)
    res = _policy(
        target=100.0, ds=0.1, plateau=0.05, plateau_window=0.5,
    ).drive(driver)

    assert res.verdict == "plateau"
    assert res.ok
    assert driver.u < 100.0               # it stopped well short
    assert "tail slope" in res.reason


def test_plateau_does_not_fire_on_a_hardening_run() -> None:
    driver = FlatteningDriver(knee=1e9)   # never flattens
    res = _policy(
        target=2.0, ds=0.1, plateau=0.05, plateau_window=0.5,
    ).drive(driver)
    assert res.verdict == "target"


def test_plateau_needs_the_window_covered() -> None:
    # A window wider than the whole run cannot show a sustained
    # tangent either way, so the run must reach its target instead.
    driver = FlatteningDriver(knee=0.0)   # flat from the first step
    res = _policy(
        target=1.0, ds=0.1, plateau=0.5, plateau_window=10.0,
    ).drive(driver)
    assert res.verdict == "target"


def test_plateau_undeclared_never_fires() -> None:
    driver = FlatteningDriver(knee=0.0)
    res = _policy(target=1.0, ds=0.1).drive(driver)
    assert res.verdict == "target"


# ---------------------------------------------------------------------------
# 5. Degenerate inputs (the adversarial probe)
# ---------------------------------------------------------------------------


def test_the_datum_is_relative_to_the_state_at_entry() -> None:
    # A model that arrives already displaced (a settled gravity state)
    # measures ITS OWN advance: `target` is not an absolute reading.
    driver = BilinearDriver(knee=1e9)
    driver.u = 5.0
    res = _policy(target=2.0).drive(driver)
    assert res.verdict == "target"
    assert res.steps > 0
    assert driver.u == pytest.approx(7.0)


def test_target_already_met_takes_no_step() -> None:
    driver = BilinearDriver()
    res = _policy(target=1e-300).drive(driver)
    assert res.verdict == "target"
    assert res.ok
    assert res.steps == 0
    assert driver.attempts == 0


def test_wall_budget_ends_the_run_as_a_failure() -> None:
    driver = BilinearDriver(knee=1e9)
    res = _policy(target=1e6, wall_budget=5.0).drive(
        driver, clock=_Clock(step=1.0),
    )
    assert res.verdict == "wall"
    assert not res.ok
    assert "wall-clock budget" in res.reason


def test_wall_budget_does_not_pre_empt_a_met_criterion() -> None:
    # Success criteria are checked BEFORE the wall guard: the target is
    # reached on the tick that would otherwise have declared the wall.
    driver = BilinearDriver(knee=1e9)
    res = _policy(target=1.0, ds=1.0, wall_budget=1.5).drive(
        driver, clock=_Clock(step=1.0),
    )
    assert res.verdict == "target"
    assert res.ok
    assert res.wall > 1.5                  # the wall WAS past, and lost


def test_frozen_control_dof_is_refused_at_the_first_step() -> None:
    with pytest.raises(ValueError, match="did not advance node 7 dof 3"):
        _policy().drive(FrozenDofDriver())


def test_nan_displacement_is_refused_not_spun_on() -> None:
    # ADR 0104 F2: a NaN control reading fails every comparison in the
    # loop — target, floor and the step-1 tripwire alike — so without
    # this guard the controller spins forever on a blown-up model.
    class NaNDispDriver(BilinearDriver):
        def disp(self, node: int, dof: int) -> float:
            return math.nan

    with pytest.raises(ValueError, match="non-finite displacement"):
        _policy().drive(NaNDispDriver())


def test_nan_load_under_a_declared_plateau_is_refused() -> None:
    # ADR 0104 F3: a NaN load fits a NaN initial tangent, which
    # silently DISABLES the declared criterion.
    class NaNLoadDriver(BilinearDriver):
        def load(self) -> float:
            return math.nan

    policy = _policy(plateau=0.05, plateau_window=0.5, target=10.0)
    with pytest.raises(ValueError, match="load reading is non-finite"):
        policy.drive(NaNLoadDriver(knee=1e9))

    # ...and it is scoped: with no plateau declared the same driver
    # runs, because nothing reads the load.
    res = _policy(target=1.0).drive(NaNLoadDriver(knee=1e9))
    assert res.verdict == "target"


# ---------------------------------------------------------------------------
# 6. Result shape
# ---------------------------------------------------------------------------


def test_result_curve_is_the_committed_rows() -> None:
    driver = BilinearDriver(knee=1e9)
    res = _policy(target=1.0, ds=0.25).drive(driver)
    assert isinstance(res, SubstepResult)
    assert len(res.curve) == res.steps == 4
    assert res.curve[-1][0] == pytest.approx(1.0)
    assert res.curve[-1][1] == pytest.approx(100.0)


def test_provenance_is_printed_loud(capsys: pytest.CaptureFixture[str]) -> None:
    # ADR 0057 §6: no silent success, and no silent subdivision.
    _policy(budget=8).drive(BilinearDriver())
    out = capsys.readouterr().out
    assert "apeGmsh substep: subdivision 1/8" in out
    assert "apeGmsh substep: TARGET after" in out


# ---------------------------------------------------------------------------
# 7. The OpenSeesPyDriver adapter (ADR 0104 D1)
# ---------------------------------------------------------------------------


class StubOps:
    """A stand-in for the openseespy module: no live backend.

    Records every ``integrator`` line so a test can see the step size
    the controller pushed in, and fails any increment above ``radius``
    once the DOF is past ``knee`` — the same stiff-then-soft shape the
    fixed-vs-substep test uses, driven through the real adapter.
    """

    def __init__(self, *, knee: float = 1e9, radius: float = 1e9) -> None:
        self.knee, self.radius = knee, radius
        self.u = 0.0
        self.t = 0.0
        self.integrators: list[tuple[object, ...]] = []
        self.steps = 0
        self._incr = 0.0

    def integrator(self, *args: object) -> None:
        self.integrators.append(args)
        self._incr = float(args[3])  # type: ignore[arg-type]

    def analyze(self, n: int) -> int:
        assert n == 1, "the driver must take exactly one step per attempt"
        if abs(self.u) >= self.knee and abs(self._incr) > self.radius:
            return -3
        self.u += self._incr
        self.t += abs(self._incr) * 100.0
        self.steps += 1
        return 0

    def nodeDisp(self, node: int, dof: int) -> float:
        return self.u

    def getTime(self) -> float:
        return self.t


def test_adapter_issues_displacement_control_then_one_step() -> None:
    ops = StubOps()
    drv = OpenSeesPyDriver(ops, node=7, dof=3)
    assert drv.analyze(0.25) == 0
    assert ops.integrators == [("DisplacementControl", 7, 3, -0.25)]
    assert ops.steps == 1


def test_adapter_forwards_the_rc_unchanged() -> None:
    ops = StubOps(knee=0.0, radius=0.0)     # every increment fails
    assert OpenSeesPyDriver(ops, node=7, dof=3).analyze(0.25) == -3


def test_adapter_reads_disp_and_load_from_the_module() -> None:
    ops = StubOps()
    ops.u, ops.t = -1.5, 42.0
    drv = OpenSeesPyDriver(ops, node=7, dof=3)
    assert drv.disp(7, 3) == -1.5
    assert drv.load() == 42.0


def test_adapter_sign_drives_the_declared_direction() -> None:
    ops = StubOps()
    OpenSeesPyDriver(ops, node=7, dof=3, sign=1.0).analyze(0.25)
    assert ops.integrators[-1] == ("DisplacementControl", 7, 3, 0.25)
    assert ops.u == pytest.approx(0.25)


def test_adapter_refuses_a_sign_that_is_not_a_direction() -> None:
    with pytest.raises(ValueError, match="sign must be -1.0 or 1.0"):
        OpenSeesPyDriver(StubOps(), node=7, dof=3, sign=0.5)


def test_adapter_plugs_into_drive_end_to_end() -> None:
    # The whole loop through the real adapter: stiff to |u| = 1.0, then
    # only |ds| <= 0.125 converges.  The controller must subdivide,
    # land on the target, and every integrator line must be the
    # DisplacementControl one it pushed the current step into.
    ops = StubOps(knee=1.0, radius=0.125)
    res = _policy(target=2.0, ds=0.5, budget=8).drive(
        OpenSeesPyDriver(ops, node=7, dof=3),
    )

    assert res.verdict == "target"
    assert res.ok
    assert res.subdivisions >= 2
    assert ops.u == pytest.approx(-2.0)          # driven NEGATIVE...
    assert res.disp == pytest.approx(2.0)        # ...measured as advance
    assert all(a[0] == "DisplacementControl" for a in ops.integrators)
    assert len(ops.integrators) == ops.steps + res.subdivisions
    # the rescue step is on the record, halved twice from the base
    assert any(a[3] == pytest.approx(-0.125) for a in ops.integrators)
