"""#2498: a build that rejects a fork option, told apart from one that has it.

The post-pin gate (``tests/opensees/warrant_rows.py``,
:func:`require_fork_features`) skips a fork-lane test only when the live
build *itself* prints that it lacks the feature: ``unknown option '-shear'``
from ``LadrunoIMKBeam``, ``element type LadrunoForceBeamColumn is unknown``.
These tests pin it with stub engines, without importing openseespy:

* the build's rejection line is read from ``sys.stderr`` and from fd 2;
* an accepted declaration, a failure for any other reason, or silence is
  no evidence of absence, so the test runs (a skip never hides a failure on
  a build that has the feature);
* the patterns do not cross (``-shearHinge`` is not ``-shear``);
* a post-pin skip stays a skip in the fork lane only for a warrant row that
  names the feature in ``post_pin``, and its member never turns green;
* every gated test asks for its features by name (AST);
* every gate expires with the ``FORK_PIN`` sha it predates: a bumped pin
  fails the lock, naming each stale gate;
* the live emitter's fork-element gate reads the same lines: a build that
  knows the element but rejects an option raises "option X is not
  supported by this build", never "requires the Ladruno fork build", which
  stays for an unknown class (or output this process cannot read).
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import apply_warrant_rule
from tests.opensees import warrant_rows as wr

ROOT = Path(__file__).resolve().parents[3]
WP200 = ROOT / "tests/opensees/integration_ladruno/test_wp200_ladruno_beams_live.py"
FBC_ROW = next(r for r in wr.ROWS if r.member == "LadrunoForceBeamColumn")

#: The lines ladruno-v20261017 printed on live-fork run 39773542334.
PIN_LINES = {
    "LadrunoIMKBeam -shear": "WARNING LadrunoIMKBeam (2D) -- unknown option '-shear'\n",
    "LadrunoIMKBeam -shearHinge":
        "WARNING LadrunoIMKBeam (2D) -- unknown option '-shearHinge'\n",
    "LadrunoCohesiveHinge -implex":
        "WARNING LadrunoCohesiveHinge tag 1: unknown option '-implex' (ignored)\n",
    "LadrunoForceBeamColumn": "WARNING element type LadrunoForceBeamColumn is unknown\n",
}


class StubOps:
    """An engine whose declaring calls print ``line`` (to ``sys.stderr`` or
    fd 2) and then raise or return, as a real build would."""

    def __init__(self, line: str = "", *, fd: bool = False,
                 raises: bool = True) -> None:
        self.line, self.fd, self.raises = line, fd, raises
        self.wipes = 0

    def wipe(self) -> None:
        self.wipes += 1

    def _declare(self, *_args: Any) -> None:
        if self.line:
            if self.fd:
                os.write(2, self.line.encode())
            else:
                sys.stderr.write(self.line)
        if self.raises and self.line:
            raise RuntimeError("See stderr output")

    element = uniaxialMaterial = _declare

    def __getattr__(self, name: str) -> Any:  # model, node, geomTransf, ...
        return lambda *a, **k: None


@pytest.mark.parametrize("name", sorted(wr.POST_PIN_FEATURES))
@pytest.mark.parametrize("fd", [False, True], ids=["sys.stderr", "fd2"])
def test_the_builds_rejection_line_is_absence(name: str, fd: bool) -> None:
    ops = StubOps(PIN_LINES[name], fd=fd)
    line = wr.probe_absence(ops, name)
    assert line is not None and line in PIN_LINES[name]
    assert ops.wipes == 2, "the probe wipes before and after its declaration"


@pytest.mark.parametrize("name", sorted(wr.POST_PIN_FEATURES))
def test_a_build_that_has_the_feature_runs_the_test(name: str) -> None:
    # Accepted silently, or refused for another reason (a wrong-arity probe
    # on a build that has the option): neither is evidence of absence.
    assert wr.probe_absence(StubOps(), name) is None
    other = StubOps("WARNING insufficient arguments for LadrunoIMKBeam\n")
    assert wr.probe_absence(other, name) is None
    unknown_other = StubOps("WARNING element type LadrunoFooBeam is unknown\n")
    assert wr.probe_absence(unknown_other, name) is None


def test_the_patterns_do_not_cross() -> None:
    for name in wr.POST_PIN_FEATURES:
        for other, line in PIN_LINES.items():
            got = wr.probe_absence(StubOps(line), name)
            assert (got is not None) == (other == name), (name, other)


def test_the_probe_restores_the_streams(capfd: pytest.CaptureFixture[str]) -> None:
    before = sys.stdout, sys.stderr
    wr.probe_absence(StubOps(PIN_LINES["LadrunoIMKBeam -shear"], fd=True),
                     "LadrunoIMKBeam -shear")
    assert (sys.stdout, sys.stderr) == before
    os.write(2, b"fd2-after-probe\n")
    print("stdout-after-probe")
    out, err = capfd.readouterr()
    assert "fd2-after-probe" in err and "stdout-after-probe" in out
    assert "unknown option" not in err, "the probe's own line stays captured"


def test_an_unknown_feature_fails_closed() -> None:
    with pytest.raises(KeyError, match="not a post-pin feature"):
        wr.probe_absence(StubOps(), "LadrunoIMKBeam -shaer")
    with pytest.raises(ValueError, match="at least one"):
        wr.require_fork_features()


def _gate(monkeypatch: pytest.MonkeyPatch, ops: StubOps) -> None:
    from apeGmsh.opensees.emitter import live

    monkeypatch.setattr(wr, "_ABSENCE_CACHE", {})
    monkeypatch.setattr(live, "get_ops", lambda: ops)


def test_the_gate_skips_naming_the_missing_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _gate(monkeypatch, StubOps(PIN_LINES["LadrunoIMKBeam -shearHinge"]))
    with pytest.raises(pytest.skip.Exception) as info:
        wr.require_fork_features("LadrunoIMKBeam -shear", "LadrunoIMKBeam -shearHinge")
    reason = str(info.value)
    assert reason.startswith(wr.POST_PIN_SKIP_PREFIX)
    assert wr.post_pin_missing(reason) == ("LadrunoIMKBeam -shearHinge",)
    assert "unknown option '-shearHinge'" in reason


def test_the_gate_lets_a_build_with_the_feature_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _gate(monkeypatch, StubOps())
    assert wr.require_fork_features(*wr.POST_PIN_FEATURES) is None


def _skip_report(nodeid: str, reason: str) -> pytest.TestReport:
    return pytest.TestReport(
        nodeid=nodeid, location=("x", 0, "x"), keywords={},
        outcome="skipped", longrepr=("x", 0, f"Skipped: {reason}"),
        when="call",
    )


def _item(*marks: str) -> SimpleNamespace:
    return SimpleNamespace(iter_markers=lambda: [SimpleNamespace(name=m) for m in marks])


def test_a_post_pin_skip_counts_for_the_lane_never_for_the_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(wr.LANE_ENV, "fork")
    reason = wr.post_pin_reason(
        {"LadrunoForceBeamColumn": PIN_LINES["LadrunoForceBeamColumn"].strip()}
    )
    report = _skip_report(FBC_ROW.fork_test, reason)
    apply_warrant_rule(_item("live", "ladruno_fork"), report)
    assert report.skipped, "the row names the feature: the lane keeps the skip"
    assert not wr.side_green(report.outcome), "and the member is not fork green"


def test_a_post_pin_skip_the_row_does_not_name_fails_the_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = wr.post_pin_reason({"LadrunoIMKBeam -shear": "unknown option '-shear'"})
    both = wr.post_pin_reason({
        "LadrunoForceBeamColumn": "x", "LadrunoIMKBeam -shear": "y",
    })
    tet10 = next(r for r in wr.ROWS if r.member == "TenNodeTetrahedron")
    fbc_reason = wr.post_pin_reason({"LadrunoForceBeamColumn": "x"})
    for lane, nodeid, why in (
        ("fork", FBC_ROW.fork_test, other),        # not the row's feature
        ("fork", FBC_ROW.fork_test, both),         # one feature too many
        ("fork", FBC_ROW.fork_test, "post_pin: hand-written"),
        ("fork", tet10.fork_test, fbc_reason),     # a row with no post_pin
        ("stock", FBC_ROW.stock_test, fbc_reason),  # never in live-stock
    ):
        monkeypatch.setenv(wr.LANE_ENV, lane)
        report = _skip_report(nodeid, why)
        apply_warrant_rule(_item("live", "ladruno_fork"), report)
        assert report.failed, (lane, nodeid, why)
        assert "fails the lane" in str(report.longrepr)


def test_every_gate_predates_the_current_fork_pin() -> None:
    """The gates expire with their pin (#2498 review F1). Once ``FORK_PIN``
    moves, the bump slice re-probes each gate on the new build and either
    re-stamps its ``predates`` (still absent) or removes it (now present)."""
    pin = wr.fork_pin_sha((ROOT / "FORK_PIN").read_text(encoding="utf-8"))
    stale = wr.stale_gates(pin)
    assert not stale, (
        "FORK_PIN moved; re-justify or remove these post-pin gates "
        "(tests/opensees/warrant_rows.py POST_PIN_FEATURES): " + "; ".join(stale)
    )


def test_a_bumped_pin_names_every_stale_gate() -> None:
    bumped = "tag=x\nsha=" + "f" * 40 + "\nrepo=r\nasset=a\n"
    stale = wr.stale_gates(wr.fork_pin_sha(bumped))
    assert len(stale) == len(wr.POST_PIN_FEATURES)
    for name in wr.POST_PIN_FEATURES:
        assert any(repr(name) in s for s in stale), name
    for bad in ("tag=x\n", "sha=a\nsha=b\n", " sha=a\n"):
        with pytest.raises(ValueError, match="exactly one sha="):
            wr.fork_pin_sha(bad)


def test_post_pin_names_are_registered() -> None:
    for row in wr.ROWS:
        assert set(row.post_pin) <= set(wr.POST_PIN_FEATURES), row.member
    assert wr.post_pin_missing("post_pin: the live build lacks Nope -- x") == ()


def _calls_gate_with(fn: ast.FunctionDef) -> ast.Call | None:
    first = fn.body[1] if ast.get_docstring(fn) else fn.body[0]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            and isinstance(first.value.func, ast.Name)
            and first.value.func.id == "require_fork_features"):
        return first.value
    return None


def test_every_gated_test_asks_for_its_features_first() -> None:
    """The wp200 tests open with ``require_fork_features(*NEEDS[<own name>])``
    and the FBC fork test with its row's ``post_pin`` names."""
    module = ast.parse(WP200.read_text(encoding="utf-8"))
    needs = next(
        ast.literal_eval(s.value) for s in module.body
        if isinstance(s, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "NEEDS" for t in s.targets)
    )
    funcs = {s.name: s for s in module.body if isinstance(s, ast.FunctionDef)}
    assert set(needs) <= set(funcs)
    for name, features in needs.items():
        assert set(features) <= set(wr.POST_PIN_FEATURES), name
        call = _calls_gate_with(funcs[name])
        assert call is not None, f"{name} does not open with the gate"
        (arg,) = call.args
        assert isinstance(arg, ast.Starred)
        assert ast.literal_eval(arg.value.slice) == name  # type: ignore[attr-defined]

    path, fn_name = FBC_ROW.fork_test.split("::")
    fbc = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    fn = next(s for s in fbc.body
              if isinstance(s, ast.FunctionDef) and s.name == fn_name)
    call = _calls_gate_with(fn)
    assert call is not None
    assert tuple(ast.literal_eval(a) for a in call.args) == FBC_ROW.post_pin


# ---------------------------------------------------------------------------
# The live emitter's message: a rejected option is not an unknown element
# ---------------------------------------------------------------------------

_FORK_REQUIRED = "requires the Ladruno fork build"
_UNKNOWN_ELEMENT = "does not know this element"


class _ElementOps:
    """A build whose ``element`` prints ``line`` and then raises, or (with
    ``raises=False``) returns without creating the element."""

    def __init__(self, line: str, *, fd: bool = False, raises: bool = True) -> None:
        self.line, self.fd, self.raises = line, fd, raises

    def element(self, *_args: Any) -> None:
        if self.fd:
            os.write(2, self.line.encode())
        else:
            sys.stderr.write(self.line)
        if self.raises:
            raise RuntimeError("See stderr output")

    def getEleTags(self) -> list[int]:  # noqa: N802 - the OpenSees name
        return []


def _bare_emitter(ops: object) -> Any:
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops
    le._in_partition = False
    le._fork_verified_types = set()
    return le


@pytest.mark.parametrize("fd", [False, True], ids=["sys.stderr", "fd2"])
@pytest.mark.parametrize("raises", [True, False], ids=["raises", "drops"])
def test_a_rejected_option_names_the_option(fd: bool, raises: bool) -> None:
    le = _bare_emitter(_ElementOps(PIN_LINES["LadrunoIMKBeam -shear"], fd=fd,
                                   raises=raises))
    with pytest.raises(RuntimeError) as info:
        le.element("LadrunoIMKBeam", 2, 1, 2, 0.04, 2e11, 1.6e-4, 1, "-shear", 8e10, 0.03)
    msg = str(info.value)
    assert "option -shear is not supported by this build" in msg
    assert "unknown option '-shear'" in msg, "it quotes the build's line"
    assert _FORK_REQUIRED not in msg and _UNKNOWN_ELEMENT not in msg


@pytest.mark.parametrize("raises", [True, False], ids=["raises", "drops"])
def test_an_unknown_class_still_requires_the_fork(raises: bool) -> None:
    unknown = "WARNING element type LadrunoIMKBeam is unknown\n"
    le = _bare_emitter(_ElementOps(unknown, raises=raises))
    with pytest.raises(RuntimeError, match=_FORK_REQUIRED) as info:
        le.element("LadrunoIMKBeam", 2, 1, 2)
    assert "is not supported by this build" not in str(info.value)


def test_another_classes_option_line_is_not_this_elements() -> None:
    le = _bare_emitter(_ElementOps(PIN_LINES["LadrunoCohesiveHinge -implex"]))
    with pytest.raises(RuntimeError, match=_FORK_REQUIRED):
        le.element("LadrunoIMKBeam", 2, 1, 2)
    # A class whose name merely contains the element's is not the element.
    le = _bare_emitter(_ElementOps("WARNING LadrunoIMKBeamX -- unknown option '-q'\n"))
    with pytest.raises(RuntimeError, match=_FORK_REQUIRED):
        le.element("LadrunoIMKBeam", 2, 1, 2)


def test_the_builds_line_still_reaches_the_user(
    capfd: pytest.CaptureFixture[str],
) -> None:
    le = _bare_emitter(_ElementOps(PIN_LINES["LadrunoIMKBeam -shear"], fd=True))
    with pytest.raises(RuntimeError):
        le.element("LadrunoIMKBeam", 2, 1, 2)
    assert "unknown option '-shear'" in capfd.readouterr().err


def test_unreadable_streams_fall_back_to_the_fork_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apeGmsh.opensees.emitter import live

    monkeypatch.setattr(live, "_dup_std_fds", lambda _streams: None)
    le = _bare_emitter(_ElementOps(PIN_LINES["LadrunoIMKBeam -shear"]))
    with pytest.raises(RuntimeError, match=_FORK_REQUIRED):
        le.element("LadrunoIMKBeam", 2, 1, 2)
