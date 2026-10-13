"""R3-S2 (#1779): the claims in "What a stage boundary does" hold for the deck.

``docs/concepts/opensees-bridge.md`` and ``skills/apegmsh/references/
opensees-bridge.md`` describe what a stage boundary does.  Each claim is
asserted here against the line order of an emitted two-stage ``py`` deck.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from apeGmsh.opensees import InertStaticDtWarning
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.unit.test_stages import _make_single_quad_fem

_CHAIN = ("constraints", "numberer", "system", "test", "algorithm",
          "integrator", "analysis")


def _deck(tmp_path: Path, *, set_time: float | None = None,
          dt2: float = 0.25, dt1: float = 0.25) -> list[str]:
    ops = apeSees(_make_single_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Left", dofs=(1, 0))
    ops.recorder.Node(file="d.out", response="disp", pg="Left", dofs=(1,))
    for name in ("s1", "s2"):
        with ops.stage(name=name) as s:
            if name == "s2" and set_time is not None:
                s.set_time(set_time)
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.load(pg="Left", forces=(1.0, 0.0))
            s.analysis(test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
                       algorithm=ops.algorithm.Newton(),
                       integrator=ops.integrator.LoadControl(dlam=0.25),
                       constraints=ops.constraints.Plain(),
                       numberer=ops.numberer.Plain(),
                       system=ops.system.BandGeneral(),
                       analysis=ops.analysis.Static())
            s.run(n_increments=4, dt=dt2 if name == "s2" else dt1)
    path = tmp_path / "deck.py"
    ops.py(str(path))
    return path.read_text().splitlines()


def _idx(lines: list[str], prefix: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    raise AssertionError(f"{prefix!r} not found after line {start}")


def test_loadconst_then_wipe_then_next_stage_chain(tmp_path: Path) -> None:
    lines = _deck(tmp_path)
    lc = _idx(lines, "ops.loadConst('-time', 0.0)")
    assert lines[lc + 1] == "ops.wipeAnalysis()"
    s2 = _idx(lines, "# === Stage: s2")
    assert lc < s2
    # the full chain is re-emitted after the wipe, in stage s2
    for verb in _CHAIN:
        assert any(ln.startswith(f"ops.{verb}(") for ln in lines[s2:]), verb
    # and not emitted globally before stage 1
    s1 = _idx(lines, "# === Stage: s1")
    assert not any(ln.startswith("ops.analysis(") for ln in lines[:s1])


def test_loadconst_closes_each_stage(tmp_path: Path) -> None:
    lines = _deck(tmp_path)
    assert sum(ln.startswith("ops.loadConst('-time', 0.0)")
               for ln in lines) == 2
    assert sum(ln == "ops.wipeAnalysis()" for ln in lines) == 2


def test_time_is_zero_unless_set_time(tmp_path: Path) -> None:
    plain = _deck(tmp_path)
    assert not any(ln.startswith("ops.setTime(") for ln in plain)
    timed = _deck(tmp_path, set_time=1.0)
    wipe = _idx(timed, "ops.wipeAnalysis()")
    st = _idx(timed, "ops.setTime(1.0)")
    assert wipe < st < _idx(timed, "ops.domainChange()", wipe)


def test_domain_and_recorders_kept(tmp_path: Path) -> None:
    lines = _deck(tmp_path)
    assert sum(ln.startswith("ops.recorder(") for ln in lines) == 1
    assert _idx(lines, "ops.recorder(") < _idx(lines, "# === Stage: s1")
    first = _idx(lines, "# === Stage: s1")
    assert not any(ln.startswith(("ops.wipe()", "ops.remove(", "ops.reset("))
                   for ln in lines[first:])
    assert sum(ln.startswith("ops.node(") for ln in lines) == 4


def test_static_ignores_dt_dlam_sets_step(tmp_path: Path) -> None:
    """dt has no effect under Static; a differing dt warns (R3 D1, #1785)."""
    with pytest.warns(InertStaticDtWarning) as rec:
        a = _deck(tmp_path, dt2=0.9, dt1=0.1)
    # s1 (dt=0.1) and s2 (dt=0.9) both differ from dlam=0.25.
    assert len(rec) == 2
    assert all(
        str(w.message).startswith(
            "dt has no effect under Static; dlam sets the step")
        for w in rec
    )
    with pytest.warns(InertStaticDtWarning):
        b = _deck(tmp_path, dt2=0.05, dt1=0.1)
    s2a = a[_idx(a, "# === Stage: s2"):]
    s2b = b[_idx(b, "# === Stage: s2"):]
    # dt reaches only the analyze call ...
    assert any("ops.analyze(1, 0.9)" in ln for ln in s2a)
    assert any("ops.analyze(1, 0.05)" in ln for ln in s2b)
    # ... the step the integrator takes is dlam, identical in both decks
    assert "ops.integrator('LoadControl', 0.25)" in s2a
    assert "ops.integrator('LoadControl', 0.25)" in s2b
    a_rest = [x.replace("0.9", "D") for x in s2a]
    b_rest = [x.replace("0.05", "D") for x in s2b]
    assert a_rest == b_rest


def test_static_dt_equal_to_dlam_is_silent(tmp_path: Path) -> None:
    """The silent half of the claim: dt == dlam does not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", InertStaticDtWarning)
        _deck(tmp_path, dt2=0.25, dt1=0.25)
