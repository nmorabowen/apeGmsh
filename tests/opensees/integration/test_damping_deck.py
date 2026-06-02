"""End-to-end deck test for ``ops.damping.rayleigh`` (ADR 0053, D1).

Builds a real (stub-fem) frame, declares global Rayleigh damping, and
asserts the ``rayleigh`` line lands in the generated Tcl *and* openseespy
decks.  No openseespy required — pure emit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.apesees import RayleighOverwriteWarning

from tests.opensees.fixtures.fem_stub import make_two_column_frame


def _frame_with_rayleigh(**rayleigh_kwargs: Any) -> apeSees:
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.damping.rayleigh(**rayleigh_kwargs)
    return ops


def _deck(ops: apeSees, tmp_path: Path, suffix: str) -> str:
    out = tmp_path / f"deck.{suffix}"
    getattr(ops, suffix)(str(out))
    return out.read_text(encoding="utf-8")


def test_raw_rayleigh_line_in_tcl_deck(tmp_path: Path) -> None:
    ops = _frame_with_rayleigh(alpha_m=0.1, beta_k=0.01)
    lines = [
        ln for ln in _deck(ops, tmp_path, "tcl").splitlines()
        if ln.strip().startswith("rayleigh")
    ]
    assert lines == ["rayleigh 0.1 0.01 0.0 0.0"]


def test_raw_rayleigh_line_in_py_deck(tmp_path: Path) -> None:
    ops = _frame_with_rayleigh(alpha_m=0.1, beta_k=0.01)
    lines = [
        ln for ln in _deck(ops, tmp_path, "py").splitlines()
        if "rayleigh(" in ln
    ]
    assert len(lines) == 1
    assert "0.1" in lines[0] and "0.01" in lines[0]


def test_ratio_form_initial_default_lands_in_betaK0(tmp_path: Path) -> None:
    # initial default → β in the 3rd slot (betaK0); the 2nd slot (betaK,
    # current tangent) stays zero — the nonlinear-safe choice (ADR 0053).
    ops = _frame_with_rayleigh(ratio=0.05, f_i=1.0, f_j=10.0)
    line = next(
        ln for ln in _deck(ops, tmp_path, "tcl").splitlines()
        if ln.strip().startswith("rayleigh")
    )
    _, alpha, beta_k, beta_k0, beta_kc = line.split()
    assert float(alpha) > 0.0
    assert float(beta_k) == 0.0
    assert float(beta_k0) > 0.0
    assert float(beta_kc) == 0.0


# --- D2: region-scoped (on=) ----------------------------------------------

def _region_rayleigh_lines(tcl_text: str) -> list[str]:
    return [
        ln.strip() for ln in tcl_text.splitlines()
        if ln.strip().startswith("region") and "-rayleigh" in ln
    ]


def test_on_pg_emits_region_ele_rayleigh(tmp_path: Path) -> None:
    ops = _frame_with_rayleigh(alpha_m=0.1, beta_k=0.02, on="Cols")
    (line,) = _region_rayleigh_lines(_deck(ops, tmp_path, "tcl"))
    toks = line.split()
    assert toks[0] == "region"
    # region $tag -ele <e1> <e2> -rayleigh 0.1 0.02 0.0 0.0
    ele = toks[toks.index("-ele") + 1: toks.index("-rayleigh")]
    assert len(ele) == 2 and all(t.lstrip("-").isdigit() for t in ele)
    assert toks[toks.index("-rayleigh") + 1:] == ["0.1", "0.02", "0.0", "0.0"]


def test_on_list_emits_one_region_per_group(tmp_path: Path) -> None:
    # The frame has one element PG; passing it twice (as a list) must emit
    # one region-rayleigh line per name.
    ops = _frame_with_rayleigh(alpha_m=0.1, on=["Cols", "Cols"])
    assert len(_region_rayleigh_lines(_deck(ops, tmp_path, "tcl"))) == 2


def test_global_plus_region_warns_overwrite(tmp_path: Path) -> None:
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.damping.rayleigh(alpha_m=0.05)               # global
    ops.damping.rayleigh(alpha_m=0.1, on="Cols")     # region
    with pytest.warns(RayleighOverwriteWarning):
        ops.tcl(str(tmp_path / "deck.tcl"))


def test_global_emits_before_region(tmp_path: Path) -> None:
    # Declared region-first, but the deck must emit the global rayleigh
    # BEFORE the region one ("region refines global").
    ops = make_two_column_frame()
    ops = apeSees(cast("object", ops))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.damping.rayleigh(alpha_m=0.1, on="Cols")     # region declared first
    ops.damping.rayleigh(alpha_m=0.05)               # global declared second
    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    lines = out.read_text(encoding="utf-8").splitlines()
    global_i = next(
        i for i, ln in enumerate(lines)
        if ln.strip().startswith("rayleigh")
    )
    region_i = next(
        i for i, ln in enumerate(lines)
        if ln.strip().startswith("region") and "-rayleigh" in ln
    )
    assert global_i < region_i


def test_on_non_element_group_fails_loud(tmp_path: Path) -> None:
    # "Top" is a NODE group — region Rayleigh needs elements for βK, so
    # resolving it to elements fails loud (BridgeError: not an element PG).
    from apeGmsh.opensees._internal.build import BridgeError

    ops = _frame_with_rayleigh(alpha_m=0.1, on="Top")
    with pytest.raises(BridgeError, match="not found"):
        ops.tcl(str(tmp_path / "deck.tcl"))


# --- D3: damping objects (Uniform / SecStif) + region -damp attach --------

def _frame_with_damping(kind: str, **kwargs: Any) -> apeSees:
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    getattr(ops.damping, kind)(**kwargs)
    return ops


def test_uniform_object_and_attach_in_deck(tmp_path: Path) -> None:
    ops = _frame_with_damping(
        "uniform", ratio=0.03, freq_lower=0.5, freq_upper=10.0, on="Cols",
    )
    lines = [ln.strip() for ln in _deck(ops, tmp_path, "tcl").splitlines()]
    # 1. the object definition line
    damp_line = next(ln for ln in lines if ln.startswith("damping Uniform"))
    dtoks = damp_line.split()
    damp_tag = dtoks[2]
    assert dtoks[3:] == ["0.03", "0.5", "10.0"]
    # 2. the region -damp attach line, referencing that exact tag
    reg_line = next(
        ln for ln in lines if ln.startswith("region") and "-damp" in ln
    )
    rtoks = reg_line.split()
    assert rtoks[rtoks.index("-damp") + 1] == damp_tag
    ele = rtoks[rtoks.index("-ele") + 1: rtoks.index("-damp")]
    assert len(ele) == 2


def test_uniform_object_emits_before_its_attach(tmp_path: Path) -> None:
    ops = _frame_with_damping(
        "uniform", ratio=0.03, freq_lower=0.5, freq_upper=10.0, on="Cols",
    )
    lines = [ln.strip() for ln in _deck(ops, tmp_path, "tcl").splitlines()]
    damp_i = next(i for i, ln in enumerate(lines) if ln.startswith("damping "))
    reg_i = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("region") and "-damp" in ln
    )
    assert damp_i < reg_i


def test_uniform_time_window_in_deck(tmp_path: Path) -> None:
    ops = _frame_with_damping(
        "uniform", ratio=0.03, freq_lower=0.5, freq_upper=10.0,
        on="Cols", activate_time=1.0,
    )
    damp_line = next(
        ln for ln in _deck(ops, tmp_path, "tcl").splitlines()
        if "damping Uniform" in ln
    )
    assert "-activateTime 1.0" in damp_line


def test_sec_stif_object_in_deck(tmp_path: Path) -> None:
    ops = _frame_with_damping("sec_stif", beta=0.002, on="Cols")
    text = _deck(ops, tmp_path, "tcl")
    assert any(
        ln.strip().split()[:2] == ["damping", "SecStif"]
        for ln in text.splitlines()
    )
    assert any(
        ln.strip().startswith("region") and "-damp" in ln
        for ln in text.splitlines()
    )


def test_damping_object_in_py_deck(tmp_path: Path) -> None:
    ops = _frame_with_damping(
        "uniform", ratio=0.03, freq_lower=0.5, freq_upper=10.0, on="Cols",
    )
    text = _deck(ops, tmp_path, "py")
    assert any("damping(" in ln and "Uniform" in ln for ln in text.splitlines())
