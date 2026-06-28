"""Global damping under partitioned (MPI) emit — ADR 0053 × ADR 0027.

Regression coverage for the plane-wave handoff's load-bearing finding #1:
a **global** ``ops.damping.rayleigh(...)`` (declared outside any stage)
was silently dropped from the partitioned (OpenSeesMP) deck — zero
``rayleigh`` lines — so an np>1 run came out *undamped* (uniform ~14 %,
max 45 % seq↔np4 surface error).  The flat (single-process) path emits
the bare ``rayleigh`` line driver-post; the partitioned path emitted no
global damping at all.

The fix (:meth:`apeSees._emit_global_damping_partitioned`) emits a bare
global ``rayleigh αM βK βK0 βKc`` ONCE outside any ``partition_open``
block, so every rank applies it to its local domain — the correct
OpenSeesMP behaviour (every partition gets the same Rayleigh damping).

This file asserts:

  1. **Non-staged**: a global ``rayleigh`` appears exactly once, OUTSIDE
     every per-rank ``if {[getPID]==K}`` block (Tcl + Py).
  2. **Staged**: the exact bug scenario — global damping declared outside
     the stage still reaches the partitioned deck, once, globally.
  3. **Fail-loud** rather than silent-drop for the forms that need
     per-rank routing the partitioned path does not yet wire: region-
     scoped Rayleigh (``on=``), Damping-object attach (``-damp``), and
     modal damping (``eigen`` / ``modalDamping``).
"""
from __future__ import annotations

import re
import warnings
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.apesees import BridgeError

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    make_two_column_frame_partitioned,
)

# Silence ADR 0027 INV-5 auto-emit warnings (constraint handler /
# numberer / system) — they fire on the non-staged path because these
# tests declare no MP-friendly chain.  They are contracted elsewhere; here
# they would only mask the damping assertions under ``-W error``.
_MP_AUTO_EMIT_FILTERS = (
    "ignore:MP constraints are present in the model:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared numberer:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared system:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _MP_AUTO_EMIT_FILTERS]


_TCL_RANK_OPEN_RE = re.compile(r"^\s*if\s*\{\[getPID\]\s*==\s*(\d+)\}\s*\{\s*$")
_PY_RANK_OPEN_RE = re.compile(r"^if\s+getPID\(\)\s*==\s*(\d+):\s*$")


def _build_columns(ops: apeSees, fem: FEMStub) -> None:
    """Emit the two-column frame's elements + base fixity (ndf 6)."""
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))


def _tcl_lines_outside_rank_blocks(deck_text: str) -> list[str]:
    """Return the deck's top-level lines — those NOT inside any
    ``if {[getPID]==K} { ... }`` block.  Used to prove a line is global.
    """
    out: list[str] = []
    depth = 0
    in_rank = False
    for raw in deck_text.splitlines():
        if not in_rank and _TCL_RANK_OPEN_RE.match(raw):
            in_rank = True
            depth = 1
            continue
        if in_rank:
            depth += raw.count("{") - raw.count("}")
            if depth <= 0:
                in_rank = False
            continue
        out.append(raw.strip())
    return out


def _py_lines_outside_rank_blocks(deck_text: str) -> list[str]:
    """Return the Py deck's column-0 lines — those NOT indented inside any
    ``if getPID()==K:`` block.
    """
    out: list[str] = []
    in_rank = False
    for raw in deck_text.splitlines():
        if not in_rank and _PY_RANK_OPEN_RE.match(raw):
            in_rank = True
            continue
        if in_rank:
            if raw.strip() == "":
                continue
            if raw.startswith((" ", "\t")):
                continue
            in_rank = False  # dedent back to column 0
            if _PY_RANK_OPEN_RE.match(raw):
                in_rank = True
                continue
        out.append(raw.strip())
    return out


# ---------------------------------------------------------------------------
# 1. Non-staged — global rayleigh emits once, outside every rank block.
# ---------------------------------------------------------------------------


def test_global_rayleigh_emits_once_outside_partition_blocks(tmp_path) -> None:
    """A global ``ops.damping.rayleigh(alpha_m=...)`` reaches the
    partitioned deck as a single bare ``rayleigh`` line outside any
    ``getPID`` block — the fix for finding #1 (was silently dropped).
    """
    alpha = 1.2566

    def _emit(side: str) -> str:
        fem = make_two_column_frame_partitioned()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _build_columns(ops, fem)
        ops.damping.rayleigh(alpha_m=alpha)  # GLOBAL — no on=, no stage
        path = tmp_path / f"deck.{side}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            getattr(ops, side)(str(path))
        return path.read_text(encoding="utf-8")

    tcl_text = _emit("tcl")
    py_text = _emit("py")

    # -- Tcl: exactly one ``rayleigh`` line, and it is top-level. --------
    tcl_rayleigh = [
        ln for ln in tcl_text.splitlines() if ln.strip().startswith("rayleigh ")
    ]
    assert len(tcl_rayleigh) == 1, (
        f"expected exactly one global rayleigh line, got {tcl_rayleigh!r}"
    )
    toks = tcl_rayleigh[0].split()
    assert toks[0] == "rayleigh"
    assert abs(float(toks[1]) - alpha) < 1e-12, tcl_rayleigh[0]
    assert [float(t) for t in toks[2:5]] == [0.0, 0.0, 0.0], tcl_rayleigh[0]
    tcl_global = _tcl_lines_outside_rank_blocks(tcl_text)
    assert any(ln.startswith("rayleigh ") for ln in tcl_global), (
        "rayleigh must be emitted OUTSIDE the per-rank blocks (global "
        "domain command, applied on every rank)"
    )

    # -- Py: exactly one ``ops.rayleigh(...)`` line, top-level. ----------
    py_rayleigh = [
        ln for ln in py_text.splitlines()
        if ln.strip().startswith("ops.rayleigh(")
    ]
    assert len(py_rayleigh) == 1, py_rayleigh
    py_global = _py_lines_outside_rank_blocks(py_text)
    assert any(ln.startswith("ops.rayleigh(") for ln in py_global), py_global


# ---------------------------------------------------------------------------
# 2. Staged — the exact finding-#1 scenario.
# ---------------------------------------------------------------------------


def _full_transient_chain(ops: apeSees) -> dict[str, object]:
    """A syntactically complete chain for emit-shape staged tests (not
    run); mirrors the partitioned-staged fixtures under this directory."""
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def test_staged_global_rayleigh_reaches_partitioned_deck(tmp_path) -> None:
    """Finding #1 verbatim: damping declared GLOBALLY (outside the stage)
    on a staged + partitioned model must still emit one bare ``rayleigh``
    line, globally.  (The handoff's workaround was to move it into the
    stage; this proves the global form now works too.)
    """
    alpha = 1.2566
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _build_columns(ops, fem)
    ops.damping.rayleigh(alpha_m=alpha)  # GLOBAL, outside the stage

    with ops.stage(name="dyn") as s:
        s.analysis(**_full_transient_chain(ops))
        s.run(n_increments=2)

    path = tmp_path / "deck.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(path))
    tcl_text = path.read_text(encoding="utf-8")

    rayleigh = [
        ln for ln in tcl_text.splitlines() if ln.strip().startswith("rayleigh ")
    ]
    assert len(rayleigh) == 1, (
        f"staged + partitioned global damping must emit exactly one "
        f"rayleigh line; got {rayleigh!r}"
    )
    assert abs(float(rayleigh[0].split()[1]) - alpha) < 1e-12, rayleigh[0]
    assert any(
        ln.startswith("rayleigh ")
        for ln in _tcl_lines_outside_rank_blocks(tcl_text)
    ), "global rayleigh must be outside the per-rank blocks"


# ---------------------------------------------------------------------------
# 3. Fail-loud (vs silent-drop) for the not-yet-wired partitioned forms.
# ---------------------------------------------------------------------------


def test_region_scoped_rayleigh_fails_loud_under_partition(tmp_path) -> None:
    """``ops.damping.rayleigh(on=...)`` needs per-rank element-tag routing
    the partitioned path does not wire — fail loud rather than silently
    drop (the same class of bug as finding #1)."""
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _build_columns(ops, fem)
    ops.damping.rayleigh(alpha_m=1.0, on="Cols")
    with pytest.raises(BridgeError, match="region-scoped"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ops.tcl(str(tmp_path / "deck.tcl"))


def test_damping_object_attach_fails_loud_under_partition(tmp_path) -> None:
    """A Damping-object region attach (``-damp``) is unsupported under
    partitioned emit — fail loud."""
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _build_columns(ops, fem)
    ops.damping.uniform(ratio=0.02, freq_lower=1.0, freq_upper=10.0, on="Cols")
    with pytest.raises(BridgeError, match="damping-object attach"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ops.tcl(str(tmp_path / "deck.tcl"))


def test_modal_damping_fails_loud_under_partition(tmp_path) -> None:
    """Modal damping (``eigen`` + ``modalDamping``) needs a parallel
    eigensolver — fail loud under partitioned emit."""
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _build_columns(ops, fem)
    ops.damping.modal(0.02, modes=2)
    with pytest.raises(BridgeError, match="modal damping"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ops.tcl(str(tmp_path / "deck.tcl"))
