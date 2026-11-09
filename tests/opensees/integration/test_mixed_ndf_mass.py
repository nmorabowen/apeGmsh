"""Nodal mass on a node whose ndf differs from the builder ndf (py target).

openseespy's ``OPS_addNodalMass`` sizes the mass matrix by the BUILDER
ndf (``OPS_GetNDF``), not the node's, so ``ops.mass(n, m, m, m)`` on a
3-dof solid node under ``ops.model('basic', '-ndm', 3, '-ndf', 6)`` dies
in ``Node::setMass`` ("incompatible matrices").  Tcl ``mass`` counts its
arguments and is unaffected.  The py and live emitters therefore bracket
an off-envelope mass in ``model basic -ndf <len(values)>``; a re-issued
``ops.model`` only sets ndm/ndf on openseespy (``OPS_model``), so the
bracket purges nothing there (ADR 0099 S7, ``model_reissue_purges``).

The live run is in ``tests/opensees/live/test_mixed_ndf_mass_live.py``.
"""
from __future__ import annotations

import re

from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

_MODEL_RE = re.compile(r"ops\.model\('basic', '-ndm', (\d), '-ndf', (\d)\)")
_MASS_RE = re.compile(r"ops\.mass\((\d+)((?:, [^,)]+)*)\)")


def _mass_builder_ndf(lines: list[str]) -> list[tuple[int, int, int]]:
    """``(node, n_values, builder ndf in force)`` for every mass line."""
    ndf = None
    out = []
    for raw in lines:
        line = raw.strip()
        if (m := _MODEL_RE.fullmatch(line)) is not None:
            ndf = int(m.group(2))
        elif (m := _MASS_RE.fullmatch(line)) is not None:
            n_vals = len([v for v in m.group(2).split(",") if v.strip()])
            out.append((int(m.group(1)), n_vals, ndf))
    return out


def test_py_mass_on_off_envelope_node_is_bracketed() -> None:
    em = PyEmitter()
    em.model(ndm=3, ndf=6)
    em.node(1, 0.0, 0.0, 0.0, ndf=3)
    em.node(2, 1.0, 0.0, 0.0, ndf=3)
    em.node(3, 2.0, 0.0, 0.0)
    em.mass(1, 1.0, 1.0, 1.0)
    em.mass(2, 2.0, 2.0, 2.0)
    em.mass(3, 3.0, 3.0, 3.0, 0.1, 0.1, 0.1)
    lines = em.lines()
    masses = _mass_builder_ndf(lines)
    assert masses == [(1, 3, 3), (2, 3, 3), (3, 6, 6)], "\n".join(lines)
    # A run of off-envelope masses shares one bracket.
    assert sum(
        1 for ln in lines if "'-ndf', 3)" in ln and "ops.model" in ln
    ) == 1, "\n".join(lines)
    # The envelope is restored after the bracket.
    last_model = [ln for ln in lines if _MODEL_RE.fullmatch(ln.strip())][-1]
    assert last_model.endswith("'-ndf', 6)")


def test_py_bracket_reopens_after_an_interleaved_line() -> None:
    em = PyEmitter()
    em.model(ndm=3, ndf=6)
    em.node(1, 0.0, 0.0, 0.0, ndf=3)
    em.node(2, 1.0, 0.0, 0.0, ndf=3)
    em.mass(1, 1.0, 1.0, 1.0)
    em.fix(2, 1, 1, 1)
    em.mass(2, 2.0, 2.0, 2.0)
    lines = em.lines()
    assert _mass_builder_ndf(lines) == [(1, 3, 3), (2, 3, 3)]
    # The fix sits under the restored envelope, not inside the bracket.
    ndf = None
    for ln in lines:
        if (m := _MODEL_RE.fullmatch(ln.strip())) is not None:
            ndf = int(m.group(2))
        elif ln.startswith("ops.fix("):
            assert ndf == 6, "\n".join(lines)


def test_py_mass_on_envelope_node_is_unchanged() -> None:
    em = PyEmitter()
    em.model(ndm=2, ndf=3)
    em.node(1, 0.0, 0.0)
    em.mass(1, 1.0, 1.0, 0.0)
    assert em.lines()[-1] == "ops.mass(1, 1.0, 1.0, 0.0)"
    assert sum(1 for ln in em.lines() if ln.startswith("ops.model(")) == 1


def test_tcl_mass_is_never_bracketed() -> None:
    """Tcl ``mass`` sizes by its argument count; a bracket would purge
    the builder-scoped registries (ADR 0099), so Tcl must not get one."""
    em = TclEmitter()
    em.model(ndm=3, ndf=6)
    em.node(1, 0.0, 0.0, 0.0, ndf=3)
    em.mass(1, 1.0, 1.0, 1.0)
    assert sum(1 for ln in em.lines() if ln.startswith("model ")) == 1


def test_py_mass_bracket_stays_inside_its_partition_block() -> None:
    """Masses in consecutive rank blocks each get their own bracket at
    their block's indent; the deck compiles."""
    em = PyEmitter()
    em.model(ndm=3, ndf=6)
    em.partition_open(0)
    em.node(1, 0.0, 0.0, 0.0, ndf=3)
    em.mass(1, 1.0, 1.0, 1.0)
    em.partition_close()
    em.partition_open(1)
    em.node(2, 1.0, 0.0, 0.0, ndf=3)
    em.mass(2, 1.0, 1.0, 1.0)
    em.partition_close()
    lines = em.lines()
    compile("\n".join(lines), "<deck>", "exec")
    assert _mass_builder_ndf(lines) == [(1, 3, 3), (2, 3, 3)]
    assert sum(
        1 for ln in lines if "ops.model(" in ln and "'-ndf', 3)" in ln
    ) == 2
