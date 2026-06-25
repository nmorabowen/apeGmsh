"""ADR 0068 P5 — tie-force recovery for equation-tied interfaces.

Two routes expose the fork OpenSees tie force ``f = M(a_raw - a_proj)`` (the
projection constraint force the ``LadrunoProjection`` handler applies, the
analogue of LS-DYNA ``*DATABASE_NCFORC``):

* **live query** — :meth:`apeSees.ladruno_projection_tie_force` /
  :meth:`LiveOpsEmitter.ladruno_projection_tie_force`, thin fork-gated wrappers
  around ``ops.ladrunoProjectionTieForce`` (ADR-30 P3);
* **recorder readback** — ``recorder ladruno -N constraintTieForce`` writes
  ``RESULTS/ON_NODES/CONSTRAINT_TIE_FORCE`` (``COMPONENTS="TFx[,TFy[,TFz]]"``),
  read back as the canonical component ``constraint_tie_force_{x,y,z}``
  (ADR-30 P4).

The pure-unit + fork-gate legs run anywhere; the live legs need the Ladruno
fork build (skipped on stock openseespy, which has no ``ladrunoProjectionTieForce``).
"""
from __future__ import annotations

from typing import cast

import pytest


# --------------------------------------------------------------------------
# Unit — recorder readback mapping (no openseespy / no fork needed)
# --------------------------------------------------------------------------

def test_constraint_tie_force_canonical_mapping():
    from apeGmsh.results.readers._mpco_translation import (
        canonical_node_component,
        has_canonical_mapping,
    )

    assert has_canonical_mapping("CONSTRAINT_TIE_FORCE")
    assert canonical_node_component("CONSTRAINT_TIE_FORCE", "TFx") == \
        "constraint_tie_force_x"
    assert canonical_node_component("CONSTRAINT_TIE_FORCE", "TFy") == \
        "constraint_tie_force_y"
    assert canonical_node_component("CONSTRAINT_TIE_FORCE", "TFz") == \
        "constraint_tie_force_z"


# --------------------------------------------------------------------------
# Fork-gate — the public bridge helper fails loud before any live analyze
# (no openseespy needed: it never reaches the live emitter)
# --------------------------------------------------------------------------

def test_apesees_tie_force_requires_live_analyze():
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees._internal.build import BridgeError
    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    bridge = apeSees(cast("object", make_two_node_beam()))
    with pytest.raises(BridgeError, match="no live analysis has run"):
        bridge.ladruno_projection_tie_force(1, 1)


# --------------------------------------------------------------------------
# LIVE (fork-only) — the wrapper returns the exact internal constraint force,
# and the recorder channel reads back through the canonical component.
# --------------------------------------------------------------------------

# T8 reference (ADR-30 P3): two masses tied by equalDOF under a constant force
# F on the master. The internal tie force on the master is F·m2/(m1+m2); the
# slave carries the equal-and-opposite reaction.
_M1, _M2, _F = 2.0, 3.0, 10.0
_F_EXACT = _F * _M2 / (_M1 + _M2)            # = +6.0 on the master (node 1)


def _fork_ops():
    """Importable openseespy with the fork tie-force query, or skip."""
    ops = pytest.importorskip("openseespy.opensees")
    if not hasattr(ops, "ladrunoProjectionTieForce"):
        pytest.skip(
            "stock openseespy has no ladrunoProjectionTieForce — needs the "
            "Ladruno fork build (run via the opensees_venv)."
        )
    return ops


def _build_t8_explicit(ops, *, recorder_file=None):
    """Build + run one step of the T8 two-mass explicit projection model.

    Optionally attach a ``recorder ladruno -N constraintTieForce`` first (the
    exact line :class:`~apeGmsh.opensees.recorder.Ladruno` emits unfiltered
    for ``nodal_responses=("constraintTieForce",)``).
    """
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)                          # retained (master)
    ops.mass(1, _M1)
    ops.node(2, 1.0)                          # constrained (slave)
    ops.mass(2, _M2)
    ops.equalDOF(1, 2, 1)
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(1, _F)
    ops.constraints("LadrunoProjection")
    ops.numberer("Plain")
    ops.system("Diagonal")
    ops.test("NormDispIncr", 1e-12, 10)
    ops.algorithm("Linear")
    ops.integrator("CentralDifferenceLadruno")
    ops.analysis("Transient")
    if recorder_file is not None:
        ops.recorder("ladruno", str(recorder_file), "-N", "constraintTieForce")
    assert ops.analyze(1, 1e-3) == 0


def test_live_tie_force_query_matches_exact():
    _fork_ops()
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    # Construct the live emitter first (wipes on construction); its ._ops is
    # the openseespy singleton we build the model through.
    e = LiveOpsEmitter(wipe=True)
    _build_t8_explicit(e._ops)

    f_master = e.ladruno_projection_tie_force(1, 1)
    f_slave = e.ladruno_projection_tie_force(2, 1)
    assert f_master == pytest.approx(_F_EXACT, abs=1e-9)
    assert f_slave == pytest.approx(-_F_EXACT, abs=1e-9)
    assert f_master + f_slave == pytest.approx(0.0, abs=1e-12)
    e._ops.wipe()


def test_live_tie_force_query_guard_wrong_handler():
    # The fork command refuses (raises) when the active handler is not
    # LadrunoProjection — the wrapper propagates that rather than returning a
    # bogus value.
    _fork_ops()
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    e = LiveOpsEmitter(wipe=True)
    o = e._ops
    o.model("basic", "-ndm", 1, "-ndf", 1)
    o.node(1, 0.0)
    o.mass(1, 1.0)
    o.node(2, 1.0)
    o.mass(2, 1.0)
    o.equalDOF(1, 2, 1)
    o.constraints("Transformation")          # NOT LadrunoProjection
    o.numberer("Plain")
    o.system("FullGeneral")
    o.test("NormDispIncr", 1e-10, 10)
    o.algorithm("Linear")
    o.integrator("CentralDifferenceLadruno")
    o.analysis("Transient")
    o.analyze(1, 1e-3)
    with pytest.raises(Exception):
        e.ladruno_projection_tie_force(1, 1)
    o.wipe()


def test_recorder_readback_constraint_tie_force(tmp_path):
    # The `-N constraintTieForce` channel writes RESULTS/ON_NODES/
    # CONSTRAINT_TIE_FORCE; the reader maps (CONSTRAINT_TIE_FORCE, "TFx") to
    # the canonical component "constraint_tie_force_x" (the single map entry
    # this PR adds). Read back == the live query == the exact tie force.
    ops = _fork_ops()
    from apeGmsh.results.readers._ladruno import LadrunoReader

    rec_file = tmp_path / "tieforce.ladruno"
    _build_t8_explicit(ops, recorder_file=rec_file)
    ops.remove("recorders")                  # flush + close the HDF5 file
    ops.wipe()

    reader = LadrunoReader(rec_file)
    sid = reader.stages()[0].id
    import numpy as np

    slab_m = reader.read_nodes(
        sid, "constraint_tie_force_x", node_ids=np.array([1]),
    )
    slab_s = reader.read_nodes(
        sid, "constraint_tie_force_x", node_ids=np.array([2]),
    )
    # last recorded step on each node
    assert float(slab_m.values[-1, 0]) == pytest.approx(_F_EXACT, abs=1e-9)
    assert float(slab_s.values[-1, 0]) == pytest.approx(-_F_EXACT, abs=1e-9)
