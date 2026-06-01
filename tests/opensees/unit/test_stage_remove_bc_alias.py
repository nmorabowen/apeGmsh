"""ADR 0051 (BL-5) — ``s.remove_bc`` is the verbatim alias of
``s.remove_sp``.

``remove_bc`` reads naturally when the released constraint was declared
with ``g.constraints.bc(...)``; it must produce byte-identical
``SPRemovalRecord`` rows and ``remove sp`` deck lines.  DOF convention:
``dofs=`` are 1-based DOF *indices* (not the 0/1 fixity flag vector that
``ops.fix`` / ``s.fix`` take).
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import _StageBuilder, apeSees
from apeGmsh.opensees._internal.build import SPRemovalRecord
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _fem() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            node_pgs={"Left": [1, 4]},
        ),
        elements=_ElementsStub(elem_pgs={
            "rock": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),)),
        }),
    )


def _full_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _build(verb: str) -> apeSees:
    """A 1-quad model with a global fix on Left, then a stage that
    releases DOFs 1,2 on Left via ``verb`` ('remove_bc' | 'remove_sp')."""
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.fix(pg="Left", dofs=(1, 1))      # global SP the stage will release
    with ops.stage(name="release") as s:
        getattr(s, verb)(pg="Left", dofs=(1, 2))
        s.fix(pg="Left", dofs=(1, 1))    # re-fix in the same stage
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    return ops


def test_remove_bc_records_identical_sp_removal_record() -> None:
    ops = _build("remove_bc")
    recs = ops._stage_records[0].remove_sp_records
    assert recs == (SPRemovalRecord(pg="Left", nodes=None, dofs=(1, 2)),)


def test_remove_bc_and_remove_sp_emit_identical_decks() -> None:
    rec_bc = RecordingEmitter()
    _build("remove_bc").build().emit(rec_bc)
    rec_sp = RecordingEmitter()
    _build("remove_sp").build().emit(rec_sp)
    assert rec_bc.calls == rec_sp.calls
    # Sanity: the deck actually contains the remove sp lines.
    assert ("remove_sp", (1, 1), {}) in rec_bc.calls
    assert ("remove_sp", (4, 1), {}) in rec_bc.calls
    assert ("remove_sp", (1, 2), {}) in rec_bc.calls
    assert ("remove_sp", (4, 2), {}) in rec_bc.calls


def test_remove_bc_rejects_both_pg_and_nodes() -> None:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(ValueError, match="exactly one"):
        with ops.stage(name="bad") as s:
            s.remove_bc(pg="Left", nodes=[1], dofs=(1,))


def test_remove_bc_rejects_empty_dofs() -> None:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(ValueError, match="at least one DOF"):
        with ops.stage(name="bad") as s:
            s.remove_bc(nodes=[1], dofs=())


def test_remove_bc_exists_on_stage_builder() -> None:
    assert callable(getattr(_StageBuilder, "remove_bc", None))


def test_remove_bc_docstring_states_dof_convention() -> None:
    doc = _StageBuilder.remove_bc.__doc__ or ""
    assert "1-based" in doc
    assert "flag" in doc            # distinguishes from the fix flag vector
    assert "remove_sp" in doc       # names the canonical method it aliases
