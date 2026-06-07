"""Tests for ``s.tied_contact(name=...)`` claim semantics (ADR 0034 follow-up).

``tied_contact`` was the last deferred stage-bound constraint sibling.
Its resolved record is a :class:`SurfaceCouplingRecord` that WRAPS one
:class:`InterpolationRecord` per slave node.  The global surface-coupling
pass consumes ``constraints.interpolations()``, which EXPANDS the wrapper
into its slave rows — so the stage-claim exclusion filter must register
the nested slave ids, not just the outer record id, or the slaves emit
BOTH globally (at t = 0) and inside the stage.

Locks the contract:

* ``s.tied_contact(name="x")`` finds the resolved
  :class:`SurfaceCouplingRecord` with ``name == "x"`` AND
  ``kind == "tied_contact"`` on ``fem.elements.constraints``, claims it,
  and appends it to the stage's constraint pool.
* The global MP-constraint pass SKIPS the claimed coupling's nested
  slaves (no double emission) — the regression the deferral guarded.
* The slaves emit AFTER the stage's open banner (inside the stage block).
* Missing name → fail-loud; double-claim across stages → fail-loud.
"""
from __future__ import annotations

import pytest

from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    SurfaceCouplingRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _make_fem_with_tied_contact(name: str = "interface"):
    """Quad host + two non-matching slave nodes coupled via tied_contact.

    Simulates the post-resolution state that
    ``g.constraints.tied_contact(master_label="host",
    slave_label="slab", name="interface")`` produces: one
    :class:`SurfaceCouplingRecord` carrying one
    :class:`InterpolationRecord` per slave node.
    """
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.4, 0.4, 0.0),  # slave 1 (interior)
                (0.6, 0.6, 0.0),  # slave 2 (interior)
            ],
            node_pgs={"Left": [1, 4], "Bottom": [1, 2]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
            },
        ),
    )
    slaves = [
        InterpolationRecord(
            kind=ConstraintKind.TIED_CONTACT,
            name=name,
            slave_node=sn,
            master_nodes=[1, 2, 3],
            weights=None,
            dofs=[1, 2, 3],
        )
        for sn in (5, 6)
    ]
    coupling = SurfaceCouplingRecord(
        kind=ConstraintKind.TIED_CONTACT,
        name=name,
        slave_records=slaves,
        master_nodes=[1, 2, 3, 4],
        slave_nodes=[5, 6],
        dofs=[1, 2, 3],
    )
    fem.add_surface_constraints([coupling])
    return fem, coupling, slaves


def _full_chain(ops):
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _build_quad_ops(fem):
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    return ops


def test_tied_contact_claim_populates_stage_pool() -> None:
    fem, coupling, slaves = _make_fem_with_tied_contact()
    ops = _build_quad_ops(fem)

    with ops.stage(name="install") as s:
        claimed = s.tied_contact(name="interface")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    # The claim returns the outer coupling record.
    assert claimed == (coupling,)
    stage = ops._stage_records[0]
    assert stage.stage_constraint_records == (coupling,)
    assert id(coupling) in ops._stage_claimed_constraint_ids


def test_tied_contact_global_emit_skips_claimed_slaves(tmp_path) -> None:
    """THE regression: the global surface-coupling pass must NOT emit the
    claimed coupling's nested slaves.  Two slaves → exactly two
    ASDEmbeddedNodeElement lines (both inside the stage), never four."""
    fem, _, _ = _make_fem_with_tied_contact()
    ops = _build_quad_ops(fem)

    with ops.stage(name="install") as s:
        s.tied_contact(name="interface")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    text = out.read_text(encoding="utf-8")

    embed_lines = [
        ln for ln in text.splitlines()
        if "ASDEmbeddedNodeElement" in ln
    ]
    assert len(embed_lines) == 2, (
        f"expected exactly 2 ASDEmbeddedNodeElement lines (one per slave, "
        f"inside the stage); got {len(embed_lines)}: {embed_lines}. "
        "More than 2 means the nested slaves leaked into the global "
        "pre-stage pass (double emission)."
    )

    # Both embed lines must appear AFTER the stage's open banner — i.e.
    # inside the stage block, not in the global pre-stage emit.
    stage_open_idx = text.index("# === Stage: install ===")
    first_embed_idx = text.index("ASDEmbeddedNodeElement")
    assert first_embed_idx > stage_open_idx, (
        "tied_contact slaves emitted in the global pre-stage block; "
        "claimed surface couplings must emit inside the owning stage."
    )


def test_tied_contact_missing_name_raises() -> None:
    fem, _, _ = _make_fem_with_tied_contact(name="interface")
    ops = _build_quad_ops(fem)

    with pytest.raises(ValueError, match=r"no resolved constraint records"):
        with ops.stage(name="install") as s:
            s.tied_contact(name="typo")
            s.analysis(**_full_chain(ops))
            s.run(n_increments=1, dt=1.0)


def test_tied_contact_double_claim_across_stages_raises() -> None:
    fem, _, _ = _make_fem_with_tied_contact()
    ops = _build_quad_ops(fem)

    with ops.stage(name="first") as s:
        s.tied_contact(name="interface")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    with pytest.raises(ValueError, match=r"already claimed by another stage"):
        with ops.stage(name="second") as s:
            s.tied_contact(name="interface")
            s.analysis(**_full_chain(ops))
            s.run(n_increments=1, dt=1.0)


def test_unclaimed_tied_contact_still_emits_globally(tmp_path) -> None:
    """A tied_contact NOT claimed by any stage stays in the global
    pre-stage emit pass — claim-by-name only routes named couplings."""
    fem, _, _ = _make_fem_with_tied_contact(name="interface")
    # A second, differently-named coupling that no stage claims.
    other = SurfaceCouplingRecord(
        kind=ConstraintKind.TIED_CONTACT,
        name="other",
        slave_records=[
            InterpolationRecord(
                kind=ConstraintKind.TIED_CONTACT,
                name="other",
                slave_node=5,
                master_nodes=[1, 2, 4],
                weights=None,
                dofs=[1, 2, 3],
            ),
        ],
        master_nodes=[1, 2, 3, 4],
        slave_nodes=[5],
        dofs=[1, 2, 3],
    )
    fem.elements.constraints._records.append(other)

    ops = _build_quad_ops(fem)
    with ops.stage(name="install") as s:
        s.tied_contact(name="interface")  # claims only "interface" (2 slaves)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    text = out.read_text(encoding="utf-8")

    embed_lines = [
        ln for ln in text.splitlines()
        if "ASDEmbeddedNodeElement" in ln
    ]
    # 2 claimed slaves (in stage) + 1 unclaimed slave (global) = 3 lines.
    assert len(embed_lines) == 3, (
        f"expected 3 ASDEmbeddedNodeElement lines (2 stage + 1 global); "
        f"got {len(embed_lines)}: {embed_lines}"
    )
    # The unclaimed "other" coupling must appear BEFORE the stage open.
    stage_open_idx = text.index("# === Stage: install ===")
    first_embed_idx = text.index("ASDEmbeddedNodeElement")
    assert first_embed_idx < stage_open_idx, (
        "the unclaimed global coupling should emit in the pre-stage block"
    )
