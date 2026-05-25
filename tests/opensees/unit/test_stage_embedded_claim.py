"""Tests for ``s.embedded(name=...)`` claim semantics.

Locks the contract:

* ``s.embedded(name="x")`` finds resolved ``InterpolationRecord`` rows
  matching ``name`` AND ``kind == "embedded"`` on
  ``fem.elements.constraints``, claims them by ``id()``, and appends
  them to the stage's constraint pool.
* The global MP-constraint pass SKIPS claimed records (no double
  emission).
* The stage's emit block emits the claimed records AFTER stage
  regions and BEFORE the stage's ``domain_change()``.
* Missing name → fail-loud at claim time (catches typos).
* Double-claim across two stages → fail-loud at the second claim.
"""
from __future__ import annotations

import pytest
from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _make_fem_with_embed_record(name: str = "cimbra_embed") -> FEMStub:
    """Tiny quad mesh + one embed record on the surface broker.

    Simulates the post-resolution state that
    ``g.constraints.embedded(host_label="host", embedded_label="emb",
    name="cimbra_embed")`` would have produced at apeGmsh time.
    """
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.5, 0.5, 0.0),  # the embedded node
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
    # Resolved record: node 5 embedded inside the host quad (nodes 1-4).
    rec = InterpolationRecord(
        kind="embedded",
        name=name,
        slave_node=5,
        master_nodes=[1, 2, 3],  # 3-node host face for ASDEmbedded
        weights=None,
        dofs=[1, 2, 3],
    )
    fem.add_surface_constraints([rec])
    return fem, rec


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


def test_embedded_claim_populates_stage_pool() -> None:
    fem, rec = _make_fem_with_embed_record()
    ops = _build_quad_ops(fem)

    with ops.stage(name="install") as s:
        claimed = s.embedded(name="cimbra_embed")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    assert claimed == (rec,)
    stage = ops._stage_records[0]
    assert stage.stage_constraint_records == (rec,)
    assert id(rec) in ops._stage_claimed_constraint_ids


def test_embedded_claim_missing_name_raises() -> None:
    fem, _ = _make_fem_with_embed_record(name="cimbra_embed")
    ops = _build_quad_ops(fem)

    with pytest.raises(ValueError, match=r"no resolved constraint records"):
        with ops.stage(name="install") as s:
            s.embedded(name="typo")
            s.analysis(**_full_chain(ops))
            s.run(n_increments=1, dt=1.0)


def test_embedded_claim_empty_name_raises() -> None:
    fem, _ = _make_fem_with_embed_record()
    ops = _build_quad_ops(fem)

    with pytest.raises(ValueError, match=r"name= must be non-empty"):
        with ops.stage(name="install") as s:
            s.embedded(name="")
            s.analysis(**_full_chain(ops))
            s.run(n_increments=1, dt=1.0)


def test_embedded_double_claim_across_stages_raises() -> None:
    fem, _ = _make_fem_with_embed_record()
    ops = _build_quad_ops(fem)

    with ops.stage(name="first") as s:
        s.embedded(name="cimbra_embed")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    with pytest.raises(ValueError, match=r"already claimed by another stage"):
        with ops.stage(name="second") as s:
            s.embedded(name="cimbra_embed")
            s.analysis(**_full_chain(ops))
            s.run(n_increments=1, dt=1.0)


def test_global_emit_skips_claimed_record(tmp_path) -> None:
    """The global MP-constraint pass must NOT emit a claimed record."""
    fem, _ = _make_fem_with_embed_record()
    ops = _build_quad_ops(fem)

    with ops.stage(name="install") as s:
        s.embedded(name="cimbra_embed")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    text = out.read_text(encoding="utf-8")

    # Exactly ONE embeddedNode line — emitted inside the stage block,
    # not duplicated in the global pre-stage pass.
    embed_lines = [
        ln for ln in text.splitlines()
        if "ASDEmbeddedNodeElement" in ln
    ]
    assert len(embed_lines) == 1, (
        f"expected exactly 1 ASDEmbeddedNodeElement line; "
        f"got {len(embed_lines)}: {embed_lines}"
    )

    # The embed line must appear AFTER the stage's open comment, not
    # before (which would mean it landed in the global pre-stage block).
    stage_open_idx = text.index("# === Stage: install ===")
    embed_idx = text.index("ASDEmbeddedNodeElement")
    assert embed_idx > stage_open_idx, (
        "embed record emitted in the global pre-stage block; "
        "claimed records must emit inside the owning stage's block"
    )


def test_unclaimed_global_record_still_emits_globally(tmp_path) -> None:
    """A record NOT claimed by any stage stays in the global pre-stage
    emit pass — claim-by-name only routes the records the user names."""
    fem, _ = _make_fem_with_embed_record(name="cimbra_embed")
    # Add a second, UNNAMED embed record that no stage claims.
    second_rec = InterpolationRecord(
        kind="embedded",
        name="other",
        slave_node=5,
        master_nodes=[1, 2, 4],
        weights=None,
        dofs=[1, 2, 3],
    )
    fem.elements.constraints._records.append(second_rec)

    ops = _build_quad_ops(fem)
    with ops.stage(name="install") as s:
        s.embedded(name="cimbra_embed")  # claims only the first
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    text = out.read_text(encoding="utf-8")

    embed_lines = [
        ln for ln in text.splitlines()
        if "ASDEmbeddedNodeElement" in ln
    ]
    # Two embed records → two lines.
    assert len(embed_lines) == 2, (
        f"expected 2 ASDEmbeddedNodeElement lines (one global, one "
        f"stage); got {len(embed_lines)}: {embed_lines}"
    )
    # The unclaimed "other" record must appear BEFORE the stage open.
    stage_open_idx = text.index("# === Stage: install ===")
    # Index of the first embed line (in the global block):
    first_embed_idx = text.index("ASDEmbeddedNodeElement")
    assert first_embed_idx < stage_open_idx
