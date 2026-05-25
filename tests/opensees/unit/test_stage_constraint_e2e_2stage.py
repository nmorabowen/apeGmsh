"""2-stage SSI E2E — locks the Cerro Lindo forcing-function fix.

The user-facing forcing function:

* Stage 1: rock + EDZ K0 install (initial_stress + BCs + body force).
* Stage 2: install cimbra (activate cimbra PG + claim embed).

Pre-fix all-global wiring puts the embed in the global pre-stage
block, so the stiff penalty constraint (K=1e8) is active from t=0
and Newton must equilibrate rock + lining + embed simultaneously
from a zero initial state — diverges on the second analyze step.

Post-fix the embed defers to stage 2's block, with ``domainChange``
firing AFTER the constraint emit and BEFORE the stage's analysis
chain.  Verifies the deck-structure contract.

The convergence-runtime proof (Newton actually converging) belongs
to the optional subprocess gate (Cerro Lindo binary acceptance), out
of scope for this unit test.
"""
from __future__ import annotations

from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _make_two_pg_fem() -> FEMStub:
    """Two-PG fem: ``rock`` (global elements) + ``cimbra`` (stage-2 element)."""
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                # cimbra nodes (away from rock so they stage-activate
                # cleanly):
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (3.0, 1.0, 0.0),
                (2.0, 1.0, 0.0),
                # The embedded slave node — lives inside the rock host
                # element conceptually.
                (0.5, 0.5, 0.0),
            ],
            node_pgs={"base": [1, 2], "left": [1, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
                "cimbra": _ElementGroupView(
                    ids=(2,), connectivity=((5, 6, 7, 8),),
                ),
            },
        ),
    )
    # Resolved embed record from g.constraints.embedded(host_label="rock",
    # embedded_label="cimbra_anchors", name="cimbra_embed") — naming the
    # constraint is the user's opt-in to stage-bound routing.
    rec = InterpolationRecord(
        kind="embedded", name="cimbra_embed",
        slave_node=9, master_nodes=[1, 2, 3],
        weights=None, dofs=[1, 2, 3],
    )
    fem.add_surface_constraints([rec])
    return fem, rec


def _full_chain(ops):
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Transformation(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _build_two_stage_ssi(tmp_path):
    fem, rec = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    # cimbra: declared globally, activated per-stage.
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)

    # Stage 1: K0 install on rock only.
    with ops.stage(name="install_rock") as s1:
        s1.fix(pg="base", dofs=(1, 1))
        s1.initial_stress(
            name="rock_k0", pg="rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=5,
        )
        s1.analysis(**_full_chain(ops))
        s1.run(n_increments=5, dt=1.0)

    # Stage 2: activate cimbra, claim embed.
    with ops.stage(name="install_cimbra") as s2:
        s2.activate(pgs=["cimbra"])
        s2.embedded(name="cimbra_embed")
        s2.analysis(**_full_chain(ops))
        s2.run(n_increments=5, dt=1.0)

    out = tmp_path / "two_stage.tcl"
    ops.tcl(str(out))
    return out.read_text(encoding="utf-8"), rec


def test_stage1_has_no_cimbra_or_embed(tmp_path) -> None:
    """Stage 1's block must NOT contain the cimbra element line OR
    the embed line — both belong to stage 2."""
    text, _ = _build_two_stage_ssi(tmp_path)

    s1_start = text.index("# === Stage: install_rock ===")
    s2_start = text.index("# === Stage: install_cimbra ===")
    stage1_block = text[s1_start:s2_start]

    # No cimbra element fan-out in stage 1.  The cimbra element fan-
    # out emits ``element quad <tag> 5 6 7 8 ...`` — assert no quad
    # line in stage 1 references the cimbra corner nodes.
    for ln in stage1_block.splitlines():
        if "element quad" in ln:
            assert "5 6 7 8" not in ln, (
                f"cimbra element leaked into stage 1 block: {ln!r}"
            )
    # No embed line in stage 1.
    assert "ASDEmbeddedNodeElement" not in stage1_block, (
        "embed record leaked into stage 1 block"
    )


def test_stage2_has_cimbra_and_embed_and_domainchange(tmp_path) -> None:
    """Stage 2's block must contain BOTH the cimbra element line AND
    the embed line, with ``domainChange`` AFTER both and BEFORE the
    analysis chain."""
    text, _ = _build_two_stage_ssi(tmp_path)

    s2_start = text.index("# === Stage: install_cimbra ===")
    s2_end = text.index("loadConst -time 0.0", s2_start)
    stage2_block = text[s2_start:s2_end]

    # cimbra element line present (5 6 7 8 connectivity).
    cimbra_lines = [
        ln for ln in stage2_block.splitlines()
        if "element quad" in ln and "5 6 7 8" in ln
    ]
    assert len(cimbra_lines) == 1, (
        f"expected cimbra element in stage 2; got {cimbra_lines}"
    )

    # Embed line present.
    embed_lines = [
        ln for ln in stage2_block.splitlines()
        if "ASDEmbeddedNodeElement" in ln
    ]
    assert len(embed_lines) == 1, (
        f"expected embed in stage 2; got {embed_lines}"
    )

    # Ordering: cimbra element comes first, then embed, then
    # domainChange, then analysis chain.
    cimbra_idx = stage2_block.index(cimbra_lines[0])
    embed_idx = stage2_block.index(embed_lines[0])
    dc_idx = stage2_block.index("domainChange")
    # Analysis directive — Static() emits ``analysis Static`` in Tcl.
    chain_idx = stage2_block.index("analysis Static")

    assert cimbra_idx < embed_idx < dc_idx < chain_idx, (
        "stage 2 emit order must be: nodes/elements -> stage-bound "
        "BCs -> stage-bound constraints -> domainChange -> analysis "
        "chain.  Got indices "
        f"cimbra={cimbra_idx}, embed={embed_idx}, "
        f"domainChange={dc_idx}, analysis={chain_idx}"
    )


def test_global_pre_stage_block_has_no_embed(tmp_path) -> None:
    """The global pre-stage MP-constraint pass must NOT emit the
    claimed embed (it's deferred to stage 2)."""
    text, _ = _build_two_stage_ssi(tmp_path)

    pre_stage = text[: text.index("# === Stage: install_rock ===")]
    assert "ASDEmbeddedNodeElement" not in pre_stage, (
        "stage-claimed embed leaked into the global pre-stage MP-"
        "constraint pass — it must emit inside its owning stage only"
    )
