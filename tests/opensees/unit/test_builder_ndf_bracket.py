"""Builder-ndf bracket for gated upstream element parsers.

``OPS_FourNodeQuad`` / ``OPS_SixNodeTri`` hard-gate on the BUILDER state
(``OPS_GetNDM() != 2 || OPS_GetNDF() != 2``) and refuse to parse, ignoring
per-node ndf — so a mixed-ndf 2D model (envelope ndf=3 because of beams,
soil nodes inferred ndf=2 per ADR 0048/0049) used to die at the first
``tri6n`` / ``quad`` line with *"WARNING -- model dimensions and/or nodal
DOF not compatible with quad element"*.

The emit orchestrators now bracket each gated element block with a
``model basic -ndf 2`` re-issue + envelope restore (re-issuing ``model``
does not wipe the domain — the same trick STKO decks use for their
per-subset ndf switches).

It DOES, however, delete the Tcl model builder, whose destructor purges
the process-global ``timeSeries`` / ``geomTransf`` / ``beamIntegration``
/ ``damping`` registries (fork ``TclModelBuilder.cpp:681``, measured on
build ``25a0647f``).  ADR 0099 hoists the gated blocks above every such
declaration on the flat path (INV-1) and fails loud everywhere else.

Covers:

* the capability table (:func:`element_builder_ndf`),
* the open/close helpers against a RecordingEmitter,
* end-to-end py + tcl decks for the flat path (mixed-ndf SixNodeTri +
  beam, FourNodeQuad variant, ungated Tri31, and the no-bracket flat
  ndf=2 case),
* the stage-activated path (ADR 0099 INV-4: refused, because the
  bracket would fire mid-deck and destroy the global declarations),
* ADR 0099 INV-1 ordering, and the INV-3 / INV-4 guards.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._element_capabilities import element_builder_ndf
from apeGmsh.opensees._internal.build import (
    BridgeError,
    close_builder_ndf_bracket,
    open_builder_ndf_bracket,
    validate_builder_scope_ordering,
)
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# =====================================================================
# Capability table
# =====================================================================

def test_builder_ndf_table() -> None:
    assert element_builder_ndf("SixNodeTri") == 2
    assert element_builder_ndf("FourNodeQuad") == 2
    assert element_builder_ndf("Tri31") is None
    assert element_builder_ndf("BezierTri6") is None
    assert element_builder_ndf("dispBeamColumn") is None
    assert element_builder_ndf("FourNodeTetrahedron") is None


# =====================================================================
# open/close helpers
# =====================================================================

def test_open_bracket_emits_and_reports(monkeypatch) -> None:
    from apeGmsh.opensees.element.solid import SixNodeTri
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    spec = SixNodeTri(
        pg="Rock", thickness=1.0,
        material=ElasticIsotropic(E=1e6, nu=0.3, rho=0.0),
    )
    e = RecordingEmitter()
    assert open_builder_ndf_bracket(e, spec, ndm=2, envelope_ndf=3) is True
    close_builder_ndf_bracket(e, ndm=2, envelope_ndf=3)
    assert e.calls == [
        ("model", (), {"ndm": 2, "ndf": 2}),
        ("model", (), {"ndm": 2, "ndf": 3}),
    ]


def test_open_bracket_noop_when_envelope_matches() -> None:
    from apeGmsh.opensees.element.solid import SixNodeTri
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    spec = SixNodeTri(
        pg="Rock", thickness=1.0,
        material=ElasticIsotropic(E=1e6, nu=0.3, rho=0.0),
    )
    e = RecordingEmitter()
    assert open_builder_ndf_bracket(e, spec, ndm=2, envelope_ndf=2) is False
    assert e.calls == []


def test_open_bracket_noop_for_ungated_spec() -> None:
    from apeGmsh.opensees.element.solid import Tri31
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    spec = Tri31(
        pg="Rock", thickness=1.0,
        material=ElasticIsotropic(E=1e6, nu=0.3, rho=0.0),
    )
    e = RecordingEmitter()
    assert open_builder_ndf_bracket(e, spec, ndm=2, envelope_ndf=3) is False
    assert e.calls == []


# =====================================================================
# End-to-end deck emission (flat path)
# =====================================================================

def _mixed_ndf_fem(*, soil_conn: tuple[int, ...]) -> FEMStub:
    """Soil element (tri6 or quad4 connectivity) + a 2-node beam on
    SEPARATE nodes (ADR 0046: ndf-2 soil may not share with ndf-3 beam)."""
    soil_nodes = {
        1: (0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0),
        4: (1.0, 0.0, 0.0), 5: (1.0, 0.5, 0.0), 6: (0.0, 0.5, 0.0),
    }
    beam_nodes = {7: (5.0, 0.0, 0.0), 8: (5.0, 1.0, 0.0)}
    ids = sorted(set(soil_conn) | set(beam_nodes))
    coords = [(soil_nodes | beam_nodes)[i] for i in ids]
    return FEMStub(
        nodes=_NodesStub(ids=ids, coords=coords, node_pgs={}),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(ids=(1,), connectivity=(soil_conn,)),
                "Liner": _ElementGroupView(ids=(2,), connectivity=((7, 8),)),
            },
        ),
    )


def _bridge_with_beam(fem: FEMStub) -> apeSees:
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    sec = ops.section.Elastic(E=2.0e8, A=0.01, Iz=1e-5)
    transf = ops.geomTransf.Linear()
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.dispBeamColumn(pg="Liner", transf=transf, integration=integ)
    return ops


def _py_deck(ops: apeSees, tmp_path) -> list[str]:
    deck = tmp_path / "deck.py"
    ops.py(str(deck))
    return deck.read_text().splitlines()


def _model_lines(lines: list[str], token: str) -> dict[str, int]:
    """Index map: header/bracket/restore model lines + first/last element."""
    models = [i for i, ln in enumerate(lines) if ln.startswith("ops.model(")]
    eles = [i for i, ln in enumerate(lines) if f"ops.element('{token}'" in ln]
    return {"models": models, "eles": eles}  # type: ignore[return-value]


def test_py_deck_brackets_tri6n_under_mixed_envelope(tmp_path) -> None:
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 3, 4, 5, 6)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.SixNodeTri(pg="Rock", thickness=1.0, material=mat)
    lines = _py_deck(ops, tmp_path)

    ix = _model_lines(lines, "tri6n")
    models, eles = ix["models"], ix["eles"]
    assert len(eles) == 1
    # header + bracket + restore — nothing else re-issues model.
    assert [lines[i] for i in models] == [
        "ops.model('basic', '-ndm', 2, '-ndf', 3)",
        "ops.model('basic', '-ndm', 2, '-ndf', 2)",
        "ops.model('basic', '-ndm', 2, '-ndf', 3)",
    ]
    # bracket directly wraps the tri6n block.
    assert models[1] < eles[0] < models[2]
    # the beam emits OUTSIDE the bracket (under the restored envelope).
    beam = next(i for i, ln in enumerate(lines) if "dispBeamColumn" in ln)
    assert not (models[1] < beam < models[2])


def test_py_deck_brackets_quad_under_mixed_envelope(tmp_path) -> None:
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 5, 6)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    lines = _py_deck(ops, tmp_path)

    ix = _model_lines(lines, "quad")
    models, eles = ix["models"], ix["eles"]
    assert len(eles) == 1
    assert len(models) == 3
    assert models[1] < eles[0] < models[2]


def test_py_deck_no_bracket_for_ungated_tri31(tmp_path) -> None:
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 3)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.Tri31(pg="Rock", thickness=1.0, material=mat)
    lines = _py_deck(ops, tmp_path)

    models = [ln for ln in lines if ln.startswith("ops.model(")]
    assert models == ["ops.model('basic', '-ndm', 2, '-ndf', 3)"]


def test_py_deck_no_bracket_when_envelope_is_two(tmp_path) -> None:
    # tri6n-only model under a flat ndf=2 envelope — no bracket needed.
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (0.0, 0.5, 0.0),
            ],
            node_pgs={},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4, 5, 6),),
                ),
            },
        ),
    )
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.SixNodeTri(pg="Rock", thickness=1.0, material=mat)
    lines = _py_deck(ops, tmp_path)

    models = [ln for ln in lines if ln.startswith("ops.model(")]
    assert models == ["ops.model('basic', '-ndm', 2, '-ndf', 2)"]


def test_tcl_deck_brackets_tri6n_under_mixed_envelope(tmp_path) -> None:
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 3, 4, 5, 6)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.SixNodeTri(pg="Rock", thickness=1.0, material=mat)
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))
    lines = deck.read_text().splitlines()

    models = [i for i, ln in enumerate(lines) if ln.startswith("model ")]
    ele = next(i for i, ln in enumerate(lines) if "tri6n" in ln)
    assert [lines[i].split("-ndf")[1].strip() for i in models] == [
        "3", "2", "3",
    ]
    assert models[1] < ele < models[2]


# =====================================================================
# ADR 0099 — builder-scoped declarations must follow the last model line
# =====================================================================

#: The four OpenSees declaration keywords a ``model`` re-issue destroys.
_SCOPED_TOKENS = ("timeSeries", "geomTransf", "beamIntegration", "damping")


def _bridge_all_four_kinds() -> apeSees:
    """Mixed-ndf model carrying every builder-scoped declaration kind.

    ``_bridge_with_beam`` already supplies section + geomTransf +
    beamIntegration + the frame element that forces the ndf=3 envelope;
    this adds the gated tri6n, a damping object and a timeSeries-backed
    pattern.
    """
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 3, 4, 5, 6)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.SixNodeTri(pg="Rock", thickness=1.0, material=mat)
    ops.damping.uniform(
        ratio=0.05, freq_lower=0.2, freq_upper=10.0, on=["Liner"],
    )
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.load(node=7, forces=(1000.0, 0.0, 0.0))
    return ops


def _tcl_deck(ops: apeSees, tmp_path) -> list[str]:
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))
    return deck.read_text().splitlines()


@pytest.mark.parametrize(
    "writer, model_prefix",
    [(_py_deck, "ops.model("), (_tcl_deck, "model ")],
    ids=["py", "tcl"],
)
def test_inv1_scoped_declarations_follow_the_last_model_line(
    tmp_path, writer, model_prefix,
) -> None:
    """INV-1, asserted as a contract rather than as line numbers.

    Pre-0099 this deck declared timeSeries / geomTransf / beamIntegration
    / damping ABOVE the tri6n bracket, so the bracket destroyed all four
    and the deck died at ``pattern Plain``.
    """
    lines = writer(_bridge_all_four_kinds(), tmp_path)
    last_model = max(
        i for i, ln in enumerate(lines) if ln.startswith(model_prefix)
    )
    offenders = [
        (i, ln) for i, ln in enumerate(lines[:last_model])
        if any(tok in ln for tok in _SCOPED_TOKENS)
    ]
    assert offenders == [], (
        f"builder-scoped declarations emitted before the last model line "
        f"(index {last_model}); the bracket destroys them: {offenders}"
    )
    # ...and the deck really does carry all four, so this is not vacuous.
    after = chr(10).join(lines[last_model:])
    for tok in _SCOPED_TOKENS:
        assert tok in after, f"{tok} missing from the deck entirely"


def test_inv1_gated_block_is_hoisted_above_the_declarations(
    tmp_path,
) -> None:
    """The mechanism: gated elements move up, the declarations do not move
    down — a timeSeries can legitimately be consumed BY an element
    (ASDAbsorbingBoundary's -fx/-fy/-fz), so deferring them is not safe.
    """
    lines = _py_deck(_bridge_all_four_kinds(), tmp_path)
    tri = next(i for i, ln in enumerate(lines) if "ops.element(" in ln
               and "tri6n" in ln)
    transf = next(i for i, ln in enumerate(lines) if "ops.geomTransf(" in ln)
    beam = next(i for i, ln in enumerate(lines) if "dispBeamColumn" in ln)
    # gated block first, then the declarations, then the ungated frame.
    assert tri < transf < beam


def test_inv3_gated_element_may_not_depend_on_a_scoped_primitive() -> None:
    """``quad(damp=...)`` is self-wiping: its own bracket destroys the
    damping object it references, and OpenSees only warns."""
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 5, 6)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    damp = ops.damping.uniform(ratio=0.05, freq_lower=0.2, freq_upper=10.0)
    ops.element.FourNodeQuad(
        pg="Rock", thickness=1.0, material=mat, damp=damp,
    )
    with pytest.raises(BridgeError, match="INV-3"):
        ops.build().emit(RecordingEmitter())


def test_inv4_stage_activated_gated_element_is_refused(tmp_path) -> None:
    """A stage-activated gated element brackets INSIDE the stage block —
    after the global declarations and, past the first stage, after a
    pattern and a completed analyze.  Hoisting cannot reach it, so it is
    refused until the declaration-replay slice lands."""
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 3, 4, 5, 6)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.SixNodeTri(pg="Rock", thickness=1.0, material=mat)

    with ops.stage(name="dig") as s:
        s.activate(pgs=["Rock"])
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.1),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=2, dt=0.5)

    with pytest.raises(BridgeError, match="stage-activated"):
        _py_deck(ops, tmp_path)


def test_inv4_non_flat_paths_are_refused() -> None:
    """The validator's path arm, unit-tested directly — a partitioned or
    split fixture here would exercise the fixture, not the contract."""
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 3, 4, 5, 6)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    gated = ops.element.SixNodeTri(pg="Rock", thickness=1.0, material=mat)
    scoped = ops.timeSeries.Linear()

    for path in ("split", "partitioned"):
        with pytest.raises(BridgeError, match=path):
            validate_builder_scope_ordering(
                [gated], [scoped], ops.fem,
                ndm=2, envelope_ndf=3, path=path,
            )

    # ...and the flat path is exactly what the hoist makes safe.
    validate_builder_scope_ordering(
        [gated], [scoped], ops.fem,
        ndm=2, envelope_ndf=3, path="flat",
    )


def test_validator_is_a_no_op_without_a_bracket_or_a_declaration() -> None:
    """No gated element, or nothing builder-scoped to lose: no opinion."""
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 3)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ungated = ops.element.Tri31(pg="Rock", thickness=1.0, material=mat)
    scoped = ops.timeSeries.Linear()

    # ungated element + a declaration -> fine on any path.
    validate_builder_scope_ordering(
        [ungated], [scoped], ops.fem,
        ndm=2, envelope_ndf=3, path="partitioned",
    )
    # gated element but nothing builder-scoped registered -> fine.
    gated = ops.element.SixNodeTri(pg="Rock", thickness=1.0, material=mat)
    validate_builder_scope_ordering(
        [gated], [mat], ops.fem,
        ndm=2, envelope_ndf=3, path="partitioned",
    )
