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

import re

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


def test_inv4_non_hoisting_paths_are_refused() -> None:
    """The validator's path arm, unit-tested directly.

    ``split`` and ``partitioned per_rank`` are the two file-per-fragment
    layouts; both need the fragment's ``source`` line moved rather than
    its element lines, so neither is fixed by the S2 / S5 hoist."""
    ops = _bridge_with_beam(_mixed_ndf_fem(soil_conn=(1, 2, 3, 4, 5, 6)))
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    gated = ops.element.SixNodeTri(pg="Rock", thickness=1.0, material=mat)
    scoped = ops.timeSeries.Linear()

    for path in ("split", "partitioned per_rank"):
        with pytest.raises(BridgeError, match=path):
            validate_builder_scope_ordering(
                [gated], [scoped], ops.fem,
                ndm=2, envelope_ndf=3, path=path,
            )

    # ...and the two hoisting paths are exactly what S2 / S5 make safe.
    for path in ("flat", "partitioned"):
        validate_builder_scope_ordering(
            [gated], [scoped], ops.fem,
            ndm=2, envelope_ndf=3, path=path,
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


# =====================================================================
# Partitioned emit (ADR 0099 S5)
# =====================================================================

#: rank -> the node ids that rank owns, per :func:`_partitioned_mixed_ndf_fem`.
_RANK_NODES = {0: {1, 2, 3, 4, 5, 6}, 1: {7, 8}, 2: {9, 10}}


def _partitioned_mixed_ndf_fem() -> FEMStub:
    """Three ranks, only ONE of which owns a gated element.

    Rank 0 gets a two-quad soil block on ndf-2 nodes; ranks 1 and 2 get a
    two-node beam each on ndf-3 nodes.  The asymmetry is the point: the
    pre-S5 damage is RANK-LOCAL, because a rank owning no gated element
    never executes the bracket and runs fine.  A fixture where every rank
    is gated would hide exactly what makes this defect non-deterministic
    in ``np`` — the same model passes on 2 ranks and dies on 4.
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        coords=[
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0),
            (0.0, 5.0, 0.0), (1.0, 5.0, 0.0),
            (0.0, 9.0, 0.0), (1.0, 9.0, 0.0),
        ],
        node_pgs={"Base": (1, 2, 3), "Anchor": (7, 9), "Tip": (8,)},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Rock": _ElementGroupView(
                ids=(1, 2), connectivity=((1, 2, 5, 4), (2, 3, 6, 5)),
            ),
            "Liner": _ElementGroupView(
                ids=(3, 4), connectivity=((7, 8), (9, 10)),
            ),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.set_partitions([
        (1, [1, 2, 3, 4, 5, 6], [1, 2]),   # rank 0 — the gated soil block
        (2, [7, 8], [3]),                  # rank 1 — beam only
        (3, [9, 10], [4]),                 # rank 2 — beam only
    ])
    return fem


def _partitioned_bridge(*, gated: bool = True, scoped: bool = True) -> apeSees:
    """Mixed-ndf partitioned model; the two flags toggle the two
    conditions the S5 hoist is gated on.

    ``scoped`` carries the beam with it — geomTransf and beamIntegration
    are what a frame element needs, so a genuinely declaration-free
    variant is a soil-only one.
    """
    ops = apeSees(_partitioned_mixed_ndf_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2000.0, nu=0.25, rho=0.0)
    sec = ops.section.Elastic(E=2.0e5, A=0.01, Iz=1.0e-4)
    if gated:
        ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    if scoped:
        transf = ops.geomTransf.Linear()
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
        ops.element.dispBeamColumn(
            pg="Liner", transf=transf, integration=integ,
        )
        ops.damping.uniform(
            ratio=0.05, freq_lower=0.2, freq_upper=10.0, on=["Liner"],
        )
        ts = ops.timeSeries.Linear()
        with ops.pattern.Plain(series=ts) as pat:
            pat.load(node=8, forces=(0.0, -1.0, 0.0))
    ops.fix(pg="Base", dofs=(1, 1))
    ops.fix(pg="Anchor", dofs=(1, 1, 1))
    return ops


_GUARD_OPEN = re.compile(r"^if \{\[getPID\] == (\d+)\} \{$")


def _rank_streams(lines: list[str]) -> dict[int, list[str]]:
    """Split a partitioned Tcl deck into one linear stream per rank.

    Under OpenSeesMP every rank executes the WHOLE file, taking the global
    lines plus the bodies of its own ``if {[getPID] == K} { ... }`` guards.
    Reconstructing that stream is what makes INV-1 checkable PER RANK,
    which is the only way rank-local damage shows up at all — the
    file-wide reading says "fine" for the two ranks that own no soil.
    """
    ranks = {int(m.group(1)) for ln in lines if (m := _GUARD_OPEN.match(ln))}
    streams: dict[int, list[str]] = {r: [] for r in ranks}
    current: int | None = None
    for ln in lines:
        m = _GUARD_OPEN.match(ln)
        if m is not None:
            current = int(m.group(1))
            continue
        if current is not None and ln == "}":
            current = None
            continue
        for rank in ranks:
            if current is None or current == rank:
                streams[rank].append(ln)
    return streams


def _inv1_offenders(stream: list[str]) -> list[tuple[int, str]]:
    """Builder-scoped declarations preceding the stream's LAST model line."""
    models = [
        i for i, ln in enumerate(stream) if ln.strip().startswith("model ")
    ]
    if not models:
        return []
    return [
        (i, ln) for i, ln in enumerate(stream[:models[-1]])
        if any(ln.strip().startswith(tok) for tok in _SCOPED_TOKENS)
    ]


def _guard_count(lines: list[str], rank: int) -> int:
    return sum(
        1 for ln in lines
        if (m := _GUARD_OPEN.match(ln)) and int(m.group(1)) == rank
    )


def test_s5_partitioned_inv1_holds_on_every_rank_stream(tmp_path) -> None:
    """INV-1 asserted per RANK, not per file.

    Pre-S5 this model was refused outright (INV-4); before the refusal
    landed it emitted a deck whose rank-0 stream declared all four
    builder-scoped kinds above the quad bracket — while the rank-1 and
    rank-2 streams were already clean, because those ranks never execute
    the bracket.  Checking the file as one text hides that.
    """
    lines = _tcl_deck(_partitioned_bridge(), tmp_path)
    streams = _rank_streams(lines)
    assert sorted(streams) == [0, 1, 2]

    for rank, stream in streams.items():
        assert _inv1_offenders(stream) == [], (
            f"rank {rank} declares a builder-scoped kind above its last "
            f"model line: {_inv1_offenders(stream)}"
        )

    # ...and this is not vacuous: rank 0 really does re-issue ``model``
    # (it owns the gated block), and every rank really does carry all four
    # declarations.
    assert sum(
        1 for ln in streams[0] if ln.strip().startswith("model ")
    ) == 3
    for rank, stream in streams.items():
        for tok in _SCOPED_TOKENS:
            assert any(ln.strip().startswith(tok) for ln in stream), (
                f"{tok} missing from rank {rank}'s stream"
            )


def test_s5_only_the_owning_rank_carries_the_bracket(tmp_path) -> None:
    """The hoisted block is emitted for the gated rank ONLY.

    An empty extra guard is dead weight in Tcl and a syntax error in the
    Python emitter, so a rank owning no gated element gets no block —
    which is the same fact that made the pre-S5 failure np-dependent.
    """
    lines = _tcl_deck(_partitioned_bridge(), tmp_path)
    streams = _rank_streams(lines)
    for rank in (1, 2):
        assert not [
            ln for ln in streams[rank] if ln.strip().startswith("element quad")
        ]
        # deck header only — no bracket, no restore.
        assert sum(
            1 for ln in streams[rank] if ln.strip().startswith("model ")
        ) == 1
        assert _guard_count(lines, rank) == 1
    assert len([
        ln for ln in streams[0] if ln.strip().startswith("element quad")
    ]) == 2
    # rank 0 carries two guards: the hoisted gated block, then its own.
    assert _guard_count(lines, 0) == 2


def test_s5_hoist_moves_node_lines_it_does_not_add_or_drop_them(
    tmp_path,
) -> None:
    """Splitting a rank's block in two must not duplicate or lose a node.

    Each rank still declares exactly its owned node set, exactly once.
    """
    streams = _rank_streams(_tcl_deck(_partitioned_bridge(), tmp_path))
    for rank, owned in _RANK_NODES.items():
        tags = [
            int(ln.split()[1]) for ln in streams[rank]
            if ln.strip().startswith("node ")
        ]
        assert len(tags) == len(set(tags)), f"rank {rank} duplicated a node"
        assert set(tags) == owned, f"rank {rank} node set changed"


def test_s5_hoisted_elements_never_forward_reference_a_node(
    tmp_path,
) -> None:
    """Every node an element names is declared earlier in the same rank
    stream — the property the hoist could plausibly break by moving
    elements above the per-rank node pass."""
    streams = _rank_streams(_tcl_deck(_partitioned_bridge(), tmp_path))
    for rank, stream in streams.items():
        declared: set[int] = set()
        for ln in stream:
            s = ln.strip()
            if s.startswith("node "):
                declared.add(int(s.split()[1]))
            elif s.startswith("element quad "):
                # element quad <tag> n1 n2 n3 n4 ...
                used = {int(t) for t in s.split()[3:7]}
                assert used <= declared, (
                    f"rank {rank}: {s!r} references undeclared nodes "
                    f"{sorted(used - declared)}"
                )


@pytest.mark.parametrize(
    "kwargs", [{"gated": False}, {"scoped": False}],
    ids=["nothing-gated", "nothing-scoped"],
)
def test_s5_no_hoist_when_either_gate_is_open(tmp_path, kwargs) -> None:
    """Both gates matter.  With nothing gated no bracket ever fires; with
    nothing builder-scoped INV-1 holds vacuously.  Either way reordering
    is pure churn, so the deck keeps the shape it had: one guard per rank
    and the declarations in registration (topo) order above them."""
    lines = _tcl_deck(_partitioned_bridge(**kwargs), tmp_path)
    for rank in (0, 1, 2):
        assert _guard_count(lines, rank) == 1
    first_guard = next(i for i, ln in enumerate(lines) if _GUARD_OPEN.match(ln))
    scoped_ix = [
        i for i, ln in enumerate(lines)
        if any(ln.startswith(tok) for tok in _SCOPED_TOKENS)
    ]
    if kwargs.get("scoped", True):
        assert scoped_ix and max(scoped_ix) < first_guard
    else:
        assert scoped_ix == []


def test_s5_per_rank_fragments_are_still_refused(tmp_path) -> None:
    """``per_rank=True`` (ADR 0061) writes file-per-rank fragments, which
    puts a FILE boundary where the single-file deck has only a brace — the
    fix there is to move the fragment's ``source`` line, the same
    mechanism ``split`` needs (ADR 0099 §"How the two deferred paths
    should actually be fixed").  Deferred, so still refused."""
    ops = _partitioned_bridge()
    with pytest.raises(BridgeError, match="per_rank"):
        ops.tcl(str(tmp_path / "d.tcl"), per_rank=True)
