"""ADR 0099 S4 — the H5/deck-replay hoist (INV-1 on the replay path).

S2 hoisted the flat bridge path's gated element blocks above the four
builder-scoped declarations.  The replay path (``from_h5 → build('tcl')``)
kept the defect: ``_replay_into`` declared ``geomTransf`` /
``beamIntegration`` / ``timeSeries`` / ``damping`` at steps 5-7b and only
then emitted the elements, so the gated element's bracket re-issued
``model BasicBuilder``, whose destructor purged all four registries — and
the deck died at the first later reference:

    TimeSeries *getTimeSeries(int tag) - none found with tag: 1
    WARNING - problem creating TimeSeries for LoadPattern 1
        while executing "pattern Plain 1 1 {..."

which is the reported Cerro Lindo Hito-3 RevA failure exactly.  Verified
on Ladruno ``25a0647f`` against the fixture below before the fix, and
clean after it.

Replay walks ALREADY-RESOLVED records, so tags are fixed and reordering
only moves deck lines — this is the one path the ADR defers that can be
genuinely fixed rather than only made loud.

The fixture is a real ``FEMData`` (not ``FEMStub``) because
``OpenSeesModel.from_h5`` reloads the neutral zone via
``FEMData.from_h5``, which rejects stub-only files.  It is the first
mixed-ndf gated fixture in this directory: every other one here sits at
an ``ndf=2`` envelope, where the gate is satisfied and no bracket fires.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.h5.test_opensees_model_build_targets import _assert_h5_equal

#: What ``~TclModelBuilder()`` purges (ADR 0099 INV-2 —
#: ``_element_capabilities._BUILDER_SCOPED_KINDS`` is the audit).
SCOPED = ("timeSeries", "geomTransf", "beamIntegration", "damping")


# ---------------------------------------------------------------------------
# Mixed-ndf fixture: a gated quad + all four builder-scoped kinds
# ---------------------------------------------------------------------------


def build_mixed_ndf_fem() -> FEMData:
    """Quad soil block (``Rock``, nodes 1-4) + a 2-node beam (``Liner``,
    nodes 5-6) on SEPARATE nodes.

    ADR 0046: an ndf-2 continuum may not share nodes with an ndf-3 frame.
    The beam forces the ``ndf=3`` envelope; the quad's parser hard-gates
    on builder ``ndf==2``, so the quad block brackets.
    """
    node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],   # 1-4: the quad
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],   # 5-6: the beam
            [5.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=1,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )

    def _sel(ids: "list[int]") -> np.ndarray:
        return np.array(ids, dtype=np.int64)

    def _coords(ids: "list[int]") -> np.ndarray:
        return node_coords[[i - 1 for i in ids]]

    pg = {
        (2, 201): {
            "name": "Rock", "node_ids": _sel([1, 2, 3, 4]),
            "node_coords": _coords([1, 2, 3, 4]), "element_ids": _sel([1]),
        },
        (1, 202): {
            "name": "Liner", "node_ids": _sel([5, 6]),
            "node_coords": _coords([5, 6]), "element_ids": _sel([2]),
        },
        (0, 203): {
            "name": "Base", "node_ids": _sel([1, 2]),
            "node_coords": _coords([1, 2]),
            "element_ids": np.array([], dtype=np.int64),
        },
        (0, 204): {
            "name": "LinerBase", "node_ids": _sel([5]),
            "node_coords": _coords([5]),
            "element_ids": np.array([], dtype=np.int64),
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={
            3: ElementGroup(
                element_type=quad_info, ids=np.array([1], dtype=np.int64),
                connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64),
            ),
            1: ElementGroup(
                element_type=line_info, ids=np.array([2], dtype=np.int64),
                connectivity=np.array([[5, 6]], dtype=np.int64),
            ),
        },
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=6, n_elems=2, bandwidth=3, types=[quad_info, line_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


def build_mixed_ndf_bridge() -> apeSees:
    """``ndf=3`` envelope + a gated ``quad`` + ALL FOUR scoped kinds.

    The damping rides the ungated beam, not the quad: ``quad(damp=...)``
    is refused up front by INV-3 (its own bracket would destroy the
    declaration it references).
    """
    ops = apeSees(build_mixed_ndf_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)

    sec = ops.section.Elastic(E=2.0e8, A=0.01, Iz=1e-5)
    transf = ops.geomTransf.Linear()
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    damp = ops.damping.uniform(
        ratio=0.03, freq_lower=0.5, freq_upper=10.0, name="frame_damp",
    )
    ops.element.dispBeamColumn(
        pg="Liner", transf=transf, integration=integ, damp=damp,
    )

    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)

    ops.fix(pg="Base", dofs=(1, 1))
    ops.fix(pg="LinerBase", dofs=(1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="LinerBase", forces=(10.0, 0.0, 0.0))
    return ops


@pytest.fixture()
def replayed_deck(tmp_path: Path) -> "list[str]":
    """``build → h5 → from_h5 → build('tcl')``, as stripped lines."""
    p = tmp_path / "model.h5"
    build_mixed_ndf_bridge().h5(str(p))
    tcl = OpenSeesModel.from_h5(str(p)).build("tcl")
    return [ln.strip() for ln in tcl.splitlines()]


def _first(lines: "list[str]", prefix: str) -> int:
    return next(i for i, ln in enumerate(lines) if ln.startswith(prefix))


# ---------------------------------------------------------------------------
# INV-1 — no builder-scoped declaration precedes the last ``model`` line
# ---------------------------------------------------------------------------


def test_inv1_replayed_deck_declares_after_the_last_model_line(
    replayed_deck: "list[str]",
) -> None:
    """THE gate.  Pre-S4 all four landed above the bracket and the deck
    died at ``pattern Plain 1 1`` against a purged ``timeSeries``."""
    lines = replayed_deck
    last_model = max(
        i for i, ln in enumerate(lines) if ln.startswith("model ")
    )
    offenders = [
        lines[i] for i in range(last_model)
        if any(lines[i].startswith(k + " ") for k in SCOPED)
    ]
    assert offenders == [], (
        "ADR 0099 INV-1: builder-scoped declaration(s) emitted above the "
        f"last model line, where the bracket destroys them: {offenders}"
    )
    # Non-vacuity: the fixture really does carry all four kinds.
    found = {k for ln in lines for k in SCOPED if ln.startswith(k + " ")}
    assert found == set(SCOPED), f"fixture lost a scoped kind: {found}"


def test_inv1_gated_element_precedes_the_declarations(
    replayed_deck: "list[str]",
) -> None:
    """The hoist moves the ELEMENTS up, not the declarations down."""
    lines = replayed_deck
    assert (
        _first(lines, "element quad")
        < _first(lines, "geomTransf")
        < _first(lines, "element dispBeamColumn")
    )


def test_declarations_still_precede_the_ungated_elements(
    replayed_deck: "list[str]",
) -> None:
    """Anti-degeneracy.

    A "fix" that instead deferred the declarations below ALL elements
    would satisfy INV-1 and still be wrong: the ungated beam resolves
    ``geomTransf`` / ``beamIntegration`` / ``-damp`` by tag on its own
    line, and an ``ASDAbsorbingBoundary`` resolves a ``timeSeries`` the
    same way.  Only the GATED elements may hoist.
    """
    lines = replayed_deck
    beam = _first(lines, "element dispBeamColumn")
    for kind in SCOPED:
        assert _first(lines, kind) < beam, (
            f"{kind} must still precede the ungated element that "
            f"resolves it by tag"
        )
    damp_ele = next(ln for ln in lines if "-damp" in ln)
    assert lines.index(damp_ele) > _first(lines, "damping")


def test_replayed_deck_has_exactly_three_model_lines(
    replayed_deck: "list[str]",
) -> None:
    """header / bracket-open / envelope-restore — nothing else re-issues.

    Pins the coalescing too: a hoist that bracketed per-spec rather than
    per-run would inflate this count.
    """
    models = [ln for ln in replayed_deck if ln.startswith("model ")]
    assert models == [
        "model BasicBuilder -ndm 2 -ndf 3",
        "model BasicBuilder -ndm 2 -ndf 2",
        "model BasicBuilder -ndm 2 -ndf 3",
    ]


# ---------------------------------------------------------------------------
# The archive-rewrite path opts out (``deck_ordering=False``)
# ---------------------------------------------------------------------------


def test_h5_reemit_is_unaffected_by_the_hoist(tmp_path: Path) -> None:
    """The mutation pin for ``deck_ordering``, taken at the SEAM.

    An archive-level comparison cannot catch a flipped default —
    ``_assert_group_equal`` sorts keys, so the element-meta group
    creation order the hoist perturbs is invisible there.  Drive
    ``_replay_into`` directly instead and compare element call order.
    """
    from apeGmsh.opensees._internal.compose import _replay_into
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    p = tmp_path / "model.h5"
    build_mixed_ndf_bridge().h5(str(p))
    om = OpenSeesModel.from_h5(str(p))

    def _element_tokens(*, deck_ordering: bool) -> "list[str]":
        # The transforms are load-bearing here, not decoration: the hoist
        # only fires when a builder-scoped declaration exists for the
        # bracket to destroy (with none, INV-1 holds vacuously and the
        # deck keeps its order).
        em = RecordingEmitter()
        _replay_into(
            em, ndm=2, ndf=3,
            transforms=om.transforms(),
            elements=om.elements(),
            deck_ordering=deck_ordering,
        )
        return [
            str(a[0]) for name, a, _ in em.calls if name == "element"
        ]

    # Archive path: record order, exactly as before S4.
    assert _element_tokens(deck_ordering=False) == [
        "dispBeamColumn", "quad",
    ]
    # Deck path: the gated element hoists ahead of the ungated one.
    assert _element_tokens(deck_ordering=True) == [
        "quad", "dispBeamColumn",
    ]


def test_h5_populate_path_actually_opts_out(tmp_path: Path) -> None:
    """The other half of the ``deck_ordering`` pin: that the ARCHIVE
    CALLER selects the opt-out.

    ``test_h5_reemit_is_unaffected_by_the_hoist`` passes ``deck_ordering``
    explicitly, so it proves ``_replay_into`` has two branches — not that
    ``_populate_emitter_h5`` picks the right one.  Deleting
    ``deck_ordering=False`` from that call site passes every other test
    in this file; this is the one that catches it.

    Observed on ``H5Emitter._elements``, the append-ordered list whose
    order sets the ``/opensees/element_meta`` group creation order.  A
    file-level comparison cannot see this: ``_assert_group_equal`` sorts
    keys.
    """
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    p = tmp_path / "model.h5"
    build_mixed_ndf_bridge().h5(str(p))
    om = OpenSeesModel.from_h5(str(p))

    emitter = H5Emitter(model_name="m", snapshot_id=None)
    om._populate_emitter_h5(emitter)
    assert [str(r.type_token) for r in emitter._elements] == [
        "dispBeamColumn", "quad",
    ], (
        "the archive-rewrite path must keep RECORD order — it emits no "
        "deck, so the ADR 0099 hoist buys it nothing and would perturb "
        "the element_meta group creation order for free"
    )


def test_partition_predicate_is_resolved_once_per_distinct_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 4b split costs O(distinct tokens), not O(elements).

    Gatedness is a property of the token at a fixed
    ``(ndm, envelope_ndf)``, so evaluating it per element is pure waste —
    and it lands on decks the hoist does nothing for.  An adversarial
    probe measured **+13-15%** on the emit of an ungated-only 200k-element
    deck before this was memoed, ~355 ns of the 430 ns per call being the
    function-level ``import`` re-executed inside the predicate.

    A call-count pin, deliberately NOT a timing assertion — wall-clock
    tests on this repo's warm bench cells are not trustworthy.
    """
    from apeGmsh.opensees._internal import build as _build
    from apeGmsh.opensees._internal.compose import _replay_into
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    class _Rec:
        __slots__ = ("tag", "type_token", "args", "connectivity", "fem_eid")

        def __init__(self, tag: int, token: str) -> None:
            self.tag = tag
            self.type_token = token
            self.args = ()
            self.connectivity = ()
            self.fem_eid = tag

    calls: "list[str]" = []
    real = _build.needs_builder_ndf_bracket_for_token

    def _spy(type_token: str, **kw: int) -> bool:
        calls.append(type_token)
        return real(type_token, **kw)

    monkeypatch.setattr(
        _build, "needs_builder_ndf_bracket_for_token", _spy,
    )

    class _Transf:
        type_token = "Linear"
        tag = 1
        vec = ()

    def _run(n: int) -> int:
        # The transform is load-bearing: it is what makes the hoist fire
        # at all (see the vacuous-case early-out in ``_replay_into``).
        calls.clear()
        recs = [
            _Rec(i, ("quad", "dispBeamColumn", "truss")[i % 3])
            for i in range(1, n + 1)
        ]
        _replay_into(
            RecordingEmitter(), ndm=2, ndf=3,
            transforms=(_Transf(),), elements=recs,
        )
        assert set(calls) == {"quad", "dispBeamColumn", "truss"}
        return len(calls)

    # Same 3 distinct tokens, 10x the elements: the count must not move.
    # Asserting invariance rather than a literal keeps this honest as
    # passes are added — S4b's INV-3 guard legitimately added one, and a
    # fixed-number assertion would just have been bumped.
    few, many = _run(300), _run(3000)
    assert few == many, (
        f"predicate calls scale with element count ({few} at 300 elements, "
        f"{many} at 3000) — gatedness is a property of the TOKEN and must "
        f"be resolved once per distinct token, not per element"
    )


def test_no_scoped_declarations_means_no_hoist(tmp_path: Path) -> None:
    """A gated element with NOTHING to destroy keeps its line position.

    INV-1 is about declarations surviving the bracket.  With no
    ``timeSeries`` / ``geomTransf`` / ``beamIntegration`` / ``damping``
    in the deck it holds vacuously, so reordering would be churn: a deck
    diff, moved line numbers, and no defect averted.  This also keeps the
    partition off the hot path for the models the hoist cannot help.
    """
    ops = apeSees(build_mixed_ndf_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    # An UNGATED element declared FIRST, so record order and hoisted
    # order are distinguishable: a Truss needs only a uniaxial material,
    # which is not builder-scoped (unlike a beam, which drags in a
    # geomTransf and would defeat the point of this fixture).
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    ops.element.Truss(pg="Liner", A=0.01, material=steel)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))

    p = tmp_path / "model.h5"
    ops.h5(str(p))
    lines = [
        ln.strip()
        for ln in OpenSeesModel.from_h5(str(p)).build("tcl").splitlines()
    ]
    assert not any(
        ln.startswith(k + " ") for ln in lines for k in SCOPED
    ), "fixture must carry no builder-scoped declaration"
    # The quad still BRACKETS — that is the ndf gate, untouched by ADR
    # 0099 — but it must not HOIST: the truss keeps its record position
    # ahead of it.  Drop the vacuous-case early-out and this inverts.
    assert _first(lines, "element Truss") < _first(lines, "element quad")


def test_h5_roundtrip_fixed_point_on_a_gated_archive(tmp_path: Path) -> None:
    """``h5 → from_h5 → to_h5`` stays a fixed point on a mixed-ndf archive.

    Smoke, NOT a mutation pin — ``_assert_h5_equal`` compares
    structurally (sorted keys), so it would still pass if the archive
    path took the hoist.  ``test_h5_reemit_is_unaffected_by_the_hoist``
    is the test that would fail.
    """
    src = tmp_path / "src.h5"
    rt = tmp_path / "rt.h5"
    build_mixed_ndf_bridge().h5(str(src))
    OpenSeesModel.from_h5(str(src)).to_h5(str(rt))
    with h5py.File(str(src), "r") as a, h5py.File(str(rt), "r") as b:
        _assert_h5_equal(a, b)
