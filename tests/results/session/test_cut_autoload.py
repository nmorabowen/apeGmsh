"""ADR 0098 S6b — persisted section cuts boot as view clips.

The retired ``section_cut`` diagram kind took its door with it, not its
contract: cuts persisted under ``/opensees/cuts/`` must come back on a
freshly booted session as clips on the view. This file is that
contract's oracle, and it is Qt-free because the load half is IR — the
render half (``MeshView.add_clip`` → ``_apply_clips``) already existed
and is not re-tested here.

**The assertions that earn this file are the SKIPS.** A
``SectionCutDef`` cut only the elements it named, optionally narrowed
by a polygon on the cut plane; a ``ViewClip`` cuts the whole view. So a
port that translated those anyway would hide more of the model than the
cut ever did — silently, because nothing on screen says a plane grew.
Every skip test therefore asserts BOTH halves: no clip was added AND
the notice names the cut. Asserting ``clips == ()`` alone would pass
just as happily against a port that read nothing at all, which is the
other way to be silently wrong here.

Two tests are the mutation guards. Delete the skip logic and
``test_an_element_subset_cut_is_skipped_and_says_so`` goes red (the
subset would become a clip). Invert it and
``test_a_whole_model_cut_boots_as_a_view_clip`` goes red (the honest
cut would be refused). Neither can be satisfied by doing nothing.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from apeGmsh.cuts import SectionCutDef, SectionSweepDef, persist_to_h5
from apeGmsh.results import Results
from apeGmsh.results.session._cuts import persisted_clips
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5, _stub_opensees_model

STAGE = "grav"
N_STEPS = 3


# =====================================================================
# Fixture — a REAL composed results.h5 whose /opensees/ zone carries
# real elements, because the whole skip rule turns on the model's
# element ids actually existing to be compared against.
# =====================================================================

@pytest.fixture
def meshed(g, tmp_path: Path):
    """``(path, tags)`` — a composed results.h5 + its OpenSees tags.

    Built the long way on purpose. ``SectionCutDef.element_ids`` are
    OpenSees element TAGS, and the bridge does not hand out tags equal
    to the FEM eids it fanned them from — a fixture that faked the tags
    would let an id-space confusion pass every test in this file.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    model_path = tmp_path / "model.h5"
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    material = ElasticIsotropic(E=2.0e10, nu=0.2)
    ops.register(material)
    ops.element.FourNodeTetrahedron(pg="Body", material=material)
    ops.h5(str(model_path))

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    disp = np.tile(node_ids.astype(np.float64), (N_STEPS, 1))
    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(N_STEPS, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.end_stage()

    with h5py.File(path, "r") as f:
        meta = f["opensees/element_meta"]
        tags = tuple(sorted(
            int(t)
            for token in meta
            for t in meta[token]["ids"][...].ravel()
        ))
    assert len(tags) >= 2, "need >= 2 elements for a strict subset to exist"
    return SimpleNamespace(path=path, tags=tags)


def _results(meshed, *, cuts=(), sweeps=()):
    """Persist ``cuts`` / ``sweeps`` into the file, then open it.

    Order matters: the model handle is loaded AFTER the write, so
    ``model.cuts()`` carries them — the same order a real run produces
    (``ops.h5`` writes the cuts, a later session opens the file).
    """
    if cuts or sweeps:
        persist_to_h5(meshed.path, cuts=list(cuts), sweeps=list(sweeps))
    return Results.from_native(
        meshed.path, model=_open_model_from_h5(meshed.path),
    )


def _whole_model_cut(meshed, **kwargs) -> SectionCutDef:
    """A cut naming every element in the model — the honest case."""
    kwargs.setdefault("plane_point", (0.0, 0.0, 0.25))
    kwargs.setdefault("plane_normal", (0.0, 0.0, 1.0))
    return SectionCutDef(element_ids=meshed.tags, **kwargs)


def _session_lines(capsys) -> list[str]:
    """The ``[session]`` lines only — gmsh chatters on the same stream."""
    return [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[session]")
    ]


# =====================================================================
# The honest translation
# =====================================================================

def test_a_whole_model_cut_boots_as_a_view_clip(meshed, capsys):
    """The contract ADR 0098 kept, end to end: persisted → booted.

    Also the INVERSION guard — a port that skipped the honest cuts and
    attached the dishonest ones would fail right here.
    """
    cut = _whole_model_cut(
        meshed,
        plane_point=(0.0, 0.0, 0.25),
        plane_normal=(0.0, 0.0, 2.0),   # not unit; the Def normalises
        label="Base shear",
    )
    session = _results(meshed, cuts=[cut]).session()

    (clip,) = session.panes[0].clips
    assert clip.name == "Base shear"
    assert clip.normal == pytest.approx((0.0, 0.0, 1.0))
    # offset = dot(plane_point, unit plane_normal)
    assert clip.offset == pytest.approx(0.25)
    assert clip.flipped is False
    assert clip.active is True
    assert _session_lines(capsys) == []


def test_a_negative_side_cut_boots_flipped(meshed):
    """``side="negative"`` is the ONLY thing ``flipped`` may come from."""
    cut = _whole_model_cut(meshed, side="negative", label="From above")
    session = _results(meshed, cuts=[cut]).session()

    (clip,) = session.panes[0].clips
    assert clip.flipped is True
    assert clip.normal == pytest.approx((0.0, 0.0, 1.0))


def test_an_unlabelled_cut_still_gets_a_sensible_name(meshed):
    """``label=None`` is legal on a Def; a nameless clip is not."""
    session = _results(meshed, cuts=[_whole_model_cut(meshed)]).session()

    (clip,) = session.panes[0].clips
    assert clip.name == "Section cut 1"


def test_a_sweeps_cuts_are_flattened_onto_the_view(meshed):
    """A sweep is a container, not a plane — each of its cuts is a clip."""
    sweep = SectionSweepDef(cuts=(
        _whole_model_cut(
            meshed, plane_point=(0.0, 0.0, 0.25), label="Level 1",
        ),
        _whole_model_cut(
            meshed, plane_point=(0.0, 0.0, 0.75), label="Level 2",
        ),
    ))
    session = _results(meshed, sweeps=[sweep]).session()

    clips = session.panes[0].clips
    assert [c.name for c in clips] == ["Level 1", "Level 2"]
    assert [c.offset for c in clips] == pytest.approx([0.25, 0.75])


def test_standalone_cuts_come_before_sweep_cuts(meshed):
    """Writer order, then sweep order — what ``load_cuts_from_h5``
    attached in, kept so a restored picture reads the same way."""
    standalone = _whole_model_cut(meshed, label="Standalone")
    sweep = SectionSweepDef(cuts=(
        _whole_model_cut(meshed, plane_point=(0.0, 0.0, 0.9), label="Swept"),
    ))
    session = _results(meshed, cuts=[standalone], sweeps=[sweep]).session()

    assert [c.name for c in session.panes[0].clips] == [
        "Standalone", "Swept",
    ]


# =====================================================================
# NOTICE-AND-SKIP — the cuts a clip cannot honestly carry
# =====================================================================

def test_a_bounded_cut_is_skipped_and_says_so(meshed, capsys):
    """A polygon narrows the cut to part of the plane. ``ViewClip`` has
    no polygon, so attaching this would extend the cut across the whole
    plane — a wider hole than the user ever asked for.

    Note the cut is otherwise WHOLE-MODEL: the polygon alone is the
    reason, which a test that also subset the elements could not tell.
    """
    cut = _whole_model_cut(
        meshed,
        label="Core wall only",
        bounding_polygon=(
            (0.0, 0.0, 0.25), (1.0, 0.0, 0.25), (1.0, 1.0, 0.25),
        ),
    )
    session = _results(meshed, cuts=[cut]).session()

    assert session.panes[0].clips == ()
    (line,) = _session_lines(capsys)
    assert "Core wall only" in line
    assert "polygon" in line


def test_an_element_subset_cut_is_skipped_and_says_so(meshed, capsys):
    """The DELETION guard. A cut naming some of the model's elements
    hid only those; a clip hides everything on its far side. Drop the
    skip and this cut becomes a clip, and this test goes red.
    """
    subset = meshed.tags[:-1]
    assert len(subset) < len(meshed.tags)
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.25),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=subset,
        label="Columns only",
    )
    session = _results(meshed, cuts=[cut]).session()

    assert session.panes[0].clips == ()
    (line,) = _session_lines(capsys)
    assert "Columns only" in line
    assert str(len(meshed.tags)) in line, "the notice must say of how many"


def test_one_cut_being_skipped_does_not_take_the_others_with_it(
    meshed, capsys,
):
    """Per-cut decision, not per-file: the honest cut still lands."""
    keeper = _whole_model_cut(meshed, label="Keeper")
    dropped = SectionCutDef(
        plane_point=(0.0, 0.0, 0.5),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=meshed.tags[:1],
        label="Dropped",
    )
    session = _results(meshed, cuts=[keeper, dropped]).session()

    assert [c.name for c in session.panes[0].clips] == ["Keeper"]
    (line,) = _session_lines(capsys)
    assert "Dropped" in line


def test_a_cut_that_cannot_be_checked_is_skipped_not_guessed(meshed, capsys):
    """No element universe to compare against → the cut MIGHT be a
    strict subset. Widening the picture on an unverified guess is the
    exact failure the rule exists to prevent, so it skips and says the
    check could not be made."""
    results = _results(meshed, cuts=[_whole_model_cut(meshed, label="Blind")])
    # A handle that reports cuts but no elements, over a file with no
    # element ids to fall back on.
    results._model = SimpleNamespace(
        cuts=lambda: (_whole_model_cut(meshed, label="Blind"),),
        sweeps=lambda: (),
        elements=lambda: (),
    )
    results._path = None

    session = results.session()

    assert session.panes[0].clips == ()
    (line,) = _session_lines(capsys)
    assert "Blind" in line
    assert "could not be checked" in line


# =====================================================================
# Nothing persisted — the common case, and it must be silent
# =====================================================================

def test_a_file_with_no_cuts_zone_boots_clean_and_quiet(meshed, capsys):
    """Most files carry no cuts. They must cost no clips and no lines —
    a notice here would fire on nearly every ``session()`` in the
    product."""
    session = _results(meshed).session()

    assert len(session.panes) == 1
    assert session.panes[0].clips == ()
    assert _session_lines(capsys) == []


def test_no_cuts_zone_is_quiet_on_the_file_branch_too(meshed, capsys):
    """Same, with no handle bound — the branch that opens the file
    itself must not narrate a schema it was never asked about."""
    results = _results(meshed)
    results._model = None

    session = results.session()

    assert session.panes[0].clips == ()
    assert _session_lines(capsys) == []


# =====================================================================
# Precedence — the retired director's rule, mirrored
# =====================================================================

def test_a_bound_model_handle_takes_precedence_over_the_file(
    meshed, capsys,
):
    """``ResultsDirector.load_cuts_from_h5`` preferred the bound
    ``OpenSeesModel`` and read the file only without one. A port that
    always read the file would silently diverge — so the handle's cut
    must be the one on screen, and the file's must not be."""
    results = _results(
        meshed, cuts=[_whole_model_cut(meshed, label="from-file")],
    )
    assert [c.label for c in results.model.cuts()] == ["from-file"]

    results._model = SimpleNamespace(
        cuts=lambda: (_whole_model_cut(
            meshed, plane_point=(0.0, 0.0, 0.75), label="from-model",
        ),),
        sweeps=lambda: (),
        elements=lambda: tuple(
            SimpleNamespace(tag=t) for t in meshed.tags
        ),
    )

    session = results.session()

    assert [c.name for c in session.panes[0].clips] == ["from-model"]
    assert _session_lines(capsys) == []


def test_with_no_handle_bound_the_file_is_read(meshed):
    """The other half of the same rule — the fallback still works."""
    results = _results(
        meshed, cuts=[_whole_model_cut(meshed, label="from-file")],
    )
    results._model = None

    session = results.session()

    assert [c.name for c in session.panes[0].clips] == ["from-file"]


def test_an_empty_handle_does_not_fall_through_to_the_file(meshed, capsys):
    """A bound handle is the answer about this model's cuts even when
    the answer is "none" — re-walking the file behind its back is the
    divergence the precedence rule forbids."""
    results = _results(
        meshed, cuts=[_whole_model_cut(meshed, label="from-file")],
    )
    results._model = SimpleNamespace(
        cuts=lambda: (), sweeps=lambda: (), elements=lambda: (),
    )

    session = results.session()

    assert session.panes[0].clips == ()
    assert _session_lines(capsys) == []


# =====================================================================
# Never fail the boot
# =====================================================================

def test_a_malformed_cuts_zone_still_returns_a_session(meshed, capsys):
    """INV-SESSION-OPEN's sibling. A cuts zone this build cannot read
    is a line, never a traceback out of ``session()`` — the alternative
    is a human who cannot render their own results."""
    with h5py.File(meshed.path, "a") as f:
        f.create_group("opensees/cuts/cut_0")   # no attrs, no datasets

    results = Results.from_native(
        meshed.path, model=_stub_opensees_model(),
    )
    results._model = None   # force the file branch onto the bad zone

    session = results.session()

    assert len(session.panes) == 1
    assert session.panes[0].clips == ()
    (line,) = _session_lines(capsys)
    assert "could not be read" in line


def test_a_handle_that_raises_does_not_fail_the_boot(meshed, capsys):
    """The same guarantee on the chain-forward branch."""
    results = _results(meshed)

    def _boom():
        raise RuntimeError("intentional")

    results._model = SimpleNamespace(
        cuts=_boom, sweeps=lambda: (), elements=lambda: (),
    )

    session = results.session()

    assert len(session.panes) == 1
    assert session.panes[0].clips == ()
    (line,) = _session_lines(capsys)
    assert "intentional" in line


# =====================================================================
# The policy without a view (same decision, testable in isolation)
# =====================================================================

def test_persisted_clips_reports_every_cut_exactly_once(meshed):
    """Silence is the only forbidden outcome: each cut read yields
    either a clip or a notice — never both, never neither."""
    cuts = [
        _whole_model_cut(meshed, label="honest"),
        _whole_model_cut(
            meshed, label="bounded",
            bounding_polygon=(
                (0.0, 0.0, 0.25), (1.0, 0.0, 0.25), (1.0, 1.0, 0.25),
            ),
        ),
        SectionCutDef(
            plane_point=(0.0, 0.0, 0.5),
            plane_normal=(0.0, 0.0, 1.0),
            element_ids=meshed.tags[:1],
            label="subset",
        ),
    ]
    clips, notices = persisted_clips(_results(meshed, cuts=cuts))

    assert len(clips) + len(notices) == len(cuts)
    assert [c.name for c in clips] == ["honest"]
    assert {"bounded", "subset"} == {
        label for label in ("bounded", "subset")
        if any(label in n for n in notices)
    }
