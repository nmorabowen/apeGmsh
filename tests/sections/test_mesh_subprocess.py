"""Tests — out-of-process meshing for the properties worker (ADR 0080 B6).

The worker meshes in a child interpreter so that Gmsh — one
process-global, non-reentrant C++ runtime — is never driven from a
background thread of a process that is also driving it from the main
thread. What has to hold:

* the mesh that comes back is the *same* mesh an in-process build
  produces (this is a relocation, not a reimplementation);
* the analyzer is rebuilt **live** in the parent, so the inspector can
  still drive it interactively with user-entered loads;
* every way the child can fail — a document error, a wedged mesh, a
  hard crash — arrives as a message, never as a hang and never as a
  traceback the panel cannot show.
"""
from __future__ import annotations

import pytest

from apeGmsh.sections import SectionDocument
from apeGmsh.sections._mesh_proc import (
    MeshSubprocessError,
    mesh_document,
)


def _plate(*, lc: float = 0.5) -> SectionDocument:
    doc = SectionDocument.new(name="plate", kind="continuum")
    doc.set_material("s", E=200e3, nu=0.3, fy=345.0)
    doc.add_shape("rect_face", id="s", b=4.0, h=2.0, material="s")
    doc.set_mesh(lc=lc, order=2)
    return doc


# ─────────────────────────────────────────────────────────────────────
# same mesh, same numbers
# ─────────────────────────────────────────────────────────────────────

def test_child_mesh_matches_an_in_process_build():
    """The relocation gate: analyzer numbers from the child's mesh equal
    the ones from a plain in-process ``doc.build()``."""
    doc = _plate()
    here = doc.build()
    there = doc.analysis_from_fem(mesh_document(doc.to_dict()))

    assert there.geometric().area == pytest.approx(here.geometric().area)
    assert there.geometric().EIxx_c == pytest.approx(here.geometric().EIxx_c)
    assert there.warping().GJ == pytest.approx(here.warping().GJ, rel=1e-9)
    assert there.plastic().Mp_xx == pytest.approx(here.plastic().Mp_xx)


def test_the_rebuilt_analyzer_is_live_not_a_snapshot_of_numbers():
    """The inspector drives ``analysis.stress(**loads)`` on loads the
    user types. A "return the numbers" split would have lost that, so
    assert the rebuilt object still solves on demand."""
    doc = _plate()
    analysis = doc.analysis_from_fem(mesh_document(doc.to_dict()))
    field = analysis.stress(N=1000.0, Mxx=5.0e5)
    assert field is not None
    assert len(field.von_mises) > 0


def test_materials_are_resolved_in_the_parent():
    """Only the mesh crosses the process boundary; the materials table
    is the parent's business, so a composite still lands with its
    per-region moduli."""
    doc = SectionDocument.new(name="src", kind="continuum")
    doc.set_material("concrete", E=25e3, nu=0.2)
    doc.set_material("steel", E=200e3, nu=0.3)
    doc.add_shape("rect_face", id="concrete", b=6.0, h=6.0)
    doc.add_shape("W_face", id="steel", bf=3.0, tf=0.2, h=3.6, tw=0.12)
    doc.add_embed("concrete", "steel")
    doc.set_mesh(lc=0.5, order=2)

    analysis = doc.analysis_from_fem(mesh_document(doc.to_dict()))
    assert set(analysis.materials) == {"concrete", "steel"}
    # composite: the rigidity form is valid, the unprefixed one is not
    assert analysis.geometric().EIxx_c > 0.0


# ─────────────────────────────────────────────────────────────────────
# failure modes — always a message, never a hang
# ─────────────────────────────────────────────────────────────────────

def test_document_error_from_the_child_carries_its_diagnosis():
    """A document the child cannot mesh reports the child's own message,
    so the panel shows the actionable text rather than 'it failed'."""
    doc = SectionDocument.new(name="nomesh", kind="continuum")
    doc.set_material("s", E=200e3, nu=0.3)
    doc.add_shape("rect_face", id="s", b=1.0, h=1.0, material="s")
    # never called set_mesh(lc=...)
    with pytest.raises(MeshSubprocessError, match="set_mesh"):
        mesh_document(doc.to_dict())


def test_a_wedged_child_is_stopped_rather_than_waited_on():
    with pytest.raises(MeshSubprocessError, match="did not finish"):
        mesh_document(_plate(lc=0.02).to_dict(), timeout=0.05)


def test_a_child_that_dies_without_reporting_is_still_a_message():
    """The abort case this module exists to keep out of the parent: the
    child vanishes mid-mesh and leaves no envelope."""
    import apeGmsh.sections._mesh_proc as mesh_proc

    real_run = mesh_proc.subprocess.run

    class _Died:
        returncode = -11          # SIGSEGV-ish
        stdout = ""
        stderr = "Fatal Python error: Aborted\n"

    def fake_run(*a, **k):
        return _Died()

    mesh_proc.subprocess.run = fake_run
    try:
        with pytest.raises(
            MeshSubprocessError, match="failed without reporting"
        ):
            mesh_document(_plate().to_dict())
    finally:
        mesh_proc.subprocess.run = real_run


def test_build_document_turns_a_child_failure_into_a_grey_panel():
    """End to end through the worker entry point: a bad document greys
    the panel with the reason instead of raising into the thread."""
    from apeGmsh.sections._properties import build_document

    doc = SectionDocument.new(name="nomesh", kind="continuum")
    doc.set_material("s", E=200e3, nu=0.3)
    doc.add_shape("rect_face", id="s", b=1.0, h=1.0, material="s")

    result = build_document(doc.to_dict())
    assert result.analysis is None
    assert result.error is not None and "set_mesh" in result.error


# ─────────────────────────────────────────────────────────────────────
# the child itself
# ─────────────────────────────────────────────────────────────────────

def test_worker_writes_an_envelope_for_a_bad_document(tmp_path):
    """The child never raises out of ``main`` — it reports through the
    envelope, so the parent always has something to read."""
    import io
    import json
    import pickle
    import sys

    from apeGmsh.sections import _mesh_worker

    out = tmp_path / "env.pkl"
    stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps({"not": "a document"}))
    try:
        assert _mesh_worker.main([str(out)]) == 0
    finally:
        sys.stdin = stdin

    payload = pickle.loads(out.read_bytes())
    assert payload["ok"] is False
    assert payload["fem"] is None
    assert "SectionDocumentError" in payload["error"]


def test_worker_rejects_a_bad_invocation():
    from apeGmsh.sections import _mesh_worker

    assert _mesh_worker.main([]) == 2
