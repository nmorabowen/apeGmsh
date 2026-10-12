"""The viewer's orientation source: the ``model.h5`` probe and resolver.

ADR 0020 collapsed the legacy ``model_h5=`` plumbing kwarg: the chain
forward ``Results.model -> OpenSeesModel`` is the primary source of
OpenSees enrichment (beam orientation, cuts), read through the
file-mediated seam against the path the OpenSeesModel was loaded from.

What stays pinned here is the shared gate: ``has_opensees_orientation``
and ``resolve_orientation_source`` (ADR 0026). The tests that built the
legacy ``ResultsViewer`` to exercise its ``_build_viewer_data`` and the
deprecated ``model_h5=`` kwarg went with that viewer (X3-d, #1734).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from apeGmsh.opensees import ModelData
from apeGmsh.viewers.data._h5_probe import has_opensees_orientation

from tests.opensees.fixtures.fem_stub import make_two_node_beam


def _write_oriented_model(
    path: Path,
    *,
    fem,
    vecxz: tuple = (1.0, 0.0, 0.0),
) -> None:
    """Write a model.h5 with /opensees/transforms + /opensees/element_meta."""
    md = ModelData(fem, ndm=3, ndf=6, model_name="oriented")
    md.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=vecxz,
    )
    md.write(str(path))


def _write_bare_model(path: Path, *, fem) -> None:
    """Write a model.h5 WITHOUT the /opensees/ orientation zone."""
    md = ModelData(fem, ndm=3, ndf=6, model_name="bare")
    md.write(str(path))


# =====================================================================
# Phase 5 — h5 probe still works (retained per spec)
# =====================================================================

def test_probe_rejects_nonexistent_file(tmp_path: Path) -> None:
    """The probe's False answer for a missing file is the contract that
    keeps the auto-resolve quiet for in-memory Results (no ``_path``)
    and for files that simply don't exist on disk."""
    assert has_opensees_orientation(tmp_path / "nope.h5") is False


def test_probe_rejects_non_hdf5_file(tmp_path: Path) -> None:
    """A non-HDF5 file (or otherwise unreadable) → False, no raise.
    The caller's contract is "should I auto-resolve?", not "is this
    file healthy?"."""
    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not actually hdf5\n")
    assert has_opensees_orientation(junk) is False


# =====================================================================
# resolve_orientation_source — the centralised gate (PR3 / ADR 0026)
# =====================================================================
#
# Two ResultsViewer call sites previously inlined the same
# ``Path + has_opensees_orientation`` block (the scene snapshot at
# ``_build_viewer_data`` and the cuts auto-load gate at
# ``_apply_pending_cuts``).  Collapsing into one helper preserves the
# probe semantics — these tests pin that contract.

def test_resolve_returns_path_when_orientation_zone_present(tmp_path):
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    fem = make_two_node_beam()
    oriented = tmp_path / "oriented.h5"
    _write_oriented_model(oriented, fem=fem)

    results = SimpleNamespace(_path=oriented)
    source = resolve_orientation_source(results)

    assert source == oriented
    assert isinstance(source, Path)


def test_resolve_returns_none_when_orientation_zone_absent(tmp_path):
    """A bare model.h5 (no /opensees/transforms or /opensees/element_meta)
    yields None — the viewer must degrade to the from_fem path."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    fem = make_two_node_beam()
    bare = tmp_path / "bare.h5"
    _write_bare_model(bare, fem=fem)

    results = SimpleNamespace(_path=bare)

    assert resolve_orientation_source(results) is None


def test_resolve_returns_none_when_results_has_no_path():
    """In-memory Results (``_path is None``) yields None — typical for
    recorder-flavoured Results and for hand-constructed test fixtures."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    results = SimpleNamespace(_path=None)

    assert resolve_orientation_source(results) is None


def test_resolve_returns_none_when_results_path_does_not_exist(tmp_path):
    """A ``_path`` that points at a nonexistent file yields None — the
    probe's "should I auto-resolve?" contract trumps any path-shape
    inference."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    results = SimpleNamespace(_path=tmp_path / "never_written.h5")

    assert resolve_orientation_source(results) is None


def test_resolve_prefers_model_path_for_fem_keyed_reads(tmp_path):
    """A paired open whose element reads relabel into fem_eid space
    (``_element_reads_fem_keyed()`` True) must resolve to
    ``_model_path`` — the scene must be built from the fem_eid-keyed
    model.h5, never from the reader's ops-tag-keyed embedded snapshot.

    The model.h5 here is deliberately BARE (no /opensees zone at all):
    the paired branch must NOT apply the ``has_opensees_orientation``
    gate, because a solid-only model records ``element_meta`` but no
    ``transforms`` group and every consumer degrades gracefully.
    """
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    fem = make_two_node_beam()
    model_h5 = tmp_path / "paired_model.h5"
    _write_bare_model(model_h5, fem=fem)

    results = SimpleNamespace(
        _path=tmp_path / "run.mpco",  # need not exist — never probed
        _model_path=model_h5,
        _element_reads_fem_keyed=lambda: True,
    )
    source = resolve_orientation_source(results)

    assert source == model_h5
    assert isinstance(source, Path)


def test_resolve_ignores_model_path_when_reads_not_fem_keyed(tmp_path):
    """``_model_path`` set but reads stay in the file's own id space
    (no pairing, or a stub pairing that never fires) → the probe falls
    through to the pre-existing ``_path`` gate."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    fem = make_two_node_beam()
    model_h5 = tmp_path / "unpaired_model.h5"
    _write_bare_model(model_h5, fem=fem)

    results = SimpleNamespace(
        _path=None,
        _model_path=model_h5,
        _element_reads_fem_keyed=lambda: False,
    )

    assert resolve_orientation_source(results) is None


def test_resolve_ignores_model_path_that_does_not_exist(tmp_path):
    """A ``_model_path`` pointing at a deleted/moved file falls through
    to the ``_path`` gate instead of returning an unreadable path."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    results = SimpleNamespace(
        _path=None,
        _model_path=tmp_path / "moved_away.h5",
        _element_reads_fem_keyed=lambda: True,
    )

    assert resolve_orientation_source(results) is None


def test_resolve_ignores_model_path_that_is_not_hdf5(tmp_path):
    """A ``_model_path`` whose file no longer opens as HDF5 (corrupted
    or replaced after open) degrades to the fallback instead of letting
    ``ViewerData.from_h5`` raise at scene build."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    corrupt = tmp_path / "corrupt.h5"
    corrupt.write_text("not an hdf5 file")

    results = SimpleNamespace(
        _path=None,
        _model_path=corrupt,
        _element_reads_fem_keyed=lambda: True,
    )

    assert resolve_orientation_source(results) is None


def test_resolve_is_duck_typed_no_results_import():
    """Verify the resolver never imports apeGmsh.results — it operates
    entirely on ``getattr(results, "_path", None)``.  Preserves
    ADR 0014 INV-1: viewers/data/ has no import edge into
    apeGmsh.results.  The probe accepts any object with a _path
    attribute; the test passes a bare SimpleNamespace."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    # No _path attribute at all → also None (getattr default fires).
    assert resolve_orientation_source(SimpleNamespace()) is None
