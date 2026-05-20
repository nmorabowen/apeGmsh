"""Parity test for the P2 seam (ADR 0018) — ``ResultsViewer``'s
auto-resolve of ``results._path`` into the OpenSees orientation zone,
through to ``ViewerData.from_h5`` and the per-element ``vecxz`` the
beam-orientation overlays consume.

End-to-end recipe:

1. Build a minimal FEM via :func:`make_two_node_beam`.
2. Write a ``model.h5`` via :class:`apeGmsh.opensees.ModelData` —
   the vanilla-OpenSees declarative writer. (``apeSees(fem).h5()`` would
   produce a byte-equivalent zone per ADR 0018 INV-16; one writer is
   enough to exercise the seam.)
3. Construct a :class:`ResultsViewer` against a Results stub whose
   ``_path`` points at the file. Asserts ``_effective_model_h5`` was
   auto-resolved.
4. Take the same path the branched scene builder would, via
   ``ViewerData.from_h5(...)``, and assert ``view.elements.vecxz_for(eid)``
   returns the injected vecxz keyed by FEM element id.

Plus the negative-case for the graceful default fallback (no zone →
auto-resolve returns None → scene takes the ``from_fem`` branch).

The viewer is *constructed*, not shown — we exercise the resolver
without spinning up Qt or OpenGL, mirroring the headless-verify
posture in scope-doc §5 and memory ``feedback_viewer_no_gpu``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from apeGmsh.opensees import ModelData
from apeGmsh.viewers.data import ViewerData
from apeGmsh.viewers.data._h5_probe import has_opensees_orientation
from apeGmsh.viewers.results_viewer import ResultsViewer

from tests.opensees.fixtures.fem_stub import make_two_node_beam


class _ResultsStub:
    """Minimal Results stand-in for ``ResultsViewer.__init__``.

    Reads:
      - ``results.fem``  (must be non-None; the guard at
        ``ResultsViewer.__init__`` raises otherwise).
      - ``results._path`` (the resolver's auto-fallback source).

    Nothing else is touched until ``show()`` runs.
    """

    def __init__(self, *, path: Optional[Path], fem: object) -> None:
        self._path = path
        self.fem = fem


def test_viewer_auto_resolves_vecxz_from_model_h5(tmp_path: Path) -> None:
    """ModelData-produced ``model.h5`` → ``ResultsViewer`` auto-resolves
    from ``results._path`` → branched scene builder routes through
    ``ViewerData.from_h5`` → ``view.elements.vecxz_for(eid)`` recovers
    the injected vecxz keyed by FEM element id (ADR 0018 P2)."""
    fem = make_two_node_beam()
    md = ModelData(fem, ndm=3, ndf=6, model_name="auto")
    md.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    out = tmp_path / "auto.h5"
    md.write(str(out))

    # Probe agrees the file is an orientation source.
    assert has_opensees_orientation(out)

    # ResultsViewer auto-resolves _effective_model_h5 from results._path.
    stub = _ResultsStub(path=out, fem=fem)
    viewer = ResultsViewer(stub)  # __init__ only — no show()
    assert viewer._effective_model_h5 == out

    # Same path the branched scene builder takes
    # (``results_viewer.py:_show_impl`` after C2). Assert the resulting
    # snapshot carries the injected vecxz keyed by FEM element id 1.
    view = ViewerData.from_h5(str(viewer._effective_model_h5))
    assert view.elements.has_vecxz
    np.testing.assert_allclose(view.elements.vecxz_for(1), [1.0, 0.0, 0.0])


def test_viewer_falls_back_when_no_orientation_zone(tmp_path: Path) -> None:
    """Negative: a model.h5 with no ``/opensees/transforms`` → probe is
    False → ``_effective_model_h5`` is None → the branched scene
    builder takes the live ``from_fem`` path; ``vecxz_for`` returns
    None and ``has_vecxz`` is False (ADR 0018 INV-11 graceful default).
    """
    fem = make_two_node_beam()
    md = ModelData(fem, ndm=3, ndf=6, model_name="bare")
    # NO oriented_elements call → the neutral zone is written, the
    # /opensees/transforms + /opensees/element_meta groups are not.
    out = tmp_path / "bare.h5"
    md.write(str(out))

    # Probe says no.
    assert not has_opensees_orientation(out)

    # Resolver returns None; the scene-build branch keeps from_fem.
    stub = _ResultsStub(path=out, fem=fem)
    viewer = ResultsViewer(stub)
    assert viewer._effective_model_h5 is None

    # The from_h5 reader on the same file also degrades gracefully:
    # zero vecxz entries → has_vecxz False, vecxz_for None.
    view = ViewerData.from_h5(str(out))
    assert not view.elements.has_vecxz
    assert view.elements.vecxz_for(1) is None


def test_explicit_model_h5_kwarg_overrides_auto_resolve(tmp_path: Path) -> None:
    """Explicit ``model_h5=`` wins unconditionally — even when the
    results file itself carries the orientation zone (covers the
    'custom layout' user override path the BLUE/RED decision served)."""
    fem = make_two_node_beam()

    # Two model.h5 files: one with vecxz=(1,0,0), one with vecxz=(0,1,0).
    md1 = ModelData(fem, ndm=3, ndf=6, model_name="a")
    md1.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    file_a = tmp_path / "a.h5"
    md1.write(str(file_a))

    md2 = ModelData(fem, ndm=3, ndf=6, model_name="b")
    md2.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(0.0, 1.0, 0.0),
    )
    file_b = tmp_path / "b.h5"
    md2.write(str(file_b))

    # Results points at file A, but the user explicitly passes file B.
    stub = _ResultsStub(path=file_a, fem=fem)
    viewer = ResultsViewer(stub, model_h5=file_b)

    # Explicit kwarg wins.
    assert viewer._effective_model_h5 == file_b
    view = ViewerData.from_h5(str(viewer._effective_model_h5))
    np.testing.assert_allclose(view.elements.vecxz_for(1), [0.0, 1.0, 0.0])


def test_probe_rejects_nonexistent_file(tmp_path: Path) -> None:
    """The probe's ``False`` answer for a missing file is the contract
    that keeps the auto-resolve quiet for in-memory Results (no
    ``_path``) and for files that simply don't exist on disk."""
    assert has_opensees_orientation(tmp_path / "nope.h5") is False


def test_probe_rejects_non_hdf5_file(tmp_path: Path) -> None:
    """A non-HDF5 file (or otherwise unreadable) → False, no raise.
    The caller's contract is 'should I auto-resolve?', not 'is this
    file healthy?'."""
    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not actually hdf5\n")
    assert has_opensees_orientation(junk) is False


def test_resolver_returns_none_for_in_memory_results() -> None:
    """A Results with ``_path=None`` (in-memory) yields no auto-fallback;
    the scene takes the live ``from_fem`` path. Explicit ``model_h5=``
    still wins, of course."""
    fem = make_two_node_beam()

    # in-memory: _path is None.
    stub = _ResultsStub(path=None, fem=fem)
    viewer = ResultsViewer(stub)
    assert viewer._effective_model_h5 is None
