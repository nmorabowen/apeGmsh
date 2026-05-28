"""Opt-in ``tolerance=`` on the solid boolean ops.

The curve-side booleans (``make_conformal``) and the dedup path already
exposed a ``Geometry.ToleranceBoolean`` lever via ``_temporary_tolerance``;
the *solid* booleans (``g.model.boolean.fuse/cut/intersect/fragment`` and
the geometry ``slice`` / ``cut_by_surface`` / ``cut_by_plane`` wrappers)
ran at the global default with no escape hatch for near-coincident faces.

These tests pin the new opt-in ``tolerance=`` param: it is applied to
``Geometry.ToleranceBoolean`` *during* the OCC op, *restored* afterwards,
and is a no-op (default ``None``) on clean geometry. The mechanism is
asserted by spying on the underlying ``gmsh.model.occ`` call rather than
relying on OCC's version-sensitive numerical coincidence behaviour.
"""
from __future__ import annotations

import gmsh
import pytest


_TOL_KEY = "Geometry.ToleranceBoolean"


def _tol() -> float:
    return gmsh.option.getNumber(_TOL_KEY)


def _spy_on(monkeypatch, occ_fn_name: str, sink: dict) -> None:
    """Wrap ``gmsh.model.occ.<occ_fn_name>`` to record the live boolean
    tolerance at call time, then delegate to the real function."""
    real = getattr(gmsh.model.occ, occ_fn_name)

    def spy(*args, **kwargs):
        sink["tol_during"] = gmsh.option.getNumber(_TOL_KEY)
        return real(*args, **kwargs)

    monkeypatch.setattr(gmsh.model.occ, occ_fn_name, spy)


def test_fuse_applies_and_restores_tolerance(g, monkeypatch) -> None:
    """``boolean.fuse(..., tolerance=X)`` sets ToleranceBoolean=X during
    the OCC fuse and restores the prior global value afterwards."""
    sink: dict = {}
    _spy_on(monkeypatch, "fuse", sink)

    before = _tol()
    a = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    b = g.model.geometry.add_box(1, 0, 0, 1, 1, 1)
    g.model.boolean.fuse([a], [b], tolerance=1e-3)

    assert sink["tol_during"] == pytest.approx(1e-3)
    assert _tol() == pytest.approx(before)


def test_fragment_applies_and_restores_tolerance(g, monkeypatch) -> None:
    """``boolean.fragment(..., tolerance=X)`` routes the override through
    ``_bool_op`` to the OCC fragment call."""
    sink: dict = {}
    _spy_on(monkeypatch, "fragment", sink)

    before = _tol()
    a = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    b = g.model.geometry.add_box(0.5, 0, 0, 1, 1, 1)
    g.model.boolean.fragment([a], [b], tolerance=2e-3)

    assert sink["tol_during"] == pytest.approx(2e-3)
    assert _tol() == pytest.approx(before)


def test_slice_threads_tolerance_to_occ_fragment(g, monkeypatch) -> None:
    """The geometry ``slice`` wrapper (which calls ``occ.fragment``
    directly via ``cut_by_surface``) honours ``tolerance=``."""
    sink: dict = {}
    _spy_on(monkeypatch, "fragment", sink)

    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.model.geometry.slice(box, axis="z", offset=0.5, tolerance=5e-4)

    assert sink["tol_during"] == pytest.approx(5e-4)


def test_tolerance_none_is_noop_on_clean_geometry(g) -> None:
    """Default ``tolerance=None`` leaves the global tolerance untouched
    and produces the correct fragment count on clean geometry; an
    explicit tolerance does not change the (correct) result either."""
    before = _tol()

    box1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    none_pieces = g.model.geometry.slice(box1, axis="z", offset=0.5)
    assert len(none_pieces) == 2
    assert _tol() == pytest.approx(before)  # None never touched it

    # Offset differs from box1's cut so the global coincident-face
    # advisory (which scans every surface in the shared model) stays quiet.
    box2 = g.model.geometry.add_box(5, 0, 0, 1, 1, 1)
    tol_pieces = g.model.geometry.slice(box2, axis="z", offset=0.3, tolerance=1e-6)
    assert len(tol_pieces) == 2          # tolerance harmless on clean geom
    assert _tol() == pytest.approx(before)  # restored after
