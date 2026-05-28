"""CAD-import health: non-mutating diagnose(), scale-aware heal, parts heal.

Covers the three OCC-headroom additions:
  A — g.model.io.diagnose() reports health without mutating, and a raw
      import emits a WarnGeomImportHealth advisory when slivers appear.
  B — heal=True / heal="auto" derive a scale-aware tolerance from the
      model bbox instead of the legacy absolute 1e-8.
  C — g.parts.import_step exposes heal/dedupe (the assembly path).
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest

from apeGmsh.core._geometry_errors import WarnGeomImportHealth
from apeGmsh.core._model_io import (
    _IO,
    ImportHealth,
    _model_bbox_diag,
    _suggested_heal_tolerance,
)


# ── B: scale-aware tolerance helper (pure) ───────────────────────────

def test_suggested_heal_tolerance_scales_with_model() -> None:
    assert _suggested_heal_tolerance(0.0) == 1e-8       # empty -> legacy
    assert _suggested_heal_tolerance(1_000.0) == pytest.approx(1e-3)
    assert _suggested_heal_tolerance(1.0) == pytest.approx(1e-6)


# ── A: diagnose() is non-mutating and sane on clean geometry ─────────

def test_diagnose_clean_box(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    report = g.model.io.diagnose()
    assert isinstance(report, ImportHealth)
    assert report.n_solids == 1
    assert not report.is_suspect              # a clean box has no slivers
    assert report.short_edges == ()
    assert report.tiny_faces == ()
    # OCC's getBoundingBox is slightly padded vs the exact extent, so
    # compare with a loose relative tolerance.
    diag = math.dist((0, 0, 0), (1, 1, 1))
    assert report.bbox_diag == pytest.approx(diag, rel=1e-4)
    assert report.suggested_tolerance == pytest.approx(1e-6 * diag, rel=1e-4)


def test_diagnose_does_not_mutate(g) -> None:
    import gmsh
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    counts_before = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
    g.model.io.diagnose(warn=True)
    counts_after = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
    assert counts_before == counts_after      # nothing added/removed


def test_diagnose_warn_silent_on_clean_geometry(g) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGeomImportHealth)
        g.model.io.diagnose(warn=True)         # must NOT raise


# ── B + A via the import path (STEP round-trip) ──────────────────────

def _save_box_step(g, tmp_path: Path, *, size: float = 10.0) -> Path:
    g.model.geometry.add_box(0, 0, 0, size, size, size)
    out = tmp_path / "box.step"
    g.model.io.save_step(str(out))
    return out


def test_load_step_heal_auto_uses_scale_aware_tolerance(
    g, tmp_path: Path, monkeypatch,
) -> None:
    """heal='auto' passes a scale-derived tolerance (>> the legacy 1e-8)
    to heal_shapes."""
    step = _save_box_step(g, tmp_path, size=10.0)

    captured: dict = {}
    real = _IO.heal_shapes

    def spy(self, *args, **kwargs):
        captured["tol"] = kwargs.get("tolerance")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(_IO, "heal_shapes", spy)
    result = g.model.io.load_step(str(step), heal="auto")

    assert 3 in result                          # a solid was imported
    assert captured["tol"] is not None
    assert captured["tol"] > 1e-8               # scale-aware, not legacy


def test_load_step_heal_true_is_also_scale_aware(
    g, tmp_path: Path, monkeypatch,
) -> None:
    """heal=True now means the same scale-aware auto (semantics change
    from the legacy fixed 1e-8)."""
    step = _save_box_step(g, tmp_path, size=10.0)
    captured: dict = {}
    real = _IO.heal_shapes
    monkeypatch.setattr(
        _IO, "heal_shapes",
        lambda self, *a, **k: (captured.setdefault("tol", k.get("tolerance")),
                               real(self, *a, **k))[1],
    )
    g.model.io.load_step(str(step), heal=True)
    assert captured["tol"] > 1e-8


def test_load_step_raw_clean_emits_no_advisory(g, tmp_path: Path) -> None:
    """A clean raw import (no heal) produces no WarnGeomImportHealth."""
    step = _save_box_step(g, tmp_path, size=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGeomImportHealth)
        g.model.io.load_step(str(step))         # clean -> no advisory


# ── C: parts.import_step heal/dedupe ─────────────────────────────────

def test_parts_import_step_accepts_heal_auto(g, tmp_path: Path) -> None:
    step = _save_box_step(g, tmp_path, size=5.0)
    inst = g.parts.import_step(str(step), label="block", heal="auto")
    assert inst is not None
    assert "block" in g.parts._instances


def test_parts_import_step_raw_still_works(g, tmp_path: Path) -> None:
    step = _save_box_step(g, tmp_path, size=5.0)
    inst = g.parts.import_step(str(step), label="block")   # default raw
    assert inst is not None
