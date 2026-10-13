"""#1854 — a ``model_hash`` mismatch across the 2.31.0 hash change names it.

At opensees 2.31.0 (A7 #1839, ``h5_reader.PROGRAM_DECL_FROM``) the
``/opensees/program.decl`` column began to carry declaration ids, and it
is hashed: every re-emitted archive's ``model_hash`` changed once. A
results file written against an older archive, opened with the re-saved
model, therefore warns "model hash mismatch". When one side's hash
includes the ids and the other's does not, the warning names that cause.
Each side's era is its ``/opensees/decls@program_decl`` completeness
marker, not its version stamp: a ``to_h5`` rewrite of an older archive
is stamped current but keeps the old hash (marker ``"absent"``).

It stays a warning (ADR 0021 INV-2). A same-era mismatch, or one whose
results file carries no ``/opensees`` zone, keeps the old text exactly.

The legacy archive is the prior-minor schema corpus file; a "different
model" is a results file whose stored ``model_hash`` is rewritten.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.lineage import WARNING_PREFIX, LineageError
from apeGmsh.results import Results
from apeGmsh.results.Results import MODEL_HASH_ERA_CAUSE
from apeGmsh.results.writers import NativeWriter

from tests.fixtures.schema import OPENSEES_PRIOR_MINOR, OPENSEES_PROGRAM_DECL_FROM
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_h5

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "schema_corpus"
_OTHER_HASH = "0" * 32


def _fresh(tmp_path: Path) -> Path:
    """A current-era emit (marker ``"complete"``)."""
    sub = tmp_path / "fresh"
    sub.mkdir()
    path, _fem = build_simple_frame_h5(sub)
    return path


def _legacy(tmp_path: Path) -> Path:
    """The prior-minor corpus archive (no marker: old hash)."""
    minor = OPENSEES_PRIOR_MINOR.rsplit(".", 1)[0]
    path = tmp_path / "legacy.h5"
    shutil.copy(CORPUS / f"opensees_{minor}.h5", path)
    return path


def _rewrite(src: Path, tmp_path: Path) -> Path:
    """``from_h5 -> to_h5``: stamped current, keeps its source's hash."""
    out = tmp_path / f"{src.stem}_rewrite.h5"
    OpenSeesModel.from_h5(str(src)).to_h5(str(out))
    return out


def _results(
    model_path: Path, tmp_path: Path, *, other_model: bool = False,
) -> Path:
    """A results file run against ``model_path``.

    ``other_model`` rewrites the stored ``model_hash``, as if the run
    used a genuinely different model of the same era.
    """
    fem = OpenSeesModel.from_h5(str(model_path)).fem
    results_path = tmp_path / f"run_{model_path.stem}.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(name="g", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((1, node_ids.size))},
        )
        w.end_stage()
    if other_model:
        with h5py.File(results_path, "r+") as f:
            f["meta/lineage"].attrs["model_hash"] = _OTHER_HASH
    return results_path


def _mismatch(results_path: Path, model_path: Path) -> tuple[str, str, str]:
    """The model-hash-mismatch warning, the stored and the model's hash."""
    model = OpenSeesModel.from_h5(str(model_path))
    with Results.from_native(results_path, model=model) as r:
        lineage = r.lineage
    with h5py.File(results_path, "r") as f:
        stored = str(f["meta/lineage"].attrs["model_hash"])
    hits = [w for w in lineage.warnings if "model hash mismatch" in w]
    assert len(hits) == 1, lineage.warnings
    assert model.lineage.model_hash
    return hits[0], stored, model.lineage.model_hash


def _old_text(stored: str, model_hash: str) -> str:
    return (
        WARNING_PREFIX
        + "model hash mismatch between Results and OpenSeesModel: "
        f"results-file stamp {stored!r} != "
        f"OpenSeesModel.lineage.model_hash {model_hash!r}"
    )


def _assert_cause(warning: str, stored: str, model_hash: str) -> None:
    assert warning.startswith(_old_text(stored, model_hash) + "; ")
    assert MODEL_HASH_ERA_CAUSE in warning
    assert f"schema {OPENSEES_PROGRAM_DECL_FROM} " in warning


# --- cross-era: the cause is named ------------------------------------------


def test_old_results_with_resaved_model_names_the_cause(tmp_path: Path) -> None:
    results = _results(_legacy(tmp_path), tmp_path)
    warning, stored, model_hash = _mismatch(results, _fresh(tmp_path))
    _assert_cause(warning, stored, model_hash)
    assert "the model's hash includes declaration ids" in warning
    assert "re-run the analysis against the re-saved model" in warning


def test_new_results_with_old_model_names_the_cause(tmp_path: Path) -> None:
    results = _results(_fresh(tmp_path), tmp_path)
    warning, stored, model_hash = _mismatch(results, _legacy(tmp_path))
    _assert_cause(warning, stored, model_hash)
    assert "the results file's hash includes declaration ids" in warning
    assert "open the results with the model.h5 they were run on" in warning


def test_rewrite_results_with_fresh_emit_names_the_cause(tmp_path: Path) -> None:
    """Both sides are stamped current, yet only the fresh emit's hash
    includes the ids: the version stamp alone would name no cause."""
    rewrite = _rewrite(_legacy(tmp_path), tmp_path)
    results = _results(rewrite, tmp_path)
    warning, stored, model_hash = _mismatch(results, _fresh(tmp_path))
    _assert_cause(warning, stored, model_hash)


def test_cross_era_stays_a_warning(tmp_path: Path) -> None:
    results = _results(_legacy(tmp_path), tmp_path)
    model = OpenSeesModel.from_h5(str(_fresh(tmp_path)))
    with Results.from_native(results, model=model) as r:
        lineage = r.lineage  # never raises (INV-2)
    with pytest.raises(LineageError):
        lineage.assert_clean()


# --- same era, or unknown: the old text, exactly ----------------------------


def test_legacy_rewrite_with_other_legacy_results_keeps_the_old_text(
    tmp_path: Path,
) -> None:
    """The rewrite is stamped current but keeps the old hash; results
    from a genuinely different old-era model must not blame the hash
    change (the version stamps differ, the markers agree)."""
    legacy = _legacy(tmp_path)
    rewrite = _rewrite(legacy, tmp_path)
    results = _results(legacy, tmp_path, other_model=True)
    warning, stored, model_hash = _mismatch(results, rewrite)
    assert warning == _old_text(stored, model_hash)


def test_same_new_era_mismatch_keeps_the_old_text(tmp_path: Path) -> None:
    fresh = _fresh(tmp_path)
    results = _results(fresh, tmp_path, other_model=True)
    warning, stored, model_hash = _mismatch(results, fresh)
    assert warning == _old_text(stored, model_hash)


def test_results_without_opensees_zone_keeps_the_old_text(
    tmp_path: Path,
) -> None:
    results = _results(_legacy(tmp_path), tmp_path)
    with h5py.File(results, "r+") as f:
        del f["opensees"]
    warning, stored, model_hash = _mismatch(results, _fresh(tmp_path))
    assert warning == _old_text(stored, model_hash)


# --- assess ------------------------------------------------------------------


def test_assess_names_the_cause_across_eras(tmp_path: Path) -> None:
    results = _results(_legacy(tmp_path), tmp_path)
    model = OpenSeesModel.from_h5(str(_fresh(tmp_path)))
    with Results.from_native(results, model=model) as r:
        report = r.assess()
    finding = next(f for f in report.findings if f.code == "RES.LINEAGE")
    assert finding.severity == "warning"
    assert MODEL_HASH_ERA_CAUSE in finding.message
    assert "not corruption" in finding.message
    assert MODEL_HASH_ERA_CAUSE in report.text


def test_assess_same_era_message_unchanged(tmp_path: Path) -> None:
    fresh = _fresh(tmp_path)
    results = _results(fresh, tmp_path, other_model=True)
    model = OpenSeesModel.from_h5(str(fresh))
    with Results.from_native(results, model=model) as r:
        report = r.assess()
    finding = next(f for f in report.findings if f.code == "RES.LINEAGE")
    assert finding.severity == "warning"
    assert finding.detail is not None
    n = finding.detail["n"]
    assert finding.message == f"{n} lineage warning(s)"
