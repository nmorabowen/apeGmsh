"""#1854 — a ``model_hash`` mismatch across the 2.31.0 hash change names it.

At opensees 2.31.0 (A7 #1839, ``h5_reader.PROGRAM_DECL_FROM``) the
``/opensees/program.decl`` column began to carry declaration ids, and it
is hashed: every re-emitted archive's ``model_hash`` changed once. A
results file written against an older archive, opened with the re-saved
model, therefore warns "model hash mismatch". When the two sides'
opensees-zone versions straddle the boundary the warning names that
cause. It stays a warning (ADR 0021 INV-2). A same-era mismatch, or one
where either version cannot be read, keeps the old text exactly.

The pair is built from one current-era archive: the results file's
stored ``model_hash`` and its opensees stamps (root + ``/model/meta``)
are rewritten to what an older run left behind.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.lineage import WARNING_PREFIX, LineageError
from apeGmsh.opensees._internal.schema_version import SchemaVersion
from apeGmsh.results import Results
from apeGmsh.results.Results import MODEL_HASH_ERA_CAUSE
from apeGmsh.results.writers import NativeWriter

from tests.fixtures.schema import (
    OPENSEES_CURRENT,
    OPENSEES_PRIOR_MINOR,
    OPENSEES_PROGRAM_DECL_FROM,
)
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_h5

_OLD_HASH = "0" * 32
_KEY = "opensees_schema_version"


def _pair(
    tmp_path: Path, *, results_version: str | None,
) -> tuple[Path, Path]:
    """A model.h5 and a results.h5 whose stored ``model_hash`` differs.

    ``results_version`` restamps the results file's opensees version
    (root and ``/model/meta``); ``None`` deletes both stamps.
    """
    model_path, fem = build_simple_frame_h5(tmp_path)
    results_path = tmp_path / "run.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(name="g", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((1, node_ids.size))},
        )
        w.end_stage()
    with h5py.File(results_path, "r+") as f:
        f["meta/lineage"].attrs["model_hash"] = _OLD_HASH
        for attrs in (f.attrs, f["model/meta"].attrs):
            if results_version is None:
                if _KEY in attrs:
                    del attrs[_KEY]
            else:
                attrs[_KEY] = results_version
    return model_path, results_path


def _restamp_model(model_path: Path, version: str) -> None:
    with h5py.File(model_path, "r+") as f:
        f["meta"].attrs[_KEY] = version


def _mismatch(results_path: Path, model_path: Path) -> tuple[str, str]:
    """The model-hash-mismatch warning, and the model's hash."""
    model = OpenSeesModel.from_h5(model_path)
    with Results.from_native(results_path, model=model) as r:
        lineage = r.lineage
    hits = [w for w in lineage.warnings if "model hash mismatch" in w]
    assert len(hits) == 1, lineage.warnings
    assert model.lineage.model_hash
    return hits[0], model.lineage.model_hash


def _old_text(model_hash: str) -> str:
    return (
        WARNING_PREFIX
        + "model hash mismatch between Results and OpenSeesModel: "
        f"results-file stamp {_OLD_HASH!r} != "
        f"OpenSeesModel.lineage.model_hash {model_hash!r}"
    )


def test_fixture_versions_straddle_the_boundary() -> None:
    def key(s: str) -> tuple[int, int, int]:
        v = SchemaVersion.parse(s)
        return (v.major, v.minor, v.patch)

    assert (
        key(OPENSEES_PRIOR_MINOR)
        < key(OPENSEES_PROGRAM_DECL_FROM)
        <= key(OPENSEES_CURRENT)
    )


def test_old_results_with_resaved_model_names_the_cause(tmp_path: Path) -> None:
    model_path, results_path = _pair(
        tmp_path, results_version=OPENSEES_PRIOR_MINOR,
    )
    warning, model_hash = _mismatch(results_path, model_path)
    assert warning.startswith(_old_text(model_hash) + "; ")
    assert MODEL_HASH_ERA_CAUSE in warning
    assert f"schema {OPENSEES_PROGRAM_DECL_FROM} " in warning
    assert f"written against {OPENSEES_PRIOR_MINOR}" in warning
    assert f"model {OPENSEES_CURRENT}" in warning
    assert "re-run the analysis against the re-saved model" in warning


def test_new_results_with_old_model_names_the_cause(tmp_path: Path) -> None:
    model_path, results_path = _pair(
        tmp_path, results_version=OPENSEES_CURRENT,
    )
    _restamp_model(model_path, OPENSEES_PRIOR_MINOR)
    warning, model_hash = _mismatch(results_path, model_path)
    assert warning.startswith(_old_text(model_hash) + "; ")
    assert MODEL_HASH_ERA_CAUSE in warning
    assert "open the results with the model.h5 they were run on" in warning


def test_cross_era_stays_a_warning(tmp_path: Path) -> None:
    model_path, results_path = _pair(
        tmp_path, results_version=OPENSEES_PRIOR_MINOR,
    )
    model = OpenSeesModel.from_h5(model_path)
    with Results.from_native(results_path, model=model) as r:
        lineage = r.lineage  # never raises (INV-2)
    with pytest.raises(LineageError):
        lineage.assert_clean()


def test_same_era_mismatch_keeps_the_old_text(tmp_path: Path) -> None:
    model_path, results_path = _pair(
        tmp_path, results_version=OPENSEES_CURRENT,
    )
    warning, model_hash = _mismatch(results_path, model_path)
    assert warning == _old_text(model_hash)


def test_same_old_era_mismatch_keeps_the_old_text(tmp_path: Path) -> None:
    model_path, results_path = _pair(
        tmp_path, results_version=OPENSEES_PRIOR_MINOR,
    )
    _restamp_model(model_path, OPENSEES_PRIOR_MINOR)
    warning, model_hash = _mismatch(results_path, model_path)
    assert warning == _old_text(model_hash)


def test_unreadable_results_version_keeps_the_old_text(tmp_path: Path) -> None:
    model_path, results_path = _pair(tmp_path, results_version=None)
    # With no opensees stamp the reader falls back to the root envelope
    # (the results zone's own version) and flags the embedded zone; the
    # era check must not read that envelope as an opensees version.
    with pytest.warns(UserWarning, match="embedded opensees zone is unavailable"):
        warning, model_hash = _mismatch(results_path, model_path)
    assert warning == _old_text(model_hash)


def test_assess_names_the_cause_across_eras(tmp_path: Path) -> None:
    model_path, results_path = _pair(
        tmp_path, results_version=OPENSEES_PRIOR_MINOR,
    )
    model = OpenSeesModel.from_h5(model_path)
    with Results.from_native(results_path, model=model) as r:
        report = r.assess()
    finding = next(f for f in report.findings if f.code == "RES.LINEAGE")
    assert finding.severity == "warning"
    assert MODEL_HASH_ERA_CAUSE in finding.message
    assert "not corruption" in finding.message
    assert MODEL_HASH_ERA_CAUSE in report.text


def test_assess_same_era_message_unchanged(tmp_path: Path) -> None:
    model_path, results_path = _pair(
        tmp_path, results_version=OPENSEES_CURRENT,
    )
    model = OpenSeesModel.from_h5(model_path)
    with Results.from_native(results_path, model=model) as r:
        report = r.assess()
    finding = next(f for f in report.findings if f.code == "RES.LINEAGE")
    assert finding.severity == "warning"
    n = finding.detail["n"]
    assert finding.message == f"{n} lineage warning(s)"
