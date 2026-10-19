"""#1911 — ``assess`` names the opensees 2.32.0 side-channel hash change.

``RES.LINEAGE`` says the ``model_hash`` mismatch likely spans the 2.32.0
step when ``Results.lineage`` names it (``SIDE_CHANNEL_HASH_CAUSE``): both
``program_decl`` markers agree, both sides carry side-channel calls, and
exactly one side's calls carry their declaration ids. A straddling pair
with no side-channel call, and a same-version mismatch, keep the plain
count. The pairs are built as in ``tests/results/test_lineage_hash_era.py``.
"""
from __future__ import annotations

from pathlib import Path

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.results import Results
from apeGmsh.results.Results import (
    MODEL_HASH_ERA_CAUSE,
    SIDE_CHANNEL_DECL_FROM,
    SIDE_CHANNEL_HASH_CAUSE,
)
from tests.results.test_lineage_hash_era import (
    _as_below_step,
    _results,
    _side_channel_model,
)


def _lineage_finding(results_path: Path, model_path: Path):
    model = OpenSeesModel.from_h5(str(model_path))
    with Results.from_native(results_path, model=model) as r:
        report = r.assess()
    finding = next(f for f in report.findings if f.code == "RES.LINEAGE")
    assert finding.severity == "warning"
    return finding, report


def test_assess_names_the_side_channel_step(tmp_path: Path) -> None:
    fresh = _side_channel_model(tmp_path)
    results = _results(_as_below_step(fresh, tmp_path), tmp_path)
    finding, report = _lineage_finding(results, fresh)
    step = ".".join(str(part) for part in SIDE_CHANNEL_DECL_FROM)
    assert (
        f"likely spans the opensees schema {step} {SIDE_CHANNEL_HASH_CAUSE}, "
        "not corruption" in finding.message
    )
    assert MODEL_HASH_ERA_CAUSE not in finding.message
    assert SIDE_CHANNEL_HASH_CAUSE in report.text


def test_assess_names_the_step_for_step_results_on_below_step_model(
    tmp_path: Path,
) -> None:
    fresh = _side_channel_model(tmp_path)
    results = _results(fresh, tmp_path)
    finding, _ = _lineage_finding(results, _as_below_step(fresh, tmp_path))
    assert SIDE_CHANNEL_HASH_CAUSE in finding.message


def test_assess_straddling_pair_without_side_channel_unchanged(
    tmp_path: Path,
) -> None:
    plain = _side_channel_model(tmp_path, zero=False)
    results = _results(_as_below_step(plain, tmp_path), tmp_path,
                       other_model=True)
    finding, _ = _lineage_finding(results, plain)
    assert finding.detail is not None
    assert finding.message == f"{finding.detail['n']} lineage warning(s)"


def test_assess_same_version_side_channel_mismatch_unchanged(
    tmp_path: Path,
) -> None:
    fresh = _side_channel_model(tmp_path)
    results = _results(fresh, tmp_path, other_model=True)
    finding, _ = _lineage_finding(results, fresh)
    assert finding.detail is not None
    assert finding.message == f"{finding.detail['n']} lineage warning(s)"
