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

The legacy archive is the corpus file of the minor before the marker; a "different
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
from apeGmsh.results.Results import (
    MODEL_HASH_ERA_CAUSE,
    SIDE_CHANNEL_DECL_FROM,
    SIDE_CHANNEL_DECL_VERBS,
    SIDE_CHANNEL_HASH_CAUSE,
    _side_channel_hash_cause,
    _side_channel_proof,
)
from apeGmsh.results.writers import NativeWriter

from tests.fixtures.schema import OPENSEES_PROGRAM_DECL_FROM
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
    """The corpus archive of the minor before the marker (no marker: old
    hash)."""
    major, below, _patch = OPENSEES_PROGRAM_DECL_FROM.split(".")
    minor = f"{major}.{int(below) - 1}"
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


# --- #1911: the 2.32.0 side-channel step --------------------------------------
#
# At 2.32.0 (#1853) the six side-channel verbs' calls began to carry their
# declaration in the hashed /opensees/program.decl column. Both sides'
# program_decl markers read "complete" there, so the rule above is blind
# to it. The cause is named when both sides carry side-channel calls and
# exactly one side's calls are proved (decl >= 0): the hashed column
# itself, so a to_h5 rewrite of a 2.31 archive (stamped current, old
# column, old hash) counts as the 2.31 side.
#
# A "2.31 emit" of the model is its current emit with every side-channel
# run's decl set to -1, stamped one minor below the step, and its
# model_hash re-stamped: the one difference #1853's golden diff shows.


def _side_channel_model(tmp_path: Path, *, zero: bool = True) -> Path:
    """A current emit whose one stage zeroes two nodes (set_node_vel /
    set_node_accel, side-channel calls); ``zero=False`` drops them."""
    from apeGmsh.opensees import apeSees
    from tests.opensees.h5.test_h5_stages_reader import (
        _chain,
        build_two_quad_fem,
    )

    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="static") as s:
        if zero:
            s.zero_velocities(nodes=[3, 4])
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    sub = tmp_path / ("zero" if zero else "plain")
    sub.mkdir()
    path = sub / "model.h5"
    ops.h5(str(path))
    return path


def _below_step() -> str:
    major, minor, _patch = SIDE_CHANNEL_DECL_FROM
    return f"{major}.{minor - 1}.0"


def _side_channel_runs(path: Path) -> list[int]:
    """The decl of each side-channel run of ``path``'s program."""
    model = OpenSeesModel.from_h5(str(path))
    return [r.decl for r in model.program()
            if r.method in SIDE_CHANNEL_DECL_VERBS]


def _as_below_step(src: Path, tmp_path: Path) -> Path:
    """``src`` as the emit one minor below the step would have written it."""
    from apeGmsh.opensees._internal.lineage import compute_model_hash

    out = tmp_path / f"{src.parent.name}_below.h5"
    shutil.copy(src, out)
    with h5py.File(out, "r+") as f:
        ds = f["opensees/program"]
        methods = [m.decode() if isinstance(m, bytes) else str(m)
                   for m in ds.attrs["methods"]]
        table = ds[()]
        for i, code in enumerate(table["method"]):
            if methods[int(code)] in SIDE_CHANNEL_DECL_VERBS:
                table["decl"][i] = -1
        ds[...] = table
        f["meta"].attrs["opensees_schema_version"] = _below_step()
        lin = f["meta/lineage"]
        lin.attrs["model_hash"] = compute_model_hash(
            str(lin.attrs["fem_hash"]), f["opensees"])
    return out


def _clean_model(path: Path) -> OpenSeesModel:
    model = OpenSeesModel.from_h5(str(path))
    assert model.lineage.warnings == (), model.lineage.warnings
    return model


def _side_channel_line(remedy: str) -> str:
    step = ".".join(str(part) for part in SIDE_CHANNEL_DECL_FROM)
    return (
        f"; likely cause: the opensees schema {step} "
        f"{SIDE_CHANNEL_HASH_CAUSE}; {remedy}"
    )


_RESAVED = ("re-run the analysis against the re-saved model, or open the "
            "results with the model.h5 they were run on")
_ORIGINAL = ("open the results with the model.h5 they were run on, or "
             "re-run the analysis against this model")


def test_fixture_differs_only_in_the_side_channel_column(
    tmp_path: Path,
) -> None:
    """The oracle's premise: the current emit proves every side-channel
    call, the below-step copy proves none, both are self-consistent and
    their hashes differ; with no side-channel call the copy's hash is
    the current one (nothing else changed at the step)."""
    fresh = _side_channel_model(tmp_path)
    below = _as_below_step(fresh, tmp_path)
    assert _side_channel_runs(fresh)
    assert all(d >= 0 for d in _side_channel_runs(fresh))
    assert all(d == -1 for d in _side_channel_runs(below))
    assert (_clean_model(fresh).lineage.model_hash
            != _clean_model(below).lineage.model_hash)
    plain = _side_channel_model(tmp_path, zero=False)
    assert _side_channel_runs(plain) == []
    assert (_clean_model(plain).lineage.model_hash
            == _clean_model(_as_below_step(plain, tmp_path)).lineage.model_hash)


def test_below_step_results_with_resaved_model_names_the_step(
    tmp_path: Path,
) -> None:
    fresh = _side_channel_model(tmp_path)
    results = _results(_as_below_step(fresh, tmp_path), tmp_path)
    warning, stored, model_hash = _mismatch(results, fresh)
    assert warning == _old_text(stored, model_hash) + _side_channel_line(
        _RESAVED)
    assert MODEL_HASH_ERA_CAUSE not in warning


def test_step_results_with_below_step_model_names_the_step(
    tmp_path: Path,
) -> None:
    fresh = _side_channel_model(tmp_path)
    results = _results(fresh, tmp_path)
    warning, stored, model_hash = _mismatch(
        results, _as_below_step(fresh, tmp_path))
    assert warning == _old_text(stored, model_hash) + _side_channel_line(
        _ORIGINAL)


def test_straddling_pair_without_side_channel_keeps_the_old_text(
    tmp_path: Path,
) -> None:
    """Stamps straddle the step and the markers agree, but neither side
    carries a side-channel call: the step changed nothing there."""
    plain = _side_channel_model(tmp_path, zero=False)
    results = _results(_as_below_step(plain, tmp_path), tmp_path,
                       other_model=True)
    warning, stored, model_hash = _mismatch(results, plain)
    assert warning == _old_text(stored, model_hash)


def test_same_version_side_channel_mismatch_keeps_the_old_text(
    tmp_path: Path,
) -> None:
    fresh = _side_channel_model(tmp_path)
    results = _results(fresh, tmp_path, other_model=True)
    warning, stored, model_hash = _mismatch(results, fresh)
    assert warning == _old_text(stored, model_hash)


def test_below_step_rewrite_keeps_its_hash(tmp_path: Path) -> None:
    """Seam: a to_h5 rewrite of a below-step archive is stamped current
    but echoes the old column, so it keeps the old hash and its own
    results pair with it cleanly."""
    below = _as_below_step(_side_channel_model(tmp_path), tmp_path)
    rewrite = _rewrite(below, tmp_path)
    assert all(d == -1 for d in _side_channel_runs(rewrite))
    model = _clean_model(rewrite)
    with Results.from_native(_results(below, tmp_path), model=model) as r:
        lineage = r.lineage
    assert not [w for w in lineage.warnings if "model hash mismatch" in w]


def test_below_step_rewrite_with_other_below_step_results_keeps_the_old_text(
    tmp_path: Path,
) -> None:
    """Seam: the stamps straddle the step (rewrite current, results below)
    and the model carries a side-channel call, yet neither hash holds the
    ids. A stamp rule would blame the step; the column does not."""
    below = _as_below_step(_side_channel_model(tmp_path), tmp_path)
    rewrite = _rewrite(below, tmp_path)
    results = _results(below, tmp_path, other_model=True)
    warning, stored, model_hash = _mismatch(results, rewrite)
    assert warning == _old_text(stored, model_hash)


def test_results_of_a_rewrite_with_resaved_model_names_the_step(
    tmp_path: Path,
) -> None:
    """Seam: results run on the rewrite carry its current stamp but the
    old column; against the re-emitted model the step is the cause."""
    fresh = _side_channel_model(tmp_path)
    rewrite = _rewrite(_as_below_step(fresh, tmp_path), tmp_path)
    results = _results(rewrite, tmp_path)
    warning, stored, model_hash = _mismatch(results, fresh)
    assert warning == _old_text(stored, model_hash) + _side_channel_line(
        _RESAVED)


def test_step_verbs_are_the_emitters_side_channel_verbs() -> None:
    """The step's verbs are the six #1853 proved. When the emitter's set
    grows, that later change is a new marker value, not this rule
    (#1911 item 3): update the marker, not SIDE_CHANNEL_DECL_VERBS."""
    from apeGmsh.opensees.emitter.h5 import _SIDE_CHANNEL_DECL_VERBS

    assert SIDE_CHANNEL_DECL_VERBS == _SIDE_CHANNEL_DECL_VERBS


@pytest.mark.parametrize(("results_proof", "model_proof"), [
    (None, None), (None, True), (False, None), (True, True), (False, False),
])
def test_no_side_channel_cause_unless_exactly_one_side_proves(
    results_proof: bool | None, model_proof: bool | None,
) -> None:
    assert _side_channel_hash_cause(results_proof, model_proof) is None


def test_side_channel_proof_reads_only_the_step_verbs() -> None:
    assert _side_channel_proof([("fix", 3), ("element", 0)]) is None
    assert _side_channel_proof([("fix", 3), ("set_node_vel", -1)]) is False
    assert _side_channel_proof(
        [("set_node_vel", -1), ("set_node_accel", 2)]) is True
    # decl 0 is the first /opensees/decls row: a proof, not "none".
    assert _side_channel_proof([("set_node_vel", 0)]) is True


def test_marker_cause_takes_precedence_over_the_side_channel_step(
    tmp_path: Path,
) -> None:
    """Results from the pre-marker corpus file (side-channel calls, all
    decl -1) against a current emit with proved side-channel calls: the
    markers disagree, so the marker cause is named exactly, and the
    side-channel step, which the columns alone would also suggest, is
    not."""
    legacy = _legacy(tmp_path)
    runs = _side_channel_runs(legacy)
    assert runs and all(d == -1 for d in runs)
    results = _results(legacy, tmp_path)
    warning, stored, model_hash = _mismatch(
        results, _side_channel_model(tmp_path))
    assert warning == (
        _old_text(stored, model_hash)
        + f"; likely cause: the opensees schema {OPENSEES_PROGRAM_DECL_FROM} "
        f"{MODEL_HASH_ERA_CAUSE} (the model's hash includes declaration "
        "ids, the results file's does not); re-run the analysis against "
        "the re-saved model, or open the results with the model.h5 they "
        "were run on"
    )
    assert SIDE_CHANNEL_HASH_CAUSE not in warning
