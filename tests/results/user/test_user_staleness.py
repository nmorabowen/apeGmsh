"""The staleness table of ADR 0130 A6, row by row."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.user import (
    Combination,
    Term,
    UserFile,
    UserWriteRefused,
    staleness,
    write_user_file,
)

from .conftest import nodal


def _with_zones(path: Path, *roots: str) -> None:
    with h5py.File(path, "a") as f:
        for root in roots:
            f.create_group(root).create_dataset("x", data=np.array([1.0]))
            f["meta"].attrs[f"{root}_schema_version"] = "1.0.0"


@pytest.fixture
def full(make_model, ladruno, tmp_path: Path) -> Path:
    path = write_user_file(
        tmp_path / "s.user.h5", make_model(), results_file=ladruno,
        results=[nodal("by results", basis="results"), nodal("by model", basis="model")],
        combinations=[
            Combination("stage only", "linear", (Term("stage", "D"),)),
            Combination("cqc", "cqc", (Term("spectrum", "c"),)),
            Combination("via cqc", "linear", (Term("combination", "cqc"), Term("stage", "D"))),
        ],
    )
    _with_zones(path, "modes", "spectra", "entities")
    return path


def _rerun(ladruno: Path) -> None:
    with h5py.File(ladruno, "a") as f:
        data = f["MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA"]
        data[data.shape[0] - 1] = data[data.shape[0] - 1] + 1.0


def test_a_matching_set_is_current(full, make_model, ladruno, tmp_path) -> None:
    got = staleness(full, model_h5=tmp_path / "model.h5", results_file=ladruno)
    assert not any(s.stale for s in got.zones.values()), got.zones
    assert not any(s.stale for s in got.results.values())
    assert got.combinations == ()


def test_a_mismatched_results_hash_stales_only_basis_results(full, ladruno, tmp_path) -> None:
    _rerun(ladruno)
    got = staleness(full, model_h5=tmp_path / "model.h5", results_file=ladruno)
    assert got.results["by results"].stale and "results_hash differs" in got.results["by results"].reason
    assert not got.results["by model"].stale
    assert got.zones["results"].stale
    for root in ("modes", "spectra", "entities", "meta", "combinations"):
        assert not got.zones[root].stale, root
    assert got.combinations == ()


def test_stamping_the_ladruno_does_not_stale_anything(full, ladruno, tmp_path) -> None:
    with h5py.File(ladruno, "a") as f:
        f["INFO/PROVENANCE"].attrs["fem_hash"] = "stamped later"
    got = staleness(full, model_h5=tmp_path / "model.h5", results_file=ladruno)
    assert not got.results["by results"].stale


@pytest.mark.parametrize(
    "kw, word",
    [({"fem": "other"}, "fem_hash"), ({"model": "other"}, "model_hash")],
)
def test_differing_model_hashes_stale_the_model_zones(full, make_model, ladruno, kw, word) -> None:
    other = make_model("other.h5", **kw)
    got = staleness(full, model_h5=other, results_file=ladruno)
    for root in ("modes", "spectra", "entities"):
        assert got.zones[root].stale and word in got.zones[root].reason
    assert got.results["by model"].stale
    assert not got.results["by results"].stale
    assert not got.zones["meta"].stale and not got.zones["combinations"].stale


def test_a_combination_never_goes_stale_itself_but_lists_a_stale_source(full, make_model, ladruno) -> None:
    got = staleness(full, model_h5=make_model("other.h5", fem="other"), results_file=ladruno)
    assert not got.zones["combinations"].stale
    assert got.combinations == ("cqc", "via cqc"), "direct and through a combination"
    table = UserFile(full).combinations()
    assert len(table.rows) == 3, "authored rows stay in force"


def test_model_hash_compares_only_when_both_sides_have_it(make_model, tmp_path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal()])
    only_fem = make_model("fem_only.h5", model=None)
    assert not staleness(path, model_h5=only_fem).results[nodal().name].stale
    path2 = write_user_file(tmp_path / "t.user.h5", only_fem, results=[nodal()])
    assert not staleness(path2, model_h5=make_model("m2.h5")).results[nodal().name].stale
    assert staleness(path2, model_h5=make_model("m3.h5", fem="x")).results[nodal().name].stale


def test_what_cannot_be_compared_is_stale(full, tmp_path: Path) -> None:
    got = staleness(full)
    assert got.results["by results"].stale and "no results file" in got.results["by results"].reason
    assert got.results["by model"].stale and "no model file" in got.results["by model"].reason
    gone = staleness(full, model_h5=tmp_path / "absent.h5", results_file=tmp_path / "absent.ladruno")
    assert gone.results["by results"].stale and gone.results["by model"].stale


def test_a_refused_zone_reports_stale_instead_of_raising(full, ladruno, tmp_path) -> None:
    with h5py.File(full, "a") as f:
        f["meta"].attrs["combinations_schema_version"] = "9.0.0"
        f["meta"].attrs["user_results_schema_version"] = "9.0.0"
    got = staleness(full, model_h5=tmp_path / "model.h5", results_file=ladruno)
    assert got.zones["combinations"].stale and "refused" in got.zones["combinations"].reason
    assert got.zones["results"].stale and "refused" in got.zones["results"].reason


# ---- a write that moves a hash must say what happens to the old values ------


def test_a_new_results_hash_refuses_to_keep_basis_results_entries(full, make_model, ladruno, tmp_path) -> None:
    _rerun(ladruno)
    before = full.read_bytes()
    with pytest.raises(UserWriteRefused, match="would become stale"):
        write_user_file(full, tmp_path / "model.h5", results_file=ladruno)
    assert full.read_bytes() == before


def test_rewriting_the_entry_in_the_same_call_is_allowed(full, ladruno, tmp_path) -> None:
    _rerun(ladruno)
    fresh = nodal("by results", basis="results", data=np.ones((2, 3, 2)))
    write_user_file(full, tmp_path / "model.h5", results_file=ladruno, results=[fresh])
    got = staleness(full, model_h5=tmp_path / "model.h5", results_file=ladruno)
    assert not got.results["by results"].stale
    assert np.all(UserFile(full).read_result("by results").data == 1.0)


def test_drop_stale_drops_only_what_the_hash_change_stales(full, ladruno, tmp_path) -> None:
    _rerun(ladruno)
    write_user_file(full, tmp_path / "model.h5", results_file=ladruno, drop_stale=True)
    uf = UserFile(full)
    assert [e.name for e in uf.results().entries] == ["by model"]
    assert {"modes", "spectra", "entities"} <= uf.zones
    assert len(uf.combinations().rows) == 3


def test_a_new_model_refuses_then_drop_stale_drops_derived_keeps_authored(full, make_model, ladruno) -> None:
    other = make_model("other.h5", fem="other")
    with pytest.raises(UserWriteRefused, match="modes"):
        write_user_file(full, other)
    write_user_file(full, other, drop_stale=True)
    uf = UserFile(full)
    assert uf.fem_hash == "other"
    assert not ({"modes", "spectra", "entities"} & uf.zones)
    assert [e.name for e in uf.results().entries] == ["by results"]
    assert len(uf.combinations().rows) == 3, "authored combinations reference states by name"
    with h5py.File(full, "r") as f:
        assert "modes_schema_version" not in f["meta"].attrs


def test_basis_results_without_any_hash_refuses(make_model, tmp_path: Path) -> None:
    with pytest.raises(UserWriteRefused, match="results_file"):
        write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal(basis="results")])
