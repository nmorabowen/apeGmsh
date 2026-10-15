"""#1948 (G8a): stage kinds on MPCO, and the stage in the ``.out`` cache key.

Kinds.  An ``.mpco`` does not tag its stages; ``Results.from_mpco(...,
model_h5=)`` kinds each stage from the archive's own ``analysis`` record
(``/opensees/stages/stage_NNN/analysis@analysis``).  ``analyze_dt`` does
not decide it: a ``Static`` stage may carry ``dt=dlam`` (silent, the
``ops.stage`` docstring's own spelling) and a ``Transient`` one may omit
``dt``.  Oracle: the archived token, ``Static`` -> ``static``,
``Transient`` / ``VariableTransient`` -> ``transient``; anything else, or
no record, raises.

Cache key.  ``Results.from_recorders`` writes the stage name and kind into
the transcoded file, but neither is in a source file, so both are in the
cache key: a second read of the same files under another name or kind is
a new file, not the first read's.  (Multi-stage ``.out`` reads are G8b,
#2035.)
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results._bind import _ANALYSIS_KIND, _bind_stage_names
from apeGmsh.results.readers._mpco import MPCOReader
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

from tests.conftest import _open_model_from_h5, _stub_model_h5_path
from tests.test_results_transcoder import _MockFem, _write_synthetic_node_out


# =====================================================================
# Stage kinds on MPCO (bind beside the stage-name pairing)
# =====================================================================

_IDS = np.array([1, 2], dtype=np.int64)


def _write_two_stage_mpco(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        f.create_group("INFO")
        for k in (1, 2):
            on = f.create_group(f"MODEL_STAGE[{k}]/RESULTS/ON_NODES")
            f.create_group(f"MODEL_STAGE[{k}]/RESULTS/ON_ELEMENTS")
            disp = on.create_group("DISPLACEMENT")
            disp.attrs["COMPONENTS"] = np.array([b"Ux,Uy"])
            disp.create_dataset("ID", data=_IDS.reshape(-1, 1))
            for s in range(3):
                ds = disp.create_dataset(f"DATA/STEP_{s}", data=np.zeros((2, 2)))
                ds.attrs["TIME"] = np.array([0.1 * (s + 1)])
    return path


def _program(*stages: tuple[str, "dict[str, Any]"]) -> Any:
    """An ``OpenSeesModel`` stand-in: ``stages()`` as ``StageRecordRO``s."""
    from apeGmsh.opensees._internal.typed_records import StageRecordRO

    records = [StageRecordRO(name=n, analyze_steps=3, analyze_dt=0.1,
                             chain_attrs=chain) for n, chain in stages]
    return SimpleNamespace(stages=lambda: list(records))


def test_kind_table_covers_every_analysis_the_bridge_emits() -> None:
    from apeGmsh.opensees.analysis import analysis

    assert set(_ANALYSIS_KIND) == set(analysis.__all__)


def test_mpco_stages_take_the_archived_analysis_kind(tmp_path: Path) -> None:
    """Both stages carry ``analyze_dt=0.1``; the analysis record decides."""
    model = _program(("gravity", {"analysis": "Static"}),
                     ("shake", {"analysis": "Transient"}))
    with MPCOReader(_write_two_stage_mpco(tmp_path / "r.mpco")) as r:
        assert [s.kind for s in r.stages()] == ["transient", "transient"]
        _bind_stage_names(r, model, model_path="model.h5")
        stages = r.stages()
    assert [(s.name, s.kind) for s in stages] == [
        ("gravity", "static"), ("shake", "transient")]
    assert [s.aliases for s in stages] == [
        ("MODEL_STAGE[1]",), ("MODEL_STAGE[2]",)]


def test_variable_transient_is_transient(tmp_path: Path) -> None:
    model = _program(("a", {"analysis": "VariableTransient"}),
                     ("b", {"analysis": "Static"}))
    with MPCOReader(_write_two_stage_mpco(tmp_path / "r.mpco")) as r:
        _bind_stage_names(r, model, model_path="model.h5")
        assert [s.kind for s in r.stages()] == ["transient", "static"]


@pytest.mark.parametrize("chain, match", [
    ({"analysis": "PFEM"}, "analysis 'PFEM', which has no stage kind"),
    ({}, "archives no analysis record"),
])
def test_unknown_or_missing_analysis_record_raises(
    tmp_path: Path, chain: "dict[str, Any]", match: str,
) -> None:
    model = _program(("a", {"analysis": "Static"}), ("b", chain))
    with MPCOReader(_write_two_stage_mpco(tmp_path / "r.mpco")) as r:
        with pytest.raises(ValueError, match=match):
            _bind_stage_names(r, model, model_path="model.h5")


def test_attach_refuses_a_kind_that_is_not_a_program_stage_kind(
    tmp_path: Path,
) -> None:
    with MPCOReader(_write_two_stage_mpco(tmp_path / "r.mpco")) as r:
        with pytest.raises(ValueError, match=r"\['mode'\] are not program"):
            r.attach_stage_names(["a", "b"], ["static", "mode"])
        with pytest.raises(ValueError, match="2 names and 1 kinds"):
            r.attach_stage_names(["a", "b"], ["static"])


# =====================================================================
# The stage name and kind are in the .out cache key
# =====================================================================

def _node_spec(fem: Any) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="r", components=("displacement_x",),
            dt=None, n_steps=None, node_ids=np.array([1, 2]),
        ),),
    )


def _read(out: Path, fem: Any, cache: Path, **kw: Any) -> Results:
    return Results.from_recorders(
        _node_spec(fem), out, fem=fem, cache_root=cache,
        model=_open_model_from_h5(_stub_model_h5_path()), **kw)


def _write_gravity(out: Path) -> None:
    out.mkdir(exist_ok=True)
    _write_synthetic_node_out(
        out / "gravity__r_disp.out", time=[0.5, 1.0], node_ids=[1, 2],
        dofs=[1], value_fn=lambda k, nid, dof: 1.0 * nid + k)


def test_the_cache_key_covers_the_stage_kind(tmp_path: Path) -> None:
    """Regression: ``stage_kind=`` was not in the key, so a second read of
    the same files with another kind returned the first read's kind."""
    fem = _MockFem([1, 2])
    _write_gravity(tmp_path / "out")
    for kind in ("static", "transient", "static"):
        with _read(tmp_path / "out", fem, tmp_path / "c",
                   stage_id="gravity", stage_kind=kind) as r:
            assert [(s.name, s.kind) for s in r.stages] == [("gravity", kind)]
            slab = r.nodes.get(component="displacement_x")
        np.testing.assert_allclose(slab.values, [[1.0, 2.0], [2.0, 3.0]])


def test_the_cache_key_covers_the_stage_name(tmp_path: Path) -> None:
    fem = _MockFem([1, 2])
    _write_gravity(tmp_path / "out")
    for name in ("gravity", "dead_load"):
        with _read(tmp_path / "out", fem, tmp_path / "c",
                   stage_id="gravity", stage_name=name) as r:
            assert [s.name for s in r.stages] == [name]


def test_compute_cache_key_separates_stage_labels(tmp_path: Path) -> None:
    from apeGmsh.results.writers._cache import compute_cache_key

    def key(*stages: "tuple[str, str, str | None]") -> str:
        return compute_cache_key([tmp_path / "a.out"], parser_version="1",
                                 fem_snapshot_id="f", stages=stages)

    base = key(("g", "static", "g"))
    assert base == key(("g", "static", "g"))
    others = {key(("g", "transient", "g")), key(("h", "static", "g")),
              key(("g", "static", None)), key(), key(("g", "static", "g"),
                                                    ("g", "static", "g"))}
    assert base not in others and len(others) == 5


