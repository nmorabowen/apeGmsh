"""#2100: ``.ladruno`` readers name and kind their stages as MPCO's do.

``results/_bind.py::_bind_stage_names`` pairs the archive's program
stage names and kinds onto a reader's non-mode stages and calls
``reader.attach_stage_names(names, kinds)``.  The MPCO pair
(``MPCOReader`` / ``MPCOMultiPartitionReader``) is the oracle: the same
stage list, written once as ``.mpco`` and once as ``.ladruno``, must come
out with the same ids, names, kinds and aliases, and the same refusals.

One structural difference is pinned here: a ``.ladruno`` stage records
its ``KIND`` and an ``.mpco`` stage does not.  On the pinned fork the
``KIND`` is the kind the stage ran (fork #986), so it is evidence: where
it disagrees with the program kind the pairing is misaligned, and per
#2079 Q7 ("map program stage names from ``/opensees`` when the count and
kinds match; otherwise keep the file's names and warn") nothing is
renamed and :class:`StageKindMismatchWarning` lists the conflicts.  The
raw ``KIND`` is compared: ``eigen`` is written on every stage of a
recorder that asks for modes, so it is no evidence, like an absent
``KIND``, and the program kind fills in for both.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import warnings

import h5py
import numpy as np
import pytest

from apeGmsh.results._bind import _bind_stage_names
from apeGmsh.results.readers._ladruno import (
    LadrunoReader,
    StageKindMismatchWarning,
)
from apeGmsh.results.readers._ladruno_multi import (
    LadrunoMultiPartitionReader,
    StageOrderMatchWarning,
)
from apeGmsh.results.readers._mpco import MPCOReader
from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader

_IDS = np.array([1, 2], dtype=np.int32)
_NAMES = ["gravity", "push", "shake"]
_KINDS = ["static", "static", "transient"]


def _write(
    path: Path, fmt: str, *, stamps=(1, 2, 3), kinds=None,
    mode_in: "int | None" = None, part: "tuple[int, int] | None" = None,
) -> Path:
    """Stages ``MODEL_STAGE[<stamp>]`` with a two-step 2-D DISPLACEMENT.

    ``mode_in`` (an index into ``stamps``) adds one recorded mode to that
    stage, which every reader lists as its own ``kind="mode"`` stage.
    ``kinds`` sets each ``.ladruno`` stage's ``KIND`` attribute (``None``
    writes none); by default it is ``_KINDS``, with ``eigen`` on the stage
    that holds the mode.  ``part`` is ``(partition_id, num_partitions)``.
    """
    if kinds is None:
        kinds = ["eigen" if k == mode_in else _KINDS[k]
                 for k in range(len(stamps))]
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        if fmt == "ladruno":
            info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
            info.attrs["FORMAT_VERSION"] = 1
            info.attrs["SPATIAL_DIM"] = 2
            if part is not None:
                info.attrs["PARTITIONED"] = 1
                info.attrs["PARTITION_ID"] = part[0]
                info.attrs["NUM_PARTITIONS"] = part[1]
        for k, stamp in enumerate(stamps):
            stage = f.create_group(f"MODEL_STAGE[{stamp}]")
            if fmt == "ladruno" and kinds[k] is not None:
                stage.attrs["KIND"] = np.bytes_(kinds[k].encode())
            nodes = stage.create_group("MODEL/NODES")
            nodes.create_dataset("ID", data=_IDS.reshape(-1, 1))
            nodes.create_dataset("COORDINATES", data=np.zeros((2, 2)))
            stage.create_group("MODEL/ELEMENTS")
            on = stage.create_group("RESULTS/ON_NODES")
            stage.create_group("RESULTS/ON_ELEMENTS")
            disp = on.create_group("DISPLACEMENT")
            disp.attrs["COMPONENTS"] = np.array([b"Ux,Uy"])
            disp.create_dataset("ID", data=_IDS.reshape(-1, 1))
            if fmt == "ladruno":
                disp.create_dataset("DATA", data=np.zeros((2, 2, 2)))
                disp.create_dataset("TIME", data=np.array([0.5, 1.0]))
                disp.create_dataset("STEP", data=np.array([1, 2]))
            else:
                for s in range(2):
                    ds = disp.create_dataset(f"DATA/STEP_{s}",
                                             data=np.zeros((2, 2)))
                    ds.attrs["TIME"] = np.array([0.5 * (s + 1)])
            if mode_in == k:
                modes = on.create_group("MODES_OF_VIBRATION(U)")
                modes.attrs["COMPONENTS"] = np.array([b"Ux,Uy"])
                modes.create_dataset("ID", data=_IDS.reshape(-1, 1))
                ds = modes.create_dataset("DATA/STEP_0/MODE_0",
                                          data=np.ones((2, 2)))
                ds.attrs["LAMBDA"] = np.array([4.0])
    return path


def _view(reader: Any) -> "list[tuple[str, str, str, tuple[str, ...]]]":
    return [(s.id, s.name, s.kind, s.aliases) for s in reader.stages()]


def _program(*stages: "tuple[str, str]") -> Any:
    """An ``OpenSeesModel`` stand-in, as ``_bind_stage_names`` reads it."""
    from apeGmsh.opensees._internal.typed_records import StageRecordRO

    records = [StageRecordRO(name=n, analyze_steps=2, analyze_dt=0.5,
                             chain_attrs={"analysis": a}) for n, a in stages]
    return SimpleNamespace(stages=lambda: list(records))


# =====================================================================
# Single file
# =====================================================================

@pytest.mark.parametrize("mode_in", [None, 0, 1])
def test_single_file_matches_mpco(tmp_path: Path, mode_in) -> None:
    """Same stages, same names, kinds and aliases; modes are untouched."""
    with MPCOReader(_write(tmp_path / "r.mpco", "mpco",
                           mode_in=mode_in)) as m, \
            LadrunoReader(_write(tmp_path / "r.ladruno", "ladruno",
                                 mode_in=mode_in)) as lad:
        m.attach_stage_names(_NAMES, _KINDS)
        lad.attach_stage_names(_NAMES, _KINDS)
        got, want = _view(lad), _view(m)
    assert got == want
    hosts = [v for v in got if v[2] != "mode"]
    assert [(v[1], v[2], v[3]) for v in hosts] == [
        ("gravity", "static", ("MODEL_STAGE[1]",)),
        ("push", "static", ("MODEL_STAGE[2]",)),
        ("shake", "transient", ("MODEL_STAGE[3]",)),
    ]
    modes = [v for v in got if v[2] == "mode"]
    assert len(modes) == (0 if mode_in is None else 1)
    assert all(v[3] == () for v in modes)


_FILE_NAMES = ["MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[3]"]


def test_agreeing_kinds_rename_silently(tmp_path: Path) -> None:
    with LadrunoReader(_write(tmp_path / "r.ladruno", "ladruno")) as lad:
        with warnings.catch_warnings():
            warnings.simplefilter("error", StageKindMismatchWarning)
            lad.attach_stage_names(_NAMES, _KINDS)
        assert [(s.name, s.kind) for s in lad.stages()] == list(
            zip(_NAMES, _KINDS))


def test_misordered_pairing_warns_and_keeps_the_file_names(
    tmp_path: Path,
) -> None:
    """Program stages in the wrong order against the recorded KINDs."""
    with LadrunoReader(_write(tmp_path / "r.ladruno", "ladruno")) as lad:
        with pytest.warns(StageKindMismatchWarning,
                          match=r"MODEL_STAGE\[1\] records KIND 'static' "
                                r"but program stage 'shake' is 'transient'"):
            lad.attach_stage_names(_NAMES[::-1], _KINDS[::-1])
        assert [(s.name, s.kind, s.aliases) for s in lad.stages()] == [
            (n, k, ()) for n, k in zip(_FILE_NAMES, _KINDS)]


def test_recorded_kind_is_not_overridden(tmp_path: Path) -> None:
    """Inverts 4804fa31's override: a recorded static/transient stands."""
    path = _write(tmp_path / "r.ladruno", "ladruno",
                  kinds=["transient", "eigen", "static"])
    with LadrunoReader(path) as lad:
        with pytest.warns(StageKindMismatchWarning) as rec:
            lad.attach_stage_names(_NAMES, _KINDS)
        msg = str(rec[0].message)
        assert "MODEL_STAGE[1] records KIND 'transient'" in msg
        assert "MODEL_STAGE[3] records KIND 'static'" in msg
        assert "MODEL_STAGE[2]" not in msg      # eigen is no evidence
        assert [(s.name, s.kind) for s in lad.stages()] == list(
            zip(_FILE_NAMES, ["transient", "static", "static"]))


def test_eigen_is_no_evidence(tmp_path: Path) -> None:
    """A transient stage recorded beside modes reads ``eigen``: no refusal."""
    path = _write(tmp_path / "r.ladruno", "ladruno",
                  kinds=["static", "eigen", "static"])
    with LadrunoReader(path) as lad:
        with warnings.catch_warnings():
            warnings.simplefilter("error", StageKindMismatchWarning)
            lad.attach_stage_names(_NAMES, ["static", "transient", "static"])
        assert [(s.name, s.kind) for s in lad.stages()] == [
            ("gravity", "static"), ("push", "transient"), ("shake", "static")]


def test_mode_host_paired_with_a_transient_program_stage_warns(
    tmp_path: Path,
) -> None:
    """A misaligned pairing puts the transient program stage on the stage
    that holds the modes (recorded ``static``); the mode stage, found by
    its modal group and not by KIND, is never paired."""
    path = _write(tmp_path / "r.ladruno", "ladruno", mode_in=0,
                  kinds=["static", "static", "transient"])
    with LadrunoReader(path) as lad:
        before = _view(lad)
        assert [v[2] for v in before] == [
            "static", "mode", "static", "transient"]
        with pytest.warns(StageKindMismatchWarning,
                          match=r"MODEL_STAGE\[1\] records KIND 'static' "
                                r"but program stage 'shake' is 'transient'"):
            lad.attach_stage_names(["shake", "push", "gravity"],
                                   ["transient", "static", "static"])
        assert _view(lad) == before


def test_unrecognised_kind_is_a_conflict(tmp_path: Path) -> None:
    path = _write(tmp_path / "r.ladruno", "ladruno",
                  kinds=["static", "buckling", "transient"])
    with LadrunoReader(path) as lad:
        with pytest.warns(StageKindMismatchWarning, match="'buckling'"):
            lad.attach_stage_names(_NAMES, _KINDS)
        assert [s.name for s in lad.stages()] == _FILE_NAMES


def test_program_kind_fills_a_stage_without_kind(tmp_path: Path) -> None:
    """Older builds wrote no KIND: the program kind fills in, no warning."""
    path = _write(tmp_path / "r.ladruno", "ladruno",
                  kinds=[None, "static", None])
    with LadrunoReader(path) as lad:
        assert [s.kind for s in lad.stages()] == ["static"] * 3
        with warnings.catch_warnings():
            warnings.simplefilter("error", StageKindMismatchWarning)
            lad.attach_stage_names(_NAMES, _KINDS)
        assert [(s.name, s.kind) for s in lad.stages()] == list(
            zip(_NAMES, _KINDS))


def test_stage_ids_survive_and_reads_still_key_on_them(tmp_path: Path) -> None:
    with LadrunoReader(_write(tmp_path / "r.ladruno", "ladruno")) as lad:
        before = [s.id for s in lad.stages()]
        lad.attach_stage_names(_NAMES, _KINDS)
        assert [s.id for s in lad.stages()] == before
        np.testing.assert_allclose(lad.time_vector("stage_2"), [0.5, 1.0])


@pytest.mark.parametrize("names, kinds, match", [
    (["a", "b"], ["static", "static"], "2 names and 2 kinds for 3"),
    (["a", "b", "c"], ["static"], "3 names and 1 kinds for 3"),
    (["a", "b", "c"], ["static", "mode", "eigen"],
     r"\['eigen', 'mode'\] are not program"),
])
def test_refusals_match_mpco(tmp_path: Path, names, kinds, match) -> None:
    """Same refusal, same message bar the file name; nothing is renamed."""
    errors = {}
    for fmt, cls in (("mpco", MPCOReader), ("ladruno", LadrunoReader)):
        with cls(_write(tmp_path / f"r.{fmt}", fmt, mode_in=1)) as r:
            before = _view(r)
            with pytest.raises(ValueError, match=match) as exc:
                r.attach_stage_names(names, kinds)
            assert _view(r) == before
            errors[fmt] = str(exc.value).replace(f"r.{fmt}", "r.<ext>")
    assert errors["ladruno"] == errors["mpco"]


def test_bind_stage_names_drives_the_ladruno_reader(tmp_path: Path) -> None:
    """The caller X wires (ADR 0122 slice 3b) names ``.ladruno`` stages."""
    model = _program(("gravity", "Static"), ("push", "Static"),
                     ("shake", "Transient"))
    with MPCOReader(_write(tmp_path / "r.mpco", "mpco", mode_in=0)) as m, \
            LadrunoReader(_write(tmp_path / "r.ladruno", "ladruno",
                                 mode_in=0)) as lad:
        _bind_stage_names(m, model, model_path="model.h5")
        _bind_stage_names(lad, model, model_path="model.h5")
        assert _view(lad) == _view(m)
        assert [s.name for s in lad.stages() if s.kind != "mode"] == _NAMES


# =====================================================================
# Partitioned
# =====================================================================

def _parts(tmp_path: Path, fmt: str, n: int = 2, **kw: Any) -> "list[Path]":
    stamps = kw.pop("stamps", None)
    return [
        _write(tmp_path / f"r.part-{i}.{fmt}", fmt,
               part=(i, n), stamps=(stamps[i] if stamps else (1, 2, 3)), **kw)
        for i in range(n)
    ]


@pytest.mark.parametrize("mode_in", [None, 1])
def test_partitioned_matches_mpco(tmp_path: Path, mode_in) -> None:
    with MPCOMultiPartitionReader(
            _parts(tmp_path, "mpco", mode_in=mode_in)) as m, \
            LadrunoMultiPartitionReader(
                _parts(tmp_path, "ladruno", mode_in=mode_in)) as lad:
        m.attach_stage_names(_NAMES, _KINDS)
        lad.attach_stage_names(_NAMES, _KINDS)
        assert _view(lad) == _view(m)
        # Every part is renamed, not only the one ``stages()`` answers from.
        for child in lad._readers:
            assert [s.name for s in child.stages()
                    if s.kind != "mode"] == _NAMES


def test_partitioned_refusal_matches_mpco(tmp_path: Path) -> None:
    with MPCOMultiPartitionReader(_parts(tmp_path, "mpco")) as m, \
            LadrunoMultiPartitionReader(_parts(tmp_path, "ladruno")) as lad:
        for r in (m, lad):
            with pytest.raises(ValueError, match="1 names and 1 kinds for 3"):
                r.attach_stage_names(["a"], ["static"])
        assert [s.name for s in lad.stages()] == [
            "MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[3]"]


def test_parts_stamped_differently_take_one_name(tmp_path: Path) -> None:
    """A rank-local stamp (WP-165) keeps its own ``MODEL_STAGE`` alias."""
    paths = _parts(tmp_path, "ladruno", stamps=[(1, 2, 3), (1, 4, 5)])
    with pytest.warns(StageOrderMatchWarning):
        lad = LadrunoMultiPartitionReader(paths)
    with lad:
        lad.attach_stage_names(_NAMES, _KINDS)
        assert [(s.name, s.kind) for s in lad.stages()] == list(
            zip(_NAMES, _KINDS))
        assert [s.aliases for s in lad._readers[1].stages()] == [
            ("MODEL_STAGE[1]",), ("MODEL_STAGE[4]",), ("MODEL_STAGE[5]",)]


def test_one_part_in_conflict_renames_no_part(tmp_path: Path) -> None:
    """The KIND verdict is taken once, over every part."""
    paths = _parts(tmp_path, "ladruno")
    with h5py.File(paths[1], "r+") as f:   # part 1 is the only witness
        f["MODEL_STAGE[3]"].attrs["KIND"] = np.bytes_(b"static")
    with h5py.File(paths[0], "r+") as f:   # part 0 records no KIND there
        del f["MODEL_STAGE[3]"].attrs["KIND"]
    with LadrunoMultiPartitionReader(paths) as lad:
        with pytest.warns(StageKindMismatchWarning) as rec:
            lad.attach_stage_names(_NAMES, _KINDS)
        assert len(rec) == 1 and "r.part-1.ladruno" in str(rec[0].message)
        for child in lad._readers:
            assert [s.name for s in child.stages()] == _FILE_NAMES
