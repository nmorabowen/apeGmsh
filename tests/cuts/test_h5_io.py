"""Round-trip tests for ``apeGmsh.cuts._h5_io`` (v4-1).

Exercises :func:`write_cuts_into` and :func:`read_cuts_and_sweeps`
against fresh tmp model.h5 fixtures. Tests are deliberately
pinhole-scoped on the I/O primitive — they don't drive
``apeSees.h5`` (v4-2) or ``persist_to_h5`` (v4-3), which land in
separate commits.

The minimal fixture is the smallest ``model.h5`` the reference reader
accepts: a single ``/meta/schema_version`` attr with a 2.x.y string.
That's all :func:`read_cuts_and_sweeps` needs to walk the rest of the
file.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.cuts import SectionCutDef, SectionSweepDef
from apeGmsh.cuts._h5_io import (
    persist_to_h5,
    read_cuts_and_sweeps,
    write_cuts_into,
)
from apeGmsh.opensees.emitter.h5_reader import SchemaVersionError


# --------------------------------------------------------------------- #
# Fixture builder — minimal model.h5 the reference reader accepts
# --------------------------------------------------------------------- #
def _make_h5_with_meta(
    path: Path, *, schema_version: str = "2.6.0",
) -> None:
    """Create a minimal ``model.h5`` carrying just ``/meta/schema_version``.

    Mirrors the convention in ``test_tag_map.py``: only what the
    reference reader requires, nothing more. The schema version
    defaults to ``2.6.0`` (within the two-version window for the
    current 2.7.0 reader); tests can pass other in-window values.
    """
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs.create(
            "schema_version", schema_version,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


# --------------------------------------------------------------------- #
# Empty input
# --------------------------------------------------------------------- #
def test_write_empty_is_noop(tmp_path: Path) -> None:
    """No cuts and no sweeps → neither group is created."""
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f)  # nothing to write

    cuts, sweeps = read_cuts_and_sweeps(path)
    assert cuts == ()
    assert sweeps == ()

    # Also assert the file is byte-shape unchanged at the group level.
    with h5py.File(path, "r") as f:
        assert "opensees/cuts" not in f
        assert "opensees/sweeps" not in f


# --------------------------------------------------------------------- #
# Round-trip — single cut
# --------------------------------------------------------------------- #
def test_roundtrip_full_shape_cut(tmp_path: Path) -> None:
    """Every SectionCutDef field populated → round-trip preserves equality."""
    cut = SectionCutDef(
        plane_point=(1.0, 2.0, 3.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(10, 20, 30, 40),
        side="negative",
        label="story 3 base shear",
        bounding_polygon=(
            (0.0, 0.0, 3.0),
            (5.0, 0.0, 3.0),
            (5.0, 5.0, 3.0),
            (0.0, 5.0, 3.0),
        ),
    )
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f, cuts=[cut])

    cuts, sweeps = read_cuts_and_sweeps(path)
    assert sweeps == ()
    assert len(cuts) == 1
    assert cuts[0] == cut


def test_roundtrip_minimal_cut(tmp_path: Path) -> None:
    """label=None and bounding_polygon=None → round-trip preserves None.

    The ``has_label`` / ``has_bounding`` flags are the critical bit —
    without them HDF5's lack of a native None would force ``""`` and
    an empty array back at read time.
    """
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(1.0, 0.0, 0.0),
        element_ids=(7,),
        # side defaults to "positive"
        # label=None, bounding_polygon=None
    )
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f, cuts=[cut])

    cuts, sweeps = read_cuts_and_sweeps(path)
    assert len(cuts) == 1
    restored = cuts[0]
    assert restored.label is None
    assert restored.bounding_polygon is None
    assert restored == cut


# --------------------------------------------------------------------- #
# Round-trip — sweep
# --------------------------------------------------------------------- #
def test_roundtrip_sweep_3(tmp_path: Path) -> None:
    """Sweep of 3 cuts → order preserved through round-trip."""
    cut0 = SectionCutDef(
        plane_point=(0.0, 0.0, 1.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1, 2),
        label="story 1",
    )
    cut1 = SectionCutDef(
        plane_point=(0.0, 0.0, 2.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(3, 4),
        label="story 2",
    )
    cut2 = SectionCutDef(
        plane_point=(0.0, 0.0, 3.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(5, 6),
        label="story 3",
    )
    sweep = SectionSweepDef(cuts=(cut0, cut1, cut2))

    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f, sweeps=[sweep])

    cuts, sweeps_out = read_cuts_and_sweeps(path)
    assert cuts == ()
    assert len(sweeps_out) == 1
    restored = sweeps_out[0]
    assert len(restored) == 3
    assert restored[0] == cut0
    assert restored[1] == cut1
    assert restored[2] == cut2
    assert restored == sweep


def test_standalone_and_sweep_coexist(tmp_path: Path) -> None:
    """Two standalone cuts + one sweep with two cuts → reader partitions.

    Standalone cuts live under ``/opensees/cuts/`` and sweep cuts
    live under ``/opensees/sweeps/sweep_0/cuts/`` — they don't bleed
    into each other.
    """
    standalone_a = SectionCutDef(
        plane_point=(0.0, 0.0, 1.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(100,),
        label="standalone A",
    )
    standalone_b = SectionCutDef(
        plane_point=(0.0, 0.0, 2.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(200,),
        label="standalone B",
    )
    sweep_cut_0 = SectionCutDef(
        plane_point=(0.0, 0.0, 10.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1000,),
        label="sweep cut 0",
    )
    sweep_cut_1 = SectionCutDef(
        plane_point=(0.0, 0.0, 20.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(2000,),
        label="sweep cut 1",
    )
    sweep = SectionSweepDef(cuts=(sweep_cut_0, sweep_cut_1))

    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(
            f,
            cuts=[standalone_a, standalone_b],
            sweeps=[sweep],
        )

    cuts, sweeps_out = read_cuts_and_sweeps(path)
    assert len(cuts) == 2
    assert cuts[0] == standalone_a
    assert cuts[1] == standalone_b
    assert len(sweeps_out) == 1
    assert sweeps_out[0] == sweep


# --------------------------------------------------------------------- #
# Forward-compat with pre-v4 files
# --------------------------------------------------------------------- #
def test_read_missing_returns_empty(tmp_path: Path) -> None:
    """Pre-v4 file with no /opensees/cuts/ + no /opensees/sweeps/ → ((), ())."""
    path = tmp_path / "model.h5"
    # Pre-v4 file is one that lacks /opensees/cuts/ — its schema must
    # still be inside the ADR 0023 two-version reader window
    # (2.6.x / 2.7.x).  Reader-window-old fixtures are out of scope
    # (INV-5 migration tooling).
    _make_h5_with_meta(path, schema_version="2.6.0")  # pre-v4 schema

    cuts, sweeps = read_cuts_and_sweeps(path)
    assert cuts == ()
    assert sweeps == ()


def test_read_only_cuts_no_sweeps(tmp_path: Path) -> None:
    """File with /opensees/cuts/ but no /opensees/sweeps/ → sweeps empty."""
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f, cuts=[cut])

    cuts, sweeps = read_cuts_and_sweeps(path)
    assert len(cuts) == 1
    assert sweeps == ()


def test_read_only_sweeps_no_cuts(tmp_path: Path) -> None:
    """File with /opensees/sweeps/ but no /opensees/cuts/ → cuts empty."""
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    sweep = SectionSweepDef(cuts=(cut,))
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f, sweeps=[sweep])

    cuts, sweeps = read_cuts_and_sweeps(path)
    assert cuts == ()
    assert len(sweeps) == 1


# --------------------------------------------------------------------- #
# Order traps
# --------------------------------------------------------------------- #
def test_natural_order_11_cuts(tmp_path: Path) -> None:
    """11 standalone cuts → reader returns them in writer order.

    Guards against HDF5's alphabetic group iteration (``cut_10``
    sorts before ``cut_2`` lexically). The reader's natural-int sort
    by index suffix must produce the writer's insertion order.
    """
    cuts_in = tuple(
        SectionCutDef(
            plane_point=(0.0, 0.0, float(i)),
            plane_normal=(0.0, 0.0, 1.0),
            element_ids=(i + 1,),
            label=f"cut #{i}",
        )
        for i in range(11)
    )
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f, cuts=list(cuts_in))

    cuts_out, _ = read_cuts_and_sweeps(path)
    assert len(cuts_out) == 11
    for i, (got, want) in enumerate(zip(cuts_out, cuts_in)):
        assert got == want, f"order mismatch at index {i}"


def test_sweep_order_attr_drives_reader(tmp_path: Path) -> None:
    """A sweep of 11 cuts round-trips via the explicit ``order`` attr."""
    sweep_cuts = tuple(
        SectionCutDef(
            plane_point=(0.0, 0.0, float(i)),
            plane_normal=(0.0, 0.0, 1.0),
            element_ids=(i + 1,),
            label=f"sweep cut #{i}",
        )
        for i in range(11)
    )
    sweep = SectionSweepDef(cuts=sweep_cuts)
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f, sweeps=[sweep])

    _, sweeps_out = read_cuts_and_sweeps(path)
    assert len(sweeps_out) == 1
    restored = sweeps_out[0]
    assert len(restored) == 11
    for i in range(11):
        assert restored[i] == sweep_cuts[i]


# --------------------------------------------------------------------- #
# Overwrite refusal
# --------------------------------------------------------------------- #
def test_write_raises_if_cuts_group_exists(tmp_path: Path) -> None:
    """Re-writing /opensees/cuts/ → ValueError (caller manages overwrite)."""
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f, cuts=[cut])
        with pytest.raises(ValueError, match="/opensees/cuts already exists"):
            write_cuts_into(f, cuts=[cut])


def test_write_raises_if_sweeps_group_exists(tmp_path: Path) -> None:
    """Re-writing /opensees/sweeps/ → ValueError."""
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    sweep = SectionSweepDef(cuts=(cut,))
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        write_cuts_into(f, sweeps=[sweep])
        with pytest.raises(ValueError, match="/opensees/sweeps already exists"):
            write_cuts_into(f, sweeps=[sweep])


# --------------------------------------------------------------------- #
# Constructor validation runs on read
# --------------------------------------------------------------------- #
def test_read_runs_post_init_validation(tmp_path: Path) -> None:
    """A hand-corrupted file → reader raises through ``__post_init__``.

    Writing a zero-vector ``plane_normal`` would never happen via the
    public writer (the dataclass refuses to construct), but a script
    or a corrupted file could in theory put one on disk. Confirm the
    reader doesn't silently accept it.
    """
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)
    with h5py.File(path, "r+") as f:
        ops = f.create_group("opensees")
        cuts_group = ops.create_group("cuts")
        cuts_group.attrs["count"] = np.int64(1)
        g = cuts_group.create_group("cut_0")
        # Deliberately bad: zero normal — fails __post_init__.
        g.attrs["plane_point"] = np.zeros(3, dtype=np.float64)
        g.attrs["plane_normal"] = np.zeros(3, dtype=np.float64)
        g.attrs.create("side", "positive",
                       dtype=h5py.string_dtype(encoding="utf-8"))
        g.attrs["has_label"] = np.int8(0)
        g.attrs.create("label", "",
                       dtype=h5py.string_dtype(encoding="utf-8"))
        g.attrs["has_bounding"] = np.int8(0)
        g.create_dataset("element_ids",
                         data=np.asarray([1], dtype=np.int64))

    with pytest.raises(ValueError, match="plane_normal must be nonzero"):
        read_cuts_and_sweeps(path)


# --------------------------------------------------------------------- #
# persist_to_h5 — append helper (v4-3)
# --------------------------------------------------------------------- #
def test_persist_to_h5_empty_is_noop(tmp_path: Path) -> None:
    """No cuts and no sweeps → file is never opened.

    Verifies the early-return before the ``h5py.File(..., "r+")``
    call — passing a nonexistent path with empty kwargs must NOT
    raise FileNotFoundError because the file isn't touched.
    """
    nonexistent = tmp_path / "never_created.h5"
    persist_to_h5(nonexistent)  # must not raise
    assert not nonexistent.exists()


def test_persist_to_h5_appends_to_in_window_file_keeps_version(
    tmp_path: Path,
) -> None:
    """In-window file → /opensees/cuts/ written; envelope unchanged.

    Per ADR 0023 the reader window is 2.5.x / 2.6.x; files below the
    window are refused at open time (INV-5 migration tooling).  The
    starting fixture is at 2.6.0 — within the current two-version
    window — so the version stays unchanged after append.
    """
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 1.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(42,),
        label="appended",
    )
    path = tmp_path / "in_window.h5"
    _make_h5_with_meta(path, schema_version="2.6.0")

    persist_to_h5(path, cuts=[cut])

    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["schema_version"] == "2.6.0"
        assert "opensees/cuts/cut_0" in f

    cuts, _ = read_cuts_and_sweeps(path)
    assert cuts == (cut,)


def test_persist_to_h5_refuses_out_of_window_file(tmp_path: Path) -> None:
    """File below the reader window (e.g. 2.5.0) → SchemaVersionError.

    Per ADR 0023 INV-5, archived files outside the two-version reader
    window need migration tooling.  ``persist_to_h5`` validates via
    :func:`h5_reader.open` and so refuses with a SchemaVersionError
    rather than silently appending and bumping the version.
    """
    from apeGmsh.opensees._internal.schema_version import SchemaVersionError

    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    path = tmp_path / "below_window.h5"
    _make_h5_with_meta(path, schema_version="2.5.0")

    with pytest.raises(SchemaVersionError):
        persist_to_h5(path, cuts=[cut])


def test_persist_to_h5_leaves_2_5_0_version_alone(tmp_path: Path) -> None:
    """File already at 2.6.0 → version unchanged after append."""
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    path = tmp_path / "already_v4.h5"
    _make_h5_with_meta(path, schema_version="2.6.0")

    persist_to_h5(path, cuts=[cut])

    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["schema_version"] == "2.6.0"


def test_persist_to_h5_preserves_future_version(tmp_path: Path) -> None:
    """Speculative 2.7.0 file → version not downgraded.

    Guards against the bump helper accidentally clobbering a forward-
    looking version when ``min_version`` is below the current value.
    """
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    path = tmp_path / "future.h5"
    _make_h5_with_meta(path, schema_version="2.7.0")

    persist_to_h5(path, cuts=[cut])

    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["schema_version"] == "2.7.0"


def test_persist_to_h5_overwrites_existing_cuts_group(
    tmp_path: Path,
) -> None:
    """Re-calling with cuts= replaces the existing /opensees/cuts/."""
    cut_old = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="old",
    )
    cut_new_a = SectionCutDef(
        plane_point=(0.0, 0.0, 1.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(2,),
        label="new a",
    )
    cut_new_b = SectionCutDef(
        plane_point=(0.0, 0.0, 2.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(3,),
        label="new b",
    )
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)

    persist_to_h5(path, cuts=[cut_old])
    persist_to_h5(path, cuts=[cut_new_a, cut_new_b])

    cuts, _ = read_cuts_and_sweeps(path)
    assert len(cuts) == 2
    assert cuts[0] == cut_new_a
    assert cuts[1] == cut_new_b


def test_persist_to_h5_only_touches_requested_groups(
    tmp_path: Path,
) -> None:
    """persist_to_h5(cuts=) must leave /opensees/sweeps/ untouched."""
    standalone = SectionCutDef(
        plane_point=(0.0, 0.0, 1.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="standalone",
    )
    sweep_cut = SectionCutDef(
        plane_point=(0.0, 0.0, 10.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(2,),
        label="sweep member",
    )
    sweep = SectionSweepDef(cuts=(sweep_cut,))
    path = tmp_path / "model.h5"
    _make_h5_with_meta(path)

    # Establish both groups.
    persist_to_h5(path, cuts=[standalone], sweeps=[sweep])

    # Re-call with cuts only — sweep must survive.
    new_standalone = SectionCutDef(
        plane_point=(0.0, 0.0, 2.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(3,),
        label="replacement",
    )
    persist_to_h5(path, cuts=[new_standalone])

    cuts, sweeps_out = read_cuts_and_sweeps(path)
    assert len(cuts) == 1
    assert cuts[0] == new_standalone
    assert len(sweeps_out) == 1
    assert sweeps_out[0] == sweep


def test_persist_to_h5_raises_on_wrong_major(tmp_path: Path) -> None:
    """Major != 2 → SchemaVersionError propagates from the reader."""
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    path = tmp_path / "wrong_major.h5"
    _make_h5_with_meta(path, schema_version="3.0.0")

    with pytest.raises(SchemaVersionError):
        persist_to_h5(path, cuts=[cut])


def test_persist_to_h5_raises_on_missing_file(tmp_path: Path) -> None:
    """Non-existent file + non-empty kwargs → FileNotFoundError."""
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    with pytest.raises((FileNotFoundError, OSError)):
        persist_to_h5(tmp_path / "nope.h5", cuts=[cut])
