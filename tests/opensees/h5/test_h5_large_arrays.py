"""A long ``*args`` array archives and replays exactly (#2404, K #2418).

``ops.timeSeries.Path(values=<16384 floats>)`` ran its deck, but the
``model.h5`` write failed with ``OSError: object header message is too
large``: the writer stored the values as one HDF5 attribute, and an
attribute's object-header message is capped at 65,535 bytes (about 8,183
float64 slots, or 4,091 variable-length string slots, measured on
h5py 3.15 / HDF5 1.14.6). The writer now tries the attribute and falls
back to same-named datasets only when HDF5 refuses it.

Oracles:

* **Bridge equality.** ``OpenSeesModel.from_h5(p).build("tcl")`` equals
  the deck the bridge itself emits, byte for byte, at a small control
  size and at 16,384 and 200,000 values.
* **Hash round-trip.** ``model_hash`` of the archive equals the one of
  ``from_h5(p).to_h5(q)``, so the identity of the large value survives a
  rewrite.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.lineage import compute_model_hash
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem

#: A control below every limit, the size #2404 hit, and a size well past
#: anything the attribute form could ever hold.
SIZES = (16, 16_384, 200_000)


def _values(n: int) -> tuple[float, ...]:
    # Non-integral, distinct values: a slot that lost its float-ness or
    # its position shows in the deck.
    return tuple(0.001 * i + 0.0005 for i in range(n))


def _frame_with_path(n: int) -> apeSees:
    ops = apeSees(cast("object", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.0125, E=2900.25, Iz=1.5e-4, Iy=2.5e-4, G=1100.75, J=3.5e-4,
    )
    ops.timeSeries.Path(values=_values(n), dt=0.01)
    return ops


def _bridge_deck(ops: apeSees, tmp_path: Path) -> str:
    p = tmp_path / "bridge.tcl"
    ops.tcl(str(p), progress=False)
    return p.read_text(encoding="utf-8")


def _hash(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return compute_model_hash("", f["opensees"])


@pytest.mark.parametrize("n", SIZES)
def test_path_series_archives_and_replays_exactly(
    n: int, tmp_path: Path,
) -> None:
    ops = _frame_with_path(n)
    bridge = _bridge_deck(ops, tmp_path)
    src = tmp_path / "model.h5"
    ops.h5(str(src))

    replay = OpenSeesModel.from_h5(str(src)).build("tcl")

    assert isinstance(replay, str)
    assert replay == bridge


@pytest.mark.parametrize("n", SIZES)
def test_path_series_model_hash_round_trips(n: int, tmp_path: Path) -> None:
    src = tmp_path / "model.h5"
    _frame_with_path(n).h5(str(src))
    out = tmp_path / "rewrite.h5"

    OpenSeesModel.from_h5(str(src)).to_h5(str(out))

    assert _hash(out) == _hash(src)


# ---------------------------------------------------------------------------
# The on-disk form: attribute below the limit, datasets above it
# ---------------------------------------------------------------------------


def _series_group(path: Path) -> tuple[set[str], set[str]]:
    """``(attribute names, child names)`` of the one time series group."""
    with h5py.File(str(path), "r") as f:
        ts = f["opensees/time_series"]
        (name,) = list(ts)
        g = ts[name]
        return set(g.attrs), set(g)


def test_small_pair_keeps_the_attribute_form(tmp_path: Path) -> None:
    src = tmp_path / "model.h5"
    _frame_with_path(16).h5(str(src))

    attrs, children = _series_group(src)

    assert {"params", "params_str"} <= attrs
    assert children == set()


def test_large_pair_is_two_same_named_datasets(tmp_path: Path) -> None:
    src = tmp_path / "model.h5"
    _frame_with_path(16_384).h5(str(src))

    attrs, children = _series_group(src)

    assert children == {"params", "params_str"}
    assert not {"params", "params_str"} & attrs


# ---------------------------------------------------------------------------
# The boundary: the attribute form wherever a 2.33 writer could write one
# ---------------------------------------------------------------------------

#: ``model_hash("", /opensees)`` of the largest archive a 2.33 writer could
#: write, as main's writer gave it (origin/main e094d9e05, run out of place
#: from a ``git archive`` copy). One more value made that writer fail with
#: ``object header message is too large``. A Path pair is bound by its
#: string half (4,091 slots: ``-dt``, ``-values`` and the values); a
#: MultiLinear pair is all numeric (8,183 slots: the tag and two per point).
MAIN_LARGEST_PATH = (4_088, "869adb87bcc7b6dd8bdefee594e5672b")
MAIN_LARGEST_MULTILINEAR = (4_091, "8c7605bb3607b720b926f506079c8047")


def _frame_with_multilinear(n_points: int) -> apeSees:
    ops = apeSees(cast("object", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.0125, E=2900.25, Iz=1.5e-4, Iy=2.5e-4, G=1100.75, J=3.5e-4,
    )
    ops.uniaxialMaterial.MultiLinear(points=tuple(
        (0.001 * (i + 1), 1.0 + 0.5 * i) for i in range(n_points)))
    return ops


def _form_and_hash(ops: apeSees, group: str, path: Path) -> tuple[str, str]:
    ops.h5(str(path))
    with h5py.File(str(path), "r") as f:
        (name,) = list(f[group])
        g = f[group][name]
        in_attrs = {"params", "params_str"} & set(g.attrs)
        in_group = {"params", "params_str"} & set(g)
        assert not (in_attrs and in_group), "one form per group"
        form = "dataset" if in_group else "attr"
        return form, compute_model_hash("", f["opensees"])


@pytest.mark.parametrize(("build", "group", "largest"), [
    (_frame_with_path, "opensees/time_series", MAIN_LARGEST_PATH),
    (_frame_with_multilinear, "opensees/materials/uniaxial",
     MAIN_LARGEST_MULTILINEAR),
], ids=["path", "multilinear"])
def test_boundary_keeps_main_bytes_then_falls_back(
    build: object, group: str, largest: tuple[int, str], tmp_path: Path,
) -> None:
    """At the largest size main's writer could write, the pair is still an
    attribute and the archive hashes as main's did; one more value and the
    pair is two datasets, with no half-written attribute left behind."""
    from collections.abc import Callable

    make = cast("Callable[[int], apeSees]", build)
    n, main_hash = largest

    assert _form_and_hash(make(n), group, tmp_path / "at.h5") == (
        "attr", main_hash)
    form, _ = _form_and_hash(make(n + 1), group, tmp_path / "past.h5")
    assert form == "dataset"


def test_set_attr_refuses_only_what_hdf5_refuses() -> None:
    """``_set_attr`` writes every string list a raw ``attrs.create`` (main's
    writer) can hold and refuses, by name, the first one it cannot."""
    from apeGmsh.opensees.emitter._h5_write import (
        AttributeTooLargeError,
        _set_attr,
    )

    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File("raw.h5", "w", driver="core", backing_store=False) as f:
        def raw_fits(n: int) -> bool:
            g = f.create_group(f"raw_{n}")
            try:
                g.attrs.create("selection", ["s"] * n, dtype=str_dtype)
            except OSError:
                return False
            return True

        lo, hi = 1, 20_000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if raw_fits(mid) else (lo, mid - 1)
        largest = lo

        g = f.create_group("ours")
        _set_attr(g, "selection", ["s"] * largest)
        assert len(g.attrs["selection"]) == largest
        g2 = f.create_group("ours_past")
        with pytest.raises(AttributeTooLargeError,
                           match=r"'selection' on /ours_past holds"):
            _set_attr(g2, "selection", ["s"] * (largest + 1))
        assert "selection" not in g2.attrs


# ---------------------------------------------------------------------------
# A declared recorder past the attribute limit
# ---------------------------------------------------------------------------


def _recorder_archive(path: Path, ids: tuple[int, ...]) -> None:
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    e.recorder_declaration_begin(
        declaration_name="d", record_name="r", category="nodes",
        components=("displacement_x",), ids=ids,
    )
    e.recorder("Node", "-file", "a.out", "-node", *ids, "-dof", 1, "disp")
    e.recorder_declaration_end()
    e.write(str(path))


def test_recorder_with_more_than_8184_ids_round_trips(tmp_path: Path) -> None:
    """8,184 int64 slots is the most one attribute holds; the ids and the
    ``-node`` tail both go to datasets and read back slot for slot."""
    from apeGmsh.opensees.emitter import h5_reader

    ids = tuple(range(1, 9_001))
    out = tmp_path / "rec.h5"
    _recorder_archive(out, ids)

    with h5py.File(str(out), "r") as f:
        (g,) = list(f["opensees/recorders"].values())
        assert {"ids", "params", "params_str"} <= set(g)
    with h5_reader.open(str(out)) as model:
        (rec,) = model.recorders()
    assert rec.decl_context is not None
    assert rec.decl_context.ids == ids
    assert rec.args == ("-file", "a.out", "-node", *ids, "-dof", 1, "disp")


# ---------------------------------------------------------------------------
# Fail closed: refusals at the writer and the reader
# ---------------------------------------------------------------------------


def test_other_oversized_sequence_refuses_by_name(tmp_path: Path) -> None:
    """A sequence attribute with no dataset form (here a declaration's
    ``selection``) refuses, naming the key and the group, instead of the
    bare HDF5 ``OSError``."""
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    e.recorder_declaration_begin(
        declaration_name="d", record_name="r", category="nodes",
        components=("displacement_x",),
        selection=tuple(f"s{i}" for i in range(5_000)),
    )
    e.recorder("Node", "-file", "a.out", "-node", 1, "-dof", 1, "disp")
    e.recorder_declaration_end()

    with pytest.raises(
        ValueError,
        match=r"attribute 'selection' on /opensees/recorders/\S+ holds "
              r"5000 values",
    ):
        e.write(str(tmp_path / "x.h5"))


def _large_archive(tmp_path: Path) -> Path:
    src = tmp_path / "model.h5"
    _frame_with_path(16_384).h5(str(src))
    return src


def _series_path(f: h5py.File) -> str:
    (name,) = list(f["opensees/time_series"])
    return f"opensees/time_series/{name}"


def test_both_forms_present_is_malformed(tmp_path: Path) -> None:
    from apeGmsh.opensees.emitter import h5_reader
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    src = _large_archive(tmp_path)
    with h5py.File(str(src), "r+") as f:
        f[_series_path(f)].attrs["params"] = [1.0, 2.0]

    with h5_reader.open(str(src)) as model:
        with pytest.raises(MalformedH5Error, match="one form"):
            model.time_series()


def test_dataset_form_below_2_34_is_malformed(tmp_path: Path) -> None:
    """No writer below 2.34.0 made a dataset form, so a file stamped
    below it that holds one is corrupt, not an old archive."""
    from apeGmsh.opensees.emitter import h5_reader
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
    from tests.fixtures.schema import OPENSEES_LARGE_ARRAY_DATASETS_FROM

    # The minor just below the one that introduced the dataset form (not
    # the prior minor, which moves on with every later bump).
    major, minor, _patch = OPENSEES_LARGE_ARRAY_DATASETS_FROM.split(".")
    below = f"{major}.{int(minor) - 1}.0"
    src = _large_archive(tmp_path)
    with h5py.File(str(src), "r+") as f:
        meta = f["meta"].attrs
        assert "opensees_schema_version" in meta
        meta["opensees_schema_version"] = below

    with h5_reader.open(str(src)) as model:
        with pytest.raises(MalformedH5Error,
                           match="is a dataset in an opensees"):
            model.time_series()


# ---------------------------------------------------------------------------
# The assembly reader sees a dataset-form region
# ---------------------------------------------------------------------------


def test_region_params_reads_a_dataset_form_damp(tmp_path: Path) -> None:
    """``refuse_region_dampings`` reads region rows with raw h5py; a long
    region attaching a damping must still be refused (#2404)."""
    from apeGmsh.assembly import AssemblyError
    from apeGmsh.assembly._rehydrate import refuse_region_dampings
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    e.region(7, "-node", *range(1, 5_001), "-damp", 3)
    out = tmp_path / "region.h5"
    e.write(str(out))
    with h5py.File(str(out), "r") as f:
        assert "params" in f["opensees/regions/region_000"]

    with pytest.raises(AssemblyError, match=r"damping tags \[3\]"):
        refuse_region_dampings("p", out)
