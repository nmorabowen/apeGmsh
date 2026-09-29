"""Self-test for ``scripts/h5_canonical_dump.py`` (the golden corpus H5 oracle).

The dump must be blind to what legitimately varies between two writes of
one model (wall-clock ``created_iso``, byte order) and see everything else:
one changed value is exactly one changed ``sha1`` line.
"""
from __future__ import annotations

import importlib.util
import shutil
import time
from pathlib import Path
from types import ModuleType
from typing import cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees

from tests.opensees.fixtures.fem_stub import make_two_column_frame

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "h5_canonical_dump.py"


@pytest.fixture(scope="module")
def dumper() -> ModuleType:
    # Loaded by path and never registered in sys.modules: nothing to restore.
    spec = importlib.util.spec_from_file_location(
        "_selftest_h5_canonical_dump", SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_model_h5(path: Path) -> None:
    ops = apeSees(cast("object", make_two_column_frame()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.h5(str(path), model_name="model")


def _created_iso(path: Path) -> str:
    with h5py.File(path, "r") as f:
        return str(f["meta"].attrs["created_iso"])


def test_same_model_at_different_times_dumps_identically(
    dumper: ModuleType, tmp_path: Path,
) -> None:
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _write_model_h5(a)
    time.sleep(0.05)
    _write_model_h5(b)
    # Precondition: the two writes really carry different wall clocks.
    assert _created_iso(a) != _created_iso(b)
    assert dumper.dump(a) == dumper.dump(b)
    assert "A /meta@created_iso " in dumper.dump(a)
    assert "sha1=masked" in dumper.dump(a)


def test_one_changed_value_changes_exactly_one_sha1_line(
    dumper: ModuleType, tmp_path: Path,
) -> None:
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _write_model_h5(a)
    shutil.copyfile(a, b)
    target = "/opensees/element_meta/elasticBeamColumn/args"
    with h5py.File(b, "r+") as f:
        ds = f[target]
        values = ds[()]
        values.flat[0] += 1.0
        ds[...] = values

    before = dumper.dump(a).splitlines()
    after = dumper.dump(b).splitlines()
    assert len(before) == len(after)
    changed = [(x, y) for x, y in zip(before, after) if x != y]
    assert len(changed) == 1
    old, new = changed[0]
    assert old.startswith(f"D {target} ") and new.startswith(f"D {target} ")
    assert old.rsplit("sha1=", 1)[0] == new.rsplit("sha1=", 1)[0]


def test_byte_order_and_chunking_do_not_change_the_dump(
    dumper: ModuleType, tmp_path: Path,
) -> None:
    values = np.arange(12, dtype="<f8").reshape(3, 4)
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    with h5py.File(a, "w") as f:
        f.create_dataset("x", data=values)
        f.attrs["n"] = np.int64(3)
    with h5py.File(b, "w") as f:
        f.create_dataset("x", data=values.astype(">f8"), chunks=(1, 4))
        f.attrs["n"] = np.array(3, dtype=">i8")
    assert dumper.dump(a) == dumper.dump(b)


def test_mask_is_exact_path_only(dumper: ModuleType, tmp_path: Path) -> None:
    """A wall-clock attr outside ``/meta`` is a finding, not masked noise."""
    path = tmp_path / "m.h5"
    with h5py.File(path, "w") as f:
        f.create_group("meta").attrs["created_iso"] = "2026-01-01T00:00:00"
        f.create_group("other").attrs["created_iso"] = "2026-01-01T00:00:00"
    lines = dumper.dump(path).splitlines()
    meta = next(ln for ln in lines if ln.startswith("A /meta@created_iso"))
    other = next(ln for ln in lines if ln.startswith("A /other@created_iso"))
    assert meta.endswith("sha1=masked")
    assert not other.endswith("sha1=masked")


def test_model_hash_mask_is_that_path_only(
    dumper: ModuleType, tmp_path: Path,
) -> None:
    """``/meta/lineage@model_hash`` is masked; the same name elsewhere is not.

    The digest summarises raw float bytes under ``/opensees`` (a last-ulp
    vecxz changes it, #1258); the datasets it covers are pinned line by line.
    """
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    for path, digest in ((a, "aaaa"), (b, "bbbb")):
        with h5py.File(path, "w") as f:
            f.create_group("meta/lineage").attrs["model_hash"] = digest
            f["meta"].attrs["model_hash"] = digest
            f.create_group("opensees/lineage").attrs["model_hash"] = digest
    la, lb = dumper.dump(a).splitlines(), dumper.dump(b).splitlines()
    masked = [ln for ln in la if ln.endswith("sha1=masked")]
    assert masked == [
        "A /meta/lineage@model_hash dtype=str[utf-8,vlen] shape=() sha1=masked",
    ]
    differing = sorted(x.split(" ", 2)[1] for x, y in zip(la, lb) if x != y)
    assert differing == ["/meta@model_hash", "/opensees/lineage@model_hash"]


def test_strings_and_compound_rows_are_hashed_by_value(
    dumper: ModuleType, tmp_path: Path,
) -> None:
    str_dt = h5py.string_dtype()
    row = np.dtype([("name", str_dt), ("dofs", "<i8", (2,))])
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    for path, name in ((a, "Base"), (b, "Top")):
        with h5py.File(path, "w") as f:
            data = np.array([(name, (1, 1))], dtype=row)
            f.create_dataset("rows", data=data)
    da, db = dumper.dump(a), dumper.dump(b)
    assert "dtype={name:str[utf-8,vlen],dofs:<i8[2]}" in da
    assert da != db
    # Re-dumping the same file is stable (no pointer bytes in the hash).
    assert dumper.dump(a) == da


@pytest.mark.parametrize(
    ("golden", "other", "same"),
    [
        # #1258: the CI runner's libm emitted these vecxz components.
        (0.25881904510252085, 0.2588190451025208, True),
        (0.9659258262890684, 0.9659258262890682, True),
        (0.25881904510252085, float(np.nextafter(0.25881904510252085, 1.0)), True),
        (0.0, -0.0, True),
        (0.25881904510252085, 0.25881904510252085 * (1 + 1e-9), False),
        (0.25881904510252085, 0.25881904510252085 * (1 + 1e-6), False),
        (200e9, 200e9 + 1.0, False),
    ],
)
def test_float_hash_absorbs_last_ulp_only(
    dumper: ModuleType, tmp_path: Path, golden: float, other: float, same: bool,
) -> None:
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    for path, value in ((a, golden), (b, other)):
        with h5py.File(path, "w") as f:
            f.create_dataset("vecxz", data=np.array([[value, 0.0, 1.0]]))
            f.attrs["scale"] = np.float64(value)
    assert (dumper.dump(a) == dumper.dump(b)) is same
