"""Schema-validation tests for the bridge model.h5 archive.

Each test builds a small model via :class:`H5Emitter`, writes it
to disk, then re-opens it via :mod:`apeGmsh.opensees.emitter.h5_reader`
and asserts schema invariants. The reader's :meth:`H5Model.validate`
returns a list of violations — every passing test asserts that the
list is empty.
"""
from __future__ import annotations

from typing import Any

import h5py
import pytest

from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.h5_reader import (
    H5Model,
    MalformedH5Error,
    SchemaVersionError,
)
from apeGmsh.opensees._internal.tag_resolution import set_element_nodes


def _open(path: str) -> H5Model:
    return h5_reader.open(path)


def test_reader_accepts_current_schema(tmp_path: Any) -> None:
    e = H5Emitter()
    e.model(ndm=2, ndf=3)
    out = tmp_path / "ok.h5"
    e.write(str(out))
    with _open(str(out)) as m:
        assert m.schema_version.startswith("1.")
        assert m.meta()["ndm"] == 2


def test_reader_refuses_wrong_major(tmp_path: Any) -> None:
    e = H5Emitter(schema_version="2.0.0")
    out = tmp_path / "wrong_major.h5"
    e.write(str(out))
    with pytest.raises(SchemaVersionError) as exc:
        _open(str(out))
    assert "major 2" in str(exc.value) or "major" in str(exc.value)


def test_reader_refuses_missing_meta(tmp_path: Any) -> None:
    out = tmp_path / "no_meta.h5"
    with h5py.File(out, "w") as f:
        f.create_group("nothing_useful")
    with pytest.raises(MalformedH5Error):
        _open(str(out))


def test_reader_refuses_empty_schema_version(tmp_path: Any) -> None:
    out = tmp_path / "no_version.h5"
    with h5py.File(out, "w") as f:
        f.create_group("meta")
    with pytest.raises(MalformedH5Error):
        _open(str(out))


def test_reader_validate_finds_no_violations_in_complete_model(
    tmp_path: Any,
) -> None:
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial(
        "Steel02", 1, 420.0e6, 200.0e9, 0.01, 20.0, 0.925, 0.15,
    )
    e.uniaxialMaterial(
        "Concrete02", 2, -30.0e6, -0.002, -25.0e6, -0.006,
        0.1, 2.5e6, 200.0e6,
    )
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.patch("rect", 2, 8, 8, -0.2, -0.2, 0.2, 0.2)
    e.fiber(0.1, 0.0, 0.001, 1)
    e.section_close()
    e.timeSeries("Linear", 1, "-factor", 1.0)
    e.pattern_open("Plain", 1, 1)
    e.load(10, 1.0, 0.0, 0.0)
    e.pattern_close()
    out = tmp_path / "complete.h5"
    e.write(str(out))
    with _open(str(out)) as m:
        violations = m.validate()
        assert violations == [], violations


def test_reader_validate_detects_dangling_material_ref(tmp_path: Any) -> None:
    """Hand-craft a file with a bad material_ref and confirm validate catches it."""
    out = tmp_path / "dangling.h5"
    # First write a valid file ...
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial("Steel02", 1, 420.0e6, 200.0e9, 0.01)
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.fiber(0.1, 0.0, 0.001, 1)
    e.section_close()
    e.write(str(out))
    # ... then mutate the fiber's material_ref to a dangling path.
    with h5py.File(out, "a") as f:
        ds = f["sections/Fiber_1/fibers"]
        rows = ds[:]
        rows[0]["material_ref"] = b"/materials/uniaxial/Nonexistent_99"
        ds[...] = rows
    with _open(str(out)) as m:
        violations = m.validate()
        assert any("Nonexistent_99" in v for v in violations)


def test_reader_accessors_return_attrs(tmp_path: Any) -> None:
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial("Steel02", 1, 420.0e6, 200.0e9, 0.01)
    e.geomTransf("PDelta", 1, 0.0, 0.0, 1.0)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 1, 1, 2, 1, 1)
    out = tmp_path / "accessors.h5"
    e.write(str(out))
    with _open(str(out)) as m:
        mats = m.materials()
        assert "uniaxial" in mats
        assert "Steel02_1" in mats["uniaxial"]
        assert mats["uniaxial"]["Steel02_1"]["type"] == "Steel02"
        tx = m.transforms()
        assert "PDelta_1" in tx
        els = m.elements()
        assert "forceBeamColumn" in els
