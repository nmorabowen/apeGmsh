"""Fixture-suite contract tests.

For each entry in :data:`FIXTURE_BUILDERS` the test:

1. Builds the fixture via its builder.
2. Writes it to a ``tmp_path``.
3. Opens it through the reference reader (or asserts open refuses,
   for ``wrong_major``).
4. Asserts the expectations in :data:`FIXTURE_EXPECTATIONS` are met.

This is the bridge team's deliverable to the viewer team: a portable
recipe for producing every viewer-integration scenario.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import pytest

from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5_reader import SchemaVersionError

from .fixtures import FIXTURE_BUILDERS, FIXTURE_EXPECTATIONS


def _has_group(handle: h5py.File, path: str) -> bool:
    """Group existence via ``in`` check on the file handle."""
    return path in handle


@pytest.mark.parametrize("name", sorted(FIXTURE_BUILDERS))
def test_fixture_builds_and_meets_expectations(
    name: str, tmp_path: Path,
) -> None:
    builder = FIXTURE_BUILDERS[name]
    expected = FIXTURE_EXPECTATIONS[name]
    out = tmp_path / f"{name}.h5"
    builder().write(str(out))
    assert out.exists() and out.stat().st_size > 0

    if not expected.get("should_open", True):
        with pytest.raises(SchemaVersionError):
            h5_reader.open(str(out))
        # Sanity-check the version marker
        with h5py.File(out, "r") as f:
            version_prefix = expected["schema_version_starts_with"]
            assert str(f["meta"].attrs["schema_version"]).startswith(version_prefix)
        return

    with h5_reader.open(str(out)) as model:
        if expected.get("should_validate"):
            violations = model.validate()
            assert violations == [], f"{name}: {violations}"
        for group in expected.get("expected_groups", []):
            assert _has_group(model.handle, group), (
                f"{name}: expected group {group!r} not present"
            )
        for group in expected.get("expected_absent_groups", []):
            assert not _has_group(model.handle, group), (
                f"{name}: unexpected group {group!r} present"
            )
        if "material_count_uniaxial" in expected:
            uni = model.materials().get("uniaxial", {})
            assert len(uni) == expected["material_count_uniaxial"]
        if "section_count" in expected:
            assert len(model.sections()) == expected["section_count"]
        if "transform_count" in expected:
            assert len(model.transforms()) == expected["transform_count"]
        if "element_type_count" in expected:
            assert len(model.elements()) == expected["element_type_count"]
        if "pattern_count" in expected:
            assert len(model.patterns()) == expected["pattern_count"]
        if "recorder_count" in expected:
            assert len(model.recorders()) == expected["recorder_count"]


def test_fixture_expectations_are_serializable_to_json(tmp_path: Path) -> None:
    """Every expectations dict is JSON-serializable.

    The viewer team consumes the same expectations as a ``.json``
    sibling per the integration doc — we don't ship those files
    (binary blob drift), but we DO commit to the dicts being
    serializable so the viewer team can dump them on demand.
    """
    for name, exp in FIXTURE_EXPECTATIONS.items():
        # Round-trip through JSON to confirm no non-serializable types.
        payload = json.dumps(exp)
        roundtrip: dict[str, Any] = json.loads(payload)
        assert roundtrip == exp, f"non-roundtrip-safe expectations: {name}"


def test_fixture_dump_json_artifacts_on_demand(tmp_path: Path) -> None:
    """Write each fixture's expectations to a sibling ``.json``.

    Demonstrates the viewer-integration team's discovery flow without
    committing the artifacts to the repo. The test passes if every
    file lands and is parseable.
    """
    for name, exp in FIXTURE_EXPECTATIONS.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(exp, indent=2, sort_keys=True))
        parsed: dict[str, Any] = json.loads(path.read_text())
        assert parsed == exp
