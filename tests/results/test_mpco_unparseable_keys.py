"""MPCO discovery names every bucket key it cannot parse (#2165).

Five discovery paths, at the Gauss, material-state, spring, fiber and
line-station levels, met a bucket key that ``parse_mpco_element_key``
refuses with a bare ``except ValueError: continue``: the bucket vanished
from listings and reads without a word. Each now names it in one
``MPCOBucketSkippedWarning`` (path, reason, file), the contract #2159 and
#2164 set, and the other buckets still read. The layer and nodal-force
levels already named such a key, in their own warning classes; the tests
here pin that too, so every bucket family names a malformed key.

A key under two alias groups of one token (``stresses`` and
``material.stress``) is named once. A spring group is shared with beams
(``basicForce``), and a key that does not parse names no class; the
spring reader is the only one that parses that group's keys, so it is the
one that names it.

The converter's spring scan (``_scan_stage``) used the spring reader's
discovery, whose warning says a bucket is "left unread", on buckets the
converter copies; and it set the ``springs`` provenance flag only for a
bucket the reader could read. A custom-rule ZeroLength file now converts
without that warning, and with its ``springs`` flag.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import mpco_gauss_group_aliases
from apeGmsh.results import Results
from apeGmsh.results.readers._mpco import MPCOBucketSkippedWarning
from apeGmsh.results.readers._mpco_element_io import discover_gauss_buckets
from apeGmsh.results.readers._mpco_fiber_io import discover_fiber_buckets
from apeGmsh.results.readers._mpco_layer_io import scan_layer_buckets
from apeGmsh.results.readers._mpco_line_io import (
    discover_line_station_buckets,
)
from apeGmsh.results.readers._mpco_material_io import (
    discover_material_state_buckets,
)
from apeGmsh.results.readers._mpco_nodal_io import scan_nodal_force_buckets
from apeGmsh.results.readers._mpco_spring_io import discover_spring_buckets
from tests.conftest import _stub_model_h5_path
from tests.test_results_mpco_element_mock import _create_mpco_skeleton

#: Keys ``parse_mpco_element_key`` refuses: no bracket, a non-integer
#: header index, a missing class tag.
_BAD_KEYS = ("not-a-bucket", "19-ZeroLength[1:0:x]", "ElasticBeam3d[1:0:0]")

_Discover = Callable[..., Any]

#: family -> (ON_ELEMENTS group, discovery function, canonical component).
_FAMILIES: "dict[str, tuple[str, _Discover, str]]" = {
    "gauss": ("stresses", discover_gauss_buckets, "stress_xx"),
    "material-state": (
        "material.damage", discover_material_state_buckets, "damage",
    ),
    "spring": ("basicForce", discover_spring_buckets, "spring_force_0"),
    "fiber": ("section.fiber.stress", discover_fiber_buckets, "fiber_stress"),
    "line-station": (
        "section.force", discover_line_station_buckets, "axial_force",
    ),
}


def _skips(record: "list[warnings.WarningMessage]") -> list[str]:
    return [
        str(w.message) for w in record
        if issubclass(w.category, MPCOBucketSkippedWarning)
    ]


def _on_elements(path: Path, groups: "dict[str, tuple[str, ...]]") -> h5py.File:
    """A file whose ON_ELEMENTS holds an empty bucket per (group, key)."""
    f = h5py.File(path, "w")
    on_el = f.create_group("RESULTS/ON_ELEMENTS")
    for group, keys in groups.items():
        g = on_el.require_group(group)
        for key in keys:
            g.create_group(key)
    return f


# ---------------------------------------------------------------------
# Discovery: one warning per unparseable key, naming path and file
# ---------------------------------------------------------------------

@pytest.mark.parametrize("family", sorted(_FAMILIES))
@pytest.mark.parametrize("bad_key", _BAD_KEYS)
def test_discovery_names_an_unparseable_key(
    family: str, bad_key: str, tmp_path: Path,
) -> None:
    group, discover, component = _FAMILIES[family]
    path = tmp_path / "bad_key.mpco"
    with _on_elements(path, {group: (bad_key,)}) as f:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _, buckets = discover(
                f["RESULTS/ON_ELEMENTS"], canonical_component=component,
            )
    assert buckets == []
    skips = _skips(record)
    assert len(skips) == 1, skips
    (msg,) = skips
    assert f"/RESULTS/ON_ELEMENTS/{group}/{bad_key}: its bracket key " \
        f"does not parse" in msg
    assert "bad_key.mpco" in msg
    assert "left 1 bucket(s) unread" in msg


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_each_unparseable_key_is_named_once(family: str, tmp_path: Path) -> None:
    """Every bad key is named, once each, in one discovery."""
    group, discover, component = _FAMILIES[family]
    path = tmp_path / "bad_keys.mpco"
    with _on_elements(path, {group: _BAD_KEYS}) as f:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            discover(f["RESULTS/ON_ELEMENTS"], canonical_component=component)
    skips = _skips(record)
    assert sorted(
        re.search(rf"/{re.escape(group)}/(.+?): its bracket key", m).group(1)
        for m in skips
    ) == sorted(_BAD_KEYS)


@pytest.mark.parametrize(("family", "primary"), [
    ("gauss", "stresses"),
    ("material-state", "damage"),
])
def test_a_key_under_two_alias_groups_is_named_once(
    family: str, primary: str, tmp_path: Path,
) -> None:
    group, discover, component = _FAMILIES[family]
    aliases = mpco_gauss_group_aliases(primary)
    assert len(aliases) >= 2, aliases     # the oracle needs two groups
    path = tmp_path / "aliases.mpco"
    with _on_elements(path, {a: ("not-a-bucket",) for a in aliases}) as f:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            discover(f["RESULTS/ON_ELEMENTS"], canonical_component=component)
    assert len(_skips(record)) == 1, _skips(record)


def test_the_spring_level_still_ignores_a_parsed_beam_key(
    tmp_path: Path,
) -> None:
    """A beam's ``basicForce`` key parses and names a non-spring class:
    it is not the spring level's to name. Only the unparseable key is."""
    path = tmp_path / "beam_and_bad.mpco"
    with _on_elements(
        path, {"basicForce": ("5-ElasticBeam3d[1:0:0]", "not-a-bucket")},
    ) as f:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            discover_spring_buckets(
                f["RESULTS/ON_ELEMENTS"], canonical_component="spring_force_0",
            )
    (msg,) = _skips(record)
    assert "/basicForce/not-a-bucket:" in msg


# ---------------------------------------------------------------------
# The two levels that already named the key keep doing so
# ---------------------------------------------------------------------

def test_the_layer_level_lists_an_unparseable_key(tmp_path: Path) -> None:
    path = tmp_path / "layer.mpco"
    with _on_elements(path, {"material.fiber.stress": ("not-a-bucket",)}) as f:
        _, buckets, unread = scan_layer_buckets(
            f["RESULTS/ON_ELEMENTS"], canonical_component="fiber_stress",
        )
    assert buckets == []
    assert [(p, r.split(" (")[0]) for p, r in unread] == [(
        "ON_ELEMENTS/material.fiber.stress/not-a-bucket",
        "its bracket key does not parse",
    )]


def test_the_nodal_level_lists_an_unparseable_key(tmp_path: Path) -> None:
    path = tmp_path / "nodal.mpco"
    with _on_elements(path, {"globalForce": ("not-a-bucket",)}) as f:
        _, buckets, skipped = scan_nodal_force_buckets(
            f["RESULTS/ON_ELEMENTS"],
            canonical_component="nodal_resisting_force_x",
        )
    assert buckets == []
    assert [s.bracket_key for s in skipped] == ["not-a-bucket"]
    assert "does not parse" in skipped[0].reason


# ---------------------------------------------------------------------
# End to end: one malformed key per family, one warning per read
# ---------------------------------------------------------------------

_READS: "dict[str, tuple[str, str]]" = {
    "gauss": ("gauss", "stress_xx"),
    "material-state": ("gauss", "damage"),
    "spring": ("springs", "spring_force_0"),
    "fiber": ("fibers", "fiber_stress"),
    "line-station": ("line_stations", "axial_force"),
}


@pytest.fixture
def every_family_malformed(tmp_path: Path) -> Path:
    path = tmp_path / "every_family.mpco"
    f, stage_name = _create_mpco_skeleton(
        path, node_ids=np.array([1, 2]), node_coords=np.zeros((2, 3)),
        n_steps=1, dt=1.0,
    )
    try:
        stage = f[stage_name]
        stage["MODEL"].create_group("ELEMENTS")
        stage["MODEL"].create_group("SECTION_ASSIGNMENTS")
        on_el = stage["RESULTS"].require_group("ON_ELEMENTS")
        for group, _discover, _component in _FAMILIES.values():
            on_el.require_group(group).create_group("not-a-bucket")
    finally:
        f.close()
    return path


@pytest.mark.parametrize("family", sorted(_READS))
def test_a_read_names_the_malformed_key_once(
    family: str, every_family_malformed: Path,
) -> None:
    level, component = _READS[family]
    group = _FAMILIES[family][0]
    with Results.from_mpco(
        str(every_family_malformed), model_h5=_stub_model_h5_path(),
    ) as r:
        s = r.stage(r.stages[0].id)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            slab = getattr(s.elements, level).get(component=component)
    assert slab.values.size == 0
    skips = _skips(record)
    assert len(skips) == 1, skips
    assert f"/{group}/not-a-bucket: its bracket key does not parse" in skips[0]
    assert "every_family.mpco" in skips[0]


# ---------------------------------------------------------------------
# The converter copies custom-rule springs: no "unread", springs flagged
# ---------------------------------------------------------------------

def test_custom_rule_springs_convert_flagged_and_quiet(tmp_path: Path) -> None:
    """``zl_springs.mpco`` with every ZeroLength bucket moved to custom
    index 2: the converter copies them verbatim, so it neither warns
    that they are left unread nor drops their ``springs`` flag."""
    from tests.test_results_mpco_to_ladruno import (
        SPRINGS,
        _bridge_copy,
        _convert,
        _prov,
        _springs_model,
    )

    model = _springs_model(tmp_path / "springs.h5")
    src = _bridge_copy(SPRINGS, model, tmp_path / "zl_custom.mpco")
    moved: list[str] = []
    with h5py.File(src, "r+") as f:
        for stage in (k for k in f if k.startswith("MODEL_STAGE[")):
            on_el = f[f"{stage}/RESULTS/ON_ELEMENTS"]
            for token in ("basicForce", "deformation"):
                grp = on_el[token]
                for key in list(grp):
                    new = key.replace("[1:0:", "[1:2:")
                    assert new != key
                    grp.move(key, new)
                    moved.append(f"{token}/{new}")
    assert moved

    # ``_convert`` turns every warning into an error.
    (dst,) = _convert(src, tmp_path / "zl_custom.ladruno", model)
    assert "springs" in json.loads(_prov(dst)["unsupported"])
    with h5py.File(dst, "r") as d:
        stage = next(k for k in d if k.startswith("MODEL_STAGE["))
        for path in moved:
            assert path in d[f"{stage}/RESULTS/ON_ELEMENTS"], path
