"""Phase 0 — native schema path builder consistency."""
from __future__ import annotations

from apeGmsh.results.schema import _native, _versions


def test_schema_version_constants_are_strings() -> None:
    assert isinstance(_versions.SCHEMA_VERSION, str)
    assert isinstance(_versions.PARSER_VERSION, str)
    # Trivial format check — semver-ish
    assert "." in _versions.SCHEMA_VERSION


def test_top_level_groups() -> None:
    assert _native.MODEL_GROUP == "/model"
    assert _native.STAGES_GROUP == "/stages"


def test_stage_paths() -> None:
    sid = _native.stage_id(0)
    assert sid == "stage_0"
    assert _native.stage_path(sid) == "/stages/stage_0"
    assert _native.stage_time_path(sid) == "/stages/stage_0/time"


def test_partition_paths() -> None:
    sid = "stage_0"
    pid = _native.partition_id(2)
    assert pid == "partition_2"
    assert _native.partitions_path(sid) == "/stages/stage_0/partitions"
    assert _native.partition_path(sid, pid) == (
        "/stages/stage_0/partitions/partition_2"
    )


def test_nodes_paths() -> None:
    sid, pid = "stage_0", "partition_0"
    assert _native.nodes_path(sid, pid) == (
        "/stages/stage_0/partitions/partition_0/nodes"
    )
    assert _native.nodes_component_path(sid, pid, "displacement_x") == (
        "/stages/stage_0/partitions/partition_0/nodes/displacement_x"
    )


def test_element_subgroup_paths() -> None:
    sid, pid, gid = "stage_0", "partition_0", "group_0"
    base = "/stages/stage_0/partitions/partition_0/elements"

    assert _native.elements_path(sid, pid) == base
    assert _native.gauss_group_path(sid, pid, gid) == f"{base}/gauss_points/group_0"
    assert _native.fibers_group_path(sid, pid, gid) == f"{base}/fibers/group_0"
    assert _native.layers_group_path(sid, pid, gid) == f"{base}/layers/group_0"
    assert _native.line_stations_group_path(sid, pid, gid) == (
        f"{base}/line_stations/group_0"
    )
    assert _native.nodal_forces_group_path(sid, pid, gid) == (
        f"{base}/nodal_forces/group_0"
    )


def test_id_conventions() -> None:
    assert _native.stage_id(0) == "stage_0"
    assert _native.stage_id(7) == "stage_7"
    assert _native.partition_id(0) == "partition_0"
    assert _native.group_id(3) == "group_3"


def test_kind_constants() -> None:
    assert _native.KIND_TRANSIENT in _native.ALL_KINDS
    assert _native.KIND_STATIC in _native.ALL_KINDS
    assert _native.KIND_MODE in _native.ALL_KINDS
    assert len(_native.ALL_KINDS) == 3


def test_attr_constants_are_distinct() -> None:
    """Catch accidental duplicate attribute names."""
    attr_names = {
        v for k, v in vars(_native).items()
        if k.startswith("ATTR_") and isinstance(v, str)
    }
    # Number of ATTR_ constants in the module
    declared = sum(1 for k in vars(_native) if k.startswith("ATTR_"))
    assert len(attr_names) == declared, "Duplicate ATTR_ values"


def test_dataset_name_constants_are_underscore_prefixed() -> None:
    """All `DSET_` constants for index/metadata datasets start with underscore.

    The exception is ``DSET_TIME`` which is a result-adjacent dataset
    (the time vector lives at the stage level, not as an index).
    """
    for k, v in vars(_native).items():
        if not k.startswith("DSET_"):
            continue
        if not isinstance(v, str):
            continue
        if k == "DSET_TIME":
            continue
        assert v.startswith("_"), (
            f"Index/metadata dataset {k}={v!r} should be underscore-prefixed"
        )


def test_path_builders_are_compositional() -> None:
    """Higher-level paths nest the lower-level ones."""
    sid, pid, gid = "stage_5", "partition_3", "group_7"
    assert _native.partitions_path(sid).startswith(_native.stage_path(sid))
    assert _native.partition_path(sid, pid).startswith(
        _native.partitions_path(sid)
    )
    assert _native.nodes_path(sid, pid).startswith(
        _native.partition_path(sid, pid)
    )
    assert _native.gauss_group_path(sid, pid, gid).startswith(
        _native.elements_path(sid, pid)
    )
