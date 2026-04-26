"""Native HDF5 schema — path builders and attribute keys.

The schema layout is documented in
``internal_docs/Results_architecture.md``. This module is the single
source of truth for HDF5 paths and attribute names; readers and
writers reference these constants rather than hard-coding strings.

Conventions
-----------
- Underscore-prefixed dataset names (``_ids``, ``_element_index``,
  ``_natural_coords``, …) are index/metadata. No-prefix names
  (``displacement_x``, ``stress_xx``, …) are result components.
- Stage and partition IDs are arbitrary strings; ``stage_<n>`` and
  ``partition_<n>`` are conventions, not requirements. The reader
  enumerates them at discovery time.
"""
from __future__ import annotations


# =====================================================================
# Top-level groups
# =====================================================================

ROOT = "/"
MODEL_GROUP = "/model"
STAGES_GROUP = "/stages"


# =====================================================================
# Root attributes
# =====================================================================

ATTR_SCHEMA_VERSION = "schema_version"
ATTR_SOURCE_TYPE = "source_type"        # "tcl_recorders" | "domain_capture" | ...
ATTR_SOURCE_PATH = "source_path"
ATTR_CREATED_AT = "created_at"
ATTR_APEGMSH_VERSION = "apegmsh_version"
ATTR_ANALYSIS_LABEL = "analysis_label"


# =====================================================================
# Source-type constants
# =====================================================================

SOURCE_TCL_RECORDERS = "tcl_recorders"
SOURCE_DOMAIN_CAPTURE = "domain_capture"


# =====================================================================
# /model attributes (FEMData snapshot)
# =====================================================================

ATTR_SNAPSHOT_ID = "snapshot_id"
ATTR_NDM = "ndm"
ATTR_NDF = "ndf"
ATTR_MODEL_NAME = "model_name"
ATTR_UNITS = "units"


# =====================================================================
# Stage attributes
# =====================================================================

ATTR_STAGE_NAME = "name"
ATTR_STAGE_KIND = "kind"

# Mode-only stage attributes
ATTR_EIGENVALUE = "eigenvalue"
ATTR_FREQUENCY_HZ = "frequency_hz"
ATTR_PERIOD_S = "period_s"
ATTR_MODE_INDEX = "mode_index"

# Stage kinds
KIND_TRANSIENT = "transient"
KIND_STATIC = "static"
KIND_MODE = "mode"

ALL_KINDS = frozenset({KIND_TRANSIENT, KIND_STATIC, KIND_MODE})


# =====================================================================
# Element group attributes
# =====================================================================

ATTR_CLASS_TAG = "class_tag"
ATTR_INT_RULE = "int_rule"
ATTR_CUSTOM_RULE_IDX = "custom_rule_idx"
ATTR_FRAME = "frame"                    # "global" | "local" for nodal_forces
ATTR_SECTION_TAG = "section_tag"
ATTR_SECTION_CLASS = "section_class"


# =====================================================================
# Index/metadata dataset names (underscore-prefixed)
# =====================================================================

DSET_IDS = "_ids"
DSET_ELEMENT_INDEX = "_element_index"
DSET_GP_INDEX = "_gp_index"
DSET_LAYER_INDEX = "_layer_index"
DSET_SUB_GP_INDEX = "_sub_gp_index"
DSET_NATURAL_COORDS = "_natural_coords"
DSET_LOCAL_AXES_QUATERNION = "_local_axes_quaternion"
DSET_STATION_NATURAL_COORD = "_station_natural_coord"
DSET_THICKNESS = "_thickness"
DSET_Y = "_y"
DSET_Z = "_z"
DSET_AREA = "_area"
DSET_MATERIAL_TAG = "_material_tag"

# Per-stage time vector
DSET_TIME = "time"


# =====================================================================
# Element-level subgroup names
# =====================================================================

GROUP_NODES = "nodes"
GROUP_ELEMENTS = "elements"
GROUP_NODAL_FORCES = "nodal_forces"
GROUP_LINE_STATIONS = "line_stations"
GROUP_GAUSS_POINTS = "gauss_points"
GROUP_FIBERS = "fibers"
GROUP_LAYERS = "layers"
GROUP_PARTITIONS = "partitions"


# =====================================================================
# Path builders
# =====================================================================

def stage_path(stage_id: str) -> str:
    """Path to a stage group: ``/stages/<stage_id>``."""
    return f"{STAGES_GROUP}/{stage_id}"


def stage_time_path(stage_id: str) -> str:
    """Path to the time vector dataset for a stage."""
    return f"{stage_path(stage_id)}/{DSET_TIME}"


def partitions_path(stage_id: str) -> str:
    """Path to the partitions group of a stage."""
    return f"{stage_path(stage_id)}/{GROUP_PARTITIONS}"


def partition_path(stage_id: str, partition_id: str) -> str:
    """Path to a single partition group within a stage."""
    return f"{partitions_path(stage_id)}/{partition_id}"


def nodes_path(stage_id: str, partition_id: str) -> str:
    """Path to the nodes/ group within a partition."""
    return f"{partition_path(stage_id, partition_id)}/{GROUP_NODES}"


def nodes_component_path(stage_id: str, partition_id: str, component: str) -> str:
    """Path to a nodal component dataset (``nodes/<component>``)."""
    return f"{nodes_path(stage_id, partition_id)}/{component}"


def elements_path(stage_id: str, partition_id: str) -> str:
    """Path to the elements/ group within a partition."""
    return f"{partition_path(stage_id, partition_id)}/{GROUP_ELEMENTS}"


def gauss_group_path(
    stage_id: str, partition_id: str, group_id: str,
) -> str:
    """Path to a single Gauss-points group: ``elements/gauss_points/<group_id>``."""
    return f"{elements_path(stage_id, partition_id)}/{GROUP_GAUSS_POINTS}/{group_id}"


def fibers_group_path(
    stage_id: str, partition_id: str, group_id: str,
) -> str:
    """Path to a single fibers group: ``elements/fibers/<group_id>``."""
    return f"{elements_path(stage_id, partition_id)}/{GROUP_FIBERS}/{group_id}"


def layers_group_path(
    stage_id: str, partition_id: str, group_id: str,
) -> str:
    """Path to a single layers group: ``elements/layers/<group_id>``."""
    return f"{elements_path(stage_id, partition_id)}/{GROUP_LAYERS}/{group_id}"


def line_stations_group_path(
    stage_id: str, partition_id: str, group_id: str,
) -> str:
    """Path to a single line-stations group: ``elements/line_stations/<group_id>``."""
    return f"{elements_path(stage_id, partition_id)}/{GROUP_LINE_STATIONS}/{group_id}"


def nodal_forces_group_path(
    stage_id: str, partition_id: str, group_id: str,
) -> str:
    """Path to a single nodal-forces group: ``elements/nodal_forces/<group_id>``."""
    return f"{elements_path(stage_id, partition_id)}/{GROUP_NODAL_FORCES}/{group_id}"


# =====================================================================
# ID conventions (for readability — no enforcement)
# =====================================================================

def stage_id(index: int) -> str:
    """Conventional stage ID: ``stage_<index>``."""
    return f"stage_{index}"


def partition_id(index: int) -> str:
    """Conventional partition ID: ``partition_<index>``."""
    return f"partition_{index}"


def group_id(index: int) -> str:
    """Conventional element-group ID: ``group_<index>``."""
    return f"group_{index}"
