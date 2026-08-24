"""Canonical machine-readable inventory of the published neutral zone.

This module is the **single source of truth** for what
``docs/design/model-h5-neutral-zone.md`` publishes.  The page is prose
*about* this table; when the two disagree, this table wins and the page
is the thing to update.

``tests/mesh/test_neutral_zone_schema_note.py`` enforces both directions:

* the emitter's output (the committed golden **and** a freshly generated
  file) must match this inventory — dtype, rank, attrs — so emitter
  drift fails there;
* every path in this inventory must appear verbatim in the published
  page, and every ``/``-path the page documents must appear here, so
  doc drift fails there too.

Scope is the **consumer-facing** neutral zone — the groups an external
reader (no ``apeGmsh`` import, no ``h5py``) needs to draw a model:
``/meta``, ``/nodes``, ``/elements``, ``/physical_groups``, ``/labels``,
``/mesh_selections``.  The rest of the neutral zone (records: loads,
masses, constraints, ties, contacts, interfaces, and the compose
provenance groups) is real and is written by the same writer, but it is
**not** published here — see :data:`KNOWN_UNPUBLISHED_ROOT_GROUPS` and
``src/apeGmsh/opensees/architecture/h5-schema.md`` for those.

Path patterns use ``{type}`` / ``{name}`` for the one variable segment
in a path; every other segment is literal.
"""
from __future__ import annotations

from dataclasses import dataclass, field


__all__ = [
    "ATTR_INT",
    "ATTR_STR",
    "DATASETS",
    "GROUPS",
    "KNOWN_UNPUBLISHED_ROOT_GROUPS",
    "PUBLISHED_ROOT_GROUPS",
    "DatasetSpec",
    "GroupSpec",
]


#: Attribute type tokens.  ``h5py`` reports a scalar string attr as
#: ``str`` and a scalar integer attr as ``numpy.int64``; both are what
#: a plain ``attrs[...]`` read yields.
ATTR_STR = "str"
ATTR_INT = "int64"


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset in the published neutral zone.

    ``dtype`` is a canonical token, not a numpy dtype string:

    * ``"int64"`` / ``"float64"`` / ``"int8"`` — plain numeric.
    * ``"vlen_utf8"`` — HDF5 variable-length UTF-8 string.  ``h5py``
      reports ``dtype == object`` and hands back ``bytes`` on read;
      a browser reader (h5wasm) sees a variable-length string type.
    """

    dtype: str
    rank: int
    required: bool = True
    note: str = ""


@dataclass(frozen=True)
class GroupSpec:
    """One group in the published neutral zone."""

    required: bool = True
    attrs: dict[str, str] = field(default_factory=dict)
    optional_attrs: dict[str, str] = field(default_factory=dict)
    note: str = ""


#: Root groups this note publishes.
PUBLISHED_ROOT_GROUPS: frozenset[str] = frozenset({
    "meta",
    "nodes",
    "elements",
    "physical_groups",
    "labels",
    "mesh_selections",
})

#: Root groups the same writer may emit that this note deliberately does
#: NOT publish.  Listed so the drift test can tell "known, out of scope"
#: from "brand-new group nobody documented".  Sourced from
#: ``mesh/_femdata_h5_io.py:write_neutral_zone``.
KNOWN_UNPUBLISHED_ROOT_GROUPS: frozenset[str] = frozenset({
    "partitions",
    "parts",
    "constraints",
    "reinforce_ties",
    "embed_ties",
    "rebar_elements",
    "contacts",
    "contact_planes",
    "interfaces",
    "loads",
    "masses",
    "composed_from",
})


GROUPS: dict[str, GroupSpec] = {
    "/meta": GroupSpec(
        required=True,
        attrs={
            "schema_version": ATTR_STR,
            "neutral_schema_version": ATTR_STR,
            "opensees_schema_version": ATTR_STR,
            "apeGmsh_version": ATTR_STR,
            "created_iso": ATTR_STR,
            "ndm": ATTR_INT,
            "ndf": ATTR_INT,
            "snapshot_id": ATTR_STR,
            "model_name": ATTR_STR,
            "tag_span_max": ATTR_INT,
        },
        note="attrs only; no datasets",
    ),
    "/meta/lineage": GroupSpec(
        required=False,
        attrs={},
        optional_attrs={
            "fem_hash": ATTR_STR,
            "model_hash": ATTR_STR,
            "results_hash": ATTR_STR,
        },
        note="ADR 0021 hash chain; broker-only files carry fem_hash alone",
    ),
    "/nodes": GroupSpec(required=True),
    "/elements": GroupSpec(
        required=True,
        note="always created, even when the snapshot has no elements",
    ),
    "/elements/{type}": GroupSpec(
        required=False,
        attrs={
            "code": ATTR_INT,
            "gmsh_name": ATTR_STR,
            "npe": ATTR_INT,
            "dim": ATTR_INT,
            "order": ATTR_INT,
        },
        note="one per gmsh element-type alias present in the snapshot",
    ),
    "/physical_groups": GroupSpec(required=False),
    "/physical_groups/node_side": GroupSpec(required=False),
    "/physical_groups/element_side": GroupSpec(required=False),
    "/physical_groups/node_side/{name}": GroupSpec(
        required=False,
        attrs={"dim": ATTR_INT, "tag": ATTR_INT, "name": ATTR_STR},
    ),
    "/physical_groups/element_side/{name}": GroupSpec(
        required=False,
        attrs={"dim": ATTR_INT, "tag": ATTR_INT, "name": ATTR_STR},
    ),
    "/labels": GroupSpec(required=False),
    "/labels/node_side": GroupSpec(required=False),
    "/labels/element_side": GroupSpec(required=False),
    "/labels/node_side/{name}": GroupSpec(
        required=False,
        attrs={"dim": ATTR_INT, "tag": ATTR_INT, "name": ATTR_STR},
    ),
    "/labels/element_side/{name}": GroupSpec(
        required=False,
        attrs={"dim": ATTR_INT, "tag": ATTR_INT, "name": ATTR_STR},
    ),
    "/mesh_selections": GroupSpec(required=False),
    "/mesh_selections/{name}": GroupSpec(
        required=False,
        attrs={"dim": ATTR_INT, "tag": ATTR_INT, "name": ATTR_STR},
        note="flat — no node_side/element_side split here",
    ),
}


DATASETS: dict[str, DatasetSpec] = {
    # ---------------------------------------------------------------- nodes
    "/nodes/ids": DatasetSpec(
        "int64", 1,
        note="node ids; gmsh-assigned, 1-based, not necessarily contiguous",
    ),
    "/nodes/coords": DatasetSpec(
        "float64", 2,
        note="(N, 3) ALWAYS — three columns even when /meta@ndm == 2",
    ),
    "/nodes/module_label": DatasetSpec(
        "vlen_utf8", 1,
        note="compose provenance; empty string for host-owned rows",
    ),
    "/nodes/ndf": DatasetSpec(
        "int8", 1, required=False,
        note="per-node DOF count; 0 = undeclared. Absent unless declared",
    ),
    "/nodes/provenance": DatasetSpec(
        "int8", 1, required=False,
        note="0 = mesh node, 1 = decoupled node. Absent when none",
    ),
    # ------------------------------------------------------------- elements
    "/elements/{type}/ids": DatasetSpec(
        "int64", 1,
        note="element ids; gmsh-assigned, globally unique across types",
    ),
    "/elements/{type}/connectivity": DatasetSpec(
        "int64", 2,
        note="(E, npe) of NODE IDS — values of /nodes/ids, not row indices",
    ),
    "/elements/{type}/module_label": DatasetSpec(
        "vlen_utf8", 1,
        note="compose provenance; empty string for host-owned rows",
    ),
    # ------------------------------------------------- physical groups
    "/physical_groups/node_side/{name}/node_ids": DatasetSpec("int64", 1),
    "/physical_groups/node_side/{name}/node_coords": DatasetSpec(
        "float64", 2,
    ),
    "/physical_groups/element_side/{name}/node_ids": DatasetSpec("int64", 1),
    "/physical_groups/element_side/{name}/node_coords": DatasetSpec(
        "float64", 2,
    ),
    "/physical_groups/element_side/{name}/element_ids": DatasetSpec(
        "int64", 1, required=False,
        note="written only when non-empty and dim >= 1; ids MAY be absent "
             "from every /elements/{type} group",
    ),
    # ----------------------------------------------------------- labels
    "/labels/node_side/{name}/node_ids": DatasetSpec("int64", 1),
    "/labels/node_side/{name}/node_coords": DatasetSpec("float64", 2),
    "/labels/element_side/{name}/node_ids": DatasetSpec("int64", 1),
    "/labels/element_side/{name}/node_coords": DatasetSpec("float64", 2),
    "/labels/element_side/{name}/element_ids": DatasetSpec(
        "int64", 1, required=False,
    ),
    # -------------------------------------------------- mesh selections
    "/mesh_selections/{name}/node_ids": DatasetSpec("int64", 1),
    "/mesh_selections/{name}/node_coords": DatasetSpec("float64", 2),
    "/mesh_selections/{name}/element_ids": DatasetSpec(
        "int64", 1, required=False,
    ),
    "/mesh_selections/{name}/connectivity": DatasetSpec(
        "int64", 2, required=False,
        note="rows align 1:1 with element_ids",
    ),
}
