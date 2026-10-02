"""Types and helpers the STKO translator modules exchange (ADR 0111).

Fixed by internal_docs/stko_translator_rules.md section 3: change it there first.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .model import ScdModel

Vec3 = tuple[float, float, float]
Vec6 = tuple[float, float, float, float, float, float]
#: (geometry id, kind, 0-based sub-shape index), kind in KIND_DIM.
SubShape = tuple[int, str, int]

KIND_DIM: dict[str, int] = {"vertices": 0, "edges": 1, "faces": 2, "solids": 3}
#: SubShapeRef.kind (interaction masters / slaves) -> kind.
REF_KIND: dict[int, str] = {1: "vertices", 2: "edges", 3: "faces", 4: "solids"}

Category = Literal[
    "element_property", "physical_property", "condition", "definition",
    "analysis_step", "interaction", "mesh_element", "option",
]


@dataclass(frozen=True, slots=True)
class Unsupported:
    """One STKO object, or one option of an object, the translator will not translate."""

    category: Category
    xobj_meta: str              # STKO type; "<type>:<attribute>" for an option
    ids: tuple[int, ...]
    names: tuple[str, ...]
    reason: str                 # "tier-only: not in this version" | "unknown STKO type" | option text


class UnsupportedSTKOTypes(NotImplementedError):
    """Raised before the session is touched; lists every offender."""

    def __init__(self, items: tuple[Unsupported, ...]) -> None:
        self.items = items
        lines = [
            f"  [{u.category}] {u.xobj_meta} ids={list(u.ids)} names={list(u.names)}: {u.reason}"
            for u in items
        ]
        super().__init__(
            f"{len(items)} STKO type(s) or option(s) are not supported by this version "
            "of the translator:\n" + "\n".join(lines)
        )


@dataclass(frozen=True, slots=True)
class Ignored:
    """An STKO object deliberately not translated (it is not part of the model)."""

    category: Category
    xobj_meta: str
    ids: tuple[int, ...]
    names: tuple[str, ...]
    reason: str
    payload: str | None = None  # e.g. the Tcl of a customCommand


@dataclass(frozen=True, slots=True)
class ElementGroup:
    """Analysis elements sharing (element property, physical property, local axis)."""

    pg: str
    dim: int                    # 1 beams, 2 shells
    element_property: int
    physical_property: int
    local_axis: Vec3            # shell: -local x vector; beam: geomTransf vecxz (local_axis())
    element_ids: tuple[int, ...]  # STKO ids, ascending


@dataclass(frozen=True, slots=True)
class DiaphragmGroup:
    """One rigid-diaphragm master and its slaves (from the interaction's links)."""

    condition: int
    perp_dirn: int              # 1, 2 or 3
    master_node: int
    slave_nodes: tuple[int, ...]  # in link order
    master_pg: str
    slave_pg: str


@dataclass(frozen=True, slots=True)
class MeshMap:
    node_ids: tuple[int, ...]                                # session nodes, ascending
    element_groups: tuple[ElementGroup, ...]
    subshape_entities: Mapping[SubShape, tuple[int, ...]]   # gmsh tags, dim KIND_DIM[kind]
    selection_set_pgs: Mapping[str, Mapping[str, str]]      # set -> kind -> PG
    condition_pgs: Mapping[int, Mapping[str, str]]           # condition id -> kind -> PG
    diaphragms: tuple[DiaphragmGroup, ...]
    carrier_ids: range                                       # synthetic element ids, never emitted


@dataclass(frozen=True, slots=True)
class PropsResult:
    primitives: Mapping[int, Any]   # STKO physical-property id -> apeSees primitive
    elements: Mapping[str, Any]     # ElementGroup.pg -> element primitive
    transforms: Mapping[Vec3, Any]  # vecxz -> geomTransf primitive


@dataclass(frozen=True, slots=True)
class TimeSeriesSpec:
    stko_id: int
    type: Literal["Linear", "Path"]
    factor: float = 1.0
    dt: float | None = None
    values: tuple[float, ...] | None = None
    start_time: float | None = None   # None: not written
    placeholder: bool = False         # values are the .scd's own, no record supplied


@dataclass(frozen=True, slots=True)
class PatternSpec:
    stko_id: int                      # STKO analysis-step id (STKO's pattern tag)
    name: str
    kind: Literal["Plain", "UniformExcitation"]
    series: int                       # TimeSeriesSpec.stko_id
    loads: Mapping[int, Vec6] = field(default_factory=dict)  # node -> summed load
    conditions: tuple[int, ...] = ()  # load + massToLoad condition ids, STKO order
    direction: int | None = None      # UniformExcitation: 1..6


@dataclass(frozen=True, slots=True)
class RayleighSpec:
    alpha_m: float
    beta_k: float
    beta_k_init: float
    beta_k_comm: float


@dataclass(frozen=True, slots=True)
class StageSpec:
    stko_id: int
    name: str
    analysis: Literal["Static", "Transient"]
    patterns: tuple[int, ...]         # PatternSpec.stko_id first active in this stage
    n_incr: int
    duration: float
    load_const: bool
    integrator: tuple[str, tuple[float, ...]]  # ("LoadControl", (dlam,)), ("Newmark", (gamma, beta))
    chain: Mapping[str, Any]          # STKO's solver choices, informational


@dataclass(frozen=True, slots=True)
class ConditionSummary:
    """A condition as intent: the definitions source for the parametric recipes."""

    stko_id: int
    name: str
    type: str
    value: tuple[float, ...]          # per area / length / node; rho for AutoEdgeMass
    per: Literal["area", "length", "node", "volume", "none"]
    pgs: Mapping[str, str]            # kind -> PG (MeshMap.condition_pgs)
    patterns: tuple[int, ...]         # load patterns using it
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConditionsPlan:
    masses: Mapping[int, Vec6]
    fixes: Mapping[tuple[int, ...], tuple[int, ...]]   # 6-DOF mask -> nodes, ascending
    diaphragm_pairs: frozenset[tuple[int, int, int]]   # (perp, master, slave)
    time_series: tuple[TimeSeriesSpec, ...]
    patterns: tuple[PatternSpec, ...]                  # STKO step order
    constraint_conditions: tuple[int, ...]              # sp + mp of the constraint pattern
    rayleigh: RayleighSpec | None
    stages: tuple[StageSpec, ...]
    implex_dt_targets: tuple[int, ...]
    summaries: tuple[ConditionSummary, ...]
    ignored: tuple[Ignored, ...]


@dataclass(frozen=True, slots=True)
class TranslateResult:
    scd: "ScdModel"
    mesh: MeshMap
    plan: ConditionsPlan
    ignored: tuple[Ignored, ...]


def quat_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix of STKO's element quaternion ``(x, y, z, w)``;
    columns are the element's local x, y, z axes."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def local_axis(scd: "ScdModel", eid: int) -> Vec3:
    """Rules M4/M5: a shell's ``-local`` vector (column 0, STKO prints %.6g) or a
    beam's ``vecxz`` (column 2, rounded to 12 significant digits for grouping)."""
    M = quat_matrix(scd.mesh.orientation[eid])
    if len(scd.mesh.elements[eid].nodes) == 2:
        return tuple(float(f"{v:.12g}") + 0.0 for v in M[:, 2])  # type: ignore[return-value]
    return tuple(float(f"{v:.6g}") + 0.0 for v in M[:, 0])       # type: ignore[return-value]


def property_references(scd: "ScdModel", pid: int) -> frozenset[int]:
    """Physical properties ``pid`` references, transitively: its INDEX / INDEX_VEC
    attributes, plus the fiber-group materials inside a fiber section's custom
    object (which are not INDEX attributes)."""
    seen: set[int] = set()
    stack = [pid]
    while stack:
        x = scd.physical_properties[stack.pop()]
        refs: list[int] = []
        for name in x.references:
            v = x.attributes[name]
            refs.extend(v if isinstance(v, tuple) else (v,))
        fs = x.attributes.get("Fiber section")
        if isinstance(fs, dict):
            for grp in ("PUNCTUAL_FIBER_GROUPS", "SURFACE_FIBER_GROUPS", "LINEAR_FIBER_GROUPS"):
                for item in (fs.get(grp) or {}).values():
                    refs.append(int(np.asarray(item["@attrs"]["PHYS_PROP_ID"]).ravel()[0]))
        for r in refs:
            if r and r in scd.physical_properties and r not in seen:
                seen.add(r)
                stack.append(r)
    return frozenset(seen)
