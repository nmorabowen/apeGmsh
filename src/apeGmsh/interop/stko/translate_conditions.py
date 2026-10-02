"""STKO conditions, definitions and analysis steps -> a plan -> the bridge (ADR 0111).

Work package C of ``internal_docs/stko_translator_rules.md`` section 8, rules
C1-C14. Three stages, so a recipe can use any of them alone:

* :func:`plan_conditions` -- pure numpy over an :class:`ScdModel`. Reproduces
  what STKO's own OpenSees exporter writes for masses, loads, fixities, rigid
  diaphragms, time series, patterns, Rayleigh damping and the analysis stages.
  No session, no bridge.
* :func:`declare_session` -- the rigid diaphragms, on ``g.constraints``.
* :func:`build_conditions` -- the plan, declared on an ``apeSees`` bridge.

The type registry (:data:`REGISTRY`) is data: a later STKO type is one entry
plus its rule. A type the registry does not list, a tier-only type, or an
option of a supported type this version does not translate is reported by
:func:`check_supported`, never skipped silently.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .model import Condition, ScdModel, XObject
from .translate_types import (
    ConditionsPlan,
    ConditionSummary,
    DiaphragmGroup,
    Ignored,
    MeshMap,
    PatternSpec,
    RayleighSpec,
    StageSpec,
    TimeSeriesSpec,
    Unsupported,
    UnsupportedSTKOTypes,
    Vec6,
    property_references,
)

__all__ = [
    "REGISTRY",
    "check_supported",
    "diaphragm_groups",
    "plan_conditions",
    "declare_session",
    "resolved_diaphragm_pairs",
    "verify_diaphragms",
    "build_conditions",
]

Status = Literal["supported", "tier_only", "ignored"]

TIER_ONLY = "tier-only: not in this version"
UNKNOWN = "unknown STKO type"

#: (category, XOBJ_META) -> status. Anything not listed is "unknown STKO type".
REGISTRY: dict[tuple[str, str], Status] = {
    # conditions
    ("condition", "Constraints.sp.fix"): "supported",
    ("condition", "Constraints.mp.rigidDiaphragm"): "supported",
    ("condition", "Mass.FaceMass"): "supported",
    ("condition", "Mass.AutoEdgeMass"): "supported",
    ("condition", "Mass.NodeMass"): "supported",
    ("condition", "Loads.Force.FaceForce"): "supported",
    ("condition", "Loads.Force.EdgeForce"): "supported",
    ("condition", "Loads.Force.NodeForce"): "supported",
    ("condition", "Constraints.mp.ASDEmbeddedNodeElement"): "tier_only",
    ("condition", "Loads.Generic.H5DRM"): "tier_only",
    # definitions
    ("definition", "timeSeries.Linear"): "supported",
    ("definition", "timeSeries.Path"): "supported",
    # analysis steps
    ("analysis_step", "Patterns.addPattern.loadPattern"): "supported",
    ("analysis_step", "Patterns.addPattern.constraintPattern"): "supported",
    ("analysis_step", "Patterns.addPattern.UniformExcitation"): "supported",
    ("analysis_step", "Misc_commands.rayleigh"): "supported",
    ("analysis_step", "Analyses.AnalysesCommand"): "supported",
    ("analysis_step", "Patterns.addPattern.H5DRM"): "tier_only",
    ("analysis_step", "Misc_commands.ASDAbsorbingBoundaryActivate"): "tier_only",
    ("analysis_step", "Recorders.MPCORecorder"): "ignored",
    ("analysis_step", "Misc_commands.region"): "ignored",
    ("analysis_step", "Misc_commands.monitor"): "ignored",
    ("analysis_step", "Misc_commands.customCommand"): "ignored",
    ("analysis_step", "Misc_commands.ImplexAutoErrorControlActivate"): "ignored",
}

_IGNORED_WHY = {
    "Recorders.MPCORecorder": "recorders are the caller's: declare them on the bridge",
    "Misc_commands.region": "regions are the caller's (a recorder helper in STKO)",
    "Misc_commands.monitor": "STKO monitors plot while STKO runs; they are not model state",
    "Misc_commands.customCommand": "a user Tcl command is not translated; its text is the payload",
    "Misc_commands.ImplexAutoErrorControlActivate": (
        "no analysis follows it, so it controls nothing (like a pattern after the last analysis)"
    ),
}

#: ignored analysis-step types that are ignored only when no analysis follows them;
#: before an analysis they change how it runs and are refused.
_IGNORED_ONLY_AFTER_LAST_ANALYSIS = frozenset({"Misc_commands.ImplexAutoErrorControlActivate"})

#: material types whose elements STKO lists as IMPL-EX dTime targets (C13).
_DTIME_TYPES = frozenset({
    "ASDConcrete1D", "ASDConcrete3D", "DamageTC1D", "DamageTC3D",
    "ASDBondSlip", "ASDSteel1D",
})

_MASS_TYPES = ("Mass.FaceMass", "Mass.AutoEdgeMass", "Mass.NodeMass")
_LOAD_TYPES = ("Loads.Force.FaceForce", "Loads.Force.EdgeForce", "Loads.Force.NodeForce")
_UE_DIRECTION = {"dx": 1, "dy": 2, "dz": 3, "rx": 4, "ry": 5, "rz": 6}
#: constrained DOFs of OpenSees' rigidDiaphragm, by perpendicular direction.
_DIAPHRAGM_DOFS = {1: [2, 3, 4], 2: [1, 3, 5], 3: [1, 2, 6]}

_PLAIN = "Patterns.addPattern.loadPattern"
_CONSTRAINT = "Patterns.addPattern.constraintPattern"
_UE = "Patterns.addPattern.UniformExcitation"
_RAYLEIGH = "Misc_commands.rayleigh"
_ANALYSES = "Analyses.AnalysesCommand"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _ids(value: Any) -> tuple[int, ...]:
    """An INDEX_VEC attribute as ids, zeros (STKO's empty slot) dropped."""
    if value is None:
        return ()
    if isinstance(value, (int, np.integer)):
        value = (value,)
    return tuple(int(v) for v in value if int(v))


def _vec3(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


class _Issues:
    """Collects :class:`Unsupported` rows, grouped by (category, type, reason)."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, str], tuple[set[int], set[str]]] = {}

    def add(self, category: str, xobj_meta: str, ident: int, name: str, reason: str) -> None:
        ids, names = self._rows.setdefault((category, xobj_meta, reason), (set(), set()))
        ids.add(int(ident))
        names.add(name)

    def rows(self) -> list[Unsupported]:
        return [
            Unsupported(
                category=c, xobj_meta=t, ids=tuple(sorted(ids)),  # type: ignore[arg-type]
                names=tuple(sorted(names)), reason=r,
            )
            for (c, t, r), (ids, names) in sorted(self._rows.items())
        ]


def _status(category: str, xobj_meta: str) -> Status | None:
    return REGISTRY.get((category, xobj_meta))


# ---------------------------------------------------------------------------
# what the document references
# ---------------------------------------------------------------------------

class _Refs:
    """Which conditions and definitions the analysis steps reference."""

    def __init__(self, scd: ScdModel) -> None:
        self.load_patterns: list[XObject] = []
        self.constraint_patterns: list[XObject] = []
        self.ue_patterns: list[XObject] = []
        self.first_analysis: int | None = None
        self.last_analysis: int | None = None
        for sid, st in scd.analysis_steps.items():
            if st.type == _PLAIN:
                self.load_patterns.append(st)
            elif st.type == _CONSTRAINT:
                self.constraint_patterns.append(st)
            elif st.type == _UE:
                self.ue_patterns.append(st)
            elif st.type == _ANALYSES:
                if self.first_analysis is None:
                    self.first_analysis = sid
                self.last_analysis = sid

    def after_last_analysis(self, sid: int) -> bool:
        """True when no Analyses command follows step ``sid`` (steps run in id order)."""
        return self.last_analysis is None or sid > self.last_analysis

    def mp_conditions(self) -> list[int]:
        """Condition ids under ``mp`` of every constraint pattern, STKO order."""
        return [cid for st in self.constraint_patterns
                for key, cid in self.pattern_conditions(st) if key == "mp"]

    def pattern_conditions(self, st: XObject) -> list[tuple[str, int]]:
        """``(attribute, condition id)`` of a pattern step, STKO order."""
        out: list[tuple[str, int]] = []
        keys = ("sp", "mp") if st.type == _CONSTRAINT else (
            "load", "eleLoad", "sp", "genericLoad", "massToLoad")
        for key in keys:
            out.extend((key, c) for c in _ids(st.attributes.get(key)))
        return out

    def referenced_conditions(self) -> dict[int, list[int]]:
        """Condition id -> ids of the pattern steps listing it."""
        out: dict[int, list[int]] = defaultdict(list)
        for st in (*self.load_patterns, *self.constraint_patterns):
            for _, cid in self.pattern_conditions(st):
                if st.id not in out[cid]:
                    out[cid].append(st.id)
        return out

    def referenced_definitions(self) -> set[int]:
        return {
            int(st.attributes["tsTag"]) for st in (*self.load_patterns, *self.ue_patterns)
            if st.attributes.get("tsTag")
        }


# ---------------------------------------------------------------------------
# geometry access (vectorised lumping)
# ---------------------------------------------------------------------------

_G = 1.0 / math.sqrt(3.0)
_GAUSS_QUAD = ((-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G))   # 2x2, weights 1
_GAUSS_LINE = (-_G, _G)                                         # 2 points, weights 1


class _Geo:
    """Mesh lookups over an :class:`ScdModel`: element connectivity and coordinates."""

    def __init__(self, scd: ScdModel) -> None:
        self.scd = scd
        self._ids = np.asarray(scd.mesh.node_ids)
        self._xyz = np.asarray(scd.mesh.coordinates, dtype=float)

    def elements(self, gid: int, kind: str, indices: Sequence[int]) -> np.ndarray:
        """Element ids meshed on the given 0-based sub-shapes."""
        dom = self.scd.mesh.domains.get((gid, kind), {})
        parts = [np.asarray(dom[i], dtype=np.int64) for i in indices if i in dom]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def connectivity(self, eids: np.ndarray) -> np.ndarray:
        """``(n, k)`` node ids of equal-length elements (``(0, 0)`` for none)."""
        if len(eids) == 0:
            return np.empty((0, 0), dtype=np.int64)
        rows = [self.scd.mesh.elements[int(e)].nodes for e in eids]
        if len({len(r) for r in rows}) != 1:
            raise ValueError("mixed element node counts on one assignment")
        return np.asarray(rows, dtype=np.int64)

    def coords(self, nodes: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self._ids, nodes)
        idx = np.clip(idx, 0, len(self._ids) - 1)
        if not np.array_equal(self._ids[idx], nodes):
            raise KeyError("an element refers to a node that is not in the mesh")
        return self._xyz[idx]

    def vertex_node(self, gid: int, index: int) -> int:
        return int(self.scd.mesh.vertex_nodes[gid][index])


def _lump(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Consistent lumping ``integral of N_i * V`` per element node.

    ``P`` is ``(n, k, 3)`` node coordinates (k = 4 quad, 2 line), ``V`` is
    ``(n, 3)`` the (constant per element) value. Returns ``(n, k, 3)``. 2x2
    Gauss on the quad, 2-point Gauss on the line: STKO's own integration rule,
    exact for a constant value (rules C1-C3).
    """
    n, k, _ = P.shape
    out = np.zeros((n, k, 3))
    if k == 4:
        for xi, eta in _GAUSS_QUAD:
            N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                 (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
            dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            t1 = np.einsum("k,nkj->nj", dxi, P)
            t2 = np.einsum("k,nkj->nj", deta, P)
            det = np.linalg.norm(np.cross(t1, t2), axis=1)
            out += N[None, :, None] * det[:, None, None] * V[:, None, :]
    elif k == 2:
        half = 0.5 * np.linalg.norm(P[:, 1] - P[:, 0], axis=1)
        for xi in _GAUSS_LINE:
            N = np.array([(1 - xi) / 2, (1 + xi) / 2])
            out += N[None, :, None] * half[:, None, None] * V[:, None, :]
    else:
        raise NotImplementedError(f"nodal lumping for {k}-node elements")
    return out


def _quat_matrices(q: np.ndarray) -> np.ndarray:
    """``(n, 3, 3)`` rotation matrices of STKO quaternions ``(x, y, z, w)``
    (same formula as :func:`translate_types.quat_matrix`)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class _Acc:
    """Sums per-node vectors that arrive in chunks."""

    def __init__(self, width: int = 3) -> None:
        self.width = width
        self._n: list[np.ndarray] = []
        self._v: list[np.ndarray] = []

    def add(self, nodes: np.ndarray, values: np.ndarray) -> None:
        self._n.append(np.asarray(nodes, dtype=np.int64).reshape(-1))
        self._v.append(np.asarray(values, dtype=float).reshape(-1, self.width))

    def sums(self) -> dict[int, np.ndarray]:
        if not self._n:
            return {}
        N = np.concatenate(self._n)
        V = np.concatenate(self._v)
        uniq, inv = np.unique(N, return_inverse=True)
        out = np.zeros((len(uniq), self.width))
        np.add.at(out, inv, V)
        return {int(u): out[i] for i, u in enumerate(uniq)}


# ---------------------------------------------------------------------------
# sections: the area AutoEdgeMass uses (rule C1)
# ---------------------------------------------------------------------------

def _section_areas(scd: ScdModel, pid: int) -> list[float]:
    """Areas of every area-bearing section in ``pid`` and what it references.

    STKO (``AutoEdgeMass._getCrossArea``) averages over the property and the
    properties it references: sections.Elastic -> ``Section/PROPS[0]`` (no
    modifier), sections.Fiber -> the SURFACE fibers' areas (rebar excluded),
    sections.RectangularFiberSection -> Width x Height.
    """
    areas: list[float] = []
    for p in (pid, *sorted(property_references(scd, pid))):
        x = scd.physical_properties[p]
        if x.type == "sections.Elastic":
            areas.append(float(np.asarray(x.attributes["Section"]["PROPS"], float).ravel()[0]))
        elif x.type == "sections.Fiber":
            fs = x.attributes["Fiber section"]
            areas.append(sum(
                float(np.asarray(it["FIBERS"], float).reshape(-1, 3)[:, 2].sum())
                for it in (fs.get("SURFACE_FIBER_GROUPS") or {}).values()
            ))
        elif x.type == "sections.RectangularFiberSection":
            areas.append(float(x.attributes["Width"]) * float(x.attributes["Height"]))
    return areas


def _edge_property(scd: ScdModel, gid: int, index: int) -> int:
    return int(scd.geometries[gid].physical_property["edges"][index])


# ---------------------------------------------------------------------------
# nodal rules, one function per STKO condition type
# ---------------------------------------------------------------------------

def _mass_face(geo: _Geo, c: Condition, acc: _Acc) -> None:
    m = _vec3(c.xobject.attributes["mass"])
    for gid, sub in c.geometry.items():
        eids = geo.elements(gid, "faces", sub.faces)
        if len(eids):
            conn = geo.connectivity(eids)
            lumped = _lump(geo.coords(conn), np.broadcast_to(m, (len(eids), 3)))
            acc.add(conn, lumped)


def _mass_edge_auto(geo: _Geo, c: Condition, acc: _Acc) -> None:
    rho = float(c.xobject.attributes["rho"])
    for gid, sub in c.geometry.items():
        for i in sub.edges:
            eids = geo.elements(gid, "edges", [i])
            if len(eids):
                ma = rho * _mean_area(geo.scd, _edge_property(geo.scd, gid, i))
                conn = geo.connectivity(eids)
                v = np.full((len(eids), 3), ma)
                acc.add(conn, _lump(geo.coords(conn), v))


def _mass_node(geo: _Geo, c: Condition, acc: _Acc) -> None:
    m = _vec3(c.xobject.attributes["mass"])
    for gid, sub in c.geometry.items():
        for i in sub.vertices:
            acc.add(np.array([geo.vertex_node(gid, i)]), m[None, :])


_MASS_RULES: dict[str, Callable[[_Geo, Condition, _Acc], None]] = {
    "Mass.FaceMass": _mass_face,
    "Mass.AutoEdgeMass": _mass_edge_auto,
    "Mass.NodeMass": _mass_node,
}


def _mean_area(scd: ScdModel, pid: int) -> float:
    areas = _section_areas(scd, pid)
    return sum(areas) / len(areas)


def _distributed_force(geo: _Geo, c: Condition, kind: str, acc: _Acc) -> None:
    a = c.xobject.attributes
    F = _vec3(a["F"])
    local = not bool(a.get("Global", True))
    for gid, sub in c.geometry.items():
        eids = geo.elements(gid, kind, sub.of(kind))
        if not len(eids):
            continue
        conn = geo.connectivity(eids)
        V = np.broadcast_to(F, (len(eids), 3))
        if local:
            q = np.array([geo.scd.mesh.orientation[int(e)] for e in eids], dtype=float)
            V = np.einsum("nij,j->ni", _quat_matrices(q), F)
        acc.add(conn, _lump(geo.coords(conn), V))


def _force_face(geo: _Geo, c: Condition, acc: _Acc) -> None:
    _distributed_force(geo, c, "faces", acc)


def _force_edge(geo: _Geo, c: Condition, acc: _Acc) -> None:
    _distributed_force(geo, c, "edges", acc)


def _force_node(geo: _Geo, c: Condition, acc: _Acc) -> None:
    F = _vec3(c.xobject.attributes["F"])
    for gid, sub in c.geometry.items():
        for i in sub.vertices:
            acc.add(np.array([geo.vertex_node(gid, i)]), F[None, :])


_LOAD_RULES: dict[str, Callable[[_Geo, Condition, _Acc], None]] = {
    "Loads.Force.FaceForce": _force_face,
    "Loads.Force.EdgeForce": _force_edge,
    "Loads.Force.NodeForce": _force_node,
}


def _mass_to_load(geo: _Geo, c: Condition, acc: _Acc) -> None:
    """Rule C5: AutoEdgeMass "Convert to load": ``integral N_i (g rho A)`` per edge."""
    a = c.xobject.attributes
    g = np.array([a["gx"], a["gy"], a["gz"]], dtype=float)
    rho = float(a["rho"])
    for gid, sub in c.geometry.items():
        for i in sub.edges:
            eids = geo.elements(gid, "edges", [i])
            if len(eids):
                ma = rho * _mean_area(geo.scd, _edge_property(geo.scd, gid, i))
                conn = geo.connectivity(eids)
                v = np.broadcast_to(g * ma, (len(eids), 3))
                acc.add(conn, _lump(geo.coords(conn), v))


def _condition_nodes(geo: _Geo, c: Condition) -> set[int]:
    """Rule C6: every node of every mesh element (analysis or not) on the
    assigned edges / faces / solids, plus the assigned vertices' nodes."""
    nodes: set[int] = set()
    for gid, sub in c.geometry.items():
        for kind in ("edges", "faces", "solids"):
            for e in geo.elements(gid, kind, sub.of(kind)):
                nodes.update(int(n) for n in geo.scd.mesh.elements[int(e)].nodes)
        nodes.update(geo.vertex_node(gid, i) for i in sub.vertices)
    return nodes


# ---------------------------------------------------------------------------
# fixities and diaphragms
# ---------------------------------------------------------------------------

_FIX_KEYS = ("Ux", "Uy", "Uz", "Rx", "Ry", "Rz/3D")


def _fix_mask(c: Condition) -> tuple[int, ...]:
    a = c.xobject.attributes
    return tuple(int(bool(a[k])) for k in _FIX_KEYS)


def _fix_table(
    geo: _Geo, scd: ScdModel, cids: Sequence[int],
) -> tuple[dict[int, tuple[int, ...]], dict[int, set[tuple[int, ...]]]]:
    """``(node -> mask, node -> {masks} for nodes with conflicting masks)``."""
    table: dict[int, tuple[int, ...]] = {}
    conflicts: dict[int, set[tuple[int, ...]]] = {}
    for cid in cids:
        c = scd.conditions[cid]
        if c.type != "Constraints.sp.fix":
            continue
        mask = _fix_mask(c)
        for n in _condition_nodes(geo, c):
            if n in table and table[n] != mask:
                conflicts.setdefault(n, {table[n]}).add(mask)
            table[n] = mask
    return table, conflicts


_RIGID_DIAPHRAGM = "Constraints.mp.rigidDiaphragm"


def _links(scd: ScdModel, cids: Sequence[int]) -> list[tuple[int, int, int, list[int]]]:
    """``(condition, perp, master, slaves)`` per link element of each listed
    rigidDiaphragm condition (``nodes[0]`` master, ``nodes[1:]`` slaves), STKO order."""
    out: list[tuple[int, int, int, list[int]]] = []
    for cid in dict.fromkeys(cids):
        c = scd.conditions.get(cid)
        if c is None or c.type != _RIGID_DIAPHRAGM:
            continue
        perp = int(c.xobject.attributes.get("perpDirn", 0))
        for iid in c.interactions:
            inter = scd.interactions.get(iid)
            for e in (inter.elements if inter is not None else ()):
                nodes = scd.mesh.elements[e].nodes
                out.append((cid, perp, int(nodes[0]), [int(s) for s in nodes[1:]]))
    return out


def diaphragm_groups(scd: ScdModel) -> tuple[DiaphragmGroup, ...]:
    """Rule C7 as groups: one :class:`DiaphragmGroup` per (referenced condition,
    master), slaves in link order without repeats.

    Only rigidDiaphragm conditions a constraint pattern lists under ``mp`` are
    groups (STKO writes no ``rigidDiaphragm`` for any other). The mesh module
    builds its carriers from exactly these groups and the plan's
    ``diaphragm_pairs`` are their pairs, so the two cannot disagree. PG names
    are ``rd:{condition}:{master}:master`` / ``...:slaves``. Pure.
    """
    groups: dict[tuple[int, int], tuple[int, list[int]]] = {}
    for cid, perp, master, slaves in _links(scd, _Refs(scd).mp_conditions()):
        _, acc = groups.setdefault((cid, master), (perp, []))
        acc.extend(slaves)
    out = []
    for (cid, master), (perp, slaves) in sorted(groups.items()):
        base = f"rd:{cid}:{master}"
        out.append(DiaphragmGroup(
            condition=cid, perp_dirn=perp, master_node=master,
            slave_nodes=tuple(dict.fromkeys(slaves)),
            master_pg=f"{base}:master", slave_pg=f"{base}:slaves",
        ))
    return tuple(out)


def _diaphragm_pairs(scd: ScdModel, cids: Sequence[int]) -> set[tuple[int, int, int]]:
    """Rule C7: ``(perp, master, slave)`` for each link element of each interaction."""
    return {(perp, m, s) for _, perp, m, slaves in _links(scd, cids) for s in slaves}


# ---------------------------------------------------------------------------
# the solver chain of an Analyses command (rule C12)
# ---------------------------------------------------------------------------

_CONSTRAINTS = {
    "Plain Constraints": "Plain", "Transformation Method": "Transformation", "Auto": "Auto",
    "Penalty Method": "Penalty", "Lagrange Multipliers": "Lagrange",
}
_NUMBERERS = {
    "Plain Numberer": "Plain", "Reverse Cuthill-McKee Numberer": "RCM",
    "Alternative_Minimum_Degree Numberer": "AMD",
    "Parallel Reverse Cuthill-McKee Numberer": "ParallelRCM",
}
_TESTS = {
    "Norm Unbalance Test": ("NormUnbalance", "NormUnbalance"),
    "Norm Displacement Increment Test": ("NormDispIncr", "NormDispIncr"),
    "Energy Increment Test": ("EnergyIncr", "EnergyIncr"),
}
_ALGORITHMS = {
    "Linear": "Linear", "Newton": "Newton", "Modified Newton": "ModifiedNewton",
    "Krylov-Newton": "KrylovNewton",
}
_TANGENT = {"-secant": "secant", "-initial": "initial"}
_TRANSIENT = ("Newmark Method", "Central Difference", "Hilber-Hughes-Taylor Method",
              "Generalized Alpha Method")


def _chain(a: Mapping[str, Any], *, static: bool) -> tuple[dict[str, Any], list[str]]:
    """STKO's solver choices of one Analyses command, normalised, and the
    reasons any of them cannot be translated (only enforced on static stages)."""
    why: list[str] = []
    chain: dict[str, Any] = {}

    c = a.get("constraints")
    if c in _CONSTRAINTS:
        chain["constraints"] = {"type": _CONSTRAINTS[c]}
        if c == "Auto":
            chain["constraints"].update(
                verbose=bool(a.get("-verbose/auto")),
                auto_penalty_oom=int(a["oom/auto"]) if a.get("Automatic") else None,
                user_penalty=float(a["userPenalty/auto"]) if a.get("User-Defined") else None,
            )
        elif c == "Penalty Method":
            chain["constraints"].update(
                alpha_sp=float(a["alphaS/penaltyMethod"]), alpha_mp=float(a["alphaM/penaltyMethod"]))
        elif c == "Lagrange Multipliers":
            optional = bool(a.get("Optional lagrangeMultipliers"))
            chain["constraints"].update(
                alpha_sp=float(a["alphaS/LagrangeMultipliers"]) if optional else None,
                alpha_mp=float(a["alphaM/LagrangeMultipliers"]) if optional else None)
    else:
        chain["constraints"] = {"type": c}
        why.append(f"constraints={c!r}")

    n = a.get("numbererType")
    chain["numberer"] = _NUMBERERS[n] if n in _NUMBERERS else n
    if n not in _NUMBERERS:
        why.append(f"numbererType={n!r}")

    s = a.get("system")
    if s == "Mumps":
        chain["system"] = {
            "type": "Mumps",
            "icntl14": int(a["ICNTL14 Value"]) if a.get("-ICNTL14") else None,
            "matrix_type": a.get("matrixType") if a.get("-matrixType") else None,
        }
        if chain["system"]["matrix_type"] not in (None, "Unsymmetric"):
            why.append(f"system Mumps matrixType={a.get('matrixType')!r}")
    elif s == "UmfPack SOE":
        chain["system"] = {"type": "UmfPack"}
        if a.get("-lvalueFact"):
            why.append("system UmfPack -lvalueFact")
    else:
        chain["system"] = {"type": s}
        why.append(f"system={s!r}")

    adaptive = bool(a.get("Adaptive Time Step"))
    t = a.get("testCommand")
    if t in _TESTS:
        short, suffix = _TESTS[t]
        # STKO (AnalysesCommand.py, test.py writeTcl_test): the test is written with
        # 2 x iter when "Adaptive Time Step" is on (iter itself stays the driver's
        # desired iteration count); pFlag only when use_pFlag, nType only when use_nType.
        desired = int(a[f"iter/{suffix}"])
        use_n = bool(a.get(f"use_nType/{suffix}"))
        chain["test"] = {
            "type": short, "tol": float(a[f"tol/{suffix}"]),
            "max_iter": 2 * desired if adaptive else desired,
            "desired_iter": desired,
            "print_flag": int(a[f"pFlag/{suffix}"]) if a.get(f"use_pFlag/{suffix}") else 0,
            "n_type": int(a[f"nType/{suffix}"]) if use_n else None,
        }
        if use_n:
            why.append(f"test {short} nType (use_nType) is not translated")
    else:
        chain["test"] = {"type": t}
        why.append(f"testCommand={t!r}")

    al = a.get("algorithm")
    if al in _ALGORITHMS:
        short = _ALGORITHMS[al]
        d: dict[str, Any] = {"type": short}
        if short in ("Linear", "Newton", "ModifiedNewton"):
            if a.get(f"use_formTangent/{short}"):
                ft = a.get(f"formTangent/{short}")
                if ft in _TANGENT:
                    d["tangent"] = _TANGENT[ft]
                else:
                    why.append(f"algorithm {al} formTangent={ft!r}")
            fo = a.get("-factorOnce" if short == "Linear" else "-factorOnce/ModifiedNewton")
            if short != "Newton":
                d["factor_once"] = bool(fo)
        else:
            tag = "KrylovNewton"
            if a.get(f"-iterate/{tag}"):
                d["iterate"] = a.get(f"tangIter/{tag}")
            if a.get(f"-increment/{tag}"):
                d["increment"] = a.get(f"tangIncr/{tag}")
            if a.get(f"-maxDim/{tag}"):
                d["max_dim"] = int(a[f"maxDim/{tag}"])
        chain["algorithm"] = d
    else:
        chain["algorithm"] = {"type": al}
        why.append(f"algorithm={al!r}")

    chain["adaptive"] = adaptive
    chain["integrator_choice"] = (a.get("staticIntegrators") if static
                                  else a.get("transientIntegrators"))
    return chain, (why if static else [])


def _stage_spec(st: XObject, patterns: tuple[int, ...]) -> tuple[StageSpec, list[str]]:
    """The :class:`StageSpec` of an Analyses command and the reasons it cannot
    be translated (empty when it can; transient stages are data and never refused
    for their chain, only for an integrator STKO writes that this does not model)."""
    a = st.attributes
    static = bool(a.get("Static"))
    n_incr = int(a["numIncr"])
    why: list[str] = []
    if static:
        duration = float(a["duration"])
        chain, why = _chain(a, static=True)
        if a.get("staticIntegrators") != "Load Control":
            why.append(f"staticIntegrators={a.get('staticIntegrators')!r}")
        if a.get("Adaptive Time Step"):
            why.append("Adaptive Time Step on a static stage")
        if not a.get("loadConst"):
            why.append("loadConst=False on a static stage")
        if duration * n_incr == 0.0:
            why.append("duration * numIncr = 0 (STKO skips the analysis)")
        integrator: tuple[str, tuple[float, ...]] = (
            "LoadControl", (duration / n_incr if n_incr else 0.0,))
    else:
        duration = float(a["duration/transient"])
        chain, _ = _chain(a, static=False)
        tr = a.get("transientIntegrators")
        if tr == "Newmark Method":
            integrator = ("Newmark", (float(a["gamma"]), float(a["beta"])))
        elif tr == "Central Difference":
            integrator = ("CentralDifference", ())
        elif tr == "Hilber-Hughes-Taylor Method":
            integrator = ("HHT", (float(a["alpha/HHT"]),))
        elif tr == "Generalized Alpha Method":
            integrator = ("GeneralizedAlpha", (float(a["alphaM"]), float(a["alphaF"])))
        else:
            integrator = (str(tr), ())
            why.append(f"transientIntegrators={tr!r}")
        if duration * n_incr == 0.0:
            why.append("duration * numIncr = 0 (STKO skips the analysis)")
        if chain["adaptive"]:
            chain["adaptive_time_step"] = {
                k: a[k] for k in ("max factor", "min factor", "max factor incr",
                                  "min factor incr") if k in a
            }
    spec = StageSpec(
        stko_id=st.id, name=st.name, analysis="Static" if static else "Transient",
        patterns=patterns, n_incr=n_incr, duration=duration,
        load_const=bool(a.get("loadConst")), integrator=integrator, chain=chain,
    )
    return spec, why


# ---------------------------------------------------------------------------
# check_supported
# ---------------------------------------------------------------------------

def check_supported(scd: ScdModel) -> list[Unsupported]:
    """Everything the conditions slice of the registry (§4) refuses.

    Scans every ``Mass.*`` condition (STKO applies all masses whatever the
    patterns say), the conditions any load / constraint pattern references, the
    definitions a pattern references and every analysis step.
    """
    iss = _Issues()
    refs = _Refs(scd)
    geo = _Geo(scd)
    ref_conditions = refs.referenced_conditions()

    # ---- analysis steps ----
    seen_transient: XObject | None = None
    for sid, st in scd.analysis_steps.items():
        status = _status("analysis_step", st.type)
        if status is None:
            iss.add("analysis_step", st.type, sid, st.name, UNKNOWN)
        elif status == "tier_only":
            iss.add("analysis_step", st.type, sid, st.name, TIER_ONLY)
        elif status == "supported":
            _check_step(scd, refs, st, iss)
        elif st.type in _IGNORED_ONLY_AFTER_LAST_ANALYSIS and not refs.after_last_analysis(sid):
            iss.add("option", f"{st.type}:before analysis", sid, st.name,
                    "an analysis follows it, so it would change how that analysis runs; "
                    "it is ignored only after the last analysis")
        if st.type == _ANALYSES:
            if st.attributes.get("Static") and seen_transient is not None:
                iss.add("option", f"{_ANALYSES}:static after transient", sid, st.name,
                        f"a static stage after the transient stage {seen_transient.name!r} "
                        f"(step {seen_transient.id}): static stages are emitted before the "
                        "time-history driver runs, so the order would change")
            elif not st.attributes.get("Static") and seen_transient is None:
                seen_transient = st

    n_rayleigh = [s for s in scd.analysis_steps.values() if s.type == _RAYLEIGH]
    for st in n_rayleigh[1:]:
        iss.add("option", f"{_RAYLEIGH}:multiple", st.id, st.name,
                "more than one rayleigh step (a later one overrides at run time)")

    # ---- definitions a pattern references ----
    for did in sorted(refs.referenced_definitions()):
        d = scd.definitions.get(did)
        if d is None:
            iss.add("option", "pattern:tsTag", did, "<missing>",
                    f"time series {did} is not defined")
            continue
        status = _status("definition", d.type)
        if status is None:
            iss.add("definition", d.type, did, d.name, UNKNOWN)
        elif status == "tier_only":
            iss.add("definition", d.type, did, d.name, TIER_ONLY)
        elif d.type == "timeSeries.Path" and not d.attributes.get("constant"):
            iss.add("option", f"{d.type}:non_constant", did, d.name,
                    "a Path series with a list of times is not translated")

    # ---- conditions: every mass, plus what a pattern references ----
    scanned = {c.id for c in scd.conditions.values() if c.type.startswith("Mass.")}
    scanned |= set(ref_conditions)
    for cid in sorted(scanned):
        c = scd.conditions.get(cid)
        if c is None:
            iss.add("condition", "<missing>", cid, "<missing>",
                    f"a pattern references condition {cid}, which does not exist")
            continue
        status = _status("condition", c.type)
        if status is None:
            iss.add("condition", c.type, cid, c.name, UNKNOWN)
        elif status == "tier_only":
            iss.add("condition", c.type, cid, c.name, TIER_ONLY)
        else:
            _check_condition(scd, geo, refs, c, ref_conditions.get(cid, []), iss)

    # ---- cross-condition conflicts ----
    cp_conditions = [cid for st in refs.constraint_patterns
                     for key, cid in refs.pattern_conditions(st) if key == "sp"]
    _, conflicts = _fix_table(geo, scd, [c for c in cp_conditions if c in scd.conditions])
    for node, masks in conflicts.items():
        iss.add("option", "Constraints.sp.fix:conflicting masks", node, f"node {node}",
                f"node {node} is fixed by two conditions with different masks {sorted(masks)}")
    _check_diaphragm_groups(scd, refs, iss)
    return iss.rows()


def _check_diaphragm_groups(scd: ScdModel, refs: _Refs, iss: _Issues) -> None:
    """Rule C7 groups the session can carry verbatim (see :func:`declare_session`).

    Each group's master and slaves are owned by two carrier entities, one per
    role, and the bridge resolves the master as the carrier node nearest the
    master's own coordinates. So: a node may sit in one group only (as master or
    slave); a master may not be its own slave; and no slave may sit at its
    master's coordinates (the nearest-node pick would be ambiguous).
    """
    masters: dict[int, set[int]] = defaultdict(set)
    for _, m, s in _diaphragm_pairs(scd, refs.mp_conditions()):
        masters[s].add(m)
    for s, ms in masters.items():
        if len(ms) > 1:
            iss.add("option", f"{_RIGID_DIAPHRAGM}:slave in two diaphragms", s,
                    f"node {s}", f"node {s} is a slave of masters {sorted(ms)}")
    groups = diaphragm_groups(scd)
    owner: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for d in groups:
        for n in (d.master_node, *d.slave_nodes):
            owner[n].add((d.condition, d.master_node))
    for n, keys in sorted(owner.items()):
        if len(keys) > 1 and len(masters.get(n, ())) < 2:   # else reported just above
            iss.add("option", f"{_RIGID_DIAPHRAGM}:node in two diaphragms", n, f"node {n}",
                    f"node {n} belongs to the diaphragm groups (condition, master) "
                    f"{sorted(keys)}")
    for d in groups:
        if d.master_node in d.slave_nodes:
            iss.add("option", f"{_RIGID_DIAPHRAGM}:master is its own slave", d.condition,
                    scd.conditions[d.condition].name,
                    f"master {d.master_node} is listed among its own slaves")
            continue
        mxyz = np.asarray(scd.mesh.node_xyz(d.master_node), dtype=float)
        same = [s for s in d.slave_nodes
                if np.array_equal(np.asarray(scd.mesh.node_xyz(s), dtype=float), mxyz)]
        if same:
            iss.add("option", f"{_RIGID_DIAPHRAGM}:slave at the master", d.condition,
                    scd.conditions[d.condition].name,
                    f"slaves {same[:5]} sit at master {d.master_node}'s coordinates: the "
                    "bridge picks the master by position and could pick a slave")


def _check_step(scd: ScdModel, refs: _Refs, st: XObject, iss: _Issues) -> None:
    a = st.attributes
    t = st.type

    def opt(key: str, reason: str) -> None:
        iss.add("option", f"{t}:{key}", st.id, st.name, reason)

    if t == _PLAIN:
        if a.get("-fact"):
            opt("-fact", "a pattern factor is not translated")
        for key in ("eleLoad", "sp", "genericLoad"):
            if _ids(a.get(key)):
                opt(key, f"`{key}` conditions in a load pattern are not translated")
    elif t == _UE:
        for flag in ("-fact", "-vel0", "-disp", "-vel", "-int"):
            if a.get(flag):
                opt(flag, f"UniformExcitation {flag} is not translated")
        if a.get("direction") not in _UE_DIRECTION:
            opt("direction", f"direction {a.get('direction')!r} is not dx..rz")
    elif t == _CONSTRAINT:
        if refs.first_analysis is not None and st.id > refs.first_analysis:
            opt("after analysis",
                "a constraint pattern after the first analysis needs stage-bound fixities")
    elif t == _RAYLEIGH:
        if refs.first_analysis is not None and st.id > refs.first_analysis:
            opt("after analysis", "rayleigh after the first analysis changes damping mid-run")
    elif t == _ANALYSES:
        _, why = _stage_spec(st, ())
        for w in why:
            opt("stage", w)


def _check_condition(
    scd: ScdModel, geo: _Geo, refs: _Refs, c: Condition, patterns: list[int], iss: _Issues,
) -> None:
    a = c.xobject.attributes
    t = c.type

    def opt(key: str, reason: str) -> None:
        iss.add("option", f"{t}:{key}", c.id, c.name, reason)

    if a.get("Mode", "constant") == "function":
        opt("Mode=function", "a spatial-function value is not translated")
    if t in _MASS_TYPES + _LOAD_TYPES:
        _check_element_types(scd, c, opt)
    if t == "Constraints.sp.fix":
        if not (a.get("3D") and not a.get("2D")):
            opt("Dimension", "only 3-D fixities are translated")
        if not a.get("U-R (Displacement+Rotation)"):
            opt("ModelType", "only U-R (Displacement+Rotation) fixities are translated")
    elif t == "Constraints.mp.rigidDiaphragm":
        if int(a.get("perpDirn", 0)) not in _DIAPHRAGM_DOFS:
            opt("perpDirn", f"perpDirn {a.get('perpDirn')!r} is not 1, 2 or 3")
        for iid in c.interactions:
            if iid not in scd.interactions:
                opt("interaction", f"interaction {iid} does not exist")
    elif t == "Mass.AutoEdgeMass":
        converts = any(
            c.id in _ids(refs_st.attributes.get("massToLoad"))
            for refs_st in refs.load_patterns
        )
        if converts and a.get("Convert to load") and a.get("Type", "load") != "load":
            opt("Type", "AutoEdgeMass converted to eleLoad is not translated")
        for gid, sub in c.geometry.items():
            for i in sub.edges:
                pid = _edge_property(scd, gid, i)
                if not pid:
                    opt("physical property", f"edge {i} of geometry {gid} has no physical property")
                    continue
                n = len(_section_areas(scd, pid))
                if n == 0:
                    opt("section area", f"physical property {pid} has no cross-section area")
                elif n > 1:
                    opt("section closure",
                        f"physical property {pid} reaches {n} sections: STKO's mean over a "
                        "closure with two sections is unverified")

    # mass conditions in a pattern's massToLoad
    for pid in patterns:
        st = scd.analysis_steps[pid]
        if c.id in _ids(st.attributes.get("massToLoad")) and t != "Mass.AutoEdgeMass":
            opt("massToLoad", "only AutoEdgeMass can be converted to a load")
        if st.type == _PLAIN and c.id in _ids(st.attributes.get("load")) and t not in _LOAD_TYPES:
            opt("in load", "a load pattern lists a condition that is not a Loads.Force")
        if st.type == _CONSTRAINT:
            slot = "sp" if c.id in _ids(st.attributes.get("sp")) else "mp"
            if (slot, t) not in (("sp", "Constraints.sp.fix"),
                                 ("mp", "Constraints.mp.rigidDiaphragm")):
                opt("in constraint pattern", f"listed under {slot!r}")


def _check_element_types(scd: ScdModel, c: Condition, opt: Callable[[str, str], None]) -> None:
    kind, want = {
        "Mass.FaceMass": ("faces", 303), "Loads.Force.FaceForce": ("faces", 303),
        "Mass.AutoEdgeMass": ("edges", 102), "Loads.Force.EdgeForce": ("edges", 102),
    }.get(c.type, (None, None))
    if kind is None:
        return
    seen: set[int] = set()
    for gid, sub in c.geometry.items():
        dom = scd.mesh.domains.get((gid, kind), {})
        for i in sub.of(kind):
            for e in dom.get(i, ()):
                seen.add(scd.mesh.elements[int(e)].type)
    for tcode in sorted(seen - {want}):
        opt("element type", f"mesh element type {tcode} on the assigned {kind}")


# ---------------------------------------------------------------------------
# plan_conditions
# ---------------------------------------------------------------------------

def plan_conditions(
    scd: ScdModel, *, records: Mapping[int, Sequence[float]] | None = None,
) -> ConditionsPlan:
    """Reproduce STKO's masses, loads, fixities, diaphragms, time series,
    patterns, Rayleigh damping and stages (rules C1-C14) as data.

    ``records`` maps a ``timeSeries.Path`` definition id to its values: the
    ``.scd`` of San Ramon holds a one-value placeholder, the real record exists
    only in the exported decks.

    Raises :class:`UnsupportedSTKOTypes` when :func:`check_supported` finds
    anything. Pure: no session, no bridge.
    """
    issues = check_supported(scd)
    if issues:
        raise UnsupportedSTKOTypes(tuple(issues))
    geo = _Geo(scd)
    refs = _Refs(scd)
    ref_conditions = refs.referenced_conditions()

    masses = _plan_masses(geo, scd)
    cp_sp = [cid for st in refs.constraint_patterns
             for key, cid in refs.pattern_conditions(st) if key == "sp"]
    cp_mp = [cid for st in refs.constraint_patterns
             for key, cid in refs.pattern_conditions(st) if key == "mp"]
    table, _ = _fix_table(geo, scd, cp_sp)
    by_mask: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for n in sorted(table):
        by_mask[table[n]].append(n)
    fixes = {m: tuple(ns) for m, ns in sorted(by_mask.items())}

    time_series = _plan_time_series(scd, records)
    patterns, stages, never_active = _plan_patterns_and_stages(geo, scd, refs)

    ignored = _plan_ignored(scd, refs, ref_conditions, never_active)
    summaries = _plan_summaries(scd, refs)
    return ConditionsPlan(
        masses={n: _vec6(m) for n, m in sorted(masses.items())},
        fixes=fixes,
        diaphragm_pairs=frozenset(_diaphragm_pairs(scd, cp_mp)),
        time_series=time_series,
        patterns=patterns,
        constraint_conditions=tuple(dict.fromkeys([*cp_sp, *cp_mp])),
        rayleigh=_plan_rayleigh(scd),
        stages=stages,
        implex_dt_targets=_implex_targets(scd),
        summaries=summaries,
        ignored=ignored,
    )


def _vec6(m: np.ndarray) -> Vec6:
    return (float(m[0]), float(m[1]), float(m[2]), 0.0, 0.0, 0.0)


def _plan_masses(geo: _Geo, scd: ScdModel) -> dict[int, np.ndarray]:
    """Rule C1: every ``Mass.*`` condition of the document, summed per node."""
    acc = _Acc()
    for c in scd.conditions.values():
        rule = _MASS_RULES.get(c.type)
        if rule is not None:
            rule(geo, c, acc)
    return acc.sums()




# ---- time series (rule C10) ----

def _plan_time_series(
    scd: ScdModel, records: Mapping[int, Sequence[float]] | None,
) -> tuple[TimeSeriesSpec, ...]:
    records = dict(records or {})
    unknown = sorted(set(records) - set(scd.definitions))
    if unknown:
        raise ValueError(f"records= has ids {unknown} that are not definitions of the document; "
                         f"the document has {sorted(scd.definitions)}")
    out: list[TimeSeriesSpec] = []
    for did, d in scd.definitions.items():
        if _status("definition", d.type) != "supported":
            continue
        a = d.attributes
        factor = float(a["cFactor"]) if a.get("-factor") else 1.0
        if d.type == "timeSeries.Linear":
            if did in records:
                raise ValueError(f"records= for time series {did}, which is a Linear series")
            out.append(TimeSeriesSpec(stko_id=did, type="Linear", factor=factor))
            continue
        if not a.get("constant"):
            continue    # a list-of-times Path: refused by check_supported only if a pattern uses it
        own = tuple(float(v) for v in (a.get("list_of_values") or ()))
        if did in records:
            values, placeholder = tuple(float(v) for v in records[did]), False
        else:
            values, placeholder = own, len(own) <= 1
        out.append(TimeSeriesSpec(
            stko_id=did, type="Path", factor=factor, dt=float(a["dt"]), values=values,
            start_time=float(a["tStart"]) if a.get("-startTime") else None,
            placeholder=placeholder,
        ))
    return tuple(out)


# ---- patterns and stages (rules C8, C9, C12) ----

def _plan_patterns_and_stages(
    geo: _Geo, scd: ScdModel, refs: _Refs,
) -> tuple[tuple[PatternSpec, ...], tuple[StageSpec, ...], tuple[int, ...]]:
    patterns: list[PatternSpec] = []
    stages: list[StageSpec] = []
    pending: list[int] = []
    for sid, st in scd.analysis_steps.items():
        a = st.attributes
        if st.type == _PLAIN:
            patterns.append(_plain_pattern(geo, scd, st))
            pending.append(sid)
        elif st.type == _UE:
            patterns.append(PatternSpec(
                stko_id=sid, name=st.name, kind="UniformExcitation", series=int(a["tsTag"]),
                direction=_UE_DIRECTION[a["direction"]],
            ))
            pending.append(sid)
        elif st.type == _ANALYSES:
            spec, _ = _stage_spec(st, tuple(pending))
            stages.append(spec)
            pending = []
    return tuple(patterns), tuple(stages), tuple(pending)


def _plain_pattern(geo: _Geo, scd: ScdModel, st: XObject) -> PatternSpec:
    """Rule C8: the loads of a pattern, summed per node. The ``load`` conditions
    first, then ``massToLoad`` (only AutoEdgeMass with "Convert to load")."""
    a = st.attributes
    acc = _Acc()
    conds: list[int] = []
    for cid in _ids(a.get("load")):
        conds.append(cid)
        c = scd.conditions[cid]
        _LOAD_RULES[c.type](geo, c, acc)
    for cid in _ids(a.get("massToLoad")):
        conds.append(cid)
        c = scd.conditions[cid]
        if c.xobject.attributes.get("Convert to load"):
            _mass_to_load(geo, c, acc)
    loads = {
        n: _vec6(v) for n, v in sorted(acc.sums().items()) if np.any(v != 0.0)
    }
    return PatternSpec(
        stko_id=st.id, name=st.name, kind="Plain", series=int(a["tsTag"]),
        loads=loads, conditions=tuple(conds),
    )


# ---- Rayleigh (rule C11) ----

def _plan_rayleigh(scd: ScdModel) -> RayleighSpec | None:
    for st in scd.analysis_steps.values():
        if st.type != _RAYLEIGH:
            continue
        a = st.attributes
        manual = a.get("Input Type") == "Manual"
        alpha_m = float(a["alphaM/RayleighUser" if manual else "alphaM/Rayleigh"])
        beta = float(a["Betak/RayleighUser" if manual else "Betak/Rayleigh"])
        return RayleighSpec(
            alpha_m=alpha_m,
            beta_k=beta if a.get("Kcurr/Rayleigh") else 0.0,
            beta_k_init=beta if a.get("Kinit/Rayleigh") else 0.0,
            beta_k_comm=beta if a.get("Kcomm/Rayleigh") else 0.0,
        )
    return None


# ---- IMPL-EX dTime targets (rule C13) ----

def _implex_targets(scd: ScdModel) -> tuple[int, ...]:
    target: set[int] = set()
    for pid, x in scd.physical_properties.items():
        if x.type.split(".")[-1] not in _DTIME_TYPES:
            continue
        a = x.attributes
        integration = a.get("integration", a.get("Integration"))
        if integration == "IMPL-EX" or float(a.get("eta", 0.0) or 0.0) != 0.0:
            target.add(pid)
    if not target:
        return ()
    reaching = {pid for pid in scd.physical_properties
                if pid in target or property_references(scd, pid) & target}
    return tuple(sorted(
        e for e, ae in scd.analysis_elements().items()
        if ae.interaction is None and ae.physical_property in reaching
    ))


# ---- ignored and summaries ----

def _plan_ignored(
    scd: ScdModel, refs: _Refs, ref_conditions: Mapping[int, list[int]],
    never_active: tuple[int, ...],
) -> tuple[Ignored, ...]:
    out: list[Ignored] = []
    grouped: dict[str, list[XObject]] = defaultdict(list)
    for st in scd.analysis_steps.values():
        if _status("analysis_step", st.type) == "ignored":
            grouped[st.type].append(st)
    for t, steps in grouped.items():
        if t == "Misc_commands.customCommand":
            for st in steps:
                out.append(Ignored(
                    category="analysis_step", xobj_meta=t, ids=(st.id,), names=(st.name,),
                    reason=_IGNORED_WHY[t], payload=str(st.attributes.get("TCLscript", "")),
                ))
        else:
            out.append(Ignored(
                category="analysis_step", xobj_meta=t, ids=tuple(s.id for s in steps),
                names=tuple(s.name for s in steps), reason=_IGNORED_WHY[t],
            ))
    for sid in never_active:
        st = scd.analysis_steps[sid]
        out.append(Ignored(
            category="analysis_step", xobj_meta=st.type, ids=(sid,), names=(st.name,),
            reason="defined after the last analysis, so it is never active",
        ))
    referenced = set(ref_conditions)
    for c in scd.conditions.values():
        if c.id in referenced or c.type.startswith("Mass."):
            continue
        out.append(Ignored(
            category="condition", xobj_meta=c.type, ids=(c.id,), names=(c.name,),
            reason="not referenced by any load or constraint pattern: STKO does not write it",
        ))
    used_ts = refs.referenced_definitions()
    for did, d in scd.definitions.items():
        listed = _status("definition", d.type) == "supported" and (
            d.type == "timeSeries.Linear" or d.attributes.get("constant"))
        if did not in used_ts and not listed:
            out.append(Ignored(
                category="definition", xobj_meta=d.type, ids=(did,), names=(d.name,),
                reason="not referenced by any pattern",
            ))
    return tuple(out)


def _summary_for(c: Condition, patterns: tuple[int, ...]) -> ConditionSummary:
    a = c.xobject.attributes
    t = c.type
    extra: dict[str, Any] = {}
    value: tuple[float, ...] = ()
    per: Literal["area", "length", "node", "volume", "none"] = "none"
    if t == "Mass.FaceMass":
        value, per = tuple(float(v) for v in _vec3(a["mass"])), "area"
    elif t == "Mass.NodeMass":
        value, per = tuple(float(v) for v in _vec3(a["mass"])), "node"
    elif t == "Mass.AutoEdgeMass":
        value, per = (float(a["rho"]),), "volume"
        extra = {"convert_to_load": bool(a.get("Convert to load")), "type": a.get("Type"),
                 "g": (float(a["gx"]), float(a["gy"]), float(a["gz"]))}
    elif t in _LOAD_TYPES:
        value = tuple(float(v) for v in _vec3(a["F"]))
        per = {"Loads.Force.FaceForce": "area", "Loads.Force.EdgeForce": "length",
               "Loads.Force.NodeForce": "node"}[t]  # type: ignore[assignment]
        extra = {"global": bool(a.get("Global", True)), "mode": a.get("Mode")}
        if a.get("Mode") == "function":
            extra["function"] = tuple(str(a.get(k)) for k in ("Fx", "Fy", "Fz"))
    elif t == "Constraints.sp.fix":
        value = tuple(float(v) for v in _fix_mask(c))
        extra = {"mask": _fix_mask(c)}
    elif t == "Constraints.mp.rigidDiaphragm":
        extra = {"perp_dirn": int(a["perpDirn"]), "interactions": c.interactions}
    return ConditionSummary(
        stko_id=c.id, name=c.name, type=t, value=value, per=per, pgs={},
        patterns=patterns, extra=extra,
    )


def _plan_summaries(scd: ScdModel, refs: _Refs) -> tuple[ConditionSummary, ...]:
    using: dict[int, list[int]] = defaultdict(list)
    for st in (*refs.load_patterns, *refs.constraint_patterns):
        for _, cid in refs.pattern_conditions(st):
            using[cid].append(st.id)
    return tuple(
        _summary_for(c, tuple(dict.fromkeys(using.get(c.id, ()))))
        for c in scd.conditions.values()
        if _status("condition", c.type) == "supported"
    )


# ---------------------------------------------------------------------------
# declare_session
# ---------------------------------------------------------------------------

def declare_session(g: Any, scd: ScdModel, plan: ConditionsPlan, mesh: MeshMap) -> None:
    """Rigid diaphragms (rule C7) on ``g.constraints``, STKO's links verbatim.

    apeGmsh has no node-level master/slave API for a rigid diaphragm: the only
    route is ``g.constraints.rigid_diaphragm(master_pg, slave_pg, ...)``, which
    the resolver (``resolve_rigid_diaphragm``) turns into one record by geometry
    -- it keeps the nodes of ``master_pg | slave_pg`` within ``plane_tolerance``
    of the plane through ``master_point`` and takes as master the node nearest
    ``master_point``. Each :class:`~.translate_types.DiaphragmGroup` is carried so
    that this geometry reproduces STKO's pairs exactly:

    * its master and its slaves are owned by two carrier entities (the mesh
      module), so ``master_pg | slave_pg`` is exactly ``{master, *slaves}``;
    * ``master_point`` is the master's own coordinates, so the nearest node of
      that set is the master (a slave at the same coordinates is refused by
      :func:`check_supported`);
    * ``plane_tolerance`` is larger than the farthest slave's distance from the
      master's plane, so no slave is dropped (OpenSees' ``rigidDiaphragm`` keeps
      an off-plane slave too, with a warning: ``RigidDiaphragm.cpp``).

    The resolved records are checked against ``plan.diaphragm_pairs`` after
    ``get_fem_data`` by :func:`verify_diaphragms` (``build_conditions`` calls it).
    Raises ``ValueError`` before touching ``g`` when the mesh's groups and the
    plan's pairs disagree.
    """
    from_mesh = {
        (d.perp_dirn, d.master_node, s) for d in mesh.diaphragms for s in d.slave_nodes
    }
    if from_mesh != set(plan.diaphragm_pairs):
        only_mesh = sorted(from_mesh - set(plan.diaphragm_pairs))[:3]
        only_plan = sorted(set(plan.diaphragm_pairs) - from_mesh)[:3]
        raise ValueError(
            "the mesh's diaphragm groups and the plan's diaphragm pairs disagree "
            f"(mesh only: {only_mesh}, plan only: {only_plan})"
        )
    if not mesh.diaphragms:
        return
    xyz = np.asarray(scd.mesh.coordinates, dtype=float)
    floor = 1.0e-6 * max(float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))), 1.0)
    for d in mesh.diaphragms:
        axis = d.perp_dirn - 1
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        master_xyz = np.asarray(scd.mesh.node_xyz(d.master_node), dtype=float)
        off = max((abs(float(scd.mesh.node_xyz(s)[axis]) - float(master_xyz[axis]))
                   for s in d.slave_nodes), default=0.0)
        g.constraints.rigid_diaphragm(
            d.master_pg, d.slave_pg,
            master_point=tuple(float(v) for v in master_xyz),
            plane_normal=tuple(normal),
            constrained_dofs=list(_DIAPHRAGM_DOFS[d.perp_dirn]),
            plane_tolerance=2.0 * off + floor,
            name=f"rd:{d.condition}:{d.master_node}",
        )


def resolved_diaphragm_pairs(fem: Any) -> list[tuple[int, int, int]]:
    """``(perp, master, slave)`` of every resolved rigid-diaphragm record of a
    FEM (``fem.nodes.constraints.rigid_diaphragms()``), repeats kept."""
    return [
        (int(perp), int(master), int(s))
        for perp, master, slaves in fem.nodes.constraints.rigid_diaphragms()
        for s in slaves
    ]


def verify_diaphragms(fem: Any, plan: ConditionsPlan) -> None:
    """Assert the FEM's resolved rigid-diaphragm records are exactly
    ``plan.diaphragm_pairs``: same set, no pair twice, nothing else.

    Raises ``ValueError`` naming the first differences. Run it on
    ``g.mesh.queries.get_fem_data(dim=None)`` of a :func:`declare_session`
    session; :func:`build_conditions` runs it on the bridge's FEM.
    """
    got = resolved_diaphragm_pairs(fem)
    want = set(plan.diaphragm_pairs)
    seen: set[tuple[int, int, int]] = set()
    twice = sorted({p for p in got if p in seen or seen.add(p)})  # type: ignore[func-returns-value]
    if set(got) != want or twice:
        raise ValueError(
            "the FEM's resolved rigid diaphragms are not STKO's link pairs "
            f"({len(set(got))} resolved, {len(want)} in the plan; resolved only: "
            f"{sorted(set(got) - want)[:5]}, plan only: {sorted(want - set(got))[:5]}, "
            f"resolved twice: {twice[:5]})"
        )


# ---------------------------------------------------------------------------
# build_conditions
# ---------------------------------------------------------------------------

#: STKO_DT_UTIL_OnBeforeAnalyze's parameters, in its order (time_increment_utils.py).
_DTIME_PARAMETERS = ("dTimeCommit", "dTimeInitial", "dTime")


def build_conditions(
    ops: Any, plan: ConditionsPlan, mesh: MeshMap, *,
    stages: bool = True, chain: Literal["serial", "stko"] = "serial",
    implex_dtime: bool = True,
) -> dict[int, Any]:
    """Declare the plan on an ``apeSees`` bridge.

    Fixities, nodal masses and Rayleigh damping are declared globally; the time
    series the patterns use are created; then, with ``stages=True``, one
    ``ops.stage`` per *static* stage holds its patterns, its analysis chain and
    its run. With ``stages=False`` every Plain pattern is declared globally and
    no analysis is (deck parity of the loads only).

    The transient stage, UniformExcitation patterns and STKO's adaptive driver
    stay in the plan as data (rules §11): the time-history driver owns them.

    ``chain="serial"`` (default) maps STKO's MPI choices to serial equivalents
    (``system Mumps`` -> Pardiso, ``numberer ParallelRCM`` -> RCM); ``"stko"``
    emits them as STKO wrote them, for partitioned decks.

    Returns the created time series by STKO definition id (a time-history
    driver builds its UniformExcitation patterns on them).

    On a real bridge (``ops.fem`` a :class:`~apeGmsh.mesh.FEMData.FEMData`),
    first :func:`verify_diaphragms`: the FEM's resolved rigid diaphragms must be
    exactly ``plan.diaphragm_pairs``, or nothing is declared.

    IMPL-EX ``dTime`` (rule C13), ``implex_dtime=True`` (default): at the start
    of every static stage, ``s.update_parameter`` sets ``dTimeCommit``,
    ``dTimeInitial`` and ``dTime`` of every ``plan.implex_dt_targets`` element
    to the stage's increment ``duration / numIncr`` -- what STKO's
    ``STKO_DT_UTIL_OnBeforeAnalyze`` does at the stage's first increment (its
    value stays the same for the stage's other increments: no adaptive static
    stage is translated). Once set, ASDConcrete stops reading OpenSees' own
    increment (``dtime_is_user_defined``, ``ASDConcrete3DMaterial.cpp``), so
    **the time-history driver must set ``dTime`` before every transient step**
    (and ``dTimeCommit`` / ``dTimeInitial`` at its first), as STKO's transient
    template does; without it the materials keep the last static increment.
    ``implex_dtime=False`` writes no reset (the materials then follow
    OpenSees' increment; the first increment of a static stage whose
    increment differs from the previous stage's then extrapolates with
    ``dt_new / dt_old`` instead of STKO's 1).
    """
    fem = _bridge_fem(ops)
    if fem is not None:
        verify_diaphragms(fem, plan)
    known = set(int(n) for n in mesh.node_ids)
    wanted = set(plan.masses) | {n for ns in plan.fixes.values() for n in ns}
    for p in plan.patterns:
        wanted.update(p.loads)
    missing = sorted(wanted - known)
    if missing:
        raise ValueError(
            f"{len(missing)} node(s) of the plan are not in the session's mesh, "
            f"first {missing[:5]}"
        )

    for mask, nodes in plan.fixes.items():
        ops.fix(nodes=list(nodes), dofs=tuple(mask))
    for node, m in plan.masses.items():
        ops.mass(nodes=[node], values=m)
    if plan.rayleigh is not None:
        r = plan.rayleigh
        ops.damping.rayleigh(
            alpha_m=r.alpha_m, beta_k=r.beta_k,
            beta_k_init=r.beta_k_init, beta_k_comm=r.beta_k_comm,
        )

    used = {p.series for p in plan.patterns}
    series: dict[int, Any] = {}
    for ts in plan.time_series:
        if ts.stko_id not in used:
            continue
        if ts.type == "Linear":
            series[ts.stko_id] = ops.timeSeries.Linear(factor=ts.factor)
        else:
            kw: dict[str, Any] = {"values": ts.values, "dt": ts.dt, "factor": ts.factor}
            if ts.start_time is not None:
                kw["start_time"] = ts.start_time
            series[ts.stko_id] = ops.timeSeries.Path(**kw)

    by_id = {p.stko_id: p for p in plan.patterns}
    if not stages:
        for p in plan.patterns:
            if p.kind == "Plain":
                _declare_pattern(ops.pattern.Plain(series=series[p.series]), p)
        return series

    names: dict[str, int] = {}
    targets = tuple(plan.implex_dt_targets) if implex_dtime else ()
    for st in plan.stages:
        if st.analysis != "Static":
            continue
        names[st.name] = names.get(st.name, 0) + 1
        name = st.name if names[st.name] == 1 else f"{st.name}#{st.stko_id}"
        with ops.stage(name) as s:
            if targets:
                # rule C13: STKO_DT_UTIL_OnBeforeAnalyze at increment 1 of the stage
                dt = st.integrator[1][0]
                for pname in _DTIME_PARAMETERS:
                    s.update_parameter(pname, dt, elements=targets)
            for pid in st.patterns:
                p = by_id[pid]
                if p.kind == "Plain":
                    _declare_pattern(s.pattern(series=series[p.series]), p)
            s.analysis(**_static_chain(ops, st, chain))
            s.run(n_increments=st.n_incr)
    return series


def _bridge_fem(ops: Any) -> Any:
    """``ops.fem`` when ``ops`` is a bridge over a real FEM, else None (a stand-in)."""
    from ...mesh.FEMData import FEMData

    if not isinstance(getattr(type(ops), "fem", None), property):
        return None
    fem = ops.fem
    return fem if isinstance(fem, FEMData) else None


def _declare_pattern(pattern: Any, spec: PatternSpec) -> None:
    with pattern as p:
        for node, forces in spec.loads.items():
            p.load(node=node, forces=forces)


def _static_chain(ops: Any, st: StageSpec, mode: str) -> dict[str, Any]:
    """The seven analysis primitives of a static stage, from its ``chain`` data."""
    c = st.chain
    cs = c["constraints"]
    if cs["type"] == "Plain":
        constraints = ops.constraints.Plain()
    elif cs["type"] == "Transformation":
        constraints = ops.constraints.Transformation()
    elif cs["type"] == "Penalty":
        constraints = ops.constraints.Penalty(alpha_sp=cs["alpha_sp"], alpha_mp=cs["alpha_mp"])
    elif cs["type"] == "Lagrange":
        constraints = (ops.constraints.Lagrange() if cs["alpha_sp"] is None else
                       ops.constraints.Lagrange(alpha_sp=cs["alpha_sp"], alpha_mp=cs["alpha_mp"]))
    elif cs["user_penalty"] is not None:
        constraints = ops.constraints.Auto(
            verbose=cs["verbose"], auto_penalty=False, user_penalty=cs["user_penalty"])
    elif cs["auto_penalty_oom"] is not None:
        constraints = ops.constraints.Auto(
            verbose=cs["verbose"], auto_penalty_oom=float(cs["auto_penalty_oom"]))
    else:
        constraints = ops.constraints.Auto(verbose=cs["verbose"])

    numb = c["numberer"]
    if numb == "ParallelRCM" and mode == "serial":
        numb = "RCM"
    numberer = getattr(ops.numberer, numb)()

    sy = c["system"]
    if sy["type"] == "Mumps":
        if mode == "serial":
            system = ops.system.Pardiso()
        else:
            system = ops.system.Mumps(icntl14=sy["icntl14"])
    else:
        system = ops.system.UmfPack()

    t = c["test"]
    test = getattr(ops.test, t["type"])(
        tol=t["tol"], max_iter=t["max_iter"], print_flag=t["print_flag"])

    al = c["algorithm"]
    kind = al["type"]
    if kind == "Linear":
        algorithm = ops.algorithm.Linear(
            tangent=al.get("tangent", "tangent"), factor_once=al["factor_once"])
    elif kind == "Newton":
        algorithm = ops.algorithm.Newton(tangent=al.get("tangent", "tangent"))
    elif kind == "ModifiedNewton":
        algorithm = ops.algorithm.ModifiedNewton(
            tangent=al.get("tangent", "tangent"), factor_once=al["factor_once"])
    else:
        algorithm = ops.algorithm.KrylovNewton(
            iterate=al.get("iterate"), increment=al.get("increment"),
            max_dim=al.get("max_dim"))

    return {
        "test": test, "algorithm": algorithm, "constraints": constraints,
        "numberer": numberer, "system": system,
        "integrator": ops.integrator.LoadControl(dlam=st.integrator[1][0]),
        "analysis": ops.analysis.Static(),
    }

