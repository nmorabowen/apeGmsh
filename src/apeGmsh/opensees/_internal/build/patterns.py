"""patterns: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Iterable

import numpy as np

from ..types import Element, Primitive
from ...pattern.pattern import (
    Plain,
    _BridgeValueError,
    _EleLoadRecord,
    _LoadRecord,
    _SPRecord,
    _beam_uniform_args,
    _membrane_pressure_args,
    _self_weight_args,
)
from ..tag_resolution import MISSING_FEM_ELEMENT_ID, resolve_tag
from .errors import BridgeError
from .pg_expand import ElementPlanRows, expand_pg_to_elements, expand_pg_to_nodes
from .recorders import _expand_label_to_elements
from .ndf import _load_dof_layout, fit_dof_vector
from .transforms import (
    _element_transf,
    compute_vecxz_for_element,
    element_jnt_offset,
    is_orientation_transform,
)


if TYPE_CHECKING:
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData
    from .ownership import FemToOpsTagMap, SortedIntToInt


# ---------------------------------------------------------------------------
# Pattern fan-out — Plain pattern PG records resolved into per-node calls
# ---------------------------------------------------------------------------

def emit_pattern_spec(
    spec: Primitive,
    emitter: "Emitter",
    tag: int,
    fem: "FEMData",
    ndf: int,
    ndm: int = 3,
    *,
    effective_ndf: "dict[int, int] | None" = None,
    element_plan: "Sequence[tuple[Element, ElementPlanRows]] | None" = None,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
) -> None:
    """Drive a pattern's emit, expanding any ``pg=`` records to per-node calls.

    For a :class:`Plain` pattern: emit ``pattern_open`` ourselves so we
    can intervene between it and ``pattern_close``, replacing the
    spec's :meth:`_emit` body. Records with ``target_kind == "node"``
    pass through unchanged; ``target_kind == "pg"`` records are
    fanned out into per-node ``emitter.load`` / ``emitter.sp`` calls.
    Any :meth:`Plain.from_model` cases are expanded last (ADR 0051):
    the geometry-resolved nodal loads / prescribed displacements tagged
    with the case are pulled from ``fem`` and emitted as ``load`` / ``sp``
    lines inside this pattern (``ndf`` maps the DOF-agnostic records),
    followed by ``eleLoad`` commands from ``spec.ele_loads`` and
    element-form ``from_model`` records.

    For non-Plain patterns (``UniformExcitation``, etc.) we delegate to
    the spec's own ``_emit`` since they have no PG-bearing records.
    """
    if not isinstance(spec, Plain):
        spec._emit(emitter, tag)
        return

    eff = effective_ndf or {}

    def ndf_of(n: int) -> int:
        return int(eff.get(int(n), ndf))

    ts_tag = resolve_tag(emitter, spec.series)
    emitter.pattern_open("Plain", tag, ts_tag)
    for rec in spec.loads:
        _emit_load_record(rec, emitter, fem, ndf_of)
    for sp_rec in spec.sps:
        _emit_sp_record(sp_rec, emitter, fem)
    for case in spec.from_model_cases:
        _emit_from_model_case(case, emitter, fem, ndf_of, ndm)
    for mt_rec in spec.moment_tensors:
        _emit_moment_tensor_record(mt_rec, emitter, fem, ndf_of, ndm)
    end_zone_loads: list[tuple[int, tuple[float, ...]]] = []
    for ele_args in resolve_ele_load_commands(
        spec,
        fem,
        ndm,
        element_plan=element_plan,
        fem_eid_to_ops_tag=fem_eid_to_ops_tag,
        nodal_loads=end_zone_loads,
    ):
        emitter.eleLoad(*ele_args)
    for node, comps in end_zone_loads:
        emitter.load(node, *fit_dof_vector(
            comps, ndf_of(node), kind="end-zone line load", node=node))
    emitter.pattern_close()


_LOAD_COMPONENT_LABELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def broker_load_components(
    rec: "Any", ndf: int, ndm: int = 3,
) -> tuple[float, ...]:
    """Map a DOF-agnostic ``NodalLoadRecord`` onto a model's ``(ndm, ndf)``.

    apeGmsh records store pure 3-D spatial force/moment vectors; the
    bridge is the only layer that knows ``ndm``/``ndf`` (per ADR 0051 /
    the records' DOF-agnostic contract). The DOF layout depends on
    **both** — ``ndf == 3`` is a planar frame ``(Fx, Fy, Mz)`` when
    ``ndm == 2`` but a 3D solid ``(Fx, Fy, Fz)`` when ``ndm == 3`` (see
    :func:`_load_dof_layout`).

    Fails loud (``BridgeError``) when a non-zero force/moment component
    maps to no DOF in this model — a silently-dropped load is a
    correctness bug, not a convenience.
    """
    spatial = (rec.force_xyz or (0.0, 0.0, 0.0)) + (
        rec.moment_xyz or (0.0, 0.0, 0.0))
    layout = _load_dof_layout(ndf, ndm)
    carried = set(layout)
    dropped = [
        f"{_LOAD_COMPONENT_LABELS[i]}={spatial[i]:g}"
        for i in range(6)
        if i not in carried and spatial[i] != 0.0
    ]
    if dropped:
        node = getattr(rec, "node_id", None)
        where = f" at node {node}" if node is not None else ""
        raise BridgeError(
            f"Load{where} has component(s) {', '.join(dropped)} that the "
            f"model (ndm={ndm}, ndf={ndf}) cannot carry — it would be "
            f"silently dropped. Check the load direction against the "
            f"model's DOFs (e.g. a 2D model has no out-of-plane force; a "
            f"solid element has no nodal moment), or use a model ndf that "
            f"includes the missing DOF."
        )
    return tuple(spatial[i] for i in layout)


_MASS_COMPONENT_LABELS = ("mx", "my", "mz", "Ixx", "Iyy", "Izz")


def broker_mass_components(
    mass: "Iterable[float]", ndf: int, ndm: int, *, node: int,
) -> tuple[float, ...]:
    """Map a broker ``(mx, my, mz, Ixx, Iyy, Izz)`` mass onto a node's DOFs.

    The mass counterpart of :func:`broker_load_components`. Broker masses
    (``fem.nodes.masses``) are spatially ordered, not DOF-ordered, so a
    positional trim is wrong in 2-D: an ``ndm=2, ndf=3`` frame node is
    ``(ux, uy, rz)`` and must receive ``(mx, my, Izz)``, not ``(mx, my, mz)``.

    * ``ndm != 2`` delegates to :func:`fit_dof_vector` (positional), which
      is already the right layout for 3-D (``ndf`` 3 or 6); 3-D decks are
      unchanged.
    * ``ndm == 2`` places components by :func:`_load_dof_layout`
      (``ndf=2`` → ``(mx, my)``, ``ndf=3`` → ``(mx, my, Izz)``). ``mz`` is
      dropped when it rides with in-plane mass: the resolver fills it by
      default (``dofs=None`` means ``mx = my = mz = m``), and a 2-D model
      has no z-translation. A z-ONLY mass (``mz != 0`` with ``mx = my =
      0``, e.g. ``dofs=[3]``) is explicit out-of-plane intent and fails
      loud, like ``Fz`` in :func:`broker_load_components`. Any other
      non-zero component with no DOF to land on also fails loud: ``Ixx`` /
      ``Iyy``, or ``Izz`` on an ``ndf=2`` node, come from an explicit
      ``rotational=`` or ``derive_rotational=True``; ``my`` on an ``ndf=1``
      node is an in-plane translation the node lacks.

    Known gap: the record does not carry the resolver's ``dofs`` mask, so
    a mixed explicit mask such as ``dofs=[1, 3]`` (``mx = mz = m``, ``my =
    0``) is indistinguishable from a default fill and its ``mz`` is
    dropped silently. Likewise, the guard sees the per-node *accumulated*
    record, so a z-only def (``dofs=[3]``) whose nodes all also carry
    in-plane mass from another def is absorbed into the default-fill case
    and its ``mz`` is dropped without error. Closing both needs the mask
    on ``MassRecord``.
    """
    if int(ndm) != 2:
        return fit_dof_vector(mass, ndf, kind="mass", node=node)
    vals = [float(v) for v in mass]
    vals += [0.0] * (6 - len(vals))
    ndf_i = int(ndf)
    if vals[2] != 0.0 and vals[0] == 0.0 and vals[1] == 0.0:
        raise BridgeError(
            f"mass on node {node} is z-translation only (mz={vals[2]:g}, "
            f"mx = my = 0), but a 2-D model (ndm=2, ndf={ndf_i}) has no "
            f"z-translation — the mass would be silently dropped. Use "
            f"in-plane DOFs (dofs=[1, 2]) for a 2-D model."
        )
    layout = _load_dof_layout(ndf_i, 2)
    carried = set(layout) | {2}
    dropped = [
        i for i in range(6) if i not in carried and vals[i] != 0.0
    ]
    if dropped:
        causes = []
        if any(i < 3 for i in dropped):
            causes.append(
                f"the node has only {ndf_i} DOF(s), so an in-plane "
                f"translational mass it lacks cannot land (check the node's "
                f"ndf, or restrict the mass with dofs=)")
        if any(i >= 3 for i in dropped):
            causes.append(
                "rotational inertia about an axis the node cannot rotate "
                "about (a 2-D node rotates only about z, and only on an "
                "ndf=3 frame node): use rotational=(0, 0, Izz), or drop "
                "derive_rotational on a node without that DOF")
        labels = ", ".join(
            f"{_MASS_COMPONENT_LABELS[i]}={vals[i]:g}" for i in dropped)
        raise BridgeError(
            f"mass on node {node} has component(s) {labels} that the 2-D "
            f"model (ndm=2, ndf={ndf_i}) cannot carry — it would be "
            f"silently dropped: {'; '.join(causes)}."
        )
    out = tuple(vals[i] for i in layout)
    return out + (0.0,) * (int(ndf) - len(out))


def _emit_from_model_case(
    case: str,
    emitter: "Emitter",
    fem: "FEMData",
    ndf_of: "Callable[[int], int]",
    ndm: int = 3,
) -> None:
    """Expand a ``Plain.from_model(case)`` import into load / sp lines.

    Nodal loads tagged with ``case`` become ``load`` lines; non-
    homogeneous (prescribed) SPs tagged with ``case`` become ``sp``
    lines. Homogeneous fixes are model-level and never imported here.
    Element-form loads are out of scope (all-nodal, ADR 0051).

    The DOF-agnostic spatial load is mapped onto **each node's** effective
    ndf (``ndf_of`` — inferred per node, envelope fallback), not the model
    envelope: a force on a 3-DOF solid node emits 3 components even in a
    6-DOF-envelope model. :func:`broker_load_components` fails loud if a
    non-zero component cannot land on that node's DOFs.
    """
    nodes = getattr(fem, "nodes", None)
    if nodes is None:
        return
    load_set = getattr(nodes, "loads", None)
    if load_set is not None:
        for rec in load_set.by_pattern(case):
            emitter.load(
                int(rec.node_id),
                *broker_load_components(rec, ndf_of(int(rec.node_id)), ndm),
            )
    sp_set = getattr(nodes, "sp", None)
    if sp_set is not None:
        for rec in sp_set.prescribed():
            if rec.pattern == case:
                emitter.sp(int(rec.node_id), rec.dof, rec.value)


# gmsh element-type code → (inverse-map host kind, corner-node count).
# Continuum hosts only; a straight-sided higher-order host maps with its
# corner kind (gmsh orders corner nodes first). Sibling of
# ``ReinforcementsComposite._GMSH_HOST_KIND`` — kept local so build.py
# does not import a core composite.
_MT_HOST_KIND: dict[int, tuple[str, int]] = {
    2: ("tri3", 3), 3: ("quad4", 4), 4: ("tet4", 4), 5: ("hex8", 8),
    9: ("tri3", 3), 10: ("quad4", 4), 11: ("tet4", 4),
    16: ("quad4", 4), 17: ("hex8", 8),
}


def _region_element_ids(fem: "FEMData", region: str) -> set[int]:
    """Resolve a ``region`` name to its element-id set (Tier 1 → Tier 2).

    Mirrors :class:`FEMDataSource._element_ids_for_target` — element-side
    labels first, then physical groups. Used to restrict the moment-tensor
    host search to a named region (e.g. the soil interior, excluding the
    absorbing skin). Fails loud if the name resolves to no element group.
    """
    from apeGmsh._kernel._label_prefix import add_prefix

    prefixed = add_prefix(region)
    for entry in fem.elements.labels._groups.values():
        if entry.get("name", "") in (region, prefixed):
            eids = entry.get("element_ids")
            if eids is not None:
                return {int(x) for x in eids}
    for entry in fem.elements.physical._groups.values():
        if entry.get("name", "") == region:
            eids = entry.get("element_ids")
            if eids is not None:
                return {int(x) for x in eids}
    raise BridgeError(
        f"moment_tensor: region {region!r} resolves to no element-side label "
        f"or physical group in the model — name the continuum region the "
        f"source must sit inside (e.g. the soil PG, not the absorbing skin)."
    )


def _collect_continuum_hosts(
    fem: "FEMData",
    region: str | None = None,
) -> tuple[list[list[int]], list[np.ndarray], list[str], int,
           np.ndarray, np.ndarray]:
    """Per continuum element: corner tags / coords / inverse-map kind.

    Walks the broker's element groups (keyed by gmsh type code), keeps the
    continuum kinds in :data:`_MT_HOST_KIND`, and restricts to the highest
    continuum dimension present (a 3D model's surface tris are not source
    hosts). When ``region`` is given, only elements in that PG/label are
    candidate hosts — a source outside it (e.g. in the absorbing skin)
    then fails loud via the out-of-continuum guard. Returns
    ``(host_node_ids, host_node_coords, host_kinds, model_ndm,
    all_node_ids, all_node_coords)`` — the last two are the global node
    cloud the ``"dipole"`` method searches.
    """
    from apeGmsh._kernel.geometry._inverse_map import HOST_KINDS

    ids = np.asarray(fem.nodes.ids)
    coords = np.asarray(fem.nodes.coords, dtype=float)
    coord_of = {int(t): coords[i] for i, t in enumerate(ids)}

    region_eids = _region_element_ids(fem, region) if region else None

    groups: list[tuple[str, int, np.ndarray]] = []
    for code, grp in fem.elements._groups.items():
        info = _MT_HOST_KIND.get(int(code))
        if info is None:
            continue
        kind, n_corner = info
        conn = np.asarray(grp.connectivity, dtype=int)
        if region_eids is not None:
            grp_ids = np.asarray(grp.ids, dtype=int)
            mask = np.array([int(e) in region_eids for e in grp_ids], dtype=bool)
            conn = conn[mask]
        if conn.shape[0]:
            groups.append((kind, n_corner, conn))

    if not groups:
        where = f" in region {region!r}" if region else ""
        raise BridgeError(
            f"moment_tensor: the model carries no continuum host elements "
            f"(tri/quad/tet/hex){where} for the source to embed into."
        )

    model_ndm = max(HOST_KINDS[kind][1] for kind, _, _ in groups)
    host_ids: list[list[int]] = []
    host_coords: list[np.ndarray] = []
    host_kinds: list[str] = []
    for kind, n_corner, conn in groups:
        if HOST_KINDS[kind][1] != model_ndm:
            continue
        for row in conn:
            corners = [int(t) for t in row[:n_corner]]
            host_ids.append(corners)
            host_coords.append(np.vstack([coord_of[t] for t in corners]))
            host_kinds.append(kind)
    return host_ids, host_coords, host_kinds, model_ndm, ids, coords


def resolve_moment_tensor_pairs(
    rec: "Any", fem: "FEMData",
) -> list[tuple[int, np.ndarray]]:
    """Resolve one ``Plain.moment_tensor`` source into ``(node, force_xyz)``.

    Builds the moment tensor in the mesh frame, locates the host (or grid
    node) in ``fem``, and turns the representation-theorem body force into
    per-node force vectors (length-3, padded). Shared by the flat emit
    (:func:`_emit_moment_tensor_record`) and the partitioned emit
    (``apeSees._owned_moment_tensor_lines``) so both paths agree.

    Raises ``NotImplementedError`` for a non-zero rupture onset (MT-4),
    and a clear MT-flavoured ``ValueError`` (not the reinforcement-worded
    inverse-map message) when the source lies outside the continuum.
    """
    from apeGmsh._kernel.geometry._moment_tensor import moment_tensor
    from apeGmsh._kernel.resolvers._moment_tensor import (
        resolve_moment_tensor_source,
    )

    if rec.t0 != 0.0:
        raise NotImplementedError(
            "moment_tensor: a non-zero rupture onset t0 (a per-source "
            "delay) is MT-4 (finite-fault) work — the v1 single source "
            "rides the pattern's shared S(t). Got t0="
            f"{rec.t0!r}."
        )

    if rec.m_ij is not None:
        M = moment_tensor(
            m_ij=np.asarray(rec.m_ij, dtype=float), M0=rec.M0, frame=rec.frame,
        )
    else:
        M = moment_tensor(
            strike=rec.strike, dip=rec.dip, rake=rec.rake,
            M0=rec.M0, frame=rec.frame,
        )

    region = getattr(rec, "region", None)
    host_ids, host_coords, host_kinds, _model_ndm, all_ids, all_coords = (
        _collect_continuum_hosts(fem, region)
    )
    try:
        return resolve_moment_tensor_source(
            position=np.asarray(rec.position, dtype=float),
            M=M,
            method=rec.method,
            host_node_ids=host_ids,
            host_node_coords=host_coords,
            host_kinds=host_kinds,
            node_ids=all_ids,
            node_coords=all_coords,
            label=f"moment_tensor @ {tuple(rec.position)}",
        )
    except ValueError as exc:
        if rec.method == "consistent" and "outside every host" in str(exc):
            where = (
                f"the region {region!r}" if region else "the continuum mesh"
            )
            raise BridgeError(
                f"moment_tensor: the source at {tuple(rec.position)} lies "
                f"outside {where} — every seismic source must sit inside an "
                f"element of the intact continuum (not on the free surface, "
                f"not in the absorbing skin). "
                + ("" if region else "Pass region= to restrict the host "
                   "search to the soil PG.")
            ) from exc
        raise


def _emit_moment_tensor_record(
    rec: "Any",
    emitter: "Emitter",
    fem: "FEMData",
    ndf_of: "Callable[[int], int]",
    ndm: int = 3,
) -> None:
    """Resolve a ``Plain.moment_tensor`` source into nodal ``load`` lines.

    Locates the host (or grid node) in ``fem``, turns the
    representation-theorem body force into per-node forces, and emits each
    as a ``load`` line mapped onto the node's effective ndf. The pattern's
    time series supplies ``S(t)``.
    """
    from apeGmsh._kernel.records._loads import NodalLoadRecord

    for node, force in resolve_moment_tensor_pairs(rec, fem):
        nl = NodalLoadRecord(
            node_id=int(node),
            force_xyz=(float(force[0]), float(force[1]), float(force[2])),
        )
        emitter.load(int(node), *broker_load_components(nl, ndf_of(int(node)), ndm))


def _emit_load_record(
    rec: _LoadRecord,
    emitter: "Emitter",
    fem: "FEMData",
    ndf_of: "Callable[[int], int]",
) -> None:
    if rec.target_kind == "node":
        node = int(rec.target)
        emitter.load(
            node, *fit_dof_vector(
                rec.forces, ndf_of(node), kind="nodal load", node=node),
        )
        return
    # PG fan-out.
    for node_tag in expand_pg_to_nodes(fem, rec.target):
        emitter.load(
            node_tag, *fit_dof_vector(
                rec.forces, ndf_of(int(node_tag)), kind="nodal load",
                node=int(node_tag)),
        )


def _emit_sp_record(
    rec: _SPRecord, emitter: "Emitter", fem: "FEMData",
) -> None:
    if rec.target_kind == "node":
        emitter.sp(int(rec.target), rec.dof, rec.value)
        return
    for node_tag in expand_pg_to_nodes(fem, rec.target):
        emitter.sp(node_tag, rec.dof, rec.value)


# ---------------------------------------------------------------------------
# Element-load resolution (Plain.ele_loads + Plain.from_model element loads)
# ---------------------------------------------------------------------------


def _clean_zero(val: float) -> float:
    return 0.0 if abs(val) < 1e-15 else float(val)


def _resolve_beam_vecxz(
    el_spec: Element | None,
    p_i: np.ndarray,
    p_j: np.ndarray,
    fem: "FEMData",
    eid: int,
) -> np.ndarray:
    """Resolve the 3D beam ``vecxz`` vector from ``el_spec.transf``."""
    if el_spec is None:
        raise BridgeError(
            f"from_model: beamUniform on 3D element {eid} requires a "
            "registered beam-column element with a geomTransf."
        )
    transf = _element_transf(el_spec)
    if transf is None:
        raise BridgeError(
            f"from_model: beamUniform on 3D element {eid} requires a "
            f"beam-column element with a geomTransf, got {type(el_spec).__name__}."
        )
    if is_orientation_transform(transf):
        orient = transf.orientation  # type: ignore[attr-defined]
        if hasattr(orient, "bind_fem"):
            orient.bind_fem(fem)
        return np.asarray(
            compute_vecxz_for_element(transf, p_i, p_j),  # type: ignore[arg-type]
            dtype=float,
        )
    vecxz = getattr(transf, "vecxz", None)
    if vecxz is None:
        raise BridgeError(
            f"from_model: 3D beam element {eid} transform {transf!r} has "
            "neither vecxz nor orientation."
        )
    return np.asarray(vecxz, dtype=float)


def _offset_line_load(
    el_spec: Element | None,
    p_i: np.ndarray,
    p_j: np.ndarray,
    e_x: np.ndarray,
    ndm: int,
    fem: "FEMData",
    eid: int,
    pattern: object,
) -> "tuple[float, float, np.ndarray] | None":
    """Where a global line load on the node chord meets a joint-offset beam.

    ``None`` when the element has no offsets. Otherwise
    ``(s_i, s_j, t)``: the flexible element runs from ``s_i`` to ``s_j``
    along the node chord (``0`` and ``L`` without end zones), shifted by
    the transverse offset ``t`` (a 3-vector).

    Refuses offsets that tilt the axis away from the chord, or leave no
    flexible length: the load is projected on the chord's local axes,
    and OpenSees builds them between the offset ends, so any other pair
    would project the load on the wrong axes, silently. Offsets that shift
    both ends alike (a cardinal point) or run along the axis (end lengths)
    leave the direction alone.
    """
    transf = _element_transf(el_spec) if el_spec is not None else None
    if transf is None or getattr(transf, "jnt_offset", None) is None:
        return None
    vecxz = None
    if ndm == 3:
        vecxz = tuple(float(c) for c in _resolve_beam_vecxz(
            el_spec, p_i, p_j, fem, eid))
    jnt = element_jnt_offset(transf, p_i, p_j, vecxz, ndm)  # type: ignore[arg-type]
    if not jnt:
        return None
    w = len(jnt) // 2
    d_i = np.zeros(3)
    d_j = np.zeros(3)
    d_i[:w] = jnt[:w]
    d_j[:w] = jnt[w:]
    length = float(np.linalg.norm(p_j - p_i))
    s_i = float(np.dot(d_i, e_x))
    s_j = length + float(np.dot(d_j, e_x))
    t_i = d_i - s_i * e_x
    t_j = d_j - (s_j - length) * e_x
    if s_j - s_i <= 0.0 or float(np.linalg.norm(t_j - t_i)) > 1e-9 * max(
            length, 1.0):
        raise BridgeError(
            f"from_model({pattern!r}): beamUniform on element {eid}: its "
            "geomTransf jnt_offset turns the element axis away from the node "
            "chord (or leaves no flexible length), so a global line load "
            "cannot be projected on the axes OpenSees uses. Declare the load "
            "in local components with Plain.beam_uniform, or give both ends "
            "the same transverse offset."
        )
    return s_i, s_j, t_i


def _segment_resultant(
    q_a: np.ndarray, q_b: np.ndarray, a: float, b: float, u: float, v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(∫ q ds, ∫ s q ds)`` over ``[u, v]`` of the load linear in ``s``
    from ``q_a`` at ``a`` to ``q_b`` at ``b``."""
    beta = (q_b - q_a) / (b - a)
    alpha = q_a - beta * a
    force = alpha * (v - u) + beta * (v * v - u * u) / 2.0
    first = alpha * (v * v - u * u) / 2.0 + beta * (v ** 3 - u ** 3) / 3.0
    return force, first


def _split_end_zone_line_load(
    q1: np.ndarray,
    q2: np.ndarray,
    a_l: float,
    b_l: float,
    length: float,
    e_x: np.ndarray,
    s_i: float,
    s_j: float,
    t: np.ndarray,
    node_i: int,
    node_j: int,
    ndm: int,
) -> "tuple[tuple[np.ndarray, np.ndarray, float, float] | None, list[tuple[int, tuple[float, ...]]]]":
    """Split a global line load on the node chord over a beam's end zones.

    The load runs linearly from ``q1`` at ``a_l * length`` to ``q2`` at
    ``b_l * length`` along the node chord, on the element axis (shifted
    by ``t``), as ETABS/CSI load a frame object over its full length.
    OpenSees' ``-beamUniform`` loads only the flexible element between
    the offset ends (``s_i`` to ``s_j``), so the part over each rigid end
    zone goes to that zone's joint as a force and the moment of that
    force about the joint (the zone is rigid, so this is exact).

    Returns ``(flexible, nodal)``: ``flexible`` is ``(q_a, q_b, aL, bL)``
    for the flexible element (fractions of its length; ``None`` when the
    load lies wholly on the end zones), and ``nodal`` the
    ``(node, components)`` loads, ``(Fx, Fy, Mz)`` in 2-D and
    ``(Fx, Fy, Fz, Mx, My, Mz)`` in 3-D.
    """
    a = a_l * length
    b = b_l * length
    nodal: list[tuple[int, tuple[float, ...]]] = []
    for node, lo, hi, s0 in (
        (node_i, a, min(b, s_i), 0.0),
        (node_j, max(a, s_j), b, length),
    ):
        if hi <= lo:
            continue
        force, first = _segment_resultant(q1, q2, a, b, lo, hi)
        # r(s) - p_node = t + (s - s0) e_x on the element axis.
        moment = np.cross(t, force) + np.cross(e_x, first - s0 * force)
        comps = (
            (force[0], force[1], moment[2]) if ndm == 2
            else (*force, *moment)
        )
        nodal.append((node, tuple(_clean_zero(float(c)) for c in comps)))
    lo, hi = max(a, s_i), min(b, s_j)
    if hi <= lo:
        return None, nodal

    def q_at(s: float) -> np.ndarray:
        return np.asarray(q1 + (q2 - q1) * (s - a) / (b - a), dtype=float)

    flex = s_j - s_i
    return (q_at(lo), q_at(hi), (lo - s_i) / flex, (hi - s_i) / flex), nodal


def _translate_fem_eids_for_ele_load(
    fem_eids: Iterable[int],
    *,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None",
    element_owner: "SortedIntToInt | Mapping[int, int] | None",
    partition_rank: int | None,
    where: str,
) -> list[int]:
    out: list[int] = []
    for raw_eid in fem_eids:
        eid = int(raw_eid)
        if fem_eid_to_ops_tag is not None:
            otag = fem_eid_to_ops_tag.get(eid)
            if otag is None:
                raise BridgeError(
                    f"{where} resolves to FEM element id {eid}, but no "
                    "OpenSees Element primitive was emitted for that element."
                )
        else:
            otag = eid
        if partition_rank is not None and element_owner is not None:
            if element_owner.get(eid) != partition_rank:
                continue
        out.append(int(otag))
    return out


def _membrane_pressure_tags(
    erec: Any,
    element_plan: "Sequence[tuple[Element, ElementPlanRows]] | None",
    element_owner: "SortedIntToInt | Mapping[int, int] | None",
    partition_rank: "int | None",
) -> list[int]:
    """OpenSees tags of the LadrunoMembranes a ``surface_load(pressure=)``
    targets: its ``ele=`` handles, or every membrane declared with its
    ``pg=``.  Fails loud when a target is not a membrane, or ``pg=`` names
    none (the record would otherwise be a beam load on whatever it hit)."""
    from ...element.shell import LadrunoMembrane

    where = "Plain.surface_load(pressure=...)"
    if element_plan is None:
        raise BridgeError(f"{where}: requires the bridge element_plan.")
    specs: "list[tuple[Element, ElementPlanRows]]"
    if erec.target_kind == "pg":
        specs = [
            (s, sub) for s, sub in element_plan
            if isinstance(s, LadrunoMembrane) and s.pg == str(erec.target)
        ]
        if not specs:
            raise BridgeError(
                f"{where}: pg={erec.target!r} names no LadrunoMembrane. "
                "pressure= loads LadrunoMembrane panels only; declare the "
                "membrane with that pg=, or target it by its handle (ele=)."
            )
    else:
        assert isinstance(erec.target, tuple)
        by_id = {id(s): (s, sub) for s, sub in element_plan}
        specs = []
        for item in erec.target:
            hit = by_id.get(id(item))
            if hit is None or not isinstance(hit[0], LadrunoMembrane):
                raise BridgeError(
                    f"{where}: target {item!r} is not a LadrunoMembrane "
                    "registered on this bridge."
                )
            specs.append(hit)
    tags: list[int] = []
    for _s, sub in specs:
        for i in range(len(sub)):
            eid_i = int(sub.eids[i])
            if (
                partition_rank is not None
                and element_owner is not None
                and eid_i != MISSING_FEM_ELEMENT_ID
                and element_owner.get(eid_i) != partition_rank
            ):
                continue
            tags.append(int(sub._tag(i)))
    return tags


def resolve_ele_load_commands(
    spec: Plain,
    fem: "FEMData",
    ndm: int = 3,
    *,
    element_plan: "Sequence[tuple[Element, ElementPlanRows]] | None" = None,
    fem_eid_to_ops_tag: "FemToOpsTagMap | None" = None,
    element_owner: "SortedIntToInt | Mapping[int, int] | None" = None,
    partition_rank: int | None = None,
    nodal_loads: "list[tuple[int, tuple[float, ...]]]",
) -> list[tuple[int | float | str, ...]]:
    """Expand ``spec.ele_loads`` and element-form ``spec.from_model_cases``
    into ``emitter.eleLoad(*args)`` argument tuples.

    ``nodal_loads`` (required, so no caller can drop them) receives the
    ``(node, components)`` joint loads of a ``from_model`` line load's
    share over a beam's rigid end zones (``geomTransf -jntOffset``): the
    caller emits each as ``load`` in the same pattern, fitted to the
    node's ndf, so the beam carries the load over its full node-to-node
    length, as ETABS/CSI do. Components are ``(Fx, Fy, Mz)`` in 2-D and
    ``(Fx, Fy, Fz, Mx, My, Mz)`` in 3-D.
    """
    cmds: list[tuple[int | float | str, ...]] = []

    # 1. Explicit Plain.ele_loads (beam_uniform / surface_load / self_weight).
    for erec in spec.ele_loads:
        ops_tags: list[int] = []
        if erec.load_type == "membranePressure":
            # surface_load(pressure=q): LadrunoMembrane macro-elements only
            # (the record refused label= and non-membrane handles).
            ops_tags = _membrane_pressure_tags(
                erec, element_plan, element_owner, partition_rank)
        elif erec.target_kind == "pg":
            pg_name = str(erec.target)
            macro_tags: list[int] = []
            if element_plan is not None:
                for el_spec, sub in element_plan:
                    if getattr(el_spec, "pg", None) == pg_name:
                        for i in range(len(sub)):
                            if int(sub.eids[i]) == MISSING_FEM_ELEMENT_ID:
                                macro_tags.append(int(sub._tag(i)))
            fem_eids_list = [eid for eid, _conn in expand_pg_to_elements(fem, pg_name)]
            if macro_tags and (
                fem_eid_to_ops_tag is None
                or not any(int(e) in fem_eid_to_ops_tag for e in fem_eids_list)
            ):
                ops_tags = macro_tags
            else:
                ops_tags = _translate_fem_eids_for_ele_load(
                    fem_eids_list,
                    fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                    element_owner=element_owner,
                    partition_rank=partition_rank,
                    where=f"Plain.{erec.load_type}(pg={pg_name!r})",
                )
                ops_tags.extend(macro_tags)
        elif erec.target_kind == "label":
            label_name = str(erec.target)
            fem_eids_t = _expand_label_to_elements(fem, label_name)
            ops_tags = _translate_fem_eids_for_ele_load(
                fem_eids_t,
                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                element_owner=element_owner,
                partition_rank=partition_rank,
                where=f"Plain.{erec.load_type}(label={label_name!r})",
            )
        else:
            assert isinstance(erec.target, tuple)
            for item in erec.target:
                if isinstance(item, Element):
                    if element_plan is None:
                        raise BridgeError(
                            f"Plain.{erec.load_type}(ele=...): Element handle "
                            "requires bridge element_plan."
                        )
                    matched = False
                    for el_spec, sub in element_plan:
                        if id(el_spec) == id(item):
                            matched = True
                            for i in range(len(sub)):
                                eid_i = int(sub.eids[i])
                                otag_i = int(sub._tag(i))
                                if (
                                    partition_rank is not None
                                    and element_owner is not None
                                    and eid_i != MISSING_FEM_ELEMENT_ID
                                    and element_owner.get(eid_i) != partition_rank
                                ):
                                    continue
                                ops_tags.append(otag_i)
                            break
                    if not matched:
                        raise BridgeError(
                            f"Plain.{erec.load_type}(ele=...): Element spec "
                            f"{item!r} was not registered on the bridge."
                        )
                else:
                    item_int = int(item)
                    macro_hit = False
                    if (
                        element_plan is not None
                        and fem_eid_to_ops_tag is not None
                        and item_int not in fem_eid_to_ops_tag
                    ):
                        for _el_spec, sub in element_plan:
                            for i in range(len(sub)):
                                if (
                                    int(sub.eids[i]) == MISSING_FEM_ELEMENT_ID
                                    and int(sub._tag(i)) == item_int
                                ):
                                    ops_tags.append(item_int)
                                    macro_hit = True
                                    break
                            if macro_hit:
                                break
                    if not macro_hit:
                        ops_tags.extend(
                            _translate_fem_eids_for_ele_load(
                                (item_int,),
                                fem_eid_to_ops_tag=fem_eid_to_ops_tag,
                                element_owner=element_owner,
                                partition_rank=partition_rank,
                                where=f"Plain.{erec.load_type}(ele={item!r})",
                            )
                        )

        if not ops_tags:
            continue
        if erec.load_type == "beamUniform":
            cmds.append(
                (
                    "-ele",
                    *ops_tags,
                    "-type",
                    "-beamUniform",
                    *_beam_uniform_args(erec, ndm),
                )
            )
        elif erec.load_type == "surfaceLoad":
            cmds.append(("-ele", *ops_tags, "-type", "-surfaceLoad"))
        elif erec.load_type == "membranePressure":
            cmds.append((
                "-ele", *ops_tags, "-type", "-beamUniform",
                *_membrane_pressure_args(erec, ndm),
            ))
        elif erec.load_type == "selfWeight":
            cmds.append(
                (
                    "-ele",
                    *ops_tags,
                    "-type",
                    "-selfWeight",
                    *_self_weight_args(erec, ndm),
                )
            )

    # 2. Element-form records from Plain.from_model(case).
    if not spec.from_model_cases:
        return cmds
    elements_obj = getattr(fem, "elements", None)
    elem_load_set = (
        getattr(elements_obj, "loads", None) if elements_obj is not None else None
    )
    if elem_load_set is None:
        return cmds

    case_records = [
        rec
        for case in spec.from_model_cases
        for rec in elem_load_set.by_pattern(case)
    ]
    if not case_records:
        return cmds

    target_eids = {int(r.element_id) for r in case_records}
    plan_info: dict[int, tuple[Element | None, tuple[int, ...], int]] = {}
    if element_plan is not None:
        for el_spec, sub in element_plan:
            for i in range(len(sub)):
                eid_i = int(sub.eids[i])
                if eid_i in target_eids:
                    _e, conn_i, tag_i = sub[i]
                    plan_info[eid_i] = (el_spec, conn_i, tag_i)
    elif elements_obj is not None:
        if getattr(elements_obj, "physical", None) is not None:
            for pg_name in elements_obj.physical.names():
                for eid_i, conn_row in expand_pg_to_elements(fem, pg_name):
                    ei = int(eid_i)
                    if ei in target_eids and ei not in plan_info:
                        otag_i = (
                            int(fem_eid_to_ops_tag.get(ei) or ei)
                            if fem_eid_to_ops_tag is not None
                            else ei
                        )
                        plan_info[ei] = (
                            None,
                            tuple(int(n) for n in conn_row),
                            otag_i,
                        )
        groups_map = getattr(elements_obj, "groups", None)
        if isinstance(groups_map, Mapping):
            for grp in groups_map.values():
                g_ids = getattr(grp, "ids", ())
                g_conn = getattr(grp, "connectivity", ())
                for eid_i, conn_row in zip(g_ids, g_conn):
                    ei = int(eid_i)
                    if ei in target_eids and ei not in plan_info:
                        otag_i = (
                            int(fem_eid_to_ops_tag.get(ei) or ei)
                            if fem_eid_to_ops_tag is not None
                            else ei
                        )
                        plan_info[ei] = (
                            None,
                            tuple(int(n) for n in conn_row),
                            otag_i,
                        )

    needed_nodes = {
        int(n) for _spec, conn, _tag in plan_info.values() for n in conn[:2]
    }
    coord_of: dict[int, np.ndarray] = {}
    if needed_nodes and getattr(fem, "nodes", None) is not None:
        ids = np.asarray(fem.nodes.ids)
        coords = np.asarray(fem.nodes.coords, dtype=float)
        coord_of = {
            int(t): coords[i]
            for i, t in enumerate(ids)
            if int(t) in needed_nodes
        }

    for rec in case_records:
        eid = int(rec.element_id)
        if element_plan is not None or fem_eid_to_ops_tag is not None:
            if eid not in plan_info:
                otag_check = (
                    fem_eid_to_ops_tag.get(eid)
                    if fem_eid_to_ops_tag is not None
                    else None
                )
                if otag_check is None:
                    raise BridgeError(
                        f"from_model({rec.pattern!r}): ElementLoadRecord "
                        f"(load_type={rec.load_type!r}) targets FEM element "
                        f"id {eid}, but no OpenSees Element primitive was "
                        "emitted for that element."
                    )
        if partition_rank is not None and element_owner is not None:
            if element_owner.get(eid) != partition_rank:
                continue

        elem_spec, conn, otag = plan_info.get(
            eid,
            (
                None,
                (),
                int(fem_eid_to_ops_tag.get(eid) or eid)
                if fem_eid_to_ops_tag is not None
                else eid,
            ),
        )
        if rec.load_type in ("beamUniform", "line_load"):
            if elem_spec is not None and _element_transf(elem_spec) is None:
                raise BridgeError(
                    f"from_model({rec.pattern!r}): {rec.load_type} on element {eid} "
                    f"requires a beam-column element with a geomTransf, got "
                    f"{type(elem_spec).__name__} (not a beam-column element)."
                )
            wx1 = float(rec.params.get("wx", 0.0))
            wy1 = float(rec.params.get("wy", 0.0))
            wz1 = float(rec.params.get("wz", 0.0))
            wx2 = float(rec.params.get("wx_end", wx1))
            wy2 = float(rec.params.get("wy_end", wy1))
            wz2 = float(rec.params.get("wz_end", wz1))
            a_l = float(rec.params.get("aL", 0.0))
            b_l = float(rec.params.get("bL", 1.0))
            if len(conn) < 2:
                raise BridgeError(
                    f"from_model({rec.pattern!r}): beamUniform on element {eid} "
                    f"requires 2 endpoint nodes, got connectivity {conn!r}."
                )
            p_i = coord_of[int(conn[0])]
            p_j = coord_of[int(conn[1])]
            edge = p_j - p_i
            length = float(np.linalg.norm(edge))
            if length <= 0.0:
                raise BridgeError(
                    f"from_model({rec.pattern!r}): beamUniform on element {eid} "
                    "has zero length."
                )
            e_x = edge / length
            if ndm == 2 and (wz1 != 0.0 or wz2 != 0.0):
                raise _BridgeValueError(
                    f"from_model({rec.pattern!r}): beamUniform on element {eid} "
                    f"has non-zero out-of-plane wz=({wz1}, {wz2}) in a 2D "
                    "model (ndm=2)."
                )
            offset = _offset_line_load(
                elem_spec, p_i, p_j, e_x, ndm, fem, eid, rec.pattern)
            if offset is not None:
                # The load covers the node chord (ETABS); the end zones'
                # share goes to the joints, the rest to the flexible span.
                flexible, end_loads = _split_end_zone_line_load(
                    np.array([wx1, wy1, wz1], dtype=float),
                    np.array([wx2, wy2, wz2], dtype=float),
                    a_l, b_l, length, e_x, *offset,
                    int(conn[0]), int(conn[1]), ndm,
                )
                nodal_loads.extend(end_loads)
                if flexible is None:
                    continue
                q_a, q_b, a_l, b_l = flexible
                wx1, wy1, wz1 = (float(c) for c in q_a)
                wx2, wy2, wz2 = (float(c) for c in q_b)
            if ndm == 2:
                e_y = np.array([-e_x[1], e_x[0], 0.0], dtype=float)
                q1 = np.array([wx1, wy1, 0.0], dtype=float)
                q2 = np.array([wx2, wy2, 0.0], dtype=float)
                wx_loc_a = _clean_zero(float(np.dot(q1, e_x)))
                wy_loc_a = _clean_zero(float(np.dot(q1, e_y)))
                wx_loc_b = _clean_zero(float(np.dot(q2, e_x)))
                wy_loc_b = _clean_zero(float(np.dot(q2, e_y)))
                erec_proj = _EleLoadRecord(
                    load_type="beamUniform",
                    target_kind="ele",
                    target=(otag,),
                    wy=(wy_loc_a, wy_loc_b),
                    wz=None,
                    wx=(wx_loc_a, wx_loc_b),
                    aL=a_l,
                    bL=b_l,
                )
            else:
                v_xz = _resolve_beam_vecxz(elem_spec, p_i, p_j, fem, eid)
                cross_y = np.cross(v_xz, e_x)
                norm_y = float(np.linalg.norm(cross_y))
                if norm_y <= 1e-12:
                    raise BridgeError(
                        f"from_model({rec.pattern!r}): beamUniform on element {eid} "
                        "has geomTransf vecxz parallel to element chord."
                    )
                e_y = cross_y / norm_y
                e_z = np.cross(e_x, e_y)
                q1 = np.array([wx1, wy1, wz1], dtype=float)
                q2 = np.array([wx2, wy2, wz2], dtype=float)
                wx_loc_a = _clean_zero(float(np.dot(q1, e_x)))
                wy_loc_a = _clean_zero(float(np.dot(q1, e_y)))
                wz_loc_a = _clean_zero(float(np.dot(q1, e_z)))
                wx_loc_b = _clean_zero(float(np.dot(q2, e_x)))
                wy_loc_b = _clean_zero(float(np.dot(q2, e_y)))
                wz_loc_b = _clean_zero(float(np.dot(q2, e_z)))
                erec_proj = _EleLoadRecord(
                    load_type="beamUniform",
                    target_kind="ele",
                    target=(otag,),
                    wy=(wy_loc_a, wy_loc_b),
                    wz=(wz_loc_a, wz_loc_b),
                    wx=(wx_loc_a, wx_loc_b),
                    aL=a_l,
                    bL=b_l,
                )
            cmds.append(
                (
                    "-ele",
                    otag,
                    "-type",
                    "-beamUniform",
                    *_beam_uniform_args(erec_proj, ndm),
                )
            )
        elif rec.load_type in ("surfacePressure", "surfaceLoad"):
            cmds.append(("-ele", otag, "-type", "-surfaceLoad"))
        elif rec.load_type in ("bodyForce", "selfWeight"):
            if "g" in rec.params:
                g_vec = rec.params["g"]
                wx_sw, wy_sw, wz_sw = (
                    float(g_vec[0]),
                    float(g_vec[1]),
                    float(g_vec[2]),
                )
            elif "bf" in rec.params:
                bf_vec = rec.params["bf"]
                wx_sw, wy_sw, wz_sw = (
                    float(bf_vec[0]),
                    float(bf_vec[1]),
                    float(bf_vec[2]),
                )
            else:
                wx_sw = float(rec.params.get("wx", 0.0))
                wy_sw = float(rec.params.get("wy", 0.0))
                wz_sw = float(rec.params.get("wz", 0.0))
            if ndm == 2:
                if wz_sw != 0.0:
                    raise _BridgeValueError(
                        f"from_model({rec.pattern!r}): bodyForce on element {eid} "
                        f"has non-zero wz={wz_sw} in a 2D model (ndm=2)."
                    )
                cmds.append(("-ele", otag, "-type", "-selfWeight", wx_sw, wy_sw))
            else:
                cmds.append(
                    ("-ele", otag, "-type", "-selfWeight", wx_sw, wy_sw, wz_sw)
                )
        else:
            raise BridgeError(
                f"from_model({rec.pattern!r}): unsupported ElementLoadRecord "
                f"load_type={rec.load_type!r} on element {eid}."
            )
    return cmds
