"""patterns: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Any, Callable, Iterable

import numpy as np

from ..types import Primitive
from ...pattern.pattern import Plain, _LoadRecord, _SPRecord
from ..tag_resolution import resolve_tag
from .errors import BridgeError
from .pg_expand import expand_pg_to_nodes
from .ndf import _load_dof_layout, fit_dof_vector


if TYPE_CHECKING:
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData


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
    lines inside this pattern (``ndf`` maps the DOF-agnostic records).

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
