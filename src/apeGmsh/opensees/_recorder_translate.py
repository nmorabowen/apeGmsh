"""Canonical-name → OpenSees recorder token translation (Phase 9).

The :class:`~apeGmsh.opensees.recorder.RecorderDeclaration` carries
components in canonical vocabulary (``"displacement_x"``,
``"reaction_force_y"``, ``"axial_force"``, …). Emitting them as
OpenSees ``recorder`` commands requires translation to the native
token shapes (``"disp"`` + ``-dof 1``, ``"reaction"`` + ``-dof 5``,
``"section force"`` per integration point, …).

This module owns the translation tables for **node-level** components
(Phase 9 commit 3a). Element-level translation (elements / gauss /
line_stations) lands in commit 3b and lifts the heavier translation
logic from :mod:`apeGmsh.results.spec._emit`.

The :class:`Node`, :class:`Element`, and :class:`MPCO` typed
primitives in :mod:`apeGmsh.opensees.recorder` continue to take raw
OpenSees tokens directly (no translation needed); only the unified
``RecorderDeclaration`` consumed by
:func:`~apeGmsh.opensees._internal.build.recorders.emit_recorder_spec` flows
through these tables.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

from ._element_capabilities import (
    SIGMA_ZZ_ELEMENT_CLASSES,
    SIGMA_ZZ_MATERIAL_CLASSES,
    STRESS_PLANE_STRAIN_RESPONSE,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from ._readback import RotationRoles

    from numpy import ndarray

    from apeGmsh.mesh.FEMData import FEMData


class StressZZNotRecordedWarning(UserWarning):
    """A ``stress_zz`` gauss record could not be promoted to the fork's
    4-component plane-strain stress response.

    Raised at emit time by :func:`element_record_response_tokens` when the
    record's elements or their materials do not expose the out-of-plane
    σ33 (see
    :func:`~apeGmsh.opensees._element_capabilities.element_records_stress_zz`).
    The recorder still writes the 3 in-plane components and the read side
    reconstructs σzz from Poisson's ratio, so this is a precision warning,
    not a failure — but at plastic Gauss points that elastic estimate is
    wrong by tens of percent, and this is the only place the user hears
    about it.
    """


# =====================================================================
# Per-axis spatial index tables
# =====================================================================
#
# A canonical node component names a SPATIAL slot, indexed
# ``(x, y, z, rx, ry, rz)`` = ``0..5``.  Which deck DOF carries that slot
# depends on the node's ``(ndm, ndf)``: a planar frame node
# (``ndm=2, ndf=3``) carries ``rz`` at DOF 3, a 3-D frame node at DOF 6,
# and a truss node not at all.  That map is the bridge's own DOF layout
# (``_internal/build/ndf.py::_load_dof_layout``, the one loads, masses,
# SPs and readback use); callers pass it in.  There is no fixed 6-DOF
# fallback: ``-dof 6`` on a planar frame node records 0, silently.

_TRANS_AXIS_TO_SPATIAL: dict[str, int] = {"x": 0, "y": 1, "z": 2}
_ROT_AXIS_TO_SPATIAL: dict[str, int] = {"x": 3, "y": 4, "z": 5}


# =====================================================================
# Canonical-prefix → (ops_recorder_token, axis_kind) table
# =====================================================================
#
# axis_kind ∈ {"trans", "rot"} selects which spatial table the axis
# suffix is read against.

_NODAL_PREFIX_TABLE: dict[str, tuple[str, str]] = {
    "displacement":           ("disp",      "trans"),
    "rotation":               ("disp",      "rot"),
    "velocity":               ("vel",       "trans"),
    "angular_velocity":       ("vel",       "rot"),
    "acceleration":           ("accel",     "trans"),
    "angular_acceleration":   ("accel",     "rot"),
    "displacement_increment": ("incrDisp",  "trans"),
    "reaction_force":         ("reaction",  "trans"),
    "reaction_moment":        ("reaction",  "rot"),
    # OpenSees ``unbalance`` returns residual nodal forces.
    "force":                  ("unbalance", "trans"),
    "moment":                 ("unbalance", "rot"),
}


# =====================================================================
# Scalar canonical names (no axis suffix)
# =====================================================================
#
# ``-1`` is a sentinel meaning "OpenSees infers the DOF" — used for
# pressure (the formulation-dependent pressure DOF). Real emitters
# decide based on the bridge's ``ndf`` whether to emit ``-dof <pdof>``
# explicitly or rely on the OpenSees default.

_NODAL_SCALAR_TABLE: dict[str, tuple[str, int]] = {
    "pore_pressure": ("pressure", -1),
}


# =====================================================================
# Public translation API
# =====================================================================


def node_component_spatial(canonical: str) -> Optional[tuple[str, int]]:
    """Map a canonical node component to ``(ops_recorder_token, spatial)``.

    ``spatial`` indexes ``(x, y, z, rx, ry, rz)`` (``0..5``); it is ``-1``
    for the scalar ``pore_pressure``, which has no spatial slot. Returns
    ``None`` if ``canonical`` is not a recognized node-level component.

    Examples
    --------
    >>> node_component_spatial("rotation_z")
    ('disp', 5)
    >>> node_component_spatial("bogus") is None
    True
    """
    if canonical in _NODAL_SCALAR_TABLE:
        return _NODAL_SCALAR_TABLE[canonical]
    if "_" not in canonical:
        return None
    prefix, axis = canonical.rsplit("_", 1)
    entry = _NODAL_PREFIX_TABLE.get(prefix)
    if entry is None:
        return None
    ops_token, axis_kind = entry
    table = (_TRANS_AXIS_TO_SPATIAL if axis_kind == "trans"
             else _ROT_AXIS_TO_SPATIAL)
    if axis not in table:
        return None
    return (ops_token, table[axis])


def node_component_to_ops(
    canonical: str, layout: tuple[int, ...],
) -> Optional[tuple[str, int]]:
    """Map a canonical node component to ``(ops_recorder_token, dof)``
    through ``layout``: the spatial slot each DOF carries, as
    ``_load_dof_layout(ndf, ndm)`` gives it.

    Returns ``None`` if ``canonical`` is not a recognized node-level
    component (caller should fall through to error). Raises
    :class:`ValueError` if the layout has no DOF for it (a rotation on a
    truss node, ``displacement_z`` in 2-D): never a guessed DOF.

    Examples
    --------
    >>> node_component_to_ops("displacement_x", (0, 1, 2, 3, 4, 5))
    ('disp', 1)
    >>> node_component_to_ops("rotation_z", (0, 1, 2, 3, 4, 5))
    ('disp', 6)
    >>> node_component_to_ops("rotation_z", (0, 1, 5))
    ('disp', 3)
    >>> node_component_to_ops("reaction_force_y", (0, 1, 5))
    ('reaction', 2)
    >>> node_component_to_ops("pore_pressure", (0, 1))
    ('pressure', -1)
    >>> node_component_to_ops("bogus", (0, 1)) is None
    True
    """
    translated = node_component_spatial(canonical)
    if translated is None:
        return None
    ops_token, spatial = translated
    if spatial < 0:
        return (ops_token, spatial)
    if spatial not in layout:
        raise ValueError(
            f"{canonical!r} has no DOF in the node layout {layout} "
            f"(spatial slots 0..5 = x, y, z, rx, ry, rz)."
        )
    return (ops_token, layout.index(spatial) + 1)


def group_node_components_by_ops_token(
    components: tuple[str, ...], layout: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    """Group node components by their OpenSees token, each mapped
    through ``layout`` (see :func:`node_component_to_ops`).

    Components sharing the same ``ops_recorder_token`` collapse into
    one ``recorder Node`` line with a ``-dof`` list of all their DOFs.
    For example, ``("displacement_x", "displacement_y", "reaction_force_z")``
    over the 3-D frame layout ``(0, 1, 2, 3, 4, 5)`` produces:

        {"disp": (1, 2), "reaction": (3,)}

    Returns
    -------
    Mapping from ops token to ordered tuple of DOFs (1-based, OpenSees
    convention). Order within each group preserves component
    declaration order; ops tokens themselves are returned in
    discovery order.

    Raises
    ------
    ValueError
        If any component is not a recognized node-level canonical
        (caller is responsible for validating against the canonical
        vocabulary upstream; this raise is a defensive backstop), or
        if it has no DOF in ``layout``.
    """
    grouped: dict[str, list[int]] = {}
    for comp in components:
        translated = node_component_to_ops(comp, layout)
        if translated is None:
            raise ValueError(
                f"_recorder_translate.group_node_components_by_ops_token: "
                f"{comp!r} is not a recognized node-level canonical."
            )
        ops_token, dof = translated
        grouped.setdefault(ops_token, []).append(dof)
    # Deduplicate DOFs per token while preserving insertion order.
    return {
        token: tuple(dict.fromkeys(dofs)) for token, dofs in grouped.items()
    }



def node_record_deck_dofs(
    slot_dofs: tuple[int, ...],
    node_ids: tuple[int, ...],
    nodes_ndf: "Mapping[int, int]",
    ndm: int,
    *,
    rotation_refusal: "Callable[[int], str | None]",
    where: str,
) -> tuple[int, ...]:
    """The ``-dof`` list a ``recorder Node`` line over ``node_ids`` records.

    ``rotation_refusal(node)`` says why a node whose layout holds a
    rotational slot carries no rotation there (``RotationRoles.refusal``:
    a u-p node's DOF ``ndm + 1`` is pore pressure), or ``None``. It is
    asked only for nodes whose layout holds a rotational slot the record
    asks for, and any refusal raises.

    ``slot_dofs`` name spatial slots as 1-based DOFs of the full
    ``(x, y, z, rx, ry, rz)`` layout (``rotation_z`` is 6), which is how
    the spec route's read side keys its columns. Each slot maps to the
    DOF each node's own layout (``_node_layout(ndf, ndm)``) places it at:
    a planar frame node (``ndm=2, ndf=3``) carries ``rotation_z`` at DOF 3.

    The one ``-dof`` list serves every node, so the nodes that carry a
    slot must agree on its DOF. A rotation some nodes lack is recorded
    anyway: their padded column is masked on read
    (:func:`mask_absent_rotations`). Raises :class:`ValueError` for a node
    missing from ``nodes_ndf``, a slot no node carries, a translation some
    nodes lack (nothing would mask its padding), or a slot the nodes place
    at different DOFs.
    """
    from ._internal.build.recorders import _node_layout

    missing = [n for n in node_ids if int(n) not in nodes_ndf]
    if missing:
        raise ValueError(
            f"{where}: nodes {missing[:10]} have no ndf in model=.nodes_ndf, "
            f"so the DOF each records cannot be named. Pass the "
            f"OpenSeesModel the deck was built from (OpenSeesModel.from_h5 "
            f"of the bridge's ops.h5(...) file)."
        )
    layouts = {int(n): _node_layout(int(nodes_ndf[int(n)]), int(ndm))
               for n in node_ids}
    names = ("x", "y", "z", "rx", "ry", "rz")
    out: list[int] = []
    for d in slot_dofs:
        slot = int(d) - 1
        if not 0 <= slot < 6:
            raise ValueError(
                f"{where}: slot DOF {d} is not one of 1..6 "
                f"(x, y, z, rx, ry, rz)."
            )
        at = {n: lay.index(slot) + 1 for n, lay in layouts.items()
              if slot in lay}
        lacking = [n for n in layouts if n not in at]
        if not at:
            raise ValueError(
                f"{where}: no recorded node carries {names[slot]!r} "
                f"(ndm={ndm}; node {node_ids[0]} has ndf "
                f"{nodes_ndf[int(node_ids[0])]}). Drop the component, or "
                f"narrow the record to nodes that carry it."
            )
        if lacking and slot < 3:
            raise ValueError(
                f"{where}: nodes {lacking[:10]} carry no {names[slot]!r} "
                f"DOF (ndm={ndm}), and a recorder pads it with 0.0. Record "
                f"them in a separate record without that component."
            )
        if slot >= 3:
            refused = [(n, why) for n in at
                       if (why := rotation_refusal(n)) is not None]
            if refused:
                listed = "; ".join(
                    f"node {n} ({why})" for n, why in refused[:3])
                more = len(refused) - min(len(refused), 3)
                raise ValueError(
                    f"{where}: {names[slot]!r} has no established DOF at "
                    f"{listed}" + (f"; and {more} more node(s)" if more else "")
                    + ". Narrow the record to nodes whose elements carry a "
                    "rotation, or drop the rotational component."
                )
        dofs = set(at.values())
        if len(dofs) > 1:
            shown = {n: at[n] for n in list(at)[:6]}
            raise ValueError(
                f"{where}: the recorded nodes carry {names[slot]!r} at "
                f"different DOFs ({shown}); one recorder line records one "
                f"DOF list. Split the record by node ndf."
            )
        out.append(dofs.pop())
    if out != sorted(set(out)):
        raise ValueError(  # pragma: no cover - laid-out layouts are monotone
            f"{where}: DOFs {out} for slots {slot_dofs} are out of order; "
            f"the read side maps columns by slot order."
        )
    return tuple(out)


def model_node_deck_dofs(
    slot_dofs: tuple[int, ...],
    node_ids: tuple[int, ...],
    model: object,
    *,
    where: str,
) -> tuple[int, ...]:
    """:func:`node_record_deck_dofs` over a built ``OpenSeesModel``: its
    ``ndm``, ``nodes_ndf`` and, for a rotational slot, the rotation roles
    of its elements (:func:`~apeGmsh.opensees._readback.rotation_roles_from_tokens`),
    the table the ``recorder.declare`` emit and ``.read`` use.

    A :class:`LiveDomainFacts` (a FEMData archive beside a live domain)
    answers from the live domain instead (#2251)."""
    if isinstance(model, LiveDomainFacts):
        return model.node_deck_dofs(slot_dofs, node_ids, where=where)
    ndm = int(model.ndm)  # type: ignore[attr-defined]
    if ndm < 1:
        raise ValueError(
            f"{where}: model= declares no ndm (ndm={ndm}), so the node "
            f"DOFs cannot be mapped. Pass the model built through "
            f"apeSees(fem) and ops.model(ndm=, ndf=)."
        )
    roles: "list[RotationRoles]" = []

    def refusal(node: int) -> "str | None":
        if not roles:
            roles.append(_model_rotation_roles(model, node_ids, where=where))
        return roles[0].refusal(int(node))

    return node_record_deck_dofs(
        slot_dofs, node_ids,
        model.nodes_ndf,  # type: ignore[attr-defined]
        ndm, rotation_refusal=refusal, where=where,
    )


def _model_rotation_roles(
    model: object, node_ids: tuple[int, ...], *, where: str,
) -> "RotationRoles":
    """The rotation roles of ``node_ids`` under ``model``'s elements.

    Only the element plan's rows count, as for :func:`rotation_roles` on
    the declared path. The emitter marks those rows through its fan-out
    side channels: a FEM id for a mesh-fanned element, a node list for a
    node-pair one (ADR 0049; no node-pair element predates the 2.17.0
    ``inline_connectivity`` that persists it). The side-list passes
    (couplings, rigid bodies, embedded ties, interface springs) clear
    both, so a record with neither (``fem_eid`` is
    ``MISSING_FEM_ELEMENT_ID``, no connectivity) is not a plan element and
    is skipped. A rehydrated plan row keeps no connectivity: its nodes come
    from the model's FEMData by ``fem_eid``. A plan row whose FEM id that
    FEMData lacks raises, unless its class is a known beam/shell/
    zero-length or translational one, since skipping that only ever
    refuses more.
    """
    from ._internal.tag_resolution import MISSING_FEM_ELEMENT_ID
    from ._readback import (
        _CLASS_FOR_TOKEN,
        _ROTATIONAL_FAMILIES,
        _TRANSLATIONAL_ONLY,
        rotation_roles_from_tokens,
    )

    records = tuple(model.elements())  # type: ignore[attr-defined]
    conn_of: "dict[int, tuple[int, ...]] | None" = None
    rows: list[tuple[str, tuple[int, ...]]] = []
    for rec in records:
        nodes = tuple(int(n) for n in rec.connectivity)
        if not nodes and int(rec.fem_eid) == MISSING_FEM_ELEMENT_ID:
            continue                # a side-list element, not in the plan
        if not nodes and int(rec.fem_eid) >= 0:
            if conn_of is None:
                conn_of = {}
                for group in model.fem.elements:  # type: ignore[attr-defined]
                    for eid, conn in group:
                        conn_of[int(eid)] = tuple(int(n) for n in conn)
            nodes = conn_of.get(int(rec.fem_eid), ())
        if not nodes:
            cls = _CLASS_FOR_TOKEN.get(rec.type_token)
            if cls is not None and issubclass(
                    cls, _ROTATIONAL_FAMILIES + _TRANSLATIONAL_ONLY):
                continue
            raise ValueError(
                f"{where}: element tag {rec.tag} ({rec.type_token}) has no "
                f"node list in model= (no connectivity, and FEM id "
                f"{rec.fem_eid} is not in its FEMData), so whether it "
                f"blocks a rotation at the recorded nodes is unknown."
            )
        rows.append((rec.type_token, nodes))
    return rotation_roles_from_tokens(rows, node_ids)


def model_h5_rotation_roles(
    model_h5: "str | Path", node_ids: "Iterable[int]", *, where: str,
) -> "RotationRoles":
    """The rotation roles (:class:`~._readback.RotationRoles`) of
    ``node_ids`` under the elements ``model_h5`` archives: the table the
    ``recorder.declare`` emit and ``.read`` use (:func:`_model_rotation_roles`),
    so a read mask and the deck agree on which slots hold a rotation."""
    from .opensees_model import OpenSeesModel

    return _model_rotation_roles(
        OpenSeesModel.from_h5(model_h5),
        tuple(int(n) for n in node_ids), where=where,
    )


#: ``ops.eleType`` class name -> the ``element`` token of the bridge class
#: whose family it is (:data:`~._readback._CLASS_FOR_TOKEN`), so a live
#: domain the bridge did not build classifies its rotational DOF slots by
#: the same family tables. Each name is what stock openseespy
#: 3.7.1 answers for an element that token makes; a class not listed is
#: unknown, and blocks the rotational slots of the nodes it touches and
#: of element-less nodes, as an unknown token does.
_TOKEN_FOR_LIVE_CLASS: "dict[str, str]" = {
    "ElasticBeam2d": "elasticBeamColumn",
    "ElasticBeam3d": "elasticBeamColumn",
    "ForceBeamColumn2d": "forceBeamColumn",
    "ForceBeamColumn3d": "forceBeamColumn",
    "DispBeamColumn2d": "dispBeamColumn",
    "DispBeamColumn3d": "dispBeamColumn",
    "ElasticTimoshenkoBeam2d": "ElasticTimoshenkoBeam",
    "ElasticTimoshenkoBeam3d": "ElasticTimoshenkoBeam",
    "ShellMITC4": "ShellMITC4",
    "ShellDKGQ": "ShellDKGQ",
    "ASDShellQ4": "ASDShellQ4",
    "ASDShellT3": "ASDShellT3",
    "ZeroLength": "zeroLength",
    "ZeroLengthSection": "zeroLengthSection",
    "CoupledZeroLength": "CoupledZeroLength",
    "TwoNodeLink": "twoNodeLink",
    "Truss": "Truss",
    "CorotTruss": "CorotTruss",
    "InertiaTruss": "InertiaTruss",
    "FourNodeQuad": "quad",
    "Tri31": "tri31",
    "SixNodeTri": "tri6n",
    "FourNodeTetrahedron": "FourNodeTetrahedron",
    "TenNodeTetrahedron": "TenNodeTetrahedron",
    "Brick": "stdBrick",
}


def _live_int(value: object, query: str) -> int:
    """The one integer an ``ops`` query answers (openseespy returns it as a
    one-element list); anything else raises, naming the query."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"ops.{query} answered {value!r}, not one integer; the live "
            "domain's DOF facts cannot be read from this module.")
    return value


def live_ndm(ops: object) -> int:
    """``ops.getNDM()``: the live domain's spatial dimension, ``0`` when no
    ``ops.model`` has been declared."""
    return _live_int(ops.getNDM(), "getNDM()")  # type: ignore[attr-defined]


def _same_position(coords: "ndarray") -> float:
    """The distance under which two FEM nodes share one position:
    ``1e-9`` of the coordinate scale."""
    import numpy as np

    return 1e-9 * max(float(np.abs(coords).max(initial=0.0)), 1.0)


#: Candidate pairs :func:`_near_points` forms at once (about 8 arrays of
#: this length are alive per chunk).
_PAIR_BUDGET = 4_000_000


def _near_points(
    coords: "ndarray", rows: "ndarray", h: float, same: float,
) -> "tuple[ndarray, ndarray]":
    """For each point ``coords[rows]``: the nearest other point farther
    than ``same`` among the ``3**d`` grid cells of side ``h`` around it
    (``inf`` when there is none), and whether another point lies within
    ``same`` of it.

    Every point within ``h`` of a query lies in those cells, so a nearest
    distance at most ``h`` is exact. NumPy only: cells are matched by
    ``np.unique`` over integer cell tuples, so no cell index can overflow.
    The query rows are taken in chunks of at most :data:`_PAIR_BUDGET`
    candidate pairs (a single row may exceed it alone), so a dense cluster
    bounds the memory, not the square of its cell's population.
    """
    import itertools

    import numpy as np

    n, d = coords.shape
    cell = np.floor((coords - coords.min(axis=0)) / h).astype(np.int64)
    offs = np.array(list(itertools.product((-1, 0, 1), repeat=d)),
                    dtype=np.int64)
    k = len(offs)
    probe = (cell[rows][:, None, :] + offs[None, :, :]).reshape(-1, d)
    inv = np.unique(np.vstack([cell, probe]), axis=0,
                    return_inverse=True)[1].ravel()
    order = np.argsort(inv[:n], kind="stable")
    sorted_cells = inv[:n][order]
    lo = np.searchsorted(sorted_cells, inv[n:], "left")
    cnt = np.searchsorted(sorted_cells, inv[n:], "right") - lo
    per_row = np.cumsum(cnt.reshape(len(rows), k).sum(axis=1))
    shared = np.zeros(len(rows), dtype=bool)
    best = np.full(len(rows), np.inf)
    start = 0
    while start < len(rows):
        done = int(per_row[start - 1]) if start else 0
        stop = max(start + 1, int(np.searchsorted(
            per_row, done + _PAIR_BUDGET, "right")))
        c = cnt[start * k:stop * k]
        owner = start + np.repeat(np.arange((stop - start) * k) // k, c)
        first = np.repeat(lo[start * k:stop * k], c)
        step = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        cand = order[first + step]
        dist = np.linalg.norm(coords[cand] - coords[rows[owner]], axis=1)
        other = cand != rows[owner]
        shared[owner[other & (dist <= same)]] = True
        far = other & (dist > same)
        np.minimum.at(best, owner[far], dist[far])
        start = stop
    return best, shared


def _identity_tolerances(coords: "ndarray", rows: "ndarray") -> "ndarray":
    """A quarter of each ``coords[rows]`` point's distance to the nearest
    point of ``coords`` at another position.

    Within that radius a point is nearer its own node than any other
    node at another position. Points closer than :func:`_same_position`
    share one position (coincident constraint nodes): coordinates cannot
    tell them apart, so they do not shrink each other's radius, and
    :meth:`LiveDomainFacts._check_shared_positions` tells them apart by
    element instead. A point with no other position gets the
    :func:`_same_position` floor.

    The search bins the points on a grid near the mean spacing and grows
    the cells fourfold for the points whose nearest neighbour lies
    farther than one cell, so a graded mesh costs a few passes, not a
    quadratic search.
    """
    import numpy as np

    same = _same_position(coords)
    nearest = np.full(len(rows), np.inf)
    spans = np.ptp(coords, axis=0)
    live = spans > same
    if live.any() and len(coords) > 1:
        h = max(float(np.prod(spans[live]) / len(coords))
                ** (1.0 / int(live.sum())), 2.0 * same)
        diag = float(np.linalg.norm(spans))
        todo = np.arange(len(rows))
        while todo.size:
            best, _ = _near_points(coords, rows[todo], h, same)
            done = (best <= h) | (h > diag)
            nearest[todo[done]] = best[done]
            todo = todo[~done]
            h *= 4.0
    out = 0.25 * nearest
    out[~np.isfinite(out)] = same
    return np.asarray(np.maximum(out, same))


class LiveDomainFacts:
    """The node DOF facts of the live domain ``ops`` holds, standing in for
    the model on the emit side when the model archive is a FEMData archive
    (``fem.to_h5``: no ``/opensees`` zone, no declared ndm; #2251).

    Each fact is read from the live domain, never inferred from the
    FEMData: ``getNDM()``, ``getNDF(node)`` for each node a record names,
    and ``eleType`` / ``eleNodes`` of every element for the rotation
    roles (:data:`_TOKEN_FOR_LIVE_CLASS`). Each recorded node's
    ``nodeCoord`` must be the FEM node's first ``ndm`` coordinates, so the
    tag names the same node in both. ``lineage`` is the archive's, so the
    snapshot check is the archive's own.

    ``elements()`` is the element pairing :meth:`pair_elements` proved
    from the live domain (#2269): a FEM element pairs with the live
    element of its own id only when that element's ``eleNodes`` is the
    FEM connectivity, in order, and its nodes pass the node identity
    check. Its type token is the live ``eleType`` through
    :data:`_TOKEN_FOR_LIVE_CLASS`. Nothing is paired until it is proved,
    so an unproved id is refused by every consumer of the pairing.

    A rotation is recorded only where it reads back as written: a FEMData
    archive holds no ``/opensees/nodes_ndf``, so the read side cannot mask
    a padded column (:func:`mask_absent_rotations`). A rotational slot a
    recorded node's layout lacks, or that its elements do not establish,
    refuses by name.
    """

    def __init__(self, ops: object, archive: object) -> None:
        ndm = live_ndm(ops)
        if ndm < 1:
            raise ValueError(
                "LiveDomainFacts: the live domain declares no ndm "
                f"(getNDM() = {ndm}); build the model with ops.model(...) "
                "before recording it.")
        self._ops = ops
        self._archive = archive
        self._ndm = ndm
        #: The FEM's first ``ndm`` coordinates and its ``{node: row}``,
        #: built on first use, and the nodes already shown to be the FEM's.
        self._coords: "ndarray | None" = None
        self._row: "dict[int, int]" = {}
        self._verified: "set[int]" = set()
        #: ``{fem_eid: ElementRecord}`` of the elements proved live.
        self._paired: "dict[int, Any]" = {}
        #: ``{fem_eid: live eleType}`` of the same elements.
        self._live_class: "dict[int, str]" = {}

    @property
    def ndm(self) -> int:
        """``ops.getNDM()`` of the live domain."""
        return self._ndm

    @property
    def lineage(self) -> object:
        """The archive's lineage (its ``fem_hash`` is the FEMData's)."""
        return self._archive.lineage  # type: ignore[attr-defined]

    def elements(self) -> "tuple[Any, ...]":
        """The element records :meth:`pair_elements` proved: one per paired
        FEM element, its ``tag`` the FEM id and its ``type_token`` the live
        class's. Read from the proof, never from the live domain, so it
        holds after the domain is wiped."""
        return tuple(self._paired[e] for e in sorted(self._paired))

    def live_class(self, fem_eid: int) -> str:
        """The live ``eleType`` of a paired FEM element."""
        return self._live_class[int(fem_eid)]

    def pair_elements(self, element_ids: "Iterable[int]", *, where: str) -> None:
        """Prove that each FEM element of ``element_ids`` is the live
        element of its own id, and pair it.

        The FEMData pairs no FEM id with an OpenSees tag and identity is
        never assumed, so each one is shown by fact: the live domain holds
        an element of that tag, its ``eleNodes`` is the FEM connectivity in
        order, its nodes are the FEM's nodes (the node identity check of
        :meth:`nodes_ndf`), and its ``eleType`` has a bridge token
        (:data:`_TOKEN_FOR_LIVE_CLASS`). A FEM id that fails any of these
        refuses, naming it; nothing is paired then.

        Two FEM elements with the same ordered connectivity (a stacked
        quad pair, a doubled spring) match any ``eleNodes`` check, so a
        deck that swapped their tags (and so their materials) would record
        each one's response under the other's id. A recorded element whose
        ordered connectivity another FEM element shares refuses, naming
        both, as a coincident node does (:meth:`_check_shared_positions`).
        """
        import numpy as np

        from ._internal.typed_records import ElementRecord

        wanted = [e for e in dict.fromkeys(int(v) for v in element_ids)
                  if e not in self._paired]
        if not wanted:
            return
        conn_of: "dict[int, list[int]]" = {}
        ask = np.asarray(wanted, dtype=np.int64)
        by_width: "dict[int, list[tuple[ndarray, ndarray]]]" = {}
        for group in self._archive.fem.elements:  # type: ignore[attr-defined]
            ids = np.asarray(group.ids, dtype=np.int64)
            conn = np.asarray(group.connectivity, dtype=np.int64)
            by_width.setdefault(conn.shape[1], []).append((ids, conn))
            for i in np.flatnonzero(np.isin(ids, ask)):
                conn_of[int(ids[i])] = [int(v) for v in conn[i]]
        absent = [e for e in wanted if e not in conn_of]
        if absent:
            raise ValueError(
                f"{where}: FEM elements {absent[:10]} are not in the FEMData "
                "the archive holds, so no live element can be shown to be "
                "them. Record the FEM elements the deck was built from.")
        twins: "list[str]" = []
        for parts in by_width.values():
            ids = np.concatenate([p[0] for p in parts])
            conn = np.concatenate([p[1] for p in parts])
            _, inv, counts = np.unique(conn, axis=0, return_inverse=True,
                                       return_counts=True)
            inv = inv.ravel()
            for i in np.flatnonzero(np.isin(ids, ask) & (counts[inv] > 1)):
                others = [int(e) for e in ids[inv == inv[i]] if e != ids[i]]
                twins.append(f"FEM element {int(ids[i])} shares nodes "
                             f"{conn_of[int(ids[i])]} with {others[:5]}")
        if twins:
            raise ValueError(
                f"{where}: " + "; ".join(sorted(twins)[:3])
                + (f"; and {len(twins) - 3} more" if len(twins) > 3 else "")
                + ". eleNodes cannot tell those elements apart, so a deck "
                "that swapped their tags would record each one's response "
                "under the other's id. Drop them from the record, or record "
                "them through the bridge's model archive (ops.h5(...)).")
        ops = self._ops
        live_tags = {int(t) for t in ops.getEleTags()}  # type: ignore[attr-defined]
        bad: "list[str]" = []
        live_nodes: "dict[int, list[int]]" = {}
        for e in wanted:
            if e not in live_tags:
                bad.append(f"FEM element {e} has nodes {conn_of[e]}, no live "
                           "element has that tag")
                continue
            live = [int(v) for v in ops.eleNodes(e)]  # type: ignore[attr-defined]
            if live != conn_of[e]:
                bad.append(f"FEM element {e} has nodes {conn_of[e]}, the live "
                           f"element {e} has {live}")
            live_nodes[e] = live
        if bad:
            raise ValueError(
                f"{where}: the live elements are not the FEMData's: "
                + "; ".join(bad[:3])
                + (f"; and {len(bad) - 3} more" if len(bad) > 3 else "")
                + ". The deck did not build these elements under their FEM "
                "ids from this FEMData, so each one's results would be "
                "another element's. Tag each element by its FEM id, with its "
                "FEM connectivity in order.")
        classes = {e: str(ops.eleType(e)) for e in wanted}  # type: ignore[attr-defined]
        unmapped: "dict[str, list[int]]" = {}
        for e, cls in classes.items():
            if cls not in _TOKEN_FOR_LIVE_CLASS:
                unmapped.setdefault(cls, []).append(e)
        if unmapped:
            listed = "; ".join(f"{cls!r}: FEM elements {ids[:10]}"
                               for cls, ids in sorted(unmapped.items()))
            raise ValueError(
                f"{where}: live element classes with no bridge element "
                f"token ({listed}), so their responses cannot be typed. "
                f"Known classes: {sorted(_TOKEN_FOR_LIVE_CLASS)}.")
        self._check_coords(
            list(dict.fromkeys(n for e in wanted for n in live_nodes[e])),
            where=where)
        for e in wanted:
            self._paired[e] = ElementRecord(
                type_token=_TOKEN_FOR_LIVE_CLASS[classes[e]], tag=e, args=(),
                connectivity=tuple(live_nodes[e]), fem_eid=e)
            self._live_class[e] = classes[e]

    def nodes_ndf(self, node_ids: "Iterable[int]", *, where: str) -> "dict[int, int]":
        """``{node: ops.getNDF(node)}`` over ``node_ids``; a node the live
        domain does not hold raises, naming it."""
        ops = self._ops
        built = {int(n) for n in ops.getNodeTags()}  # type: ignore[attr-defined]
        wanted = [int(n) for n in node_ids]
        missing = [n for n in wanted if n not in built]
        if missing:
            raise ValueError(
                f"{where}: nodes {missing[:10]} are not in the live domain, "
                "so the DOFs each records cannot be read from it. Record "
                "the nodes the deck built.")
        self._check_coords(list(dict.fromkeys(wanted)), where=where)
        return {n: _live_int(ops.getNDF(n), f"getNDF({n})")  # type: ignore[attr-defined]
                for n in dict.fromkeys(wanted)}

    def _check_coords(self, nodes: "list[int]", *, where: str) -> None:
        """Refuse, by node id, a recorded node the live domain does not
        show to be the FEM node of that tag.

        A tag match alone does not make the live node the FEM node: a deck
        built from another FEMData shares tags, and a planar deck of a
        column along global Z maps (x, z) to (x, y), so its
        ``displacement_y`` would be the FEM's z motion. So the live
        ``nodeCoord`` must be the FEM node's first ``ndm`` coordinates:
        exactly, or within the node's own tolerance
        (:func:`_identity_tolerances`), which a deck's rounding (6
        significant digits, float32, millimetre decimals) on a mesh coarser
        than the rounding stays inside. Coordinates cannot tell apart nodes
        at one position, so those are checked by element
        (:meth:`_check_shared_positions`).
        """
        import numpy as np

        nodes = [n for n in nodes if n not in self._verified]
        if not nodes:
            return
        if self._coords is None:
            fem_nodes = self._archive.fem.nodes  # type: ignore[attr-defined]
            self._row = {int(n): i for i, n in enumerate(
                np.asarray(fem_nodes.ids, dtype=np.int64))}
            self._coords = np.asarray(
                fem_nodes.coords, dtype=np.float64)[:, :self._ndm]
        coords, row = self._coords, self._row
        absent = [n for n in nodes if n not in row]
        if absent:
            raise ValueError(
                f"{where}: nodes {absent[:10]} are not in the FEMData the "
                "archive holds, so whether the live node is the FEM node "
                "cannot be checked.")
        moved: "list[tuple[int, list[float], list[float]]]" = []
        inexact: "list[tuple[int, ndarray]]" = []
        for n in nodes:
            xyz = self._ops.nodeCoord(n)  # type: ignore[attr-defined]
            live = np.asarray(xyz, dtype=np.float64)
            fem = coords[row[n]]
            if live.shape != fem.shape:
                moved.append((n, live.tolist(), fem.tolist()))
            elif not np.array_equal(live, fem):
                inexact.append((n, live))
        if inexact:
            tols = _identity_tolerances(
                coords, np.array([row[n] for n, _ in inexact]))
            moved.extend(
                (n, live.tolist(), coords[row[n]].tolist())
                for (n, live), tol in zip(inexact, tols.tolist())
                if float(np.linalg.norm(live - coords[row[n]])) > tol)
        if moved:
            moved.sort()
            listed = "; ".join(f"node {n}: live {lv}, FEM {fv}"
                               for n, lv, fv in moved[:3])
            more = len(moved) - min(len(moved), 3)
            raise ValueError(
                f"{where}: the live domain's node coordinates are not the "
                f"FEMData's first {self._ndm} at {listed}"
                + (f"; and {more} more node(s)" if more else "")
                + ". The deck was not built from this FEMData (or maps its "
                "axes), so its DOFs would record another node's or another "
                "axis's motion. Pass the FEMData archive the deck was built "
                "from, with the deck's node coordinates.")
        self._check_shared_positions(nodes, where=where)
        self._verified.update(nodes)

    def _check_shared_positions(self, nodes: "list[int]", *, where: str) -> None:
        """Refuse a recorded node that shares its position with another FEM
        node unless the live elements pin its tag.

        Two nodes at one position (a zero-length spring's ends, an
        ``equalDOF`` or rigid-link pair) match any coordinate check, so a
        deck that swaps their tags would record each one's motion under
        the other. Each FEM element touching such a node must exist live
        under its FEM id with ``eleNodes`` equal to its FEM connectivity,
        in order; a shared-position node no FEM element touches refuses
        by name, since nothing tells its tag from the other node's.
        """
        import numpy as np

        coords, row = self._coords, self._row
        assert coords is not None
        same = _same_position(coords)
        _, shared = _near_points(
            coords, np.array([row[n] for n in nodes]), 2.0 * same, same)
        at = [n for n, s in zip(nodes, shared.tolist()) if s]
        if not at:
            return
        wanted = np.array(at, dtype=np.int64)
        touching: "dict[int, list[tuple[int, list[int]]]]" = {n: [] for n in at}
        for group in self._archive.fem.elements:  # type: ignore[attr-defined]
            conn = np.asarray(group.connectivity, dtype=np.int64)
            hit = np.isin(conn, wanted)
            for i in np.flatnonzero(hit.any(axis=1)):
                nodes_i = [int(v) for v in conn[i]]
                for v in conn[i][hit[i]]:
                    touching[int(v)].append((int(group.ids[i]), nodes_i))
        lone = [n for n in at if not touching[n]]
        if lone:
            raise ValueError(
                f"{where}: nodes {lone[:10]} share their position with "
                "another FEM node, and no FEM element touches them, so "
                "neither coordinates nor elements tell their live tags from "
                "the other node's. Drop them from the record, or record "
                "them through the bridge's model archive (ops.h5(...)).")
        ops = self._ops
        live_tags = {int(t) for t in ops.getEleTags()}  # type: ignore[attr-defined]
        bad: "list[str]" = []
        for n in at:
            for eid, conn_i in touching[n]:
                live = ([int(v) for v in ops.eleNodes(eid)]  # type: ignore[attr-defined]
                        if eid in live_tags else None)
                if live != conn_i:
                    bad.append(
                        f"node {n}: FEM element {eid} has nodes {conn_i}, "
                        + ("no live element has that tag" if live is None
                           else f"the live element {eid} has {live}"))
                    break
        if bad:
            raise ValueError(
                f"{where}: nodes that share their position with another FEM "
                "node are not pinned by their elements: "
                + "; ".join(bad[:3])
                + (f"; and {len(bad) - 3} more" if len(bad) > 3 else "")
                + ". The deck may have swapped the tags of coincident nodes, "
                "which would record each one's motion under the other.")

    def rotation_roles(self, node_ids: "Iterable[int]") -> "RotationRoles":
        """The rotation roles of ``node_ids`` under the live domain's
        element classes, by the family tables the bridge path uses."""
        import numpy as np

        from ._readback import _CLASS_FOR_TOKEN, _roles_from_rows

        ops = self._ops
        by_class: "dict[str, list[int]]" = {}
        for tag in ops.getEleTags():  # type: ignore[attr-defined]
            name = str(ops.eleType(int(tag)))  # type: ignore[attr-defined]
            by_class.setdefault(name, []).extend(
                int(n) for n in ops.eleNodes(int(tag)))  # type: ignore[attr-defined]
        rows = []
        for name, nodes in by_class.items():
            token = _TOKEN_FOR_LIVE_CLASS.get(name)
            cls = None if token is None else _CLASS_FOR_TOKEN[token]
            rows.append((cls, name, np.unique(np.asarray(nodes, dtype=np.int64))))
        return _roles_from_rows(rows, node_ids)

    def node_deck_dofs(
        self, slot_dofs: tuple[int, ...], node_ids: tuple[int, ...], *,
        where: str,
    ) -> tuple[int, ...]:
        """:func:`node_record_deck_dofs` from the live domain's facts."""
        from ._internal.build.recorders import _node_layout

        ndf = self.nodes_ndf(node_ids, where=where)
        names = ("x", "y", "z", "rx", "ry", "rz")
        for d in slot_dofs:
            slot = int(d) - 1
            if not 3 <= slot < 6:
                continue
            lacking = [n for n, k in ndf.items()
                       if slot not in _node_layout(k, self._ndm)]
            if lacking:
                raise ValueError(
                    f"{where}: nodes {lacking[:10]} carry no {names[slot]!r} "
                    f"DOF in the live domain (ndm={self._ndm}, ndf "
                    f"{ndf[lacking[0]]}). A FEMData archive holds no per-node "
                    "ndf to mask their padded column on read; record them in "
                    "a separate record without that component."
                )
        roles: "list[RotationRoles]" = []

        def refusal(node: int) -> "str | None":
            if not roles:
                roles.append(self.rotation_roles(node_ids))
            return roles[0].refusal(int(node))

        return node_record_deck_dofs(
            slot_dofs, node_ids, ndf, self._ndm,
            rotation_refusal=refusal, where=where,
        )


def mask_absent_rotations(
    components: "dict[str, ndarray]",
    node_ids: "ndarray",
    model_h5: "Optional[str | Path]",
    fem: "FEMData",
) -> None:
    """NaN, in place, each rotation column at a node whose DOF layout
    (the emit's own ``_node_layout(ndf, ndm)``) has no such slot, or whose
    elements say that slot holds no rotation.

    One ``recorder Node`` line over a mixed-ndf model writes every
    requested DOF for every node and pads the DOFs a node lacks with
    0.0: a tet node (ndf 3) beside a shell reads rotation 0.0 (#1971).
    The mask keys on the node's layout, never on the file's DOF index
    (``rotation_z`` is DOF 6 in the ``-dof`` list whatever the ndm). An
    ndf without a known layout (4 or 5: a u-p node's pressure DOF) carries
    translations only, never a rotation.

    The layout alone cannot tell a 2-D frame node (ndf 3: ux, uy, rz)
    from a 2-D ``LadrunoUP`` node (ndf 3: ux, uy, p): both lay their third
    DOF in the ``rotation_z`` slot. So a slot the layout keeps is NaN too
    where the model's DOF roles (:func:`model_h5_rotation_roles`) refuse
    a rotation at the node: a u-p element touches it, no beam/shell
    element does, or (an element-less node) the model holds a u-p class.
    A pore pressure never reads as a rotation (#2029).

    ``components`` maps a canonical name to a ``(T, N)`` array over
    ``node_ids``. Values are kept as written when ``model_h5`` is
    ``None`` or the archive predates ``/opensees/nodes_ndf``. Raises
    :class:`ValueError` when a recorded node is missing from that
    group, since its rotation columns may then be padding.
    """
    slots = {c: (node_component_spatial(c) or ("", -1))[1] for c in components}
    if model_h5 is None or max(slots.values(), default=-1) < 3:
        return                      # no model, or no rotation recorded
    import numpy as np

    from ._internal.build.recorders import _node_layout
    from .emitter import h5_reader

    with h5_reader.open(str(model_h5)) as m:
        nodes_ndf = m.nodes_ndf()
        if nodes_ndf is None:
            return                  # an archive older than nodes_ndf
        ndm = h5_reader.read_spatial_ndm(
            m.meta(), m.handle, coords=lambda: np.asarray(fem.nodes.coords),
        )
    missing = sorted({int(n) for n in node_ids} - nodes_ndf.keys())
    if missing:
        raise ValueError(
            f"nodes {missing[:10]} are not in {model_h5}'s "
            "/opensees/nodes_ndf, so whether their rotation columns hold "
            "values or padding is unknown. Pass the model the deck ran."
        )
    layouts = [_node_layout(nodes_ndf[int(n)], ndm) for n in node_ids]
    refused: "ndarray | None" = None
    for comp, slot in slots.items():
        if slot >= 3:               # rx, ry, rz are spatial slots 3..5
            absent = np.array([slot not in lay for lay in layouts], dtype=bool)
            if not absent.all():
                # The layout keeps the slot at some nodes; their elements
                # say whether it holds a rotation, as RotationMask asks.
                if refused is None:
                    roles = model_h5_rotation_roles(
                        model_h5, node_ids,
                        where=f"{model_h5}: the rotation mask",
                    )
                    refused = np.array(
                        [roles.refusal(int(n)) is not None for n in node_ids],
                        dtype=bool)
                absent |= refused
            components[comp] = np.where(absent, np.nan, components[comp])

# =====================================================================
# Element-level keyword resolution (Phase 9 commit 3b)
# =====================================================================
#
# Element-level recorders (``recorder Element ...``) take a *single*
# response token (or multi-token phrase like ``section force``) that
# applies to the whole record. The token depends on which component
# family the record carries:
#
#   - ``"stress_*"`` / shell membrane/curvature etc. → ``"stresses"``
#   - ``"strain_*"`` / shell strain etc.            → ``"strains"``
#   - ``"axial_force"`` etc. (line-stations)        → ``"section"`` + ``"force"``
#   - ``"nodal_resisting_force_*"`` (global frame)  → ``"globalForce"``
#   - ``"nodal_resisting_force_local_*"`` (local)   → ``"localForce"``
#
# The work-conjugate check rejects records that mix families
# (e.g. ``stress_xx`` + ``strain_yy`` in one ``gauss`` record) — those
# can't share a single ``ops.eleResponse`` call.


_ELEMENT_LEVEL_CATEGORIES: frozenset[str] = frozenset({
    "elements", "line_stations", "gauss",
})


def element_record_response_tokens(
    category: str,
    components: tuple[str, ...],
    *,
    record_name: str | None = None,
    sigma_zz_capable: bool | None = None,
) -> Optional[tuple[str, ...]]:
    """Resolve the ``recorder Element`` response phrase for a record.

    Returns the response tokens as a tuple — usually one token
    (``("stresses",)``) but occasionally two (``("section", "force")``)
    that OpenSees parses as a multi-word response phrase. Returns
    ``None`` when no components route through the category's
    topology (caller skips silently).

    ``sigma_zz_capable`` reports whether every element the record targets
    can emit a real out-of-plane σzz (see
    :func:`~apeGmsh.opensees._element_capabilities.element_records_stress_zz`):
    ``True`` promotes a ``stress_zz``-bearing gauss record onto the
    4-component superset response, ``False`` keeps the 3-component one and
    warns, and ``None`` — the caller has no element plan to check against —
    keeps it silently.

    Raises ``ValueError`` if components mix work-conjugate families
    (stress + strain in one gauss record, global + local frame in one
    elements record).
    """
    if category not in _ELEMENT_LEVEL_CATEGORIES:
        raise ValueError(
            f"_recorder_translate.element_record_response_tokens: "
            f"category {category!r} is not element-level."
        )
    topology = {
        "gauss":         None,
        "line_stations": "line_stations",
        "elements":      "nodal_forces",
    }[category]

    # Lazy import to avoid a top-level cycle with the response catalog
    # module.
    from ._response_catalog import gauss_keyword_for_canonical

    keywords: set[str] = set()
    for comp in components:
        kw = gauss_keyword_for_canonical(comp, topology=topology)
        if kw is not None:
            keywords.add(kw)
    if not keywords:
        return None
    if len(keywords) > 1:
        rec_label = f"record {record_name!r}" if record_name else "record"
        raise ValueError(
            f"_recorder_translate.element_record_response_tokens: "
            f"{rec_label} (category={category!r}) mixes work-conjugate "
            f"families ({sorted(keywords)}); split into separate records "
            "(one per ops keyword)."
        )
    keyword = next(iter(keywords))
    if keyword == "stresses" and category == "gauss" and (
        "stress_zz" in components
    ):
        return _stress_zz_tokens(record_name, sigma_zz_capable)
    # The line_stations keyword "section.force" is OpenSees-emitted as
    # two tokens ("section" "force"); same for ``section.deformation``.
    if "." in keyword:
        return tuple(keyword.split("."))
    return (keyword,)


def stress_zz_keyword(sigma_zz_capable: bool | None) -> str:
    """The ops response keyword a ``stress_zz`` gauss record resolves to.

    The mapping half of :func:`_stress_zz_tokens`, without the warning —
    for READ sites (the ``.out`` transcoder) that must land on the same
    catalog family the emit side wrote, but would only be duplicating a
    complaint the emit side already made.
    """
    return STRESS_PLANE_STRAIN_RESPONSE if sigma_zz_capable else "stresses"


def _stress_zz_tokens(
    record_name: str | None, sigma_zz_capable: bool | None,
) -> tuple[str, ...]:
    """Record-level superset promotion for a ``stress_zz`` gauss record.

    ``stress_zz`` routes onto the plain ``stresses`` token, which carries
    only the three in-plane components — so requesting it is otherwise a
    silent no-op.  The fork's ``stressesPlaneStrain`` is a strict superset
    (σxx, σyy, σxy, σzz per Gauss point), so ``stress_xx`` / ``_yy`` /
    ``_xy`` declared in the SAME record ride along unchanged and the
    promotion stays record-level.

    Promotion is gated because ``NDMaterial::getStressZZ()`` returns NaN
    unless the material overrides it: recording the superset on an
    incapable element writes an all-NaN column that poisons every invariant
    built on it, where the un-promoted token at least lets
    :mod:`apeGmsh.results._plane_recovery` reconstruct σzz from ν.
    """
    keyword = stress_zz_keyword(sigma_zz_capable)
    if sigma_zz_capable:
        return (keyword,)
    if sigma_zz_capable is False:
        rec_label = f"record {record_name!r}" if record_name else "record"
        warnings.warn(
            f"{rec_label} requests 'stress_zz', but not every element it "
            f"targets can report the out-of-plane stress: recording "
            f"{STRESS_PLANE_STRAIN_RESPONSE!r} there would write NaN. The "
            f"3-component 'stresses' response is recorded instead and "
            f"stress_zz is reconstructed at read time as nu*(sxx+syy), "
            f"which is exact only while the material is elastic. The "
            f"out-of-plane stress is reported by "
            f"{sorted(SIGMA_ZZ_ELEMENT_CLASSES)} under "
            f"plane_type='PlaneStrain' with one of "
            f"{sorted(SIGMA_ZZ_MATERIAL_CLASSES)}.",
            StressZZNotRecordedWarning, stacklevel=4,
        )
    return (keyword,)
