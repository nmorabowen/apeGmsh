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
from typing import TYPE_CHECKING, Optional

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
    the table the ``recorder.declare`` emit and ``.read`` use."""
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
    roles = model_h5_rotation_roles(
        model_h5, node_ids, where=f"{model_h5}: the rotation mask",
    )
    refused = np.array(
        [roles.refusal(int(n)) is not None for n in node_ids], dtype=bool)
    for comp, slot in slots.items():
        if slot >= 3:               # rx, ry, rz are spatial slots 3..5
            absent = refused | np.array(
                [slot not in lay for lay in layouts], dtype=bool)
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
