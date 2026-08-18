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
:func:`~apeGmsh.opensees._internal.build.emit_recorder_spec` flows
through these tables.
"""
from __future__ import annotations

import warnings
from typing import Optional

from ._element_capabilities import (
    SIGMA_ZZ_ELEMENT_CLASSES,
    SIGMA_ZZ_MATERIAL_CLASSES,
    STRESS_PLANE_STRAIN_RESPONSE,
)


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
# Per-axis DOF index tables
# =====================================================================

_TRANS_AXIS_TO_DOF: dict[str, int] = {"x": 1, "y": 2, "z": 3}
_ROT_AXIS_TO_DOF: dict[str, int] = {"x": 4, "y": 5, "z": 6}


# =====================================================================
# Canonical-prefix → (ops_recorder_token, axis_kind) table
# =====================================================================
#
# axis_kind ∈ {"trans", "rot"} selects which DOF table the axis
# suffix is read against. ``displacement_z`` → ("disp", 3); a 2-D
# ``rotation_z`` → ("disp", 6) (z-rotation in ndf=3 lives at DOF 3,
# but OpenSees promotes to the 3-D convention by default; emit-time
# clipping handles ndm/ndf consistency via the canonical-name set
# the bridge already validated).

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


def node_component_to_ops(canonical: str) -> Optional[tuple[str, int]]:
    """Map a canonical node component to ``(ops_recorder_token, dof)``.

    Returns ``None`` if ``canonical`` is not a recognized node-level
    component (caller should fall through to error).

    Examples
    --------
    >>> node_component_to_ops("displacement_x")
    ('disp', 1)
    >>> node_component_to_ops("rotation_z")
    ('disp', 6)
    >>> node_component_to_ops("reaction_force_y")
    ('reaction', 2)
    >>> node_component_to_ops("pore_pressure")
    ('pressure', -1)
    >>> node_component_to_ops("bogus") is None
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
    dof_table = _TRANS_AXIS_TO_DOF if axis_kind == "trans" else _ROT_AXIS_TO_DOF
    if axis not in dof_table:
        return None
    return (ops_token, dof_table[axis])


def group_node_components_by_ops_token(
    components: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    """Group node components by their OpenSees token.

    Components sharing the same ``ops_recorder_token`` collapse into
    one ``recorder Node`` line with a ``-dof`` list of all their DOFs.
    For example, ``("displacement_x", "displacement_y", "reaction_force_z")``
    produces:

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
        vocabulary upstream; this raise is a defensive backstop).
    """
    grouped: dict[str, list[int]] = {}
    for comp in components:
        translated = node_component_to_ops(comp)
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
    if sigma_zz_capable:
        return (STRESS_PLANE_STRAIN_RESPONSE,)
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
    return ("stresses",)
