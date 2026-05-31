"""
Element capability map for the OpenSees bridge.

Contains ``_ElemSpec`` dataclass, ``_ELEM_REGISTRY`` dictionary mapping
OpenSees element names to their mesh/material requirements, and the
Tcl/Python command renderers.

Lives next to :mod:`apeGmsh.opensees._response_catalog` because the
registry is OpenSees-class metadata that the bridge owns.  The recorder
declaration helper consults it during ``Recorders.resolve(...)`` to
validate per-element capability flags (``has_gauss``, ``has_fibers``,
``has_layers``, ``has_line_stations``, ``has_nodal_forces``).

Relocated from ``apeGmsh.solvers._element_specs`` in Phase 8.3b.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from apeGmsh._types import DimTag  # noqa: F401  — re-exported by OpenSees.py


# ---------------------------------------------------------------------------
# Gmsh element type -> (corner_node_count, topological_dim)
# ---------------------------------------------------------------------------
_ETYPE_INFO: dict[int, tuple[int, int]] = {
    1:  (2, 1),   # 2-node line
    2:  (3, 2),   # 3-node triangle
    3:  (4, 2),   # 4-node quad
    4:  (4, 3),   # 4-node tet
    5:  (8, 3),   # 8-node hex
    6:  (6, 3),   # 6-node prism
    7:  (5, 3),   # 5-node pyramid
    9:  (6, 2),   # 6-node second-order triangle
    11: (10, 3),  # 10-node second-order tet
    16: (8, 2),   # 8-node second-order quad
    17: (20, 3),  # 20-node second-order hex
    15: (1, 0),   # 1-node point
}

# ---------------------------------------------------------------------------
# Slot default values used when the caller omits an extra parameter
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "bodyForce": [0.0, 0.0, 0.0],
    "thick"    : 1.0,
    "eleType"  : "PlaneStress",
    "A"        : 1.0,
    "E"        : 200e9,
    "G"        : 77e9,
    "Jx"       : 1.0,
    "Iy"       : 1.0,
    "Iz"       : 1.0,
    "Avy"      : 1.0,
    "Avz"      : 1.0,
}

# ---------------------------------------------------------------------------
# Element specification
# ---------------------------------------------------------------------------

@dataclass
class _ElemSpec:
    """
    Full description of what a single OpenSees element type requires
    from the mesh, material registries, and model dimensions.
    """
    mat_family  : str                         # "nd" | "uni" | "section" | "none"
    needs_transf: bool                        # must reference a geomTransf
    ndm_ok      : frozenset[int]              # valid ndm values
    ndf_ok      : frozenset[int]              # valid ndf values
    gmsh_etypes : frozenset[int]              # acceptable gmsh element-type codes
    node_reorder: dict[int, tuple[int, ...]]  # etype -> local-index permutation
    slots       : tuple[str, ...]  = ()       # ndm-independent slot order
    slots_2d    : tuple[str, ...] | None = None  # overrides slots for ndm=2
    slots_3d    : tuple[str, ...] | None = None  # overrides slots for ndm=3

    # Recorder capabilities — used by recorder resolution
    # to validate at declaration time. ``has_*`` describes whether the
    # element class *can* expose results at that topology level —
    # specific instances may or may not, but the class supports it.
    has_gauss          : bool = False     # continuum integration points
    has_fibers         : bool = False     # fiber-section beams
    has_layers         : bool = False     # layered shells / through-thickness
    has_line_stations  : bool = False     # beam section forces along length

    # C++ class name — the key used by RESPONSE_CATALOG entries in
    # _element_response.py. When None, the registry key (the user-facing
    # ops_type) IS the C++ class name (e.g. "Tri31", "SSPquad"). Set
    # explicitly when they differ, e.g. ops_type="stdBrick" → "Brick",
    # ops_type="quad" → "FourNodeQuad". Used to populate
    # ResolvedRecorderRecord.element_class_name so the .out transcoder
    # can disambiguate elements that share a flat-size (e.g. Tri31 and
    # SSPquad both produce 3-column 2D stress records).
    cpp_class_name     : str | None = None

    # Per-node DOF FLOOR keyed by ndm, for ADR 0048 ndf inference. This is the
    # element's *minimum* dof/node at a given ndm — distinct from ``ndf_ok``
    # (the *tolerance* set of node-ndf values the element can operate at).
    # Leave ``None`` for elements whose ``ndf_ok`` is single-valued (the floor
    # is that sole value); supply a map ONLY for multi-ndm elements where the
    # set cannot be collapsed by a rule (a 3D beam needs 6, not the set-min 3).
    ndf_required       : dict[int, int] | None = None

    def get_slots(self, ndm: int) -> tuple[str, ...]:
        if ndm == 2 and self.slots_2d is not None:
            return self.slots_2d
        if ndm == 3 and self.slots_3d is not None:
            return self.slots_3d
        return self.slots

    def required_floor(self, ndm: int, local_index: int | None = None) -> int:
        """Minimum dof/node this element requires at *ndm* (ADR 0048).

        Used by the per-node ndf inference: a mesh node's ndf is the ``max`` of
        ``required_floor`` over its incident elements, then validated against
        every incident element's :attr:`ndf_ok` (the shell-on-solid / quad+beam
        ``∩`` gate).

        ``local_index`` is **reserved** for mixed ``u-p`` elements whose corner
        and mid-side nodes carry different ndf (e.g.
        ``Twenty_Eight_Node_BrickUP``: corners 4, mid-sides 3 —
        ``Twenty_Eight_Node_BrickUP.cpp:489``). Every element currently in the
        registry is position-uniform and ignores it; the parameter exists so
        the ``u-p`` family slots in without a signature change (ADR 0048
        position-aware seam).
        """
        if self.ndf_required is not None:
            try:
                return self.ndf_required[ndm]
            except KeyError:
                raise ValueError(
                    f"required_floor: no ndf_required entry for ndm={ndm} "
                    f"(have {sorted(self.ndf_required)})"
                ) from None
        if len(self.ndf_ok) == 1:
            return next(iter(self.ndf_ok))
        raise ValueError(
            f"required_floor: multi-valued ndf_ok={sorted(self.ndf_ok)} needs "
            f"an explicit ndf_required map on this _ElemSpec"
        )

    def supports(self, recorder_category: str) -> bool:
        """True if this element class supports the given recorder category.

        Categories are the ones used by ``ops.recorder.*`` —
        ``"gauss"``, ``"fibers"``, ``"layers"``, ``"line_stations"``.
        ``"nodes"`` and ``"elements"`` (per-element-node forces) are
        always supported and not checked here.
        """
        if recorder_category == "gauss":
            return self.has_gauss
        if recorder_category == "fibers":
            return self.has_fibers
        if recorder_category == "layers":
            return self.has_layers
        if recorder_category == "line_stations":
            return self.has_line_stations
        if recorder_category in ("nodes", "elements"):
            return True
        return False

    @property
    def expected_pg_dim(self) -> int | None:
        """
        Physical-group entity dimension inferred from element topology.
        Returns ``None`` when gmsh_etypes spans more than one topological dim.
        """
        dims = {_ETYPE_INFO[et][1] for et in self.gmsh_etypes if et in _ETYPE_INFO}
        return next(iter(dims)) if len(dims) == 1 else None


# ---------------------------------------------------------------------------
# Element registry
# ---------------------------------------------------------------------------
_ELEM_REGISTRY: dict[str, _ElemSpec] = {

    # ── 3-D solid ──────────────────────────────────────────────────────────
    "FourNodeTetrahedron": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({3}),
        gmsh_etypes=frozenset({4}),
        node_reorder={4: (0, 1, 2, 3)},
        slots=("nodes", "matTag", "bodyForce"),
        has_gauss=True,
    ),
    "TenNodeTetrahedron": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({3}),
        gmsh_etypes=frozenset({11}),
        node_reorder={11: (0,1,2,3,4,5,6,7,8,9)},
        slots=("nodes", "matTag", "bodyForce"),
        has_gauss=True,
    ),
    "stdBrick": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({3}),
        gmsh_etypes=frozenset({5}),
        node_reorder={5: (0,1,2,3,4,5,6,7)},
        slots=("nodes", "matTag", "bodyForce"),
        has_gauss=True,
        cpp_class_name="Brick",
    ),
    "bbarBrick": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({3}),
        gmsh_etypes=frozenset({5}),
        node_reorder={5: (0,1,2,3,4,5,6,7)},
        slots=("nodes", "matTag", "bodyForce"),
        has_gauss=True,
        cpp_class_name="BbarBrick",
    ),
    "SSPbrick": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({3}),
        gmsh_etypes=frozenset({5}),
        node_reorder={5: (0,1,2,3,4,5,6,7)},
        slots=("nodes", "matTag", "bodyForce"),
        has_gauss=True,
    ),

    # ── 2-D solid ──────────────────────────────────────────────────────────
    "quad": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({2}), ndf_ok=frozenset({2}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "thick", "eleType", "matTag"),
        has_gauss=True,
        cpp_class_name="FourNodeQuad",
    ),
    "tri31": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({2}), ndf_ok=frozenset({2}),
        gmsh_etypes=frozenset({2}),
        node_reorder={2: (0,1,2)},
        slots=("nodes", "thick", "eleType", "matTag"),
        has_gauss=True,
        cpp_class_name="Tri31",
    ),
    # Gmsh tri6 (etype 9) node ordering matches the OpenSees SixNodeTri
    # shape-function ordering 1-on-1 (corners 1-3, then mid-edges
    # 1-2, 2-3, 3-1) — see SixNodeTri.cpp:1322 shape functions.
    "tri6n": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({2}), ndf_ok=frozenset({2}),
        gmsh_etypes=frozenset({9}),
        node_reorder={9: (0,1,2,3,4,5)},
        slots=("nodes", "thick", "eleType", "matTag"),
        has_gauss=True,
        cpp_class_name="SixNodeTri",
    ),
    "SSPquad": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({2}), ndf_ok=frozenset({2}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "matTag", "thick", "eleType"),
        has_gauss=True,
    ),

    # ── 3-D shell (section-based) ──────────────────────────────────────────
    # Shells use a section; with a layered section they expose
    # through-thickness layer responses (has_layers).
    "ShellMITC3": _ElemSpec(
        mat_family="section", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({6}),
        gmsh_etypes=frozenset({2}),
        node_reorder={2: (0, 1, 2)},
        slots=("nodes", "secTag"),
        has_gauss=True, has_layers=True,
    ),
    "ShellMITC4": _ElemSpec(
        mat_family="section", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({6}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "secTag"),
        has_gauss=True, has_layers=True,
    ),
    "ShellDKGQ": _ElemSpec(
        mat_family="section", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({6}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "secTag"),
        has_gauss=True, has_layers=True,
    ),
    "ASDShellQ4": _ElemSpec(
        mat_family="section", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({6}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "secTag"),
        has_gauss=True, has_layers=True,
    ),

    # ── 1-D truss (uniaxial material) ──────────────────────────────────────
    # Trusses report force as a single per-element scalar (no GPs,
    # no along-length stations, no fibers).
    "truss": _ElemSpec(
        mat_family="uni", needs_transf=False,
        ndm_ok=frozenset({2, 3}), ndf_ok=frozenset({2, 3, 6}),
        gmsh_etypes=frozenset({1}),
        node_reorder={1: (0, 1)},
        slots=("nodes", "A", "matTag"),
        cpp_class_name="Truss",
        ndf_required={2: 2, 3: 3},  # truss adapts; floor = ndm (uses first ndm dofs)
    ),
    "corotTruss": _ElemSpec(
        mat_family="uni", needs_transf=False,
        ndm_ok=frozenset({2, 3}), ndf_ok=frozenset({2, 3, 6}),
        gmsh_etypes=frozenset({1}),
        node_reorder={1: (0, 1)},
        slots=("nodes", "A", "matTag"),
        cpp_class_name="CorotTruss",
        ndf_required={2: 2, 3: 3},
    ),

    # ── 1-D beam (no section material; section props as scalars + geomTransf)
    "elasticBeamColumn": _ElemSpec(
        mat_family="none", needs_transf=True,
        ndm_ok=frozenset({2, 3}), ndf_ok=frozenset({3, 6}),
        gmsh_etypes=frozenset({1}),
        node_reorder={1: (0, 1)},
        slots_2d=("nodes", "A", "E", "Iz", "transfTag"),
        slots_3d=("nodes", "A", "E", "G", "Jx", "Iy", "Iz", "transfTag"),
        has_line_stations=True,
        ndf_required={2: 3, 3: 6},  # 2D beam: ux,uy,rz; 3D beam: 3 disp + 3 rot
    ),
    "ElasticTimoshenkoBeam": _ElemSpec(
        mat_family="none", needs_transf=True,
        ndm_ok=frozenset({2, 3}), ndf_ok=frozenset({3, 6}),
        gmsh_etypes=frozenset({1}),
        node_reorder={1: (0, 1)},
        slots_2d=("nodes", "E", "G", "A", "Iz", "Avy", "transfTag"),
        slots_3d=("nodes", "E", "G", "A", "Jx", "Iy", "Iz", "Avy", "Avz", "transfTag"),
        has_line_stations=True,
        ndf_required={2: 3, 3: 6},
    ),
}


# ---------------------------------------------------------------------------
# Per-node DOF requirement lookup (shell-on-solid node-sharing guard)
# ---------------------------------------------------------------------------

#: Python ``Element`` subclass name -> :data:`_ELEM_REGISTRY` token, for
#: the handful of classes whose class name differs from the OpenSees
#: token they emit (``ops.element(<token>, ...)``).  Every other element
#: class name equals its token.
_CLASS_TOKEN_ALIASES: dict[str, str] = {
    "FourNodeQuad": "quad",
    "Tri31": "tri31",
    "SixNodeTri": "tri6n",
    "Truss": "truss",
    "CorotTruss": "corotTruss",
}

#: ``ndf_ok`` for ``Element`` subclasses that ship a ``_emit`` but have
#: no :data:`_ELEM_REGISTRY` entry yet.  Without this the shell-on-solid
#: guard would silently skip them.  ``ASDShellT3`` is the load-bearing
#: one — a 3-node ``ndf=6`` shell; missing it would leave an
#: ASDShellT3-on-solid model unguarded (the exact trap ADR 0046
#: targets).  Multi-ndf beams (``forceBeamColumn`` / ``dispBeamColumn``,
#: ``{3, 6}``) are intentionally NOT listed: they intersect both solids
#: and shells, so the guard never flags them anyway.
_EXTRA_CLASS_NDF_OK: dict[str, "frozenset[int]"] = {
    "ASDShellT3": frozenset({6}),
}


def element_class_ndf_ok(class_name: str) -> "frozenset[int] | None":
    """Return the set of per-node ``ndf`` values an ``Element`` subclass
    accepts, or ``None`` when the class is unclassifiable (no
    :data:`_ELEM_REGISTRY` entry and no :data:`_EXTRA_CLASS_NDF_OK`
    fallback).

    Used by the build-time shell-on-solid node-sharing guard
    (:func:`apeGmsh.opensees._internal.build.validate_node_ndf_element_compat`):
    a node shared by two elements whose ``ndf_ok`` sets are disjoint
    cannot be assembled by OpenSees (``FE_Element::setID`` sizes the
    element's equation map to its own ``numDOF`` and truncates when a
    node carries more DOFs than the element expects).

    ``None`` is the conservative "unknown — do not constrain" answer:
    the guard never fires on an element type it cannot classify, so a
    missing entry yields a false negative (silent), never a false
    positive (spurious raise).
    """
    token = _CLASS_TOKEN_ALIASES.get(class_name, class_name)
    spec = _ELEM_REGISTRY.get(token)
    if spec is not None:
        return spec.ndf_ok
    return _EXTRA_CLASS_NDF_OK.get(class_name)


def element_required_floor(
    class_name: str, ndm: int, local_index: "int | None" = None,
) -> "int | None":
    """Return an ``Element`` subclass's minimum dof/node at *ndm* (ADR 0048),
    or ``None`` when the class is unclassifiable (no :data:`_ELEM_REGISTRY`
    entry and no single-valued :data:`_EXTRA_CLASS_NDF_OK` fallback).

    Companion to :func:`element_class_ndf_ok`: that returns the *tolerance*
    set, this returns the *floor* the node-ndf inference maxes over. ``None``
    is the conservative "unknown" answer — the inference fails loud on a node
    whose incident element it cannot classify rather than guessing.

    ``local_index`` is reserved for mixed ``u-p`` elements (see
    :meth:`_ElemSpec.required_floor`); ignored for every element currently
    registered.
    """
    token = _CLASS_TOKEN_ALIASES.get(class_name, class_name)
    spec = _ELEM_REGISTRY.get(token)
    if spec is not None:
        return spec.required_floor(ndm, local_index)
    extra = _EXTRA_CLASS_NDF_OK.get(class_name)
    if extra is not None and len(extra) == 1:
        return next(iter(extra))
    return None


# ---------------------------------------------------------------------------
# Element command renderers
# ---------------------------------------------------------------------------

def _render_tcl(
    ops_id    : int,
    ops_type  : str,
    slots     : tuple[str, ...],
    nodes     : tuple[int, ...],
    mat_tag   : int | None,
    sec_tag   : int | None,
    transf_tag: int | None,
    extra     : dict,
    pg_name   : str,
) -> str:
    parts = [f"element {ops_type} {ops_id}"]
    for slot in slots:
        if slot == "nodes":
            parts.append(" ".join(str(n) for n in nodes))
        elif slot == "matTag":
            parts.append(str(mat_tag))
        elif slot == "secTag":
            parts.append(str(sec_tag))
        elif slot == "transfTag":
            parts.append(str(transf_tag))
        elif slot == "bodyForce":
            bf = extra.get("bodyForce", _DEFAULTS["bodyForce"])
            parts.append(" ".join(str(v) for v in bf))
        else:
            val = extra.get(slot, _DEFAULTS.get(slot, ""))
            parts.append(
                " ".join(str(v) for v in val)
                if isinstance(val, (list, tuple))
                else str(val)
            )
    return "  ".join(parts) + f"  ;# {pg_name}"


def _render_py(
    ops_id    : int,
    ops_type  : str,
    slots     : tuple[str, ...],
    nodes     : tuple[int, ...],
    mat_tag   : int | None,
    sec_tag   : int | None,
    transf_tag: int | None,
    extra     : dict,
    pg_name   : str,
) -> str:
    args: list[str] = [repr(ops_type), str(ops_id)]
    for slot in slots:
        if slot == "nodes":
            args.extend(str(n) for n in nodes)
        elif slot == "matTag":
            args.append(str(mat_tag))
        elif slot == "secTag":
            args.append(str(sec_tag))
        elif slot == "transfTag":
            args.append(str(transf_tag))
        elif slot == "bodyForce":
            bf = extra.get("bodyForce", _DEFAULTS["bodyForce"])
            args.extend(repr(v) for v in bf)
        else:
            val = extra.get(slot, _DEFAULTS.get(slot, ""))
            if isinstance(val, (list, tuple)):
                args.extend(repr(float(v)) if isinstance(v, np.floating) else repr(v) for v in val)
            else:
                args.append(repr(float(val)) if isinstance(val, np.floating) else repr(val))
    return f"ops.element({', '.join(args)})  # {pg_name}"


# ---------------------------------------------------------------------------
# Transf-arg-tail slot lookup (shared single source of truth, ADR 0018 INV-3)
# ---------------------------------------------------------------------------

#: Beam-column element types whose geomTransf tag is the first positional
#: arg after connectivity (``args`` tail index 0).  These are absent from
#: :data:`_ELEM_REGISTRY` (which only carries the scalar-property beam
#: forms); the position is a stable OpenSees convention:
#: ``element forceBeamColumn $ele $iN $jN $transfTag $integrationTag``.
_FORCE_DISP_BEAMS: frozenset = frozenset({"forceBeamColumn", "dispBeamColumn"})


def _transf_arg_tail_index(
    type_token: str, ndm: int, registry: dict,
) -> "int | None":
    """Return the ``args``-tail index of the geomTransf tag, or ``None``.

    ``args`` is the element's positional list *after* the connectivity
    prefix is dropped (h5-schema.md ``/opensees/element_meta``).  In the
    vocabulary the connectivity prefix is the leading ``"nodes"`` slot,
    so the tail index is ``slots.index("transfTag") - 1``.  ``None``
    means "this element type carries no geomTransf" (solids, trusses,
    shells) — the caller skips it.

    Single source of truth for the writer/reader join (ADR 0018 INV-3).
    Both :class:`H5Emitter.add_oriented_elements` (writer) and
    :meth:`h5_reader.H5Model.element_local_axes_vecxz` (reader) consult
    this function so the transf tag is always written at the slot the
    reader will read.
    """
    if type_token in _FORCE_DISP_BEAMS:
        return 0
    spec = registry.get(type_token)
    if spec is None:
        return None
    if ndm == 2:
        slots = getattr(spec, "slots_2d", None)
    elif ndm == 3:
        slots = getattr(spec, "slots_3d", None)
    else:
        slots = None
    if slots is None:
        slots = (
            getattr(spec, "slots_3d", None)
            or getattr(spec, "slots_2d", None)
            or getattr(spec, "slots", None)
        )
    if not slots or "transfTag" not in slots:
        return None
    return slots.index("transfTag") - 1


def known_beam_type_tokens(ndm: int) -> tuple[str, ...]:
    """Return the sorted tuple of beam-type tokens with a transf slot at ``ndm``.

    Used at inject time (ADR 0018 INV-7) to validate a user-supplied
    ``ele_type`` and produce a helpful raise message listing the valid
    options.
    """
    tokens = sorted(_FORCE_DISP_BEAMS)
    for token, spec in _ELEM_REGISTRY.items():
        if not getattr(spec, "needs_transf", False):
            continue
        if _transf_arg_tail_index(token, ndm, _ELEM_REGISTRY) is not None:
            tokens.append(token)
    return tuple(sorted(set(tokens)))
