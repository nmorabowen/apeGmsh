"""
Element specification data for the OpenSees bridge.

Contains ``_ElemSpec`` dataclass, ``_ELEM_REGISTRY`` dictionary mapping
OpenSees element names to their mesh/material requirements, and the
Tcl/Python command renderers.

Extracted from OpenSees.py to reduce file size.
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

    def get_slots(self, ndm: int) -> tuple[str, ...]:
        if ndm == 2 and self.slots_2d is not None:
            return self.slots_2d
        if ndm == 3 and self.slots_3d is not None:
            return self.slots_3d
        return self.slots

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
    ),
    "TenNodeTetrahedron": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({3}),
        gmsh_etypes=frozenset({11}),
        node_reorder={11: (0,1,2,3,4,5,6,7,8,9)},
        slots=("nodes", "matTag", "bodyForce"),
    ),
    "stdBrick": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({3}),
        gmsh_etypes=frozenset({5}),
        node_reorder={5: (0,1,2,3,4,5,6,7)},
        slots=("nodes", "matTag", "bodyForce"),
    ),
    "bbarBrick": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({3}),
        gmsh_etypes=frozenset({5}),
        node_reorder={5: (0,1,2,3,4,5,6,7)},
        slots=("nodes", "matTag", "bodyForce"),
    ),
    "SSPbrick": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({3}),
        gmsh_etypes=frozenset({5}),
        node_reorder={5: (0,1,2,3,4,5,6,7)},
        slots=("nodes", "matTag", "bodyForce"),
    ),

    # ── 2-D solid ──────────────────────────────────────────────────────────
    "quad": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({2}), ndf_ok=frozenset({2}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "thick", "eleType", "matTag"),
    ),
    "tri31": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({2}), ndf_ok=frozenset({2}),
        gmsh_etypes=frozenset({2}),
        node_reorder={2: (0,1,2)},
        slots=("nodes", "thick", "eleType", "matTag"),
    ),
    "SSPquad": _ElemSpec(
        mat_family="nd", needs_transf=False,
        ndm_ok=frozenset({2}), ndf_ok=frozenset({2}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "matTag", "thick", "eleType"),
    ),

    # ── 3-D shell (section-based) ──────────────────────────────────────────
    "ShellMITC3": _ElemSpec(
        mat_family="section", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({6}),
        gmsh_etypes=frozenset({2}),
        node_reorder={2: (0, 1, 2)},
        slots=("nodes", "secTag"),
    ),
    "ShellMITC4": _ElemSpec(
        mat_family="section", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({6}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "secTag"),
    ),
    "ShellDKGQ": _ElemSpec(
        mat_family="section", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({6}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "secTag"),
    ),
    "ASDShellQ4": _ElemSpec(
        mat_family="section", needs_transf=False,
        ndm_ok=frozenset({3}), ndf_ok=frozenset({6}),
        gmsh_etypes=frozenset({3}),
        node_reorder={3: (0,1,2,3)},
        slots=("nodes", "secTag"),
    ),

    # ── 1-D truss (uniaxial material) ──────────────────────────────────────
    "truss": _ElemSpec(
        mat_family="uni", needs_transf=False,
        ndm_ok=frozenset({2, 3}), ndf_ok=frozenset({2, 3, 6}),
        gmsh_etypes=frozenset({1}),
        node_reorder={1: (0, 1)},
        slots=("nodes", "A", "matTag"),
    ),
    "corotTruss": _ElemSpec(
        mat_family="uni", needs_transf=False,
        ndm_ok=frozenset({2, 3}), ndf_ok=frozenset({2, 3, 6}),
        gmsh_etypes=frozenset({1}),
        node_reorder={1: (0, 1)},
        slots=("nodes", "A", "matTag"),
    ),

    # ── 1-D beam (no section material; section props as scalars + geomTransf)
    "elasticBeamColumn": _ElemSpec(
        mat_family="none", needs_transf=True,
        ndm_ok=frozenset({2, 3}), ndf_ok=frozenset({3, 6}),
        gmsh_etypes=frozenset({1}),
        node_reorder={1: (0, 1)},
        slots_2d=("nodes", "A", "E", "Iz", "transfTag"),
        slots_3d=("nodes", "A", "E", "G", "Jx", "Iy", "Iz", "transfTag"),
    ),
    "ElasticTimoshenkoBeam": _ElemSpec(
        mat_family="none", needs_transf=True,
        ndm_ok=frozenset({2, 3}), ndf_ok=frozenset({3, 6}),
        gmsh_etypes=frozenset({1}),
        node_reorder={1: (0, 1)},
        slots_2d=("nodes", "E", "G", "A", "Iz", "Avy", "transfTag"),
        slots_3d=("nodes", "E", "G", "A", "Jx", "Iy", "Iz", "Avy", "Avz", "transfTag"),
    ),
}


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
