"""Element response metadata catalog (Phase 11a).

Three sites in the Results module need to decode element-level recorder
output (gauss / fibers / layers / line stations / per-element-node
forces) into per-component arrays:

- ``results.readers._mpco.MPCOReader`` — reads MPCO ``ON_ELEMENTS``
  buckets. MPCO carries a ``META`` self-description for column layout,
  but does **not** carry natural Gauss-point coordinates for standard
  integration rules; we look those up here.
- ``results.transcoders._recorder.RecorderTranscoder`` — parses
  OpenSees ``.out`` files. There is no META; the catalog is the only
  source of truth for column ordering.
- ``results.capture._domain.DomainCapture`` — calls
  ``ops.eleResponse(eid, *tokens)`` and gets a flat list of doubles.
  Same: catalog is the only authority on what each entry means.

This module is the single shared keystone. v1 covers the few
``_ELEM_REGISTRY`` classes whose response shape is fixed by their
class + integration rule alone; richer cases (custom rules,
heterogeneous fiber sections, layered shells) are deferred.

Out of scope for v1 (raise / log on encounter)
----------------------------------------------
- ``CustomIntegrationRule`` (1000) — force-based beams with user IPs;
  these store their own ``GP_X`` in MPCO.
- ``CUSTOM_INTEGRATION_RULE_DIMENSION == 2`` (MVLEM family).
- ``state_variable_<n>`` material outputs.
- Multiple ``headerIdx`` buckets in MPCO (heterogeneous fiber sections,
  variable-length section responses) — first bucket only in v1.

References
----------
- Source-of-truth for integration-rule enum values and per-geometry
  parent domains: ``mpco-recorder/references/integration-rules-and-gauss.md``.
- Element class tags: ``OpenSees/SRC/classTags.h``.
- Gauss-point natural coordinates verified against
  ``OpenSees/SRC/element/tetrahedron/{FourNodeTetrahedron,TenNodeTetrahedron}.cpp``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from ..results._vocabulary import (
    SHELL_GENERALIZED_STRAINS,
    SHELL_STRESS_RESULTANTS,
    STRAIN,
    STRAIN_2D,
    STRESS,
    STRESS_2D,
)


# =====================================================================
# Integration-rule enum (subset; values mirror mpco::ElementIntegrationRuleType)
# =====================================================================

class IntRule:
    """Integer codes that appear in MPCO bracket keys ``[<rule>:<cust>:<hdr>]``.

    Values match ``mpco::ElementIntegrationRuleType`` in
    ``MPCORecorder.cpp`` line 1313.
    """
    NoIntegrationRule = 0

    Line_GL_1 = 1
    Line_GL_2 = 2
    Line_GL_3 = 3

    Triangle_GL_1 = 100
    Triangle_GL_2 = 101
    Triangle_GL_2B = 102
    Triangle_GL_2C = 103

    Quad_GL_1 = 200
    Quad_GL_2 = 201
    Quad_GL_3 = 202

    Tet_GL_1 = 300
    Tet_GL_2 = 301

    Hex_GL_1 = 400
    Hex_GL_2 = 401
    Hex_GL_3 = 402

    Custom = 1000


# =====================================================================
# Element class tags (subset; values mirror SRC/classTags.h)
# =====================================================================
#
# We hard-code the tags we use rather than depend on the C++ header,
# because (a) the catalog needs Python-level constants for keys and
# (b) the values are stable parts of the OpenSees ABI.

ELE_TAG_FourNodeTetrahedron = 179
ELE_TAG_TenNodeTetrahedron = 256
ELE_TAG_Brick = 56
ELE_TAG_BbarBrick = 57
ELE_TAG_SSPbrick = 121
ELE_TAG_FourNodeQuad = 31
ELE_TAG_Tri31 = 33
ELE_TAG_SSPquad = 119
ELE_TAG_Twenty_Node_Brick = 49
ELE_TAG_EightNodeQuad = 208
# Shells
ELE_TAG_ShellMITC4 = 53
ELE_TAG_ShellMITC9 = 54
ELE_TAG_ShellDKGQ = 156
ELE_TAG_ShellNLDKGQ = 157
ELE_TAG_ShellDKGT = 167
ELE_TAG_ShellNLDKGT = 168
ELE_TAG_ASDShellQ4 = 203
ELE_TAG_ASDShellT3 = 204
# Line elements — trusses
ELE_TAG_Truss = 12
ELE_TAG_TrussSection = 13
ELE_TAG_CorotTruss = 14
ELE_TAG_CorotTrussSection = 15
ELE_TAG_Truss2 = 138
ELE_TAG_CorotTruss2 = 139
ELE_TAG_InertiaTruss = 218
# Line elements — beam-columns (Phase 11b)
# Force-/disp-based families with per-instance integration rules
# (custom rule, line-stations topology). Tags from
# ``OpenSees/SRC/classTags.h`` lines 651–810.
ELE_TAG_ElasticBeam2d = 3
ELE_TAG_ModElasticBeam2d = 4
ELE_TAG_ElasticBeam3d = 5
ELE_TAG_DispBeamColumn2d = 62
ELE_TAG_DispBeamColumn3d = 64
ELE_TAG_ForceBeamColumn2d = 73
ELE_TAG_ForceBeamColumnWarping2d = 731
ELE_TAG_ForceBeamColumn3d = 74
ELE_TAG_ElasticForceBeamColumn2d = 75
ELE_TAG_ElasticForceBeamColumn3d = 76
ELE_TAG_ForceBeamColumnCBDI2d = 77
ELE_TAG_ElasticTimoshenkoBeam2d = 145
ELE_TAG_ElasticTimoshenkoBeam3d = 146


# =====================================================================
# ResponseLayout — describes one (class, rule, token) entry
# =====================================================================

@dataclass(frozen=True)
class ResponseLayout:
    """How a single ``ops.eleResponse(eid, token)`` flat array unflattens.

    The flat array layout is ``[gp0_c0, gp0_c1, ..., gp0_cK,
    gp1_c0, ..., gp(G-1)_c(K-1)]`` — Gauss point varies slowest,
    component varies fastest. This matches both OpenSees's element
    ``setResponse`` ordering and MPCO's META block layout.

    Parameters
    ----------
    n_gauss_points
        Number of integration points the response covers.
    natural_coords
        ``(n_gauss_points, dim)`` parent-domain coordinates of the GPs.
        The parent domain depends on the geometry — see ``coord_system``.
    coord_system
        Tag describing the parent domain so the reader can interpret
        ``natural_coords`` correctly:

        - ``"isoparametric"`` — ``[-1, +1]^dim`` (line, quad, hex).
        - ``"barycentric_tet"`` — volume coordinates ``(L1, L2, L3)``
          with the implicit fourth ``L4 = 1 - L1 - L2 - L3`` (tet).
        - ``"barycentric_tri"`` — area coordinates ``(L1, L2)`` with
          ``L3 = 1 - L1 - L2`` (triangle).
    n_components_per_gp
        Number of scalar components emitted per Gauss point.
    component_layout
        Canonical apeGmsh names (in flat order) for the
        ``n_components_per_gp`` columns at each GP.
    class_tag
        OpenSees ``ELE_TAG_*`` integer for this element class. Lets
        callers stamp the class tag onto the native HDF5 group.
    flat_size_per_element
        Convenience: ``n_gauss_points * n_components_per_gp``.
    """

    n_gauss_points: int
    natural_coords: ndarray
    coord_system: str
    n_components_per_gp: int
    component_layout: tuple[str, ...]
    class_tag: int

    @property
    def flat_size_per_element(self) -> int:
        return self.n_gauss_points * self.n_components_per_gp

    def __post_init__(self) -> None:
        if self.natural_coords.shape[0] != self.n_gauss_points:
            raise ValueError(
                f"natural_coords first dim {self.natural_coords.shape[0]} "
                f"does not match n_gauss_points={self.n_gauss_points}."
            )
        if len(self.component_layout) != self.n_components_per_gp:
            raise ValueError(
                f"component_layout has {len(self.component_layout)} names "
                f"but n_components_per_gp={self.n_components_per_gp}."
            )


# =====================================================================
# CustomRuleLayout — beam-columns with per-instance integration
# =====================================================================
#
# Force-/disp-based beam-column elements let the user attach any
# beamIntegration scheme (Lobatto, Legendre, Radau, NewtonCotes,
# Simpson, HingeMidpoint, HingeRadau, FixedLocation, UserDefined, …).
# Both ``n_IP`` and IP locations are per-element metadata: they live
# on the assigned ``beamIntegration`` object, not on the C++ class.
# MPCO classifies all such rules as ``CustomIntegrationRule = 1000``
# and stores per-element ``GP_X`` natural coordinates as an attribute
# on the connectivity dataset.
#
# The catalog can therefore declare only the *structural* identity of
# the response (class tag, parent domain), not its concrete shape. The
# per-element shape — ``n_IP``, ``natural_coords``, and the
# ``component_layout`` (which depends on the assigned section's
# ``getType()`` codes) — is filled in by ``resolve_layout_from_gp_x``
# at read/capture time, producing a concrete ``ResponseLayout`` that
# the existing ``unflatten`` keystone can consume directly.

@dataclass(frozen=True)
class CustomRuleLayout:
    """Structural layout for an element class with per-instance integration.

    Used by ``CUSTOM_RULE_CATALOG`` entries for force- and
    displacement-based beam-columns. Concrete per-element layout —
    ``n_IP``, GP natural coordinates, and component names ordered to
    match the section's response codes — is built at runtime via
    :func:`resolve_layout_from_gp_x` from the per-element ``GP_X``
    array (MPCO) or ``ops.eleResponse(eid, "integrationPoints")``
    output (DomainCapture) plus the section's response code vector.

    Parameters
    ----------
    class_tag
        OpenSees ``ELE_TAG_*`` integer for this element class.
    coord_system
        Tag describing the parent domain so the resolver can stamp
        the right value onto the produced ``ResponseLayout``. For
        line beam-columns this is always ``"isoparametric_1d"``
        (parent ξ ∈ [-1, +1]). MVLEM-family 2-D rules (out of scope
        for v1) would use a 2-D variant.
    """

    class_tag: int
    coord_system: str


# =====================================================================
# NodalForceLayout — closed-form elastic beams (no integration points)
# =====================================================================
#
# ElasticBeam{2d,3d}, ElasticTimoshenkoBeam{2d,3d}, ModElasticBeam2d
# and similar closed-form line elements have no per-IP state to
# probe. ``ops.eleResponse(eid, "globalForce")`` returns a single
# flat per-element-node force vector packed node-slowest /
# component-fastest::
#
#     flat[t, e, n * K + k] = component k at element-node n
#
# matching the same packing convention as the GP/component layout
# in ``ResponseLayout`` (substitute "node" for "GP"). The layout is
# fixed by element class + frame ("global" or "local") and does not
# depend on the assigned section, so we can fully declare it in the
# catalog without a runtime resolver.

@dataclass(frozen=True)
class NodalForceLayout:
    """Per-element-node force/moment layout for closed-form line elements.

    Parameters
    ----------
    n_nodes_per_element
        Number of element-end nodes carrying the force vector
        (always 2 for elastic beams in v1).
    n_components_per_node
        Force/moment components emitted per element-node (3 in 2D
        ndf=3, 6 in 3D ndf=6).
    component_layout
        Canonical apeGmsh names (in flat per-node order) for the
        ``n_components_per_node`` columns at each element-node.
        Ordered to match OpenSees's per-node DOF order.
    class_tag
        OpenSees ``ELE_TAG_*`` integer for this element class.
    frame
        ``"global"`` or ``"local"``. Distinguishes the two recorder
        tokens (``globalForce`` / ``localForce``) which return the
        same data in different reference frames.
    """

    n_nodes_per_element: int
    n_components_per_node: int
    component_layout: tuple[str, ...]
    class_tag: int
    frame: str

    @property
    def flat_size_per_element(self) -> int:
        return self.n_nodes_per_element * self.n_components_per_node

    def __post_init__(self) -> None:
        if len(self.component_layout) != self.n_components_per_node:
            raise ValueError(
                f"component_layout has {len(self.component_layout)} names "
                f"but n_components_per_node={self.n_components_per_node}."
            )
        if self.frame not in ("global", "local"):
            raise ValueError(
                f"frame must be 'global' or 'local'; got {self.frame!r}."
            )


# =====================================================================
# Gauss-point coordinate tables
# =====================================================================

# Tet_GL_1 — single point at the volume centroid.
# FourNodeTetrahedron.cpp:226 — sg = {0.25}, weight 1/6.
_TET_GL_1_COORDS: ndarray = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)

# Tet_GL_2 — 4-point Hammer-Stroud rule.
# TenNodeTetrahedron.cpp:223–226 — sg = {alpha, beta, beta, beta}
# with alpha = (5 + 3*sqrt(5))/20, beta = (5 - sqrt(5))/20, all weights 1/24.
# Loop body picks (sg[k], sg[|1-k|], sg[|2-k|]) for k = 0..3.
_TET_GL_2_ALPHA = (5.0 + 3.0 * math.sqrt(5.0)) / 20.0
_TET_GL_2_BETA = (5.0 - math.sqrt(5.0)) / 20.0
_TET_GL_2_COORDS: ndarray = np.array([
    # k=0: (alpha, beta, beta)
    [_TET_GL_2_ALPHA, _TET_GL_2_BETA, _TET_GL_2_BETA],
    # k=1: (beta, alpha, beta)   — sg[|1-1|]=sg[0]=alpha, sg[|2-1|]=sg[1]=beta
    [_TET_GL_2_BETA, _TET_GL_2_ALPHA, _TET_GL_2_BETA],
    # k=2: (beta, beta, alpha)   — sg[|1-2|]=sg[1]=beta, sg[|2-2|]=sg[0]=alpha
    [_TET_GL_2_BETA, _TET_GL_2_BETA, _TET_GL_2_ALPHA],
    # k=3: (beta, beta, beta)    — sg[|1-3|]=sg[2]=beta, sg[|2-3|]=sg[1]=beta
    [_TET_GL_2_BETA, _TET_GL_2_BETA, _TET_GL_2_BETA],
], dtype=np.float64)

# Hex_GL_1 — single point at the parent-cube centroid.
# SSPbrick.cpp uses one Gauss point at (0, 0, 0) in [-1, +1]³.
_HEX_GL_1_COORDS: ndarray = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

# Hex_GL_2 — tensor product of 1-D 2-point Gauss-Legendre on [-1, +1]³.
# Brick.cpp:124–125 — sg = {-1/√3, +1/√3}, weights all 1.0.
# Triple loop ``for i: for j: for k: gaussPoint = (sg[i], sg[j], sg[k])``
# (Brick.cpp:540–542) advances ζ fastest, then η, then ξ. So GP index
# i*4 + j*2 + k.
_HEX_GL_2_M = -1.0 / math.sqrt(3.0)
_HEX_GL_2_P = +1.0 / math.sqrt(3.0)
_HEX_GL_2_COORDS: ndarray = np.array([
    [_HEX_GL_2_M, _HEX_GL_2_M, _HEX_GL_2_M],   # 0  (-, -, -)
    [_HEX_GL_2_M, _HEX_GL_2_M, _HEX_GL_2_P],   # 1  (-, -, +)
    [_HEX_GL_2_M, _HEX_GL_2_P, _HEX_GL_2_M],   # 2  (-, +, -)
    [_HEX_GL_2_M, _HEX_GL_2_P, _HEX_GL_2_P],   # 3  (-, +, +)
    [_HEX_GL_2_P, _HEX_GL_2_M, _HEX_GL_2_M],   # 4  (+, -, -)
    [_HEX_GL_2_P, _HEX_GL_2_M, _HEX_GL_2_P],   # 5  (+, -, +)
    [_HEX_GL_2_P, _HEX_GL_2_P, _HEX_GL_2_M],   # 6  (+, +, -)
    [_HEX_GL_2_P, _HEX_GL_2_P, _HEX_GL_2_P],   # 7  (+, +, +)
], dtype=np.float64)

# Quad_GL_1 — single point at the parent-square centroid.
# SSPquad and similar single-IP plane elements use this rule.
_QUAD_GL_1_COORDS: ndarray = np.array([[0.0, 0.0]], dtype=np.float64)

# Quad_GL_2 — 4 GPs at (±1/√3, ±1/√3) in [-1, +1]².
# FourNodeQuad.cpp:298–305 stores the points in counter-clockwise
# order around the parent square (NOT the i-slowest tensor product
# Brick uses): (−,−), (+,−), (+,+), (−,+).
_QUAD_GL_2_M = -1.0 / math.sqrt(3.0)
_QUAD_GL_2_P = +1.0 / math.sqrt(3.0)
_QUAD_GL_2_COORDS: ndarray = np.array([
    [_QUAD_GL_2_M, _QUAD_GL_2_M],   # 0  (-, -)
    [_QUAD_GL_2_P, _QUAD_GL_2_M],   # 1  (+, -)
    [_QUAD_GL_2_P, _QUAD_GL_2_P],   # 2  (+, +)
    [_QUAD_GL_2_M, _QUAD_GL_2_P],   # 3  (-, +)
], dtype=np.float64)

# Triangle_GL_1 — single point at the area-coord centroid (1/3, 1/3).
# The third area coord is implicit: L3 = 1 - L1 - L2 = 1/3.
_TRI_GL_1_COORDS: ndarray = np.array(
    [[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64,
)

# Hex_GL_3 — 27 GPs in OpenSees Twenty_Node_Brick's corner-edge-face-
# centroid order (NOT a tensor product). The element shares its
# 27-GP rule with the 27-node hex via ``shp3dv.cpp::brcshl``
# (lines 251–277), where parent-cube node positions are stored as
# RA/SA/TA in {-0.5, 0.0, +0.5} and scaled by ``G = 2*sqrt(3/5)`` to
# place each GP at the same parent-cube point as the corresponding
# node would be in a 27-node element. The order is:
#   L=0..7   : 8 corners        (all sign combinations of ±√(3/5))
#   L=8..19  : 12 edge midpoints (one coord 0, two are ±√(3/5))
#   L=20..25 : 6 face centers   (two coords 0, one is ±√(3/5))
#   L=26     : 1 body centroid  (all zero)
_HEX_GL_3_RA: tuple[float, ...] = (
    -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5,
     0.0,  0.5,  0.0, -0.5,  0.0,  0.5,  0.0, -0.5,
    -0.5,  0.5,  0.5, -0.5,  0.5,  0.0,  0.0, -0.5,
     0.0,  0.0,  0.0,
)
_HEX_GL_3_SA: tuple[float, ...] = (
    -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5,
    -0.5,  0.0,  0.5,  0.0, -0.5,  0.0,  0.5,  0.0,
    -0.5, -0.5,  0.5,  0.5,  0.0,  0.5,  0.0,  0.0,
    -0.5,  0.0,  0.0,
)
_HEX_GL_3_TA: tuple[float, ...] = (
    -0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5,
    -0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5,
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.5,  0.0,
     0.0, -0.5,  0.0,
)
_HEX_GL_3_G = 2.0 * math.sqrt(3.0 / 5.0)
_HEX_GL_3_COORDS: ndarray = _HEX_GL_3_G * np.array(
    list(zip(_HEX_GL_3_RA, _HEX_GL_3_SA, _HEX_GL_3_TA)),
    dtype=np.float64,
)

# Quad_GL_3 (EightNodeQuad order) — 9 GPs at all combinations of
# (-√(3/5), 0, +√(3/5))² in EightNodeQuad's corner-edge-centroid
# order: 4 corners CCW, 4 edge midpoints CCW starting at the bottom
# edge, then the centroid. EightNodeQuad.cpp:128–144 stores the
# literal ±0.7745966 coordinates explicitly.
_QUAD_GL_3_S = math.sqrt(3.0 / 5.0)
_QUAD_GL_3_COORDS_QUAD8: ndarray = np.array([
    [-_QUAD_GL_3_S, -_QUAD_GL_3_S],   # 0  corner (-, -)
    [+_QUAD_GL_3_S, -_QUAD_GL_3_S],   # 1  corner (+, -)
    [+_QUAD_GL_3_S, +_QUAD_GL_3_S],   # 2  corner (+, +)
    [-_QUAD_GL_3_S, +_QUAD_GL_3_S],   # 3  corner (-, +)
    [          0.0, -_QUAD_GL_3_S],   # 4  bottom edge (0, -)
    [+_QUAD_GL_3_S,           0.0],   # 5  right edge  (+, 0)
    [          0.0, +_QUAD_GL_3_S],   # 6  top edge    (0, +)
    [-_QUAD_GL_3_S,           0.0],   # 7  left edge   (-, 0)
    [          0.0,           0.0],   # 8  centroid
], dtype=np.float64)

# Quad_GL_3 (ShellMITC9 order) — same 9 GP positions, different walk
# order: alternating corner-edge CCW around the parent square, with
# the centroid last. ShellMITC9.cpp:107–124 sg/tg arrays.
_QUAD_GL_3_COORDS_MITC9: ndarray = np.array([
    [-_QUAD_GL_3_S, -_QUAD_GL_3_S],   # 0  corner SW
    [          0.0, -_QUAD_GL_3_S],   # 1  edge   S
    [+_QUAD_GL_3_S, -_QUAD_GL_3_S],   # 2  corner SE
    [+_QUAD_GL_3_S,           0.0],   # 3  edge   E
    [+_QUAD_GL_3_S, +_QUAD_GL_3_S],   # 4  corner NE
    [          0.0, +_QUAD_GL_3_S],   # 5  edge   N
    [-_QUAD_GL_3_S, +_QUAD_GL_3_S],   # 6  corner NW
    [-_QUAD_GL_3_S,           0.0],   # 7  edge   W
    [          0.0,           0.0],   # 8  centroid
], dtype=np.float64)

# Triangle_GL_2B — 3 GPs at the area-coord midpoints of the triangle
# edges. Used by ASDShellT3 (ASDShellT3.cpp:148–150 mid-edge XI/ETA).
# The (XI, ETA) values from the source map directly to (L1, L2)
# pairs with L3 = 1 - L1 - L2 implicit.
_TRI_GL_2B_COORDS: ndarray = np.array([
    [0.5, 0.5],   # edge between vertices 1-2 (L3 = 0)
    [0.0, 0.5],   # edge between vertices 1-3 (L1 = 0)
    [0.5, 0.0],   # edge between vertices 2-3 (L2 = 0)
], dtype=np.float64)

# Triangle_GL_2C — 4-point quadrature used by ShellDKGT / ShellNLDKGT
# (ShellDKGT.cpp:238–250 sg/tg arrays). One centroid + three
# barycentric points each at (1/5, 1/5, 3/5)-type permutations.
_TRI_GL_2C_COORDS: ndarray = np.array([
    [1.0 / 3.0, 1.0 / 3.0],   # 0  centroid
    [1.0 / 5.0, 3.0 / 5.0],   # 1  near vertex 2
    [3.0 / 5.0, 1.0 / 5.0],   # 2  near vertex 1
    [1.0 / 5.0, 1.0 / 5.0],   # 3  near vertex 3
], dtype=np.float64)

# Line_GL_1 — single point at the parent-line midpoint. Used by
# trusses, zero-length elements, and simple bearings (anything in
# the MPCO Line_2N + Line_GaussLegendre_1 bucket). One coordinate
# in 1-D parametric space ξ ∈ [-1, +1].
_LINE_GL_1_COORDS: ndarray = np.array([[0.0]], dtype=np.float64)


# =====================================================================
# RESPONSE_CATALOG — the master table
# =====================================================================
#
# Key: (class_name, int_rule, response_token) where:
#   - class_name is the C++ class as it appears in MPCO bracket keys
#     (Element::getClassType()), NOT necessarily the Tcl element name.
#   - int_rule is the IntRule enum integer.
#   - response_token is the broad OpenSees response keyword
#     (e.g. "stress", "strain", "globalForce").
#
# The "stress" and "strain" tokens emit identical layouts (both are
# rank-2 symmetric tensors at every GP), so they share constructors.

def _continuum_layout(
    *,
    n_gp: int,
    natural_coords: ndarray,
    coord_system: str,
    component_names: tuple[str, ...],
    class_tag: int,
) -> ResponseLayout:
    return ResponseLayout(
        n_gauss_points=n_gp,
        natural_coords=natural_coords,
        coord_system=coord_system,
        n_components_per_gp=len(component_names),
        component_layout=component_names,
        class_tag=class_tag,
    )


RESPONSE_CATALOG: dict[tuple[str, int, str], ResponseLayout] = {
    # ── FourNodeTetrahedron (1 GP) ────────────────────────────────────
    ("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress"): _continuum_layout(
        n_gp=1, natural_coords=_TET_GL_1_COORDS,
        coord_system="barycentric_tet",
        component_names=STRESS,
        class_tag=ELE_TAG_FourNodeTetrahedron,
    ),
    ("FourNodeTetrahedron", IntRule.Tet_GL_1, "strain"): _continuum_layout(
        n_gp=1, natural_coords=_TET_GL_1_COORDS,
        coord_system="barycentric_tet",
        component_names=STRAIN,
        class_tag=ELE_TAG_FourNodeTetrahedron,
    ),

    # ── TenNodeTetrahedron (4 GPs) ────────────────────────────────────
    # NOTE: This catalog entry is correct for MPCO reads (which probe
    # materials directly). It is **not** correct for ``ops.eleResponse``
    # in builds before the upstream fix at
    # ``SRC/element/tetrahedron/TenNodeTetrahedron.cpp:1845``: that file
    # declares ``static Vector stresses(6)`` but writes 24 floats,
    # causing heap corruption and returning only 6 values. Until that
    # is patched to ``Vector(24)``, DomainCapture and the .out
    # transcoder will see broken stress on TenNodeTet.
    ("TenNodeTetrahedron", IntRule.Tet_GL_2, "stress"): _continuum_layout(
        n_gp=4, natural_coords=_TET_GL_2_COORDS,
        coord_system="barycentric_tet",
        component_names=STRESS,
        class_tag=ELE_TAG_TenNodeTetrahedron,
    ),
    ("TenNodeTetrahedron", IntRule.Tet_GL_2, "strain"): _continuum_layout(
        n_gp=4, natural_coords=_TET_GL_2_COORDS,
        coord_system="barycentric_tet",
        component_names=STRAIN,
        class_tag=ELE_TAG_TenNodeTetrahedron,
    ),

    # ── Brick (8-node, 8 GPs Hex_GL_2) ───────────────────────────────
    # C++ class: Brick (Tcl element name: ``stdBrick``).
    ("Brick", IntRule.Hex_GL_2, "stress"): _continuum_layout(
        n_gp=8, natural_coords=_HEX_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=STRESS,
        class_tag=ELE_TAG_Brick,
    ),
    ("Brick", IntRule.Hex_GL_2, "strain"): _continuum_layout(
        n_gp=8, natural_coords=_HEX_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=STRAIN,
        class_tag=ELE_TAG_Brick,
    ),

    # ── BbarBrick (8-node, 8 GPs Hex_GL_2) ───────────────────────────
    # C++ class: BbarBrick (Tcl element name: ``bbarBrick``).
    # Uses the same setResponse keyword and component layout as Brick.
    # NOTE: in current OpenSees builds, BbarBrick's
    # ``ops.eleResponse(eid, "stresses")`` returns zeros even after a
    # converged analysis (the resisting forces and ``"strains"`` paths
    # are correct). MPCO probes materials directly and works around
    # this; DomainCapture and the .out transcoder hit the broken path.
    # Catalog entry retained for MPCO read support.
    ("BbarBrick", IntRule.Hex_GL_2, "stress"): _continuum_layout(
        n_gp=8, natural_coords=_HEX_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=STRESS,
        class_tag=ELE_TAG_BbarBrick,
    ),
    ("BbarBrick", IntRule.Hex_GL_2, "strain"): _continuum_layout(
        n_gp=8, natural_coords=_HEX_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=STRAIN,
        class_tag=ELE_TAG_BbarBrick,
    ),

    # ── SSPbrick (8-node, 1 GP Hex_GL_1) ─────────────────────────────
    # C++ class: SSPbrick. setResponse delegates "stresses" / "strains"
    # to the bound nDMaterial; for ElasticIsotropic and the standard
    # plasticity materials, the material returns the same 6-component
    # vector layout as Brick.
    ("SSPbrick", IntRule.Hex_GL_1, "stress"): _continuum_layout(
        n_gp=1, natural_coords=_HEX_GL_1_COORDS,
        coord_system="isoparametric",
        component_names=STRESS,
        class_tag=ELE_TAG_SSPbrick,
    ),
    ("SSPbrick", IntRule.Hex_GL_1, "strain"): _continuum_layout(
        n_gp=1, natural_coords=_HEX_GL_1_COORDS,
        coord_system="isoparametric",
        component_names=STRAIN,
        class_tag=ELE_TAG_SSPbrick,
    ),

    # ── FourNodeQuad (4 GPs Quad_GL_2) ───────────────────────────────
    # C++ class: FourNodeQuad (Tcl element name: ``quad``).
    # Plane element: 3 stress components per GP (σ_xx, σ_yy, σ_xy).
    # FourNodeQuad.cpp:1370 accepts both ``stresses`` and ``stress``.
    ("FourNodeQuad", IntRule.Quad_GL_2, "stress"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=STRESS_2D,
        class_tag=ELE_TAG_FourNodeQuad,
    ),
    ("FourNodeQuad", IntRule.Quad_GL_2, "strain"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=STRAIN_2D,
        class_tag=ELE_TAG_FourNodeQuad,
    ),

    # ── Tri31 (3-node triangle, 1 GP Triangle_GL_1) ──────────────────
    # C++ class: Tri31 (Tcl element name: ``tri31``).
    # Single Gauss point at the area-coord centroid (1/3, 1/3, 1/3).
    ("Tri31", IntRule.Triangle_GL_1, "stress"): _continuum_layout(
        n_gp=1, natural_coords=_TRI_GL_1_COORDS,
        coord_system="barycentric_tri",
        component_names=STRESS_2D,
        class_tag=ELE_TAG_Tri31,
    ),
    ("Tri31", IntRule.Triangle_GL_1, "strain"): _continuum_layout(
        n_gp=1, natural_coords=_TRI_GL_1_COORDS,
        coord_system="barycentric_tri",
        component_names=STRAIN_2D,
        class_tag=ELE_TAG_Tri31,
    ),

    # ── SSPquad (4-node, 1 GP Quad_GL_1) ─────────────────────────────
    # C++ class: SSPquad. Stabilized single-point quad; setResponse
    # exposes its own ``stress`` / ``strain`` keywords (Vector(3))
    # plus delegates to the material for ``stresses`` / ``strains``.
    ("SSPquad", IntRule.Quad_GL_1, "stress"): _continuum_layout(
        n_gp=1, natural_coords=_QUAD_GL_1_COORDS,
        coord_system="isoparametric",
        component_names=STRESS_2D,
        class_tag=ELE_TAG_SSPquad,
    ),
    ("SSPquad", IntRule.Quad_GL_1, "strain"): _continuum_layout(
        n_gp=1, natural_coords=_QUAD_GL_1_COORDS,
        coord_system="isoparametric",
        component_names=STRAIN_2D,
        class_tag=ELE_TAG_SSPquad,
    ),

    # ── Twenty_Node_Brick (20-node, 27 GPs Hex_GL_3) ─────────────────
    # C++ class: Twenty_Node_Brick. Tcl name: ``20NodeBrick``.
    # 27-GP rule emitted in corner-edge-face-centroid order (see the
    # _HEX_GL_3_* tables above). Twenty_Node_Brick.cpp:1769 declares
    # ``static Vector stresses(162)`` (correctly sized: 27 × 6).
    # NOTE: the 20-node serendipity hex uses an undocumented node
    # ordering; an incorrect order triggers ``exit(-1)`` from
    # ``Jacobian3d`` (line ~1995) — see the live-ops test skip in
    # ``test_results_catalog_solids_real.py``. The catalog GP layout
    # is absolute and unaffected.
    ("Twenty_Node_Brick", IntRule.Hex_GL_3, "stress"): _continuum_layout(
        n_gp=27, natural_coords=_HEX_GL_3_COORDS,
        coord_system="isoparametric",
        component_names=STRESS,
        class_tag=ELE_TAG_Twenty_Node_Brick,
    ),
    ("Twenty_Node_Brick", IntRule.Hex_GL_3, "strain"): _continuum_layout(
        n_gp=27, natural_coords=_HEX_GL_3_COORDS,
        coord_system="isoparametric",
        component_names=STRAIN,
        class_tag=ELE_TAG_Twenty_Node_Brick,
    ),

    # ── EightNodeQuad (8-node, 9 GPs Quad_GL_3) ──────────────────────
    # C++ class: EightNodeQuad. 9-point 3×3 rule in corner-edge-centroid
    # order (EightNodeQuad.cpp:128–144). 3 components per GP (plane
    # stress / plane strain).
    ("EightNodeQuad", IntRule.Quad_GL_3, "stress"): _continuum_layout(
        n_gp=9, natural_coords=_QUAD_GL_3_COORDS_QUAD8,
        coord_system="isoparametric",
        component_names=STRESS_2D,
        class_tag=ELE_TAG_EightNodeQuad,
    ),
    ("EightNodeQuad", IntRule.Quad_GL_3, "strain"): _continuum_layout(
        n_gp=9, natural_coords=_QUAD_GL_3_COORDS_QUAD8,
        coord_system="isoparametric",
        component_names=STRAIN_2D,
        class_tag=ELE_TAG_EightNodeQuad,
    ),

    # =================================================================
    # Shell elements — surface-GP stress resultants (8 components per GP).
    # =================================================================
    #
    # Every shell class returns the same 8-vector per surface GP from
    # ``ops.eleResponse(eid, "stresses")``: 3 membrane forces + 3
    # bending moments + 2 transverse shears. The component layout is
    # identical across shell classes; only the GP rule and walk-order
    # differ. Layered shells (LayeredShellFiberSection) appear here
    # with the *same* entry as their non-layered siblings — the
    # layered behavior is on the section, transparent to the element
    # at this topology level. Through-thickness layer probing maps to
    # apeGmsh's ``layers/`` topology level (Phase 11c).

    # ── Quad shells, 4 GPs (Quad_GL_2) ───────────────────────────────
    # ShellMITC4, ShellDKGQ, ShellNLDKGQ, ASDShellQ4 all share the same
    # CCW 4-GP order at (±1/√3, ±1/√3) — same array as FourNodeQuad.
    # ShellMITC4.cpp:234, ShellDKGQ.cpp:223, ShellNLDKGQ.cpp:227,
    # ASDShellQ4.cpp:173–175.
    #
    # NOTE: ShellMITC4 / ShellDKGQ / ShellNLDKGQ have a broken
    # ``ops.eleResponse(eid, "stresses")`` path in current OpenSees
    # builds — ``materialPointers[i]->getStressResultant()`` returns
    # zeros after a converged analysis (the analysis itself runs
    # fine, displacements and ``"forces"`` are correct, only the
    # per-GP probe is broken). MPCO works around this via direct
    # section probing. Catalog entries retained for MPCO support.
    # ASDShellQ4 is correct via DomainCapture as well.
    ("ShellMITC4", IntRule.Quad_GL_2, "stress"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=SHELL_STRESS_RESULTANTS,
        class_tag=ELE_TAG_ShellMITC4,
    ),
    ("ShellMITC4", IntRule.Quad_GL_2, "strain"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=SHELL_GENERALIZED_STRAINS,
        class_tag=ELE_TAG_ShellMITC4,
    ),
    ("ShellDKGQ", IntRule.Quad_GL_2, "stress"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=SHELL_STRESS_RESULTANTS,
        class_tag=ELE_TAG_ShellDKGQ,
    ),
    ("ShellDKGQ", IntRule.Quad_GL_2, "strain"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=SHELL_GENERALIZED_STRAINS,
        class_tag=ELE_TAG_ShellDKGQ,
    ),
    ("ShellNLDKGQ", IntRule.Quad_GL_2, "stress"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=SHELL_STRESS_RESULTANTS,
        class_tag=ELE_TAG_ShellNLDKGQ,
    ),
    ("ShellNLDKGQ", IntRule.Quad_GL_2, "strain"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=SHELL_GENERALIZED_STRAINS,
        class_tag=ELE_TAG_ShellNLDKGQ,
    ),
    ("ASDShellQ4", IntRule.Quad_GL_2, "stress"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=SHELL_STRESS_RESULTANTS,
        class_tag=ELE_TAG_ASDShellQ4,
    ),
    ("ASDShellQ4", IntRule.Quad_GL_2, "strain"): _continuum_layout(
        n_gp=4, natural_coords=_QUAD_GL_2_COORDS,
        coord_system="isoparametric",
        component_names=SHELL_GENERALIZED_STRAINS,
        class_tag=ELE_TAG_ASDShellQ4,
    ),

    # ── ShellMITC9 (9-node quad, 9 GPs Quad_GL_3, alt CCW order) ─────
    # Distinct GP walk order from EightNodeQuad — see
    # _QUAD_GL_3_COORDS_MITC9 above. ShellMITC9.cpp:107–124.
    # NOTE: ShellMITC9.cpp:518 has ``static Vector stresses(84)`` but
    # the loop fills only 9*8 = 72 entries — ``ops.eleResponse(eid,
    # "stresses")`` returns 84 values (the last 12 are uninitialized).
    # The catalog reflects the *correct* 72-component layout that
    # MPCO writes; DomainCapture against this element rejects the
    # 84-value return. Catalog correct for MPCO.
    ("ShellMITC9", IntRule.Quad_GL_3, "stress"): _continuum_layout(
        n_gp=9, natural_coords=_QUAD_GL_3_COORDS_MITC9,
        coord_system="isoparametric",
        component_names=SHELL_STRESS_RESULTANTS,
        class_tag=ELE_TAG_ShellMITC9,
    ),
    ("ShellMITC9", IntRule.Quad_GL_3, "strain"): _continuum_layout(
        n_gp=9, natural_coords=_QUAD_GL_3_COORDS_MITC9,
        coord_system="isoparametric",
        component_names=SHELL_GENERALIZED_STRAINS,
        class_tag=ELE_TAG_ShellMITC9,
    ),

    # ── Triangle shells, 4 GPs (Triangle_GL_2C) ──────────────────────
    # ShellDKGT and ShellNLDKGT share the 4-point degree-3 rule.
    # ShellDKGT.cpp:238–250.
    # NOTE: same broken-eleResponse pattern as the older quad shells
    # above — ``ops.eleResponse(eid, "stresses")`` returns zeros.
    # Catalog correct for MPCO. Use ASDShellT3 if you need
    # DomainCapture support on triangles.
    ("ShellDKGT", IntRule.Triangle_GL_2C, "stress"): _continuum_layout(
        n_gp=4, natural_coords=_TRI_GL_2C_COORDS,
        coord_system="barycentric_tri",
        component_names=SHELL_STRESS_RESULTANTS,
        class_tag=ELE_TAG_ShellDKGT,
    ),
    ("ShellDKGT", IntRule.Triangle_GL_2C, "strain"): _continuum_layout(
        n_gp=4, natural_coords=_TRI_GL_2C_COORDS,
        coord_system="barycentric_tri",
        component_names=SHELL_GENERALIZED_STRAINS,
        class_tag=ELE_TAG_ShellDKGT,
    ),
    ("ShellNLDKGT", IntRule.Triangle_GL_2C, "stress"): _continuum_layout(
        n_gp=4, natural_coords=_TRI_GL_2C_COORDS,
        coord_system="barycentric_tri",
        component_names=SHELL_STRESS_RESULTANTS,
        class_tag=ELE_TAG_ShellNLDKGT,
    ),
    ("ShellNLDKGT", IntRule.Triangle_GL_2C, "strain"): _continuum_layout(
        n_gp=4, natural_coords=_TRI_GL_2C_COORDS,
        coord_system="barycentric_tri",
        component_names=SHELL_GENERALIZED_STRAINS,
        class_tag=ELE_TAG_ShellNLDKGT,
    ),

    # ── ASDShellT3 (3-node triangle, 3 mid-edge GPs Triangle_GL_2B) ──
    # ASDShellT3.cpp:148–150. Petracca's shell.
    ("ASDShellT3", IntRule.Triangle_GL_2B, "stress"): _continuum_layout(
        n_gp=3, natural_coords=_TRI_GL_2B_COORDS,
        coord_system="barycentric_tri",
        component_names=SHELL_STRESS_RESULTANTS,
        class_tag=ELE_TAG_ASDShellT3,
    ),
    ("ASDShellT3", IntRule.Triangle_GL_2B, "strain"): _continuum_layout(
        n_gp=3, natural_coords=_TRI_GL_2B_COORDS,
        coord_system="barycentric_tri",
        component_names=SHELL_GENERALIZED_STRAINS,
        class_tag=ELE_TAG_ASDShellT3,
    ),

    # =================================================================
    # Truss family — single GP at the line midpoint, scalar axial force.
    # =================================================================
    #
    # All MPCO-classified ``Line_2N + Line_GL_1`` (rule 1) elements
    # whose response is a single scalar ``axial_force``.
    # ``ops.eleResponse(eid, "axialForce")`` returns Vector(1) for
    # these. MPCO writes them under
    # ``ON_ELEMENTS/axialForce/<tag>-<Class>[1:0:0]``.
    #
    # NOTE: this is the only catalog topology where the component
    # layout is a 1-tuple of a scalar name (no tensor / axis suffix).
    # The prefix splitter recognises ``"axial_force"`` as a scalar
    # canonical via the full-name fallback in
    # ``gauss_keyword_for_canonical``.
    #
    # The TrussSection / CorotTrussSection family is *not* covered
    # here — they expose section-level forces (Vector(3) for the basic
    # axial-shear-moment system) under a different response keyword.
    # That's Round B (line_stations / nodal_forces topology).
    ("Truss", IntRule.Line_GL_1, "axial_force"): _continuum_layout(
        n_gp=1, natural_coords=_LINE_GL_1_COORDS,
        coord_system="isoparametric_1d",
        component_names=("axial_force",),
        class_tag=ELE_TAG_Truss,
    ),
    ("CorotTruss", IntRule.Line_GL_1, "axial_force"): _continuum_layout(
        n_gp=1, natural_coords=_LINE_GL_1_COORDS,
        coord_system="isoparametric_1d",
        component_names=("axial_force",),
        class_tag=ELE_TAG_CorotTruss,
    ),
    ("Truss2", IntRule.Line_GL_1, "axial_force"): _continuum_layout(
        n_gp=1, natural_coords=_LINE_GL_1_COORDS,
        coord_system="isoparametric_1d",
        component_names=("axial_force",),
        class_tag=ELE_TAG_Truss2,
    ),
    ("CorotTruss2", IntRule.Line_GL_1, "axial_force"): _continuum_layout(
        n_gp=1, natural_coords=_LINE_GL_1_COORDS,
        coord_system="isoparametric_1d",
        component_names=("axial_force",),
        class_tag=ELE_TAG_CorotTruss2,
    ),
    ("InertiaTruss", IntRule.Line_GL_1, "axial_force"): _continuum_layout(
        n_gp=1, natural_coords=_LINE_GL_1_COORDS,
        coord_system="isoparametric_1d",
        component_names=("axial_force",),
        class_tag=ELE_TAG_InertiaTruss,
    ),
}


# =====================================================================
# CUSTOM_RULE_CATALOG — beam-columns with per-instance integration
# =====================================================================
#
# Phase 11b. Keyed on ``(class_name, token)`` (no integration-rule
# field — every entry implicitly carries ``IntRule.Custom``). The
# concrete per-element ``ResponseLayout`` is built at read/capture
# time by :func:`resolve_layout_from_gp_x` from the per-element
# ``GP_X`` array and the assigned section's response codes.
#
# v1 covers ``token == "section_force"`` only. The conjugate
# ``"section_deformation"`` token will be added when its canonical
# vocabulary names land (curvature_y / curvature_z and the line-
# axial strain are not in ``LINE_DIAGRAMS`` yet).
#
# Notes on disp-based beam-columns: ``DispBeamColumn{2d,3d}`` does not
# expose ``ops.eleResponse(eid, "integrationPoints")`` — DomainCapture
# v1 cannot drive these directly. MPCO writes their ``GP_X`` to disk,
# so the read path works fine. See
# :doc:`internal_docs/plan_phase_11b_line_stations` §"Custom-rule
# complexity" for the Tier 1/2/3 discovery story.

CUSTOM_RULE_CATALOG: dict[tuple[str, str], CustomRuleLayout] = {
    # ── Force-based beam-columns ──────────────────────────────────────
    ("ForceBeamColumn2d", "section_force"): CustomRuleLayout(
        class_tag=ELE_TAG_ForceBeamColumn2d,
        coord_system="isoparametric_1d",
    ),
    ("ForceBeamColumn3d", "section_force"): CustomRuleLayout(
        class_tag=ELE_TAG_ForceBeamColumn3d,
        coord_system="isoparametric_1d",
    ),
    ("ForceBeamColumnCBDI2d", "section_force"): CustomRuleLayout(
        class_tag=ELE_TAG_ForceBeamColumnCBDI2d,
        coord_system="isoparametric_1d",
    ),
    ("ForceBeamColumnWarping2d", "section_force"): CustomRuleLayout(
        class_tag=ELE_TAG_ForceBeamColumnWarping2d,
        coord_system="isoparametric_1d",
    ),
    ("ElasticForceBeamColumn2d", "section_force"): CustomRuleLayout(
        class_tag=ELE_TAG_ElasticForceBeamColumn2d,
        coord_system="isoparametric_1d",
    ),
    ("ElasticForceBeamColumn3d", "section_force"): CustomRuleLayout(
        class_tag=ELE_TAG_ElasticForceBeamColumn3d,
        coord_system="isoparametric_1d",
    ),
    # ── Displacement-based beam-columns ───────────────────────────────
    ("DispBeamColumn2d", "section_force"): CustomRuleLayout(
        class_tag=ELE_TAG_DispBeamColumn2d,
        coord_system="isoparametric_1d",
    ),
    ("DispBeamColumn3d", "section_force"): CustomRuleLayout(
        class_tag=ELE_TAG_DispBeamColumn3d,
        coord_system="isoparametric_1d",
    ),
}


# =====================================================================
# NODAL_FORCE_CATALOG — closed-form elastic beams
# =====================================================================
#
# Phase 11b. Keyed on ``(class_name, token)`` where ``token`` is the
# OpenSees recorder token in apeGmsh-canonical form
# (``"global_force"`` / ``"local_force"``). Component layouts use
# the ``nodal_resisting_*`` canonical names from
# :data:`apeGmsh.results._vocabulary.PER_ELEMENT_NODAL_FORCES` —
# distinct from the global-frame ``force_*`` / ``moment_*`` names
# (which are applied nodal forces, a different topology level).
#
# Per-node component order matches OpenSees's per-node DOF order:
# 2D ndf=3 → (Fx, Fy, Mz); 3D ndf=6 → (Fx, Fy, Fz, Mx, My, Mz).

_BEAM_NODAL_2D_GLOBAL: tuple[str, ...] = (
    "nodal_resisting_force_x",
    "nodal_resisting_force_y",
    "nodal_resisting_moment_z",
)
_BEAM_NODAL_2D_LOCAL: tuple[str, ...] = (
    "nodal_resisting_force_local_x",
    "nodal_resisting_force_local_y",
    "nodal_resisting_moment_local_z",
)
_BEAM_NODAL_3D_GLOBAL: tuple[str, ...] = (
    "nodal_resisting_force_x",
    "nodal_resisting_force_y",
    "nodal_resisting_force_z",
    "nodal_resisting_moment_x",
    "nodal_resisting_moment_y",
    "nodal_resisting_moment_z",
)
_BEAM_NODAL_3D_LOCAL: tuple[str, ...] = (
    "nodal_resisting_force_local_x",
    "nodal_resisting_force_local_y",
    "nodal_resisting_force_local_z",
    "nodal_resisting_moment_local_x",
    "nodal_resisting_moment_local_y",
    "nodal_resisting_moment_local_z",
)


def _nodal_force_2d(class_tag: int, frame: str) -> NodalForceLayout:
    layout = _BEAM_NODAL_2D_GLOBAL if frame == "global" else _BEAM_NODAL_2D_LOCAL
    return NodalForceLayout(
        n_nodes_per_element=2,
        n_components_per_node=3,
        component_layout=layout,
        class_tag=class_tag,
        frame=frame,
    )


def _nodal_force_3d(class_tag: int, frame: str) -> NodalForceLayout:
    layout = _BEAM_NODAL_3D_GLOBAL if frame == "global" else _BEAM_NODAL_3D_LOCAL
    return NodalForceLayout(
        n_nodes_per_element=2,
        n_components_per_node=6,
        component_layout=layout,
        class_tag=class_tag,
        frame=frame,
    )


NODAL_FORCE_CATALOG: dict[tuple[str, str], NodalForceLayout] = {
    # ── ElasticBeam ──────────────────────────────────────────────────
    ("ElasticBeam2d", "global_force"):
        _nodal_force_2d(ELE_TAG_ElasticBeam2d, "global"),
    ("ElasticBeam2d", "local_force"):
        _nodal_force_2d(ELE_TAG_ElasticBeam2d, "local"),
    ("ElasticBeam3d", "global_force"):
        _nodal_force_3d(ELE_TAG_ElasticBeam3d, "global"),
    ("ElasticBeam3d", "local_force"):
        _nodal_force_3d(ELE_TAG_ElasticBeam3d, "local"),
    # ── ModElasticBeam (2D only in v1) ───────────────────────────────
    ("ModElasticBeam2d", "global_force"):
        _nodal_force_2d(ELE_TAG_ModElasticBeam2d, "global"),
    ("ModElasticBeam2d", "local_force"):
        _nodal_force_2d(ELE_TAG_ModElasticBeam2d, "local"),
    # ── ElasticTimoshenkoBeam ────────────────────────────────────────
    ("ElasticTimoshenkoBeam2d", "global_force"):
        _nodal_force_2d(ELE_TAG_ElasticTimoshenkoBeam2d, "global"),
    ("ElasticTimoshenkoBeam2d", "local_force"):
        _nodal_force_2d(ELE_TAG_ElasticTimoshenkoBeam2d, "local"),
    ("ElasticTimoshenkoBeam3d", "global_force"):
        _nodal_force_3d(ELE_TAG_ElasticTimoshenkoBeam3d, "global"),
    ("ElasticTimoshenkoBeam3d", "local_force"):
        _nodal_force_3d(ELE_TAG_ElasticTimoshenkoBeam3d, "local"),
}


# =====================================================================
# Public API
# =====================================================================

class CatalogLookupError(KeyError):
    """Raised when ``(class_name, int_rule, token)`` is not in the catalog.

    Subclass of ``KeyError`` so callers can ``except KeyError`` if
    they want lenient skip-on-miss behaviour.
    """


def lookup(class_name: str, int_rule: int, token: str) -> ResponseLayout:
    """Catalog access with a helpful error message.

    Parameters
    ----------
    class_name
        OpenSees C++ class name as it appears in MPCO bracket keys
        (e.g. ``"FourNodeTetrahedron"``).
    int_rule
        Integration-rule enum integer (see ``IntRule``).
    token
        Broad response keyword (``"stress"``, ``"strain"``, ...).
    """
    key = (class_name, int_rule, token)
    try:
        return RESPONSE_CATALOG[key]
    except KeyError:
        raise CatalogLookupError(
            f"No ResponseLayout for (class={class_name!r}, "
            f"int_rule={int_rule}, token={token!r}). "
            f"Add an entry to RESPONSE_CATALOG in "
            f"src/apeGmsh/solvers/_element_response.py if this "
            f"combination should be supported."
        ) from None


def is_catalogued(class_name: str, int_rule: int, token: str) -> bool:
    """True if ``(class_name, int_rule, token)`` has a layout entry."""
    return (class_name, int_rule, token) in RESPONSE_CATALOG


# ---------------------------------------------------------------------
# Custom-rule (line-stations) catalog access + resolver
# ---------------------------------------------------------------------
#
# OpenSees section response codes (``SECTION_RESPONSE_*`` in
# ``SRC/material/section/SectionForceDeformation.h`` lines 52–57)
# identify each entry of a section's force / deformation vector. The
# mapping is fixed across all sections — what varies is the *order*
# in which a particular section's ``getType()`` returns these codes.
# A bare ``FiberSection3d`` returns ``[P, Mz, My, T]`` (codes
# 2, 1, 4, 6); a ``SectionAggregator`` adding shears returns
# ``[P, Mz, My, T, Vy, Vz]`` (codes 2, 1, 4, 6, 3, 5). Per-element
# section codes therefore drive the concrete ``ResponseLayout``.

SECTION_RESPONSE_TO_CANONICAL: dict[int, str] = {
    1: "bending_moment_z",   # SECTION_RESPONSE_MZ
    2: "axial_force",        # SECTION_RESPONSE_P
    3: "shear_y",            # SECTION_RESPONSE_VY
    4: "bending_moment_y",   # SECTION_RESPONSE_MY
    5: "shear_z",            # SECTION_RESPONSE_VZ
    6: "torsion",            # SECTION_RESPONSE_T
}


def lookup_custom_rule(class_name: str, token: str) -> CustomRuleLayout:
    """Look up the structural layout for a custom-rule (line-stations) entry.

    See :data:`CUSTOM_RULE_CATALOG`. Raises
    :class:`CatalogLookupError` on miss with a helpful message.
    """
    key = (class_name, token)
    try:
        return CUSTOM_RULE_CATALOG[key]
    except KeyError:
        raise CatalogLookupError(
            f"No CustomRuleLayout for (class={class_name!r}, "
            f"token={token!r}). Add an entry to CUSTOM_RULE_CATALOG "
            f"in src/apeGmsh/solvers/_element_response.py if this "
            f"line-station element should be supported."
        ) from None


def is_custom_rule_catalogued(class_name: str, token: str) -> bool:
    """True if ``(class_name, token)`` has a custom-rule layout entry."""
    return (class_name, token) in CUSTOM_RULE_CATALOG


def resolve_layout_from_gp_x(
    custom: CustomRuleLayout,
    gp_x: ndarray,
    section_codes: tuple[int, ...],
) -> ResponseLayout:
    """Build a concrete :class:`ResponseLayout` for one custom-rule bucket.

    The structural :class:`CustomRuleLayout` declares only what is
    fixed by the element class (class tag, parent domain). Concrete
    shape — how many integration points there are, where they sit in
    the parent domain, and what each component column means — comes
    from the per-element data this function takes as input.

    Parameters
    ----------
    custom
        Catalog entry for ``(class_name, token)``.
    gp_x
        Per-element natural coordinates in ``[-1, +1]``. Shape
        ``(n_IP,)``. From MPCO ``MODEL/ELEMENTS/<bucket>/@GP_X`` or
        from ``ops.eleResponse(eid, "integrationPoints")`` after
        normalising the physical-length values OpenSees returns
        (``ξ_natural = 2 * ξ_physical/L - 1``).
    section_codes
        OpenSees ``SECTION_RESPONSE_*`` codes giving the per-IP
        section's response order (``[P, Mz, My, T]`` =
        ``(2, 1, 4, 6)`` for a bare 3D fiber section). Drives both
        ``n_components_per_gp`` (= ``len(section_codes)``) and the
        canonical ``component_layout`` order. Code values come from
        the MPCO ``META/COMPONENTS`` string (Step 2a) or from a
        section-introspection probe at capture time (Step 2b).

    Returns
    -------
    ResponseLayout
        A concrete fixed-rule layout that the existing
        :func:`unflatten` / :func:`flatten` keystone consumes.

    Raises
    ------
    KeyError
        If ``section_codes`` contains an unrecognised value (e.g. a
        warping or asymmetric-section code outside
        :data:`SECTION_RESPONSE_TO_CANONICAL`).
    """
    gp_x = np.asarray(gp_x, dtype=np.float64).reshape(-1)
    component_layout = tuple(
        SECTION_RESPONSE_TO_CANONICAL[int(c)] for c in section_codes
    )
    return ResponseLayout(
        n_gauss_points=gp_x.size,
        natural_coords=gp_x.reshape(-1, 1),
        coord_system=custom.coord_system,
        n_components_per_gp=len(component_layout),
        component_layout=component_layout,
        class_tag=custom.class_tag,
    )


# ---------------------------------------------------------------------
# Nodal-force catalog access
# ---------------------------------------------------------------------

def lookup_nodal_force(class_name: str, token: str) -> NodalForceLayout:
    """Look up the per-element-node force layout for a closed-form line element.

    See :data:`NODAL_FORCE_CATALOG`. Raises
    :class:`CatalogLookupError` on miss with a helpful message.
    """
    key = (class_name, token)
    try:
        return NODAL_FORCE_CATALOG[key]
    except KeyError:
        raise CatalogLookupError(
            f"No NodalForceLayout for (class={class_name!r}, "
            f"token={token!r}). Add an entry to NODAL_FORCE_CATALOG "
            f"in src/apeGmsh/solvers/_element_response.py if this "
            f"closed-form line element should be supported."
        ) from None


def is_nodal_force_catalogued(class_name: str, token: str) -> bool:
    """True if ``(class_name, token)`` has a nodal-force layout entry."""
    return (class_name, token) in NODAL_FORCE_CATALOG


# ---------------------------------------------------------------------
# Section-shape inference + parent-coordinate normalisation
# ---------------------------------------------------------------------
#
# These helpers serve every Phase 11b consumer that has to decode
# line-station data without a META block — i.e. DomainCapture (which
# probes ops live) and the .out transcoder (which reads text recorder
# files). MPCO read does not use them: META/COMPONENTS carries the
# section codes verbatim.

# Inferred section codes by ``(dimension, n_components)`` under
# canonical aggregation order. Codes match
# :data:`SECTION_RESPONSE_TO_CANONICAL`.
INFERRED_SECTION_CODES_TABLE: dict[tuple[int, int], tuple[int, ...]] = {
    # 2D
    (2, 2): (2, 1),                # P, Mz
    (2, 3): (2, 1, 3),             # P, Mz, Vy
    # 3D
    (3, 3): (2, 1, 4),             # P, Mz, My
    (3, 4): (2, 1, 4, 6),          # P, Mz, My, T
    (3, 5): (2, 1, 4, 6, 3),       # P, Mz, My, T, Vy
    (3, 6): (2, 1, 4, 6, 3, 5),    # P, Mz, My, T, Vy, Vz
}


def class_dimension(class_name: str) -> int:
    """Infer 2D vs 3D from an OpenSees beam-column class name suffix."""
    lower = class_name.lower()
    if lower.endswith("2d"):
        return 2
    if lower.endswith("3d"):
        return 3
    raise ValueError(
        f"Cannot infer dimension from class name {class_name!r}; "
        f"expected a suffix of '2d' or '3d'."
    )


def infer_section_codes(
    class_name: str, n_components: int,
) -> tuple[int, ...]:
    """Map ``(class_dim, n_components)`` to canonical section codes.

    Used when the section's ``getType()`` is not directly available
    — e.g. from a ``.out`` recorder file (no META) or live openseespy
    (no section-introspection API). Assumes canonical aggregation
    order (P, Mz, My, T, Vy, Vz in 3D — inner section codes first,
    aggregated codes last in user-listed order). Non-canonical
    SectionAggregator orderings cannot be reliably decoded; users
    with such sections should use MPCO recording where META carries
    the actual code names.

    Raises ``ValueError`` for shapes outside the canonical table.
    """
    dim = class_dimension(class_name)
    key = (dim, int(n_components))
    if key in INFERRED_SECTION_CODES_TABLE:
        return INFERRED_SECTION_CODES_TABLE[key]
    raise ValueError(
        f"Cannot infer section codes for {class_name} with "
        f"{n_components} section.force components. Canonical "
        f"layouts: 2D ∈ {{2 (P,Mz), 3 (P,Mz,Vy)}}; "
        f"3D ∈ {{3 (P,Mz,My), 4 (+T), 5 (+Vy), 6 (+Vy,Vz)}}. "
        f"Non-canonical SectionAggregator orderings are not "
        f"supported by inference; use MPCO recording instead."
    )


def normalise_integration_points(
    xi_phys: ndarray, L: float,
) -> ndarray:
    """Map physical IP positions ``[0, L]`` to natural ``[-1, +1]``.

    OpenSees's ``ops.eleResponse(eid, "integrationPoints")`` returns
    physical positions ``pts[i] * L`` along the beam (per
    ``ForceBeamColumn3d.cpp:3338–3346``). MPCO's ``GP_X`` and
    apeGmsh's catalog use natural ξ ∈ [-1, +1]; this helper bridges.
    """
    if L <= 0:
        raise ValueError(f"Element length {L} must be positive.")
    return 2.0 * xi_phys / L - 1.0


# ---------------------------------------------------------------------
# Canonical-component prefix routing (shared by all three sites)
# ---------------------------------------------------------------------
#
# A canonical Gauss-level component name decomposes into ``<prefix>_<suffix>``
# where ``<suffix>`` is a tensor index (``xx`` / ``yy`` / ``zz`` / ``xy`` /
# ``yz`` / ``xz``) or vector axis (``x`` / ``y`` / ``z``). For shell
# resultants the prefix is multi-word (``membrane_force_xx`` →
# prefix=``membrane_force``); a naive split-on-first-underscore would
# break those, so we strip a known suffix instead.
#
# The prefix in turn maps to an OpenSees ``setResponse`` keyword
# (``stresses`` / ``strains``) — the same keyword used by
# ``ops.eleResponse``, the same MPCO ``ON_ELEMENTS/<token>/`` group
# name, and the same routing for the .out transcoder. Shell
# resultants (``membrane_force_*``, ``bending_moment_*``,
# ``transverse_shear_*``) share the ``stresses`` keyword with
# continuum stress; their per-class catalog entry carries the
# 8-component layout.

_KNOWN_COMPONENT_SUFFIXES: tuple[str, ...] = (
    # Tensor indices (try longer first so ``stress_xz`` doesn't match ``z``).
    "xx", "yy", "zz", "xy", "yz", "xz",
    # Vector axes.
    "x", "y", "z",
)


def split_canonical_component(name: str) -> tuple[str, str] | None:
    """Split a canonical name into ``(prefix, suffix)``.

    Recognizes tensor-index suffixes (``xx`` / ``yy`` / ``zz`` / ``xy``
    / ``yz`` / ``xz``) and vector-axis suffixes (``x`` / ``y`` / ``z``).
    Returns ``None`` for scalar names without an axis suffix
    (e.g. ``"pore_pressure"``, ``"damage"``).

    Examples
    --------
    ``"stress_xx"``           → ``("stress", "xx")``
    ``"membrane_force_xy"``   → ``("membrane_force", "xy")``
    ``"transverse_shear_xz"`` → ``("transverse_shear", "xz")``
    ``"displacement_x"``      → ``("displacement", "x")``
    ``"pore_pressure"``       → ``None``
    """
    for suf in _KNOWN_COMPONENT_SUFFIXES:
        sep = "_" + suf
        if name.endswith(sep):
            return name[: -len(sep)], suf
    return None


# Canonical prefix (or full scalar name) → OpenSees ``setResponse``
# keyword. The keyword is *also* the on-disk MPCO group name
# (``ON_ELEMENTS/<keyword>/``).
#
# Routing depends on the *topology level* the component belongs to —
# the same canonical name can hit different recorder keywords:
# ``axial_force`` is the Truss scalar (``axialForce``) at the
# gauss-points topology, but the first column of ``section.force``
# at the line-stations topology. Likewise ``bending_moment_y`` is a
# shell resultant (``stresses``) at gauss-points but a beam section
# moment (``section.force``) at line-stations.
#
# Each topology has its own table; the routing helpers take a
# ``topology`` keyword. The default ``topology=None`` preserves the
# Phase 11a behaviour (continuum stress/strain, shell resultants,
# truss axial — everything that lives at the gauss-points topology).
_GAUSS_PREFIX_TO_KEYWORD: dict[str, str] = {
    # Continuum stress / strain.
    "stress": "stresses",
    "strain": "strains",
    # Shell stress resultants.
    "membrane_force": "stresses",
    "bending_moment": "stresses",
    "transverse_shear": "stresses",
    # Shell generalized strains.
    "membrane_strain": "strains",
    "curvature": "strains",
    "transverse_shear_strain": "strains",
    # Truss / line scalar — full canonical name (no axis suffix).
    # OpenSees Truss::setResponse accepts ``"axialForce"``,
    # ``"basicForce"``, and ``"basicForces"`` as aliases for the same
    # scalar response (Truss.cpp:1194–1196).
    "axial_force": "axialForce",
}


# Line-stations topology — force-/disp-based beam-columns expose
# section-level forces under a single recorder keyword
# (``section.force``). All apeGmsh ``LINE_DIAGRAMS`` canonicals
# route there. The MPCO bucket name on disk is identical
# (``ON_ELEMENTS/section.force/<bracket>``).
_LINE_STATION_PREFIX_TO_KEYWORD: dict[str, str] = {
    # Scalars (full canonical names, no axis suffix).
    "axial_force": "section.force",
    "torsion": "section.force",
    # Vector-suffixed (``shear_y``, ``shear_z``, ``bending_moment_y``,
    # ``bending_moment_z``) — the splitter strips ``_y`` / ``_z`` and
    # we route on the prefix.
    "shear": "section.force",
    "bending_moment": "section.force",
}


# Nodal-forces topology — closed-form elastic beams expose per-
# element-node force vectors under ``globalForce`` / ``localForce``.
# The canonical names are the ``nodal_resisting_*`` family from
# :data:`apeGmsh.results._vocabulary.PER_ELEMENT_NODAL_FORCES`.
_NODAL_FORCE_PREFIX_TO_KEYWORD: dict[str, str] = {
    # Global frame.
    "nodal_resisting_force": "globalForce",
    "nodal_resisting_moment": "globalForce",
    # Local frame — distinct prefixes (the splitter strips only the
    # final ``_x`` / ``_y`` / ``_z`` axis suffix, leaving
    # ``nodal_resisting_force_local`` intact).
    "nodal_resisting_force_local": "localForce",
    "nodal_resisting_moment_local": "localForce",
}


# OpenSees keyword → catalog token (the third element of
# ``RESPONSE_CATALOG`` keys for the gauss-points topology).
_KEYWORD_TO_CATALOG_TOKEN: dict[str, str] = {
    "stresses": "stress",
    "strains": "strain",
    "axialForce": "axial_force",
}


# Line-stations topology — keyword → catalog token (second element
# of ``CUSTOM_RULE_CATALOG`` keys).
_LINE_STATION_KEYWORD_TO_CATALOG_TOKEN: dict[str, str] = {
    "section.force": "section_force",
}


# Nodal-forces topology — keyword → catalog token (second element
# of ``NODAL_FORCE_CATALOG`` keys).
_NODAL_FORCE_KEYWORD_TO_CATALOG_TOKEN: dict[str, str] = {
    "globalForce": "global_force",
    "localForce": "local_force",
}


# Per-topology dispatch — keeps the routing helpers small and lets
# us add new topologies without growing more conditional branches.
_TOPOLOGY_PREFIX_TABLES: dict[str | None, dict[str, str]] = {
    None: _GAUSS_PREFIX_TO_KEYWORD,
    "line_stations": _LINE_STATION_PREFIX_TO_KEYWORD,
    "nodal_forces": _NODAL_FORCE_PREFIX_TO_KEYWORD,
}

_TOPOLOGY_KEYWORD_TABLES: dict[str | None, dict[str, str]] = {
    None: _KEYWORD_TO_CATALOG_TOKEN,
    "line_stations": _LINE_STATION_KEYWORD_TO_CATALOG_TOKEN,
    "nodal_forces": _NODAL_FORCE_KEYWORD_TO_CATALOG_TOKEN,
}


def gauss_keyword_for_canonical(
    name: str, *, topology: str | None = None,
) -> str | None:
    """Return the ``ops.eleResponse`` keyword for a component.

    Vectors / tensors decompose via :func:`split_canonical_component`
    and route through their prefix (``"stress_xx"`` →
    ``"stresses"``, ``"membrane_force_xy"`` → ``"stresses"``).
    Scalar canonical names (``"axial_force"``, no axis suffix) fall
    through to a full-name lookup in the same table.

    The ``topology`` keyword selects which routing table to use:

    - ``None`` (default) — gauss-points topology: continuum
      stress/strain, shell resultants, truss axial.
    - ``"line_stations"`` — beam-column section forces. The same
      canonical name (``axial_force``) maps to ``section.force``
      here rather than ``axialForce``.
    - ``"nodal_forces"`` — closed-form elastic beam per-node forces
      (``nodal_resisting_*`` canonicals → ``globalForce`` /
      ``localForce``).

    Returns ``None`` for components without a routing in the
    requested topology.
    """
    table = _TOPOLOGY_PREFIX_TABLES.get(topology)
    if table is None:
        raise ValueError(
            f"Unknown topology {topology!r}. Expected one of "
            f"{sorted(k for k in _TOPOLOGY_PREFIX_TABLES if k is not None)} "
            f"or None."
        )
    parts = split_canonical_component(name)
    if parts is not None:
        prefix, _ = parts
        keyword = table.get(prefix)
        if keyword is not None:
            return keyword
    # Scalar fallback: the full canonical name *is* the prefix.
    return table.get(name)


def catalog_token_for_keyword(
    keyword: str, *, topology: str | None = None,
) -> str | None:
    """Map an OpenSees keyword to its catalog token in the given topology.

    Default topology preserves Phase 11a behaviour
    (``stresses`` → ``"stress"``, ``axialForce`` → ``"axial_force"``).
    Other topologies use their own keyword→token tables.
    """
    table = _TOPOLOGY_KEYWORD_TABLES.get(topology)
    if table is None:
        raise ValueError(
            f"Unknown topology {topology!r}. Expected one of "
            f"{sorted(k for k in _TOPOLOGY_KEYWORD_TABLES if k is not None)} "
            f"or None."
        )
    return table.get(keyword)


def gauss_routing_for_canonical(
    name: str, *, topology: str | None = None,
) -> tuple[str, str] | None:
    """Return ``(ops_keyword, catalog_token)`` for a component.

    Convenience wrapper combining
    :func:`gauss_keyword_for_canonical` and
    :func:`catalog_token_for_keyword`. Used by reader/transcoder
    sites that need both pieces at once. Returns ``None`` if the
    component has no routing in ``topology``.
    """
    keyword = gauss_keyword_for_canonical(name, topology=topology)
    if keyword is None:
        return None
    catalog_token = catalog_token_for_keyword(keyword, topology=topology)
    if catalog_token is None:
        return None
    return (keyword, catalog_token)


# ---------------------------------------------------------------------
# MPCO bracket-key parser
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MPCOElementKey:
    """Parsed MPCO bracket key for an element bucket.

    Two on-disk forms are supported:

    - ``MODEL/ELEMENTS/<tag>-<Class>[<rule>:<cust>]`` — connectivity.
    - ``RESULTS/ON_ELEMENTS/<token>/<tag>-<Class>[<rule>:<cust>:<hdr>]`` —
      results buckets with an extra ``<hdr>`` (response-shape index).
    """
    class_tag: int
    class_name: str
    int_rule: int
    custom_rule_idx: int
    header_idx: int   # 0 when the source path was MODEL/ELEMENTS

    @property
    def is_custom_rule(self) -> bool:
        return self.int_rule == IntRule.Custom


_MPCO_KEY_RE = re.compile(
    r"""
    ^\s*
    (?P<tag>-?\d+)              # signed for safety; tags are normally positive
    -
    (?P<name>[^[\]]+?)
    \[
    (?P<rule>-?\d+)
    :
    (?P<cust>-?\d+)
    (?: : (?P<hdr>-?\d+) )?     # optional header index (only on results path)
    \]
    \s*$
    """,
    re.VERBOSE,
)


def parse_mpco_element_key(key: str) -> MPCOElementKey:
    """Parse ``"<tag>-<Class>[<rule>:<cust>(:<hdr>)?]"``.

    Examples (verified against the mpco-recorder skill's
    ``hdf5-layout.md`` §3 worked examples)::

        parse_mpco_element_key("179-FourNodeTetrahedron[300:0]")
        # MPCOElementKey(class_tag=179, class_name='FourNodeTetrahedron',
        #                int_rule=300, custom_rule_idx=0, header_idx=0)

        parse_mpco_element_key("31-ASDShellQ4[202:0:0]")
        # ... header_idx=0

        parse_mpco_element_key("73-ForceBeamColumn3d[1000:3:0]")
        # ... header_idx=0, is_custom_rule=True
    """
    m = _MPCO_KEY_RE.match(key)
    if m is None:
        raise ValueError(
            f"Could not parse MPCO element key {key!r}; expected "
            f"'<tag>-<Class>[<rule>:<cust>]' or "
            f"'<tag>-<Class>[<rule>:<cust>:<hdr>]'."
        )
    hdr = m.group("hdr")
    return MPCOElementKey(
        class_tag=int(m.group("tag")),
        class_name=m.group("name"),
        int_rule=int(m.group("rule")),
        custom_rule_idx=int(m.group("cust")),
        header_idx=int(hdr) if hdr is not None else 0,
    )


# ---------------------------------------------------------------------
# unflatten — the keystone shape transform
# ---------------------------------------------------------------------

def unflatten(
    flat: ndarray,
    layout: ResponseLayout,
) -> dict[str, ndarray]:
    """Convert ``(T, E_g, flat_size)`` → per-component ``(T, E_g, n_GP)``.

    ``flat`` is the raw response array, with the canonical ordering
    GP-slowest / component-fastest::

        flat[t, e, g * K + k] = component k at GP g of element e at time t

    where ``G = layout.n_gauss_points`` and
    ``K = layout.n_components_per_gp``.

    Returns
    -------
    dict[str, ndarray]
        ``{component_name: (T, E_g, G)}`` keyed by canonical apeGmsh
        names from ``layout.component_layout``.
    """
    flat = np.asarray(flat)
    if flat.ndim != 3:
        raise ValueError(
            f"unflatten expects a 3-D flat array (T, E_g, flat_size); "
            f"got shape {flat.shape}."
        )
    T, E_g, flat_size = flat.shape
    expected = layout.flat_size_per_element
    if flat_size != expected:
        raise ValueError(
            f"flat_size {flat_size} does not match layout's "
            f"n_gauss_points * n_components_per_gp = "
            f"{layout.n_gauss_points} * {layout.n_components_per_gp} "
            f"= {expected}."
        )

    # Reshape (T, E_g, G * K) → (T, E_g, G, K) — GP slowest.
    reshaped = flat.reshape(T, E_g, layout.n_gauss_points,
                             layout.n_components_per_gp)
    return {
        comp_name: np.ascontiguousarray(reshaped[:, :, :, k])
        for k, comp_name in enumerate(layout.component_layout)
    }


def flatten(
    components: dict[str, ndarray],
    layout: ResponseLayout,
) -> ndarray:
    """Inverse of ``unflatten`` — pack per-component arrays into the flat form.

    Used by tests and (potentially) by transcoder code paths that build
    flat arrays from already-decoded per-component data.

    Parameters
    ----------
    components
        ``{component_name: (T, E_g, n_GP)}`` keyed by canonical names.
        Must contain every name in ``layout.component_layout``; extra
        keys raise.
    layout
        The matching ``ResponseLayout``.

    Returns
    -------
    ndarray
        Shape ``(T, E_g, layout.flat_size_per_element)``.
    """
    expected_names = set(layout.component_layout)
    got_names = set(components.keys())
    if got_names != expected_names:
        missing = expected_names - got_names
        extra = got_names - expected_names
        raise ValueError(
            f"flatten requires exactly the catalog's components. "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )

    first = components[layout.component_layout[0]]
    if first.ndim != 3:
        raise ValueError(
            f"components must have shape (T, E_g, n_GP); got {first.shape}."
        )
    T, E_g, n_gp = first.shape
    if n_gp != layout.n_gauss_points:
        raise ValueError(
            f"component arrays have {n_gp} GPs but layout specifies "
            f"{layout.n_gauss_points}."
        )

    out = np.empty(
        (T, E_g, layout.flat_size_per_element), dtype=np.float64,
    )
    for k, name in enumerate(layout.component_layout):
        arr = np.asarray(components[name])
        if arr.shape != (T, E_g, n_gp):
            raise ValueError(
                f"component {name!r} has shape {arr.shape}; expected "
                f"{(T, E_g, n_gp)}."
            )
        # GP-slowest packing: stride n_components_per_gp through flat axis.
        out[:, :, k::layout.n_components_per_gp] = arr
    return out


def unflatten_nodal(
    flat: ndarray,
    layout: NodalForceLayout,
) -> dict[str, ndarray]:
    """Convert ``(T, E, n_nodes * K)`` → per-component ``(T, E, n_nodes)``.

    Closed-form line elements (``ElasticBeam*``,
    ``ElasticTimoshenkoBeam*``, ``ModElasticBeam*``) emit per-element-
    node force vectors with the same node-slowest / component-fastest
    packing convention used elsewhere in this catalog::

        flat[t, e, n * K + k] = component k at element-node n

    where ``n_nodes = layout.n_nodes_per_element`` and
    ``K = layout.n_components_per_node``. The output dict keys are
    canonical apeGmsh names (e.g. ``"nodal_resisting_force_x"``) and
    each value is a ``(T, E, n_nodes)`` array suitable for one
    :class:`apeGmsh.results._slabs.ElementSlab` per component.
    """
    flat = np.asarray(flat)
    if flat.ndim != 3:
        raise ValueError(
            f"unflatten_nodal expects a 3-D flat array (T, E, "
            f"flat_size); got shape {flat.shape}."
        )
    T, E, flat_size = flat.shape
    expected = layout.flat_size_per_element
    if flat_size != expected:
        raise ValueError(
            f"flat_size {flat_size} does not match layout's "
            f"n_nodes_per_element * n_components_per_node = "
            f"{layout.n_nodes_per_element} * "
            f"{layout.n_components_per_node} = {expected}."
        )

    reshaped = flat.reshape(
        T, E,
        layout.n_nodes_per_element,
        layout.n_components_per_node,
    )
    return {
        name: np.ascontiguousarray(reshaped[:, :, :, k])
        for k, name in enumerate(layout.component_layout)
    }
