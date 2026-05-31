"""Load definitions — pre-mesh user-facing intent.

Each :class:`LoadDef` subclass describes a load applied to a
geometric/PG/label target before meshing. The corresponding resolved
records (post-mesh) live in :mod:`apeGmsh.mesh.records._loads`; the
machinery that translates defs into records lives in
:mod:`apeGmsh.mesh._load_resolver`.

These classes have no Gmsh dependency, no session plumbing, and no
factory methods — pure data containers consumed by the resolver.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoadDef:
    """Base class for all load definitions."""
    kind: str
    target: object                  # part label, PG name, label name, mesh selection, or DimTag list
    pattern: str = "default"        # pattern grouping name
    name: str | None = None
    reduction: str = "tributary"    # "tributary" | "consistent"
    target_form: str = "nodal"      # "nodal" | "element"
    target_source: str = "auto"     # "auto" | "pg" | "label"


@dataclass
class PointLoadDef(LoadDef):
    """Concentrated force/moment at a node (or set of nodes).

    All targeted nodes receive the **same** force/moment.  Use
    :meth:`PointLoadDef.force_xyz` for translational forces and
    :meth:`PointLoadDef.moment_xyz` for moments (3D rotational DOFs).
    Either may be ``None``.
    """
    kind: str = field(init=False, default="point")
    force_xyz: tuple[float, float, float] | None = None
    moment_xyz: tuple[float, float, float] | None = None


@dataclass
class PointClosestLoadDef(PointLoadDef):
    """Concentrated load at the mesh node(s) closest to a coordinate.

    Coordinate-driven targeting (no PG/label required). At resolve time,
    the composite snaps ``xyz_request`` to the nearest mesh node — or, if
    ``tol`` is given, to every node within that radius. Pass ``within``
    (PG/label/part/DimTag list) to restrict the candidate node pool.

    The actual snap distance is written back to ``snap_distance`` after
    :meth:`LoadsComposite.resolve`, so it surfaces in ``summary()``.
    """
    kind: str = field(init=False, default="point_closest")
    xyz_request: tuple[float, float, float] = (0.0, 0.0, 0.0)
    within: object | None = None
    within_source: str = "auto"
    tol: float | None = None
    snap_distance: float | None = None


@dataclass
class LineLoadDef(LoadDef):
    """Distributed load along a 1-D entity (curve / beam element).

    Three ways to specify the load vector:

    * ``magnitude`` + ``direction`` — scalar magnitude (force per unit
      length) and a direction vector or axis name (``"x"``, ``"y"``,
      ``"z"``).
    * ``q_xyz`` — explicit ``(qx, qy, qz)`` force-per-length vector.
    * ``normal=True`` + ``away_from=(x0, y0, z0)`` — pressure
      perpendicular to each edge, in the plane of the loaded curves
      (any plane; fitted from the geometry).  The in-plane normal is
      sign-flipped per edge so it points away from ``away_from``;
      ``magnitude`` is then force per unit length along that normal.

    ``magnitude`` may be a constant float **or** a callable
    ``q(xyz) -> float`` evaluated per edge midpoint (spatially varying
    line load); the resolver in :mod:`apeGmsh.mesh._load_resolver`
    handles both.
    """
    kind: str = field(init=False, default="line")
    magnitude: object = 0.0                 # float | Callable[[xyz], float]
    direction: object = (0.0, 0.0, -1.0)   # tuple or "x"/"y"/"z"
    q_xyz: tuple[float, float, float] | None = None
    normal: bool = False
    away_from: tuple[float, float, float] | None = None


@dataclass
class SurfaceLoadDef(LoadDef):
    """Pressure or traction on a 2-D entity.

    * ``normal=True``: scalar pressure perpendicular to the face
      (positive into the face).
    * ``normal=False``: vector traction in the given direction.
    """
    kind: str = field(init=False, default="surface")
    magnitude: float = 0.0
    normal: bool = True
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)


@dataclass
class GravityLoadDef(LoadDef):
    """Body load from gravity = ρ·g over a volume.

    If ``density`` is ``None``, the solver bridge is expected to read
    density from the assigned material/section.
    """
    kind: str = field(init=False, default="gravity")
    g: tuple[float, float, float] = (0.0, 0.0, -9.81)
    density: float | None = None


@dataclass
class BodyLoadDef(LoadDef):
    """Generic per-volume body force vector (force per unit volume)."""
    kind: str = field(init=False, default="body")
    force_per_volume: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class FaceLoadDef(LoadDef):
    """Concentrated force/moment at face centroid, distributed to face nodes.

    ``force_xyz`` is split equally among all face nodes (``F / N``).
    ``moment_xyz`` is converted to statically equivalent nodal forces
    via a least-norm distribution such that ``Sum(r_i x f_i) = M`` and
    ``Sum(f_i) = 0``.

    A scalar ``magnitude`` (total Newtons, NOT pressure) can be combined
    with either ``normal=True`` or an explicit ``direction`` to produce
    the equivalent ``force_xyz`` without manually computing the face
    normal.  Sign convention: ``magnitude * direction_unit`` always —
    i.e. ``+magnitude`` with ``normal=True`` acts along the
    area-weighted average outward normal ``+n_avg`` (and
    ``-magnitude`` flips it, matching :class:`SurfaceLoadDef`'s
    "into-face" pressure when desired).  Composes with ``moment_xyz``;
    combining with ``force_xyz`` is an error.

    Use this instead of a reference node + coupling when you only need
    to apply a load to a face without structural coupling to another
    element.
    """
    kind: str = field(init=False, default="face_load")
    force_xyz: tuple[float, float, float] | None = None
    moment_xyz: tuple[float, float, float] | None = None
    magnitude: float = 0.0
    normal: bool = False
    direction: tuple[float, float, float] | None = None


@dataclass
class FaceSPDef(LoadDef):
    """Prescribed displacement/rotation at face centroid, mapped to face nodes.

    Maps a rigid-body motion at the face centroid to per-node
    displacements using ``u_i = disp_xyz + rot_xyz x r_i``.

    When ``disp_xyz``, ``rot_xyz``, and ``magnitude`` are all None /
    zero, the result is a homogeneous fix.

    A scalar ``magnitude`` (displacement, in mesh length units) can be
    combined with ``normal=True`` or an explicit ``direction`` to
    derive the centroid translation without computing the face normal
    by hand.  Sign convention matches :class:`FaceLoadDef`: total =
    ``magnitude * unit_direction`` along ``+n_avg`` (or the normalised
    ``direction``).  Composes with ``rot_xyz``; combining with
    ``disp_xyz`` is an error.

    Parameters
    ----------
    dofs : list[int]
        Restraint mask — ``1`` for constrained DOFs, ``0`` for free.
    disp_xyz : tuple or None
        Prescribed translation at the face centroid.
    rot_xyz : tuple or None
        Prescribed rotation about the face centroid.
    magnitude : float
        Scalar centroid translation, routed via ``normal``/``direction``.
    normal : bool
        When True, use the area-weighted face normal as the direction.
    direction : tuple or None
        Explicit unit direction (auto-normalised); mutually exclusive
        with ``normal=True``.
    """
    kind: str = field(init=False, default="face_sp")
    dofs: list[int] = field(default_factory=lambda: [1, 1, 1])
    disp_xyz: tuple[float, float, float] | None = None
    rot_xyz: tuple[float, float, float] | None = None
    magnitude: float = 0.0
    normal: bool = False
    direction: tuple[float, float, float] | None = None


@dataclass
class PointSPDef(LoadDef):
    """Prescribed displacement/rotation applied directly at node(s).

    Each targeted node receives one ``SPRecord`` per DOF marked in
    ``dofs`` (1 = constrained, 0 = free). ``values`` gives the prescribed
    value per DOF index (aligned with ``dofs``); ``None`` = homogeneous
    (all zero). Authored via ``g.displacements.point`` (ADR 0050).

    Unlike :class:`FaceSPDef` there is no centroid / rigid-body mapping —
    the value is applied verbatim at every targeted node.
    """
    kind: str = field(init=False, default="point_sp")
    dofs: list[int] = field(default_factory=lambda: [1, 1, 1])
    values: tuple[float, ...] | None = None


__all__ = [
    "LoadDef",
    "PointLoadDef",
    "PointClosestLoadDef",
    "LineLoadDef",
    "SurfaceLoadDef",
    "GravityLoadDef",
    "BodyLoadDef",
    "FaceLoadDef",
    "FaceSPDef",
    "PointSPDef",
]
