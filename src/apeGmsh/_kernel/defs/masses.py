"""Mass definitions — pre-mesh user-facing intent.

Each :class:`MassDef` subclass describes a mass applied to a
geometric/PG/label target before meshing. The corresponding resolved
records (post-mesh) live in :mod:`apeGmsh.mesh.records._masses`; the
machinery that translates defs into records lives in
:mod:`apeGmsh.mesh._mass_resolver`.

These classes have no Gmsh dependency, no session plumbing, and no
factory methods — pure data containers consumed by the resolver.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MassDef:
    """Base class for all mass definitions.

    ``dofs`` masks which translational DOFs (subset of ``{1, 2, 3}``)
    receive the resolved mass.  ``None`` (default) applies mass to all
    three translational components.  Rotational DOFs (4, 5, 6) are
    controlled separately via ``rotational=(Ixx, Iyy, Izz)`` on the
    individual def subclasses — ``dofs`` deliberately does not cover
    them because ``density × volume`` (and the line / area analogues)
    cannot derive rotational inertia without element-shape awareness.
    """
    kind: str
    target: object              # part label, PG name, label name, mesh selection, or DimTag list
    name: str | None = None
    reduction: str = "lumped"   # "lumped" | "consistent"
    target_source: str = "auto" # "auto" | "pg" | "label"
    dofs: list[int] | None = None   # None → translational (1, 2, 3)


@dataclass
class PointMassDef(MassDef):
    """Concentrated mass at a node (or set of nodes).

    The same scalar mass (and optional rotational inertia) is applied
    to every node in the target.  Useful for representing equipment,
    point fixtures, or any localised lumped mass.
    """
    kind: str = field(init=False, default="point")
    mass: float = 0.0                          # translational mass per node
    rotational: tuple[float, float, float] | None = None  # (Ixx, Iyy, Izz)


@dataclass
class LineMassDef(MassDef):
    """Distributed line mass — linear density along curve(s).

    ``linear_density`` is in mass per unit length (e.g. kg/m).  The
    total mass per curve is ``linear_density × curve_length`` and is
    distributed to the curve's mesh nodes.

    ``rotational`` (optional) attaches a fixed rotational inertia
    triple ``(Ixx, Iyy, Izz)`` to **every** node receiving mass from
    this def.  apeGmsh does not derive rotational inertia from
    ``linear_density`` — the user supplies the value.  Useful when an
    ndf=6 model needs rotational inertia along an edge (e.g. cable
    with cross-sectional moment of inertia).
    """
    kind: str = field(init=False, default="line")
    linear_density: float = 0.0
    rotational: tuple[float, float, float] | None = None


@dataclass
class SurfaceMassDef(MassDef):
    """Distributed surface mass — areal density on face(s).

    ``areal_density`` is in mass per unit area (e.g. kg/m²).  The
    total mass per face is ``areal_density × face_area`` distributed
    to the face's mesh nodes.

    ``rotational`` (optional) attaches a fixed rotational inertia
    triple ``(Ixx, Iyy, Izz)`` to **every** node receiving mass from
    this def.  apeGmsh does not derive rotational inertia from
    ``areal_density`` — the user supplies the value.
    """
    kind: str = field(init=False, default="surface")
    areal_density: float = 0.0
    rotational: tuple[float, float, float] | None = None
    derive_rotational: bool = False


@dataclass
class VolumeMassDef(MassDef):
    """Distributed volume mass — material density on volume(s).

    ``density`` is in mass per unit volume (e.g. kg/m³).  The total
    mass per element is ``density × element_volume`` distributed to
    the element's nodes.

    ``rotational`` (optional) attaches a fixed rotational inertia
    triple ``(Ixx, Iyy, Izz)`` to **every** node receiving mass from
    this def.  apeGmsh does not derive rotational inertia from
    ``density × volume`` — that would need a moment-of-inertia
    integration over the element, which is element-shape aware.
    Until that's implemented, this kwarg lets the user attach a
    pre-computed rotational inertia uniformly.

    ``derive_rotational`` (optional, default ``False``) instead
    *computes* per-node rotational inertia from the element shape
    functions: ``I_xx^(I) = ∫ρ N_I (y²+z²) dV − M_I (y_I²+z_I²)``
    (and cyclic), the about-node parallel-axis form whose assembled
    sum reproduces the continuum rigid-rotation kinetic energy
    exactly.  Requires ``reduction='consistent'`` and is mutually
    exclusive with a fixed ``rotational=`` tuple.  Per-node emitted
    values may be negative (they are parallel-axis corrections, not
    standalone inertias); the physical guarantee is on the assembled
    total.

    Note
    ----
    The user is responsible for setting the OpenSees material's
    ``rho=0`` (or equivalent in other solvers) to avoid double
    counting.  This composite always emits explicit nodal mass
    via ``ops.mass(...)`` commands.
    """
    kind: str = field(init=False, default="volume")
    density: float = 0.0
    rotational: tuple[float, float, float] | None = None
    derive_rotational: bool = False


__all__ = [
    "MassDef",
    "PointMassDef",
    "LineMassDef",
    "SurfaceMassDef",
    "VolumeMassDef",
]
