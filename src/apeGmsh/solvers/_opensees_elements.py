"""
_Elements — OpenSees element declaration sub-composite.

Accessed via ``g.opensees.elements``.  Groups the three things you do
on a per-physical-group basis before ``build()``:

* ``add_geom_transf`` — geometric transformations for beam elements
* ``assign``          — assign an OpenSees element type to a PG
                        (renamed from ``assign_element``)
* ``fix``             — homogeneous single-point constraints
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ._element_specs import _ELEM_REGISTRY

if TYPE_CHECKING:
    from .OpenSees import OpenSees


Vec3 = tuple[float, float, float]


def _normalize3(v) -> Vec3:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError(f"zero-length vector: {(x, y, z)}")
    return (x / n, y / n, z / n)


class _Elements:
    """Element formulations, geom transforms, and fix constraints."""

    def __init__(self, parent: "OpenSees") -> None:
        self._opensees = parent

    # ------------------------------------------------------------------
    # Geometric transformations (beam elements)
    # ------------------------------------------------------------------

    @staticmethod
    def vecxz(
        axis    : Vec3 | list[float],
        local_z : Vec3 | list[float] = (0.0, 0.0, 1.0),
        roll_deg: float = 0.0,
    ) -> Vec3:
        """
        Compute a valid ``vecxz`` for :meth:`add_geom_transf` given the
        beam axis and a reference local-z direction, with optional
        section rotation about the beam axis.

        Convention
        ----------
        At ``roll_deg = 0`` the returned vecxz equals ``local_z``.
        ``roll_deg`` rotates the section about ``axis`` via the
        right-hand rule (positive angle → right-handed rotation).
        The user is responsible for supplying section properties
        (``Iy``, ``Iz``, ``J``) consistent with this frame.

        Parameters
        ----------
        axis     : beam direction (local-x); need not be unit length.
        local_z  : reference direction for local-z when ``roll_deg=0``.
                   Defaults to global +Z.
        roll_deg : section rotation about ``axis`` in degrees.

        Returns
        -------
        tuple[float, float, float] : a unit vector suitable for
        ``add_geom_transf(vecxz=...)``.

        Raises
        ------
        ValueError : if ``axis`` or ``local_z`` is zero-length, or if
                     ``local_z`` is collinear with ``axis`` (no unique
                     x-z plane exists).

        Example
        -------
        ::

            # Horizontal beam, gravity downward, strong axis vertical
            vxz = g.opensees.elements.vecxz(axis=(1, 0, 0))
            # → (0.0, 0.0, 1.0)

            # Same beam, section rotated 90° (weak axis now vertical)
            vxz = g.opensees.elements.vecxz(axis=(1, 0, 0), roll_deg=90)
            # → (0.0, -1.0, 0.0)
        """
        ax = _normalize3(axis)
        lz = _normalize3(local_z)
        if abs(ax[0] * lz[0] + ax[1] * lz[1] + ax[2] * lz[2]) > 1.0 - 1e-9:
            raise ValueError(
                "local_z is collinear with axis; no unique x-z plane. "
                f"axis={ax}, local_z={lz}"
            )
        if roll_deg == 0.0:
            return lz
        # Rodrigues' rotation: rotate lz about ax by roll_deg
        th = math.radians(roll_deg)
        c, s = math.cos(th), math.sin(th)
        kx, ky, kz = ax
        vx, vy, vz = lz
        # k × v
        cx = ky * vz - kz * vy
        cy = kz * vx - kx * vz
        cz = kx * vy - ky * vx
        # k · v
        kdv = kx * vx + ky * vy + kz * vz
        rx = vx * c + cx * s + kx * kdv * (1 - c)
        ry = vy * c + cy * s + ky * kdv * (1 - c)
        rz = vz * c + cz * s + kz * kdv * (1 - c)
        return (rx, ry, rz)

    def add_geom_transf(
        self,
        name       : str,
        transf_type: str,
        *,
        vecxz      : list[float] | None = None,
        csys       : object | None = None,
        roll_deg   : float = 0.0,
        **extra,
    ) -> "_Elements":
        """
        Register a geometric transformation for beam elements.

        Two ways to specify the local-z orientation in 3-D:

        * **Explicit** ``vecxz=[vx, vy, vz]`` — single Cartesian vector
          shared by every beam under this transform.  Same as before.
        * **Coordinate system** ``csys=<Cartesian|Cylindrical|Spherical>``
          — orientation rule evaluated per element from the gmsh tangent
          and the CS triad.  Curved members (ring beams, dome ribs) get a
          per-element ``vecxz`` automatically; the build step emits one
          ``geomTransf`` per distinct vecxz it observes.

        ``vecxz`` and ``csys`` are mutually exclusive.  In 2-D both are
        ignored.

        Parameters
        ----------
        name        : identifier referenced in :meth:`assign`
        transf_type : ``"Linear"``, ``"PDelta"``, or ``"Corotational"``
        vecxz       : 3-D only — local x-z plane vector ``[vx, vy, vz]``
        csys        : 3-D only — :class:`Cartesian`, :class:`Cylindrical`,
                      or :class:`Spherical` from
                      :mod:`apeGmsh.solvers._opensees_csys`.
        roll_deg    : 3-D only — additional section rotation about the
                      beam tangent (right-hand rule), applied after the
                      CS rule.  Only meaningful with ``csys=``.

        Example
        -------
        ::

            from apeGmsh.solvers import Cartesian, Cylindrical

            # 2-D frame
            g.opensees.elements.add_geom_transf("Cols", "PDelta")

            # 3-D frame, explicit vector
            g.opensees.elements.add_geom_transf(
                "Cols", "Linear", vecxz=[0, 0, 1],
            )

            # 3-D frame, "Z up" rule (default)
            g.opensees.elements.add_geom_transf(
                "Beams", "Linear", csys=Cartesian(),
            )

            # 3-D ring beam — radial strong axis
            g.opensees.elements.add_geom_transf(
                "RingBeam", "Linear",
                csys=Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1)),
            )
        """
        if vecxz is not None and csys is not None:
            raise ValueError(
                f"add_geom_transf({name!r}): pass either vecxz= or csys=, "
                f"not both."
            )
        if roll_deg != 0.0 and csys is None:
            raise ValueError(
                f"add_geom_transf({name!r}): roll_deg is only meaningful "
                f"with csys=; supply a csys or pre-rotate the explicit "
                f"vecxz with elements.vecxz(..., roll_deg=...)."
            )
        self._opensees._geom_transfs[name] = {
            "transf_type": transf_type,
            "vecxz"      : vecxz,
            "csys"       : csys,
            "roll_deg"   : float(roll_deg),
            **extra,
        }
        self._opensees._log(
            f"add_geom_transf({name!r}, {transf_type!r})"
        )
        return self

    # ------------------------------------------------------------------
    # Element assignment
    # ------------------------------------------------------------------

    def assign(
        self,
        pg_name    : str,
        ops_type   : str,
        *,
        material   : str | None = None,
        geom_transf: str | None = None,
        dim        : int | None = None,
        **extra,
    ) -> "_Elements":
        """
        Declare that every mesh element in physical group *pg_name*
        should be written as an OpenSees *ops_type* element.

        Validation is deferred to :meth:`OpenSees.build` so multiple
        assignments can be chained before gmsh is queried.

        Parameters
        ----------
        pg_name     : physical-group name
        ops_type    : OpenSees element type (must be in the element registry)
        material    : material / section name from the matching registry:

                      ``"nd"``      -> ``g.opensees.materials.add_nd_material``
                      ``"uni"``     -> ``g.opensees.materials.add_uni_material``
                      ``"section"`` -> ``g.opensees.materials.add_section``
                      ``"none"``    -> omit (beam elements with scalar props)
        geom_transf : name from :meth:`add_geom_transf` — required for
                      beam elements (``elasticBeamColumn`` etc.)
        dim         : physical-group dimension hint for name disambiguation
        **extra     : element-specific scalar parameters.  Keys must
                      match the slot names for *ops_type* (see
                      ``_ELEM_REGISTRY``).

        Example
        -------
        ::

            g.opensees.elements.assign(
                "Body", "FourNodeTetrahedron",
                material="Concrete",
                bodyForce=[0, 0, -9.81 * 2400],
            )

            g.opensees.elements.assign(
                "Diags", "corotTruss",
                material="Steel", A=3.14e-4,
            )

            g.opensees.elements.assign(
                "Cols", "elasticBeamColumn",
                geom_transf="ColTransf",
                A=0.04, E=200e9, G=77e9,
                Jx=1e-4, Iy=2e-4, Iz=2e-4,
            )
        """
        if ops_type not in _ELEM_REGISTRY:
            raise ValueError(
                f"elements.assign: unknown ops_type {ops_type!r}. "
                f"Supported: {sorted(_ELEM_REGISTRY)}"
            )
        self._opensees._elem_assignments[pg_name] = {
            "ops_type"   : ops_type,
            "material"   : material,
            "geom_transf": geom_transf,
            "dim"        : dim,
            "extra"      : extra,
        }
        self._opensees._log(
            f"assign({pg_name!r}, {ops_type!r}, "
            f"material={material!r}, geom_transf={geom_transf!r})"
        )
        return self

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def fix(
        self,
        pg_name: str,
        *,
        dofs   : list[int],
        dim    : int | None = None,
    ) -> "_Elements":
        """
        Apply homogeneous single-point constraints to every node in a
        physical group.

        Parameters
        ----------
        pg_name : physical-group name
        dofs    : restraint mask of length ``ndf`` — ``1`` fixed, ``0`` free
        dim     : physical-group dimension hint for name disambiguation

        Example
        -------
        ::

            g.opensees.elements.fix("BasePlate", dofs=[1, 1, 1])
            g.opensees.elements.fix("PinnedEnd", dofs=[1, 1, 1, 0, 0, 0])
        """
        if len(dofs) != self._opensees._ndf:
            raise ValueError(
                f"elements.fix({pg_name!r}): len(dofs)={len(dofs)} "
                f"!= ndf={self._opensees._ndf}"
            )
        self._opensees._bcs[pg_name] = {"dofs": list(dofs), "dim": dim}
        self._opensees._log(f"fix({pg_name!r}, dofs={dofs})")
        return self
