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

from typing import TYPE_CHECKING

from ._element_specs import _ELEM_REGISTRY

if TYPE_CHECKING:
    from .OpenSees import OpenSees


class _Elements:
    """Element formulations, geom transforms, and fix constraints."""

    def __init__(self, parent: "OpenSees") -> None:
        self._opensees = parent

    # ------------------------------------------------------------------
    # Geometric transformations (beam elements)
    # ------------------------------------------------------------------

    def add_geom_transf(
        self,
        name       : str,
        transf_type: str,
        *,
        vecxz      : list[float] | None = None,
        **extra,
    ) -> "_Elements":
        """
        Register a geometric transformation for beam elements.

        Parameters
        ----------
        name        : identifier referenced in :meth:`assign`
        transf_type : ``"Linear"``, ``"PDelta"``, or ``"Corotational"``
        vecxz       : 3-D only — the local x-z plane vector ``[vx, vy, vz]``
                      (the vector in the x-z plane, not the z-axis).
                      Ignored for 2-D models.

        Example
        -------
        ::

            # 2-D frame (no vecxz needed)
            g.opensees.elements.add_geom_transf("Cols", "PDelta")

            # 3-D frame
            g.opensees.elements.add_geom_transf(
                "Cols", "Linear", vecxz=[0, 0, 1],
            )
        """
        self._opensees._geom_transfs[name] = {
            "transf_type": transf_type,
            "vecxz"      : vecxz,
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

                      ``"nd"``      → ``g.opensees.materials.add_nd_material``
                      ``"uni"``     → ``g.opensees.materials.add_uni_material``
                      ``"section"`` → ``g.opensees.materials.add_section``
                      ``"none"``    → omit (beam elements with scalar props)
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
