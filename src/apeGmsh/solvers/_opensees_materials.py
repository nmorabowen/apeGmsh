"""
_Materials — OpenSees material registry sub-composite.

Accessed via ``g.opensees.materials``.  Holds the three material-like
registries that OpenSees keeps separate:

* ``add_nd_material`` — nDMaterial (solids: tetrahedron, brick, quad, ...)
* ``add_uni_material`` — uniaxialMaterial (trusses, zeroLength springs)
* ``add_section`` — section (shells)

Geometric transformations live on the sibling ``g.opensees.elements``
sub-composite because they are attached to element declarations, not
to material properties.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .OpenSees import OpenSees


class _Materials:
    """OpenSees material / section registries."""

    def __init__(self, parent: "OpenSees") -> None:
        self._opensees = parent

    # ------------------------------------------------------------------
    # nDMaterial
    # ------------------------------------------------------------------

    def add_nd_material(
        self, name: str, ops_type: str, **params
    ) -> "_Materials":
        """
        Register an OpenSees ``nDMaterial``.

        Used by solid elements: ``FourNodeTetrahedron``, ``stdBrick``,
        ``quad``, ``tri31``, ``SSPquad``, ``SSPbrick``, ``bbarBrick``.

        Parameters
        ----------
        name     : identifier referenced in ``g.opensees.elements.assign``
        ops_type : e.g. ``"ElasticIsotropic"``, ``"J2Plasticity"``,
                   ``"DruckerPrager"``
        **params : forwarded verbatim to the OpenSees material command
                   in declaration order.

        Example
        -------
        ::

            g.opensees.materials.add_nd_material(
                "Soil", "DruckerPrager",
                K=80e6, G=60e6, sigmaY=20e3,
                rho=0.0, rhoBar=0.0, Kinf=0.0, Ko=0.0,
                delta1=0.0, delta2=0.0, H=0.0, theta=0.0,
            )
        """
        self._opensees._nd_materials[name] = {
            "ops_type": ops_type, "params": params,
        }
        self._opensees._log(f"add_nd_material({name!r}, {ops_type!r})")
        return self

    # ------------------------------------------------------------------
    # uniaxialMaterial
    # ------------------------------------------------------------------

    def add_uni_material(
        self, name: str, ops_type: str, **params
    ) -> "_Materials":
        """
        Register an OpenSees ``uniaxialMaterial``.

        Used by truss elements (``truss``, ``corotTruss``) and
        ``zeroLength`` spring elements.

        Parameters
        ----------
        name     : identifier referenced in ``g.opensees.elements.assign``
        ops_type : e.g. ``"Steel01"``, ``"Elastic"``, ``"ENT"``
        **params : forwarded verbatim to the material command.

        Example
        -------
        ::

            g.opensees.materials.add_uni_material(
                "Steel", "Steel01", Fy=250e6, E=200e9, b=0.01,
            )
        """
        self._opensees._uni_materials[name] = {
            "ops_type": ops_type, "params": params,
        }
        self._opensees._log(f"add_uni_material({name!r}, {ops_type!r})")
        return self

    # ------------------------------------------------------------------
    # section
    # ------------------------------------------------------------------

    def add_section(
        self, name: str, section_type: str, **params
    ) -> "_Materials":
        """
        Register an OpenSees ``section``.

        Used by shell elements (``ShellMITC4``, ``ShellDKGQ``,
        ``ASDShellQ4``).  The most common shell section is
        ``ElasticMembranePlateSection``.

        Parameters
        ----------
        name         : identifier referenced in ``g.opensees.elements.assign``
        section_type : e.g. ``"ElasticMembranePlateSection"``
        **params     : forwarded verbatim, e.g. ``E, nu, h, rho``.

        Example
        -------
        ::

            g.opensees.materials.add_section(
                "Slab", "ElasticMembranePlateSection",
                E=30e9, nu=0.2, h=0.2, rho=2400,
            )
        """
        self._opensees._sections[name] = {
            "section_type": section_type, "params": params,
        }
        self._opensees._log(f"add_section({name!r}, {section_type!r})")
        return self
