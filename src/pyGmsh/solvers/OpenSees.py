from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import gmsh
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase

from ._element_specs import (
    Tag, DimTag, _ETYPE_INFO, _DEFAULTS,
    _ElemSpec, _ELEM_REGISTRY, _render_tcl, _render_py,
)


# ---------------------------------------------------------------------------
# OpenSees composite class
# ---------------------------------------------------------------------------

class OpenSees:
    """
    OpenSees bridge composite attached to a ``pyGmsh`` instance as
    ``g.opensees``.

    Translates a gmsh mesh (nodes, elements, physical groups) into an
    OpenSees finite-element model in three stages:

    **1 — Declare**  (before ``build()``)

    Register materials in the correct registry, assign element formulations
    to physical groups, declare boundary conditions and load patterns::

        (g.opensees
           .set_model(ndm=3, ndf=3)
           # nD material for solid elements
           .add_nd_material("Concrete", "ElasticIsotropic",
                            E=30e9, nu=0.2, rho=2400)
           # element assignment — topology and material family are validated
           .assign_element("Body", "FourNodeTetrahedron",
                           material="Concrete",
                           bodyForce=[0, 0, -2400*9.81])
           .fix("Base",    dofs=[1, 1, 1])
           .add_nodal_load("Wind", "TopFace", force=[1e4, 0, 0]))

    For beam elements a geometric transformation is required::

        (g.opensees
           .set_model(ndm=3, ndf=6)
           .add_geom_transf("ColTransf", "Linear", vecxz=[0, 0, 1])
           .assign_element("Columns", "elasticBeamColumn",
                           geom_transf="ColTransf",
                           A=0.04, E=200e9, G=77e9,
                           Jx=1e-4, Iy=2e-4, Iz=2e-4))

    For truss/corotTruss elements a uniaxial material is required::

        (g.opensees
           .add_uni_material("Steel", "Steel01",
                             Fy=250e6, E=200e9, b=0.01)
           .assign_element("Diagonals", "truss",
                           material="Steel", A=0.002))

    **2 — Build**  (after ``g.mesh.generate()`` and all declarations)::

        g.opensees.build()

    **3 — Inspect and export**::

        print(g.opensees.summary())
        g.opensees.export_tcl("model.tcl").export_py("model.py")

    Parameters
    ----------
    parent : _SessionBase
        The owning instance.
    """

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent
        self._ndm: int = 3
        self._ndf: int = 3

        # ── material registries ────────────────────────────────────────────
        self._nd_materials : dict[str, dict] = {}  # name → {ops_type, params}
        self._uni_materials: dict[str, dict] = {}
        self._sections     : dict[str, dict] = {}
        self._geom_transfs : dict[str, dict] = {}  # name → {transf_type, vecxz, ...}

        # ── element assignments ────────────────────────────────────────────
        # pg_name → {ops_type, material, geom_transf, dim, extra}
        self._elem_assignments: dict[str, dict] = {}

        # ── constraints and loads ──────────────────────────────────────────
        self._bcs          : dict[str, list[int]] = {}   # pg_name → dof mask
        self._load_patterns: dict[str, list[dict]] = {}  # name → [load_def, ...]

        # ── post-build state ───────────────────────────────────────────────
        self._built        = False
        self._node_map: dict[int, int] = {}   # gmsh tag → ops id (1-based)
        self._nodes_df    = pd.DataFrame()
        self._elements_df = pd.DataFrame()
        # sequential integer tags assigned at build()
        self._nd_mat_tags  : dict[str, int] = {}
        self._uni_mat_tags : dict[str, int] = {}
        self._sec_tags     : dict[str, int] = {}
        self._transf_tags  : dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[OpenSees] {msg}")

    def _require_built(self, method: str) -> None:
        if not self._built:
            raise RuntimeError(
                f"OpenSees.{method}() must be called after build()."
            )

    def _all_pg_names(self) -> list[str]:
        return [
            gmsh.model.getPhysicalName(d, t)
            for d, t in gmsh.model.getPhysicalGroups()
        ]

    def _find_pg(
        self,
        name        : str,
        expected_dim: int | None = None,
    ) -> DimTag:
        """
        Return ``(dim, tag)`` for the named physical group.

        Raises ``ValueError`` if the name does not exist or is ambiguous
        across multiple dimensions (pass ``expected_dim`` to disambiguate).
        """
        matches = [
            (d, t) for d, t in gmsh.model.getPhysicalGroups()
            if gmsh.model.getPhysicalName(d, t) == name
        ]
        if not matches:
            raise ValueError(
                f"Physical group {name!r} not found in the model.\n"
                f"Available: {self._all_pg_names()}"
            )
        if len(matches) == 1:
            return matches[0]
        # ambiguous — try to resolve with expected_dim
        if expected_dim is not None:
            dim_matches = [(d, t) for d, t in matches if d == expected_dim]
            if len(dim_matches) == 1:
                return dim_matches[0]
        dims = sorted(d for d, _ in matches)
        raise ValueError(
            f"Ambiguous physical-group name {name!r}: "
            f"exists in dimensions {dims}. "
            f"Pass `dim=<dim>` to assign_element / fix / add_nodal_load "
            f"to resolve the ambiguity."
        )

    def _nodes_for_pg(self, pg_name: str, dim: int | None = None) -> list[int]:
        """Return OpenSees node IDs for all mesh nodes in a physical group.

        After Numberer.renumber() syncs IDs back to the Gmsh model,
        ``getNodesForPhysicalGroup`` already returns solver-ready IDs.
        We still filter through ``_node_map`` when it is populated
        (for backward compatibility with workflows that build their
        own mapping), but fall back to the raw tags when the map is
        empty — which is the expected path after renumbering.
        """
        pg_dim, pg_tag = self._find_pg(pg_name, dim)
        raw, _ = gmsh.model.mesh.getNodesForPhysicalGroup(pg_dim, pg_tag)
        if self._node_map:
            return [
                self._node_map[int(t)]
                for t in raw
                if int(t) in self._node_map
            ]
        # Post-renumber path: Gmsh tags ARE the solver IDs
        return sorted(int(t) for t in raw)

    # ------------------------------------------------------------------
    # Stage 1 — Declarations
    # ------------------------------------------------------------------

    def set_model(self, *, ndm: int = 3, ndf: int = 3) -> OpenSees:
        """
        Set spatial dimension and DOFs per node.

        Parameters
        ----------
        ndm : 2 or 3
        ndf : DOFs per node.

              Typical combinations:

              ``ndm=2, ndf=2``  2-D solid (ux, uy)
              ``ndm=2, ndf=3``  2-D frame (ux, uy, θz)
              ``ndm=3, ndf=3``  3-D solid (ux, uy, uz)
              ``ndm=3, ndf=6``  3-D frame/shell (ux, uy, uz, θx, θy, θz)
        """
        self._ndm = ndm
        self._ndf = ndf
        self._log(f"set_model(ndm={ndm}, ndf={ndf})")
        return self

    # ── material registries ────────────────────────────────────────────────

    def add_nd_material(
        self, name: str, ops_type: str, **params
    ) -> OpenSees:
        """
        Register an OpenSees ``nDMaterial``.

        Used by solid elements: ``FourNodeTetrahedron``, ``stdBrick``,
        ``quad``, ``tri31``, ``SSPquad``, ``SSPbrick``, ``bbarBrick``.

        Parameters
        ----------
        name     : identifier referenced in ``assign_element``
        ops_type : e.g. ``"ElasticIsotropic"``, ``"J2Plasticity"``,
                   ``"DruckerPrager"``
        **params : forwarded verbatim to the OpenSees material command
                   in declaration order.

        Example
        -------
        ::

            g.opensees.add_nd_material("Soil", "DruckerPrager",
                                       K=80e6, G=60e6, sigmaY=20e3,
                                       rho=0.0, rhoBar=0.0,
                                       Kinf=0.0, Ko=0.0,
                                       delta1=0.0, delta2=0.0,
                                       H=0.0, theta=0.0)
        """
        self._nd_materials[name] = {"ops_type": ops_type, "params": params}
        self._log(f"add_nd_material({name!r}, {ops_type!r})")
        return self

    def add_uni_material(
        self, name: str, ops_type: str, **params
    ) -> OpenSees:
        """
        Register an OpenSees ``uniaxialMaterial``.

        Used by truss elements (``truss``, ``corotTruss``) and
        ``zeroLength`` spring elements.

        Parameters
        ----------
        name     : identifier referenced in ``assign_element``
        ops_type : e.g. ``"Steel01"``, ``"Elastic"``, ``"ENT"``
        **params : forwarded verbatim to the material command.

        Example
        -------
        ::

            g.opensees.add_uni_material("Steel", "Steel01",
                                        Fy=250e6, E=200e9, b=0.01)
        """
        self._uni_materials[name] = {"ops_type": ops_type, "params": params}
        self._log(f"add_uni_material({name!r}, {ops_type!r})")
        return self

    def add_section(
        self, name: str, section_type: str, **params
    ) -> OpenSees:
        """
        Register an OpenSees ``section``.

        Used by shell elements (``ShellMITC4``, ``ShellDKGQ``,
        ``ASDShellQ4``).  The most common shell section is
        ``ElasticMembranePlateSection``.

        Parameters
        ----------
        name         : identifier referenced in ``assign_element``
        section_type : e.g. ``"ElasticMembranePlateSection"``
        **params     : forwarded verbatim, e.g. ``E, nu, h, rho``.

        Example
        -------
        ::

            g.opensees.add_section("Slab", "ElasticMembranePlateSection",
                                   E=30e9, nu=0.2, h=0.2, rho=2400)
        """
        self._sections[name] = {"section_type": section_type, "params": params}
        self._log(f"add_section({name!r}, {section_type!r})")
        return self

    def add_geom_transf(
        self,
        name       : str,
        transf_type: str,
        *,
        vecxz      : list[float] | None = None,
        **extra,
    ) -> OpenSees:
        """
        Register a geometric transformation for beam elements.

        Parameters
        ----------
        name        : identifier referenced in ``assign_element``
        transf_type : ``"Linear"``, ``"PDelta"``, or ``"Corotational"``
        vecxz       : 3-D only — the local x-z plane vector ``[vx,vy,vz]``
                      (the vector in the x-z plane, not the z-axis).
                      Ignored for 2-D models.

        Example
        -------
        ::

            # 2-D frame (no vecxz needed)
            g.opensees.add_geom_transf("Cols", "PDelta")

            # 3-D frame
            g.opensees.add_geom_transf("Cols", "Linear", vecxz=[0, 0, 1])
        """
        self._geom_transfs[name] = {
            "transf_type": transf_type,
            "vecxz"      : vecxz,
            **extra,
        }
        self._log(f"add_geom_transf({name!r}, {transf_type!r})")
        return self

    # ── element assignment ─────────────────────────────────────────────────

    def assign_element(
        self,
        pg_name    : str,
        ops_type   : str,
        *,
        material   : str | None = None,
        geom_transf: str | None = None,
        dim        : int | None = None,
        **extra,
    ) -> OpenSees:
        """
        Declare that every mesh element in physical group *pg_name* should
        be written as an OpenSees *ops_type* element.

        Validation is deferred to ``build()`` so multiple assignments can
        be chained before gmsh is queried.

        Parameters
        ----------
        pg_name     : physical-group name
        ops_type    : OpenSees element type (must be in the element registry)
        material    : material / section name from the matching registry:

                      ``"nd"``      → ``add_nd_material``
                      ``"uni"``     → ``add_uni_material``
                      ``"section"`` → ``add_section``
                      ``"none"``    → omit (beam elements with scalar props)
        geom_transf : name from ``add_geom_transf`` — required for beam
                      elements (``elasticBeamColumn`` etc.)
        dim         : physical-group dimension hint for name disambiguation
        **extra     : element-specific scalar parameters.  Keys must match
                      the slot names for *ops_type* (see ``_ELEM_REGISTRY``).

        Example
        -------
        ::

            g.opensees.assign_element("Body", "FourNodeTetrahedron",
                                      material="Concrete",
                                      bodyForce=[0, 0, -9.81*2400])

            g.opensees.assign_element("Diags", "corotTruss",
                                      material="Steel", A=3.14e-4)

            g.opensees.assign_element("Cols",  "elasticBeamColumn",
                                      geom_transf="ColTransf",
                                      A=0.04, E=200e9, G=77e9,
                                      Jx=1e-4, Iy=2e-4, Iz=2e-4)
        """
        if ops_type not in _ELEM_REGISTRY:
            raise ValueError(
                f"assign_element: unknown ops_type {ops_type!r}. "
                f"Supported: {sorted(_ELEM_REGISTRY)}"
            )
        self._elem_assignments[pg_name] = {
            "ops_type"   : ops_type,
            "material"   : material,
            "geom_transf": geom_transf,
            "dim"        : dim,
            "extra"      : extra,
        }
        self._log(
            f"assign_element({pg_name!r}, {ops_type!r}, "
            f"material={material!r}, geom_transf={geom_transf!r})"
        )
        return self

    # ── constraints ───────────────────────────────────────────────────────

    def fix(
        self,
        pg_name: str,
        *,
        dofs   : list[int],
        dim    : int | None = None,
    ) -> OpenSees:
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

            g.opensees.fix("BasePlate", dofs=[1, 1, 1])          # 3-D solid
            g.opensees.fix("PinnedEnd", dofs=[1, 1, 1, 0, 0, 0]) # 3-D frame
        """
        if len(dofs) != self._ndf:
            raise ValueError(
                f"fix({pg_name!r}): len(dofs)={len(dofs)} != ndf={self._ndf}"
            )
        self._bcs[pg_name] = {"dofs": list(dofs), "dim": dim}
        self._log(f"fix({pg_name!r}, dofs={dofs})")
        return self

    # ── loads ─────────────────────────────────────────────────────────────

    def add_nodal_load(
        self,
        pattern_name: str,
        pg_name     : str,
        *,
        force       : list[float],
        dim         : int | None = None,
    ) -> OpenSees:
        """
        Schedule an equal nodal force on every node in a physical group.

        Multiple calls with the same *pattern_name* accumulate loads inside
        one OpenSees ``pattern Plain`` block.

        Parameters
        ----------
        pattern_name : load-pattern label
        pg_name      : physical-group name
        force        : force vector of length ``ndf``
        dim          : dimension hint for disambiguation

        Example
        -------
        ::

            g.opensees.add_nodal_load("Wind", "WindwardFace",
                                      force=[1e4, 0, 0])
        """
        if len(force) != self._ndf:
            raise ValueError(
                f"add_nodal_load: len(force)={len(force)} != ndf={self._ndf}"
            )
        if pattern_name not in self._load_patterns:
            self._load_patterns[pattern_name] = []
        self._load_patterns[pattern_name].append({
            "type"   : "nodal",
            "pg_name": pg_name,
            "force"  : list(force),
            "dim"    : dim,
        })
        self._log(
            f"add_nodal_load(pattern={pattern_name!r}, "
            f"pg={pg_name!r}, force={force})"
        )
        return self

    # ------------------------------------------------------------------
    # Stage 2 — Build
    # ------------------------------------------------------------------

    def build(self) -> OpenSees:
        """
        Extract the active gmsh mesh and construct all model tables.

        Performs full validation:

        * element types match the physical-group topology
        * material references point to the correct registry
        * geomTransf is declared for beam elements
        * ``ndf``/``ndm`` are compatible with each element spec
        * warns when higher-order gmsh nodes are downgraded to first-order

        Must be called after ``g.mesh.generate()`` and all declarations.
        """
        # ── 1. Global node numbering ─────────────────────────────────────
        # Use getNodes() with no args → returns the unique node cache.
        # getNodes(dim=-1, ..., includeBoundary=True) can return the
        # same physical node multiple times (once per entity it touches),
        # producing duplicates that break the model.
        raw_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
        if len(raw_tags) == 0:
            raise RuntimeError(
                "OpenSees.build(): no mesh nodes found — "
                "call g.mesh.generate() first."
            )
        coords_arr = np.array(coords_flat).reshape(-1, 3)
        self._node_map = {
            int(gt): ops_id
            for ops_id, gt in enumerate(raw_tags, start=1)
        }
        self._nodes_df = pd.DataFrame({
            'ops_id': np.arange(1, len(raw_tags) + 1, dtype=np.int64),
            'x'     : coords_arr[:, 0],
            'y'     : coords_arr[:, 1],
            'z'     : coords_arr[:, 2],
        }).set_index('ops_id')
        self._log(f"build(): {len(raw_tags)} nodes mapped")

        # ── 2. Sequential registry tags ──────────────────────────────────
        self._nd_mat_tags  = {n: i for i, n in enumerate(self._nd_materials,  start=1)}
        self._uni_mat_tags = {n: i for i, n in enumerate(self._uni_materials, start=1)}
        self._sec_tags     = {n: i for i, n in enumerate(self._sections,      start=1)}
        self._transf_tags  = {n: i for i, n in enumerate(self._geom_transfs,  start=1)}

        # ── 3. Elements per assigned physical group ──────────────────────
        elem_rows: list[dict] = []
        ops_elem_id = 1
        _warned_ho: set[str] = set()   # track high-order warnings per PG

        for pg_name, asgn in self._elem_assignments.items():
            ops_type   = asgn["ops_type"]
            mat_name   = asgn["material"]
            transf_name= asgn["geom_transf"]
            hint_dim   = asgn["dim"]
            extra      = asgn["extra"]
            spec       = _ELEM_REGISTRY[ops_type]  # already validated in assign_element

            # ── ndm / ndf check ──────────────────────────────────────────
            if self._ndm not in spec.ndm_ok:
                raise ValueError(
                    f"assign_element({pg_name!r}, {ops_type!r}): "
                    f"ndm={self._ndm} is not valid for this element type "
                    f"(allowed: {sorted(spec.ndm_ok)}). "
                    f"Call set_model() with the correct ndm."
                )
            if self._ndf not in spec.ndf_ok:
                raise ValueError(
                    f"assign_element({pg_name!r}, {ops_type!r}): "
                    f"ndf={self._ndf} is not valid for this element type "
                    f"(allowed: {sorted(spec.ndf_ok)}). "
                    f"Call set_model() with the correct ndf."
                )

            # ── material reference check ─────────────────────────────────
            mat_tag = sec_tag = None
            if spec.mat_family == "nd":
                if mat_name is None or mat_name not in self._nd_materials:
                    raise ValueError(
                        f"assign_element({pg_name!r}, {ops_type!r}): "
                        f"requires an nDMaterial; "
                        f"{mat_name!r} not found in add_nd_material registry. "
                        f"Available: {sorted(self._nd_materials)}"
                    )
                mat_tag = self._nd_mat_tags[mat_name]
            elif spec.mat_family == "uni":
                if mat_name is None or mat_name not in self._uni_materials:
                    raise ValueError(
                        f"assign_element({pg_name!r}, {ops_type!r}): "
                        f"requires a uniaxialMaterial; "
                        f"{mat_name!r} not found in add_uni_material registry. "
                        f"Available: {sorted(self._uni_materials)}"
                    )
                mat_tag = self._uni_mat_tags[mat_name]
            elif spec.mat_family == "section":
                if mat_name is None or mat_name not in self._sections:
                    raise ValueError(
                        f"assign_element({pg_name!r}, {ops_type!r}): "
                        f"requires a section; "
                        f"{mat_name!r} not found in add_section registry. "
                        f"Available: {sorted(self._sections)}"
                    )
                sec_tag = self._sec_tags[mat_name]
            # spec.mat_family == "none": no material needed

            # ── geomTransf check ─────────────────────────────────────────
            transf_tag = None
            if spec.needs_transf:
                if transf_name is None or transf_name not in self._geom_transfs:
                    raise ValueError(
                        f"assign_element({pg_name!r}, {ops_type!r}): "
                        f"requires a geomTransf; "
                        f"{transf_name!r} not found in add_geom_transf registry. "
                        f"Available: {sorted(self._geom_transfs)}"
                    )
                transf_tag = self._transf_tags[transf_name]

            # ── physical group lookup ────────────────────────────────────
            expected_dim = hint_dim if hint_dim is not None else spec.expected_pg_dim
            pg_dim, pg_tag = self._find_pg(pg_name, expected_dim)
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)

            # ── extract elements ─────────────────────────────────────────
            slots = spec.get_slots(self._ndm)

            for ent_tag in entity_tags:
                etypes, elem_tags_list, enodes_list = \
                    gmsh.model.mesh.getElements(dim=pg_dim, tag=ent_tag)

                for etype, elem_tags_arr, enodes in zip(
                    etypes, elem_tags_list, enodes_list
                ):
                    # topology compatibility check
                    if etype not in spec.gmsh_etypes:
                        raise ValueError(
                            f"assign_element({pg_name!r}, {ops_type!r}): "
                            f"physical group contains gmsh element type {etype} "
                            f"which is incompatible with {ops_type!r}. "
                            f"Expected gmsh types: {sorted(spec.gmsh_etypes)}. "
                            f"Check that the physical-group dimension and mesh "
                            f"algorithm match the element formulation."
                        )

                    n_corner, _      = _ETYPE_INFO[etype]
                    n_per            = len(enodes) // len(elem_tags_arr)
                    reorder          = spec.node_reorder.get(etype, tuple(range(n_corner)))

                    # high-order warning (issued once per PG)
                    if n_per > n_corner and pg_name not in _warned_ho:
                        warnings.warn(
                            f"OpenSees.build(): physical group {pg_name!r} "
                            f"contains {n_per}-node elements (gmsh type {etype}), "
                            f"but {ops_type!r} uses only {n_corner} corner nodes. "
                            f"Mid-side nodes are discarded — the exported model "
                            f"is first-order only.",
                            UserWarning, stacklevel=2,
                        )
                        _warned_ho.add(pg_name)

                    for k in range(len(elem_tags_arr)):
                        raw_ns    = enodes[k * n_per:(k + 1) * n_per]
                        corner_ns = [raw_ns[i] for i in reorder]
                        ops_nodes = tuple(
                            self._node_map[int(ni)]
                            for ni in corner_ns
                            if int(ni) in self._node_map
                        )
                        if len(ops_nodes) != n_corner:
                            continue  # unmapped node — skip silently

                        elem_rows.append({
                            'ops_id'    : ops_elem_id,
                            'gmsh_id'   : int(elem_tags_arr[k]),
                            'ops_type'  : ops_type,
                            'pg_name'   : pg_name,
                            'mat_name'  : mat_name,
                            'mat_tag'   : mat_tag,
                            'sec_tag'   : sec_tag,
                            'transf_tag': transf_tag,
                            'n_nodes'   : n_corner,
                            'nodes'     : ops_nodes,
                            'slots'     : slots,
                            'extra'     : extra,
                        })
                        ops_elem_id += 1

        # ── 3b. Filter to model-active nodes only ─────────────────────
        # Gmsh's getNodes(dim=-1) returns ALL mesh nodes including
        # geometric vertex points that are not part of any element.
        # We prune down to nodes referenced by at least one element
        # OR by a boundary-condition / load physical group, then
        # renumber sequentially.
        reverse_map = {v: k for k, v in self._node_map.items()}
        connected_gmsh: set[int] = set()
        for row in elem_rows:
            for oid in row['nodes']:
                connected_gmsh.add(reverse_map[oid])

        # Also keep nodes needed by boundary conditions and loads —
        # these may sit on physical groups (e.g. base supports at
        # dim=0) that have no assigned OpenSees element.
        all_gmsh_tags = set(self._node_map.keys())
        for pg_name, bc in self._bcs.items():
            pg_dim, pg_tag = self._find_pg(pg_name, bc.get("dim"))
            raw_pg, _ = gmsh.model.mesh.getNodesForPhysicalGroup(
                pg_dim, pg_tag
            )
            connected_gmsh.update(
                int(t) for t in raw_pg if int(t) in all_gmsh_tags
            )

        for loads in self._load_patterns.values():
            for load_def in loads:
                if load_def["type"] != "nodal":
                    continue
                pg_name = load_def["pg_name"]
                pg_dim, pg_tag = self._find_pg(
                    pg_name, load_def.get("dim")
                )
                raw_pg, _ = gmsh.model.mesh.getNodesForPhysicalGroup(
                    pg_dim, pg_tag
                )
                connected_gmsh.update(
                    int(t) for t in raw_pg if int(t) in all_gmsh_tags
                )

        n_total   = len(self._node_map)
        n_pruned  = n_total - len(connected_gmsh)
        if n_pruned > 0:
            self._log(
                f"build(): pruned {n_pruned} disconnected node(s) "
                f"({n_total} → {len(connected_gmsh)})"
            )

            # Build new sequential mapping (gmsh tag → new ops id)
            sorted_connected = sorted(connected_gmsh)
            new_node_map = {
                gt: new_id
                for new_id, gt in enumerate(sorted_connected, 1)
            }

            # Remap element node tuples from old ops ids to new ones
            old_to_new = {
                self._node_map[gt]: new_node_map[gt]
                for gt in connected_gmsh
            }
            for row in elem_rows:
                row['nodes'] = tuple(old_to_new[oid] for oid in row['nodes'])

            # Rebuild coordinate table for connected nodes only
            raw_tag_to_idx = {int(t): i for i, t in enumerate(raw_tags)}
            connected_indices = [raw_tag_to_idx[gt] for gt in sorted_connected]
            cc = coords_arr[connected_indices]

            self._node_map = new_node_map
            self._nodes_df = pd.DataFrame({
                'ops_id': np.arange(1, len(sorted_connected) + 1,
                                    dtype=np.int64),
                'x': cc[:, 0],
                'y': cc[:, 1],
                'z': cc[:, 2],
            }).set_index('ops_id')

        cols = [
            'gmsh_id', 'ops_type', 'pg_name', 'mat_name', 'mat_tag',
            'sec_tag', 'transf_tag', 'n_nodes', 'nodes', 'slots', 'extra',
        ]
        self._elements_df = (
            pd.DataFrame(elem_rows).set_index('ops_id')
            if elem_rows
            else pd.DataFrame(columns=cols)
        )
        self._log(
            f"build(): {len(elem_rows)} elements from "
            f"{len(self._elem_assignments)} group(s)"
        )
        self._built = True
        return self

    # ------------------------------------------------------------------
    # Stage 3a — Inspect
    # ------------------------------------------------------------------

    def node_table(self) -> pd.DataFrame:
        """
        Node coordinates plus declared nodal annotations (post-build).

        The returned table is indexed by OpenSees node ID and includes:

        * ``x``, ``y``, ``z`` coordinate columns
        * ``fix_i`` boolean columns for each constrained DOF
        * ``load_i`` float columns with the cumulative nodal load per DOF
        """
        self._require_built("node_table")
        df = self._nodes_df.copy()

        for dof_idx in range(1, self._ndf + 1):
            df[f"fix_{dof_idx}"] = False
            df[f"load_{dof_idx}"] = 0.0

        if df.empty:
            return df

        for pg_name, bc in self._bcs.items():
            ops_ids = self._nodes_for_pg(pg_name, bc.get("dim"))
            if not ops_ids:
                continue
            for dof_idx, is_fixed in enumerate(bc["dofs"], start=1):
                if is_fixed:
                    df.loc[ops_ids, f"fix_{dof_idx}"] = True

        for loads in self._load_patterns.values():
            for load_def in loads:
                if load_def["type"] != "nodal":
                    continue
                ops_ids = self._nodes_for_pg(
                    load_def["pg_name"],
                    load_def.get("dim"),
                )
                if not ops_ids:
                    continue
                for dof_idx, force in enumerate(load_def["force"], start=1):
                    if force:
                        df.loc[ops_ids, f"load_{dof_idx}"] += float(force)

        return df

    def element_table(self) -> pd.DataFrame:
        """Element connectivity table (post-build).  Indexed by OpenSees element ID."""
        self._require_built("element_table")
        return self._elements_df.copy()

    def summary(self) -> str:
        """Human-readable model description (works before and after build)."""
        lines = [
            f"OpenSees bridge — model: {self._parent.model_name!r}",
            f"  ndm={self._ndm}  ndf={self._ndf}",
            "",
            f"  nDMaterials ({len(self._nd_materials)}):",
        ]
        for i, (name, m) in enumerate(self._nd_materials.items(), 1):
            p = "  ".join(f"{k}={v}" for k, v in m["params"].items())
            lines.append(f"    [{i}] {name!r}  →  {m['ops_type']}  {p}")

        lines += [f"  uniaxialMaterials ({len(self._uni_materials)}):"]
        for i, (name, m) in enumerate(self._uni_materials.items(), 1):
            p = "  ".join(f"{k}={v}" for k, v in m["params"].items())
            lines.append(f"    [{i}] {name!r}  →  {m['ops_type']}  {p}")

        lines += [f"  Sections ({len(self._sections)}):"]
        for i, (name, s) in enumerate(self._sections.items(), 1):
            p = "  ".join(f"{k}={v}" for k, v in s["params"].items())
            lines.append(f"    [{i}] {name!r}  →  {s['section_type']}  {p}")

        lines += [f"  GeomTransfs ({len(self._geom_transfs)}):"]
        for i, (name, t) in enumerate(self._geom_transfs.items(), 1):
            lines.append(f"    [{i}] {name!r}  →  {t['transf_type']}")

        lines += ["", f"  Element assignments ({len(self._elem_assignments)}):"]
        for pg, a in self._elem_assignments.items():
            mat_info = (
                f"mat={a['material']!r}" if a['material']
                else f"transf={a['geom_transf']!r}"
            )
            extra_str = (
                "  " + "  ".join(f"{k}={v}" for k, v in a['extra'].items())
                if a['extra'] else ""
            )
            lines.append(
                f"    PG {pg!r}  →  {a['ops_type']}  ({mat_info}){extra_str}"
            )

        lines += ["", f"  Boundary conditions ({len(self._bcs)}):"]
        for pg, bc in self._bcs.items():
            lines.append(f"    PG {pg!r}  →  fix {bc['dofs']}")

        lines += ["", f"  Load patterns ({len(self._load_patterns)}):"]
        for pat, loads in self._load_patterns.items():
            for ld in loads:
                lines.append(
                    f"    {pat!r}  PG {ld['pg_name']!r}  "
                    f"{ld['type']}  force={ld.get('force')}"
                )

        if self._built:
            lines += ["", "  ── built ──", f"  nodes    : {len(self._nodes_df)}"]
            if not self._elements_df.empty:
                by = (
                    self._elements_df
                    .groupby(['ops_type', 'pg_name'])
                    .size()
                    .reset_index(name='n')
                )
                lines.append(f"  elements : {len(self._elements_df)}")
                for _, r in by.iterrows():
                    lines.append(
                        f"    {r.ops_type:32s}  PG {r.pg_name!r}  n={r.n}"
                    )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 3b — Export: Tcl
    # ------------------------------------------------------------------

    def export_tcl(self, path: Path | str) -> OpenSees:
        """Write an OpenSees Tcl input script to *path*."""
        self._require_built("export_tcl")
        path  = Path(path)
        lines : list[str] = []

        def hdr(title: str) -> None:
            lines.extend(["", f"# {'─'*62}", f"# {title}", f"# {'─'*62}"])

        lines += [
            "# OpenSees Tcl script",
            f"# Model : {self._parent.model_name}",
            "# Source: pyGmsh / OpenSees composite",
        ]

        hdr("Model builder")
        lines.append(
            f"model BasicBuilder  -ndm {self._ndm}  -ndf {self._ndf}"
        )

        hdr(f"Nodes  ({len(self._nodes_df)})")
        for ops_id, row in self._nodes_df.iterrows():
            xyz = [row.x, row.y, row.z][: self._ndm]
            lines.append(
                f"node {ops_id}  " + "  ".join(f"{v:.10g}" for v in xyz)
            )

        hdr(f"nDMaterials  ({len(self._nd_materials)})")
        for name, m in self._nd_materials.items():
            tag    = self._nd_mat_tags[name]
            params = "  ".join(str(v) for v in m["params"].values())
            lines.append(
                f"nDMaterial {m['ops_type']}  {tag}  {params}  ;# {name}"
            )

        hdr(f"uniaxialMaterials  ({len(self._uni_materials)})")
        for name, m in self._uni_materials.items():
            tag    = self._uni_mat_tags[name]
            params = "  ".join(str(v) for v in m["params"].values())
            lines.append(
                f"uniaxialMaterial {m['ops_type']}  {tag}  {params}  ;# {name}"
            )

        hdr(f"Sections  ({len(self._sections)})")
        for name, s in self._sections.items():
            tag    = self._sec_tags[name]
            params = "  ".join(str(v) for v in s["params"].values())
            lines.append(
                f"section {s['section_type']}  {tag}  {params}  ;# {name}"
            )

        hdr(f"GeomTransfs  ({len(self._geom_transfs)})")
        for name, t in self._geom_transfs.items():
            tag    = self._transf_tags[name]
            vecxz  = t.get("vecxz")
            suffix = (
                "  " + "  ".join(str(v) for v in vecxz)
                if (vecxz and self._ndm == 3) else ""
            )
            lines.append(
                f"geomTransf {t['transf_type']}  {tag}{suffix}  ;# {name}"
            )

        hdr(f"Elements  ({len(self._elements_df)})")
        for ops_id, row in self._elements_df.iterrows():
            lines.append(
                _render_tcl(
                    ops_id, row.ops_type, row.slots,
                    row.nodes, row.mat_tag, row.sec_tag, row.transf_tag,
                    row.extra, row.pg_name,
                )
            )

        hdr("Single-point constraints  (fix)")
        for pg_name, bc in self._bcs.items():
            ops_ids = self._nodes_for_pg(pg_name, bc.get("dim"))
            dof_str = "  ".join(str(d) for d in bc["dofs"])
            lines.append(f";# PG: {pg_name!r}  —  {len(ops_ids)} nodes")
            for nid in ops_ids:
                lines.append(f"fix {nid}  {dof_str}")

        hdr("Load patterns")
        for pat_idx, (pat_name, loads) in enumerate(
            self._load_patterns.items(), start=1
        ):
            lines.append(f"pattern Plain {pat_idx} Linear {{")
            lines.append(f"    ;# pattern: {pat_name!r}")
            for ld in loads:
                if ld["type"] == "nodal":
                    ops_ids = self._nodes_for_pg(ld["pg_name"], ld.get("dim"))
                    f_str   = "  ".join(str(v) for v in ld["force"])
                    lines.append(
                        f"    ;# PG: {ld['pg_name']!r}  —  {len(ops_ids)} nodes"
                    )
                    for nid in ops_ids:
                        lines.append(f"    load {nid}  {f_str}")
            lines.append("}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._log(f"export_tcl → {path}  ({len(lines)} lines)")
        return self

    # ------------------------------------------------------------------
    # Stage 3b — Export: openseespy
    # ------------------------------------------------------------------

    def export_py(self, path: Path | str) -> OpenSees:
        """Write an openseespy Python script to *path*."""
        self._require_built("export_py")
        path  = Path(path)
        lines : list[str] = []

        def hdr(title: str) -> None:
            lines.extend(["", f"# {'─'*62}", f"# {title}"])

        lines += [
            "# openseespy script",
            f"# Model : {self._parent.model_name}",
            "# Source: pyGmsh / OpenSees composite",
            "import openseespy.opensees as ops",
        ]

        hdr("Model builder")
        lines.append(
            f"ops.model('basic', '-ndm', {self._ndm}, '-ndf', {self._ndf})"
        )

        hdr(f"Nodes  ({len(self._nodes_df)})")
        for ops_id, row in self._nodes_df.iterrows():
            xyz = [row.x, row.y, row.z][: self._ndm]
            lines.append(
                f"ops.node({ops_id}, "
                + ", ".join(f"{v:.10g}" for v in xyz)
                + ")"
            )

        hdr(f"nDMaterials  ({len(self._nd_materials)})")
        for name, m in self._nd_materials.items():
            tag    = self._nd_mat_tags[name]
            params = ", ".join(str(v) for v in m["params"].values())
            lines.append(
                f"ops.nDMaterial('{m['ops_type']}', {tag}, {params})  # {name}"
            )

        hdr(f"uniaxialMaterials  ({len(self._uni_materials)})")
        for name, m in self._uni_materials.items():
            tag    = self._uni_mat_tags[name]
            params = ", ".join(str(v) for v in m["params"].values())
            lines.append(
                f"ops.uniaxialMaterial('{m['ops_type']}', {tag}, {params})"
                f"  # {name}"
            )

        hdr(f"Sections  ({len(self._sections)})")
        for name, s in self._sections.items():
            tag    = self._sec_tags[name]
            params = ", ".join(str(v) for v in s["params"].values())
            lines.append(
                f"ops.section('{s['section_type']}', {tag}, {params})  # {name}"
            )

        hdr(f"GeomTransfs  ({len(self._geom_transfs)})")
        for name, t in self._geom_transfs.items():
            tag   = self._transf_tags[name]
            vecxz = t.get("vecxz")
            suffix = (
                ", " + ", ".join(repr(v) for v in vecxz)
                if (vecxz and self._ndm == 3) else ""
            )
            lines.append(
                f"ops.geomTransf('{t['transf_type']}', {tag}{suffix})  # {name}"
            )

        hdr(f"Elements  ({len(self._elements_df)})")
        for ops_id, row in self._elements_df.iterrows():
            lines.append(
                _render_py(
                    ops_id, row.ops_type, row.slots,
                    row.nodes, row.mat_tag, row.sec_tag, row.transf_tag,
                    row.extra, row.pg_name,
                )
            )

        hdr("Single-point constraints")
        for pg_name, bc in self._bcs.items():
            ops_ids = self._nodes_for_pg(pg_name, bc.get("dim"))
            dof_str = ", ".join(str(d) for d in bc["dofs"])
            lines.append(f"# PG: {pg_name!r}  —  {len(ops_ids)} nodes")
            for nid in ops_ids:
                lines.append(f"ops.fix({nid}, {dof_str})")

        hdr("Load patterns")
        for pat_idx, (pat_name, loads) in enumerate(
            self._load_patterns.items(), start=1
        ):
            lines.append(
                f"ops.pattern('Plain', {pat_idx}, 'Linear')  # {pat_name!r}"
            )
            for ld in loads:
                if ld["type"] == "nodal":
                    ops_ids = self._nodes_for_pg(ld["pg_name"], ld.get("dim"))
                    f_str   = ", ".join(str(v) for v in ld["force"])
                    lines.append(
                        f"# PG: {ld['pg_name']!r}  —  {len(ops_ids)} nodes"
                    )
                    for nid in ops_ids:
                        lines.append(f"ops.load({nid}, {f_str})")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._log(f"export_py → {path}  ({len(lines)} lines)")
        return self

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "built" if self._built else "not built"
        return (
            f"OpenSees(model={self._parent.model_name!r}, "
            f"ndm={self._ndm}, ndf={self._ndf}, "
            f"nd_mats={len(self._nd_materials)}, "
            f"uni_mats={len(self._uni_materials)}, "
            f"sections={len(self._sections)}, "
            f"transfs={len(self._geom_transfs)}, "
            f"assignments={len(self._elem_assignments)}, "
            f"status={status})"
        )
