"""
OpenSees — bridge composite attached to an ``apeGmsh`` session as
``g.opensees``.

``g.opensees`` is a thin composition container.  Every declaration
method lives in a focused sub-composite:

* ``g.opensees.materials`` — ``add_nd_material``, ``add_uni_material``,
                              ``add_section``
* ``g.opensees.elements``  — ``add_geom_transf``, ``assign``, ``fix``
* ``g.opensees.ingest``    — ``loads(fem)``, ``masses(fem)`` —
                              ingest resolved records from a FEMData
                              snapshot
* ``g.opensees.inspect``   — ``node_table``, ``element_table``,
                              ``summary`` (post-build introspection)
* ``g.opensees.export``    — ``tcl(path)``, ``py(path)`` (script emission)

Plus two lifecycle entry points that stay flat on the bridge itself:

* ``g.opensees.set_model(ndm, ndf)`` — set spatial / DOF dimensions
* ``g.opensees.build()``             — extract the mesh and build all
                                        internal tables

Example
-------
::

    (g.opensees.materials
        .add_nd_material("Concrete", "ElasticIsotropic",
                         E=30e9, nu=0.2, rho=2400))
    g.opensees.elements.assign("Body", "FourNodeTetrahedron",
                               material="Concrete")
    g.opensees.elements.fix("Base", dofs=[1, 1, 1])

    with g.loads.pattern("Wind"):
        g.loads.point("TopFace", force_xyz=(1e4, 0, 0))
    g.masses.volume("Body", density=2400)

    g.mesh.generation.generate(3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    (g.opensees
        .set_model(ndm=3, ndf=3))
    (g.opensees.ingest
        .loads(fem)
        .masses(fem))
    g.opensees.build()

    print(g.opensees.inspect.summary())
    g.opensees.export.tcl("model.tcl").py("model.py")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gmsh
import pandas as pd

from ._element_specs import DimTag
from ._opensees_build import run_build
from ._opensees_materials import _Materials
from ._opensees_elements import _Elements
from ._opensees_ingest import _Ingest
from ._opensees_inspect import _Inspect
from ._opensees_export import _Export

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession


__all__ = ["OpenSees"]


from apeGmsh._logging import _HasLogging


class OpenSees(_HasLogging):
    """
    Thin composition container for the OpenSees bridge.  All
    declaration state lives on this class; the sub-composites are
    stateless namespaces that mutate it.

    Parameters
    ----------
    parent : apeGmsh
        The owning session.
    """

    _log_prefix = "OpenSees"

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self._ndm: int = 3
        self._ndf: int = 3

        # ── material registries ────────────────────────────────────────
        self._nd_materials : dict[str, dict] = {}
        self._uni_materials: dict[str, dict] = {}
        self._sections     : dict[str, dict] = {}
        self._geom_transfs : dict[str, dict] = {}

        # ── element assignments ────────────────────────────────────────
        self._elem_assignments: dict[str, dict] = {}

        # ── constraints, loads, mass ───────────────────────────────────
        self._bcs          : dict[str, dict]       = {}
        self._load_patterns: dict[str, list[dict]] = {}
        self._sp_records   : list[dict]            = []
        self._mass_records : list[dict]            = []

        # Constraint records ingested from fem (Phase 11a).
        # Populated by ``ingest.constraints(fem)``.
        # Element-side (SurfaceConstraintSet) consumed by
        # ``emit_tie_elements()`` at the end of ``build()``.
        self._constraint_records: Any = None   # SurfaceConstraintSet | None
        # Node-side (NodeConstraintSet) for future node-pair emission.
        self._node_constraint_records: Any = None  # NodeConstraintSet | None
        self._tie_penalty: float | None = None
        # Populated by ``emit_tie_elements`` during build.  Each entry:
        #   {"ele_tag": int, "cNode": int, "rNodes": list[int],
        #    "use_rot": bool, "penalty": float | None,
        #    "source_kind": str}
        self._tie_elements: list[dict] = []

        # ── post-build state ───────────────────────────────────────────
        self._built        = False
        self._node_map    : dict[int, int] = {}
        self._nodes_df    = pd.DataFrame()
        self._elements_df = pd.DataFrame()
        self._nd_mat_tags : dict[str, int] = {}
        self._uni_mat_tags: dict[str, int] = {}
        self._sec_tags    : dict[str, int] = {}
        self._transf_tags : dict[str, int] = {}

        # ── sub-composites ─────────────────────────────────────────────
        self.materials = _Materials(self)
        self.elements  = _Elements(self)
        self.ingest    = _Ingest(self)
        self.inspect   = _Inspect(self)
        self.export    = _Export(self)
        # Recorders is fully standalone; surfaced here for ergonomic
        # ``g.opensees.recorders.*`` access. The back-reference is
        # consumed at ``resolve()`` time only, to populate
        # :class:`LayerSectionMetadata` on layered-shell records from
        # ``self._sections`` + ``self._elem_assignments`` (data the
        # live openseespy domain doesn't surface). Other categories
        # ignore the back-reference.
        from .Recorders import Recorders as _Recorders
        self.recorders = _Recorders(opensees=self)

    # ------------------------------------------------------------------
    # Internal helpers (used by sub-composites via self._opensees._*)
    # ------------------------------------------------------------------

    def _require_built(self, method: str) -> None:
        if not self._built:
            raise RuntimeError(
                f"OpenSees.{method}() must be called after build()."
            )

    def _all_pg_names(self) -> list[str]:
        from apeGmsh.core.Labels import is_label_pg, strip_prefix
        names = []
        for d, t in gmsh.model.getPhysicalGroups():
            raw = gmsh.model.getPhysicalName(d, t)
            if is_label_pg(raw):
                names.append(strip_prefix(raw) + "  [label]")
            else:
                names.append(raw)
        return names

    def _find_pg(
        self,
        name        : str,
        expected_dim: int | None = None,
    ) -> DimTag:
        """
        Return ``(dim, tag)`` for a named physical group **or** label.

        Resolution order:

        1. Exact match on a user-facing PG name.
        2. Fallback: match on ``_label:{name}`` (the internal label
           prefix).  This lets the broker accept label names directly
           so the user can skip ``promote_to_physical``.

        Raises ``ValueError`` if neither is found, or if the name is
        ambiguous across multiple dimensions (pass ``expected_dim``
        to disambiguate).
        """
        from apeGmsh.core.Labels import LABEL_PREFIX

        # Try user PG name first, then label-prefixed name.
        candidates = [name, LABEL_PREFIX + name]
        matches: list[tuple[int, int]] = []
        for d, t in gmsh.model.getPhysicalGroups():
            pg_name = gmsh.model.getPhysicalName(d, t)
            if pg_name in candidates:
                matches.append((d, t))
        if not matches:
            raise ValueError(
                f"Physical group or label {name!r} not found in the "
                f"model.\nAvailable: {self._all_pg_names()}"
            )
        if len(matches) == 1:
            return matches[0]
        if expected_dim is not None:
            dim_matches = [(d, t) for d, t in matches if d == expected_dim]
            if len(dim_matches) == 1:
                return dim_matches[0]
        dims = sorted(d for d, _ in matches)
        raise ValueError(
            f"Ambiguous physical-group name {name!r}: "
            f"exists in dimensions {dims}. "
            f"Pass `dim=<dim>` to elements.assign / elements.fix "
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
    # Lifecycle entry points (flat)
    # ------------------------------------------------------------------

    def set_model(self, *, ndm: int = 3, ndf: int = 3) -> "OpenSees":
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

    def build(self) -> "OpenSees":
        """
        Extract the active gmsh mesh and construct all model tables.

        Performs full validation:

        * element types match the physical-group topology
        * material references point to the correct registry
        * geomTransf is declared for beam elements
        * ``ndf`` / ``ndm`` are compatible with each element spec
        * warns when higher-order gmsh nodes are downgraded to
          first-order

        Must be called after ``g.mesh.generation.generate()`` and
        all declarations on ``g.opensees.materials`` /
        ``g.opensees.elements`` / ``g.opensees.ingest``.
        """
        run_build(self)
        return self

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "built" if self._built else "not built"
        return (
            f"OpenSees(model={self._parent.name!r}, "
            f"ndm={self._ndm}, ndf={self._ndf}, "
            f"nd_mats={len(self._nd_materials)}, "
            f"uni_mats={len(self._uni_materials)}, "
            f"sections={len(self._sections)}, "
            f"transfs={len(self._geom_transfs)}, "
            f"assignments={len(self._elem_assignments)}, "
            f"status={status})"
        )
