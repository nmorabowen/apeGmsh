"""Recorders — declarative recorder spec composite.

Surfaced on ``g.opensees.recorders`` for ergonomic access, but the
class itself is **standalone**: ``Recorders()`` works without any
parent session. This decoupling protects against the future
``apeSees`` migration — the class moves with the OpenSees bridge as
a unit; the surfacing path changes, the API doesn't.

Usage
-----
::

    g.opensees.recorders.nodes(
        pg="Top", components=["displacement", "rotation"], dt=0.01,
    )
    g.opensees.recorders.gauss(
        selection="Body_clip", components=["stress_xx", "von_mises_stress"],
    )
    g.opensees.recorders.modal(n_modes=10)

    fem = g.mesh.queries.get_fem_data(dim=3)
    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=6)
    # spec is a ResolvedRecorderSpec — feeds Phase 5/7/8 emission paths.

Notes
-----
- Selection vocabulary mirrors :class:`FEMData.nodes.get` exactly:
  ``pg=`` / ``label=`` / ``selection=`` / ``ids=``. Multiple named
  selectors combine as union; ``ids=`` is mutex with the named ones.
- Components accept canonical names or shorthands from the Phase 0
  vocabulary (``"displacement"`` → ``displacement_x/y/z`` etc.).
  Shorthand expansion happens at ``resolve()`` time, when ndm/ndf are
  known.
- Cadence: at most one of ``dt=`` / ``n_steps=`` per record; both
  ``None`` means every analysis step.
- Phase 4 does **not** do element-capability validation
  (e.g. "this PG has no GPs"). That validation lives at emission
  time (Phase 5) when the OpenSees bridge's element class
  assignments are in scope. The ``_ElemSpec.has_*`` flags are in
  place for that downstream use.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from numpy import ndarray

from ..results._vocabulary import (
    DERIVED_SCALARS,
    FIBER,
    LINE_DIAGRAMS,
    MATERIAL_STATE,
    NODAL_FORCES,
    NODAL_KINEMATICS,
    PER_ELEMENT_NODAL_FORCES,
    STRAIN,
    STRESS,
    expand_many,
    is_canonical,
)
from ._recorder_specs import (
    ALL_CATEGORIES,
    LayerSectionDef,
    LayerSectionMetadata,
    RecorderRecord,
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from .OpenSees import OpenSees


# Per-category sets of allowed canonical components. ``state_variable_*``
# is allowed in any element-level category; checked separately.
_CATEGORY_COMPONENTS: dict[str, frozenset[str]] = {
    "nodes": frozenset(NODAL_KINEMATICS + NODAL_FORCES),
    "elements": frozenset(PER_ELEMENT_NODAL_FORCES),
    "line_stations": frozenset(LINE_DIAGRAMS),
    "gauss": frozenset(STRESS + STRAIN + DERIVED_SCALARS + MATERIAL_STATE),
    "fibers": frozenset(FIBER + MATERIAL_STATE),
    # Layered-shell records use the same fiber canonicals as beam
    # fibers (one stress/strain value per fiber/layer), aliased
    # under the LAYER_CATALOG keys ``fiber_stress`` / ``fiber_strain``.
    "layers": frozenset(FIBER + MATERIAL_STATE),
}

_ELEMENT_LEVEL_CATEGORIES = frozenset({
    "elements", "line_stations", "gauss", "fibers", "layers",
})


# =====================================================================
# Recorders composite
# =====================================================================

class Recorders:
    """Declarative recorder spec builder.

    Instances are constructable standalone (``Recorders()``) or
    surfaced on an OpenSees bridge (``g.opensees.recorders``). The
    OpenSees back-reference is optional: most categories
    (``nodes`` / ``gauss`` / ``line_stations`` / ``elements`` /
    ``fibers`` / ``modal``) resolve from the FEMData alone. The one
    exception is ``layers`` — :class:`LayerSectionMetadata` is
    populated at resolve time from ``opensees._sections`` and the
    per-PG element assignments. Without an OpenSees handle, layer
    records resolve without metadata, and DomainCapture's
    ``_LayerCapturer`` will surface a clear error.
    """

    def __init__(self, opensees: "OpenSees | None" = None) -> None:
        self._records: list[RecorderRecord] = []
        self._auto_id: int = 0
        self._opensees = opensees

    # ------------------------------------------------------------------
    # Declaration methods (one per category)
    # ------------------------------------------------------------------

    def nodes(
        self,
        *,
        components: str | Iterable[str],
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        name: str | None = None,
    ) -> "Recorders":
        """Declare a nodal recorder.

        Records per-node values across the analysis. Reads back as a
        :class:`NodeSlab` with shape ``(T, N)``.

        **Components** (canonical names):

        - Kinematics — ``displacement_x``, ``displacement_y``,
          ``displacement_z``, ``rotation_x``, ``rotation_y``,
          ``rotation_z``, ``velocity_x``, ``velocity_y``,
          ``velocity_z``, ``angular_velocity_x``, ``angular_velocity_y``,
          ``angular_velocity_z``, ``acceleration_x``,
          ``acceleration_y``, ``acceleration_z``,
          ``angular_acceleration_x``, ``angular_acceleration_y``,
          ``angular_acceleration_z``, ``displacement_increment_x/y/z``.
        - Forces / reactions — ``force_x/y/z``, ``moment_x/y/z``,
          ``reaction_force_x/y/z``, ``reaction_moment_x/y/z``,
          ``pore_pressure``, ``pore_pressure_rate``.

        **Shorthands** (clipped to the active ``ndm``/``ndf``):

        - ``"displacement"`` → ``displacement_x/y/z``.
        - ``"rotation"`` → ``rotation_x/y/z``.
        - ``"velocity"`` / ``"acceleration"`` / ``"angular_velocity"`` /
          ``"angular_acceleration"`` / ``"displacement_increment"`` —
          analogous.
        - ``"force"`` / ``"moment"`` — analogous (point loads).
        - ``"reaction_force"`` / ``"reaction_moment"`` — granular.
        - ``"reaction"`` — both forces and moments in one pass.

        **Selectors** (provide one — they're mutually exclusive):

        - ``pg=`` — physical group name(s) (Tier 2).
        - ``label=`` — apeGmsh label name(s) (Tier 1).
        - ``selection=`` — post-mesh ``g.mesh_selection`` set name(s).
        - ``ids=`` — raw node IDs.

        **Cadence** (at most one):

        - ``dt=`` — wall-clock cadence.
        - ``n_steps=`` — step-count cadence.
        - both ``None`` (default) — every analysis step.

        Example::

            g.opensees.recorders.nodes(
                components=["displacement", "reaction_force"],
                pg="Top",
                dt=0.01,
            )
        """
        self._declare(
            "nodes", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
        )
        return self

    def elements(
        self,
        *,
        components: str | Iterable[str],
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        name: str | None = None,
    ) -> "Recorders":
        """Declare a per-element-node force recorder.

        Records the resisting force vector at each element node, in
        either the global or local frame. Reads back as an
        :class:`ElementSlab` with shape ``(T, E, npe)``.

        **Components** (canonical names):

        - Global frame — ``nodal_resisting_force_x/y/z``,
          ``nodal_resisting_moment_x/y/z``.
        - Local frame — ``nodal_resisting_force_local_x/y/z``,
          ``nodal_resisting_moment_local_x/y/z``.

        Components from different frames cannot mix in one record —
        split into separate records (one ``globalForce`` + one
        ``localForce``).

        **Selectors** — same vocabulary as :meth:`nodes`
        (``pg=`` / ``label=`` / ``selection=`` / ``ids=``); resolve to
        element IDs.

        **Cadence** — same as :meth:`nodes` (``dt=`` / ``n_steps=`` /
        every step).

        Example::

            g.opensees.recorders.elements(
                components=["nodal_resisting_force_x"],
                pg="Frame",
            )
        """
        self._declare(
            "elements", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
        )
        return self

    def line_stations(
        self,
        *,
        components: str | Iterable[str],
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        name: str | None = None,
    ) -> "Recorders":
        """Declare a beam line-station recorder.

        Records section forces / strains at each integration point
        along beam-column elements. Reads back as a
        :class:`LineStationSlab` with shape ``(T, sum_S)``.

        **Components — section forces** (in section-local frame):

        - ``axial_force``, ``shear_y``, ``shear_z``, ``torsion``,
          ``bending_moment_y``, ``bending_moment_z``.

        Note: ``bending_moment_y/z`` (in section-local frame) are
        intentionally distinct from ``moment_x/y/z`` on :meth:`nodes`
        (applied nodal moments, in global frame).

        Section deformations (``axial_strain`` / ``curvature_*`` /
        etc.) live in the vocabulary but are not currently accepted
        by this declaration path — they're work-conjugate to the
        forces and emerge on the read side via the
        ``section_deformation`` MPCO bucket.

        **Shorthand**:

        - ``"section_force"`` → all six forces.

        **Selectors / cadence** — same as :meth:`nodes`.

        Example::

            g.opensees.recorders.line_stations(
                components=["axial_force", "bending_moment_y"],
                label="frame",
            )
        """
        self._declare(
            "line_stations", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
        )
        return self

    def gauss(
        self,
        *,
        components: str | Iterable[str],
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        name: str | None = None,
    ) -> "Recorders":
        """Declare a continuum Gauss-point recorder.

        Records stress / strain / derived scalars at integration
        points of continuum elements. Reads back as a
        :class:`GaussSlab` with shape ``(T, sum_GP)`` plus
        ``natural_coords (sum_GP, dim)`` for visualisation.

        **Components — stress** (Cauchy, tensor indices):

        - 3D — ``stress_xx``, ``stress_yy``, ``stress_zz``,
          ``stress_xy``, ``stress_yz``, ``stress_xz``.
        - 2D plane elements — ``stress_xx``, ``stress_yy``,
          ``stress_xy`` (three independent components).

        **Components — strain** (small-strain, conjugate to stress):

        - 3D — ``strain_xx``, ``strain_yy``, ``strain_zz``,
          ``strain_xy``, ``strain_yz``, ``strain_xz``.
        - 2D plane elements — ``strain_xx``, ``strain_yy``,
          ``strain_xy``.

        **Components — derived scalars**:

        - ``von_mises_stress``, ``pressure_hydrostatic``,
          ``principal_stress_1/2/3``, ``equivalent_plastic_strain``.

        **Components — material state**:

        - ``damage`` (single-component models),
          ``damage_tension`` / ``damage_compression`` (split models).
        - ``equivalent_plastic_strain_tension`` /
          ``equivalent_plastic_strain_compression``.
        - ``state_variable_<n>`` — generic material-state outputs
          (any non-negative integer ``<n>``).

        Stress and strain components cannot mix in one record (they
        come from different OpenSees recorder tokens). Split into
        separate records.

        **Shell stress resultants / generalised strains** — recorded
        via this method on layered shells (one set per surface GP);
        see also :meth:`layers` for through-thickness layer values.

        **Shorthands** (clipped to the element's tensor dimension):

        - ``"stress"`` → all six stress components in 3D, three in 2D.
        - ``"strain"`` → analogous.

        **Selectors / cadence** — same as :meth:`nodes`.

        Example::

            g.opensees.recorders.gauss(
                components=["stress", "von_mises_stress"],
                pg="Body",
            )
        """
        self._declare(
            "gauss", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
        )
        return self

    def fibers(
        self,
        *,
        components: str | Iterable[str],
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        name: str | None = None,
    ) -> "Recorders":
        """Declare a fiber-section recorder (uniaxial along fiber axis).

        Records per-fiber stress / strain in fiber sections
        (``FiberSection2d`` / ``FiberSection3d``). Reads back as a
        :class:`FiberSlab` with shape ``(T, sum_F)`` plus per-fiber
        ``y / z / area / material_tag`` location metadata.

        **Components** (canonical):

        - ``fiber_stress``, ``fiber_strain`` — uniaxial along the
          fiber axis (1D constitutive law).

        **Indexed variants** (handled at read-time, not enumerated):

        - ``fiber_stress_<n>`` / ``fiber_strain_<n>`` for shell layered
          sections that emit a vector per layer (see :meth:`layers`).

        **Material state**:

        - ``damage``, ``equivalent_plastic_strain``, etc. — same
          vocabulary as :meth:`gauss`. Fiber sections expose these
          when the fiber's uniaxial material does.

        **Coverage caveat**:

        - Fiber records cannot be emitted via the classic recorder
          path (``spec.emit_recorders`` / ``export.tcl/py`` text
          files) — the per-fiber row count would explode the file
          format. They are supported by :meth:`spec.capture` (in-
          process) and :meth:`spec.emit_mpco` / MPCO export (which
          uses ``section.fiber.stress`` natively).

        **Selectors / cadence** — same as :meth:`nodes`; resolve to
        beam-column element IDs.

        Example::

            g.opensees.recorders.fibers(
                components=["fiber_stress", "fiber_strain"],
                pg="RC_Columns",
            )
        """
        self._declare(
            "fibers", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
        )
        return self

    def layers(
        self,
        *,
        components: str | Iterable[str],
        pg: str | Iterable[str] | None = None,
        label: str | Iterable[str] | None = None,
        selection: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        name: str | None = None,
    ) -> "Recorders":
        """Declare a layered-shell layer recorder.

        Records per-layer stress / strain through the thickness of
        layered-shell sections (``LayeredShellFiberSection`` on
        ``ASDShellQ4`` / ``ShellMITC4`` / ``ShellDKGQ`` / etc.). Reads
        back as a :class:`LayerSlab` with shape ``(T, sum_L)`` plus
        per-layer ``thickness`` / ``layer_index`` / ``sub_gp_index``
        / ``local_axes_quaternion`` metadata.

        **Components** (canonical):

        - ``fiber_stress``, ``fiber_strain`` — when each layer emits
          one scalar (uniaxial-equivalent layer behaviour).
        - ``fiber_stress_<n>`` / ``fiber_strain_<n>`` — when each
          layer emits a vector (e.g. plane-stress + transverse-shear,
          5 components per layer). Component count is auto-discovered
          on first probe.

        **Material state** — same vocabulary as :meth:`gauss` /
        :meth:`fibers` (``damage_tension``, ``damage_compression``,
        ``state_variable_<n>``, etc.).

        **Coverage caveat**:

        - Same as :meth:`fibers` — not emittable via classic recorder
          ``.out`` files. Use :meth:`spec.capture` (in-process; needs
          an OpenSees back-reference for the per-layer thickness +
          material-tag metadata) or :meth:`spec.emit_mpco` / MPCO
          export.

        **Selectors / cadence** — same as :meth:`nodes`; resolve to
        shell element IDs.

        Example::

            g.opensees.recorders.layers(
                components=["fiber_stress", "fiber_strain"],
                pg="Slab",
            )
        """
        self._declare(
            "layers", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
        )
        return self

    def modal(
        self,
        n_modes: int,
        *,
        dt: float | None = None,
        n_steps: int | None = None,
        name: str | None = None,
    ) -> "Recorders":
        """Declare a modal-shape recorder.

        Triggers an eigenvalue analysis at execution time. Each
        requested mode lands as its own stage with ``kind="mode"`` in
        the resulting file, exposing the eigenvalue, frequency, and
        period as stage attributes. Read back via ``results.modes``.

        Modal records have **no components or selectors** — they
        capture all node displacement DOFs of every mode shape.

        Parameters
        ----------
        n_modes
            Number of modes to extract (positive integer).
        dt, n_steps
            Cadence — same semantics as the other declaration
            methods.
        name
            Optional record name (auto-generated otherwise).

        **Coverage caveat**:

        - Modal records can only be executed via :meth:`spec.capture`
          (which calls ``ops.eigen()`` itself) or :meth:`spec.emit_mpco`
          (which records the ``modesOfVibration`` token natively).
          The classic recorder path (:meth:`spec.emit_recorders` /
          ``export.tcl``) **cannot** drive eigenvalue analysis and
          will raise / skip modal records.

        Example::

            g.opensees.recorders.modal(n_modes=10)

            # ... then with spec.capture(...) as cap: cap.capture_modes(10)
            # ... or with spec.emit_mpco(...): writes modesOfVibration
        """
        if not isinstance(n_modes, int) or n_modes <= 0:
            raise ValueError(f"n_modes must be a positive int (got {n_modes!r}).")
        _validate_cadence(dt, n_steps)
        rec_name = name or f"modal_{self._auto_id}"
        self._auto_id += 1
        self._records.append(RecorderRecord(
            category="modal",
            components=(),
            name=rec_name,
            dt=dt, n_steps=n_steps,
            n_modes=n_modes,
        ))
        return self

    # ------------------------------------------------------------------
    # Introspection — what's available to declare
    # ------------------------------------------------------------------

    @staticmethod
    def categories() -> tuple[str, ...]:
        """Return the recorder categories you can declare.

        These are the seven declaration methods on this class
        (``nodes``, ``elements``, ``line_stations``, ``gauss``,
        ``fibers``, ``layers``, ``modal``) and the ``category``
        values stored on :class:`RecorderRecord`.

        Example::

            >>> Recorders.categories()
            ('nodes', 'elements', 'line_stations', 'gauss',
             'fibers', 'layers', 'modal')
        """
        return ALL_CATEGORIES

    @staticmethod
    def components_for(category: str) -> tuple[str, ...]:
        """Return the canonical component names allowed in ``category``.

        Returns an empty tuple for ``category="modal"`` (modal records
        have no components — they capture mode shapes wholesale).
        Indexed variants (``state_variable_<n>``,
        ``fiber_stress_<n>``, ``spring_force_<n>``) are not enumerated
        here; pass them directly and :meth:`resolve` will validate.

        Example::

            >>> Recorders.components_for("nodes")[:3]
            ('displacement_x', 'displacement_y', 'displacement_z')
            >>> Recorders.components_for("gauss")[:3]
            ('stress_xx', 'stress_yy', 'stress_zz')

        Raises
        ------
        KeyError
            If ``category`` is not one of :meth:`categories`.
        """
        if category == "modal":
            return ()
        if category not in _CATEGORY_COMPONENTS:
            raise KeyError(
                f"Unknown category {category!r}. Valid categories: "
                f"{Recorders.categories()}."
            )
        return tuple(sorted(_CATEGORY_COMPONENTS[category]))

    @staticmethod
    def shorthands_for(category: str) -> dict[str, tuple[str, ...]]:
        """Return shorthand → expansion mapping valid in ``category``.

        Shorthands always expand to the *full* (3D, ndf=6) component
        set; :meth:`resolve` clips to the active ``ndm``/``ndf``.
        Returns an empty dict for ``"modal"`` (no shorthands) and for
        unknown categories.

        Example::

            >>> sh = Recorders.shorthands_for("nodes")
            >>> sh["displacement"]
            ('displacement_x', 'displacement_y', 'displacement_z')
            >>> Recorders.shorthands_for("gauss")["stress"][:3]
            ('stress_xx', 'stress_yy', 'stress_zz')
        """
        from ..results import _vocabulary as _voc

        if category in ("modal",):
            return {}

        # Build a per-category subset of the vocabulary's shorthand
        # tables. Shorthands are valid when their expansion is a
        # subset of the category's allowed canonicals.
        allowed = _CATEGORY_COMPONENTS.get(category, frozenset())
        if not allowed:
            return {}

        out: dict[str, tuple[str, ...]] = {}
        all_tables = (
            _voc._SHORTHAND_TRANSLATIONAL,
            _voc._SHORTHAND_ROTATIONAL,
            _voc._SHORTHAND_TENSOR,
            _voc._SHORTHAND_LINE_STATION,
        )
        for table in all_tables:
            for shorthand, expansion in table.items():
                if all(c in allowed for c in expansion):
                    out[shorthand] = expansion

        # The "reaction" mega-shorthand expands to forces + moments
        # together; only valid on nodes.
        if category == "nodes":
            out["reaction"] = _voc._SHORTHAND_REACTION

        return out

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        fem: "FEMData",
        *,
        ndm: int = 3,
        ndf: int = 6,
    ) -> ResolvedRecorderSpec:
        """Resolve all records against a FEMData snapshot.

        Steps per record:
        1. Expand shorthand components (``"displacement"`` → ``displacement_x/y/z``,
           clipped to ``ndm``/``ndf``).
        2. Validate that every expanded component is allowed in the
           record's category (``stress_xx`` is invalid on nodes, etc.).
        3. Resolve selectors (``pg`` / ``label`` / ``selection``) to
           concrete ID arrays via FEMData's named-group lookups.
        4. Bundle into :class:`ResolvedRecorderRecord`.

        Element-capability validation (e.g. "this PG has no GPs") is
        not done here — it requires OpenSees-side knowledge of which
        element class is assigned to which PG, which lives on the
        OpenSees bridge. That validation runs at emission time
        (Phase 5+).

        Returns
        -------
        ResolvedRecorderSpec
            A snapshot tied to ``fem.snapshot_id``. Re-meshing
            produces a new hash; the spec refuses to bind.
        """
        from ..results._vocabulary import expand_many

        resolved: list[ResolvedRecorderRecord] = []
        for rec in self._records:
            resolved.append(self._resolve_one(rec, fem, ndm=ndm, ndf=ndf))
        return ResolvedRecorderSpec(
            fem_snapshot_id=fem.snapshot_id,
            records=tuple(resolved),
        )

    def _resolve_one(
        self,
        rec: RecorderRecord,
        fem: "FEMData",
        *,
        ndm: int,
        ndf: int,
    ) -> ResolvedRecorderRecord:
        if rec.category == "modal":
            return ResolvedRecorderRecord(
                category="modal",
                name=rec.name,
                components=(),
                dt=rec.dt,
                n_steps=rec.n_steps,
                n_modes=rec.n_modes,
                source=rec,
            )

        # Expand shorthand and validate per-category
        expanded = expand_many(rec.components, ndm=ndm, ndf=ndf)
        if not expanded:
            raise ValueError(
                f"Record {rec.name!r} ({rec.category}) expanded to zero "
                f"components in ndm={ndm}, ndf={ndf}. "
                f"Original: {list(rec.components)}"
            )
        _validate_components_for_category(rec.category, expanded)

        # Resolve selectors
        if rec.category == "nodes":
            ids_array = _resolve_node_selectors(fem, rec)
            return ResolvedRecorderRecord(
                category="nodes",
                name=rec.name,
                components=expanded,
                dt=rec.dt, n_steps=rec.n_steps,
                node_ids=ids_array,
                source=rec,
            )
        # All other (non-modal) categories are element-level.
        ids_array = _resolve_element_selectors(fem, rec)
        layer_metadata = None
        if rec.category == "layers":
            layer_metadata = self._resolve_layer_section_metadata(
                fem, ids_array,
            )
        return ResolvedRecorderRecord(
            category=rec.category,
            name=rec.name,
            components=expanded,
            dt=rec.dt, n_steps=rec.n_steps,
            element_ids=ids_array,
            layer_section_metadata=layer_metadata,
            source=rec,
        )

    def _resolve_layer_section_metadata(
        self,
        fem: "FEMData",
        element_ids: ndarray,
    ) -> Optional[LayerSectionMetadata]:
        """Build LayerSectionMetadata from the OpenSees back-reference.

        Walks ``opensees._elem_assignments`` to learn which physical
        group each element belongs to and which named section is
        attached, then pulls layered-shell composition from
        ``opensees._sections``. Elements assigned to a non-layered
        section are silently absent from the map; if no element in
        the record uses a layered section, returns ``None``.
        """
        if self._opensees is None:
            return None

        sections_registry = self._opensees._sections
        elem_assignments = self._opensees._elem_assignments

        # Reverse-build: pg_name → layered-section_name (skip if section
        # isn't LayeredShell, or if assignment lacks a section).
        pg_to_layered_section: dict[str, str] = {}
        for pg_name, assign in elem_assignments.items():
            sec_name = assign.get("material")
            if sec_name is None or sec_name not in sections_registry:
                continue
            sec_def = sections_registry[sec_name]
            if not _is_layered_shell_section(sec_def):
                continue
            pg_to_layered_section[pg_name] = sec_name

        if not pg_to_layered_section:
            return None

        # Map element_id → section_name by intersecting the record's
        # element_ids with each PG's element_ids.
        record_id_set = set(int(e) for e in element_ids.tolist())
        eid_to_sec_name: dict[int, str] = {}
        for pg_name, sec_name in pg_to_layered_section.items():
            try:
                pg_eids = fem.elements.physical.element_ids(pg_name)
            except Exception:
                continue
            for eid in np.asarray(pg_eids, dtype=np.int64).tolist():
                if int(eid) in record_id_set:
                    eid_to_sec_name[int(eid)] = sec_name

        if not eid_to_sec_name:
            return None

        # Tag → def cache (use enumerated sec_tag from build, or
        # fall back to a stable hash-of-name when build hasn't run).
        sec_tags = getattr(self._opensees, "_sec_tags", {}) or {}
        sections: dict[int, LayerSectionDef] = {}
        eid_to_sec_tag: dict[int, int] = {}
        for eid, sec_name in eid_to_sec_name.items():
            sec_tag = int(
                sec_tags.get(sec_name, _stable_section_tag(sec_name)),
            )
            eid_to_sec_tag[eid] = sec_tag
            if sec_tag in sections:
                continue
            sec_def = sections_registry[sec_name]
            sections[sec_tag] = _layered_shell_section_def(
                sec_tag=sec_tag, name=sec_name, raw=sec_def,
                opensees=self._opensees,
            )

        return LayerSectionMetadata(
            sections=sections,
            element_to_section=eid_to_sec_tag,
        )

    # ------------------------------------------------------------------
    # Inspection / lifecycle
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterable[RecorderRecord]:
        return iter(self._records)

    def clear(self) -> None:
        """Remove all declared records."""
        self._records.clear()
        self._auto_id = 0

    def __repr__(self) -> str:
        if not self._records:
            return "Recorders(empty)"
        lines = [f"Recorders({len(self._records)} records):"]
        for r in self._records:
            sel = self._record_selectors_repr(r)
            cad = self._record_cadence_repr(r)
            comps = list(r.components) if r.components else f"n_modes={r.n_modes}"
            lines.append(f"  - {r.category} {r.name!r}: {sel}, {cad}, {comps}")
        return "\n".join(lines)

    @staticmethod
    def _record_selectors_repr(r: RecorderRecord) -> str:
        if r.category == "modal":
            return ""
        parts = []
        if r.pg:
            parts.append(f"pg={list(r.pg)}")
        if r.label:
            parts.append(f"label={list(r.label)}")
        if r.selection:
            parts.append(f"selection={list(r.selection)}")
        if r.ids is not None:
            parts.append(f"ids=[{len(r.ids)} entries]")
        return ", ".join(parts) if parts else "all"

    @staticmethod
    def _record_cadence_repr(r: RecorderRecord) -> str:
        if r.dt is not None:
            return f"dt={r.dt}"
        if r.n_steps is not None:
            return f"every {r.n_steps} steps"
        return "every step"

    # ------------------------------------------------------------------
    # Declaration helper
    # ------------------------------------------------------------------

    def _declare(
        self,
        category: str,
        components: str | Iterable[str],
        *,
        pg, label, selection, ids,
        dt: float | None,
        n_steps: int | None,
        name: str | None,
    ) -> None:
        if category not in ALL_CATEGORIES:
            raise ValueError(
                f"Unknown recorder category {category!r}. "
                f"Must be one of {ALL_CATEGORIES}."
            )
        comp_tuple = _normalize_components(components)
        if not comp_tuple:
            raise ValueError(
                f"At least one component is required for {category!r}."
            )

        _validate_cadence(dt, n_steps)
        _validate_selector_exclusivity(pg, label, selection, ids)

        rec_name = name or f"{category}_{self._auto_id}"
        self._auto_id += 1

        self._records.append(RecorderRecord(
            category=category,
            components=comp_tuple,
            name=rec_name,
            pg=_to_str_tuple(pg),
            label=_to_str_tuple(label),
            selection=_to_str_tuple(selection),
            ids=tuple(int(i) for i in ids) if ids is not None else None,
            dt=dt,
            n_steps=n_steps,
        ))


# =====================================================================
# Validation helpers
# =====================================================================

def _normalize_components(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _to_str_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _validate_cadence(dt: float | None, n_steps: int | None) -> None:
    if dt is not None and n_steps is not None:
        raise ValueError(
            "Provide at most one of dt= or n_steps= "
            "(both None means every analysis step)."
        )
    if dt is not None and dt <= 0:
        raise ValueError(f"dt must be positive (got {dt}).")
    if n_steps is not None and n_steps <= 0:
        raise ValueError(f"n_steps must be a positive int (got {n_steps}).")


def _validate_selector_exclusivity(pg, label, selection, ids) -> None:
    if ids is None:
        return
    named = [x for x in (pg, label, selection) if x is not None]
    if named:
        raise ValueError(
            "Provide one of pg=, label=, selection=, or ids= (not multiple)."
        )


def _validate_components_for_category(
    category: str, components: tuple[str, ...],
) -> None:
    """Each component must be canonical AND valid for the category."""
    allowed = _CATEGORY_COMPONENTS.get(category, frozenset())
    for comp in components:
        if not is_canonical(comp):
            raise ValueError(
                f"Component {comp!r} is not a canonical apeGmsh name. "
                f"Use shorthands (``displacement``, ``stress``, …) or "
                f"explicit canonical names from the Phase 0 vocabulary."
            )
        # state_variable_<n> is always allowed in element-level categories.
        if comp.startswith("state_variable_") and category in (
            "gauss", "fibers", "layers",
        ):
            continue
        if comp not in allowed:
            raise ValueError(
                f"Component {comp!r} is not valid for recorder category "
                f"{category!r}. Valid components for {category!r}: "
                f"{sorted(allowed)[:8]}{'...' if len(allowed) > 8 else ''}"
            )


# =====================================================================
# Selector resolution helpers
# =====================================================================

def _resolve_node_selectors(fem: "FEMData", rec: RecorderRecord) -> ndarray:
    """Resolve a node-side recorder's selectors to a concrete ID array.

    ``None``-equivalent (no selectors) means ALL nodes — return all
    fem node IDs.
    """
    if rec.ids is not None:
        return np.asarray(rec.ids, dtype=np.int64)

    if not rec.pg and not rec.label and not rec.selection:
        return np.asarray(fem.nodes.ids, dtype=np.int64)

    chunks: list[ndarray] = []
    for name in rec.pg:
        chunks.append(np.asarray(
            fem.nodes.physical.node_ids(name), dtype=np.int64,
        ))
    for name in rec.label:
        chunks.append(np.asarray(
            fem.nodes.labels.node_ids(name), dtype=np.int64,
        ))
    if rec.selection:
        store = getattr(fem, "mesh_selection", None)
        if store is None:
            raise RuntimeError(
                f"Record {rec.name!r} uses selection=, but "
                f"fem.mesh_selection is None (no post-mesh selections "
                f"were declared on the session)."
            )
        for name in rec.selection:
            chunks.append(store.node_ids(name))

    if not chunks:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(chunks))


def _resolve_element_selectors(
    fem: "FEMData", rec: RecorderRecord,
) -> ndarray:
    """Resolve an element-side recorder's selectors to an ID array."""
    if rec.ids is not None:
        return np.asarray(rec.ids, dtype=np.int64)

    if not rec.pg and not rec.label and not rec.selection:
        return np.asarray(fem.elements.ids, dtype=np.int64)

    chunks: list[ndarray] = []
    for name in rec.pg:
        chunks.append(np.asarray(
            fem.elements.physical.element_ids(name), dtype=np.int64,
        ))
    for name in rec.label:
        chunks.append(np.asarray(
            fem.elements.labels.element_ids(name), dtype=np.int64,
        ))
    if rec.selection:
        store = getattr(fem, "mesh_selection", None)
        if store is None:
            raise RuntimeError(
                f"Record {rec.name!r} uses selection=, but "
                f"fem.mesh_selection is None."
            )
        for name in rec.selection:
            chunks.append(store.element_ids(name))

    if not chunks:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(chunks))


# =====================================================================
# Layered-shell section helpers
# =====================================================================
#
# ``g.opensees.materials.add_section(name, "LayeredShell", ...)`` shape:
# the params dict carries ``layers`` as a list of ``(matTag, thickness)``
# pairs (or, for legacy callers, separate ``thicknesses`` and
# ``materials`` lists). We support both shapes.

_LAYERED_SHELL_TYPES = frozenset({
    "LayeredShell",
    "LayeredShellFiberSection",
})


def _is_layered_shell_section(sec_def: dict) -> bool:
    return sec_def.get("section_type", "") in _LAYERED_SHELL_TYPES


def _layered_shell_section_def(
    *, sec_tag: int, name: str, raw: dict,
    opensees: "OpenSees | None" = None,
) -> "LayerSectionDef":
    """Normalise a registered LayeredShell section's params.

    Accepts either:
    - ``params={"layers": [(mat, thickness), ...]}``
    - ``params={"thicknesses": [...], "materials": [...]}``

    Materials may be int tags or string names from the parent
    ``OpenSees`` material registries. Names are resolved against
    ``_nd_mat_tags`` / ``_uni_mat_tags`` (populated at build time);
    if those tables are empty (build hasn't run), we fall back to
    a stable hash so the LayerSlab still has a usable column.
    """
    params = raw.get("params") or {}
    layers = params.get("layers")
    if layers is not None:
        thicknesses = [float(t) for _, t in layers]
        material_refs = [m for m, _ in layers]
    else:
        thicknesses = [float(t) for t in (params.get("thicknesses") or ())]
        material_refs = list(params.get("materials") or ())
    material_tags = [
        _resolve_material_tag(m, opensees) for m in material_refs
    ]
    if len(thicknesses) != len(material_tags):
        raise ValueError(
            f"LayeredShell section {name!r}: thickness count "
            f"({len(thicknesses)}) does not match material-tag count "
            f"({len(material_tags)})."
        )
    if not thicknesses:
        raise ValueError(
            f"LayeredShell section {name!r} has zero layers — check "
            f"params={params!r}."
        )
    return LayerSectionDef(
        section_tag=sec_tag,
        section_name=name,
        n_layers=len(thicknesses),
        thickness=np.asarray(thicknesses, dtype=np.float64),
        material_tags=np.asarray(material_tags, dtype=np.int64),
    )


def _resolve_material_tag(
    mat_ref, opensees: "OpenSees | None",
) -> int:
    """Convert an int (already-a-tag) or a name to an integer tag."""
    if isinstance(mat_ref, (int, np.integer)):
        return int(mat_ref)
    name = str(mat_ref)
    if opensees is not None:
        nd_tags = getattr(opensees, "_nd_mat_tags", {}) or {}
        uni_tags = getattr(opensees, "_uni_mat_tags", {}) or {}
        if name in nd_tags:
            return int(nd_tags[name])
        if name in uni_tags:
            return int(uni_tags[name])
    return _stable_section_tag(name)


def _stable_section_tag(name: str) -> int:
    """Deterministic positive int tag derived from a section name.

    Used as a fallback when build() hasn't run yet (so
    ``opensees._sec_tags`` is empty). Stable across processes.
    """
    h = abs(hash(name))
    # Keep it within int32 to avoid surprises in HDF5 attrs.
    return (h % (2 ** 31 - 1)) or 1
