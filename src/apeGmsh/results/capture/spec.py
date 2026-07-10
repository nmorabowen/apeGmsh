"""DomainCaptureSpec — declarative spec for in-process domain capture.

Phase 9 commit 5 introduces this results-side sibling of the bridge's
file-emit recorder declaration. Two separate systems for two separate
responsibilities:

- :func:`apeGmsh.opensees.ops.recorder.declare` produces a typed
  :class:`apeGmsh.opensees.recorder.RecorderDeclaration` for the
  **file-emit** path (classic ``.out`` recorders, MPCO).
- :class:`DomainCaptureSpec` produces a
  :class:`ResolvedDomainCaptureSpec` for the **in-process**
  :class:`apeGmsh.results.capture.DomainCapture` path (native HDF5
  written from a live openseespy domain).

Both validate against the same canonical vocabulary in
:mod:`apeGmsh._vocabulary`; both source ``ndm`` / ``ndf`` implicitly
per Phase 9 D8 (no consumer args).

Usage
-----
::

    # Live: ndm/ndf sourced from the attached bridge.
    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(pg="Top", components=["displacement", "reaction"])
    spec.gauss(pg="Body", components=["stress", "von_mises_stress"])
    with ops.domain_capture(spec, path="run.h5") as cap:
        cap.begin_stage("gravity", kind="static")
        for _ in range(n):
            ops.analyze(1, 1.0)
            cap.step(t=ops.getTime())
        cap.end_stage()

    # File: ndm/ndf sourced from model.h5 ``/meta``.
    spec = DomainCaptureSpec()
    spec.nodes(components=["displacement"], ids=[1, 2, 3])
    with DomainCapture.from_h5(
        "model.h5", spec=spec, fem=fem, output="run.h5",
    ) as cap:
        ...

Why a sibling instead of reusing the bridge's typed declaration?
-----------------------------------------------------------------
The file-emit path's declaration only stores selectors as strings
(``pg=`` / ``label=`` / ``selection=`` / ``ids=``) and resolves them
at emit time inside the bridge's build pipeline. DomainCapture
needs **already-resolved** concrete ID arrays per record (the
capturers walk them on every ``step()`` call). The two systems also
diverge in supported categories: DomainCapture handles all seven
(including fibers / layers / modal), while file-emit handles four
(nodes / elements / line_stations / gauss).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

import numpy as np
from numpy import ndarray

from ..._vocabulary import (
    DERIVED_SCALARS,
    FIBER,
    LINE_DIAGRAMS,
    MATERIAL_STATE,
    NODAL_FORCES,
    NODAL_KINEMATICS,
    PER_ELEMENT_NODAL_FORCES,
    PLASTIC_STRAIN,
    STRAIN,
    STRESS,
    expand_many,
    is_canonical,
)
# LayerSectionDef / LayerSectionMetadata are pure-data types reused
# from the resolved spec module — they describe layered-shell
# composition, no coupling to either spec system.
from ..spec._resolved import LayerSectionDef, LayerSectionMetadata

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData


# Allowed categories — these match the user-facing methods on
# ``DomainCaptureSpec``. ``modal`` is special: no target, has
# ``n_modes``.
ALL_CATEGORIES: tuple[str, ...] = (
    "nodes",
    "elements",
    "line_stations",
    "gauss",
    "fibers",
    "layers",
    "modal",
)


# ``displacement_increment_*`` is recorder-only: OpenSees exposes the
# per-step displacement increment through the ``incrDisp`` recorder
# column, but openseespy has no in-process per-step increment query, so
# the live capture path cannot produce it (``nodeDisp`` returns the
# total displacement, not the increment). Excluded from the nodes
# category so ``components_for`` / ``where_does`` / validation all agree.
_CAPTURE_UNSUPPORTED_NODE: frozenset[str] = frozenset((
    "displacement_increment_x",
    "displacement_increment_y",
    "displacement_increment_z",
))


# Per-category sets of allowed canonical components. ``state_variable_*``
# is allowed in any element-level category; checked separately.
_CATEGORY_COMPONENTS: dict[str, frozenset[str]] = {
    "nodes": frozenset(NODAL_KINEMATICS + NODAL_FORCES) - _CAPTURE_UNSUPPORTED_NODE,
    "elements": frozenset(PER_ELEMENT_NODAL_FORCES),
    "line_stations": frozenset(LINE_DIAGRAMS),
    "gauss": frozenset(
        STRESS + STRAIN + PLASTIC_STRAIN + DERIVED_SCALARS + MATERIAL_STATE
    ),
    "fibers": frozenset(FIBER + MATERIAL_STATE),
    "layers": frozenset(FIBER + MATERIAL_STATE),
}


# =====================================================================
# Declarative record (pre-resolution)
# =====================================================================

@dataclass(frozen=True)
class DomainCaptureRecord:
    """One declarative DomainCapture entry — what the user wrote.

    Selectors (``pg`` / ``label`` / ``selection`` / ``ids``) follow the
    same vocabulary as the read-side composite API. Multiple named
    selectors combine as union; ``ids`` is mutually exclusive with the
    named selectors. ``modal`` records ignore selectors and use
    ``n_modes`` instead.

    Same shape as the legacy
    :class:`apeGmsh.results.spec._resolved.RecorderRecord` (the helper
    that retired in commit 5), with a separate identity so the
    DomainCapture and file-emit systems can diverge cleanly.
    """

    category: str
    components: tuple[str, ...]
    name: str

    pg: tuple[str, ...] = ()
    label: tuple[str, ...] = ()
    selection: tuple[str, ...] = ()
    ids: Optional[tuple[int, ...]] = None

    dt: Optional[float] = None
    n_steps: Optional[int] = None

    n_modes: Optional[int] = None

    # Optional user-supplied OpenSees C++ class name override for the
    # element-level categories. DomainCapture uses ``ops.eleType(eid)``
    # at capture time so this is rarely needed, but the field is kept
    # for symmetry and to allow forcing a specific class group at the
    # capturer level when the live domain reports ambiguous classes.
    element_class_name: Optional[str] = None


# =====================================================================
# Resolved record (post-resolution against FEMData)
# =====================================================================

@dataclass(frozen=True)
class ResolvedDomainCaptureRecord:
    """A :class:`DomainCaptureRecord` after resolution against FEMData.

    All selectors collapse to concrete ID arrays. The original
    declaration is preserved on ``source`` for inspection / debugging.
    Carries the same fields as the legacy ``ResolvedRecorderRecord``
    so DomainCapture's per-category capturers can consume either type
    via attribute access (structurally identical; nominally distinct).
    """

    category: str
    name: str
    components: tuple[str, ...]
    dt: Optional[float]
    n_steps: Optional[int]

    node_ids: Optional[ndarray] = None
    element_ids: Optional[ndarray] = None

    element_class_name: Optional[str] = None

    n_modes: Optional[int] = None

    # ``category="layers"``-only — populated when the resolver has
    # an OpenSees back-reference and the record's elements are
    # assigned a LayeredShellFiberSection. ``None`` otherwise.
    layer_section_metadata: Optional[LayerSectionMetadata] = None

    source: Optional[DomainCaptureRecord] = None


# =====================================================================
# Resolved spec
# =====================================================================

@dataclass(frozen=True)
class ResolvedDomainCaptureSpec:
    """A complete DomainCapture spec resolved against a FEMData snapshot.

    Carries the FEMData ``snapshot_id`` (for inspection / paranoia),
    a tuple of :class:`ResolvedDomainCaptureRecord`, and the
    ``ndm`` / ``ndf`` that were in effect at resolve time. Per
    Phase 9 D8 the spec is the single source of truth for those
    dimensions — :class:`DomainCapture` reads them off the spec
    rather than taking redundant kwargs.

    Consumed only by :class:`apeGmsh.results.capture.DomainCapture`
    — file-emit paths use the bridge's typed
    :class:`apeGmsh.opensees.recorder.RecorderDeclaration` instead.
    """

    fem_snapshot_id: str
    records: tuple[ResolvedDomainCaptureRecord, ...] = field(default_factory=tuple)
    ndm: int = 3
    ndf: int = 6

    def __iter__(self) -> Iterator[ResolvedDomainCaptureRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def by_category(
        self, category: str,
    ) -> tuple[ResolvedDomainCaptureRecord, ...]:
        """Records whose ``category`` matches."""
        return tuple(r for r in self.records if r.category == category)

    def __repr__(self) -> str:
        lines = [
            f"ResolvedDomainCaptureSpec("
            f"fem_snapshot_id={self.fem_snapshot_id[:8]}…, "
            f"{len(self.records)} records)"
        ]
        for r in self.records:
            sel = self._record_selection_summary(r)
            cad = self._record_cadence_summary(r)
            lines.append(
                f"  - {r.category} {r.name!r}: {sel}, {cad}, "
                f"components={list(r.components)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _record_selection_summary(r: ResolvedDomainCaptureRecord) -> str:
        if r.category == "modal":
            return f"n_modes={r.n_modes}"
        if r.category == "nodes":
            n = 0 if r.node_ids is None else r.node_ids.size
            return f"{n} nodes"
        n = 0 if r.element_ids is None else r.element_ids.size
        return f"{n} elements"

    @staticmethod
    def _record_cadence_summary(r: ResolvedDomainCaptureRecord) -> str:
        if r.dt is not None:
            return f"dt={r.dt}"
        if r.n_steps is not None:
            return f"every {r.n_steps} steps"
        return "every step"


# =====================================================================
# DomainCaptureSpec composite
# =====================================================================

class DomainCaptureSpec:
    """Declarative spec builder for :class:`DomainCapture`.

    Standalone ``DomainCaptureSpec()`` supports the full declarative
    surface (every per-category method + introspection); it requires
    an OpenSees bridge attachment (``DomainCaptureSpec(opensees=ops)``)
    for :meth:`resolve` so ``ndm`` / ``ndf`` are unambiguous per
    Phase 9 D8.

    The :class:`DomainCapture` class wraps the resolved output and
    drives the in-process capture pipeline. Use one of:

    - ``ops.domain_capture(spec, path="run.h5")`` for live capture
      (bridge attached; resolve happens inside the bridge entry point).
    - ``DomainCapture.from_h5("model.h5", spec=spec, fem=fem,
      output="run.h5")`` for a non-bridge setup (``ndm`` / ``ndf``
      sourced from the model file's ``/meta``).

    Layered-shell section metadata is populated by :meth:`resolve`
    when an OpenSees bridge is attached; for the ``from_h5`` path the
    metadata is built from the model file's section registry inside
    :meth:`DomainCapture.from_h5`.
    """

    def __init__(self, opensees: Any = None) -> None:
        self._records: list[DomainCaptureRecord] = []
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
    ) -> "DomainCaptureSpec":
        """Declare a nodal capture record.

        Components and selectors mirror the file-emit path
        (:func:`ops.recorder.declare`). See the module docstring for
        the full vocabulary; this method differs in that records
        resolve to concrete IDs for in-process capture rather than
        fanning out to ``.out`` files, and cadence is driven by the
        user's ``step(t)`` loop (per-record ``dt`` / ``n_steps`` are
        not honored on the capture path — see :func:`_validate_cadence`).

        Example::

            spec.nodes(
                components=["displacement", "reaction_force"],
                pg="Top",
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
        element_class_name: str | None = None,
    ) -> "DomainCaptureSpec":
        """Declare a per-element-node force capture record."""
        self._declare(
            "elements", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
            element_class_name=element_class_name,
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
        element_class_name: str | None = None,
    ) -> "DomainCaptureSpec":
        """Declare a beam line-station capture record."""
        self._declare(
            "line_stations", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
            element_class_name=element_class_name,
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
        element_class_name: str | None = None,
    ) -> "DomainCaptureSpec":
        """Declare a continuum Gauss-point capture record."""
        self._declare(
            "gauss", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
            element_class_name=element_class_name,
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
        element_class_name: str | None = None,
    ) -> "DomainCaptureSpec":
        """Declare a fiber-section capture record (uniaxial)."""
        self._declare(
            "fibers", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
            element_class_name=element_class_name,
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
        element_class_name: str | None = None,
    ) -> "DomainCaptureSpec":
        """Declare a layered-shell capture record (per-layer)."""
        self._declare(
            "layers", components,
            pg=pg, label=label, selection=selection, ids=ids,
            dt=dt, n_steps=n_steps, name=name,
            element_class_name=element_class_name,
        )
        return self

    def modal(
        self,
        n_modes: int,
        *,
        dt: float | None = None,
        n_steps: int | None = None,
        name: str | None = None,
    ) -> "DomainCaptureSpec":
        """Declare a modal-shape capture record.

        Modal records trigger an eigenvalue analysis at capture time
        via :meth:`DomainCapture.capture_modes`. Each mode lands as
        its own stage with ``kind="mode"`` in the resulting file.
        """
        if not isinstance(n_modes, int) or n_modes <= 0:
            raise ValueError(
                f"n_modes must be a positive int (got {n_modes!r})."
            )
        _validate_cadence(dt, n_steps)
        rec_name = name or f"modal_{self._auto_id}"
        self._auto_id += 1
        self._records.append(DomainCaptureRecord(
            category="modal",
            components=(),
            name=rec_name,
            dt=dt, n_steps=n_steps,
            n_modes=n_modes,
        ))
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @staticmethod
    def categories() -> tuple[str, ...]:
        """Return the seven categories you can declare on a DomainCaptureSpec."""
        return ALL_CATEGORIES

    @staticmethod
    def components_for(category: str) -> tuple[str, ...]:
        """Return the canonical component names allowed in ``category``.

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
                f"{DomainCaptureSpec.categories()}."
            )
        return tuple(sorted(_CATEGORY_COMPONENTS[category]))

    @staticmethod
    def shorthands_for(category: str) -> dict[str, tuple[str, ...]]:
        """Return shorthand → expansion mapping valid in ``category``."""
        from ... import _vocabulary as _voc

        if category == "modal":
            return {}

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

        if category == "nodes":
            out["reaction"] = _voc._SHORTHAND_REACTION

        return out

    @staticmethod
    def where_does(component: str) -> tuple[str, ...]:
        """Return the categories that accept ``component``.

        Reverse of :meth:`components_for` — given a canonical component
        name, returns the tuple of ``DomainCaptureSpec`` categories
        where ``spec.<category>(components=component, ...)`` would
        validate. Empty tuple if no category accepts it.

        Mirrors :func:`_validate_components_for_category` exactly,
        including the ``state_variable_*`` wildcard which routes to
        ``gauss`` / ``fibers`` / ``layers`` for any suffix.

        Examples
        --------
        >>> DomainCaptureSpec.where_does("axial_force")
        ('line_stations',)
        >>> DomainCaptureSpec.where_does("displacement_x")
        ('nodes',)
        >>> DomainCaptureSpec.where_does("state_variable_42")
        ('fibers', 'gauss', 'layers')
        >>> DomainCaptureSpec.where_does("not_a_component")
        ()
        """
        matches: set[str] = {
            cat for cat, allowed in _CATEGORY_COMPONENTS.items()
            if component in allowed
        }
        if component.startswith("state_variable_"):
            matches |= {"gauss", "fibers", "layers"}
        return tuple(sorted(matches))

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, fem: "FEMData") -> ResolvedDomainCaptureSpec:
        """Resolve all records against a FEMData snapshot.

        Per Phase 9 D8 sources ``ndm`` / ``ndf`` from the attached
        bridge. Standalone instances must use one of the
        :class:`DomainCapture` construction paths
        (``ops.domain_capture`` / ``DomainCapture.from_h5``) that
        provide ``ndm`` / ``ndf`` explicitly.

        Raises
        ------
        RuntimeError
            If no bridge is attached, or if the attached bridge has
            not yet had ``ops.model(ndm=, ndf=)`` called.
        """
        if self._opensees is None:
            raise RuntimeError(
                "DomainCaptureSpec.resolve(fem): no OpenSees bridge "
                "attached. Standalone DomainCaptureSpec() supports "
                "declaration and introspection only; .resolve() needs "
                "a bridge to source ndm/ndf (Phase 9 D8). Either "
                "construct as DomainCaptureSpec(opensees=ops) and "
                "call ops.model(ndm=, ndf=) first, or use "
                "DomainCapture.from_h5(\"model.h5\", spec=spec, ...) "
                "which sources ndm/ndf from the file's /meta."
            )
        ndm = getattr(self._opensees, "_ndm", None)
        ndf = getattr(self._opensees, "_ndf", None)
        if ndm is None or ndf is None:
            raise RuntimeError(
                "DomainCaptureSpec.resolve(fem): the attached bridge "
                "has no ndm/ndf set. Call ops.model(ndm=, ndf=) "
                "before resolving (Phase 9 D8 binds ndm/ndf at "
                "resolve time)."
            )
        return self._resolve_with_explicit_ndm_ndf(fem, ndm=ndm, ndf=ndf)

    def _resolve_with_explicit_ndm_ndf(
        self,
        fem: "FEMData",
        *,
        ndm: int,
        ndf: int,
    ) -> ResolvedDomainCaptureSpec:
        """Internal resolve used by both the bridge and the from_h5 paths.

        Not part of the user-facing surface — callers go through
        :meth:`resolve` (bridge path) or :meth:`DomainCapture.from_h5`
        (file path), which supply ``ndm`` / ``ndf`` from the
        appropriate source.
        """
        resolved: list[ResolvedDomainCaptureRecord] = []
        for rec in self._records:
            resolved.append(self._resolve_one(rec, fem, ndm=ndm, ndf=ndf))
        return ResolvedDomainCaptureSpec(
            fem_snapshot_id=fem.snapshot_id,
            records=tuple(resolved),
            ndm=ndm,
            ndf=ndf,
        )

    def _resolve_one(
        self,
        rec: DomainCaptureRecord,
        fem: "FEMData",
        *,
        ndm: int,
        ndf: int,
    ) -> ResolvedDomainCaptureRecord:
        if rec.category == "modal":
            return ResolvedDomainCaptureRecord(
                category="modal",
                name=rec.name,
                components=(),
                dt=rec.dt,
                n_steps=rec.n_steps,
                n_modes=rec.n_modes,
                source=rec,
            )

        expanded = expand_many(rec.components, ndm=ndm, ndf=ndf)
        if not expanded:
            raise ValueError(
                f"Record {rec.name!r} ({rec.category}) expanded to zero "
                f"components in ndm={ndm}, ndf={ndf}. "
                f"Original: {list(rec.components)}"
            )
        _validate_components_for_category(rec.category, expanded)

        if rec.category == "nodes":
            ids_array = _resolve_node_selectors(fem, rec)
            return ResolvedDomainCaptureRecord(
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
        class_hint = rec.element_class_name
        if class_hint is None:
            class_hint = self._lookup_class_hint_for_pgs(rec.pg)
        return ResolvedDomainCaptureRecord(
            category=rec.category,
            name=rec.name,
            components=expanded,
            dt=rec.dt, n_steps=rec.n_steps,
            element_ids=ids_array,
            element_class_name=class_hint,
            layer_section_metadata=layer_metadata,
            source=rec,
        )

    def _lookup_class_hint_for_pgs(
        self, pgs: tuple[str, ...],
    ) -> str | None:
        """Resolve PG names to a single C++ element class name.

        Walks the attached bridge's ``_elem_assignments`` and
        ``_ELEM_REGISTRY``; returns ``None`` if there's no bridge, no
        PGs were named, or multiple element classes are involved.
        """
        if not pgs or self._opensees is None:
            return None
        elem_assignments = getattr(
            self._opensees, "_elem_assignments", None,
        ) or {}
        if not elem_assignments:
            return None
        from ...opensees._element_capabilities import _ELEM_REGISTRY
        names: set[str] = set()
        for pg_name in pgs:
            asgn = elem_assignments.get(pg_name)
            if asgn is None:
                continue
            ops_type = asgn.get("ops_type")
            if ops_type is None:
                continue
            spec = _ELEM_REGISTRY.get(ops_type)
            if spec is None:
                continue
            names.add(spec.cpp_class_name or ops_type)
        if len(names) == 1:
            return next(iter(names))
        return None

    def _resolve_layer_section_metadata(
        self,
        fem: "FEMData",
        element_ids: ndarray,
    ) -> Optional[LayerSectionMetadata]:
        """Build LayerSectionMetadata from the OpenSees back-reference."""
        if self._opensees is None:
            return None

        sections_registry = self._opensees._sections
        elem_assignments = self._opensees._elem_assignments

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

    def __iter__(self) -> Iterator[DomainCaptureRecord]:
        return iter(self._records)

    def clear(self) -> None:
        """Remove all declared records."""
        self._records.clear()
        self._auto_id = 0

    def __repr__(self) -> str:
        if not self._records:
            return "DomainCaptureSpec(empty)"
        lines = [f"DomainCaptureSpec({len(self._records)} records):"]
        for r in self._records:
            sel = self._record_selectors_repr(r)
            cad = self._record_cadence_repr(r)
            comps = list(r.components) if r.components else f"n_modes={r.n_modes}"
            lines.append(f"  - {r.category} {r.name!r}: {sel}, {cad}, {comps}")
        return "\n".join(lines)

    @staticmethod
    def _record_selectors_repr(r: DomainCaptureRecord) -> str:
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
    def _record_cadence_repr(r: DomainCaptureRecord) -> str:
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
        pg: str | Iterable[str] | None,
        label: str | Iterable[str] | None,
        selection: str | Iterable[str] | None,
        ids: Iterable[int] | None,
        dt: float | None,
        n_steps: int | None,
        name: str | None,
        element_class_name: str | None = None,
    ) -> None:
        if category not in ALL_CATEGORIES:
            raise ValueError(
                f"Unknown DomainCaptureSpec category {category!r}. "
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

        self._records.append(DomainCaptureRecord(
            category=category,
            components=comp_tuple,
            name=rec_name,
            pg=_to_str_tuple(pg),
            label=_to_str_tuple(label),
            selection=_to_str_tuple(selection),
            ids=tuple(int(i) for i in ids) if ids is not None else None,
            dt=dt,
            n_steps=n_steps,
            element_class_name=element_class_name,
        ))


# =====================================================================
# Validation helpers
# =====================================================================

def _normalize_components(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _to_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _validate_cadence(dt: float | None, n_steps: int | None) -> None:
    """Per-record cadence is not honored on the in-process capture path.

    DomainCapture records a snapshot on every ``DomainCapture.step(t)``
    call — the user's analysis loop sets the cadence, not the record.
    A per-record ``dt`` / ``n_steps`` would be silently ignored here, so
    we reject it loudly rather than imply a sub-sampling that never
    happens. For sub-sampled, per-record cadence use the file-emit path
    (``ops.recorder.declare(..., dt=/n_steps=)``).
    """
    if dt is not None or n_steps is not None:
        raise ValueError(
            "dt=/n_steps= are not supported on the in-process capture "
            "path: DomainCapture records on every step(t) call, so the "
            "analysis loop controls cadence. Drive step(t) at the rate "
            "you want, or use ops.recorder.declare(..., dt=/n_steps=) "
            "for sub-sampled file output."
        )


def _validate_selector_exclusivity(
    pg: Any, label: Any, selection: Any, ids: Any,
) -> None:
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
        if comp in _CAPTURE_UNSUPPORTED_NODE:
            raise ValueError(
                f"Component {comp!r} cannot be captured in-process: "
                f"OpenSees exposes displacement increments only through "
                f"the 'incrDisp' recorder column, and openseespy has no "
                f"per-step increment query (nodeDisp returns the total "
                f"displacement). Use the file-emit path "
                f"(ops.recorder.declare(...)), or capture 'displacement' "
                f"and difference it post hoc."
            )
        if comp.startswith("state_variable_") and category in (
            "gauss", "fibers", "layers",
        ):
            continue
        if comp not in allowed:
            raise ValueError(
                f"Component {comp!r} is not valid for DomainCaptureSpec "
                f"category {category!r}. Valid components for "
                f"{category!r}: "
                f"{sorted(allowed)[:8]}{'...' if len(allowed) > 8 else ''}"
            )


# =====================================================================
# Selector resolution helpers
# =====================================================================

def _resolve_node_selectors(
    fem: "FEMData", rec: DomainCaptureRecord,
) -> ndarray:
    """Resolve a node-side record's selectors to a concrete ID array."""
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
    fem: "FEMData", rec: DomainCaptureRecord,
) -> ndarray:
    """Resolve an element-side record's selectors to a concrete ID array."""
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

_LAYERED_SHELL_TYPES = frozenset({
    "LayeredShell",
    "LayeredShellFiberSection",
})


def _is_layered_shell_section(sec_def: dict[str, Any]) -> bool:
    return sec_def.get("section_type", "") in _LAYERED_SHELL_TYPES


def _layered_shell_section_def(
    *, sec_tag: int, name: str, raw: dict[str, Any],
    opensees: Any = None,
) -> "LayerSectionDef":
    """Normalise a registered LayeredShell section's params."""
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
    mat_ref: Any, opensees: Any,
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
    """Deterministic positive int tag derived from a section name."""
    h = abs(hash(name))
    return (h % (2 ** 31 - 1)) or 1


__all__ = [
    "ALL_CATEGORIES",
    "DomainCaptureRecord",
    "DomainCaptureSpec",
    "ResolvedDomainCaptureRecord",
    "ResolvedDomainCaptureSpec",
]
