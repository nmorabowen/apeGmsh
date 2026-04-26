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
    RecorderRecord,
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData


# Per-category sets of allowed canonical components. ``state_variable_*``
# is allowed in any element-level category; checked separately.
_CATEGORY_COMPONENTS: dict[str, frozenset[str]] = {
    "nodes": frozenset(NODAL_KINEMATICS + NODAL_FORCES),
    "elements": frozenset(PER_ELEMENT_NODAL_FORCES),
    "line_stations": frozenset(LINE_DIAGRAMS),
    "gauss": frozenset(STRESS + STRAIN + DERIVED_SCALARS + MATERIAL_STATE),
    "fibers": frozenset(FIBER + MATERIAL_STATE),
    "layers": frozenset(STRESS + STRAIN),
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
    class has no required parent reference — records and ``resolve()``
    operate on plain data plus a :class:`FEMData` argument.
    """

    def __init__(self) -> None:
        self._records: list[RecorderRecord] = []
        self._auto_id: int = 0

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
        """Declare a nodal recorder. See module docstring for selectors."""
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
        """Declare a per-element-node recorder (globalForce / localForce)."""
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
        """Declare a line-stations recorder (beam section forces)."""
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
        """Declare a continuum Gauss-point recorder (stress / strain / …)."""
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
        """Declare a fiber-section recorder (uniaxial along fiber axis)."""
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
        """Declare a layered-shell layer recorder."""
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

        Each requested mode lands as a stage with ``kind="mode"`` in
        the resulting native HDF5 (per Phase 1 design).
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
        return ResolvedRecorderRecord(
            category=rec.category,
            name=rec.name,
            components=expanded,
            dt=rec.dt, n_steps=rec.n_steps,
            element_ids=ids_array,
            source=rec,
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
