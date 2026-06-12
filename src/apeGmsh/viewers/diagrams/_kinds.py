"""Declarative diagram-kind registry (ADR 0058 S0).

One registration per diagram kind, declared in the diagram's own module
via the :func:`register_diagram_kind` class decorator. This is the
single source of truth for every per-kind table that used to be
hand-maintained in parallel — and drift apart:

* ``ui/_add_diagram_dialog.py`` ``_KINDS`` (label / class / style
  factory) + the derived ``_KIND_TO_TOPOLOGY``;
* ``_kind_catalog.py`` ``_KIND_DEFINITIONS`` (kind availability for a
  Results file — labels had already drifted from the dialog's);
* ``_session.py`` ``_KIND_TO_STYLE`` (had silently dropped ``loads``
  and ``reactions``, so those layers never survived a session restore);
* ``_style_presets.py`` ``KIND_TO_STYLE_CLASS``.

Adding a new diagram kind is now: write the class file (whose
decorator registers it) + tests. Consumers iterate :func:`all_kinds`
or look up :func:`kind_def` at call time — by then the package import
below has populated the registry.

Import-order note: every diagram module imports this one, and the
package ``__init__`` imports every diagram module — so importing
*anything* under ``apeGmsh.viewers.diagrams`` populates the registry.
The accessors still call :func:`_ensure_populated` so a consumer that
somehow grabbed this module first (e.g. via a direct submodule import
in a test) sees the full registry, not a partial one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ._styles import DiagramStyle


# Sentinel: "derive data_topology from the class's ``topology`` attr".
# Distinct from an explicit ``None`` (= no Results composite feeds this
# kind; skipped by every component-enumeration consumer — section cut).
_FROM_CLASS = "__from_class__"


@dataclass(frozen=True)
class DiagramKindDef:
    """One registered diagram kind.

    Attributes
    ----------
    kind_id
        ``DiagramSpec.kind`` discriminator — read from the class-level
        ``kind`` attribute at registration so the two can't drift.
    label
        The user-facing display name (Add Diagram dialog, settings-tab
        cards, kind catalog — one label everywhere).
    diagram_class
        The :class:`Diagram` subclass to instantiate.
    style_class
        Style dataclass for session / preset (de)serialization.
    style_factory
        Optional per-component default-style builder. ``None`` means
        ``style_class()`` (no-arg defaults).
    order
        Sort key for UI listings (dialog kind combo, catalog).
    in_catalog
        Whether the kind appears in the kind-availability catalog
        (:func:`._kind_catalog.build_catalog`) and therefore in the
        settings-tab creation panel. ``deformed_shape`` (legacy view
        modifier) and ``section_cut`` (its own creation flow) opt out.
    requires_data
        Whether the creation form needs a Data/Component combo.
    data_topology
        Results-composite key whose ``available_components()``
        populates the Component combo (``"nodes"``, ``"gauss"``,
        ``"line_stations"``, …). ``None`` = no composite to enumerate;
        component-driven consumers skip the kind.
    """

    kind_id: str
    label: str
    diagram_class: type
    style_class: type
    style_factory: Optional[Callable[[str], Any]]
    order: int
    in_catalog: bool = True
    requires_data: bool = True
    data_topology: Optional[str] = None

    def make_default_style(self, component: str) -> "DiagramStyle":
        """Default style for a fresh diagram of this kind."""
        if self.style_factory is not None:
            return self.style_factory(component)
        return self.style_class()


_REGISTRY: dict[str, DiagramKindDef] = {}


def register_diagram_kind(
    *,
    label: str,
    style_class: type,
    style_factory: Optional[Callable[[str], Any]] = None,
    order: int,
    in_catalog: bool = True,
    requires_data: bool = True,
    data_topology: Any = _FROM_CLASS,
) -> Callable[[type], type]:
    """Class decorator — register a :class:`Diagram` subclass as a kind.

    ``kind_id`` is read from the class-level ``kind`` attribute;
    ``data_topology`` defaults to the class-level ``topology`` attribute
    (pass ``data_topology=None`` explicitly for kinds with no Results
    composite to enumerate).
    """
    def _decorate(cls: type) -> type:
        kind_id = getattr(cls, "kind", "")
        if not kind_id:
            raise ValueError(
                f"{cls.__name__} has no class-level 'kind' attribute — "
                "set it before registering."
            )
        existing = _REGISTRY.get(kind_id)
        if existing is not None and (
            existing.diagram_class.__module__ != cls.__module__
            or existing.diagram_class.__qualname__ != cls.__qualname__
        ):
            raise ValueError(
                f"Diagram kind {kind_id!r} already registered by "
                f"{existing.diagram_class.__module__}."
                f"{existing.diagram_class.__qualname__}."
            )
        # Same module+qualname re-registering = module reload; overwrite.
        topo = (
            getattr(cls, "topology", None)
            if data_topology is _FROM_CLASS else data_topology
        )
        _REGISTRY[kind_id] = DiagramKindDef(
            kind_id=kind_id,
            label=label,
            diagram_class=cls,
            style_class=style_class,
            style_factory=style_factory,
            order=order,
            in_catalog=in_catalog,
            requires_data=requires_data,
            data_topology=topo or None,
        )
        return cls
    return _decorate


def _ensure_populated() -> None:
    """Make sure every diagram module has run its registration."""
    import apeGmsh.viewers.diagrams  # noqa: F401  (package init imports all)


def all_kinds() -> tuple[DiagramKindDef, ...]:
    """Every registered kind, in UI listing order."""
    _ensure_populated()
    return tuple(sorted(_REGISTRY.values(), key=lambda d: d.order))


def kind_def(kind_id: str) -> Optional[DiagramKindDef]:
    """Look up one kind by id. ``None`` for unknown kinds."""
    _ensure_populated()
    return _REGISTRY.get(kind_id)


def style_class_for(kind_id: str) -> Optional[type]:
    """Style dataclass for ``kind_id`` — the (de)serialization discriminator."""
    entry = kind_def(kind_id)
    return entry.style_class if entry is not None else None


def kind_ids() -> tuple[str, ...]:
    """All registered kind ids, in UI listing order."""
    return tuple(d.kind_id for d in all_kinds())


__all__ = [
    "DiagramKindDef",
    "all_kinds",
    "kind_def",
    "kind_ids",
    "register_diagram_kind",
    "style_class_for",
]
