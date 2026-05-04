"""Kind catalog — what diagram kinds are available for a Results file.

Replaces the topology-first ``_family_catalog`` with a kind-first
walker. The user's mental model is to **stack layers** (contour +
vector glyph + line force + …); each layer is identified by its
*kind*, and the data that feeds it is a secondary choice inside the
creation form.

The catalog is read once per viewer session. Each :class:`KindEntry`
records the data options the kind can render against this particular
file — empty options means the kind has no recordings to draw, so the
UI greys it out with a tooltip.

Kinds with ``requires_data=False`` (e.g. Constraints) have no Data
dropdown — the UI skips that row in the creation form. The Loads
kind uses ``requires_data=True`` with the data combo bound to load
pattern names rather than canonical components.

Note: ``deformed_shape`` is intentionally absent. Deformation is now a
global view modifier in the Session panel (``☐ Deform`` + scale
spinner) that mutates ``scene.grid.points`` at every step; every layer
that paints on the substrate inherits the deformed shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

from ...solvers._element_response import split_canonical_component

if TYPE_CHECKING:
    from ._director import ResultsDirector


# ---------------------------------------------------------------------
# Catalog rows
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class KindEntry:
    """One catalog row — a diagram kind with its data options for this file.

    Attributes
    ----------
    kind_id
        Diagram-class registry key (``"contour"``, ``"vector_glyph"``, …).
    label
        User-facing display name in the Kind dropdown.
    requires_data
        Whether the creation form needs a Data combo. Loads /
        Constraints / Gauss markers don't pick a component.
    data_options
        Component / prefix names this kind can render. For
        ``vector_glyph`` these are *prefixes* (``"displacement"``)
        because the kind reads x/y/z together; for everything else
        these are full canonical names (``"stress_xx"``).
        Empty = nothing in this file feeds the kind → the UI disables
        the row.
    default_data
        Pre-selected entry in the Data combo. ``None`` for kinds with
        ``requires_data=False`` or empty ``data_options``.
    topology_hint
        Optional topology to set on the diagram's style at construction
        time (``"gauss"`` for Contour against gauss data, etc.).
        ``None`` lets the diagram class decide.
    enabled
        Mirror of ``bool(data_options) or not requires_data`` — kept
        as an explicit attribute so UI code reads it directly.
    """
    kind_id: str
    label: str
    requires_data: bool
    data_options: tuple[str, ...]
    default_data: Optional[str] = None
    topology_hint: Optional[str] = None
    enabled: bool = True


# ---------------------------------------------------------------------
# Component prefix helpers
# ---------------------------------------------------------------------

# Canonical axis ordering for sorting components within a family.
_AXIS_ORDER: dict[str, int] = {
    "x":  0, "y":  1, "z":  2,
    "xx": 10, "yy": 11, "zz": 12, "xy": 13, "yz": 14, "xz": 15,
}


def _component_sort_key(name: str) -> tuple[int, str]:
    parts = split_canonical_component(name)
    if parts is None:
        return (999, name)
    return (_AXIS_ORDER.get(parts[1], 99), name)


def _vector_prefixes(components: Iterable[str]) -> list[str]:
    """Return prefixes that have ≥ 2 axis components in the iterable.

    Vector glyph needs x and y (and optionally z) together to draw
    arrows; a prefix with only one axis recorded is not a vector.
    """
    by_prefix: dict[str, set[str]] = {}
    for name in components:
        parts = split_canonical_component(name)
        if parts is None:
            continue
        prefix, suf = parts
        # Skip tensor suffixes — vectors are x/y/z only.
        if suf not in {"x", "y", "z"}:
            continue
        by_prefix.setdefault(prefix, set()).add(suf)
    return sorted(p for p, axes in by_prefix.items() if len(axes) >= 2)


# ---------------------------------------------------------------------
# Available-components per topology (mirrors _add_diagram_dialog routing)
# ---------------------------------------------------------------------

def _available_components(scoped: object, topology: str) -> list[str]:
    """Pull ``available_components()`` from the right composite."""
    try:
        if topology == "nodes":
            return sorted(scoped.nodes.available_components())
        if topology == "elements":
            return sorted(scoped.elements.nodal_forces.available_components())
        if topology == "gauss":
            return sorted(scoped.elements.gauss.available_components())
        if topology == "line_stations":
            return sorted(
                scoped.elements.line_stations.available_components()
            )
        if topology == "fibers":
            return sorted(scoped.elements.fibers.available_components())
        if topology == "layers":
            return sorted(scoped.elements.layers.available_components())
        if topology == "springs":
            return sorted(scoped.elements.springs.available_components())
    except Exception:
        return []
    return []


def _load_patterns_with_forces(director: "ResultsDirector") -> list[str]:
    """Return load pattern names that have at least one non-zero force.

    Patterns with only moments are excluded — the LoadsDiagram draws
    forces only, so a moments-only pattern would attach to nothing.
    Data source is ``director.fem.nodes.loads`` (FEM-side records),
    not the Results composite.
    """
    fem = getattr(director, "fem", None)
    if fem is None:
        return []
    try:
        ns_loads = fem.nodes.loads
    except Exception:
        return []
    out: list[str] = []
    try:
        patterns = list(ns_loads.patterns())
    except Exception:
        return []
    for p in patterns:
        try:
            records = ns_loads.by_pattern(p)
        except Exception:
            continue
        for r in records:
            f = getattr(r, "force_xyz", None)
            if f is None:
                continue
            if any(abs(float(v)) > 0.0 for v in f):
                out.append(p)
                break
    return out


def _union_across_stages(
    director: "ResultsDirector", topology: str,
) -> list[str]:
    """Union ``available_components()`` for ``topology`` across every stage."""
    out: set[str] = set()
    try:
        stages = list(director.stages())
    except Exception:
        return []
    for s in stages:
        sid = getattr(s, "id", None)
        if sid is None:
            continue
        try:
            scoped = director.results.stage(sid)
        except Exception:
            continue
        for comp in _available_components(scoped, topology):
            out.add(comp)
    return sorted(out)


# ---------------------------------------------------------------------
# Public — build the catalog for a Results
# ---------------------------------------------------------------------

# Kinds in the order they appear in the Add layer dropdown. Kept
# stable; UI orders enabled-first then disabled-last but preserves
# this order within each group.
_KIND_DEFINITIONS: tuple[tuple[str, str, bool, str, Optional[str]], ...] = (
    # (kind_id, label, requires_data, primary_topology, fallback_topology)
    ("contour",       "Contour",                 True,  "nodes", "gauss"),
    ("vector_glyph",  "Vector glyph (arrows)",   True,  "nodes", None),
    ("line_force",    "Line force diagram",      True,  "line_stations", None),
    ("fiber_section", "Fiber section",           True,  "fibers", None),
    ("layer_stack",   "Layer stack (shells)",    True,  "layers", None),
    # Gauss point markers — sphere glyphs at GP world positions,
    # colored by a scalar. The kind needs a data component (the
    # value driving the colormap), pulled from the gauss composite.
    ("gauss_marker",  "Gauss point markers",     True,  "gauss",  None),
    ("spring_force",  "Spring force",            True,  "springs", None),
    # Applied loads — arrows for nodal force vectors, scoped per
    # load pattern. Data combo is populated from
    # ``fem.nodes.loads.patterns()`` rather than a Results composite,
    # since loads live on FEM, not the recorder output.
    ("loads",         "Applied loads",           True,  "loads", None),
    # Reactions — recorded reaction forces and moments at constrained
    # nodes, scaled per step. The kind is enabled iff the file
    # actually carries ``reaction_force_*`` or ``reaction_moment_*``
    # nodal recordings; no Data combo (force + moment families are
    # both rendered when present).
    ("reactions",     "Reactions",               False, "nodes", None),
)


def build_catalog(director: "ResultsDirector") -> list[KindEntry]:
    """Walk the file once and emit one :class:`KindEntry` per kind.

    Returns kinds in :data:`_KIND_DEFINITIONS` order. Disabled kinds
    (no data feeds them) are still returned — the UI greys them out
    rather than hiding them, so the user can see what *isn't* in the
    file.
    """
    out: list[KindEntry] = []
    for kind_id, label, requires_data, primary, fallback in _KIND_DEFINITIONS:
        # Compute the candidate Data list for each kind.
        if kind_id == "contour":
            # Union of nodes + gauss scalars; sorted by canonical axis.
            comps = sorted(
                set(_union_across_stages(director, "nodes"))
                | set(_union_across_stages(director, "gauss")),
                key=_component_sort_key,
            )
            data_options = tuple(comps)
            topology_hint = None
        elif kind_id == "vector_glyph":
            # Prefixes with ≥ 2 axes recorded on nodes.
            data_options = tuple(_vector_prefixes(
                _union_across_stages(director, "nodes")
            ))
            topology_hint = "nodes"
        elif kind_id == "loads":
            # Pattern names that have at least one record with a
            # non-zero force vector. Moments-only patterns are hidden
            # since the diagram only renders forces today.
            data_options = tuple(_load_patterns_with_forces(director))
            topology_hint = None
        elif requires_data:
            # All other data-bound kinds use their primary topology
            # only — no fallback. Primary topology empty → kind
            # disabled.
            comps = sorted(
                _union_across_stages(director, primary),
                key=_component_sort_key,
            )
            data_options = tuple(comps)
            topology_hint = None
        elif kind_id == "reactions":
            # Reactions kind has no Data combo — both force and
            # moment families render together when present. Enabled
            # iff the file actually carries any reaction recording.
            nodal = set(_union_across_stages(director, "nodes"))
            has_reaction = any(
                comp in nodal for comp in (
                    "reaction_force_x",
                    "reaction_force_y",
                    "reaction_force_z",
                    "reaction_moment_x",
                    "reaction_moment_y",
                    "reaction_moment_z",
                )
            )
            out.append(KindEntry(
                kind_id=kind_id, label=label,
                requires_data=False,
                data_options=(),
                default_data=None,
                topology_hint=None,
                enabled=has_reaction,
            ))
            continue
        else:
            # No-data kinds (gauss markers): "enabled" iff their
            # primary topology has *any* recordings (so we don't show
            # a Gauss-points marker layer for a file with no gauss
            # data).
            primary_has_data = bool(
                _union_across_stages(director, primary)
            )
            data_options = ()
            topology_hint = None
            out.append(KindEntry(
                kind_id=kind_id, label=label,
                requires_data=False,
                data_options=(),
                default_data=None,
                topology_hint=topology_hint,
                enabled=primary_has_data,
            ))
            continue

        default = data_options[0] if data_options else None
        out.append(KindEntry(
            kind_id=kind_id, label=label,
            requires_data=True,
            data_options=data_options,
            default_data=default,
            topology_hint=topology_hint,
            enabled=bool(data_options),
        ))
    return out
