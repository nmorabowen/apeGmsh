"""Slot record → ``DiagramSpec`` — the S1 mapping table (ADR 0098 §4).

The session IR stores opaque broker tokens (S0 decision 3); this module
interprets them against the data and the existing per-kind vocabulary.
One entry per §4 category:

| Slot | Kind | Token becomes |
|---|---|---|
| `contour` | `contour` | component + `ContourStyle.topology/averaging` (a quantity recorded at BOTH levels resolves to `nodes`) |
| `vector` | `vector_glyph` or `principal_glyph` | routed by the component suffix |
| `gauss` | `gauss_marker` | component (gauss composite) |
| `line` | `line_force` | component → `COMPONENT_TO_LOCAL_AXIS` inside the kind |
| `sand` | `sand` | component (nodal composite) |
| `loads` | `loads` | **pattern name** (the kind reads it as `selector.component`) |
| `reactions` | `reactions` | `"reactions"` → all axes (`_AXIS_FROM_COMPONENT`) |

Every spec is director-free: ``stage_id`` is pinned on the spec so the
diagram's reads scope without a ``_stage_pin_resolver``.

Scope (§3 INV-MESH-1) rides on ``selector.ids``. ``SlabSelector.ids``
short-circuits BOTH ``resolve_node_ids`` and ``resolve_element_ids``,
so each kind must be handed the id family IT consumes — node ids for
the node-reading kinds, element ids for the element-reading ones.
:data:`ID_FAMILY` records which is which, verified per kind against
the attribute each one actually reads.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

from ...results import _shell_thickness
from ...results._derived import available_derived, is_shell_derived
from ..diagrams._base import DiagramSpec
from ..diagrams._kinds import kind_def
from ..diagrams._selectors import normalize as normalize_selector

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.results.session import Slot
    from ..data import ViewerData
    from ._scope import ScopedSet

#: Which resolved kind reads which id family from its selector.
#:
#: ``contour`` is deliberately ABSENT: it reads node ids or element ids
#: depending on the topology resolved per instance, so a lookup here
#: must fail loudly rather than quietly answer "no restriction" — that
#: answer would drop a view's scope and paint the contour over the
#: whole mesh while every other layer stayed scoped.
ID_FAMILY: dict[str, str] = {
    "vector_glyph": "nodes",    # _vector_glyph.py:109 _resolved_node_ids
    "principal_glyph": "elements",  # _principal_glyph.py:103
    "gauss_marker": "elements",     # _gauss_marker.py:109
    "line_force": "elements",       # _line_force.py:139
    "sand": "elements",             # _sand.py:335
    "reactions": "nodes",           # _reactions.py:152
    "loads": "none",            # reads no resolved ids — see below
}

#: Kinds whose emitted surface is opaque and coincident with the grey
#: substrate, so the substrate must not be drawn under them
#: (INV-MESH-2, generalized from ``Diagram.occludes_substrate``).
OCCLUDING_KINDS: frozenset = frozenset({"contour", "sand"})

#: Component suffixes that name a tensor vs a vector axis. These decide
#: the `vector` slot's kind (see :func:`resolve_vector_kind`).
_TENSOR_SUFFIXES = frozenset(
    {"xx", "yy", "zz", "xy", "yz", "xz", "yx", "zy", "zx"}
)
_AXIS_SUFFIXES = frozenset({"x", "y", "z"})


def resolve_contour_topology(
    quantity: str, results: "Results", stage_id: str,
) -> str:
    """``"nodes"`` or ``"gauss"`` — where ``quantity`` comes from.

    Loud on a quantity the realized stage cannot produce at either
    level: a still of a slot nobody recorded would be a lie, not an
    empty picture.

    **Derived scalars count as available.** ``von_mises_stress`` and the
    rest are computed on read from the stored tensor and so never appear
    in the recorded sets — which used to make this refuse them, while
    ``Gauss("von_mises_stress")`` on the same run painted happily. ADR
    0098 §5 asks a contour and a Gauss slot of the same quantity to share
    one legend, and that is unreachable while the contour cannot accept
    the quantity at all.

    :func:`available_derived` is the right oracle and the reader's
    ``available_components`` is not: that lists stored datasets, so it
    answers the same recorded-only question this already asks. What
    ``available_derived`` adds is "and what can be COMPUTED from them",
    which is the actual question.

    **The shell scalars need a thickness, and it has to be READABLE.**
    ``von_mises_shell`` recovers σ = N/t ± 6M/t², and a slot carries a
    bare component name and nothing else. The read path resolves the
    thickness per element from the model's section records, so the slot
    works — but only for a model whose shell sections apeGmsh can read.
    When it cannot, this refuses here rather than letting the slot
    resolve and fail at read, which is the ADR 0098 A6 defect one level
    deeper. ``available_derived`` is deliberately not consulted for this:
    it answers "what can this data PRODUCE", and its other callers (the
    Gauss composite, the studio namespace) can pass ``thickness=``
    themselves.
    """
    nodal, gauss = _recorded(results, stage_id)
    if quantity in nodal:
        return "nodes"
    if quantity in gauss:
        return "gauss"
    if is_shell_derived(quantity) and not _shell_thickness.is_resolvable(
        getattr(results, "model", None),
    ):
        raise ValueError(
            f"Contour quantity {quantity!r} needs the shell thickness, "
            f"and none could be read from this model's section records "
            f"(apeGmsh reads ElasticMembranePlateSection, LayeredShell "
            f"and LayeredShellFiberSection). Read it directly with an "
            f"explicit thickness instead: "
            f"results.elements.gauss.get(component={quantity!r}, "
            f"thickness=<t>)."
        )
    # Nodes first, matching the recorded lookup above: a quantity
    # derivable at both levels must route the same way in both passes.
    if quantity in available_derived(nodal):
        return "nodes"
    if quantity in available_derived(gauss):
        return "gauss"
    raise ValueError(
        f"Contour quantity {quantity!r} is not recorded in stage "
        f"{stage_id!r}, and cannot be derived from what is "
        f"(nodal: {sorted(nodal)}; gauss: {sorted(gauss)}). "
        f"Use results.inspect.diagnose({quantity!r}) for the routing "
        f"detail."
    )


def resolve_vector_kind(
    quantity: str, results: "Results", stage_id: str,
) -> str:
    """``"vector_glyph"`` or ``"principal_glyph"`` for one §4 `vector`
    token (the ADR collapses both kinds into the one slot).

    A **component suffix decides the kind**, and the recorded sets
    only disambiguate bare prefixes and validate: a tensor suffix is a
    tensor whether or not someone also recorded it nodally, and an
    axis suffix is a vector component either way. Deciding by "which
    composite records it" instead would make a doubly-recorded token
    route one way here and the other way in
    :func:`resolve_contour_topology` (which resolves nodes-first), and
    would hand ``vector_glyph`` a tensor token — it then synthesises
    ``stress_xx_x/_y/_z``, reads nothing, and draws an empty picture
    with no error. Bare prefixes (``"displacement"``, ``"stress"``)
    are legal for both kinds and are resolved against the data.
    """
    nodal, gauss = _recorded(results, stage_id)
    suffix = quantity.rsplit("_", 1)[-1] if "_" in quantity else ""
    if suffix in _TENSOR_SUFFIXES:
        return "principal_glyph"
    if suffix in _AXIS_SUFFIXES:
        return "vector_glyph"
    # A bare prefix: whichever family records it under that prefix.
    if f"{quantity}_xx" in gauss or quantity in gauss:
        return "principal_glyph"
    if f"{quantity}_x" in nodal or quantity in nodal:
        return "vector_glyph"
    raise ValueError(
        f"Vector quantity {quantity!r} is not recorded in stage "
        f"{stage_id!r} — neither as a nodal vector (looked for "
        f"{quantity!r} and {quantity + '_x'!r} in {sorted(nodal)}) nor "
        f"as a Gauss tensor (looked for {quantity!r} and "
        f"{quantity + '_xx'!r} in {sorted(gauss)})."
    )


def slot_spec(
    category: str,
    record: "Slot",
    results: "Results",
    stage_id: str,
    scoped: "ScopedSet",
    view: "ViewerData",
) -> "tuple[str, DiagramSpec]":
    """``(kind_id, DiagramSpec)`` realizing one occupied §4 slot."""
    if category == "contour":
        topology = resolve_contour_topology(record.quantity, results, stage_id)
        kind_id, component = "contour", record.quantity
        family = "nodes" if topology == "nodes" else "elements"
    elif category == "vector":
        kind_id = resolve_vector_kind(record.quantity, results, stage_id)
        component, topology = record.quantity, None
        family = ID_FAMILY[kind_id]
    elif category == "gauss":
        kind_id, component, topology = "gauss_marker", record.quantity, None
        family = ID_FAMILY[kind_id]
    elif category == "line":
        kind_id, component, topology = "line_force", record.component, None
        family = ID_FAMILY[kind_id]
    elif category == "sand":
        kind_id, component, topology = "sand", record.quantity, None
        family = ID_FAMILY[kind_id]
    elif category == "loads":
        kind_id = "loads"
        component = _resolve_load_pattern(record.pattern, view)
        topology, family = None, ID_FAMILY[kind_id]
    elif category == "reactions":
        kind_id, component, topology = "reactions", "reactions", None
        family = ID_FAMILY[kind_id]
    else:  # pragma: no cover - the §4 catalog is closed
        raise ValueError(f"Unknown slot category {category!r}.")

    kdef = kind_def(kind_id)
    assert kdef is not None  # shipped kinds — the registry always has them
    style = kdef.make_default_style(component)
    if topology is not None:
        style = replace(
            style,
            topology=topology,
            averaging=(
                "discrete" if record.averaging == "unaveraged" else "averaged"
            ),
        )
    return kind_id, DiagramSpec(
        kind=kind_id,
        selector=normalize_selector(
            component=component, ids=_scope_ids(scoped, family),
        ),
        style=style,
        stage_id=stage_id,
    )


def _scope_ids(scoped: "ScopedSet", family: str) -> Any:
    """The scoped ids of ``family``, or ``None`` when unrestricted.

    The array is handed over as-is: ``normalize_selector`` already
    converts to an int tuple, and pre-converting here would run that
    O(N) Python-level pass twice on every realize of a scoped view.
    """
    if scoped.is_unscoped or family == "none":
        return None
    return scoped.node_ids if family == "nodes" else scoped.element_ids


def _resolve_load_pattern(
    pattern: Optional[str], view: "ViewerData",
) -> str:
    """The pattern the `loads` slot draws (§4: ``None`` = the default
    the realize layer resolves).

    ``None`` resolves to the one recorded pattern; with several,
    picking silently would draw an arbitrary subset of the model's
    loads, so it refuses and names them.
    """
    try:
        available = list(view.nodes.loads.patterns())
    except Exception:
        available = []
    if pattern is not None:
        if available and pattern not in available:
            raise ValueError(
                f"Load pattern {pattern!r} is not in this model "
                f"(have: {available})."
            )
        return pattern
    if not available:
        raise ValueError(
            "The loads slot needs a load pattern, and this model "
            "carries no applied nodal loads."
        )
    if len(available) > 1:
        raise ValueError(
            f"This model has {len(available)} load patterns "
            f"({available}) — name one on the slot, e.g. "
            f"Loads(pattern={available[0]!r})."
        )
    return available[0]


def _recorded(
    results: "Results", stage_id: str,
) -> "tuple[set[str], set[str]]":
    components = results.inspect.components(stage=stage_id)
    return set(components.get("nodes", ())), set(components.get("gauss", ()))


__all__ = [
    "ID_FAMILY",
    "OCCLUDING_KINDS",
    "resolve_contour_topology",
    "resolve_vector_kind",
    "slot_spec",
]
