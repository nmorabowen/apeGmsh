"""transforms: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Any, Iterable, TypeAlias

import numpy as np

from ...transform import Corotational, Linear, PDelta
from ...element.beam_column import (
    ElasticTimoshenkoBeam,
    LadrunoDispBeamColumn,
    LadrunoIMKBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from ..types import Element, GeomTransf, Primitive
from ..tag_allocator import TagAllocator
from ..._orientation import _div3, _norm3, _sub3, _vec3, resolve_vecxz
from .errors import BridgeError
from .pg_expand import expand_pg_to_elements


if TYPE_CHECKING:
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData
    from ..tag_plan import TagPlan


#: Tolerance for considering two ``vecxz`` triples equal during the
#: orientation-derived fan-out's deduplication step. Two elements
#: whose per-element ``vecxz`` agrees to this tolerance share one
#: ``geomTransf`` line.
VECXZ_TOL: float = 1e-9


# ---------------------------------------------------------------------------
# orientation → per-element vecxz computation
# ---------------------------------------------------------------------------

_AnyTransf = Linear | PDelta | Corotational


_TRANSF_TYPE_TOKEN: dict[type[GeomTransf], str] = {
    Linear:       "Linear",
    PDelta:       "PDelta",
    Corotational: "Corotational",
}


def is_orientation_transform(t: Primitive) -> bool:
    """True if ``t`` is a Linear / PDelta / Corotational with an
    ``orientation=`` parameter set (and hence needs per-element
    vecxz fan-out at build time).

    A transform with explicit ``vecxz=`` does NOT need fan-out: it
    emits one ``geomTransf`` line with the spec's allocated tag. The
    bridge checks this flag to decide whether to drive the fan-out or
    let ``spec._emit`` run once with the spec's tag.
    """
    if not isinstance(t, (Linear, PDelta, Corotational)):
        return False
    return t.orientation is not None and t.vecxz is None


def compute_vecxz_for_element(
    transf: _AnyTransf,
    p_i: np.ndarray,
    p_j: np.ndarray,
) -> tuple[float, float, float]:
    """Return the per-element ``vecxz`` for one element under ``transf``.

    Reads the orientation triad at the element midpoint, computes the
    unit tangent from ``p_i``/``p_j``, and runs the orientation rule
    (ADR 0010) via :func:`resolve_vecxz`. ``transf`` MUST be an
    orientation-bearing transform (caller checks via
    :func:`is_orientation_transform` first).
    """
    if transf.orientation is None:
        raise BridgeError(
            f"compute_vecxz_for_element: transform {transf!r} has no "
            "orientation; caller should have used the explicit vecxz "
            "path."
        )
    # Scalar fixed-order math, not numpy reductions: the vecxz is written
    # into the deck and must be bit-identical on every host (#1279).
    a, b = _vec3(p_i), _vec3(p_j)
    edge = _sub3(b, a)
    norm = _norm3(edge)
    if norm <= 0.0:
        raise BridgeError(
            "compute_vecxz_for_element: element has zero-length edge; "
            f"p_i={a}, p_j={b}."
        )
    tangent = _div3(edge, norm)
    midpoint = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2]))
    e1, e2, e3 = transf.orientation.triad_at(midpoint)
    return resolve_vecxz(tangent, e1, e2, e3, transf.roll_deg)


def _vecxz_key(v: tuple[float, float, float]) -> tuple[int, int, int]:
    """Quantize a ``vecxz`` triple to integer cells of size :data:`VECXZ_TOL`
    so dict keys agree on the dedupe."""
    inv = 1.0 / VECXZ_TOL
    return (int(round(v[0] * inv)), int(round(v[1] * inv)), int(round(v[2] * inv)))


def _element_transf(spec: Element) -> GeomTransf | None:
    """Return ``spec.transf`` for elements that compose a transform; else None.

    Truss / shell / solid elements have no transform — only beam-column
    family elements do. We isinstance-dispatch rather than reading
    ``getattr(spec, "transf", None)`` so we don't accidentally pick up
    an attribute on a future Element family that means something else.
    """
    if isinstance(
        spec,
        (
            elasticBeamColumn,
            forceBeamColumn,
            dispBeamColumn,
            ElasticTimoshenkoBeam,
            LadrunoDispBeamColumn,
            LadrunoIMKBeam,
        ),
    ):
        return spec.transf
    return None


# ---------------------------------------------------------------------------
# orientation-bearing transform fan-out — emit one geomTransf line per
# distinct vecxz observed across the elements that reference the spec.
# ---------------------------------------------------------------------------

#: One ``geomTransf`` line of an orientation fan-out: ``(tag, vecxz)``.
TransformLine: TypeAlias = "tuple[int, tuple[float, float, float]]"


@dataclass(frozen=True, slots=True)
class TransformFanout:
    """:func:`plan_transform_specs`'s result: every transform's lines.

    ``specs`` holds one ``(transform, lines)`` pair per transform spec, in
    the order the emit walks them. ``lines`` is ``None`` for a transform
    with no orientation fan-out (it emits its own line under its own
    tag), and otherwise the ``(tag, vecxz)`` line of each distinct
    ``vecxz``, first-seen order: the first reuses the spec's own tag, the
    rest are the planned tags (empty when no element references it).
    ``overrides`` maps ``(id(transform), element id)`` to the planned tag
    of every element whose ``vecxz`` is not the spec's own. ``fem`` is
    the FEM snapshot the fan-out was planned over.
    """

    specs: "tuple[tuple[GeomTransf, tuple[TransformLine, ...] | None], ...]"
    overrides: "dict[tuple[int, int], int]"
    fem: Any


def plan_transform_specs(
    transforms: Iterable[GeomTransf],
    elements: Iterable[Element],
    fem: "FEMData",
    tags: TagAllocator,
    spec_to_own_tag: dict[int, int],
    ndm: int = 3,
) -> TransformFanout:
    """Plan the orientation fan-out's ``geomTransf`` tags (ADR 0010).

    The allocation loop of :func:`emit_transform_specs`, moved out of the
    emit (ADR 0114 D4, amended): the build's tag plan runs it once per
    emit mode with the planner allocator, and
    :func:`emit_transform_specs` writes what it planned.

    For each orientation-bearing transform, the bridge:

      1. Walks every element spec whose ``transf`` IS this transform.
      2. For each element in the spec's PG, computes the per-element
         vecxz via :func:`compute_vecxz_for_element`.
      3. Deduplicates across all elements: distinct vecxz keys produce
         distinct ``geomTransf`` tags. The first encountered vecxz
         reuses the spec's own allocated tag (so the spec's tag is
         never wasted); subsequent distinct vecxz get freshly-allocated
         transform tags.
      4. Records one line per distinct ``vecxz`` and a per-element
         override map, so the element fan-out can install
         element-specific resolvers for elements whose transform vecxz
         is not the spec's "own" tag.

    A transform with no orientation records ``None``: its single line
    is the spec's own, written by :func:`emit_transform_specs`.
    """
    # Pre-bin elements by transform spec.
    elems_by_transf: dict[int, list[Element]] = {}
    for ele in elements:
        t = _element_transf(ele)
        if t is None:
            continue
        elems_by_transf.setdefault(id(t), []).append(ele)

    specs: "list[tuple[GeomTransf, tuple[TransformLine, ...] | None]]" = []
    overrides: dict[tuple[int, int], int] = {}

    for transf in transforms:
        own_tag = spec_to_own_tag[id(transf)]
        if not is_orientation_transform(transf):
            specs.append((transf, None))
            continue

        # Guard: orientation= is meaningless in OpenSees 2-D.  The
        # 2-D ``geomTransf <Type> $tag`` command takes no trailing
        # vecxz argument; if we proceed into the fan-out with ndm=2,
        # the emitter would silently produce the 3-D form (three
        # extra floats) which OpenSees rejects at parse time.
        # Refuse loudly with a clear message instead of producing
        # invalid output.  Lifting this restriction (supporting
        # in-plane Cylindrical(axis=(0,0,1)) etc.) is tracked in
        # ``architecture/_DEFERRED.md`` § "Cylindrical / Spherical
        # in 2-D models".
        if ndm == 2:
            raise BridgeError(
                f"geomTransf {type(transf).__name__}: orientation= is "
                "not supported with ndm=2 (OpenSees 2-D transforms "
                "take no vecxz argument). Drop the orientation= "
                "kwarg, or construct the bridge with "
                "``apeSees(fem, default_orientation=None)`` so the "
                "default Cartesian is not auto-applied.  See "
                "architecture/_DEFERRED.md § \"Cylindrical / Spherical "
                "in 2-D models\" for the planned lift."
            )

        # FEM-aware orientations (e.g. AlongBeam) declare a bind_fem
        # hook that materializes any FEM-derived state (reference-curve
        # segments, tangents) into the orientation instance before
        # per-element queries. The fixed-geometry orientations
        # (Cartesian, Cylindrical, Spherical) don't declare it — the
        # hasattr check makes the hook opt-in without polluting the
        # simple cases with a no-op method.
        # is_orientation_transform already narrowed `transf` to a
        # Linear / PDelta / Corotational that has a non-None
        # orientation, but the static GeomTransf base type doesn't
        # advertise the attribute.
        orient = transf.orientation  # type: ignore[attr-defined]
        if hasattr(orient, "bind_fem"):
            orient.bind_fem(fem)

        # orientation path: walk every element whose transf IS this
        # transform, compute per-element vecxz, dedupe.
        elems = elems_by_transf.get(id(transf), [])
        if not elems:
            # No elements reference this transform — emit nothing. The
            # spec is effectively a dead declaration; the user can
            # find this with introspection.
            specs.append((transf, ()))
            continue

        # Gather per-element vecxz, keyed by element id.
        per_element_vecxz: list[tuple[int, tuple[float, float, float]]] = []
        for ele_spec in elems:
            for eid, node_ids in expand_pg_to_elements(fem, ele_spec.pg):  # type: ignore[attr-defined]
                if len(node_ids) != 2:
                    pg = ele_spec.pg  # type: ignore[attr-defined]
                    raise BridgeError(
                        f"orientation transform {type(transf).__name__}: "
                        f"element {eid} in PG {pg!r} has "
                        f"{len(node_ids)} nodes; orientation-driven "
                        "vecxz fan-out requires line elements (2 nodes)."
                    )
                p_i = _node_coord(fem, int(node_ids[0]))
                p_j = _node_coord(fem, int(node_ids[1]))
                vec = compute_vecxz_for_element(transf, p_i, p_j)  # type: ignore[arg-type]
                per_element_vecxz.append((eid, vec))

        # Dedupe by quantized key. First-seen vecxz reuses the spec's
        # own tag; later distinct vecxz claim fresh transform tags.
        key_to_tag: dict[tuple[int, int, int], int] = {}
        lines: list[TransformLine] = []
        for eid, vec in per_element_vecxz:
            k = _vecxz_key(vec)
            if k not in key_to_tag:
                if not key_to_tag:
                    # First distinct vecxz — reuse the spec's own tag.
                    key_to_tag[k] = own_tag
                else:
                    key_to_tag[k] = tags.allocate("geomTransf")
                lines.append((key_to_tag[k], vec))

            assigned = key_to_tag[k]
            if assigned != own_tag:
                overrides[(id(transf), eid)] = assigned
        specs.append((transf, tuple(lines)))

    return TransformFanout(specs=tuple(specs), overrides=overrides, fem=fem)


def emit_transform_specs(
    transforms: Iterable[GeomTransf],
    emitter: "Emitter",
    fem: "FEMData",
    tag_plan: "TagPlan",
    spec_to_own_tag: dict[int, int],
    ndm: int = 3,
    replay_log: "list[tuple[Any, ...]] | None" = None,
) -> dict[tuple[int, int], int]:
    """Emit ``geomTransf`` lines for every transform spec.

    The tags come from :func:`plan_transform_specs` (ADR 0114 D4,
    amended): this reads the fan-out ``tag_plan`` holds, and a plan made
    for other transforms raises :class:`TagPlanMiss`.

    ``replay_log`` (ADR 0099 S7): when a stage-activated gated element
    will bracket mid-deck, the staged path must be able to re-declare
    these lines at bracket close — but the orientation fan-out ALLOCATES
    per-vecxz tags, so a re-run cannot reproduce them.  A non-None log
    captures one entry per emitted line, in emit order, for
    :func:`replay_builder_scoped_declarations` to re-drive verbatim.

    For non-orientation transforms (explicit ``vecxz=``), one line
    per spec using the spec's own allocated tag — that's the path
    :class:`Linear` / :class:`PDelta` / :class:`Corotational` already
    handle in their ``_emit``. When ``ndm == 2`` and such a transform
    has neither ``vecxz`` nor ``orientation``, the bare 2-D form
    ``geomTransf <Type> $tag`` is emitted here instead (the primitive
    ``_emit`` requires a ``vecxz`` and doesn't know ``ndm``).

    For orientation-bearing transforms, one ``geomTransf`` line per
    distinct ``vecxz``, as :func:`plan_transform_specs` planned them.

    Returns
    -------
    dict[(id(transf_spec), element_id), int]
        Per-element override tags. Elements not in the dict use the
        spec's own resolver lookup (which yields the spec's own tag).
    """
    fanout = tag_plan.transforms.fanout_for(list(transforms), fem)

    for transf, lines in fanout.specs:
        own_tag = spec_to_own_tag[id(transf)]
        if lines is None:
            # No orientation fan-out. Either an explicit vecxz= (3D —
            # one line, the spec's own _emit) or the bare 2D form
            # (``geomTransf <Type> $tag`` with no vecxz vector, which
            # is required in 2D and invalid in 3D). The primitive's
            # _emit can't take this branch because it doesn't know ndm.
            bare_2d = ndm == 2 and type(transf) in _TRANSF_TYPE_TOKEN
            if bare_2d:
                # An explicit vecxz in 2-D is dropped: Tcl's 2-D
                # ``geomTransf`` rejects any trailing args (and exits
                # 0), while openseespy silently ignores them.  Only a
                # vector along global Z matches what a 2-D model can
                # mean (local z = global Z); anything else is a 3-D
                # intent the 2-D transform cannot honor.
                vecxz = getattr(transf, "vecxz", None)
                if vecxz is not None and not _is_global_z(vecxz):
                    raise BridgeError(
                        f"geomTransf {type(transf).__name__}: "
                        f"vecxz={tuple(vecxz)!r} with ndm=2. OpenSees "
                        "2-D transforms take no vecxz (local z is always "
                        "global Z); drop the vecxz= kwarg."
                    )
                emitter.geomTransf(_TRANSF_TYPE_TOKEN[type(transf)], own_tag)
                if replay_log is not None:
                    replay_log.append(
                        ("line", _TRANSF_TYPE_TOKEN[type(transf)],
                         own_tag, ()),
                    )
            else:
                transf._emit(emitter, own_tag)
                if replay_log is not None:
                    replay_log.append(("spec", transf, own_tag))
            continue

        type_token = _TRANSF_TYPE_TOKEN[type(transf)]
        for tag, vec in lines:
            emitter.geomTransf(type_token, tag, *vec)
            if replay_log is not None:
                replay_log.append(("line", type_token, tag, vec))

    return fanout.overrides


def _is_global_z(v: "tuple[float, float, float]") -> bool:
    """True when ``v`` is a non-zero vector along global ±Z."""
    x, y, z = (float(c) for c in v)
    return abs(z) > 0.0 and float(np.hypot(x, y)) <= VECXZ_TOL * abs(z)


def _node_coord(fem: "FEMData", node_id: int) -> np.ndarray:
    """Return the 3-D coordinates of ``node_id`` from the FEM snapshot."""
    idx = fem.nodes.index(node_id)
    return np.asarray(fem.nodes.coords[idx], dtype=float)
