"""transforms: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Any, Iterable, TypeAlias

import numpy as np

from ...joint_offset import InsertionPoint, flatten_jnt_offset
from ...transform import Corotational, Linear, PDelta
from ...element.beam_column import (
    ElasticTimoshenkoBeam,
    LadrunoDispBeamColumn,
    LadrunoForceBeamColumn,
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
            LadrunoForceBeamColumn,
            LadrunoIMKBeam,
        ),
    ):
        return spec.transf
    return None


# ---------------------------------------------------------------------------
# orientation-bearing transform fan-out — emit one geomTransf line per
# distinct vecxz observed across the elements that reference the spec.
# A per-element joint-offset rule (InsertionPoint) fans out the same way,
# keyed on (vecxz, -jntOffset).
# ---------------------------------------------------------------------------

#: One ``geomTransf`` line of a fan-out: ``(tag, vecxz, jnt_offset)``.
#: ``vecxz`` is empty in 2-D; ``jnt_offset`` is the flat ``-jntOffset``
#: tail, empty when the line has none.
TransformLine: TypeAlias = (
    "tuple[int, tuple[float, ...], tuple[float, ...]]"
)


def _jnt_rule(t: Primitive) -> InsertionPoint | None:
    """The transform's per-element joint-offset rule, else ``None``."""
    rule = getattr(t, "jnt_offset", None)
    return rule if isinstance(rule, InsertionPoint) else None


def _constant_jnt(t: Primitive) -> tuple[float, ...]:
    """The transform's constant ``-jntOffset`` tail; ``()`` for none,
    for an all-zero pair and for a per-element rule."""
    jnt = getattr(t, "jnt_offset", None)
    if jnt is None or isinstance(jnt, InsertionPoint):
        return ()
    flat = flatten_jnt_offset(jnt, type(t).__name__)
    return flat if any(v != 0.0 for v in flat) else ()


def needs_fanout(t: Primitive) -> bool:
    """True when ``t`` emits one line per distinct per-element value: an
    ``orientation=`` transform (ADR 0010) or one whose ``jnt_offset=`` is
    an :class:`InsertionPoint` rule."""
    return is_orientation_transform(t) or _jnt_rule(t) is not None


def _check_jnt_offset(transf: GeomTransf, ndm: int) -> None:
    """Refuse a joint offset this model cannot emit faithfully."""
    jnt = getattr(transf, "jnt_offset", None)
    if jnt is None:
        return
    name = type(transf).__name__
    if ndm == 3 and isinstance(transf, Corotational):
        raise BridgeError(
            f"geomTransf {name}: jnt_offset= is not supported in 3-D. "
            "OpenSees' CorotCrdTransf3d (stock and Ladruno) zeroes the "
            "offsets (\"rigid joint zones not implemented yet\"), so the "
            "element would solve without them. Use Linear or PDelta, or "
            "model the offsets with explicit nodes and rigidLink beam."
        )
    if isinstance(jnt, InsertionPoint):
        return
    width = len(flatten_jnt_offset(jnt, name)) // 2
    if width != ndm:
        raise BridgeError(
            f"geomTransf {name}: jnt_offset has {width}-component ends but "
            f"the model is {ndm}-D; OpenSees reads -jntOffset as {ndm} "
            f"global components per end."
        )


def _jnt_key(jnt: tuple[float, ...]) -> tuple[int, ...]:
    """Quantize a ``-jntOffset`` tail to cells of :data:`VECXZ_TOL` (in
    model length units) so equal offsets share one ``geomTransf``."""
    inv = 1.0 / VECXZ_TOL
    return tuple(int(round(v * inv)) for v in jnt)


def element_jnt_offset(
    transf: GeomTransf,
    p_i: np.ndarray,
    p_j: np.ndarray,
    vecxz: "tuple[float, float, float] | None",
    ndm: int,
) -> tuple[float, ...]:
    """The flat ``-jntOffset`` tail ``transf`` gives one element.

    A constant pair is the same for every element; an
    :class:`InsertionPoint` rule is evaluated on the element's joints
    (their first ``ndm`` coordinates) and ``vecxz``. All-zero -> ``()``.
    """
    rule = _jnt_rule(transf)
    if rule is None:
        return _constant_jnt(transf)
    pair = rule.offsets(
        [float(c) for c in p_i[:ndm]], [float(c) for c in p_j[:ndm]],
        vecxz if ndm == 3 else None)
    flat = tuple(float(c) for end in pair for c in end)
    return flat if any(v != 0.0 for v in flat) else ()


# ---------------------------------------------------------------------------
# Member ends (ETABS): refuse what the mesh alone cannot place
# ---------------------------------------------------------------------------
#
# ETABS applies a frame object's end offsets at the object's two ends. A
# member meshed into several elements is a straight collinear run of line
# elements, and so is a continuous girder of several members; without each
# element's source curve (#2551) the two cannot be told apart. So an offset
# that is not the same pure transverse shift at both ends (an end length, an
# axial component, unequal ends) refuses on any collinear run of the
# transform's elements. One-element members, chains that turn a corner, and
# an equal transverse shift on every element are unaffected.

#: One element of a transform: ``(eid, node_i, node_j, p_i, p_j, vecxz)``.
_ElementRow: TypeAlias = (
    "tuple[int, int, int, np.ndarray, np.ndarray, tuple[float, ...]]"
)

#: The issue that carries each element's source curve (member ends).
MEMBER_CURVES_ISSUE = "#2551"


def _pad3(v: "Iterable[float]") -> np.ndarray:
    out = np.zeros(3)
    vals = [float(c) for c in v]
    out[:len(vals)] = vals
    return out


def _transverse_only(
    jnt: tuple[float, ...], p_i: np.ndarray, p_j: np.ndarray,
) -> bool:
    """True when ``jnt`` is the same shift at both ends with no component
    along the element: it reads the same on a meshed member and on a
    girder of several members."""
    if not jnt:
        return True
    w = len(jnt) // 2
    d_i, d_j = _pad3(jnt[:w]), _pad3(jnt[w:])
    axis = p_j - p_i
    length = float(np.linalg.norm(axis))
    tol = VECXZ_TOL * max(length, 1.0)
    return (float(np.linalg.norm(d_i - d_j)) <= tol
            and abs(float(d_i @ axis)) / length <= tol)


def _refuse_ambiguous_members(
    transf: GeomTransf,
    rows: "list[_ElementRow]",
    jnt_of: "dict[int, tuple[float, ...]]",
) -> None:
    """Refuse end-sensitive offsets on a collinear run of the transform's
    elements: a meshed member and a continuous girder look alike there."""
    by_node: dict[int, list[int]] = {}
    for k, r in enumerate(rows):
        by_node.setdefault(r[1], []).append(k)
        by_node.setdefault(r[2], []).append(k)
    for node, ks in by_node.items():
        for x, ka in enumerate(ks):
            for kb in ks[x + 1:]:
                a, b = rows[ka], rows[kb]
                u = (a[4] - a[3]) / np.linalg.norm(a[4] - a[3])
                v = (b[4] - b[3]) / np.linalg.norm(b[4] - b[3])
                if float(np.linalg.norm(np.cross(u, v))) > VECXZ_TOL:
                    continue   # a corner ends the member there
                bad = [r for r in (a, b)
                       if not _transverse_only(jnt_of[r[0]], r[3], r[4])]
                if bad:
                    raise BridgeError(
                        f"geomTransf {type(transf).__name__}: jnt_offset with "
                        "end lengths, an axial component or unequal end "
                        f"offsets on elements {a[0]} and {b[0]}, which run "
                        f"collinear through node {node}. Without each "
                        "element's source curve apeGmsh cannot tell a member "
                        "meshed into several elements from a continuous "
                        "girder of several members, so it cannot put the "
                        "offsets at the member's ends, as ETABS does "
                        f"(member ends by curve: {MEMBER_CURVES_ISSUE}). Mesh "
                        "each member as one element and give collinear "
                        "members separate transforms, or use offsets that "
                        "are the same transverse shift at both ends.")


def _transform_rows(
    transf: GeomTransf, elements: "Iterable[Element]", fem: "FEMData", ndm: int,
) -> "list[_ElementRow]":
    """Every element using ``transf``, with its joints and ``vecxz``."""
    oriented = is_orientation_transform(transf)
    spec_vecxz = getattr(transf, "vecxz", None)
    rows: list[_ElementRow] = []
    for ele_spec in elements:
        if _element_transf(ele_spec) is not transf:
            continue
        for eid, node_ids in expand_pg_to_elements(fem, ele_spec.pg):  # type: ignore[attr-defined]
            if len(node_ids) != 2:
                pg = ele_spec.pg  # type: ignore[attr-defined]
                raise BridgeError(
                    f"geomTransf fan-out {type(transf).__name__}: "
                    f"element {eid} in PG {pg!r} has "
                    f"{len(node_ids)} nodes; a per-element vecxz or "
                    "joint offset requires line elements (2 nodes)."
                )
            p_i = _node_coord(fem, int(node_ids[0]))
            p_j = _node_coord(fem, int(node_ids[1]))
            vec: tuple[float, ...]
            if oriented:
                vec = compute_vecxz_for_element(transf, p_i, p_j)  # type: ignore[arg-type]
            elif ndm == 3 and spec_vecxz is not None:
                vec = tuple(float(c) for c in spec_vecxz)
            else:
                vec = ()
            rows.append(
                (int(eid), int(node_ids[0]), int(node_ids[1]), p_i, p_j, vec))
    return rows


def element_offsets_by_id(
    transf: GeomTransf, rows: "list[_ElementRow]", ndm: int,
) -> dict[int, tuple[float, ...]]:
    """Each element's ``-jntOffset`` tail under ``transf``, by element id,
    after refusing what only the members' curves could place."""
    out: dict[int, tuple[float, ...]] = {}
    for eid, _ni, _nj, p_i, p_j, vec in rows:
        try:
            out[eid] = element_jnt_offset(
                transf, p_i, p_j, vec if len(vec) == 3 else None,
                ndm)
        except ValueError as exc:
            raise BridgeError(
                f"geomTransf {type(transf).__name__}: element {eid} "
                f"jnt_offset: {exc}"
            ) from exc
    _refuse_ambiguous_members(transf, rows, out)
    return out


def transform_element_offsets(
    transf: GeomTransf, elements: "Iterable[Element]", fem: "FEMData", ndm: int,
) -> dict[int, tuple[float, ...]]:
    """The ``-jntOffset`` tail each element using ``transf`` was emitted
    with, for a caller that needs it again (a line load over the end
    zones)."""
    return element_offsets_by_id(
        transf, _transform_rows(transf, elements, fem, ndm), ndm)


@dataclass(frozen=True, slots=True)
class TransformFanout:
    """:func:`plan_transform_specs`'s result: every transform's lines.

    ``specs`` holds one ``(transform, lines)`` pair per transform spec, in
    the order the emit walks them. ``lines`` is ``None`` for a transform
    with no fan-out (it emits its own line under its own tag), and
    otherwise the ``(tag, vecxz, jnt_offset)`` line of each distinct
    per-element value, first-seen order: the first reuses the spec's own
    tag, the rest are the planned tags (empty when no element references
    it). ``overrides`` maps ``(id(transform), element id)`` to the
    planned tag of every element whose line is not the spec's own.
    ``fem`` is the FEM snapshot the fan-out was planned over.
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
    """Plan the fan-out's ``geomTransf`` tags (ADR 0010).

    The allocation loop of :func:`emit_transform_specs`, moved out of the
    emit (ADR 0114 D4, amended): the build's tag plan runs it once per
    emit mode with the planner allocator, and
    :func:`emit_transform_specs` writes what it planned.

    For each transform that :func:`needs_fanout` (an ``orientation=``
    transform, or a ``jnt_offset=`` :class:`InsertionPoint` rule), the
    bridge:

      1. Walks every element spec whose ``transf`` IS this transform.
      2. For each element in the spec's PG, computes the per-element
         vecxz (:func:`compute_vecxz_for_element`, else the spec's own
         ``vecxz``; none in 2-D) and ``-jntOffset``
         (:func:`element_jnt_offset`).
      3. Deduplicates across all elements: distinct ``(vecxz, offset)``
         keys produce distinct ``geomTransf`` tags. The first encountered
         key reuses the spec's own allocated tag (so the spec's tag is
         never wasted); subsequent distinct keys get freshly-allocated
         transform tags.
      4. Records one line per distinct key and a per-element override
         map, so the element fan-out can install element-specific
         resolvers for elements whose transform is not the spec's "own"
         tag.

    A transform with no fan-out records ``None``: its single line is the
    spec's own, written by :func:`emit_transform_specs`. Every transform's
    ``jnt_offset`` is checked against ``ndm`` here, before any line is
    written.
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
        _check_jnt_offset(transf, ndm)
        if not needs_fanout(transf):
            constant = _constant_jnt(transf)
            if constant:
                # One line for every element; still refuse it where only
                # the members' curves could place it (#2551).
                rows = _transform_rows(
                    transf, elems_by_transf.get(id(transf), []), fem, ndm)
                _refuse_ambiguous_members(
                    transf, rows, {r[0]: constant for r in rows})
            specs.append((transf, None))
            continue
        oriented = is_orientation_transform(transf)

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
        if ndm == 2 and oriented:
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
        spec_vecxz = getattr(transf, "vecxz", None)
        if ndm == 2:
            _refuse_2d_vecxz(transf)
        elif not oriented and spec_vecxz is None:
            raise BridgeError(
                f"geomTransf {type(transf).__name__}: a jnt_offset= "
                "InsertionPoint needs each element's local axes in 3-D; "
                "give the transform vecxz= or orientation=."
            )

        if oriented:
            # FEM-aware orientations (e.g. AlongBeam) declare a bind_fem
            # hook that materializes any FEM-derived state
            # (reference-curve segments, tangents) into the orientation
            # instance before per-element queries. The fixed-geometry
            # orientations (Cartesian, Cylindrical, Spherical) don't
            # declare it — the hasattr check makes the hook opt-in
            # without polluting the simple cases with a no-op method.
            # is_orientation_transform already narrowed `transf` to a
            # Linear / PDelta / Corotational that has a non-None
            # orientation, but the static GeomTransf base type doesn't
            # advertise the attribute.
            orient = transf.orientation  # type: ignore[attr-defined]
            if hasattr(orient, "bind_fem"):
                orient.bind_fem(fem)

        # Walk every element whose transf IS this transform.
        elems = elems_by_transf.get(id(transf), [])
        if not elems:
            # No elements reference this transform — emit nothing. The
            # spec is effectively a dead declaration; the user can
            # find this with introspection.
            specs.append((transf, ()))
            continue

        # Gather per-element (vecxz, jnt_offset), keyed by element id.
        rows = _transform_rows(transf, elems, fem, ndm)
        jnt_of = element_offsets_by_id(transf, rows, ndm)
        per_element: list[
            tuple[int, tuple[float, ...], tuple[float, ...]]] = [
            (eid, vec, jnt_of[eid]) for eid, _ni, _nj, _pi, _pj, vec in rows]

        # Dedupe by quantized key. First-seen key reuses the spec's own
        # tag; later distinct keys claim fresh transform tags.
        key_to_tag: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        lines: list[TransformLine] = []
        for eid, vec, jnt in per_element:
            k = (_vecxz_key(vec) if vec else (), _jnt_key(jnt))  # type: ignore[arg-type]
            if k not in key_to_tag:
                if not key_to_tag:
                    # First distinct key — reuse the spec's own tag.
                    key_to_tag[k] = own_tag
                else:
                    key_to_tag[k] = tags.allocate("geomTransf")
                lines.append((key_to_tag[k], vec, jnt))

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
    these lines at bracket close — but the fan-out ALLOCATES per-line
    tags, so a re-run cannot reproduce them.  A non-None log captures
    one entry per emitted line, in emit order, for
    :func:`replay_builder_scoped_declarations` to re-drive verbatim.

    For transforms with no fan-out (explicit ``vecxz=``, constant or no
    ``jnt_offset``), one line per spec using the spec's own allocated
    tag — that's the path :class:`Linear` / :class:`PDelta` /
    :class:`Corotational` already handle in their ``_emit``. When
    ``ndm == 2`` and such a transform has neither ``vecxz`` nor
    ``orientation``, the bare 2-D form ``geomTransf <Type> $tag
    [-jntOffset dXi dYi dXj dYj]`` is emitted here instead (the primitive
    ``_emit`` requires a ``vecxz`` and doesn't know ``ndm``).

    For fan-out transforms, one ``geomTransf`` line per distinct
    ``(vecxz, jnt_offset)``, as :func:`plan_transform_specs` planned them.

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
            # No fan-out. Either an explicit vecxz= (3D — one line, the
            # spec's own _emit) or the bare 2D form (``geomTransf <Type>
            # $tag`` with no vecxz vector, which is required in 2D and
            # invalid in 3D). The primitive's _emit can't take this
            # branch because it doesn't know ndm.
            bare_2d = ndm == 2 and type(transf) in _TRANSF_TYPE_TOKEN
            if bare_2d:
                _refuse_2d_vecxz(transf)
                token = _TRANSF_TYPE_TOKEN[type(transf)]
                jnt = _constant_jnt(transf)
                emitter.geomTransf(token, own_tag, **_jnt_kw(jnt))
                if replay_log is not None:
                    replay_log.append(("line", token, own_tag, (), jnt))
            else:
                transf._emit(emitter, own_tag)
                if replay_log is not None:
                    replay_log.append(("spec", transf, own_tag))
            continue

        type_token = _TRANSF_TYPE_TOKEN[type(transf)]
        for tag, vec, jnt in lines:
            emitter.geomTransf(type_token, tag, *vec, **_jnt_kw(jnt))
            if replay_log is not None:
                replay_log.append(("line", type_token, tag, vec, jnt))

    return fanout.overrides


def _jnt_kw(jnt: tuple[float, ...]) -> "dict[str, tuple[float, ...]]":
    """The emitter kwarg for a ``-jntOffset`` tail; ``{}`` for none, so an
    offset-free line is the call it always was."""
    return {"jnt_offset": jnt} if jnt else {}


def _refuse_2d_vecxz(transf: GeomTransf) -> None:
    """An explicit vecxz in 2-D is dropped: Tcl's 2-D ``geomTransf``
    rejects any trailing args (and exits 0), while openseespy silently
    ignores them.  Only a vector along global Z matches what a 2-D model
    can mean (local z = global Z); anything else is a 3-D intent the 2-D
    transform cannot honor."""
    vecxz = getattr(transf, "vecxz", None)
    if vecxz is not None and not _is_global_z(vecxz):
        raise BridgeError(
            f"geomTransf {type(transf).__name__}: "
            f"vecxz={tuple(vecxz)!r} with ndm=2. OpenSees "
            "2-D transforms take no vecxz (local z is always "
            "global Z); drop the vecxz= kwarg."
        )


def _is_global_z(v: "tuple[float, float, float]") -> bool:
    """True when ``v`` is a non-zero vector along global ±Z."""
    x, y, z = (float(c) for c in v)
    return abs(z) > 0.0 and float(np.hypot(x, y)) <= VECXZ_TOL * abs(z)


def _node_coord(fem: "FEMData", node_id: int) -> np.ndarray:
    """Return the 3-D coordinates of ``node_id`` from the FEM snapshot."""
    idx = fem.nodes.index(node_id)
    return np.asarray(fem.nodes.coords[idx], dtype=float)
