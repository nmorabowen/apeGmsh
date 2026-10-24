"""
Rigid joint offsets for ``geomTransf`` (``-jntOffset``) and the ETABS
cardinal-point / insertion-point helper.

OpenSees command shape (stock and Ladruno fork, Tcl and Python)::

    geomTransf <Linear|PDelta|Corotational> tag vecxzX vecxzY vecxzZ \\
        -jntOffset dXi dYi dZi dXj dYj dZj           # 3-D
    geomTransf <Linear|PDelta|Corotational> tag \\
        -jntOffset dXi dYi dXj dYj                   # 2-D

``dI`` is the vector, in **global** coordinates, from node I to the end
of the flexible element at I (``dJ`` likewise at J). The element's
local axes and length are those of the offset ends, and the offset is a
rigid arm: no mass, no load, no flexibility. ``CorotCrdTransf3d``
prints "rigid joint zones not implemented yet" and zeroes the offsets,
so the bridge refuses a 3-D ``Corotational`` with a joint offset
instead of emitting a line that solves without them.

In the bridge a joint offset is a ``jnt_offset=`` of the transform
(:class:`~apeGmsh.opensees.transform.Linear` / ``PDelta`` /
``Corotational``): either one constant pair
``((dXi, dYi, dZi), (dXj, dYj, dZj))`` (2-D: 2-tuples) for every element
that uses the transform, or an :class:`InsertionPoint` rule the bridge
evaluates per element and fans out as one ``geomTransf`` per distinct
``(vecxz, offsets)``, as it already does for ``orientation=`` (ADR 0010).

The ETABS insertion point
=========================

:func:`insertion_point_offsets` (and the :class:`InsertionPoint` rule)
turn an ETABS / SAP2000 frame insertion assignment into the
``jnt_offset`` pair. Conventions (CSI Analysis Reference Manual,
"Insertion Points", Figure 32, and "End Offsets"):

* **ETABS local axes.** ``1`` runs from joint I to joint J; ``2`` is the
  section's depth direction ("top" is ``+2``); ``3 = 1 x 2``. Viewed
  with ``+1`` into the page, ``+2`` points up and ``+3`` to the right.
  The nominal axes come from the joints (before any offset), and joint
  offsets given in the local system use them, as ETABS does.
* **Mapping onto OpenSees.** OpenSees builds its local axes from
  ``vecxz`` (``y = vecxz x x``, ``z = x x y``, normalised). ``axis_2``
  names the OpenSees local axis that carries ETABS ``2``: ``"y"``
  (``vecxz`` is the ETABS ``3`` direction; ``Iz`` is ``I33``) or ``"z"``
  (``vecxz`` is the ETABS ``2`` direction, e.g. ``vecxz=(0, 0, 1)`` on a
  horizontal beam; ``Iy`` is ``I33``). ETABS ``3`` is then ``1 x 2``.
* **Cardinal points.** 1 bottom-left, 2 bottom-center, 3 bottom-right,
  4 middle-left, 5 middle-center, 6 middle-right, 7 top-left,
  8 top-center, 9 top-right, 10 centroid, 11 shear center. Bottom/top
  are the ``-2``/``+2`` faces of the section's bounding box, left/right
  the ``-3``/``+3`` faces, middle and center the box's mid-lines. Point
  11 is taken as the centroid, which holds only for a doubly symmetric
  bounding box; an asymmetric one refuses it.
* **Sign.** The cardinal point sits on the joint, so the centroid (the
  OpenSees element axis) is displaced from the joint by minus the
  cardinal point's position in the section: cardinal 8 (top-center) on
  a section of depth ``d`` gives ``-d/2`` along ``+2``; the beam hangs
  below the joint.
* **Joint offsets** (``offset_i``/``offset_j``) run from the joint to the
  cardinal point, in global ``(X, Y, Z)`` or in local ``(1, 2, 3)``
  components. The total is ``joint offset + (centroid - cardinal)``.
* **End-length offsets** (``end_offset_i``/``end_offset_j``, the ETABS
  rigid zones) run along the element axis between the insertion ends,
  inward from each end. Only ``rigid_zone_factor`` times each length is
  rigid; the rest stays part of the flexible element, as in ETABS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

from ._orientation import _cross3, _norm3, _sub3

__all__ = [
    "JntOffset",
    "InsertionPoint",
    "insertion_point_offsets",
    "flatten_jnt_offset",
    "pair_of_flat",
]


_Vec2 = tuple[float, float]
_Vec3 = tuple[float, float, float]

#: A ``-jntOffset`` pair ``(dI, dJ)`` in global coordinates: 3-tuples in
#: a 3-D model, 2-tuples in a 2-D one.
JntOffset: TypeAlias = "tuple[_Vec3, _Vec3] | tuple[_Vec2, _Vec2]"

#: Tolerance (relative to the section size) for a bounding box to count
#: as symmetric about the centroid (cardinal point 11).
_SYM_RTOL = 1e-9


def flatten_jnt_offset(jnt: object, where: str) -> tuple[float, ...]:
    """Validate a constant ``jnt_offset`` pair and return it flat.

    Returns ``(dXi, dYi[, dZi], dXj, dYj[, dZj])``: the ``-jntOffset``
    tokens in OpenSees order. Raises :class:`ValueError` unless ``jnt``
    is two sequences of 2 or 3 finite numbers each, of equal length.
    """
    if not isinstance(jnt, (tuple, list)) or len(jnt) != 2:
        raise ValueError(
            f"{where}: jnt_offset must be a pair ((dXi, dYi[, dZi]), "
            f"(dXj, dYj[, dZj])) in global coordinates, got {jnt!r}."
        )
    out: list[float] = []
    widths: list[int] = []
    for end in jnt:
        if not isinstance(end, (tuple, list)) or len(end) not in (2, 3):
            raise ValueError(
                f"{where}: each jnt_offset end must be a 2-tuple (2-D) or a "
                f"3-tuple (3-D) of global components, got {end!r}."
            )
        widths.append(len(end))
        for c in end:
            if isinstance(c, bool) or not isinstance(c, (int, float)):
                raise ValueError(
                    f"{where}: jnt_offset component {c!r} is not a number."
                )
            v = float(c)
            if not math.isfinite(v):
                raise ValueError(
                    f"{where}: jnt_offset component {v!r} is not finite."
                )
            out.append(v)
    if widths[0] != widths[1]:
        raise ValueError(
            f"{where}: jnt_offset ends have {widths[0]} and {widths[1]} "
            "components; both ends are 2-D or both 3-D."
        )
    return tuple(out)


def pair_of_flat(flat: Sequence[float]) -> JntOffset:
    """The inverse of :func:`flatten_jnt_offset`: 4 or 6 floats -> pair."""
    n = len(flat) // 2
    if n == 3:
        return ((flat[0], flat[1], flat[2]), (flat[3], flat[4], flat[5]))
    return ((flat[0], flat[1]), (flat[2], flat[3]))


def _point(p: Sequence[float], where: str) -> _Vec3:
    if len(p) not in (2, 3):
        raise ValueError(f"{where}: a node coordinate has 2 or 3 components, got {p!r}.")
    return (float(p[0]), float(p[1]), float(p[2]) if len(p) == 3 else 0.0)


def _unit(v: _Vec3, what: str) -> _Vec3:
    n = _norm3(v)
    if n <= 0.0:
        raise ValueError(f"insertion_point_offsets: {what} has zero length.")
    return (v[0] / n, v[1] / n, v[2] / n)


def _axpy(*terms: tuple[float, _Vec3]) -> _Vec3:
    """``sum(k * v)`` in a fixed operation order (deck bits are host-independent)."""
    x = y = z = 0.0
    for k, v in terms:
        x += k * v[0]
        y += k * v[1]
        z += k * v[2]
    return (x, y, z)


def _cardinal_coords(
    point: int,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    """``(c2, c3)``: cardinal ``point``'s position from the centroid along
    ETABS local 2 and 3, on the bounding box ``(min2, max2, min3, max3)``."""
    min2, max2, min3, max3 = bbox
    if point == 10:
        return 0.0, 0.0
    if point == 11:
        size = max(max2 - min2, max3 - min3)
        if (abs(max2 + min2) > _SYM_RTOL * size
                or abs(max3 + min3) > _SYM_RTOL * size):
            raise ValueError(
                "insertion_point_offsets: cardinal point 11 (shear center) "
                "is taken as the centroid, which holds only for a doubly "
                f"symmetric section; the bounding box {bbox!r} is not "
                "symmetric about the centroid. Pass the shear-center "
                "position as offset_i/offset_j from cardinal point 10."
            )
        return 0.0, 0.0
    row, col = divmod(point - 1, 3)
    c2 = (min2, 0.5 * (min2 + max2), max2)[row]
    c3 = (min3, 0.5 * (min3 + max3), max3)[col]
    return c2, c3


def _section_bbox(
    depth: float | None,
    width: float | None,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if bbox is not None:
        if depth is not None or width is not None:
            raise ValueError(
                "insertion_point_offsets: give depth=/width= or bbox=, not both."
            )
        if len(bbox) != 4:
            raise ValueError(
                "insertion_point_offsets: bbox= is (min2, max2, min3, max3) "
                f"from the centroid, got {bbox!r}."
            )
        min2, max2, min3, max3 = (float(v) for v in bbox)
        if not (min2 <= 0.0 <= max2 and min3 <= 0.0 <= max3):
            raise ValueError(
                "insertion_point_offsets: bbox= is measured from the "
                f"centroid, so it must contain 0 on both axes; got {bbox!r}."
            )
        return min2, max2, min3, max3
    if depth is None:
        raise ValueError(
            "insertion_point_offsets: give the section depth= (along ETABS "
            "local 2) and width= (along 3), or its bbox=."
        )
    d = float(depth)
    b = 0.0 if width is None else float(width)
    if d < 0.0 or b < 0.0:
        raise ValueError(
            f"insertion_point_offsets: depth={depth!r} and width={width!r} "
            "must be non-negative."
        )
    return -0.5 * d, 0.5 * d, -0.5 * b, 0.5 * b


def insertion_point_offsets(
    *,
    p_i: Sequence[float],
    p_j: Sequence[float],
    cardinal_point: int = 10,
    depth: float | None = None,
    width: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    vecxz: tuple[float, float, float] | None = None,
    axis_2: Literal["y", "z"] = "y",
    mirror_2: bool = False,
    mirror_3: bool = False,
    offset_i: Sequence[float] | None = None,
    offset_j: Sequence[float] | None = None,
    offset_system: Literal["global", "local"] = "global",
    end_offset_i: float = 0.0,
    end_offset_j: float = 0.0,
    rigid_zone_factor: float = 1.0,
) -> JntOffset:
    """The ``jnt_offset`` pair of an ETABS insertion-point assignment.

    See the module docstring for every sign convention.

    Parameters
    ----------
    p_i, p_j
        Joint coordinates of ends I and J: 3 components in a 3-D model,
        2 in a 2-D one (the result has the same width).
    cardinal_point
        ETABS cardinal point, 1-11 (default 10, the centroid).
    depth, width
        Section depth along ETABS local 2 and width along 3, for a
        section whose bounding box is centred on the centroid.
    bbox
        Instead of ``depth``/``width``: the bounding box
        ``(min2, max2, min3, max3)`` measured from the centroid.
    vecxz
        The element's ``geomTransf`` vector (3-D). Needed whenever a
        local direction is: a cardinal point off the centroid, local
        joint offsets. 2-D ignores it (local y is local x turned +90 deg).
    axis_2
        OpenSees local axis carrying ETABS local 2: ``"y"`` or ``"z"``.
        2-D accepts only ``"y"`` (``z`` is out of plane).
    mirror_2, mirror_3
        ETABS ``Mirror2``/``Mirror3``: the section is mirrored about its
        local 2 (resp. 3) axis before the cardinal point is located, so
        an asymmetric bounding box flips (``min3, max3 -> -max3, -min3``).
    offset_i, offset_j
        ETABS joint offsets, from each joint to its cardinal point.
    offset_system
        ``"global"`` (``(X, Y[, Z])``) or ``"local"`` (``(1, 2[, 3])`` on
        the nominal axes) for ``offset_i``/``offset_j``.
    end_offset_i, end_offset_j
        ETABS end-length offsets (rigid zones) along the axis.
    rigid_zone_factor
        ETABS rigid-zone factor in ``[0, 1]``; the rigid length at each
        end is ``rigid_zone_factor * end_offset``.

    Returns
    -------
    ``((dXi, dYi, dZi), (dXj, dYj, dZj))`` in 3-D, 2-tuples in 2-D.
    """
    if len(p_i) != len(p_j) or len(p_i) not in (2, 3):
        raise ValueError(
            "insertion_point_offsets: p_i and p_j are both 2-D or both 3-D "
            f"coordinates, got {p_i!r} and {p_j!r}."
        )
    ndm = len(p_i)
    if isinstance(cardinal_point, bool) or cardinal_point not in range(1, 12):
        raise ValueError(
            f"insertion_point_offsets: cardinal_point must be 1-11, got "
            f"{cardinal_point!r}."
        )
    if axis_2 not in ("y", "z"):
        raise ValueError(
            f"insertion_point_offsets: axis_2 must be 'y' or 'z', got {axis_2!r}."
        )
    if offset_system not in ("global", "local"):
        raise ValueError(
            "insertion_point_offsets: offset_system must be 'global' or "
            f"'local', got {offset_system!r}."
        )
    if not 0.0 <= float(rigid_zone_factor) <= 1.0:
        raise ValueError(
            "insertion_point_offsets: rigid_zone_factor must be in [0, 1], "
            f"got {rigid_zone_factor!r}."
        )
    if float(end_offset_i) < 0.0 or float(end_offset_j) < 0.0:
        raise ValueError(
            "insertion_point_offsets: end offsets are lengths (>= 0), got "
            f"{end_offset_i!r} and {end_offset_j!r}."
        )
    if ndm == 2 and axis_2 != "y":
        raise ValueError(
            "insertion_point_offsets: a 2-D frame's section depth lies along "
            "local y (local z is out of plane); axis_2 must be 'y'."
        )
    pi = _point(p_i, "insertion_point_offsets")
    pj = _point(p_j, "insertion_point_offsets")
    e1 = _unit(_sub3(pj, pi), "the element (p_j - p_i)")
    min2, max2, min3, max3 = _section_bbox(depth, width, bbox)
    if mirror_2:  # mirrored about the 2 axis: the 3 extent flips
        min3, max3 = -max3, -min3
    if mirror_3:  # mirrored about the 3 axis: the 2 extent flips
        min2, max2 = -max2, -min2
    c2, c3 = _cardinal_coords(int(cardinal_point), (min2, max2, min3, max3))

    ends: list[_Vec3] = []
    for name, off in (("offset_i", offset_i), ("offset_j", offset_j)):
        if off is None:
            ends.append((0.0, 0.0, 0.0))
            continue
        if len(off) != ndm:
            raise ValueError(
                f"insertion_point_offsets: {name} has {len(off)} components; "
                f"a {ndm}-D model takes {ndm}."
            )
        ends.append(_point(off, "insertion_point_offsets"))

    need_local = (c2 != 0.0 or c3 != 0.0
                  or (offset_system == "local"
                      and any(v != (0.0, 0.0, 0.0) for v in ends)))
    e2: _Vec3 = (0.0, 0.0, 0.0)
    e3: _Vec3 = (0.0, 0.0, 0.0)
    if need_local:
        if ndm == 2:
            e2 = (-e1[1], e1[0], 0.0)
            e3 = (0.0, 0.0, 1.0)
            if c3 != 0.0:
                raise ValueError(
                    f"insertion_point_offsets: cardinal point {cardinal_point} "
                    "sits off the section's 2-axis (left/right, along local "
                    "3), which is out of plane in a 2-D model; use 2, 5, 8, "
                    "10 or 11."
                )
        else:
            if vecxz is None:
                raise ValueError(
                    "insertion_point_offsets: a 3-D cardinal point off the "
                    "centroid, or a local joint offset, needs the element's "
                    "local axes: pass vecxz= (the geomTransf vector)."
                )
            v = (float(vecxz[0]), float(vecxz[1]), float(vecxz[2]))
            y = _cross3(v, e1)
            if _norm3(y) <= 1e-12 * _norm3(v):
                raise ValueError(
                    f"insertion_point_offsets: vecxz={tuple(vecxz)!r} is "
                    "parallel to the element axis; it cannot fix the local "
                    "axes."
                )
            ey = _unit(y, "vecxz x axis")
            ez = _cross3(e1, ey)
            e2 = ey if axis_2 == "y" else ez
            e3 = _cross3(e1, e2)
        if offset_system == "local":
            ends = [_axpy((o[0], e1), (o[1], e2), (o[2], e3)) for o in ends]
        if ndm == 2 and offset_system == "local" and any(o[2] != 0.0 for o in ends):
            raise ValueError("insertion_point_offsets: 2-D local offsets are (1, 2).")

    # The cardinal point sits on the joint: the centroid is at minus its
    # section position (cardinal 8 on depth d: -d/2 along +2).
    shift = _axpy((-c2, e2), (-c3, e3))
    d_i = _axpy((1.0, ends[0]), (1.0, shift))
    d_j = _axpy((1.0, ends[1]), (1.0, shift))

    rz_i = float(rigid_zone_factor) * float(end_offset_i)
    rz_j = float(rigid_zone_factor) * float(end_offset_j)
    if float(end_offset_i) != 0.0 or float(end_offset_j) != 0.0:
        a = _axpy((1.0, pi), (1.0, d_i))
        b = _axpy((1.0, pj), (1.0, d_j))
        clear = _norm3(_sub3(b, a))
        if float(end_offset_i) + float(end_offset_j) >= clear:
            raise ValueError(
                f"insertion_point_offsets: end offsets {end_offset_i!r} + "
                f"{end_offset_j!r} reach past the element's length "
                f"{clear!r} between its insertion points."
            )
        axis = _unit(_sub3(b, a), "the offset element")
        d_i = _axpy((1.0, d_i), (rz_i, axis))
        d_j = _axpy((1.0, d_j), (-rz_j, axis))

    if ndm == 2:
        if d_i[2] != 0.0 or d_j[2] != 0.0:  # pragma: no cover - guarded above
            raise ValueError("insertion_point_offsets: a 2-D offset left the plane.")
        return ((d_i[0], d_i[1]), (d_j[0], d_j[1]))
    return (d_i, d_j)


@dataclass(frozen=True, kw_only=True, slots=True)
class InsertionPoint:
    """A per-element ``jnt_offset`` rule: an ETABS insertion point.

    Pass it as ``jnt_offset=`` of a ``geomTransf``. At build the bridge
    evaluates :func:`insertion_point_offsets` on every element that uses
    the transform, with that element's joints and ``vecxz``, and emits
    one ``geomTransf`` per distinct ``(vecxz, offsets)`` (the ADR 0010
    fan-out). Beams of one section running in different directions can
    share one rule: their left/right cardinal points and end-length
    offsets resolve per element. Fields are those of
    :func:`insertion_point_offsets` without the per-element inputs.
    """

    cardinal_point: int = 10
    depth: float | None = None
    width: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    axis_2: Literal["y", "z"] = "y"
    mirror_2: bool = False
    mirror_3: bool = False
    offset_i: tuple[float, ...] | None = None
    offset_j: tuple[float, ...] | None = None
    offset_system: Literal["global", "local"] = "global"
    end_offset_i: float = 0.0
    end_offset_j: float = 0.0
    rigid_zone_factor: float = 1.0

    def __post_init__(self) -> None:
        # Refuse a bad section-level field at construction, not at build.
        if isinstance(self.cardinal_point, bool) or self.cardinal_point not in range(1, 12):
            raise ValueError(
                f"InsertionPoint: cardinal_point must be 1-11, got "
                f"{self.cardinal_point!r}."
            )
        _cardinal_coords(
            self.cardinal_point, _section_bbox(self.depth, self.width, self.bbox))

    def offsets(
        self,
        p_i: Sequence[float],
        p_j: Sequence[float],
        vecxz: tuple[float, float, float] | None,
    ) -> JntOffset:
        """The ``jnt_offset`` pair for one element (joints, ``vecxz``)."""
        return insertion_point_offsets(
            p_i=p_i, p_j=p_j, vecxz=vecxz,
            cardinal_point=self.cardinal_point, depth=self.depth,
            width=self.width, bbox=self.bbox, axis_2=self.axis_2,
            mirror_2=self.mirror_2, mirror_3=self.mirror_3,
            offset_i=self.offset_i, offset_j=self.offset_j,
            offset_system=self.offset_system,
            end_offset_i=self.end_offset_i, end_offset_j=self.end_offset_j,
            rigid_zone_factor=self.rigid_zone_factor,
        )

