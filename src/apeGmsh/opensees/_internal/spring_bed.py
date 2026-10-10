"""``ops.spring_bed`` — a grounded zeroLength spring/dashpot bed (ADR 0119 D2).

A builder, not an emit pass: it reads the frozen FEM snapshot once, at
the call, and registers ordinary primitives on the bridge — one shared
``Elastic 1.0`` and ``Viscous 1.0 1.0``, one ``Parallel -factors k c``
per spring and direction, one node-pair ``ZeroLength`` per spring, the
ground ``fix`` and the ``ndf`` statements of the decoupled nodes — so the
emit path, the tag law and the ``ndf`` gates see nothing new.

The nodes come from the session (ADR 0049 / 0119 D1): ``ground`` is a
``g.decouple_node_set(...)`` handle; ``at`` (optional) is a second set on
the same source, typically the 3-dof side nodes created with
``tie_dofs=(1, 2, 3)``. Spring ``i`` joins ``ground`` node ``i`` to the
``at`` node with the same source node, or to the source node itself.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from numpy.typing import NDArray

from apeGmsh._kernel.defs.decoupled import DecoupledNodeSetDef

from ..element.zero_length import ZeroLength, ZeroLengthMatDir
from ..material.uniaxial import ElasticMaterial, Parallel, Viscous
from .types import UniaxialMaterial

if TYPE_CHECKING:
    from ..apesees import apeSees


__all__ = ["SpringBed", "build_spring_bed", "nodal_tributary_area"]


#: Per-spring values: an ``(n, len(dirs))`` array, one row broadcast to
#: every spring, or a callable ``f(xyz, area)`` returning the array.
SpringValues = Union[
    Sequence[float], Sequence[Sequence[float]], NDArray[Any],
    Callable[[NDArray[np.float64], "NDArray[np.float64] | None"], Any],
]

#: Per-spring ``-orient``: a 6-tuple, an ``(n, 6)`` array or ``f(xyz)``.
OrientValues = Union[
    Sequence[float], Sequence[Sequence[float]], NDArray[Any],
    Callable[[NDArray[np.float64]], Any],
]

#: Provenance keys of the two shared unit materials. They are registered
#: *unnamed* (``synthesised=``): no entry in the bridge's name table, so a
#: user material can never be picked up as the unit spring (a user
#: ``name="spring_bed_unit_elastic"`` is just a user material).
_UNIT_ELASTIC = "spring_bed:unit_elastic"
_UNIT_VISCOUS = "spring_bed:unit_viscous"


@dataclass(frozen=True, eq=False)
class SpringBed:
    """What :func:`build_spring_bed` registered, per spring, in source
    order: node tags, the stiffness and dashpot factors (``c`` is
    ``None`` without dashpots), the tributary areas (``None`` without
    ``tributary=``), the orientations and the ``ZeroLength`` specs."""

    name: str | None
    source: tuple[int, ...]
    ground: tuple[int, ...]
    nodes: tuple[int, ...]
    dirs: tuple[int, ...]
    k: NDArray[np.float64]
    c: NDArray[np.float64] | None
    area: NDArray[np.float64] | None
    orient: NDArray[np.float64] | None
    elements: tuple[ZeroLength, ...]

    def __len__(self) -> int:
        return len(self.ground)


def nodal_tributary_area(fem: Any, group: str) -> dict[int, float]:
    """``{node: area}`` over the 2-D elements of a label or physical group.

    Each element's area is shared equally among its corner nodes (tri3 /
    tri6: 3 corners; quad4 / quad8 / quad9: 4, area of the two triangles
    of the (0, 2) diagonal). Midside nodes of higher-order elements get
    none. Raises ``ValueError`` when ``group`` holds no 2-D elements.
    """
    conn: Any = None
    for groups in (fem.nodes.labels, fem.nodes.physical):
        if groups is not None and group in groups:
            try:
                conn = groups.connectivity(group, dim=2)
            except (KeyError, ValueError):
                conn = None
            if conn is not None:
                break
    if conn is None or len(conn) == 0:
        raise ValueError(
            f"spring_bed: tributary={group!r} names no label or physical "
            f"group with 2-D elements on the FEM snapshot."
        )
    xyz = {int(n): np.asarray(c, dtype=float)
           for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    area: dict[int, float] = {}
    for row in np.asarray(conn):
        nodes = [int(n) for n in row if int(n) >= 0]
        n_corner = {3: 3, 6: 3, 4: 4, 8: 4, 9: 4}.get(len(nodes))
        if n_corner is None:
            raise ValueError(
                f"spring_bed: tributary={group!r} holds a {len(nodes)}-node "
                f"element; only tri3/tri6/quad4/quad8/quad9 carry an area."
            )
        corners = nodes[:n_corner]
        p = [xyz[n] for n in corners]
        a = 0.5 * float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
        if n_corner == 4:
            a += 0.5 * float(np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[0])))
        for n in corners:
            area[n] = area.get(n, 0.0) + a / n_corner
    return area


def _per_spring(
    what: str, value: Any, n: int, width: int,
    xyz: NDArray[np.float64], area: NDArray[np.float64] | None,
    *, call_with_area: bool = True,
) -> NDArray[np.float64]:
    if callable(value):
        value = value(xyz.copy(), None if area is None else area.copy()) \
            if call_with_area else value(xyz.copy())
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and arr.shape == (width,):
        arr = np.broadcast_to(arr, (n, width)).copy()
    if arr.shape != (n, width):
        raise ValueError(
            f"spring_bed: {what} must be {width} values, an ({n}, {width}) "
            f"array or a callable returning one; got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"spring_bed: {what} holds a non-finite value.")
    return arr


def _set_def(ref: object, role: str) -> DecoupledNodeSetDef:
    if not isinstance(ref, DecoupledNodeSetDef):
        raise TypeError(
            f"spring_bed: {role}= must be a g.decouple_node_set(...) handle, "
            f"got {type(ref).__name__}."
        )
    if ref.tags is None or ref.source_ids is None:
        raise ValueError(
            f"spring_bed: the {role}= node set {ref.label or ref.source!r} "
            f"has no resolved tags — call g.mesh.queries.get_fem_data(...) "
            f"before building the bridge."
        )
    return ref


def _unit(ops: "apeSees", key: str, make: Callable[[], UniaxialMaterial]) -> UniaxialMaterial:
    """The bridge's one shared unit material under ``key``, made on first use."""
    units = ops._spring_bed_units
    if key not in units:
        units[key] = ops._register(make(), synthesised=key)
    return units[key]


def build_spring_bed(
    ops: "apeSees",
    ground: DecoupledNodeSetDef,
    *,
    at: DecoupledNodeSetDef | None = None,
    k: SpringValues,
    c: SpringValues | None = None,
    orient: OrientValues | None = None,
    tributary: str | None = None,
    dirs: Sequence[int] = (1, 2, 3),
    do_rayleigh: bool = False,
    fix: bool = True,
    ndf: int = 3,
    name: str | None = None,
) -> SpringBed:
    """Register the springs of one bed on ``ops``; see ``apeSees.spring_bed``."""
    gset = _set_def(ground, "ground")
    dirs_t = tuple(int(d) for d in dirs)
    if isinstance(ndf, bool) or not isinstance(ndf, int) or not 1 <= ndf <= 6:
        raise ValueError(f"spring_bed: ndf must be an int in 1..6, got {ndf!r}.")
    if (not dirs_t or len(set(dirs_t)) != len(dirs_t)
            or any(d < 1 or d > ndf for d in dirs_t)):
        raise ValueError(
            f"spring_bed: dirs must be distinct DOFs in 1..ndf={ndf}; "
            f"got {tuple(dirs)!r}."
        )
    assert gset.source_ids is not None and gset.tags is not None
    source = tuple(gset.source_ids)
    ground_tags = tuple(gset.tags)
    if at is None:
        j_tags = source
    else:
        aset = _set_def(at, "at")
        pairs = aset.pairs()
        missing = [s for s in source if s not in pairs]
        if missing:
            raise ValueError(
                f"spring_bed: the at= set {aset.label or aset.source!r} has "
                f"no node for source nodes {missing[:5]} of the ground set; "
                f"declare both sets on the same source."
            )
        j_tags = tuple(pairs[s] for s in source)
        # The springs act on the ``at`` node's translations; unless they
        # are tied to the structure the bed is inert (the analysis
        # converges, the ground reaction is 0.0). ``orient`` rotates the
        # spring axes, so every translation is coupled then.
        need = set(dirs_t) | ({1, 2, 3} if orient is not None else set())
        tied = set(aset.tie_dofs or ())
        if not tied or not need <= tied:
            have = sorted(tied)
            raise ValueError(
                f"spring_bed: the at= set {aset.label or aset.source!r} is "
                f"not tied to the structure on DOFs {sorted(need - tied)} "
                f"(declared tie_dofs={have if have else None}); the bed "
                f"would carry no load. Declare it with "
                f"g.decouple_node_set(..., tie_dofs={tuple(sorted(need))}) "
                f"or a superset."
            )

    fem = ops.fem
    row_of = {int(t): i for i, t in enumerate(fem.nodes.ids)}
    coords = np.asarray(fem.nodes.coords, dtype=float)
    absent = [t for t in source + ground_tags + j_tags if t not in row_of]
    if absent:
        raise ValueError(
            f"spring_bed: nodes {absent[:5]} are not on this bridge's FEM "
            f"snapshot — build the bridge from the snapshot the node sets "
            f"were resolved into."
        )
    xyz = coords[[row_of[t] for t in source]]
    n, width = len(source), len(dirs_t)

    area: NDArray[np.float64] | None = None
    if tributary is not None:
        trib = nodal_tributary_area(fem, tributary)
        no_area = [t for t in source if t not in trib]
        if no_area:
            raise ValueError(
                f"spring_bed: source nodes {no_area[:5]} carry no area in "
                f"tributary={tributary!r}."
            )
        area = np.array([trib[t] for t in source], dtype=float)

    k_arr = _per_spring("k", k, n, width, xyz, area)
    c_arr = None if c is None else _per_spring("c", c, n, width, xyz, area)
    for what, arr in (("k", k_arr), ("c", c_arr)):
        if arr is not None and np.any(arr < 0.0):
            raise ValueError(f"spring_bed: {what} must be >= 0.")
    o_arr = None if orient is None else _per_spring(
        "orient", orient, n, 6, xyz, area, call_with_area=False)

    unit_e = _unit(ops, _UNIT_ELASTIC, lambda: ElasticMaterial(E=1.0))
    unit_v = None if c_arr is None else _unit(
        ops, _UNIT_VISCOUS, lambda: Viscous(C=1.0, alpha=1.0))

    elements: list[ZeroLength] = []
    for i in range(n):
        mat_dirs: list[ZeroLengthMatDir] = []
        for jd, dof in enumerate(dirs_t):
            if unit_v is None or c_arr is None:
                par = Parallel(materials=(unit_e,), factors=(float(k_arr[i, jd]),))
            else:
                par = Parallel(
                    materials=(unit_e, unit_v),
                    factors=(float(k_arr[i, jd]), float(c_arr[i, jd])),
                )
            mat_dirs.append(ZeroLengthMatDir(material=ops._register(par), dof=dof))
        o6 = None
        if o_arr is not None:
            r = o_arr[i]
            o6 = (float(r[0]), float(r[1]), float(r[2]),
                  float(r[3]), float(r[4]), float(r[5]))
        spec = ZeroLength(
            nodes=(ground_tags[i], j_tags[i]), mat_dirs=tuple(mat_dirs),
            orient=o6, do_rayleigh=bool(do_rayleigh),
        )
        elements.append(ops._register(spec))

    if fix:
        ops.fix(nodes=ground_tags, dofs=(1,) * ndf)
    for tag in ground_tags:
        ops.ndf(tag, ndf=ndf)
    if at is not None:
        for tag in j_tags:
            ops.ndf(tag, ndf=ndf)

    return SpringBed(
        name=name, source=source, ground=ground_tags, nodes=j_tags,
        dirs=dirs_t, k=k_arr, c=c_arr, area=area, orient=o_arr,
        elements=tuple(elements),
    )
