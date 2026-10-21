"""Site-response helpers on the ``apeSees`` bridge.

:func:`lysmer_base` builds the compliant (Lysmer–Kuhlemeyer) base of a
soil model: one viscous dashpot per base node, from a fixed ground node,
with the tributary coefficient ``C_i = rho Vs A_i`` on each horizontal
DOF and either ``rho Vp A_i`` or a fixity on the vertical one. The
returned :class:`LysmerBase` then adds the matching input to a load
pattern: the force ``C_i v(t)`` per node for an outcrop velocity, which
is ``2 rho Vs A_i v_inc(t)`` for the incident wave (Joyner and Chen,
1975). Everything it registers is an ordinary bridge primitive, so it
emits, archives and stages like any other declaration.

The ground nodes come from the session, one per base node:
``g.decouple_node_set("Base", label="base_ground")`` before meshing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from apeGmsh._kernel.defs.decoupled import DecoupledNodeSetDef

from .element.zero_length import ZeroLength, ZeroLengthMatDir

if TYPE_CHECKING:
    from .apesees import apeSees
    from .pattern.pattern import Plain


__all__ = ["LysmerBase", "lysmer_base", "base_tributary_area"]


@dataclass(frozen=True, eq=False)
class LysmerBase:
    """What :func:`lysmer_base` registered, per base node, in ascending
    node order: the base and ground node tags, the tributary areas and
    the dashpot coefficients (``c_p`` is ``None`` when the vertical DOF
    is fixed), the DOF numbering and the ``ZeroLength`` specs."""

    nodes: tuple[int, ...]
    ground: tuple[int, ...]
    area: tuple[float, ...]
    c_s: tuple[float, ...]
    c_p: tuple[float, ...] | None
    horizontal: tuple[int, ...]
    vertical: int
    ndf: int
    elements: tuple[ZeroLength, ...]

    def add_input(
        self,
        pattern: "Plain",
        *,
        direction: int = 1,
        motion: Literal["outcrop", "incident"] = "outcrop",
    ) -> None:
        """Load every base node of ``pattern`` with the base-input force.

        ``pattern`` is a ``Plain`` pattern (``ops.pattern.Plain`` or a
        stage's ``s.pattern``) whose time series is the input *velocity*
        ``v(t)``. Node ``i`` gets ``C_i`` along ``direction`` for an
        ``"outcrop"`` velocity (twice the incident wave, as a rock
        outcrop record is) and ``2 C_i`` for an ``"incident"`` one, with
        ``C_i = rho Vs A_i``: the series times this factor is the force
        ``2 rho Vs A_i v_inc(t)``.
        """
        if direction not in self.horizontal:
            raise ValueError(
                f"LysmerBase.add_input: direction must be a horizontal DOF "
                f"{self.horizontal}, got {direction!r}."
            )
        if motion not in ("outcrop", "incident"):
            raise ValueError(
                f"LysmerBase.add_input: motion must be 'outcrop' or "
                f"'incident', got {motion!r}."
            )
        scale = 1.0 if motion == "outcrop" else 2.0
        for node, c in zip(self.nodes, self.c_s):
            forces = [0.0] * self.ndf
            forces[direction - 1] = scale * c
            pattern.load(node=node, forces=tuple(forces))


def base_tributary_area(
    fem: Any, base: str, *, ndm: int, thickness: float = 1.0,
) -> dict[int, float]:
    """``{node: area}`` of the base ``base`` (a label or physical group).

    In 2-D the base is a curve: each line element's length is shared
    equally by its two end nodes and multiplied by ``thickness``. In 3-D
    it is a surface: each face's area is shared by its corner nodes.
    Midside nodes of higher-order elements get none.
    """
    if ndm == 3:
        from ._internal.spring_bed import nodal_tributary_area
        return nodal_tributary_area(fem, base)
    conn: Any = None
    for groups in (fem.nodes.labels, fem.nodes.physical):
        if groups is not None and base in groups:
            try:
                conn = groups.connectivity(base, dim=1)
            except (KeyError, ValueError):
                conn = None  # this namespace has no curve of that name
            if conn is not None:
                break
    if conn is None or len(conn) == 0:
        raise ValueError(
            f"lysmer_base: base={base!r} names no label or physical group "
            f"with line elements on the FEM snapshot (a 2-D base is a "
            f"curve; mesh it with the model)."
        )
    xyz = {int(n): np.asarray(c, dtype=float)
           for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    area: dict[int, float] = {}
    for row in np.asarray(conn):
        ends = [int(n) for n in row if int(n) >= 0][:2]
        length = float(np.linalg.norm(xyz[ends[1]] - xyz[ends[0]]))
        for n in ends:
            area[n] = area.get(n, 0.0) + 0.5 * length * float(thickness)
    return area


def _tied_slave_dofs(fem: Any) -> set[tuple[int, int]]:
    """``(node, dof)`` of every slave DOF an ``equal_dof`` record ties."""
    out: set[tuple[int, int]] = set()
    for rec in getattr(fem.nodes, "constraints", None) or ():
        if getattr(rec, "kind", None) == "equal_dof":
            for d in rec.dofs:
                out.add((int(rec.slave_node), int(d)))
    return out


def lysmer_base(
    ops: "apeSees",
    base: str,
    ground: DecoupledNodeSetDef,
    *,
    rho: float,
    vs: float,
    vp: float | None = None,
    thickness: float = 1.0,
) -> LysmerBase:
    """Build a Lysmer–Kuhlemeyer viscous base under ``base``.

    For each node of the ``base`` curve (2-D) or surface (3-D) it
    registers one ``ZeroLength`` from that node's ``ground`` node to it,
    with a ``Viscous`` dashpot ``C_s = rho Vs A_i`` on every horizontal
    DOF (1 in 2-D; 1 and 2 in 3-D) and, when ``vp`` is given,
    ``C_p = rho Vp A_i`` on the vertical one (2 in 2-D, 3 in 3-D). It
    fixes every ground node, sizes their ``ndf``, and, with no ``vp``,
    fixes the vertical DOF of every base node — except a DOF a
    ``g.constraints.periodic`` / ``equal_dof`` tie already slaves to a
    fixed master, which the tie carries. Add the input motion with
    :meth:`LysmerBase.add_input` on the shaking pattern.

    Parameters
    ----------
    ops
        The bridge, after ``ops.model(ndm=, ndf=)``.
    base
        Label or physical group of the base boundary.
    ground
        The ``g.decouple_node_set(base, ...)`` handle: one ground node
        per base node (the source must be ``base``).
    rho, vs, vp
        Mass density, shear- and compression-wave velocities of the
        underlying half-space, in the model's units.
    thickness
        Out-of-plane thickness of a 2-D model (the base area per unit
        length); ignored in 3-D.
    """
    if ops._ndm is None or ops._ndf is None:
        raise ValueError(
            "lysmer_base: call ops.model(ndm=, ndf=) before building the base.")
    ndm, ndf = int(ops._ndm), int(ops._ndf)
    if ndm not in (2, 3):
        raise ValueError(f"lysmer_base: ndm must be 2 or 3, got {ndm}.")
    for what, value in (("rho", rho), ("vs", vs), ("thickness", thickness)):
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"lysmer_base: {what} must be > 0, got {value!r}.")
    if vp is not None and not (np.isfinite(vp) and vp > vs):
        raise ValueError(
            f"lysmer_base: vp must exceed vs ({vs!r}), got {vp!r}.")
    if not isinstance(ground, DecoupledNodeSetDef):
        raise TypeError(
            f"lysmer_base: ground= must be a g.decouple_node_set(...) "
            f"handle, got {type(ground).__name__}."
        )
    pairs = ground.pairs()
    fem = ops.fem
    area = base_tributary_area(fem, base, ndm=ndm, thickness=thickness)
    nodes = tuple(sorted(area))
    missing = [n for n in nodes if n not in pairs]
    if missing:
        raise ValueError(
            f"lysmer_base: base nodes {missing[:5]} have no ground node in "
            f"the set {ground.label or ground.source!r}; declare it with "
            f"g.decouple_node_set({base!r}, ...)."
        )
    horizontal: tuple[int, ...] = (1,) if ndm == 2 else (1, 2)
    vertical = ndm
    a = np.array([area[n] for n in nodes])
    c_s = rho * vs * a
    c_p = None if vp is None else rho * vp * a

    dashpots: dict[float, Any] = {}

    def _dashpot(c: float) -> Any:
        # Equal tributary areas give coefficients equal to round-off:
        # one material each.
        key = float(f"{float(c):.12g}")
        if key not in dashpots:
            dashpots[key] = ops.uniaxialMaterial.Viscous(C=key, alpha=1.0)
        return dashpots[key]

    gtags = tuple(int(pairs[n]) for n in nodes)
    elements: list[ZeroLength] = []
    for i, (node, gtag) in enumerate(zip(nodes, gtags)):
        mat_dirs = [ZeroLengthMatDir(material=_dashpot(c_s[i]), dof=d)
                    for d in horizontal]
        if c_p is not None:
            mat_dirs.append(
                ZeroLengthMatDir(material=_dashpot(c_p[i]), dof=vertical))
        elements.append(ops.element.ZeroLength(
            nodes=(gtag, node), mat_dirs=tuple(mat_dirs)))
    ops.fix(nodes=list(gtags), dofs=(1,) * ndf)
    for gtag in gtags:
        ops.ndf(gtag, ndf=ndf)
    if c_p is None:
        tied = _tied_slave_dofs(fem)
        free = [n for n in nodes if (n, vertical) not in tied]
        mask = tuple(1 if d == vertical else 0 for d in range(1, ndf + 1))
        ops.fix(nodes=free, dofs=mask)
    return LysmerBase(
        nodes=nodes, ground=gtags, area=tuple(float(v) for v in a),
        c_s=tuple(float(v) for v in c_s),
        c_p=None if c_p is None else tuple(float(v) for v in c_p),
        horizontal=horizontal, vertical=vertical, ndf=ndf,
        elements=tuple(elements),
    )
