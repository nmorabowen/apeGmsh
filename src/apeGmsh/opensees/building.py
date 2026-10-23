"""Story and pier aggregation on the live OpenSees domain.

The building tables an ETABS model reports — *Story Forces*, *Story
Drifts*, *Pier Forces* and *Mass Summary by Story* / *Centers of Mass* —
computed in-process from the domain the bridge built
(:func:`~apeGmsh.opensees.emitter.live.get_ops`, or an explicit ``ops=``).

Every quantity is read from the **current** domain state, so the same
object serves a static case, the gravity state of a P-Delta run, and each
mode of a response-spectrum analysis.

Extractors
----------
:class:`StoryForces`, :class:`StoryDrifts` and :class:`PierForces` are
*extractors*: callables ``extractor(ops) -> ndarray`` of a fixed length,
with a matching ``labels`` tuple (the :class:`Extractor` protocol). The
domain is scanned once, at construction (node coordinates, element
connectivity, the elements and nodes each cut needs, the checks below);
a call only reads element forces and node displacements for those, both
at the **trial** state (``eleForce`` is the trial resisting force; a
displacement is ``nodeDisp``, the committed value, plus ``nodeResponse``
``IncrDisp``, trial minus committed). After a committed solve the two
states agree; an uncommitted trial state, such as gravity plus one mode
left by ``responseSpectrumAnalysis -increment``, is read as is. A modal driver
evaluates an extractor at each mode and combines each entry of the
vector across modes; :meth:`StoryForces.table` (and the other ``table``
methods) reshapes any such vector, combined or not, into a
:class:`StoryTable`. The plain functions :func:`story_forces`,
:func:`story_drifts`, :func:`pier_forces` and :func:`story_masses` are
the one-off forms.

The free body
-------------
A story or pier force is the resultant the part above a horizontal cut
transmits to the part below. The cut is **node-aligned**: it runs through
the nodes at the cut elevation, and an element belongs to the part above
when all its nodes are at or above the cut and it does not lie in the
cut plane. Elements in the plane (the floor slab, the level's beams)
belong to the part below; nodal loads on cut nodes act on the part below.
``eleForce`` (the ``forces`` response: ``Element::getResistingForce``, the
global nodal force vector the integrator assembles, element loads
included) of the elements above that touch the cut gives, at each cut
node, the force the node applies to them; the force they transmit is its
negative. Hence, for a structure loaded only above the cut,

    ``(VX, VY, P) = sum of the loads applied above the cut``

exactly (to solver round-off), and at the base it equals ``-sum(R)``
(:func:`base_shear_from_reactions`). Negate the six components for the
reaction convention. The top of a story (``location="top"``) is the cut
just below a level, read from the elements below it that touch the level.

Refusals
--------
- an element whose nodes lie strictly on both sides of a cut (the cut is
  not node-aligned there);
- an MP constraint (rigid link, equalDOF, ...) joining a node of the part
  above to a node of the part below: it carries force past the elements,
  so the element free body would miss it;
- any EQ constraint in the domain (``equationConstraint``, the fork's
  ``LadrunoTie``): OpenSees does not report their nodes, so whether one
  crosses a cut cannot be checked. The count comes from a ``printModel``
  dump to a temporary file, which :class:`StoryForces` writes once at
  construction (about 30 µs per node and element);
- in :class:`StoryDrifts` joint mode, a story none of whose joints has a
  node directly below it;
- an element whose ``forces`` vector is not one value per DOF of its nodes
  (``eleType`` is named);
- nodes with an ``ndf`` other than 3 or 6, and 2-D models.

Code checks
-----------
The tables carry what an ASCE 7 check reads; naming columns is the
caller's. ``AvgX`` / ``AvgY`` are the ASCE 7 average of the values at the
two extreme points, ``(|max| + |min|) / 2`` of the story's (or level's)
signed entries, not a mean over every joint, so the max/avg ratio of
Table 12.3-1 and ``Ax`` (Eq. 12.8-14) do not move with the mesh. For
example, apeETABS's ``TorsionCheck`` drifts frame::

    t = drifts.story_max(combined).to_frame().rename_axis("Story").reset_index()
    h = dict(zip(drifts.stories, drifts.heights))
    t["Elevation"] = list(drifts.levels[1:])
    t["drift_max"] = t["DriftX"] * t["Story"].map(h)
    t["drift_avg"] = t["AvgX"] * t["Story"].map(h)

and its displacements frame (for ``Ax``) is ``StoryDisplacements.story_max()``
with ``MaxX, AvgX`` renamed ``Maximum, Average``.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from ._internal.build.errors import BridgeError

if TYPE_CHECKING:
    import pandas

    from ..mesh.FEMData import FEMData
    from .apesees import apeSees

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

__all__ = [
    "Extractor",
    "Pier",
    "PierForces",
    "StoryDisplacements",
    "StoryDrifts",
    "StoryForces",
    "StoryTable",
    "base_shear_from_reactions",
    "levels_from_pgs",
    "pier_forces",
    "piers_by_story",
    "story_drifts",
    "story_forces",
    "story_masses",
]

#: Component order of a story force row (ETABS *Story Forces*).
STORY_FORCE_COMPONENTS: tuple[str, ...] = ("VX", "VY", "P", "MX", "MY", "MZ")
#: Component order of a pier force row (ETABS *Pier Forces*).
PIER_FORCE_COMPONENTS: tuple[str, ...] = ("P", "V2", "V3", "T", "M2", "M3")
#: Columns of :func:`story_masses`.
STORY_MASS_COLUMNS: tuple[str, ...] = ("MassX", "MassY", "MassZ", "XCM", "YCM")


class Extractor(Protocol):
    """A live quantity: ``extractor(ops)`` returns a 1-D ``float64`` array
    of ``len(labels)`` entries, read from the current domain state."""

    labels: tuple[str, ...]

    def __call__(self, ops: ModuleType | None = None) -> FloatArray: ...


@dataclass(frozen=True, eq=False)
class StoryTable:
    """A small labelled table: ``values[i, j]`` is row ``rows[i]``,
    column ``columns[j]``."""

    rows: tuple[str, ...]
    columns: tuple[str, ...]
    values: FloatArray

    def column(self, name: str) -> FloatArray:
        """The column ``name``, one value per row."""
        return np.asarray(self.values[:, self.columns.index(name)])

    def row(self, name: str) -> FloatArray:
        """The row ``name``, one value per column."""
        return np.asarray(self.values[self.rows.index(name), :])

    def to_frame(self) -> "pandas.DataFrame":
        """The table as a :class:`pandas.DataFrame` (pandas imported lazily)."""
        import pandas as pd

        return pd.DataFrame(self.values, index=list(self.rows),
                            columns=list(self.columns))

    def __repr__(self) -> str:
        return (f"StoryTable(rows={len(self.rows)}, "
                f"columns={list(self.columns)})")


# ---------------------------------------------------------------------------
# Domain snapshot
# ---------------------------------------------------------------------------


def _resolve_ops(ops: ModuleType | None) -> ModuleType:
    if ops is not None:
        return ops
    from .emitter.live import get_ops

    return get_ops()


def _as_int(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


@dataclass(frozen=True, eq=False)
class _Domain:
    """Node coordinates and element connectivity of the live domain, read once."""

    node_tags: IntArray
    coords: FloatArray           # (N, 3)
    ndf: IntArray                # (N,)
    row_of: dict[int, int]
    ele_tags: IntArray           # (E,)
    ele_rows: tuple[IntArray, ...]
    ele_zmin: FloatArray
    ele_zmax: FloatArray
    tol: float

    def ele_type(self, ops: ModuleType, tag: int) -> str:
        try:
            return str(ops.eleType(int(tag)))
        except Exception:  # noqa: BLE001 - a name for an error message only
            return "?"


def _capture(ops: ModuleType, tol: float | None, where: str) -> _Domain:
    node_tags = np.asarray([int(t) for t in ops.getNodeTags()], dtype=np.int64)
    if node_tags.size == 0:
        raise BridgeError(f"{where}: the live OpenSees domain has no nodes.")
    coords_l: list[list[float]] = []
    ndf_l: list[int] = []
    for t in node_tags:
        c = [float(v) for v in ops.nodeCoord(int(t))]
        if len(c) != 3:
            raise BridgeError(
                f"{where}: node {int(t)} has {len(c)} coordinates; story and "
                "pier aggregation needs a 3-D model (ndm 3, Z vertical)."
            )
        coords_l.append(c)
        ndf_l.append(_as_int(ops.getNDF(int(t))))
    coords = np.asarray(coords_l, dtype=np.float64)
    ndf = np.asarray(ndf_l, dtype=np.int64)
    row_of = {int(t): i for i, t in enumerate(node_tags)}
    if tol is None:
        span = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        tol = 1e-6 * max(span, 1.0)
    tol = float(tol)
    if not (tol > 0.0 and math.isfinite(tol)):
        raise ValueError(f"{where}: tol must be a positive finite number, got {tol}.")
    ele_tags = np.asarray([int(t) for t in ops.getEleTags()], dtype=np.int64)
    rows: list[IntArray] = []
    zmin = np.empty(ele_tags.size)
    zmax = np.empty(ele_tags.size)
    for k, e in enumerate(ele_tags):
        r = np.asarray([row_of[int(n)] for n in ops.eleNodes(int(e))],
                       dtype=np.int64)
        rows.append(r)
        z = coords[r, 2]
        zmin[k] = z.min() if z.size else np.nan
        zmax[k] = z.max() if z.size else np.nan
    return _Domain(
        node_tags=node_tags, coords=coords, ndf=ndf, row_of=row_of,
        ele_tags=ele_tags, ele_rows=tuple(rows), ele_zmin=zmin,
        ele_zmax=zmax, tol=tol,
    )


def _mp_pairs(ops: ModuleType, dom: _Domain) -> IntArray:
    """``(constrained_row, retained_row)`` of every MP constraint, ``(M, 2)``."""
    out: list[tuple[int, int]] = []
    for t in dom.node_tags:
        retained = ops.getRetainedNodes(int(t))
        if not retained:
            continue
        if not isinstance(retained, (list, tuple)):
            retained = [retained]
        c = dom.row_of[int(t)]
        for r in retained:
            out.append((c, dom.row_of[int(r)]))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


#: The line ``Domain::Print`` writes before the EQ constraints (openseespy
#: >= 3.8 and the fork).
_EQ_HEADER = "EQ_Constraints: numConstraints:"


def _eq_constraint_count(ops: ModuleType, where: str) -> int:
    """The number of EQ constraints (``equationConstraint``, the fork's
    ``LadrunoTie``) in the live domain.

    OpenSees has no query for them (``getRetainedNodes`` sees MP
    constraints only), so the count is read from the ``printModel`` dump,
    the one place the domain reports it. A build without
    ``equationConstraint`` cannot hold one; any other build whose dump
    lacks the count raises.
    """
    with tempfile.TemporaryDirectory(prefix="apegmsh_eq_") as d:
        path = os.path.join(d, "domain.txt")
        ops.printModel("-file", path)
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith(_EQ_HEADER):
                        return int(line[len(_EQ_HEADER):].split()[0])
    if not hasattr(ops, "equationConstraint"):
        return 0
    raise BridgeError(
        f"{where}: printModel did not report the domain's EQ constraint "
        f"count ({_EQ_HEADER!r}), so an equationConstraint across a cut "
        "cannot be ruled out."
    )


def _check_levels(levels: Sequence[float], where: str) -> FloatArray:
    z = np.asarray([float(v) for v in levels], dtype=np.float64)
    if z.ndim != 1 or z.size < 2:
        raise ValueError(f"{where}: levels needs at least two elevations "
                         f"(the base and one level); got {list(z)}.")
    if not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0.0):
        raise ValueError(f"{where}: levels must be finite and strictly "
                         f"increasing; got {list(z)}.")
    return z


def _level_names(names: Sequence[str] | None, n_levels: int, where: str) -> tuple[str, ...]:
    if names is None:
        return ("Base",) + tuple(f"Story{i}" for i in range(1, n_levels))
    out = tuple(str(s) for s in names)
    if len(out) != n_levels:
        raise ValueError(f"{where}: names has {len(out)} entries; levels has "
                         f"{n_levels} (one name per level, the base first).")
    if len(set(out)) != len(out):
        raise ValueError(f"{where}: names must be unique; got {list(out)}.")
    return out


# ---------------------------------------------------------------------------
# Cuts and the element-force gather
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Cut:
    """One node-aligned horizontal cut: which element-force entries to sum."""

    z: float
    sign: float              # -1: elements above (force on them negated); +1: below
    ent_elem: IntArray       # index into the gatherer's element list
    ent_off: IntArray        # DOF offset of the node inside that element's vector
    ent_ndf: IntArray
    ent_xyz: FloatArray      # (k, 3) node coordinates
    ent_node: IntArray       # node rows


class _Gatherer:
    """The elements a set of cuts reads, and one ``eleForce`` per element per call."""

    def __init__(self) -> None:
        self.tags: list[int] = []
        self.sizes: list[int] = []
        self._pos: dict[int, int] = {}

    def index(self, tag: int, size: int) -> int:
        k = self._pos.get(tag)
        if k is None:
            k = self._pos[tag] = len(self.tags)
            self.tags.append(tag)
            self.sizes.append(size)
        return k

    def freeze(self) -> None:
        self.starts = np.concatenate(
            [[0], np.cumsum(np.asarray(self.sizes, dtype=np.int64))]
        ).astype(np.int64)

    def gather(self, ops: ModuleType, dom: _Domain, where: str) -> FloatArray:
        """Every element's force vector, concatenated, plus one trailing 0."""
        parts: list[FloatArray] = []
        for tag, size in zip(self.tags, self.sizes):
            f = np.asarray(ops.eleForce(int(tag)), dtype=np.float64).ravel()
            if f.size != size:
                raise BridgeError(
                    f"{where}: element {tag} ({dom.ele_type(ops, tag)}) "
                    f"answered eleForce with {f.size} values; its nodes carry "
                    f"{size} DOFs. Its 'forces' response is not the global "
                    "nodal force vector, so it cannot enter a cut free body."
                )
            parts.append(f)
        parts.append(np.zeros(1))
        return np.concatenate(parts)

    def resultant(self, flat: FloatArray, cut: _Cut, ref: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``(force (3,), moment (3,))`` of ``cut`` about ``ref``."""
        if cut.ent_elem.size == 0:
            return np.zeros(3), np.zeros(3)
        base = self.starts[cut.ent_elem] + cut.ent_off
        f = flat[base[:, None] + np.arange(3)] * cut.sign
        pad = flat.size - 1
        rot_idx = np.where((cut.ent_ndf == 6)[:, None],
                           base[:, None] + 3 + np.arange(3), pad)
        m = flat[rot_idx] * cut.sign
        arm = cut.ent_xyz - ref
        force = f.sum(axis=0)
        moment = np.cross(arm, f).sum(axis=0) + m.sum(axis=0)
        return force, moment


def _check_ndf(dom: _Domain, rows: IntArray, where: str) -> None:
    bad = rows[~np.isin(dom.ndf[rows], (3, 6))]
    if bad.size:
        r = int(bad[0])
        raise BridgeError(
            f"{where}: node {int(dom.node_tags[r])} has ndf {int(dom.ndf[r])}; "
            "a cut reads translations (ndf 3) or translations and rotations "
            "(ndf 6) only."
        )


def _build_cut(
    ops: ModuleType,
    dom: _Domain,
    gatherer: _Gatherer,
    z: float,
    side: Literal["above", "below"],
    elements: IntArray | None,
    where: str,
    *,
    check_mp: IntArray | None,
) -> _Cut:
    """The cut at ``z`` read from the elements on ``side`` (all, or ``elements``)."""
    tol = dom.tol
    if elements is None:
        cand = np.arange(dom.ele_tags.size)
    else:
        cand = elements
    zmin = dom.ele_zmin[cand]
    zmax = dom.ele_zmax[cand]
    span = cand[(zmin < z - tol) & (zmax > z + tol)]
    if span.size:
        tags = [int(dom.ele_tags[k]) for k in span[:8]]
        kinds = sorted({dom.ele_type(ops, t) for t in tags})
        raise BridgeError(
            f"{where}: element(s) {tags}{' ...' if span.size > 8 else ''} "
            f"({', '.join(kinds)}) cross the cut at z={z:g} without a node on "
            "it. A cut is node-aligned: mesh a node line at every level."
        )
    if check_mp is not None and check_mp.size:
        nz = dom.coords[:, 2]
        upper = (nz > z + tol) if side == "above" else (nz >= z - tol)
        cross = check_mp[upper[check_mp[:, 0]] != upper[check_mp[:, 1]]]
        if cross.size:
            pairs = [(int(dom.node_tags[a]), int(dom.node_tags[b]))
                     for a, b in cross[:5]]
            raise BridgeError(
                f"{where}: MP constraint(s) (constrained, retained) {pairs} join "
                f"a node above the cut at z={z:g} to one below it. Their force "
                "bypasses the elements, so the element free body would miss "
                "it; keep constraints inside a level or within one side of it."
            )
    touch = cand[(zmin <= z + tol) & (zmax >= z - tol)
                 & ~((zmax - zmin) <= 2.0 * tol)]
    if side == "above":
        touch = touch[dom.ele_zmin[touch] >= z - tol]
    else:
        touch = touch[dom.ele_zmax[touch] <= z + tol]
    e_idx: list[int] = []
    offs: list[int] = []
    ndfs: list[int] = []
    nodes: list[int] = []
    for k in touch:
        rows = dom.ele_rows[int(k)]
        nd = dom.ndf[rows]
        size = int(nd.sum())
        gi = gatherer.index(int(dom.ele_tags[k]), size)
        off = np.concatenate([[0], np.cumsum(nd)[:-1]])
        on = np.abs(dom.coords[rows, 2] - z) <= tol
        for r, o, n in zip(rows[on], off[on], nd[on]):
            e_idx.append(gi)
            offs.append(int(o))
            ndfs.append(int(n))
            nodes.append(int(r))
    node_rows = np.asarray(nodes, dtype=np.int64)
    _check_ndf(dom, node_rows, where)
    return _Cut(
        z=float(z),
        sign=-1.0 if side == "above" else 1.0,
        ent_elem=np.asarray(e_idx, dtype=np.int64),
        ent_off=np.asarray(offs, dtype=np.int64),
        ent_ndf=np.asarray(ndfs, dtype=np.int64),
        ent_xyz=dom.coords[node_rows] if node_rows.size else np.zeros((0, 3)),
        ent_node=node_rows,
    )


# ---------------------------------------------------------------------------
# Story forces
# ---------------------------------------------------------------------------


class StoryForces:
    """Story shear and overturning per story, as an :class:`Extractor`.

    Parameters
    ----------
    levels
        Level elevations, strictly increasing, the base first. Story ``i``
        (``i = 1..n``) spans ``levels[i-1]`` to ``levels[i]`` and is named
        after its top level.
    names
        One name per level (default ``"Base", "Story1", ...``).
    location
        ``"bottom"`` (default): the cut just above ``levels[i-1]``, the
        resultant of every load above that level. ``"top"``: the cut just
        below ``levels[i]``, the resultant of every load at or above it.
    reference
        Plan point ``(x, y)`` the moments are taken about, at each cut's
        elevation (default the origin). ``MZ`` is the story torsion.
    ops
        The OpenSees module; default :func:`~apeGmsh.opensees.emitter.live.get_ops`.
    tol
        Elevation tolerance; default ``1e-6`` of the model's size.

    A call returns ``[VX, VY, P, MX, MY, MZ]`` per story, stories in level
    order (``labels`` names each entry ``"<story>:<component>"``).
    """

    def __init__(
        self,
        levels: Sequence[float],
        *,
        names: Sequence[str] | None = None,
        location: Literal["bottom", "top"] = "bottom",
        reference: tuple[float, float] = (0.0, 0.0),
        ops: ModuleType | None = None,
        tol: float | None = None,
    ) -> None:
        where = "StoryForces"
        if location not in ("bottom", "top"):
            raise ValueError(f"{where}: location must be 'bottom' or 'top', "
                             f"got {location!r}.")
        z = _check_levels(levels, where)
        level_names = _level_names(names, z.size, where)
        self.levels: tuple[float, ...] = tuple(float(v) for v in z)
        self.stories: tuple[str, ...] = level_names[1:]
        self.location = location
        self._ref_xy = (float(reference[0]), float(reference[1]))
        module = _resolve_ops(ops)
        self._dom = _capture(module, tol, where)
        n_eq = _eq_constraint_count(module, where)
        if n_eq:
            raise BridgeError(
                f"{where}: the domain holds {n_eq} EQ constraint(s) "
                "(equationConstraint, LadrunoTie, or an enforce='equation' "
                "tie). OpenSees does not report their nodes, so a cut "
                "cannot tell whether one carries force past the elements, "
                "which the element free body would miss. Story forces "
                "refuse any EQ constraint; tie with equalDOF / rigid links "
                "inside a level, or read the forces from a model without them."
            )
        mp = _mp_pairs(module, self._dom)
        self._gather = _Gatherer()
        cuts: list[_Cut] = []
        for i in range(1, z.size):
            if location == "bottom":
                cuts.append(_build_cut(module, self._dom, self._gather, z[i - 1],
                                       "above", None, f"{where} {self.stories[i-1]}",
                                       check_mp=mp))
            else:
                cuts.append(_build_cut(module, self._dom, self._gather, z[i],
                                       "below", None, f"{where} {self.stories[i-1]}",
                                       check_mp=mp))
        self._gather.freeze()
        self._cuts = tuple(cuts)
        self.labels: tuple[str, ...] = tuple(
            f"{s}:{c}" for s in self.stories for c in STORY_FORCE_COMPONENTS)

    def __call__(self, ops: ModuleType | None = None) -> FloatArray:
        module = _resolve_ops(ops)
        flat = self._gather.gather(module, self._dom, "StoryForces")
        out = np.empty((len(self._cuts), 6))
        for i, cut in enumerate(self._cuts):
            ref = np.array([self._ref_xy[0], self._ref_xy[1], cut.z])
            f, m = self._gather.resultant(flat, cut, ref)
            out[i, :3] = f
            out[i, 3:] = m
        return out.ravel()

    def table(self, values: FloatArray | None = None, *,
              ops: ModuleType | None = None) -> StoryTable:
        """``values`` (a call's vector, or any combination of them) as a
        story × component table; evaluates now when ``values`` is ``None``."""
        v = self(ops) if values is None else _check_len(values, self.labels)
        return StoryTable(self.stories, STORY_FORCE_COMPONENTS,
                          v.reshape(len(self.stories), 6))


def _check_len(values: FloatArray | Sequence[float], labels: tuple[str, ...]) -> FloatArray:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size != len(labels):
        raise ValueError(f"values has {v.size} entries; this extractor has "
                         f"{len(labels)} labels.")
    return v


def story_forces(
    levels: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    location: Literal["bottom", "top"] = "bottom",
    reference: tuple[float, float] = (0.0, 0.0),
    ops: ModuleType | None = None,
    tol: float | None = None,
) -> StoryTable:
    """One-off :class:`StoryForces` table at the current domain state."""
    return StoryForces(levels, names=names, location=location,
                       reference=reference, ops=ops, tol=tol).table(ops=ops)


def base_shear_from_reactions(
    z_base: float,
    *,
    reference: tuple[float, float] = (0.0, 0.0),
    ops: ModuleType | None = None,
    tol: float | None = None,
) -> FloatArray:
    """``-sum(R)`` over the nodes at or below ``z_base``, as
    ``[VX, VY, P, MX, MY, MZ]`` about ``(reference, z_base)``.

    The cross-check of the base story's :class:`StoryForces` row: the two
    agree when no load is applied at or below the base. Calls
    ``ops.reactions()``.
    """
    module = _resolve_ops(ops)
    dom = _capture(module, tol, "base_shear_from_reactions")
    module.reactions()
    ref = np.array([float(reference[0]), float(reference[1]), float(z_base)])
    force = np.zeros(3)
    moment = np.zeros(3)
    for r in np.flatnonzero(dom.coords[:, 2] <= float(z_base) + dom.tol):
        reac = np.asarray(module.nodeReaction(int(dom.node_tags[r])), dtype=np.float64)
        f = reac[:3]
        force += f
        moment += np.cross(dom.coords[r] - ref, f)
        if reac.size >= 6:
            moment += reac[3:6]
    return np.asarray(-np.concatenate([force, moment]), dtype=np.float64)


# ---------------------------------------------------------------------------
# Story drifts
# ---------------------------------------------------------------------------


class StoryDrifts:
    """Interstory drift ratios ``(u_top - u_bottom) / h``, as an :class:`Extractor`.

    Parameters
    ----------
    levels, names, ops, tol
        As :class:`StoryForces`.
    points
        Plan points ``(x, y)`` to report. ``None`` (default) reports every
        *joint*: each node at a level that has a node at the same plan
        position (within ``tol``) on the level below — ETABS *Joint
        Drifts*, whose maximum per story (:meth:`story_max`) is ETABS
        *Story Drifts*. A story with no such joint raises.
    method
        How a point reads a level's displacement. ``"nearest"``: the
        nearest node of the level in plan, which must lie within
        ``max_offset`` (default ``tol``). ``"rigid"``: the least-squares
        in-plane rigid motion (``ux, uy, rz``) of all the level's nodes,
        evaluated at the point (exact for a rigid diaphragm).

    A call returns ``[DX, DY]`` per entry, entries per story in level
    order; ``labels`` reads ``"<story>:<node>:DX"`` (joints) or
    ``"<story>:P<j>:DX"`` (points). Displacements are the trial
    displacements (see the module notes).
    """

    def __init__(
        self,
        levels: Sequence[float],
        *,
        points: Sequence[tuple[float, float]] | None = None,
        names: Sequence[str] | None = None,
        method: Literal["nearest", "rigid"] = "nearest",
        max_offset: float | None = None,
        ops: ModuleType | None = None,
        tol: float | None = None,
    ) -> None:
        where = "StoryDrifts"
        if method not in ("nearest", "rigid"):
            raise ValueError(f"{where}: method must be 'nearest' or 'rigid', "
                             f"got {method!r}.")
        z = _check_levels(levels, where)
        level_names = _level_names(names, z.size, where)
        self.levels: tuple[float, ...] = tuple(float(v) for v in z)
        self.stories: tuple[str, ...] = level_names[1:]
        self.heights: tuple[float, ...] = tuple(float(h) for h in np.diff(z))
        module = _resolve_ops(ops)
        dom = _capture(module, tol, where)
        self._dom = dom
        on_level = [np.flatnonzero(np.abs(dom.coords[:, 2] - zk) <= dom.tol)
                    for zk in z]
        for k, rows in enumerate(on_level):
            if rows.size == 0:
                raise BridgeError(f"{where}: no node lies on level "
                                  f"{level_names[k]!r} (z={z[k]:g}).")
        # Each entry: (story index, top operator, bottom operator); an
        # operator maps the level's needed node displacements to (ux, uy).
        self._story: list[int] = []
        self._ops_top: list[tuple[IntArray, FloatArray]] = []
        self._ops_bot: list[tuple[IntArray, FloatArray]] = []
        labels: list[str] = []
        if points is None:
            for i in range(1, z.size):
                top, bot = on_level[i], on_level[i - 1]
                d = np.linalg.norm(dom.coords[top, None, :2]
                                   - dom.coords[None, bot, :2], axis=2)
                n_before = len(self._story)
                for a, row in enumerate(top):
                    b = int(np.argmin(d[a]))
                    if d[a, b] > dom.tol:
                        continue
                    self._add(i - 1, _pick(int(row)), _pick(int(bot[b])))
                    tag = int(dom.node_tags[row])
                    labels += [f"{self.stories[i-1]}:{tag}:DX",
                               f"{self.stories[i-1]}:{tag}:DY"]
                if len(self._story) == n_before:
                    raise BridgeError(
                        f"{where}: story {self.stories[i-1]!r} has no joint: "
                        f"no node on level {level_names[i]!r} (z={z[i]:g}) has "
                        f"a node at the same plan position on level "
                        f"{level_names[i-1]!r} (z={z[i-1]:g}), as at a transfer "
                        "level or an offset column. Its drift would read zero; "
                        "pass points= (method='rigid' for a rigid diaphragm)."
                    )
        else:
            pts = [(float(p[0]), float(p[1])) for p in points]
            if not pts:
                raise ValueError(f"{where}: points is empty; pass None for every joint.")
            off = dom.tol if max_offset is None else float(max_offset)
            per_level: list[list[tuple[IntArray, FloatArray]]] = []
            for k, rows in enumerate(on_level):
                if method == "nearest":
                    per_level.append([
                        _nearest(dom, rows, p, off, f"{where} level "
                                 f"{level_names[k]!r}") for p in pts])
                else:
                    per_level.append(_rigid(dom, rows, pts,
                                            f"{where} level {level_names[k]!r}"))
            for i in range(1, z.size):
                for j in range(len(pts)):
                    self._add(i - 1, per_level[i][j], per_level[i - 1][j])
                    labels += [f"{self.stories[i-1]}:P{j}:DX",
                               f"{self.stories[i-1]}:P{j}:DY"]
        need =sorted({int(r) for op in self._ops_top + self._ops_bot for r in op[0]})
        self._need = np.asarray(need, dtype=np.int64)
        pos = {r: k for k, r in enumerate(need)}
        self._top_idx = [np.asarray([pos[int(r)] for r in op[0]], dtype=np.int64)
                         for op in self._ops_top]
        self._bot_idx = [np.asarray([pos[int(r)] for r in op[0]], dtype=np.int64)
                         for op in self._ops_bot]
        self._h = np.asarray([self.heights[s] for s in self._story])
        self.labels: tuple[str, ...] = tuple(labels)

    def _add(self, story: int, top: tuple[IntArray, FloatArray],
             bot: tuple[IntArray, FloatArray]) -> None:
        self._story.append(story)
        self._ops_top.append(top)
        self._ops_bot.append(bot)

    def __call__(self, ops: ModuleType | None = None) -> FloatArray:
        module = _resolve_ops(ops)
        u = np.empty((self._need.size, 2))
        for k, r in enumerate(self._need):
            u[k] = _trial_xy(module, int(self._dom.node_tags[r]))
        out = np.empty((len(self._story), 2))
        for e in range(len(self._story)):
            ut = self._ops_top[e][1] @ u[self._top_idx[e]].ravel()
            ub = self._ops_bot[e][1] @ u[self._bot_idx[e]].ravel()
            out[e] = (ut - ub) / self._h[e]
        return out.ravel()

    def table(self, values: FloatArray | None = None, *,
              ops: ModuleType | None = None) -> StoryTable:
        """Every entry as a row (``"<story>:<node|P<j>>"``), columns ``DX, DY``."""
        v = self(ops) if values is None else _check_len(values, self.labels)
        rows = tuple(lab.rsplit(":", 1)[0] for lab in self.labels[::2])
        return StoryTable(rows, ("DX", "DY"), v.reshape(-1, 2))

    def story_max(self, values: FloatArray | None = None, *,
                  ops: ModuleType | None = None) -> StoryTable:
        """The largest ``|DX|`` and ``|DY|`` of each story (ETABS *Story
        Drifts*), and beside them ``AvgX``, ``AvgY``: the ASCE 7 average of
        the story drifts at the two extreme points, ``(|max| + |min|) / 2``
        of the story's entries (:func:`_max_avg`), the denominator of the
        Table 12.3-1 torsional irregularity ratio. For a modal combination,
        pass the combined vector: the maximum of the combined joint drifts,
        not the combination of maxima."""
        v = self(ops) if values is None else _check_len(values, self.labels)
        return StoryTable(self.stories, ("DriftX", "DriftY", "AvgX", "AvgY"),
                          _max_avg(v, self._story, len(self.stories)))


def _max_avg(values: FloatArray, story: Sequence[int], n: int) -> FloatArray:
    """``(n, 4)``: per story ``max|v|`` in X and Y, then the ASCE 7 average.

    The average is that of the two extreme-point values, the largest and
    the smallest signed value of the story's entries:
    ``(|max v| + |min v|) / 2``. Under a rigid diaphragm a level's
    in-plane displacement is linear in plan, so these are the values at
    the structure's two extreme points, whatever the number of points
    between them; a mean over every entry would move with the mesh.
    With all values of one sign (a combined vector) it is
    ``(max|v| + min|v|) / 2``.
    """
    v = values.reshape(-1, 2)
    st = np.asarray(story)
    out = np.empty((n, 4))
    for s in range(n):
        sel = v[st == s]
        if sel.shape[0] == 0:
            raise BridgeError(f"story index {s} has no entry.")
        hi = sel.max(axis=0)
        lo = sel.min(axis=0)
        out[s, :2] = np.abs(sel).max(axis=0)
        out[s, 2:] = 0.5 * (np.abs(hi) + np.abs(lo))
    return out


#: ``NodeResponseType::IncrDisp`` (trial minus committed displacement); OPS_Globals.h.
_INCR_DISP = 4


def _trial_xy(ops: ModuleType, tag: int) -> tuple[float, float]:
    """Trial ``(ux, uy)`` of node ``tag``. ``nodeDisp`` (and ``nodeResponse``
    ``Disp``) answer the committed value; ``IncrDisp`` adds trial - committed,
    which ``setTrialDisp``, ``incrTrialDisp``, ``commitState`` and
    ``revertToLastCommit`` all keep current (``Node.cpp``)."""
    d = ops.nodeDisp(tag)
    return (float(d[0]) + float(ops.nodeResponse(tag, 1, _INCR_DISP)),
            float(d[1]) + float(ops.nodeResponse(tag, 2, _INCR_DISP)))


def _pick(row: int) -> tuple[IntArray, FloatArray]:
    return np.asarray([row], dtype=np.int64), np.eye(2)


def _nearest(dom: _Domain, rows: IntArray, p: tuple[float, float],
             max_offset: float, where: str) -> tuple[IntArray, FloatArray]:
    d = np.hypot(dom.coords[rows, 0] - p[0], dom.coords[rows, 1] - p[1])
    k = int(np.argmin(d))
    if d[k] > max_offset:
        raise BridgeError(
            f"{where}: the nearest node to point {p} is node "
            f"{int(dom.node_tags[rows[k]])}, {d[k]:g} away (max_offset "
            f"{max_offset:g}). Raise max_offset or use method='rigid'."
        )
    return _pick(int(rows[k]))


def _rigid(dom: _Domain, rows: IntArray, pts: list[tuple[float, float]],
           where: str) -> list[tuple[IntArray, FloatArray]]:
    xy = dom.coords[rows, :2]
    c = xy.mean(axis=0)
    n = rows.size
    a = np.zeros((2 * n, 3))
    a[0::2, 0] = 1.0
    a[0::2, 2] = -(xy[:, 1] - c[1])
    a[1::2, 1] = 1.0
    a[1::2, 2] = xy[:, 0] - c[0]
    if np.linalg.matrix_rank(a) < 3:
        raise BridgeError(f"{where}: the level's nodes do not fix an in-plane "
                          "rigid motion (one plan position); use method='nearest'.")
    pinv = np.linalg.pinv(a)
    out = []
    for p in pts:
        b = np.array([[1.0, 0.0, -(p[1] - c[1])], [0.0, 1.0, p[0] - c[0]]])
        out.append((rows, b @ pinv))
    return out


def story_drifts(
    levels: Sequence[float],
    *,
    points: Sequence[tuple[float, float]] | None = None,
    names: Sequence[str] | None = None,
    method: Literal["nearest", "rigid"] = "nearest",
    max_offset: float | None = None,
    ops: ModuleType | None = None,
    tol: float | None = None,
) -> StoryTable:
    """One-off drift table: the story maximum over every joint
    (``points=None``), else one row per story and point."""
    d = StoryDrifts(levels, points=points, names=names, method=method,
                    max_offset=max_offset, ops=ops, tol=tol)
    return d.story_max(ops=ops) if points is None else d.table(ops=ops)


class StoryDisplacements:
    """Level displacements ``UX, UY``, as an :class:`Extractor`.

    Every node on each level above the base (``points=None``), or the
    given plan points read as :class:`StoryDrifts` reads them (``method``,
    ``max_offset``). :meth:`story_max` gives each level's largest
    ``|UX|``, ``|UY|`` and the ASCE 7 extreme-point average
    ``(|max| + |min|) / 2``: the input of a torsional-irregularity or
    torsional-amplification check. ``labels`` read ``"<story>:<node>:UX"``
    or ``"<story>:P<j>:UX"``; the other parameters are as for
    :class:`StoryDrifts`.
    """

    def __init__(
        self,
        levels: Sequence[float],
        *,
        points: Sequence[tuple[float, float]] | None = None,
        names: Sequence[str] | None = None,
        method: Literal["nearest", "rigid"] = "nearest",
        max_offset: float | None = None,
        ops: ModuleType | None = None,
        tol: float | None = None,
    ) -> None:
        where = "StoryDisplacements"
        if method not in ("nearest", "rigid"):
            raise ValueError(f"{where}: method must be 'nearest' or 'rigid', "
                             f"got {method!r}.")
        z = _check_levels(levels, where)
        level_names = _level_names(names, z.size, where)
        self.levels: tuple[float, ...] = tuple(float(v) for v in z)
        self.stories: tuple[str, ...] = level_names[1:]
        module = _resolve_ops(ops)
        self._dom = dom = _capture(module, tol, where)
        self._story: list[int] = []
        self._ops: list[tuple[IntArray, FloatArray]] = []
        labels: list[str] = []
        for i in range(1, z.size):
            w = f"{where} level {level_names[i]!r}"
            rows = np.flatnonzero(np.abs(dom.coords[:, 2] - z[i]) <= dom.tol)
            if rows.size == 0:
                raise BridgeError(f"{w}: no node lies on it (z={z[i]:g}).")
            if points is None:
                ops_i = [_pick(int(r)) for r in rows]
                keys = [str(int(dom.node_tags[r])) for r in rows]
            else:
                pts = [(float(p[0]), float(p[1])) for p in points]
                off = dom.tol if max_offset is None else float(max_offset)
                ops_i = ([_nearest(dom, rows, p, off, w) for p in pts]
                         if method == "nearest" else _rigid(dom, rows, pts, w))
                keys = [f"P{j}" for j in range(len(pts))]
            for op, key in zip(ops_i, keys):
                self._story.append(i - 1)
                self._ops.append(op)
                labels += [f"{level_names[i]}:{key}:UX", f"{level_names[i]}:{key}:UY"]
        self.labels: tuple[str, ...] = tuple(labels)

    def __call__(self, ops: ModuleType | None = None) -> FloatArray:
        module = _resolve_ops(ops)
        tags = self._dom.node_tags
        out = np.empty((len(self._ops), 2))
        for e, (rows, mat) in enumerate(self._ops):
            u = np.array([_trial_xy(module, int(tags[r])) for r in rows])
            out[e] = mat @ u.ravel()
        return out.ravel()

    def table(self, values: FloatArray | None = None, *,
              ops: ModuleType | None = None) -> StoryTable:
        """Every entry as a row (``"<story>:<node|P<j>>"``), columns ``UX, UY``."""
        v = self(ops) if values is None else _check_len(values, self.labels)
        rows = tuple(lab.rsplit(":", 1)[0] for lab in self.labels[::2])
        return StoryTable(rows, ("UX", "UY"), v.reshape(-1, 2))

    def story_max(self, values: FloatArray | None = None, *,
                  ops: ModuleType | None = None) -> StoryTable:
        """Per level ``MaxX, MaxY, AvgX, AvgY``: the largest ``|UX|``,
        ``|UY|`` over its entries, and the ASCE 7 average of the
        displacements at the two extreme points, ``(|max| + |min|) / 2``
        (:func:`_max_avg`; ``delta_avg`` of Eq. 12.8-14). Pass a combined
        vector for a modal combination."""
        v = self(ops) if values is None else _check_len(values, self.labels)
        return StoryTable(self.stories, ("MaxX", "MaxY", "AvgX", "AvgY"),
                          _max_avg(v, self._story, len(self.stories)))


# ---------------------------------------------------------------------------
# Pier forces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pier:
    """A wall pier: OpenSees element tags between two node-aligned cuts.

    ``bottom`` / ``top`` default to the lowest / highest node of the
    elements. ``axis2`` is the plan direction of the pier's length (local
    axis 2); by default the principal direction of the bottom cut's nodes,
    oriented toward +X (+Y when it runs along Y). Axis 1 is +Z and axis
    3 = axis 1 × axis 2.
    """

    name: str
    elements: tuple[int, ...]
    bottom: float | None = None
    top: float | None = None
    axis2: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        tags = tuple(int(t) for t in self.elements)
        if not tags:
            raise ValueError(f"Pier {self.name!r}: elements is empty.")
        object.__setattr__(self, "elements", tags)

    @classmethod
    def from_pg(
        cls,
        bridge: "apeSees",
        pg: str,
        *,
        name: str | None = None,
        bottom: float | None = None,
        top: float | None = None,
        axis2: tuple[float, float] | None = None,
    ) -> "Pier":
        """The pier of physical group ``pg``'s elements in ``bridge``.

        FEM element ids map to OpenSees tags directly under
        ``element_tags="fem"``; otherwise through the bridge's live read
        session (after a live ``analyze``), which raises when there is none.
        """
        from ._readback import live_session

        where = f"Pier.from_pg({pg!r})"
        ids = [int(e) for e in bridge.fem.elements.select(pg=pg).ids]
        if not ids:
            raise ValueError(f"{where}: the physical group has no elements.")
        if bridge._element_tags == "fem":
            tags = tuple(ids)
        else:
            rows = live_session(bridge, where).element_rows(ids, where)
            tags = tuple(t for _spec, t in rows)
        return cls(name=pg if name is None else name, elements=tags,
                   bottom=bottom, top=top, axis2=axis2)


def piers_by_story(
    pier: Pier,
    levels: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    ops: ModuleType | None = None,
    tol: float | None = None,
) -> tuple[Pier, ...]:
    """Split ``pier`` (one PG over many stories) into one pier per story.

    Each story's pier holds the elements between its two levels and is
    named ``"<pier>/<story>"``; a story with none is left out. An element
    that crosses a level raises.
    """
    where = f"piers_by_story({pier.name!r})"
    z = _check_levels(levels, where)
    level_names = _level_names(names, z.size, where)
    module = _resolve_ops(ops)
    dom = _capture(module, tol, where)
    idx = _element_index(dom, pier, where)
    out: list[Pier] = []
    assigned = np.zeros(idx.size, dtype=bool)
    for i in range(1, z.size):
        sel = ((dom.ele_zmin[idx] >= z[i - 1] - dom.tol)
               & (dom.ele_zmax[idx] <= z[i] + dom.tol))
        assigned |= sel
        if sel.any():
            out.append(Pier(
                name=f"{pier.name}/{level_names[i]}",
                elements=tuple(int(dom.ele_tags[k]) for k in idx[sel]),
                bottom=float(z[i - 1]), top=float(z[i]), axis2=pier.axis2))
    if not assigned.all():
        bad = [int(dom.ele_tags[k]) for k in idx[~assigned][:8]]
        raise BridgeError(f"{where}: element(s) {bad} do not lie between two "
                          "consecutive levels (they cross a level, or lie "
                          "outside the levels).")
    return tuple(out)


def _element_index(dom: _Domain, pier: Pier, where: str) -> IntArray:
    pos = {int(t): k for k, t in enumerate(dom.ele_tags)}
    missing = [t for t in pier.elements if t not in pos]
    if missing:
        raise BridgeError(f"{where}: element(s) {missing[:8]} are not in the "
                          "live domain.")
    return np.asarray(sorted({pos[t] for t in pier.elements}), dtype=np.int64)


@dataclass(frozen=True, eq=False)
class _PierCuts:
    bottom: _Cut
    top: _Cut
    axes: FloatArray           # (3, 3): rows e1, e2, e3
    c_bottom: FloatArray       # (3,)
    c_top: FloatArray


def _section_centroid(dom: _Domain, cut: _Cut, z: float) -> FloatArray:
    """Length-weighted centroid of the cut's element edges; node mean if none."""
    if cut.ent_node.size == 0:
        raise BridgeError(f"no element of the pier touches its cut at z={z:g}.")
    total = 0.0
    acc = np.zeros(2)
    for e in np.unique(cut.ent_elem):
        xy = cut.ent_xyz[cut.ent_elem == e, :2]
        if xy.shape[0] < 2:
            continue
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        a, b = np.unravel_index(int(np.argmax(d)), d.shape)
        length = float(d[a, b])
        total += length
        acc += length * 0.5 * (xy[a] + xy[b])
    xy_c = acc / total if total > 0.0 else cut.ent_xyz[:, :2].mean(axis=0)
    return np.array([xy_c[0], xy_c[1], z])


def _pier_axes(dom: _Domain, cut: _Cut, axis2: tuple[float, float] | None,
               where: str) -> FloatArray:
    if axis2 is not None:
        v = np.array([float(axis2[0]), float(axis2[1])])
        n = float(np.linalg.norm(v))
        if not (n > 0.0 and math.isfinite(n)):
            raise ValueError(f"{where}: axis2 must be a nonzero plan vector, got {axis2}.")
        v = v / n
    else:
        xy = np.unique(cut.ent_xyz[:, :2], axis=0)
        if xy.shape[0] < 2:
            raise BridgeError(f"{where}: the bottom cut has one plan position; "
                              "pass axis2= to orient the pier.")
        w, vec = np.linalg.eigh(np.cov((xy - xy.mean(axis=0)).T))
        v = vec[:, int(np.argmax(w))]
        if v[0] < -1e-9 or (abs(v[0]) <= 1e-9 and v[1] < 0.0):
            v = -v
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.array([v[0], v[1], 0.0])
    return np.vstack([e1, e2, np.cross(e1, e2)])


class PierForces:
    """Pier forces at the bottom and top of each pier, as an :class:`Extractor`.

    ``P, V2, V3, T, M2, M3`` in pier local axes (axis 1 up, axis 2 along
    the pier in plan, axis 3 = 1 × 2), moments about the centroid of the
    pier section at the cut (the length-weighted centroid of the element
    edges on the cut, which assumes a uniform thickness). Both cuts give
    the force the part above transmits to the part below, so ``P`` is
    negative in compression and, under a lateral load ``H`` along axis 2 at
    height ``L`` above the cut, ``V2 = H`` and ``M3 = H L``.

    A call returns, per pier, the bottom row then the top row; ``labels``
    reads ``"<pier>:bottom:P"``.
    """

    def __init__(
        self,
        piers: Sequence[Pier],
        *,
        ops: ModuleType | None = None,
        tol: float | None = None,
    ) -> None:
        where = "PierForces"
        self.piers: tuple[Pier, ...] = tuple(piers)
        if not self.piers:
            raise ValueError(f"{where}: piers is empty.")
        names = [p.name for p in self.piers]
        if len(set(names)) != len(names):
            raise ValueError(f"{where}: pier names must be unique; got {names}.")
        module = _resolve_ops(ops)
        dom = _capture(module, tol, where)
        self._dom = dom
        self._gather = _Gatherer()
        cuts: list[_PierCuts] = []
        for p in self.piers:
            w = f"{where} {p.name!r}"
            idx = _element_index(dom, p, w)
            zb = float(dom.ele_zmin[idx].min()) if p.bottom is None else float(p.bottom)
            zt = float(dom.ele_zmax[idx].max()) if p.top is None else float(p.top)
            if not zt > zb + dom.tol:
                raise ValueError(f"{w}: top ({zt:g}) must lie above bottom ({zb:g}).")
            outside = idx[(dom.ele_zmin[idx] < zb - dom.tol)
                          | (dom.ele_zmax[idx] > zt + dom.tol)]
            if outside.size:
                bad = [int(dom.ele_tags[k]) for k in outside[:8]]
                raise BridgeError(
                    f"{w}: element(s) {bad} reach outside the pier's cuts "
                    f"(z {zb:g} to {zt:g}); split a multi-story pier with "
                    "piers_by_story()."
                )
            bot = _build_cut(module, dom, self._gather, zb, "above", idx, w,
                             check_mp=None)
            top = _build_cut(module, dom, self._gather, zt, "below", idx, w,
                             check_mp=None)
            axes = _pier_axes(dom, bot, p.axis2, w)
            cuts.append(_PierCuts(bot, top, axes, _section_centroid(dom, bot, zb),
                                  _section_centroid(dom, top, zt)))
        self._gather.freeze()
        self._cuts = tuple(cuts)
        self.labels: tuple[str, ...] = tuple(
            f"{p.name}:{loc}:{c}" for p in self.piers for loc in ("bottom", "top")
            for c in PIER_FORCE_COMPONENTS)

    def centroids(self) -> StoryTable:
        """Each cut's section centroid ``(x, y, z)``: rows ``"<pier>:bottom|top"``."""
        rows = tuple(f"{p.name}:{loc}" for p in self.piers for loc in ("bottom", "top"))
        vals = np.vstack([c for pc in self._cuts for c in (pc.c_bottom, pc.c_top)])
        return StoryTable(rows, ("X", "Y", "Z"), vals)

    def axes(self, name: str) -> FloatArray:
        """The ``(3, 3)`` local axes of pier ``name``, rows axis 1, 2, 3."""
        return np.asarray(self._cuts[[p.name for p in self.piers].index(name)].axes)

    def __call__(self, ops: ModuleType | None = None) -> FloatArray:
        module = _resolve_ops(ops)
        flat = self._gather.gather(module, self._dom, "PierForces")
        out = np.empty((len(self._cuts), 2, 6))
        for i, pc in enumerate(self._cuts):
            for j, (cut, c) in enumerate(((pc.bottom, pc.c_bottom), (pc.top, pc.c_top))):
                f, m = self._gather.resultant(flat, cut, c)
                fl = pc.axes @ f
                ml = pc.axes @ m
                out[i, j] = (fl[0], fl[1], fl[2], ml[0], ml[1], ml[2])
        return out.ravel()

    def table(self, values: FloatArray | None = None, *,
              ops: ModuleType | None = None) -> StoryTable:
        """Rows ``"<pier>:bottom|top"``, columns ``P, V2, V3, T, M2, M3``."""
        v = self(ops) if values is None else _check_len(values, self.labels)
        rows = tuple(f"{p.name}:{loc}" for p in self.piers for loc in ("bottom", "top"))
        return StoryTable(rows, PIER_FORCE_COMPONENTS, v.reshape(-1, 6))


def pier_forces(
    piers: Sequence[Pier],
    *,
    ops: ModuleType | None = None,
    tol: float | None = None,
) -> StoryTable:
    """One-off :class:`PierForces` table at the current domain state."""
    return PierForces(piers, ops=ops, tol=tol).table(ops=ops)


# ---------------------------------------------------------------------------
# Story masses
# ---------------------------------------------------------------------------


def _element_translational_mass(ops: ModuleType, dom: _Domain) -> FloatArray:
    """``(N, 3)`` translational mass each node receives from the elements.

    Row ``d`` of ``M e_d`` per element, from ``eleDynamicalForce``
    (``getResistingForceIncInertia``) at the trial acceleration plus a unit
    acceleration in ``d`` less the same at the committed one, so damping
    and stiffness forces cancel. Exact for lumped mass; the row sums of a
    consistent mass. The trial accelerations end at the committed ones.
    """
    out = np.zeros((dom.node_tags.size, 3))
    if dom.ele_tags.size == 0:
        return out
    acc0 = {int(r): np.asarray(ops.nodeAccel(int(dom.node_tags[r])), dtype=np.float64)
            for r in range(dom.node_tags.size)}

    def _set(d: int, bump: float) -> None:
        for r, a in acc0.items():
            if a.size > d:
                ops.setNodeAccel(int(dom.node_tags[r]), d + 1, float(a[d]) + bump)

    def _forces() -> list[FloatArray]:
        return [np.asarray(ops.eleDynamicalForce(int(t)), dtype=np.float64).ravel()
                for t in dom.ele_tags]

    try:
        _set(0, 0.0)
        f0 = _forces()
        for d in range(3):
            _set(d, 1.0)
            f1 = _forces()
            _set(d, 0.0)
            for k, rows in enumerate(dom.ele_rows):
                nd = dom.ndf[rows]
                if f1[k].size != int(nd.sum()) or f0[k].size != f1[k].size:
                    raise BridgeError(
                        f"story_masses: element {int(dom.ele_tags[k])} "
                        f"({dom.ele_type(ops, int(dom.ele_tags[k]))}) answered "
                        f"eleDynamicalForce with {f1[k].size} values; its "
                        f"nodes carry {int(nd.sum())} DOFs."
                    )
                off = np.concatenate([[0], np.cumsum(nd)[:-1]])
                ok = nd > d
                np.add.at(out[:, d], rows[ok], (f1[k] - f0[k])[off[ok] + d])
    finally:
        _set(0, 0.0)
    return out


def story_masses(
    levels: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    include_elements: bool = True,
    ops: ModuleType | None = None,
    tol: float | None = None,
) -> StoryTable:
    """Translational mass and center of mass per level (ETABS *Mass
    Summary by Story* / *Centers of Mass*).

    Rows are the level names (the base first); columns ``MassX, MassY,
    MassZ, XCM, YCM``. A node's mass is the nodal ``mass`` (``nodeMass``)
    plus, with ``include_elements``, what the elements lump to it (see
    :func:`_element_translational_mass`: element ``rho``, frame mass per
    length, shell ``rho * h``, whatever the build defines). A node on a
    level goes to that level; a node between two levels is split between
    them linearly in height (a lumped story element's mass goes half to
    each level); a node at or below the base goes to the base row, one
    above the top level to the top row. ``XCM, YCM`` weigh the horizontal
    mass ``(MassX + MassY) / 2``. ``modalProperties`` counts only free
    DOFs, so with a fixed base its total mass is the sum of the rows above
    the base.
    """
    where = "story_masses"
    z = _check_levels(levels, where)
    level_names = _level_names(names, z.size, where)
    module = _resolve_ops(ops)
    dom = _capture(module, tol, where)
    m = np.zeros((dom.node_tags.size, 3))
    for r, t in enumerate(dom.node_tags):
        nm = np.asarray(module.nodeMass(int(t)), dtype=np.float64).ravel()
        k = min(3, nm.size)
        m[r, :k] = nm[:k]
    if include_elements:
        m += _element_translational_mass(module, dom)
    nz = dom.coords[:, 2]
    w = np.zeros((dom.node_tags.size, z.size))     # node -> level weights
    for r in range(dom.node_tags.size):
        zr = nz[r]
        if zr <= z[0] + dom.tol:
            w[r, 0] = 1.0
        elif zr >= z[-1] - dom.tol:
            w[r, -1] = 1.0
        else:
            i = int(np.searchsorted(z, zr))
            if abs(zr - z[i - 1]) <= dom.tol:
                w[r, i - 1] = 1.0
            elif abs(zr - z[i]) <= dom.tol:
                w[r, i] = 1.0
            else:
                t = (zr - z[i - 1]) / (z[i] - z[i - 1])
                w[r, i - 1] = 1.0 - t
                w[r, i] = t
    mass = w.T @ m                                    # (L, 3)
    mh = 0.5 * (m[:, 0] + m[:, 1])
    wh = w.T @ mh
    with np.errstate(invalid="ignore", divide="ignore"):
        xcm = np.where(wh > 0.0, (w.T @ (mh * dom.coords[:, 0])) / wh, np.nan)
        ycm = np.where(wh > 0.0, (w.T @ (mh * dom.coords[:, 1])) / wh, np.nan)
    return StoryTable(level_names, STORY_MASS_COLUMNS,
                      np.column_stack([mass, xcm, ycm]))


# ---------------------------------------------------------------------------
# Levels from physical groups
# ---------------------------------------------------------------------------


def levels_from_pgs(
    fem: "FEMData",
    pgs: Sequence[str],
    *,
    tol: float | None = None,
) -> tuple[float, ...]:
    """The elevation of each physical group in ``pgs`` (one per level).

    Every node of a group must lie at one elevation within ``tol``
    (default ``1e-6`` of the model's size); the result keeps ``pgs``'s
    order, which must be strictly increasing.
    """
    where = "levels_from_pgs"
    if tol is None:
        all_xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
        span = float(np.linalg.norm(all_xyz.max(axis=0) - all_xyz.min(axis=0)))
        tol = 1e-6 * max(span, 1.0)
    out: list[float] = []
    for name in pgs:
        xyz = np.asarray(fem.nodes.select(pg=name).coords, dtype=np.float64)
        if xyz.size == 0:
            raise ValueError(f"{where}: physical group {name!r} has no nodes.")
        zs = xyz.reshape(-1, 3)[:, 2]
        if float(zs.max() - zs.min()) > tol:
            raise ValueError(
                f"{where}: physical group {name!r} spans z {zs.min():g} to "
                f"{zs.max():g}; a level group must be horizontal (tol {tol:g})."
            )
        out.append(float(zs.mean()))
    _check_levels(out, where)
    return tuple(out)
