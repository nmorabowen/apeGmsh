"""
Selection — the interactive viewer pick-result type for apeGmsh.

This is the frozen ``DimTag`` snapshot exposed by an interactive
viewer's ``.selection`` property (``g.model.viewer()`` /
``g.mesh.viewer()`` → ``ModelViewer.selection`` /
``MeshViewer.selection``), reflecting what was picked in the viewport.
It is retained **by architecture** as the viewer pick-result type (it
is *not* a ``g.model.select`` migration path — for programmatic entity
selection use ``g.model.select(...)`` which returns an
:class:`~apeGmsh.core._selection.EntitySelection`).

A picked :class:`Selection` supports:

* set operations  — ``|``, ``&``, ``-``, ``^``
* refinement     — ``.filter(**kw)``, ``.limit(n)``, ``.sorted_by(...)``
* geometry       — ``.bbox()``, ``.centers()``, ``.masses()``
* conversion     — ``.to_list()``, ``.to_tags()``, ``.to_dataframe()``,
                   ``.to_physical(name, tag=-1)``

Typical usage after an interactive pick::

    v = g.model.viewer()
    # ... pick entities in the viewer ...
    sel = v.selection
    cols = sel.filter(vertical=True)
    cols.to_physical("columns")
"""

from __future__ import annotations

import fnmatch
import math
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import gmsh
import numpy as np
import pandas as pd

from apeGmsh._logging import _HasLogging
from apeGmsh._types import Tag, DimTag

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _SessionBase
    from apeGmsh.core.Model import Model

BBox    = tuple[float, float, float, float, float, float]

_DIM_PREFIX = {0: 'P', 1: 'C', 2: 'S', 3: 'V'}
_DIM_NAME   = {0: 'points', 1: 'curves', 2: 'surfaces', 3: 'volumes'}
_AXIS_IDX   = {'x': 0, 'y': 1, 'z': 2}


_FILTER_DOC_COMMON = """
Filter families (all keyword-only, default ``None``; combined with AND).

Identity:

* ``tags=[t, ...]`` — keep entities with these tags.
* ``exclude_tags=[t, ...]`` — drop entities with these tags.
* ``labels="glob" | [...]`` — keep entities whose label matches (fnmatch).
* ``kinds="box" | [...]`` — keep entities by registered kind.
* ``physical="name" | tag`` — keep entities in this physical group.

Spatial:

* ``in_box=(x0, y0, z0, x1, y1, z1)`` — inside an axis-aligned bbox.
* ``in_sphere=(cx, cy, cz, r)`` — centroid within radius ``r``.
* ``on_plane=(axis, value, tol)`` — bbox intersects ``axis=value`` within
  ``tol`` (``axis`` is ``"x"``, ``"y"``, or ``"z"``).
* ``on_axis=(axis, tol)`` — centroid lies on the given coordinate axis
  (off-axis components within ``tol``).
* ``at_point=(x, y, z, tol)`` — bbox contains ``(x, y, z)`` within ``tol``.

Metrics / orientation (dim-specific; silently skipped on other dims):

* ``length_range=(lo, hi)`` — curves (dim=1).
* ``area_range=(lo, hi)`` — surfaces (dim=2).
* ``volume_range=(lo, hi)`` — volumes (dim=3).
* ``aligned=(axis, atol_deg)`` — curves within ``atol_deg`` of the axis.
* ``horizontal=True`` — curves perpendicular to ``z``.
* ``vertical=True`` — curves parallel to ``z``.

Escape hatch:

* ``predicate=fn`` — callable ``(dim, tag) -> bool``.

Example::

    sel = v.selection.filter(
        vertical=True, length_range=(2.0, 4.0),
    ).filter(physical="columns")
"""


# ===========================================================================
# Selection — immutable result object
# ===========================================================================

class Selection:
    """
    Frozen snapshot of a set of Gmsh ``DimTag``s with set-algebra,
    refinement, and conversion helpers.

    ``Selection`` objects are exposed by an interactive viewer's
    ``.selection`` property (``g.model.viewer()`` /
    ``g.mesh.viewer()``) — you do not normally instantiate them
    directly.

    Attributes
    ----------
    dim : int
        Common dimension of all entries, or ``-1`` if mixed (only
        possible after cross-dim set operations).
    dimtags : tuple[DimTag, ...]
        Frozen tuple of (dim, tag) pairs.
    """

    __slots__ = ('_dimtags', '_dim', '_parent')

    _dimtags: tuple[DimTag, ...]
    _dim: int
    _parent: _SessionBase

    def __init__(
        self,
        dimtags : Iterable[DimTag],
        parent: _SessionBase,
    ) -> None:
        # Deduplicate while preserving order
        seen: set[DimTag] = set()
        uniq: list[DimTag] = []
        for dt in dimtags:
            t = (int(dt[0]), int(dt[1]))
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        object.__setattr__(self, '_dimtags', tuple(uniq))
        dims = {d for d, _ in uniq}
        object.__setattr__(self, '_dim', dims.pop() if len(dims) == 1 else -1)
        object.__setattr__(self, '_parent', parent)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dimtags(self) -> tuple[DimTag, ...]:
        return self._dimtags

    @property
    def tags(self) -> tuple[Tag, ...]:
        """Tuple of entity tags (requires homogeneous dim)."""
        if self._dim == -1:
            raise ValueError(
                "Selection spans multiple dims — use .dimtags instead."
            )
        return tuple(t for _, t in self._dimtags)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dimtags)

    def __iter__(self):
        return iter(self._dimtags)

    def __bool__(self) -> bool:
        return bool(self._dimtags)

    def __contains__(self, dt: DimTag) -> bool:
        return (int(dt[0]), int(dt[1])) in self._dimtags

    def __repr__(self) -> str:
        n = len(self._dimtags)
        if self._dim == -1:
            return f"<Selection mixed n={n}>"
        prefix = _DIM_PREFIX.get(self._dim, '?')
        preview = ", ".join(f"{prefix}{t}" for _, t in self._dimtags[:6])
        more = f", …(+{n - 6})" if n > 6 else ""
        return f"<Selection dim={self._dim} n={n} [{preview}{more}]>"

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def _combine(self, other: Selection, op: str) -> Selection:
        if not isinstance(other, Selection):
            return NotImplemented
        a = set(self._dimtags)
        b = set(other._dimtags)
        if op == 'or':
            result = a | b
        elif op == 'and':
            result = a & b
        elif op == 'sub':
            result = a - b
        elif op == 'xor':
            result = a ^ b
        else:
            raise ValueError(op)
        # Preserve order from self, then other
        ordered = [dt for dt in self._dimtags if dt in result]
        ordered += [dt for dt in other._dimtags if dt in result and dt not in set(ordered)]
        return Selection(ordered, self._parent)

    def __or__(self, other: Selection) -> Selection:
        return self._combine(other, 'or')

    def __and__(self, other: Selection) -> Selection:
        return self._combine(other, 'and')

    def __sub__(self, other: Selection) -> Selection:
        return self._combine(other, 'sub')

    def __xor__(self, other: Selection) -> Selection:
        return self._combine(other, 'xor')

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def filter(
        self,
        *,
        tags           : Sequence[Tag] | None = None,
        exclude_tags   : Sequence[Tag] | None = None,
        labels         : str | Sequence[str] | None = None,
        kinds          : str | Sequence[str] | None = None,
        physical       : str | Tag | None     = None,
        in_box         : BBox | None          = None,
        in_sphere      : tuple[float, float, float, float] | None = None,
        on_plane       : tuple[str, float, float] | None = None,
        on_axis        : tuple[str, float] | None = None,
        at_point       : tuple[float, float, float, float] | None = None,
        length_range   : tuple[float, float] | None = None,
        area_range     : tuple[float, float] | None = None,
        volume_range   : tuple[float, float] | None = None,
        aligned        : tuple[str, float] | None = None,
        horizontal     : bool | None          = None,
        vertical       : bool | None          = None,
        predicate      : Callable[[int, int], bool] | None = None,
    ) -> Selection:
        """Re-apply filters to this selection."""
        if self._dim == -1:
            raise ValueError("Cannot filter a mixed-dim selection.")
        filtered = _apply_filters(
            list(self._dimtags), dim=self._dim, parent=self._parent,
            tags=tags, exclude_tags=exclude_tags, labels=labels,
            kinds=kinds, physical=physical, in_box=in_box,
            in_sphere=in_sphere, on_plane=on_plane, on_axis=on_axis,
            at_point=at_point, length_range=length_range,
            area_range=area_range, volume_range=volume_range,
            aligned=aligned, horizontal=horizontal, vertical=vertical,
            predicate=predicate,
        )
        return Selection(filtered, self._parent)
    filter.__doc__ = (filter.__doc__ or "") + "\n" + _FILTER_DOC_COMMON

    def limit(self, n: int) -> Selection:
        """Keep at most *n* entries."""
        return Selection(self._dimtags[:n], self._parent)

    def sorted_by(
        self,
        key: str | Callable[[DimTag], float] = "x",
    ) -> Selection:
        """
        Sort by a coordinate (``"x"``, ``"y"``, ``"z"``), a metric
        (``"length"``, ``"area"``, ``"volume"``, ``"mass"``), or a
        callable ``(dim, tag) -> float``.
        """
        if self._dim == -1:
            raise ValueError("Cannot sort a mixed-dim selection.")

        if callable(key):
            scored = [(key(dt), dt) for dt in self._dimtags]
        elif key in _AXIS_IDX:
            idx = _AXIS_IDX[key]
            centers = self.centers()
            scored = list(zip(centers[:, idx].tolist(), self._dimtags))
        elif key in ("length", "area", "volume", "mass"):
            masses = self.masses()
            scored = list(zip(masses.tolist(), self._dimtags))
        else:
            raise ValueError(f"Unknown sort key: {key!r}")
        scored.sort(key=lambda pair: pair[0])
        return Selection([dt for _, dt in scored], self._parent)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def bbox(self) -> BBox:
        """Return the axis-aligned bounding box of the union."""
        if not self._dimtags:
            raise ValueError("Empty selection — no bounding box.")
        xmins, ymins, zmins = [], [], []
        xmaxs, ymaxs, zmaxs = [], [], []
        for d, t in self._dimtags:
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(d, t)
            xmins.append(x0)
            ymins.append(y0)
            zmins.append(z0)
            xmaxs.append(x1)
            ymaxs.append(y1)
            zmaxs.append(z1)
        return (min(xmins), min(ymins), min(zmins),
                max(xmaxs), max(ymaxs), max(zmaxs))

    def centers(self) -> np.ndarray:
        """(N, 3) array of entity centroids."""
        pts = np.empty((len(self._dimtags), 3))
        for i, (d, t) in enumerate(self._dimtags):
            pts[i] = _entity_center(d, t)
        return pts

    def masses(self) -> np.ndarray:
        """(N,) array of length/area/volume values (via ``occ.getMass``)."""
        out = np.empty(len(self._dimtags))
        for i, (d, t) in enumerate(self._dimtags):
            if d == 0:
                out[i] = 0.0
            else:
                try:
                    out[i] = gmsh.model.occ.getMass(d, t)
                except Exception:
                    out[i] = float('nan')
        return out

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_list(self) -> list[DimTag]:
        """Return a plain list of ``(dim, tag)`` tuples."""
        return list(self._dimtags)

    def to_tags(self) -> list[Tag]:
        """Return a plain list of tags (requires homogeneous dim)."""
        return list(self.tags)

    def to_physical(self, name: str, *, tag: Tag = -1) -> Tag:
        """
        Promote this selection to a physical group.

        Parameters
        ----------
        name : physical-group name
        tag  : requested physical-group tag (``-1`` = auto-assign)

        Returns
        -------
        Tag the physical-group tag assigned by Gmsh.
        """
        if self._dim == -1:
            raise ValueError(
                "to_physical requires a homogeneous-dim selection."
            )
        if not self._dimtags:
            raise ValueError("Cannot promote an empty selection to a physical group.")
        return self._parent.physical.add(
            self._dim, list(self.tags), name=name, tag=tag,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns
        ``dim, tag, kind, label, x, y, z, mass``.

        ``kind`` is read from ``Model._metadata`` (entity metadata).
        ``label`` is read from ``g.labels`` (the single source of truth).
        """
        reg = self._parent.model._metadata
        # Build label reverse map from g.labels
        labels_comp = getattr(self._parent, 'labels', None)
        label_map: dict[tuple[int, int], str] = {}
        if labels_comp is not None:
            try:
                label_map = labels_comp.reverse_map()
            except Exception:
                pass
        centers = self.centers() if self._dimtags else np.empty((0, 3))
        masses  = self.masses()  if self._dimtags else np.empty((0,))
        rows = []
        for i, (d, t) in enumerate(self._dimtags):
            info = reg.get((d, t), {})
            rows.append({
                'dim'   : d,
                'tag'   : t,
                'kind'  : info.get('kind'),
                'label' : label_map.get((d, t), ''),
                'x'     : centers[i, 0],
                'y'     : centers[i, 1],
                'z'     : centers[i, 2],
                'mass'  : masses[i],
            })
        return pd.DataFrame(
            rows,
            columns=['dim', 'tag', 'kind', 'label', 'x', 'y', 'z', 'mass'],
        )


# ===========================================================================
# Filter engine
# ===========================================================================

def _apply_filters(
    dimtags : list[DimTag],
    *,
    dim     : int,
    parent: _SessionBase,
    # — filters —
    tags           : Sequence[Tag] | None = None,
    exclude_tags   : Sequence[Tag] | None = None,
    labels         : str | Sequence[str] | None = None,
    kinds          : str | Sequence[str] | None = None,
    physical       : str | Tag | None     = None,
    in_box         : BBox | None          = None,
    in_sphere      : tuple[float, float, float, float] | None = None,
    on_plane       : tuple[str, float, float] | None = None,
    on_axis        : tuple[str, float] | None = None,
    at_point       : tuple[float, float, float, float] | None = None,
    length_range   : tuple[float, float] | None = None,
    area_range     : tuple[float, float] | None = None,
    volume_range   : tuple[float, float] | None = None,
    aligned        : tuple[str, float] | None = None,
    horizontal     : bool | None          = None,
    vertical       : bool | None          = None,
    predicate      : Callable[[int, int], bool] | None = None,
) -> list[DimTag]:
    """
    Apply a battery of filters (AND-combined) to a list of DimTags.
    """
    if not dimtags:
        return []

    out = list(dimtags)

    # ---- identity filters -------------------------------------------
    out = _filter_by_identity(
        out, dim, parent, tags=tags, exclude_tags=exclude_tags,
        labels=labels, kinds=kinds, physical=physical,
    )

    # ---- spatial filters --------------------------------------------
    out = _filter_by_spatial(
        out, dim, in_box=in_box, in_sphere=in_sphere,
        on_plane=on_plane, on_axis=on_axis, at_point=at_point,
    )

    # ---- metric / orientation filters --------------------------------
    out = _filter_by_metrics(
        out, dim, length_range=length_range, area_range=area_range,
        volume_range=volume_range, horizontal=horizontal,
        vertical=vertical, aligned=aligned,
    )

    # ---- predicate escape hatch --------------------------------------
    if predicate is not None:
        out = [dt for dt in out if predicate(*dt)]

    return out


def _filter_by_identity(
    out: list[DimTag],
    dim: int,
    parent: _SessionBase,
    *,
    tags: Sequence[Tag] | None,
    exclude_tags: Sequence[Tag] | None,
    labels: str | Sequence[str] | None,
    kinds: str | Sequence[str] | None,
    physical: str | Tag | None,
) -> list[DimTag]:
    """Tag, label, kind, and physical-group membership filters."""
    if tags is not None:
        keep = set(int(t) for t in tags)
        out = [dt for dt in out if dt[1] in keep]
    if exclude_tags is not None:
        drop = set(int(t) for t in exclude_tags)
        out = [dt for dt in out if dt[1] not in drop]

    if labels is not None:
        patterns = [labels] if isinstance(labels, str) else list(labels)
        labels_comp = getattr(parent, 'labels', None)
        label_map: dict[DimTag, str] = {}
        if labels_comp is not None:
            try:
                label_map = labels_comp.reverse_map()
            except Exception:
                pass
        out = [dt for dt in out
               if any(fnmatch.fnmatch(label_map.get(dt, ''), p)
                      for p in patterns)]

    if kinds is not None:
        wanted = {kinds} if isinstance(kinds, str) else set(kinds)
        reg = parent.model._metadata
        out = [dt for dt in out
               if reg.get(dt, {}).get('kind') in wanted]

    if physical is not None:
        keep_dt = set(_entities_of_physical(physical, dim))
        out = [dt for dt in out if dt in keep_dt]

    return out


def _filter_by_spatial(
    out: list[DimTag],
    dim: int,
    *,
    in_box: BBox | None,
    in_sphere: tuple[float, float, float, float] | None,
    on_plane: tuple[str, float, float] | None,
    on_axis: tuple[str, float] | None,
    at_point: tuple[float, float, float, float] | None,
) -> list[DimTag]:
    """Bounding-box, sphere, plane, axis, and point proximity filters."""
    if in_box is not None:
        x0, y0, z0, x1, y1, z1 = in_box
        in_box_set = set(
            gmsh.model.getEntitiesInBoundingBox(x0, y0, z0, x1, y1, z1, dim=dim)
        )
        out = [dt for dt in out if dt in in_box_set]

    if in_sphere is not None:
        cx, cy, cz, r = in_sphere
        r2 = r * r
        center = np.array([cx, cy, cz])
        out = [dt for dt in out
               if float(((_entity_center(*dt) - center) ** 2).sum()) <= r2]

    if on_plane is not None:
        axis, val, atol = on_plane
        i = _AXIS_IDX[axis.lower()]
        def _on_plane(dt: DimTag) -> bool:
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(*dt)
            return ((x0, y0, z0)[i] - atol) <= val <= ((x1, y1, z1)[i] + atol)
        out = [dt for dt in out if _on_plane(dt)]

    if on_axis is not None:
        axis, atol = on_axis
        i = _AXIS_IDX[axis.lower()]
        out = [dt for dt in out
               if all(abs(_entity_center(*dt)[j]) <= atol
                      for j in range(3) if j != i)]

    if at_point is not None:
        px, py, pz, atol = at_point
        def _at_point(dt: DimTag) -> bool:
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(*dt)
            return (x0 - atol <= px <= x1 + atol and
                    y0 - atol <= py <= y1 + atol and
                    z0 - atol <= pz <= z1 + atol)
        out = [dt for dt in out if _at_point(dt)]

    return out


def _filter_by_metrics(
    out: list[DimTag],
    dim: int,
    *,
    length_range: tuple[float, float] | None,
    area_range: tuple[float, float] | None,
    volume_range: tuple[float, float] | None,
    horizontal: bool | None,
    vertical: bool | None,
    aligned: tuple[str, float] | None,
) -> list[DimTag]:
    """Size range and orientation filters."""
    if length_range is not None and dim == 1:
        lo, hi = length_range
        out = [dt for dt in out if lo <= _safe_mass(*dt) <= hi]
    if area_range is not None and dim == 2:
        lo, hi = area_range
        out = [dt for dt in out if lo <= _safe_mass(*dt) <= hi]
    if volume_range is not None and dim == 3:
        lo, hi = volume_range
        out = [dt for dt in out if lo <= _safe_mass(*dt) <= hi]

    if dim == 1:
        if horizontal is True:
            out = [dt for dt in out if _is_axis_aligned(dt, "z",
                                                        perpendicular=True)]
        if vertical is True:
            out = [dt for dt in out if _is_axis_aligned(dt, "z",
                                                        perpendicular=False)]
        if aligned is not None:
            if isinstance(aligned, str):
                axis_name, atol_deg = aligned, 5.0
            else:
                axis_name, atol_deg = aligned
            out = [dt for dt in out
                   if _is_axis_aligned(dt, axis_name, atol_deg=atol_deg)]

    return out


# ===========================================================================
# Low-level helpers
# ===========================================================================

def _entity_center(dim: int, tag: Tag) -> np.ndarray:
    """Return the geometric centroid of a BRep entity as a length-3 array."""
    try:
        if dim == 0:
            return np.asarray(gmsh.model.getValue(0, tag, []), dtype=float)
        if dim == 3:
            return np.asarray(gmsh.model.occ.getCenterOfMass(3, tag),
                              dtype=float)
        # dim == 1 or 2 -> parametric midpoint
        bounds = gmsh.model.getParametrizationBounds(dim, tag)
        mids = [0.5 * (bounds[0][i] + bounds[1][i]) for i in range(dim)]
        return np.asarray(gmsh.model.getValue(dim, tag, mids), dtype=float)
    except Exception:
        # fallback to bounding-box centre
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
        return np.array([0.5*(x0+x1), 0.5*(y0+y1), 0.5*(z0+z1)])


def _safe_mass(dim: int, tag: Tag) -> float:
    try:
        return float(gmsh.model.occ.getMass(dim, tag))
    except Exception:
        return float('nan')


def _is_axis_aligned(
    dt: DimTag,
    axis: str,
    *,
    perpendicular: bool = False,
    atol_deg     : float = 1.0,
) -> bool:
    """
    Test whether a *curve* (dim=1) is aligned with a coordinate axis.

    ``perpendicular=False`` -> curve is parallel to ``axis``.
    ``perpendicular=True``  -> curve is perpendicular to ``axis`` (i.e.
                              lies in the plane normal to that axis).
    """
    d, t = dt
    if d != 1:
        return False
    i = _AXIS_IDX[axis.lower()]
    # use bounding-box extents as a proxy for curve direction —
    # works for straight-line curves which is the common case
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(d, t)
    extents = np.array([x1 - x0, y1 - y0, z1 - z0])
    total = float(np.linalg.norm(extents))
    if total < 1e-12:
        return False
    along_axis = abs(extents[i]) / total  # 1 -> parallel, 0 -> perpendicular
    # angle between curve direction and axis
    ang_deg = math.degrees(math.acos(min(1.0, along_axis)))
    if perpendicular:
        # curve is perpendicular to axis when its axis-component is ~0
        return along_axis <= math.sin(math.radians(atol_deg))
    else:
        return ang_deg <= atol_deg


def _entities_of_physical(
    physical : str | Tag,
    dim      : int,
) -> list[DimTag]:
    """Return all DimTags belonging to a physical group (by name or tag)."""
    # Find the physical-group tag
    if isinstance(physical, str):
        pg_tag = None
        for d, t in gmsh.model.getPhysicalGroups(dim=dim):
            try:
                if gmsh.model.getPhysicalName(d, t) == physical:
                    pg_tag = t
                    break
            except Exception:
                continue
        if pg_tag is None:
            return []
    else:
        pg_tag = int(physical)
    ents = gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag)
    return [(dim, int(t)) for t in ents]