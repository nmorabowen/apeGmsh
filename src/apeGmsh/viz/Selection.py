"""
Selection — model-entity selection composite for apeGmsh.

Exposed as ``g.model.selection``.  Provides five query entry points
(``select_points``, ``select_curves``, ``select_surfaces``,
``select_volumes``, ``select_all``) plus three topology helpers
(``boundary_of``, ``adjacent_to``, ``closest_to``).  Each returns a
frozen :class:`Selection` snapshot of ``DimTag``s.

The resulting :class:`Selection` supports:

* set operations  — ``|``, ``&``, ``-``, ``^``
* refinement     — ``.filter(**kw)``, ``.limit(n)``, ``.sorted_by(...)``
* geometry       — ``.bbox()``, ``.centers()``, ``.masses()``
* conversion     — ``.to_list()``, ``.to_tags()``, ``.to_dataframe()``,
                   ``.to_physical(name, tag=-1)``

Typical usage after an IGES import::

    sel = g.model.selection
    cols = sel.select_curves(vertical=True)
    cols.to_physical("columns")
    beams = sel.select_curves(horizontal=True)
    beams.to_physical("beams")
    base = sel.select_points(on_plane=("z", 0.0, 1e-3))
    base.to_physical("fixed_support")
"""

from __future__ import annotations

import fnmatch
import math
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import gmsh
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase
    from apeGmsh.core.Model import Model


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tag     = int
DimTag  = tuple[int, int]
BBox    = tuple[float, float, float, float, float, float]

_DIM_PREFIX = {0: 'P', 1: 'C', 2: 'S', 3: 'V'}
_DIM_NAME   = {0: 'points', 1: 'curves', 2: 'surfaces', 3: 'volumes'}
_AXIS_IDX   = {'x': 0, 'y': 1, 'z': 2}


# ===========================================================================
# Selection — immutable result object
# ===========================================================================

class Selection:
    """
    Frozen snapshot of a set of Gmsh ``DimTag``s with set-algebra,
    refinement, and conversion helpers.

    ``Selection`` objects are returned by the query methods on the
    :class:`SelectionComposite` (``g.model.selection``) — you do not
    normally instantiate them directly.

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

    def filter(self, **kwargs) -> Selection:
        """Re-apply filters to this selection."""
        if self._dim == -1:
            raise ValueError("Cannot filter a mixed-dim selection.")
        filtered = _apply_filters(
            list(self._dimtags), dim=self._dim,
            parent=self._parent, **kwargs,
        )
        return Selection(filtered, self._parent)

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
            xmins.append(x0); ymins.append(y0); zmins.append(z0)
            xmaxs.append(x1); ymaxs.append(y1); zmaxs.append(z1)
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

    # ------------------------------------------------------------------
    # Bridges to mesh data
    # ------------------------------------------------------------------

    def to_mesh_nodes(self) -> dict:
        """Return mesh node data for the entities in this selection.

        Requires the mesh to have been generated.  Combines nodes from
        all entities in the selection (deduplicated, sorted by tag).

        Returns
        -------
        dict
            ``'tags'``   : ndarray(N,) — global node tags
            ``'coords'`` : ndarray(N, 3) — XYZ coordinates

        Raises
        ------
        RuntimeError
            If the mesh has not been generated.

        Example
        -------
        ::

            top = g.model.selection.select_surfaces(on_plane=("z", 10))
            nodes = top.to_mesh_nodes()
            # → {'tags': [12, 18, 22, ...], 'coords': [[...], ...]}
        """
        if not self._dimtags:
            return {'tags': np.array([], dtype=np.int64),
                    'coords': np.empty((0, 3), dtype=np.float64)}

        all_tags: list[int] = []
        all_coords: list = []
        seen: set[int] = set()
        for d, t in self._dimtags:
            try:
                ntags, ncoords, _ = gmsh.model.mesh.getNodes(d, t, includeBoundary=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to retrieve mesh nodes for ({d}, {t}). "
                    f"Has the mesh been generated? Original error: {exc}"
                )
            ncoords = np.asarray(ncoords).reshape(-1, 3)
            for i, nt in enumerate(ntags):
                nt = int(nt)
                if nt not in seen:
                    seen.add(nt)
                    all_tags.append(nt)
                    all_coords.append(ncoords[i])

        order = np.argsort(all_tags)
        return {
            'tags':   np.asarray(all_tags, dtype=np.int64)[order],
            'coords': np.asarray(all_coords, dtype=np.float64)[order],
        }

    def to_mesh_elements(self) -> dict:
        """Return mesh element data for the entities in this selection.

        Combines elements from all entities at this selection's dim.
        Only available for homogeneous-dim selections (dim >= 1).

        Returns
        -------
        dict
            ``'element_ids'``  : ndarray(E,)
            ``'connectivity'`` : ndarray(E, npe)

        Raises
        ------
        ValueError
            If the selection is mixed-dim or dim==0 (points have no elements).
        RuntimeError
            If the mesh has not been generated.
        """
        if self._dim == -1:
            raise ValueError(
                "to_mesh_elements requires a homogeneous-dim selection."
            )
        if self._dim == 0:
            raise ValueError("Points (dim=0) have no elements.")

        eids: list[int] = []
        conn_blocks: list[np.ndarray] = []
        for d, t in self._dimtags:
            try:
                etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to retrieve elements for ({d}, {t}). "
                    f"Has the mesh been generated? Original error: {exc}"
                )
            for etags, enodes in zip(etags_list, enodes_list):
                if len(etags) == 0:
                    continue
                npe = len(enodes) // len(etags)
                conn = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                eids.extend(int(e) for e in etags)
                conn_blocks.append(conn)

        if not conn_blocks:
            return {
                'element_ids': np.array([], dtype=np.int64),
                'connectivity': np.empty((0, 0), dtype=np.int64),
            }
        # Pad if mixed npe — use the largest, fill with -1 sentinel
        max_npe = max(b.shape[1] for b in conn_blocks)
        if all(b.shape[1] == max_npe for b in conn_blocks):
            conn = np.vstack(conn_blocks)
        else:
            padded = []
            for b in conn_blocks:
                if b.shape[1] < max_npe:
                    pad = np.full((b.shape[0], max_npe - b.shape[1]), -1, dtype=np.int64)
                    b = np.hstack([b, pad])
                padded.append(b)
            conn = np.vstack(padded)
        return {
            'element_ids':  np.asarray(eids, dtype=np.int64),
            'connectivity': conn,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns
        ``dim, tag, kind, label, x, y, z, mass``.

        ``kind`` / ``label`` are read from ``Model._registry`` when
        available; otherwise they are ``None`` / ``""``.
        """
        reg = self._parent.model._registry
        centers = self.centers() if self._dimtags else np.empty((0, 3))
        masses  = self.masses()  if self._dimtags else np.empty((0,))
        rows = []
        for i, (d, t) in enumerate(self._dimtags):
            info = reg.get((d, t), {})
            rows.append({
                'dim'   : d,
                'tag'   : t,
                'kind'  : info.get('kind'),
                'label' : info.get('label', ''),
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
# SelectionComposite — attached to Model as `model.selection`
# ===========================================================================

class SelectionComposite:
    """
    Query-entry composite attached to :class:`Model` as
    ``g.model.selection``.

    All query methods return a :class:`Selection` snapshot.  They are
    *not* chainable on this composite (the result is a value, not a
    builder) — but the returned :class:`Selection` *is* chainable:

        g.model.selection.select_curves(vertical=True).filter(length_range=(2,4))

    Requires that the geometry be synchronised
    (``g.model.sync()``) before calling — OCC topology queries read
    the synced kernel state.
    """

    def __init__(self, parent: _SessionBase, model: Model) -> None:
        self._parent = parent
        self._model  = model

    def picker(
        self,
        physical_group: str | None = None,
        *,
        dims: list[int] | None = None,
        **kwargs,
    ):
        """
        Open an interactive Qt viewer (pyvistaqt + VTK) for this model's BRep
        and manage physical groups from the UI.

        Typical usage
        -------------
        Call with no arguments — create, rename, delete, and populate
        physical groups entirely from the toolbar and tree, then close
        the window::

            m.model.selection.picker()

        *Inside the viewer:* click **New Group** on the toolbar to start
        a group, pick entities (or LMB-drag to box-select), repeat for
        more groups, use **Rename** / **Delete** or right-click a group
        in the tree for per-group actions.  For Assembly models an
        **Instances** tree root lets you select every entity of one
        instance in a click or create a group from it via right-click.
        Every group staged during the session is written to Gmsh on
        close (existing groups of the same name are replaced).

        Shortcut — edit one existing group's members directly::

            m.model.selection.picker(physical_group="supports")

        When ``physical_group`` is given and the group already exists,
        its members are pre-loaded as the active working set.

        Layout
        ------
        3D viewport on the left; on the right a **model tree**
        (*Physical groups* / *Unassigned* / *Instances*) and a
        **preferences** dock with live sliders for point size, line
        width, surface opacity, edges, AA, dark/light theme.  Toolbar:
        New / Rename / Delete group, ⊥ Parallel ↔ perspective toggle,
        ⤢ Fit view, ? Help.

        Requires ``pyvistaqt`` + a Qt binding (PyQt5/6 or PySide2/6).

        Keyboard shortcuts inside the 3D viewport:
            Pick filter:  [1] points  [2] curves  [3] surfaces
                          [4] volumes  [0] all
            Visibility:   [H] hide picks  [I] isolate picks
                          [R] reveal all
            Edit:         [U] undo    [Tab] cycle overlapping entities
                          [Esc] deselect all    [Q] close window

        Mouse bindings:
            LEFT click         : pick entity (pixel-accurate)
            LEFT drag          : rubber-band box-select
                                 (L→R = window, R→L = crossing)
            Ctrl+LEFT click    : unpick entity under cursor
            Ctrl+LEFT drag     : rubber-band box-UNselect
            MIDDLE drag        : pan camera
            Shift+MIDDLE drag  : rotate camera (orbit)
            RIGHT drag         : pan camera
            WHEEL              : zoom
            hover              : highlight entity under cursor (gold)

        Parameters
        ----------
        physical_group : str or None, optional
            *Shortcut for edit-one-group mode.* When given, pre-loads
            the existing group's members as the active working set so
            you can adjust its membership directly.  Leave as ``None``
            (the default) to create groups from the UI.
        dims : list of int
            BRep dimensions rendered in the picker.  Default
            ``[0, 1, 2, 3]``.  Use e.g. ``dims=[0]`` to show only
            points from the start.
        **kwargs : forwarded to :class:`SelectionPicker`
            (``n_curve_samples``, ``n_surf_samples``, ``point_size``,
            ``line_width``, ``surface_opacity``, ``show_surface_edges``).

        Returns
        -------
        SelectionPicker — with ``.selection``, ``.tags``,
        ``.active_group``, and ``.to_physical(name)`` available after close.
        """
        from apeGmsh.viewers.model_viewer import ModelViewer
        p = ModelViewer(
            parent=self._parent, model=self._model,
            physical_group=physical_group, dims=dims, **kwargs,
        )
        p.show()
        return p

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[Selection] {msg}")

    def _query(self, dim: int, **kwargs) -> Selection:
        gmsh.model.occ.synchronize()
        universe: list[DimTag] = list(gmsh.model.getEntities(dim=dim))
        filtered = _apply_filters(
            universe, dim=dim, parent=self._parent, **kwargs,
        )
        sel = Selection(filtered, self._parent)
        self._log(f"select dim={dim} → {len(sel)} / {len(universe)} entities")
        return sel

    # ------------------------------------------------------------------
    # Per-dim query methods
    # ------------------------------------------------------------------

    def select_points(self, **kwargs) -> Selection:
        """Select BRep points (dim=0) matching all given filters."""
        return self._query(0, **kwargs)

    def select_curves(self, **kwargs) -> Selection:
        """Select BRep curves (dim=1) matching all given filters."""
        return self._query(1, **kwargs)

    def select_surfaces(self, **kwargs) -> Selection:
        """Select BRep surfaces (dim=2) matching all given filters."""
        return self._query(2, **kwargs)

    def select_volumes(self, **kwargs) -> Selection:
        """Select BRep volumes (dim=3) matching all given filters."""
        return self._query(3, **kwargs)

    def select_all(self, dim: int = -1, **kwargs) -> Selection:
        """
        Select entities of any (or given) dimension.

        When ``dim=-1`` returns entities across all dims; filters
        that require a single dim (e.g. ``length_range``) will be
        applied per-dim-slice.
        """
        if dim >= 0:
            return self._query(dim, **kwargs)
        parts: list[DimTag] = []
        for d in (0, 1, 2, 3):
            universe = list(gmsh.model.getEntities(dim=d))
            parts.extend(_apply_filters(
                universe, dim=d, parent=self._parent, **kwargs,
            ))
        return Selection(parts, self._parent)

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def boundary_of(self, sel: Selection, *, combined: bool = False) -> Selection:
        """
        Return the boundary of *sel* as a :class:`Selection`.

        Wraps ``gmsh.model.getBoundary(..., oriented=False)``.
        """
        if not sel:
            return Selection([], self._parent)
        gmsh.model.occ.synchronize()
        bnd = gmsh.model.getBoundary(
            list(sel.dimtags), oriented=False, combined=combined,
        )
        return Selection(bnd, self._parent)

    def adjacent_to(
        self,
        sel: Selection,
        dim_target: int,
    ) -> Selection:
        """
        Entities of ``dim_target`` whose boundary touches any entity
        in *sel*.  Useful for e.g. "all surfaces bounded by these
        curves" (``dim_target=2``) or "all volumes on these surfaces"
        (``dim_target=3``).
        """
        gmsh.model.occ.synchronize()
        src_set = set(sel.dimtags)
        out: list[DimTag] = []
        for d, t in gmsh.model.getEntities(dim=dim_target):
            bnd = gmsh.model.getBoundary(
                [(d, t)], oriented=False, recursive=False,
            )
            if any(b in src_set for b in bnd):
                out.append((d, t))
        return Selection(out, self._parent)

    def closest_to(
        self,
        x: float, y: float, z: float,
        *,
        dim: int = 0,
        n  : int = 1,
    ) -> Selection:
        """
        Return the *n* entities of dimension *dim* whose centroid is
        closest to the point ``(x, y, z)``.
        """
        gmsh.model.occ.synchronize()
        ents = list(gmsh.model.getEntities(dim=dim))
        if not ents:
            return Selection([], self._parent)
        q = np.array([x, y, z])
        scored = []
        for d, t in ents:
            c = _entity_center(d, t)
            scored.append((float(np.linalg.norm(c - q)), (d, t)))
        scored.sort(key=lambda pair: pair[0])
        return Selection([dt for _, dt in scored[:max(n, 0)]], self._parent)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        counts = {
            _DIM_NAME[d]: len(gmsh.model.getEntities(dim=d)) for d in (0, 1, 2, 3)
        }
        return f"SelectionComposite({counts})"


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

    # ---- tag / exclude_tags ------------------------------------------
    if tags is not None:
        keep = set(int(t) for t in tags)
        out = [dt for dt in out if dt[1] in keep]
    if exclude_tags is not None:
        drop = set(int(t) for t in exclude_tags)
        out = [dt for dt in out if dt[1] not in drop]

    # ---- labels (glob over Model._registry labels) -------------------
    if labels is not None:
        patterns = [labels] if isinstance(labels, str) else list(labels)
        reg = parent.model._registry
        def _match_label(dt: DimTag) -> bool:
            info = reg.get(dt)
            if not info:
                return False
            lbl = info.get('label', '')
            return any(fnmatch.fnmatch(lbl, p) for p in patterns)
        out = [dt for dt in out if _match_label(dt)]

    # ---- kinds -------------------------------------------------------
    if kinds is not None:
        wanted = {kinds} if isinstance(kinds, str) else set(kinds)
        reg = parent.model._registry
        out = [dt for dt in out
               if reg.get(dt, {}).get('kind') in wanted]

    # ---- physical-group membership -----------------------------------
    if physical is not None:
        pg_dimtags = _entities_of_physical(physical, dim)
        keep = set(pg_dimtags)
        out = [dt for dt in out if dt in keep]

    # ---- spatial: in_box ---------------------------------------------
    if in_box is not None:
        x0, y0, z0, x1, y1, z1 = in_box
        in_box_set = set(
            gmsh.model.getEntitiesInBoundingBox(x0, y0, z0, x1, y1, z1, dim=dim)
        )
        out = [dt for dt in out if dt in in_box_set]

    # ---- spatial: in_sphere (centroid within radius of center) -------
    if in_sphere is not None:
        cx, cy, cz, r = in_sphere
        r2 = r * r
        center = np.array([cx, cy, cz])
        def _in_sphere(dt: DimTag) -> bool:
            c = _entity_center(*dt)
            d = c - center
            return float(d @ d) <= r2
        out = [dt for dt in out if _in_sphere(dt)]

    # ---- spatial: on_plane -------------------------------------------
    if on_plane is not None:
        axis, val, atol = on_plane
        i = _AXIS_IDX[axis.lower()]
        def _on_plane(dt: DimTag) -> bool:
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(*dt)
            lo = (x0, y0, z0)[i]; hi = (x1, y1, z1)[i]
            return (lo - atol) <= val <= (hi + atol)
        out = [dt for dt in out if _on_plane(dt)]

    # ---- spatial: on_axis (centroid lies on axis line through origin)
    if on_axis is not None:
        axis, atol = on_axis
        i = _AXIS_IDX[axis.lower()]
        def _on_axis(dt: DimTag) -> bool:
            c = _entity_center(*dt)
            # the two *other* coords must be ~0
            others = [c[j] for j in range(3) if j != i]
            return all(abs(v) <= atol for v in others)
        out = [dt for dt in out if _on_axis(dt)]

    # ---- spatial: at_point -------------------------------------------
    if at_point is not None:
        px, py, pz, atol = at_point
        np.array([px, py, pz])
        def _at_point(dt: DimTag) -> bool:
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(*dt)
            return (x0 - atol <= px <= x1 + atol and
                    y0 - atol <= py <= y1 + atol and
                    z0 - atol <= pz <= z1 + atol)
        out = [dt for dt in out if _at_point(dt)]

    # ---- metric ranges -----------------------------------------------
    if length_range is not None and dim == 1:
        lo, hi = length_range
        out = [dt for dt in out
               if lo <= _safe_mass(*dt) <= hi]
    if area_range is not None and dim == 2:
        lo, hi = area_range
        out = [dt for dt in out
               if lo <= _safe_mass(*dt) <= hi]
    if volume_range is not None and dim == 3:
        lo, hi = volume_range
        out = [dt for dt in out
               if lo <= _safe_mass(*dt) <= hi]

    # ---- orientation (curves only) -----------------------------------
    if dim == 1:
        if horizontal is True:
            out = [dt for dt in out if _is_axis_aligned(dt, "z",
                                                        perpendicular=True)]
        if vertical is True:
            out = [dt for dt in out if _is_axis_aligned(dt, "z",
                                                        perpendicular=False)]
        if aligned is not None:
            if isinstance(aligned, str):
                axis, atol_deg = aligned, 5.0
            else:
                axis, atol_deg = aligned
            out = [dt for dt in out
                   if _is_axis_aligned(dt, axis, atol_deg=atol_deg)]

    # ---- predicate escape hatch --------------------------------------
    if predicate is not None:
        out = [dt for dt in out if predicate(*dt)]

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
        # dim == 1 or 2 → parametric midpoint
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

    ``perpendicular=False`` → curve is parallel to ``axis``.
    ``perpendicular=True``  → curve is perpendicular to ``axis`` (i.e.
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
    along_axis = abs(extents[i]) / total  # 1 → parallel, 0 → perpendicular
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