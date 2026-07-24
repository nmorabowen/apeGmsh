from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

from ._model_boolean import _Boolean
from ._model_geometry import _Geometry
from ._model_io import _IO
from ._model_queries import _Queries
from ._model_transforms import _Transforms

if TYPE_CHECKING:
    from apeGmsh._types import SessionProtocol as _SessionBase

from apeGmsh._logging import _HasLogging
from apeGmsh._types import Tag, DimTag, TagsLike


class Model(_HasLogging):
    """
    Geometry composite attached to an ``apeGmsh`` instance as
    ``g.model``.  Owns five focused sub-composites:

    * ``g.model.geometry`` — point / curve / surface / solid primitives
    * ``g.model.boolean`` — fuse, cut, intersect, fragment
    * ``g.model.transforms`` — translate, rotate, scale, mirror, copy,
      extrude, revolve, sweep, thru_sections
    * ``g.model.io`` — load/save STEP, IGES, DXF, MSH, heal_shapes
    * ``g.model.queries`` — bounding_box, center_of_mass, mass,
      boundary, adjacencies, entities_in_bounding_box, registry

    Plus entity selection:

    * ``g.model.select(...)`` — fluent spatial entity selection

    And top-level utilities on the Model itself:

    * ``g.model.sync()`` — flush the OCC kernel
    * ``g.model.viewer()`` — open the interactive Qt viewer
    * ``g.model.gui()`` / ``g.model.launch_picker()`` — native Gmsh viewers

    Example
    -------
    ::

        # Solid boolean workflow
        box  = g.model.geometry.add_box(0, 0, 0, 10, 10, 10)
        hole = g.model.geometry.add_cylinder(5, 5, 0, 0, 0, 10, 2)
        part = g.model.boolean.cut(box, hole)

        # Wire-frame -> surface workflow
        p1   = g.model.geometry.add_point(0, 0, 0)
        p2   = g.model.geometry.add_point(10, 0, 0)
        p3   = g.model.geometry.add_point(10, 5, 0)
        p4   = g.model.geometry.add_point(0, 5, 0)
        l1   = g.model.geometry.add_line(p1, p2)
        l2   = g.model.geometry.add_line(p2, p3)
        l3   = g.model.geometry.add_line(p3, p4)
        l4   = g.model.geometry.add_line(p4, p1)
        loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
        surf = g.model.geometry.add_plane_surface(loop)

    Parameters
    ----------
    parent : _SessionBase
        Owning session — used to read ``_verbose`` and ``name``.
    """

    _log_prefix = "Model"

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent
        # (dim, tag) -> {kind, ...}  (labels live in g.labels, not here)
        self._metadata: dict[DimTag, dict] = {}

        # Five focused sub-composites — each one holds a reference to self
        self.geometry = _Geometry(self)
        self.boolean = _Boolean(self)
        self.transforms = _Transforms(self)
        self.io = _IO(self)
        self.queries = _Queries(self)

    # ------------------------------------------------------------------
    # Internal helpers (used by sub-composites via self._model._*)
    # ------------------------------------------------------------------

    def _resolve_dim(self, tag: int, default_dim: int) -> int:
        """Resolve tag dimension from the live Gmsh model. See :func:`_helpers.resolve_dim`."""
        from ._helpers import resolve_dim
        return resolve_dim(tag, default_dim)

    def _as_dimtags(self, tags: TagsLike, default_dim: int = 3) -> list[DimTag]:
        """Normalize tag input to [(dim, tag), ...]. See :func:`_helpers.as_dimtags`."""
        from ._helpers import as_dimtags
        return as_dimtags(tags, default_dim)

    def _register(self, dim: int, tag: Tag, label: str | None, kind: str) -> Tag:
        # Phase 3B.2d / ADR 0038 — fail-loud on geometry mutation after
        # the broker has been extracted.  Every geometry primitive
        # (add_*, boolean ops, transforms, model.io.load_*) funnels
        # through here, so this single gate covers ~all geometry
        # mutation surfaces.
        from ._compose_errors import chain_phase_guard
        chain_phase_guard(self._parent, f"g.model.<geometry>({kind})")
        # Metadata only — labels live exclusively in g.labels (Gmsh PGs).
        self._metadata[(dim, tag)] = {'kind': kind}

        # When the owning session has ``_auto_pg_from_label`` set
        # (both Part and apeGmsh sessions), automatically create a
        # **label PG** (Tier 1 — geometry bookkeeping, prefixed with
        # ``_label:``) so the user-supplied label can be resolved by
        # ``g.labels.entities("name")`` and travels through the STEP
        # sidecar into the Assembly.  This does NOT create a solver-
        # facing physical group — the user promotes labels to PGs
        # explicitly via ``g.labels.promote_to_physical("name")``.
        # Label creation must never break geometry creation, but
        # failures are logged so they don't hide real bugs.
        if label and getattr(self._parent, '_auto_pg_from_label', False):
            labels_comp = getattr(self._parent, 'labels', None)
            if labels_comp is not None:
                try:
                    labels_comp.add(dim, [tag], name=label)
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"Label {label!r} (dim={dim}, tag={tag}) could "
                        f"not be created: {exc}",
                        stacklevel=3,
                    )
        return tag

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(self) -> "Model":
        """
        Synchronise the OCC kernel with the gmsh model topology.

        Call this explicitly when you have been batching operations with
        ``sync=False``.  Returns ``self`` for chaining.
        """
        gmsh.model.occ.synchronize()
        self._log("OCC kernel synchronised")
        return self

    # ------------------------------------------------------------------
    # Unified fluent selection (entity family) — additive
    # ------------------------------------------------------------------

    def select(self, target=None, *, dim: int | None = None):
        """Select geometry entities (faces, curves, volumes, points)
        to label or group them before meshing.

        Use this to identify geometry for physical groups, boundary
        conditions, or mesh sizing.  Results are consumed with
        ``.to_label()`` / ``.to_physical()`` / ``.to_dataframe()``::

            # mark all bottom faces as a label for later use
            (g.model.select("BottomFaces")
                .in_box((0, 0, 0), (10, 10, 0.01))
                .to_label("base"))

            # all surfaces that the z=1.5 plane crosses
            (g.model.select(None, dim=2)
                .crossing_plane({'z': 1.5}))

            # all surfaces that straddle a plane through 3 points
            (g.model.select(None, dim=2)
                .crossing_plane([(0,0,0), (1,0,0), (0,1,0)]))

        Returns an :class:`~apeGmsh.core._selection.EntitySelection`
        (entity family) that chains spatial-refinement verbs and
        terminates at ``.to_label()`` / ``.to_physical()`` /
        ``.to_dataframe()``.  ``.result()`` is an alias that yields
        the payload directly.

        .. note::
            **Entity family** — ``.in_box`` tests BRep **bounding-box
            containment** (always closed, ~1e-8 tolerance), *not*
            centroids.  Passing ``inclusive=`` raises ``TypeError``.
            Use ``.on_plane(...)`` or ``.crossing_plane(...)`` for
            exact boundary predicates.  For mesh-level centroid-based
            selection use :meth:`fem.nodes.select` /
            :meth:`fem.elements.select`.

        Parameters
        ----------
        target :
            Label name, physical group name, part name,
            ``(dim, tag)`` pair, raw int tag, or a list thereof.
            A string resolves through label → PG → part name in
            that order.  Pass ``None`` (with ``dim=``) to select
            every entity at that dimension.
        dim :
            Topological dimension for bare int tags and
            ``target=None`` (0=point, 1=curve, 2=surface,
            3=volume).  A label / physical group / part enumerates
            every dimension it occupies; ``dim`` is **not** a
            post-filter.  When ``dim`` is given explicitly and the
            target has **no** entity at that dimension, ``select``
            **raises** (rather than silently returning another
            dimension) — chain ``.boundary()`` to walk a volume down
            to its faces.  Defaults to 3 when omitted (bare int tags /
            ``target=None``).

        Refining verbs
        --------------
        Each returns a new ``EntitySelection`` and composes freely.

        - ``.in_box(lo, hi)`` — entities whose BRep bbox falls inside
          the query box (always closed, ~1e-8 expanded).  No
          ``inclusive=`` kwarg.
        - ``.in_sphere(center, radius)``
        - ``.on_plane(point, normal, *, tol)`` — entities entirely on
          the plane within ``tol``.
        - ``.crossing_plane(spec, *, tol=1e-6, mode="crossing")`` —
          entities that straddle, lie on, or avoid a geometric
          primitive.

          ``spec`` accepts:

          .. code-block:: python

              {'z': 0}                          # axis-aligned plane
              {'x': 3.5}                        # axis-aligned plane
              [(0,0,0), (1,0,0), (0,1,0)]       # plane through 3 pts
              [(0,0,0), (0,0,1)]                # infinite line, 2 pts

          ``mode``:

          - ``"crossing"`` *(default)* — straddles the primitive
            (corners on both sides).
          - ``"on"`` — lies entirely on the primitive (all corners
            within ``tol``).
          - ``"not_crossing"`` / ``"not_on"`` — negations.

        - ``.nearest_to(point, *, count=1)``
        - ``.where(predicate)``
        - ``|`` ``&`` ``-`` ``^`` (set algebra).

        Terminals
        ---------
        - ``.to_label(name)`` — assign the selection as a label.
        - ``.to_physical(name)`` — assign as a physical group.
        - ``.to_dataframe()`` — ``DataFrame`` with dim/tag columns.
        - ``.result()`` — raw :class:`~apeGmsh.core._selection.Selection`
          payload (also exposes ``.tags()`` / ``.to_label()`` /
          ``.to_physical()``).
        """
        # Deferred import — the established idiom (mirrors
        # mesh/_mesh_structured.py).  ``_selection`` is same-package, so
        # this adds no eager cross-package edge
        # (tests/test_import_dag_polarity.py stays green with the
        # baseline unchanged).
        from ._helpers import resolve_to_dimtags
        from ._selection import EntitySelection

        if target is None and dim is None:
            raise ValueError(
                "model.select(): pass a target (label / PG / part / "
                "(dim, tag) / int / list) or a dim= to select every "
                "entity at that dimension."
            )
        dimtags = resolve_to_dimtags(
            target,
            default_dim=3 if dim is None else dim,
            session=self._parent,
        )
        # ``dim=`` is not a post-filter: a label / PG / part resolves to
        # the dimension(s) it occupies, ignoring ``dim=``.  When an
        # explicit ``dim=`` matches none of them, returning the other
        # dimension silently is the reported footgun (``select("soil",
        # dim=2)`` on a volume yields the volume, so ``.to_physical(...)``
        # registers a *volume* group).  Fail loud instead.
        if dim is not None and dimtags:
            present = sorted({d for d, _ in dimtags})
            if dim not in present:
                raise ValueError(
                    f"model.select({target!r}, dim={dim}): the target "
                    f"resolves to geometry at dimension(s) {present}, none "
                    f"at dim={dim}. `dim=` is not a post-filter — a label / "
                    f"physical group / part enumerates the dimension(s) it "
                    f"occupies. To get the boundary faces of a volume chain "
                    f"`.boundary()` (e.g. "
                    f"`select({target!r}).boundary().on_plane(...)`); to "
                    f"select every entity at a dimension pass "
                    f"`target=None` (e.g. `select(None, dim={dim})`)."
                )
        # selection-unification-v2: the host hook returns the v2
        # terminal ``EntitySelection`` (the entity-family
        # chain==terminal). Same deferred-import idiom; no new eager
        # cross-package edge.
        return EntitySelection(dimtags, _engine=self.queries)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def viewer(self, **kwargs):
        """Open the interactive Qt model viewer.

        Displays BRep geometry with selectable entities, parts,
        physical groups, and labels.  This is a **geometry-only**
        viewer — loads, constraints, and masses are mesh-resolved
        concepts and live on ``g.mesh.viewer()`` instead.

        Parameters
        ----------
        **kwargs :
            Forwarded to
            :class:`~apeGmsh.viewers.model_viewer.ModelViewer`
            (e.g. ``physical_group``, ``dims``, ``point_size``,
            ``line_width``, ``surface_opacity``).
        """
        # selection-unification v2 P3-R / §6.3 §4 SC-7: inline the
        # former ``self.selection.picker(**kwargs)`` (SelectionComposite
        # removed by M-STOP-2).  ``model=self`` is the identical object
        # the deleted ``picker`` passed as ``model=self._model``.
        from apeGmsh.viewers.model_viewer import ModelViewer
        p = ModelViewer(parent=self._parent, model=self, **kwargs)
        p.show()
        return p

    def preview(
        self,
        *,
        dims: list[int] | None = None,
        browser: bool = False,
        return_fig: bool = False,
    ):
        """Interactive WebGL preview of the BRep geometry.

        Zero Qt dependency — works inline in Jupyter / VS Code / Colab,
        or in a dedicated browser tab when ``browser=True``. Hover over
        a cell to see its ``dim`` and ``tag``.

        Parameters
        ----------
        dims : list of int, optional
            BRep dimensions to render. Defaults to ``[0, 1, 2, 3]``.
        browser : bool
            If ``True``, open in a new browser tab (temp HTML file)
            instead of rendering inline. Useful when the notebook
            output is cluttered or you want a dedicated window.
        return_fig : bool
            If ``True``, skip display and return the raw
            :class:`plotly.graph_objects.Figure` for saving with
            ``fig.write_html('path.html')`` or composing a notebook
            layout.
        """
        from apeGmsh.viz.NotebookPreview import preview_model
        return preview_model(
            self._parent,
            dims=dims,
            browser=browser,
            return_fig=return_fig,
        )

    def gui(self) -> None:
        """Open the interactive Gmsh FLTK GUI window."""
        gmsh.fltk.run()

    def launch_picker(
        self,
        *,
        show_points: bool = True,
        show_curves: bool = True,
        show_surfaces: bool = True,
        show_volumes: bool = False,
        verbose: bool = True,
    ) -> None:
        """Open Gmsh's native FLTK viewer with entity labels pre-enabled."""
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Geometry.PointLabels",   int(show_points))
        gmsh.option.setNumber("Geometry.CurveLabels",   int(show_curves))
        gmsh.option.setNumber("Geometry.SurfaceLabels", int(show_surfaces))
        gmsh.option.setNumber("Geometry.VolumeLabels",  int(show_volumes))
        gmsh.option.setNumber("Geometry.Points",   1)
        gmsh.option.setNumber("Geometry.Curves",   1)
        gmsh.option.setNumber("Geometry.Surfaces", 1)
        if verbose:
            print("[launch_picker] Opening Gmsh FLTK window.")
            print("  Labels visible — read tags off the 3D view.")
            print("  Close the window to return here.")
        gmsh.fltk.run()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Model(name={self._parent.name!r}, entities={len(self._metadata)})"
