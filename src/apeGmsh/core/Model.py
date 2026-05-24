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

        # Per-node ndf overrides — shell-to-solid coupling feature.
        # Each entry is a :class:`apeGmsh._kernel.defs.node_ndf.NodeNDFDef`
        # registered via :meth:`set_node_ndf`.  Resolved at FEM-build
        # time by ``_fem_factory`` into the per-node ``ndf`` vector on
        # :class:`NodeComposite`.  Pre-mesh declarations only — the
        # broker is the single source of truth for resolved values.
        from apeGmsh._kernel.defs.node_ndf import NodeNDFDef  # local: keep import-time graph clean
        self._node_ndf_defs: list[NodeNDFDef] = []

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
    # Per-node ndf overrides (shell-to-solid coupling feature)
    # ------------------------------------------------------------------

    def set_node_ndf(
        self,
        target,
        *,
        ndf: int,
        name: str | None = None,
    ):
        """Register an explicit per-node ``ndf`` (DOF count) override.

        The broker's default rule derives each node's ``ndf`` from
        the **dim** of its incident element groups (frame-on-line
        and shell-on-surface both default to 6; solid-on-volume
        defaults to 3 — see :data:`IMPLICIT_NDF_BY_DIM` in
        :mod:`apeGmsh._kernel.records._node_ndf`).  Call this method
        to override that default for a specific named region:

        - Truss-only models — pin truss nodes to ``ndf=3``
          (lines would otherwise default to 6).
        - 2D plane-stress models — pin every node to ``ndf=2``.
        - Future shell-on-solid bridges (S2) — pin the shell-base
          interface nodes to ``ndf=6`` even when the implicit max
          rule already yields the same value, for explicit
          intent-capture.

        Resolution happens at FEM-build time
        (``g.mesh.queries.get_fem_data()`` / ``FEMData.from_gmsh()``)
        via the same label-then-PG-then-part precedence chain that
        loads and masses use.  The resulting per-node value is
        recorded with ``source='explicit'`` on the broker (see
        :class:`~apeGmsh._kernel.records._node_ndf.NodeNDFRecord`).

        Parameters
        ----------
        target :
            Label name, physical-group name, part label, raw
            ``[(dim, tag), ...]`` list, or mesh-selection name —
            anything the shared resolver accepts.
        ndf : int, keyword-only
            Number of DOFs to pin at every resolved node.  Must be
            in ``[1, 6]``.
        name : str, optional
            Friendly tag stored on the def for debugging.

        Returns
        -------
        NodeNDFDef
            The stored def (mostly useful for tests / introspection).

        Examples
        --------
        ::

            # truss model — bring all line nodes down from default 6 to 3
            g.model.set_node_ndf("Truss", ndf=3)

            # 2D plane stress — every node carries only ux, uy
            g.model.set_node_ndf("Plate", ndf=2)
        """
        from apeGmsh._kernel.defs.node_ndf import NodeNDFDef

        if not isinstance(ndf, int) or ndf < 1 or ndf > 6:
            raise ValueError(
                f"set_node_ndf: ndf must be an int in [1, 6]; got {ndf!r}"
            )
        defn = NodeNDFDef(target=target, ndf=int(ndf), name=name)
        self._node_ndf_defs.append(defn)
        return defn

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
        """Start a fluent, daisy-chainable CAD-entity selection.

        This is the **geometry entry** of the unified selection family
        (``docs/plans/selection-unification-v2.md``).  It is the single
        entity-selection surface: the former
        ``g.model.queries.select(tags, on=/crossing=...)`` predicate
        selector and the former ``g.model.selection.select_*`` entity
        composite have been removed; their behaviour is folded into the
        verbs below.

        ``select()`` returns an
        :class:`~apeGmsh.core._selection.EntitySelection` (entity
        family) whose verbs (``in_box``, ``in_sphere``, ``on_plane``,
        ``crossing_plane``, ``nearest_to``, ``where``, ``|`` ``&`` ``-``
        ``^``) compose, and whose direct terminals ``.to_label()`` /
        ``.to_physical()`` / ``.to_dataframe()`` consume it without a
        ``.result()`` step.  ``.result()`` is a zero-cost identity alias
        yielding the :class:`~apeGmsh.core._selection.Selection`
        payload, on which ``.tags()`` / ``.to_label()`` /
        ``.to_physical()`` are also available.

        Name resolution is delegated **verbatim** to the existing,
        contract-locked geometry resolver
        (:func:`apeGmsh.core._helpers.resolve_to_dimtags`): a string is
        resolved label (Tier 1) -> physical group (Tier 2) -> part
        (Tier 3); ``(dim, tag)`` / ``int`` / lists pass through with the
        same semantics as everywhere else.  This method re-implements
        none of that tier logic, and adds **no** scoping or
        boundary-walk behaviour of its own — what
        ``resolve_to_dimtags`` returns is exactly what seeds the chain
        (no silent truncation).  To narrow to a specific dimension,
        seed with a dimension-appropriate label / PG, or refine with a
        spatial verb (``.in_box`` / ``.on_plane`` / ...).

        Parameters
        ----------
        target :
            What to seed the chain with.  Any reference
            ``resolve_to_dimtags`` accepts — a label / PG / part name
            string, a bare int tag, a ``(dim, tag)`` pair, or a list
            thereof.  ``None`` selects every entity at ``dim`` (which
            then must be given), via ``resolve_to_dimtags(None,
            default_dim=dim)``.
        dim :
            The ``default_dim`` forwarded to ``resolve_to_dimtags`` —
            i.e. the dimension used for bare int tags and for
            ``target=None``.  It is **not** a post-filter (that would be
            a silent truncation the resolution contract forbids); a
            multi-dim label still enumerates every dim it occupies, per
            the locked contract.  Defaults to ``3`` (the resolver's own
            default) when not given.

        Returns
        -------
        EntitySelection

        Example
        -------
        ::

            # seed with a face physical group, then refine spatially
            (g.model.select("BottomFaces")
                .in_box((0, 0, 0), (1, 1, 0.5))
                .to_physical("lower_faces"))
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
