from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

from ._helpers import Tag, TagsLike
from ._geometry_errors import WarnGeomImportHealth

if TYPE_CHECKING:
    from .Model import Model


# ── CAD-health diagnostics (non-mutating) ────────────────────────────
#: Edge/face "tiny" threshold, relative to the model's bbox diagonal.
#: An edge shorter than ``_REL_TINY · diag`` (or a face below that
#: squared) is flagged as a sliver that commonly defeats meshing /
#: booleans.  Advisory only — never mutates.
_REL_TINY: float = 1e-4

#: Suggested heal tolerance, relative to the bbox diagonal.  Used both
#: by :meth:`_IO.diagnose` (what to print) and by ``heal=True`` /
#: ``heal="auto"`` on import (what to actually pass to ``healShapes``).
#: Conservative — small enough not to merge genuine features, large
#: enough to close the µm-scale gaps typical of exported CAD.
_REL_HEAL: float = 1e-6


def _model_bbox_diag() -> float:
    """Bounding-box diagonal of the whole model, or ``0.0`` when the
    model is empty (``getBoundingBox`` returns non-finite extents)."""
    try:
        xn, yn, zn, xx, yx, zx = gmsh.model.getBoundingBox(-1, -1)
    except Exception:
        return 0.0
    pts = (xn, yn, zn, xx, yx, zx)
    if not all(math.isfinite(v) for v in pts):
        return 0.0
    return math.dist((xn, yn, zn), (xx, yx, zx))


def _suggested_heal_tolerance(diag: float) -> float:
    """Scale-aware heal tolerance for a model whose bbox diagonal is
    ``diag``.  Falls back to the legacy absolute ``1e-8`` for an empty
    / zero-extent model."""
    return _REL_HEAL * diag if diag > 0 else 1e-8


@dataclass(frozen=True)
class ImportHealth:
    """Non-mutating health report for imported / current CAD geometry.

    Returned by :meth:`_IO.diagnose`.  Carries the per-dimension entity
    counts, the sliver tallies, the model scale, and a suggested
    ``heal=`` tolerance — but never changes the geometry.
    """

    dim_counts: dict[int, int]
    highest_dim: int
    bbox_diag: float
    short_edges: tuple[int, ...]
    tiny_faces: tuple[int, ...]
    suggested_tolerance: float

    @property
    def n_solids(self) -> int:
        return self.dim_counts.get(3, 0)

    @property
    def is_suspect(self) -> bool:
        """True when slivers are present (the unambiguous dirty-CAD
        signal).  A surface-only import (``highest_dim < 3``) is *not*
        treated as suspect on its own — shell models import that way on
        purpose — so the advisory does not false-positive on them."""
        return bool(self.short_edges or self.tiny_faces)

    def advisory(self) -> str:
        return (
            f"imported geometry: {self.n_solids} solid(s), "
            f"{len(self.short_edges)} edge(s) shorter than "
            f"{_REL_TINY:.0e}·diag, {len(self.tiny_faces)} tiny face(s). "
            f"Slivers commonly defeat meshing / booleans — re-import "
            f"with heal='auto' (≈ {self.suggested_tolerance:.2e}) or "
            f"dedupe=True, or call g.model.io.diagnose() to inspect."
        )

    def __str__(self) -> str:
        return (
            f"ImportHealth(solids={self.n_solids}, "
            f"dims={self.dim_counts}, highest_dim={self.highest_dim}, "
            f"short_edges={len(self.short_edges)}, "
            f"tiny_faces={len(self.tiny_faces)}, "
            f"bbox_diag={self.bbox_diag:.4g}, "
            f"suggested_tolerance={self.suggested_tolerance:.2e})"
        )


def _compute_health() -> ImportHealth:
    """Scan the current model (non-mutating) and build an
    :class:`ImportHealth`.  Sub-entities are read straight from gmsh,
    so the sliver tallies work even when the import used
    ``highest_dim_only=True`` (the solids' boundary edges/faces still
    live in the OCC kernel)."""
    counts = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
    highest = max((d for d in range(4) if counts[d]), default=-1)
    diag = _model_bbox_diag()
    short_edges: list[int] = []
    tiny_faces: list[int] = []
    if diag > 0:
        edge_tol = _REL_TINY * diag
        face_tol = edge_tol * edge_tol
        for _, t in gmsh.model.getEntities(1):
            try:
                length = gmsh.model.occ.getMass(1, t)
            except Exception:
                continue  # degenerate edge — advisory scan skips it
            if 0.0 < length < edge_tol:
                short_edges.append(int(t))
        for _, t in gmsh.model.getEntities(2):
            try:
                area = gmsh.model.occ.getMass(2, t)
            except Exception:
                continue
            if 0.0 < area < face_tol:
                tiny_faces.append(int(t))
    return ImportHealth(
        dim_counts=counts,
        highest_dim=highest,
        bbox_diag=diag,
        short_edges=tuple(short_edges),
        tiny_faces=tuple(tiny_faces),
        suggested_tolerance=_suggested_heal_tolerance(diag),
    )


class _DXFImporter:
    """Encapsulates the DXF -> OCC geometry conversion pipeline.

    Owns point deduplication, per-entity-type conversion, and
    post-dedup layer rebuilding.  Instantiated by :meth:`_IO.load_dxf`.
    """

    def __init__(self, model: "Model", tol: float) -> None:
        self._model = model
        self._tol = tol
        self._pt_cache: dict[tuple[int, int, int], Tag] = {}
        self._geom_to_layer: dict[tuple[float, ...], str] = {}
        self._pt_to_layer: dict[tuple[int, int, int], str] = {}

    # -- helpers ----------------------------------------------------------

    def _point_key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        inv = 1.0 / self._tol
        return (round(x * inv), round(y * inv), round(z * inv))

    def _get_or_add_point(self, x: float, y: float, z: float) -> Tag:
        key = self._point_key(x, y, z)
        if key in self._pt_cache:
            return self._pt_cache[key]
        tag = gmsh.model.occ.addPoint(x, y, z)
        self._pt_cache[key] = tag
        self._model._register(0, tag, None, 'dxf_point')
        return tag

    @staticmethod
    def _bbox_key(
        x0: float, y0: float, z0: float,
        x1: float, y1: float, z1: float,
    ) -> tuple[float, ...]:
        return (
            round(min(x0, x1), 8), round(min(y0, y1), 8),
            round(min(z0, z1), 8), round(max(x0, x1), 8),
            round(max(y0, y1), 8), round(max(z0, z1), 8),
        )

    # -- per-entity-type converters ---------------------------------------

    def _convert_point(self, entity) -> None:
        pt = entity.dxf.location
        self._get_or_add_point(pt.x, pt.y, pt.z)
        self._pt_to_layer[self._point_key(pt.x, pt.y, pt.z)] = entity.dxf.layer

    def _convert_line(self, entity) -> None:
        s, e = entity.dxf.start, entity.dxf.end
        p1 = self._get_or_add_point(s.x, s.y, s.z)
        p2 = self._get_or_add_point(e.x, e.y, e.z)
        gmsh.model.occ.addLine(p1, p2)
        self._geom_to_layer[self._bbox_key(s.x, s.y, s.z, e.x, e.y, e.z)] = entity.dxf.layer

    def _convert_arc(self, entity) -> None:
        c = entity.dxf.center
        r = entity.dxf.radius
        a1 = math.radians(entity.dxf.start_angle)
        a2 = math.radians(entity.dxf.end_angle)
        if a2 <= a1:
            a2 += 2.0 * math.pi
        gmsh.model.occ.addCircle(c.x, c.y, c.z, r, angle1=a1, angle2=a2)
        sx = c.x + r * math.cos(a1)
        sy = c.y + r * math.sin(a1)
        ex = c.x + r * math.cos(a2)
        ey = c.y + r * math.sin(a2)
        self._geom_to_layer[self._bbox_key(sx, sy, c.z, ex, ey, c.z)] = entity.dxf.layer

    def _convert_circle(self, entity) -> None:
        c = entity.dxf.center
        r = entity.dxf.radius
        gmsh.model.occ.addCircle(c.x, c.y, c.z, r)
        self._geom_to_layer[self._bbox_key(
            c.x - r, c.y - r, c.z, c.x + r, c.y + r, c.z,
        )] = entity.dxf.layer

    def _convert_polyline(self, entity) -> None:
        etype = entity.dxftype()
        layer = entity.dxf.layer
        if etype == 'LWPOLYLINE':
            vertices = list(entity.get_points(format='xyz'))  # type: ignore[attr-defined]
        else:
            vertices = [
                (v.dxf.location.x, v.dxf.location.y, v.dxf.location.z)
                for v in entity.vertices  # type: ignore[attr-defined]
            ]
        pts = [self._get_or_add_point(vx, vy, vz) for vx, vy, vz in vertices]

        is_closed = (
            getattr(entity.dxf, 'flags', 0) & 1
            if etype == 'POLYLINE' else entity.closed  # type: ignore[attr-defined]
        )
        vert_pairs = list(zip(vertices, vertices[1:]))
        if is_closed and len(vertices) > 2:
            vert_pairs.append((vertices[-1], vertices[0]))

        pt_pairs = list(zip(pts, pts[1:]))
        if is_closed and len(pts) > 2:
            pt_pairs.append((pts[-1], pts[0]))

        for (v_s, v_e), (p1, p2) in zip(vert_pairs, pt_pairs):
            gmsh.model.occ.addLine(p1, p2)
            self._geom_to_layer[self._bbox_key(
                v_s[0], v_s[1], v_s[2], v_e[0], v_e[1], v_e[2],
            )] = layer

    def _convert_spline(self, entity) -> None:
        ctrl_pts: list[Tag] = []
        for cp in entity.control_points:  # type: ignore[attr-defined]
            ctrl_pts.append(self._get_or_add_point(
                cp[0], cp[1], cp[2] if len(cp) > 2 else 0.0,
            ))
        if len(ctrl_pts) < 2:
            return
        gmsh.model.occ.addBSpline(ctrl_pts)
        cps = entity.control_points  # type: ignore[attr-defined]
        xs = [c[0] for c in cps]
        ys = [c[1] for c in cps]
        zs = [c[2] if len(c) > 2 else 0.0 for c in cps]
        self._geom_to_layer[self._bbox_key(
            min(xs), min(ys), min(zs), max(xs), max(ys), max(zs),
        )] = entity.dxf.layer

    # -- dispatch table ---------------------------------------------------

    _CONVERTERS: dict[str, str] = {
        'POINT': '_convert_point',
        'LINE': '_convert_line',
        'ARC': '_convert_arc',
        'CIRCLE': '_convert_circle',
        'LWPOLYLINE': '_convert_polyline',
        'POLYLINE': '_convert_polyline',
        'SPLINE': '_convert_spline',
    }

    # -- main entry point -------------------------------------------------

    def run(
        self,
        file_path: Path,
        create_physical_groups: bool,
        sync: bool,
    ) -> dict[str, dict[int, list[Tag]]]:
        try:
            import ezdxf
        except ImportError:
            raise ImportError(
                "ezdxf is required for DXF import.  "
                "Install it with:  pip install ezdxf"
            )

        if not file_path.exists():
            raise FileNotFoundError(f"DXF file not found: {file_path}")

        doc = ezdxf.readfile(str(file_path))
        msp = doc.modelspace()

        # Convert entities
        for entity in msp:
            etype = entity.dxftype()
            method_name = self._CONVERTERS.get(etype)
            if method_name:
                getattr(self, method_name)(entity)
            else:
                self._model._log(
                    f"DXF: skipped unsupported entity {etype} "
                    f"on layer '{entity.dxf.layer}'"
                )

        # Merge duplicates & synchronise
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # Rebuild layer mapping from surviving entities
        layers = self._rebuild_layers()

        if create_physical_groups:
            for layer_name, dim_tags in layers.items():
                for dim, tags in dim_tags.items():
                    if tags:
                        gmsh.model.addPhysicalGroup(dim, tags, name=layer_name)

        layer_summary = {
            name: {d: len(ts) for d, ts in ents.items()}
            for name, ents in layers.items()
        }
        self._model._log(f"loaded DXF <- {file_path.name}  layers={layer_summary}")
        return layers

    def _rebuild_layers(self) -> dict[str, dict[int, list[Tag]]]:
        layers: dict[str, dict[int, list[Tag]]] = {}
        for dim, tag in gmsh.model.getEntities(1):
            bb = gmsh.model.getBoundingBox(dim, tag)
            bbox_key = self._bbox_key(*bb)
            layer_name = self._geom_to_layer.get(bbox_key, "_unmatched")
            self._model._register(dim, tag, None, 'dxf')
            layers.setdefault(layer_name, {}).setdefault(1, []).append(tag)
        for dim, tag in gmsh.model.getEntities(0):
            self._model._register(dim, tag, None, 'dxf_point')
        return layers


class _IO:
    """IO sub-composite — import/export IGES, STEP, DXF, MSH."""

    def __init__(self, model: "Model") -> None:
        self._model = model

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def _import_shapes(
        self,
        file_path      : Path,
        kind           : str,
        highest_dim_only: bool,
        sync           : bool,
        heal           : bool | float | str = False,
        dedupe         : bool | float = False,
        fuse           : bool = False,
        label          : str | None = None,
    ) -> dict[int, list[Tag]]:
        """
        Core import helper shared by ``load_iges`` and ``load_step``.

        Calls ``gmsh.model.occ.importShapes``, captures the returned
        (dim, tag) pairs, registers every imported entity, and returns a
        dimension-indexed dict so callers can address entities immediately.

        Order of operations when optional steps are enabled:
        ``import -> heal -> dedupe -> fuse -> label``.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` — e.g. ``{3: [1, 2], 2: [5, 6, 7]}``
        """
        will_mutate = bool(heal) or bool(dedupe)

        # Snapshot pre-existing entities so we can re-derive the
        # "imported set" even after heal/dedupe rewrite the kernel.
        snapshot: dict[int, set[Tag]] = (
            {
                d: {t for _, t in gmsh.model.getEntities(d)}
                for d in range(4)
            }
            if will_mutate else {}
        )

        raw: list[tuple[int, int]] = gmsh.model.occ.importShapes(
            str(file_path),
            highestDimOnly=highest_dim_only,
        )
        if sync or will_mutate:
            gmsh.model.occ.synchronize()

        # Defer label until after optional fuse — fuse consumes its
        # inputs and re-labels the survivor, so pre-labeling would
        # orphan the PG.
        for dim, tag in raw:
            self._model._register(dim, tag, None, kind)

        heal_tol: float | None = None
        if heal:
            # ``heal=True`` / ``heal="auto"`` derive a scale-aware
            # tolerance from the model bbox (a fixed 1e-8 is meaningless
            # across unit systems); an explicit float overrides.
            if heal is True or heal == "auto":
                heal_tol = _suggested_heal_tolerance(_model_bbox_diag())
            else:
                heal_tol = float(heal)
            if raw:
                self.heal_shapes(list(raw), tolerance=heal_tol, sync=True)

        if dedupe:
            dedupe_tol = None if dedupe is True else float(dedupe)
            self._model._parent.queries.remove_duplicates(
                tolerance=dedupe_tol, sync=True,
            )

        # Re-derive the surviving imported set.
        if will_mutate:
            result: dict[int, list[Tag]] = {}
            for d in range(4):
                live = {t for _, t in gmsh.model.getEntities(d)}
                new = sorted(live - snapshot.get(d, set()))
                if new:
                    result[d] = new
                    for t in new:
                        if (d, t) not in self._model._metadata:
                            self._model._register(d, t, None, kind)
        else:
            result = {}
            for dim, tag in raw:
                result.setdefault(dim, []).append(tag)

        fused = False
        if fuse and result:
            top_dim = max(result)
            top_tags = result[top_dim]
            if len(top_tags) >= 2:
                merged = self._model.boolean.fuse(
                    top_tags[:1], top_tags[1:],
                    dim=top_dim, sync=sync, label=label,
                )
                # Lower-dim sub-imports (only present when
                # highest_dim_only=False) are invalidated by the
                # volume fuse — drop them from the returned dict.
                result = {top_dim: merged}
                fused = True

        if label is not None and not fused:
            labels_comp = getattr(self._model._parent, 'labels', None)
            if labels_comp is not None:
                for dim, tags in result.items():
                    if tags:
                        labels_comp.add(dim, tags, name=label)

        dim_summary = {d: len(ts) for d, ts in result.items()}
        suffix = ""
        if heal:
            suffix += f"  healed(tol={heal_tol:.2e})"
        if dedupe:
            suffix += "  deduped" + (
                f"(tol={float(dedupe)})" if dedupe is not True else ""
            )
        if fused:
            suffix += "  fused"
        if label:
            suffix += f"  label={label!r}"
        self._model._log(
            f"loaded {kind.upper()} <- {file_path.name}  {dim_summary}{suffix}"
        )

        # Advisory: a raw (un-healed) import that shows slivers gets one
        # non-mutating WarnGeomImportHealth so the user knows to re-run
        # with heal=. Skipped when the user already healed (the slivers
        # would be gone) or fused (single survivor, scan is moot).
        if not heal and not fused:
            self.diagnose(warn=True)

        return result

    def load_iges(
        self,
        file_path       : Path | str,
        *,
        highest_dim_only: bool = True,
        sync            : bool = True,
        heal            : bool | float | str = False,
        dedupe          : bool | float = False,
        fuse            : bool = False,
        label           : str | None = None,
    ) -> dict[int, list[Tag]]:
        """
        Import an IGES file into the current model.

        All imported entities are registered and their tags are returned so
        you can immediately use them in boolean ops or transforms.

        Parameters
        ----------
        highest_dim_only : bool
            If True (default) only the highest-dimension entities are
            returned and registered (volumes for solids, surfaces for
            surface models).  Set to False to capture every sub-entity
            (faces, edges, vertices) as well.
        heal : bool, float, or "auto"
            Run ``heal_shapes`` on the imported entities immediately
            after import.  ``True`` and ``"auto"`` derive a
            **scale-aware** tolerance from the model bounding box
            (``≈ 1e-6 · bbox_diagonal``) — a fixed absolute tolerance is
            meaningless across unit systems.  A float overrides it
            (e.g. ``heal=1e-3``).  ``False`` (default) imports raw and,
            if the result shows slivers, emits a non-mutating
            :class:`WarnGeomImportHealth` advisory (see
            :meth:`diagnose`).  For non-tolerance knobs, call
            ``heal_shapes()`` directly.
        dedupe : bool or float
            Run ``g.model.queries.remove_duplicates`` after the import
            (and after heal, if enabled).  ``True`` uses the current
            Gmsh tolerance; a float overrides it for the call.
        fuse : bool
            If True, union all imported top-dimension entities into a
            single survivor via ``g.model.boolean.fuse``.  No-op when
            the import yields fewer than two entities at the top
            dimension.  Combined with ``highest_dim_only=False``, the
            lower-dim sub-imports are discarded since the volume fuse
            invalidates them.
        label : str, optional
            Global label attached to all imported entities (or to the
            fused survivor, when ``fuse=True``).  Resolvable via
            ``g.labels.entities(name)``.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` indexed by dimension.

        Example
        -------
        ::

            imported = g.model.io.load_iges("part.iges")
            bodies   = imported[3]           # all imported volume tags
            flange   = bodies[0]             # first imported volume

            boss = g.model.geometry.add_cylinder(10, 10, 0,  0, 0, 5,  3)
            result = g.model.boolean.fuse(flange, boss)
        """
        return self._import_shapes(
            Path(file_path), 'iges', highest_dim_only, sync,
            heal=heal, dedupe=dedupe, fuse=fuse, label=label,
        )

    def load_step(
        self,
        file_path       : Path | str,
        *,
        highest_dim_only: bool = True,
        sync            : bool = True,
        heal            : bool | float | str = False,
        dedupe          : bool | float = False,
        fuse            : bool = False,
        label           : str | None = None,
    ) -> dict[int, list[Tag]]:
        """
        Import a STEP file into the current model.

        All imported entities are registered and their tags are returned so
        you can immediately use them in boolean ops or transforms.

        Parameters
        ----------
        highest_dim_only : bool
            If True (default) only the highest-dimension entities are
            returned and registered.  Set to False to include all
            sub-entities.
        heal : bool, float, or "auto"
            Run ``heal_shapes`` on the imported entities immediately
            after import.  ``True`` and ``"auto"`` derive a
            **scale-aware** tolerance from the model bounding box
            (``≈ 1e-6 · bbox_diagonal``) — a fixed absolute tolerance is
            meaningless across unit systems.  A float overrides it
            (e.g. ``heal=1e-3``).  ``False`` (default) imports raw and,
            if the result shows slivers, emits a non-mutating
            :class:`WarnGeomImportHealth` advisory (see
            :meth:`diagnose`).  For non-tolerance knobs, call
            ``heal_shapes()`` directly.
        dedupe : bool or float
            Run ``g.model.queries.remove_duplicates`` after the import
            (and after heal, if enabled).  ``True`` uses the current
            Gmsh tolerance; a float overrides it for the call.
        fuse : bool
            If True, union all imported top-dimension entities into a
            single survivor via ``g.model.boolean.fuse``.  No-op when
            the import yields fewer than two entities at the top
            dimension.  Combined with ``highest_dim_only=False``, the
            lower-dim sub-imports are discarded since the volume fuse
            invalidates them.
        label : str, optional
            Global label attached to all imported entities (or to the
            fused survivor, when ``fuse=True``).  Resolvable via
            ``g.labels.entities(name)``.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` indexed by dimension.

        Example
        -------
        ::

            # one-shot: import an assembly, clean + fuse + label
            imported = g.model.io.load_step(
                "assembly.step",
                heal=True, dedupe=True, fuse=True, label="frame",
            )
            body = imported[3][0]   # single fused volume
        """
        return self._import_shapes(
            Path(file_path), 'step', highest_dim_only, sync,
            heal=heal, dedupe=dedupe, fuse=fuse, label=label,
        )

    def heal_shapes(
        self,
        tags: TagsLike | None = None,
        *,
        dim             : int   = 3,
        tolerance       : float = 1e-8,
        fix_degenerated : bool  = True,
        fix_small_edges : bool  = True,
        fix_small_faces : bool  = True,
        sew_faces       : bool  = True,
        make_solids     : bool  = True,
        sync            : bool  = True,
    ) -> _IO:
        """
        Heal topology issues in imported CAD geometry (STEP / IGES).

        Wraps ``gmsh.model.occ.healShapes`` which fixes common issues
        such as degenerate edges, tiny faces, gaps between faces, and
        open shells that should be solids.

        Parameters
        ----------
        tags : entities to heal (default: all entities in the model).
        dim : default dimension for bare integer tags.
        tolerance : healing tolerance (default 1e-8).
        fix_degenerated : fix degenerate edges/faces.
        fix_small_edges : remove edges smaller than tolerance.
        fix_small_faces : remove faces smaller than tolerance.
        sew_faces : reconnect open shells at shared edges.
        make_solids : close healed shells into solids.
        sync : synchronise OCC kernel (default True).

        Returns
        -------
        self — for method chaining.

        Example
        -------
        ::

            imported = g.model.io.load_step("legacy_part.step")
            g.model.io.heal_shapes(tolerance=1e-3)
        """
        # Phase 3B.2d / ADR 0038 — chain-phase freeze.  ``heal_shapes``
        # mutates the OCC kernel even for entities already in the
        # broker, so it's gated unconditionally rather than relying on
        # ``_register`` (which only runs for new outputs).
        from ._compose_errors import chain_phase_guard
        chain_phase_guard(self._model._parent, "g.model.io.heal_shapes")
        if tags is not None:
            dt = self._model._as_dimtags(tags, dim)
        else:
            dt = []  # empty = heal everything

        out: list[tuple[int, int]] = gmsh.model.occ.healShapes(
            dimTags=dt,
            tolerance=tolerance,
            fixDegenerated=fix_degenerated,
            fixSmallEdges=fix_small_edges,
            fixSmallFaces=fix_small_faces,
            sewFaces=sew_faces,
            makeSolids=make_solids,
        )
        if sync:
            gmsh.model.occ.synchronize()

        for d, t in out:
            if (d, t) not in self._model._metadata:
                self._model._register(d, t, None, 'healed')
        self._model._log(
            f"heal_shapes(tol={tolerance}) -> {len(out)} entities output"
        )
        return self

    def diagnose(self, *, warn: bool = False) -> ImportHealth:
        """Report CAD health of the current model **without mutating it**.

        Scans the live OCC geometry and returns an :class:`ImportHealth`
        with per-dimension entity counts, sliver tallies (edges / faces
        far below the model scale), the bbox diagonal, and a suggested
        ``heal=`` tolerance.  Nothing is healed, deduped, or
        renumbered — this is the look-before-you-leap counterpart to
        :meth:`heal_shapes` (which *does* mutate and renumber).

        Parameters
        ----------
        warn : bool, default False
            When True, emit a :class:`WarnGeomImportHealth` advisory if
            the report :attr:`~ImportHealth.is_suspect` (slivers
            present).  ``load_step`` / ``load_iges`` use this internally
            on a raw (un-healed) import.

        Returns
        -------
        ImportHealth

        Example
        -------
        ::

            g.model.io.load_step("messy.step")        # raw
            report = g.model.io.diagnose()
            if report.is_suspect:
                g.model.io.load_step("messy.step", heal="auto", dedupe=True)
        """
        health = _compute_health()
        if warn and health.is_suspect:
            warnings.warn(WarnGeomImportHealth(health.advisory()), stacklevel=2)
        return health

    def save_iges(self, file_path: Path | str) -> None:
        """
        Export the current model to IGES.

        The ``.iges`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.iges')
        gmsh.write(str(file_path))
        self._model._log(f"saved IGES -> {file_path}")

    def save_step(self, file_path: Path | str) -> None:
        """
        Export the current model to STEP.

        The ``.step`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.step')
        gmsh.write(str(file_path))
        self._model._log(f"saved STEP -> {file_path}")

    # ------------------------------------------------------------------
    # DXF (AutoCAD) — parsed with ezdxf, geometry built via OCC kernel
    # ------------------------------------------------------------------

    def load_dxf(
        self,
        file_path: Path | str,
        *,
        point_tolerance: float = 1e-6,
        create_physical_groups: bool = True,
        sync: bool = True,
    ) -> dict[str, dict[int, list[Tag]]]:
        """
        Import a DXF file into the current model.

        Uses ``ezdxf`` to parse the DXF (supports all AutoCAD versions
        from R12 to 2024+), then builds Gmsh geometry through the OCC
        kernel.  AutoCAD **layers** become Gmsh physical groups
        automatically.

        Supported DXF entity types: ``LINE``, ``ARC``, ``CIRCLE``,
        ``LWPOLYLINE``, ``POLYLINE``, ``SPLINE``, ``POINT``.

        Parameters
        ----------
        file_path : Path or str
            Path to the ``.dxf`` file.
        point_tolerance : float
            Distance below which two DXF endpoints are considered
            coincident and share a single Gmsh point.  Default ``1e-6``.
        create_physical_groups : bool
            If True (default), a physical group is created for each DXF
            layer.  If False, entities are created but no physical groups
            are made (useful when you want to assign groups manually).
        sync : bool
            Synchronise the OCC kernel after import (default True).

        Returns
        -------
        dict[str, dict[int, list[Tag]]]
            ``{layer_name: {dim: [tag, ...]}}``

            Each key is a DXF layer name.  Values map entity dimension
            to lists of Gmsh tags created from that layer.

        Example
        -------
        ::

            # AutoCAD drawing with layers: "C80x80", "V30x50"
            layers = g.model.io.load_dxf("frame_2D.dxf")

            # layers == {
            #     "C80x80": {1: [1, 2, 3, 4]},
            #     "V30x50": {1: [5, 6, 7, 8, 9]},
            # }

            # Physical groups are already created — ready for meshing.
            # Access beam curves:
            beam_curves = layers["V30x50"][1]
        """
        importer = _DXFImporter(self._model, point_tolerance)
        return importer.run(Path(file_path), create_physical_groups, sync)

    def save_dxf(self, file_path: Path | str) -> None:
        """
        Export the current model to DXF.

        The ``.dxf`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.dxf')
        gmsh.write(str(file_path))
        self._model._log(f"saved DXF -> {file_path}")

    def save_msh(self, file_path: Path | str) -> None:
        """
        Export the current model to Gmsh's native MSH format.

        Unlike STEP/IGES, this preserves **everything**: geometry, mesh,
        physical groups, and partition data.

        The ``.msh`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.msh')
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(file_path))
        self._model._log(f"saved MSH -> {file_path}")

    def load_msh(
        self,
        file_path: Path | str,
    ) -> dict[int, list[Tag]]:
        """
        Import a Gmsh ``.msh`` file using ``gmsh.merge``.

        Unlike ``load_iges`` / ``load_step``, this preserves physical
        groups, mesh data, and partition info — because ``.msh`` is
        Gmsh's native format.

        Parameters
        ----------
        file_path : Path or str
            Path to the ``.msh`` file.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` of all entities after merge.
        """
        # Phase 3B.2d / ADR 0038 — chain-phase freeze.
        from ._compose_errors import chain_phase_guard
        chain_phase_guard(self._model._parent, "g.model.io.load_msh")
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"MSH file not found: {file_path}")

        gmsh.merge(str(file_path))

        result: dict[int, list[Tag]] = {}
        for d in range(4):
            for dim, tag in gmsh.model.getEntities(d):
                result.setdefault(dim, []).append(tag)

        dim_summary = {d: len(ts) for d, ts in result.items()}
        self._model._log(f"loaded MSH <- {file_path.name}  {dim_summary}")
        return result

    def load_geo(
        self,
        file_path: Path | str,
    ) -> dict[int, list[Tag]]:
        """
        Import a Gmsh ``.geo`` script using ``gmsh.merge``.

        The script is executed in the active model, so any ``Mesh N;``
        statements inside the file will run.  The CAD kernel used for
        synchronization is auto-detected by scanning the file head for
        ``SetFactory("OpenCASCADE")``:

        - found  -> ``gmsh.model.occ.synchronize()``
        - absent -> ``gmsh.model.geo.synchronize()``

        Parameters
        ----------
        file_path : Path or str
            Path to the ``.geo`` file.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` of all entities after merge.
        """
        # Phase 3B.2d / ADR 0038 — chain-phase freeze.
        from ._compose_errors import chain_phase_guard
        chain_phase_guard(self._model._parent, "g.model.io.load_geo")
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"GEO file not found: {file_path}")

        head = file_path.read_text(encoding="utf-8", errors="ignore")[:4096]
        use_occ = "SetFactory(\"OpenCASCADE\")" in head

        gmsh.merge(str(file_path))
        if use_occ:
            gmsh.model.occ.synchronize()
            kernel = "occ"
        else:
            gmsh.model.geo.synchronize()
            kernel = "geo"

        result: dict[int, list[Tag]] = {}
        for d in range(4):
            for dim, tag in gmsh.model.getEntities(d):
                result.setdefault(dim, []).append(tag)

        dim_summary = {d: len(ts) for d, ts in result.items()}
        self._model._log(
            f"loaded GEO <- {file_path.name} [{kernel}]  {dim_summary}"
        )
        return result
