from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

from ._helpers import Tag, TagsLike
from ._geometry_errors import WarnGeomHealSkipsSewing, WarnGeomImportHealth

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


def _free_curves_and_points() -> list[tuple[int, int]]:
    """Curves and points that bound nothing higher (beam / column
    lines, reference points)."""
    return [
        (d, t)
        for d in (1, 0)
        for _, t in gmsh.model.getEntities(d)
        if len(gmsh.model.getAdjacencies(d, t)[0]) == 0
    ]


def _top_level_entities(
    dimtags: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """The entities of ``dimtags`` that bound nothing higher, once each,
    in order: a CAD import's shapes (volumes, faces of a shell, free
    beam / column curves) without their sub-entities."""
    seen: set[tuple[int, int]] = set()
    top: list[tuple[int, int]] = []
    for d, t in dimtags:
        if (d, t) in seen:
            continue
        seen.add((d, t))
        if d == 3 or len(gmsh.model.getAdjacencies(d, t)[0]) == 0:
            top.append((d, t))
    return top


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


class WarnDxfLayerMismatch(UserWarning):
    """Advisory: :meth:`_IO.load_dxf` could not map every DXF curve to
    exactly one layer after OCC's duplicate removal (#1532).

    Two cases, one warning each: a DXF entity that no surviving OCC
    curve matched (its layer PG is short one curve, or absent when the
    layer lost every curve), and a surviving curve that matched entities
    on several layers (an exact duplicate drawn on two layers: the curve
    joins every such layer PG).  The geometry itself is imported either
    way, so this is a warning and not an error: raising here would leave
    the OCC writes already made in the model with no layer PGs at all.

    Subclass of :class:`UserWarning` so it can be silenced with
    ``warnings.simplefilter('ignore', WarnDxfLayerMismatch)``.
    """


#: ``gmsh.model.getBoundingBox`` pads an OCC curve's box by OCC's
#: ``Precision::Confusion`` (1e-7) on every side; the endpoint
#: coordinates ``gmsh.model.getValue`` returns are not padded.  A
#: matching tolerance below this pad never matches (#1532).
_OCC_BBOX_PAD: float = 1e-7

_Point3 = tuple[float, float, float]
_Bbox = tuple[float, float, float, float, float, float]


def _bbox_of(points: list[_Point3]) -> _Bbox:
    xs, ys, zs = zip(*points)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _distinct_points(points: tuple[_Point3, ...], tol: float) -> tuple[_Point3, ...]:
    """*points* with those within *tol* (per axis) of an earlier one dropped."""
    out: list[_Point3] = []
    for p in points:
        if not any(_points_close(p, q, tol) for q in out):
            out.append(p)
    return tuple(out)


def _points_close(p: _Point3, q: _Point3, tol: float) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(p, q))


@dataclass(frozen=True)
class _DxfCurveRecord:
    """One OCC curve a DXF converter created, as :meth:`_DXFImporter.
    _rebuild_layers` recognises it after ``removeAllDuplicates``
    renumbered the tags.

    ``ends`` are the curve's endpoint coordinates (a full circle holds
    its OCC seam vertex, ``centre + (r, 0, 0)``: gmsh builds the circle
    on the global X axis).  ``bbox`` is the curve's exact axis-aligned
    box when ``bbox_exact``; for a B-spline it is the control-polygon
    hull, which OCC's own box lies within but does not equal.
    """

    layer: str
    ends: tuple[_Point3, ...]
    bbox: _Bbox
    bbox_exact: bool

    def matches(self, ends: tuple[_Point3, ...], bbox: _Bbox, tol: float) -> bool:
        mine = _distinct_points(self.ends, tol)
        if len(mine) != len(ends):
            return False
        if not all(any(_points_close(p, q, tol) for q in mine) for p in ends):
            return False
        if self.bbox_exact:
            return all(abs(a - b) <= tol for a, b in zip(self.bbox, bbox))
        lo_ok = all(a - tol <= b for a, b in zip(self.bbox[:3], bbox[:3]))
        hi_ok = all(b <= a + tol for a, b in zip(self.bbox[3:], bbox[3:]))
        return lo_ok and hi_ok


class _DXFImporter:
    """Encapsulates the DXF -> OCC geometry conversion pipeline.

    Owns point deduplication, per-entity-type conversion, and
    post-dedup layer rebuilding.  Instantiated by :meth:`_IO.load_dxf`.
    """

    def __init__(self, model: "Model", tol: float) -> None:
        self._model = model
        self._tol = tol
        self._pt_cache: dict[tuple[int, int, int], Tag] = {}
        self._records: list[_DxfCurveRecord] = []
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

    def _record_segment(self, layer: str, s: _Point3, e: _Point3) -> None:
        self._records.append(_DxfCurveRecord(
            layer=layer, ends=(s, e), bbox=_bbox_of([s, e]), bbox_exact=True,
        ))

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
        self._record_segment(entity.dxf.layer, (s.x, s.y, s.z), (e.x, e.y, e.z))

    def _convert_arc(self, entity) -> None:
        c = entity.dxf.center
        r = entity.dxf.radius
        a1 = math.radians(entity.dxf.start_angle)
        a2 = math.radians(entity.dxf.end_angle)
        if a2 <= a1:
            a2 += 2.0 * math.pi
        gmsh.model.occ.addCircle(c.x, c.y, c.z, r, angle1=a1, angle2=a2)
        start = (c.x + r * math.cos(a1), c.y + r * math.sin(a1), c.z)
        end = (c.x + r * math.cos(a2), c.y + r * math.sin(a2), c.z)
        # The arc's box spans its endpoints and every axis extreme
        # (angles k*pi/2) the arc passes through.
        extremes = [start, end]
        quarter = 0.5 * math.pi
        k = math.ceil(a1 / quarter)
        while k * quarter <= a2:
            extremes.append((
                c.x + r * math.cos(k * quarter), c.y + r * math.sin(k * quarter), c.z,
            ))
            k += 1
        self._records.append(_DxfCurveRecord(
            layer=entity.dxf.layer, ends=(start, end),
            bbox=_bbox_of(extremes), bbox_exact=True,
        ))

    def _convert_circle(self, entity) -> None:
        c = entity.dxf.center
        r = entity.dxf.radius
        gmsh.model.occ.addCircle(c.x, c.y, c.z, r)
        self._records.append(_DxfCurveRecord(
            layer=entity.dxf.layer, ends=((c.x + r, c.y, c.z),),
            bbox=(c.x - r, c.y - r, c.z, c.x + r, c.y + r, c.z), bbox_exact=True,
        ))

    def _convert_polyline(self, entity) -> None:
        etype = entity.dxftype()
        layer = entity.dxf.layer
        if etype == 'LWPOLYLINE':
            # A lightweight polyline is planar: 2-D vertices at one
            # elevation.  ezdxf has no 'z' format code ('xyz' yields
            # pairs and the unpack below raised).
            z = float(entity.dxf.elevation)
            vertices = [
                (x, y, z)
                for x, y in entity.get_points(format='xy')  # type: ignore[attr-defined]
            ]
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
            self._record_segment(
                layer, (v_s[0], v_s[1], v_s[2]), (v_e[0], v_e[1], v_e[2]),
            )

    def _convert_spline(self, entity) -> None:
        cps: list[_Point3] = [
            (cp[0], cp[1], cp[2] if len(cp) > 2 else 0.0)
            for cp in entity.control_points  # type: ignore[attr-defined]
        ]
        ctrl_pts = [self._get_or_add_point(*cp) for cp in cps]
        if len(ctrl_pts) < 2:
            return
        gmsh.model.occ.addBSpline(ctrl_pts)
        # A clamped B-spline ends at its first and last control points;
        # OCC bounds it inside the control-polygon hull, not on it.
        self._records.append(_DxfCurveRecord(
            layer=entity.dxf.layer, ends=(cps[0], cps[-1]),
            bbox=_bbox_of(cps), bbox_exact=False,
        ))

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

        # Refuse a layer name the layer PGs cannot take before any
        # gmsh write, so a refused load imports nothing.  The set holds
        # every name _rebuild_layers can produce: the file's layers,
        # plus ``_unmatched`` for any curve whose layer it cannot match
        # (curves already in the model included).
        if create_physical_groups:
            self._refuse_layer_pg_names({
                entity.dxf.layer for entity in msp
                if entity.dxftype() in self._CONVERTERS
            } | {"_unmatched"})

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
            # Through g.physical.add, never a raw addPhysicalGroup: add()
            # upserts a name that exists at this dim, where gmsh would
            # leave a second same-named PG unnamed.
            physical = self._model._parent.physical
            for layer_name, dim_tags in layers.items():
                for dim, tags in dim_tags.items():
                    if tags:
                        physical.add(dim, tags, name=layer_name)

        layer_summary = {
            name: {d: len(ts) for d, ts in ents.items()}
            for name, ents in layers.items()
        }
        self._model._log(f"loaded DXF <- {file_path.name}  layers={layer_summary}")
        return layers

    def _refuse_layer_pg_names(self, names: set[str]) -> None:
        """Raise if a layer PG cannot take one of *names* (#1364).

        Layer PGs hold curves (``_rebuild_layers`` maps dim 1 only).
        A name with the reserved label prefix, or one a PG holds at
        another dim, is refused; ``g.physical.add`` would refuse the
        latter too, but only after earlier layers had been written.
        """
        # Lazy: keeps the eager core -> _kernel import graph unchanged
        # (tests/test_import_dag_polarity.py).
        from apeGmsh._kernel._label_prefix import LABEL_PREFIX, is_label_pg
        reserved = sorted(n for n in names if is_label_pg(n))
        if reserved:
            raise ValueError(
                f"load_dxf: layer name(s) {reserved} start with the "
                f"reserved {LABEL_PREFIX!r} prefix.  Rename the layer(s), "
                f"or pass create_physical_groups=False."
            )
        physical = self._model._parent.physical
        held = sorted(
            (n, d) for n in names for d in (0, 2, 3)
            if physical.get_tag(d, n) is not None
        )
        if held:
            raise ValueError(
                f"load_dxf: layer physical groups are curves (dim=1), but "
                f"(name, dim) {held} already name physical groups at "
                f"another dim.  A physical-group name maps to one "
                f"dimension: rename the layer(s) or those groups, or pass "
                f"create_physical_groups=False."
            )

    def _match_tolerance(self) -> float:
        """Per-axis distance under which two coordinates are the same.

        Floored at ten times OCC's bbox pad (below it nothing matches,
        #1532) and at the user's ``point_tolerance`` (closer points were
        merged into one gmsh point), plus a relative term for round-off
        at large coordinates (a plan in mm at 1e5 adds 1e-3).
        """
        extent = max(
            (abs(c) for rec in self._records for c in rec.bbox), default=0.0,
        )
        return max(10.0 * _OCC_BBOX_PAD, self._tol, 1e-8 * extent)

    def _rebuild_layers(self) -> dict[str, dict[int, list[Tag]]]:
        """Map every curve now in the model to the layer(s) of the DXF
        entities it came from, or to ``_unmatched``.

        ``removeAllDuplicates`` renumbers tags, so the match is geometric:
        endpoint coordinates (exact) and the bounding box (padded by OCC,
        hence the tolerance) against the records the converters kept.
        Records are bucketed by endpoint cell so a curve checks only the
        records sharing one endpoint cell or a neighbour.
        """
        tol = self._match_tolerance()
        cells: dict[tuple[int, int, int], list[int]] = {}
        for i, rec in enumerate(self._records):
            for p in rec.ends:
                cells.setdefault(self._cell(p, tol), []).append(i)
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

        hits_per_record = [0] * len(self._records)
        multi_layer: list[tuple[Tag, list[str]]] = []
        layers: dict[str, dict[int, list[Tag]]] = {}
        for dim, tag in gmsh.model.getEntities(1):
            x0, y0, z0, x1, y1, z1 = (float(c) for c in gmsh.model.getBoundingBox(dim, tag))
            bbox: _Bbox = (x0, y0, z0, x1, y1, z1)
            bnd = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False)
            raw_ends: list[_Point3] = []
            for _, ptag in bnd:
                px, py, pz = (float(c) for c in gmsh.model.getValue(0, ptag, []))
                raw_ends.append((px, py, pz))
            ends = _distinct_points(tuple(raw_ends), tol)
            candidates: set[int] = set()
            if ends:
                cx, cy, cz = self._cell(ends[0], tol)
                for dx, dy, dz in offsets:
                    candidates.update(cells.get((cx + dx, cy + dy, cz + dz), ()))
            hits = sorted(
                i for i in candidates if self._records[i].matches(ends, bbox, tol)
            )
            for i in hits:
                hits_per_record[i] += 1
            names = sorted({self._records[i].layer for i in hits}) or ["_unmatched"]
            if len(names) > 1:
                multi_layer.append((tag, names))
            self._model._register(dim, tag, None, 'dxf')
            for name in names:
                layers.setdefault(name, {}).setdefault(1, []).append(tag)
        for dim, tag in gmsh.model.getEntities(0):
            self._model._register(dim, tag, None, 'dxf_point')

        self._warn_mismatches(hits_per_record, multi_layer, layers)
        return layers

    @staticmethod
    def _cell(p: _Point3, tol: float) -> tuple[int, int, int]:
        # Two points within tol per axis differ by at most one cell per axis.
        return (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))

    def _warn_mismatches(
        self,
        hits_per_record: list[int],
        multi_layer: list[tuple[Tag, list[str]]],
        layers: dict[str, dict[int, list[Tag]]],
    ) -> None:
        lost: dict[str, int] = {}
        for rec, n in zip(self._records, hits_per_record):
            if n == 0:
                lost[rec.layer] = lost.get(rec.layer, 0) + 1
        if lost:
            empty = sorted(name for name in lost if name not in layers)
            warnings.warn(WarnDxfLayerMismatch(
                f"load_dxf: {sum(lost.values())} DXF curve(s) matched none "
                f"of the imported curves after duplicate removal, per layer "
                f"{lost}; layer(s) {empty} matched nothing and get no "
                f"physical group.  Unmatched imported curves, if any, are "
                f"in the '_unmatched' group."
            ), stacklevel=5)
        if multi_layer:
            shown = ", ".join(f"curve {t} -> {names}" for t, names in multi_layer[:5])
            more = f" (+{len(multi_layer) - 5} more)" if len(multi_layer) > 5 else ""
            warnings.warn(WarnDxfLayerMismatch(
                f"load_dxf: {len(multi_layer)} imported curve(s) match DXF "
                f"entities on several layers (duplicates drawn on more than "
                f"one layer); each joins every such layer group: {shown}{more}."
            ), stacklevel=5)


class _IO:
    """IO sub-composite — import/export IGES, STEP, DXF, MSH."""

    # The importers mutate the model and carry the freeze guard.  The
    # exporters only read it out, so they take the kernel guard and
    # stay legal on a live post-extraction session, which still has a
    # model worth writing.
    _H5_ALTERNATIVE = (
        "g.save(path) to persist the chain-phase model as model.h5; to "
        "emit a CAD/mesh file, run the export in the source session "
        "that still owns the gmsh geometry"
    )

    def __init__(self, model: "Model") -> None:
        self._model = model

    def _require_kernel(self, verb: str) -> None:
        """Refuse an export that would read the absent gmsh kernel."""
        from ._compose_errors import raise_if_no_live_kernel
        raise_if_no_live_kernel(
            self._model._parent, verb, alternative=self._H5_ALTERNATIVE,
        )

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
        Core import helper shared by ``load_iges``, ``load_step`` and
        ``load_brep``.

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
                # The shapes only: OCC heals a list one entity at a
                # time, so re-healing every sub-entity is slow and adds
                # nothing.
                self.heal_shapes(
                    _top_level_entities(list(raw)),
                    tolerance=heal_tol, sync=True,
                )

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
            # importShapes repeats shared sub-entities (an edge once per
            # face it bounds): keep each tag once.
            result = {}
            for dim, tag in dict.fromkeys(raw):
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
            # The imported shapes carry the label, not their
            # sub-entities (one label per dim they span).
            labels_comp = getattr(self._model._parent, 'labels', None)
            if labels_comp is not None:
                shapes: dict[int, list[Tag]] = {}
                for d, t in _top_level_entities(
                    [(d, t) for d, ts in result.items() for t in ts]
                ):
                    shapes.setdefault(d, []).append(t)
                for dim, tags in shapes.items():
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
        highest_dim_only: bool = False,
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
            False (default): import every shape in the file and return
            every entity (volumes, faces, edges, vertices), so free
            lower-dimension shapes beside the top dimension (beam /
            column curves next to a shell) come in too.  True: import
            only the highest dimension (volumes for solids, surfaces for
            surface models); free lower-dimension shapes are dropped.
            ``heal=`` and ``label=`` act on the imported shapes either
            way, not on their sub-entities.
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
        highest_dim_only: bool = False,
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
            False (default): import every shape in the file and return
            every entity (volumes, faces, edges, vertices), so free
            lower-dimension shapes beside the top dimension (beam /
            column curves next to a shell) come in too.  True: import
            only the highest dimension (volumes for solids, surfaces for
            surface models); free lower-dimension shapes are dropped.
            ``heal=`` and ``label=`` act on the imported shapes either
            way, not on their sub-entities.
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

    def load_brep(
        self,
        file_path       : Path | str,
        *,
        highest_dim_only: bool = False,
        sync            : bool = True,
        heal            : bool | float | str = False,
        dedupe          : bool | float = False,
        fuse            : bool = False,
        label           : str | None = None,
    ) -> dict[int, list[Tag]]:
        """
        Import an OpenCASCADE BREP file into the current model.

        BREP is OCC's native format (what STKO and other OCC-based
        pre-processors write), so it keeps exact curve types and the
        shared topology that STEP / IGES exports can lose.  Same
        signature, options and return shape as :meth:`load_step`.

        Parameters
        ----------
        highest_dim_only, heal, dedupe, fuse, label
            As in :meth:`load_step`.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` indexed by dimension.

        Example
        -------
        ::

            # shell + free columns: both come in
            imported = g.model.io.load_brep("frame.brep")
            slabs_and_walls = imported[2]
        """
        return self._import_shapes(
            Path(file_path), 'brep', highest_dim_only, sync,
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
            OCC heals an explicit list one entity at a time, so only
            the default sews faces together.
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

        Notes
        -----
        Sewing everything drops every free curve and point (they belong
        to no face): in a frame + shell model that deletes the columns.
        So when ``tags=None`` and the model has faces plus free curves
        or points, the call heals without sewing and emits
        :class:`WarnGeomHealSkipsSewing`.

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
            free = _free_curves_and_points() if sew_faces else []
            if free and gmsh.model.getEntities(2):
                n_curves = sum(1 for d, _ in free if d == 1)
                warnings.warn(WarnGeomHealSkipsSewing(
                    f"heal_shapes(): sewing faces would delete the model's "
                    f"{n_curves} free curve(s) and {len(free) - n_curves} "
                    f"free point(s); healing without sewing instead. To sew, "
                    f"heal the faces before adding the free curves."
                ), stacklevel=2)
                sew_faces = False

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
        # Guarded before the write: on a kernel-less session gmsh.write
        # does not fail, it emits whatever empty model it holds — a
        # plausible-looking CAD file with nothing in it.
        self._require_kernel("g.model.io.save_iges()")
        file_path = Path(file_path).with_suffix('.iges')
        gmsh.write(str(file_path))
        self._model._log(f"saved IGES -> {file_path}")

    def save_step(self, file_path: Path | str) -> None:
        """
        Export the current model to STEP.

        The ``.step`` extension is appended automatically if omitted.
        """
        self._require_kernel("g.model.io.save_step()")
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
        self._require_kernel("g.model.io.save_dxf()")
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
        self._require_kernel("g.model.io.save_msh()")
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
