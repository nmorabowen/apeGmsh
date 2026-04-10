from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

from ._helpers import Tag, DimTag, TagsLike

if TYPE_CHECKING:
    from .Model import Model


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
    ) -> dict[int, list[Tag]]:
        """
        Core import helper shared by ``load_iges`` and ``load_step``.

        Calls ``gmsh.model.occ.importShapes``, captures the returned
        (dim, tag) pairs, registers every imported entity, and returns a
        dimension-indexed dict so callers can address entities immediately.

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` — e.g. ``{3: [1, 2], 2: [5, 6, 7]}``
        """
        raw: list[tuple[int, int]] = gmsh.model.occ.importShapes(
            str(file_path),
            highestDimOnly=highest_dim_only,
        )
        if sync:
            gmsh.model.occ.synchronize()

        result: dict[int, list[Tag]] = {}
        for dim, tag in raw:
            self._model._register(dim, tag, None, kind)
            result.setdefault(dim, []).append(tag)

        dim_summary = {d: len(ts) for d, ts in result.items()}
        self._model._log(f"loaded {kind.upper()} ← {file_path.name}  {dim_summary}")
        return result

    def load_iges(
        self,
        file_path       : Path | str,
        *,
        highest_dim_only: bool = True,
        sync            : bool = True,
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
            Path(file_path), 'iges', highest_dim_only, sync
        )

    def load_step(
        self,
        file_path       : Path | str,
        *,
        highest_dim_only: bool = True,
        sync            : bool = True,
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

        Returns
        -------
        dict[int, list[Tag]]
            ``{dim: [tag, ...]}`` indexed by dimension.

        Example
        -------
        ::

            imported = g.model.io.load_step("assembly.step")
            bodies   = imported[3]
            g.model.transforms.translate(bodies, 0, 0, 50)   # lift the whole import
        """
        return self._import_shapes(
            Path(file_path), 'step', highest_dim_only, sync
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
            if (d, t) not in self._model._registry:
                self._model._register(d, t, None, 'healed')
        self._model._log(
            f"heal_shapes(tol={tolerance}) → {len(out)} entities output"
        )
        return self

    def save_iges(self, file_path: Path | str) -> None:
        """
        Export the current model to IGES.

        The ``.iges`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.iges')
        gmsh.write(str(file_path))
        self._model._log(f"saved IGES → {file_path}")

    def save_step(self, file_path: Path | str) -> None:
        """
        Export the current model to STEP.

        The ``.step`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.step')
        gmsh.write(str(file_path))
        self._model._log(f"saved STEP → {file_path}")

    # ------------------------------------------------------------------
    # DXF (AutoCAD) — parsed with ezdxf, geometry built via OCC kernel
    # ------------------------------------------------------------------

    @staticmethod
    def _dxf_point_key(
        x: float, y: float, z: float, tol: float,
    ) -> tuple[int, int, int]:
        """Discretise coordinates into a grid cell for O(1) dedup."""
        inv = 1.0 / tol
        return (round(x * inv), round(y * inv), round(z * inv))

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
        try:
            import ezdxf
        except ImportError:
            raise ImportError(
                "ezdxf is required for DXF import.  "
                "Install it with:  pip install ezdxf"
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"DXF file not found: {file_path}")

        doc = ezdxf.readfile(str(file_path))
        msp = doc.modelspace()

        # -- Point deduplication ------------------------------------------
        tol = point_tolerance
        _pt_cache: dict[tuple[int, int, int], Tag] = {}

        def _get_or_add_point(x: float, y: float, z: float) -> Tag:
            key = self._dxf_point_key(x, y, z, tol)
            if key in _pt_cache:
                return _pt_cache[key]
            tag = gmsh.model.occ.addPoint(x, y, z)
            _pt_cache[key] = tag
            self._model._register(0, tag, None, 'dxf_point')
            return tag

        # -- Entity conversion by type ------------------------------------
        # We store each curve's geometry fingerprint (sorted bounding-box
        # coords) → layer name so we can rebuild the mapping after
        # removeAllDuplicates() potentially renumbers tags.

        def _bbox_key(
            x0: float, y0: float, z0: float,
            x1: float, y1: float, z1: float,
        ) -> tuple[float, ...]:
            """Canonical bounding box: min coords then max coords."""
            return (
                round(min(x0, x1), 8), round(min(y0, y1), 8),
                round(min(z0, z1), 8), round(max(x0, x1), 8),
                round(max(y0, y1), 8), round(max(z0, z1), 8),
            )

        # fingerprint → layer name  (for curves, dim=1)
        _geom_to_layer: dict[tuple[float, ...], str] = {}
        # point key → layer name  (for dim=0 DXF POINT entities)
        _pt_to_layer: dict[tuple[int, int, int], str] = {}

        for entity in msp:
            layer = entity.dxf.layer
            etype = entity.dxftype()

            if etype == 'POINT':
                pt = entity.dxf.location
                _get_or_add_point(pt.x, pt.y, pt.z)
                key = self._dxf_point_key(pt.x, pt.y, pt.z, tol)
                _pt_to_layer[key] = layer

            elif etype == 'LINE':
                s = entity.dxf.start
                e = entity.dxf.end
                p1 = _get_or_add_point(s.x, s.y, s.z)
                p2 = _get_or_add_point(e.x, e.y, e.z)
                gmsh.model.occ.addLine(p1, p2)
                _geom_to_layer[_bbox_key(s.x, s.y, s.z, e.x, e.y, e.z)] = layer

            elif etype == 'ARC':
                c = entity.dxf.center
                r = entity.dxf.radius
                a1 = math.radians(entity.dxf.start_angle)
                a2 = math.radians(entity.dxf.end_angle)
                if a2 <= a1:
                    a2 += 2.0 * math.pi
                gmsh.model.occ.addCircle(
                    c.x, c.y, c.z, r, angle1=a1, angle2=a2,
                )
                # Compute arc endpoints for the fingerprint
                sx = c.x + r * math.cos(a1)
                sy = c.y + r * math.sin(a1)
                ex = c.x + r * math.cos(a2)
                ey = c.y + r * math.sin(a2)
                _geom_to_layer[_bbox_key(sx, sy, c.z, ex, ey, c.z)] = layer

            elif etype == 'CIRCLE':
                c = entity.dxf.center
                r = entity.dxf.radius
                gmsh.model.occ.addCircle(c.x, c.y, c.z, r)
                _geom_to_layer[_bbox_key(
                    c.x - r, c.y - r, c.z, c.x + r, c.y + r, c.z,
                )] = layer

            elif etype in ('LWPOLYLINE', 'POLYLINE'):
                pts: list[Tag] = []
                if etype == 'LWPOLYLINE':
                    vertices = list(entity.get_points(format='xyz'))
                else:
                    vertices = [
                        (v.dxf.location.x, v.dxf.location.y,
                         v.dxf.location.z)
                        for v in entity.vertices
                    ]
                for vx, vy, vz in vertices:
                    pts.append(_get_or_add_point(vx, vy, vz))

                is_closed = (
                    getattr(entity.dxf, 'flags', 0) & 1
                    if etype == 'POLYLINE' else entity.closed
                )
                vert_pairs = list(zip(vertices, vertices[1:]))
                if is_closed and len(vertices) > 2:
                    vert_pairs.append((vertices[-1], vertices[0]))

                pt_pairs = list(zip(pts, pts[1:]))
                if is_closed and len(pts) > 2:
                    pt_pairs.append((pts[-1], pts[0]))

                for (v_s, v_e), (p1, p2) in zip(vert_pairs, pt_pairs):
                    gmsh.model.occ.addLine(p1, p2)
                    _geom_to_layer[_bbox_key(
                        v_s[0], v_s[1], v_s[2],
                        v_e[0], v_e[1], v_e[2],
                    )] = layer

            elif etype == 'SPLINE':
                ctrl_pts: list[Tag] = []
                for cp in entity.control_points:
                    ctrl_pts.append(
                        _get_or_add_point(
                            cp[0], cp[1],
                            cp[2] if len(cp) > 2 else 0.0,
                        )
                    )
                if len(ctrl_pts) >= 2:
                    gmsh.model.occ.addBSpline(ctrl_pts)
                    cps = entity.control_points
                    xs = [c[0] for c in cps]
                    ys = [c[1] for c in cps]
                    zs = [c[2] if len(c) > 2 else 0.0 for c in cps]
                    _geom_to_layer[_bbox_key(
                        min(xs), min(ys), min(zs),
                        max(xs), max(ys), max(zs),
                    )] = layer

            else:
                self._model._log(f"DXF: skipped unsupported entity {etype} "
                          f"on layer '{layer}'")

        # -- Merge duplicate points & synchronise --------------------------
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # -- Rebuild layer mapping from surviving entities -----------------
        layers: dict[str, dict[int, list[Tag]]] = {}

        for dim, tag in gmsh.model.getEntities(1):
            bb = gmsh.model.getBoundingBox(dim, tag)
            key = _bbox_key(*bb)
            layer_name = _geom_to_layer.get(key)
            if layer_name:
                self._model._register(dim, tag, None, 'dxf')
                layers.setdefault(layer_name, {}).setdefault(1, []).append(tag)
            else:
                # Fallback: assign to "_unmatched"
                self._model._register(dim, tag, None, 'dxf')
                layers.setdefault("_unmatched", {}).setdefault(1, []).append(tag)

        for dim, tag in gmsh.model.getEntities(0):
            self._model._register(dim, tag, None, 'dxf_point')

        # -- Physical groups from layers ----------------------------------
        if create_physical_groups:
            for layer_name, dim_tags in layers.items():
                for dim, tags in dim_tags.items():
                    if tags:
                        gmsh.model.addPhysicalGroup(
                            dim, tags, name=layer_name,
                        )

        # -- Summary ------------------------------------------------------
        layer_summary = {
            name: {d: len(ts) for d, ts in ents.items()}
            for name, ents in layers.items()
        }
        self._model._log(f"loaded DXF ← {file_path.name}  layers={layer_summary}")
        return layers

    def save_dxf(self, file_path: Path | str) -> None:
        """
        Export the current model to DXF.

        The ``.dxf`` extension is appended automatically if omitted.
        """
        file_path = Path(file_path).with_suffix('.dxf')
        gmsh.write(str(file_path))
        self._model._log(f"saved DXF → {file_path}")

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
        self._model._log(f"saved MSH → {file_path}")

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
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"MSH file not found: {file_path}")

        gmsh.merge(str(file_path))

        result: dict[int, list[Tag]] = {}
        for d in range(4):
            for dim, tag in gmsh.model.getEntities(d):
                result.setdefault(dim, []).append(tag)

        dim_summary = {d: len(ts) for d, ts in result.items()}
        self._model._log(f"loaded MSH ← {file_path.name}  {dim_summary}")
        return result
