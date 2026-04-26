from __future__ import annotations
import math
from typing import TYPE_CHECKING, Literal

import gmsh
import numpy as np
from numpy import ndarray

from ._helpers import Tag
from .Labels import pg_preserved, cleanup_label_pgs

if TYPE_CHECKING:
    from .Model import Model


# Unit vectors for the three coordinate axes — used by
# :meth:`_Geometry.add_axis_cutting_plane`.  Module-level constants so
# they are not rebuilt on every call.  Callers that rotate the
# returned vector MUST ``.copy()`` first.
_AXIS_UNIT_VEC: dict[str, ndarray] = {
    'x': np.array([1.0, 0.0, 0.0]),
    'y': np.array([0.0, 1.0, 0.0]),
    'z': np.array([0.0, 0.0, 1.0]),
}


class _Geometry:
    """Points, curves, surfaces, and solid primitive creation methods."""

    def __init__(self, model: "Model") -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Points  (dim = 0)
    # ------------------------------------------------------------------

    def add_point(
        self,
        x: float, y: float, z: float,
        *,
        mesh_size: float      = 0.0,
        lc       : float | None = None,
        label    : str | None = None,
        sync     : bool       = True,
    ) -> Tag:
        """
        Add a single point.

        Parameters
        ----------
        x, y, z   : coordinates
        mesh_size : target element size at this point (0 = use global size)
        lc        : alias for *mesh_size* (Gmsh characteristic length)

        Returns
        -------
        int tag of the new point.
        """
        if lc is not None:
            mesh_size = lc
        tag = gmsh.model.occ.addPoint(x, y, z, meshSize=mesh_size)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_point({x}, {y}, {z}) -> tag {tag}")
        return self._model._register(0, tag, label, 'point')

    # ------------------------------------------------------------------
    # Curves  (dim = 1)
    # ------------------------------------------------------------------

    def add_line(
        self,
        start: Tag,
        end  : Tag,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a straight line segment between two existing points.

        Parameters
        ----------
        start, end : point tags
        """
        tag = gmsh.model.occ.addLine(start, end)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_line({start} -> {end}) -> tag {tag}")
        return self._model._register(1, tag, label, 'line')

    def add_imperfect_line(
        self,
        start: Tag,
        end  : Tag,
        *,
        magnitude : float = 0.0,
        direction : tuple[float, float, float],
        shape     : Literal['kink', 'sine', 'multi_mode'] = 'kink',
        n_segments: int = 8,
        modes     : list[tuple[int, float]] | None = None,
        label     : str | None = None,
        sync      : bool = True,
    ) -> list[Tag]:
        """
        Add a line with a built-in geometric imperfection.

        Used for seeding initial out-of-straightness on columns, struts,
        or braces before running a corotational / nonlinear buckling
        analysis. The imperfection is baked into the **geometry** as a
        polyline through intermediate points — there is no solver-side
        perturbation. The resulting line segments are a drop-in
        replacement for a single :meth:`add_line` call and can be
        grouped into a single physical group via
        ``m.physical.add_curve(tags=[...])``.

        Parameters
        ----------
        start, end : point tags
            Endpoints of the imperfect line (straight-line length L).
        magnitude : float
            Peak perpendicular offset of the imperfection envelope.
            Typical engineering choices are ``L/500`` to ``L/1000``.
            Ignored when ``shape='multi_mode'`` — amplitudes come from
            the per-mode entries of ``modes``.
        direction : (dx, dy, dz)
            Direction hint for the offset. The vector is projected onto
            the plane perpendicular to the line axis and normalized.
            Only the perpendicular component matters; e.g. for a
            diagonal brace you can pass ``(0, 1, 0)`` to request an
            out-of-plane-Y offset and the method takes care of
            orthogonality. Raises ``ValueError`` if the vector is
            parallel to the line axis.
        shape : str
            * ``'kink'``   — single midspan intermediate point, two
              line segments. Produces a triangular bent; pedagogically
              clean but not physically smooth.
            * ``'sine'``   — half-sine envelope
              ``y(s) = magnitude · sin(π·s/L)`` discretized into
              ``n_segments`` pieces. Matches the first Euler buckling
              mode exactly; recommended for quantitative work.
            * ``'multi_mode'`` — superposition of multiple sinusoidal
              modes, ``y(s) = Σ_k  a_k · sin(k·π·s/L)`` where
              ``(k, a_k)`` pairs come from ``modes``. Used to seed
              more than one buckling mode at once.
        n_segments : int
            Number of line pieces the imperfect line is split into.
            Only meaningful for ``'sine'`` (default 8) and
            ``'multi_mode'`` (default 8). ``'kink'`` always uses
            exactly 2 segments regardless of this value.
        modes : list[(int, float)], optional
            Required when ``shape='multi_mode'``. Each entry is a
            ``(mode_number, amplitude)`` pair. The mode number ``k``
            must be a positive integer; the amplitude is absolute (not
            relative to ``magnitude``).
        label : str, optional
            Label applied to *all* resulting line segments. The
            intermediate interior points remain anonymous (no labels).
        sync : bool
            Whether to synchronise the OCC kernel after creation.

        Returns
        -------
        list[Tag]
            Line tags in geometric order from ``start`` to ``end``.
            Pass the list directly to ``m.physical.add_curve``.

        Examples
        --------
        Kinked brace with an L/1000 midspan offset in the global-Y
        direction::

            tags = m.model.geometry.add_imperfect_line(
                p_base, p_top,
                magnitude=L_brace/1000,
                direction=(0, 1, 0),
                shape='kink',
                label='brace',
            )

        Half-sine imperfection discretised into 16 segments::

            tags = m.model.geometry.add_imperfect_line(
                p1, p2,
                magnitude=L/500,
                direction=(1, 0, 0),
                shape='sine',
                n_segments=16,
                label='column',
            )

        First + third mode seeding::

            tags = m.model.geometry.add_imperfect_line(
                p1, p2,
                direction=(0, 0, 1),
                shape='multi_mode',
                modes=[(1, L/1000), (3, L/5000)],
                n_segments=24,
            )
        """
        # ── Resolve endpoint coordinates ────────────────────────
        p0 = np.asarray(
            gmsh.model.getValue(0, int(start), []), dtype=float)
        p1 = np.asarray(
            gmsh.model.getValue(0, int(end),   []), dtype=float)
        axis_vec = p1 - p0
        L = float(np.linalg.norm(axis_vec))
        if L <= 0.0:
            raise ValueError(
                f"add_imperfect_line: zero-length axis "
                f"({start} -> {end}); endpoints coincide.")
        axis_unit = axis_vec / L

        # ── Project direction perpendicular to axis ─────────────
        dir_arr = np.asarray(direction, dtype=float)
        if dir_arr.shape != (3,):
            raise ValueError(
                f"direction must be a length-3 vector, got shape "
                f"{dir_arr.shape}")
        perp = dir_arr - float(np.dot(dir_arr, axis_unit)) * axis_unit
        perp_mag = float(np.linalg.norm(perp))
        if perp_mag < 1e-12 * max(L, 1.0):
            raise ValueError(
                f"direction {tuple(direction)} is (nearly) parallel to "
                f"the line axis {tuple(axis_unit)}; pick a direction "
                f"with a perpendicular component.")
        perp_unit = perp / perp_mag

        # ── Build the (arc-length fraction, offset) pairs ───────
        if shape == 'kink':
            s_list     = [0.5]
            offset_list = [magnitude]
            n_interior = 1
        elif shape == 'sine':
            if n_segments < 2:
                raise ValueError(
                    f"n_segments must be >= 2 for shape='sine', "
                    f"got {n_segments}")
            s_list = [i / n_segments for i in range(1, n_segments)]
            offset_list = [
                magnitude * math.sin(math.pi * s) for s in s_list
            ]
            n_interior = n_segments - 1
        elif shape == 'multi_mode':
            if not modes:
                raise ValueError(
                    "shape='multi_mode' requires a non-empty 'modes' "
                    "list of (mode_number, amplitude) pairs.")
            if n_segments < 2:
                raise ValueError(
                    f"n_segments must be >= 2 for shape='multi_mode', "
                    f"got {n_segments}")
            s_list = [i / n_segments for i in range(1, n_segments)]
            offset_list = []
            for s in s_list:
                total = 0.0
                for k, a in modes:
                    if k <= 0 or int(k) != k:
                        raise ValueError(
                            f"mode number must be a positive integer, "
                            f"got {k}")
                    total += float(a) * math.sin(int(k) * math.pi * s)
                offset_list.append(total)
            n_interior = n_segments - 1
        else:
            raise ValueError(
                f"shape must be 'kink', 'sine', or 'multi_mode'; "
                f"got {shape!r}")

        # ── Create intermediate points + segment lines ──────────
        line_tags: list[Tag] = []
        prev = int(start)
        for s, off in zip(s_list, offset_list):
            pos = p0 + s * (p1 - p0) + off * perp_unit
            pt = gmsh.model.occ.addPoint(
                float(pos[0]), float(pos[1]), float(pos[2]))
            ln = gmsh.model.occ.addLine(prev, pt)
            line_tags.append(ln)
            prev = pt

        # Final segment from last interior point to the end.
        ln = gmsh.model.occ.addLine(prev, int(end))
        line_tags.append(ln)

        if sync:
            gmsh.model.occ.synchronize()

        # Register metadata for every segment (kind='imperfect_line').
        # Labels are skipped here: if we passed ``label`` to ``_register``
        # it would call ``labels.add`` once per segment and emit a
        # "duplicate label" warning on every call after the first. We
        # batch them in a single ``labels.add`` below so the label PG
        # contains all segments from the start.
        for ln in line_tags:
            self._model._register(1, ln, None, 'imperfect_line')

        # Batch-label all segments under one name (if requested).
        if label and getattr(self._model._parent, '_auto_pg_from_label', False):
            labels_comp = getattr(self._model._parent, 'labels', None)
            if labels_comp is not None:
                try:
                    labels_comp.add(1, list(line_tags), name=label)
                except Exception as exc:
                    import warnings as _warn
                    _warn.warn(
                        f"Label {label!r} (dim=1, tags={line_tags}) could "
                        f"not be created: {exc}",
                        stacklevel=2,
                    )

        self._model._log(
            f"add_imperfect_line({start} -> {end}, shape={shape!r}, "
            f"n_interior={n_interior}, magnitude={magnitude}) "
            f"-> {len(line_tags)} segment(s) {line_tags}")
        return list(line_tags)

    def replace_line(
        self,
        line_tag  : Tag,
        *,
        magnitude : float = 0.0,
        direction : tuple[float, float, float],
        shape     : Literal['kink', 'sine', 'multi_mode'] = 'kink',
        n_segments: int = 8,
        modes     : list[tuple[int, float]] | None = None,
        sync      : bool = True,
    ) -> list[Tag]:
        """
        Retrofit an existing straight line with a geometric imperfection.

        Use this when you built the frame with plain :meth:`add_line`
        calls and then want to introduce an imperfection on a specific
        member without rebuilding the whole geometry. The method:

        1. Validates that ``line_tag`` points to a straight line
           (``kind='line'`` in the model metadata; arcs, splines, and
           already-imperfect lines are rejected).
        2. Looks up the two endpoint points from the line's boundary.
        3. Records every physical group (user-facing PGs **and** label
           PGs of the form ``_label:…``) that contains the line.
        4. Deletes the old curve — endpoints are preserved because
           other geometry likely references them.
        5. Calls :meth:`add_imperfect_line` between the same endpoints
           to build the new polyline.
        6. Re-wires every recorded physical group: the old line tag is
           swapped out and the new segment tags are spliced in, so any
           PG that used to reference the straight line now references
           the full imperfect polyline.

        Parameters
        ----------
        line_tag : int
            Tag of the existing straight line to replace.
        magnitude, direction, shape, n_segments, modes :
            Same semantics as :meth:`add_imperfect_line`.
        sync : bool
            Whether to synchronise the OCC kernel at the end.

        Returns
        -------
        list[Tag]
            New line tags in geometric order. Same layout as
            :meth:`add_imperfect_line` would return.
        """
        line_tag = int(line_tag)

        # 1. Reject anything that isn't a plain straight line.
        meta = self._model._metadata.get((1, line_tag), {})
        kind = meta.get('kind')
        if kind != 'line':
            raise ValueError(
                f"replace_line only works on straight lines created via "
                f"add_line (kind='line'), got kind={kind!r} for tag "
                f"{line_tag}. Rebuild the geometry explicitly if you "
                f"need to replace an arc, spline, or already-imperfect "
                f"line.")

        # 2. Recover endpoints via the curve's boundary.
        try:
            bnd = gmsh.model.getBoundary(
                [(1, line_tag)], oriented=False)
        except Exception as exc:
            raise ValueError(
                f"Line {line_tag} does not exist in the OCC kernel: "
                f"{exc}") from exc
        point_tags = [int(t) for (d, t) in bnd if d == 0]
        if len(point_tags) != 2:
            raise ValueError(
                f"Line {line_tag} has {len(point_tags)} boundary "
                f"point(s); expected exactly 2.")
        start_tag, end_tag = point_tags

        # 3. Record every dim-1 PG that contains this line. Capture
        # both the PG tag (for removal) and its name + entity list
        # (for re-creation) in a single scan so we don't walk the PG
        # table twice.
        old_pgs: list[tuple[int, str, list[int]]] = []
        for (pg_dim, pg_tag) in gmsh.model.getPhysicalGroups(dim=1):
            ents = [
                int(e) for e in
                gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
            ]
            if line_tag not in ents:
                continue
            try:
                name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
            except Exception:
                name = ''
            other = [e for e in ents if e != line_tag]
            old_pgs.append((int(pg_tag), name, other))

        # 4. Delete the old physical groups FIRST, then the line.
        # Removing the PGs first prevents Gmsh's internal synchronise
        # from seeing a dangling entity reference when the line is
        # deleted — otherwise it fires "Unknown entity ... in physical
        # group" warnings.
        if old_pgs:
            gmsh.model.removePhysicalGroups(
                [(1, pg_tag) for (pg_tag, _n, _o) in old_pgs])

        # Delete the line itself (recursive=False keeps the endpoints).
        gmsh.model.occ.remove([(1, line_tag)], recursive=False)
        gmsh.model.occ.synchronize()

        # Drop the old line from model metadata.
        self._model._metadata.pop((1, line_tag), None)

        # 5. Build the new polyline. Pass ``label=None`` so no label
        # PGs are auto-created — we re-create the captured PGs
        # (including the ``_label:…`` one if present) below in a
        # single pass. ``sync=True`` is important: the new line tags
        # must be committed into Gmsh's model state before we add
        # physical groups referencing them, otherwise Gmsh accepts
        # the PG at the model level but warns about "Unknown entity"
        # on the next OCC synchronise.
        new_tags = self.add_imperfect_line(
            start_tag, end_tag,
            magnitude=magnitude,
            direction=direction,
            shape=shape,
            n_segments=n_segments,
            modes=modes,
            label=None,
            sync=True,
        )

        # 6. Re-create every captured PG with (other_ents ∪ new_tags)
        # and the original name. We pass ``name=`` to ``addPhysicalGroup``
        # in one atomic call (rather than ``addPhysicalGroup`` + a
        # separate ``setPhysicalName``) to avoid a brief inconsistent-
        # -state window that some Gmsh builds complain about after a
        # recent ``removePhysicalGroups``.
        new_tag_set = set(int(t) for t in new_tags)
        for (_old_pg_tag, name, other_ents) in old_pgs:
            merged = sorted(set(other_ents) | new_tag_set)
            try:
                gmsh.model.addPhysicalGroup(
                    1, merged, tag=-1, name=name or "")
            except Exception as exc:
                import warnings as _warn
                _warn.warn(
                    f"replace_line: could not re-create PG "
                    f"{name!r}: {exc}",
                    stacklevel=2,
                )

        if sync:
            gmsh.model.occ.synchronize()

        self._model._log(
            f"replace_line({line_tag}, shape={shape!r}, "
            f"magnitude={magnitude}) -> {len(new_tags)} segment(s) "
            f"{new_tags}, re-wired {len(old_pgs)} PG(s)")
        return list(new_tags)

    def sweep(
        self,
        profile_face  : Tag,
        path_curves   : list[Tag],
        *,
        label         : str | None = None,
        cleanup       : bool       = True,
        sync          : bool       = True,
    ) -> dict:
        """
        Sweep a planar profile face along a chain of curves (a polyline
        path) to produce a 3-D solid volume.

        Wraps ``gmsh.model.occ.addWire`` + ``gmsh.model.occ.addPipe``
        and — optionally — cleans up the intermediate geometry that
        would otherwise cause trouble downstream:

        * **The profile face** that was used as the input to the pipe.
          It persists as an orphan at the first station of the sweep,
          which means when you try to identify the start cap by bbox
          you pick up two coincident surfaces and their mesh nodes
          double up.
        * **The path curves** themselves. These live along the
          centroid line of the swept solid — i.e. **inside** the
          volume — and Gmsh happily meshes them into ``line2``
          elements whose nodes sit interior to the tet mesh and are
          *not* shared with any tet4 element. Those become floating
          null-space DOFs if you emit them to OpenSees.

        With ``cleanup=True`` (the default) both are removed after the
        pipe has produced the volume, so the only surfaces left in the
        model are the ones that actually bound the solid (the two end
        caps + the ``n_path_segments * n_profile_edges`` ruled side
        surfaces).

        Parameters
        ----------
        profile_face : int
            Tag of a planar surface that serves as the cross-section.
            Most commonly built with ``addCurveLoop`` + ``addPlaneSurface``
            on a closed polyline of vertices. Must be perpendicular —
            at least roughly — to the start of the path; Gmsh orients
            the local frame automatically using the Frenet trihedron.
        path_curves : list[int]
            Ordered list of curve tags that form the path the profile
            is swept along. Typically the return value of
            :meth:`add_imperfect_line` or :meth:`replace_line`.
        label : str, optional
            If given, the resulting volume is labelled. End caps and
            side surfaces remain unlabelled — run the usual
            ``select_surfaces(in_box=…)`` + ``to_physical`` pass to
            group them explicitly.
        cleanup : bool
            Remove the original profile face and path curves after
            the pipe is built. Defaults to True. Set to False if you
            want to preserve the profile/path for downstream use
            (e.g. another sweep on a branched path).
        sync : bool
            Whether to synchronise the OCC kernel at the end.

        Returns
        -------
        dict
            ``{'volume': tag, 'start_cap': tag, 'end_cap': tag}``.
            The caps are identified by scanning every new dim-2
            entity's bounding box for one whose ``x_min == x_max ==
            path_endpoint_x``. If either cap cannot be identified its
            entry is ``None``.

        Examples
        --------
        Swept solid I-beam with a half-sine imperfection in the
        weak-axis direction::

            # 1. Imperfect path
            p0 = g.model.geometry.add_point(0, 0, 0, lc=200)
            p1 = g.model.geometry.add_point(L, 0, 0, lc=200)
            path = g.model.geometry.replace_line(
                g.model.geometry.add_line(p0, p1),
                magnitude=L/1000, direction=(0, 1, 0),
                shape='sine', n_segments=16,
            )

            # 2. Rectangular profile at x = 0
            corners = [
                (0, -t/2, -h/2), (0, +t/2, -h/2),
                (0, +t/2, +h/2), (0, -t/2, +h/2),
            ]
            pts = [gmsh.model.occ.addPoint(*c) for c in corners]
            lns = [gmsh.model.occ.addLine(pts[i], pts[(i+1) % 4])
                   for i in range(4)]
            loop = gmsh.model.occ.addCurveLoop(lns)
            profile = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()

            # 3. Sweep
            swept = g.model.geometry.sweep(profile, path, label='beam')
            # swept['volume']    — the solid tag
            # swept['start_cap'] — surface tag at path start
            # swept['end_cap']   — surface tag at path end
        """
        # Snapshot the set of existing volumes + surfaces so we can
        # identify the ones the pipe creates.
        before_vols  = set(int(t) for (_d, t) in gmsh.model.getEntities(3))
        before_surfs = set(int(t) for (_d, t) in gmsh.model.getEntities(2))

        # Start position: centroid of the profile face (it sits at the
        # start of the path). End position: walk the path curves and
        # collect every endpoint's 3-D coord, then pick the one
        # farthest from the start. This avoids having to know how
        # Gmsh orients the boundary points of each curve.
        prof_com = np.asarray(
            gmsh.model.occ.getCenterOfMass(2, int(profile_face)),
            dtype=float)
        start_xyz = prof_com

        all_path_pts: list[np.ndarray] = []
        for t in path_curves:
            try:
                bnd = gmsh.model.getBoundary(
                    [(1, int(t))], oriented=False)
            except Exception:
                continue
            for (d, pt) in bnd:
                if d == 0:
                    all_path_pts.append(np.asarray(
                        gmsh.model.getValue(0, int(pt), []),
                        dtype=float))
        if all_path_pts:
            # End of path = point farthest from start_xyz.
            end_xyz = max(
                all_path_pts,
                key=lambda p: float(np.linalg.norm(p - start_xyz)))
        else:
            end_xyz = start_xyz

        # Build the wire + pipe.
        wire = gmsh.model.occ.addWire(
            [int(t) for t in path_curves], checkClosed=False)
        gmsh.model.occ.addPipe(
            [(2, int(profile_face))], wire)
        gmsh.model.occ.synchronize()

        # Cleanup — remove the original profile face (recursive, so
        # its boundary lines/points go too) and the path curves that
        # live along the volume centroid.
        if cleanup:
            try:
                gmsh.model.occ.remove(
                    [(2, int(profile_face))], recursive=True)
            except Exception:
                pass
            try:
                gmsh.model.occ.remove(
                    [(1, int(t)) for t in path_curves], recursive=False)
            except Exception:
                pass
            gmsh.model.occ.synchronize()

        # Find the new volume and end caps.
        after_vols = [
            int(t) for (_d, t) in gmsh.model.getEntities(3)
            if int(t) not in before_vols
        ]
        volume_tag = after_vols[0] if after_vols else None

        # Find the caps by projecting each new surface centroid onto
        # the path direction and taking the extremes. The "new"
        # surfaces that are actual end caps will project to approx.
        # 0 (start) and |end - start| (end); the ruled side surfaces
        # will project to values in between.
        TOL = 0.1
        start_cap = None
        end_cap   = None
        path_vec  = end_xyz - start_xyz
        path_len  = float(np.linalg.norm(path_vec))
        if path_len > TOL:
            path_unit = path_vec / path_len
            cands: list[tuple[float, int]] = []
            for (d, t) in gmsh.model.getEntities(2):
                if int(t) in before_surfs:
                    continue
                try:
                    com = np.asarray(
                        gmsh.model.occ.getCenterOfMass(2, int(t)),
                        dtype=float)
                except Exception:
                    continue
                proj = float(np.dot(com - start_xyz, path_unit))
                cands.append((proj, int(t)))
            if cands:
                cands.sort()
                start_cap = cands[0][1]
                end_cap   = cands[-1][1]

        if volume_tag is not None and label:
            self._model._register(3, volume_tag, label, 'swept_solid')

        if sync:
            gmsh.model.occ.synchronize()

        self._model._log(
            f"sweep(profile={profile_face}, n_path={len(path_curves)}) "
            f"-> volume={volume_tag}, start_cap={start_cap}, "
            f"end_cap={end_cap}")

        return {
            'volume':    volume_tag,
            'start_cap': start_cap,
            'end_cap':   end_cap,
        }

    def add_arc(
        self,
        start : Tag,
        center: Tag,
        end   : Tag,
        *,
        label : str | None = None,
        sync  : bool       = True,
    ) -> Tag:
        """
        Add a circular arc defined by three existing points.

        Parameters
        ----------
        start  : point tag — start of the arc
        center : point tag — centre of the circle (not on the arc)
        end    : point tag — end of the arc

        Note
        ----
        All three points must be equidistant from the implied circle centre.
        The arc is the *shorter* of the two possible arcs unless you reverse
        the start/end order.
        """
        tag = gmsh.model.occ.addCircleArc(start, center, end)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_arc(start={start}, centre={center}, end={end}) -> tag {tag}")
        return self._model._register(1, tag, label, 'arc')

    def add_circle(
        self,
        cx: float, cy: float, cz: float,
        radius: float,
        *,
        angle1: float     = 0.0,
        angle2: float     = 2 * math.pi,
        label : str | None = None,
        sync  : bool       = True,
    ) -> Tag:
        """
        Add a full circle (or arc sector) as a single curve entity.

        Unlike ``add_arc``, this does **not** require pre-existing point
        tags — it creates the circle directly from centre + radius.

        Parameters
        ----------
        cx, cy, cz : centre
        radius     : radius
        angle1     : start angle in radians (default 0)
        angle2     : end angle in radians   (default 2π = full circle)
        """
        tag = gmsh.model.occ.addCircle(cx, cy, cz, radius,
                                        angle1=angle1, angle2=angle2)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(
            f"add_circle(centre=({cx},{cy},{cz}), r={radius}, "
            f"[{math.degrees(angle1):.1f}°->{math.degrees(angle2):.1f}°]) -> tag {tag}"
        )
        return self._model._register(1, tag, label, 'circle')

    def add_ellipse(
        self,
        cx: float, cy: float, cz: float,
        r_major: float, r_minor: float,
        *,
        angle1: float      = 0.0,
        angle2: float      = 2 * math.pi,
        label : str | None = None,
        sync  : bool       = True,
    ) -> Tag:
        """
        Add a full ellipse (or elliptic arc) as a single curve entity.

        Parameters
        ----------
        cx, cy, cz : centre
        r_major    : semi-major axis (along X before any rotation)
        r_minor    : semi-minor axis
        angle1     : start angle in radians
        angle2     : end angle in radians
        """
        tag = gmsh.model.occ.addEllipse(cx, cy, cz, r_major, r_minor,
                                         angle1=angle1, angle2=angle2)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(
            f"add_ellipse(centre=({cx},{cy},{cz}), a={r_major}, b={r_minor}) -> tag {tag}"
        )
        return self._model._register(1, tag, label, 'ellipse')

    def add_spline(
        self,
        point_tags: list[Tag],
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a C2-continuous spline curve **through** the given points
        (interpolating spline).

        Parameters
        ----------
        point_tags : ordered list of point tags the spline passes through.
                     Minimum 2 points; for a closed spline repeat the first
                     tag at the end.

        Example
        -------
        ::

            p1 = g.model.geometry.add_point(0, 0, 0)
            p2 = g.model.geometry.add_point(1, 1, 0)
            p3 = g.model.geometry.add_point(2, 0, 0)
            s  = g.model.geometry.add_spline([p1, p2, p3])
        """
        if len(point_tags) < 2:
            raise ValueError("add_spline requires at least 2 point tags.")
        tag = gmsh.model.occ.addSpline(point_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_spline({point_tags}) -> tag {tag}")
        return self._model._register(1, tag, label, 'spline')

    def add_bspline(
        self,
        point_tags   : list[Tag],
        *,
        degree        : int         = 3,
        weights       : list[float] | None = None,
        knots         : list[float] | None = None,
        multiplicities: list[int]   | None = None,
        label         : str | None  = None,
        sync          : bool        = True,
    ) -> Tag:
        """
        Add a B-spline curve with explicit control points.

        Control points are **not** interpolated (the curve is attracted to
        them, not forced through them), which is different from
        ``add_spline``.

        Parameters
        ----------
        point_tags     : control-point tags
        degree         : polynomial degree (default 3 = cubic)
        weights        : optional rational weights (len = len(point_tags))
        knots          : optional knot vector
        multiplicities : optional knot multiplicities
        """
        if len(point_tags) < 2:
            raise ValueError("add_bspline requires at least 2 point tags.")
        tag = gmsh.model.occ.addBSpline(
            point_tags,
            degree=degree,
            weights=weights        or [],
            knots=knots            or [],
            multiplicities=multiplicities or [],
        )
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_bspline(ctrl_pts={point_tags}, degree={degree}) -> tag {tag}")
        return self._model._register(1, tag, label, 'bspline')

    def add_bezier(
        self,
        point_tags: list[Tag],
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a Bézier curve.

        Parameters
        ----------
        point_tags : control-point tags.  The curve starts at the first
                     point and ends at the last; intermediate points are
                     control handles (not interpolated).
        """
        if len(point_tags) < 2:
            raise ValueError("add_bezier requires at least 2 point tags.")
        tag = gmsh.model.occ.addBezier(point_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_bezier({point_tags}) -> tag {tag}")
        return self._model._register(1, tag, label, 'bezier')

    # ------------------------------------------------------------------
    # Wire / surface builders  (dim = 1 -> 2)
    # ------------------------------------------------------------------

    def add_wire(
        self,
        curve_tags: list[Tag],
        *,
        check_closed: bool       = False,
        label       : str | None = None,
        sync        : bool       = True,
    ) -> Tag:
        """
        Assemble an ordered list of curve tags into an OpenCASCADE wire
        (open or closed).  Wires are the path input for sweep operations
        (:meth:`sweep`) and the section input for lofted volumes
        (:meth:`thru_sections`).

        Unlike :meth:`add_curve_loop`, a wire does **not** need to be
        closed.  This is what makes it suitable as a sweep path.

        Parameters
        ----------
        curve_tags : ordered curve tags.  Curves must be connected
            end-to-end but may share only geometrically identical
            endpoints (OCC allows topologically distinct but coincident
            points).
        check_closed : if True, the underlying OCC call verifies that
            the wire forms a closed loop and raises otherwise.
        label : registry label (for later resolution by name).

        Returns
        -------
        int tag of the new wire (a dim-1 entity).

        Example
        -------
        ::

            p0 = g.model.geometry.add_point(0, 0, 0, sync=False)
            p1 = g.model.geometry.add_point(1, 0, 0, sync=False)
            p2 = g.model.geometry.add_point(1, 1, 0, sync=False)
            p3 = g.model.geometry.add_point(1, 1, 2, sync=False)
            l1 = g.model.geometry.add_line(p0, p1, sync=False)
            l2 = g.model.geometry.add_line(p1, p2, sync=False)
            l3 = g.model.geometry.add_line(p2, p3, sync=False)
            path = g.model.geometry.add_wire([l1, l2, l3], label="sweep_path")
        """
        tag = gmsh.model.occ.addWire(curve_tags, checkClosed=check_closed)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_wire({curve_tags}, closed={check_closed}) -> tag {tag}")
        return self._model._register(1, tag, label, 'wire')

    def add_curve_loop(
        self,
        curve_tags: list[Tag],
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Assemble an ordered list of curve tags into a closed wire (curve
        loop).  The result is used as input to ``add_plane_surface`` or
        ``add_surface_filling``.

        Parameters
        ----------
        curve_tags : ordered curve tags forming a closed loop.
                     Use negative tags to reverse orientation of a curve.

        Example
        -------
        ::

            loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
            surf = g.model.geometry.add_plane_surface(loop)
        """
        tag = gmsh.model.occ.addCurveLoop(curve_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_curve_loop({curve_tags}) -> tag {tag}")
        return self._model._register(1, tag, label, 'curve_loop')

    def add_plane_surface(
        self,
        wire_tags: Tag | list[Tag],
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Create a planar surface bounded by one or more curve loops.

        Parameters
        ----------
        wire_tags : tag (or list of tags) of curve loops.  The first loop
                    is the outer boundary; any additional loops define holes.

        Example
        -------
        ::

            outer = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
            hole  = g.model.geometry.add_curve_loop([h1, h2, h3, h4])
            surf  = g.model.geometry.add_plane_surface([outer, hole])
        """
        if isinstance(wire_tags, int):
            wire_tags = [wire_tags]
        tag = gmsh.model.occ.addPlaneSurface(wire_tags)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_plane_surface(wires={wire_tags}) -> tag {tag}")
        return self._model._register(2, tag, label, 'plane_surface')

    def add_surface_filling(
        self,
        wire_tag: Tag,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Create a surface filling bounded by a single curve loop, using a
        Coons-patch style interpolation (non-planar surfaces).

        Parameters
        ----------
        wire_tag : tag of the bounding curve loop
        """
        tag = gmsh.model.occ.addSurfaceFilling(wire_tag)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"add_surface_filling(wire={wire_tag}) -> tag {tag}")
        return self._model._register(2, tag, label, 'surface_filling')

    def add_rectangle(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float,
        *,
        rounded_radius: float = 0.0,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a rectangular planar surface in the XY plane.

        The rectangle is created at **(x, y, z)** with extents **dx** along
        X and **dy** along Y.  Combine with :meth:`rotate` and
        :meth:`translate` to orient it arbitrarily.

        Useful as a cutting tool for :meth:`fragment` — a 2D rectangle
        fragmented against a 3D solid splits the solid along the
        rectangle's plane.

        Parameters
        ----------
        x, y, z : float
            Corner of the rectangle.
        dx, dy : float
            Extents along X and Y.
        rounded_radius : float
            If > 0, rounds the four corners with this radius.
        label : str, optional
            Human-readable label stored in the internal registry.
        sync : bool
            Synchronise the OCC kernel after creation (default True).

        Returns
        -------
        Tag
            Surface tag of the new rectangle.

        Example
        -------
        ::

            # Split a solid at mid-height with a cutting plane
            bb = gmsh.model.getBoundingBox(3, 1)
            xmin, ymin, zmin, xmax, ymax, zmax = bb
            zmid = (zmin + zmax) / 2
            pad = 1.0
            rect = m1.model.geometry.add_rectangle(
                xmin - pad, ymin - pad, zmid,
                (xmax - xmin) + 2*pad,
                (ymax - ymin) + 2*pad,
            )
            result = m1.model.boolean.fragment(objects=[1], tools=[rect], dim=3)
        """
        tag = gmsh.model.occ.addRectangle(x, y, z, dx, dy, roundedRadius=rounded_radius)
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(
            f"add_rectangle(origin=({x},{y},{z}), size=({dx},{dy})"
            f"{f', r={rounded_radius}' if rounded_radius else ''}) -> tag {tag}"
        )
        return self._model._register(2, tag, label, 'rectangle')

    def add_cutting_plane(
        self,
        point          : list[float] | ndarray,
        normal_vector  : list[float] | ndarray,
        *,
        size           : float | None = None,
        label          : str | None   = None,
        sync           : bool         = True,
    ) -> Tag:
        """
        Create a square planar surface through ``point`` with the given
        normal, suitable for clipping / section / visualisation views.

        The surface is a plain BRep face built from 4 points + 4 lines
        + a curve loop + a plane surface, so it behaves exactly like
        any other registered surface (it can be selected, meshed as a
        discrete 2-D grid, exported to STEP, etc.).  It is *not* a
        Gmsh clipping plane in the rendering sense — it is real
        geometry.

        Parameters
        ----------
        point : array-like of 3 floats
            A point on the plane.  The square is centred here.
        normal_vector : array-like of 3 floats
            Plane normal.  Need not be unit-length — it is normalised
            internally.
        size : float, optional
            Edge length of the square.  When ``None`` (default), size
            is picked as ``2 × max(model_bbox_diagonal, 1.0)`` so the
            square comfortably overhangs the current model.
        label : str, optional
            Human-readable label stored in the internal registry.
        sync : bool, optional
            Synchronise the OCC kernel after creation (default True).

        Returns
        -------
        Tag
            Surface tag of the new cutting plane.

        Example
        -------
        ::

            # A vertical plane through (0, 0, 0) with normal (1, 0, 0)
            g.model.geometry.add_cutting_plane(
                point=(0, 0, 0), normal_vector=(1, 0, 0),
            )
        """
        p = np.asarray(point, dtype=float)
        n = np.asarray(normal_vector, dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm == 0.0:
            raise ValueError("normal_vector must be non-zero")
        n = n / n_norm

        # Pick size from the current model bounding box when not given.
        # ``getBoundingBox`` requires a synchronised OCC state, so this
        # is the only hard reason for a pre-sync.  When the caller
        # supplies an explicit ``size``, we do not sync until the end.
        if size is None:
            gmsh.model.occ.synchronize()
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
            size = 2.0 * max(diag, 1.0)

        # Orthonormal basis (u, v) spanning the plane.  v does not need
        # explicit normalisation: if ``n`` and ``u`` are unit and
        # orthogonal, then ``cross(n, u)`` is also unit.
        ref = (
            np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = np.cross(n, ref)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        half = size / 2.0
        corner_coeffs = [(-half, -half), (half, -half), (half, half), (-half, half)]
        corners = [p + a * u + b * v for a, b in corner_coeffs]

        # Delegate to the existing BRep primitives so the new points,
        # lines, loop, and surface all flow through ``_register`` /
        # ``_log`` with the correct kinds.  Defer every sub-sync so we
        # sync exactly once at the end (or zero times if sync=False).
        pt_tags  = [
            self.add_point(float(c[0]), float(c[1]), float(c[2]), sync=False)
            for c in corners
        ]
        ln_tags  = [
            self.add_line(pt_tags[i], pt_tags[(i + 1) % 4], sync=False)
            for i in range(4)
        ]
        loop_tag = self.add_curve_loop(ln_tags, sync=False)
        # Defer the label until AFTER the sync below: a label PG bind
        # via gmsh.model.addPhysicalGroup requires the entity to be
        # visible to Gmsh, which only happens post-synchronize.
        tag      = self.add_plane_surface(loop_tag, sync=False)

        if sync:
            gmsh.model.occ.synchronize()

        self._model._log(
            f"add_cutting_plane(point={tuple(p)}, normal={tuple(n)}, "
            f"size={size}) -> tag {tag}"
        )
        # Re-register as a cutting plane (overwrites the
        # ``plane_surface`` metadata kind from add_plane_surface) and,
        # now that the surface is synced, bind the label PG.  When
        # sync=False the caller has opted out of syncing, so we only
        # update the metadata kind and skip label registration to
        # avoid the "Unknown entity" warning.
        if sync:
            self._model._register(2, tag, label, 'cutting_plane')
        else:
            self._model._metadata[(2, tag)] = {'kind': 'cutting_plane'}
        entry = self._model._metadata.get((2, tag))
        if entry is not None:
            entry['point']  = tuple(float(x) for x in p)
            entry['normal'] = tuple(float(x) for x in n)
        return tag

    def add_axis_cutting_plane(
        self,
        axis          : Literal['x', 'y', 'z'],
        offset        : float = 0.0,
        *,
        origin        : list[float] | ndarray | None = None,
        rotation      : float                        = 0.0,
        rotation_about: Literal['x', 'y', 'z'] | None = None,
        label         : str | None                   = None,
        sync          : bool                         = True,
    ) -> Tag:
        """
        Add an axis-aligned cutting plane, optionally tilted by a rotation.

        Convenience wrapper around :meth:`add_cutting_plane`.  The plane is
        initially defined as **normal to** ``axis`` (so ``axis='z'`` produces
        a horizontal XY-plane).  It is then:

        1. Offset along its base normal by ``offset``.
        2. Rotated by ``rotation`` degrees about ``rotation_about``
           (if both are given), producing a tilted plane through the same
           anchor point.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Axis the plane is **normal to**.  ``'z'`` -> XY plane, etc.
        offset : float, optional
            Signed distance along the base normal from ``origin``
            (or from the global origin if ``origin`` is None).
        origin : array-like of 3 floats, optional
            Anchor point before the offset is applied.  Defaults to (0, 0, 0).
        rotation : float, optional
            Rotation angle in **degrees**.  Requires ``rotation_about``
            to have any effect — passing ``rotation`` without
            ``rotation_about`` raises ``ValueError`` so silent
            no-ops do not sneak through.
        rotation_about : {'x', 'y', 'z'}, optional
            Axis about which the base normal is rotated.  Must differ
            from ``axis`` for the rotation to have any effect.
        label : str, optional
            Human-readable label stored in the internal registry.
        sync : bool, optional
            Synchronise the OCC kernel after creation (default True).

        Returns
        -------
        Tag
            Surface tag of the new cutting plane.

        Examples
        --------
        Horizontal plane at z = 3::

            g.model.geometry.add_axis_cutting_plane('z', offset=3.0)

        Vertical YZ-plane passing through x = 1.5::

            g.model.geometry.add_axis_cutting_plane('x', offset=1.5)

        Horizontal plane tilted 15° about the y-axis::

            g.model.geometry.add_axis_cutting_plane(
                'z', offset=0.0,
                rotation=15.0, rotation_about='y',
            )
        """
        if axis not in _AXIS_UNIT_VEC:
            raise ValueError(
                f"axis must be one of 'x', 'y', 'z'; got {axis!r}"
            )
        if rotation != 0.0 and rotation_about is None:
            raise ValueError(
                "rotation was provided without rotation_about — pass "
                "rotation_about='x'|'y'|'z' to tilt the plane, or drop "
                "the rotation argument to keep it axis-aligned."
            )

        base_normal = _AXIS_UNIT_VEC[axis].copy()

        if rotation_about is not None and rotation != 0.0:
            if rotation_about not in _AXIS_UNIT_VEC:
                raise ValueError(
                    f"rotation_about must be one of 'x', 'y', 'z'; "
                    f"got {rotation_about!r}"
                )
            if rotation_about == axis:
                self._model._log(
                    f"add_axis_cutting_plane: rotation_about='{rotation_about}' "
                    f"equals axis='{axis}'; rotation has no effect."
                )
            else:
                theta = np.deg2rad(rotation)
                k     = _AXIS_UNIT_VEC[rotation_about]
                c, s  = np.cos(theta), np.sin(theta)
                # Rodrigues' rotation formula — k is a unit vector.
                base_normal = (
                    base_normal * c
                    + np.cross(k, base_normal) * s
                    + k * np.dot(k, base_normal) * (1.0 - c)
                )

        if origin is None:
            origin_arr = np.zeros(3)
        else:
            origin_arr = np.asarray(origin, dtype=float)
        if origin_arr.shape != (3,):
            raise ValueError(
                f"origin must be a length-3 vector; got shape {origin_arr.shape}"
            )

        point = origin_arr + offset * base_normal

        self._model._log(
            f"add_axis_cutting_plane(axis={axis!r}, offset={offset}, "
            f"origin={tuple(origin_arr)}, rotation={rotation}, "
            f"rotation_about={rotation_about!r}) "
            f"-> point={tuple(point)}, normal={tuple(base_normal)}"
        )

        return self.add_cutting_plane(
            point=point,
            normal_vector=base_normal,
            label=label,
            sync=sync,
        )

    # ------------------------------------------------------------------
    # Cutting operations
    # ------------------------------------------------------------------

    def _collect_volume_tags(self) -> list[Tag]:
        """Return every 3-D entity tag in the model."""
        return [int(t) for _, t in gmsh.model.getEntities(3)]

    def _resolve_label_to_tags(self, label: str) -> list[Tag]:
        """Resolve a label string to dim=3 entity tags.

        Delegates to the labels composite (Tier 1, backed by Gmsh
        PGs) — the single source of truth for label→tag resolution.

        Returns an empty list when no match is found — the caller
        decides whether to raise.
        """
        labels_comp = getattr(self._model._parent, 'labels', None)
        if labels_comp is not None:
            try:
                return labels_comp.entities(label, dim=3)
            except KeyError:
                pass
        return []

    def _normalize_solid_input(
        self,
        solid: Tag | str | list[Tag | str] | None,
        collector,
    ) -> list[Tag]:
        """Coerce the ``solid`` argument into a concrete list of tags.

        Accepts:

        * ``None`` — every registered volume (via ``collector()``).
        * ``int`` — a single tag, wrapped into a one-element list.
        * ``str`` — a registry label; resolved to every dim=3 entity
          whose ``label`` field matches.  This is the key feature
          for chained slicing: after a slice the original tag is
          consumed but the label can be propagated to the fragments,
          so the next slice can still say ``slice("shaft", ...)``.
        * ``list`` — a mix of ints and/or strings, each resolved
          individually and concatenated.
        """
        if solid is None:
            tags = collector()
        elif isinstance(solid, str):
            tags = self._resolve_label_to_tags(solid)
        elif isinstance(solid, int):
            tags = [int(solid)]
        elif isinstance(solid, (list, tuple)):
            tags = []
            for item in solid:
                if isinstance(item, str):
                    tags.extend(self._resolve_label_to_tags(item))
                else:
                    tags.append(int(item))
        else:
            tags = [int(solid)]
        if not tags:
            raise ValueError(
                "no solids to cut — pass an explicit tag list, a "
                "label string, or register at least one volume "
                "before calling the cut"
            )
        return tags

    def cut_by_surface(
        self,
        solid          : Tag | str | list[Tag | str] | None,
        surface        : Tag | str,
        *,
        keep_surface   : bool = True,
        remove_original: bool = True,
        label          : str | None = None,
        sync           : bool = True,
    ) -> list[Tag]:
        """
        Split one or more solids with an arbitrary cutting surface.

        Uses OCC's ``fragment`` operation under the hood, which splits
        every input shape at its intersections and keeps **all**
        resulting sub-shapes.  Unlike :meth:`cut_by_plane`, this method
        does not classify the output pieces — callers that need
        "above/below" semantics should use :meth:`cut_by_plane` (which
        delegates here and adds the classification step).

        Parameters
        ----------
        solid : Tag, list[Tag], or None
            Volume(s) to cut.  When ``None``, every registered volume
            in the model is cut against the surface.
        surface : Tag or str
            The cutting surface.  Can be any registered 2-D entity —
            a plane from :meth:`add_cutting_plane`, a STEP-imported
            trimmed surface, a Coons patch, etc.  A string is resolved
            against the labels composite at dim=2.
        keep_surface : bool, default True
            Leave the (now-trimmed) surface in the model after the
            cut.  Useful when you want to mesh the cut interface as a
            shared face for conformal ties.  Set to ``False`` to
            delete it.
        remove_original : bool, default True
            Consume the original solid(s) so only the cut pieces
            remain.  When ``False``, OCC keeps the originals alongside
            the pieces, which usually produces overlapping geometry
            and is rarely what you want.
        label : str, optional
            Label applied to every new volume fragment in the
            registry.  Pass ``None`` to leave the fragments unlabelled.
        sync : bool, default True
            Synchronise the OCC kernel after the cut.

        Returns
        -------
        list[Tag]
            Solid tags of the fragments produced by the cut, in the
            order OCC returns them.  An empty list means the cut
            produced nothing new (shouldn't happen unless the surface
            misses every input solid entirely).

        Example
        -------
        ::

            box   = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            plane = g.model.geometry.add_axis_cutting_plane('z', offset=0.5)
            pieces = g.model.geometry.cut_by_surface(box, plane)
        """
        solid_tags = self._normalize_solid_input(solid, self._collect_volume_tags)

        # Resolve a string surface ref via the labels composite —
        # mirrors the flexible-ref behaviour on the ``solid`` side so
        # ``cut_by_surface('box', 'cutter')`` works after add_cutting_plane(label=...).
        if isinstance(surface, str):
            labels_comp = getattr(self._model._parent, 'labels', None)
            if labels_comp is None:
                raise ValueError(
                    f"surface={surface!r} is a string but no labels "
                    f"composite is available to resolve it"
                )
            surface_tags = labels_comp.entities(surface, dim=2)
            if not surface_tags:
                raise ValueError(
                    f"label {surface!r} resolved to no dim=2 entities"
                )
        else:
            surface_tags = [int(surface)]

        obj_dt = [(3, int(t)) for t in solid_tags]
        tool_dt = [(2, int(t)) for t in surface_tags]

        # Collect the original labels BEFORE the boolean so we can
        # propagate them to fragments afterwards.
        inherited_label = label
        if inherited_label is None and remove_original:
            labels_comp = getattr(self._model._parent, 'labels', None)
            if labels_comp is not None:
                original_labels: set[str] = set()
                for t in solid_tags:
                    original_labels.update(
                        labels_comp.labels_for_entity(3, int(t))
                    )
                if len(original_labels) == 1:
                    inherited_label = original_labels.pop()

        with pg_preserved() as pg:
            out_dimtags, result_map = gmsh.model.occ.fragment(
                obj_dt,
                tool_dt,
                removeObject=remove_original,
                removeTool=not keep_surface,
            )
            # Always sync before PG remap — remap needs topology visible.
            gmsh.model.occ.synchronize()
            pg.set_result(obj_dt + tool_dt, result_map)

        new_volume_tags: list[Tag] = [
            int(t) for (d, t) in out_dimtags if d == 3
        ]

        if remove_original:
            for t in solid_tags:
                self._model._metadata.pop((3, int(t)), None)

        # Register metadata for new fragments. Label= is only passed
        # when the user explicitly provided one; inherited labels are
        # already handled by the snapshot-remap above.
        for t in new_volume_tags:
            self._model._register(3, t, label, 'cut_fragment')

        if keep_surface:
            surviving_surfaces = [
                int(t) for (d, t) in out_dimtags if d == 2
            ]
            for t in surviving_surfaces:
                if (2, t) not in self._model._metadata:
                    self._model._register(2, t, None, 'cut_interface')

        self._model._log(
            f"cut_by_surface(solids={solid_tags}, surface={surface_tags}) "
            f"-> {len(new_volume_tags)} volume fragment(s): {new_volume_tags}"
        )
        return new_volume_tags

    def cut_by_plane(
        self,
        solid          : Tag | str | list[Tag | str] | None,
        plane          : Tag,
        *,
        keep_plane     : bool = True,
        remove_original: bool = True,
        above_direction: list[float] | ndarray | None = None,
        label_above    : str | None = None,
        label_below    : str | None = None,
        sync           : bool = True,
    ) -> tuple[list[Tag], list[Tag]]:
        """
        Split one or more solids with a plane and classify the
        resulting pieces by which side of the plane they sit on.

        Thin wrapper around :meth:`cut_by_surface` that additionally
        computes which fragments are "above" (same side as the plane
        normal) vs "below" the plane.  The normal direction is
        resolved from, in order of priority:

        1. An explicit ``above_direction`` argument.
        2. The ``normal`` and ``point`` stashed in the registry by
           :meth:`add_cutting_plane` / :meth:`add_axis_cutting_plane`.
        3. ``gmsh.model.getNormal`` sampled at the parametric centre
           of the plane surface.

        Parameters
        ----------
        solid : Tag, list[Tag], or None
            Volume(s) to cut.  ``None`` = every registered volume.
        plane : Tag
            Planar surface to cut with.  Ideally built by
            :meth:`add_cutting_plane` so its normal and point are in
            the registry; other planar surfaces work too but require
            an explicit ``above_direction`` or fall back to querying
            Gmsh.
        keep_plane : bool, default True
            Leave the trimmed plane in the model as a registered
            surface (useful for meshing the cut interface).
        remove_original : bool, default True
            Consume the original solid(s).
        above_direction : array-like of 3 floats, optional
            Override the plane's normal direction.  Pieces whose
            centroid dotted with this vector (relative to the plane
            point) is positive are classified as "above".
        label_above, label_below : str, optional
            Labels applied to the above / below fragment solids.
        sync : bool, default True
            Synchronise the OCC kernel after the cut.

        Returns
        -------
        tuple[list[Tag], list[Tag]]
            ``(above_tags, below_tags)`` — solid tags on each side of
            the plane, classified by the sign of
            ``(centroid - plane_point) · normal``.

        Example
        -------
        ::

            col = g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
            pl  = g.model.geometry.add_axis_cutting_plane('z', offset=1.5)

            top, bot = g.model.geometry.cut_by_plane(
                col, pl,
                label_above="col_upper", label_below="col_lower",
            )
        """
        plane_tag = int(plane)
        normal, point = self._resolve_plane_normal(
            plane_tag, above_direction,
        )

        # Perform the actual cut via the general surface method.
        # sync=True so the PG remap and classify step see synced topology.
        fragments = self.cut_by_surface(
            solid,
            plane_tag,
            keep_surface=keep_plane,
            remove_original=remove_original,
            label=None,          # we re-label by side below
            sync=True,
        )

        above_tags, below_tags = self._classify_fragments(
            fragments, normal, point,
            label_above=label_above,
            label_below=label_below,
        )

        if not above_tags or not below_tags:
            self._model._log(
                f"cut_by_plane: WARNING plane {plane_tag} produced "
                f"only one side ({len(above_tags)} above, "
                f"{len(below_tags)} below) — the plane may not "
                f"intersect the solid(s)"
            )

        if sync:
            gmsh.model.occ.synchronize()

        self._model._log(
            f"cut_by_plane(plane={plane_tag}) -> "
            f"above={above_tags}, below={below_tags}"
        )
        return above_tags, below_tags

    def _resolve_plane_normal(
        self,
        plane_tag: int,
        above_direction: list[float] | ndarray | None,
    ) -> tuple[ndarray, ndarray]:
        """Resolve a cutting plane's normal and a point on the plane.

        Tries three sources in order: explicit *above_direction*,
        stashed metadata from ``add_cutting_plane``, and finally
        ``gmsh.model.getNormal``.

        Returns ``(unit_normal, point_on_plane)`` as numpy arrays.
        """
        entry = self._model._metadata.get((2, plane_tag))
        stashed_normal = entry.get('normal') if entry else None
        stashed_point  = entry.get('point')  if entry else None

        if above_direction is not None:
            normal = np.asarray(above_direction, dtype=float)
            norm_len = float(np.linalg.norm(normal))
            if norm_len == 0.0:
                raise ValueError("above_direction must be non-zero")
            normal = normal / norm_len
            if stashed_point is not None:
                point = np.asarray(stashed_point, dtype=float)
            else:
                point = self._any_point_on_surface(plane_tag)
            return normal, point

        if stashed_normal is not None and stashed_point is not None:
            return (
                np.asarray(stashed_normal, dtype=float),
                np.asarray(stashed_point, dtype=float),
            )

        # Fall back to gmsh.model.getNormal at the parametric midpoint.
        gmsh.model.occ.synchronize()
        try:
            nxyz = gmsh.model.getNormal(plane_tag, [0.5, 0.5])
        except Exception as exc:
            raise ValueError(
                f"cut_by_plane: plane tag {plane_tag} has no registry "
                f"normal and Gmsh could not compute one — pass "
                f"above_direction=... explicitly. ({exc})"
            ) from exc
        normal = np.asarray(nxyz, dtype=float)
        norm_len = float(np.linalg.norm(normal))
        if norm_len == 0.0:
            raise ValueError(
                f"cut_by_plane: plane tag {plane_tag} returned a zero "
                f"normal from Gmsh; pass above_direction=..."
            )
        return normal / norm_len, self._any_point_on_surface(plane_tag)

    def _classify_fragments(
        self,
        fragments: list[Tag],
        normal: ndarray,
        point: ndarray,
        *,
        label_above: str | None,
        label_below: str | None,
    ) -> tuple[list[Tag], list[Tag]]:
        """Classify volume fragments as above/below a plane.

        Sorts *fragments* by the sign of ``(centroid - point) . normal``
        and optionally labels each side via ``g.labels``.
        """
        above_tags: list[Tag] = []
        below_tags: list[Tag] = []
        labels_comp = getattr(self._model._parent, 'labels', None)

        for t in fragments:
            com = np.asarray(
                gmsh.model.occ.getCenterOfMass(3, int(t)), dtype=float,
            )
            signed = float(np.dot(com - point, normal))
            if signed >= 0.0:
                above_tags.append(t)
                self._try_label(labels_comp, label_above, t)
            else:
                below_tags.append(t)
                self._try_label(labels_comp, label_below, t)

        return above_tags, below_tags

    @staticmethod
    def _try_label(labels_comp, label: str | None, tag: Tag) -> None:
        """Apply a label to a volume tag, warning on failure."""
        if label is None or labels_comp is None:
            return
        try:
            labels_comp.add(3, [tag], name=label)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Label {label!r} for fragment {tag} "
                f"could not be created: {exc}",
                stacklevel=3,
            )

    def _any_point_on_surface(self, surface_tag: Tag) -> ndarray:
        """Return a point in space known to lie on ``surface_tag``.

        Used by :meth:`cut_by_plane` when the plane's defining point
        is not in the registry — we ask OCC for the centre of mass
        of the surface, which always lies on the surface for planar
        faces (and is a reasonable proxy for curved ones).
        """
        com = gmsh.model.occ.getCenterOfMass(2, int(surface_tag))
        return np.asarray(com, dtype=float)

    def _cleanup_slice_orphans(
        self,
        pre_entities: dict[int, set[int]],
        keep_vol_tags: set[int],
    ) -> None:
        """Remove entities created during a slice that aren't part of the result.

        Syncs the OCC kernel, walks the boundary hierarchy of
        *keep_vol_tags*, then removes anything new (not in
        *pre_entities*) that isn't a boundary of a surviving volume.
        """
        gmsh.model.occ.synchronize()

        keep_dimtags: set[tuple[int, int]] = {(3, t) for t in keep_vol_tags}
        for vol_tag in keep_vol_tags:
            try:
                for dt in gmsh.model.getBoundary(
                    [(3, vol_tag)], oriented=False, recursive=True,
                ):
                    keep_dimtags.add((abs(dt[0]), abs(dt[1])))
            except Exception:
                pass

        removed_dts: list[tuple[int, int]] = []
        for d in [2, 1, 0]:
            for _, t in gmsh.model.getEntities(d):
                if t in pre_entities.get(d, set()):
                    continue
                if (d, t) in keep_dimtags:
                    continue
                try:
                    gmsh.model.occ.remove([(d, t)], recursive=False)
                except Exception:
                    pass
                self._model._metadata.pop((d, t), None)
                removed_dts.append((d, t))

        if removed_dts:
            cleanup_label_pgs(removed_dts)

    # ------------------------------------------------------------------
    # Slice (atomic cut + cleanup)
    # ------------------------------------------------------------------

    def slice(
        self,
        solid   : Tag | str | list[Tag | str] | None = None,
        *,
        axis    : Literal['x', 'y', 'z'],
        offset  : float = 0.0,
        classify: bool = False,
        label   : str | None = None,
        sync    : bool = True,
    ) -> list[Tag] | tuple[list[Tag], list[Tag]]:
        """
        Slice solids at an axis-aligned plane in one atomic call.

        Internally creates a temporary cutting plane, fragments the
        solids, removes the cutting plane (and any trimmed surfaces
        it left behind), and returns the volume fragments.  No
        orphaned geometry is left in the model.

        Parameters
        ----------
        solid : Tag, list[Tag], or None
            Volume(s) to slice.  ``None`` slices every registered
            volume in the model.
        axis : {'x', 'y', 'z'}
            Axis the plane is **normal to**.  ``'z'`` slices with
            a horizontal XY-plane, etc.
        offset : float, default 0.0
            Signed distance along the axis from the origin.
        classify : bool, default False
            When True, returns ``(positive_side, negative_side)``
            classified by the plane's normal direction (the positive
            axis direction).  When False (default), returns all
            fragments as a flat list.
        label : str, optional
            Label applied to every fragment in the registry.
        sync : bool, default True
            Synchronise the OCC kernel after the operation.

        Returns
        -------
        list[Tag]
            All volume fragments (when ``classify=False``).
        tuple[list[Tag], list[Tag]]
            ``(positive_side, negative_side)`` fragments classified
            by which side of the plane each piece's centroid sits on
            (when ``classify=True``).

        Example
        -------
        ::

            # Slice a box at y = 0.5
            box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            pieces = g.model.geometry.slice(box, axis='y', offset=0.5)

            # Slice and classify
            top, bot = g.model.geometry.slice(
                box, axis='z', offset=0.5, classify=True,
            )

            # Slice all volumes at x = 0
            g.model.geometry.slice(axis='x', offset=0.0)
        """
        # Snapshot ALL entities before creating the cutting plane.
        # After the slice, anything that was created during the
        # operation but isn't a surviving volume fragment gets
        # removed.  This catches the cutting-plane corner points,
        # edges, curve loops, and trimmed surfaces that the old
        # registry-based cleanup missed.
        pre_entities: dict[int, set[int]] = {}
        for d in range(4):
            pre_entities[d] = {t for _, t in gmsh.model.getEntities(d)}

        plane_tag = self.add_axis_cutting_plane(
            axis, offset=offset, sync=False,
        )

        if classify:
            above, below = self.cut_by_plane(
                solid, plane_tag,
                keep_plane=False,
                label_above=label,
                label_below=label,
                sync=False,
            )
            result: list[Tag] | tuple[list[Tag], list[Tag]] = (above, below)
        else:
            fragments = self.cut_by_surface(
                solid, plane_tag,
                keep_surface=False,
                label=label,
                sync=False,
            )
            result = fragments

        # Determine which volume tags are the real output.
        if classify:
            keep_vol_tags = set(above) | set(below)
        else:
            keep_vol_tags = set(fragments)

        self._cleanup_slice_orphans(pre_entities, keep_vol_tags)

        if sync:
            gmsh.model.occ.synchronize()

        self._model._log(
            f"slice(axis={axis!r}, offset={offset}, classify={classify})"
        )
        return result

    # ------------------------------------------------------------------
    # Primitives  (dim = 3 solids)
    # ------------------------------------------------------------------

    def _add_solid(
        self, tag: int, kind: str, desc: str,
        *, label: str | None, sync: bool,
    ) -> Tag:
        """Common tail for every solid primitive: sync, log, register."""
        if sync:
            gmsh.model.occ.synchronize()
        self._model._log(f"{desc} -> tag {tag}")
        return self._model._register(3, tag, label, kind)

    def add_box(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float, dz: float,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add an axis-aligned box.

        Parameters
        ----------
        x, y, z       : origin corner
        dx, dy, dz    : extents along X, Y, Z
        """
        tag = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
        return self._add_solid(
            tag, 'box', f"add_box(origin=({x},{y},{z}), size=({dx},{dy},{dz}))",
            label=label, sync=sync,
        )

    def add_sphere(
        self,
        cx: float, cy: float, cz: float,
        radius: float,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """Add a sphere centred at (cx, cy, cz) with the given radius."""
        tag = gmsh.model.occ.addSphere(cx, cy, cz, radius)
        return self._add_solid(
            tag, 'sphere', f"add_sphere(centre=({cx},{cy},{cz}), r={radius})",
            label=label, sync=sync,
        )

    def add_cylinder(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float, dz: float,
        radius: float,
        *,
        angle: float      = 2 * math.pi,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a cylinder.

        Parameters
        ----------
        x, y, z    : base-circle centre
        dx, dy, dz : axis direction vector (length = height of cylinder)
        radius     : base radius
        angle      : sweep angle in radians (default 2π = full cylinder)
        """
        tag = gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, radius, angle=angle)
        return self._add_solid(
            tag, 'cylinder',
            f"add_cylinder(base=({x},{y},{z}), axis=({dx},{dy},{dz}), r={radius})",
            label=label, sync=sync,
        )

    def add_cone(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float, dz: float,
        r1: float, r2: float,
        *,
        angle: float      = 2 * math.pi,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a cone / truncated cone.

        Parameters
        ----------
        x, y, z    : base-circle centre
        dx, dy, dz : axis vector
        r1         : base radius
        r2         : top radius (0 = sharp cone)
        angle      : sweep angle in radians
        """
        tag = gmsh.model.occ.addCone(x, y, z, dx, dy, dz, r1, r2, angle=angle)
        return self._add_solid(
            tag, 'cone', f"add_cone(base=({x},{y},{z}), r1={r1}, r2={r2})",
            label=label, sync=sync,
        )

    def add_torus(
        self,
        cx: float, cy: float, cz: float,
        r1: float, r2: float,
        *,
        angle: float      = 2 * math.pi,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a torus.

        Parameters
        ----------
        cx, cy, cz : centre
        r1         : major radius (axis to tube centre)
        r2         : minor radius (tube cross-section)
        angle      : sweep angle in radians
        """
        tag = gmsh.model.occ.addTorus(cx, cy, cz, r1, r2, angle=angle)
        return self._add_solid(
            tag, 'torus', f"add_torus(centre=({cx},{cy},{cz}), R={r1}, r={r2})",
            label=label, sync=sync,
        )

    def add_wedge(
        self,
        x: float, y: float, z: float,
        dx: float, dy: float, dz: float,
        ltx: float,
        *,
        label: str | None = None,
        sync : bool       = True,
    ) -> Tag:
        """
        Add a right-angle wedge.

        Parameters
        ----------
        x, y, z    : origin corner
        dx, dy, dz : extents
        ltx        : top X extent (0 = sharp wedge)
        """
        tag = gmsh.model.occ.addWedge(x, y, z, dx, dy, dz, ltx=ltx)
        return self._add_solid(
            tag, 'wedge',
            f"add_wedge(origin=({x},{y},{z}), size=({dx},{dy},{dz}), ltx={ltx})",
            label=label, sync=sync,
        )
