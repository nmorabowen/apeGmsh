"""
PartEdit — whole-Part transforms and structural operations.
============================================================

Accessed as ``part.edit``.  Operations split into two groups by
session-state requirement:

**Active-session methods** — must be called inside a ``with part:``
block.  They mutate the Part's geometry in place.

* :meth:`translate`, :meth:`rotate`, :meth:`mirror`,
  :meth:`scale`, :meth:`dilate`, :meth:`affine`, :meth:`delete`

These return ``self`` (the PartEdit composite) for fluent chaining::

    part.edit.translate(0, 0, 100).rotate(math.pi/2, 1, 0, 0).scale(0.5)

The underlying Part stays accessible as ``part`` after any chain — the
return value is only useful for chaining itself.

**Producer methods** — return brand-new Part objects without entering
their sessions.  Safe to call whether the source Part is currently
active (inside ``with part:``) or already auto-persisted.

* :meth:`copy` — single duplicate with a required new label.

**Pattern methods** — produce N translated/rotated copies.  Require
the source Part to be **non-active** (outside its ``with`` block,
auto-persisted).  This is because each copy needs its own Gmsh
session to bake the per-copy transform, and Gmsh holds at most one
session at a time.

* :meth:`pattern_linear`, :meth:`pattern_polar`

Patterns return ``list[Part]`` of length ``n``.

Label clash handling
--------------------
For :meth:`copy` and the patterns: if the requested label already
names another live Part in this process, a 4-character random hex
suffix is appended (``my_part_a3f2``) and a warning emitted.  This
keeps the call non-fatal while letting you spot the clash in logs.
"""
from __future__ import annotations

import secrets
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

from ._part_anchors import sidecar_path

if TYPE_CHECKING:
    from .Part import Part


# Process-global registry of live Part names so copy/pattern can
# detect clashes.  Weakly held — a Part being garbage-collected
# automatically vacates its slot.
import weakref
_LIVE_PART_NAMES: "weakref.WeakValueDictionary[str, object]" = (
    weakref.WeakValueDictionary()
)


def _register_part_name(name: str, part) -> None:
    _LIVE_PART_NAMES[name] = part


def _name_in_use(name: str) -> bool:
    return name in _LIVE_PART_NAMES


def _resolve_unique_name(requested: str) -> str:
    """Return ``requested`` if free, else ``{requested}_{hex}`` with warning."""
    if not _name_in_use(requested):
        return requested
    fallback = f"{requested}_{secrets.token_hex(2)}"
    while _name_in_use(fallback):
        fallback = f"{requested}_{secrets.token_hex(2)}"
    warnings.warn(
        f"Part name {requested!r} is already in use; using {fallback!r} "
        f"instead.  Pass a different label to suppress this warning.",
        stacklevel=3,
    )
    return fallback


class PartEdit:
    """Whole-Part operations composite. Registered as ``part.edit``."""

    __slots__ = ("_part",)

    def __init__(self, part: "Part") -> None:
        self._part = part

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_active(self, op: str) -> None:
        if not self._part._active:
            raise RuntimeError(
                f"Part.edit.{op}() requires an active session — "
                f"call inside a `with part:` block."
            )

    def _require_inactive(self, op: str) -> None:
        if self._part._active:
            raise RuntimeError(
                f"Part.edit.{op}() requires the source Part's session to "
                f"be CLOSED (auto-persisted) — exit its `with` block first."
            )

    def _require_has_file(self, op: str) -> None:
        if not self._part.has_file:
            raise RuntimeError(
                f"Part.edit.{op}() requires the source Part to have been "
                f"saved.  Either let auto-persist run (exit the `with` "
                f"block) or call part.save() explicitly."
            )

    def _all_dimtags(self) -> list[tuple[int, int]]:
        return list(gmsh.model.getEntities())

    # ------------------------------------------------------------------
    # Translate / rotate (the foundations)
    # ------------------------------------------------------------------

    def translate(self, dx: float, dy: float, dz: float) -> "PartEdit":
        """Translate every entity in the Part by ``(dx, dy, dz)``.

        Parameters
        ----------
        dx, dy, dz : float
            Translation components in model units.

        Returns
        -------
        PartEdit
            ``self`` for chaining.

        Raises
        ------
        RuntimeError
            If the Part's session is not active.
        """
        self._require_active("translate")
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return self
        dimtags = self._all_dimtags()
        if dimtags:
            gmsh.model.occ.translate(dimtags, float(dx), float(dy), float(dz))
            gmsh.model.occ.synchronize()
        return self

    def rotate(
        self,
        angle: float,
        ax: float,
        ay: float,
        az: float,
        *,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "PartEdit":
        """Rotate every entity by ``angle`` (radians) about an axis.

        Parameters
        ----------
        angle : float
            Rotation angle in **radians**.  Right-hand rule: thumb
            along ``(ax, ay, az)``, fingers curl positive.
        ax, ay, az : float
            Axis direction.  Auto-normalized by gmsh.
        center : (cx, cy, cz), default (0, 0, 0)
            Point that the axis passes through.

        Returns
        -------
        PartEdit
            ``self`` for chaining.
        """
        self._require_active("rotate")
        if angle == 0.0:
            return self
        dimtags = self._all_dimtags()
        if dimtags:
            cx, cy, cz = center
            gmsh.model.occ.rotate(
                dimtags,
                float(cx), float(cy), float(cz),
                float(ax), float(ay), float(az),
                float(angle),
            )
            gmsh.model.occ.synchronize()
        return self

    # ------------------------------------------------------------------
    # Mirror / scale / dilate / affine
    # ------------------------------------------------------------------

    def mirror(
        self,
        *,
        plane: str | None = None,
        normal: tuple[float, float, float] | None = None,
        point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "PartEdit":
        """Reflect every entity across a plane.

        Specify the plane in one of two equivalent ways:

        * ``plane="xy"`` / ``"xz"`` / ``"yz"`` — coordinate plane
          through ``point`` (default origin).
        * ``normal=(nx, ny, nz)`` — explicit plane normal; the plane
          passes through ``point`` perpendicular to this vector.

        Pass exactly one of ``plane`` or ``normal``.

        Returns
        -------
        PartEdit
            ``self`` for chaining.

        Raises
        ------
        ValueError
            If neither or both of ``plane`` / ``normal`` are given,
            or ``plane`` is not one of the recognized names.
        """
        self._require_active("mirror")
        if (plane is None) == (normal is None):
            raise ValueError(
                "mirror() requires exactly one of `plane=` or `normal=`."
            )
        if plane is not None:
            named = {
                "xy": (0.0, 0.0, 1.0),
                "xz": (0.0, 1.0, 0.0),
                "yz": (1.0, 0.0, 0.0),
            }
            if plane not in named:
                raise ValueError(
                    f"plane must be one of {sorted(named)}; got {plane!r}"
                )
            nx, ny, nz = named[plane]
        else:
            nx, ny, nz = (float(c) for c in normal)
        # Plane equation a·x + b·y + c·z + d = 0; d shifts the plane
        # through ``point``.
        px, py, pz = point
        d = -(nx * px + ny * py + nz * pz)
        dimtags = self._all_dimtags()
        if dimtags:
            gmsh.model.occ.mirror(dimtags, nx, ny, nz, d)
            gmsh.model.occ.synchronize()
        return self

    def scale(
        self,
        factor: float,
        *,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "PartEdit":
        """Uniform scale every entity by ``factor`` about ``center``.

        ``factor=2.0`` doubles size, ``factor=0.001`` is mm→m.
        """
        self._require_active("scale")
        if factor == 1.0:
            return self
        return self.dilate(factor, factor, factor, center=center)

    def dilate(
        self,
        sx: float,
        sy: float,
        sz: float,
        *,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "PartEdit":
        """Non-uniform scale by ``(sx, sy, sz)`` about ``center``."""
        self._require_active("dilate")
        if sx == 1.0 and sy == 1.0 and sz == 1.0:
            return self
        dimtags = self._all_dimtags()
        if dimtags:
            cx, cy, cz = center
            gmsh.model.occ.dilate(
                dimtags,
                float(cx), float(cy), float(cz),
                float(sx), float(sy), float(sz),
            )
            gmsh.model.occ.synchronize()
        return self

    def affine(self, matrix4x4) -> "PartEdit":
        """Apply a general 4×4 affine transform.

        Parameters
        ----------
        matrix4x4 : 16-element sequence, 4×4 nested list, or ndarray
            Row-major.  Last row typically ``[0, 0, 0, 1]`` (gmsh
            ignores it but it must be present).
        """
        self._require_active("affine")
        flat = _flatten_matrix(matrix4x4)
        if len(flat) != 16:
            raise ValueError(
                f"affine() requires a 4x4 matrix (16 values); got {len(flat)}"
            )
        dimtags = self._all_dimtags()
        if dimtags:
            gmsh.model.occ.affineTransform(dimtags, flat)
            gmsh.model.occ.synchronize()
        return self

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self) -> None:
        """Remove every entity from the Part's Gmsh session.

        Useful when scrapping and rebuilding within the same ``with``
        block.  Labels that pointed at the deleted entities are now
        stale.

        Returns
        -------
        None
        """
        self._require_active("delete")
        dimtags = self._all_dimtags()
        if dimtags:
            gmsh.model.occ.remove(dimtags, recursive=True)
            gmsh.model.occ.synchronize()

    # ------------------------------------------------------------------
    # Copy — produces a new Part by file cloning
    # ------------------------------------------------------------------

    def copy(self, *, label: str) -> "Part":
        """Create a duplicate Part with a new label.

        The duplicate is a brand-new :class:`Part` with its own
        STEP file (and sidecar copy if present), `_owns_file=True`
        so its tempfile is reclaimed when it's garbage-collected.

        The duplicate is **not** entered as an active session — it
        sits on disk ready to be consumed by ``g.parts.add()`` or
        re-entered with ``with new_part:`` if you need to edit it
        further.

        Works whether the source Part is currently active or not:

        * **Active source** — current geometry is dumped to a fresh
          tempfile via ``gmsh.write`` (does not disturb the source's
          ``file_path``).
        * **Inactive source** — the existing STEP and sidecar are
          file-copied via ``shutil``.

        Parameters
        ----------
        label : str, required
            New Part name.  If the name is already in use by another
            live Part in this process, a 4-char random suffix is
            appended and a warning is emitted.

        Returns
        -------
        Part
            New Part with ``has_file=True``, not yet active.

        Raises
        ------
        ValueError
            If ``label`` is empty.
        RuntimeError
            If the source has no current geometry to copy.
        """
        if not isinstance(label, str) or not label:
            raise ValueError("copy() requires a non-empty `label=` argument.")

        # Lazy import to avoid circular dependency
        from .Part import Part
        import tempfile

        new_label = _resolve_unique_name(label)

        new_temp_dir = Path(
            tempfile.mkdtemp(prefix=f"apeGmsh_part_{new_label}_")
        )
        new_step = new_temp_dir / f"{new_label}.step"
        new_sidecar = sidecar_path(new_step)

        if self._part._active:
            # Source is live — dump current state to the new path.
            entities = self._all_dimtags()
            if not entities:
                shutil.rmtree(new_temp_dir, ignore_errors=True)
                raise RuntimeError(
                    "copy() called on an active Part with no geometry."
                )
            gmsh.model.occ.synchronize()
            gmsh.write(str(new_step))
            # Write the sidecar from current PG anchors
            from ._part_anchors import collect_anchors, write_sidecar
            anchors = collect_anchors(gmsh)
            if anchors:
                write_sidecar(new_step, anchors, part_name=new_label)
        else:
            # Source is inactive — copy the existing files.
            self._require_has_file("copy")
            assert self._part.file_path is not None
            shutil.copy2(self._part.file_path, new_step)
            src_sidecar = sidecar_path(self._part.file_path)
            if src_sidecar.exists():
                shutil.copy2(src_sidecar, new_sidecar)

        new_part = Part(new_label)
        new_part.file_path = new_step.resolve()
        new_part._owns_file = True
        new_part._temp_dir = new_temp_dir
        new_part._register_finalizer()
        return new_part

    # ------------------------------------------------------------------
    # Patterns — N translated/rotated copies, each in its own STEP
    # ------------------------------------------------------------------

    def pattern_linear(
        self,
        *,
        label: str,
        n: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> list["Part"]:
        """Create ``n`` translated copies along a line.

        Each copy ``i`` (1..n) is shifted by ``(i*dx, i*dy, i*dz)``
        from the source.  The source itself is **not** modified and
        is **not** included in the returned list.

        Source Part must be **non-active** (outside its ``with`` block).

        Parameters
        ----------
        label : str
            Base name.  Generated names are ``{label}_1`` … ``{label}_n``.
            Clashes get a random suffix per item, with a warning.
        n : int
            Number of copies (>= 1).
        dx, dy, dz : float
            Per-step translation increment.

        Returns
        -------
        list[Part]
            ``n`` new Parts, each with its translated geometry baked
            into its own STEP file.
        """
        self._require_inactive("pattern_linear")
        self._require_has_file("pattern_linear")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer; got {n!r}")
        if not isinstance(label, str) or not label:
            raise ValueError("pattern_linear() requires a non-empty `label=`.")

        return [
            self._make_pattern_item(
                label=f"{label}_{i}",
                translate_offset=(i * dx, i * dy, i * dz),
                rotate_spec=None,
            )
            for i in range(1, n + 1)
        ]

    def pattern_polar(
        self,
        *,
        label: str,
        n: int,
        axis: tuple[float, float, float],
        total_angle: float,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> list["Part"]:
        """Create ``n`` rotated copies around an axis.

        Each copy ``i`` (1..n) is rotated by ``i * total_angle / n``
        about the axis through ``center``.  ``total_angle`` is in
        **radians**.  For a full revolution use ``total_angle=2*pi``;
        for ``n=4`` evenly spaced this gives 90° increments.

        Source Part must be **non-active**.

        Parameters
        ----------
        label : str
            Base name; copies labeled ``{label}_1`` … ``{label}_n``.
        n : int
            Number of copies.
        axis : (ax, ay, az)
            Rotation axis direction.
        total_angle : float
            Total swept angle in radians (last copy at this angle).
        center : (cx, cy, cz), default (0, 0, 0)
            Point on the rotation axis.
        """
        self._require_inactive("pattern_polar")
        self._require_has_file("pattern_polar")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer; got {n!r}")
        if not isinstance(label, str) or not label:
            raise ValueError("pattern_polar() requires a non-empty `label=`.")

        ax, ay, az = (float(c) for c in axis)
        cx, cy, cz = (float(c) for c in center)
        step = float(total_angle) / float(n)

        return [
            self._make_pattern_item(
                label=f"{label}_{i}",
                translate_offset=(0.0, 0.0, 0.0),
                rotate_spec=(i * step, ax, ay, az, cx, cy, cz),
            )
            for i in range(1, n + 1)
        ]

    # ------------------------------------------------------------------
    # Alignment — translate self so a labeled feature lands somewhere
    # ------------------------------------------------------------------

    def align_to(
        self,
        other: "Part",
        *,
        source: str,
        target: str,
        on: "str | tuple[str, ...]",
        offset: float = 0.0,
    ) -> "PartEdit":
        """Translate this Part so its ``source`` label aligns with
        ``other``'s ``target`` label along the chosen axes.

        Computes ``source`` centroid in this Part's **live session**,
        reads ``target`` centroid from ``other``'s STEP sidecar (so
        ``other`` must have been saved — auto-persist counts), then
        applies the masked translation via :meth:`translate`.

        Parameters
        ----------
        other : Part
            Reference Part.  Must have a saved sidecar (``has_file``
            true and a ``.apegmsh.json`` written next to the STEP).
            Passing an Instance is rejected — use
            :meth:`Instance.edit.align_to` for the assembly side.
        source : str
            Label name on this Part (the feature that moves).
        target : str
            Label name on ``other`` (the feature it lands on).
        on : {"x", "y", "z", "all"} or iterable of those
            Axes on which to match centroids.  Other axes untouched.
        offset : float, default 0.0
            Signed gap along the single ``on`` axis.  Combining a
            non-zero offset with multi-axis ``on`` raises ValueError.

        Returns
        -------
        PartEdit
            ``self`` for chaining.

        Raises
        ------
        RuntimeError
            If this Part is not active or ``other`` has no sidecar.
        TypeError
            If ``other`` is not a Part (e.g. an Instance).
        LookupError
            If ``source`` or ``target`` cannot be resolved.
        """
        from ._align import (
            compute_align_translation,
            label_centroid_from_sidecar,
            label_centroid_live,
        )

        self._require_active("align_to")
        # Duck-type check (not isinstance) so the call survives module
        # re-imports in test infra that purges apeGmsh from sys.modules.
        # Reject Instances by their distinguishing attribute.
        if hasattr(other, 'entities') and hasattr(other, 'label_names'):
            raise TypeError(
                "align_to() got an Instance for `other`; for "
                "Instance-to-Instance alignment use Instance.edit.align_to()."
            )
        if not (
            hasattr(other, 'has_file')
            and hasattr(other, 'file_path')
            and hasattr(other, 'name')
        ):
            raise TypeError(
                f"align_to() expects a Part as `other`; got "
                f"{type(other).__name__}."
            )
        if other is self._part:
            raise ValueError(
                "align_to() requires a different Part as `other`."
            )
        if not other.has_file:
            raise RuntimeError(
                f"align_to() requires `other` ({other.name!r}) to have "
                f"been saved.  Exit its `with` block (auto-persist) or "
                f"call other.save() first."
            )

        source_com = label_centroid_live(source)
        target_com = label_centroid_from_sidecar(target, other.file_path)

        dx, dy, dz = compute_align_translation(
            source_com, target_com, on, offset,
        )
        return self.translate(dx, dy, dz)

    def align_to_point(
        self,
        point: tuple[float, float, float],
        *,
        source: str,
        on: "str | tuple[str, ...]",
        offset: float = 0.0,
    ) -> "PartEdit":
        """Translate this Part so its ``source`` label centroid lands
        at ``point`` along the chosen axes.

        Like :meth:`align_to` but the target is a coordinate rather
        than another Part's labeled feature.  No sidecar lookup
        needed.

        Parameters
        ----------
        point : (px, py, pz)
            World point in this Part's local frame (i.e. the same
            frame in which ``source`` lives).
        source : str
            Label on this Part.
        on : {"x", "y", "z", "all"} or iterable
            Axes to align.
        offset : float, default 0.0
            Signed gap along the single ``on`` axis.
        """
        from ._align import (
            compute_align_translation,
            label_centroid_live,
        )

        self._require_active("align_to_point")
        source_com = label_centroid_live(source)
        target = (float(point[0]), float(point[1]), float(point[2]))
        dx, dy, dz = compute_align_translation(
            source_com, target, on, offset,
        )
        return self.translate(dx, dy, dz)

    # ------------------------------------------------------------------
    # Pattern internals
    # ------------------------------------------------------------------

    def _make_pattern_item(
        self,
        *,
        label: str,
        translate_offset: tuple[float, float, float],
        rotate_spec: tuple[float, float, float, float, float, float, float] | None,
    ) -> "Part":
        """Build one pattern entry: copy source STEP, open it, transform,
        let auto-persist write the transformed geometry.
        """
        # File-clone first (no gmsh interaction)
        item = self.copy(label=label)
        # Then open + transform + auto-persist
        # Reset file_path so auto-persist on exit writes a fresh STEP
        # with the transform baked in (otherwise end() skips persist).
        cloned_step = item.file_path
        item.file_path = None
        item._owns_file = False  # the original tempfile will be reused
        try:
            with item:
                gmsh.merge(str(cloned_step))
                gmsh.model.occ.synchronize()
                if rotate_spec is not None:
                    angle, ax, ay, az, cx, cy, cz = rotate_spec
                    item.edit.rotate(angle, ax, ay, az, center=(cx, cy, cz))
                tx, ty, tz = translate_offset
                if tx or ty or tz:
                    item.edit.translate(tx, ty, tz)
        finally:
            # The new auto-persisted file lives in a new tempdir;
            # the old cloned_step is now stale.
            try:
                cloned_step.unlink(missing_ok=True)
                # If the parent dir is empty (the old _temp_dir from
                # copy()), remove it too
                if cloned_step.parent.exists() and not any(cloned_step.parent.iterdir()):
                    cloned_step.parent.rmdir()
            except OSError:
                pass
        return item


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _flatten_matrix(matrix4x4) -> list[float]:
    """Coerce a 4×4 matrix in any common form to a flat list of 16 floats."""
    try:
        seq = list(matrix4x4)
    except TypeError as exc:
        raise TypeError(
            f"affine() expected a 4x4 matrix or 16-element sequence; "
            f"got {type(matrix4x4).__name__}"
        ) from exc

    if len(seq) == 16 and all(_is_scalar(v) for v in seq):
        return [float(v) for v in seq]
    if len(seq) == 4 and all(_has_len(row, 4) for row in seq):
        return [float(v) for row in seq for v in row]
    if hasattr(matrix4x4, "shape") and getattr(matrix4x4, "shape", None) == (4, 4):
        return [float(v) for v in matrix4x4.flatten().tolist()]
    raise ValueError(
        f"affine() could not interpret matrix of shape "
        f"{_describe_shape(matrix4x4)} as 4x4."
    )


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float))


def _has_len(v, n: int) -> bool:
    try:
        return len(v) == n
    except TypeError:
        return False


def _describe_shape(m) -> str:
    if hasattr(m, "shape"):
        return str(m.shape)
    try:
        return f"len={len(m)}"
    except TypeError:
        return type(m).__name__
