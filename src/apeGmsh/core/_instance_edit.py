"""
InstanceEdit — whole-Instance transforms in the assembly session.
==================================================================

Accessed as ``inst.edit`` after ``inst = g.parts.add(...)``.  Mirrors
:class:`apeGmsh.core._part_edit.PartEdit` but operates on the
Instance's dimtags within the **live assembly's** Gmsh session.

Differences from ``Part.edit``:

* Operates on a subset of entities (``inst.entities``), not the
  whole gmsh model.
* :meth:`delete` also unregisters the Instance from the parent
  ``parts._instances`` registry.
* The session active state is the assembly's, not a Part's.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from ._parts_registry import Instance, PartsRegistry


class InstanceEdit:
    """Operations on a placed :class:`Instance`. Registered as ``inst.edit``."""

    __slots__ = ("_inst", "_registry", "_deleted")

    def __init__(self, instance: "Instance", registry: "PartsRegistry") -> None:
        self._inst = instance
        self._registry = registry
        self._deleted = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_alive(self, op: str) -> None:
        if self._deleted:
            raise RuntimeError(
                f"Instance.edit.{op}() called on a deleted instance "
                f"({self._inst.label!r})."
            )

    def _top_dimtags(self) -> list[tuple[int, int]]:
        """Return dimtags for the top-dimension entities of this instance.

        Mirrors what ``_apply_transforms`` operates on — the lower-dim
        boundary entities follow their parents under OCC transforms,
        so we don't need to enumerate them.
        """
        if not self._inst.entities:
            return []
        top_dim = max(self._inst.entities)
        return [(top_dim, t) for t in self._inst.entities[top_dim]]

    def _refresh_bbox(self) -> None:
        """Recompute the cached bounding box after a transform."""
        all_dimtags = [
            (d, t) for d, ts in self._inst.entities.items() for t in ts
        ]
        self._inst.bbox = self._registry._compute_bbox(all_dimtags)

    # ------------------------------------------------------------------
    # Translate / rotate
    # ------------------------------------------------------------------

    def translate(self, dx: float, dy: float, dz: float) -> "InstanceEdit":
        """Translate the instance by ``(dx, dy, dz)``.

        Returns ``self`` for chaining.
        """
        self._require_alive("translate")
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return self
        dimtags = self._top_dimtags()
        if dimtags:
            gmsh.model.occ.translate(dimtags, float(dx), float(dy), float(dz))
            gmsh.model.occ.synchronize()
            self._refresh_bbox()
        return self

    def rotate(
        self,
        angle: float,
        ax: float,
        ay: float,
        az: float,
        *,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "InstanceEdit":
        """Rotate the instance by ``angle`` (radians) about an axis.

        See :meth:`Part.edit.rotate` for the parameter reference.
        """
        self._require_alive("rotate")
        if angle == 0.0:
            return self
        dimtags = self._top_dimtags()
        if dimtags:
            cx, cy, cz = center
            gmsh.model.occ.rotate(
                dimtags,
                float(cx), float(cy), float(cz),
                float(ax), float(ay), float(az),
                float(angle),
            )
            gmsh.model.occ.synchronize()
            self._refresh_bbox()
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
    ) -> "InstanceEdit":
        """Reflect the instance across a plane.

        See :meth:`Part.edit.mirror` for the parameter reference.
        """
        self._require_alive("mirror")
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
        px, py, pz = point
        d = -(nx * px + ny * py + nz * pz)
        dimtags = self._top_dimtags()
        if dimtags:
            gmsh.model.occ.mirror(dimtags, nx, ny, nz, d)
            gmsh.model.occ.synchronize()
            self._refresh_bbox()
        return self

    def scale(
        self,
        factor: float,
        *,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "InstanceEdit":
        """Uniform scale by ``factor`` about ``center``."""
        self._require_alive("scale")
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
    ) -> "InstanceEdit":
        """Non-uniform scale by ``(sx, sy, sz)`` about ``center``."""
        self._require_alive("dilate")
        if sx == 1.0 and sy == 1.0 and sz == 1.0:
            return self
        dimtags = self._top_dimtags()
        if dimtags:
            cx, cy, cz = center
            gmsh.model.occ.dilate(
                dimtags,
                float(cx), float(cy), float(cz),
                float(sx), float(sy), float(sz),
            )
            gmsh.model.occ.synchronize()
            self._refresh_bbox()
        return self

    def affine(self, matrix4x4) -> "InstanceEdit":
        """Apply a general 4×4 affine transform.

        See :meth:`Part.edit.affine` for the parameter reference.
        """
        self._require_alive("affine")
        from ._part_edit import _flatten_matrix

        flat = _flatten_matrix(matrix4x4)
        if len(flat) != 16:
            raise ValueError(
                f"affine() requires a 4x4 matrix (16 values); got {len(flat)}"
            )
        dimtags = self._top_dimtags()
        if dimtags:
            gmsh.model.occ.affineTransform(dimtags, flat)
            gmsh.model.occ.synchronize()
            self._refresh_bbox()
        return self

    # ------------------------------------------------------------------
    # Delete — also unregisters from the parent registry
    # ------------------------------------------------------------------

    def delete(self) -> None:
        """Remove the instance's entities from the assembly session
        and unregister from ``g.parts._instances``.

        After ``delete()``, subsequent calls on this ``edit`` object
        raise ``RuntimeError``.  The label is freed and may be reused
        by a fresh ``parts.add()``.
        """
        self._require_alive("delete")
        all_dimtags = [
            (d, t) for d, ts in self._inst.entities.items() for t in ts
        ]
        if all_dimtags:
            gmsh.model.occ.remove(all_dimtags, recursive=True)
            gmsh.model.occ.synchronize()
        # Unregister from the parent registry so the label is free again
        self._registry._instances.pop(self._inst.label, None)
        # Wipe the entity map so any lingering references see an empty inst
        self._inst.entities = {}
        self._inst.bbox = None
        self._deleted = True

    # ------------------------------------------------------------------
    # Copy — duplicate the instance's geometry + labels
    # ------------------------------------------------------------------

    def copy(self, *, label: str) -> "Instance":
        """Duplicate this instance's geometry into a new Instance.

        Uses ``gmsh.model.occ.copy()`` to clone the dimtags (the new
        entities live in the same assembly session).  All Part-level
        labels carried by this instance are recreated under the new
        instance's label prefix — so e.g. ``b1.top_flange`` becomes
        ``b2.top_flange`` on the copy.

        Parameters
        ----------
        label : str, required
            New instance label.  If the requested label is already
            taken in this session, a 4-character random hex suffix is
            appended and a warning emitted.

        Returns
        -------
        Instance
            The new Instance, registered in ``g.parts`` and ready
            for further edits.

        Raises
        ------
        RuntimeError
            If this instance has already been deleted.
        ValueError
            If ``label`` is empty.
        """
        self._require_alive("copy")
        if not isinstance(label, str) or not label:
            raise ValueError("copy() requires a non-empty `label=` argument.")

        from ._parts_registry import Instance
        from ._part_edit import _resolve_unique_name as _resolve_unique_part
        # Resolve clash against existing instance labels
        new_label = _resolve_unique_instance_label(label, self._registry)

        src_dimtags = [
            (d, t) for d, ts in self._inst.entities.items() for t in ts
        ]
        if not src_dimtags:
            raise RuntimeError(
                "copy() called on an instance with no geometry."
            )

        new_dimtags = gmsh.model.occ.copy(src_dimtags)
        gmsh.model.occ.synchronize()

        # Build src_tag -> new_tag map per dim (gmsh preserves order)
        tag_map: dict[int, dict[int, int]] = {}
        for (sd, st), (nd, nt) in zip(src_dimtags, new_dimtags):
            tag_map.setdefault(sd, {})[st] = nt

        # New entities dict
        new_entities: dict[int, list[int]] = {}
        for (nd, nt) in new_dimtags:
            new_entities.setdefault(nd, []).append(nt)

        # Recreate labels under the new prefix
        new_label_names = self._rebrand_labels(
            tag_map, new_entities, new_label,
        )

        new_inst = Instance(
            label=new_label,
            part_name=self._inst.part_name,
            entities=new_entities,
            bbox=self._registry._compute_bbox(new_dimtags),
            label_names=new_label_names,
        )
        self._registry._register_instance(new_inst)
        return new_inst

    def _rebrand_labels(
        self,
        tag_map: dict[int, dict[int, int]],
        new_entities: dict[int, list[int]],
        new_label: str,
    ) -> list[str]:
        """Create copies of every Part-level label carried by this
        instance under the new label's prefix.

        Returns the list of label names registered for the new
        instance (mirroring ``Instance.label_names``).
        """
        labels_comp = getattr(self._registry._parent, "labels", None)
        if labels_comp is None:
            # Bare-bones session without label support (unlikely, but
            # don't crash on it).
            return []

        src_prefix = f"{self._inst.label}."
        new_prefix = f"{new_label}."
        new_label_names: list[str] = []

        # Iterate Part-prefixed labels (e.g. "b1.top_flange") and
        # rebrand each.  The umbrella label (== self._inst.label
        # exactly, no dot) is handled separately below.
        for src_name in list(self._inst.label_names):
            if src_name == self._inst.label:
                continue  # umbrella, handled below
            if not src_name.startswith(src_prefix):
                continue  # unexpected format, skip
            suffix = src_name[len(src_prefix):]
            new_name = new_prefix + suffix

            # The label may live at different dims; query each dim
            # we have a tag mapping for.
            new_tags_by_dim: dict[int, list[int]] = {}
            for dim, mapping in tag_map.items():
                try:
                    src_tags = labels_comp.entities(src_name, dim=dim)
                except (KeyError, ValueError):
                    continue
                mapped = [mapping[t] for t in src_tags if t in mapping]
                if mapped:
                    new_tags_by_dim[dim] = mapped

            for dim, new_tags in new_tags_by_dim.items():
                try:
                    labels_comp.add(dim, new_tags, name=new_name)
                except Exception:
                    continue
            if new_tags_by_dim:
                new_label_names.append(new_name)

        # Umbrella label (covers all top-dim entities)
        if new_entities:
            top_dim = max(new_entities)
            try:
                labels_comp.add(top_dim, new_entities[top_dim], name=new_label)
                new_label_names.append(new_label)
            except Exception:
                pass

        return new_label_names

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def pattern_linear(
        self,
        *,
        label: str,
        n: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> list["Instance"]:
        """Create ``n`` translated copies of this instance.

        Each copy ``i`` (1..n) is shifted by ``(i*dx, i*dy, i*dz)``
        from the source.  The source itself is not modified.

        Returns a list of ``n`` new :class:`Instance` objects, all
        registered in ``g.parts``.
        """
        self._require_alive("pattern_linear")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer; got {n!r}")
        if not isinstance(label, str) or not label:
            raise ValueError("pattern_linear() requires a non-empty `label=`.")

        out: list["Instance"] = []
        for i in range(1, n + 1):
            new_inst = self.copy(label=f"{label}_{i}")
            new_inst.edit.translate(i * dx, i * dy, i * dz)
            out.append(new_inst)
        return out

    def pattern_polar(
        self,
        *,
        label: str,
        n: int,
        axis: tuple[float, float, float],
        total_angle: float,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> list["Instance"]:
        """Create ``n`` rotated copies of this instance.

        Each copy ``i`` (1..n) is rotated by ``i * total_angle / n``
        about ``axis`` through ``center``.  ``total_angle`` is in
        radians (``2*pi`` for a full revolution).
        """
        self._require_alive("pattern_polar")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer; got {n!r}")
        if not isinstance(label, str) or not label:
            raise ValueError("pattern_polar() requires a non-empty `label=`.")

        ax, ay, az = (float(c) for c in axis)
        step = float(total_angle) / float(n)

        out: list["Instance"] = []
        for i in range(1, n + 1):
            new_inst = self.copy(label=f"{label}_{i}")
            new_inst.edit.rotate(i * step, ax, ay, az, center=center)
            out.append(new_inst)
        return out

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def align_to(
        self,
        other: "Instance",
        *,
        source: str,
        target: str,
        on: "str | tuple[str, ...]",
        offset: float = 0.0,
    ) -> "InstanceEdit":
        """Translate this instance so its ``source`` label aligns
        with ``other``'s ``target`` label along the chosen axes.

        Both instances must live in the same session (the assembly).
        Both centroids are read live from gmsh — no sidecar lookup
        needed.

        Parameters
        ----------
        other : Instance
            Reference instance.  Cross-Part alignment is rejected
            (Parts live in their own sessions; use Part.edit.align_to
            instead).
        source : str
            Label suffix on this instance (e.g. ``"top_flange"``).
            Resolved to ``f"{self.label}.{source}"``.
        target : str
            Label suffix on ``other``, resolved to
            ``f"{other.label}.{target}"``.
        on : {"x","y","z","all"} or iterable
            Axes on which to match centroids.
        offset : float, default 0.0
            Signed gap along the (single) ``on`` axis.

        Returns
        -------
        InstanceEdit
            ``self`` for chaining.
        """
        from ._align import (
            compute_align_translation,
            label_centroid_live,
        )

        self._require_alive("align_to")
        # Duck-type check (resilient to module re-imports).
        if not (
            hasattr(other, 'entities')
            and hasattr(other, 'label_names')
            and hasattr(other, 'label')
        ):
            raise TypeError(
                f"align_to() expects an Instance as `other`; got "
                f"{type(other).__name__}.  For Part-to-Part alignment "
                f"use Part.edit.align_to() instead."
            )
        if other is self._inst:
            raise ValueError(
                "align_to() requires a different Instance as `other`."
            )

        source_suffix = _normalize_label_arg(source, self._inst.label)
        target_suffix = _normalize_label_arg(target, other.label)
        source_full = f"{self._inst.label}.{source_suffix}"
        target_full = f"{other.label}.{target_suffix}"

        source_com = label_centroid_live(source_full)
        target_com = label_centroid_live(target_full)

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
    ) -> "InstanceEdit":
        """Translate this instance so its ``source`` label centroid
        lands at ``point`` along the chosen axes.
        """
        from ._align import (
            compute_align_translation,
            label_centroid_live,
        )

        self._require_alive("align_to_point")
        source_suffix = _normalize_label_arg(source, self._inst.label)
        source_full = f"{self._inst.label}.{source_suffix}"
        source_com = label_centroid_live(source_full)
        target = (float(point[0]), float(point[1]), float(point[2]))
        dx, dy, dz = compute_align_translation(
            source_com, target, on, offset,
        )
        return self.translate(dx, dy, dz)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _normalize_label_arg(arg: str, expected_prefix: str) -> str:
    """Accept a label as either a bare suffix (``"top_flange"``) or a
    fully-qualified name (``"beam_1.top_flange"``) and return the bare
    suffix in either case.

    Supports the IDE-autocomplete ergonomics of
    ``inst.edit.align_to(other, source=inst.labels.top_flange, ...)``
    where ``inst.labels.top_flange`` returns the fully-qualified
    string.

    If the qualified prefix does not match ``expected_prefix``, raises
    :class:`ValueError` — this catches mistakes where a user passes a
    label that belongs to a different instance.
    """
    if '.' not in arg:
        return arg
    head, _, tail = arg.partition('.')
    if head != expected_prefix:
        raise ValueError(
            f"Label {arg!r} belongs to instance {head!r}, but this "
            f"call expects a label on instance {expected_prefix!r}.  "
            f"Pass the bare suffix or the matching instance's qualified "
            f"label."
        )
    return tail


def _resolve_unique_instance_label(
    requested: str,
    registry: "PartsRegistry",
) -> str:
    """Return ``requested`` if free in the registry, else ``{requested}_{hex}``
    with a warning."""
    import secrets
    import warnings

    if requested not in registry._instances:
        return requested
    fallback = f"{requested}_{secrets.token_hex(2)}"
    while fallback in registry._instances:
        fallback = f"{requested}_{secrets.token_hex(2)}"
    warnings.warn(
        f"Instance label {requested!r} is already in use; using "
        f"{fallback!r} instead.",
        stacklevel=3,
    )
    return fallback
