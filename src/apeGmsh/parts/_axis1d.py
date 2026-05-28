"""
Axis1D — 1-D layered axis description for parametric part primitives.

Each Axis1D carries an ordered list of contiguous segments, each
tagged with a *region* label and an element count.  Used by
:class:`DRMBox` to describe the X / Y / Z stacking of the
Domain-Reduction-Method soil box.

The class is geometry-agnostic: it knows the breakpoints along an
axis and the per-segment element count, but never touches Gmsh.  The
DRM-box builder (:mod:`apeGmsh.parts.drm_box`) uses the breakpoints
to slice geometry and the ``region_of`` / ``count_for`` lookups to
classify sub-volumes after import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Axis1D:
    """A 1-D axis split into named, contiguous segments.

    Parameters
    ----------
    name : str
        Axis name — only used for error messages.
    segments : tuple of ``(region, lo, hi, count)``
        Ordered contiguous segments.  ``lo`` of each segment must
        match ``hi`` of the previous one; ``hi > lo``; ``count >= 1``.
    """

    name: str
    segments: Tuple[Tuple[str, float, float, int], ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError(f"Axis1D({self.name!r}): segments cannot be empty.")
        prev_hi: float | None = None
        for i, seg in enumerate(self.segments):
            if len(seg) != 4:
                raise ValueError(
                    f"Axis1D({self.name!r}) segment {i}: expected "
                    f"(region, lo, hi, count), got {seg!r}."
                )
            region, lo, hi, count = seg
            if not isinstance(region, str) or not region:
                raise ValueError(
                    f"Axis1D({self.name!r}) segment {i}: region must "
                    f"be a non-empty string, got {region!r}."
                )
            lo_f, hi_f = float(lo), float(hi)
            if not (hi_f > lo_f):
                raise ValueError(
                    f"Axis1D({self.name!r}) segment {i}: hi must be "
                    f"strictly greater than lo, got lo={lo_f}, hi={hi_f}."
                )
            if int(count) < 1:
                raise ValueError(
                    f"Axis1D({self.name!r}) segment {i}: count must "
                    f"be >= 1, got {count}."
                )
            if prev_hi is not None and abs(lo_f - prev_hi) > 1e-9:
                raise ValueError(
                    f"Axis1D({self.name!r}) segment {i}: lo={lo_f} "
                    f"does not match previous segment's hi={prev_hi}."
                )
            prev_hi = hi_f

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    @property
    def lo(self) -> float:
        """Leftmost / lowest coordinate."""
        return float(self.segments[0][1])

    @property
    def hi(self) -> float:
        """Rightmost / highest coordinate."""
        return float(self.segments[-1][2])

    @property
    def size(self) -> float:
        """Total span ``hi - lo``."""
        return self.hi - self.lo

    @property
    def breaks(self) -> list[float]:
        """Ordered list of all break coordinates, length ``n_segments + 1``.

        The first entry is the axis ``lo``, the last is ``hi``, and the
        interior entries are the boundaries between segments.
        """
        out = [float(self.segments[0][1])]
        for seg in self.segments:
            out.append(float(seg[2]))
        return out

    def slice_offsets(self) -> list[float]:
        """Interior break coordinates (everything in :attr:`breaks` except endpoints).

        These are the offsets the geometry builder slices the box at.
        """
        return self.breaks[1:-1]

    def witnesses(self) -> list[tuple[str, float]]:
        """One ``(region, midpoint)`` per segment, in order.

        Useful as test fixtures and as sanity-check probe points.
        """
        return [
            (str(region), 0.5 * (float(lo) + float(hi)))
            for region, lo, hi, _ in self.segments
        ]

    # ------------------------------------------------------------------
    # Region / count lookup
    # ------------------------------------------------------------------

    def region_of(self, value: float, *, tol: float = 1e-9) -> str:
        """Return the region whose segment contains ``value``.

        Points exactly on a segment boundary are assigned to the
        segment ending at that boundary (``value == hi``) when on the
        very last break, otherwise to the segment beginning there
        (``value == lo``).  ``tol`` widens both endpoint comparisons.

        Raises ``ValueError`` when ``value`` is outside ``[lo, hi]``.
        """
        v = float(value)
        if v < self.lo - tol or v > self.hi + tol:
            raise ValueError(
                f"Axis1D({self.name!r}).region_of({v}): value outside "
                f"axis range [{self.lo}, {self.hi}]."
            )
        for i, (region, lo, hi, _) in enumerate(self.segments):
            lo_f, hi_f = float(lo), float(hi)
            # Each interior boundary is owned by the segment to its
            # right; the very last boundary is owned by the last
            # segment so ``value == hi`` doesn't raise.
            is_last = i == len(self.segments) - 1
            if is_last:
                if lo_f - tol <= v <= hi_f + tol:
                    return str(region)
            else:
                if lo_f - tol <= v < hi_f - tol:
                    return str(region)
        # Fallback for the case where ``value`` lands exactly on the
        # final break (covered above) or the loop logic missed an
        # edge case — defensive only.
        return str(self.segments[-1][0])

    def count_for(self, value: float, *, tol: float = 1e-9) -> int:
        """Return the element count of the segment containing ``value``."""
        v = float(value)
        if v < self.lo - tol or v > self.hi + tol:
            raise ValueError(
                f"Axis1D({self.name!r}).count_for({v}): value outside "
                f"axis range [{self.lo}, {self.hi}]."
            )
        for i, (_region, lo, hi, count) in enumerate(self.segments):
            lo_f, hi_f = float(lo), float(hi)
            is_last = i == len(self.segments) - 1
            if is_last:
                if lo_f - tol <= v <= hi_f + tol:
                    return int(count)
            else:
                if lo_f - tol <= v < hi_f - tol:
                    return int(count)
        return int(self.segments[-1][3])

    # ------------------------------------------------------------------
    # Standard layouts used by DRMBox
    # ------------------------------------------------------------------

    @classmethod
    def symmetric_layered(
        cls,
        name: str,
        *,
        inner: tuple[float, int],
        layer: tuple[float, int],
        outer: tuple[float, int],
    ) -> "Axis1D":
        """5-segment symmetric layout — outer | layer | inner | layer | outer.

        The inner segment is centred on zero; the layer mirrors on
        both sides; the outer mirrors beyond the layer.  Each tuple
        is ``(size, n_elements)``.
        """
        inner_size, n_inner = float(inner[0]), int(inner[1])
        layer_size, n_layer = float(layer[0]), int(layer[1])
        outer_size, n_outer = float(outer[0]), int(outer[1])
        for s, who in (
            (inner_size, "inner"),
            (layer_size, "layer"),
            (outer_size, "outer"),
        ):
            if s <= 0:
                raise ValueError(
                    f"Axis1D.symmetric_layered({name!r}): "
                    f"{who} size must be > 0, got {s}."
                )

        x0 = -(inner_size / 2.0 + layer_size + outer_size)
        x1 = -(inner_size / 2.0 + layer_size)
        x2 = -inner_size / 2.0
        x3 = +inner_size / 2.0
        x4 = +(inner_size / 2.0 + layer_size)
        x5 = +(inner_size / 2.0 + layer_size + outer_size)

        return cls(
            name=name,
            segments=(
                ("outer", x0, x1, n_outer),
                ("layer", x1, x2, n_layer),
                ("inner", x2, x3, n_inner),
                ("layer", x3, x4, n_layer),
                ("outer", x4, x5, n_outer),
            ),
        )

    @classmethod
    def downward_layered(
        cls,
        name: str,
        *,
        top: tuple[float, int],
        mid: tuple[float, int],
        bottom: tuple[float, int],
    ) -> "Axis1D":
        """3-segment downward layout — bottom | mid | top, with ``hi = 0``.

        Convention for the DRM box: the free surface sits at z = 0
        (top of the inner box), and the stack descends downward.
        Each tuple is ``(size, n_elements)``.
        """
        top_size, n_top = float(top[0]), int(top[1])
        mid_size, n_mid = float(mid[0]), int(mid[1])
        bottom_size, n_bottom = float(bottom[0]), int(bottom[1])
        for s, who in (
            (top_size, "top"),
            (mid_size, "mid"),
            (bottom_size, "bottom"),
        ):
            if s <= 0:
                raise ValueError(
                    f"Axis1D.downward_layered({name!r}): "
                    f"{who} size must be > 0, got {s}."
                )

        z3 = 0.0
        z2 = -top_size
        z1 = -(top_size + mid_size)
        z0 = -(top_size + mid_size + bottom_size)

        return cls(
            name=name,
            segments=(
                ("bottom", z0, z1, n_bottom),
                ("mid",    z1, z2, n_mid),
                ("top",    z2, z3, n_top),
            ),
        )
