"""
Fiber section — typed primitives for OpenSees ``section Fiber``.

The Fiber section is a **block-emit** primitive: it opens a section
context, populates it with patches / layers / individual fibers, and
closes the block. The Protocol exposes this through the
``section_open`` / ``section_close`` pair plus the ``patch`` /
``layer`` / ``fiber`` calls.

Per the OpenSees Tcl manual:

* ``section Fiber $tag <-GJ $GJ> { ... patch / layer / fiber ... }``
* ``patch rect $matTag $numSubdivIJ $numSubdivJK $yI $zI $yJ $zJ``
* ``layer straight $matTag $numFibers $area $yI $zI $yJ $zJ``
* ``fiber $y $z $area $matTag``

This module supplies four classes:

* :class:`Fiber` — the section primitive (a :class:`Section`).
* :class:`RectPatch`, :class:`StraightLayer`, :class:`FiberPoint` —
  value objects (frozen dataclasses, not :class:`Primitive`) that
  describe individual sub-elements of a fiber section. They are not
  registered standalone; they hold typed references to the
  :class:`UniaxialMaterial` instances they integrate.

OPEN COORDINATOR QUESTION — material-tag resolution
====================================================

OpenSees's ``patch`` / ``layer`` / ``fiber`` commands take material
tags **positionally**. Phase 0 stores tags in
``apeSees._tags`` (an external dict), not on the primitive. So
``Fiber._emit(emitter, tag)`` cannot inline the material's tag — it
needs a resolver.

Phase 1C ships the interim contract in :mod:`section._tag_resolver`:
the bridge attaches a callable to the emitter; sections look it up.
This is opt-in (the Protocol is unchanged) and unit tests can
install a manual resolver.

The coordinator will harmonize this before Phase 2 (elements
reference sections + transforms — the same problem at one more
level of composition). See the docstring of
:mod:`section._tag_resolver` for the full discussion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import Primitive, Section, UniaxialMaterial
from ._tag_resolver import resolve_mat_tag

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Fiber",
    "FiberPoint",
    "RectPatch",
    "StraightLayer",
]


# ---------------------------------------------------------------------------
# Value objects — patches, layers, individual fibers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class RectPatch:
    """One ``patch rect`` cell of a Fiber section.

    A value object — not a :class:`Primitive`, no tag, not registered
    standalone. Held in a tuple by the parent :class:`Fiber`.

    Parameters
    ----------
    material
        The uniaxial material this patch's fibers integrate.
    ny
        Number of subdivisions along the local y-axis (the I-J edge).
    nz
        Number of subdivisions along the local z-axis (the J-K edge).
    yI, zI
        Coordinates of corner I (one diagonal corner).
    yJ, zJ
        Coordinates of corner J (the opposite diagonal corner).
    """

    material: UniaxialMaterial
    ny: int
    nz: int
    yI: float
    zI: float
    yJ: float
    zJ: float

    def __post_init__(self) -> None:
        if self.ny <= 0 or self.nz <= 0:
            raise ValueError(
                f"RectPatch: ny and nz must be > 0, got "
                f"ny={self.ny}, nz={self.nz}."
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class StraightLayer:
    """One ``layer straight`` of a Fiber section.

    A value object representing a row of identical bars between two
    points.

    Parameters
    ----------
    material
        The uniaxial material the bars are made of.
    n_bars
        Number of bars in the layer. Must be >= 1.
    area
        Cross-sectional area of one bar.
    yI, zI
        Coordinates of the first bar's centroid.
    yJ, zJ
        Coordinates of the last bar's centroid.
    """

    material: UniaxialMaterial
    n_bars: int
    area: float
    yI: float
    zI: float
    yJ: float
    zJ: float

    def __post_init__(self) -> None:
        if self.n_bars < 1:
            raise ValueError(
                f"StraightLayer: n_bars must be >= 1, got {self.n_bars}."
            )
        if self.area <= 0:
            raise ValueError(
                f"StraightLayer: area must be > 0, got {self.area}."
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class FiberPoint:
    """One ``fiber`` point of a Fiber section.

    A value object: a single fiber at coordinates ``(y, z)`` with the
    given area and uniaxial material.

    Parameters
    ----------
    material
        The uniaxial material this fiber integrates.
    y, z
        Local-axis coordinates of the fiber centroid.
    area
        Cross-sectional area of the fiber.
    """

    material: UniaxialMaterial
    y: float
    z: float
    area: float

    def __post_init__(self) -> None:
        if self.area <= 0:
            raise ValueError(
                f"FiberPoint: area must be > 0, got {self.area}."
            )


# ---------------------------------------------------------------------------
# Fiber section primitive
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Fiber(Section):
    """``section Fiber`` — block-emit fiber section.

    A Fiber section is a collection of :class:`RectPatch`,
    :class:`StraightLayer`, and :class:`FiberPoint` value objects,
    each carrying a typed :class:`UniaxialMaterial` reference. The
    section emits as a Tcl-style block (``section_open`` →
    patch/layer/fiber calls → ``section_close``).

    Parameters
    ----------
    patches
        Tuple of :class:`RectPatch`. Defaults to empty.
    fibers
        Tuple of :class:`FiberPoint`. Defaults to empty.
    layers
        Tuple of :class:`StraightLayer`. Defaults to empty.
    GJ
        Optional torsional stiffness (``-GJ $GJ`` flag). Defaults to
        ``None`` (omit the flag).

    Notes
    -----
    At least one of ``patches`` / ``fibers`` / ``layers`` must be
    non-empty.

    Material-tag resolution at emit time uses the emitter-attached
    resolver from :mod:`section._tag_resolver` — see the open
    coordinator question in this module's docstring.
    """

    patches: tuple[RectPatch, ...] = ()
    fibers:  tuple[FiberPoint, ...] = ()
    layers:  tuple[StraightLayer, ...] = ()
    GJ: float | None = None

    def __post_init__(self) -> None:
        if not (self.patches or self.fibers or self.layers):
            raise ValueError(
                "Fiber: at least one of patches / fibers / layers "
                "must be non-empty."
            )
        if self.GJ is not None and self.GJ <= 0:
            raise ValueError(
                f"Fiber: GJ must be > 0 if supplied, got {self.GJ}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        """Return the unique materials referenced by patches / layers
        / fibers, in iteration order (deduped by ``id``)."""
        seen: dict[int, UniaxialMaterial] = {}
        for patch in self.patches:
            seen.setdefault(id(patch.material), patch.material)
        for layer in self.layers:
            seen.setdefault(id(layer.material), layer.material)
        for fpt in self.fibers:
            seen.setdefault(id(fpt.material), fpt.material)
        return tuple(seen.values())

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        if self.GJ is not None:
            emitter.section_open("Fiber", tag, "-GJ", self.GJ)
        else:
            emitter.section_open("Fiber", tag)

        for patch in self.patches:
            mat_tag = resolve_mat_tag(emitter, patch.material)
            emitter.patch(
                "rect",
                mat_tag,
                patch.ny, patch.nz,
                patch.yI, patch.zI,
                patch.yJ, patch.zJ,
            )

        for layer in self.layers:
            mat_tag = resolve_mat_tag(emitter, layer.material)
            emitter.layer(
                "straight",
                mat_tag,
                layer.n_bars,
                layer.area,
                layer.yI, layer.zI,
                layer.yJ, layer.zJ,
            )

        for fpt in self.fibers:
            mat_tag = resolve_mat_tag(emitter, fpt.material)
            emitter.fiber(fpt.y, fpt.z, fpt.area, mat_tag)

        emitter.section_close()
