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

from .._internal.tag_resolution import resolve_tag
from .._internal.types import Primitive, Section, UniaxialMaterial

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "CircPatch",
    "Fiber",
    "FiberPoint",
    "RectPatch",
    "StraightLayer",
    "W_fiber",
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
class CircPatch:
    """One ``patch circ`` cell of a Fiber section (ADR 0080 B2).

    A value object like :class:`RectPatch`. Emits ``patch circ
    $matTag $nCirc $nRad $yC $zC $intRad $extRad $startAng $endAng``
    (angles always emitted explicitly, degrees).

    Parameters
    ----------
    material
        The uniaxial material this patch's fibers integrate.
    n_circ, n_rad
        Subdivisions along the circumference / the radius.
    yC, zC
        Centre coordinates.
    int_rad, ext_rad
        Inner / outer radius (``0 <= int_rad < ext_rad``; a solid
        disc has ``int_rad=0``).
    start_ang, end_ang
        Angular extent in degrees (default full circle).
    """

    material: UniaxialMaterial
    n_circ: int
    n_rad: int
    yC: float
    zC: float
    int_rad: float
    ext_rad: float
    start_ang: float = 0.0
    end_ang: float = 360.0

    def __post_init__(self) -> None:
        if self.n_circ <= 0 or self.n_rad <= 0:
            raise ValueError(
                f"CircPatch: n_circ and n_rad must be > 0, got "
                f"n_circ={self.n_circ}, n_rad={self.n_rad}."
            )
        if not (0.0 <= self.int_rad < self.ext_rad):
            raise ValueError(
                f"CircPatch: need 0 <= int_rad < ext_rad, got "
                f"int_rad={self.int_rad}, ext_rad={self.ext_rad}."
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

    patches: tuple["RectPatch | CircPatch", ...] = ()
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
            mat_tag = resolve_tag(emitter, patch.material)
            if isinstance(patch, CircPatch):
                emitter.patch(
                    "circ",
                    mat_tag,
                    patch.n_circ, patch.n_rad,
                    patch.yC, patch.zC,
                    patch.int_rad, patch.ext_rad,
                    patch.start_ang, patch.end_ang,
                )
            else:
                emitter.patch(
                    "rect",
                    mat_tag,
                    patch.ny, patch.nz,
                    patch.yI, patch.zI,
                    patch.yJ, patch.zJ,
                )

        for layer in self.layers:
            mat_tag = resolve_tag(emitter, layer.material)
            emitter.layer(
                "straight",
                mat_tag,
                layer.n_bars,
                layer.area,
                layer.yI, layer.zI,
                layer.yJ, layer.zJ,
            )

        for fpt in self.fibers:
            mat_tag = resolve_tag(emitter, fpt.material)
            emitter.fiber(fpt.y, fpt.z, fpt.area, mat_tag)

        emitter.section_close()


# ---------------------------------------------------------------------------
# Parametric builders — boilerplate-free Fiber-section constructors
# ---------------------------------------------------------------------------

def W_fiber(
    *,
    bf: float,
    tf: float,
    hw: float,
    tw: float,
    material: UniaxialMaterial,
    ny_flange: int = 2,
    nz_flange: int = 8,
    ny_web: int = 8,
    nz_web: int = 1,
    GJ: float | None = None,
) -> Fiber:
    """Build a fiber :class:`Fiber` section for a built-up W shape.

    Replaces ~30 lines of :class:`RectPatch` boilerplate.  Section
    geometry follows the canonical OpenSees convention:

    * ``y`` is the major-bending axis (vertical, upward positive),
    * ``z`` is the minor-bending axis (horizontal),
    * the centroid of the section sits at the origin (``y=0``, ``z=0``).

    The three rectangular regions:

    * **Top flange** — ``y ∈ [hw/2, hw/2 + tf]``,
      ``z ∈ [-bf/2, +bf/2]``
    * **Bottom flange** — ``y ∈ [-hw/2 - tf, -hw/2]``,
      ``z ∈ [-bf/2, +bf/2]``
    * **Web** — ``y ∈ [-hw/2, +hw/2]``, ``z ∈ [-tw/2, +tw/2]``

    Parameters
    ----------
    bf
        Flange width (z extent).
    tf
        Flange thickness (y extent).
    hw
        **Clear** web height between the flanges — the y-extent of
        the web region only.  This is NOT the catalog total depth
        ``d``.  For an AISC catalog shape with depth ``d`` and flange
        thickness ``tf``, pass ``hw = d - 2*tf``.

        Worked example — W360×64 (d = 347 mm, bf = 203 mm, tf = 13.5 mm,
        tw = 7.7 mm)::

            sec = ops.section.W_fiber(
                bf=203e-3, tf=13.5e-3,
                hw=347e-3 - 2 * 13.5e-3,   # = 0.320 m clear web
                tw=7.7e-3,
                material=steel,
            )

        The emitted section's full y-extent is ``hw + 2*tf`` (i.e. the
        catalog ``d``).
    tw
        Web thickness (z extent).
    material
        Uniform uniaxial material for every fiber.  Per-band
        residual-stress variants — e.g. the ECCS welded-I pattern
        (+0.5/-0.3/+0.2 Fy) — are not supported by this builder
        directly; for that, build the section manually or wrap the
        base material via :class:`InitialStress` at the patch level.
    ny_flange, nz_flange
        Fiber subdivision count for each flange patch (``ny`` is the
        y-direction count, ``nz`` is the z-direction count).
    ny_web, nz_web
        Fiber subdivision count for the web patch.
    GJ
        Optional torsional stiffness ``GJ``.

    Returns
    -------
    Fiber
        A typed :class:`Fiber` primitive — NOT registered with any
        bridge.  Register it on the bridge via
        ``ops.section.W_fiber(...)`` (which wraps this helper) or
        pass it through ``apeSees._register`` directly.

    Raises
    ------
    ValueError
        If any dimension is non-positive or any subdivision count is
        not strictly positive.
    """
    if bf <= 0 or tf <= 0 or hw <= 0 or tw <= 0:
        raise ValueError(
            f"W_fiber: all dimensions must be > 0, got "
            f"bf={bf}, tf={tf}, hw={hw}, tw={tw}."
        )
    if tw >= bf:
        raise ValueError(
            f"W_fiber: web thickness ({tw}) must be smaller than "
            f"flange width ({bf})."
        )

    top_flange = RectPatch(
        material=material,
        ny=ny_flange, nz=nz_flange,
        yI=hw / 2.0,        zI=-bf / 2.0,
        yJ=hw / 2.0 + tf,   zJ=+bf / 2.0,
    )
    bottom_flange = RectPatch(
        material=material,
        ny=ny_flange, nz=nz_flange,
        yI=-hw / 2.0 - tf,  zI=-bf / 2.0,
        yJ=-hw / 2.0,       zJ=+bf / 2.0,
    )
    web = RectPatch(
        material=material,
        ny=ny_web, nz=nz_web,
        yI=-hw / 2.0, zI=-tw / 2.0,
        yJ=+hw / 2.0, zJ=+tw / 2.0,
    )
    return Fiber(
        patches=(top_flange, bottom_flange, web),
        GJ=GJ,
    )
