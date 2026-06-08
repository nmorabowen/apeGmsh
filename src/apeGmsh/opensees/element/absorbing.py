"""
Typed primitive for the ``ASDAbsorbingBoundary3D`` OpenSees element.

ASDEA's staged wave-absorbing boundary brick (Lysmer-Kuhlemeyer dashpots +
enforced free-field column).  One element per skin-region hex of a plane-wave
box (see :func:`apeGmsh.parts.plane_wave_box.build_plane_wave_box`).  ADR 0054,
slice AB-2.

Tcl signature::

    element ASDAbsorbingBoundary3D $tag $n1 ... $n8  $G $v $rho  $btype \\
        <-fx $tsTag> <-fy $tsTag> <-fz $tsTag>

The element takes **raw** ``G`` (shear modulus), ``v`` (Poisson), ``rho``
(density) doubles — not a material tag.  The user-facing facade
(``ops.element.ASDAbsorbingBoundary3D`` / ``ops.element.absorbing_boundary``)
accepts either those raw numbers or an ``ElasticIsotropic`` material it reads
``G = E / (2(1+v))`` from at construction; the frozen spec only ever stores the
three floats, so the material is never emitted and is not a dependency.

``btype`` is the boundary-position string (``L``/``R``/``F``/``K``/``B`` and
their OR-combinations) the element uses to orient its dashpots and free-field
column.  The ``-fx/-fy/-fz`` base-input time series are only consumed by the
element on **bottom** (``B``-containing) boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.tag_resolution import current_element_nodes, resolve_tag
from .._internal.types import Element, Primitive, TimeSeries

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = ["ASDAbsorbingBoundary3D"]

# The OpenSees-accepted face letters and their canonical order.
_BTYPE_ORDER = "BLRFK"
_BTYPE_LETTERS = frozenset(_BTYPE_ORDER)


@dataclass(frozen=True, kw_only=True, slots=True)
class ASDAbsorbingBoundary3D(Element):
    """``element ASDAbsorbingBoundary3D`` — staged absorbing-boundary brick.

    Fan-out-driven by the bridge: the per-element node tags are pushed into the
    emitter before each ``_emit``; ``G``/``v``/``rho``/``btype`` are homogeneous
    across the fan-out (one declaration per skin btype).
    """

    pg: str
    G: float
    v: float
    rho: float
    btype: str
    fx: TimeSeries | None = None
    fy: TimeSeries | None = None
    fz: TimeSeries | None = None

    def __post_init__(self) -> None:
        if self.G <= 0.0:
            raise ValueError(
                f"ASDAbsorbingBoundary3D: G (shear modulus) must be > 0, "
                f"got {self.G}."
            )
        if not (0.0 <= self.v < 0.5):
            raise ValueError(
                f"ASDAbsorbingBoundary3D: v (Poisson) must be in [0, 0.5), "
                f"got {self.v}."
            )
        if self.rho < 0.0:
            raise ValueError(
                f"ASDAbsorbingBoundary3D: rho must be >= 0, got {self.rho}."
            )
        bt = self.btype
        if not bt:
            raise ValueError("ASDAbsorbingBoundary3D: btype must be non-empty.")
        bad = sorted(set(bt) - _BTYPE_LETTERS)
        if bad:
            raise ValueError(
                f"ASDAbsorbingBoundary3D: btype {bt!r} has illegal letter(s) "
                f"{bad} — only {sorted(_BTYPE_LETTERS)} are allowed."
            )
        if len(set(bt)) != len(bt):
            raise ValueError(
                f"ASDAbsorbingBoundary3D: btype {bt!r} repeats a letter."
            )
        # Opposite faces never coexist on a real box cell and the element has
        # no branch for them (it would silently mis-size).
        if "L" in bt and "R" in bt:
            raise ValueError(
                f"ASDAbsorbingBoundary3D: btype {bt!r} pairs opposite faces "
                f"L and R."
            )
        if "F" in bt and "K" in bt:
            raise ValueError(
                f"ASDAbsorbingBoundary3D: btype {bt!r} pairs opposite faces "
                f"F and K."
            )
        # Base-input time series are only consumed on bottom boundaries
        # (OpenSees gates -fx/-fy/-fz on BND_BOTTOM); attaching one elsewhere
        # is a silent no-op at best, a parse error at worst.
        if "B" not in bt and (
            self.fx is not None or self.fy is not None or self.fz is not None
        ):
            raise ValueError(
                f"ASDAbsorbingBoundary3D: base input (-fx/-fy/-fz) is only "
                f"valid on a bottom boundary (btype containing 'B'), got "
                f"btype {bt!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        # The material is read for its numbers at the facade, never emitted —
        # only the (optional) base-input time series are real dependencies.
        return tuple(
            ts for ts in (self.fx, self.fy, self.fz) if ts is not None
        )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 8:
            raise ValueError(
                f"ASDAbsorbingBoundary3D: expected 8 node tags, got "
                f"{len(nodes)}."
            )
        args: list[int | float | str] = [
            *nodes, self.G, self.v, self.rho, self.btype,
        ]
        if self.fx is not None:
            args += ["-fx", resolve_tag(emitter, self.fx)]
        if self.fy is not None:
            args += ["-fy", resolve_tag(emitter, self.fy)]
        if self.fz is not None:
            args += ["-fz", resolve_tag(emitter, self.fz)]
        emitter.element("ASDAbsorbingBoundary3D", tag, *args)
