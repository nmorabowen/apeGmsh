"""
Beam-line sections — typed primitives for OpenSees ``section Elastic``.

The :class:`ElasticSection` wrapper covers both the 2-D and 3-D forms
of the OpenSees ``Elastic`` section. Per the OpenSees Tcl manual:

* 2-D: ``section Elastic $tag $E $A $Iz [$G $alphaY]``
* 3-D: ``section Elastic $tag $E $A $Iz $Iy $G $J [$alphaY $alphaZ]``

A single typed dataclass exposes both. The 3-D-only parameters
(``Iy``, ``J``) default to ``None``; supplying any of them switches
the section to the 3-D variant. ``G`` is shared (the optional shear
modulus in 2-D, required in 3-D).

The OpenSees command type token is **``Elastic``** (capital E,
matching the manual).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import Primitive, Section

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = ["ElasticSection"]


@dataclass(frozen=True, kw_only=True, slots=True)
class ElasticSection(Section):
    """``section Elastic`` — 2-D or 3-D linear-elastic beam section.

    The 3-D-only parameters (``Iy``, ``J``) are optional. Supplying
    either switches the section to the 3-D variant; in that case
    ``G`` and the partner ``Iy`` / ``J`` become required.

    Parameters
    ----------
    E
        Young's modulus.
    A
        Cross-sectional area.
    Iz
        Second moment of area about the local z-axis.
    Iy
        Second moment of area about the local y-axis. 3-D only.
    G
        Shear modulus. Required in 3-D; optional in 2-D (paired with
        ``alphaY``).
    J
        Torsional moment of inertia. 3-D only.
    alphaY
        Optional shear-area correction in the local y-axis.
        In 2-D this is paired with ``G``; in 3-D paired with
        ``alphaZ``.
    alphaZ
        Optional shear-area correction in the local z-axis. 3-D only.

    Notes
    -----
    The split between 2-D and 3-D mirrors OpenSees's
    ``ElasticSection2d`` and ``ElasticSection3d`` C++ classes, both
    of which the Tcl ``section Elastic ...`` command dispatches on
    based on argument count.
    """

    E:  float
    A:  float
    Iz: float
    Iy: float | None = None
    G:  float | None = None
    J:  float | None = None
    alphaY: float | None = None
    alphaZ: float | None = None

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(
                f"ElasticSection: E must be > 0, got {self.E}."
            )
        if self.A <= 0:
            raise ValueError(
                f"ElasticSection: A must be > 0, got {self.A}."
            )
        if self.Iz <= 0:
            raise ValueError(
                f"ElasticSection: Iz must be > 0, got {self.Iz}."
            )

        # 3-D variant is selected when any 3-D-only parameter is
        # supplied (Iy, J, alphaZ). All three of Iy, G, J are then
        # required.
        is_3d = any(
            v is not None for v in (self.Iy, self.J, self.alphaZ)
        )
        if is_3d:
            missing = [
                name for name, val in (
                    ("Iy", self.Iy), ("G", self.G), ("J", self.J),
                ) if val is None
            ]
            if missing:
                raise ValueError(
                    "ElasticSection: 3-D variant requires Iy, G, J. "
                    f"Missing: {', '.join(missing)}."
                )
        else:
            # 2-D: alphaY may only be supplied if G is also supplied
            # (the OpenSees 2-D form pairs them).
            if self.alphaY is not None and self.G is None:
                raise ValueError(
                    "ElasticSection (2-D): alphaY requires G to also "
                    "be supplied (the OpenSees 2-D form pairs them)."
                )

        if self.Iy is not None and self.Iy <= 0:
            raise ValueError(
                f"ElasticSection: Iy must be > 0, got {self.Iy}."
            )
        if self.G is not None and self.G <= 0:
            raise ValueError(
                f"ElasticSection: G must be > 0, got {self.G}."
            )
        if self.J is not None and self.J <= 0:
            raise ValueError(
                f"ElasticSection: J must be > 0, got {self.J}."
            )

    def _is_3d(self) -> bool:
        return any(v is not None for v in (self.Iy, self.J, self.alphaZ))

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        params: list[float | str]
        if self._is_3d():
            # 3-D requires Iy, G, J — guaranteed by __post_init__.
            assert self.Iy is not None
            assert self.G is not None
            assert self.J is not None
            params = [self.E, self.A, self.Iz, self.Iy, self.G, self.J]
            if self.alphaY is not None or self.alphaZ is not None:
                # OpenSees expects the alphaY/alphaZ pair together;
                # default the absent one to 1.0 (the OpenSees default).
                params.append(
                    self.alphaY if self.alphaY is not None else 1.0
                )
                params.append(
                    self.alphaZ if self.alphaZ is not None else 1.0
                )
        else:
            params = [self.E, self.A, self.Iz]
            if self.G is not None:
                # 2-D optional pair: G then alphaY.
                params.append(self.G)
                if self.alphaY is not None:
                    params.append(self.alphaY)
        emitter.section("Elastic", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
