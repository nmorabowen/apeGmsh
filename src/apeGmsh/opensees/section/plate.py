"""
Plate / shell sections — typed primitives for OpenSees plate-bending
section commands.

Three section types live here:

* :class:`ElasticMembranePlateSection` — single-layer linear-elastic
  plate (membrane + bending) used by ``ShellMITC4``, ``ShellDKGQ``,
  ``ASDShellQ4``.
* :class:`LayeredShell` — stacked nDMaterial layers (a thin-shell
  composite section) for ``ShellMITC4`` and friends.
* :class:`LayeredShellFiberSection` — same shape as ``LayeredShell``
  but routed through OpenSees's ``LayeredShellFiberSection`` C++
  class (the in-tree fiber-based stack).

The ``LayeredShell*`` sections compose a tuple of nDMaterials and
declare them via :meth:`dependencies`; their ``_emit`` references
resolved nDMaterial tags via the same closure-based resolver Fiber
uses (see :mod:`section.fiber`).

The OpenSees commands per the manual:

* ``section ElasticMembranePlateSection $tag $E $nu $h <$rho>``
* ``section LayeredShell $tag $nLayers $matTag1 $h1 ... $matTagN $hN``
* ``section LayeredShellFiberSection $tag $nLayers $matTag1 $h1 ...
  $matTagN $hN``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import NDMaterial, Primitive, Section
from ._tag_resolver import resolve_mat_tag

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "ElasticMembranePlateSection",
    "LayeredShell",
    "LayeredShellFiberSection",
    "ShellLayer",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class ElasticMembranePlateSection(Section):
    """``section ElasticMembranePlateSection`` — single-layer plate.

    Linear-elastic plate-bending section with isotropic membrane
    stiffness. Used by ``ShellMITC4``, ``ShellDKGQ``, ``ASDShellQ4``.

    Parameters
    ----------
    E
        Young's modulus.
    nu
        Poisson's ratio. Must satisfy ``0 <= nu < 0.5``.
    h
        Section thickness.
    rho
        Mass density per unit volume. Defaults to 0.0 (no mass).
    """

    E:   float
    nu:  float
    h:   float
    rho: float = 0.0

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(
                f"ElasticMembranePlateSection: E must be > 0, "
                f"got {self.E}."
            )
        if not (0.0 <= self.nu < 0.5):
            raise ValueError(
                f"ElasticMembranePlateSection: nu must be in [0, 0.5), "
                f"got {self.nu}."
            )
        if self.h <= 0:
            raise ValueError(
                f"ElasticMembranePlateSection: h must be > 0, "
                f"got {self.h}."
            )
        if self.rho < 0:
            raise ValueError(
                f"ElasticMembranePlateSection: rho must be >= 0, "
                f"got {self.rho}."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        emitter.section(
            "ElasticMembranePlateSection",
            tag,
            self.E, self.nu, self.h, self.rho,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LayeredShell / LayeredShellFiberSection — composite layered sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ShellLayer:
    """One layer of a ``LayeredShell`` / ``LayeredShellFiberSection``.

    A value object — not a :class:`Primitive`, no tag, not registered
    standalone. Held in a tuple by the parent section.

    Parameters
    ----------
    material
        The nDMaterial constitutive law for this layer.
    thickness
        Layer thickness.
    """

    material: NDMaterial
    thickness: float

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError(
                f"ShellLayer: thickness must be > 0, got {self.thickness}."
            )


def _validate_layers(
    cls_name: str, layers: tuple[ShellLayer, ...]
) -> None:
    if not layers:
        raise ValueError(
            f"{cls_name}: at least one ShellLayer is required."
        )


def _layer_dependencies(
    layers: tuple[ShellLayer, ...]
) -> tuple[Primitive, ...]:
    """Deduplicate materials referenced by layers, preserving order."""
    seen: dict[int, NDMaterial] = {}
    for layer in layers:
        seen.setdefault(id(layer.material), layer.material)
    return tuple(seen.values())


@dataclass(frozen=True, kw_only=True, slots=True)
class LayeredShell(Section):
    """``section LayeredShell`` — stacked nDMaterial layers.

    Parameters
    ----------
    layers
        Tuple of :class:`ShellLayer` describing each through-thickness
        layer (bottom to top). At least one layer is required.

    Notes
    -----
    Material-tag resolution at emit time is delegated to a
    closure-captured resolver attached to the emitter; see
    :mod:`section._tag_resolver` for the contract and the open
    coordinator question flagged for the Phase 4 emitters.
    """

    layers: tuple[ShellLayer, ...]

    def __post_init__(self) -> None:
        _validate_layers("LayeredShell", self.layers)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        params: list[float | str] = [len(self.layers)]
        for layer in self.layers:
            mat_tag = resolve_mat_tag(emitter, layer.material)
            params.append(mat_tag)
            params.append(layer.thickness)
        emitter.section("LayeredShell", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return _layer_dependencies(self.layers)


@dataclass(frozen=True, kw_only=True, slots=True)
class LayeredShellFiberSection(Section):
    """``section LayeredShellFiberSection`` — stacked nDMaterial layers
    (fiber-based variant).

    Same input shape as :class:`LayeredShell`; emits the OpenSees
    ``LayeredShellFiberSection`` type token instead. The C++ class
    behind it integrates the layers as fibers through the thickness.

    Parameters
    ----------
    layers
        Tuple of :class:`ShellLayer` describing each through-thickness
        layer.
    """

    layers: tuple[ShellLayer, ...]

    def __post_init__(self) -> None:
        _validate_layers("LayeredShellFiberSection", self.layers)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        params: list[float | str] = [len(self.layers)]
        for layer in self.layers:
            mat_tag = resolve_mat_tag(emitter, layer.material)
            params.append(mat_tag)
            params.append(layer.thickness)
        emitter.section("LayeredShellFiberSection", tag, *params)

    def dependencies(self) -> tuple[Primitive, ...]:
        return _layer_dependencies(self.layers)
