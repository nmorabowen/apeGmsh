"""
ZeroLength elements — typed primitives for ``element zeroLength`` and
``element zeroLengthSection``.

OpenSees commands::

    element zeroLength tag iNode jNode \
        -mat $matTag1 $matTag2 ... -dir $dir1 $dir2 ... \
        [-orient $x1 $x2 $x3 $yp1 $yp2 $yp3] [-doRayleigh $rFlag]

    element zeroLengthSection tag iNode jNode secTag \
        [-orient $x1 $x2 $x3 $yp1 $yp2 $yp3] [-doRayleigh $rFlag]

Both elements take 2 nodes that are typically coincident (zero
length); they couple the two nodes through a list of (material, dof)
pairs (``zeroLength``) or through a single section (``zeroLengthSection``).

Element fan-out
===============

Same contract as :mod:`.truss`: the bridge fans the spec across its
physical group at build time and sets the per-element node tags on
the emitter via
:func:`apeGmsh.opensees._internal.tag_resolution.set_element_nodes`.
The typed class reads them via :func:`current_element_nodes` and
resolves composed material / section tags via
:func:`~apeGmsh.opensees._internal.tag_resolution.resolve_tag`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.tag_resolution import (
    current_element_nodes,
    damp_args,
    resolve_tag,
)
from .._internal.types import (
    Damping,
    Element,
    Primitive,
    Section,
    UniaxialMaterial,
)

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "ZeroLength",
    "ZeroLengthMatDir",
    "ZeroLengthSection",
]


# ---------------------------------------------------------------------------
# Value object — one (material, dof) pair on a ZeroLength element
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ZeroLengthMatDir:
    """One ``(material, dof)`` pair on a :class:`ZeroLength` element.

    A value object — not a :class:`Primitive`, no tag, not registered
    standalone. Held in a tuple by the parent :class:`ZeroLength`.

    Parameters
    ----------
    material
        The :class:`UniaxialMaterial` providing the constitutive
        response along ``dof``.
    dof
        OpenSees DOF index (1-based: 1=Ux, 2=Uy, 3=Uz, 4=Rx, 5=Ry,
        6=Rz). Must be >= 1.
    """

    material: UniaxialMaterial
    dof: int

    def __post_init__(self) -> None:
        if self.dof < 1:
            raise ValueError(
                f"ZeroLengthMatDir: dof must be >= 1, got {self.dof!r}"
            )


# ---------------------------------------------------------------------------
# ZeroLength
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ZeroLength(Element):
    """``element zeroLength`` — coupled (material, dof) springs.

    OpenSees command::

        element zeroLength tag iNode jNode \
            -mat $matTag1 $matTag2 ... -dir $dir1 $dir2 ... \
            [-orient $x1 $x2 $x3 $yp1 $yp2 $yp3] \
            [-doRayleigh $rFlag]

    Parameters
    ----------
    pg
        Physical-group label whose 2-node "line" entries (typically
        coincident-node pairs) receive this spec.
    mat_dirs
        Tuple of :class:`ZeroLengthMatDir` value objects, each binding
        a uniaxial material to one local DOF. At least one pair is
        required; multiple pairs may share the same material.
    orient
        Optional 6-tuple ``(x1, x2, x3, yp1, yp2, yp3)`` giving the
        local x and y'-vector that orient the element's local frame.
        ``None`` means OpenSees uses its default global orientation.
    do_rayleigh
        Include the element in Rayleigh damping (``-doRayleigh``).
    """

    pg: str
    mat_dirs: tuple[ZeroLengthMatDir, ...]
    orient: tuple[float, float, float, float, float, float] | None = None
    do_rayleigh: bool = False
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if not self.mat_dirs:
            raise ValueError(
                "ZeroLength: at least one (material, dof) pair required."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        # Multiple springs can share a material — dedup by id, keep
        # iteration order.
        seen: dict[int, UniaxialMaterial] = {}
        for md in self.mat_dirs:
            seen.setdefault(id(md.material), md.material)
        deps: tuple[Primitive, ...] = tuple(seen.values())
        if self.damp is not None:
            deps += (self.damp,)
        return deps

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 2:
            raise ValueError(
                f"ZeroLength: expected 2 node tags, got {len(nodes)}"
            )
        mat_tags = tuple(
            resolve_tag(emitter, md.material) for md in self.mat_dirs
        )
        dirs = tuple(md.dof for md in self.mat_dirs)
        args: list[int | float | str] = [
            *nodes,
            "-mat", *mat_tags,
            "-dir", *dirs,
        ]
        if self.orient is not None:
            args += ["-orient", *self.orient]
        if self.do_rayleigh:
            args += ["-doRayleigh", 1]
        args.extend(damp_args(emitter, self.damp))
        emitter.element("zeroLength", tag, *args)


# ---------------------------------------------------------------------------
# ZeroLengthSection
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ZeroLengthSection(Element):
    """``element zeroLengthSection`` — section-coupled zero-length element.

    OpenSees command::

        element zeroLengthSection tag iNode jNode secTag \
            [-orient $x1 $x2 $x3 $yp1 $yp2 $yp3] \
            [-doRayleigh $rFlag]

    Parameters
    ----------
    pg
        Physical-group label whose 2-node entries receive this spec.
    section
        The :class:`Section` whose force-deformation response governs
        the element's coupling. Typically a fiber section.
    orient
        Optional 6-tuple ``(x1, x2, x3, yp1, yp2, yp3)`` giving the
        local x and y'-vector. ``None`` means OpenSees uses its
        default global orientation.
    do_rayleigh
        Include the element in Rayleigh damping (``-doRayleigh``).
    """

    pg: str
    section: Section
    orient: tuple[float, float, float, float, float, float] | None = None
    do_rayleigh: bool = False

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 2:
            raise ValueError(
                f"ZeroLengthSection: expected 2 node tags, got "
                f"{len(nodes)}"
            )
        sec_tag = resolve_tag(emitter, self.section)
        args: list[int | float | str] = [*nodes, sec_tag]
        if self.orient is not None:
            args += ["-orient", *self.orient]
        if self.do_rayleigh:
            args += ["-doRayleigh", 1]
        emitter.element("zeroLengthSection", tag, *args)
