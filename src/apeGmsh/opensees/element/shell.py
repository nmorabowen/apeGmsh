"""
Shell element primitives — typed wrappers over the OpenSees ``element
Shell*`` family.

Five element types live here:

* :class:`ShellMITC4` — 4-node MITC shell (membrane + bending).
* :class:`ShellMITC3` — 3-node MITC shell.
* :class:`ShellDKGQ` — 4-node Discrete Kirchhoff (Quadrilateral) shell.
* :class:`ASDShellQ4` — 4-node ASD shell (Petracca/ASDEA Software).
* :class:`ASDShellT3` — 3-node ASD shell.

The OpenSees commands per the manual:

* ``element ShellMITC4 tag i j k l secTag``
* ``element ShellMITC3 tag i j k secTag``
* ``element ShellDKGQ  tag i j k l secTag``
* ``element ASDShellQ4 tag i j k l secTag [-corotational]``
  ``[-drillingNT alpha] [-localCS x1 x2 x3 y1 y2 y3]``
* ``element ASDShellT3 tag i j k secTag [-corotational]``
  ``[-drillingDOF dof_id] [-localCS x1 x2 x3 y1 y2 y3]``

Element fan-out contract
========================

Each element class composes one :class:`Section` (referenced via its
allocated tag at emit time) and one or more nodes. Per the contract in
:mod:`apeGmsh.opensees._internal.tag_resolution`, the bridge fans out
the element's physical group at build time and sets:

* the section's allocated tag via the resolver attached to the emitter
  (looked up via :func:`resolve_tag`).
* the node tags for **the current element** of the fan-out via
  :func:`set_element_nodes` on the emitter (read by ``_emit`` via
  :func:`current_element_nodes`).

Tests install both contexts manually with :func:`set_tag_resolver` +
:func:`set_element_nodes`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.tag_resolution import (
    current_element_nodes,
    damp_args,
    resolve_tag,
)
from .._internal.types import Damping, Element, Primitive, Section

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "ShellMITC3",
    "ShellMITC4",
    "ShellDKGQ",
    "ASDShellQ4",
    "ASDShellT3",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_local_cs(type_name: str, local_cs: tuple[float, ...]) -> None:
    """``-localCS`` requires exactly six floats (x1, x2, x3, y1, y2, y3)."""
    if len(local_cs) != 6:
        raise ValueError(
            f"{type_name}: local_cs= must be a 6-tuple "
            f"(x1, x2, x3, y1, y2, y3), got {len(local_cs)} entries."
        )


# ---------------------------------------------------------------------------
# ShellMITC4 — 4-node MITC shell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ShellMITC4(Element):
    """``element ShellMITC4`` — 4-node MITC shell.

    Parameters
    ----------
    pg
        Physical group whose surface (quadrilateral) cells receive
        this element. The bridge fans out at build time; per element
        the four node tags are read from the emitter context by
        :func:`current_element_nodes`.
    section
        The plate / shell :class:`Section` (typically
        :class:`~apeGmsh.opensees.section.plate.ElasticMembranePlateSection`,
        :class:`~apeGmsh.opensees.section.plate.LayeredShell`, or
        :class:`~apeGmsh.opensees.section.plate.LayeredShellFiberSection`).
    """

    pg: str
    section: Section
    damp: Damping | None = None

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"ShellMITC4: expected 4 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        emitter.element(
            "ShellMITC4", tag, *nodes, sec_tag,
            *damp_args(emitter, self.damp),
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.section, self.damp)
        return (self.section,)


# ---------------------------------------------------------------------------
# ShellMITC3 — 3-node MITC shell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ShellMITC3(Element):
    """``element ShellMITC3`` — 3-node MITC shell.

    Parameters
    ----------
    pg
        Physical group whose surface (triangular) cells receive this
        element.
    section
        The plate / shell :class:`Section`.
    """

    pg: str
    section: Section

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 3:
            raise ValueError(
                f"ShellMITC3: expected 3 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        emitter.element("ShellMITC3", tag, *nodes, sec_tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)


# ---------------------------------------------------------------------------
# ShellDKGQ — 4-node Discrete Kirchhoff (Quadrilateral) shell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ShellDKGQ(Element):
    """``element ShellDKGQ`` — 4-node Discrete Kirchhoff Quadrilateral.

    Parameters
    ----------
    pg
        Physical group whose surface (quadrilateral) cells receive
        this element.
    section
        The plate / shell :class:`Section`.
    """

    pg: str
    section: Section
    damp: Damping | None = None

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"ShellDKGQ: expected 4 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        emitter.element(
            "ShellDKGQ", tag, *nodes, sec_tag,
            *damp_args(emitter, self.damp),
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.section, self.damp)
        return (self.section,)


# ---------------------------------------------------------------------------
# ASDShellQ4 — 4-node ASD shell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ASDShellQ4(Element):
    """``element ASDShellQ4`` — 4-node ASD shell.

    Phase 2γ scope ships the canonical positional arguments plus the
    three most-used optional flags: ``-corotational``, ``-drillingNT
    alpha``, and ``-localCS x1 x2 x3 y1 y2 y3``. Other ASDShell* flags
    (e.g. ``-noEAS``, ``-drillingStab K``) are deferred — see the
    Phase 2γ report.

    Parameters
    ----------
    pg
        Physical group whose surface (quadrilateral) cells receive
        this element.
    section
        The plate / shell :class:`Section`.
    corotational
        Append the ``-corotational`` flag.
    drilling_nt_alpha
        If supplied, append ``-drillingNT <alpha>``.
    local_cs
        If supplied, append ``-localCS <x1> <x2> <x3> <y1> <y2> <y3>``.
        Must be a 6-tuple.
    """

    pg: str
    section: Section
    corotational: bool = False
    drilling_nt_alpha: float | None = None
    local_cs: tuple[float, ...] | None = None
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if self.local_cs is not None:
            _check_local_cs("ASDShellQ4", self.local_cs)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 4:
            raise ValueError(
                f"ASDShellQ4: expected 4 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        args: list[int | float | str] = [*nodes, sec_tag]
        if self.corotational:
            args.append("-corotational")
        if self.drilling_nt_alpha is not None:
            args.append("-drillingNT")
            args.append(self.drilling_nt_alpha)
        if self.local_cs is not None:
            args.append("-localCS")
            args.extend(self.local_cs)
        args.extend(damp_args(emitter, self.damp))
        emitter.element("ASDShellQ4", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        if self.damp is not None:
            return (self.section, self.damp)
        return (self.section,)


# ---------------------------------------------------------------------------
# ASDShellT3 — 3-node ASD shell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ASDShellT3(Element):
    """``element ASDShellT3`` — 3-node ASD shell.

    Phase 2γ scope ships the canonical positional arguments plus the
    three most-used optional flags: ``-corotational``, ``-drillingDOF
    dof_id``, and ``-localCS x1 x2 x3 y1 y2 y3``. Other ASDShell*
    flags are deferred — see the Phase 2γ report.

    Parameters
    ----------
    pg
        Physical group whose surface (triangular) cells receive this
        element.
    section
        The plate / shell :class:`Section`.
    corotational
        Append the ``-corotational`` flag.
    drilling_dof
        If supplied, append ``-drillingDOF <dof_id>``.
    local_cs
        If supplied, append ``-localCS <x1> <x2> <x3> <y1> <y2> <y3>``.
        Must be a 6-tuple.
    """

    pg: str
    section: Section
    corotational: bool = False
    drilling_dof: int | None = None
    local_cs: tuple[float, ...] | None = None
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if self.local_cs is not None:
            _check_local_cs("ASDShellT3", self.local_cs)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 3:
            raise ValueError(
                f"ASDShellT3: expected 3 node tags, got {len(nodes)}."
            )
        sec_tag = resolve_tag(emitter, self.section)
        args: list[int | float | str] = [*nodes, sec_tag]
        if self.corotational:
            args.append("-corotational")
        if self.drilling_dof is not None:
            args.append("-drillingDOF")
            args.append(self.drilling_dof)
        if self.local_cs is not None:
            args.append("-localCS")
            args.extend(self.local_cs)
        args.extend(damp_args(emitter, self.damp))
        emitter.element("ASDShellT3", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)
