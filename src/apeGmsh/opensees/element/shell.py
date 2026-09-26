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
    """``-local`` takes the local x axis: exactly three floats, not all zero.

    The ASD shells' parsers (``ASDShellQ4.cpp`` / ``ASDShellT3.cpp``,
    ``strcmp(type, "-local")``) read three components and build the rest of
    the frame themselves (x projected onto the shell plane, z the normal,
    y = z cross x). They have no ``-localCS``: that token, and its six
    values, fell through the option loop unread, so the element silently
    kept its default frame. A 6-tuple is refused rather than truncated: a
    script that passed one was running on the default frame, and quietly
    switching it to the requested one would change its answers unannounced.
    """
    if len(local_cs) == 6:
        raise ValueError(
            f"{type_name}: local_cs= takes the local x axis only, "
            f"(x1, x2, x3), emitted as '-local x1 x2 x3'. The old 6-tuple "
            f"form emitted '-localCS', which the element does not parse, so "
            f"it always ran on the default frame. Pass local_cs="
            f"{tuple(local_cs[:3])!r}; the element derives y from the "
            f"normal."
        )
    if len(local_cs) != 3:
        raise ValueError(
            f"{type_name}: local_cs= must be a 3-tuple (x1, x2, x3), the "
            f"local x axis; got {len(local_cs)} entries."
        )
    if not any(float(c) != 0.0 for c in local_cs):
        raise ValueError(
            f"{type_name}: local_cs= must not be the zero vector (the element "
            f"reads a zero -local vector as 'use the default frame')."
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

    Optional flags, as ``OPS_ASDShellQ4`` parses them: ``-corotational``,
    ``-drillingStab $v`` / ``-drillingNL`` (mutually exclusive) and
    ``-local $x1 $x2 $x3``. ``-noeas`` is not exposed.

    Parameters
    ----------
    pg
        Physical group whose surface (quadrilateral) cells receive
        this element.
    section
        The plate / shell :class:`Section`.
    corotational
        Append the ``-corotational`` flag.
    drilling_stab
        If supplied, append ``-drillingStab <v>``: the drilling
        stabilization factor, in ``[0, 1]`` (the parser clamps to that
        range; the default without the flag is 0.01).
    drilling_nl
        Append ``-drillingNL`` (nonlinear drilling DOF treatment). The
        parser refuses it together with ``-drillingStab``.
    drilling_nt_alpha
        **Refused.** It emitted ``-drillingNT``, which ASDShellQ4 does not
        parse (the token fell through unread). Use ``drilling_stab`` or
        ``drilling_nl``.
    local_cs
        The local x axis ``(x1, x2, x3)``, emitted as ``-local x1 x2 x3``.
        The element projects it onto the shell plane and takes y = z cross
        x. This is also the section frame the layers of a layered section
        see, so a ``PlateRebar`` angle is measured from it.
    """

    pg: str
    section: Section
    corotational: bool = False
    drilling_stab: float | None = None
    drilling_nl: bool = False
    drilling_nt_alpha: float | None = None
    local_cs: tuple[float, ...] | None = None
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if self.local_cs is not None:
            _check_local_cs("ASDShellQ4", self.local_cs)
        if self.drilling_nt_alpha is not None:
            raise ValueError(
                "ASDShellQ4: drilling_nt_alpha emitted '-drillingNT', which "
                "the element does not parse; it was silently ignored. Use "
                "drilling_stab=<v in [0, 1]> ('-drillingStab') or "
                "drilling_nl=True ('-drillingNL')."
            )
        if self.drilling_stab is not None:
            if not (0.0 <= float(self.drilling_stab) <= 1.0):
                raise ValueError(
                    f"ASDShellQ4: drilling_stab must be in [0, 1], got "
                    f"{self.drilling_stab!r} (the parser would clamp it)."
                )
            if self.drilling_nl:
                raise ValueError(
                    "ASDShellQ4: drilling_stab and drilling_nl are mutually "
                    "exclusive (the parser refuses -drillingStab with "
                    "-drillingNL)."
                )

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
        if self.drilling_stab is not None:
            args.append("-drillingStab")
            args.append(float(self.drilling_stab))
        if self.drilling_nl:
            args.append("-drillingNL")
        if self.local_cs is not None:
            args.append("-local")
            args.extend(float(c) for c in self.local_cs)
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
    Optional flags, as ``OPS_ASDShellT3`` parses them: ``-corotational``,
    ``-drillingNL`` and ``-local $x1 $x2 $x3``. ``-reducedIntegration`` is
    not exposed.

    Parameters
    ----------
    pg
        Physical group whose surface (triangular) cells receive this
        element.
    section
        The plate / shell :class:`Section`.
    corotational
        Append the ``-corotational`` flag.
    drilling_nl
        Append ``-drillingNL`` (nonlinear drilling DOF treatment).
    drilling_dof
        **Refused.** It emitted ``-drillingDOF <dof_id>``, which ASDShellT3
        does not parse (the token and its value fell through unread). The
        drilling DOF is always the element's sixth; use ``drilling_nl`` for
        the only drilling option the parser has.
    local_cs
        The local x axis ``(x1, x2, x3)``, emitted as ``-local x1 x2 x3``
        (the parser has no ``-localCS``; see ``ASDShellQ4``).
    """

    pg: str
    section: Section
    corotational: bool = False
    drilling_nl: bool = False
    drilling_dof: int | None = None
    local_cs: tuple[float, ...] | None = None
    damp: Damping | None = None

    def __post_init__(self) -> None:
        if self.local_cs is not None:
            _check_local_cs("ASDShellT3", self.local_cs)
        if self.drilling_dof is not None:
            raise ValueError(
                "ASDShellT3: drilling_dof emitted '-drillingDOF', which the "
                "element does not parse; it was silently ignored. The only "
                "drilling option ASDShellT3 has is drilling_nl=True "
                "('-drillingNL')."
            )

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
        if self.drilling_nl:
            args.append("-drillingNL")
        if self.local_cs is not None:
            args.append("-local")
            args.extend(float(c) for c in self.local_cs)
        args.extend(damp_args(emitter, self.damp))
        emitter.element("ASDShellT3", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.section,)
