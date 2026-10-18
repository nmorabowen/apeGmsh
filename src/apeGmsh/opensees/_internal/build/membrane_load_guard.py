"""Refuse a ``LadrunoMembrane`` whose ``pressure=`` no pattern activates (#2151).

The fork's ``LadrunoMembrane`` fills its load vector only in
``LadrunoMembrane::addLoad``, from an ``eleLoad`` inside a pattern
(FORK_PIN ``100eb9ae1``, ``LadrunoMembrane.cpp``; fork guide section 4).
``-pressure`` is a reference value that the pattern's ``-surfaceLoad``
(or, without ``rho``/thickness, ``-selfWeight``) scales. A membrane that
sets ``pressure=`` with no ``Plain`` pattern targeting it therefore
carries no load at all: ``analyze`` returns 0 and every support reaction
reads exactly 0.0 (#2143). :func:`refuse_inert_membrane_pressure` turns
that silent zero into an error at ``ops.build()``.

``rho=`` is not refused here: with a thickness it also builds the
element's mass matrix (``LadrunoMembrane::getMass``), so a modal or
``UniformExcitation`` run reads it with no ``Plain`` pattern at all.

A target counts the way the emit resolves it
(``patterns.resolve_ele_load_commands``):

* ``ele=`` an Element handle hits the membrane it is;
* ``ele=`` an int hits the membrane whose planned OpenSees tag it is;
* ``pg=`` hits a membrane declared with that same ``pg=``;
* ``label=`` resolves to mesh cells only, and a membrane is a macro
  element with no mesh cell, so it never hits one.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from ...element.shell import LadrunoMembrane
from ...pattern.pattern import Plain
from .errors import BridgeError

if TYPE_CHECKING:
    from ...apesees import BuiltModel


__all__ = ["InertMembranePressureError", "refuse_inert_membrane_pressure"]


#: The ``Plain`` element-load types (``surface_load``, ``self_weight``,
#: ``beam_uniform``) that reach ``LadrunoMembrane::addLoad``. They are every
#: type ``Plain`` records today; an unknown one fails closed below.
#: ``-beamUniform`` loads the membrane from its own ``w`` values and does not
#: read ``-pressure``; it counts because the element is then loaded.
_ACTIVATING: frozenset[str] = frozenset({"surfaceLoad", "selfWeight", "beamUniform"})


class InertMembranePressureError(BridgeError):
    """A ``LadrunoMembrane`` sets ``pressure=`` that no pattern activates."""


def _plain_patterns(bm: "BuiltModel") -> Iterator[Plain]:
    """Every ``Plain`` of the model, top-level and stage-scoped.

    A stage's ``s.pattern(...)`` registers its ``Plain`` on the bridge like
    a top-level one, so it is in ``bm.primitives`` too; the stage only
    claims it for its own block (ADR 0051 BL-3, ``_claimed_pattern_ids``).
    """
    for prim in bm.primitives:
        if isinstance(prim, Plain):
            yield prim


def _planned_membrane_tags(bm: "BuiltModel") -> dict[int, int]:
    """``id(membrane) -> OpenSees tag`` from the model's element tag plan."""
    from ..tag_plan import emit_mode, plan_tags

    plan = plan_tags(bm, emit_mode(bm, split=False, supports_partitions=False))
    out: dict[int, int] = {}
    for spec, rows in plan.elements.specs:
        if isinstance(spec, LadrunoMembrane):
            for _eid, _conn, tag in rows:
                out[id(spec)] = int(tag)
    return out


def refuse_inert_membrane_pressure(bm: "BuiltModel") -> None:
    """Raise when a pressured ``LadrunoMembrane`` has no activating pattern.

    A membrane with ``pressure`` nonzero must be targeted by a ``Plain``
    pattern's ``surface_load``, ``self_weight`` or ``beam_uniform``, top
    level or inside a stage. A membrane without ``pressure`` (stiffness
    or mass only) is never refused.

    Raises
    ------
    InertMembranePressureError
        Naming the membrane, its inert ``pressure``, and the fix.
    BridgeError
        When a ``Plain`` holds an element-load type this guard does not
        know (fail closed: a new type must say whether it activates).
    """
    pressured = [
        p for p in bm.primitives
        if isinstance(p, LadrunoMembrane)
        and p.pressure is not None and p.pressure != 0.0
    ]
    if not pressured:
        return

    hit: set[int] = set()
    pgs: set[str] = set()
    ints: set[int] = set()
    for pattern in _plain_patterns(bm):
        for rec in pattern.ele_loads:
            if rec.load_type not in _ACTIVATING:
                raise BridgeError(
                    f"LadrunoMembrane pressure guard: Plain element-load type "
                    f"{rec.load_type!r} is unknown; it must be classed as "
                    "activating a membrane's pressure= or not (#2151)."
                )
            if rec.target_kind == "pg":
                pgs.add(str(rec.target))
            elif rec.target_kind == "ele":
                assert isinstance(rec.target, tuple)
                for item in rec.target:
                    if isinstance(item, LadrunoMembrane):
                        hit.add(id(item))
                    elif isinstance(item, int):
                        ints.add(item)
            elif rec.target_kind != "label":
                raise BridgeError(
                    f"LadrunoMembrane pressure guard: Plain element-load "
                    f"target kind {rec.target_kind!r} is unknown (#2151)."
                )

    hit.update(id(p) for p in pressured if p.pg is not None and p.pg in pgs)
    unhit = [p for p in pressured if id(p) not in hit]
    if unhit and ints:
        tags = _planned_membrane_tags(bm)
        unhit = [p for p in unhit if tags[id(p)] not in ints]
    if not unhit:
        return

    membrane = unhit[0]
    if membrane.pg is not None:
        where, target = f"pg={membrane.pg!r}", f"pg={membrane.pg!r}"
    else:
        where, target = f"nodes={membrane.nodes!r}", "ele=<membrane>"
    more = (
        f" {len(unhit) - 1} more pressured membrane(s) have no activating "
        "pattern either." if len(unhit) > 1 else ""
    )
    raise InertMembranePressureError(
        f"LadrunoMembrane({where}) sets pressure={membrane.pressure!r}, but no "
        "Plain pattern activates it, so it applies no load: the fork reads "
        "-pressure only inside LadrunoMembrane::addLoad, and every support "
        "reaction would read 0.0. Target the membrane from a pattern: "
        "`with ops.pattern.Plain(series=...) as p: "
        f"p.surface_load({target})`. A stage's s.pattern(...) counts too; "
        "label= does not, since it resolves to mesh cells and a membrane "
        f"has none.{more}"
    )
