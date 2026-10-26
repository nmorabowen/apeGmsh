"""Refuse a ``LadrunoMembrane`` whose ``pressure=`` no pattern activates (#2151).

The fork's ``LadrunoMembrane`` fills its load vector only in
``LadrunoMembrane::addLoad``, from an ``eleLoad`` inside a pattern
(FORK_PIN ``100eb9ae1``, ``LadrunoMembrane.cpp``; fork guide section 4),
and ``-pressure`` is read on two of its paths only:

* ``-surfaceLoad`` always scales ``-pressure``;
* ``-selfWeight`` scales ``rho`` x thickness when the membrane has both
  (``rho`` from ``-rho``, else from the ``-mat`` material), and falls back
  to ``-pressure`` only when it has not;
* ``-beamUniform`` loads the membrane from its own ``w`` values and never
  reads ``-pressure``.  ``p.surface_load(..., pressure=q)`` (the
  ``membranePressure`` record) emits exactly that, ``-beamUniform 0 q 0``:
  the pattern carries its own magnitude, so it never activates the
  element's ``pressure=`` either.

A membrane that sets ``pressure=`` with no pattern reaching it on one of
those paths applies none of it: with no other load, ``analyze`` returns 0
and every support reaction reads exactly 0.0 (#2143).
:func:`refuse_inert_membrane_pressure` turns that silent zero into an
error at ``ops.build()``.

``rho=`` is not refused here: with a thickness it also builds the
element's mass matrix (``LadrunoMembrane::getMass``), so a modal or
``UniformExcitation`` run reads it with no ``Plain`` pattern at all.

A target counts where the emit would resolve it to the membrane
(``patterns.resolve_ele_load_commands``):

* ``ele=`` an Element handle hits the membrane it is;
* ``ele=`` an int is not matched: a membrane's OpenSees tag is planned
  at emit, and the tag law keeps planning off the build side, so an int
  that is the membrane's tag is refused like a miss (fail closed; the
  message says to pass the handle);
* ``pg=`` hits a membrane declared with that same ``pg=``;
* ``label=`` resolves to mesh cells only, and a membrane is a macro
  element with no mesh cell, so it never hits one.

The module also holds :func:`guard_membrane_interior` (#2468), the other
build-time ``LadrunoMembrane`` guard: the interior mesh nodes of a ``pg=``
panel are not emitted, and no record may target one.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Sequence

from ...element.shell import LadrunoMembrane
from ...pattern.pattern import Plain
from ...section.plate import ElasticMembranePlateSection
from ..types import Recorder
from .errors import BridgeError
from .ndf import (
    membrane_interior_nodes,
    validate_membrane_interior_untargeted,
    warn_membrane_hole_nodes_free,
)
from .recorders import recorder_node_targets

if TYPE_CHECKING:
    from ...apesees import BuiltModel
    from ..types import Element


__all__ = [
    "InertMembranePressureError",
    "guard_membrane_interior",
    "refuse_inert_membrane_pressure",
]


#: Every element-load type ``Plain`` records today; an unknown one fails
#: closed below, since it must be classed before it can count.
_KNOWN_TYPES: frozenset[str] = frozenset(
    {"surfaceLoad", "selfWeight", "beamUniform", "membranePressure"})


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


def _may_have_rho_t(m: LadrunoMembrane) -> bool:
    """True unless ``m`` surely has no ``rho`` x thickness in the fork.

    It mirrors what ``m`` emits: ``-rho`` is ``rho=``, else the section's
    ``rho``; the thickness is ``thickness=`` (from ``stiff=``, ``E=``/``nu=``
    or ``mat=``), else the section's ``h``. With ``-rho`` zero the fork
    reads the ``mat=`` material's density. A material whose density this
    bridge cannot read counts as having one (fail closed: ``self_weight``
    then does not count as reading ``pressure=``).
    """
    section = m.section if isinstance(m.section, ElasticMembranePlateSection) else None
    thick = m.thickness if m.thickness is not None else (
        float(section.h) if section is not None else 0.0
    )
    if not thick > 0.0:
        return False
    rho = m.rho if m.rho is not None else (
        float(section.rho) if section is not None else 0.0
    )
    if rho > 0.0:
        return True
    if m.mat is None:
        return False
    mat_rho = getattr(m.mat, "rho", None)
    return mat_rho is None or float(mat_rho) > 0.0


def _reads_pressure(
    load_type: str, vector: tuple[float, ...], m: LadrunoMembrane,
) -> bool:
    """True iff a ``load_type`` load with ``vector`` reaches ``m``'s
    ``-pressure`` (the paths in the module docstring); ``membranePressure``
    and ``beamUniform`` never do."""
    if load_type == "surfaceLoad":
        return True
    if load_type == "selfWeight":
        return any(c != 0.0 for c in vector) and not _may_have_rho_t(m)
    return False


def refuse_inert_membrane_pressure(bm: "BuiltModel") -> None:
    """Raise when a pressured ``LadrunoMembrane`` has no activating pattern.

    A membrane with ``pressure`` nonzero must be targeted by a ``Plain``
    pattern, top level or inside a stage, on a path that reads it: a
    ``surface_load``, or a nonzero ``self_weight`` when the membrane has no
    ``rho`` x thickness. A membrane without ``pressure`` (stiffness or mass
    only) is never refused.

    Raises
    ------
    InertMembranePressureError
        Naming the membrane, its inert ``pressure``, and the fix.
    BridgeError
        When a ``Plain`` holds an element-load type or target kind this
        guard does not know (fail closed: a new one must be classed).
    """
    pressured = [
        p for p in bm.primitives
        if isinstance(p, LadrunoMembrane)
        and p.pressure is not None and p.pressure != 0.0
    ]
    if not pressured:
        return

    by_pg: dict[str, list[LadrunoMembrane]] = {}
    for p in pressured:
        if p.pg is not None:
            by_pg.setdefault(p.pg, []).append(p)
    pressured_ids = {id(p) for p in pressured}

    activated: set[int] = set()
    int_targets = False
    for pattern in _plain_patterns(bm):
        for rec in pattern.ele_loads:
            if rec.load_type not in _KNOWN_TYPES:
                raise BridgeError(
                    f"LadrunoMembrane pressure guard: Plain element-load type "
                    f"{rec.load_type!r} is unknown; it must be classed as "
                    "reading a membrane's pressure= or not (#2151)."
                )
            targets: list[LadrunoMembrane] = []
            if rec.target_kind == "pg":
                targets = by_pg.get(str(rec.target), [])
            elif rec.target_kind == "ele":
                assert isinstance(rec.target, tuple)
                for item in rec.target:
                    if isinstance(item, LadrunoMembrane):
                        if id(item) in pressured_ids:
                            targets.append(item)
                    elif isinstance(item, int):
                        int_targets = True
            elif rec.target_kind != "label":
                raise BridgeError(
                    f"LadrunoMembrane pressure guard: Plain element-load "
                    f"target kind {rec.target_kind!r} is unknown (#2151)."
                )
            vector = (*rec.wx, *rec.wy, *(rec.wz or (0.0, 0.0)))
            activated.update(
                id(m) for m in targets
                if _reads_pressure(rec.load_type, vector, m)
            )

    unhit = [p for p in pressured if id(p) not in activated]
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
    if int_targets:
        more += (
            " An integer ele= tag is not matched to a membrane here: a "
            "membrane's tag is planned at emit, so pass its handle instead."
        )
    raise InertMembranePressureError(
        f"LadrunoMembrane({where}) sets pressure={membrane.pressure!r}, but no "
        "Plain pattern activates it, so it applies no load: the fork reads "
        "-pressure only inside LadrunoMembrane::addLoad, on a surface_load "
        "without pressure=, or "
        "on a nonzero self_weight when the membrane has no rho x thickness; "
        "beam_uniform and surface_load(pressure=...) never read it. Target "
        "the membrane from a pattern: "
        "`with ops.pattern.Plain(series=...) as p: "
        f"p.surface_load({target})`, or drop pressure= if the model is modal "
        "or stiffness only, or when every pattern gives its own magnitude "
        "with surface_load(..., pressure=q). A stage's s.pattern(...) counts too; label= does "
        f"not, since it resolves to mesh cells and a membrane has none.{more}"
    )


def guard_membrane_interior(
    bm: "BuiltModel", elements: "Sequence[Element]", *, archival: bool,
) -> "frozenset[int]":
    """The nodes :func:`~.ndf.membrane_interior_nodes` keeps out of
    ``bm``'s deck, after checking that nothing targets one (#2468).

    Warns on a free hole node (:func:`~.ndf.warn_membrane_hole_nodes_free`,
    skipped for an ``archival`` emit, which never solves), then refuses
    every record that targets an interior node
    (:func:`~.ndf.validate_membrane_interior_untargeted`): the global and
    staged ``fix`` / support / ``mass`` records, every ``Plain``'s loads,
    sps and ``from_model`` cases, the broker constraints and interfaces,
    the staged constraints, ``mass_from_model`` and the recorders' node
    lists. Each node-emit site calls it once and skips what it returns.
    """
    fem = bm.fem
    skip = membrane_interior_nodes(fem, elements)
    if not archival:
        warn_membrane_hole_nodes_free(fem, elements)
    if not skip:
        return skip
    stages = bm.stage_records
    # A stage-claimed Plain is in bm.primitives too: dedup by id.
    plains_by_id: dict[int, Plain] = {id(p): p for p in _plain_patterns(bm)}
    for st in stages:
        for p in st.pattern_specs:
            plains_by_id[id(p)] = p
    plains = list(plains_by_id.values())
    recorders = (
        *(p for p in bm.primitives if isinstance(p, Recorder)),
        *(r for st in stages for r in st.recorder_specs),
    )
    validate_membrane_interior_untargeted(
        fem, skip,
        fix_records=(
            *bm.fix_records,
            *(r for st in stages for r in st.fix_records),
            *(r for st in stages for r in st.support_records),
        ),
        mass_records=(
            *bm.mass_records,
            *(r for st in stages for r in st.mass_records),
        ),
        load_records=tuple(ld for p in plains for ld in p.loads),
        sp_records=tuple(sp for p in plains for sp in p.sps),
        constraint_records=(
            *fem.nodes.constraints,
            *fem.elements.constraints,
            *fem.elements.interfaces,
            *(r for st in stages for r in st.stage_constraint_records),
        ),
        from_model_cases=tuple(c for p in plains for c in p.from_model_cases),
        mass_from_model=bm.mass_from_model,
        node_targets=tuple(
            t for r in recorders for t in recorder_node_targets(r, fem)),
    )
    return skip
