"""The class-wide ``updateMaterialStage`` flag: the build check and stage 0.

The stage flag of every class in
:data:`~apeGmsh.opensees.stage._builder.CLASS_WIDE_STAGE_FLAG` is a C++
``static``.  It holds one value per process, so
``updateMaterialStage -material A`` switches every material of A's group.
Two things follow, and both live here.

* :func:`validate_class_wide_stage_flags` runs at emit, beside validator
  V7.  It refuses a stage flip that names some of a group's live
  materials but not all of them.  It sees the final model, so a material
  registered after the stage block counts, and it models
  ``s.remove_element``: a material whose elements are all removed is not
  live, so it need not be named.
* :class:`StageZeroLedger` is how a deck or live emitter writes
  ``updateMaterialStage -material <tag> -stage 0`` for the ManzariDafalias
  group.  Upstream OpenSees starts ManzariDafalias elastoplastic (the
  static starts at 1) and the Ladruno fork starts it elastic (its
  constructors set 0), so one deck gave two answers.  An explicit stage 0
  makes them agree.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

from ..types import Element, NDMaterial, Primitive
from .errors import BridgeError
from .pg_expand import expand_pg_to_elements, expand_spec_to_elements

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData

    from .records import (
        MaterialStageRecord,
        StageRecord,
        UpdateParameterRecord,
    )


def stage_zero_classes() -> frozenset[str]:
    """The nDMaterial classes that share ``ManzariDafalias::mElastFlag``."""
    from ...stage._builder import CLASS_WIDE_STAGE_FLAG

    flag = CLASS_WIDE_STAGE_FLAG["ManzariDafalias"]
    return frozenset(
        cls for cls, static in CLASS_WIDE_STAGE_FLAG.items()
        if static == flag)


def stage_flips(
    mat_stage_records: "Iterable[MaterialStageRecord]",
    parameter_records: "Iterable[UpdateParameterRecord]",
) -> list[tuple[int, int]]:
    """``(material tag, stage value)`` for every flip of one stage.

    Both routes set the static: ``s.update_material_stage``, and an
    ``s.update_parameter`` of a
    :data:`~apeGmsh.opensees.stage._builder.STAGE_FLAG_PARAMETERS` name on
    a material (the material casts that double with ``(int)``).
    """
    from ...stage._builder import STAGE_FLAG_PARAMETERS

    flips: list[tuple[int, int]] = [
        (int(tag), int(rec.stage))
        for rec in mat_stage_records
        for tag in rec.mat_tags
    ]
    flips.extend(
        (int(prec.mat_tag), int(prec.value))
        for prec in parameter_records
        if prec.mat_tag is not None and prec.name in STAGE_FLAG_PARAMETERS
    )
    return flips


class StageZeroLedger:
    """The ``updateMaterialStage ... -stage 0`` lines an emitter owes.

    ``updateMaterialStage`` acts only through elements already in the
    domain: ``MaterialStageParameter::setDomain`` walks them.  A line beside
    the ``nDMaterial`` declaration therefore does nothing.  The emitter
    instead asks :meth:`due` at each solve verb (``analyze``, ``eigen``).
    The answer is every ManzariDafalias-group tag declared so far, when an
    element was emitted since the last answer.  After a user
    ``updateMaterialStage`` names one of those tags, or a user
    ``updateParameter`` sets its stage flag, the answer is always empty,
    because the user owns the static from then on.  So stage 0
    comes after the elements and before any user stage line, and a user
    stage 1 still wins.

    The ledger reads only the emitter's own calls.  The bridge's deck and
    the deck replayed from its ``model.h5`` make the same calls, so they
    carry the same lines.  The archive stores none of them, so
    ``model_hash`` does not change.
    """

    __slots__ = ("_classes", "_params", "_tags", "_user", "owed")

    def __init__(self) -> None:
        from ...stage._builder import STAGE_FLAG_PARAMETERS

        self._classes = stage_zero_classes()
        self._params = STAGE_FLAG_PARAMETERS
        self._tags: list[int] = []
        self._user = False
        #: Set by the emitter's ``element``: a line is due at the next
        #: solve verb.  A bare attribute, since it is stored per element.
        self.owed = False

    def material(self, mat_type: str, tag: int) -> None:
        """Note an ``nDMaterial`` declaration."""
        if mat_type in self._classes:
            self._tags.append(int(tag))

    def user_flip(self, tag: int) -> None:
        """Note a user ``updateMaterialStage``; the group is then theirs."""
        if int(tag) in self._tags:
            self._user = True

    def parameter(self, args: "tuple[str | int, ...]") -> None:
        """Note a user ``parameter`` on ``args``; a stage-flag name on one
        of the group's tags hands the group to the user too."""
        if len(args) > 1 and args[0] in self._params:
            self.user_flip(int(args[1]))

    def due(self) -> tuple[int, ...]:
        """The tags to write ``-stage 0`` for now, in declaration order."""
        if self._user or not self.owed or not self._tags:
            return ()
        self.owed = False
        return tuple(self._tags)


def validate_class_wide_stage_flags(
    *,
    stage_records: "Iterable[StageRecord]",
    primitives: "Iterable[Primitive]",
    tag_for: Mapping[int, int],
    fem: "FEMData",
    element_owner_stage: Mapping[int, int],
) -> None:
    """Refuse a stage flip that names part of a class-wide group.

    At each stage that flips a group, every material of the group that is
    live at the flip must be named, all with one stage value.  A material
    is live when an element using it is in the domain at the flip line:
    the element emitted globally or in this stage or an earlier one
    (``element_owner_stage``), and not every one of its elements removed
    by ``s.remove_element`` in this stage or an earlier one (the removals
    emit before the flips).  A flip whose named materials are none of
    them live reaches no element, and nothing switches; that is refused
    too.

    Validator V7 runs first and refuses a named material that no element
    uses.  Here every material is the final model's, so one registered
    after the stage block counts.
    """
    from ...stage._builder import CLASS_WIDE_STAGE_FLAG

    stages = tuple(stage_records)
    flips = [stage_flips(st.update_material_stage_records,
                         st.update_parameter_records) for st in stages]
    if not any(flips):
        return
    # Each element spec that uses a group material:
    # (spec, owning stage or -1, {tag: class}).
    users: list[tuple[Element, int, dict[int, str]]] = []
    for spec in primitives:
        if not isinstance(spec, Element):
            continue
        mats: dict[int, str] = {}
        seen: set[int] = set()
        frontier = list(spec.dependencies())
        while frontier:
            dep = frontier.pop()
            if id(dep) in seen:
                continue
            seen.add(id(dep))
            frontier.extend(dep.dependencies())
            cls = type(dep).__name__
            if isinstance(dep, NDMaterial) and cls in CLASS_WIDE_STAGE_FLAG:
                mats[int(tag_for[id(dep)])] = cls
        if mats:
            users.append((spec, element_owner_stage.get(id(spec), -1), mats))
    # tag -> its group's static, for every group material an element uses.
    group_of = {tag: CLASS_WIDE_STAGE_FLAG[cls]
                for _s, _o, mats in users for tag, cls in mats.items()}
    removed: set[int] = set()
    spec_eids: dict[int, frozenset[int]] = {}
    for idx, stage in enumerate(stages):
        for rem in stage.remove_element_records:
            if rem.pg is not None:
                removed.update(
                    int(eid) for eid, _c in expand_pg_to_elements(fem, rem.pg))
            else:
                removed.update(int(e) for e in rem.elements or ())
        if not flips[idx]:
            continue
        # tag -> class of every group material live at this flip.
        live: dict[int, str] = {}
        for spec, owner, mats in users:
            if owner > idx:
                continue
            if removed:
                if id(spec) not in spec_eids:
                    spec_eids[id(spec)] = frozenset(
                        int(eid) for eid, _c in
                        expand_spec_to_elements(fem, spec))
                eids = spec_eids[id(spec)]
                if eids and eids <= removed:
                    continue
            live.update(mats)
        _check_stage(stage.name, flips[idx], live, group_of,
                     CLASS_WIDE_STAGE_FLAG)


def _check_stage(
    name: str,
    flips: "Iterable[tuple[int, int]]",
    live: Mapping[int, str],
    group_of: Mapping[int, str],
    table: Mapping[str, str],
) -> None:
    """Refuse one stage's flips, per group: two values, no named material
    live, or a live material left unnamed."""
    named: dict[str, dict[int, set[int]]] = {}
    for tag, value in flips:
        group = group_of.get(tag)
        if group is not None:
            named.setdefault(group, {}).setdefault(tag, set()).add(value)
    for group, tags in named.items():
        classes = " / ".join(sorted(
            cls for cls, static in table.items() if static == group))
        values = set().union(*tags.values())
        if len(values) > 1:
            raise BridgeError(
                f"Stage {name!r}: the {classes} "
                f"materials are flipped to different stages "
                f"{sorted(values)} in one stage, but their stage flag is "
                f"the class-wide C++ static {group}: it holds one value, "
                "so the last line decides the stage of every one of them.  "
                "Flip them all to the same stage."
            )
        members = {t: c for t, c in live.items() if table[c] == group}
        if not set(tags) & set(members):
            raise BridgeError(
                f"Stage {name!r} flips {classes} "
                f"nDMaterial tag(s) {sorted(tags)}, but every element "
                "using them has been removed (s.remove_element) by this "
                "point.  updateMaterialStage reaches a material through "
                "the domain's elements, so the flip would print "
                "``no effect`` and switch nothing.  Drop the flip, or "
                "move it before the removal."
            )
        unnamed = [f"{cls} (nDMaterial tag {t})"
                   for t, cls in sorted(members.items()) if t not in tags]
        if unnamed:
            raise BridgeError(
                f"Stage {name!r}: this stage flips "
                f"some {classes} materials but not {', '.join(unnamed)}.  "
                f"Their stage flag is the class-wide C++ static {group}, "
                "so updateMaterialStage on one material switches all of "
                "them, and an unnamed material skips its own flip "
                "(Elastic2Plastic on a flip to 1): its answer matches "
                "neither stage.  Name every one of them in this stage, "
                "with one stage value."
            )
