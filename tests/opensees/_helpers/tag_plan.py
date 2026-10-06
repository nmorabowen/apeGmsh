"""Tag-plan allocators for direct test calls of the ``build.py`` writers.

The MP-element, interface and contact writers read their tags from the
build's tag plan, through the emit allocator a :class:`TagPlan` forks
(ADR 0114 D4, amended). Until K1-3d S6 they also accept a plain
:class:`TagAllocator` and plan their own rows from it; S6 removes that
fallback, so a test that calls a writer directly hands it a plan's emit
allocator instead (K1-3d S6a, #1494).

:func:`emit_tags` makes that plan for a FEM snapshot the way
:func:`~apeGmsh.opensees._internal.tag_plan.plan_tags` makes it for a
built model, through the same planning walks, so the planned tags are
the tags a bridge emit of that snapshot writes. A test does not build a
model; it hands the snapshot (often a stub carrying only the streams a
writer reads) and, where a writer takes bare records, :func:`stub_fem`
wraps them in one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees._internal.tag_plan import (
    ContactTagPlan,
    ElementTagPlan,
    InterfaceTagPlan,
    MPElementTagPlan,
    ParameterTagPlan,
    RegionTagPlan,
    TagMode,
    TagPlan,
    TransformTagPlan,
    _plan_contacts,
    _plan_mp_elements_and_interfaces,
)


@dataclass(frozen=True)
class StageClaims:
    """One stage's claimed MP constraints and interfaces.

    The two streams of a ``StageRecord`` the MP-element and interface
    planning reads.
    """

    stage_constraint_records: Sequence[Any] = ()
    stage_interface_records: Sequence[Any] = ()


@dataclass(frozen=True)
class _PlanInputs:
    """The inputs of a built model that the MP-element, interface and
    contact planning of a flat emit reads: the FEM snapshot and the
    stages' claims."""

    fem: Any
    stage_records: tuple[StageClaims, ...] = field(default=())

    def _claimed_constraint_ids(self) -> set[int]:
        from apeGmsh.opensees.apesees import BuiltModel

        return BuiltModel._claimed_constraint_ids(self)  # type: ignore[arg-type]

    def _claimed_interface_ids(self) -> set[int]:
        from apeGmsh.opensees.apesees import BuiltModel

        return BuiltModel._claimed_interface_ids(self)  # type: ignore[arg-type]


def emit_tags(
    fem: Any,
    *,
    stages: Iterable[StageClaims] = (),
    tags: TagAllocator | None = None,
) -> TagAllocator:
    """The emit allocator of a flat emit's tag plan over ``fem``.

    The plan holds the MP elements, interfaces and contacts a flat emit
    of ``fem`` writes, in the order it writes them: the global
    MP-constraint pass, the reinforce and embed ties, the unclaimed
    interfaces, the rebar cells, then each stage's claimed constraints
    and interfaces (``stages``), then every contact and contact plane.

    ``tags`` seeds the planner: the plan continues its counters, as a
    bridge plan continues the seeded primitive and element tags. A fresh
    allocator is used when it is ``None``. The planner is frozen once
    planned; the returned fork continues it and refuses a mint of any
    family the plan holds.
    """
    planner = TagAllocator() if tags is None else tags
    inputs = _PlanInputs(fem=fem, stage_records=tuple(stages))
    mode = TagMode(
        split=False, partitioned=False, staged=bool(inputs.stage_records))
    mp, interfaces = _plan_mp_elements_and_interfaces(
        inputs, mode, planner)  # type: ignore[arg-type]
    contacts = _plan_contacts(inputs, mode, planner)  # type: ignore[arg-type]
    planner.freeze()
    plan = TagPlan(
        mode=mode,
        allocator=planner,
        elements=ElementTagPlan(),
        transforms=TransformTagPlan(),
        regions=RegionTagPlan(),
        parameters=ParameterTagPlan(),
        mp_elements=MPElementTagPlan(mp=mp),
        interfaces=InterfaceTagPlan(interfaces=interfaces),
        contacts=ContactTagPlan(contacts=contacts),
    )
    return plan.emit_allocator()


class _NodeConstraints:
    """``fem.nodes.constraints``: iterable over node-level records."""

    def __init__(self, records: Sequence[Any]) -> None:
        self._records = tuple(records)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._records)


class _SurfaceConstraints:
    """``fem.elements.constraints``: its ``interpolations()`` stream."""

    def __init__(self, records: Sequence[Any]) -> None:
        self._records = tuple(records)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._records)

    def interpolations(self):  # type: ignore[no-untyped-def]
        return iter(self._records)


def stub_fem(
    *,
    node_constraints: Sequence[Any] | None = None,
    interpolations: Sequence[Any] | None = None,
    **side_lists: Sequence[Any],
) -> Any:
    """A FEM snapshot stub holding exactly the given streams.

    ``node_constraints`` become ``fem.nodes.constraints`` and
    ``interpolations`` ``fem.elements.constraints.interpolations()``;
    every other keyword (``reinforce_ties``, ``embed_ties``,
    ``rebar_elements``, ``interfaces``, ``contacts``, ``contact_planes``)
    a side list on ``fem.elements``. Writers that take bare records
    (``_emit_kinematic_couplings``, ``_emit_one_interpolation``,
    ``emit_stage_interfaces``) are planned over such a stub.
    """
    elements: dict[str, Any] = {k: list(v) for k, v in side_lists.items()}
    if interpolations is not None:
        elements["constraints"] = _SurfaceConstraints(interpolations)
    nodes: dict[str, Any] = {}
    if node_constraints is not None:
        nodes["constraints"] = _NodeConstraints(node_constraints)
    fem = type("StubFem", (), {})()
    fem.elements = type("StubElements", (), elements)()
    fem.nodes = type("StubNodes", (), nodes)()
    return fem
