"""Tag plans for direct test calls of the ``build.py`` writers.

The MP-element, interface, contact and recorder-region writers read their
tags from the build's :class:`TagPlan`, which every emit helper takes in
place of an allocator (ADR 0114 D4, amended; K1-3d S6a #1494, S6 #1458),
so a test that calls a writer directly hands it a plan.

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
from typing import Any, Iterable, Iterator, Sequence

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
    plan_regions,
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
    recorders: Iterable[Any] = (),
    tags: TagAllocator | None = None,
) -> TagPlan:
    """The tag plan of a flat emit over ``fem``.

    The plan holds the MP elements, interfaces and contacts a flat emit
    of ``fem`` writes, in the order it writes them: the global
    MP-constraint pass, the reinforce and embed ties, the unclaimed
    interfaces, the rebar cells, then each stage's claimed constraints
    and interfaces (``stages``), then every contact and contact plane.
    Its regions are those of ``recorders``, filtered recorder specs, one
    site each in the given order, keyed as the build keys them
    (``("recorder", id(spec))``, :meth:`region_keys`).

    ``tags`` seeds the planner: the plan continues its counters, as a
    bridge plan continues the seeded primitive and element tags. A fresh
    allocator is used when it is ``None``. The planner is frozen once
    planned, so the plan mints nothing.
    """
    planner = TagAllocator() if tags is None else tags
    inputs = _PlanInputs(fem=fem, stage_records=tuple(stages))
    mode = TagMode(
        split=False, partitioned=False, staged=bool(inputs.stage_records))
    mp, interfaces = _plan_mp_elements_and_interfaces(
        inputs, mode, planner)  # type: ignore[arg-type]
    contacts = _plan_contacts(inputs, mode, planner)  # type: ignore[arg-type]
    regions = plan_regions(
        [(("recorder", id(spec)), spec.region_keys()) for spec in recorders],
        planner,
    )
    planner.freeze()
    return TagPlan(
        mode=mode,
        allocator=planner,
        elements=ElementTagPlan(),
        transforms=TransformTagPlan(),
        regions=RegionTagPlan(regions=regions, fem=fem),
        parameters=ParameterTagPlan(),
        mp_elements=MPElementTagPlan(mp=mp),
        interfaces=InterfaceTagPlan(interfaces=interfaces),
        contacts=ContactTagPlan(contacts=contacts),
    )


def parameter_plan(
    sites: Sequence[Any], *, partitioned: bool = False,
) -> TagPlan:
    """A staged emit's tag plan holding only the parameter ``sites``.

    The sites (``build.ParameterSite``, from ``initial_stress_sites`` or
    ``parameter_flip_sites``) are numbered by ``build.plan_parameters``,
    the loop :func:`~apeGmsh.opensees._internal.tag_plan.plan_tags` runs,
    from a fresh planner, so a parameter writer (``emit_activate_absorbing``,
    ``emit_update_parameters``) can be called directly with a plan.
    """
    from apeGmsh.opensees._internal.build import plan_parameters

    planner = TagAllocator()
    lines = plan_parameters(sites, planner)
    planner.freeze()
    return TagPlan(
        mode=TagMode(split=False, partitioned=partitioned, staged=True),
        allocator=planner,
        elements=ElementTagPlan(),
        transforms=TransformTagPlan(),
        regions=RegionTagPlan(),
        parameters=ParameterTagPlan(
            lines=lines,
            owners=tuple((site.record, site.rank) for site in sites)),
        mp_elements=MPElementTagPlan(),
        interfaces=InterfaceTagPlan(),
        contacts=ContactTagPlan(),
    )


class _NodeConstraints:
    """``fem.nodes.constraints``: iterable over node-level records."""

    def __init__(self, records: Sequence[Any]) -> None:
        self._records = tuple(records)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._records)


class _SurfaceConstraints:
    """``fem.elements.constraints``: its ``interpolations()`` stream."""

    def __init__(self, records: Sequence[Any]) -> None:
        self._records = tuple(records)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._records)

    def interpolations(self) -> Iterator[Any]:
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
