"""The tag plan's oracle (ADR 0114 D4, amended): the plan is what the emit writes.

For every model the K1-3 tag-law pins drive (``_tag_streams.models()``,
and the contact cases) and the emit mode it takes, the plan the emit read
(``BuiltModel.emit`` memoises one per mode) must plan exactly the derived
``(kind, tag)`` rows the emit writes. The oracle is the emit itself, read
twice:

* the K1-3 tap records the ``(kind, tag)`` row of every emitted verb;
* a mint log records every ``TagAllocator`` call the emit makes, with the
  function that made it. The calls on the plan's own allocator are the
  planning (the seeding, then the migrated families' mints); every other
  call is an emit-time mint. ``_MINT_SITES`` names the family of each
  minting function (an unknown one raises), so every tag has an owner.

A tapped row of a derived verb (``VERB_KIND``) then belongs to:

* nobody, when it is a registered primitive's own tag (a material, a
  transform's first ``geomTransf``);
* the family whose function minted it at emit time;
* while the element family is pending, the element family, for an
  ``element`` row nobody minted inside the emit's reserved range (a FEM
  id under ``element_tags="fem"``);
* otherwise the plan: a tag the emit read rather than minted, which some
  migrated family's plan must hold.

``test_plan_equals_tapped_stream`` runs per family. A pending family's
case is ``xfail(strict=True)``: its ``stream()`` raises. A migrated
family's case fails if the emit still mints any of its tags, and otherwise
compares its plan, exactly, with the planned rows of its kinds less the
other migrated families' plans in those kinds. Rows compare by
``(verb, tag)``: an element spec picks its type token in its own
``_emit``, so the plan's element rows carry the bare verb. A partitioned
deck writes one region tag in every rank block that holds the region's
members, so its region rows compare as distinct rows (``PER_RANK_KINDS``);
a flat deck's compare as a multiset. An initial stress declares its three
``parameter`` tags through ``step_hook_ramp``'s targets, which the K1-3 tap
does not see, so the oracle's own tap (``_ramp_tapped``) records one
``("step_hook_ramp", tag)`` row per target; ``addToParameter`` only
references those tags. The
``MIGRATED`` flag in ``tag_plan.py`` drives the marker, so a migration
slice turns its cases into real comparisons in the same commit;
``test_every_family_has_rows`` keeps every family's comparison
non-vacuous, and ``test_short_or_empty_element_plan_fails_the_oracle``
proves the comparison catches a plan that drops rows.

Beside the oracle, this file pins what S1 and S2 ship:
``TagAllocator.freeze()`` and ``fork()``, ``TagLawError``, a plan seeded
exactly as the emit used to seed its allocator (``element_tags="fem"``
included), the plan each emit path reads (``plan_of``), and the refusal
of ``reset()`` on every fork.
"""
from __future__ import annotations

import inspect
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
import pytest

from apeGmsh.opensees._internal.tag_allocator import TagAllocator, TagLawError
from apeGmsh.opensees._internal.tag_plan import (
    FAMILIES,
    FAMILY_PLANS,
    TagMode,
    TagPlan,
    emit_mode,
    plan_of,
    plan_tags,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.contract import _tag_streams as ts

Row = tuple[str, int]

#: Tap verb -> the allocator kind of its tag, for every verb whose tag the
#: build mints (or, for a primitive's own verb, may mint).
VERB_KIND: dict[str, str] = {
    "element": "element",
    "embeddedNode": "element",
    "embedded_rebar": "element",
    "embedded_node": "element",
    "geomTransf": "geomTransf",
    "uniaxialMaterial": "uniaxialMaterial",
    "region": "region",
    "step_hook_ramp": "parameter",
    "update_parameter": "parameter",
    "flip_element_stage": "parameter",
    "contact_surface": "contactSurface",
    "contact": "contact",
    "contact_plane": "contact",
}

#: Tap verbs whose tag is a node id, a registered primitive's tag that
#: no family mints, or a reference to a tag minted elsewhere (an
#: ``addToParameter`` names a ramp tag its ``step_hook_ramp`` declared:
#: ``test_add_to_parameter_names_a_planned_ramp_tag``).
NON_DERIVED_VERBS: frozenset[str] = frozenset({
    "node", "fix", "mass", "load", "sp", "remove_element", "timeSeries",
    "pattern_open", "nDMaterial", "section", "section_open",
    "beamIntegration", "damping", "addToParameter",
})

#: The function that mints at emit time -> the family its tags belong to.
_MINT_SITES: dict[str, str] = {
    "allocate_element_tags": "elements",
    "emit_element_spec": "elements",
    "plan_transform_specs": "transforms",
    "plan_regions": "regions",
    "plan_parameters": "parameters",
    "plan_mp_elements": "mp_elements",
    "plan_interface_tags": "interfaces",
    "plan_contacts": "contacts",
}

#: The owner of a row some migrated family's plan must hold.
PLANNED = "<planned>"

#: Kinds whose one tag a partitioned deck writes once per rank that holds
#: the object's members (a region: ADR 0027 INV-4). On a partitioned deck
#: their emitted rows compare as distinct ``(verb, tag)`` rows; every other
#: kind, and every kind on a flat deck, compares as a multiset, so a tag
#: written twice is caught.
PER_RANK_KINDS: frozenset[str] = frozenset({"region"})


def _contact_ranks_fem(n_ranks: int, *, partitioned: bool = True) -> Any:
    """A truss stub with one contact and one contact plane per rank.

    Rank ``r`` natively holds the master pad ``b+1..b+4`` and the plane's
    slave node ``b+7`` (``b = 100 (r + 1)``), so it owns contact ``c{r}``
    and plane ``p{r}``. The contact's slave nodes ``b+5, b+6`` live on the
    next rank, so the owner declares them as ghosts. Contacts and planes
    are declared in reverse rank order, so the partitioned deck, which
    numbers them rank by rank, numbers them differently from the flat
    one. The stub's elements are not iterable, so the owner pick is the
    node tally (ADR 0092 INV-1).
    """
    from apeGmsh._kernel.records._constraints import (
        ContactPlaneRecord,
        ContactRecord,
    )

    from tests.opensees.fixtures.fem_stub import (
        FEMStub,
        _ElementGroupView,
        _ElementsStub,
        _NodesStub,
    )

    ids: list[int] = []
    coords: list[tuple[float, float, float]] = []
    bar_ids: list[int] = []
    bars: list[tuple[int, int]] = []
    rank_nodes: dict[int, list[int]] = {r: [] for r in range(n_ranks)}
    rank_elems: dict[int, list[int]] = {r: [] for r in range(n_ranks)}
    contacts: list[Any] = []
    planes: list[Any] = []
    for r in range(n_ranks):
        b, x, nxt = 100 * (r + 1), 10.0 * r, (r + 1) % n_ranks
        ids += [b + k for k in range(1, 8)]
        coords += [(x, 0.0, 0.0), (x + 1, 0.0, 0.0), (x + 1, 1.0, 0.0),
                   (x, 1.0, 0.0), (x + 0.2, 0.2, 0.5), (x + 0.8, 0.8, 0.5),
                   (x, 0.0, -1.0)]
        rank_nodes[r] += [b + 1, b + 2, b + 3, b + 4, b + 7]
        rank_nodes[nxt] += [b + 5, b + 6]
        e = 10 * (r + 1)
        for eid, conn, rank in ((e + 1, (b + 1, b + 2), r),
                                (e + 2, (b + 3, b + 7), r),
                                (e + 3, (b + 5, b + 6), nxt)):
            bar_ids.append(eid)
            bars.append(conn)
            rank_elems[rank].append(eid)
        contacts.insert(0, ContactRecord(
            kind="contact", name=f"c{r}", formulation="nts",
            master_faces=np.array([[b + 1, b + 2, b + 3, b + 4]],
                                  dtype=np.int64),
            master_nps=4, slave_nodes=[b + 5, b + 6],
            kn=1.0e6, kt=0.0, mu=0.0,
        ))
        planes.insert(0, ContactPlaneRecord(
            kind="contact_plane", name=f"p{r}", slave_nodes=[b + 7],
            normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, -2.0), kn=1.0e6,
        ))
    fem = FEMStub(
        nodes=_NodesStub(ids=ids, coords=coords, node_pgs={}),
        elements=_ElementsStub(elem_pgs={"Bars": _ElementGroupView(
            ids=tuple(bar_ids), connectivity=tuple(bars))}),
    )
    if partitioned:
        fem.set_partitions([
            (r, rank_nodes[r], rank_elems[r]) for r in range(n_ranks)])
    fem.elements.contacts = contacts
    fem.elements.contact_planes = planes
    return fem


def _contact_ranks(n_ranks: int, *, partitioned: bool = True) -> Any:
    from typing import cast

    from apeGmsh.opensees import apeSees

    ops = apeSees(cast(Any, _contact_ranks_fem(
        n_ranks, partitioned=partitioned)))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6)
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    return ops


#: The contact cases: 2 and 4 ranks, and the 4-rank stub emitted flat.
_CONTACT_MODELS: dict[str, Callable[[], Any]] = {
    "contact_ranks_2/partitioned": lambda: _contact_ranks(2),
    "contact_ranks_4/partitioned": lambda: _contact_ranks(4),
    "contact_ranks_4/flat": lambda: _contact_ranks(4, partitioned=False),
}

def _mp_ranks_fem(
    n_ranks: int, *, partitioned: bool = True, embed: bool = False,
) -> Any:
    """A truss stub with one of every MP element on each rank.

    Rank ``r`` (``b = 100 (r + 1)``) natively holds a triangle
    ``b+1..b+3`` (with ``b+4``), a kinematic coupling ``kc{r}``
    (``b+6 -> b+7``), an element rigid body ``rb{r}`` (``b+8, b+9``), a
    reinforce tie ``rt{r}`` of ``b+10`` in the triangle, and a rebar
    cell ``b+10 - b+11`` on bar ``bar{r}``; its penalty tie ``tie{r}``
    binds ``b+5``, which lives on the next rank, so the tie's host rank
    declares it as a ghost. Every record is declared in reverse rank
    order, so the partitioned deck, which numbers each rank's elements
    in its block, numbers them differently from the flat one. ``embed``
    adds a ``g.embed`` tie ``et{r}`` of ``b+11`` in the triangle (flat
    only: the partitioned emit refuses ``g.embed``).
    """
    from apeGmsh._kernel.records._constraints import (
        EmbedTieRecord,
        InterpolationRecord,
        NodeGroupRecord,
        ReinforceTieRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind
    from apeGmsh._kernel.records._rebar import RebarElementRecord

    from tests.opensees.fixtures.fem_stub import (
        FEMStub,
        _ElementGroupView,
        _ElementsStub,
        _NodesStub,
    )

    ids: list[int] = []
    coords: list[tuple[float, float, float]] = []
    bar_ids: list[int] = []
    bars: list[tuple[int, int]] = []
    rank_nodes: dict[int, list[int]] = {r: [] for r in range(n_ranks)}
    rank_elems: dict[int, list[int]] = {r: [] for r in range(n_ranks)}
    node_recs: list[Any] = []
    interps: list[Any] = []
    ties: list[Any] = []
    embeds: list[Any] = []
    rebar: list[Any] = []
    for r in range(n_ranks):
        b, x, nxt = 100 * (r + 1), 10.0 * r, (r + 1) % n_ranks
        ids += [b + k for k in range(1, 12)]
        coords += [(x, 0.0, 0.0), (x + 1, 0.0, 0.0), (x + 1, 1.0, 0.0),
                   (x, 1.0, 0.0), (x + 0.3, 0.3, 0.0), (x, 0.0, 2.0),
                   (x + 1, 0.0, 2.0), (x, 1.0, 2.0), (x + 1, 1.0, 2.0),
                   (x + 0.5, 0.4, 0.0), (x + 0.5, 0.4, 1.0)]
        rank_nodes[r] += [b + k for k in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11)]
        rank_nodes[nxt].append(b + 5)
        e = 10 * (r + 1)
        for k, conn in enumerate(((b + 1, b + 2), (b + 2, b + 3),
                                  (b + 3, b + 4), (b + 6, b + 7),
                                  (b + 8, b + 9))):
            bar_ids.append(e + k)
            bars.append(conn)
            rank_elems[r].append(e + k)
        node_recs[:0] = [
            NodeGroupRecord(
                kind=ConstraintKind.RIGID_BODY, master_node=b + 8,
                slave_nodes=[b + 9], as_element=True, name=f"rb{r}"),
            NodeGroupRecord(
                kind=ConstraintKind.KINEMATIC_COUPLING, master_node=b + 6,
                slave_nodes=[b + 7], dofs=[1, 2, 3], name=f"kc{r}"),
        ]
        interps.insert(0, InterpolationRecord(
            kind=ConstraintKind.TIE, slave_node=b + 5,
            master_nodes=[b + 1, b + 2, b + 3], dofs=[1, 2, 3],
            enforce="penalty", stiffness=1.0e10, name=f"tie{r}"))
        ties.insert(0, ReinforceTieRecord(
            kind="reinforce", name=f"rt{r}", rebar_node=b + 10,
            host_nodes=[b + 1, b + 2, b + 3],
            weights=np.array([0.2, 0.4, 0.4]),
            direction=np.array([0.0, 0.0, 1.0]), perfect=1.0e8))
        embeds.insert(0, EmbedTieRecord(
            kind="embed", name=f"et{r}", node=b + 11,
            host_nodes=[b + 1, b + 2, b + 3],
            weights=np.array([0.2, 0.4, 0.4]), k=1.0e8))
        rebar.insert(0, RebarElementRecord(
            pg=f"bar{r}", element="truss", material="steel", area=1.0e-4,
            connectivity=((b + 10, b + 11),)))
    fem = FEMStub(
        nodes=_NodesStub(ids=ids, coords=coords, node_pgs={}),
        elements=_ElementsStub(elem_pgs={"Bars": _ElementGroupView(
            ids=tuple(bar_ids), connectivity=tuple(bars))}),
    )
    if partitioned:
        fem.set_partitions([
            (r, rank_nodes[r], rank_elems[r]) for r in range(n_ranks)])
    fem.add_node_constraints(node_recs)
    fem.add_surface_constraints(interps)
    fem.elements.reinforce_ties = ties
    fem.elements.rebar_elements = rebar
    if embed:
        fem.elements.embed_ties = embeds
    return fem


def _mp_ranks(
    n_ranks: int, *, partitioned: bool = True, staged: bool = False,
    embed: bool = False,
) -> Any:
    """:func:`_mp_ranks_fem` under a truss; ``staged`` claims rank 0's
    coupling and rigid body and rank 1's tie in a second stage."""
    from typing import cast

    from apeGmsh.opensees import apeSees

    ops = apeSees(cast(Any, _mp_ranks_fem(
        n_ranks, partitioned=partitioned, embed=embed)))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6)
    ops.uniaxialMaterial.ElasticMaterial(E=2.0e5, name="steel")
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    if staged:
        def chain() -> dict[str, Any]:
            return {
                "test": ops.test.NormDispIncr(tol=1e-4, max_iter=10),
                "algorithm": ops.algorithm.Newton(),
                "integrator": ops.integrator.LoadControl(dlam=1.0),
                "constraints": ops.constraints.Transformation(),
                "numberer": (ops.numberer.ParallelPlain() if partitioned
                             else ops.numberer.Plain()),
                "system": (ops.system.Mumps() if partitioned
                           else ops.system.BandGeneral()),
                "analysis": ops.analysis.Static(),
            }
        with ops.stage(name="s1") as s:
            s.analysis(**chain())
            s.run(n_increments=1)
        with ops.stage(name="s2") as s:
            s.kinematic_coupling(name="kc0")
            s.rigid_link(name="rb0")
            s.tie(name="tie1")
            s.analysis(**chain())
            s.run(n_increments=1)
    return ops


def _iface(**kw: Any) -> Any:
    from tests.opensees.integration import (
        test_interface_partitioned_emit as ip,
    )
    return ip._quad_ops(ip._fem(embed=True, **kw))


def _iface_staged(*, partitioned: bool) -> Any:
    from tests.opensees.integration import (
        test_interface_partitioned_emit as ip,
    )
    from tests.opensees.integration import (
        test_interface_partitioned_staged_emit as ips,
    )
    if partitioned:
        return ips._staged(ips._quad_ops(ips._fem()), service=True)
    ops = ips._quad_ops(ip._fem())

    def chain() -> dict[str, Any]:
        return {
            "test": ops.test.NormDispIncr(tol=1e-6, max_iter=25),
            "algorithm": ops.algorithm.Newton(),
            "integrator": ops.integrator.LoadControl(dlam=1.0),
            "constraints": ops.constraints.Transformation(),
            "numberer": ops.numberer.Plain(),
            "system": ops.system.BandGeneral(),
            "analysis": ops.analysis.Static(),
        }
    with ops.stage(name="ground") as s:
        s.analysis(**chain())
        s.run(n_increments=1, dt=1.0)
    with ops.stage(name="install") as s:
        s.activate(pgs=["liner"])
        s.interface(name="RockLiner")
        s.analysis(**chain())
        s.run(n_increments=1, dt=1.0)
    return ops


def _param_ranks_fem(n_ranks: int, *, partitioned: bool = True) -> Any:
    """A truss stub of two bars per rank, for the parameter sites.

    Rank ``r`` (``b = 100 (r + 1)``, ``e = 10 (r + 1)``) natively holds the
    nodes ``b+1..b+3`` and the bars ``e`` (``b+1 - b+2``) and ``e+1``
    (``b+2 - b+3``). ``Bars`` groups every bar; ``Edge`` the second bar
    of every rank but rank 0, so rank 0 holds none of it.
    """
    from tests.opensees.fixtures.fem_stub import (
        FEMStub,
        _ElementGroupView,
        _ElementsStub,
        _NodesStub,
    )

    ids: list[int] = []
    coords: list[tuple[float, float, float]] = []
    bar_ids: list[int] = []
    bars: list[tuple[int, int]] = []
    edge_ids: list[int] = []
    edge: list[tuple[int, int]] = []
    rank_nodes: dict[int, list[int]] = {}
    rank_elems: dict[int, list[int]] = {}
    for r in range(n_ranks):
        b, e, x = 100 * (r + 1), 10 * (r + 1), 10.0 * r
        ids += [b + 1, b + 2, b + 3]
        coords += [(x, 0.0, 0.0), (x + 1, 0.0, 0.0), (x + 2, 0.0, 0.0)]
        bar_ids += [e, e + 1]
        bars += [(b + 1, b + 2), (b + 2, b + 3)]
        if r:
            edge_ids.append(e + 1)
            edge.append((b + 2, b + 3))
        rank_nodes[r] = [b + 1, b + 2, b + 3]
        rank_elems[r] = [e, e + 1]
    fem = FEMStub(
        nodes=_NodesStub(ids=ids, coords=coords, node_pgs={}),
        elements=_ElementsStub(elem_pgs={
            "Bars": _ElementGroupView(
                ids=tuple(bar_ids), connectivity=tuple(bars)),
            "Edge": _ElementGroupView(
                ids=tuple(edge_ids), connectivity=tuple(edge)),
        }),
    )
    if partitioned:
        fem.set_partitions([
            (r, rank_nodes[r], rank_elems[r]) for r in range(n_ranks)])
    return fem


def _param_ranks(
    n_ranks: int, *, partitioned: bool = True, staged: bool = True,
) -> Any:
    """:func:`_param_ranks_fem` under a truss, with every parameter site.

    A global initial stress ``g`` on ``Bars``. Unless ``staged`` is false,
    two stages follow. ``s1``: an initial stress ``s1`` on rank 0's first
    bar, an absorbing flip on ``Edge`` (no element on rank 0), then the
    updates ``E`` (the last rank's first bar only) and ``A`` (``Bars``).
    ``s2``: an initial stress ``s2`` on ``Edge``, then an absorbing flip
    of rank 0's and the last rank's first bars.
    """
    from typing import cast

    from apeGmsh.opensees import apeSees

    ops = apeSees(cast(Any, _param_ranks_fem(
        n_ranks, partitioned=partitioned)))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6)
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    ops.initial_stress(name="g", pg="Bars", sigma_xx=-1.0, sigma_yy=-2.0,
                       sigma_zz=-3.0, ramp_steps=2)
    if not staged:
        return ops
    last = 10 * n_ranks

    def chain() -> dict[str, Any]:
        return {
            "test": ops.test.NormDispIncr(tol=1e-4, max_iter=10),
            "algorithm": ops.algorithm.Newton(),
            "integrator": ops.integrator.LoadControl(dlam=1.0),
            "constraints": ops.constraints.Transformation(),
            "numberer": (ops.numberer.ParallelPlain() if partitioned
                         else ops.numberer.Plain()),
            "system": (ops.system.Mumps() if partitioned
                       else ops.system.BandGeneral()),
            "analysis": ops.analysis.Static(),
        }
    with ops.stage(name="s1") as s:
        s.initial_stress(name="s1", elements=[10], sigma_xx=-1.0,
                         sigma_yy=-1.0, sigma_zz=-1.0, ramp_steps=1)
        s.activate_absorbing(pg="Edge")
        s.update_parameter("E", 2.0e6, elements=[last])
        s.update_parameter("A", 0.02, pg="Bars")
        s.analysis(**chain())
        s.run(n_increments=1)
    with ops.stage(name="s2") as s:
        s.initial_stress(name="s2", pg="Edge", sigma_xx=-1.0,
                         sigma_yy=-1.0, sigma_zz=-1.0, ramp_steps=1)
        s.activate_absorbing(elements=[10, last])
        s.analysis(**chain())
        s.run(n_increments=1)
    return ops


#: The parameter cases (K1-3d S5): every parameter site, flat and staged,
#: and partitioned over 2 and 4 ranks, staged or not.
_PARAM_MODELS: dict[str, Callable[[], Any]] = {
    "param_ranks_2/partitioned": lambda: _param_ranks(2, staged=False),
    "param_ranks_4/staged": lambda: _param_ranks(4, partitioned=False),
    "param_ranks_2/staged_partitioned": lambda: _param_ranks(2),
    "param_ranks_4/staged_partitioned": lambda: _param_ranks(4),
}


#: The MP-element and interface cases (K1-3d S3c): every MP site over 2
#: and 4 ranks, flat, and staged both ways; interfaces beside an
#: element-minting embedded tie, flat, over 2 and 4 ranks, and staged
#: both ways.
_MP_MODELS: dict[str, Callable[[], Any]] = {
    "mp_ranks_2/partitioned": lambda: _mp_ranks(2),
    "mp_ranks_4/partitioned": lambda: _mp_ranks(4),
    "mp_ranks_4/flat": lambda: _mp_ranks(4, partitioned=False),
    "mp_ranks_embed_4/flat": lambda: _mp_ranks(
        4, partitioned=False, embed=True),
    "mp_ranks_2/staged": lambda: _mp_ranks(
        2, partitioned=False, staged=True),
    "mp_ranks_2/staged_partitioned": lambda: _mp_ranks(2, staged=True),
    "iface_embed/flat": lambda: _iface(),
    "iface_embed_ranks_2/partitioned": lambda: _iface(cut="split"),
    "iface_embed_ranks_4/partitioned": lambda: _iface(n_parts=4),
    "iface_staged/staged": lambda: _iface_staged(partitioned=False),
    "iface_staged_ranks_2/staged_partitioned": (
        lambda: _iface_staged(partitioned=True)),
}

_MODELS = {**ts.models(), **_CONTACT_MODELS, **_MP_MODELS, **_PARAM_MODELS}
CASES: tuple[str, ...] = tuple(sorted(_MODELS))


def _verb(kind: str) -> str:
    return kind.split(":", 1)[0]


# ---------------------------------------------------------------------------
# The mint log
# ---------------------------------------------------------------------------


class _Op(NamedTuple):
    allocator: int          # id() of the TagAllocator
    method: str
    args: tuple[Any, ...]
    result: Any
    caller: str             # the function that called the method


_LOGGED = ("allocate", "allocate_block", "allocate_for", "reserve_through")


@contextmanager
def _mint_log() -> Iterator[list[_Op]]:
    """Record every outermost ``TagAllocator`` call made inside the block."""
    log: list[_Op] = []
    depth = [0]
    originals = {m: getattr(TagAllocator, m) for m in _LOGGED}

    def wrap(method: str, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self: TagAllocator, *args: Any) -> Any:
            caller = sys._getframe(1).f_code.co_name
            depth[0] += 1
            try:
                out = orig(self, *args)
            finally:
                depth[0] -= 1
            if depth[0] == 0:
                log.append(_Op(id(self), method, args, out, caller))
            return out
        return wrapper

    for m, f in originals.items():
        setattr(TagAllocator, m, wrap(m, f))
    try:
        yield log
    finally:
        for m, f in originals.items():
            setattr(TagAllocator, m, f)


class Case(NamedTuple):
    bm: Any
    stream: tuple[Row, ...]
    plan: TagPlan                         # the plan the emit read
    seed: dict[str, int]                  # the plan's counters before its mints
    seeded: frozenset[tuple[str, int]]    # (kind, tag) of each seeded primitive
    planned: dict[str, list[int]]         # kind -> tags the planner minted
    mints: dict[tuple[str, int], str]     # (kind, tag) minted at emit -> family
    minted: dict[str, list[int]]          # kind -> tags minted at emit, in order


def _mint_range(op: _Op) -> range:
    first = int(op.result)
    n = 1 if op.method == "allocate" else int(op.args[1])
    return range(first, first + n)


def _mint_family(name: str, op: _Op) -> str:
    if op.caller not in _MINT_SITES:
        raise AssertionError(
            f"{name}: {op.caller}() mints {op.args[0]!r}; name its family "
            "in _MINT_SITES"
        )
    return _MINT_SITES[op.caller]


def _read_log(name: str, log: list[_Op], plan: TagPlan) -> tuple[
    dict[str, int], frozenset[tuple[str, int]], dict[str, list[int]],
    dict[tuple[str, int], str], dict[str, list[int]],
]:
    """Split the log into the planning and the emit-time mints."""
    planner = id(plan.allocator)
    assert any(op.allocator == planner for op in log), (
        f"{name}: the emit's plan was not made inside this emit")
    emit_allocators = {op.allocator for op in log} - {planner}
    assert len(emit_allocators) <= 1, (
        f"{name}: the emit minted from {len(emit_allocators)} allocators "
        "besides the plan's; it mints from one emit_allocator() fork"
    )
    seed: dict[str, int] = {}
    seeded: set[tuple[str, int]] = set()
    planned: dict[str, list[int]] = {}
    mints: dict[tuple[str, int], str] = {}
    minted: dict[str, list[int]] = {}
    for op in log:
        if op.allocator != planner:
            assert op.method not in ("allocate_for", "reserve_through"), (
                f"{name}: {op.caller}() seeds the emit allocator with "
                f"{op.method}; the plan seeds it")
            family = _mint_family(name, op)
            for tag in _mint_range(op):
                mints[(op.args[0], tag)] = family
                minted.setdefault(op.args[0], []).append(tag)
            continue
        if op.method in ("allocate_for", "reserve_through"):
            assert not planned, (
                f"{name}: {op.method} after the planner's first mint")
            if op.method == "allocate_for":
                kind, tag = op.args[1], int(op.result)
                seeded.add((kind, tag))
            else:
                kind, tag = op.args[0], int(op.args[1])
            seed[kind] = max(seed.get(kind, 0), tag)
            continue
        family = _mint_family(name, op)
        assert FAMILY_PLANS[family].MIGRATED, (
            f"{name}: the planner mints for pending family {family}")
        planned.setdefault(op.args[0], []).extend(_mint_range(op))
    return seed, frozenset(seeded), planned, mints, minted


def _ramp_tapped(cls: type) -> type:
    """:func:`ts.tapped` ``cls``, whose ``step_hook_ramp`` also records a
    ``("step_hook_ramp", tag)`` row per ramp target.

    The ramp declares its ``parameter`` tags through ``targets``, not a
    Protocol tag argument, so the K1-3 tap records no row for it. These
    rows are the declarations the parameter plan's ramp rows compare to;
    the ``addToParameter`` lines only reference them.
    """
    base = ts.tapped(cls)

    def step_hook_ramp(self: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        if self._tap_depth == 0:
            self.tap.extend(
                ("step_hook_ramp", int(tag)) for tag, _value in kwargs["targets"])
        return base.step_hook_ramp(self, name, *args, **kwargs)

    return type(f"Ramp{base.__name__}", (base,), {
        "step_hook_ramp": step_hook_ramp})


def _oracle_stream(bm: Any, cls: type = RecordingEmitter) -> list[Row]:
    """``bm``'s emit through :func:`_ramp_tapped` ``cls``."""
    import warnings

    emitter = _ramp_tapped(cls)()
    emitter.tap = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm.emit(emitter)
    stream: list[Row] = emitter.tap
    return stream


@lru_cache(maxsize=None)
def _case(name: str) -> Case:
    """Everything one case's oracle reads, built once."""
    return _build_case(name)


def _build_case(name: str) -> Case:
    """Build case ``name`` afresh (a mutation test patches the emit first)."""
    bm = _MODELS[name]().build()
    cls: type = RecordingEmitter
    mode = emit_mode(
        bm, split=False,
        supports_partitions=getattr(cls, "supports_partitions", True),
    )
    assert mode not in bm._tag_plans
    with _mint_log() as log:
        stream = _oracle_stream(bm, cls)
    plan = bm._tag_plans[mode]
    seed, seeded, planned, mints, minted = _read_log(name, log, plan)
    return Case(bm, tuple(stream), plan, seed, seeded, planned, mints,
                minted)


def _owner(case: Case, row: Row) -> str | None:
    """The family a tapped row belongs to (module docstring)."""
    kind = VERB_KIND.get(_verb(row[0]))
    if kind is None:
        return None
    key = (kind, row[1])
    if key in case.seeded and _verb(row[0]) != "element":
        return None
    if key in case.mints:
        return case.mints[key]
    if (kind == "element" and not FAMILY_PLANS["elements"].MIGRATED
            and row[1] <= case.seed.get("element", 0)):
        return "elements"      # a FEM id, inside the reserved range
    return PLANNED


def _per_rank_once(case: Case, rows: Counter[Row]) -> Counter[Row]:
    """``rows`` with each :data:`PER_RANK_KINDS` row counted once, on a
    partitioned deck only: a flat deck writes each region once, so a
    region written twice there is caught."""
    if not case.plan.mode.partitioned:
        return rows
    for row in rows:
        if VERB_KIND[row[0]] in PER_RANK_KINDS:
            rows[row] = 1
    return rows


def _rows(case: Case, owner: str, kinds: frozenset[str]) -> Counter[Row]:
    """The tapped rows of ``owner`` in ``kinds``, as ``(verb, tag)``."""
    return _per_rank_once(case, Counter(
        (_verb(r[0]), r[1]) for r in case.stream
        if _owner(case, r) == owner and VERB_KIND[_verb(r[0])] in kinds
    ))


def _in_kinds(rows: Any, kinds: frozenset[str]) -> Counter[Row]:
    """Planned ``rows`` in ``kinds``, as ``(verb, tag)``."""
    return Counter(
        (_verb(r[0]), r[1]) for r in rows if VERB_KIND[_verb(r[0])] in kinds)


def _family_mismatch(
    case: Case, family: str, planned_rows: Any,
) -> str | None:
    """Why ``planned_rows`` is not ``family``'s rows of ``case``, or None."""
    kinds = FAMILY_PLANS[family].KINDS
    still_minted = _rows(case, family, kinds)
    if still_minted:
        return (f"{family} is MIGRATED, but the emit still mints its rows "
                f"{sorted(still_minted.elements())}")
    others: Counter[Row] = Counter()
    for other in FAMILIES:
        cls = FAMILY_PLANS[other]
        if other != family and cls.MIGRATED and cls.KINDS & kinds:
            others += _in_kinds(case.plan.family(other).stream(), kinds)
    pool = _rows(case, PLANNED, kinds)
    if others - pool:
        return ("other families plan rows the emit does not write: "
                f"{sorted((others - pool).elements())}")
    expected = pool - others
    planned = _in_kinds(planned_rows, kinds)
    if planned != expected:
        return (f"[{case.plan.mode}] the {family} plan and the emit "
                f"disagree: planned only "
                f"{sorted((planned - expected).elements())}, emitted only "
                f"{sorted((expected - planned).elements())}")
    return None


def _family_param(name: str, family: str) -> Any:
    marks = []
    if not FAMILY_PLANS[family].MIGRATED:
        marks.append(pytest.mark.xfail(
            raises=NotImplementedError, strict=True,
            reason=f"{family}: tags still minted at emit time",
        ))
    return pytest.param(name, family, marks=marks, id=f"{name}-{family}")


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, family",
    [_family_param(n, f) for n in CASES for f in FAMILIES],
)
def test_plan_equals_tapped_stream(name: str, family: str) -> None:
    case = _case(name)
    planned = case.plan.family(family).stream()
    problem = _family_mismatch(case, family, planned)
    assert problem is None, f"{name}: {problem}"


def test_short_or_empty_element_plan_fails_the_oracle() -> None:
    """The element comparison is not vacuous: a dropped row is caught.

    Every case with elements must reject the element plan with its last
    row dropped, and with every row dropped, while accepting the real one.
    """
    checked = 0
    for name in CASES:
        case = _case(name)
        rows = case.plan.elements.stream()
        if not rows:
            continue
        checked += 1
        assert _family_mismatch(case, "elements", rows) is None, name
        assert _family_mismatch(case, "elements", rows[:-1]), name
        assert _family_mismatch(case, "elements", ()), name
    assert checked >= 10


def test_short_or_empty_transform_plan_fails_the_oracle() -> None:
    """The transform comparison is not vacuous: a dropped row is caught.

    Every case whose orientation fan-out plans a tag must reject the
    transform plan with its last row dropped, and with every row
    dropped, while accepting the real one.
    """
    checked = 0
    for name in CASES:
        case = _case(name)
        rows = case.plan.transforms.stream()
        if not rows:
            continue
        checked += 1
        assert _family_mismatch(case, "transforms", rows) is None, name
        assert _family_mismatch(case, "transforms", rows[:-1]), name
        assert _family_mismatch(case, "transforms", ()), name
    assert checked >= 4      # the arch fixture, in each of its four modes


# ---------------------------------------------------------------------------
# The transform family (K1-3d S3a)
# ---------------------------------------------------------------------------


def _arch_transform_inputs(case: Case) -> tuple[list[Any], list[Any]]:
    from apeGmsh.opensees._internal.build import topological_order
    from apeGmsh.opensees._internal.types import Element, GeomTransf

    ordered = topological_order(case.bm.primitives)
    return ([p for p in ordered if isinstance(p, GeomTransf)],
            [p for p in ordered if isinstance(p, Element)])


def test_geomtransf_is_frozen_in_the_emit_fork() -> None:
    """A stray ``geomTransf`` mint at emit time raises where it is."""
    plan = _case("arch_with_orientation_fan_out/flat").plan
    assert "geomTransf" in plan.frozen_kinds
    tags = plan.emit_allocator()
    with pytest.raises(TagLawError, match="planned and frozen"):
        tags.allocate("geomTransf")
    with pytest.raises(TagLawError):
        tags.allocate_block("geomTransf", 1)


def test_emit_transform_specs_is_two_way() -> None:
    """Fork: read the plan. Plain allocator: plan through the same loop.

    The plain-allocator path (a direct caller, until K1-3d S6) writes the
    same lines and overrides as the planned path; any other fork raises.
    """
    from apeGmsh.opensees._internal.build import emit_transform_specs

    case = _case("arch_with_orientation_fan_out/flat")
    bm, plan = case.bm, case.plan
    transforms, elements = _arch_transform_inputs(case)

    def run(tags: TagAllocator) -> tuple[list[Row], Any]:
        em = ts.tapped(RecordingEmitter)()
        em.tap = []
        overrides = emit_transform_specs(
            transforms, elements, em, bm.fem, tags, bm.tag_for, ndm=bm.ndm)
        return list(em.tap), overrides

    planned_rows, planned_over = run(plan.emit_allocator())
    plain_rows, plain_over = run(_seeded_like_the_planner(bm))
    assert planned_rows == plain_rows and planned_over == plain_over
    assert planned_over == plan.transforms.fanout_for(
        transforms, bm.fem).overrides
    assert planned_over      # the arch fans out past the spec's own tag

    for other in (plan.allocator.fork(), plan.allocator,
                  plan.allocator.fork({"geomTransf"}, origin=object())):
        with pytest.raises(TagLawError, match="carries no tag plan"):
            run(other)


def test_emit_transform_specs_refuses_a_fork_of_another_models_plan() -> None:
    """A real foreign plan's fork is refused as such, not as a miscount.

    The same arch recipe built twice gives two models with their own FEM
    snapshots. Model B's emit handed model A's emit allocator must say
    the plan is another model's, before any count or order check.
    """
    from apeGmsh.opensees._internal.build import emit_transform_specs

    case = _case("arch_with_orientation_fan_out/flat")
    other = _MODELS["arch_with_orientation_fan_out/flat"]().build()
    assert other.fem is not case.bm.fem
    o_transforms, o_elements = _arch_transform_inputs(_Inputs(other))
    em = RecordingEmitter()
    with pytest.raises(TagLawError, match="another model's tag plan"):
        emit_transform_specs(
            o_transforms, o_elements, em, other.fem,
            case.plan.emit_allocator(), other.tag_for, ndm=other.ndm)
    assert not em.calls


class _Inputs(NamedTuple):
    """A model wrapped like a :class:`Case`, for the input helpers."""

    bm: Any


def _seeded_like_the_planner(bm: Any) -> TagAllocator:
    """A plain allocator seeded with ``bm``'s primitives, as a direct
    caller of an emit helper seeds one."""
    from apeGmsh.opensees.apesees import _kind_of

    tags = TagAllocator()
    for prim in bm.primitives:
        tags.allocate_for(prim, _kind_of(prim))
    return tags


def test_emit_refuses_a_transform_plan_for_other_specs() -> None:
    case = _case("arch_with_orientation_fan_out/flat")
    transforms, _ = _arch_transform_inputs(case)
    assert transforms
    sub = case.plan.transforms
    fem = case.bm.fem
    assert sub.fanout_for(transforms, fem) is sub.fanout
    for wrong in ([], [*transforms, transforms[0]], transforms[:-1]):
        with pytest.raises(TagLawError, match="transform plan holds"):
            sub.fanout_for(wrong, fem)
    with pytest.raises(TagLawError, match="another model's tag plan"):
        sub.fanout_for(transforms, ts.synthesised_elements_fem())


def test_transform_plan_derives_its_rows_from_its_fanout() -> None:
    from apeGmsh.opensees._internal.tag_plan import TransformTagPlan

    with pytest.raises(TagLawError, match="fanout, not rows"):
        TransformTagPlan(rows=(("geomTransf", 2),))
    with pytest.raises(TagLawError, match="no fan-out"):
        TransformTagPlan().stream()


# ---------------------------------------------------------------------------
# The contact family (K1-3d S3b)
# ---------------------------------------------------------------------------


_CONTACT_CASES = (
    "synthesised_elements/flat", "synthesised_elements_fem_ids/flat",
    *_CONTACT_MODELS,
)


def _contact_rows(stream: Any) -> list[Row]:
    return [(_verb(k), t) for k, t in stream
            if _verb(k) in ("contact_surface", "contact", "contact_plane")]


def test_short_or_empty_contact_plan_fails_the_oracle() -> None:
    """The contact comparison is not vacuous: a dropped row is caught.

    Every case with contacts (flat, and partitioned over 2 and 4 ranks)
    must reject the contact plan with its last row dropped, and with
    every row dropped, while accepting the real one.
    """
    checked = 0
    for name in CASES:
        case = _case(name)
        rows = case.plan.contacts.stream()
        if not rows:
            continue
        checked += 1
        assert _family_mismatch(case, "contacts", rows) is None, name
        assert _family_mismatch(case, "contacts", rows[:-1]), name
        assert _family_mismatch(case, "contacts", ()), name
    assert checked == len(_CONTACT_CASES)


@pytest.mark.parametrize("name", _CONTACT_CASES)
def test_contact_rows_are_written_in_plan_order(name: str) -> None:
    """The emit writes the planned contact rows in the plan's own order."""
    case = _case(name)
    assert _contact_rows(case.stream) == list(case.plan.contacts.stream())


@pytest.mark.parametrize("n_ranks", [2, 4])
def test_partitioned_contacts_number_rank_by_rank(n_ranks: int) -> None:
    """The partitioned plan numbers contacts rank by rank, as the deck did.

    Rank ``r`` owns ``c{r}`` and ``p{r}``, declared in reverse rank
    order. The partitioned deck writes each rank's block in turn, so
    rank ``r``'s contact takes surfaces ``3r+1, 3r+2`` and contact
    ``2r+1``, and its plane surface ``3r+3`` and contact ``2r+2``. The
    flat deck numbers every contact, then every plane, in declaration
    order. Each owner declares its contact's slave nodes as ghosts.
    """
    case = _case(f"contact_ranks_{n_ranks}/partitioned")
    lines = case.plan.contacts.contacts
    assert lines is not None
    got = {line.record.name: (line.tags, line.owner_rank)
           for line in lines.lines}
    want: dict[str, Any] = {}
    for r in range(n_ranks):
        want[f"c{r}"] = ((3 * r + 1, 3 * r + 2, 2 * r + 1), r)
        want[f"p{r}"] = ((3 * r + 3, 2 * r + 2), r)
    assert got == want
    for line in lines.lines:
        b = 100 * (line.owner_rank + 1)
        ghosts = (b + 5, b + 6) if line.kind == "contact" else ()
        assert tuple(sorted(line.ghost_node_ids)) == ghosts

    if n_ranks == 4:
        flat = _case("contact_ranks_4/flat").plan.contacts.contacts
        assert flat is not None
        assert [(ln.record.name, ln.tags) for ln in flat.lines] == [
            ("c3", (1, 2, 1)), ("c2", (3, 4, 2)), ("c1", (5, 6, 3)),
            ("c0", (7, 8, 4)), ("p3", (9, 5)), ("p2", (10, 6)),
            ("p1", (11, 7)), ("p0", (12, 8)),
        ]
        assert all(ln.owner_rank is None for ln in flat.lines)


def test_contact_kinds_are_frozen_on_every_emit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every emit path's allocator refuses a ``contactSurface`` or
    ``contact`` mint: flat, staged, and partitioned over 1, 2 and
    4 ranks."""
    from apeGmsh.opensees.apesees import BuiltModel

    seen: list[tuple[str, TagAllocator]] = []

    def spy(path: str) -> Callable[..., Any]:
        orig = getattr(BuiltModel, path)
        sig = inspect.signature(orig)

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append((path, sig.bind(self, *args, **kwargs)
                         .arguments["tags"]))
            return orig(self, *args, **kwargs)
        return wrapper

    for path in _PATHS:
        monkeypatch.setattr(BuiltModel, path, spy(path))
    names = {*_first_case_per_mode().values(), *_CONTACT_MODELS}
    reached: set[str] = set()
    for name in sorted(names):
        ts.emit_stream(_MODELS[name]().build(), RecordingEmitter)
    for path, tags in seen:
        reached.add(path)
        assert {"contactSurface", "contact"} <= tags.frozen_kinds, path
        for kind in ("contactSurface", "contact"):
            with pytest.raises(TagLawError, match="planned and frozen"):
                tags.allocate(kind)
            with pytest.raises(TagLawError):
                tags.allocate_block(kind, 1)
    assert reached == set(_PATHS)


@pytest.mark.parametrize("name", _CONTACT_CASES)
def test_a_stray_contact_mint_raises(
    name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: an emit that mints a contact tag raises where it mints.

    The flat path is mutated back to minting their contact
    tags (``_planned_contact_lines`` plans from the emit allocator); the
    partitioned path's writer is mutated to mint one ``contact`` tag per
    line, over 2 and 4 ranks. Each emit must raise, not write a deck.
    """
    import apeGmsh.opensees.apesees as apesees_mod
    from apeGmsh.opensees._internal import build
    from apeGmsh.opensees.apesees import BuiltModel

    def minting_lines(fem: Any, tags: TagAllocator, kind: str) -> Any:
        entries = [(kind, r, None, ()) for r in build.contact_records(fem, kind)]
        return build.plan_contacts(fem, entries, tags).lines

    monkeypatch.setattr(build, "_planned_contact_lines", minting_lines)

    fork: list[TagAllocator] = []
    orig_partitioned = BuiltModel._emit_partitioned

    def spy(self: Any, **kwargs: Any) -> Any:
        fork.append(kwargs["tags"])
        return orig_partitioned(self, **kwargs)

    orig_writer = apesees_mod.write_planned_contact

    def minting_writer(emitter: Any, line: Any, *, ndm: int) -> None:
        fork[-1].allocate("contact")
        orig_writer(emitter, line, ndm=ndm)

    monkeypatch.setattr(BuiltModel, "_emit_partitioned", spy)
    monkeypatch.setattr(apesees_mod, "write_planned_contact", minting_writer)

    with pytest.raises(TagLawError, match="planned and frozen"):
        ts.emit_stream(_MODELS[name]().build(), RecordingEmitter)
    assert bool(fork) == name.endswith("/partitioned")


@pytest.mark.parametrize("name", _CONTACT_CASES)
@pytest.mark.parametrize("cut", ["short", "empty"])
def test_emit_refuses_a_short_or_empty_contact_plan(
    name: str, cut: str,
) -> None:
    """Mutation: the emit refuses a contact plan that dropped interactions.

    The memoised plan's contact lines are cut (the last one dropped, or
    all of them); the next emit, flat or partitioned, must raise rather
    than write a deck that lacks them.
    """
    import dataclasses

    from apeGmsh.opensees._internal.tag_plan import ContactTagPlan

    bm = _MODELS[name]().build()
    mode = emit_mode(bm, split=False, supports_partitions=True)
    plan = bm._tag_plan(mode)
    contacts = plan.contacts.contacts
    assert contacts is not None and contacts.lines
    lines = contacts.lines[:-1] if cut == "short" else ()
    bm._tag_plans[mode] = dataclasses.replace(
        plan, contacts=ContactTagPlan(
            contacts=dataclasses.replace(contacts, lines=lines)))
    with pytest.raises(TagLawError, match="contact plan holds"):
        ts.emit_stream(bm, RecordingEmitter)


@pytest.mark.parametrize("name", [
    "contact_ranks_4/flat", "contact_ranks_2/partitioned",
    "contact_ranks_4/partitioned",
])
def test_emit_refuses_a_contact_plan_with_a_swapped_record(name: str) -> None:
    """Mutation: same count, other records. ``check_covers`` compares
    record identities, not only how many there are.

    The memoised plan's ``c1`` line is given ``c0``'s record, so the plan
    still holds one line per contact but plans ``c0`` twice and ``c1``
    never. The next emit must raise rather than write ``c0`` twice.
    """
    import dataclasses

    from apeGmsh.opensees._internal.tag_plan import ContactTagPlan

    bm = _MODELS[name]().build()
    mode = emit_mode(bm, split=False, supports_partitions=True)
    plan = bm._tag_plan(mode)
    contacts = plan.contacts.contacts
    assert contacts is not None
    by_name = {ln.record.name: ln.record for ln in contacts.lines}
    lines = tuple(
        ln._replace(record=by_name["c0"]) if ln.record.name == "c1" else ln
        for ln in contacts.lines
    )
    assert len(lines) == len(contacts.lines)
    bm._tag_plans[mode] = dataclasses.replace(
        plan, contacts=ContactTagPlan(
            contacts=dataclasses.replace(contacts, lines=lines)))
    with pytest.raises(TagLawError, match="contact plan holds"):
        ts.emit_stream(bm, RecordingEmitter)


def test_emit_contacts_is_two_way() -> None:
    """Fork: read the plan. Plain allocator: plan through the same loop.

    The plain-allocator path (a direct caller, until K1-3d S6) writes the
    same rows as the planned path. A plain fork, the frozen planner
    allocator and a fork for another origin carry no plan; a fork of a
    real foreign plan is refused as another model's.
    """
    from apeGmsh.opensees._internal.build import (
        emit_contact_planes,
        emit_contacts,
    )

    case = _case("synthesised_elements/flat")
    bm, plan = case.bm, case.plan

    def run(tags: TagAllocator, fem: Any = None) -> list[Row]:
        em = ts.tapped(RecordingEmitter)()
        em.tap = []
        emit_contacts(em, bm.fem if fem is None else fem, tags, ndm=bm.ndm)
        emit_contact_planes(em, bm.fem if fem is None else fem, tags)
        return list(em.tap)

    planned = run(plan.emit_allocator())
    assert planned == run(_seeded_like_the_planner(bm))
    assert planned == list(plan.contacts.stream())
    assert planned       # one contact and one plane

    for other in (plan.allocator.fork(), plan.allocator,
                  plan.allocator.fork({"contact"}, origin=object())):
        with pytest.raises(TagLawError, match="carries no tag plan"):
            run(other)
    foreign = _case("contact_ranks_4/flat").plan
    with pytest.raises(TagLawError, match="another model's tag plan"):
        run(foreign.emit_allocator())
    with pytest.raises(TagLawError, match="another model's tag plan"):
        run(plan.emit_allocator(), fem=foreign.contacts.contacts.fem)


def test_emit_contacts_refuses_a_plan_for_other_records() -> None:
    from apeGmsh.opensees._internal.build import ContactPlan

    flat = _case("contact_ranks_4/flat").plan.contacts.for_fem(
        _case("contact_ranks_4/flat").bm.fem)
    part = _case("contact_ranks_4/partitioned").plan.contacts.contacts
    assert isinstance(part, ContactPlan)
    records = [ln.record for ln in flat.lines if ln.kind == "contact"]
    assert flat.lines_for("contact", records)
    for wrong in ([], records[:-1], [*records, records[0]], records[::-1]):
        with pytest.raises(TagLawError, match="contact plan holds"):
            flat.lines_for("contact", wrong)
    # A partitioned plan numbers rank by rank: not the flat walk's order.
    part_records = [r for r in part.fem.elements.contacts]
    with pytest.raises(TagLawError, match="contact plan holds"):
        part.lines_for("contact", part_records)


def test_contact_plan_derives_its_rows_from_its_lines() -> None:
    from apeGmsh.opensees._internal.tag_plan import ContactTagPlan

    with pytest.raises(TagLawError, match="contacts, not rows"):
        ContactTagPlan(rows=(("contact", 1),))
    with pytest.raises(TagLawError, match="no contact plan"):
        ContactTagPlan().stream()
    with pytest.raises(TagLawError, match="no contact plan"):
        ContactTagPlan().for_fem(object())


def test_partitioned_routing_runs_once_and_warns_once_per_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The routing runs once per plan; its warning fires once per emit.

    Before the plan, every partitioned emit re-ran
    ``_plan_partitioned_contacts`` and warned once (the F3 partial
    element-ownership warning). Now the plan runs it once, and each emit
    repeats the warning it noted: two emits, one routing, two warnings.
    """
    import warnings

    from apeGmsh.opensees.apesees import BuiltModel

    from tests.opensees.integration.test_contact_partitioned_review_fixes import (
        _cut_master_stub,
        _stub_ops,
    )

    calls: list[int] = []
    orig = BuiltModel._plan_partitioned_contacts

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(BuiltModel, "_plan_partitioned_contacts", spy)
    bm = _stub_ops(_cut_master_stub(kn=1.0e6, second_facet="unowned")).build()
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bm.emit(RecordingEmitter())
        hits = [w for w in caught
                if "absent from every PartitionRecord" in str(w.message)]
        assert len(hits) == 1
        assert hits[0].category is UserWarning
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# The region family (K1-3d S4)
# ---------------------------------------------------------------------------


#: A case of every region site: named regions whose rank order differs
#: from their declaration order (flat, partitioned), and every global and
#: stage-bound site with stage-claimed recorders (staged, partitioned).
_REGION_CASES = (
    "two_rank_regions/flat", "two_rank_regions/partitioned",
    "stage_claimed_regions/staged", "stage_claimed_regions/staged_partitioned",
)


def test_short_or_empty_region_plan_fails_the_oracle() -> None:
    """The region comparison is not vacuous: a dropped row is caught.

    Every case that writes a region must reject the region plan with its
    last row dropped, and with every row dropped, while accepting the real
    one. A partitioned deck writes a region once per holder rank, so the
    oracle compares its region rows as distinct rows (``PER_RANK_KINDS``).
    """
    checked = 0
    for name in CASES:
        case = _case(name)
        rows = case.plan.regions.stream()
        if not rows:
            continue
        checked += 1
        assert _family_mismatch(case, "regions", rows) is None, name
        assert _family_mismatch(case, "regions", rows[:-1]), name
        assert _family_mismatch(case, "regions", ()), name
    assert checked >= 20      # every golden recording cell, and the above


@pytest.mark.parametrize(("name", "want"), [
    # Flat: named, damping, then the global recorder pass; then the stage.
    ("stage_claimed_regions/staged", [
        ("named", "east"), ("named", "west"), ("rayleigh", (0, 0)),
        ("damping", (0, 0)), ("recorder", "filter"),
        ("named", "p_east"), ("named", "p_west"), ("rayleigh", (0, 0)),
        ("damping", (0, 0)), ("recorder", "filter"), ("recorder", "filter"),
        ("recorder", "energy"),
    ]),
    # Partitioned: the global recorder pass first (its regions are written
    # in every rank block), named regions on their first holder rank
    # (rank 0 holds node 2, rank 1 node 4); the stage-claimed recorders
    # only in their stage (#1446).
    ("stage_claimed_regions/staged_partitioned", [
        ("recorder", "filter"), ("named", "west"), ("named", "east"),
        ("rayleigh", (0, 0)), ("damping", (0, 0)),
        ("named", "p_west"), ("named", "p_east"), ("rayleigh", (0, 0)),
        ("damping", (0, 0)), ("recorder", "filter"), ("recorder", "filter"),
        ("recorder", "energy"),
    ]),
])
def test_stage_claimed_regions_plan(name: str, want: list[Any]) -> None:
    """Closed form: the #1446 fixture's twelve regions, in mint order."""
    regions = _case(name).plan.regions.regions
    assert regions is not None
    assert [(r.site[0], r.key) for r in regions] == want
    assert [r.tag for r in regions] == list(range(1, len(want) + 1))


def _path_allocators(
    monkeypatch: pytest.MonkeyPatch, names: Any,
) -> list[tuple[str, TagAllocator]]:
    """Emit ``names`` and return each emit path's allocator."""
    from apeGmsh.opensees.apesees import BuiltModel

    seen: list[tuple[str, TagAllocator]] = []

    def spy(path: str) -> Callable[..., Any]:
        orig = getattr(BuiltModel, path)
        sig = inspect.signature(orig)

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append((path, sig.bind(self, *args, **kwargs)
                         .arguments["tags"]))
            return orig(self, *args, **kwargs)
        return wrapper

    for path in _PATHS:
        monkeypatch.setattr(BuiltModel, path, spy(path))
    for name in sorted(names):
        _emit_case(name)
    return seen


def _emit_case(name: str, bm: Any = None) -> list[Row]:
    return ts.emit_stream(bm or _MODELS[name]().build(), RecordingEmitter)


def test_region_is_frozen_on_every_emit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every emit path's allocator refuses a ``region`` mint: flat,
    staged, and partitioned (staged or not)."""
    names = {*_first_case_per_mode().values(), *_REGION_CASES}
    reached: set[str] = set()
    for path, tags in _path_allocators(monkeypatch, names):
        reached.add(path)
        assert "region" in tags.frozen_kinds, path
        with pytest.raises(TagLawError, match="planned and frozen"):
            tags.allocate("region")
        with pytest.raises(TagLawError):
            tags.allocate_block("region", 1)
    assert reached == set(_PATHS)


def _minting_recorder_tags(self: Any, fem: Any, tags: TagAllocator) -> Any:
    from apeGmsh.opensees._internal.tag_plan import plan_regions

    rows = plan_regions([(("recorder", id(self)), self.region_keys())], tags)
    return {row.key: row.tag for row in rows}


def _minting_damping_tags(
    self: Any, kind: str, stage: Any, recs: Any, tags: TagAllocator,
) -> Any:
    from apeGmsh.opensees._internal.tag_plan import plan_regions

    site = (kind, None if stage is None else id(stage))
    rows = plan_regions([(site, self._damping_region_keys(recs))], tags)
    return {row.key: row.tag for row in rows}


#: Each region writer mutated back to minting, with the cases that reach
#: it: the recorder writer on every emit path, the named-region and the
#: damping writers on every path that has them.
_STRAY_MINTS = {
    "recorder": ("two_column_frame/flat", "two_column_frame/partitioned",
                 "two_column_frame/staged",
                 "two_column_frame/staged_partitioned"),
    "named": _REGION_CASES,
    "damping": ("stage_claimed_regions/staged",
                "stage_claimed_regions/staged_partitioned"),
}


@pytest.mark.parametrize(("writer", "name"), [
    (w, n) for w, names in _STRAY_MINTS.items() for n in names])
def test_a_stray_region_mint_raises(
    writer: str, name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: a region writer that mints raises where it mints.

    Each writer (a filtered recorder's, the named regions', the damping
    regions') is mutated to mint its tags from the emit allocator instead
    of reading the plan. Every emit path that reaches it must raise, not
    write a deck.
    """
    from apeGmsh.opensees.apesees import BuiltModel
    from apeGmsh.opensees.recorder import FilterableRecorder

    if writer == "recorder":
        monkeypatch.setattr(
            FilterableRecorder, "planned_region_tags", _minting_recorder_tags)
    elif writer == "damping":
        monkeypatch.setattr(
            BuiltModel, "_planned_damping_region_tags", _minting_damping_tags)
    else:
        orig = BuiltModel._planned_named_regions

        def minting_named(self: Any, tags: TagAllocator, stage: Any) -> Any:
            return tuple(r._replace(tag=tags.allocate("region"))
                         for r in orig(self, tags, stage))
        monkeypatch.setattr(BuiltModel, "_planned_named_regions", minting_named)
    with pytest.raises(TagLawError, match="planned and frozen"):
        _emit_case(name)


def _memoised_region_plan(name: str) -> tuple[Any, TagMode, TagPlan]:
    bm = _MODELS[name]().build()
    mode = emit_mode(bm, split=False, supports_partitions=True)
    return bm, mode, bm._tag_plan(mode)


def _with_regions(plan: TagPlan, rows: Any) -> TagPlan:
    import dataclasses

    from apeGmsh.opensees._internal.tag_plan import RegionTagPlan

    sub = plan.regions
    return dataclasses.replace(plan, regions=RegionTagPlan(
        regions=tuple(rows), named=sub.named, partitioned=sub.partitioned,
        fem=sub.fem))


@pytest.mark.parametrize("name", _REGION_CASES)
@pytest.mark.parametrize("cut", ["short", "empty"])
def test_emit_refuses_a_short_or_empty_region_plan(name: str, cut: str) -> None:
    """Mutation: the emit refuses a region plan that dropped regions.

    The memoised plan's regions are cut (the last one dropped, or all of
    them); the next emit must raise rather than write a deck that lacks
    them.
    """
    bm, mode, plan = _memoised_region_plan(name)
    rows = plan.regions.regions
    assert rows
    bm._tag_plans[mode] = _with_regions(
        plan, rows[:-1] if cut == "short" else ())
    with pytest.raises(TagLawError, match="not made for this emit"):
        ts.emit_stream(bm, RecordingEmitter)


@pytest.mark.parametrize(("name", "site_kind"), [
    ("two_rank_regions/flat", "named"),
    ("two_rank_regions/partitioned", "named"),
    ("stage_claimed_regions/staged", "named"),
    ("stage_claimed_regions/staged_partitioned", "named"),
    ("stage_claimed_regions/staged", "recorder"),
    ("stage_claimed_regions/staged_partitioned", "recorder"),
])
def test_emit_refuses_a_region_plan_with_two_regions_swapped(
    name: str, site_kind: str,
) -> None:
    """Mutation: two regions of one site trade keys; the emit refuses it.

    Each keeps its slot and tag, so the swapped plan holds the very rows
    of the real one, same count, same kinds, same tags: a count, or a
    comparison of rows alone, would pass it and write the two regions'
    tags swapped. The emit must raise instead.
    """
    bm, mode, plan = _memoised_region_plan(name)
    rows = list(plan.regions.regions or ())
    by_site: dict[Any, list[int]] = {}
    for i, row in enumerate(rows):
        if row.site[0] == site_kind:
            by_site.setdefault(row.site, []).append(i)
    i, j = next(ix for ix in by_site.values() if len(ix) >= 2)[:2]
    rows[i], rows[j] = (rows[i]._replace(key=rows[j].key),
                        rows[j]._replace(key=rows[i].key))
    swapped = _with_regions(plan, rows)
    assert Counter(swapped.regions.stream()) == Counter(plan.regions.stream())
    bm._tag_plans[mode] = swapped
    with pytest.raises(TagLawError, match="not made for this emit"):
        ts.emit_stream(bm, RecordingEmitter)


def test_region_plan_rows_keep_their_mint_order() -> None:
    """Two rows moved out of mint order (their tags no longer rise), a
    region planned twice, or rows passed as ``rows``, are refused."""
    from apeGmsh.opensees._internal.tag_plan import RegionTagPlan

    plan = _case("stage_claimed_regions/staged").plan
    rows = list(plan.regions.regions or ())
    moved = [rows[1], rows[0], *rows[2:]]
    with pytest.raises(TagLawError, match="only rise"):
        _with_regions(plan, moved)
    twice = [*rows, rows[0]._replace(tag=rows[-1].tag + 1)]
    with pytest.raises(TagLawError, match="planned twice"):
        _with_regions(plan, twice)
    with pytest.raises(TagLawError, match="regions, not rows"):
        RegionTagPlan(rows=(("region", 1),))
    with pytest.raises(TagLawError, match="carries no regions"):
        RegionTagPlan().stream()


def test_planned_region_tags_is_two_way() -> None:
    """Fork: read the plan. Plain allocator: plan through the same loop.

    A plain fork, the frozen planner allocator and a fork for another
    origin carry no plan; a fork of a real foreign plan is refused as
    another model's.
    """
    from apeGmsh.opensees.recorder import Ladruno

    case = _case("stage_claimed_regions/staged")
    bm, plan = case.bm, case.plan
    (spec,) = [p for p in bm.primitives if isinstance(p, Ladruno)]
    assert spec.region_keys() == ("filter", "energy")
    assert spec.planned_region_tags(bm.fem, plan.emit_allocator()) == {
        "filter": 11, "energy": 12}
    assert spec.planned_region_tags(bm.fem, TagAllocator()) == {
        "filter": 1, "energy": 2}
    for other in (plan.allocator.fork(), plan.allocator,
                  plan.allocator.fork({"region"}, origin=object())):
        with pytest.raises(TagLawError, match="carries no tag plan"):
            spec.planned_region_tags(bm.fem, other)
    foreign = _case("stage_claimed_regions/staged_partitioned").plan
    with pytest.raises(TagLawError, match="another model's tag plan"):
        spec.planned_region_tags(bm.fem, foreign.emit_allocator())


def test_mint_site_names_are_unique() -> None:
    """Each ``_MINT_SITES`` name defines one function in the bridge.

    The mint log attributes a mint by the bare name of the function that
    made it. A name defined twice (a generic one, as a recorder's
    ``materialize`` was before S4) would credit one family's mints to the
    other without an error.
    """
    import ast
    from pathlib import Path

    import apeGmsh.opensees as bridge

    defs: Counter[str] = Counter()
    for path in Path(bridge.__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defs.update(
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in _MINT_SITES)
    assert set(defs) == set(_MINT_SITES), "a mint site no longer exists"
    assert {n: c for n, c in defs.items() if c != 1} == {}


def test_a_duplicated_flat_region_line_fails_the_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: a stage writes its named regions twice on a flat deck.

    A flat deck writes each region once, so the oracle compares its region
    rows as a multiset: the duplicated ``region`` lines leave the emitted
    rows one row longer per region than the plan, and the comparison
    fails. Collapsing region rows on every deck (only a partitioned one
    writes a region once per holder rank) would hide it.
    """
    from apeGmsh.opensees.apesees import BuiltModel

    name = "stage_claimed_regions/staged"
    real = _case(name)
    assert _family_mismatch(real, "regions", real.plan.regions.stream()) is None
    orig = BuiltModel._emit_stage_regions

    def twice(self: Any, *args: Any, **kwargs: Any) -> None:
        orig(self, *args, **kwargs)
        orig(self, *args, **kwargs)

    monkeypatch.setattr(BuiltModel, "_emit_stage_regions", twice)
    case = _build_case(name)
    problem = _family_mismatch(case, "regions", case.plan.regions.stream())
    assert problem is not None and "emitted only [('region', 6)" in problem


# ---------------------------------------------------------------------------
# The parameter family (K1-3d S5)
# ---------------------------------------------------------------------------


#: Every case whose emit writes a parameter tag: the K1-3 H5 fixtures (a
#: flat initial stress, a staged one, a staged absorbing flip) and every
#: parameter site flat, staged, and partitioned over 2 and 4 ranks.
_PARAM_CASES = (
    "initial_stress_frame/flat", "two_stage_initial_stress/staged",
    "kitchen_sink_absorbing/staged", *_PARAM_MODELS,
)


def test_short_or_empty_parameter_plan_fails_the_oracle() -> None:
    """The parameter comparison is not vacuous: a dropped row is caught.

    Every case that writes a parameter tag (flat, staged, and partitioned
    over 2 and 4 ranks, staged or not) must reject the parameter plan with
    its last row dropped, and with every row dropped, while accepting the
    real one.
    """
    checked: set[str] = set()
    for name in CASES:
        case = _case(name)
        rows = case.plan.parameters.stream()
        if not rows:
            continue
        checked.add(name)
        assert _family_mismatch(case, "parameters", rows) is None, name
        assert _family_mismatch(case, "parameters", rows[:-1]), name
        assert _family_mismatch(case, "parameters", ()), name
    assert set(_PARAM_CASES) <= checked
    modes = {_case(n).plan.mode for n in _PARAM_CASES}
    assert modes == {TagMode(False, p, s) for p in (False, True)
                     for s in (False, True)}


def _param_site(line: Any) -> tuple[Any, ...]:
    """A planned parameter line as ``(verb, label, rank, tags)``; the label
    is the record's name, else its ``pg``, else its element ids."""
    rec = line.record
    label = getattr(rec, "name", None) or rec.pg or tuple(rec.elements)
    return line.verb, label, line.rank, line.tags


def _ramp(label: str, first: int) -> tuple[Any, ...]:
    return ("step_hook_ramp", label, None, (first, first + 1, first + 2))


def _flip(label: Any, rank: int | None, *tags: int) -> tuple[Any, ...]:
    return ("flip_element_stage", label, rank, tags)


def _update(label: str, rank: int | None, *tags: int) -> tuple[Any, ...]:
    return ("update_parameter", label, rank, tags)


@pytest.mark.parametrize(("name", "want"), [
    # Flat: the global ramp, then per stage its ramp, flips, updates.
    ("param_ranks_4/staged", [
        _ramp("g", 1), _ramp("s1", 4), _flip("Edge", None, 7),
        _update("E", None, 8), _update("A", None, 9),
        _ramp("s2", 10), _flip((10, 40), None, 13),
    ]),
    # Partitioned, unstaged: the global ramp, written outside every block.
    ("param_ranks_2/partitioned", [_ramp("g", 1)]),
    # Partitioned: each flip and update pass rank by rank; a rank that
    # owns none of a record's elements gives it no tag (rank 0 holds no
    # ``Edge`` bar; only the last rank holds ``E``'s bar).
    ("param_ranks_2/staged_partitioned", [
        _ramp("g", 1), _ramp("s1", 4),
        _flip("Edge", 0), _flip("Edge", 1, 7),
        _update("E", 0), _update("A", 0, 8),
        _update("E", 1, 9), _update("A", 1, 10),
        _ramp("s2", 11), _flip((10, 20), 0, 14), _flip((10, 20), 1, 15),
    ]),
    ("param_ranks_4/staged_partitioned", [
        _ramp("g", 1), _ramp("s1", 4),
        _flip("Edge", 0), _flip("Edge", 1, 7), _flip("Edge", 2, 8),
        _flip("Edge", 3, 9),
        _update("E", 0), _update("A", 0, 10), _update("E", 1),
        _update("A", 1, 11), _update("E", 2), _update("A", 2, 12),
        _update("E", 3, 13), _update("A", 3, 14),
        _ramp("s2", 15),
        _flip((10, 40), 0, 18), _flip((10, 40), 1), _flip((10, 40), 2),
        _flip((10, 40), 3, 19),
    ]),
])
def test_parameters_number_in_emit_order(name: str, want: list[Any]) -> None:
    """Closed form: every parameter site of the fixture, in mint order."""
    case = _case(name)
    assert [_param_site(ln) for ln in case.plan.parameters.planned()] == want
    # The deck writes each planned flip and update tag once, in plan order.
    written = [(k, t) for k, t in case.stream
               if k in ("flip_element_stage", "update_parameter",
                        "step_hook_ramp")]
    assert written == list(case.plan.parameters.stream())


@pytest.mark.parametrize("name", _PARAM_CASES)
def test_add_to_parameter_names_a_planned_ramp_tag(name: str) -> None:
    """An ``addToParameter`` references a tag its ramp declared.

    The oracle compares the ramp declarations; each ``addToParameter`` line
    names one of them, and every declared ramp is named by some element
    (each fixture's initial stresses cover elements on some rank).
    """
    case = _case(name)
    ramps = {t for v, t in case.plan.parameters.stream()
             if v == "step_hook_ramp"}
    named = {t for k, t in case.stream if k == "addToParameter"}
    assert named == ramps


def test_no_kind_is_minted_at_emit() -> None:
    """K1-3d S5 planned the last family: every emit fork freezes every
    family's kinds, and no case mints a tag at emit time."""
    kinds = frozenset().union(*(cls.KINDS for cls in FAMILY_PLANS.values()))
    assert all(cls.MIGRATED for cls in FAMILY_PLANS.values())
    for name in CASES:
        case = _case(name)
        assert case.plan.frozen_kinds == kinds, name
        assert case.minted == {}, name


def test_parameter_is_frozen_on_every_emit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every emit path's allocator refuses a ``parameter`` mint: flat,
    staged, and partitioned (staged or not) over 2 and 4 ranks."""
    names = {*_first_case_per_mode().values(), *_PARAM_CASES}
    reached: set[str] = set()
    for path, tags in _path_allocators(monkeypatch, names):
        reached.add(path)
        assert "parameter" in tags.frozen_kinds, path
        with pytest.raises(TagLawError, match="planned and frozen"):
            tags.allocate("parameter")
        with pytest.raises(TagLawError):
            tags.allocate_block("parameter", 1)
    assert reached == set(_PATHS)


@pytest.mark.parametrize(("verb", "name", "path"), [
    ("step_hook_ramp", "initial_stress_frame/flat", "_emit_flat"),
    ("step_hook_ramp", "param_ranks_2/partitioned", "_emit_partitioned"),
    ("step_hook_ramp", "two_stage_initial_stress/staged", "_emit_stages_flat"),
    ("flip_element_stage", "param_ranks_4/staged", "_emit_stages_flat"),
    ("update_parameter", "param_ranks_4/staged", "_emit_stages_flat"),
    ("flip_element_stage", "param_ranks_2/staged_partitioned",
     "_emit_stages_partitioned"),
    ("update_parameter", "param_ranks_2/staged_partitioned",
     "_emit_stages_partitioned"),
    ("flip_element_stage", "param_ranks_4/staged_partitioned",
     "_emit_stages_partitioned"),
    ("update_parameter", "param_ranks_4/staged_partitioned",
     "_emit_stages_partitioned"),
])
def test_a_stray_parameter_mint_raises(
    verb: str, name: str, path: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: a parameter writer that mints raises where it mints.

    The writer of ``verb`` is mutated back to minting its tags from the
    emit allocator, as it did before the plan. The emit must raise on the
    path named, not write a deck.
    """
    from apeGmsh.opensees._internal import build

    orig = build._planned_parameters

    def minting(sites: Any, tags: TagAllocator) -> Any:
        if sites and sites[0].verb == verb:
            return list(build.plan_parameters(sites, tags))
        return orig(sites, tags)

    monkeypatch.setattr(build, "_planned_parameters", minting)
    with pytest.raises(TagLawError, match="planned and frozen") as err:
        _emit_case(name)
    assert path in {entry.name for entry in err.traceback}


def _with_parameters(plan: TagPlan, lines: Any, owners: Any) -> TagPlan:
    import dataclasses

    from apeGmsh.opensees._internal.tag_plan import ParameterTagPlan

    return dataclasses.replace(plan, parameters=ParameterTagPlan(
        lines=tuple(lines), owners=tuple(owners)))


@pytest.mark.parametrize("name", _PARAM_CASES)
@pytest.mark.parametrize("cut", ["short", "empty"])
def test_emit_refuses_a_short_or_empty_parameter_plan(
    name: str, cut: str,
) -> None:
    """Mutation: the emit refuses a parameter plan that dropped sites.

    The memoised plan's sites are cut (the last one dropped, or all of
    them), its owners with them, so the plan is consistent; the next emit
    must raise rather than write a deck that lacks them.
    """
    bm, mode, plan = _memoised_region_plan(name)
    sub = plan.parameters
    keep = len(sub.planned()) - 1 if cut == "short" else 0
    bm._tag_plans[mode] = _with_parameters(
        plan, sub.planned()[:keep], sub.owners[:keep])
    with pytest.raises(TagLawError, match="parameter plan holds no"):
        ts.emit_stream(bm, RecordingEmitter)


@pytest.mark.parametrize("name", _PARAM_CASES)
def test_emit_refuses_a_parameter_plan_for_other_records(name: str) -> None:
    """Mutation: same count, same kinds, other records.

    The memoised plan is given the parameter plan of a second build of the
    same recipe: every site, verb and tag is the real plan's, but its
    records are another model's objects. The emit looks each record up by
    identity, so it must raise rather than write the other model's tags.
    """
    bm, mode, plan = _memoised_region_plan(name)
    other = _MODELS[name]().build()._tag_plan(mode).parameters
    assert other.stream() == plan.parameters.stream()
    bm._tag_plans[mode] = _with_parameters(plan, other.planned(), other.owners)
    with pytest.raises(TagLawError, match="parameter plan holds no"):
        ts.emit_stream(bm, RecordingEmitter)


@pytest.mark.parametrize("name", [
    "param_ranks_4/staged", "param_ranks_2/staged_partitioned",
    "param_ranks_4/staged_partitioned",
])
def test_parameter_plan_refuses_a_swapped_dropped_or_doubled_site(
    name: str,
) -> None:
    """The parameter plan covers the model's records exactly, by identity.

    Two sites of one verb at one rank trade records, each keeping its
    slot and tags: the swapped plan holds the very rows of the real one,
    same count, same kinds, same tags, so a count of sites would pass it
    and the emit would write the two records' tags swapped. Each swap,
    each dropped site and each doubled one is refused when the plan is
    made.
    """
    from apeGmsh.opensees._internal.tag_plan import ParameterTagPlan

    sub = _case(name).plan.parameters
    lines, owners = sub.planned(), sub.owners
    assert ParameterTagPlan(lines=lines, owners=owners).lines == lines
    pairs = [
        (i, j) for i, a in enumerate(lines) for j, b in enumerate(lines)
        if i < j and (a.verb, a.rank) == (b.verb, b.rank)
        and a.record is not b.record
    ]
    assert {lines[i].verb for i, _ in pairs} == {
        "step_hook_ramp", "flip_element_stage", "update_parameter"}
    for i, j in pairs:
        swapped = list(lines)
        swapped[i] = lines[i]._replace(record=lines[j].record)
        swapped[j] = lines[j]._replace(record=lines[i].record)
        assert len(swapped) == len(owners)
        with pytest.raises(TagLawError, match="does not cover"):
            ParameterTagPlan(lines=tuple(swapped), owners=owners)
    for i, line in enumerate(lines):
        with pytest.raises(TagLawError, match="does not cover"):
            ParameterTagPlan(lines=lines[:i] + lines[i + 1:], owners=owners)
        with pytest.raises(TagLawError, match="twice"):
            ParameterTagPlan(lines=(*lines, line), owners=(*owners, owners[i]))


def test_parameter_plan_keeps_its_mint_order_and_counts() -> None:
    """Sites out of mint order, a site with the wrong number of tags, an
    unknown verb, or rows passed as ``rows``, are refused."""
    from apeGmsh.opensees._internal.tag_plan import ParameterTagPlan

    sub = _case("param_ranks_4/staged").plan.parameters
    lines, owners = list(sub.planned()), list(sub.owners)
    with pytest.raises(TagLawError, match="only rise"):
        ParameterTagPlan(lines=(lines[1], lines[0], *lines[2:]),
                         owners=(owners[1], owners[0], *owners[2:]))
    for bad in (lines[0]._replace(tags=lines[0].tags[:2]),
                lines[2]._replace(tags=(7, 8)),
                lines[2]._replace(verb="parameter")):
        with pytest.raises(TagLawError, match="the verbs and their counts"):
            ParameterTagPlan(lines=(bad, *lines[1:]), owners=tuple(owners))
    with pytest.raises(TagLawError, match="lines, not rows"):
        ParameterTagPlan(rows=(("step_hook_ramp", 1),))
    with pytest.raises(TagLawError, match="no parameter plan"):
        ParameterTagPlan().stream()
    with pytest.raises(TagLawError, match="no parameter plan"):
        ParameterTagPlan()[(lines[0].record, None)]


def test_parameter_writers_are_two_way() -> None:
    """Fork: read the plan. Plain allocator: plan through the same loop.

    The three writers, handed a plain allocator (a direct caller, until
    K1-3d S6), write the same rows as when handed the emit allocator, and
    those rows are the plan's, flat and over 2 ranks. A plain fork, the
    frozen planner allocator and a fork for another origin carry no plan;
    a fork of another model's plan holds none of these records.
    """
    from apeGmsh.opensees._internal.build import (
        FemToOpsTagMap,
        build_element_partition_owner,
        emit_activate_absorbing,
        emit_initial_stress_global,
        emit_update_parameters,
        runtime_rank_from_partition_record,
    )

    for name in ("param_ranks_4/staged", "param_ranks_2/staged_partitioned"):
        bm, plan = _case(name).bm, _case(name).plan
        eid_to_tag = FemToOpsTagMap.from_plan(plan.elements.specs)
        owner = ranks = None
        if plan.mode.partitioned:
            owner = build_element_partition_owner(bm.fem)
            ranks = [runtime_rank_from_partition_record(p, i)
                     for i, p in enumerate(bm.fem.partitions)]

        def run(tags: TagAllocator) -> list[Row]:
            em = _ramp_tapped(RecordingEmitter)()
            em.tap = []
            emit_initial_stress_global(bm.initial_stress_records, em, tags)
            for stage in bm.stage_records:
                emit_initial_stress_global(
                    stage.initial_stress_records, em, tags)
                for emit, records in (
                    (emit_activate_absorbing,
                     stage.activate_absorbing_records),
                    (emit_update_parameters, stage.update_parameter_records),
                ):
                    for rank in ranks or [None]:
                        emit(records, em, bm.fem, eid_to_tag, tags,
                             element_owner=owner, partition_rank=rank)
            return list(em.tap)

        planned = run(plan.emit_allocator())
        assert planned == run(TagAllocator()), name
        assert planned == list(plan.parameters.stream()), name
        for other in (plan.allocator.fork(), plan.allocator,
                      plan.allocator.fork({"parameter"}, origin=object())):
            with pytest.raises(TagLawError, match="carries no tag plan"):
                run(other)
        foreign = _MODELS[name]().build()._tag_plan(plan.mode)
        with pytest.raises(TagLawError, match="parameter plan holds no"):
            run(foreign.emit_allocator())


@pytest.mark.parametrize(("verb", "match"), [
    ("activate_absorbing", "activate_absorbing: element id 999"),
    ("update_parameter", "update_parameter 'E': element id 999"),
])
def test_an_unknown_flip_element_raises_when_the_emit_plans(
    verb: str, match: str,
) -> None:
    """A flat staged flip or update naming an element no primitive emits
    raises its ``BridgeError`` when the emit plans its tags, before any
    line is written, with the message the writer gave before the plan."""
    from apeGmsh.opensees._internal.build import BridgeError

    ops = _param_ranks(2, partitioned=False, staged=False)
    with ops.stage(name="bad") as s:
        if verb == "activate_absorbing":
            s.activate_absorbing(elements=[10, 999])
        else:
            s.update_parameter("E", 1.0, elements=[999])
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-4, max_iter=10),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Transformation(),
            numberer=ops.numberer.Plain(),
            system=ops.system.BandGeneral(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    bm = ops.build()
    em = RecordingEmitter()
    with pytest.raises(BridgeError, match=match) as err:
        bm.emit(em)
    assert "plan_tags" in {entry.name for entry in err.traceback}
    assert em.calls == [("model", (), {"ndm": 3, "ndf": 3})]


def test_parameter_writers_refuse_a_line_they_do_not_write() -> None:
    """The writers write the planned lines they are given, and refuse a
    line of another verb, a tag count that does not match the line's
    elements, or a list of elements per line of another length."""
    from apeGmsh.opensees._internal.build import (
        BridgeError,
        ParameterSite,
        plan_parameters,
        write_planned_flips,
        write_planned_ramps,
    )

    sub = _case("param_ranks_4/staged").plan.parameters
    ramp, _s1, flip, update, *_ = sub.planned()
    em = RecordingEmitter()
    assert write_planned_ramps(em, [ramp]) == {"g": (1, 2, 3)}
    write_planned_flips(em, [flip, update], [(9,), (9,)])
    assert [c[:2] for c in em.calls[1:]] == [
        ("flip_element_stage", (7, (9,))),
        ("update_parameter", (8, (9,), ("E",), 2.0e6))]
    for lines, ele in (([flip], [()]), ([flip], [(9,), (9,)]),
                       ([ramp], [(9,)]),
                       ([flip._replace(tags=(7, 8))], [(9,)])):
        with pytest.raises(BridgeError):
            write_planned_flips(RecordingEmitter(), lines, ele)
    with pytest.raises(BridgeError, match="not an initial-stress ramp"):
        write_planned_ramps(RecordingEmitter(), [flip])
    for verb, n in (("update_parameter", 2), ("step_hook_ramp", 1),
                    ("parameter", 1)):
        with pytest.raises(BridgeError, match=f"declares {n} parameter tags"):
            plan_parameters([ParameterSite(None, None, verb, n)],
                            TagAllocator())


# ---------------------------------------------------------------------------
# The MP-element and interface families (K1-3d S3c)
# ---------------------------------------------------------------------------


#: Cases whose emit writes MP elements / interfaces (``_MP_MODELS`` plus
#: the synthesised truss, flat and under ``element_tags="fem"``).
_MP_CASES = (
    *(n for n in _MP_MODELS if n.startswith("mp_") or "embed" in n),
    "synthesised_elements/flat", "synthesised_elements_fem_ids/flat",
)
_IFACE_CASES = (
    *(n for n in _MP_MODELS if n.startswith("iface")),
    "synthesised_elements/flat", "synthesised_elements_fem_ids/flat",
)


@pytest.mark.parametrize("family, names", [
    ("mp_elements", _MP_CASES), ("interfaces", _IFACE_CASES)])
def test_short_or_empty_mp_or_interface_plan_fails_the_oracle(
    family: str, names: tuple[str, ...],
) -> None:
    """Neither comparison is vacuous: a dropped row is caught.

    Every case that writes the family's tags (flat, staged, and
    partitioned over 2 and 4 ranks) must reject its plan with the last
    row dropped, and with every row dropped, while accepting the real one.
    """
    checked: set[str] = set()
    for name in CASES:
        case = _case(name)
        rows = case.plan.family(family).stream()
        if not rows:
            continue
        checked.add(name)
        assert _family_mismatch(case, family, rows) is None, name
        assert _family_mismatch(case, family, rows[:-1]), name
        assert _family_mismatch(case, family, ()), name
    assert set(names) <= checked
    modes = {_case(n).plan.mode for n in names}
    assert TagMode(False, True, True) in modes      # staged partitioned
    assert TagMode(False, False, True) in modes     # staged flat


def test_mp_elements_number_rank_by_rank() -> None:
    """The partitioned plan numbers MP elements rank by rank, as the deck did.

    The 20 bars take the element plan's tags, up to ``e``. Rank ``r``'s
    block then writes its rigid body, coupling, tie (the global MP pass),
    its reinforce tie and its rebar cell, so they take ``e + 1 + 5r`` to
    ``e + 5 + 5r``. The flat deck writes every rigid body, then every
    coupling, every tie, every reinforce tie, every rebar cell, each in
    declaration order (rank 3 first).
    """
    def named(name: str) -> list[tuple[str, int]]:
        mp = _case(name).plan.mp_elements.planned()
        return [(getattr(ln.record, "name", None) or ln.record.pg, ln.tag)
                for ln in mp.lines]

    e = max(t for _, t in _case("mp_ranks_4/flat").plan.elements.stream())
    assert e == max(
        t for _, t in _case("mp_ranks_4/partitioned").plan.elements.stream())
    assert named("mp_ranks_4/partitioned") == [
        (f"{kind}{r}", e + 1 + 5 * r + k)
        for r in range(4)
        for k, kind in enumerate(("rb", "kc", "tie", "rt", "bar"))
    ]
    assert named("mp_ranks_4/flat") == [
        (f"{kind}{r}", e + 1 + 4 * k + (3 - r))
        for k, kind in enumerate(("rb", "kc", "tie", "rt", "bar"))
        for r in (3, 2, 1, 0)
    ]


def test_staged_mp_elements_number_after_the_global_pass() -> None:
    """Stage-claimed MP elements take their tags in the stage's pass.

    Stage ``s2`` claims rank 0's rigid body and coupling and rank 1's
    tie. The global pass skips them; the stage pass numbers them after
    every global element, rank by rank on the partitioned deck.
    """
    for name in ("mp_ranks_2/staged", "mp_ranks_2/staged_partitioned"):
        plan = _case(name).plan
        e = max(t for _, t in plan.elements.stream())
        mp = plan.mp_elements.planned()
        assert [ln.record.name for ln in mp.lines[-3:]] == [
            "rb0", "kc0", "tie1"], name
        assert [ln.tag for ln in mp.lines] == list(range(e + 1, e + 11))
        written = _case(name).stream
        assert [t for k, t in written if _verb(k) in (
            "element", "embeddedNode", "embedded_rebar")][-3:] == [
            e + 8, e + 9, e + 10], name


def test_element_and_material_kinds_are_frozen_on_every_emit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every emit path's allocator refuses an ``element`` or
    ``uniaxialMaterial`` mint: flat, staged, and partitioned and
    staged-partitioned over 2 and 4 ranks."""
    from apeGmsh.opensees.apesees import BuiltModel

    seen: list[tuple[str, TagAllocator]] = []

    def spy(path: str) -> Callable[..., Any]:
        orig = getattr(BuiltModel, path)
        sig = inspect.signature(orig)

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append((path, sig.bind(self, *args, **kwargs)
                         .arguments["tags"]))
            return orig(self, *args, **kwargs)
        return wrapper

    for path in _PATHS:
        monkeypatch.setattr(BuiltModel, path, spy(path))
    names = {*_first_case_per_mode().values(), *_MP_CASES, *_IFACE_CASES}
    for name in sorted(names):
        ts.emit_stream(_MODELS[name]().build(), RecordingEmitter)
    assert {path for path, _ in seen} == set(_PATHS)
    for path, tags in seen:
        assert {"element", "uniaxialMaterial"} <= tags.frozen_kinds, path
        for kind in ("element", "uniaxialMaterial"):
            with pytest.raises(TagLawError, match="planned and frozen"):
                tags.allocate(kind)
            with pytest.raises(TagLawError):
                tags.allocate_block(kind, 1)


@pytest.mark.parametrize("name", sorted({*_MP_CASES, *_IFACE_CASES}))
def test_a_stray_element_mint_raises(
    name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: an emit whose writers mint again raises where they mint.

    The MP-element writers are mutated to mint each element's tag from
    the emit allocator, and ``allocate_interface_tags`` to allocate as
    it did before the plan (at both of its call sites). Every emit that
    writes an MP element or an interface, on every path, must raise
    rather than write a deck.
    """
    import apeGmsh.opensees.apesees as apesees_mod
    from apeGmsh.opensees._internal import build

    def minting_tagger(tags: TagAllocator) -> Any:
        return lambda entry: tags.allocate("element")

    def minting_interfaces(records: Any, tags: TagAllocator) -> Any:
        return {id(r): (tags.allocate("uniaxialMaterial"),
                        tags.allocate("uniaxialMaterial"),
                        tags.allocate("element")) for r in records}

    monkeypatch.setattr(build, "_mp_element_tagger", minting_tagger)
    monkeypatch.setattr(build, "allocate_interface_tags", minting_interfaces)
    monkeypatch.setattr(
        apesees_mod, "allocate_interface_tags", minting_interfaces)
    with pytest.raises(TagLawError, match="planned and frozen"):
        ts.emit_stream(_MODELS[name]().build(), RecordingEmitter)


def _plain_like_the_planner(bm: Any, plan: TagPlan) -> TagAllocator:
    """A plain allocator seeded as the planner was, its ``element``
    counter past the element fan-out: a direct caller's allocator."""
    tags = _seeded_like_the_planner(bm)
    last = max((t for _, t in plan.elements.stream()), default=0)
    if last:
        tags.reserve_through("element", last)
    return tags


def test_mp_writers_are_two_way() -> None:
    """Fork: read the plan. Plain allocator: plan through the same loop.

    The flat MP writers, handed a plain allocator (a direct caller, until
    K1-3d S6), write the same rows as when handed the emit allocator, and
    those rows are the plan's. A plain fork, the frozen planner allocator
    and a fork for another origin carry no plan; a fork of a real foreign
    plan is refused as another model's.
    """
    from apeGmsh.opensees._internal.build import (
        emit_embed_ties,
        emit_mp_constraints,
        emit_rebar_elements,
        emit_reinforce_ties,
    )

    case = _case("mp_ranks_4/flat")
    bm, plan = case.bm, case.plan

    def run(tags: TagAllocator) -> list[Row]:
        em = ts.tapped(RecordingEmitter)()
        em.tap = []
        emit_mp_constraints(em, bm.fem, tags)
        emit_reinforce_ties(em, bm.fem, tags, name_to_tag=bm.name_to_tag)
        emit_embed_ties(em, bm.fem, tags)
        emit_rebar_elements(em, bm.fem, tags, name_to_tag=bm.name_to_tag)
        return [(_verb(k), t) for k, t in em.tap]

    planned = run(plan.emit_allocator())
    assert planned == run(_plain_like_the_planner(bm, plan))
    assert planned == list(plan.mp_elements.stream())
    assert len(planned) == 20

    for other in (plan.allocator.fork(), plan.allocator,
                  plan.allocator.fork({"element"}, origin=object())):
        with pytest.raises(TagLawError, match="carries no tag plan"):
            run(other)
    foreign = _case("mp_ranks_4/partitioned").plan
    with pytest.raises(TagLawError, match="another FEM snapshot"):
        run(foreign.emit_allocator())


def test_allocate_interface_tags_is_two_way() -> None:
    """Fork: read the plan. Plain allocator: plan through the same loop."""
    from apeGmsh.opensees._internal.build import (
        allocate_interface_tags,
        interface_records,
    )

    case = _case("iface_embed/flat")
    plan = case.plan
    records = interface_records(case.bm.fem)
    planned = allocate_interface_tags(records, plan.emit_allocator())
    lines = plan.interfaces.planned().lines
    assert planned == {id(ln.record): ln.tags for ln in lines}
    assert len(planned) == len(records) >= 3

    plain = TagAllocator()
    n0, _t0, e0 = lines[0].tags
    plain.reserve_through("uniaxialMaterial", n0 - 1)
    plain.reserve_through("element", e0 - 1)
    assert allocate_interface_tags(records, plain) == planned

    for other in (plan.allocator.fork(), plan.allocator):
        with pytest.raises(TagLawError, match="carries no tag plan"):
            allocate_interface_tags(records, other)
    foreign = _case("iface_embed_ranks_2/partitioned").plan
    with pytest.raises(TagLawError, match="interface plan holds no tags"):
        allocate_interface_tags(records, foreign.emit_allocator())


def test_partitioned_mp_routing_runs_once_per_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The global pass's rank routing is resolved by the plan, once.

    Before the plan, every partitioned emit ran ``_plan_rank_constraints``
    once per rank. Now the plan runs it once per rank and every emit
    reads it: two emits of the 4-rank model, four routings.
    """
    from apeGmsh.opensees._internal import build

    calls: list[int] = []
    orig = build._plan_rank_constraints

    def spy(**kwargs: Any) -> Any:
        calls.append(kwargs["partition_rank"])
        return orig(**kwargs)

    monkeypatch.setattr(build, "_plan_rank_constraints", spy)
    bm = _MODELS["mp_ranks_4/partitioned"]().build()
    first = ts.emit_stream(bm, RecordingEmitter)
    assert sorted(calls) == [0, 1, 2, 3]
    assert ts.emit_stream(bm, RecordingEmitter) == first
    assert sorted(calls) == [0, 1, 2, 3]


def _foreign_line(case_name: str, site: str) -> Any:
    """A planned line of ``site`` from a second build of ``case_name``: the
    same kind of record, but another model's object.

    A rebar cell is keyed by content, ``(pg, i, j)``, which a second build
    repeats, so its stranger is a cell of a bar the FEM does not carry.
    """
    other = _MODELS[case_name]().build()
    mp = other._tag_plan(_case(case_name).plan.mode).mp_elements.planned()
    line = next(ln for ln in mp.lines if ln.site == site)
    if site == "rebar_cell":
        line = line._replace(key=("no_such_bar", *line.key[1:]))
    return line


_MP_SITE_CASES = {
    "rigid_body": ("mp_ranks_4/flat", "mp_ranks_4/partitioned"),
    "kinematic_coupling": ("mp_ranks_4/flat", "mp_ranks_4/partitioned"),
    "interpolation": ("mp_ranks_4/flat", "mp_ranks_4/partitioned"),
    "reinforce_tie": ("mp_ranks_4/flat", "mp_ranks_4/partitioned"),
    # The partitioned emit refuses g.embed, so embed ties are flat only.
    "embed_tie": ("mp_ranks_embed_4/flat",),
    "rebar_cell": ("mp_ranks_4/flat", "mp_ranks_4/partitioned"),
}


@pytest.mark.parametrize("site", sorted(_MP_SITE_CASES))
def test_mp_plan_refuses_a_swapped_dropped_or_doubled_element(
    site: str,
) -> None:
    """The MP-element plan covers its FEM exactly, checked by key.

    Each element of ``site`` in turn is swapped for the same kind of
    element the FEM does not hold (the count is unchanged), dropped, or
    planned twice; every variant is refused when it is made.
    """
    import dataclasses

    from apeGmsh.opensees._internal.build import MP_ELEMENT_SITES, MPElementPlan

    assert set(_MP_SITE_CASES) == set(MP_ELEMENT_SITES)
    for name in _MP_SITE_CASES[site]:
        mp = _case(name).plan.mp_elements.planned()
        assert MPElementPlan(
            fem=mp.fem, lines=mp.lines, partitioned=mp.partitioned,
            claimed_ids=mp.claimed_ids,
        ).lines == mp.lines
        stranger = _foreign_line(name, site)
        at = [i for i, ln in enumerate(mp.lines) if ln.site == site]
        assert len(at) == 4
        for i in at:
            swapped = (*mp.lines[:i],
                       stranger._replace(tag=mp.lines[i].tag),
                       *mp.lines[i + 1:])
            dropped = mp.lines[:i] + mp.lines[i + 1:]
            doubled = (*mp.lines, mp.lines[i])
            for lines, match in ((swapped, "does not cover"),
                                 (dropped, "does not cover"),
                                 (doubled, "twice")):
                with pytest.raises(TagLawError, match=match):
                    dataclasses.replace(mp, lines=lines)


def test_interface_plan_refuses_a_swapped_dropped_or_doubled_record() -> None:
    import dataclasses

    from apeGmsh.opensees._internal.build import PlannedInterface

    plan = _case("iface_embed/flat").plan.interfaces.planned()
    other = _MODELS["iface_embed/flat"]().build()
    stranger = other._tag_plan(
        _case("iface_embed/flat").plan.mode).interfaces.planned().lines[0]
    for i, line in enumerate(plan.lines):
        swapped = (*plan.lines[:i],
                   PlannedInterface(stranger.record, line.tags),
                   *plan.lines[i + 1:])
        dropped = plan.lines[:i] + plan.lines[i + 1:]
        for lines, match in ((swapped, "does not cover"),
                             (dropped, "does not cover"),
                             ((*plan.lines, line), "twice")):
            with pytest.raises(TagLawError, match=match):
                dataclasses.replace(plan, lines=lines)


def test_mp_plan_refuses_other_reads() -> None:
    """A record, a rank, a FEM or a claim set the plan was not made for."""
    from apeGmsh.opensees._internal.build import mp_element_entry

    flat = _case("mp_ranks_4/flat").plan.mp_elements.planned()
    part = _case("mp_ranks_4/partitioned").plan.mp_elements.planned()
    stranger = _foreign_line("mp_ranks_4/flat", "rigid_body")
    with pytest.raises(TagLawError, match="holds no rigid_body element"):
        flat.tag_of(mp_element_entry("rigid_body", stranger.record))
    with pytest.raises(TagLawError, match="carries no rank routing"):
        flat.rank_constraints(0)
    assert sorted(part.rank_plans) == [0, 1, 2, 3]
    with pytest.raises(TagLawError, match="routes no rank 7"):
        part.rank_constraints(7)
    with pytest.raises(TagLawError, match="another FEM snapshot"):
        flat.for_fem(part.fem)
    with pytest.raises(TagLawError, match="other stage claims"):
        flat.check_claims(frozenset({1}))
    staged = _case("mp_ranks_2/staged").plan.mp_elements.planned()
    assert len(staged.claimed_ids) == 3
    with pytest.raises(TagLawError, match="other stage claims"):
        staged.check_claims(frozenset())


def test_mp_and_interface_plans_derive_their_rows() -> None:
    from apeGmsh.opensees._internal.tag_plan import (
        InterfaceTagPlan,
        MPElementTagPlan,
    )

    with pytest.raises(TagLawError, match="pass mp, not rows"):
        MPElementTagPlan(rows=(("element", 1),))
    with pytest.raises(TagLawError, match="no MP-element plan"):
        MPElementTagPlan().stream()
    with pytest.raises(TagLawError, match="pass interfaces, not rows"):
        InterfaceTagPlan(rows=(("element", 1),))
    with pytest.raises(TagLawError, match="no interface plan"):
        InterfaceTagPlan().stream()


_ALL_MIGRATED = all(cls.MIGRATED for cls in FAMILY_PLANS.values())


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=[] if _ALL_MIGRATED else [pytest.mark.xfail(
        raises=NotImplementedError, strict=True,
        reason="families still pending: " + ", ".join(
            f for f in FAMILIES if not FAMILY_PLANS[f].MIGRATED),
    )])
    for n in CASES
])
def test_whole_plan_equals_tapped_stream(name: str) -> None:
    """``sorted(plan.stream())`` is every derived row the emit writes."""
    case = _case(name)
    planned = sorted((_verb(k), t) for k, t in case.plan.stream())
    derived = sorted(_per_rank_once(case, Counter(
        (_verb(k), t) for k, t in case.stream
        if _owner(case, (k, t)) is not None)).elements())
    assert planned == derived, (
        f"{name} [{case.plan.mode}]: the plan and the emit disagree")


def test_every_family_has_rows() -> None:
    """Each family's comparison runs on at least one case with rows."""
    for family in FAMILIES:
        kinds = FAMILY_PLANS[family].KINDS
        assert any(
            _rows(_case(n), family, kinds) or (
                FAMILY_PLANS[family].MIGRATED
                and _case(n).plan.family(family).stream())
            for n in CASES
        ), f"no corpus case writes a {family} tag; the oracle is vacuous"


@pytest.mark.parametrize("name", CASES)
def test_every_derived_row_has_an_owner(name: str) -> None:
    """Every tagged verb is classified, and a planned row has a planner."""
    case = _case(name)
    migrated_kinds: set[str] = set()
    for cls in FAMILY_PLANS.values():
        if cls.MIGRATED:
            migrated_kinds |= cls.KINDS
    for row in case.stream:
        verb = _verb(row[0])
        assert verb in VERB_KIND or verb in NON_DERIVED_VERBS, (
            f"{name}: classify the tagged verb {verb!r}")
        if _owner(case, row) == PLANNED:
            assert VERB_KIND[verb] in migrated_kinds, (
                f"{name}: {row} was neither minted at emit nor seeded, and "
                "no migrated family plans its kind"
            )


def test_mint_sites_name_families() -> None:
    assert set(_MINT_SITES.values()) <= set(FAMILIES)


@pytest.mark.parametrize("name", CASES)
def test_plan_seed_is_the_emit_seed(name: str) -> None:
    """The planner seeds, then mints its families, then the emit runs on.

    The planner's seed is read from the mint log: its counters after the
    primitive seeding and the ``element_tags="fem"`` reservation, before
    its first mint (``test_plan_seeds_every_registered_tag`` holds the
    seed to ``bm.tag_for``). Its mints in a kind run on, without a gap,
    from one above the seed, and the frozen counter ends there. Every
    emit-time mint in a kind then runs on, without a gap, from one above
    the plan's counter.
    """
    case = _case(name)
    kinds = set(case.seed) | {
        k for cls in FAMILY_PLANS.values() for k in cls.KINDS}
    for kind in sorted(kinds):
        first = case.seed.get(kind, 0) + 1
        tags = case.planned.get(kind, [])
        assert sorted(tags) == list(range(first, first + len(tags))), (
            f"{name}: planned {kind} tags are not the run from {first}")
        assert case.plan.allocator.last(kind) == first - 1 + len(tags), (
            f"{name}: the plan's {kind!r} counter is "
            f"{case.plan.allocator.last(kind)}, its seed and mints end at "
            f"{first - 1 + len(tags)}"
        )
    for kind, tags in case.minted.items():
        first = case.plan.allocator.last(kind) + 1
        assert sorted(tags) == list(range(first, first + len(tags))), (
            f"{name}: emitted {kind} tags {sorted(tags)} are not the run "
            f"from {first}, one above the plan's counter"
        )


def test_fem_ids_case_reserves_past_the_carrier() -> None:
    """The ``element_tags="fem"`` case: FEM ids from 33, the planned MP
    elements and interface from 51, nothing minted at emit."""
    case = _case("synthesised_elements_fem_ids/flat")
    assert case.seed["element"] == ts._SYNTH_CARRIER_EID
    assert min(case.planned["element"]) == ts._SYNTH_CARRIER_EID + 1
    assert "element" not in case.minted
    elements = sorted(t for k, t in case.stream if k == "element:Truss")
    assert elements[0] == ts._SYNTH_FIRST_EID


def test_corpus_reaches_every_mode() -> None:
    modes = {_case(n).plan.mode for n in CASES}
    for split, partitioned, staged in (
        (False, False, False), (False, True, False),
        (False, False, True), (False, True, True),
    ):
        assert TagMode(split, partitioned, staged) in modes


def test_two_rank_regions_number_differently_by_mode() -> None:
    """The invariant-8 fixture: rank order is not declaration order.

    ``east`` (rank 1's nodes) is declared before ``west`` (rank 0's).
    The flat deck numbers the named regions in declaration order; the
    partitioned deck numbers the MPCO filter region first and then each
    named region on the first rank that emits it. The region plan has to
    reproduce both, until the canonical-numbering slice makes them one.
    """
    def regions(name: str) -> list[int]:
        return [t for k, t in _case(name).stream if k == "region"]

    assert regions("two_rank_regions/flat") == [1, 2, 3]
    assert regions("two_rank_regions/partitioned") == [2, 1, 3, 1]


# ---------------------------------------------------------------------------
# plan_tags
# ---------------------------------------------------------------------------


def test_plan_tags_freezes_its_allocator() -> None:
    plan = _case("two_column_frame/flat").plan
    assert plan.allocator.frozen
    with pytest.raises(TagLawError):
        plan.allocator.allocate("element")


def test_plan_seeds_every_registered_tag() -> None:
    case = _case("initial_stress_frame/flat")
    bm, plan = case.bm, case.plan
    for prim in bm.primitives:
        assert plan.allocator.tag_for(prim) == bm.tag_for[id(prim)]


def test_a_kind_shared_with_a_pending_family_stays_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``element`` is frozen only once every family minting it has moved.

    The elements, MP-element and interface families all mint ``element``
    and are all planned (K1-3d S3c), so the emit fork freezes
    ``element``, and ``uniaxialMaterial`` with the interfaces. Were one of
    them still pending, the fork would leave the shared kind open, and
    the oracle's still-minted check would be what catches a mint the
    migration missed.
    """
    from apeGmsh.opensees._internal.tag_plan import MPElementTagPlan

    plan = _case("two_column_frame/flat").plan
    assert plan.migrated == tuple(
        f for f in FAMILIES if FAMILY_PLANS[f].MIGRATED)
    assert {"elements", "mp_elements", "interfaces"} <= set(plan.migrated)
    assert {"element", "uniaxialMaterial"} <= plan.frozen_kinds
    monkeypatch.setattr(MPElementTagPlan, "MIGRATED", False)
    assert "mp_elements" not in plan.migrated
    assert "element" not in plan.frozen_kinds
    assert "uniaxialMaterial" in plan.frozen_kinds


def test_element_plan_rows_come_from_its_specs() -> None:
    from apeGmsh.opensees._internal.tag_plan import ElementTagPlan

    with pytest.raises(TagLawError, match="specs, not rows"):
        ElementTagPlan(rows=(("element", 1),))
    plan = _case("two_column_frame/flat").plan
    assert plan.elements.stream() == tuple(
        ("element", t) for _s, sub in plan.elements.specs
        for _e, _c, t in sub)
    assert plan.elements.stream()


# ---------------------------------------------------------------------------
# The plan each emit path reads (plan_of)
# ---------------------------------------------------------------------------


_PATHS = ("_emit_flat", "_emit_partitioned",
          "_emit_stages_flat", "_emit_stages_partitioned")


def _first_case_per_mode() -> dict[TagMode, str]:
    out: dict[TagMode, str] = {}
    for name in CASES:
        out.setdefault(_case(name).plan.mode, name)
    return out


def test_every_emit_path_reads_the_memoised_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``plan_of(tags)`` is the memoised plan, on every path and every emit.

    Each emit path (flat, staged flat, partitioned, staged
    partitioned) receives a fresh ``emit_allocator()`` fork of the one plan
    ``BuiltModel.emit`` memoises for its mode; a second emit reuses it.
    """
    from apeGmsh.opensees.apesees import BuiltModel

    seen: list[tuple[str, TagAllocator]] = []

    def spy(path: str) -> Callable[..., Any]:
        orig = getattr(BuiltModel, path)
        sig = inspect.signature(orig)

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(self, *args, **kwargs)
            seen.append((path, bound.arguments["tags"]))
            return orig(self, *args, **kwargs)
        return wrapper

    for path in _PATHS:
        monkeypatch.setattr(BuiltModel, path, spy(path))

    reached: set[str] = set()
    for mode, name in _first_case_per_mode().items():
        bm = _MODELS[name]().build()
        cls: type = RecordingEmitter
        forks: list[TagAllocator] = []
        plans: list[TagPlan] = []
        for _ in range(2):
            seen.clear()
            ts.emit_stream(bm, cls)
            assert seen, f"{name}: no emit path ran"
            memo = bm._tag_plans[mode]
            for path, tags in seen:
                reached.add(path)
                assert plan_of(tags) is memo, f"{name}: {path}"
            forks.append(seen[0][1])
            plans.append(plan_of(seen[0][1]))
        # The second emit reads the plan the first one made.
        assert plans[0] is plans[1], f"{name}: re-planned on the second emit"
        assert forks[0] is not forks[1], f"{name}: one fork, two emits"
        assert list(bm._tag_plans) == [mode]
    assert reached == set(_PATHS)


def test_plan_of_refuses_an_allocator_without_a_plan() -> None:
    plan = _case("two_column_frame/flat").plan
    for tags in (TagAllocator(), plan.allocator, plan.allocator.fork(),
                 plan.allocator.fork({"element"}, origin=object())):
        with pytest.raises(TagLawError, match="carries no tag plan"):
            plan_of(tags)
    assert plan_of(plan.emit_allocator()) is plan


def test_emit_refuses_an_element_plan_for_other_specs() -> None:
    from apeGmsh.opensees.apesees import _planned_element_specs

    plan = _case("kitchen_sink_absorbing/staged").plan
    specs = [s for s, _ in plan.elements.specs]
    assert len({id(s) for s in specs}) == len(specs) >= 2
    tags = plan.emit_allocator()
    got = [s for s, _ in _planned_element_specs(tags, specs)]
    assert [id(s) for s in got] == [id(s) for s in specs]
    wrongs = {
        "short": specs[:-1],
        "extra": [*specs, specs[0]],
        "rotated": [specs[-1], *specs[:-1]],
        "swapped": [specs[1], specs[0], *specs[2:]],
        "reversed": specs[::-1],
        "empty": [],
    }
    for label, wrong in wrongs.items():
        with pytest.raises(TagLawError, match="element plan"):
            _planned_element_specs(tags, wrong)
            pytest.fail(f"{label}: accepted")


def _fresh_stream(name: str) -> list[Row]:
    return list(ts.emit_stream(_MODELS[name]().build(), RecordingEmitter))


def test_replace_gives_a_fresh_memo_and_fresh_tags() -> None:
    """``dataclasses.replace`` never reuses its source's plan.

    The source is emitted first, so its memo is populated; the copy, with
    ``element_tags`` or ``fem`` replaced, must emit what a freshly built
    model of the same inputs emits.
    """
    import dataclasses

    bm = _MODELS["synthesised_elements/flat"]().build()
    ts.emit_stream(bm, RecordingEmitter)
    assert bm._tag_plans

    as_fem = dataclasses.replace(bm, element_tags="fem")
    assert as_fem._tag_plans == {}
    assert (ts.emit_stream(as_fem, RecordingEmitter)
            == _fresh_stream("synthesised_elements_fem_ids/flat"))

    new_fem = dataclasses.replace(bm, fem=ts.synthesised_elements_fem())
    assert new_fem._tag_plans == {}
    assert (ts.emit_stream(new_fem, RecordingEmitter)
            == _fresh_stream("synthesised_elements/flat"))
    (mode,) = bm._tag_plans
    assert not bm._tag_plans[mode].planned_for(new_fem)


def test_copy_shares_the_memo_only_while_the_inputs_match() -> None:
    """``copy.copy`` shares the memo dict; the plan checks its inputs.

    A copy emitted as is reuses the plan and writes the same tags. A copy
    whose input changes (here through ``object.__setattr__``, the only way
    to change a frozen model in place) re-plans instead of reading a plan
    made for other inputs.
    """
    import copy

    bm = _MODELS["synthesised_elements/flat"]().build()
    first = ts.emit_stream(bm, RecordingEmitter)
    (mode,) = bm._tag_plans
    plan = bm._tag_plans[mode]

    same = copy.copy(bm)
    assert same._tag_plans is bm._tag_plans
    assert ts.emit_stream(same, RecordingEmitter) == first
    assert bm._tag_plans[mode] is plan

    changed = copy.copy(bm)
    object.__setattr__(changed, "element_tags", "fem")
    assert (ts.emit_stream(changed, RecordingEmitter)
            == _fresh_stream("synthesised_elements_fem_ids/flat"))
    assert not plan.planned_for(changed)


def test_emit_allocator_continues_the_plan() -> None:
    plan = _case("arch_with_orientation_fan_out/flat").plan
    tags = plan.emit_allocator()
    assert not tags.frozen
    assert tags.frozen_kinds == plan.frozen_kinds
    before = {k: plan.allocator.last(k)
              for k in ("element", "geomTransf", "region")}
    for kind in before:
        if kind not in plan.frozen_kinds:
            assert tags.allocate(kind) == before[kind] + 1
    # The plan's own allocator is untouched by the fork's mints.
    assert {k: plan.allocator.last(k) for k in before} == before
    assert before["element"] > 1      # the planned element fan-out


def test_unknown_family_raises() -> None:
    plan = _case("two_column_frame/flat").plan
    with pytest.raises(KeyError, match="unknown tag family"):
        plan.family("nodes")


def test_plan_tags_refuses_a_non_mode() -> None:
    bm = _case("two_column_frame/flat").bm
    with pytest.raises(TypeError, match="TagMode"):
        plan_tags(bm, (False, False, False))  # type: ignore[arg-type]


def test_tag_plan_refuses_an_open_allocator() -> None:
    plan = _case("two_column_frame/flat").plan
    fields = {f: getattr(plan, f) for f in FAMILIES}
    with pytest.raises(TagLawError, match="must be frozen"):
        TagPlan(mode=plan.mode, allocator=TagAllocator(), **fields)


def test_pending_subplan_cannot_carry_rows() -> None:
    for name, cls in FAMILY_PLANS.items():
        if cls.MIGRATED:
            continue
        with pytest.raises(TagLawError, match="pending"):
            cls(rows=(("region", 1),))


# ---------------------------------------------------------------------------
# TagAllocator.freeze() / fork()
# ---------------------------------------------------------------------------


def _seeded_allocator() -> TagAllocator:
    tags = TagAllocator()
    tags.allocate("element")
    tags.allocate_block("element", 3)
    tags.allocate("region")
    return tags


@pytest.mark.parametrize("call", [
    lambda t: t.allocate("element"),
    lambda t: t.allocate("brand_new_kind"),
    lambda t: t.allocate_block("element", 2),
    lambda t: t.allocate_block("element", 0),
    lambda t: t.allocate_for(object(), "element"),
    lambda t: t.reserve_through("element", 1),
    lambda t: t.reset(),
], ids=["allocate", "allocate_new_kind", "allocate_block",
        "allocate_block_0", "allocate_for", "reserve_through", "reset"])
def test_frozen_allocator_refuses_every_mint(call: Any) -> None:
    tags = _seeded_allocator()
    tags.freeze()
    with pytest.raises(TagLawError):
        call(tags)
    assert tags.last("element") == 4 and tags.last("region") == 1


def test_frozen_allocator_refuses_allocate_for_of_a_known_primitive() -> None:
    tags = TagAllocator()
    prim = object()
    assert tags.allocate_for(prim, "uniaxialMaterial") == 1
    tags.freeze()
    with pytest.raises(TagLawError):
        tags.allocate_for(prim, "uniaxialMaterial")
    assert tags.tag_for(prim) == 1


def test_freeze_is_idempotent_and_reads_stay_legal() -> None:
    tags = _seeded_allocator()
    assert not tags.frozen
    tags.freeze()
    tags.freeze()
    assert tags.frozen
    assert tags.last("element") == 4
    assert tags.last("never_minted") == 0


def test_fork_freezes_only_the_named_kinds() -> None:
    parent = _seeded_allocator()
    parent.freeze()
    child = parent.fork({"element"})
    assert not child.frozen
    assert child.frozen_kinds == frozenset({"element"})
    with pytest.raises(TagLawError, match="planned and frozen"):
        child.allocate("element")
    with pytest.raises(TagLawError):
        child.allocate_block("element", 1)
    with pytest.raises(TagLawError):
        child.reserve_through("element", 99)
    assert child.allocate("region") == 2
    assert child.allocate("geomTransf") == 1
    # The parent is unchanged by the child's mints.
    assert parent.last("region") == 1
    assert parent.last("geomTransf") == 0


def test_fork_keeps_assignments_and_inherits_frozen_kinds() -> None:
    parent = TagAllocator()
    prim = object()
    parent.allocate_for(prim, "section")
    child = parent.fork({"region"})
    assert child.tag_for(prim) == 1
    grandchild = child.fork({"element"})
    assert grandchild.frozen_kinds == frozenset({"region", "element"})
    with pytest.raises(TagLawError):
        grandchild.allocate("region")


def test_fork_with_a_frozen_kind_refuses_reset() -> None:
    child = _seeded_allocator().fork({"region"})
    with pytest.raises(TagLawError, match="frozen kinds"):
        child.reset()
    assert child.allocate("element") == 5


def test_fork_with_no_kinds_is_an_open_copy() -> None:
    parent = _seeded_allocator()
    parent.freeze()
    child = parent.fork()
    assert child.allocate("element") == 5
    assert child.allocate_for(object(), "element") == 6


@pytest.mark.parametrize("make", [
    lambda: _seeded_allocator().fork(),
    lambda: _seeded_allocator().fork({"region"}),
    lambda: _case("two_column_frame/flat").plan.emit_allocator(),
], ids=["no_frozen_kinds", "frozen_kind", "emit_allocator"])
def test_every_fork_refuses_reset(make: Any) -> None:
    """A fork never clears: that would re-mint tags its parent handed out.

    Before S2 a fork with no frozen kinds still reset; now that the emit
    mints from ``plan.emit_allocator()``, every fork refuses.
    """
    child = make()
    last = child.last("element")
    with pytest.raises(TagLawError, match="forked allocator"):
        child.reset()
    assert child.last("element") == last


@pytest.mark.parametrize("bad", ["", 3, None])
def test_fork_refuses_a_non_kind(bad: Any) -> None:
    with pytest.raises(TypeError, match="non-empty str"):
        TagAllocator().fork({bad})


def test_unfrozen_allocator_is_unchanged() -> None:
    """Before a freeze the allocator counts exactly as it always has."""
    tags = TagAllocator()
    assert tags.allocate("element") == 1
    assert tags.allocate_block("element", 3) == 2
    assert tags.allocate_block("element", 0) == 5
    tags.reserve_through("element", 10)
    assert tags.allocate("element") == 11
    prim = object()
    assert tags.allocate_for(prim, "region") == 1
    assert tags.allocate_for(prim, "region") == 1
    tags.reset()
    assert tags.last("element") == 0 and tags.tag_for(prim) is None
