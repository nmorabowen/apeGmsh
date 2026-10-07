"""Canonical numbering: a tag is rank-invariant (ADR 0114 D4, amended, item 4).

A tag is an archive fact, so one owner has one tag whatever the emit
mode. One model carries every derived tag kind: ``element`` (the bars,
the MP elements, the interface ``zeroLength``), ``uniaxialMaterial`` (the
interface's two materials), ``geomTransf`` (an orientation fan-out),
``contactSurface`` and ``contact`` (a contact and a contact plane),
``region`` (named regions, a region-scoped Rayleigh, a stage's region)
and ``parameter`` (initial-stress ramps, an absorbing flip and an
update). It is emitted flat and partitioned over 2 and 4 ranks; every
owner must take the same tag in all three, from the plan each emit held,
and the decks must write exactly those tags.

Every record is declared in reverse block order, and some records span
two blocks, so a rank-by-rank numbering would differ from the flat one
(it did, before canonical numbering: the 2- and 4-rank maps then differ
from the flat map, which ``test_rank_by_rank_numbering_fails_the_oracle``
proves the comparison would catch).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees._internal.tag_plan import TagMode, TagPlan, emit_mode
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.contract import _tag_streams as ts

#: Four blocks of the model; a partitioned emit puts block ``k`` on rank
#: ``k * n // 4``.
_BLOCKS = 4

#: The emit modes the model is driven in: ``None`` is flat (unpartitioned).
_RANKS: tuple[int | None, ...] = (None, 2, 4)

#: The model's two variants (:func:`_model`).
_VARIANTS: tuple[str, ...] = ("contact", "staged")


def _fem(n_ranks: int | None, *, contact: bool) -> Any:
    """The model's FEM stub, unpartitioned or cut into ``n_ranks`` ranks.

    Block ``k`` (``b = 100 (k + 1)``, ``e = 10 (k + 1)``, ``x = 10 k``):

    * nodes ``b+1..b+11`` and bars ``e..e+4``: a kinematic coupling
      ``kc{k}``, a penalty tie ``tie{k}`` (its slave
      ``b+5`` lives on the next block), a reinforce tie ``rt{k}`` and a
      rebar cell on ``bar{k}``;
    * nodes ``b+21..b+27`` and bars ``e+5..e+7``: a contact ``c{k}`` of the
      pad ``b+21..b+24`` (its slaves ``b+25, b+26`` on the next block) and
      a contact plane ``p{k}`` of ``b+27`` (when ``contact``);
    * nodes ``b+31, b+32`` (coincident) and ``b+33``, bars ``e+8``
      (backing the interface) and ``e+9``: an interface ``b+31 / b+32``;
    * nodes ``b+41..b+43`` and beams ``1000 + e``, ``1001 + e`` (``Beams``),
      whose spherical orientation fans the transform out.
    """
    from apeGmsh._kernel.records._constraints import (
        ContactPlaneRecord,
        ContactRecord,
        InterfaceRecord,
        InterpolationRecord,
        NodeGroupRecord,
        NormalLaw,
        ReinforceTieRecord,
        TangentialLaw,
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
    beam_ids: list[int] = []
    beams: list[tuple[int, int]] = []
    block_nodes: dict[int, list[int]] = {k: [] for k in range(_BLOCKS)}
    block_elems: dict[int, list[int]] = {k: [] for k in range(_BLOCKS)}
    node_recs: list[Any] = []
    interps: list[Any] = []
    ties: list[Any] = []
    rebar: list[Any] = []
    contacts: list[Any] = []
    planes: list[Any] = []
    interfaces: list[Any] = []
    for k in range(_BLOCKS):
        b, e, x, nxt = 100 * (k + 1), 10 * (k + 1), 10.0 * k, (k + 1) % _BLOCKS
        own = [b + i for i in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11,
                               21, 22, 23, 24, 27, 31, 32, 33, 41, 42, 43)]
        block_nodes[k] += own
        block_nodes[nxt] += [b + 5, b + 25, b + 26]
        ids += [b + i for i in (*range(1, 12), *range(21, 28),
                                31, 32, 33, 41, 42, 43)]
        coords += [
            (x, 0.0, 0.0), (x + 1, 0.0, 0.0), (x + 1, 1.0, 0.0),
            (x, 1.0, 0.0), (x + 0.3, 0.3, 0.0), (x, 0.0, 2.0),
            (x + 1, 0.0, 2.0), (x, 1.0, 2.0), (x + 1, 1.0, 2.0),
            (x + 0.5, 0.4, 0.0), (x + 0.5, 0.4, 1.0),
            (x, 3.0, 0.0), (x + 1, 3.0, 0.0), (x + 1, 4.0, 0.0),
            (x, 4.0, 0.0), (x + 0.2, 3.2, 0.5), (x + 0.8, 3.8, 0.5),
            (x, 3.0, -1.0),
            (x + 5, 0.0, 0.0), (x + 5, 0.0, 0.0), (x + 6, 0.0, 0.0),
            (x + 2, 6.0, 1.0), (x + 3, 6.0, 1.0), (x + 3, 7.0, 2.0),
        ]
        for i, conn, blk in (
            (0, (b + 1, b + 2), k), (1, (b + 2, b + 3), k),
            (2, (b + 3, b + 4), k), (3, (b + 6, b + 7), k),
            (4, (b + 8, b + 9), k), (5, (b + 21, b + 22), k),
            (6, (b + 23, b + 27), k), (7, (b + 25, b + 26), nxt),
            (8, (b + 31, b + 33), k), (9, (b + 32, b + 33), k),
        ):
            bar_ids.append(e + i)
            bars.append(conn)
            block_elems[blk].append(e + i)
        for i, conn in ((0, (b + 41, b + 42)), (1, (b + 42, b + 43))):
            beam_ids.append(1000 + e + i)
            beams.append(conn)
            block_elems[k].append(1000 + e + i)
        node_recs[:0] = [
            NodeGroupRecord(
                kind=ConstraintKind.KINEMATIC_COUPLING, master_node=b + 6,
                slave_nodes=[b + 7], dofs=[1, 2, 3], name=f"kc{k}"),
        ]
        interps.insert(0, InterpolationRecord(
            kind=ConstraintKind.TIE, slave_node=b + 5,
            master_nodes=[b + 1, b + 2, b + 3], dofs=[1, 2, 3],
            enforce="penalty", stiffness=1.0e10, name=f"tie{k}"))
        ties.insert(0, ReinforceTieRecord(
            kind="reinforce", name=f"rt{k}", rebar_node=b + 10,
            host_nodes=[b + 1, b + 2, b + 3],
            weights=np.array([0.2, 0.4, 0.4]),
            direction=np.array([0.0, 0.0, 1.0]), perfect=1.0e8))
        rebar.insert(0, RebarElementRecord(
            pg=f"bar{k}", element="truss", material="steel", area=1.0e-4,
            connectivity=((b + 10, b + 11),)))
        contacts.insert(0, ContactRecord(
            kind="contact", name=f"c{k}", formulation="nts",
            master_faces=np.array([[b + 21, b + 22, b + 23, b + 24]],
                                  dtype=np.int64),
            master_nps=4, slave_nodes=[b + 25, b + 26],
            kn=1.0e6, kt=0.0, mu=0.0,
        ))
        planes.insert(0, ContactPlaneRecord(
            kind="contact_plane", name=f"p{k}", slave_nodes=[b + 27],
            normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, -2.0), kn=1.0e6,
        ))
        interfaces.insert(0, InterfaceRecord(
            kind=ConstraintKind.INTERFACE, master_node=b + 31,
            slave_node=b + 32, backing_element=e + 8,
            orient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            a_trib=0.25,
            normal_law=NormalLaw(kind="ent", k_per_area=1.0e6),
            tangential_law=TangentialLaw(
                kind="epp", k_per_area=1.0e5, tau_b=250.0),
        ))
    fem = FEMStub(
        nodes=_NodesStub(ids=ids, coords=coords, node_pgs={}),
        elements=_ElementsStub(elem_pgs={
            "Bars": _ElementGroupView(
                ids=tuple(bar_ids), connectivity=tuple(bars)),
            "Beams": _ElementGroupView(
                ids=tuple(beam_ids), connectivity=tuple(beams)),
        }),
    )
    if n_ranks is not None:
        rank_nodes: dict[int, set[int]] = {}
        rank_elems: dict[int, list[int]] = {}
        for k in range(_BLOCKS):
            rank = k * n_ranks // _BLOCKS
            rank_nodes.setdefault(rank, set()).update(block_nodes[k])
            rank_elems.setdefault(rank, []).extend(block_elems[k])
        fem.set_partitions([
            (r, sorted(rank_nodes[r]), rank_elems[r])
            for r in range(n_ranks)])
    fem.add_node_constraints(node_recs)
    fem.add_surface_constraints(interps)
    fem.elements.reinforce_ties = ties
    fem.elements.rebar_elements = rebar
    if contact:
        fem.elements.contacts = contacts
        fem.elements.contact_planes = planes
    fem.elements.interfaces = interfaces
    return fem


def _model(n_ranks: int | None, variant: str) -> Any:
    """The model over :func:`_fem`, with every region and parameter site
    its variant allows.

    ``contact`` is unstaged and carries the contacts; ``staged`` drops
    them (a staged partitioned emit refuses contact, ADR 0092 S4) and adds
    a stage with a named region, an absorbing flip and two updates, which
    a partitioned deck writes rank by rank.
    """
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.transform import Spherical

    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    partitioned = n_ranks is not None
    ops = apeSees(cast(Any, _fem(n_ranks, contact=variant == "contact")))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6)
    ops.uniaxialMaterial.ElasticMaterial(E=2.0e5, name="steel")
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    ops.element.elasticBeamColumn(
        pg="Beams",
        transf=ops.geomTransf.Linear(
            orientation=Spherical(origin=(0.0, 0.0, 0.0))),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    for k in reversed(range(_BLOCKS)):
        ops.region(name=f"blk{k}", nodes=[100 * (k + 1) + 1])
    ops.damping.rayleigh(alpha_m=0.01, beta_k=0.001, on="Bars")
    ops.initial_stress(name="g", pg="Bars", sigma_xx=-1.0, sigma_yy=-2.0,
                       sigma_zz=-3.0, ramp_steps=2)
    if variant == "contact":
        return ops

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
    last = 10 * _BLOCKS
    with ops.stage(name="s1") as s:
        s.region(name="probe", nodes=[last * 10 + 2, 102])
        s.activate_absorbing(elements=[last + 1, 11])
        s.update_parameter("A", 0.02, pg="Bars")
        s.update_parameter("E", 2.0e6, elements=[last])
        s.analysis(**chain())
        s.run(n_increments=1)
    return ops


def _plan(
    n_ranks: int | None, variant: str,
) -> tuple[Any, TagPlan, list[tuple[str, int]]]:
    """Build and emit the model; return the model, the plan its emit
    held, and the ``(kind, tag)`` stream it wrote."""
    bm = _model(n_ranks, variant).build()
    mode = emit_mode(bm, split=False, supports_partitions=True)
    assert mode == TagMode(False, n_ranks is not None, variant == "staged")
    stream = ts.emit_stream(bm, RecordingEmitter)
    return bm, bm._tag_plans[mode], stream


def _label(record: Any) -> Any:
    """A record's label, the same object-free value in every build."""
    for attr in ("name", "pg"):
        value = getattr(record, attr, None)
        if value:
            return value
    elements = getattr(record, "elements", None)
    if elements is not None:
        return tuple(elements)
    return (record.master_node, record.slave_node)


def owner_tags(bm: Any, plan: TagPlan) -> dict[Any, Any]:
    """Every derived owner of ``plan`` -> its tag(s), by object-free labels.

    A partitioned plan writes an owner once per rank that holds it (a
    region, a flip, an update); each owner must carry one tag across
    them, so a second, different tag raises here.
    """
    out: dict[Any, Any] = {}

    def put(key: Any, tags: Any) -> None:
        held = out.setdefault(key, tags)
        assert held == tags, f"{key!r} holds {held} and {tags}"

    for spec, rows in plan.elements.specs:
        for eid, _conn, tag in rows:
            put(("element", spec.pg, int(eid)), tag)
    fanout = plan.transforms.fanout
    assert fanout is not None
    for transf, lines in fanout.specs:
        for tag, vecxz in lines or ():
            put(("geomTransf", tuple(round(v, 9) for v in vecxz)), tag)
    for (_tid, eid), tag in fanout.overrides.items():
        put(("geomTransf of element", eid), tag)
    for line in plan.mp_elements.planned().lines:
        key = line.key if line.site == "rebar_cell" else None
        put((line.site, _label(line.record), key), line.tag)
    for ifc in plan.interfaces.planned().lines:
        put(("interface", _label(ifc.record)), ifc.tags)
    contacts = plan.contacts.contacts
    assert contacts is not None
    for c in contacts.lines:
        put((c.kind, c.record.name), c.tags)
    scopes = {id(st): st.name for st in bm.stage_records}
    regions = plan.regions.regions
    assert regions is not None
    for r in regions:
        kind, scope = r.site
        put(("region", kind, scopes.get(scope, scope), r.key), r.tag)
    for p in plan.parameters.planned():
        if p.tags:
            put((p.verb, _label(p.record)), p.tags)
    return out


_EMITS: dict[str, Any] = {}


@pytest.fixture(params=_VARIANTS)
def emits(request: pytest.FixtureRequest) -> dict[
        int | None, tuple[Any, TagPlan, list[tuple[str, int]]]]:
    variant = request.param
    if variant not in _EMITS:
        _EMITS[variant] = {n: _plan(n, variant) for n in _RANKS}
    return cast(dict[int | None, Any], _EMITS[variant])


def test_every_kind_is_planned() -> None:
    """The model is not vacuous: between its variants, each derived kind
    takes tags on every emit, and the partitioned plans route over 2 and 4
    ranks. The contact variant alone writes every one of the seven
    kinds."""
    want = {
        "contact": ("element", "uniaxialMaterial", "geomTransf",
                    "contact_surface", "contact", "contact_plane", "region",
                    "step_hook_ramp"),
        "staged": ("element", "uniaxialMaterial", "geomTransf", "region",
                   "step_hook_ramp", "flip_element_stage",
                   "update_parameter"),
    }
    for variant, kinds in want.items():
        for n in _RANKS:
            _bm, plan, stream = _plan(n, variant)
            planned = {k.split(":")[0] for k, _ in plan.stream()}
            written = {k.split(":")[0] for k, _ in stream}
            for kind in kinds:
                # The tap sees a ramp's tags where ``addToParameter``
                # references them (``test_tag_plan_oracle``'s docstring).
                tapped = ("addToParameter" if kind == "step_hook_ramp"
                          else kind)
                assert kind in planned and tapped in written, (
                    variant, n, kind)
            assert plan.mode.partitioned == (n is not None)
            if n is not None:
                assert len(list(plan.mp_elements.planned().rank_plans)) == n


def test_every_owner_takes_one_tag_at_1_2_and_4_ranks(emits: Any) -> None:
    """The oracle: the owner -> tag map is identical in every mode."""
    maps = {n: owner_tags(bm, plan) for n, (bm, plan, _s) in emits.items()}
    flat = maps[None]
    assert len(flat) > 60
    for n in (2, 4):
        assert maps[n] == flat, n


def test_the_decks_write_the_planned_tags(emits: Any) -> None:
    """Each deck writes, per derived verb, exactly the set of tags the
    flat deck writes (a partitioned deck writes some more than once, one
    per holder rank)."""
    derived = ("element", "uniaxialMaterial", "geomTransf", "region",
               "contact_surface", "contact", "contact_plane",
               "addToParameter", "flip_element_stage", "update_parameter",
               "embeddedNode", "embedded_rebar")

    def written(stream: list[tuple[str, int]]) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for kind, tag in stream:
            verb = kind.split(":")[0]
            if verb in derived:
                out.setdefault(verb, set()).add(tag)
        return out

    flat = written(emits[None][2])
    for n in (2, 4):
        assert written(emits[n][2]) == flat, n


def test_rank_by_rank_numbering_fails_the_oracle() -> None:
    """The comparison is not vacuous: renumbering the contacts in the
    partitioned plan's own row order (the pre-canonical, rank-by-rank
    numbering) breaks the equality."""
    from dataclasses import replace

    bm, plan, _ = _plan(4, "contact")
    flat_bm, flat_plan, _ = _plan(None, "contact")
    assert owner_tags(bm, plan) == owner_tags(flat_bm, flat_plan)
    contacts = plan.contacts.contacts
    assert contacts is not None
    surf = iter(range(1, 100))
    cont = iter(range(1, 100))
    rank_major = tuple(
        line._replace(tags=(*(next(surf) for _ in line.tags[:-1]),
                            next(cont)))
        for line in contacts.lines)
    mutated = replace(
        plan, contacts=replace(
            plan.contacts, contacts=replace(contacts, lines=rank_major)))
    assert owner_tags(bm, mutated) != owner_tags(flat_bm, flat_plan)
