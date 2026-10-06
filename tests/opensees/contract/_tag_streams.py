"""The ``(kind, tag)`` stream one ``BuiltModel`` emit drives (K1-3, #1361).

Shared by the tag-law pins:

* ``tests/opensees/contract/test_tag_streams.py`` compares the
  ``(verb, tag)`` multiset across the Recording, Tcl, Py and H5 emits of
  one ``BuiltModel``;
* ``tests/opensees/subprocess/test_tag_determinism.py`` compares the
  stream built in-process against the one a fresh interpreter builds.

A stream is read by a *tap*: a subclass of the concrete emitter whose
every Protocol verb records ``(kind, tag)`` and then defers to the real
method, so the bridge sees the class it would see anyway (its
``isinstance`` and ``supports_partitions`` branches are unchanged).
Only the outermost verb is recorded, so an emitter that calls its own
verbs internally does not add rows.

Which argument is the tag is derived from the ``Emitter`` Protocol's
signatures, not from a hand table: the first parameter named in
:data:`TAG_PARAMS`. When that parameter is the second one, the first is
the type token (``element('quad', 7, ...)``) and joins the kind.
"""
from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.verbs import VERBS

#: Protocol parameter names that carry an object's tag.
TAG_PARAMS: tuple[str, ...] = ("tag", "ele_tag", "pid")

Stream = list[tuple[str, int]]


def tag_positions() -> dict[str, int]:
    """Protocol verb -> index of its tag argument (verbs with no tag omitted)."""
    out: dict[str, int] = {}
    for verb, row in VERBS.items():
        if row.via != "protocol":
            continue
        params = [
            p for p in inspect.signature(getattr(Emitter, verb)).parameters.values()
            if p.name != "self"
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        for i, p in enumerate(params):
            if p.name in TAG_PARAMS:
                out[verb] = i
                break
    return out


_POSITIONS = tag_positions()


def _project(verb: str, args: tuple[Any, ...]) -> tuple[str, int] | None:
    i = _POSITIONS.get(verb)
    if i is None:
        return None
    tag = args[i]
    if isinstance(tag, bool) or not isinstance(tag, (int, np.integer)):
        raise TypeError(
            f"{verb}: argument {i} should be a tag, got {tag!r}; the "
            "Protocol signature and the call disagree"
        )
    kind = f"{verb}:{args[0]}" if i == 1 else verb
    return kind, int(tag)


def tapped(cls: type) -> type:
    """A subclass of ``cls`` whose Protocol verbs append to ``self.tap``."""
    ns: dict[str, Any] = {}
    for verb, row in VERBS.items():
        if row.via != "protocol":
            continue
        orig = getattr(cls, verb, None)
        if not callable(orig):
            raise AttributeError(f"{cls.__name__} lacks Protocol verb {verb!r}")

        def make(verb: str, orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                if self._tap_depth == 0:
                    row = _project(verb, args)
                    if row is not None:
                        self.tap.append(row)
                self._tap_depth += 1
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    self._tap_depth -= 1
            wrapper.__name__ = verb
            return wrapper

        ns[verb] = make(verb, orig)
    ns["tap"] = None
    ns["_tap_depth"] = 0
    return type(f"Tapped{cls.__name__}", (cls,), ns)


def emit_stream(
    bm: Any, emitter_cls: type, *, split: bool = False,
    **emitter_kwargs: Any,
) -> Stream:
    """Emit the ``BuiltModel`` ``bm`` through a tapped ``emitter_cls``."""
    emitter = tapped(emitter_cls)(**emitter_kwargs)
    emitter.tap = []
    with warnings.catch_warnings():
        # Advisory nudges (auto numberer/system on a partitioned mesh) are
        # not what these pins are about.
        warnings.simplefilter("ignore")
        bm.emit(emitter, split=split)
    stream: Stream = emitter.tap
    return stream


def tag_stream(
    ops: apeSees, emitter_cls: type, *, split: bool = False,
    **emitter_kwargs: Any,
) -> Stream:
    """Build ``ops`` and emit it through a tapped ``emitter_cls``."""
    return emit_stream(ops.build(), emitter_cls, split=split, **emitter_kwargs)


# ---------------------------------------------------------------------------
# The models the pins drive
# ---------------------------------------------------------------------------


#: Golden-corpus modes the streams cover: the flat, partitioned and staged
#: emit paths (``per_rank`` is the partitioned path sliced by the Tcl writer).
GOLDEN_MODES: tuple[str, ...] = (
    "flat", "partitioned", "staged", "staged_partitioned",
)


def models() -> dict[str, Callable[[], apeSees]]:
    """Name -> a fresh ``apeSees`` recipe, covering every emit path but split.

    The golden grid's ``recording`` cells carry node, element and MPCO
    recorders (region tags), and the arch fixture fans out one
    ``geomTransf`` per element. The H5 suites' fixtures add parameter
    tags: a flat initial stress, a staged one, and a staged absorbing flip.
    :func:`two_rank_regions` adds named regions owned by different ranks,
    and :func:`stage_claimed_regions` every global and stage-bound region
    site, with stage-claimed filtered recorders.
    """
    from tests.opensees.golden import builder as golden
    from tests.opensees.h5.test_h5_initial_stress import _build_frame
    from tests.opensees.h5.test_h5_stages_reader import (
        _real_kitchen_sink_bridge,
        _real_two_stage_bridge,
    )

    out: dict[str, Callable[[], apeSees]] = {}
    for fixture in golden.FIXTURES:
        for mode in GOLDEN_MODES:
            if golden.applicability(fixture, mode, "recording") is not None:
                continue
            out[f"{fixture}/{mode}"] = partial(
                golden.build_model, fixture, mode, "recording")
    out["initial_stress_frame/flat"] = partial(
        _build_frame, with_initial_stress=True)
    out["two_stage_initial_stress/staged"] = _real_two_stage_bridge
    out["kitchen_sink_absorbing/staged"] = _real_kitchen_sink_bridge
    for mode in GOLDEN_MODES:
        out[f"two_rank_regions/{mode}"] = partial(two_rank_regions, mode)
    for mode in STAGE_CLAIMED_MODES:
        out[f"stage_claimed_regions/{mode}"] = partial(
            stage_claimed_regions, mode)
    out["synthesised_elements/flat"] = partial(
        synthesised_elements, element_tags="sequential")
    out["synthesised_elements_fem_ids/flat"] = partial(
        synthesised_elements, element_tags="fem")
    return out


#: The FEM element ids of :func:`synthesised_elements_fem`: the emitted
#: bars start at 33, and an unemitted point carrier holds the largest id.
_SYNTH_FIRST_EID = 33
_SYNTH_CARRIER_EID = 50


def synthesised_elements_fem() -> Any:
    """A truss whose FEM carries every stream the bridge synthesises tags for.

    Thirteen 3-dof nodes and seven bars (``Bars``, ids 33-39) carry:

    * MP elements: a kinematic coupling 11 -> 12 (an element) and a
      penalty tie of node 13 to the triangle 1-2-3 (an ``embeddedNode``);
    * an interface between the coincident nodes 9 and 10 (a
      ``zeroLength`` and its two materials);
    * a node-to-surface contact of nodes 5 and 6 on the face 1-2-3-4, and
      a rigid-plane contact of nodes 7 and 8.

    A one-node group (``Carrier``, id 50) is never emitted, so under
    ``element_tags="fem"`` the reservation reaches past the emitted ids.
    """
    from apeGmsh._kernel.records._constraints import (
        ContactPlaneRecord,
        ContactRecord,
        InterfaceRecord,
        InterpolationRecord,
        NodeGroupRecord,
        NormalLaw,
        TangentialLaw,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind
    from apeGmsh.mesh._element_types import ElementGroup, make_type_info
    from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import (
        ElementComposite,
        FEMData,
        MeshInfo,
        NodeComposite,
    )

    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.2, 0.2, 0.5], [0.8, 0.8, 0.5],
        [0.0, 0.0, -1.0], [1.0, 0.0, -1.0],
        [2.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, 0.0, 0.0],
    ], dtype=np.float64)
    node_ids = np.arange(1, len(coords) + 1, dtype=np.int64)
    conn = np.array([
        [1, 5], [2, 6], [3, 4], [7, 8], [9, 11], [10, 12], [13, 2],
    ], dtype=np.int64)
    eids = np.arange(_SYNTH_FIRST_EID, _SYNTH_FIRST_EID + len(conn),
                     dtype=np.int64)
    line = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=len(conn))
    point = make_type_info(
        code=15, gmsh_name="Point", dim=0, order=1, npe=1, count=1)
    groups = {
        1: ElementGroup(element_type=line, ids=eids, connectivity=conn),
        15: ElementGroup(
            element_type=point,
            ids=np.array([_SYNTH_CARRIER_EID], dtype=np.int64),
            connectivity=np.array([[13]], dtype=np.int64)),
    }
    pg = {(1, 100): {
        "name": "Bars",
        "node_ids": node_ids,
        "node_coords": coords,
        "element_ids": eids,
    }}
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        constraints=[NodeGroupRecord(
            kind=ConstraintKind.KINEMATIC_COUPLING, master_node=11,
            slave_nodes=[12], dofs=[1, 2, 3],
        )],
    )
    elements = ElementComposite(
        groups=groups,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        constraints=[InterpolationRecord(
            kind=ConstraintKind.TIE, slave_node=13, master_nodes=[1, 2, 3],
            dofs=[1, 2, 3], enforce="penalty",
        )],
        interfaces=[InterfaceRecord(
            kind=ConstraintKind.INTERFACE, master_node=9, slave_node=10,
            backing_element=int(eids[4]),
            orient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            a_trib=0.25,
            normal_law=NormalLaw(kind="ent", k_per_area=1.0e6),
            tangential_law=TangentialLaw(
                kind="epp", k_per_area=1.0e5, tau_b=250.0),
        )],
        contacts=[ContactRecord(
            kind="contact", name="pad", formulation="nts",
            master_faces=np.array([[1, 2, 3, 4]], dtype=np.int64),
            master_nps=4, slave_nodes=[5, 6], kn=1.0e6, kt=0.0, mu=0.0,
        )],
        contact_planes=[ContactPlaneRecord(
            kind="contact_plane", name="floor", slave_nodes=[7, 8],
            normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, -2.0), kn=1.0e6,
        )],
    )
    info = MeshInfo(
        n_nodes=len(node_ids), n_elems=len(conn) + 1, bandwidth=1,
        types=[line, point],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


def synthesised_elements(*, element_tags: str) -> apeSees:
    """:func:`synthesised_elements_fem` under a plain truss declaration."""
    ops = apeSees(synthesised_elements_fem(), element_tags=element_tags)
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6)
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    return ops


def two_rank_regions(mode: str) -> apeSees:
    """Two named regions owned by different ranks, plus a filtered MPCO.

    The golden two-column frame with its recording set (whose MPCO
    recorder is filtered by ``nodes_pg`` / ``elements_pg``, so it takes a
    region tag) and two named regions: ``east`` holds only rank 1's
    nodes and is declared first, ``west`` holds only rank 0's nodes and
    is declared second. The flat deck numbers them in declaration order;
    the partitioned deck numbers each on the first rank that emits it,
    so the two orders differ. The tag plan must reproduce both.
    """
    from tests.opensees.golden import builder as golden

    ops = golden.build_model("two_column_frame", mode, "recording")
    split = golden.FIXTURES["two_column_frame"].partitions
    assert split is not None
    rank_nodes = {rank: nodes for rank, nodes, _ in split}
    ops.region(name="east", nodes=rank_nodes[1])
    ops.region(name="west", nodes=rank_nodes[0])
    return ops


#: The emit modes :func:`stage_claimed_regions` is driven in.
STAGE_CLAIMED_MODES: tuple[str, ...] = ("staged", "staged_partitioned")


def stage_claimed_regions(mode: str) -> apeSees:
    """Every region site, global and stage-bound, plus stage-claimed recorders.

    The golden two-column frame, staged, with its recording set (a global
    MPCO filtered by ``nodes_pg`` / ``elements_pg``), then:

    * global named regions ``east`` (rank 1's top node) and ``west``
      (rank 0's), declared in that order, and a region-scoped Rayleigh
      and a ``Uniform`` damping attach on ``Cols``;
    * a third stage, ``probe``, with its own named regions (``p_east``
      then ``p_west``), a scoped Rayleigh, a damping attach, and two
      claimed recorders: an MPCO filtered by ``Top`` / ``Cols`` and a
      Ladruno filtered by ``Top`` with a decoupled ``energy_pg``.

    Under ``staged_partitioned`` this is the #1446 model: before the fix,
    the partitioned pre-plan also gave each stage-claimed recorder region
    a tag, which the stage pass then minted again, so three region tags
    were written per rank and never referenced.
    """
    from tests.opensees.golden import builder as golden

    if mode not in STAGE_CLAIMED_MODES:
        raise ValueError(f"stage_claimed_regions: unknown mode {mode!r}")
    ops = golden.build_model("two_column_frame", mode, "recording")
    ops.region(name="east", nodes=[4])
    ops.region(name="west", nodes=[2])
    ops.damping.rayleigh(alpha_m=0.01, beta_k=0.001, on="Cols")
    ops.damping.uniform(ratio=0.02, freq_lower=1.0, freq_upper=10.0,
                        on="Cols")
    with ops.stage(name="probe") as s:
        s.region(name="p_east", nodes=[4])
        s.region(name="p_west", pg="Top")
        s.damping.rayleigh(alpha_m=0.02, beta_k=0.002, on="Cols")
        s.damping.uniform(ratio=0.03, freq_lower=1.0, freq_upper=10.0,
                          on="Cols")
        s.recorder(ops.recorder.MPCO(
            file="out/probe.mpco", nodal_responses=("displacement",),
            nodes_pg="Top", elements_pg="Cols",
        ))
        s.recorder(ops.recorder.Ladruno(
            file="out/probe.ladruno", nodal_responses=("displacement",),
            nodes_pg="Top", energy_pg="Cols",
        ))
        s.analysis(**golden._chain(
            ops, mode in golden.PARTITIONED_MODES))
        s.run(n_increments=1)
    return ops


def split_model() -> apeSees:
    """The golden two-module frame, emitted with ``split=True`` (Tcl/Py only)."""
    from tests.opensees.golden import builder as golden
    return golden.build_model("two_module_frame", "flat", "recording")


def all_streams() -> dict[str, list[list[object]]]:
    """Every model's Recording stream plus the split Tcl stream, JSON-ready."""
    from apeGmsh.opensees.emitter.recording import RecordingEmitter
    from apeGmsh.opensees.emitter.tcl import TclEmitter

    out: dict[str, list[list[object]]] = {}
    for name, recipe in models().items():
        out[name] = [list(r) for r in tag_stream(recipe(), RecordingEmitter)]
    out["two_module_frame/split"] = [
        list(r) for r in tag_stream(split_model(), TclEmitter, split=True)
    ]
    return out
