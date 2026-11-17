"""A Ladruno recorder that filters nodes AND elements emits two regions.

An OpenSees ``MeshRegion`` cannot hold a node list and an element list at
once: ``MeshRegion::setNodes`` / ``setElements`` each wipe both, and
``-ele`` rebuilds the nodes from the elements. The single
``region $tag -node ... -ele ...`` line the bridge used to write therefore
kept only the element nodes, and ``nodes_pg`` was lost with nothing but an
OpenSees warning (apeGmsh_Examples 02: the embedded-rebar nodes were never
recorded, so the cage drew undeformed).

Fork WP-234 adds ``recorder ladruno ... -R $eleTag -RN $nodeTag``. These
tests pin the bridge side: two regions (``-ele`` only, ``-node`` only),
``-RN`` beside ``-R`` and before ``-G``, on the flat, staged and
partitioned routes, and the build guard on the deck and live routes.
MPCO (stock, no ``-RN``) keeps its single region.
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.stage_emit import (
    LADRUNO_NODE_REGION_CAPABILITY,
    LADRUNO_NODE_REGION_GUARD,
)
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.recorder import Ladruno

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
    make_two_column_frame_partitioned,
)
from tests.opensees.integration.test_deck_md_static_guard import schema_json

_SHA = "f0b9a1df699f9634fd89c5942d69d90fe5ad27c4"


def _fem() -> FEMStub:
    """Quad 1 on Rock (nodes 1-4); nodes 5, 6 sit above it with no
    element of their own in the recorder (the embedded-bar shape)."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0), (1.0, 2.0, 0.0), (0.0, 2.0, 0.0)],
            node_pgs={"Rock": [1, 2, 3, 4], "All": [1, 2, 3, 4, 5, 6],
                      "Base": [1, 2]},
        ),
        elements=_ElementsStub(elem_pgs={
            "Rock": _ElementGroupView(ids=(1,),
                                      connectivity=((1, 2, 3, 4),)),
        }),
    )


def _model(**rec: Any) -> tuple[apeSees, Ladruno]:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    r = ops.recorder.Ladruno(
        file="run.ladruno", nodal_responses=("displacement",),
        elem_responses=("stresses",), **rec)
    return ops, r


def _flag(args: tuple, flag: str) -> list[int]:
    if flag not in args:
        return []
    out, i = [], list(args).index(flag) + 1
    while i < len(args) and not (isinstance(args[i], str)
                                 and args[i].startswith("-")):
        out.append(int(args[i]))
        i += 1
    return out


def _split(calls: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    regions = [c for c in calls if c[0] == "region"]
    lad = [c for c in calls if c[0] == "recorder" and c[1][0] == "ladruno"]
    return regions, lad


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


class TestSpec:
    def test_both_selectors_split(self) -> None:
        r = Ladruno(file="r.ladruno", nodal_responses=("displacement",),
                    elem_responses=("stresses",), nodes_pg="All",
                    elements_pg="Rock")
        assert r.splits_filter()
        assert r.region_keys() == ("filter", "nodes")

    @pytest.mark.parametrize("kw", [
        {"nodes_pg": "All", "nodal_responses": ("displacement",)},
        {"elements_pg": "Rock", "elem_responses": ("stresses",)},
    ])
    def test_one_selector_does_not_split(self, kw: dict) -> None:
        r = Ladruno(file="r.ladruno", **kw)
        assert not r.splits_filter()
        assert r.region_keys() == ("filter",)

    def test_energy_region_keeps_its_place_after_nodes(self) -> None:
        r = Ladruno(file="r.ladruno", nodal_responses=("displacement",),
                    elem_responses=("stresses",), nodes=(1,),
                    elements=(1,), energy_pg="Rock")
        assert r.region_keys() == ("filter", "nodes", "energy")

    def test_emit_puts_RN_after_R_and_before_G(self) -> None:
        r = Ladruno(file="r.ladruno", elem_responses=("stresses",),
                    nodal_responses=("displacement",), nodes=(5,),
                    elements=(1,), energy=True, nsteps=2,
                    _region_tag=3, _node_region_tag=4)
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls[0][1] == (
            "ladruno", "r.ladruno", "-N", "displacement", "-E", "stresses",
            "-T", "nsteps", 2, "-R", 3, "-RN", 4, "-G", "energy", 3)


# ---------------------------------------------------------------------------
# Flat and staged builds
# ---------------------------------------------------------------------------


def test_flat_build_emits_two_regions() -> None:
    ops, _r = _model(nodes_pg="All", elements_pg="Rock")
    rec = RecordingEmitter()
    ops.build().emit(rec)
    regions, lad = _split(rec.calls)
    assert len(regions) == 2 and len(lad) == 1
    by_kind = {("-ele" in c[1]): c[1] for c in regions}
    ele_region, node_region = by_kind[True], by_kind[False]
    assert "-node" not in ele_region and "-ele" not in node_region
    assert sorted(_flag(node_region, "-node")) == [1, 2, 3, 4, 5, 6]
    assert len(_flag(ele_region, "-ele")) == 1
    args = lad[0][1]
    assert _flag(args, "-R") == [int(ele_region[0])]
    assert _flag(args, "-RN") == [int(node_region[0])]
    assert int(ele_region[0]) != int(node_region[0])


def test_single_selector_build_is_unchanged() -> None:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.recorder.Ladruno(file="run.ladruno",
                         nodal_responses=("displacement",), nodes_pg="All")
    rec = RecordingEmitter()
    ops.build().emit(rec)
    regions, lad = _split(rec.calls)
    assert len(regions) == 1
    assert "-RN" not in lad[0][1]


def test_staged_build_emits_two_regions_inside_the_stage() -> None:
    ops = apeSees(_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="probe") as s:
        s.recorder(ops.recorder.Ladruno(
            file="run.ladruno", nodal_responses=("displacement",),
            elem_responses=("stresses",), nodes_pg="All",
            elements_pg="Rock"))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.1),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static())
        s.run(n_increments=2)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    regions, lad = _split(rec.calls)
    assert len(regions) == 2
    opened = names.index("stage_open")
    assert all(rec.calls.index(c) > opened for c in regions)
    tags = {int(c[1][0]) for c in regions}
    assert {_flag(lad[0][1], "-R")[0], _flag(lad[0][1], "-RN")[0]} == tags


# ---------------------------------------------------------------------------
# Partitioned build
# ---------------------------------------------------------------------------


def test_partitioned_build_splits_each_rank() -> None:
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    ops.recorder.Ladruno(
        file="out/run.ladruno", nodal_responses=("displacement",),
        elem_responses=("section.force",), nodes_pg="Top",
        elements_pg="Cols")
    rec = RecordingEmitter()
    ops.build().emit(rec)

    per_rank: dict[int, list[tuple]] = {}
    rank: int | None = None
    glob: list[tuple] = []
    for c in rec.calls:
        if c[0] == "partition_open":
            rank = int(c[1][0])
            per_rank[rank] = []
        elif c[0] == "partition_close":
            rank = None
        elif rank is None:
            glob.append(c)
        else:
            per_rank[rank].append(c)
    _r, lad = _split(glob)
    assert len(lad) == 1
    r_tag = _flag(lad[0][1], "-R")[0]
    rn_tag = _flag(lad[0][1], "-RN")[0]
    for k, calls in per_rank.items():
        regions = {int(c[1][0]): c[1] for c in calls if c[0] == "region"}
        assert set(regions) == {r_tag, rn_tag}, (k, regions)
        assert "-node" not in regions[r_tag]
        assert "-ele" not in regions[rn_tag]
        assert len(_flag(regions[rn_tag], "-node")) == 1   # one Top node


# ---------------------------------------------------------------------------
# The build guard
# ---------------------------------------------------------------------------


def test_capability_name_matches_the_fork() -> None:
    assert LADRUNO_NODE_REGION_CAPABILITY == "recorderLadrunoNodeRegion"
    assert LADRUNO_NODE_REGION_GUARD.capability == "recorderLadrunoNodeRegion"


@pytest.mark.parametrize("emitter_cls", [TclEmitter, PyEmitter])
def test_deck_checks_the_build_before_the_recorder(emitter_cls: type) -> None:
    ops, _r = _model(nodes_pg="All", elements_pg="Rock")
    e = emitter_cls()
    ops.build().emit(e)
    text = "\n".join(e.lines())
    assert "recorderLadrunoNodeRegion" in text
    assert text.index("recorderLadrunoNodeRegion") < text.index("-RN")


@pytest.mark.parametrize("emitter_cls", [TclEmitter, PyEmitter])
def test_deck_without_RN_has_no_guard(emitter_cls: type) -> None:
    ops2 = apeSees(_fem(), default_orientation=None)
    ops2.model(ndm=2, ndf=2)
    mat = ops2.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops2.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops2.recorder.Ladruno(file="run.ladruno",
                          elem_responses=("stresses",), elements_pg="Rock")
    e = emitter_cls()
    ops2.build().emit(e)
    assert "recorderLadrunoNodeRegion" not in "\n".join(e.lines())


@pytest.mark.parametrize("caps, runs", [
    (("mdStageFlagPreserved", "recorderLadrunoNodeRegion"), True),
    (("mdStageFlagPreserved",), False),
    (None, False),                                            # v0 schema
])
def test_py_deck_guard_runs(caps: "tuple[str, ...] | None", runs: bool) -> None:
    """The emitted py guard, executed against a stub fork engine."""
    from types import SimpleNamespace

    from apeGmsh.opensees._internal.build.stage_emit import fork_guard_lines

    code = "\n".join(fork_guard_lines(LADRUNO_NODE_REGION_GUARD, "py"))
    ops = SimpleNamespace(ladrunoBuild=lambda: _SHA,
                          ladrunoSchema=lambda: schema_json(caps=caps))
    if runs:
        exec(code, {"ops": ops})
    else:
        with pytest.raises(SystemExit, match="recorderLadrunoNodeRegion"):
            exec(code, {"ops": ops})


class _Engine:
    def __init__(self, *, build: "str | None" = None,
                 schema: "str | None" = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        if build is not None:
            self.ladrunoBuild = lambda: build
        if schema is not None:
            self.ladrunoSchema = lambda: schema

    def __getattr__(self, name: str) -> Any:
        if name.startswith(("ladruno", "__")):
            raise AttributeError(name)

        def call(*args: Any) -> Any:
            self.calls.append((name, args))
            return 0
        return call


def _live(engine: _Engine) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = engine  # type: ignore[assignment]
    le._real_ops = engine  # type: ignore[assignment]
    le._in_partition = False
    le._step_hooks_registered = False
    le._fork_verified_types = set()
    le._partition_warned = True
    return le


_RN_ARGS = ("run.ladruno", "-N", "displacement", "-R", 1, "-RN", 2)


@pytest.mark.parametrize("engine", [
    _Engine(build=_SHA, schema=schema_json(caps=("mdStageFlagPreserved",))),
    _Engine(build=_SHA, schema=schema_json(caps=None)),       # v0
    _Engine(build=_SHA),                                      # no schema
])
def test_live_refuses_an_older_fork_by_name(engine: _Engine) -> None:
    with pytest.raises(BridgeError, match="recorderLadrunoNodeRegion"):
        _live(engine).recorder("ladruno", *_RN_ARGS)
    assert not any(n == "recorder" for n, _a in engine.calls)


def test_live_passes_a_fork_listing_the_capability() -> None:
    engine = _Engine(build=_SHA, schema=schema_json(
        caps=("mdStageFlagPreserved", "recorderLadrunoNodeRegion")))
    _live(engine).recorder("ladruno", *_RN_ARGS)
    assert ("recorder", ("ladruno", *_RN_ARGS)) in engine.calls


def test_live_without_RN_never_asks() -> None:
    engine = _Engine(build=_SHA)          # would be refused if asked
    _live(engine).recorder("ladruno", "run.ladruno", "-N", "displacement",
                           "-R", 1)
    assert engine.calls and engine.calls[0][0] == "recorder"
