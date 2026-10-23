"""#2465: an archived stage flip runs the stage close's route checks on replay.

The stage close refuses a stage-flag update that reaches nothing, the wrong
material, or only part of what it must (#2442, #2443, #2445, #2458, #2460).
An archive written before those checks landed can still hold such a flip,
and ``OpenSeesModel.from_h5(...).build`` re-emitted it silently.  The
replay now builds the same ``RouteModel`` from the archive's declarations
and runs the same checks (``stage._builder.check_archived_stage_routes``),
so such an archive is refused by name, and an archive the stage close
accepts replays byte for byte (the R1b oracle).

A pre-fix archive is written here with the checks switched off, the way
the bridge wrote it before them: ``check_stage_routes`` at stage close and,
for the class-wide group, ``validate_class_wide_stage_flags`` at emit.

The live tests run the replayed openseespy deck: a valid archive gives the
bridge deck's answer, and the tag-blind neighbour archive, replayed with the
check off, gives the wrong one that the refusal stops.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees import apesees as _apesees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._target import OpenSeesTarget
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.stage import _builder

# The ManzariDafalias constants of the #2462 tests.
_MD = dict(
    G0=125.0, nu=0.05, e_init=0.8, Mc=1.25, c=0.712, lambda_c=0.019,
    e0=0.934, ksi=0.7, P_atm=100.0, m=0.01, h0=7.05, Ch=0.968, nb=1.1,
    A0=0.704, nd=3.5, z_max=4.0, cz=600.0, rho=1.42,
)
_DP = dict(
    K=1e5, G=5e4, sigmaY=10.0, rho=0.0, rhoBar=0.0, Kinf=10.0, Ko=10.0,
    delta1=0.0, delta2=0.0, H=1000.0, theta=1.0,
)
_TRI6 = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5),
         (0.0, 0.5)]
_QUAD4 = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
#: Physical group -> (shape, element count).  Every element has its own
#: nodes, 3 units apart along x, so no two elements touch.
_GROUPS = (("A", _TRI6, 1), ("B", _TRI6, 1), ("Q", _QUAD4, 2),
           ("P", _QUAD4, 1), ("C", _TRI6, 1))


def _fem() -> FEMData:
    """Disconnected SixNodeTri ``A`` (FEM element 1) and ``B`` (2), two
    quads ``Q`` (3, 4), one quad ``P`` (5) and a SixNodeTri ``C`` (6);
    ``Base`` is every node on y = 0."""
    coords: list[tuple[float, float, float]] = []
    rows: dict[str, list[tuple[int, list[int]]]] = {}
    eid = 0
    for name, shape, n in _GROUPS:
        for _ in range(n):
            eid += 1
            first = len(coords) + 1
            coords += [(x + 3.0 * eid, y, 0.0) for x, y in shape]
            rows.setdefault(name, []).append(
                (eid, list(range(first, first + len(shape)))))
    xyz = np.array(coords, dtype=np.float64)
    ids = lambda v: np.array(list(v), dtype=np.int64)  # noqa: E731
    by_npe: dict[int, list[tuple[int, list[int]]]] = {}
    for name_rows in rows.values():
        for e, conn in name_rows:
            by_npe.setdefault(len(conn), []).append((e, conn))
    t6 = make_type_info(code=9, gmsh_name="Triangle 6", dim=2, order=2,
                        npe=6, count=len(by_npe[6]))
    q4 = make_type_info(code=3, gmsh_name="Quadrangle 4", dim=2, order=1,
                        npe=4, count=len(by_npe[4]))
    groups = {
        info.code: ElementGroup(
            element_type=info, ids=ids(e for e, _c in by_npe[info.npe]),
            connectivity=np.array([c for _e, c in by_npe[info.npe]],
                                  dtype=np.int64))
        for info in (t6, q4)
    }

    def pg(nodes: "list[int]", eles: "list[int]") -> dict[str, Any]:
        return {"node_ids": ids(nodes), "element_ids": ids(eles),
                "node_coords": xyz[[n - 1 for n in nodes]]}

    pgs: dict[tuple[int, int], dict[str, Any]] = {}
    for k, (name, name_rows) in enumerate(rows.items()):
        nodes = sorted({n for _e, conn in name_rows for n in conn})
        pgs[(2, 300 + k)] = {"name": name,
                             **pg(nodes, [e for e, _c in name_rows])}
    base = [i + 1 for i, (_x, y, _z) in enumerate(coords) if y == 0.0]
    pgs[(0, 399)] = {"name": "Base", **pg(base, [])}
    physical = PhysicalGroupSet(pgs)
    return FEMData(
        nodes=NodeComposite(node_ids=ids(range(1, len(coords) + 1)),
                            node_coords=xyz, physical=physical,
                            labels=LabelSet({})),
        elements=ElementComposite(groups=groups, physical=physical,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=len(coords), n_elems=eid, bandwidth=6,
                      types=[t6, q4]),
        model_name="stage_route_replay",
    )


def _chain(ops: apeSees, *, system: str = "UmfPack") -> dict[str, object]:
    return {
        "test": ops.test.NormUnbalance(tol=1e-6, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.RCM(),
        "system": getattr(ops.system, system)(),
        "analysis": ops.analysis.Static(),
    }


def _ops(mode: "str | None" = None) -> apeSees:
    target = None if mode is None else OpenSeesTarget(mode=mode)  # type: ignore[arg-type]
    ops = apeSees(_fem(), default_orientation=None, opensees=target)
    ops.model(ndm=2, ndf=2)
    return ops


def _md(ops: apeSees, name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ops.nDMaterial.ManzariDafalias(name=name, **_MD)


def _host(ops: apeSees, pg: str, mat: Any) -> None:
    if pg in ("A", "B", "C"):
        ops.element.SixNodeTri(pg=pg, thickness=1.0, material=mat)
    else:
        ops.element.FourNodeQuad(pg=pg, thickness=1.0, material=mat)


def _stage(ops: apeSees, name: str, body: "Callable[[Any], None]",
           *, system: str = "UmfPack") -> None:
    with ops.stage(name=name) as s:
        body(s)
        s.analysis(**_chain(ops, system=system))
        s.run(n_increments=1)


# ---------------------------------------------------------------------------
# The models: each REFUSED one is a flip the stage close now refuses
# ---------------------------------------------------------------------------


def _update_material_stage_parameter() -> apeSees:
    ops = _ops()
    sand = _md(ops, "sand")
    _host(ops, "A", sand)
    _stage(ops, "push", lambda s: s.update_parameter(
        "updateMaterialStage", 1.0, pg="A", material=sand))
    return ops


def _intercepting_host() -> apeSees:
    ops = _ops()
    sand = _md(ops, "sand")
    _host(ops, "P", sand)
    _stage(ops, "push", lambda s: s.update_parameter(
        "materialState", 1.0, pg="P", material=sand))
    return ops


def _sand_beside_clay() -> "tuple[apeSees, Any]":
    ops = _ops()
    sand = _md(ops, "sand")
    clay = ops.nDMaterial.DruckerPrager(name="clay", **_DP)
    _host(ops, "A", sand)
    _host(ops, "B", clay)
    return ops, sand


def _tag_blind_neighbour() -> apeSees:
    ops, sand = _sand_beside_clay()
    _stage(ops, "push", lambda s: s.update_parameter(
        "materialState", 1.0, elements=[1, 2], material=sand))
    return ops


def _target_without_the_material() -> apeSees:
    ops, sand = _sand_beside_clay()
    _stage(ops, "push", lambda s: s.update_parameter(
        "materialState", 1.0, pg="B", material=sand))
    return ops


def _part_of_a_class_wide_group() -> apeSees:
    ops = _ops()
    sand_a, sand_b = _md(ops, "sand_a"), _md(ops, "sand_b")
    _host(ops, "A", sand_a)
    _host(ops, "B", sand_b)
    _stage(ops, "push", lambda s: s.update_material_stage(
        materials=[sand_a], stage=1))
    return ops


def _multi_element_on_quads() -> apeSees:
    ops = _ops()
    sand = _md(ops, "sand")
    _host(ops, "Q", sand)
    _stage(ops, "push", lambda s: s.update_material_stage(
        materials=[sand], stage=1))
    return ops


def _flip_then_activate(*, system: str = "UmfPack", reflip: bool = False,
                        own_material: bool = False,
                        mode: "str | None" = None) -> apeSees:
    """Sand on ``A`` flipped plastic, then ``C`` activated.  With
    ``own_material``, ``C`` carries a second ManzariDafalias (the fork's
    constructor reset, #2443) instead of the same sand (#2445).  ``mode``
    pins the bridge's ``OpenSeesTarget``."""
    ops = _ops(mode)
    sand = _md(ops, "sand")
    _host(ops, "A", sand)
    _host(ops, "C", _md(ops, "sand_c") if own_material else sand)
    _stage(ops, "plastic", lambda s: s.update_material_stage(
        materials=[sand], stage=1))

    def install(s: Any) -> None:
        s.activate(pgs=["C"])
        if reflip:
            s.update_material_stage(materials=[sand], stage=1)

    _stage(ops, "install", install, system=system)
    return ops


REFUSED: dict[str, tuple[Callable[[], apeSees], str, str]] = {
    "updateMaterialStage parameter": (
        _update_material_stage_parameter, "#2460",
        "s.update_parameter('updateMaterialStage', 1.0) on pg 'A' does not "
        "carry its value"),
    "materialState through FourNodeQuad": (
        _intercepting_host, "#2460",
        "targets FourNodeQuad elements, which are not measured"),
    "tag-blind neighbour": (
        _tag_blind_neighbour, "#2458",
        "carry DruckerPrager (nDMaterial tag 2), whose setParameter takes "
        "materialState without checking the tag"),
    "target without the material": (
        _target_without_the_material, "#2460",
        "its targets (SixNodeTri on pg 'B') do not carry it"),
    "part of a class-wide group": (
        _part_of_a_class_wide_group, "but not ManzariDafalias (nDMaterial "
        "tag 2)", "ManzariDafalias::mElastFlag"),
    "multi-element flip on FourNodeQuad": (
        _multi_element_on_quads, "#2445",
        "but FourNodeQuad carries it, and that class drops the "
        "materialState update"),
    "activation after the plastic flip": (
        _flip_then_activate, "#2445",
        "SixNodeTri on pg 'C' carries nDMaterial tag 1, whose stage flag "
        "an earlier stage left plastic"),
    "fork constructor reset": (
        lambda: _flip_then_activate(system="Pardiso", own_material=True),
        "#2444", "reset the class-wide C++ static "
        "ManzariDafalias::mElastFlag to 0"),
}


# ---------------------------------------------------------------------------
# The models a stage close accepts: they must replay byte for byte
# ---------------------------------------------------------------------------


def valid_bridge(case: str) -> apeSees:
    """A staged flip the stage close accepts.  An R1b model: this module's
    tests run the oracle of ``tests/opensees/golden/test_roundtrip_oracle.py``
    on it."""
    if case == "both_routes":
        # One sand on two SixNodeTri, swept, then updated by materialState.
        ops = _ops()
        sand = _md(ops, "sand")
        _host(ops, "A", sand)
        _host(ops, "B", sand)
        _stage(ops, "gravity", lambda s: s.update_material_stage(
            materials=[sand], stage=0))
        _stage(ops, "push", lambda s: (
            s.update_material_stage(materials=[sand], stage=1),
            s.update_parameter("materialState", 1.0, elements=[1, 2],
                               material=sand)))
        return ops
    if case == "beside_clay":
        # DruckerPrager is tag-blind for materialState only; the flips
        # reach the sand's elements alone.
        ops, sand = _sand_beside_clay()
        _stage(ops, "push", lambda s: (
            s.update_material_stage(materials=[sand], stage=1),
            s.update_parameter("materialState", 1.0, pg="A",
                               material=sand)))
        return ops
    if case == "whole_group_elastic":
        # The whole class-wide group, at stage 0, on quads.
        ops = _ops()
        sand_a, sand_b = _md(ops, "sand_a"), _md(ops, "sand_b")
        _host(ops, "A", sand_a)
        _host(ops, "Q", sand_b)
        _stage(ops, "gravity", lambda s: s.update_material_stage(
            materials=[sand_a, sand_b], stage=0))
        return ops
    if case == "reflip_on_activation":
        return _flip_then_activate(reflip=True)
    if case == "own_material_on_stock":
        # The fork-reset model on a stock chain: not known to run the fork.
        return _flip_then_activate(own_material=True)
    raise KeyError(case)


VALID = ("both_routes", "beside_clay", "whole_group_elastic",
         "reflip_on_activation", "own_material_on_stock")


# ---------------------------------------------------------------------------
# Writing an archive, with the checks off or on
# ---------------------------------------------------------------------------


def _write(build: "Callable[[], apeSees]", path: Path,
           monkeypatch: pytest.MonkeyPatch, *, pre_fix: bool) -> Path:
    """``build``'s archive.  ``pre_fix``: written as the bridge did before
    the route checks, with ``check_stage_routes`` and the emit-time
    class-wide check switched off; the patch is undone before returning,
    so the replay runs the real checks."""
    with monkeypatch.context() as m:
        if pre_fix:
            m.setattr(_builder, "check_stage_routes",
                      lambda *a, **k: None)
            m.setattr(_apesees, "validate_class_wide_stage_flags",
                      lambda **k: None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ops = build()
            ops.h5(str(path))
    return path


def _replay(path: Path, target: str = "tcl", **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OpenSeesModel.from_h5(str(path)).build(target, **kwargs)


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_the_bridge_refuses_each_case(case: str, tmp_path: Path) -> None:
    """Each REFUSED model is one the current bridge refuses, at stage close
    or (the class-wide group) at emit, with the message the replay gives:
    the archive is pre-fix by construction.  The bridge knows it runs the
    fork only under a pinned ``mode="fork"``."""
    build, issue, phrase = REFUSED[case]
    if case == "fork constructor reset":
        def build() -> apeSees:
            return _flip_then_activate(system="Pardiso", own_material=True,
                                       mode="fork")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BridgeError) as exc:
            build().tcl(str(tmp_path / "deck.tcl"))
    msg = str(exc.value)
    assert issue in msg and phrase in msg, msg


@pytest.mark.parametrize("case", sorted(REFUSED))
@pytest.mark.parametrize("target", ["tcl", "py", "h5"])
def test_a_pre_fix_archive_is_refused_on_replay(
    case: str, target: str, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build, issue, phrase = REFUSED[case]
    path = _write(build, tmp_path / "pre_fix.h5", monkeypatch, pre_fix=True)
    kwargs = {"out": str(tmp_path / "rewrite.h5")} if target == "h5" else {}
    with pytest.raises(BridgeError) as exc:
        _replay(path, target, **kwargs)
    msg = str(exc.value)
    assert issue in msg and phrase in msg, msg
    assert not (tmp_path / "rewrite.h5").exists()
