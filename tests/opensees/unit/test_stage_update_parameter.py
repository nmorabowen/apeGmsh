"""``s.update_parameter(name, value, ...)`` — the typed pass-through
over the OpenSees ``parameter`` / ``addToParameter`` /
``updateParameter`` primitive.

Two target shapes, both addressed through an element because
``OPS_Parameter`` / ``OPS_addToParameter`` only accept ``node`` /
``element`` / ``region`` / ``loadPattern``:

* an ELEMENT parameter (``xPerm`` on ``LadrunoUP``) — no trailing arg;
* a MATERIAL parameter (``poissonRatio`` on ManzariDafalias) — the
  material tag rides as the trailing argv, which is what
  ``ManzariDafalias::setParameter`` matches on (``argv[1]``).

Locks the emitted deck text on both targets and both emit targets
(Tcl + openseespy), plus the emit slot.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    UpdateParameterRecord,
)
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_two_quad_fem() -> FEMStub:
    """Two quads in PG ``soil`` (fem eids 1, 2)."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
            ],
            node_pgs={"Left": [1, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "soil": _ElementGroupView(
                    ids=(1, 2), connectivity=((1, 2, 3, 4), (2, 5, 6, 3)),
                ),
            },
        ),
    )


def _full_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormUnbalance(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _soil_ops() -> tuple[apeSees, object]:
    """Bridge on the two-quad fem; returns (ops, material handle)."""
    ops = apeSees(_make_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="soil", thickness=1.0, material=mat)
    return ops, mat


def _ops_tags_of_soil(lines: list[str]) -> list[int]:
    """OpenSees element tags in emit order, read back from the deck."""
    return [
        int(ln.split()[2]) for ln in lines if ln.startswith("element ")
    ]


# ===========================================================================
# Builder surface
# ===========================================================================


def test_element_parameter_records_no_material_tag() -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="perm") as s:
        rec = s.update_parameter("xPerm", 1e-5, pg="soil")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert rec == UpdateParameterRecord(
        name="xPerm", value=1e-5, pg="soil", elements=None, mat_tag=None,
    )
    assert ops._stage_records[0].update_parameter_records == (rec,)


def test_material_parameter_records_the_material_tag() -> None:
    ops, mat = _soil_ops()
    with ops.stage(name="soften") as s:
        rec = s.update_parameter(
            "poissonRatio", 0.35, pg="soil", material=mat,
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert rec.mat_tag == ops.tag_for(mat)
    assert rec.name == "poissonRatio"


def test_rejects_both_pg_and_elements() -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="exactly one of pg= or elements="):
            s.update_parameter("xPerm", 1.0, pg="soil", elements=[1])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_rejects_neither_pg_nor_elements() -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="exactly one of pg= or elements="):
            s.update_parameter("xPerm", 1.0)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_rejects_empty_name() -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="name= must be non-empty"):
            s.update_parameter("", 1.0, pg="soil")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_unregistered_material_fails_loud() -> None:
    """A material with no bridge tag would put nothing in the argv and
    the update would silently no-op."""
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    ops, _mat = _soil_ops()
    stray = ElasticIsotropic(E=1.0, nu=0.1, rho=0.0)
    with ops.stage(name="bad") as s:
        with pytest.raises(BridgeError, match="never registered"):
            s.update_parameter(
                "poissonRatio", 0.35, pg="soil", material=stray,
            )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


# ===========================================================================
# Tcl deck text
# ===========================================================================


def test_tcl_element_parameter_block() -> None:
    """``xPerm`` on LadrunoUP-style element parameters: name only, no
    trailing tag (fork ``LadrunoUP::setParameter`` matches argv[0])."""
    ops, _mat = _soil_ops()
    with ops.stage(name="perm") as s:
        s.update_parameter("xPerm", 1e-5, pg="soil")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()
    e1, e2 = _ops_tags_of_soil(lines)
    block = [
        ln for ln in lines
        if ln.startswith(("parameter ", "addToParameter ",
                          "updateParameter ", "remove parameter "))
    ]
    pid = int(block[0].split()[1])
    assert block == [
        f"parameter {pid}",
        f"addToParameter {pid} element {e1} xPerm",
        f"addToParameter {pid} element {e2} xPerm",
        f"updateParameter {pid} 1e-05",
        f"remove parameter {pid}",
    ]


def test_tcl_material_parameter_block_carries_the_material_tag() -> None:
    """``poissonRatio`` is a MATERIAL parameter: the element forwards the
    argv tail to its GP materials, and ManzariDafalias matches on
    ``argv[1] == its own tag`` — so the tag must be in the line."""
    ops, mat = _soil_ops()
    mat_tag = ops.tag_for(mat)
    with ops.stage(name="soften") as s:
        s.update_parameter("poissonRatio", 0.35, pg="soil", material=mat)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()
    e1, e2 = _ops_tags_of_soil(lines)
    add = [ln for ln in lines if ln.startswith("addToParameter ")]
    pid = int(add[0].split()[1])
    assert add == [
        f"addToParameter {pid} element {e1} poissonRatio {mat_tag}",
        f"addToParameter {pid} element {e2} poissonRatio {mat_tag}",
    ]
    assert f"updateParameter {pid} 0.35" in lines


def test_tcl_elements_form_targets_only_the_listed_eids() -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="perm") as s:
        s.update_parameter("yPerm", 2.0, elements=[2])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()
    _e1, e2 = _ops_tags_of_soil(lines)
    add = [ln for ln in lines if ln.startswith("addToParameter ")]
    assert len(add) == 1
    assert add[0].split()[3:] == [str(e2), "yPerm"]


def test_tcl_two_records_get_distinct_parameter_tags() -> None:
    ops, mat = _soil_ops()
    with ops.stage(name="both") as s:
        s.update_parameter("xPerm", 1.0, pg="soil")
        s.update_parameter("poissonRatio", 0.35, pg="soil", material=mat)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    decl = [
        int(ln.split()[1]) for ln in emitter.lines()
        if ln.startswith("parameter ")
    ]
    assert len(decl) == 2
    assert decl[0] != decl[1]


def test_tcl_unknown_element_id_fails_loud() -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="perm") as s:
        s.update_parameter("xPerm", 1.0, elements=[999])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="not registered with any Element"):
        ops.build().emit(TclEmitter())


# ===========================================================================
# openseespy deck text
# ===========================================================================


def test_py_emits_the_same_block() -> None:
    ops, mat = _soil_ops()
    mat_tag = ops.tag_for(mat)
    with ops.stage(name="soften") as s:
        s.update_parameter("poissonRatio", 0.35, elements=[1], material=mat)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = PyEmitter()
    ops.build().emit(emitter)
    lines = [ln.strip() for ln in emitter.lines()]
    add = [ln for ln in lines if ln.startswith("ops.addToParameter(")]
    assert len(add) == 1
    pid = int(add[0].split("(")[1].split(",")[0])
    assert f"ops.parameter({pid})" in lines
    assert add[0].endswith(f", 'poissonRatio', {mat_tag})")
    assert f"ops.updateParameter({pid}, 0.35)" in lines
    assert f"ops.remove('parameter', {pid})" in lines


# ===========================================================================
# Emit slot
# ===========================================================================


def test_update_parameter_emits_after_the_chain_and_before_analyze() -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="perm") as s:
        s.update_parameter("xPerm", 1.0, pg="soil")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    up_i = names.index("update_parameter")
    assert names.index("domain_change") < up_i < names.index("analyze")


def test_update_parameter_is_scoped_to_its_own_stage() -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="a") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="b") as s:
        s.update_parameter("xPerm", 1.0, pg="soil")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    stage_idx = -1
    owner: list[int] = []
    for name, _args, _kw in rec.calls:
        if name == "stage_open":
            stage_idx += 1
        elif name == "update_parameter":
            owner.append(stage_idx)
    assert owner == [1]


def test_absent_verb_emits_nothing() -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="plain") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    assert "update_parameter" not in [c[0] for c in rec.calls]


# ===========================================================================
# H5 archival — deferred, fail-loud
# ===========================================================================


def test_h5_archive_refuses_a_stage_that_updates_a_parameter(tmp_path) -> None:
    ops, _mat = _soil_ops()
    with ops.stage(name="perm") as s:
        s.update_parameter("xPerm", 1.0, pg="soil")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(NotImplementedError, match="s.update_parameter"):
        ops.h5(str(tmp_path / "model.h5"))
