"""A rotated instance turns its authored loads, never its weight (#1597).

The maintainer's ruling on #1597 (option A), recorded in ADR 0117's load
note: a force, moment, line load or pressure direction authored in the
module turns by ``R v`` and never translates; gravity ``g`` and a body
force ``bf`` stay global, so rotating a module never tilts its weight; DOF
indices never transform. Before, the merge copied every load vector from
the source.

Oracles, each independent of the code under test:

* closed form: ``R`` is 90 degrees about +x, ``(x, y, z) -> (x, -z, y)``.
  One unit test per field family on records built by hand.
* lock: every field of every load record kind, and every element-load
  parameter, is classified; an unclassified one raises.
* deck: a v2 block instance turned 90 degrees about x, with a self-weight
  of ``rho g V = 80`` along -Z and a tip force ``(0, 5, 0)``, writes its
  weight along -Z (summing to 80) and its force as ``(0, 0, 5)``.
* live (stock openseespy, in a subprocess): the base reactions of that
  instance sum to ``-(F + W) = (0, 0, 75)``. A turned weight would put 80
  on Y; an unturned force would put 5 on Y.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.records import _loads as LR
from apeGmsh.mesh._compose import (
    ComposeError,
    _ELEMENT_LOAD_PARAMS,
    _LOAD_FIELDS,
    _place_load_record,
)

ROT_X_90 = (1.0, 0.0, 0.0, math.pi / 2.0)
S = 2.0
RHO, G = 1.0, 10.0
W = RHO * G * S ** 3          # 80, along -Z
F_SRC = (0.0, 5.0, 0.0)


def _r(v) -> tuple[float, ...]:
    """The closed-form ``R v``: 90 degrees about +x."""
    x, y, z = v
    return (x, -z, y)


def _place(rec):
    return _place_load_record(rec, rotate=ROT_X_90)


# ---------------------------------------------------------------------------
# One unit test per field family (closed form)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["point", "point_closest", "line",
                                    "surface", "face_load"])
def test_authored_nodal_force_and_moment_turn(source):
    rec = LR.NodalLoadRecord(pattern="p", node_id=1, force_xyz=(1.0, 2.0, 3.0),
                             moment_xyz=(0.0, 4.0, 0.0), source=source)
    got = _place(rec)
    assert got.force_xyz == pytest.approx(_r((1.0, 2.0, 3.0)), abs=1e-12)
    assert got.moment_xyz == pytest.approx(_r((0.0, 4.0, 0.0)), abs=1e-12)
    assert got.node_id == 1


@pytest.mark.parametrize("source", ["gravity", "body"])
def test_reduced_self_weight_stays_global(source):
    rec = LR.NodalLoadRecord(pattern="p", node_id=1, force_xyz=(0.0, 0.0, -9.0),
                             source=source)
    assert _place(rec).force_xyz == (0.0, 0.0, -9.0)


def test_unknown_source_refuses_a_rotation_but_not_a_translation():
    rec = LR.NodalLoadRecord(pattern="p", node_id=1, force_xyz=(1.0, 0.0, 0.0),
                             source=None)
    with pytest.raises(ComposeError, match="source None"):
        _place(rec)
    assert _place_load_record(rec, rotate=None) is rec


def test_element_load_directions_turn_and_weight_stays():
    beam = LR.ElementLoadRecord(pattern="p", element_id=3, load_type="beamUniform",
                                params={"wx": 1.0, "wy": 2.0, "wz": 3.0})
    got = _place(beam).params
    assert (got["wx"], got["wy"], got["wz"]) == pytest.approx(_r((1.0, 2.0, 3.0)),
                                                              abs=1e-12)
    pres = LR.ElementLoadRecord(pattern="p", element_id=3, load_type="surfacePressure",
                                params={"p": 2.0, "normal": False,
                                        "direction": [0.0, 1.0, 0.0]})
    got = _place(pres).params
    assert got["direction"] == pytest.approx(_r((0.0, 1.0, 0.0)), abs=1e-12)
    assert (got["p"], got["normal"]) == (2.0, False)
    for params in ({"g": (0.0, 0.0, -9.81), "density": 2.0},
                   {"bf": (0.0, 0.0, -3.0)}):
        body = LR.ElementLoadRecord(pattern="p", element_id=3,
                                    load_type="bodyForce", params=params)
        assert _place(body).params == params


def test_sp_dof_and_value_never_transform():
    rec = LR.SPRecord(pattern="p", node_id=1, dof=2, value=0.5)
    got = _place(rec)
    assert (got.dof, got.value) == (2, 0.5)


def test_unclassified_load_type_or_param_raises():
    with pytest.raises(TypeError, match="_ELEMENT_LOAD_PARAMS"):
        _place(LR.ElementLoadRecord(pattern="p", element_id=1,
                                    load_type="beamPoint", params={"P": 1.0}))
    with pytest.raises(TypeError, match="not classified"):
        _place(LR.ElementLoadRecord(pattern="p", element_id=1,
                                    load_type="bodyForce",
                                    params={"g": (0, 0, -1), "tilt": 1.0}))


# ---------------------------------------------------------------------------
# Lock: every load kind and field is classified
# ---------------------------------------------------------------------------

def test_every_load_kind_and_field_is_classified():
    kinds = {cls.__name__: cls for cls in vars(LR).values()
             if isinstance(cls, type) and issubclass(cls, LR.LoadRecord)
             and cls is not LR.LoadRecord}
    assert set(kinds) == set(_LOAD_FIELDS)
    for name, cls in kinds.items():
        fields = {f.name for f in dataclasses.fields(cls)}
        assert fields == set(_LOAD_FIELDS[name]), (name, fields ^ set(_LOAD_FIELDS[name]))
    assert set(_ELEMENT_LOAD_PARAMS) == {"beamUniform", "surfacePressure", "bodyForce"}
    # Gravity and a body force are never turned (the ruling on #1597).
    turned = {k for grps, _ in _ELEMENT_LOAD_PARAMS.values() for g in grps for k in g}
    assert not turned & {"g", "bf"}


# ---------------------------------------------------------------------------
# Deck and live: a rotated block carrying its weight and a tip force
# ---------------------------------------------------------------------------

def _archive(workdir: Path) -> Path:
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name="blk", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, S, S, S, label="v")
        g.physical.add_volume("v", name="Vol")
        g.model.sync()
        tip = g.model.select(None, dim=0).in_box(
            (S - 0.1,) * 3, (S + 0.1,) * 3).result().tags()
        g.physical.add(0, tip, name="Tip")
        bot = g.model.select(None, dim=2).in_box(
            (-0.1, -0.1, -0.1), (S + 0.1, S + 0.1, 0.1)).result().tags()
        g.physical.add_surface(bot, name="bot")
        with g.loads.case("dead"):
            g.loads.gravity("Vol", g=(0.0, 0.0, -G), density=RHO)
            g.loads.point.force("Tip", force=F_SRC)
        g.mesh.recipe.structured(size=1.0, fallback="strict")
        fem = g.mesh.queries.get_fem_data(dim=None)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1000.0, nu=0.0, name="m")
    ops.element.stdBrick(pg="Vol", material=mat)
    path = workdir / "blk.h5"
    ops.h5(str(path))
    return path


def _bridge(workdir: Path):
    from apeGmsh.assembly import Assembly

    ops = (Assembly("a")
           .instance("c", _archive(workdir), rotate=((1.0, 0.0, 0.0), math.pi / 2),
                     translate=(10.0, 20.0, 30.0))
           .bridge(ndm=3, ndf=3))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.from_model("dead")
    return ops


def test_rotated_instance_deck_keeps_weight_and_turns_force(tmp_path):
    ops = _bridge(tmp_path)
    (tip,) = (int(i) for i in ops.fem.nodes.select(pg="c.Tip").ids)
    deck = tmp_path / "a.tcl"
    ops.tcl(str(deck), flat=True)
    total = np.zeros(3)
    at_tip = np.zeros(3)
    for ln in deck.read_text(encoding="utf-8").splitlines():
        tok = ln.split()
        if tok[:1] == ["load"]:
            vec = np.array([float(v) for v in tok[2:5]])
            total += vec
            if int(tok[1]) == tip:
                at_tip += vec
    np.testing.assert_allclose(total, np.array(_r(F_SRC)) + (0.0, 0.0, -W), atol=1e-9)
    # The tip carries its share of the weight plus the turned force.
    assert at_tip[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert at_tip[2] > _r(F_SRC)[2] - W


def _solve(workdir: Path) -> dict:
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    ops = _bridge(workdir)
    base = sorted(int(i) for i in ops.fem.nodes.select(pg="c.bot").ids)
    ops.fix(nodes=base, dofs=(1, 1, 1))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    live = emitter.ops
    live.reactions()
    return {"R": [sum(live.nodeReaction(t, d) for t in base) for d in (1, 2, 3)]}


def _main() -> None:
    """``python -c`` entry: argv[1] is the work directory."""
    print("RESULT " + json.dumps(_solve(Path(sys.argv[1]))))


@pytest.mark.live
def test_rotated_instance_reactions_balance_weight_and_turned_force(tmp_path):
    from apeGmsh.opensees.emitter.live import _get_ops

    try:
        _get_ops()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), env.get("PYTHONPATH", "")])
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    proc = subprocess.run(
        [sys.executable, "-W", "ignore::UserWarning", "-c",
         "from tests.assembly.test_instance_load_frame import _main; _main()",
         str(tmp_path)],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"subprocess failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    reactions = json.loads(lines[-1][len("RESULT "):])["R"]
    want = -(np.array(_r(F_SRC)) + (0.0, 0.0, -W))     # (0, 0, 75)
    np.testing.assert_allclose(reactions, want, atol=1e-8)


# ---------------------------------------------------------------------------
# Routing: element-form loads through a rotated instance
# ---------------------------------------------------------------------------

def test_rotated_instance_turns_line_loads_and_keeps_body_force(tmp_path):
    from apeGmsh.assembly import Assembly
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.opensees import apeSees

    path = tmp_path / "elem_loads.h5"
    with apeGmsh(model_name="el", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, S, S, S, label="v")
        p0 = g.model.geometry.add_point(0.0, 0.0, 5.0)
        p1 = g.model.geometry.add_point(4.0, 0.0, 5.0)
        beam = g.model.geometry.add_line(p0, p1)
        g.model.sync()
        g.physical.add_volume("v", name="Vol")
        g.physical.add(1, [beam], name="Beam")
        with g.loads.case("dead"):
            g.loads.gravity("Vol", g=(0.0, 0.0, -G), density=RHO,
                            target_form="element")
            g.loads.line("Beam", q_xyz=F_SRC, target_form="element")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.mesh.queries.get_fem_data(dim=None).to_h5(str(path))

    src = list(FEMData.from_h5(str(path)).elements.loads)
    ops = apeSees(FEMData.from_h5(str(path)))
    ops.model(ndm=3, ndf=6)
    archive = tmp_path / "elem_loads_model.h5"
    ops.h5(str(archive))
    # The module at the identity, then a rotated copy: the merge appends
    # the copy's rows after the first instance's.
    fem = (Assembly("el")
           .instance("host", archive)
           .instance("m", archive, rotate=(ROT_X_90[:3], ROT_X_90[3]))
           .bridge(ndm=3, ndf=6)).fem
    rows = list(fem.elements.loads)
    assert len(rows) == 2 * len(src)
    by_type = {"beamUniform": 0, "bodyForce": 0}
    for s, got in zip(src, rows[len(src):]):
        by_type[s.load_type] += 1
        if s.load_type == "beamUniform":
            w = (got.params["wx"], got.params["wy"], got.params["wz"])
            assert w == pytest.approx(_r(F_SRC), abs=1e-12)
        else:
            assert tuple(got.params["g"]) == (0.0, 0.0, -G)
    assert by_type["beamUniform"] and by_type["bodyForce"], by_type
