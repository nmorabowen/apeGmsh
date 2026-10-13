"""The side-channel verbs' ``/opensees/program`` ``decl`` (#1853, opensees 2.32.0).

``step_hook_ramp``, ``addToParameter``, ``flip_element_stage``,
``update_parameter``, ``set_node_vel`` and ``set_node_accel`` write
through the K1-6 side-channel stores, whose records' declarations the
bridge hands in after the emit. From 2.32.0 the stage emit
(``_internal/build/stage_emit.py``) opens each record's declaration
around the calls that record drives, so the tape records it like a
``fix``: a proof at emit time, never a guess.

Oracles:

* **The archive, not the writer.** :func:`_expected` (the A7 table join)
  finds each side-channel call's record from the record's own stored
  data (its ``param_tags`` or ``nodes``) and expects that record's
  ``/opensees/decls/rows`` entry. Driven over flat, staged and
  partitioned models with every verb, and a stage whose zeroings
  overlap; every side-channel call must be proved.
* **First record wins.** A node two ``s.zero_velocities`` records name
  is zeroed once, as the first one's; the per-record split concatenates
  to the flat node list, so the deck is unchanged.
* **Nothing leaks.** After a record's calls the declaration is closed:
  the next ``/opensees/commands`` row carries ``-1``, also when a call
  inside the bracket raised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator, cast

import pytest

from apeGmsh.opensees._internal.build import stage_emit
from apeGmsh.opensees._internal.build.records import ZeroVelocityRecord
from apeGmsh.opensees.apesees import _ArchiveDeclCursor, apeSees
from apeGmsh.opensees.emitter.h5 import H5Emitter
from tests.opensees.contract.test_tag_rank_invariance import _fem
from tests.opensees.h5.test_h5_decls import _param_stage_model
from tests.opensees.h5.test_h5_param_tags import (
    _absorb_and_stress_bridge,
    _flat_quad_initial_stress_bridge,
    _update_then_stress_bridge,
    _zero_some_bridge,
)
from tests.opensees.h5.test_h5_program_decl import (
    SIDE,
    Call,
    _decls_by_method,
    _expected,
    spy,
)
from tests.opensees.h5.test_h5_stages_reader import _chain, build_two_quad_fem

__all__ = ["spy"]  # the fixture, re-exported for this module's tests


def _overlap_bridge() -> apeSees:
    """One stage, three zeroings: ``[3, 4]``, the whole domain, ``[4]``.
    Nodes 3 and 4 are the first record's, every other node the second's,
    and the third record zeroes nothing."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="static") as s:
        s.zero_velocities(nodes=[3, 4])
        s.zero_velocities()
        s.zero_velocities(nodes=[4])
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    return ops


def _partitioned_staged() -> apeSees:
    """Two ranks (the rank-invariance model's blocks): a global initial
    stress, then a stage with its own initial stress, an absorbing flip,
    two updates and three overlapping zeroings, each rank writing its
    share inside its partition block."""
    ops = apeSees(cast(Any, _fem(2, contact=False)))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6)
    ops.uniaxialMaterial.ElasticMaterial(E=2.0e5, name="steel")  # rebar
    ops.element.Truss(pg="Bars", A=0.01, material=mat)
    ops.initial_stress(name="g", pg="Bars", sigma_xx=-1.0, sigma_yy=-2.0,
                       sigma_zz=-3.0, ramp_steps=2)
    with ops.stage(name="s1") as s:
        s.initial_stress(name="st", pg="Bars", sigma_xx=-4.0,
                         sigma_yy=-5.0, sigma_zz=-6.0, ramp_steps=3)
        s.activate_absorbing(elements=[41, 11])
        s.update_parameter("A", 0.02, pg="Bars")
        s.update_parameter("E", 2.0e6, elements=[40])
        s.zero_velocities(nodes=[101, 102, 401])
        s.zero_velocities(nodes=[102, 301, 401])
        s.zero_velocities()
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-4, max_iter=10),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Transformation(),
            numberer=ops.numberer.ParallelPlain(),
            system=ops.system.Mumps(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    return ops


MODELS: dict[str, Callable[[], apeSees]] = {
    "param_stage": _param_stage_model,
    "update_then_stress": _update_then_stress_bridge,
    "flat_initial_stress": _flat_quad_initial_stress_bridge,
    "absorb_and_stress": _absorb_and_stress_bridge,
    "zero_some": _zero_some_bridge,
    "zero_overlap": _overlap_bridge,
    "partitioned_staged": _partitioned_staged,
}


def _side_joins(
    tmp_path: Path, spy: list[Call], name: str,
) -> Iterator[tuple[str, int, int, int]]:
    """``(verb, stage, got, want)`` for each side-channel call of model
    ``name``, after asserting the whole emit joins (:func:`_expected`)."""
    spy.clear()
    p = tmp_path / f"{name}.h5"
    MODELS[name]().h5(str(p))
    got, want, where = _expected(p, spy)
    assert got == want, name
    for g, w, (verb, stage) in zip(got, want, where):
        if verb in SIDE:
            yield verb, stage, g, w


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_side_channel_call_joins_its_record(
    tmp_path: Path, spy: list[Call], name: str,
) -> None:
    """Each side-channel call's decl is its archived record's, and every
    one is proved (none is ``-1``)."""
    joins = list(_side_joins(tmp_path, spy, name))
    assert joins, name
    assert all(w != -1 for _v, _s, _g, w in joins), joins


def test_every_side_channel_verb_is_exercised(
    tmp_path: Path, spy: list[Call],
) -> None:
    """The oracle is not vacuous: over the models every one of the six
    verbs is called and proved, on a flat and on a partitioned emit."""
    seen: dict[str, set[str]] = {}
    for name in MODELS:
        for verb, _stage, _got, want in _side_joins(tmp_path, spy, name):
            assert want != -1
            seen.setdefault(verb, set()).add(name)
    assert set(seen) == SIDE
    partitioned = {v for v, names in seen.items()
                   if "partitioned_staged" in names}
    assert partitioned == SIDE


def test_a_node_two_zeroings_name_is_the_first_ones(
    tmp_path: Path, spy: list[Call],
) -> None:
    """``[3, 4]``, the whole domain, ``[4]``: nodes 3 and 4 carry the first
    record's declaration, the rest the second's, the third none."""
    p = tmp_path / "overlap.h5"
    _overlap_bridge().h5(str(p))
    got, want, where = _expected(p, spy)
    assert got == want
    from apeGmsh.opensees.emitter import h5_reader

    with h5_reader.open(str(p)) as m:
        rows = list(m.declarations().rows[
            "stages/stage_000/zero_velocities"])
    assert len(set(rows)) == 3
    by_node: dict[int, set[int]] = {}
    for (verb, args, _kw), decl in zip(spy, got):
        if verb in ("set_node_vel", "set_node_accel"):
            by_node.setdefault(int(args[0]), set()).add(decl)
    assert by_node.pop(3) == by_node.pop(4) == {rows[0]}
    assert by_node and all(d == {rows[1]} for d in by_node.values())


def test_the_per_record_split_keeps_the_flat_order() -> None:
    """Concatenated, the per-record node lists are the flat list (the deck
    is unchanged), and each node sits under the first record naming it."""
    recs = [ZeroVelocityRecord(nodes=(5, 3)),
            ZeroVelocityRecord(nodes=None),
            ZeroVelocityRecord(nodes=(3, 9))]
    domain = (1, 3, 5, 7)
    split = stage_emit.zero_velocity_targets_by_record(recs, domain)
    assert [rec for rec, _n in split] == recs
    assert [nodes for _r, nodes in split] == [[5, 3], [1, 7], [9]]
    assert [n for _r, nodes in split for n in nodes] == (
        stage_emit.zero_velocity_target_nodes(recs, domain))


class _Ramp:
    """The fields :func:`stage_emit.write_planned_ramps` reads."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.sigma_xx = self.sigma_yy = self.sigma_zz = -1.0
        self.lambda_install = 1.0
        self.ramp_steps = 2


def _archive() -> H5Emitter:
    e = H5Emitter()
    e.model(ndm=2, ndf=2)
    e.set_declarations(
        [("opensees/initial_stress/a", "initial_stress", "", False),
         ("opensees/initial_stress/b", "initial_stress", "", False)],
        [],
    )
    return e


def test_the_next_commands_row_carries_minus_one() -> None:
    """A ramp's bracket closes: the ``/opensees/commands`` row written
    after it (none open) carries ``-1``, the ramps their own decls."""
    e = _archive()
    a, b = _Ramp("a"), _Ramp("b")
    cursor = _ArchiveDeclCursor(e, {id(a): 0, id(b): 1})
    stage_emit.write_planned_ramps(e, [
        stage_emit.PlannedParameter(a, None, "step_hook_ramp", (1, 2, 3)),
        stage_emit.PlannedParameter(b, None, "step_hook_ramp", (4, 5, 6)),
    ], cursor=cursor)
    e.modal_damping(0.05)
    got = _decls_by_method(e)
    assert got["step_hook_ramp"] == [0, 1]
    assert got["modal_damping"] == [-1]
    assert e._decl_open == -1


def test_a_raising_call_still_closes_its_bracket() -> None:
    """A ramp that refuses (a second ramp of one name with other tags)
    leaves no declaration open behind it."""
    e = _archive()
    a, b = _Ramp("a"), _Ramp("a")
    cursor = _ArchiveDeclCursor(e, {id(a): 0, id(b): 1})
    with pytest.raises(RuntimeError, match="ramps twice"):
        stage_emit.write_planned_ramps(e, [
            stage_emit.PlannedParameter(a, None, "step_hook_ramp", (1, 2, 3)),
            stage_emit.PlannedParameter(b, None, "step_hook_ramp", (4, 5, 6)),
        ], cursor=cursor)
    assert e._decl_open == -1
    e.modal_damping(0.05)
    assert _decls_by_method(e)["modal_damping"] == [-1]


def test_the_no_op_cursor_opens_nothing() -> None:
    """``compose.py``'s replays pass no cursor: the ramp carries ``-1``."""
    e = _archive()
    stage_emit.write_planned_ramps(e, [stage_emit.PlannedParameter(
        _Ramp("a"), None, "step_hook_ramp", (1, 2, 3))])
    assert _decls_by_method(e)["step_hook_ramp"] == [-1]
