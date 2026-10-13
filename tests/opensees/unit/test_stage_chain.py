"""R3-S1 (#1785): the shared stage chain, the Static ``dt`` rule (D1),
and a stage ``Linear`` without a test.

Oracles:

* **Byte identity.** The staged strip footing of the skill's
  ``model-scripts.md`` emits the same py and tcl deck whether its stages
  take the chain as a dict splat or as ``ops.analysis.chain``.
* **D1 (warn-iff).** Under ``Static`` + ``LoadControl`` a ``dt`` that
  differs from ``dlam`` raises :class:`InertStaticDtWarning` at stage
  close, naming the stage, ``dt`` and ``dlam``; an equal or omitted ``dt``
  and a ``Transient`` stage are silent (asserted with the warning turned
  into an error).  Run with ``-W error::apeGmsh.opensees.InertStaticDtWarning``.
* **Linear without a test.** The stage emits no ``test`` line, the H5
  archive replays to the same deck, and a strategy ladder still needs a
  test.  The live half (same periods and drifts) is in
  ``tests/opensees/live/test_stage_linear_without_test_live.py``.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import InertStaticDtWarning, OpenSeesModel
from apeGmsh.opensees.analysis.algorithm import (
    BFGS,
    Broyden,
    KrylovNewton,
    Linear,
    ModifiedNewton,
    Newton,
    NewtonLineSearch,
)
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.stage import StageChain

from tests.opensees.unit.test_label_selector_fix_merge import (
    build_footing_fem as build_plate_fem,
)
from tests.opensees.unit.test_label_selector_fix_merge import footing as plate


# ---------------------------------------------------------------------------
# The staged strip footing (skill model-scripts.md, "Staged strip footing")
# ---------------------------------------------------------------------------

HALF_SOIL, SOIL_DEPTH, HALF_FOOTING = 8.0, 8.0, 1.0
N_GEO, N_FOOT = 10, 30


def _footing_fem() -> Any:
    with apeGmsh(model_name="r3s1_footing", verbose=False) as g:
        geo = g.model.geometry
        geo.add_point(-HALF_SOIL, -SOIL_DEPTH, 0.0, label="base_left")
        geo.add_point(HALF_SOIL, -SOIL_DEPTH, 0.0, label="base_right")
        geo.add_point(HALF_SOIL, 0.0, 0.0, label="ground_right")
        geo.add_point(HALF_FOOTING, 0.0, 0.0, label="footing_right_edge")
        geo.add_point(0.0, 0.0, 0.0, label="footing_centre")
        geo.add_point(-HALF_FOOTING, 0.0, 0.0, label="footing_left_edge")
        geo.add_point(-HALF_SOIL, 0.0, 0.0, label="ground_left")
        geo.add_line("base_left", "base_right", label="base")
        geo.add_line("base_right", "ground_right", label="right_side")
        geo.add_line("ground_right", "footing_right_edge", label="grs")
        geo.add_line("footing_right_edge", "footing_centre", label="frh")
        geo.add_line("footing_centre", "footing_left_edge", label="flh")
        geo.add_line("footing_left_edge", "ground_left", label="gls")
        geo.add_line("ground_left", "base_left", label="left_side")
        geo.add_curve_loop(
            ["base", "right_side", "grs", "frh", "flh", "gls", "left_side"],
            label="soil_outline",
        )
        geo.add_plane_surface("soil_outline", label="soil")
        g.physical.add_surface("soil", name="soil")
        g.physical.add_curve("base", name="base")
        g.physical.add_curve(["left_side", "right_side"], name="sides")
        g.physical.add_curve(["flh", "frh"], name="footing")
        g.physical.add_point("footing_centre", name="footing_centre")
        g.constraints.bc("base", dofs=[1, 1])
        g.constraints.bc("sides", dofs=[1, 0])
        with g.loads.case("self_weight"):
            g.loads.gravity("soil", g=(0.0, -9.81, 0.0), density=1.8)
        with g.loads.case("footing_pressure"):
            g.loads.line(
                "footing", magnitude=300.0, direction=(0.0, -1.0, 0.0),
                reduction="consistent",
            )
        near = g.mesh.field.distance(curves="footing")
        grading = g.mesh.field.threshold(
            near, size_min=0.5, size_max=2.0, dist_min=1.0, dist_max=8.0,
        )
        g.mesh.field.set_background(grading)
        g.mesh.generation.generate(dim=2)
        g.mesh.generation.set_order(2)
        return g.mesh.queries.get_fem_data(dim=2)


@pytest.fixture(scope="module")
def footing_fem() -> Any:
    return _footing_fem()


def _footing(fem: Any, form: str) -> apeSees:
    """The footing, its two stages sharing one chain in ``form``."""
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    clay = ops.nDMaterial.J2Plasticity(
        K=2.0e4, G=6.0e3, sig0=60.0, sigInf=60.0, delta=0.0, H=0.0,
    )
    ops.element.SixNodeTri(
        pg="soil", thickness=1.0, material=clay, plane_type="PlaneStrain",
    )
    ops.fix_from_model()
    ops.recorder.Node(
        file="uy.out", response="disp", pg="footing_centre", dofs=(2,),
    )
    slots: dict[str, Any] = dict(
        test=ops.test.NormDispIncr(tol=1e-8, max_iter=50),
        algorithm=ops.algorithm.Newton(),
        constraints=ops.constraints.Plain(),
        numberer=ops.numberer.RCM(),
        system=ops.system.UmfPack(),
        analysis=ops.analysis.Static(),
    )
    solver = ops.analysis.chain(**slots) if form == "chain" else None
    for name, case, n in (
        ("geostatic", "self_weight", N_GEO),
        ("footing", "footing_pressure", N_FOOT),
    ):
        step = 1.0 / n
        with ops.stage(name=name) as s:
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.from_model(case)
            integ = ops.integrator.LoadControl(dlam=step)
            if solver is None:
                s.analysis(integrator=integ, **slots)
            else:
                s.analysis(solver, integrator=integ)
            s.run(n_increments=n, dt=step)
    return ops


def _deck(ops: apeSees, kind: str) -> str:
    emitter = PyEmitter() if kind == "py" else TclEmitter()
    ops.build().emit(emitter)
    return "\n".join(emitter.lines())


@pytest.mark.parametrize("kind", ["py", "tcl"])
def test_footing_deck_is_byte_identical_with_chain(
    footing_fem: Any, kind: str,
) -> None:
    splat = _deck(_footing(footing_fem, "splat"), kind)
    chained = _deck(_footing(footing_fem, "chain"), kind)
    assert chained == splat
    # The chain reached both stages: one Newton line per stage.
    assert splat.count("Newton") == 2


@pytest.mark.parametrize("kind", ["py", "tcl"])
def test_footing_files_are_byte_identical_with_chain(
    footing_fem: Any, kind: str, tmp_path: Path,
) -> None:
    """The user-facing writers (``ops.py`` / ``ops.tcl``) agree too."""
    paths = []
    for form in ("splat", "chain"):
        ops = _footing(footing_fem, form)
        path = tmp_path / f"{form}.{kind}"
        getattr(ops, kind)(str(path), progress=False)
        paths.append(path)
    assert paths[0].read_bytes() == paths[1].read_bytes()


# ---------------------------------------------------------------------------
# ops.analysis.chain: a value object, validated
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plate_fem() -> Any:
    return build_plate_fem()


def _slots(ops: apeSees, **over: Any) -> dict[str, Any]:
    slots: dict[str, Any] = dict(
        test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
        algorithm=ops.algorithm.Newton(),
        constraints=ops.constraints.Plain(),
        numberer=ops.numberer.RCM(),
        system=ops.system.BandGeneral(),
        analysis=ops.analysis.Static(),
    )
    slots.update(over)
    return slots


def test_chain_registers_nothing(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    slots = _slots(ops)
    before = len(ops._primitives)
    solver = ops.analysis.chain(**slots)
    assert isinstance(solver, StageChain)
    assert len(ops._primitives) == before
    with pytest.raises(AttributeError):
        solver.test = None  # type: ignore[misc]  # frozen


def test_chain_refuses_a_wrong_slot_type(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    slots = _slots(ops)
    slots["system"], slots["numberer"] = slots["numberer"], slots["system"]
    with pytest.raises(TypeError, match="numberer= must be a Numberer"):
        ops.analysis.chain(**slots)


def test_chain_refuses_an_iterating_algorithm_without_test(
    plate_fem: Any,
) -> None:
    ops = plate(plate_fem)
    slots = _slots(ops)
    del slots["test"]
    with pytest.raises(ValueError, match="test= is required with algorithm Newton"):
        ops.analysis.chain(**slots)
    slots["algorithm"] = ops.algorithm.Linear()
    assert ops.analysis.chain(**slots).test is None


def test_needs_test_is_declared_on_every_algorithm() -> None:
    assert Linear.needs_test is False
    for cls in (Newton, ModifiedNewton, NewtonLineSearch, KrylovNewton,
                BFGS, Broyden):
        assert cls.needs_test is True, cls.__name__


# ---------------------------------------------------------------------------
# s.analysis: chain form, keyword form, and the refusals
# ---------------------------------------------------------------------------


def test_chain_binds_the_same_slots_as_the_keyword_form(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    slots = _slots(ops)
    integ = ops.integrator.LoadControl(dlam=1.0)
    with ops.stage(name="kw") as s:
        s.analysis(integrator=integ, **slots)
        s.run(n_increments=1)
    with ops.stage(name="chain") as s:
        s.analysis(ops.analysis.chain(**slots), integrator=integ)
        s.run(n_increments=1)
    kw, ch = ops._stage_records
    for slot in (*slots, "integrator"):
        assert getattr(ch, slot) is getattr(kw, slot), slot


@pytest.mark.parametrize("slot", ["test", "algorithm", "system", "analysis"])
def test_chain_plus_the_same_slot_is_refused(plate_fem: Any, slot: str) -> None:
    ops = plate(plate_fem)
    slots = _slots(ops)
    solver = ops.analysis.chain(**slots)
    with pytest.raises(ValueError, match=f"{slot} given both by the chain"):
        with ops.stage(name="s") as s:
            s.analysis(
                solver, integrator=ops.integrator.LoadControl(dlam=1.0),
                **{slot: slots[slot]},
            )
    assert ops._stage_records == []


def test_chain_plus_an_explicit_none_slot_is_refused(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    solver = ops.analysis.chain(**_slots(ops))
    with pytest.raises(ValueError, match="test given both by the chain"):
        with ops.stage(name="s") as s:
            s.analysis(
                solver, integrator=ops.integrator.LoadControl(dlam=1.0),
                test=None,
            )


def test_a_dict_in_the_chain_position_is_refused(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    with pytest.raises(TypeError, match="must be a chain from ops.analysis.chain"):
        with ops.stage(name="s") as s:
            s.analysis(
                _slots(ops),  # type: ignore[arg-type]
                integrator=ops.integrator.LoadControl(dlam=1.0),
            )


def test_keyword_form_still_refuses_an_omitted_slot(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    slots = _slots(ops)
    del slots["numberer"]
    with pytest.raises(TypeError, match="missing numberer"):
        with ops.stage(name="s") as s:
            s.analysis(integrator=ops.integrator.LoadControl(dlam=1.0), **slots)


# ---------------------------------------------------------------------------
# D1: under Static, a dt that differs from dlam warns (InertStaticDtWarning)
# ---------------------------------------------------------------------------


@contextmanager
def _silent() -> Iterator[None]:
    """The warning is an error inside: a silent case that warns fails."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", InertStaticDtWarning)
        yield


def _static_stage(ops: apeSees, *, dlam: float, **run: Any) -> None:
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(pg="Plate", forces=(1.0, 0.0))
        s.analysis(
            ops.analysis.chain(**_slots(ops)),
            integrator=ops.integrator.LoadControl(dlam=dlam),
        )
        s.run(n_increments=4, **run)


def test_static_dt_differing_from_dlam_warns(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    with pytest.warns(InertStaticDtWarning) as record:
        _static_stage(ops, dlam=0.25, dt=1.0)
    assert len(record) == 1
    msg = str(record[0].message)
    assert msg.startswith("dt has no effect under Static; dlam sets the step")
    assert "'push'" in msg and "dt=1.0" in msg and "dlam=0.25" in msg
    # A warning, not a refusal: the stage is recorded, dt unchanged.
    assert ops._stage_records[0].dt == 1.0
    assert issubclass(InertStaticDtWarning, UserWarning)


def test_static_dt_differing_from_ladruno_dlam_warns(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    with pytest.warns(InertStaticDtWarning, match="dlam=0.5"):
        with ops.stage(name="push") as s:
            s.analysis(
                ops.analysis.chain(**_slots(ops)),
                integrator=ops.integrator.LadrunoLoadControl(dlam=0.5),
            )
            s.run(n_increments=2, dt=0.1)


def test_static_dt_equal_to_dlam_is_silent(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    with _silent():
        _static_stage(ops, dlam=0.25, dt=0.25)
    assert ops._stage_records[0].dt == 0.25


def test_static_dt_equal_to_dlam_up_to_roundoff_is_silent(
    plate_fem: Any,
) -> None:
    ops = plate(plate_fem)
    with _silent():
        _static_stage(ops, dlam=1.0 / 3.0, dt=1.0 - 2.0 / 3.0)
    assert ops._stage_records[0].dt is not None


def test_static_dt_omitted_is_silent(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    with _silent():
        _static_stage(ops, dlam=0.25)
    assert ops._stage_records[0].dt is None


def test_transient_keeps_its_dt_silently(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    with _silent():
        with ops.stage(name="shake") as s:
            s.analysis(
                ops.analysis.chain(
                    **_slots(ops, analysis=ops.analysis.Transient()),
                ),
                # A LoadControl here, its dlam unequal to dt, isolates the
                # analysis guard: under Transient dt is the step, so the
                # mismatch is not D1's business.
                integrator=ops.integrator.LoadControl(dlam=0.25),
            )
            s.run(n_increments=4, dt=0.01)
    assert ops._stage_records[0].dt == 0.01
    assert "ops.analyze(1, 0.01)" in _deck(ops, "py")


def test_static_under_displacement_control_is_not_d1(plate_fem: Any) -> None:
    """D1 reads ``dlam``; an integrator without one is left alone."""
    ops = plate(plate_fem)
    node = int(plate_fem.nodes.ids[-1])
    with _silent():
        with ops.stage(name="push") as s:
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.load(node=node, forces=(1.0, 0.0))
            s.analysis(
                ops.analysis.chain(**_slots(ops)),
                integrator=ops.integrator.DisplacementControl(
                    node=node, dof=1, dU=0.001,
                ),
            )
            s.run(n_increments=2, dt=1.0)
    assert ops._stage_records[0].dt == 1.0


# ---------------------------------------------------------------------------
# A stage Linear without a test
# ---------------------------------------------------------------------------


def _linear_stage(ops: apeSees, *, with_test: bool, form: str) -> None:
    slots = _slots(ops, algorithm=ops.algorithm.Linear())
    if not with_test:
        del slots["test"]
    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(pg="Plate", forces=(0.0, -1.0))
        integ = ops.integrator.LoadControl(dlam=1.0)
        if form == "chain":
            s.analysis(ops.analysis.chain(**slots), integrator=integ)
        else:
            s.analysis(integrator=integ, **slots)
        s.run(n_increments=1)


@pytest.mark.parametrize("form", ["keyword", "chain"])
def test_linear_stage_without_test_emits_no_test_line(
    plate_fem: Any, form: str,
) -> None:
    ops = plate(plate_fem)
    _linear_stage(ops, with_test=False, form=form)
    assert ops._stage_records[0].test is None
    deck = _deck(ops, "tcl")
    assert "algorithm Linear" in deck
    assert "\ntest " not in deck
    with_test = plate(plate_fem)
    _linear_stage(with_test, with_test=True, form=form)
    other = _deck(with_test, "tcl").splitlines()
    # The only difference is the one test line.
    assert [ln for ln in other if not ln.startswith("test ")] == deck.splitlines()


def test_iterating_stage_without_test_is_refused(plate_fem: Any) -> None:
    ops = plate(plate_fem)
    slots = _slots(ops)
    del slots["test"]
    with pytest.raises(ValueError, match="test= is required with algorithm Newton"):
        with ops.stage(name="s") as s:
            s.analysis(integrator=ops.integrator.LoadControl(dlam=1.0), **slots)


def test_linear_stage_without_test_refuses_a_strategy_ladder(
    plate_fem: Any,
) -> None:
    ops = plate(plate_fem)
    slots = _slots(ops, algorithm=ops.algorithm.Linear())
    del slots["test"]
    with pytest.raises(ValueError, match="strategy ladder needs a convergence test"):
        with ops.stage(name="s") as s:
            s.analysis(integrator=ops.integrator.LoadControl(dlam=1.0), **slots)
            s.run(n_increments=1, strategy=ops.strategy.profile("non-smooth"))


def test_linear_stage_without_test_round_trips_through_h5(
    plate_fem: Any, tmp_path: Path,
) -> None:
    """The archive stores no ``analysis@test`` and replays without one.

    The oracle is the same model with a test: its replay differs from
    this one's by exactly the one ``test`` line.
    """
    import h5py

    replays = {}
    for with_test in (False, True):
        ops = plate(plate_fem)
        _linear_stage(ops, with_test=with_test, form="chain")
        h5 = tmp_path / f"model_{with_test}.h5"
        ops.h5(str(h5))
        with h5py.File(h5, "r") as f:
            chain_attrs: list[dict[str, Any]] = []
            f.visititems(
                lambda name, obj: chain_attrs.append(dict(obj.attrs))
                if name.endswith("analysis") and "algorithm" in obj.attrs
                else None,
            )
        assert chain_attrs, "no stage analysis group in the archive"
        assert all(("test" in a) is with_test for a in chain_attrs)
        replays[with_test] = OpenSeesModel.from_h5(str(h5)).build("tcl")
    without = replays[False].splitlines()
    assert "algorithm Linear" in without
    assert not any(ln.startswith("test ") for ln in without)
    assert [
        ln for ln in replays[True].splitlines() if not ln.startswith("test ")
    ] == without
