"""B5-b (#1726, fixes #1723): ``recorder.declare`` node components map
through the per-node DOF layout, never the fixed 6-DOF table.

The emit-level half of the slice; the deck-run oracle (tip ``rotation_z``
against ``PL^2/2EI``) is
``tests/opensees/live/test_recorder_declare_rotation_live.py``.

- ndm=2 / ndf=3 beam nodes: every rotational and moment component lands
  on DOF 3, translations on DOFs 1-2.
- ndm=3 / ndf=6: unchanged (DOFs 1-3 translate, 4-6 rotate).
- A rotation where no beam-column or shell establishes one refuses: a
  truss-only ``ndf=2`` model, a u-p node (DOF 3 is pore pressure), the
  truss half of a mixed-ndf model. Nothing is recorded for the refused
  component.
- ``ModelData`` (a hand-written deck: no element plan) maps through the
  layout of its declared ``(ndm, ndf)``.
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._recorder_translate import (
    group_node_components_by_ops_token,
    node_component_to_ops,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.model_data import ModelData

from tests.opensees.fixtures.fem_stub import make_two_node_beam
from tests.opensees.subprocess.test_mixed_ndf_2d_node_ndf import (
    _mixed_ndf_2d_fem,
    _mixed_ndf_2d_ops,
)


def _node_recorders(ops: apeSees) -> list[tuple[Any, ...]]:
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return [args for name, args, _ in rec.calls
            if name == "recorder" and args[0] == "Node"]


def _dofs_and_token(args: tuple[Any, ...]) -> tuple[tuple[int, ...], str]:
    i = args.index("-dof")
    return tuple(args[i + 1:-1]), str(args[-1])


def _beam_2d() -> apeSees:
    ops = apeSees(cast("object", make_two_node_beam(ndm=2)),
                  default_orientation=None)
    ops.model(ndm=2, ndf=3)
    ops.element.elasticBeamColumn(
        pg="Cols", transf=ops.geomTransf.Linear(), A=0.01, E=200e9, Iz=1e-4,
    )
    return ops


def _beam_3d() -> apeSees:
    ops = apeSees(cast("object", make_two_node_beam(ndm=3)))
    ops.model(ndm=3, ndf=6)
    ops.element.elasticBeamColumn(
        pg="Cols", transf=ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    return ops


# ---------------------------------------------------------------------------
# The translator takes a layout; there is no 6-DOF default
# ---------------------------------------------------------------------------


def test_translator_maps_through_the_layout_it_is_given() -> None:
    planar = (0, 1, 5)
    assert node_component_to_ops("rotation_z", planar) == ("disp", 3)
    assert node_component_to_ops("moment_z", planar) == ("unbalance", 3)
    assert node_component_to_ops("velocity_y", planar) == ("vel", 2)
    assert node_component_to_ops("rotation_z", (0, 1, 2, 3, 4, 5)) == (
        "disp", 6)
    assert node_component_to_ops("bogus", planar) is None


def test_translator_refuses_a_component_with_no_slot() -> None:
    with pytest.raises(ValueError, match="rotation_z.*no DOF"):
        node_component_to_ops("rotation_z", (0, 1))
    with pytest.raises(ValueError, match="displacement_z.*no DOF"):
        group_node_components_by_ops_token(("displacement_z",), (0, 1, 5))


# ---------------------------------------------------------------------------
# ndm=2 / ndf=3: the #1723 case
# ---------------------------------------------------------------------------


def test_planar_beam_rotation_is_dof_3() -> None:
    ops = _beam_2d()
    ops.recorder.declare(nodes=("displacement_y", "rotation_z"), pg="Top")
    calls = _node_recorders(ops)
    assert [_dofs_and_token(a) for a in calls] == [((2, 3), "disp")]


@pytest.mark.parametrize(("component", "token"), [
    ("rotation_z", "disp"),
    ("angular_velocity_z", "vel"),
    ("angular_acceleration_z", "accel"),
    ("reaction_moment_z", "reaction"),
    ("moment_z", "unbalance"),
])
def test_planar_beam_every_rotational_component_is_dof_3(
    component: str, token: str,
) -> None:
    ops = _beam_2d()
    ops.recorder.declare(nodes=(component,), pg="Top")
    assert [_dofs_and_token(a) for a in _node_recorders(ops)] == [
        ((3,), token)]


def test_planar_beam_translations_are_dofs_1_2() -> None:
    ops = _beam_2d()
    ops.recorder.declare(
        nodes=("velocity_x", "velocity_y", "acceleration_x",
               "acceleration_y", "force_x", "reaction_force_y"),
        pg="Top",
    )
    got = {tok: dofs for dofs, tok in map(_dofs_and_token,
                                          _node_recorders(ops))}
    assert got == {"vel": (1, 2), "accel": (1, 2), "unbalance": (1,),
                   "reaction": (2,)}


@pytest.mark.parametrize("component", ["rotation_x", "rotation_y"])
def test_planar_beam_out_of_plane_rotation_refuses(component: str) -> None:
    ops = _beam_2d()
    ops.recorder.declare(nodes=(component,), pg="Top")
    with pytest.raises(BridgeError, match=component):
        _node_recorders(ops)


# ---------------------------------------------------------------------------
# ndm=3 / ndf=6: unchanged
# ---------------------------------------------------------------------------


def test_3d_frame_dofs_are_unchanged() -> None:
    ops = _beam_3d()
    ops.recorder.declare(
        nodes=("displacement", "rotation", "reaction_moment_x", "moment_y"),
        pg="Top",
    )
    got = [_dofs_and_token(a) for a in _node_recorders(ops)]
    assert got == [((1, 2, 3, 4, 5, 6), "disp"), ((4,), "reaction"),
                   ((5,), "unbalance")]


def test_3d_frame_tcl_lines_are_the_pre_fix_bytes(tmp_path: Any) -> None:
    """The recorder lines of a 3-D frame are the bytes origin/main wrote
    before the fix (captured at e101340a)."""
    ops = _beam_3d()
    ops.recorder.declare(nodes=("displacement", "rotation"), pg="Top",
                         file_root="out", name="r")
    deck = tmp_path / "m.tcl"
    ops.tcl(str(deck))
    lines = [ln for ln in deck.read_text().splitlines()
             if ln.startswith("recorder Node")]
    assert lines == [
        "recorder Node -file out/r__default__disp.out -time -node 2 "
        "-dof 1 2 3 4 5 6 disp",
    ]


# ---------------------------------------------------------------------------
# Refusals: no slot, or a slot no beam/shell establishes
# ---------------------------------------------------------------------------


def test_truss_model_rotation_refuses() -> None:
    """ndm=2 / ndf=2: a truss node has no rotation."""
    ops = apeSees(cast("object", make_two_node_beam(ndm=2)),
                  default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    ops.element.Truss(pg="Cols", A=0.01, material=mat)
    ops.recorder.declare(nodes=("rotation_z",), pg="Top")
    with pytest.raises(BridgeError, match=r"rotation_z.*node 2"):
        _node_recorders(ops)


def test_up_node_rotation_refuses_and_records_no_pressure() -> None:
    """A LadrunoUP node at ndm=2 / ndf=3: DOF 3 is pore pressure, so a
    declared ``rotation_z`` refuses instead of recording it."""
    from tests.opensees.unit.test_label_selector_fix_merge import (
        build_footing_fem,
    )

    ops = apeSees(build_footing_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=0.0)
    ops.element.LadrunoUP(pg="Plate", material=mat, Kf=2.2e6, poro=0.4,
                          rhoF=1.0, perm=(1e-4, 1e-4))
    ops.recorder.declare(nodes=("displacement_y", "rotation_z"),
                         pg="Bottom")
    rec = RecordingEmitter()
    with pytest.raises(BridgeError, match=r"rotation_z.*LadrunoUP"):
        ops.build().emit(rec)
    assert not [c for c in rec.calls if c[0] == "recorder"]

    # The translation of the same node is DOF 2, as before.
    ops2 = apeSees(build_footing_fem(), default_orientation=None)
    ops2.model(ndm=2, ndf=3)
    mat2 = ops2.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=0.0)
    ops2.element.LadrunoUP(pg="Plate", material=mat2, Kf=2.2e6, poro=0.4,
                           rhoF=1.0, perm=(1e-4, 1e-4))
    ops2.recorder.declare(nodes=("displacement_y",), pg="Bottom")
    assert [_dofs_and_token(a) for a in _node_recorders(ops2)] == [
        ((2,), "disp")]


def test_mixed_ndf_maps_each_node_by_its_own_layout() -> None:
    """Truss nodes (ndf 2) and beam nodes (ndf 3) in one ndm=2 model."""
    fem = _mixed_ndf_2d_fem()
    # Translations: DOFs 1-2 on every node, one recorder line.
    ops = _mixed_ndf_2d_ops(fem)
    ops.recorder.declare(nodes=("displacement_x", "displacement_y"),
                         ids=(1, 2, 3, 4))
    calls = _node_recorders(ops)
    assert [_dofs_and_token(a) for a in calls] == [((1, 2), "disp")]
    # The beam nodes' rotation is DOF 3.
    ops = _mixed_ndf_2d_ops(fem)
    ops.recorder.declare(nodes=("rotation_z",), ids=(3, 4))
    assert [_dofs_and_token(a) for a in _node_recorders(ops)] == [
        ((3,), "disp")]
    # A truss node has none: the whole record refuses, naming it.
    ops = _mixed_ndf_2d_ops(fem)
    ops.recorder.declare(nodes=("rotation_z",), ids=(2, 4))
    with pytest.raises(BridgeError, match=r"rotation_z.*node 2"):
        _node_recorders(ops)


# ---------------------------------------------------------------------------
# ModelData: a hand-written deck maps through its declared (ndm, ndf)
# ---------------------------------------------------------------------------


def test_model_data_planar_rotation_is_dof_3() -> None:
    md = ModelData(make_two_node_beam(ndm=2), ndm=2, ndf=3)
    md.recorders(nodes=("rotation_z",), pg="Top")
    lines = [ln for ln in md.recorder_commands(target="tcl")
             if ln.startswith("recorder Node")]
    assert len(lines) == 1 and lines[0].endswith("-dof 3 disp"), lines


def test_model_data_ndf2_rotation_refuses() -> None:
    md = ModelData(make_two_node_beam(ndm=2), ndm=2, ndf=2)
    md.recorders(nodes=("rotation_z",), pg="Top")
    with pytest.raises(BridgeError, match="rotation_z.*no DOF"):
        md.recorder_commands(target="tcl")
