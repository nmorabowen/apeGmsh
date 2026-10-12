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

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.types import Element
from apeGmsh.opensees._readback import rotation_roles
from apeGmsh.opensees.element.beam_column import elasticBeamColumn
from apeGmsh.opensees.element.solid import LadrunoUP
from apeGmsh.opensees.element.zero_length import (
    CoupledZeroLength,
    ZeroLength,
    ZeroLengthMatDir,
    ZeroLengthSection,
)
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


# ---------------------------------------------------------------------------
# The shared rotation-role table (``_readback.rotation_roles``): the rule the
# declare emit, ``ops.results`` and ``.read`` all apply
# ---------------------------------------------------------------------------


class _Rows:
    """Plan rows stand-in: connectivity only."""

    def __init__(self, *conn: tuple[int, ...]) -> None:
        self.conn = np.asarray(conn, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.conn)


def _spec(cls: type) -> Any:
    """An instance of ``cls`` for an ``isinstance`` role check."""
    return object.__new__(cls)


def test_role_table_blocked_wins_over_rotational() -> None:
    """A beam and a LadrunoUP share node 2: the u-p class blocks it."""
    roles = rotation_roles([(_spec(elasticBeamColumn), _Rows((1, 2))),
                            (_spec(LadrunoUP), _Rows((2, 3, 4, 5)))])
    assert roles.refusal(1) is None
    assert "LadrunoUP" in str(roles.refusal(2))


def test_role_table_unknown_class_blocks() -> None:
    class Unknown(Element):
        def dependencies(self) -> tuple[Any, ...]:
            return ()

        def _emit(self, emitter: Any, tag: int) -> None:
            raise AssertionError("never emitted")

    roles = rotation_roles([(_spec(elasticBeamColumn), _Rows((1, 2))),
                            (_spec(Unknown), _Rows((2, 3)))])
    assert "Unknown" in str(roles.refusal(2))
    assert "Unknown" in str(roles.refusal(3))


@pytest.mark.parametrize(
    "cls", [ZeroLength, ZeroLengthSection, CoupledZeroLength])
def test_role_table_zero_length_family_is_rotational(cls: type) -> None:
    roles = rotation_roles([(_spec(cls), _Rows((1, 2)))])
    assert roles.refusal(1) is None and roles.refusal(2) is None


def test_role_table_element_less_node_needs_a_model_without_up() -> None:
    """Node 9 is touched by no element: it carries a rotation in a frame,
    but not beside a u-p class, whose pressure DOF a constraint could tie
    to its slot."""
    frame = [(_spec(elasticBeamColumn), _Rows((1, 2)))]
    assert rotation_roles(frame).refusal(9) is None
    mixed = frame + [(_spec(LadrunoUP), _Rows((3, 4, 5, 6)))]
    assert "LadrunoUP" in str(rotation_roles(mixed).refusal(9))


# ---------------------------------------------------------------------------
# No false refusals: the zero-length family and a decoupled master record
# as on main (byte-identical lines in 3-D)
# ---------------------------------------------------------------------------


def _tcl_recorder_lines(ops: apeSees, tmp_path: Any) -> list[str]:
    deck = tmp_path / "m.tcl"
    ops.tcl(str(deck))
    return [ln for ln in deck.read_text().splitlines()
            if ln.startswith("recorder Node")]


def test_3d_zero_length_rotational_spring_records_dof_6(tmp_path: Any) -> None:
    ops = apeSees(cast("object", make_two_node_beam(ndm=3)))
    ops.model(ndm=3, ndf=6)
    k = ops.uniaxialMaterial.ElasticMaterial(E=1e6)
    ops.element.ZeroLength(
        pg="Cols", mat_dirs=(ZeroLengthMatDir(material=k, dof=6),))
    ops.recorder.declare(nodes=("rotation_z", "reaction_moment_z"),
                         pg="Top", file_root="out", name="r")
    assert _tcl_recorder_lines(ops, tmp_path) == [
        "recorder Node -file out/r__default__disp.out -time -node 2 "
        "-dof 6 disp",
        "recorder Node -file out/r__default__reaction.out -time -node 2 "
        "-dof 6 reaction",
    ]


def test_3d_zero_length_section_base_records_rotations(tmp_path: Any) -> None:
    ops = apeSees(cast("object", make_two_node_beam(ndm=3)))
    ops.model(ndm=3, ndf=6)
    k = ops.uniaxialMaterial.ElasticMaterial(E=1e6)
    agg = ops.section.Aggregator(materials_by_dof={"P": k, "Mz": k, "My": k})
    ops.element.ZeroLengthSection(pg="Cols", section=agg)
    ops.recorder.declare(nodes=("rotation",), pg="Base",
                         file_root="out", name="r")
    assert _tcl_recorder_lines(ops, tmp_path) == [
        "recorder Node -file out/r__default__disp.out -time -node 1 "
        "-dof 4 5 6 disp",
    ]


def _frame_with_master(name: str) -> tuple[Any, Any, Any]:
    """A beam line plus an element-less decoupled master node."""
    from apeGmsh import apeGmsh

    g = apeGmsh(model_name=name)
    g.begin()
    a = g.model.geometry.add_point(0.0, 0.0, 0.0)
    b = g.model.geometry.add_point(1.0, 0.0, 0.0)
    g.physical.add(1, [g.model.geometry.add_line(a, b)], name="Beam")
    h = g.decouple_node(coords=(0.5, 1.0, 0.0), label="master")
    g.mesh.generation.generate(dim=1)
    return g, g.mesh.queries.get_fem_data(dim=1), h


def test_decoupled_ndf6_master_records_rotations(tmp_path: Any) -> None:
    g, fem, h = _frame_with_master("b5_master_3d")
    try:
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        ops.element.elasticBeamColumn(
            pg="Beam", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
        )
        ops.ndf(h, ndf=6)
        ops.recorder.declare(nodes=("rotation",), ids=(int(h.tag),),
                             file_root="out", name="m")
        assert _tcl_recorder_lines(ops, tmp_path) == [
            f"recorder Node -file out/m__default__disp.out -time -node "
            f"{int(h.tag)} -dof 4 5 6 disp",
        ]
    finally:
        g.end()


def test_decoupled_planar_master_rotation_is_dof_3(tmp_path: Any) -> None:
    g, fem, h = _frame_with_master("b5_master_2d")
    try:
        ops = apeSees(fem, default_orientation=None)
        ops.model(ndm=2, ndf=3)
        ops.element.elasticBeamColumn(
            pg="Beam", transf=ops.geomTransf.Linear(), A=0.01, E=200e9,
            Iz=1e-4)
        ops.ndf(h, ndf=3)
        ops.recorder.declare(nodes=("rotation_z",), ids=(int(h.tag),),
                             file_root="out", name="m")
        assert _tcl_recorder_lines(ops, tmp_path) == [
            f"recorder Node -file out/m__default__disp.out -time -node "
            f"{int(h.tag)} -dof 3 disp",
        ]
    finally:
        g.end()


@pytest.mark.parametrize(("ndm", "rot_dir", "line_dofs"), [
    (2, 3, "-dof 3"), (3, 6, "-dof 4 5 6")])
def test_two_node_link_end_node_records_rotation(
    ndm: int, rot_dir: int, line_dofs: str, tmp_path: Any,
) -> None:
    """A twoNodeLink with a rotational ``-dir`` carries the rotation of its
    end node: DOF 3 in 2-D, DOFs 4-6 in 3-D."""
    ops = apeSees(cast("object", make_two_node_beam(ndm=ndm)),
                  default_orientation=None)
    ops.model(ndm=ndm, ndf=3 if ndm == 2 else 6)
    k = ops.uniaxialMaterial.ElasticMaterial(E=1e6)
    ops.element.TwoNodeLink(
        pg="Cols", mat_dirs=(ZeroLengthMatDir(material=k, dof=rot_dir),))
    comps = ("rotation_z",) if ndm == 2 else ("rotation",)
    ops.recorder.declare(nodes=comps, pg="Top", file_root="out", name="r")
    assert _tcl_recorder_lines(ops, tmp_path) == [
        f"recorder Node -file out/r__default__disp.out -time -node 2 "
        f"{line_dofs} disp",
    ]


def test_overlay_master_in_an_ndf3_solid_model_names_the_cause() -> None:
    """Open gap: an element-less master given ``ops.ndf(h, 6)`` in an
    ``ndf=3`` solid model. The emit cannot see the overlay, so it refuses
    and says why, rather than guess a DOF."""
    from tests.opensees.unit.test_node_pair_zerolength import (
        _box_with_ground,
    )

    g, fem, h = _box_with_ground("master")
    try:
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
        ops.element.FourNodeTetrahedron(pg="Body", material=mat)
        ops.ndf(h, ndf=6)
        ops.recorder.declare(nodes=("rotation_x",), ids=(int(h.tag),))
        tag = int(h.tag)
        with pytest.raises(BridgeError, match=(
                rf"node {tag} .*'rotation_x' has no DOF.*envelope ndf=3.*"
                rf"ops\.ndf raised node {tag}.*temporary regression.*#1745")):
            _node_recorders(ops)
    finally:
        g.end()


# ---------------------------------------------------------------------------
# pore_pressure: never a silent ``-dof -1``
# ---------------------------------------------------------------------------


def test_pore_pressure_refuses_on_a_frame() -> None:
    ops = _beam_2d()
    ops.recorder.declare(nodes=("pore_pressure",), pg="Top")
    with pytest.raises(BridgeError, match="pore_pressure.*PFEM"):
        _node_recorders(ops)


def test_pore_pressure_refuses_on_a_up_node() -> None:
    """The OpenSees ``pressure`` node response is the PFEM pressure
    constraint: on a u-p node it is not the pore pressure either."""
    from tests.opensees.unit.test_label_selector_fix_merge import (
        build_footing_fem,
    )

    ops = apeSees(build_footing_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=0.0)
    ops.element.LadrunoUP(pg="Plate", material=mat, Kf=2.2e6, poro=0.4,
                          rhoF=1.0, perm=(1e-4, 1e-4))
    ops.recorder.declare(nodes=("pore_pressure",), pg="Bottom")
    with pytest.raises(BridgeError, match=r"pore_pressure.*DOF \(3\)"):
        _node_recorders(ops)
