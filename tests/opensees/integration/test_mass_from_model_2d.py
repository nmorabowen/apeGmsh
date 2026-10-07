"""mass_from_model() on 2-D models — broker masses map by (ndm, ndf).

Broker masses (``fem.nodes.masses``) are spatial ``(mx, my, mz, Ixx, Iyy,
Izz)`` 6-vectors. They used to be trimmed POSITIONALLY to the node ndf, which
is only right in 3-D: a 2-D frame node ``(ux, uy, rz)`` received
``(mx, my, mz)`` (translational mass landing on rz as a rotational inertia),
and a 2-D solid node (ndf=2) raised on the resolver-filled ``mz``. The
3-D byte-identity pin lives in ``test_mass_from_model.py``.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    broker_mass_components,
    fit_dof_vector,
)


def _mass_lines(ops) -> "dict[int, tuple[float, ...]]":
    fd, path = tempfile.mkstemp(suffix=".tcl")
    os.close(fd)
    try:
        ops.tcl(path)
        with open(path, encoding="utf-8") as f:
            out = {}
            for ln in f:
                tok = ln.split()
                if tok and tok[0] == "mass":
                    out[int(tok[1])] = tuple(float(v) for v in tok[2:])
            return out
    finally:
        os.remove(path)


def _frame_fem(partitions: int = 0, **mass_kw):
    with apeGmsh(model_name="mfm2d_frame", verbose=False) as g:
        p0 = g.model.geometry.add_point(0, 0, 0)
        p1 = g.model.geometry.add_point(10, 0, 0)
        g.model.geometry.add_line(p0, p1, label="beam")
        g.physical.add_curve("beam", name="Beam")
        g.masses.line("Beam", linear_density=100.0, **mass_kw)
        g.mesh.sizing.set_global_size(2.5)
        g.mesh.generation.generate(dim=1)
        if partitions:
            g.mesh.partitioning.partition(partitions)
        fem = g.mesh.queries.get_fem_data(dim=1)
    assert len(fem.nodes.masses) > 0
    return fem


def _frame_ops(fem):
    ops = apeSees(fem, _artifacts=False)  # partitioned snapshot: no automatic model.h5
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(
        pg="Beam", transf=transf, A=0.01, E=200e9, Iz=1e-4,
    )
    ops.mass_from_model()
    return ops


@pytest.mark.parametrize("partitions", [0, 2])
def test_2d_frame_mass_lands_on_translations_not_rz(partitions):
    fem = _frame_fem(partitions)
    lines = _mass_lines(_frame_ops(fem))
    expected = {int(m.node_id): m.mass for m in fem.nodes.masses}
    assert set(lines) == set(expected)
    for nid, vec in expected.items():
        mx, my, _mz, _ixx, _iyy, izz = (float(v) for v in vec)
        assert mx > 0.0
        # (mx, my, Izz) — the old positional trim emitted (mx, my, mz).
        assert lines[nid] == pytest.approx((mx, my, izz))
        assert lines[nid][2] == 0.0


def test_2d_frame_izz_lands_on_rz():
    fem = _frame_fem(rotational=(0.0, 0.0, 7.0))
    lines = _mass_lines(_frame_ops(fem))
    expected = {int(m.node_id): m.mass for m in fem.nodes.masses}
    assert set(lines) == set(expected)
    for nid, vec in expected.items():
        # rotational= accumulates per receiving element: 7 or 14 per node.
        assert float(vec[5]) in (7.0, 14.0)
        assert lines[nid] == pytest.approx(
            (float(vec[0]), float(vec[1]), float(vec[5])))


def test_2d_frame_z_only_mass_fails_loud():
    # An explicit dofs=[3] mass has nowhere to land in 2-D; it must not be
    # zeroed silently (mirrors Fz in broker_load_components).
    fem = _frame_fem(dofs=[3])
    with pytest.raises(BridgeError, match="z-translation only"):
        _mass_lines(_frame_ops(fem))


def test_2d_solid_mass_emits_two_components_instead_of_raising():
    with apeGmsh(model_name="mfm2d_solid", verbose=False) as g:
        g.model.geometry.add_rectangle(0, 0, 0, 4, 2, label="plate")
        g.physical.add_surface("plate", name="Plate")
        g.masses.surface("Plate", areal_density=500.0)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    assert len(fem.nodes.masses) > 0

    ops = apeSees(fem, _artifacts=False)  # partitioned snapshot: no automatic model.h5
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.Tri31(pg="Plate", thickness=0.2, material=mat)
    ops.mass_from_model()
    lines = _mass_lines(ops)

    expected = {int(m.node_id): m.mass for m in fem.nodes.masses}
    assert set(lines) == set(expected)
    for nid, vec in expected.items():
        assert lines[nid] == pytest.approx((float(vec[0]), float(vec[1])))


# ---------------------------------------------------------------------------
# broker_mass_components — the mapping itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndf", [3, 6])
def test_3d_is_the_positional_fit(ndf):
    vec = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0) if ndf == 3 else (1, 2, 3, 4, 5, 6)
    assert broker_mass_components(vec, ndf, 3, node=1) == fit_dof_vector(
        vec, ndf, kind="mass", node=1)


def test_2d_layout_by_ndf():
    vec = (5.0, 5.0, 5.0, 0.0, 0.0, 7.0)
    assert broker_mass_components(vec, 3, 2, node=1) == (5.0, 5.0, 7.0)
    vec2 = (5.0, 5.0, 5.0, 0.0, 0.0, 0.0)
    assert broker_mass_components(vec2, 2, 2, node=1) == (5.0, 5.0)


@pytest.mark.parametrize(
    "ndf, vec, label",
    [
        (3, (1.0, 1.0, 1.0, 2.0, 0.0, 0.0), "Ixx=2"),
        (3, (1.0, 1.0, 1.0, 0.0, 3.0, 0.0), "Iyy=3"),
        (2, (1.0, 1.0, 1.0, 0.0, 0.0, 4.0), "Izz=4"),
    ],
)
def test_2d_uncarried_rotational_inertia_fails_loud(ndf, vec, label):
    with pytest.raises(BridgeError, match=label) as exc:
        broker_mass_components(vec, ndf, 2, node=9)
    assert "rotational inertia" in str(exc.value)


def test_2d_translation_the_node_lacks_names_the_real_cause():
    with pytest.raises(BridgeError, match="my=1") as exc:
        broker_mass_components((1.0, 1.0, 1.0, 0, 0, 0), 1, 2, node=9)
    assert "rotational" not in str(exc.value)


def test_2d_mixed_x_z_mask_drops_mz_known_gap():
    # dofs=[1, 3] is indistinguishable from a default fill without the mask
    # on MassRecord; pinned so closing the gap is a deliberate test change.
    assert broker_mass_components(
        (2.0, 0.0, 2.0, 0, 0, 0), 3, 2, node=1) == (2.0, 0.0, 0.0)
