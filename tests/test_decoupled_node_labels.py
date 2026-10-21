"""A decoupled node's ``label=`` names it on the FEM snapshot (#2389).

``g.decouple_node(..., label="ground")`` used to keep the label on the
session def only, so ``ops.element.ZeroLength(nodes=("ground", n))``
raised ``node label 'ground' not found`` and a script fell back to
``fem.nodes.decoupled_ids[0]`` (the YOO site-response column). The FEM
factory now registers each decoupled label as a dim-0 node label of the
snapshot, so it resolves wherever a node label does and survives
``model.h5``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir


def _box(decl):
    """A 1x1x1 tet box (Body) with the decoupled nodes ``decl(g)`` adds."""
    g = apeGmsh(model_name="dn_labels", verbose=False)
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.add_volume("body", name="Body")
    out = decl(g)
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    return g, fem, out


def _spring_deck(fem, ground, other, tmp_path: Path) -> str:
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="Body", material=mat)
    k = ops.uniaxialMaterial.ElasticMaterial(E=1e6)
    ops.element.ZeroLength(
        nodes=(ground, other),
        mat_dirs=(ZeroLengthMatDir(material=k, dof=1),),
    )
    ops.ndf(ground, ndf=3)
    ops.fix(label=ground, dofs=(1, 1, 1))
    out = tmp_path / "m.tcl"
    ops.tcl(str(out))
    return out.read_text(encoding="utf-8")


def test_decoupled_label_is_a_snapshot_node_label() -> None:
    g, fem, h = _box(lambda g: g.decouple_node(
        coords=(0.0, 0.0, -1.0), label="ground"))
    try:
        assert "ground" in fem.nodes.labels
        assert [int(n) for n in fem.nodes.select(label="ground").ids] \
            == [h.tag]
    finally:
        g.end()


def test_zero_length_and_fix_resolve_the_decoupled_label(
        tmp_path: Path) -> None:
    g, fem, h = _box(lambda g: g.decouple_node(
        coords=(0.0, 0.0, 0.0), label="ground"))
    try:
        mesh_tag = int(fem.nodes.ids[0])
        text = _spring_deck(fem, "ground", mesh_tag, tmp_path)
    finally:
        g.end()
    zl = [ln.split() for ln in text.splitlines()
          if ln.strip().startswith("element zeroLength ")]
    assert len(zl) == 1 and zl[0][3:5] == [str(h.tag), str(mesh_tag)]
    assert f"fix {h.tag} 1 1 1" in text


def test_decoupled_label_survives_model_h5(tmp_path: Path) -> None:
    g, fem, h = _box(lambda g: g.decouple_node(
        coords=(0.0, 0.0, 0.0), label="ground"))
    try:
        path = tmp_path / "model.h5"
        fem.to_h5(str(path))
        mesh_tag = int(fem.nodes.ids[0])
    finally:
        g.end()
    back = FEMData.from_h5(str(path))
    assert [int(n) for n in back.nodes.select(label="ground").ids] == [h.tag]
    text = _spring_deck(back, "ground", mesh_tag, tmp_path)
    assert f"fix {h.tag} 1 1 1" in text


def test_decoupled_node_set_label_holds_every_set_node() -> None:
    g, fem, s = _box(lambda g: g.decouple_node_set(
        "Body", offset=(0.0, 0.0, -1.0), label="ground_set"))
    try:
        got = sorted(int(n) for n in fem.nodes.select(label="ground_set").ids)
        assert got == sorted(s.tags) and len(got) > 1
    finally:
        g.end()


def test_label_shared_by_two_decoupled_nodes_refuses_a_pair_endpoint(
        tmp_path: Path) -> None:
    def decl(g):
        g.decouple_node(coords=(0.0, 0.0, -1.0), label="gnd")
        g.decouple_node(coords=(1.0, 0.0, -1.0), label="gnd")
    g, fem, _ = _box(decl)
    try:
        assert len(fem.nodes.select(label="gnd").ids) == 2
        with pytest.raises(BridgeError, match="EXACTLY one node"):
            _spring_deck(fem, "gnd", int(fem.nodes.ids[0]), tmp_path)
    finally:
        g.end()


def test_decoupled_label_that_is_a_geometry_label_raises() -> None:
    with pytest.raises(ValueError, match="already geometry labels"):
        g, _, _ = _box(lambda g: g.decouple_node(
            coords=(0.0, 0.0, -1.0), label="body"))
    import gmsh
    if gmsh.isInitialized():
        gmsh.finalize()


def test_pair_endpoint_accepts_a_single_node_physical_group(
        tmp_path: Path) -> None:
    def decl(g):
        h = g.decouple_node(coords=(0.0, 0.0, -1.0), label="ground")
        g.physical.add(0, [g.model.queries.entities_in_bounding_box(
            -1e-6, -1e-6, -1e-6, 1e-6, 1e-6, 1e-6, dim=0)[0][1]],
            name="Corner")
        return h
    g, fem, h = _box(decl)
    try:
        corner = [int(n) for n in fem.nodes.select(pg="Corner").ids]
        assert len(corner) == 1
        text = _spring_deck(fem, "ground", "Corner", tmp_path)
    finally:
        g.end()
    zl = [ln.split() for ln in text.splitlines()
          if ln.strip().startswith("element zeroLength ")]
    assert zl[0][3:5] == [str(h.tag), str(corner[0])]


def test_unknown_pair_endpoint_names_both_namespaces(tmp_path: Path) -> None:
    g, fem, _ = _box(lambda g: g.decouple_node(
        coords=(0.0, 0.0, -1.0), label="ground"))
    try:
        with pytest.raises(BridgeError, match="neither a node label nor a "
                                              "physical group"):
            _spring_deck(fem, "ground", "nowhere", tmp_path)
    finally:
        g.end()


def test_rbe2_master_after_a_first_snapshot_is_not_a_clash() -> None:
    """After the first ``get_fem_data`` the session validates through the
    snapshot, which now carries the decoupled label: that label is the
    node itself, not a second target of the same name."""
    def decl(g):
        g.model.select(dim=2).on_plane(
            (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), tol=1e-3,
        ).to_physical("Top")
        return g.decouple_node(coords=(0.5, 0.5, 2.0), label="work_pt")
    g, _fem, h = _box(decl)
    try:
        g.mesh.queries.get_fem_data()        # the default call caches g._fem
        assert g._fem is not None and "work_pt" in g._fem.nodes.labels
        g.constraints.kinematic_coupling("work_pt", "Top", name="rbe2")
        fem2 = g.mesh.queries.get_fem_data(dim=3)
        assert h.tag in {int(n) for n in fem2.nodes.decoupled_ids}
    finally:
        g.end()


def test_decoupled_label_that_is_also_a_pg_still_clashes_after_a_snapshot() -> None:
    """A decoupled label shared with a physical group stays ambiguous for
    an RBE2 role whether or not a first snapshot was taken."""
    def decl(g):
        g.model.select(dim=2).on_plane(
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), tol=1e-3,
        ).to_physical("ground")
        g.model.select(dim=2).on_plane(
            (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), tol=1e-3,
        ).to_physical("Top")
        return g.decouple_node(coords=(0.5, 0.5, 2.0), label="ground")
    g, _fem, _h = _box(decl)
    try:
        g.mesh.queries.get_fem_data()
        assert g._fem is not None
        with pytest.raises(ValueError, match="names both a decoupled node"):
            g.constraints.kinematic_coupling("ground", "Top", name="bad")
    finally:
        g.end()


def test_nodes_one_reaches_a_decoupled_label() -> None:
    """``ops.nodes.one(label=)`` is the single-node route of ``p.load`` and
    ``ops.equation_constraint``; it reaches a decoupled label too."""
    g, fem, h = _box(lambda g: g.decouple_node(
        coords=(0.0, 0.0, -1.0), label="ground"))
    try:
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        assert int(ops.nodes.one(label="ground").tag) == h.tag
    finally:
        g.end()
