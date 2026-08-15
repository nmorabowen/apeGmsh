"""Decoupled work-point as RBE2/RBE3 master (ADR 0049 OQ2).

A labelled ``g.decouple_node`` is a first-class constraint role for
``kinematic_coupling`` (RBE2) and ``distributing_coupling`` (RBE3) —
no post-``get_fem_data`` retarget. Essential (``ops.fix``) and natural
(``p.load``) BCs address the knot after ``ops.ndf(h, ndf=6)``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees


_WORK_PT = (0.5, 0.5, 1.0)


def _box_session(*, decouple_label: str | None = "work_pt", mesh: bool = True):
    """1×1×1 tet box, top face PG ``end_face``, optional labelled knot."""
    g = apeGmsh(model_name="dn_rbe_master", verbose=False)
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.model.sync()
    g.physical.add_volume("body", name="Body")
    g.model.select(dim=2).on_plane(
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), tol=1e-3,
    ).to_physical("end_face")
    h = None
    if decouple_label is not None:
        h = g.decouple_node(coords=_WORK_PT, label=decouple_label)
    if mesh:
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=3)
    return g, h


def _face_node_tags(g) -> set[int]:
    """Mesh node tags on the ``end_face`` physical group."""
    import gmsh
    from apeGmsh.core._helpers import resolve_to_dimtags

    tags: set[int] = set()
    for dim, tag in resolve_to_dimtags(
            "end_face", default_dim=2, session=g):
        nt, _, _ = gmsh.model.mesh.getNodes(
            dim=int(dim), tag=int(tag),
            includeBoundary=True, returnParametricCoord=False,
        )
        tags.update(int(t) for t in nt)
    return tags


# ---------------------------------------------------------------------
# Positive — clean API
# ---------------------------------------------------------------------

def test_rbe2_by_string_label_uses_decoupled_master() -> None:
    g, h = _box_session()
    try:
        g.constraints.kinematic_coupling(
            "work_pt", "end_face",
            master_point=_WORK_PT,
            name="support",
        )
        fem = g.mesh.queries.get_fem_data(dim=3)
        assert int(h.tag) in {int(x) for x in fem.nodes.decoupled_ids}

        rec = next(
            r for r in fem.nodes.constraints
            if getattr(r, "name", None) == "support"
        )
        assert rec.kind == ConstraintKind.KINEMATIC_COUPLING
        assert int(rec.master_node) == int(h.tag)
        assert int(h.tag) not in {int(s) for s in rec.slave_nodes}

        face = _face_node_tags(g)
        assert set(int(s) for s in rec.slave_nodes) == face
        assert rec.offsets is not None and len(rec.offsets) == len(face)

        xyz = {
            int(i): np.asarray(c, dtype=float)
            for i, c in zip(fem.nodes.ids, fem.nodes.coords)
        }
        mx = xyz[int(h.tag)]
        by_slave = {
            int(s): np.asarray(o, dtype=float)
            for s, o in zip(rec.slave_nodes, rec.offsets)
        }
        for s, off in by_slave.items():
            assert np.allclose(off, xyz[s] - mx)
    finally:
        g.end()


def test_rbe3_by_string_label_uses_decoupled_reference() -> None:
    g, h = _box_session()
    try:
        g.constraints.distributing_coupling(
            "work_pt", "end_face",
            master_point=_WORK_PT,
            weighting="uniform",
            name="load_intro",
        )
        fem = g.mesh.queries.get_fem_data(dim=3)

        rec = next(
            r for r in fem.elements.constraints
            if getattr(r, "name", None) == "load_intro"
        )
        assert rec.kind == ConstraintKind.DISTRIBUTING
        assert int(rec.slave_node) == int(h.tag)
        assert int(h.tag) not in {int(m) for m in rec.master_nodes}
        assert set(int(m) for m in rec.master_nodes) == _face_node_tags(g)
        assert rec.weights is None
    finally:
        g.end()


def test_rbe2_handle_as_first_arg_normalizes_to_label() -> None:
    g, h = _box_session()
    try:
        g.constraints.kinematic_coupling(
            h, "end_face",
            master_point=_WORK_PT,
            name="support",
        )
        assert g.constraints.constraint_defs[-1].master_label == "work_pt"
        fem = g.mesh.queries.get_fem_data(dim=3)
        rec = next(
            r for r in fem.nodes.constraints
            if getattr(r, "name", None) == "support"
        )
        assert int(rec.master_node) == int(h.tag)
    finally:
        g.end()


def test_emit_without_retarget_deck_references_work_pt(tmp_path: Path) -> None:
    """Acceptance: RBE2 + RBE3 + fix/load emit the decoupled tag directly."""
    g, h = _box_session()
    try:
        g.constraints.kinematic_coupling(
            "work_pt", "end_face", name="support",
        )
        g.constraints.distributing_coupling(
            "work_pt", "end_face",
            weighting="uniform", name="load_intro",
        )
        fem = g.mesh.queries.get_fem_data(dim=3)

        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
        ops.element.FourNodeTetrahedron(pg="Body", material=mat)
        ops.ndf(h, ndf=6)
        ops.fix(nodes=[int(h.tag)], dofs=(1, 1, 1, 1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.load(node=int(h.tag), forces=(0.0, 0.0, -1.0e3, 0.0, 0.0, 0.0))

        out = tmp_path / "work_pt.tcl"
        ops.tcl(str(out))
        text = out.read_text(encoding="utf-8")

        assert "LadrunoKinematicCoupling" in text
        assert "LadrunoDistributingCoupling" in text
        kc = [ln for ln in text.splitlines() if "LadrunoKinematicCoupling" in ln]
        dc = [ln for ln in text.splitlines() if "LadrunoDistributingCoupling" in ln]
        assert any(str(h.tag) in ln for ln in kc), kc
        assert any(str(h.tag) in ln for ln in dc), dc
        assert any(
            ln.strip().startswith(f"fix {h.tag} ")
            for ln in text.splitlines()
        ), text
        assert any(
            f"load {h.tag} " in ln for ln in text.splitlines()
        ), text
    finally:
        g.end()


def test_declare_after_first_get_fem_data_still_binds_knot() -> None:
    """Live sessions cache ``g._fem``; both _add_def phases must allow it."""
    g, h = _box_session()
    try:
        _ = g.mesh.queries.get_fem_data(dim=3)
        g.constraints.kinematic_coupling(
            "work_pt", "end_face", name="support",
        )
        fem = g.mesh.queries.get_fem_data(dim=3)
        rec = next(
            r for r in fem.nodes.constraints
            if getattr(r, "name", None) == "support"
        )
        assert int(rec.master_node) == int(h.tag)
    finally:
        g.end()


# ---------------------------------------------------------------------
# Fail-loud
# ---------------------------------------------------------------------

def test_equal_dof_rejects_decoupled_label() -> None:
    g, _h = _box_session()
    try:
        with pytest.raises(ValueError, match=r"decoupled|kinematic_coupling"):
            g.constraints.equal_dof("work_pt", "end_face")
    finally:
        g.end()


def test_label_collision_with_physical_group_raises() -> None:
    g = apeGmsh(model_name="dn_rbe_collide", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
        g.model.sync()
        g.physical.add_volume("body", name="Body")
        g.model.select(dim=2).on_plane(
            (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), tol=1e-3,
        ).to_physical("work_pt")
        g.decouple_node(coords=_WORK_PT, label="work_pt")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=3)
        with pytest.raises(ValueError, match=r"both a decoupled|unambiguous"):
            g.constraints.kinematic_coupling(
                "work_pt", "Body", name="bad",
            )
    finally:
        g.end()


def test_duplicate_decoupled_labels_raise_at_declare() -> None:
    g = apeGmsh(model_name="dn_rbe_dup", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
        g.model.sync()
        g.physical.add_volume("body", name="Body")
        g.model.select(dim=2).on_plane(
            (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), tol=1e-3,
        ).to_physical("end_face")
        g.decouple_node(coords=_WORK_PT, label="work_pt")
        g.decouple_node(coords=(0.5, 0.5, 0.0), label="work_pt")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=3)
        with pytest.raises(ValueError, match=r"matches .* g.decouple_node"):
            g.constraints.kinematic_coupling(
                "work_pt", "end_face", name="bad",
            )
    finally:
        g.end()


def test_handle_without_label_raises() -> None:
    g, _ = _box_session(decouple_label=None)
    try:
        h = g.decouple_node(coords=_WORK_PT)  # no label=
        with pytest.raises(ValueError, match=r"no label="):
            g.constraints.kinematic_coupling(h, "end_face")
    finally:
        g.end()


# ---------------------------------------------------------------------
# Documented hazard (unchanged Part-bbox nearest-in-set path)
# ---------------------------------------------------------------------

def test_part_bbox_may_accidentally_include_decoupled_not_supported() -> None:
    """Explicit decoupled *name* now wins; Part-label bbox path is unchanged.

    A Part whose bbox swallows the knot can still surface it in
    ``node_map`` when the *Part* label is the master role — that is not
    a supported decoupled-master API.
    """
    pytest.skip(
        "documented hazard only: Part bbox + nearest-in-set is not a "
        "supported decoupled-master API; explicit decoupled label= takes "
        "precedence in _resolve_nodes (ADR 0049 OQ2)."
    )
