"""A tie onto quad4 master faces emits a 3-retained-node
``ASDEmbeddedNodeElement``.

``ASDEmbeddedNodeElement`` reads 3 retained nodes as a triangle and 4 as a
tetrahedron (``ASDEmbeddedNodeElement.cpp``, ``getTangentStiff``: a
4-node element, constrained node included, is the triangle). A tie
record holds the corners of the master *face* it projected onto, so a
hexahedral master gave 4 coplanar corners: a zero-volume tet whose
stiffness is singular (the fork fails the first solve). The emit now
writes the triangle of the face's (0, 2) diagonal split that holds the
projected point, the same split ``embedded`` applies to quad hosts.

Oracles, independent of the code under test:

* the deck: every tie line carries 3 retained nodes, and the slave lies
  inside (or on) that triangle by its own barycentric coordinates,
  computed here from the node coordinates;
* the record: the FEM keeps all 4 corners and their bilinear weights
  (the ``equation`` route reads them), only the element changes;
* the engine (``APEGMSH_OPENSEES_BIN``): the tied stack solves a static
  step and carries the load across the tie.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees._internal.build import _embedded_retained_nodes

E, NU = 30_000.0, 0.2
K = 1.0e8
LOW, HIGH = 2.0, 1.0          # base block side, tied block side
SHIFT = (0.3, 0.15)           # the tied block's offset; off the diagonals
TOL = 1.0e-6


def _tie_record(weights, kind=ConstraintKind.TIE, n=4) -> InterpolationRecord:
    return InterpolationRecord(
        kind=kind, slave_node=100, master_nodes=list(range(1, n + 1)),
        weights=np.asarray(weights, dtype=float), dofs=[1, 2, 3],
    )


class TestRetainedNodes:
    def test_quad_tie_below_the_diagonal_takes_triangle_012(self) -> None:
        # xi > eta: w1 > w3.
        from apeGmsh._kernel.resolvers._constraint_resolver import _shape_quad4
        rec = _tie_record(_shape_quad4(0.5, -0.5))
        assert _embedded_retained_nodes(rec) == [1, 2, 3]

    def test_quad_tie_above_the_diagonal_takes_triangle_023(self) -> None:
        from apeGmsh._kernel.resolvers._constraint_resolver import _shape_quad4
        rec = _tie_record(_shape_quad4(-0.5, 0.5))
        assert _embedded_retained_nodes(rec) == [1, 3, 4]

    def test_a_point_on_the_diagonal_takes_triangle_012(self) -> None:
        from apeGmsh._kernel.resolvers._constraint_resolver import _shape_quad4
        rec = _tie_record(_shape_quad4(0.2, 0.2))
        assert _embedded_retained_nodes(rec) == [1, 2, 3]

    def test_tri_tie_passes_through(self) -> None:
        rec = _tie_record([0.2, 0.3, 0.5], n=3)
        assert _embedded_retained_nodes(rec) == [1, 2, 3]

    def test_embedded_tet_host_keeps_its_four_nodes(self) -> None:
        rec = _tie_record([0.25] * 4, kind=ConstraintKind.EMBEDDED)
        assert _embedded_retained_nodes(rec) == [1, 2, 3, 4]

    def test_quad_tie_without_weights_fails_loud(self) -> None:
        rec = InterpolationRecord(
            kind=ConstraintKind.TIE, slave_node=100,
            master_nodes=[1, 2, 3, 4], weights=None, dofs=[1, 2, 3],
        )
        with pytest.raises(ValueError, match="quad4 master face"):
            _embedded_retained_nodes(rec)


def _tied_stack():
    """A LOW x LOW x 1 block (quad4 top faces, 0.5 grid) and a HIGH-side
    block standing on it, shifted by SHIFT so its bottom nodes fall
    inside the master quads; ``top`` tied to ``bot``."""
    from apeGmsh import apeGmsh

    with apeGmsh(model_name="tie_quad_face", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, LOW, LOW, 1.0, label="low")
        g.model.geometry.add_box(SHIFT[0], SHIFT[1], 1.0, HIGH, HIGH, 1.0,
                                 label="high")
        g.physical.add_volume(["low", "high"], name="Vol")
        top = g.model.select(None, dim=2).in_box(
            (-TOL, -TOL, 1.0 - TOL), (LOW + TOL, LOW + TOL, 1.0 + TOL))
        g.physical.add_surface(
            [t for t in top.result().tags()
             if _face_side(g, t) == LOW], name="top")
        g.physical.add_surface(
            [t for t in top.result().tags()
             if _face_side(g, t) == HIGH], name="bot")
        base = g.model.select(None, dim=2).in_box(
            (-TOL, -TOL, -TOL), (LOW + TOL, LOW + TOL, TOL))
        g.physical.add_surface(list(base.result().tags()), name="Base")
        cap = g.model.select(None, dim=2).in_box(
            (-TOL, -TOL, 2.0 - TOL), (LOW + TOL, LOW + TOL, 2.0 + TOL))
        g.physical.add_surface(list(cap.result().tags()), name="Cap")
        g.constraints.tie("top", "bot", dofs=[1, 2, 3], stiffness=K)
        g.mesh.recipe.structured(size=0.5, fallback="strict")
        return g.mesh.queries.get_fem_data(dim=None)


def _face_side(g, tag: int) -> float:
    import gmsh
    x0, y0, _z0, x1, _y1, _z1 = gmsh.model.getBoundingBox(2, int(tag))
    return round(x1 - x0, 6)


def _ops(fem):
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU, rho=0.0)
    ops.element.stdBrick(pg="Vol", material=mat)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    return ops


def _tie_lines(deck: Path) -> list[list[str]]:
    return [ln.split() for ln in deck.read_text(encoding="utf-8").splitlines()
            if ln.startswith("element ASDEmbeddedNodeElement ")]


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    d = tmp_path_factory.mktemp("tie_quad_face")
    fem = _tied_stack()
    deck = d / "stack.tcl"
    _ops(fem).tcl(str(deck), flat=True)
    return fem, deck


def test_every_tie_line_has_three_retained_nodes(stack) -> None:
    from apeGmsh.opensees._internal.build import interpolation_records

    fem, deck = stack
    lines = _tie_lines(deck)
    recs = [r for r in interpolation_records(fem.elements.constraints)
            if r.kind == ConstraintKind.TIE]
    assert lines, "the tie emitted no ASDEmbeddedNodeElement"
    assert len(lines) == len(recs)
    for tok in lines:
        k = tok.index("-K")
        assert len(tok[4:k]) == 3, " ".join(tok)


def test_the_slave_lies_in_its_triangle(stack) -> None:
    fem, deck = stack
    xyz = dict(zip((int(i) for i in fem.nodes.ids),
                   np.asarray(fem.nodes.coords, dtype=float)))
    interior = 0
    for tok in _tie_lines(deck):
        s = xyz[int(tok[3])]
        a, b, c = (xyz[int(t)] for t in tok[4:7])
        # Barycentric coordinates of s in the (planar, z = 1) triangle.
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        lam = np.linalg.solve(m, s[:2] - a[:2])
        bary = np.array([1.0 - lam.sum(), lam[0], lam[1]])
        assert bary.min() >= -1e-9, (tok, bary)
        interior += bool(bary.min() > 1e-9)
    # The shift puts slave nodes strictly inside master faces, so the
    # check is not only met by corner coincidences.
    assert interior > 0


def test_the_record_keeps_the_four_quad_corners(stack) -> None:
    fem, _deck = stack
    from apeGmsh.opensees._internal.build import interpolation_records

    recs = [r for r in interpolation_records(fem.elements.constraints)
            if r.kind == ConstraintKind.TIE]
    assert recs
    assert {len(r.master_nodes) for r in recs} == {4}
    for r in recs:
        assert abs(float(np.sum(r.weights)) - 1.0) < 1e-12


def _dist_bin() -> "Path | None":
    d = os.environ.get("APEGMSH_OPENSEES_BIN")
    if not d:
        return None
    p = Path(d)
    return p if (p / "OpenSees.exe").is_file() else None


@pytest.mark.subprocess
@pytest.mark.skipif(
    _dist_bin() is None,
    reason="APEGMSH_OPENSEES_BIN unset or does not hold OpenSees.exe",
)
def test_the_tied_stack_solves(stack, tmp_path) -> None:
    """Before the fix the 4-node tie elements were zero-volume tets and
    the first solve failed on a singular matrix."""
    fem, _deck = stack
    ops = _ops(fem)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Cap", forces=(0.0, 0.0, -1.0))
    ops.tcl(str(tmp_path / "model.tcl"), flat=True)
    cap_ids = sorted({int(n) for n in fem.nodes.physical.node_ids("Cap")})
    probe = cap_ids[0]
    driver = tmp_path / "run.tcl"
    driver.write_text(
        "source model.tcl\n"
        "constraints Transformation\nnumberer RCM\nsystem UmfPack\n"
        "test NormDispIncr 1e-10 20\nalgorithm Newton\n"
        "integrator LoadControl 1.0\nanalysis Static\n"
        "set ok [analyze 1]\n"
        f"puts \"RESULT $ok [nodeDisp {probe} 3]\"\n",
        encoding="utf-8",
    )
    dist = _dist_bin()
    assert dist is not None
    r = subprocess.run(
        [str(dist / "OpenSees.exe"), str(driver)], cwd=tmp_path,
        capture_output=True, text=True, timeout=300,
    )
    out = r.stdout + r.stderr
    line = next(ln for ln in out.splitlines() if ln.startswith("RESULT"))
    ok, uz = line.split()[1:3]
    assert ok == "0", out[-2000:]
    assert np.isfinite(float(uz)) and float(uz) < 0.0, out[-2000:]
