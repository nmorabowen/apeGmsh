"""Ops-only element tags land above every FEM id under ``"sequential"`` (#2926).

ADR 0111 D2, the sequential rule. Under the default
``element_tags="sequential"`` a physical-group element keeps the tag the
one-pass plan gave it (the block just above the element primitives' own
seed, spec by spec), while every element OpenSees alone knows (an
*ops-only* element: a node-pair spec, an MP element, an interface
``zeroLength``) takes a tag strictly above the snapshot's largest FEM
element id. Before #2926 such a tag continued the physical-group run and
could equal a paired element's FEM id, so one id named two elements of
the archive (#2076; the reader warns ``WarnElementIdCollision``, #2929).

The oracle is a closed form: the largest FEM id of each fixture is read
from its snapshot, and every ops-only tag must exceed it; the paired tags
are the contiguous run above the element seed. Each case fails with the
fix reverted (the reverted plan puts the ops-only tags at the end of the
paired run, inside the FEM-id range).
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir

_ELEMENT = re.compile(r"^element\s+(\S+)\s+(\d+)\s+(.*)$")


def _element_lines(path: Path) -> list[tuple[str, int, str]]:
    """``(type, tag, rest)`` of every ``element`` line, in deck order."""
    out = []
    for ln in path.read_text().splitlines():
        m = _ELEMENT.match(ln.strip())
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3)))
    return out


# ---------------------------------------------------------------------------
# The #2926 repro: a stdBrick block with surface groups and five springs
# ---------------------------------------------------------------------------

def _block_with_springs(tmp_path: Path) -> tuple[Any, Path, Path]:
    """4x4x4 ``stdBrick`` block whose boundary quads (groups ``top`` and
    ``bot``) take the low FEM ids, plus five node-pair ``ZeroLength``
    ground springs. Returns ``(fem, deck, model_h5)``."""
    side = 10.0
    with apeGmsh(model_name="block", verbose=False, units="kN_m_s") as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, side, side, side, label="v")
        g.physical.add_volume("v", name="Vol")
        for z, name in ((side, "top"), (0.0, "bot")):
            tags = g.model.select(None, dim=2).in_box(
                (-1, -1, z - 0.1), (side + 1, side + 1, z + 0.1),
            ).result().tags()
            g.physical.add_surface(tags, name=name)
        grounds = [
            g.decouple_node(coords=(0.0, 0.0, -1.0 - k), label=f"gnd_{k}")
            for k in range(5)
        ]
        g.mesh.recipe.structured(size=2.5, fallback="strict")
        fem = g.mesh.queries.get_fem_data(dim=None)
    decoupled = {int(n) for n in fem.nodes.decoupled_ids}
    mesh_nodes = [int(n) for n in fem.nodes.ids if int(n) not in decoupled]
    ops = apeSees(fem, _artifacts=False)          # the default: sequential
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=2.0e8, nu=0.3, rho=0.0, name="s")
    ops.element.stdBrick(pg="Vol", material=steel)
    spring = ops.uniaxialMaterial.ElasticMaterial(E=2.0e8)
    dirs = tuple(ZeroLengthMatDir(material=spring, dof=d) for d in (1, 2, 3))
    for k, gnd in enumerate(grounds):
        ops.element.ZeroLength(nodes=(gnd, mesh_nodes[k]), mat_dirs=dirs)
        ops.fix(nodes=[int(gnd.tag)], dofs=(1, 1, 1))
    deck, h5 = tmp_path / "block.tcl", tmp_path / "model.h5"
    ops.tcl(str(deck))
    ops.h5(str(h5))
    return fem, deck, h5


def test_block_springs_land_above_every_fem_id(tmp_path: Path) -> None:
    fem, deck, h5 = _block_with_springs(tmp_path)
    top = int(np.max(fem.elements.ids))
    bricks = np.asarray(fem.elements.select(pg="Vol").ids, dtype=np.int64)
    # The quads take FEM ids 1..32, the bricks 33..96, and the snapshot's
    # lower-dimensional elements run on above them.
    assert bricks.min() == 33 and bricks.max() == 96 < top, "fixture"

    lines = _element_lines(deck)
    brick_tags = [t for kind, t, _ in lines if kind == "stdBrick"]
    spring_tags = [t for kind, t, _ in lines if kind == "zeroLength"]
    # Six element primitives seed tags 1..6; the 64 bricks keep 7..70,
    # the run they had before #2926.
    assert brick_tags == list(range(7, 71))
    # Every ops-only tag clears the largest FEM id, in declaration order.
    assert spring_tags == list(range(top + 1, top + 6))

    # The archive pairs every brick with its FEM id and lists the springs
    # without one; no sentinel tag is a paired FEM id, so a read through
    # the archive warns nothing.
    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from apeGmsh.results.readers._tag_translation import (
        ElementTagTranslator,
        WarnElementFilterUnread,
        WarnElementIdCollision,
    )

    model = OpenSeesModel.from_h5(str(h5))
    recs = model.elements()
    paired = {int(r.fem_eid) for r in recs if int(r.fem_eid) >= 0}
    sentinel = {int(r.tag) for r in recs if int(r.fem_eid) < 0}
    assert sentinel == set(spring_tags)
    assert not sentinel & paired
    t = ElementTagTranslator.from_model(model)
    t.declare_recorded(np.array([int(r.tag) for r in recs]))
    assert t.describes_files()
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnElementIdCollision)
        warnings.simplefilter("error", WarnElementFilterUnread)
        bucket, back = t.read_translation(bricks)
        np.testing.assert_array_equal(back(bucket), bricks)
        bucket, back = t.read_translation(np.asarray(spring_tags))
        np.testing.assert_array_equal(back(bucket), spring_tags)


# ---------------------------------------------------------------------------
# A node-pair spec between two physical-group specs
# ---------------------------------------------------------------------------

def test_interleaved_spring_keeps_plan_order_and_the_paired_run(
    tmp_path: Path,
) -> None:
    """Beams B1 (FEM ids 30, 10), a spring, beams B2 (FEM id 20).

    The paired rows take one contiguous run above the three element
    primitives' seed (4, 5, 6), in spec order; the spring, declared
    between them, takes its tag after the reserve (31), and the deck
    still writes the specs in declaration order."""
    with apeGmsh(model_name="interleave", verbose=False) as g:
        e1 = gmsh.model.addDiscreteEntity(1)
        e2 = gmsh.model.addDiscreteEntity(1)
        gmsh.model.mesh.addNodes(
            1, e1, [1, 2, 3], [0, 0, 0, 1, 0, 0, 2, 0, 0])
        gmsh.model.mesh.addNodes(
            1, e2, [4, 5], [3, 0, 0, 3, 0, 0])
        gmsh.model.mesh.addElementsByType(e1, 1, [30, 10], [1, 2, 2, 3])
        gmsh.model.mesh.addElementsByType(e2, 1, [20], [3, 4])
        gmsh.model.addPhysicalGroup(1, [e1], name="B1")
        gmsh.model.addPhysicalGroup(1, [e2], name="B2")
        fem = g.mesh.queries.get_fem_data(dim=1)
        ops = apeSees(fem, _artifacts=False)
        ops.model(ndm=3, ndf=6)
        beam: dict[str, float] = dict(
            A=1.0, E=1.0, G=1.0, J=1.0, Iy=1.0, Iz=1.0)
        # One root per spec, declared in spec order, so the topological
        # order (Kahn, stable) is B1, spring, B2.
        tr1 = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        k = ops.uniaxialMaterial.ElasticMaterial(E=1.0)
        tr2 = ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))
        ops.element.elasticBeamColumn(pg="B1", transf=tr1, **beam)
        ops.element.ZeroLength(
            nodes=(4, 5), mat_dirs=(ZeroLengthMatDir(material=k, dof=1),))
        ops.element.elasticBeamColumn(pg="B2", transf=tr2, **beam)
        deck = tmp_path / "interleave.tcl"
        ops.tcl(str(deck))
    assert int(np.max(fem.elements.ids)) == 30
    assert [(kind, t) for kind, t, _ in _element_lines(deck)] == [
        ("elasticBeamColumn", 4), ("elasticBeamColumn", 5),
        ("zeroLength", 31),
        ("elasticBeamColumn", 6),
    ]


# ---------------------------------------------------------------------------
# MP elements and interface zeroLength
# ---------------------------------------------------------------------------

def test_coupling_element_lands_above_every_fem_id(tmp_path: Path) -> None:
    """A kinematic coupling (an MP element) under ``"sequential"``: the
    beams keep 2.., the coupling element clears the largest FEM id, which
    is a point element that is not emitted."""
    with apeGmsh(model_name="seq_kc", verbose=False) as g:
        p0 = g.model.geometry.add_point(0, 0, 0)
        p1 = g.model.geometry.add_point(2, 0, 0)
        p2 = g.model.geometry.add_point(2, 1, 0)
        line = g.model.geometry.add_line(p0, p1)
        g.physical.add(1, [line], name="B")
        g.physical.add(0, [p1], name="M")
        g.physical.add(0, [p2], name="S")
        g.constraints.kinematic_coupling(
            "M", "S", master_point=(2, 0, 0), name="kc")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data()
    beams = sorted(int(e) for e in fem.elements.select(pg="B").ids)
    top = int(np.max(fem.elements.ids))
    assert top > len(beams) + 1, "fixture: a one-pass tag would be a FEM id"

    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="B", transf=tr, A=1.0, E=1.0, G=1.0, J=1.0, Iy=1.0, Iz=1.0)
    deck = tmp_path / "kc.tcl"
    ops.tcl(str(deck))
    lines = _element_lines(deck)
    assert [t for kind, t, _ in lines if kind == "elasticBeamColumn"] == (
        list(range(2, 2 + len(beams))))
    assert [t for kind, t, _ in lines
            if kind == "LadrunoKinematicCoupling"] == [top + 1]


_NORMAL = NormalLaw(kind="ent", k_per_area=1.0e9)
_TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)


def _curve_at_x(surface: int, x: float) -> int:
    for _dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < 1e-6 and abs(bb[3] - x) < 1e-6:
            return abs(tag)
    raise AssertionError(f"no boundary curve at x={x}")


def _interface_deck(partition: int, tmp_path: Path) -> tuple[Any, Path]:
    with apeGmsh(model_name="seq_iface", verbose=False) as g:
        left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, left), (2, right)], n=4)
        g.mesh.generation.generate(2)
        g.physical.add(2, [left], name="rock")
        g.physical.add(2, [right], name="liner")
        g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
        g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
        g.constraints.interface(
            "face", "wire", normal=_NORMAL, tangential=_TANGENTIAL,
            thickness=0.5, name="RockLiner")
        if partition:
            g.mesh.partitioning.partition(partition)
        fem = g.mesh.queries.get_fem_data()
    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in ("rock", "liner"):
        ops.element.FourNodeQuad(
            pg=pg, thickness=0.5, material=mat, plane_type="PlaneStrain")
    deck = tmp_path / f"iface_{partition}.tcl"
    ops.tcl(str(deck))
    return fem, deck


def test_interface_zero_length_lands_above_every_fem_id_on_every_rank(
    tmp_path: Path,
) -> None:
    """The interface ``zeroLength`` elements clear the largest FEM id, and
    a partitioned emit (2 ranks) writes every element with the flat
    emit's tag (ADR 0027 tag determinism)."""
    decks = {}
    for partition in (0, 2):
        fem, deck = _interface_deck(partition, tmp_path)
        top = int(np.max(fem.elements.ids))
        n_quads = sum(len(fem.elements.select(pg=pg).ids)
                      for pg in ("rock", "liner"))
        lines = _element_lines(deck)
        quads = sorted({t for kind, t, _ in lines if kind == "quad"})
        zl = sorted({t for kind, t, _ in lines if kind == "zeroLength"})
        assert quads == list(range(3, 3 + n_quads)), partition
        assert top > quads[-1], "fixture: a one-pass tag would be a FEM id"
        assert zl and zl == list(range(top + 1, top + 1 + len(zl))), partition
        decks[partition] = {(kind, t, rest) for kind, t, rest in lines}
    assert decks[0] == decks[2]
