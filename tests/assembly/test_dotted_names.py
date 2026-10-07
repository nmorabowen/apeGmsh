"""AS4-c (#1549): an instance's dotted names follow one rule on both sides.

A source physical group whose name already holds a ``.`` or ``/`` is
prefixed by the merge engine with the ADR 0038 alternation (the new
separator is ``.`` after an even number of separators, ``/`` after an odd
number). The rehydrator used to write ``{label}.{pg}`` regardless, so its
element spec named a group the merged FEM does not hold and the build
raised. It now asks the merge engine's rule.

Oracles, each naming the right answer independently of the code under test:

* the merged group names, written out by hand from the ADR 0038 rule:
  ``deck.slab`` -> ``A/deck.slab``, ``deck/slab`` -> ``A/deck/slab``,
  ``bay.deck/slab`` -> ``A.bay.deck/slab``;
* parity (INV-4): one instance's deck is the source deck, ids shifted back
  by the closed-form relocation ``1_000_000 - source_min``;
* shared namespace: two instances of one dotted source hold two distinct
  groups and twice the source's elements;
* reload: the assembly archive carries the dotted names, and the archive
  itself bridges as an instance (``X``), whose names nest once more by the
  same rule (``X.A/deck.slab``, ``X/A.Right``);
* bridge names (materials, the carried rebar material) stay flat
  ``{label}.{name}`` on both sides, dotted or not.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh

GRANULE = 1_000_000

#: Source PG name -> the name instance ``A`` holds it under (ADR 0038).
MERGED = {
    "deck.slab": "A/deck.slab",
    "deck/slab": "A/deck/slab",
    "bay.deck/slab": "A.bay.deck/slab",
}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def block_fem(dotted: str):
    """A 2x2x2 hex8 block: PG ``dotted`` (x<5) and PG ``Right`` (x>5)."""
    with apeGmsh(model_name="dotted", verbose=False) as g:
        a = g.model.geometry.add_box(0.0, 0.0, 0.0, 5.0, 10.0, 10.0)
        b = g.model.geometry.add_box(5.0, 0.0, 0.0, 5.0, 10.0, 10.0)
        g.model.boolean.fragment([a], [b])
        g.model.sync()
        vols = sorted(g.model.select(None, dim=3).result().tags())
        g.physical.add(3, [vols[0]], name=dotted)
        g.physical.add(3, [vols[1]], name="Right")
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        return g.mesh.queries.get_fem_data(dim=None)


def declare_block(ops, dotted: str) -> None:
    ops.model(ndm=3, ndf=3)
    conc = ops.nDMaterial.ElasticIsotropic(E=30_000.0, nu=0.2, rho=2.4e-9,
                                           name="conc.c30")
    steel = ops.nDMaterial.ElasticIsotropic(E=200_000.0, nu=0.3, rho=7.85e-9,
                                            name="steel")
    ops.element.stdBrick(pg=dotted, material=conc)
    ops.element.stdBrick(pg="Right", material=steel)


def cage_fem():
    """A tet-meshed column with one conformal bar on material ``bar.steel``."""
    from apeGmsh._kernel.defs.rebar import Cage

    with apeGmsh(model_name="cage", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="ConcreteVol")
        bar = g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)],
                          db=0.0254, material="bar.steel", name="L1")
        g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                      coupling="conformal", emit_elements=True)
        g.physical.add_volume("ConcreteVol", name="Conc")
        g.mesh.sizing.set_global_size(0.25)
        g.mesh.generation.generate(dim=3)
        return g.mesh.queries.get_fem_data(dim=3)


def declare_cage(ops) -> None:
    ops.model(ndm=3, ndf=3)
    ops.uniaxialMaterial.ElasticMaterial(E=200_000.0, name="bar.steel")
    conc = ops.nDMaterial.ElasticIsotropic(E=30_000.0, nu=0.2, rho=0.0,
                                           name="conc")
    ops.element.FourNodeTetrahedron(pg="Conc", material=conc)


def _write(path: Path, fem, declare, *args) -> Path:
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    declare(ops, *args)
    ops.h5(str(path))
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("as4c")
    out = {name: _write(d / f"block{k}.h5", block_fem(name), declare_block, name)
           for k, name in enumerate(MERGED)}
    out["cage"] = _write(d / "cage.h5", cage_fem(), declare_cage)
    return out


def _assembly(*instances):
    from apeGmsh.assembly import Assembly

    asm = Assembly("asm")
    for label, path in instances:
        asm.instance(label, path)
    return asm


def _bridge(asm, ndf: int = 3):
    from apeGmsh.assembly import AssemblyRankWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssemblyRankWarning)
        return asm.bridge(ndm=3, ndf=ndf)


def _deck(ops, path: Path) -> str:
    ops.tcl(str(path), flat=True)
    return path.read_text(encoding="utf-8")


def _source_min(path: Path) -> int:
    from apeGmsh.mesh import FEMData

    src = FEMData.from_h5(str(path))
    return min([int(np.min(src.nodes.ids))]
               + [int(np.min(g.ids)) for g in src.elements if len(g.ids)])


def _names(ops, path: Path) -> set[tuple[str, str]]:
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    ops.h5(str(path))
    return {(n, k) for n, k, _ in OpenSeesModel.from_h5(str(path)).names()}


# ---------------------------------------------------------------------------
# #1549: the element spec names the group the merged FEM holds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dotted", sorted(MERGED))
def test_one_instance_on_a_dotted_pg_deck_equals_the_source_deck(
        files, dotted, tmp_path):
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees import apeSees

    src = files[dotted]
    ref = apeSees(FEMData.from_h5(str(src)), element_tags="fem")
    declare_block(ref, dotted)
    want = _deck(ref, tmp_path / "ref.tcl")

    ops = _bridge(_assembly(("A", src)))
    assert set(ops.fem.elements.physical.names()) == {MERGED[dotted], "A.Right"}
    got = _deck(ops, tmp_path / "asm.tcl")
    off = GRANULE - _source_min(src)
    got = re.sub(r"(?<![\w.\-])\d{7,}(?![\w.])",
                 lambda m: str(int(m.group(0)) - off), got)
    assert got == want
    assert want.count("element stdBrick") == 8


@pytest.mark.parametrize("dotted", sorted(MERGED))
def test_two_instances_of_a_dotted_pg_do_not_collide(files, dotted, tmp_path):
    ops = _bridge(_assembly(("A", files[dotted]), ("B", files[dotted])))
    b_name = "B" + MERGED[dotted][1:]
    assert set(ops.fem.elements.physical.names()) == {
        MERGED[dotted], b_name, "A.Right", "B.Right"}
    for pg in (MERGED[dotted], b_name):
        assert len(ops.fem.elements.select(pg=pg).ids) == 4
    assert _deck(ops, tmp_path / "two.tcl").count("element stdBrick") == 16
    assert {("A.conc.c30", "nDMaterial"), ("B.conc.c30", "nDMaterial"),
            ("A.steel", "nDMaterial"), ("B.steel", "nDMaterial")} <= _names(
        ops, tmp_path / "two.h5")


# ---------------------------------------------------------------------------
# Reload: the archive carries the names and bridges as an instance itself
# ---------------------------------------------------------------------------

def test_the_archive_round_trips_and_re_instances_by_the_same_rule(
        files, tmp_path):
    from apeGmsh.assembly import Assembly
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    asm = _assembly(("A", files["deck.slab"]))
    _bridge(asm)
    archive = tmp_path / "archive.h5"
    asm.h5(archive, model_name="dotted")

    back = FEMData.from_h5(str(archive))
    assert set(back.elements.physical.names()) == {"A/deck.slab", "A.Right"}
    assert ("A.conc.c30", "nDMaterial") in {
        (n, k) for n, k, _ in OpenSeesModel.from_h5(str(archive)).names()}
    assert [i.label for i in Assembly.from_h5(archive).instances] == ["A"]

    # The archive's own element specs name 'A/deck.slab' and 'A.Right';
    # composed again under 'X' they nest once more (ADR 0038).
    ops = _bridge(_assembly(("X", archive)))
    assert set(ops.fem.elements.physical.names()) == {"X.A/deck.slab",
                                                      "X/A.Right"}
    assert _deck(ops, tmp_path / "x.tcl").count("element stdBrick") == 8
    assert {("X.A.conc.c30", "nDMaterial"), ("X.A.steel", "nDMaterial")} \
        <= _names(ops, tmp_path / "x.h5")


# ---------------------------------------------------------------------------
# Bridge names: one flat rule on both sides, the carried rebar included
# ---------------------------------------------------------------------------

def test_a_dotted_rebar_material_binds_to_the_rehydrated_material(
        files, tmp_path):
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    (bar,) = FEMData.from_h5(str(files["cage"])).elements.rebar_elements
    assert bar.material == "bar.steel"
    ops = _bridge(_assembly(("a", files["cage"]), ("b", files["cage"])))
    assert [r.material for r in ops.fem.elements.rebar_elements] == [
        "a.bar.steel", "b.bar.steel"]

    out = tmp_path / "cage.h5"
    ops.h5(str(out))
    tag = {n: t for n, k, t in OpenSeesModel.from_h5(str(out)).names()
           if k == "uniaxialMaterial"}
    assert set(tag) == {"a.bar.steel", "b.bar.steel"}
    trusses = [ln.split() for ln in _deck(ops, tmp_path / "cage.tcl").splitlines()
               if ln.startswith("element CorotTruss")]
    assert len(trusses) == 2 * len(bar.connectivity)
    assert sorted(int(t[-1]) for t in trusses) == sorted(
        [tag["a.bar.steel"]] * len(bar.connectivity)
        + [tag["b.bar.steel"]] * len(bar.connectivity))
