"""ADR 0117 P1 (AS1): two instances of one file, one tie, rehydration parity.

Oracles, each naming the right answer independently of the code under test:

* INV-1 — every instance-owned name is ``{label}.{local}``; a bad or
  clashing label raises before anything is recorded.
* same file twice — both instances carry their own ``steel``
  (``pier_1.steel``, ``pier_2.steel``: two ``nDMaterial`` lines), each
  element row points at its own instance's tag, and the source is read once.
* INV-4 — one instance at the identity transform emits the deck the source
  script emits on its own FEM, once its FEM ids are shifted back by the
  closed-form relocation ``1_000_000 - source_min``. Run for a material
  (block) and a section (plate).
* INV-6 — instancing is exact: coordinates are ``R x + t`` to 1e-12 and
  connectivity is the source's plus the closed-form offset.
* INV-7 — a tie that resolves no record raises naming both ports; a bare
  port raises listing the instances.
* reload — two builds of one script emit identical decks.
* INV-3 — no module of the package names ``TagAllocator`` (the AST lock in
  ``tests/opensees/contract/test_tag_law_lock.py`` covers ``allocate*``).
"""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh

# Units: N, mm, MPa.
E = 200_000.0
#: Nonzero and distinct, so a rehydrator that swaps two params changes the deck.
NU = 0.3
RHO = 7.85e-9
SIDE = 10.0
H = 10.0
TOL = 0.01
#: The merge engine's reservation granularity: instance k (1-based) of a
#: source whose ids are below it starts at ``k * GRANULE`` (ADR 0038).
GRANULE = 1_000_000


def _faces_at_z(g, z: float) -> list[int]:
    lo = (-SIDE, -SIDE, z - TOL)
    hi = (2 * SIDE, 2 * SIDE, z + TOL)
    return g.model.select(None, dim=2).in_box(lo, hi).result().tags()


def block_fem(workdir: Path):
    """A 2x2x2 hex8 block, PGs ``Vol``, ``bot`` (z=0), ``top`` (z=H)."""
    with apeGmsh(model_name="block", save_to=str(workdir / "block_mesh.h5"),
                 overwrite=True) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, SIDE, SIDE, H, label="v")
        g.physical.add_volume("v", name="Vol")
        g.physical.add_surface(_faces_at_z(g, 0.0), name="bot")
        g.physical.add_surface(_faces_at_z(g, H), name="top")
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        return g.mesh.queries.get_fem_data(dim=None)


def declare_block(ops, *, extra: bool = False) -> None:
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU, rho=RHO, name="steel")
    ops.element.stdBrick(pg="Vol", material=steel)
    if extra:  # a material AS1 does not rehydrate
        ops.uniaxialMaterial.ElasticPP(E=1.0, epsyP=0.01)


def plate_fem(workdir: Path):
    """A 2x2 quad4 plate, PG ``Slab``."""
    with apeGmsh(model_name="plate", save_to=str(workdir / "plate_mesh.h5"),
                 overwrite=True) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, SIDE, SIDE, label="p")
        g.physical.add_surface("p", name="Slab")
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        return g.mesh.queries.get_fem_data(dim=None)


def declare_plate(ops) -> None:
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(
        E=E, nu=0.2, h=0.5, rho=2.4e-9, name="slab")
    ops.element.ShellMITC4(pg="Slab", section=sec)


def write_instance(path: Path, fem, declare, **kw) -> Path:
    """Write ``fem`` plus its declared model content as ``path``."""
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    declare(ops, **kw)
    ops.h5(str(path))
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("as1")
    bfem = block_fem(d)
    pfem = plate_fem(d)
    return {
        "block": write_instance(d / "block.h5", bfem, declare_block),
        "block_extra": write_instance(
            d / "block_extra.h5", bfem, declare_block, extra=True),
        "plate": write_instance(d / "plate.h5", pfem, declare_plate),
        "dir": d,
    }


def _stack(files, **tie_kw):
    from apeGmsh.assembly import Assembly

    asm = Assembly("stack")
    asm.instance("pier_1", files["block"])
    asm.instance("pier_2", files["block"], translate=(0.0, 0.0, H))
    asm.tie("pier_1.top", "pier_2.bot", **({"enforce": "equation",
                                            "dofs": [1, 2, 3]} | tie_kw))
    return asm


def _deck(ops, path: Path) -> str:
    ops.tcl(str(path), flat=True)
    return path.read_text(encoding="utf-8")


def _source_min(path: Path) -> int:
    from apeGmsh.mesh import FEMData

    src = FEMData.from_h5(str(path))
    ids = [int(np.min(src.nodes.ids))] + [
        int(np.min(g.ids)) for g in src.elements if len(g.ids)]
    return min(ids)


# ---------------------------------------------------------------------------
# INV-1 — labels and names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["", "a.b", "a/b", "a b", "_a", "a_"])
def test_bad_instance_label_raises_before_recording(files, label):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = Assembly("x")
    with pytest.raises(AssemblyError):
        asm.instance(label, files["block"])
    assert asm.instances == ()


def test_clashing_label_and_missing_file_raise_before_recording(files):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = Assembly("x").instance("p", files["block"])
    with pytest.raises(AssemblyError, match="already declared"):
        asm.instance("p", files["block"])
    with pytest.raises(AssemblyError, match="no file"):
        asm.instance("q", files["dir"] / "nope.h5")
    with pytest.raises(AssemblyError, match="axis is zero"):
        asm.instance("q", files["block"], rotate=((0.0, 0.0, 0.0), 1.0))
    assert [i.label for i in asm.instances] == ["p"]


def test_same_file_twice_registers_two_namespaced_materials(files, monkeypatch):
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    reads: list[str] = []
    real = OpenSeesModel.from_h5.__func__

    def counting(cls, path, **kw):
        reads.append(str(path))
        return real(cls, path, **kw)

    monkeypatch.setattr(OpenSeesModel, "from_h5", classmethod(counting))
    ops = _stack(files).bridge(ndm=3, ndf=3)
    assert reads == [str(files["block"])], "the shared source is read once"
    monkeypatch.undo()

    out = files["dir"] / "stack_names.h5"
    ops.h5(str(out))
    names = OpenSeesModel.from_h5(str(out)).names()
    assert names == (("pier_1.steel", "nDMaterial", 1),
                     ("pier_2.steel", "nDMaterial", 2))

    deck = _deck(ops, files["dir"] / "stack.tcl")
    mats = [ln for ln in deck.splitlines() if ln.startswith("nDMaterial")]
    assert mats == [f"nDMaterial ElasticIsotropic {k} 200000.0 0.3 7.85e-09"
                    for k in (1, 2)]
    # Each brick's last token is its material tag; FEM ids name the instance.
    by_instance: dict[int, set[str]] = {1: set(), 2: set()}
    for ln in deck.splitlines():
        if ln.startswith("element stdBrick"):
            tok = ln.split()
            by_instance[int(tok[2]) // GRANULE].add(tok[-1])
    assert by_instance == {1: {"1"}, 2: {"2"}}


# ---------------------------------------------------------------------------
# INV-4 — rehydration parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["block", "plate"])
def test_one_instance_deck_equals_the_source_deck(files, kind, tmp_path):
    """The source script's own deck, from the assembly's rehydration."""
    from apeGmsh.assembly import Assembly
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees import apeSees

    src = files[kind]
    declare = declare_block if kind == "block" else declare_plate
    ref = apeSees(FEMData.from_h5(str(src)), element_tags="fem")
    declare(ref)
    want = _deck(ref, tmp_path / "ref.tcl")

    ndf = 3 if kind == "block" else 6
    got = _deck(Assembly("one").instance("inst", src).bridge(ndm=3, ndf=ndf),
                tmp_path / "asm.tcl")
    off = GRANULE - _source_min(src)

    def shift_back(m: re.Match) -> str:
        return str(int(m.group(0)) - off)

    # Integer tokens >= GRANULE are relocated FEM ids (node and element
    # tags); material/section tags and coordinates never reach it.
    got = re.sub(r"(?<![\w.\-])\d{7,}(?![\w.])", shift_back, got)
    assert got == want
    assert "element" in want and ("nDMaterial" in want or "section" in want)


def test_unsupported_model_content_raises(files):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = Assembly("x").instance("p", files["block_extra"])
    with pytest.raises(AssemblyError, match="uniaxialMaterial 'ElasticPP'"):
        asm.bridge(ndm=3, ndf=3)


# ---------------------------------------------------------------------------
# INV-6 — instancing is exact
# ---------------------------------------------------------------------------

def test_instancing_is_rotation_then_translation_with_relocated_ids(files):
    from apeGmsh.assembly import Assembly
    from apeGmsh.mesh import FEMData

    theta = math.pi / 2
    t = np.array([30.0, 0.0, 0.0])
    asm = (Assembly("placed")
           .instance("a", files["block"])
           .instance("b", files["block"], translate=tuple(t),
                     rotate=((0.0, 0.0, 1.0), theta)))
    fem = asm.bridge(ndm=3, ndf=3).fem
    src = FEMData.from_h5(str(files["block"]))
    smin = _source_min(files["block"])
    rot = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                    [math.sin(theta), math.cos(theta), 0.0],
                    [0.0, 0.0, 1.0]])
    coord_of = dict(zip((int(i) for i in fem.nodes.ids),
                        np.asarray(fem.nodes.coords, dtype=float)))
    conn_of = {int(e): tuple(int(n) for n in c)
               for g in fem.elements for e, c in zip(g.ids, g.connectivity)}
    for k, (R, tr) in enumerate([(np.eye(3), np.zeros(3)), (rot, t)], start=1):
        off = k * GRANULE - smin
        for nid, x in zip(src.nodes.ids, np.asarray(src.nodes.coords, float)):
            np.testing.assert_allclose(
                coord_of[int(nid) + off], R @ x + tr, rtol=0, atol=1e-12)
        for g in src.elements:
            for e, c in zip(g.ids, g.connectivity):
                assert conn_of[int(e) + off] == tuple(int(n) + off for n in c)


# ---------------------------------------------------------------------------
# INV-7 — ties and ports fail loud
# ---------------------------------------------------------------------------

def test_tie_resolving_no_record_names_both_ports(files):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = (Assembly("apart")
           .instance("pier_1", files["block"])
           .instance("pier_2", files["block"], translate=(0.0, 0.0, 5 * H))
           .tie("pier_1.top", "pier_2.bot", enforce="equation"))
    with pytest.raises(AssemblyError) as info:
        asm.bridge(ndm=3, ndf=3)
    assert "pier_1.top" in str(info.value) and "pier_2.bot" in str(info.value)


def test_tie_to_a_missing_group_raises_at_bridge(files):
    from apeGmsh.assembly import AssemblyError

    with pytest.raises(AssemblyError, match="pier_2.nope"):
        _stack_with_slave(files, "pier_2.nope").bridge(ndm=3, ndf=3)


def _stack_with_slave(files, slave: str):
    from apeGmsh.assembly import Assembly

    return (Assembly("s").instance("pier_1", files["block"])
            .instance("pier_2", files["block"], translate=(0.0, 0.0, H))
            .tie("pier_1.top", slave, enforce="equation"))


@pytest.mark.parametrize("port, match", [
    ("top", r"names no assembly object.*\['pier_1', 'pier_2'\]"),
    ("ghost.top", r"unknown instance 'ghost'.*\['pier_1', 'pier_2'\]"),
    ("pier_2.", "empty name"),
])
def test_bad_port_raises_at_declaration(files, port, match):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = (Assembly("s").instance("pier_1", files["block"])
           .instance("pier_2", files["block"], translate=(0.0, 0.0, H)))
    with pytest.raises(AssemblyError, match=match):
        asm.tie("pier_1.top", port)
    with pytest.raises(AssemblyError, match="contains '.'"):
        asm.tie("pier_1.top", "pier_2.bot", name="a.b")
    assert asm.ties == ()


def test_v1_and_v2_declarations_do_not_mix(files):
    from apeGmsh.assembly import Assembly, AssemblyError

    with pytest.raises(AssemblyError, match="v1"):
        Assembly("x").add("h", str(files["block"])).instance("p", files["block"])
    with pytest.raises(AssemblyError, match="instance/tie"):
        Assembly("x").instance("p", files["block"]).add("h", str(files["block"]))
    with pytest.raises(AssemblyError, match="no parts"):
        Assembly("x").instance("p", files["block"]).materialize()


# ---------------------------------------------------------------------------
# Reload — determinism
# ---------------------------------------------------------------------------

def test_two_builds_emit_identical_decks(files, tmp_path):
    a = _stack(files).bridge(ndm=3, ndf=3)
    b = _stack(files).bridge(ndm=3, ndf=3)
    da, db = _deck(a, tmp_path / "a.tcl"), _deck(b, tmp_path / "b.tcl")
    assert da == db
    assert da.count("equationConstraint") == 27  # 9 interface nodes x 3 dofs
    a.tcl(str(tmp_path / "pa.tcl"))
    b.tcl(str(tmp_path / "pb.tcl"))
    assert (tmp_path / "pa.tcl").read_bytes() == (tmp_path / "pb.tcl").read_bytes()


# ---------------------------------------------------------------------------
# INV-3 — no allocator in the package
# ---------------------------------------------------------------------------

def test_assembly_package_names_no_tag_allocator():
    root = Path(__file__).resolve().parents[2] / "src" / "apeGmsh" / "assembly"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = (node.id if isinstance(node, ast.Name)
                    else node.attr if isinstance(node, ast.Attribute)
                    else node.name if isinstance(node, ast.alias) else None)
            if name is not None and "TagAllocator" in name:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


# ---------------------------------------------------------------------------
# Review round 1 (#1529)
# ---------------------------------------------------------------------------

def test_dedup_key_keeps_two_files_that_share_a_fem_hash_apart(files):
    """``block.h5`` and ``block_extra.h5`` share their mesh, so their FEM
    hash, but only the second carries an ``ElasticPP``. Keyed by hash alone,
    the second instance would reuse the first file's model and build."""
    from apeGmsh.assembly import Assembly, AssemblyError
    from apeGmsh.mesh import FEMData

    a, b = files["block"], files["block_extra"]
    assert (FEMData.from_h5(str(a)).snapshot_id
            == FEMData.from_h5(str(b)).snapshot_id)
    asm = (Assembly("pair").instance("a", a)
           .instance("b", b, translate=(0.0, 0.0, H)))
    with pytest.raises(AssemblyError, match="instance 'b'.*ElasticPP"):
        asm.bridge(ndm=3, ndf=3)


@pytest.mark.parametrize("kw, match", [
    ({"enforce": "bogus"}, "bogus"),
    ({"method": "mortar"}, "mortar"),       # mortar needs enforce="equation"
])
def test_bad_tie_options_raise_at_declaration(files, kw, match):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = (Assembly("s").instance("pier_1", files["block"])
           .instance("pier_2", files["block"], translate=(0.0, 0.0, H)))
    with pytest.raises(AssemblyError, match=match):
        asm.tie("pier_1.top", "pier_2.bot", **kw)
    assert asm.ties == ()


def test_ndf_mismatch_raises(files):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = Assembly("x").instance("slab", files["plate"])
    with pytest.raises(AssemblyError, match="ndf=6.*ndf=3"):
        asm.bridge(ndm=3, ndf=3)
