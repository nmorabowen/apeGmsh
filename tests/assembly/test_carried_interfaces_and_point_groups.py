"""Assembly v2: sources with ``/interfaces`` or a point group bridge (#1586, #1587).

Gaps G4 and G5 of the AS5-a parity table (#1585). Oracles, each naming the
right answer independently of the code under test:

* INV-4 parity, interfaces (G4): a 2-D module with one ``ent``/``epp``
  interface across x=1 and a named user material. The build synthesises
  each pair's ``ENT`` / ``ElasticPP`` uniaxials and ``zeroLength`` from the
  carried ``/interfaces`` stream, so the instance's deck is the source's
  own deck with every FEM id shifted by the closed-form relocation
  ``1_000_000 - source_min`` and the record name prefixed ``{inst}.``.
* INV-4 parity, point group (G5): a hex8 block with a dim-0 ``Corner``
  group. The deck is the source's, shifted the same way, and the group
  arrives as ``{inst}.Corner`` on the one node at the corner plus the
  instance's translation.
* refusals: an interface record that matches no archived ``zeroLength``
  row, or a matched row whose material is named or of a non-D1 type,
  raises :class:`AssemblyError` instead of dropping or re-declaring it.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh

GRANULE = 1_000_000


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _curve_at_x(surface: int, x: float) -> int:
    for _, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < 1e-6 and abs(bb[3] - x) < 1e-6:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def interface_fem():
    """Two abutting unit squares with one interface ``RL`` across x=1."""
    from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw

    with apeGmsh(model_name="iface", verbose=False) as g:
        left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, left), (2, right)], n=2)
        g.mesh.generation.generate(2)
        g.physical.add(2, [left], name="rock")
        g.physical.add(2, [right], name="liner")
        g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
        g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
        g.constraints.interface(
            "face", "wire",
            normal=NormalLaw(kind="ent", k_per_area=1.0e9),
            tangential=TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5),
            thickness=0.5, name="RL")
        return g.mesh.queries.get_fem_data()


def declare_interface(ops) -> None:
    ops.model(ndm=2, ndf=2)
    ops.uniaxialMaterial.ElasticMaterial(E=3.0e7, name="k")


def corner_fem():
    """A 2x2x2 hex8 block ``Vol`` with a point group ``Corner`` at (10,10,10)."""
    with apeGmsh(model_name="corner", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        g.model.sync()
        g.physical.add(3, list(g.model.select(None, dim=3).result().tags()),
                       name="Vol")
        corner = [t for _, t in gmsh.model.getEntities(0)
                  if np.allclose(gmsh.model.getValue(0, t, []), (10, 10, 10))]
        assert len(corner) == 1
        g.physical.add(0, corner, name="Corner")
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        return g.mesh.queries.get_fem_data(dim=None)


def declare_corner(ops) -> None:
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=200_000.0, nu=0.3, rho=0.0, name="m")
    ops.element.stdBrick(pg="Vol", material=mat)


def _write(path: Path, fem, declare) -> Path:
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    declare(ops)
    ops.h5(str(path))
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("g4g5")
    return {
        "iface": _write(d / "iface.h5", interface_fem(), declare_interface),
        "corner": _write(d / "corner.h5", corner_fem(), declare_corner),
    }


def _deck(ops, path: Path) -> str:
    ops.tcl(str(path), flat=True)
    return path.read_text(encoding="utf-8")


def _source_min(path: Path) -> int:
    from apeGmsh.mesh import FEMData

    src = FEMData.from_h5(str(path))
    return min([int(np.min(src.nodes.ids))] + [
        int(np.min(g.ids)) for g in src.elements if len(g.ids)])


def _bridge(*instances, ndm: int, ndf: int):
    from apeGmsh.assembly import Assembly

    asm = Assembly("asm")
    for label, path, kw in instances:
        asm.instance(label, path, **kw)
    return asm.bridge(ndm=ndm, ndf=ndf)


def _parity(src: Path, declare, ndm: int, ndf: int, tmp_path) -> tuple[str, str]:
    """``(want, got)``: the source deck and the one-instance deck, ids shifted back."""
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees import apeSees

    ref = apeSees(FEMData.from_h5(str(src)), element_tags="fem")
    declare(ref)
    want = _deck(ref, tmp_path / "ref.tcl")
    got = _deck(_bridge(("inst", src, {}), ndm=ndm, ndf=ndf), tmp_path / "asm.tcl")
    off = GRANULE - _source_min(src)
    got = re.sub(r"(?<![\w.\-])\d{7,}(?![\w.])",
                 lambda m: str(int(m.group(0)) - off), got)
    return want, got


# ---------------------------------------------------------------------------
# G4 (#1586): /interfaces
# ---------------------------------------------------------------------------

def test_an_interface_source_deck_equals_the_source_deck(files, tmp_path):
    want, got = _parity(files["iface"], declare_interface, 2, 2, tmp_path)
    assert want.count("element zeroLength") == 2
    assert want.count("uniaxialMaterial ENT") == 2
    assert want.count("uniaxialMaterial ElasticPP") == 2
    assert "uniaxialMaterial Elastic 1 30000000.0" in want   # user material first
    assert got.count("# inst.RL") == 2          # compose namespaces the record
    assert got.replace("# inst.RL", "# RL") == want


def test_two_interface_instances_each_carry_their_own_pairs(files, tmp_path):
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    ops = _bridge(("a", files["iface"], {}),
                  ("b", files["iface"], {"translate": (5.0, 0.0, 0.0)}),
                  ndm=2, ndf=2)
    assert sorted(r.name for r in ops.fem.elements.interfaces) == \
        ["a.RL", "a.RL", "b.RL", "b.RL"]
    deck = _deck(ops, tmp_path / "two.tcl")
    assert deck.count("element zeroLength") == 4
    assert deck.count("uniaxialMaterial ENT") == 4
    out = tmp_path / "two.h5"
    ops.h5(str(out))
    names = {(n, k) for n, k, _ in OpenSeesModel.from_h5(str(out)).names()}
    assert names == {("a.k", "uniaxialMaterial"), ("b.k", "uniaxialMaterial")}


def _iface_stub(*, rows, mats, names_ok=True):
    from apeGmsh.opensees._internal.typed_records import (
        ElementRecord,
        MaterialRecord,
    )

    rec = SimpleNamespace(name="RL", master_node=2, slave_node=5,
                          phantom_node=None, orient=(1, 0, 0, 0, 1, 0))
    els = tuple(ElementRecord(type_token="zeroLength", tag=t, args=a,
                              connectivity=a[:2], fem_eid=-1) for t, a in rows)
    uni = tuple(MaterialRecord(type_token=tok, tag=t, params=(1.0,))
                for t, tok in mats)
    fem = SimpleNamespace(elements=SimpleNamespace(interfaces=[rec]))
    return SimpleNamespace(fem=fem, elements=lambda: els,
                           materials_by_family=lambda: {"uniaxial": uni})


_ROW = (2, 5, "-mat", 1, 2, "-dir", 1, 2, "-orient", 1, 0, 0, 0, 1, 0)


@pytest.mark.parametrize("rows, mats, names, match", [
    ((), ((1, "ENT"), (2, "ElasticPP")), {}, "matches 0 archived"),
    (((7, (2, 6) + _ROW[2:]),), ((1, "ENT"), (2, "ElasticPP")), {},
     "matches 0 archived"),
    (((7, _ROW), (8, _ROW)), ((1, "ENT"), (2, "ElasticPP")), {},
     "matches 2 archived"),
    (((7, _ROW),), ((1, "ENT"), (2, "Steel01")), {}, "material tag 2"),
    (((7, _ROW),), ((1, "ENT"),), {}, "material tag 2"),
    (((7, _ROW),), ((1, "ENT"), (2, "ElasticPP")),
     {("uniaxialMaterial", 1): "user"}, "material tag 1"),
])
def test_an_interface_unit_the_stream_does_not_rebuild_raises(rows, mats, names,
                                                              match):
    from apeGmsh.assembly import AssemblyError
    from apeGmsh.assembly._rehydrate import _interface_rows

    with pytest.raises(AssemblyError, match=match):
        _interface_rows("p", _iface_stub(rows=rows, mats=mats), names)


def test_the_matched_interface_unit_is_skipped():
    from apeGmsh.assembly._rehydrate import _interface_rows

    model = _iface_stub(rows=((7, _ROW),), mats=((1, "ENT"), (2, "ElasticPP")))
    assert _interface_rows("p", model, {}) == ({7}, {1, 2})


# ---------------------------------------------------------------------------
# G5 (#1587): a dim-0 physical group
# ---------------------------------------------------------------------------

def test_a_point_group_source_deck_equals_the_source_deck(files, tmp_path):
    want, got = _parity(files["corner"], declare_corner, 3, 3, tmp_path)
    assert want.count("element stdBrick") == 8
    assert got == want


def test_a_point_group_travels_as_the_instance_group(files):
    ops = _bridge(("a", files["corner"], {}),
                  ("b", files["corner"], {"translate": (20.0, 0.0, 0.0)}),
                  ndm=3, ndf=3)
    nodes = ops.fem.nodes
    assert {"a.Corner", "b.Corner"} <= set(nodes.physical.names(dim=0))
    for label, x in (("a", 10.0), ("b", 30.0)):
        sel = nodes.select(pg=f"{label}.Corner")
        assert len(sel.ids) == 1
        np.testing.assert_allclose(np.asarray(sel.coords), [[x, 10.0, 10.0]])
