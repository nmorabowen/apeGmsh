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


#: ``(normal law, tangential law)`` per fixture: together every uniaxial the
#: ADR 0093 D1 table synthesises (``ENT``, ``ElasticPPGap``, ``ElasticPP``,
#: ``Elastic``).
LAWS = {
    "ent_epp": (dict(kind="ent", k_per_area=1.0e9),
                dict(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)),
    "gap_elastic": (dict(kind="epp_gap", k_per_area=1.0e9, tau_b_n=4.0e5,
                         gap=-1.0e-4),
                    dict(kind="elastic", k_per_area=1.0e8)),
    "elastic_epp": (dict(kind="elastic", k_per_area=1.0e9),
                    dict(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)),
}


def interface_fem(laws: str = "ent_epp"):
    """Two abutting unit squares with one interface ``RL`` across x=1."""
    from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw

    normal, tangential = LAWS[laws]

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
            normal=NormalLaw(**normal), tangential=TangentialLaw(**tangential),
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
        **{laws: _write(d / f"{laws}.h5", interface_fem(laws), declare_interface)
           for laws in LAWS},
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

@pytest.mark.parametrize("laws, normal, tangential", [
    ("ent_epp", "ENT", "ElasticPP"),
    ("gap_elastic", "ElasticPPGap", "Elastic"),
    ("elastic_epp", "Elastic", "ElasticPP"),
])
def test_an_interface_source_deck_equals_the_source_deck(files, laws, normal,
                                                         tangential, tmp_path):
    want, got = _parity(files[laws], declare_interface, 2, 2, tmp_path)
    assert want.count("element zeroLength") == 2
    for token in (normal, tangential):
        assert len(re.findall(rf"^uniaxialMaterial {token} [2-5] ", want,
                              re.M)) == 2, token      # one per pair, after "k"
    assert "uniaxialMaterial Elastic 1 30000000.0" in want   # user material first
    assert got.count("# inst.RL") == 2          # compose namespaces the record
    assert got.replace("# inst.RL", "# RL") == want


def test_two_interface_instances_each_carry_their_own_pairs(files, tmp_path):
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    ops = _bridge(("a", files["ent_epp"], {}),
                  ("b", files["ent_epp"], {"translate": (5.0, 0.0, 0.0)}),
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


# ---------------------------------------------------------------------------
# #1590 review F3: a 3-D surface master, a phantom (mixed-ndf) pair, and a
# user node-pair zeroLength on an interface pair
# ---------------------------------------------------------------------------

def _surface_at_z(volume: int, z: float) -> int:
    for _, tag in gmsh.model.getBoundary([(3, volume)], oriented=False):
        bb = gmsh.model.getBoundingBox(2, abs(tag))
        if abs(bb[2] - z) < 1e-6 and abs(bb[5] - z) < 1e-6:
            return abs(tag)
    raise AssertionError(f"no boundary surface of volume {volume} at z={z}")


def interface_3d_fem():
    """Two stacked unit cubes with one interface ``SF`` across z=1."""
    from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw

    normal, tangential = LAWS["ent_epp"]
    with apeGmsh(model_name="iface3d", verbose=False) as g:
        soil = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        footing = g.model.geometry.add_box(0, 0, 1, 1, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(3, soil), (3, footing)], n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [soil], name="soil")
        g.physical.add(3, [footing], name="footing")
        g.physical.add(2, [_surface_at_z(soil, 1.0)], name="face")
        g.physical.add(2, [_surface_at_z(footing, 1.0)], name="skin")
        g.constraints.interface(
            "face", "skin",
            normal=NormalLaw(**normal), tangential=TangentialLaw(**tangential),
            name="SF")
        return g.mesh.queries.get_fem_data(dim=3)


def declare_3d(ops) -> None:
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0, name="m")
    for pg in ("soil", "footing"):
        ops.element.stdBrick(pg=pg, material=mat)


def test_a_3d_interface_source_deck_equals_the_source_deck(tmp_path):
    """A 3-D surface master: one ``zeroLength`` per node pair of the one-quad
    face, each ``-mat n t t -dir 1 2 3``: two uniaxials per pair (F2)."""
    src = _write(tmp_path / "iface3d.h5", interface_3d_fem(), declare_3d)
    want, got = _parity(src, declare_3d, 3, 3, tmp_path)
    zl = [ln.split() for ln in want.splitlines()
          if ln.startswith("element zeroLength")]
    assert len(zl) == 4                       # transfinite n=2: 4 face nodes
    for tok in zl:
        mats = tok[tok.index("-mat") + 1:tok.index("-dir")]
        assert len(mats) == 3 and mats[1] == mats[2]
    assert len(re.findall(r"^uniaxialMaterial ENT ", want, re.M)) == 4
    assert len(re.findall(r"^uniaxialMaterial ElasticPP ", want, re.M)) == 4
    assert got.replace("# inst.SF", "# SF") == want


def phantom_fem():
    """``interface_fem`` with a 3-dof beam slave: each pair gets a phantom."""
    from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw

    normal, tangential = LAWS["ent_epp"]
    with apeGmsh(model_name="iface_ph", verbose=False) as g:
        left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, left), (2, right)], n=2)
        g.mesh.generation.generate(2)
        g.physical.add(2, [left], name="rock")
        g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
        g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
        g.constraints.interface(
            "face", "wire",
            normal=NormalLaw(**normal), tangential=TangentialLaw(**tangential),
            thickness=0.5, slave_ndf=3, name="RW")
        return g.mesh.queries.get_fem_data()


def declare_phantom(ops) -> None:
    ops.model(ndm=2, ndf=2)
    ops.element.elasticBeamColumn(
        pg="wire", transf=ops.geomTransf.Linear(name="lin"),
        A=0.02, E=200e9, Iz=1.0e-4)


def test_a_phantom_interface_carries_its_phantoms_per_instance(tmp_path):
    """Mixed ndf (ADR 0093 D4): the source deck has one ``-ndf 2`` phantom
    node per pair; one instance reproduces it, two carry two sets."""
    src = _write(tmp_path / "phantom.h5", phantom_fem(), declare_phantom)
    want, got = _parity(src, declare_phantom, 2, 2, tmp_path)
    phantoms = [ln for ln in want.splitlines()
                if ln.startswith("node ") and ln.endswith("-ndf 2")]
    assert len(phantoms) == 2
    assert want.count("element zeroLength") == 2
    assert got.replace("# inst.RW", "# RW") == want

    ops = _bridge(("a", src, {}), ("b", src, {"translate": (5.0, 0.0, 0.0)}),
                  ndm=2, ndf=2)
    recs = list(ops.fem.elements.interfaces)
    assert len(recs) == 4 and all(r.phantom_node is not None for r in recs)
    assert len({int(r.phantom_node) for r in recs}) == 4
    deck = _deck(ops, tmp_path / "two.tcl")
    assert deck.count("element zeroLength") == 4


def test_a_user_zero_length_on_an_interface_pair_is_refused(tmp_path):
    """A node-pair ``zeroLength`` the source declares on an interface pair
    makes two archived rows on that pair: the bridge refuses rather than
    guess which one the interface synthesises, and names the user
    element, not a stage claim (F1)."""
    from apeGmsh.assembly import AssemblyError
    from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir

    fem = interface_fem("ent_epp")
    rec = next(iter(fem.elements.interfaces))

    def declare(ops) -> None:
        declare_interface(ops)
        k = ops.uniaxialMaterial.ElasticMaterial(E=1.0e6, name="spring")
        ops.element.ZeroLength(
            nodes=(int(rec.master_node), int(rec.slave_node)),
            mat_dirs=(ZeroLengthMatDir(material=k, dof=1),))

    src = _write(tmp_path / "user_zl.h5", fem, declare)
    with pytest.raises(AssemblyError, match=(
            r"matches 2 archived node-pair zeroLength rows.*declare it on "
            r"the bridge")) as info:
        _bridge(("a", src, {}), ndm=2, ndf=2)
    assert "stage" not in str(info.value)
