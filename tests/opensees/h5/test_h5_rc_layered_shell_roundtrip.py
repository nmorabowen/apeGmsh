"""H5 round-trip of a layered RC shell: ``PlateRebar`` / ``PlateFiber`` layers.

``PlateRebar`` is an nD material that references a *uniaxial* tag, and
``PlateFiber`` an nD that references another nD. The archive stores
materials as generic ``(type_token, tag, params)`` rows and
``OpenSeesModel.build`` replays every uniaxial before any nD, then the nDs
in their archived (dependency) order. So the replayed deck must carry the
same material and section lines, each after every tag it references.
"""
from __future__ import annotations

from pathlib import Path

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.section.plate import RebarMesh

from tests.opensees.h5.test_h5_stages_reader import build_two_quad_fem

_KEYS = ("uniaxialMaterial ", "nDMaterial ", "section ")


def _bridge() -> apeSees:
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel01(fy=420e6, E=200e9, b=0.01)
    conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
    conc_layer = ops.nDMaterial.PlateFiber(material=conc)
    t = 113.1e-6 / 0.15
    sec = ops.section.RCLayeredShell(h=0.2, concrete=conc_layer, n_concrete=8, meshes=[
        RebarMesh(material=steel, angle=0.0, area_per_width=t, cover=0.03, face="bottom"),
        RebarMesh(material=steel, angle=90.0, area_per_width=t, cover=0.042, face="bottom"),
        RebarMesh(material=steel, angle=0.0, area_per_width=t, cover=0.03, face="top"),
        RebarMesh(material=steel, angle=90.0, area_per_width=t, cover=0.042, face="top"),
    ])
    for pg in ("Rock", "Fill"):
        ops.element.ASDShellQ4(pg=pg, section=sec, local_cs=(1.0, 0.0, 0.0))
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    return ops


def _decl_lines(deck: str) -> list[str]:
    return [ln.strip() for ln in deck.splitlines() if ln.strip().startswith(_KEYS)]


def _num(tok: str) -> "float | str":
    try:
        return float(tok)
    except ValueError:
        return tok


def _normalised(lines: list[str]) -> list[tuple["float | str", ...]]:
    """Token tuples with numbers compared by value: the replay writes a
    whole float as ``30000000000``, the bridge as ``30000000000.0``."""
    return sorted(
        (tuple(_num(t) for t in ln.split()) for ln in lines), key=repr,
    )


def _assert_defined_before_use(lines: list[str]) -> None:
    uni: set[str] = set()
    nd: set[str] = set()
    for ln in lines:
        tok = ln.split()
        kind, typ, tag = tok[0], tok[1], tok[2]
        if kind == "uniaxialMaterial":
            uni.add(tag)
        elif kind == "nDMaterial":
            if typ == "PlateRebar":
                assert tok[3] in uni, ln
            elif typ == "PlateFiber":
                assert tok[3] in nd, ln
            nd.add(tag)
        else:  # section LayeredShell tag n m1 t1 ...
            assert all(m in nd for m in tok[4::2]), ln


def test_rc_layered_shell_round_trips_through_h5(tmp_path: Path) -> None:
    tcl = tmp_path / "bridge.tcl"
    _bridge().tcl(str(tcl), progress=False)
    expected = _decl_lines(tcl.read_text(encoding="utf-8"))
    assert sum("PlateRebar" in ln for ln in expected) == 2
    assert sum("PlateFiber" in ln for ln in expected) == 1
    assert sum(ln.startswith("section LayeredShell") for ln in expected) == 1

    h5 = tmp_path / "model.h5"
    _bridge().h5(str(h5))
    replayed = _decl_lines(OpenSeesModel.from_h5(str(h5)).build("tcl") or "")

    assert _normalised(replayed) == _normalised(expected)
    _assert_defined_before_use(expected)
    _assert_defined_before_use(replayed)
