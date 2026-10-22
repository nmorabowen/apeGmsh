"""Joint offsets survive the archive: H5 replay and a rotated instance.

* ``OpenSeesModel.from_h5(...).build("tcl")`` writes every ``geomTransf``
  line of the source deck, ``-jntOffset`` included (the fan-out of an
  ``InsertionPoint`` rule replays as its resolved lines).
* An ``Assembly`` instance turned by ``R`` (120 deg about (1, 1, 1), the
  cyclic map x -> y -> z -> x) writes ``R d`` for each offset half: an
  offset is a direction (node -> element end), never translated.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from apeGmsh import apeGmsh
from apeGmsh.opensees.joint_offset import InsertionPoint

ROT = ((1.0, 1.0, 1.0), 2.0 * math.pi / 3.0)
COL_JNT = ((0.05, -0.04, 0.0), (0.05, -0.04, -0.3))
PROPS = dict(A=0.25, E=30e9, Iz=5e-3, Iy=3e-3, G=12e9, J=6e-3)


def _frame_fem():
    """One column (0,0,0)-(0,0,3) and two beams at z = 3 along +X, +Y."""
    with apeGmsh(model_name="jnt", verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(0.0, 0.0, 3.0)
        p2 = geo.add_point(5.0, 0.0, 3.0)
        p3 = geo.add_point(0.0, 4.0, 3.0)
        col = geo.add_line(p0, p1)
        bx = geo.add_line(p1, p2)
        by = geo.add_line(p1, p3)
        g.model.sync()
        g.physical.add(1, [col], name="Col")
        g.physical.add(1, [bx, by], name="Beams")
        g.physical.add(0, [p0], name="Base")
        g.mesh.sizing.set_global_size(100.0)
        g.mesh.generation.generate(1)
        return g.mesh.queries.get_fem_data(dim=1)


def _archive(tmp_path: Path) -> tuple[Path, list[str]]:
    from apeGmsh.opensees import apeSees

    ops = apeSees(_frame_fem())
    ops.model(ndm=3, ndf=6)
    tc = ops.geomTransf.PDelta(vecxz=(1.0, 0.0, 0.0), jnt_offset=COL_JNT, name="tc")
    tb = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), name="tb", jnt_offset=InsertionPoint(
        cardinal_point=9, depth=0.6, width=0.3, axis_2="z", end_offset_i=0.25))
    ops.element.elasticBeamColumn(pg="Col", transf=tc, **PROPS)
    ops.element.elasticBeamColumn(pg="Beams", transf=tb, **PROPS)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    path = tmp_path / "jnt.h5"
    ops.h5(str(path))
    deck = tmp_path / "src.tcl"
    ops.tcl(str(deck))
    return path, _transf_lines(deck.read_text(encoding="utf-8"))


def _transf_lines(text: str) -> list[str]:
    return sorted(ln.strip() for ln in text.splitlines()
                  if ln.strip().startswith("geomTransf"))


def _offsets(lines: list[str]) -> list[tuple[float, ...]]:
    out = []
    for ln in lines:
        tok = ln.split()
        k = tok.index("-jntOffset")
        out.append(tuple(round(float(v), 12) + 0.0 for v in tok[k + 1:k + 7]))
    return sorted(out)


def test_h5_replay_writes_the_offset_lines(tmp_path: Path) -> None:
    from apeGmsh.opensees import OpenSeesModel

    path, src = _archive(tmp_path)
    assert len(src) == 3 and all("-jntOffset" in ln for ln in src)
    replayed = _transf_lines(OpenSeesModel.from_h5(str(path)).build("tcl"))
    assert replayed == src


def test_model_data_rewrite_keeps_the_offsets(tmp_path: Path) -> None:
    import warnings

    from apeGmsh.opensees import ModelData
    from apeGmsh.opensees.emitter import h5_reader

    path, _src = _archive(tmp_path)
    with h5_reader.open(str(path)) as m:
        want = sorted((r.tag, r.jnt_offset) for r in m.transforms())
    out = tmp_path / "rewritten.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ModelData.from_h5(str(path)).write(str(out))
    with h5_reader.open(str(out)) as m:
        got = sorted((r.tag, r.jnt_offset) for r in m.transforms())
    assert got == want and all(j for _t, j in got)


def test_rotated_instance_turns_each_offset_half(tmp_path: Path) -> None:
    from apeGmsh.assembly import Assembly

    path, src = _archive(tmp_path)
    ops = (Assembly("turned")
           .instance("c", path, rotate=ROT, translate=(100.0, -50.0, 7.0))
           .bridge(ndm=3, ndf=6))
    out = tmp_path / "turned.tcl"
    ops.tcl(str(out), flat=True)
    got = _offsets(_transf_lines(out.read_text(encoding="utf-8")))

    def r(v: np.ndarray) -> np.ndarray:   # (x, y, z) -> (z, x, y)
        return np.array([v[2], v[0], v[1]])

    want = []
    for d in _offsets(src):
        a, b = np.asarray(d[:3]), np.asarray(d[3:])
        want.append(tuple(round(float(c), 12) + 0.0
                          for c in (*r(a), *r(b))))
    assert got == sorted(want)
