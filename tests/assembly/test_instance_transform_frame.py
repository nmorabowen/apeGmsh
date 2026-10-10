"""A rotated instance turns its archived ``geomTransf`` vecxz (#1597).

The instance frame rule (ADR 0117 INV-6 note): an instance is authored in
its source frame, so a direction turns by ``R v`` and never translates. The
merge turned the instance's node coordinates, but rehydration re-registered
each archived ``geomTransf`` with its source ``vecxz``, so a rotated beam got
the wrong local axes with no error.

Oracles, each independent of the code under test:

* closed form: ``R`` is 120 degrees about ``(1, 1, 1)``, the cyclic map
  ``x -> y -> z -> x``, so it changes every source vecxz used here. The deck
  must write ``geomTransf <type> <tag> R*vecxz`` for Linear, PDelta and
  Corotational.
* lock: every field of the archived ``TransformRecord`` is classified in
  ``_TRANSFORM_FIELDS``, and ``vec`` is the one direction.
* live (stock openseespy, in a subprocess): three cantilever columns, one per
  transform type, with ``Iy != Iz``. In the source a load along global X
  bends each column about its local y, so the tip moves ``P L^3 / (3 E Iy)``.
  Rotated, the load ``R X = Y`` must give the same deflection along Y. With
  the source vecxz the local axes are wrong (here vecxz lies along the
  rotated column axis, which OpenSees refuses).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh

E = 200_000.0
IY = 2.0e3
IZ = 8.0e3
L = 3.0
P = 1.0e-3
ROT = ((1.0, 1.0, 1.0), 2.0 * math.pi / 3.0)
#: Transform type -> (column PG, tip PG, source vecxz). The columns stand
#: along +z; local z is the vecxz direction (global X for each).
COLUMNS = {
    "Linear": ("ColLin", "TipLin", (1.0, 0.0, 0.0)),
    "PDelta": ("ColPD", "TipPD", (1.0, 0.0, 0.0)),
    "Corotational": ("ColCor", "TipCor", (1.0, 0.0, 0.0)),
}
#: A transform whose source vecxz is not along X, so the cyclic R moves
#: a second axis too (Z -> X).
BEAM_VECXZ = (0.0, 0.0, 1.0)


def _r(v) -> np.ndarray:
    """The closed-form cyclic rotation: (x, y, z) -> (z, x, y)."""
    x, y, z = v
    return np.array([z, x, y], dtype=float)


def _columns_fem(workdir: Path):
    """Three cantilever columns along +z at x = 0, 5, 10, plus a beam."""
    with apeGmsh(model_name="cols", verbose=False) as g:
        geo = g.model.geometry
        base, tips = [], []
        for k, (col, tip, _) in enumerate(COLUMNS.values()):
            p0 = geo.add_point(5.0 * k, 0.0, 0.0)
            p1 = geo.add_point(5.0 * k, 0.0, L)
            ln = geo.add_line(p0, p1)
            base.append(p0)
            tips.append((tip, p1, col, ln))
        b0 = geo.add_point(0.0, 10.0, 0.0)
        b1 = geo.add_point(4.0, 10.0, 0.0)
        beam = geo.add_line(b0, b1)
        g.model.sync()
        for tip, p1, col, ln in tips:
            g.physical.add(1, [ln], name=col)
            g.physical.add(0, [p1], name=tip)
        g.physical.add(0, base + [b0], name="Base")
        g.physical.add(1, [beam], name="Beam")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(1)
        return g.mesh.queries.get_fem_data(dim=1)


def _archive(workdir: Path) -> Path:
    from apeGmsh.opensees import apeSees

    ops = apeSees(_columns_fem(workdir))
    ops.model(ndm=3, ndf=6)
    props = dict(A=100.0, E=E, Iz=IZ, Iy=IY, G=8.0e4, J=5.0e3)
    for token, (col, _, vec) in COLUMNS.items():
        t = getattr(ops.geomTransf, token)(vecxz=vec, name=f"t{token}")
        ops.element.elasticBeamColumn(pg=col, transf=t, **props)
    tb = ops.geomTransf.Linear(vecxz=BEAM_VECXZ, name="tBeam")
    ops.element.elasticBeamColumn(pg="Beam", transf=tb, **props)
    path = workdir / "cols.h5"
    ops.h5(str(path))
    return path


def _bridge(workdir: Path, *, rotate=ROT):
    from apeGmsh.assembly import Assembly

    return (Assembly("turned")
            .instance("c", _archive(workdir), rotate=rotate,
                      translate=(100.0, -50.0, 7.0))
            .bridge(ndm=3, ndf=6))


def _transf_lines(ops, path: Path) -> dict[str, list[tuple[float, ...]]]:
    """``{type: sorted vecxz rows}`` from the deck, rounded to 1e-12."""
    ops.tcl(str(path), flat=True)
    out: dict[str, list[tuple[float, ...]]] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        tok = ln.split()
        if tok[:1] == ["geomTransf"]:
            out.setdefault(tok[1], []).append(
                tuple(round(float(v), 12) + 0.0 for v in tok[3:6]))
    return {k: sorted(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Deck: closed form for every transform type
# ---------------------------------------------------------------------------

def test_rotated_instance_writes_turned_vecxz(tmp_path):
    got = _transf_lines(_bridge(tmp_path), tmp_path / "turned.tcl")
    assert set(got) == set(COLUMNS)
    for token, (_, _, vec) in COLUMNS.items():
        vecs = [vec] + ([BEAM_VECXZ] if token == "Linear" else [])
        want = sorted(tuple(float(x) for x in _r(v)) for v in vecs)
        assert got[token] == want, token


def test_unrotated_instance_keeps_source_vecxz(tmp_path):
    got = _transf_lines(_bridge(tmp_path, rotate=None), tmp_path / "plain.tcl")
    assert got["Linear"] == sorted([(1.0, 0.0, 0.0), BEAM_VECXZ])
    assert got["PDelta"] == [(1.0, 0.0, 0.0)]
    assert got["Corotational"] == [(1.0, 0.0, 0.0)]


# ---------------------------------------------------------------------------
# Lock: every archived transform field is classified
# ---------------------------------------------------------------------------

def test_every_transform_field_is_classified():
    from apeGmsh.assembly._rehydrate import _TRANSFORM_FIELDS, _TRANSFORMS
    from apeGmsh.opensees._internal.typed_records import TransformRecord

    names = {f.name for f in dataclasses.fields(TransformRecord)}
    assert names == set(_TRANSFORM_FIELDS), names ^ set(_TRANSFORM_FIELDS)
    assert {k for k, v in _TRANSFORM_FIELDS.items() if v == "direction"} == {"vec"}
    assert set(_TRANSFORMS) == set(COLUMNS)


# ---------------------------------------------------------------------------
# Live, on whatever backend CI installs (stock openseespy in live-stock)
# ---------------------------------------------------------------------------

def _solve(workdir: Path) -> dict:
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    ops = _bridge(workdir)
    fem = ops.fem
    ops.fix(nodes=sorted(int(i) for i in fem.nodes.select(pg="c.Base").ids),
            dofs=(1, 1, 1, 1, 1, 1))
    load = tuple(_r((P, 0.0, 0.0)))
    tips = {token: int(fem.nodes.select(pg=f"c.{tip}").ids[0])
            for token, (_, tip, _) in COLUMNS.items()}
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        for tip in tips.values():
            pat.load(node=tip, forces=load + (0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=20)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    live = emitter.ops
    return {token: [live.nodeDisp(t, d) for d in (1, 2, 3)]
            for token, t in tips.items()}


def _main() -> None:
    """``python -c`` entry: argv[1] is the work directory."""
    print("RESULT " + json.dumps(_solve(Path(sys.argv[1]))))


@pytest.mark.live
def test_rotated_cantilevers_bend_about_their_turned_axes(tmp_path):
    from apeGmsh.opensees.emitter.live import _get_ops

    try:
        _get_ops()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), env.get("PYTHONPATH", "")])
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    proc = subprocess.run(
        [sys.executable, "-W", "ignore::UserWarning", "-c",
         "from tests.assembly.test_instance_transform_frame import _main; _main()",
         str(tmp_path)],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"subprocess failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    res = json.loads(lines[-1][len("RESULT "):])
    u = P * L ** 3 / (3.0 * E * IY)       # bending about local y (uses Iy)
    want = _r((u, 0.0, 0.0))
    for token, disp in res.items():
        # Corotational adds an axial shortening of order u**2 / L.
        assert disp == pytest.approx(list(want), rel=1e-6, abs=1e-6 * u), token
