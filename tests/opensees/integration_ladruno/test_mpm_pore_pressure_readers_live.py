"""The fork's PORE_PRESSURE output, read back (#3121; fork WP-232l, OpenSees#1116, P12).

Fork release ``ladruno-v20261123b`` (``3eb195bbc``) adds one reserved particle
field to Contract 2 (232a §2.6): ``PORE_PRESSURE``, ``COMPONENTS`` ``p``,
``DIMENSION`` ``F/L^2``, ``DATA_TYPE`` 0, the excess pore pressure of a two-phase
particle, compression positive. This test runs the fork, in a fresh process, on a
mode-F (``mpm mode explicit``, route S) two-phase column under gravity, drained at
the top (``mpm pfix ymax 0``), written by ``mpm output -fields POSITION STRESS
PORE_PRESSURE``. Then:

* the fork's own checker at the pin (``tests/fixtures/mpm/fork_3eb195bbc/``)
  passes the file (``validate`` returns ``[]``);
* the Python reader (``Results.from_ladruno``) reads ``PORE_PRESSURE/p`` as h5py
  reads it: finite, and larger at the base of the column than at its top;
* the file's ``PORE_PRESSURE`` group has the encoding of the committed fixture
  ``tests/fixtures/mpm/particles_pore_pressure_v1m2.ladruno`` (the same
  attributes, each of the same HDF5 type and value, and a float64 one-column
  ``DATA``). The viewer's ``ladrunoParticles.test.ts`` reads that fixture, so this
  is what ties its offline test to what the fork writes: the live-fork lane
  installs no Node toolchain to run the viewer reader here.

The bridge offers no two-phase member yet, so the deck is written here, not by
``apeSees`` (the pattern of ``test_mpm_bspline_readers_live.py``).
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import apeGmsh
from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno_particles import PARTICLE_FIELDS

pytestmark = pytest.mark.ladruno_fork

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mpm"
_PIN = _FIXTURES / "fork_3eb195bbc"
_FIXTURE = _FIXTURES / "particles_pore_pressure_v1m2.ladruno"
_FIXTURE_GROUP = "MODEL_STAGE[1]/RESULTS/ON_PARTICLES/PORE_PRESSURE"

# A 1 m x 4 m column on a flush 2 x 8 grid of 0.5 m cells (the shape of the fork's T4p
# hydrostatic column, ``ladruno_mpm_twophase_cases_harness.run_t4``), run in mode F.
_DECK = """\
from apeGmsh.opensees.emitter.live import get_ops
ops = get_ops()


def first(v):
    return float(v[0] if isinstance(v, (list, tuple)) else v)


ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)
ops.nDMaterial("ElasticIsotropic", 1, 1.0e4, 0.2, 2.0)
ops.mpm("grid", 232, "-origin", 0.0, 0.0, "-h", 0.5, "-cells", 2, 8, "-basis", "gimp")
ops.mpm("mode", "explicit")
ops.mpm("body", 1, "box", 0.0, 0.0, 1.0, 4.0, "-ppc", 2, "-mat", 1, "-rho", 2.0)
ops.mpm("twoPhase", 1, "-Kf", 4444.0, "-poro", 0.4, "-rhoF", 1.0, "-perm", 1.0e-4, 1.0e-4,
        "-stab", "auto", 0.25)
ops.mpm("gravity", 0.0, -9.81)
ops.mpm("fix", "xmin", "x")
ops.mpm("fix", "xmax", "x")
ops.mpm("fix", "ymin", "y")
ops.mpm("pfix", "ymax", 0.0)
ops.mpm("units", "m", "kN", "s")
dt = 0.5 * first(ops.mpm("criticalTimeStep"))
ops.mpm("output", {out!r}, "-every", 50, "-fields", "POSITION", "STRESS", "PORE_PRESSURE")
ops.mpm("analyze", 200, dt)
ops.mpm("output", "-close")
print("DECK-DONE", dt)
"""


def _env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])])
    return env


@pytest.fixture
def pin_format():
    """The fork's ``ladruno_format`` at the pin, with ``sys.path`` / ``sys.modules`` restored."""
    names = ("ladruno_format", "ladruno_basis")
    saved_path = list(sys.path)
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    sys.path.insert(0, str(_PIN))
    try:
        yield importlib.import_module("ladruno_format")
    finally:
        sys.path[:] = saved_path
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def _encoding(group: "h5py.Group") -> dict:
    """Each attribute's HDF5 type, shape and value, and DATA's dtype and width."""
    out = {}
    for k in sorted(group.attrs):
        aid = group.attrs.get_id(k)
        t = aid.get_type()
        if isinstance(t, h5py.h5t.TypeStringID):
            kind = ("str", t.get_size(), t.is_variable_str(), t.get_strpad(), t.get_cset())
        else:
            kind = (type(t).__name__, t.get_size())
        out[k] = (kind, aid.get_space().get_simple_extent_dims(),
                  np.asarray(group.attrs[k]).tolist())
    out["DATA"] = (group["DATA"].dtype.str, group["DATA"].shape[1])
    return out


def test_the_forks_pore_pressure_reads_and_matches_the_fixture(tmp_path, pin_format) -> None:
    assert PARTICLE_FIELDS["PORE_PRESSURE"] == ("p", "p")
    out = tmp_path / "column.ladruno"
    deck = tmp_path / "column_deck.py"
    deck.write_text(_DECK.format(out=str(out)), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(deck)], cwd=tmp_path, env=_env(),
                          capture_output=True, text=True, check=False, timeout=300)
    assert proc.returncode == 0 and "DECK-DONE" in proc.stdout, (
        proc.stdout[-3000:] + proc.stderr[-3000:])

    # The fork's own checker at the pin.
    assert pin_format.validate(str(out)) == []

    with h5py.File(out, "r") as f:
        stages = [k for k in f if k.startswith("MODEL_STAGE[")
                  and "MODEL/PARTICLES" in f[k]]
        assert len(stages) == 1, list(f)
        (stage,) = stages
        on = f[f"{stage}/RESULTS/ON_PARTICLES"]
        assert {"POSITION", "STRESS", "PORE_PRESSURE"} <= set(on)
        raw_p = on["PORE_PRESSURE/DATA"][...]
        raw_off = on["OFFSETS"][...]
        live = _encoding(on["PORE_PRESSURE"])

    # The Python reader.
    with Results.from_ladruno(out) as r:
        assert "PORE_PRESSURE/p" in r.particles.components()
        s = r.particles.get(component="PORE_PRESSURE/p")
        assert s.offsets.size >= 3, "at least two outputs"
        np.testing.assert_array_equal(s.offsets, raw_off[:s.offsets.size])
        values = np.asarray(s.values).reshape(-1)
        np.testing.assert_array_equal(values, raw_p[:int(s.offsets[-1])].reshape(-1))
        assert np.all(np.isfinite(values))
        pos = r.particles.get(component="POSITION", time=-1)
        last = r.particles.get(component="PORE_PRESSURE/p", time=-1)
        y = np.asarray(pos.values)[:, 1]
        p = np.asarray(last.values).reshape(-1)
        base, top = p[y <= y.min() + 0.25], p[y >= y.max() - 0.25]
        assert base.size and top.size
        assert base.mean() > top.mean(), (base, top)
        assert base.mean() > 0.0, "compression positive under the column's weight"

    # The committed fixture the viewer's test reads holds the field as the fork writes it.
    with h5py.File(_FIXTURE, "r") as f:
        assert _encoding(f[_FIXTURE_GROUP]) == live
