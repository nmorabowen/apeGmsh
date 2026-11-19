"""The fork's B-spline grid output, read back (#3013; fork WP-232i, OpenSees#1099).

Fork release ``ladruno-v20261118`` amends 232a §2.4: in Contract 2 the grid's
``BASIS`` is one of ``gimp | linear | bspline2 | bspline3``, with no minor bump.
For each B-spline basis this test runs the fork, in a fresh process, on the
deck of the fork's own row G7 (``tests/test_ladruno_mpm_bspline_basis.py``,
``test_g7_ladruno_output_round_trip``): a mode-A body on a ``-basis`` grid,
written by ``mpm output``. Then:

* the Python reader (``Results.from_ladruno``) opens the file and reports the
  basis, and reads every output's positions as h5py reads them;
* the file's ``GRID/BASIS`` has the encoding of the committed fixture
  ``tests/fixtures/mpm/particles_<basis>_v1m2.ladruno`` (a 1-element,
  fixed-length, null-terminated ASCII string), and the grid has the same four
  attributes. The viewer's ``ladrunoParticles.test.ts`` reads that fixture, so
  this is what ties its offline test to what the fork writes: the live-fork
  lane installs no Node toolchain to run the viewer reader here.

The bridge does not emit ``-basis bspline*`` yet (Contract 1 v2 is out of
scope, #3013), so the deck is written here, not by ``apeSees``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import apeGmsh
from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno_particles import C2_BASES

pytestmark = pytest.mark.ladruno_fork

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mpm"

_DECK = """\
from apeGmsh.opensees.emitter.live import get_ops
ops = get_ops()
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)
ops.nDMaterial("ElasticIsotropic", 1, 1.0e3, 0.3)
ops.mpm("grid", 232, "-origin", 0.0, 0.0, "-h", 0.25, "-cells", {n}, {n}, "-basis", "{basis}")
ops.mpm("body", 1, "box", 0.25, 0.25, 0.75, 0.75, "-ppc", 2, "-mat", 1, "-rho", 1.0)
ops.mpm("fix", "ymin", "y")
ops.mpm("velocity", 1, "uniform", 0.1, -0.05)
ops.mpm("units", "m", "N", "s")
ops.mpm("output", {out!r}, "-every", 2, "-fields", "POSITION", "STRESS")
ops.mpm("analyze", 6, 1.0e-3)
ops.mpm("output", "-close")
print("DECK-DONE")
"""


def _env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])])
    return env


def _basis_type(path: Path, grid: str) -> "tuple[int, bool, int, int, tuple]":
    with h5py.File(path, "r") as f:
        aid = f[grid].attrs.get_id("BASIS")
        t = aid.get_type()
        return (t.get_size(), t.is_variable_str(), t.get_strpad(), t.get_cset(),
                aid.get_space().get_simple_extent_dims())


@pytest.mark.parametrize("basis", ["bspline2", "bspline3"])
def test_the_forks_bspline_ladruno_opens_and_reports_its_basis(tmp_path, basis) -> None:
    assert basis in C2_BASES
    out = tmp_path / f"{basis}.ladruno"
    deck = tmp_path / f"{basis}_deck.py"
    # G7's 4 x 4 grid for both: a B-spline grid needs >= 2 cells per axis, and a
    # particle is admissible iff its centre is in the real box (interface B3).
    n = 4
    deck.write_text(_DECK.format(n=n, basis=basis, out=str(out)), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(deck)], cwd=tmp_path, env=_env(),
                          capture_output=True, text=True, check=False, timeout=300)
    assert proc.returncode == 0 and "DECK-DONE" in proc.stdout, (
        proc.stdout[-3000:] + proc.stderr[-3000:])

    with h5py.File(out, "r") as f:
        stages = [k for k in f if k.startswith("MODEL_STAGE[")
                  and "MODEL/PARTICLES" in f[k]]
        assert len(stages) == 1, list(f)
        (stage,) = stages
        grid = f[f"{stage}/MODEL/PARTICLES/GRID"]
        assert grid.attrs["BASIS"].tolist() == [basis.encode()]
        on = f[f"{stage}/RESULTS/ON_PARTICLES"]
        raw_pos = on["POSITION/DATA"][...]
        raw_off = on["OFFSETS"][...]
        grid_attrs = sorted(grid.attrs)

    # The Python reader.
    with Results.from_ladruno(out) as r:
        g = r.particles.grid()
        assert g.basis == basis
        np.testing.assert_array_equal(g.num_cells, [n, n])
        np.testing.assert_array_equal(g.cell_size, [0.25, 0.25])
        s = r.particles.get(component="POSITION")
        assert s.offsets.size >= 2
        np.testing.assert_array_equal(s.offsets, raw_off[:s.offsets.size])
        np.testing.assert_array_equal(s.values, raw_pos[:int(s.offsets[-1])])

    # The committed fixture the viewer's test reads holds BASIS as the fork writes it.
    fixture = _FIXTURES / f"particles_{basis}_v1m2.ladruno"
    fixture_grid = "MODEL_STAGE[1]/MODEL/PARTICLES/GRID"
    assert _basis_type(out, f"{stage}/MODEL/PARTICLES/GRID") == \
        _basis_type(fixture, fixture_grid)
    with h5py.File(fixture, "r") as f:
        assert sorted(f[fixture_grid].attrs) == grid_attrs
