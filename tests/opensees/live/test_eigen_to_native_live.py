"""``EigenResult.to_native`` on a real solve (#1972, brief #1961 §6).

The stock proof runs in CI's ``live-stock`` lane (pip ``openseespy``); a
local run beside the Ladruno fork checks the fork half of invariant 6.
The backend is the module the bridge resolves (``live.get_ops``), never
``openseespy`` imported by name.

A 2-D cantilever at ``ndf=3`` with lumped masses, ``ops.eigen(3)``:

- the file's eigenvalues are the solve's, bit for bit, and
  ``frequency_hz`` / ``period_s`` follow from them;
- every written shape equals ``ops.nodeEigenvector`` (atol 0);
  ``displacement_x/y`` only (``ndm=2``, envelope ``ndf=3`` < 6);
- the root attrs name the module that solved;
- ``path=None`` writes under ``tempfile.gettempdir()`` and the file
  opens alone; ``results()`` opens the same modes;
- a later ``modal_properties`` makes the result stale.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.results import Results
from apeGmsh.sections._mc import backend_available

pytestmark = pytest.mark.live

L_BEAM = 3.0
N_NODES = 5
E = 2.0e11
A_BEAM = 1.0e-2
I_BEAM = 8.0e-6
M_NODE = 50.0


@pytest.fixture(autouse=True)
def _backend() -> None:
    if not backend_available():
        pytest.skip("no OpenSees backend importable (fork or openseespy)")


@pytest.fixture(scope="module")
def cantilever() -> Any:
    with apeGmsh(model_name="eigen_to_native_live", verbose=False) as g:
        geo = g.model.geometry
        base = geo.add_point(0.0, 0.0, 0.0)
        tip = geo.add_point(L_BEAM, 0.0, 0.0)
        line = geo.add_line(base, tip)
        gmsh.model.mesh.setTransfiniteCurve(line, N_NODES)
        g.physical.add(1, [line], name="Beam")
        g.labels.add(0, [base], "base")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _bridge(fem: Any) -> apeSees:
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Beam", transf=t, A=A_BEAM, E=E,
                                  Iz=I_BEAM)
    ops.fix(label="base", dofs=(1, 1, 1))
    ops.mass(pg="Beam", values=(M_NODE, M_NODE, 1.0e-3))
    return ops


def test_modes_file_matches_the_solve(cantilever: Any) -> None:
    ops = _bridge(cantilever)
    r = ops.eigen(3)
    assert np.all(r.eigenvalues > 0)

    p = r.to_native()

    assert Path(tempfile.gettempdir()).resolve() in p.resolve().parents
    node_ids = [int(n) for n in cantilever.nodes.ids]
    shapes = {(n, k): r.mode_shape(n, k)
              for n in node_ids for k in (1, 2, 3)}
    with Results.from_native(p, model=OpenSeesModel.from_h5(p)) as res:
        modes = sorted(res.modes, key=lambda m: m.mode_index)
        assert [m.mode_index for m in modes] == [1, 2, 3]
        for m, lam in zip(modes, r.eigenvalues):
            assert m.eigenvalue == float(lam)                # bit for bit
            omega = math.sqrt(float(lam))
            assert m.frequency_hz == pytest.approx(
                omega / (2.0 * math.pi), rel=1e-15)
            assert m.period_s == pytest.approx(
                2.0 * math.pi / omega, rel=1e-15)
            comps = set(m.nodes.available_components())
            assert comps == {"displacement_x", "displacement_y"}
            for d, comp in enumerate(("displacement_x", "displacement_y")):
                got = m.nodes.get(component=comp)
                by_node = dict(zip((int(i) for i in got.node_ids),
                                   got.values[0]))
                want = {n: float(shapes[(n, m.mode_index)][d])
                        for n in node_ids}
                assert by_node == want                       # atol 0

    info = live.get_backend_info()
    with h5py.File(p, "r") as f:
        assert f.attrs["source_type"] == "eigen"
        assert f.attrs["opensees_backend"] == info.kind
        if info.kind == "fork":
            assert f.attrs["opensees_build"] == info.build
        else:
            assert "opensees_build" not in f.attrs


def test_results_opens_the_modes(cantilever: Any) -> None:
    r = _bridge(cantilever).eigen(3)
    with r.results() as res:
        eig = sorted(res.eigen_modes, key=lambda m: m.mode_index)
    assert [m.eigenvalue for m in eig] == [float(v) for v in r.eigenvalues]


def test_modal_properties_after_eigen_is_refused(
    cantilever: Any, tmp_path: Path,
) -> None:
    ops = _bridge(cantilever)
    r = ops.eigen(3)
    ops.modal_properties(3)
    with pytest.raises(RuntimeError, match="rebuilt since this eigen"):
        r.to_native(tmp_path / "modes.h5")
