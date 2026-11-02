"""Live oracles for ``/modes`` and ``/spectra`` on a frame (ADR 0130 slice VU6).

A 2-D frame, two bays by three storeys, lumped floor masses, gravity loads, run
on whichever engine the bridge binds (the Ladruno fork, or stock openseespy):

(a) ``/modes`` read back equals ``modalProperties -return``: ``lambda`` and
    ``period`` bit for bit, ``participation`` bit for bit, the ratios as the
    percentages over 100. The generalised mass equals ``φᵀMφ`` computed here
    from ``nodeEigenvector`` and ``nodeMass``, an oracle outside the table.
(b) ``/spectra`` read back holds ``spectrum_after_gravity``'s in-memory
    per-mode values and reference values bit for bit.
(c) CQC over the file's values, by the textbook formula with the file's
    ``lambda``, ``damping`` and ``sa_period``, equals the driver's own
    :meth:`SpectrumModalResult.combine`.

Engine-agnostic: the spectrum driver takes its static route on stock and the
fork's ``-increment`` route where the build has it; both must pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.spectrum import DesignSpectrum, NodeDisp
from apeGmsh.opensees.emitter.live import get_ops
from apeGmsh.results.user import UserFile, mode_table, spectrum_cases, write_user_file
from apeGmsh.results.user._spectra import same_bits

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)
from tests.results.user._user_helpers import cqc_reference

pytest.importorskip("openseespy.opensees")

E = 30e9
BAY, STORY = 6.0, 3.0
N_BAY, N_STORY = 2, 3
A_COL, I_COL = 0.25, 0.0052
A_BM, I_BM = 0.18, 0.0054
M_FLOOR = 20e3
G_LOAD = 20e3 * 9.81
TN = (0.0, 0.5, 1.0, 2.0)
SA = (3.0, 6.0, 4.0, 2.0)
SCALE = 0.75
XI = 0.05
N_MODES = 4


def _tag(s: int, c: int) -> int:
    return s * (N_BAY + 1) + c + 1


COORDS = [(c * BAY, s * STORY) for s in range(N_STORY + 1) for c in range(N_BAY + 1)]
ELEMS: list[tuple[int, int, str]] = []
for _s in range(N_STORY):
    ELEMS += [(_tag(_s, c), _tag(_s + 1, c), "col") for c in range(N_BAY + 1)]
    ELEMS += [(_tag(_s + 1, c), _tag(_s + 1, c + 1), "beam") for c in range(N_BAY)]
BASE = [_tag(0, c) for c in range(N_BAY + 1)]
FLOOR = [_tag(s, c) for s in range(1, N_STORY + 1) for c in range(N_BAY + 1)]


def _frame() -> apeSees:
    nodes = _NodesStub(
        ids=list(range(1, len(COORDS) + 1)),
        coords=[(x, y, 0.0) for x, y in COORDS],
        node_pgs={"Base": BASE, "Floor": FLOOR},
    )
    groups = {
        kind: _ElementGroupView(
            ids=tuple(i + 1 for i, e in enumerate(ELEMS) if e[2] == key),
            connectivity=tuple(e[:2] for e in ELEMS if e[2] == key))
        for kind, key in (("Cols", "col"), ("Beams", "beam"))
    }
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs=groups))
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Cols", transf=t, A=A_COL, E=E, Iz=I_COL)
    ops.element.elasticBeamColumn(pg="Beams", transf=t, A=A_BM, E=E, Iz=I_BM)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.mass(pg="Floor", values=(M_FLOOR, M_FLOOR, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Floor", forces=(0.0, -G_LOAD, 0.0))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()
    return ops


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "frame.h5"
    with h5py.File(path, "w") as f:
        f.create_group("meta/lineage").attrs["fem_hash"] = "frame-under-test"
    return path


@pytest.mark.live
def test_a_modes_table_equals_modal_properties(tmp_path: Path) -> None:
    result = _frame().modal_properties(N_MODES, solver="-fullGenLapack")
    props = result.properties
    path = write_user_file(tmp_path / "frame.user.h5", _model(tmp_path),
                           modes=mode_table(props, host="stage_0"))
    t = UserFile(path).modes()
    assert len(t) == N_MODES
    assert t.lam.tobytes() == np.asarray(props["eigenLambda"], dtype="f8").tobytes()
    assert t.period.tobytes() == np.asarray(props["eigenPeriod"], dtype="f8").tobytes()
    assert "MX" in t.components and "MY" in t.components
    for i, c in enumerate(t.components):
        assert t.participation[:, i].tobytes() == np.asarray(
            props[f"partiFactor{c}"], dtype="f8").tobytes(), c
        assert np.array_equal(t.mass_ratio[:, i], np.asarray(props[f"partiMassRatios{c}"]) / 100.0)
        assert np.array_equal(t.cumulative_ratio[:, i],
                              np.asarray(props[f"partiMassRatiosCumu{c}"]) / 100.0)

    ops: Any = get_ops()
    mgen = np.zeros(N_MODES)
    for k in range(N_MODES):
        for node in ops.getNodeTags():
            phi = np.asarray(result.mode_shape(int(node), k + 1), dtype="f8")
            m = np.asarray(ops.nodeMass(int(node)), dtype="f8")
            mgen[k] += float(np.sum(m[:phi.size] * phi * phi))
    np.testing.assert_allclose(t.generalized_mass, mgen, rtol=1e-9)


@pytest.mark.live
def test_b_c_spectra_hold_the_drivers_values_and_cqc_over_them_is_its_own(tmp_path: Path) -> None:
    roof_column = [_tag(s, 0) for s in range(1, N_STORY + 1)]
    result = _frame().spectrum_after_gravity(
        N_MODES, spectra=[DesignSpectrum(TN, SA, direction=1, scale=SCALE)],
        extractors={"disp": NodeDisp(nodes=roof_column, dofs=(1, 2))},
        solver="-fullGenLapack")
    cases = spectrum_cases(result, ["E_x"], damping=XI, nodal=["disp"])
    path = write_user_file(tmp_path / "frame.user.h5", _model(tmp_path), spectra=cases)
    index = UserFile(path).spectra()
    assert index.refused == ()
    (back,) = index.cases
    assert same_bits(back, cases[0])

    disp = back.result("DISPLACEMENT")
    assert disp.ids.tolist() == roof_column and disp.components == ("Ux", "Uy")
    assert disp.data.reshape(N_MODES, -1).tobytes() == result.per_mode["disp"][0].tobytes()
    shear = back.result("base_shear")
    assert shear.data.reshape(N_MODES, -1).tobytes() == result.per_mode["base_shear"][0].tobytes()
    ref = {q.label: q for q in back.reference}
    assert ref["DISPLACEMENT"].data.ravel().tobytes() == result.gravity_values["disp"].tobytes()
    assert back.lam.tobytes() == np.asarray(result.modal.eigenvalues, dtype="f8").tobytes()

    for name, q in (("disp", disp), ("base_shear", shear)):
        mine = cqc_reference(q.data.reshape(N_MODES, -1), back.lam, back.damping, back.sa_period)
        np.testing.assert_allclose(mine, result.combine(name, damping=XI)[0], rtol=1e-12)
