"""Live: a combination over a case VU6 wrote equals the driver's own (ADR 0130 slice VU2b).

A 2-D frame, two bays by three storeys, with lumped floor masses and gravity
loads, runs ``spectrum_after_gravity`` under an x and a y spectrum on whichever
engine the bridge binds. ``spectrum_cases`` and ``write_user_file`` write the two
cases, :class:`UserSpectra` reads them back, and the evaluator combines them.
The oracle is the driver's own combination of its in-memory values:

* CQC and SRSS of each case, for each roof-column node's ``Ux`` and ``Uy``
  and for the base shear, equal :meth:`SpectrumModalResult.combine`;
* 100/30 over the two CQC cases equals :meth:`SpectrumModalResult.directional`.

The provider is the one ``Results.combinations()`` builds, and the recorded
source is the one ``UserCombinations.evaluate`` builds, so the spectrum terms
take the production path, on the nodes and on the domain.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.spectrum import DesignSpectrum, NodeDisp
from apeGmsh.results.user import (
    Combination,
    Term,
    UserSpectra,
    spectrum_cases,
    write_user_file,
)
from apeGmsh.results.user._descriptor import resolve
from apeGmsh.results.user._evaluate import evaluate
from apeGmsh.results.user._states import _RecordedSource

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

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
XI = 0.05
N_MODES = 5
RTOL = 1e-12


def _tag(s: int, c: int) -> int:
    return s * (N_BAY + 1) + c + 1


COORDS = [(c * BAY, s * STORY) for s in range(N_STORY + 1) for c in range(N_BAY + 1)]
ELEMS: list[tuple[int, int, str]] = []
for _s in range(N_STORY):
    ELEMS += [(_tag(_s, c), _tag(_s + 1, c), "col") for c in range(N_BAY + 1)]
    ELEMS += [(_tag(_s + 1, c), _tag(_s + 1, c + 1), "beam") for c in range(N_BAY)]
BASE = [_tag(0, c) for c in range(N_BAY + 1)]
FLOOR = [_tag(s, c) for s in range(1, N_STORY + 1) for c in range(N_BAY + 1)]
ROOF_COLUMN = [_tag(s, 0) for s in range(1, N_STORY + 1)]


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


TABLE = resolve([
    Combination("Ex_cqc", "cqc", (Term("spectrum", "E_x"),)),
    Combination("Ey_cqc", "cqc", (Term("spectrum", "E_y"),)),
    Combination("Ex_srss", "srss", (Term("spectrum", "E_x"),)),
    Combination("dir", "directional",
                (Term("combination", "Ex_cqc"), Term("combination", "Ey_cqc")), fraction=0.3),
])


@pytest.mark.live
def test_a_cqc_over_a_written_case_equals_the_drivers_own(tmp_path: Path) -> None:
    result = _frame().spectrum_after_gravity(
        N_MODES,
        spectra=[DesignSpectrum(TN, SA, direction=1, scale=0.75),
                 DesignSpectrum(TN, SA, direction=2, scale=0.5)],
        extractors={"disp": NodeDisp(nodes=ROOF_COLUMN, dofs=(1, 2))},
        solver="-fullGenLapack")
    cases = spectrum_cases(result, ["E_x", "E_y"], damping=XI, nodal=["disp"])
    model = tmp_path / "frame.h5"
    with h5py.File(model, "w") as f:
        f.create_group("meta/lineage").attrs["fem_hash"] = "frame-under-test"
    path = write_user_file(tmp_path / "frame.user.h5", model, spectra=cases)
    spectra = UserSpectra(path, model_h5=model)

    # NodeDisp values are node-major: column 2 n + d is node n, DOF d.
    cqc = result.combine("disp", damping=XI)               # (2, 2 n)
    srss = result.combine("disp", rule="SRSS")
    directional = result.directional("disp", damping=XI)   # (2 n,)
    for d, component in enumerate(("displacement_x", "displacement_y")):
        source: Any = _RecordedSource(
            cast(Any, None), "nodes", component, None, spectra, "live")
        cols = slice(d, None, 2)
        for name, want in (("Ex_cqc", cqc[0, cols]), ("Ey_cqc", cqc[1, cols]),
                           ("Ex_srss", srss[0, cols]), ("dir", directional[cols])):
            got = evaluate(TABLE, name, source).field
            assert got.keys["id"].tolist() == ROOF_COLUMN
            np.testing.assert_allclose(got.values[:, 0], want, rtol=RTOL, err_msg=f"{name} {component}")

    shear: Any = _RecordedSource(cast(Any, None), "domain", "base_shear", None, spectra, "live")
    for name, want in (("Ex_cqc", result.combine("base_shear", damping=XI)[0]),
                       ("Ey_cqc", result.combine("base_shear", damping=XI)[1]),
                       ("Ex_srss", result.combine("base_shear", rule="SRSS")[0]),
                       ("dir", result.directional("base_shear", damping=XI))):
        got = evaluate(TABLE, name, shear).field
        assert got.components == ("1", "2")
        np.testing.assert_allclose(got.values[0], want, rtol=RTOL, err_msg=name)
