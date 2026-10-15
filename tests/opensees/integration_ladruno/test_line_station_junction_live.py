"""Fork lane: elastic-beam line stations meet the force beams at a junction,
read back from MPCO and ``.ladruno`` (#2009).

The bar, the oracle and the closed form are those of
``tests/test_results_line_station_junction_live.py`` (DomainCapture and
``.out``): force beam [0, 1], elastic beam [1, 2], force beam [2, 3], fixed
at x = 0, tip-loaded, solved in two steps through the bridge with
sequential tags. Here the bridge declares an MPCO and a ``.ladruno``
recorder of ``localForce`` before the run, and each elastic station read
back must equal the adjacent force beam's live ``section force`` at the
shared node, at ``rtol=1e-10``, for every component in 2-D and 3-D and for
each elastic class the bridge can build.

MPCO and ``.ladruno`` also take a force or disp beam's line stations from
``localForce`` when no ``section.force`` is recorded, and the one sign
table applies to those too. So ``Mid`` is also each sectioned family that
records ``localForce`` (:data:`SECTIONED`): its signed ``localForce``
stations must meet the force beams at both junctions the same way. A disp
beam's own ``section force`` reports no shear (Euler-Bernoulli kinematics),
so the junction, not its own sections, is the oracle for every family.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

from tests.test_results_line_station_junction_live import (
    COMPS, ELASTIC, LAMS, _bar_fem, _ndm, build_junction, close,
    closed_form, fem_to_ops, ids, junction_truth, station_values,
)

pytestmark = pytest.mark.ladruno_fork

KINDS = ("mpco", "ladruno")


def _sectioned(make: str) -> Any:
    def build(ops: Any, tr: Any, integ: Any) -> Any:
        return getattr(ops.element, make)(pg="Mid", transf=tr,
                                          integration=integ)
    return build


#: sectioned ``Mid`` class -> bridge builder on the force beams' section
SECTIONED = {
    f"{name}{d}d": _sectioned(make)
    for name, make in (("ForceBeamColumn", "forceBeamColumn"),
                       ("DispBeamColumn", "dispBeamColumn"),
                       ("LadrunoDispBeamColumn", "LadrunoDispBeamColumn"))
    for d in (2, 3)
}


def _solve(tmp: Path, cls: str) -> dict[str, Any]:
    fem = _bar_fem(f"junction_fork_{cls}")
    eid = ids(fem)
    paths = {k: str(tmp / f"bar.{k}") for k in KINDS}

    def recorders(ops: Any) -> None:
        # A node channel too: an element-only MPCO file read 0 steps (#1970).
        ops.recorder.MPCO(file=paths["mpco"],
                          nodal_responses=("displacement",),
                          elem_responses=("localForce",))
        ops.recorder.Ladruno(file=paths["ladruno"],
                             elem_responses=("localForce",))

    ops = build_junction(fem, cls, recorders, SECTIONED.get(cls))
    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    to_ops = fem_to_ops(model_h5)
    assert any(to_ops[e] != e for e in eid.values())    # not vacuous
    tags = {pg: to_ops[e] for pg, e in eid.items()}
    assert emitter.ops.eleType(tags["Mid"]) == cls

    truth = []
    for _ in LAMS:
        assert emitter.analyze(steps=1) == 0
        truth.append(junction_truth(emitter.ops, cls, tags))
    emitter.ops.remove("recorders")
    readers = {"mpco": Results.from_mpco, "ladruno": Results.from_ladruno}
    return {
        "results": {k: readers[k](p, fem=fem, model_h5=model_h5)
                    for k, p in paths.items()},
        "mid": eid["Mid"],
        "truth": np.stack(truth),                       # (T, 2, n_comp)
    }


@pytest.fixture(scope="module", params=sorted(ELASTIC) + sorted(SECTIONED))
def solved(request: pytest.FixtureRequest,
           tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    sol = _solve(tmp_path_factory.mktemp(f"junction_fork_{request.param}"),
                 request.param)
    sol["cls"] = request.param
    return sol


def test_force_beam_sections_are_statics(solved: dict[str, Any]) -> None:
    """The oracle: each live junction section force is the closed form."""
    ndm = _ndm(solved["cls"])
    for k, comp in enumerate(COMPS[ndm]):
        want = np.outer(LAMS, [closed_form(ndm, comp, 1.0),
                               closed_form(ndm, comp, 2.0)])
        close(solved["truth"][:, :, k], want, f"{solved['cls']} {comp}")


#: ``.ladruno`` spells a 2-D disp beam's ``localForce`` ``N1,V1,M1,..`` (no
#: underscore), which its station reader does not parse: the element gets
#: no stations at all. A read gap, not a sign; pinned so its fix shows.
UNREAD = {("ladruno", "DispBeamColumn2d"), ("ladruno", "LadrunoDispBeamColumn2d")}


@pytest.mark.parametrize("kind", KINDS)
def test_elastic_stations_meet_the_force_beams(
    solved: dict[str, Any], kind: str, request: pytest.FixtureRequest,
) -> None:
    """Station 1 equals the left force beam at x = 1 and station 2 the
    right one at x = 2: every component is continuous at both junctions."""
    if (kind, solved["cls"]) in UNREAD:
        request.applymarker(pytest.mark.xfail(
            strict=True, raises=AssertionError,
            reason=".ladruno reads no 2-D disp-beam localForce station "
                   "(N1,V1,M1 spelling)"))
    ndm = _ndm(solved["cls"])
    r = solved["results"][kind]
    for k, comp in enumerate(COMPS[ndm]):
        close(station_values(r, comp, solved["mid"]),
              solved["truth"][:, :, k],
              f"{kind} {solved['cls']} {comp} at the junctions")
