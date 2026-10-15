"""Fork lane: an MPCO line-stations *spec* record reads elastic beams (#2051).

Before #2051 ``spec.emit_mpco`` turned a line-stations record into
``-E section.force`` only. Elastic beams do not answer ``section.force``, so
their stations read back empty with no warning. The emit now also records
``localForce`` when the record names an elastic class, and the MPCO reader
builds the two end stations from it (#2020, #2048).

Every model is a cantilever along +x, fixed at x = 0, loaded at the tip
``x = L`` and solved in two load steps (``lam`` = 0.5, 1.0). The oracle is
the closed form in the force-beam ``section force`` convention every route
shares (#2009; derived in ``test_elastic_beam_stations_live.py``), at the
station's own ``x``::

    N = lam Fx, Vy = -lam Fy, Vz = lam Fz, T = lam Mx,
    My = -lam Fz (L - x), Mz = lam Fy (L - x).

A statically determinate cantilever has these forces whatever the element's
stiffness, so the same closed form checks Euler-Bernoulli, Timoshenko and
modified-stiffness elastic beams, and the force beam's sections.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from apeGmsh.opensees.emitter.live import get_ops
from apeGmsh.results.readers._mpco import MPCOReader
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
L = 4.0
FX, FY, FZ, MX = 1.0e3, -2.0e3, 3.0e3, 4.0e2
LAMS = (0.5, 1.0)
E, G, A = 2.0e11, 8.0e10, 0.03
IZ, IY, J = 5.0e-4, 4.0e-4, 3.0e-4
COMPS_3D = ("axial_force", "shear_y", "shear_z", "torsion",
            "bending_moment_y", "bending_moment_z")
COMPS_2D = ("axial_force", "shear_y", "bending_moment_z")


def _closed(comp: str, x: float, lam: float) -> float:
    return lam * {
        "axial_force": FX, "shear_y": -FY, "shear_z": FZ, "torsion": MX,
        "bending_moment_y": -FZ * (L - x), "bending_moment_z": FY * (L - x),
    }[comp]


#: element tag (nodes are 2*tag - 1 at x = 0 and 2*tag at x = L) -> command
Builder = Callable[[Any, int, int, int], None]

FAMILIES_2D: dict[str, Builder] = {
    "ElasticBeam2d": lambda o, t, i, j: o.element(
        "elasticBeamColumn", t, i, j, A, E, IZ, 1),
    "ElasticTimoshenkoBeam2d": lambda o, t, i, j: o.element(
        "ElasticTimoshenkoBeam", t, i, j, E, G, A, IZ, 0.8 * A, 1),
    "ModElasticBeam2d": lambda o, t, i, j: o.element(
        "ModElasticBeam2d", t, i, j, A, E, IZ, 4.0, 4.0, 2.0, 1),
}
FAMILIES_3D: dict[str, Builder] = {
    "ElasticBeam3d": lambda o, t, i, j: o.element(
        "elasticBeamColumn", t, i, j, A, E, G, J, IY, IZ, 1),
    "ElasticTimoshenkoBeam3d": lambda o, t, i, j: o.element(
        "ElasticTimoshenkoBeam", t, i, j, E, G, A, J, IY, IZ,
        0.8 * A, 0.8 * A, 1),
}


def _force_beam_3d(o: Any, t: int, i: int, j: int) -> None:
    o.section("Elastic", 1, E, A, IZ, IY, G, J)
    o.beamIntegration("Lobatto", 1, 1, 4)
    o.element("forceBeamColumn", t, i, j, 1, 1)


def _solve(path: Path, ndm: int, builders: list[Builder],
           records: tuple[ResolvedRecorderRecord, ...]) -> list[float]:
    """Build one cantilever per builder, record MPCO, solve; return times."""
    o = get_ops()
    o.wipe()
    ndf = 6 if ndm == 3 else 3
    o.model("basic", "-ndm", ndm, "-ndf", ndf)
    if ndm == 3:
        o.geomTransf("Linear", 1, 0.0, 0.0, 1.0)     # local y = Y, z = Z
    else:
        o.geomTransf("Linear", 1)
    o.timeSeries("Linear", 1)
    o.pattern("Plain", 1, 1)
    for k, build in enumerate(builders):
        t, i, j = k + 1, 2 * k + 1, 2 * k + 2
        y = 2.0 * k
        if ndm == 3:
            o.node(i, 0.0, y, 0.0)
            o.node(j, L, y, 0.0)
            o.fix(i, *([1] * 6))
            o.load(j, FX, FY, FZ, MX, 0.0, 0.0)
        else:
            o.node(i, 0.0, y)
            o.node(j, L, y)
            o.fix(i, 1, 1, 1)
            o.load(j, FX, FY, 0.0)
        build(o, t, i, j)
    o.constraints("Plain")
    o.numberer("RCM")
    o.system("BandGeneral")
    o.test("NormDispIncr", 1e-12, 10)
    # Newton, not Linear: under Linear, ModElasticBeam2d answers
    # localForce one step late (it forms its end forces in
    # getResistingForce only; eleResponse reads 0 at lam = 0.5).
    o.algorithm("Newton")
    o.integrator("LoadControl", LAMS[0])
    o.analysis("Static")
    spec = ResolvedRecorderSpec(fem_snapshot_id="raw-ops-2051", records=records)
    times = []
    with spec.emit_mpco(str(path), ops=o):
        for _ in LAMS:
            assert o.analyze(1) == 0
            times.append(o.getTime())
    o.wipe()
    return times


def _line(cls: str | None, eids: list[int], comps: tuple[str, ...],
          name: str = "beam") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="line_stations", name=name, components=comps,
        dt=None, n_steps=None, element_ids=np.array(eids),
        element_class_name=cls,
    )


def _nodes(n: int) -> ResolvedRecorderRecord:
    # A node channel: an element-only MPCO file reads 0 steps (#1970).
    return ResolvedRecorderRecord(
        category="nodes", name="u", components=("displacement_x",),
        dt=None, n_steps=None, node_ids=np.arange(1, 2 * n + 1),
    )


def _assert_closed_form(path: Path, eids: list[int],
                        comps: tuple[str, ...], label: str) -> None:
    with MPCOReader(str(path)) as r:
        sid = r.stages()[0].id
        assert len(r.time_vector(sid)) == len(LAMS)
        for comp in comps:
            slab = r.read_line_stations(sid, comp, element_ids=np.array(eids))
            ei = np.asarray(slab.element_index)
            assert sorted({int(e) for e in ei}) == sorted(eids), (label, comp)
            values = np.asarray(slab.values)
            assert values.shape == (len(LAMS), ei.size)
            x = (np.asarray(slab.station_natural_coord) + 1.0) * L / 2.0
            want = np.array([[_closed(comp, xs, lam) for xs in x]
                             for lam in LAMS])
            np.testing.assert_allclose(
                values, want, rtol=RTOL, atol=RTOL * abs(FY) * L,
                err_msg=f"{comp} of {label}",
            )


@pytest.mark.parametrize("cls", sorted(FAMILIES_2D) + sorted(FAMILIES_3D))
def test_elastic_family_reads_its_end_stations(tmp_path: Path, cls: str) -> None:
    ndm = 3 if cls in FAMILIES_3D else 2
    build = {**FAMILIES_2D, **FAMILIES_3D}[cls]
    comps = COMPS_3D if ndm == 3 else COMPS_2D
    path = tmp_path / "run.mpco"
    _solve(path, ndm, [build], (_line(cls, [1], comps), _nodes(1)))
    with MPCOReader(str(path)) as r:
        slab = r.read_line_stations(r.stages()[0].id, comps[0],
                                    element_ids=np.array([1]))
        np.testing.assert_array_equal(slab.station_natural_coord, [-1.0, 1.0])
    _assert_closed_form(path, [1], comps, cls)


def test_mixed_elastic_and_force_beam_record_reads_both(tmp_path: Path) -> None:
    """One record over an ElasticBeam3d (tag 1) and a forceBeamColumn (tag 2).

    The force beam reads its four Lobatto sections from ``section.force``,
    the elastic beam its two ends from ``localForce``.
    """
    path = tmp_path / "run.mpco"
    _solve(path, 3, [FAMILIES_3D["ElasticBeam3d"], _force_beam_3d],
           (_line("ElasticBeam3d", [1, 2], COMPS_3D), _nodes(2)))
    with MPCOReader(str(path)) as r:
        slab = r.read_line_stations(r.stages()[0].id, "bending_moment_z",
                                    element_ids=np.array([1, 2]))
        ei = np.asarray(slab.element_index)
        assert int((ei == 1).sum()) == 2 and int((ei == 2).sum()) == 4
    _assert_closed_form(path, [1, 2], COMPS_3D, "mixed record")


def test_no_class_warns_and_the_elastic_beam_reads_empty(tmp_path: Path) -> None:
    """The warning is not a false alarm: without a class the read is empty."""
    path = tmp_path / "run.mpco"
    with pytest.warns(UserWarning, match=r"'beam' names no element class"):
        _solve(path, 3, [FAMILIES_3D["ElasticBeam3d"]],
               (_line(None, [1], COMPS_3D), _nodes(1)))
    with MPCOReader(str(path)) as r:
        sid = r.stages()[0].id
        assert len(r.time_vector(sid)) == len(LAMS)
        slab = r.read_line_stations(sid, "bending_moment_z",
                                    element_ids=np.array([1]))
        assert np.asarray(slab.values).size == 0
