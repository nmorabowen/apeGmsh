"""Phase 11b Step 2b — real openseespy line-stations capture test.

Builds a tiny ForceBeamColumn3d cantilever directly via openseespy,
runs a static analysis, captures section forces through
``DomainCapture._LineStationCapturer``, and reads back. Verifies
that ``ops.eleResponse(eid, "integrationPoints")`` is correctly
normalised to ``[-1, +1]`` and that section codes are inferred
correctly from the per-IP force vector length.

Skipped if openseespy isn't importable.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

openseespy = pytest.importorskip(
    "openseespy.opensees", reason="openseespy required",
)
ops = openseespy


class _MinimalFem:
    """Tiny synthetic FEMData backed by a real snapshot_id."""

    def __init__(self, node_ids: np.ndarray, coords: np.ndarray) -> None:
        self.nodes = SimpleNamespace(ids=node_ids, coords=coords)
        self.elements = []

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group) -> None:
        group.attrs["snapshot_id"] = self.snapshot_id
        group.attrs["ndm"] = 3
        group.attrs["ndf"] = 6
        group.attrs["model_name"] = ""
        group.attrs["units"] = ""
        nodes_grp = group.create_group("nodes")
        nodes_grp.create_dataset("ids", data=self.nodes.ids)
        nodes_grp.create_dataset("coords", data=self.nodes.coords)
        group.create_group("elements")


def test_force_beam_3d_lobatto5_aggregated(tmp_path: Path) -> None:
    """ForceBeamColumn3d, Lobatto-5, aggregated [P,Mz,My,T,Vy,Vz] section.

    Pure axial tip load → axial force = applied load at every IP, all
    other components zero (rigid linear analysis).
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    L = 5.0
    P_axial = 1.0e3

    coords = np.array([
        [0.0, 0.0, 0.0],
        [L,   0.0, 0.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ops.fix(1, 1, 1, 1, 1, 1, 1)

    ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)
    # Aggregated 6-comp section: [P, Mz, My, T, Vy, Vz].
    ops.section("Elastic", 1, 200e9, 0.01, 1.0e-4, 1.5e-4, 80e9, 1.0e-4)
    ops.uniaxialMaterial("Elastic", 100, 1.0e9)
    ops.uniaxialMaterial("Elastic", 101, 1.0e9)
    ops.section("Aggregator", 11, 100, "Vy", 101, "Vz", "-section", 1)
    ops.beamIntegration("Lobatto", 1, 11, 5)
    ops.element("forceBeamColumn", 1, 1, 2, 1, 1)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, P_axial, 0.0, 0.0, 0.0, 0.0, 0.0)

    ops.system("ProfileSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    # ── Build a synthetic FEMData matching the OpenSees domain ──────
    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    from apeGmsh.results import Results
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="line_stations", name="beam",
                components=(
                    "axial_force", "bending_moment_z",
                    "bending_moment_y", "torsion",
                    "shear_y", "shear_z",
                ),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
            ),
        ),
    )

    path = tmp_path / "cap.h5"
    with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=ops) as cap:
        cap.begin_stage("static", kind="static")
        ops.analyze(1)
        cap.step(t=ops.getTime())
        cap.end_stage()
        # Sanity: capturer didn't skip the element.
        (lc,) = cap._line_station_capturers
        assert lc.skipped_elements == [], (
            f"unexpected skipped elements: {lc.skipped_elements}"
        )

    # ── Read back through the public API ────────────────────────────
    expected_gp_x = np.array(
        [-1.0, -0.65465367, 0.0, 0.65465367, 1.0],
        dtype=np.float64,
    )
    with Results.from_native(path) as r:
        s = r.stage(r.stages[0].id)
        comps = set(s.elements.line_stations.available_components())
        for name in (
            "axial_force", "bending_moment_z", "bending_moment_y",
            "torsion", "shear_y", "shear_z",
        ):
            assert name in comps, f"missing {name} in {sorted(comps)}"

        slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 5)
        assert slab.element_index.tolist() == [1, 1, 1, 1, 1]
        np.testing.assert_allclose(
            slab.station_natural_coord, expected_gp_x, atol=1e-5,
        )
        # Axial force = applied tip load at every IP.
        np.testing.assert_allclose(
            slab.values[0], np.full(5, P_axial), rtol=1e-9, atol=1e-6,
        )
        # All other components zero under pure axial load.
        for name in ("shear_y", "shear_z", "torsion",
                     "bending_moment_y", "bending_moment_z"):
            other = s.elements.line_stations.get(component=name)
            np.testing.assert_allclose(
                other.values[0], np.zeros(5), atol=1e-6,
                err_msg=f"{name} should be zero under pure axial load",
            )


def test_force_beam_3d_disp_beam_skipped(tmp_path: Path) -> None:
    """DispBeamColumn3d should be skipped by DomainCapture (Tier 2/3).

    OpenSees disp-based beam-columns don't expose
    ``"integrationPoints"`` directly; the capturer should record them
    as skipped and produce no on-disk entries.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    L = 4.0
    coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], dtype=np.float64)
    node_ids = np.array([1, 2], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ops.fix(1, 1, 1, 1, 1, 1, 1)

    ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)
    ops.section("Elastic", 1, 200e9, 0.01, 1.0e-4, 1.5e-4, 80e9, 1.0e-4)
    ops.beamIntegration("Lobatto", 1, 1, 3)
    ops.element("dispBeamColumn", 1, 1, 2, 1, 1)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0e3, 0.0, 0.0, 0.0, 0.0, 0.0)
    ops.system("ProfileSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    from apeGmsh.results import Results
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="line_stations", name="beam",
                components=("axial_force",),
                dt=None, n_steps=None,
                element_ids=np.array([1], dtype=np.int64),
            ),
        ),
    )

    path = tmp_path / "cap.h5"
    with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=ops) as cap:
        cap.begin_stage("static", kind="static")
        ops.analyze(1)
        # If DispBeamColumn happens to support integrationPoints in
        # this OpenSees build, the capturer succeeds and the test below
        # must read non-empty data. Either outcome is acceptable; we
        # just verify no crash and consistent on-disk state.
        cap.step(t=ops.getTime())
        cap.end_stage()
        (lc,) = cap._line_station_capturers
        skipped_eids = [eid for eid, _ in lc.skipped_elements]
        succeeded = lc._groups is not None and any(
            lc._groups.values()
        )

    if skipped_eids == [1]:
        # Skipped path: no line-station data on disk.
        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 0)
    else:
        # If the OpenSees build DOES expose integrationPoints for
        # dispBeamColumn (some forks add it), verify the read works.
        assert succeeded, (
            "dispBeamColumn was neither skipped nor captured — "
            "unexpected DomainCapture state."
        )
        with Results.from_native(path) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.line_stations.get(component="axial_force")
        assert slab.values.shape == (1, 3)
