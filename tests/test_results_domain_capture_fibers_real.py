"""Phase 11e — DomainCapture fiber path against real openseespy.

Builds a tiny standalone OpenSees fiber-section beam model (single
beam, simple rectangular fiber section) without any apeGmsh
involvement, runs a static analysis, and captures fiber stress +
strain via ``DomainCapture``. The point is to prove the new
``_FiberCapturer`` works against the real ``ops`` API, complementing
the mocked tests in ``test_results_domain_capture_fibers.py``.

We exercise both element families covered by ``FIBER_CATALOG``:

- ``ForceBeamColumn3d`` (force-based)
- ``DispBeamColumn3d`` (displacement-based)

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
    """Synthetic FEMData backed by a real snapshot_id."""
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


# =====================================================================
# Helpers
# =====================================================================

# Simple square fiber section: 3×3 grid, side length 0.2 m.
_N_FIBERS_PER_SIDE = 3
_SECTION_HALF_WIDTH = 0.1
_FIBER_AREA = (2 * _SECTION_HALF_WIDTH / _N_FIBERS_PER_SIDE) ** 2


def _build_fiber_section(sec_tag: int, mat_tag: int) -> int:
    """Define a 9-fiber rectangular section. Returns the fiber count."""
    # 3D fiber section requires torsional rigidity — supply a stiff value.
    ops.section("Fiber", sec_tag, "-GJ", 1.0e10)
    yzs = np.linspace(
        -_SECTION_HALF_WIDTH + (_SECTION_HALF_WIDTH / _N_FIBERS_PER_SIDE),
        _SECTION_HALF_WIDTH - (_SECTION_HALF_WIDTH / _N_FIBERS_PER_SIDE),
        _N_FIBERS_PER_SIDE,
    )
    n = 0
    for y in yzs:
        for z in yzs:
            ops.fiber(float(y), float(z), float(_FIBER_AREA), mat_tag)
            n += 1
    return n


def _build_cantilever_with_element(
    element_kind: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a 2-node cantilever along x. Returns (node_ids, coords, ele_tag).

    ``element_kind`` ∈ {``"forceBeamColumn"``, ``"dispBeamColumn"``}.
    Both elements use the same fiber section and Lobatto integration
    with 3 IPs.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), *map(float, xyz))
    ops.fix(1, 1, 1, 1, 1, 1, 1)

    # Linear elastic uniaxial — keeps the math predictable.
    mat_tag = 1
    ops.uniaxialMaterial("Elastic", mat_tag, 210e9)
    sec_tag = 1
    n_fibers = _build_fiber_section(sec_tag, mat_tag)
    assert n_fibers == _N_FIBERS_PER_SIDE ** 2

    transf_tag = 1
    ops.geomTransf("Linear", transf_tag, 0.0, 0.0, 1.0)
    ele_tag = 1
    ops.beamIntegration("Lobatto", 1, sec_tag, 3)    # 3 IPs
    if element_kind == "forceBeamColumn":
        ops.element(
            "forceBeamColumn", ele_tag, 1, 2, transf_tag, 1,
        )
    elif element_kind == "dispBeamColumn":
        ops.element(
            "dispBeamColumn", ele_tag, 1, 2, transf_tag, 1,
        )
    else:
        raise ValueError(f"Unknown element_kind={element_kind!r}")

    return node_ids, coords, ele_tag


def _setup_static_analysis(load_at_tip_y: float) -> None:
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, float(load_at_tip_y), 0.0, 0.0, 0.0, 0.0)
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


# =====================================================================
# Force-based beam capture
# =====================================================================

def test_force_beam_fiber_capture(tmp_path: Path) -> None:
    """Capture fibers on a force-based cantilever and check shape + signs."""
    node_ids, coords, ele_tag = _build_cantilever_with_element(
        "forceBeamColumn",
    )
    _setup_static_analysis(load_at_tip_y=1.0e5)        # 100 kN transverse

    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="fibers", name="fbc_fibers",
                components=("fiber_stress", "fiber_strain"),
                dt=None, n_steps=None,
                element_ids=np.array([ele_tag]),
            ),
        ),
    )

    cap_path = tmp_path / "fbc.h5"
    with DomainCapture(spec, cap_path, fem, ndm=3, ndf=6) as cap:
        cap.begin_stage("push", kind="static")
        for _ in range(3):
            assert ops.analyze(1) == 0
            cap.step(t=ops.getTime())
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(cap_path, fem=fem) as r:
        slab = r.elements.fibers.get(component="fiber_stress")
        # 1 element × 3 IPs × 9 fibers = 27 fiber rows; 3 time steps.
        assert slab.values.shape == (3, 27)
        assert np.unique(slab.element_index).tolist() == [ele_tag]
        # Three integration points expected.
        assert sorted(np.unique(slab.gp_index).tolist()) == [0, 1, 2]
        # Bending under transverse load: stress varies linearly with
        # y (one side in tension, the other in compression). Convention
        # depends on OpenSees curvature sign — we assert the *gradient*
        # rather than a specific tension-side.
        last_step = slab.values[-1]
        # Pick the first IP (closest to the fixed end), where the
        # bending moment is largest.
        gp0 = slab.gp_index == 0
        ys_gp0 = slab.y[gp0]
        sigmas_gp0 = last_step[gp0]
        # Sort by y; verify monotonic stress change across the section
        # (linear elastic + pure bending → linear σ(y)).
        order = np.argsort(ys_gp0)
        sorted_sigmas = sigmas_gp0[order]
        # Stress at extreme y has opposite sign to stress at extreme -y,
        # both with magnitude above floating-point noise.
        assert np.sign(sorted_sigmas[0]) == -np.sign(sorted_sigmas[-1])
        assert abs(sorted_sigmas[0]) > 1.0
        assert abs(sorted_sigmas[-1]) > 1.0
        # Linear elastic + LoadControl with dt=1.0 → stress at any
        # given fiber should scale with the cumulative load factor.
        # Pick the most-loaded fiber (max |σ|) and check the ratio
        # across the first and last steps.
        first_step = slab.values[0]
        peak_idx = int(np.argmax(np.abs(last_step)))
        ratio = abs(last_step[peak_idx]) / abs(first_step[peak_idx])
        # 3 LoadControl steps with dt=1 each → final factor / first
        # factor = 3.0.
        assert ratio == pytest.approx(3.0, rel=1e-6)

        # Strain comes through as well.
        strain_slab = r.elements.fibers.get(component="fiber_strain")
        # Linear elastic σ = E·ε → ε = σ / E with E = 210e9.
        np.testing.assert_allclose(
            strain_slab.values, slab.values / 210e9, rtol=1e-6, atol=1e-12,
        )


# =====================================================================
# Disp-based beam capture
# =====================================================================

def test_disp_beam_fiber_capture(tmp_path: Path) -> None:
    """Capture fibers on a disp-based cantilever — same checks as above."""
    node_ids, coords, ele_tag = _build_cantilever_with_element(
        "dispBeamColumn",
    )
    _setup_static_analysis(load_at_tip_y=1.0e5)

    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="fibers", name="dbc_fibers",
                components=("fiber_stress",),
                dt=None, n_steps=None,
                element_ids=np.array([ele_tag]),
            ),
        ),
    )

    cap_path = tmp_path / "dbc.h5"
    with DomainCapture(spec, cap_path, fem, ndm=3, ndf=6) as cap:
        cap.begin_stage("push", kind="static")
        for _ in range(2):
            assert ops.analyze(1) == 0
            cap.step(t=ops.getTime())
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(cap_path, fem=fem) as r:
        slab = r.elements.fibers.get(component="fiber_stress")
        # 1 element × 3 IPs × 9 fibers = 27 cols; 2 steps.
        assert slab.values.shape == (2, 27)
        # Bending sign check at the fixed-end IP, last step.
        gp0 = slab.gp_index == 0
        ys = slab.y[gp0]
        sigmas = slab.values[-1, gp0]
        order = np.argsort(ys)
        sorted_sigmas = sigmas[order]
        # Linear elastic + pure bending → linear σ(y) with opposite
        # signs at extreme +y vs -y (sign convention is OpenSees-defined).
        assert np.sign(sorted_sigmas[0]) == -np.sign(sorted_sigmas[-1])
        assert abs(sorted_sigmas[0]) > 0
        assert abs(sorted_sigmas[-1]) > 0
        # Strain wasn't requested → reader yields an empty slab.
        strain = r.elements.fibers.get(component="fiber_strain")
        assert strain.values.shape[1] == 0


# =====================================================================
# Geometry round-trip
# =====================================================================

def test_geometry_matches_section_definition(tmp_path: Path) -> None:
    """Captured (y, z, area) per fiber matches the ``ops.fiber`` calls."""
    node_ids, coords, ele_tag = _build_cantilever_with_element(
        "forceBeamColumn",
    )
    _setup_static_analysis(load_at_tip_y=1.0e3)
    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="fibers", name="r",
                components=("fiber_stress",),
                dt=None, n_steps=None,
                element_ids=np.array([ele_tag]),
            ),
        ),
    )

    cap_path = tmp_path / "geom.h5"
    with DomainCapture(spec, cap_path, fem, ndm=3, ndf=6) as cap:
        cap.begin_stage("g", kind="static")
        ops.analyze(1)
        cap.step(t=ops.getTime())
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(cap_path, fem=fem) as r:
        slab = r.elements.fibers.get(component="fiber_stress")
        # First IP has 9 fibers — same y/z grid we built.
        gp0 = slab.gp_index == 0
        assert gp0.sum() == 9
        # Fiber area is uniform.
        np.testing.assert_allclose(
            slab.area[gp0], _FIBER_AREA, rtol=1e-9,
        )
        # All fibers reference material_tag=1.
        assert (slab.material_tag[gp0] == 1).all()
        # y / z form the expected 3×3 grid.
        ys_unique = sorted(set(np.round(slab.y[gp0], 6)))
        zs_unique = sorted(set(np.round(slab.z[gp0], 6)))
        assert len(ys_unique) == _N_FIBERS_PER_SIDE
        assert len(zs_unique) == _N_FIBERS_PER_SIDE
