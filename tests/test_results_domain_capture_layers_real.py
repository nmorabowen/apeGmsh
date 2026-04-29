"""Phase 11f — DomainCapture layer path against real openseespy.

Builds a tiny standalone OpenSees layered-shell model (one ASDShellQ4
with a 3-layer LayeredShellFiberSection), runs a static analysis,
captures via ``DomainCapture``, and reads back. Cross-checks shape,
geometry plumbing (per-layer thickness from session, identity
quaternion from a unit-quad in xy), and that the per-layer probe
returns nonzero stress under load.

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


def _build_one_quad_layered_shell() -> tuple[np.ndarray, np.ndarray, int]:
    """Build a unit-quad ASDShellQ4 cantilever in the xy-plane.

    Returns (node_ids, coords, ele_tag).
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), *map(float, xyz))
    # Clamp the back edge.
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.fix(4, 1, 1, 1, 1, 1, 1)

    # Two NDMaterials → wrapped as PlateFiber → assembled into a
    # 3-layer LayeredShellFiberSection.
    ops.nDMaterial("ElasticIsotropic", 1, 30e9, 0.2)        # concrete
    ops.nDMaterial("ElasticIsotropic", 2, 200e9, 0.3)       # steel
    ops.section(
        "LayeredShell", 7, 3,
        1, 0.10,    # layer 1: concrete, 10 cm
        2, 0.005,   # layer 2: steel, 0.5 cm
        1, 0.10,    # layer 3: concrete, 10 cm
    )

    ele_tag = 1
    ops.element("ASDShellQ4", ele_tag, 1, 2, 3, 4, 7)

    return node_ids, coords, ele_tag


def _setup_static_analysis(load_at_node: int, fz: float) -> None:
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(load_at_node, 0.0, 0.0, fz, 0.0, 0.0, 0.0)
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


# =====================================================================
# End-to-end real layered-shell capture
# =====================================================================

def test_layered_shell_capture_round_trip(tmp_path: Path) -> None:
    """Capture a 3-layer ASDShellQ4 + read back the LayerSlab."""
    node_ids, coords, ele_tag = _build_one_quad_layered_shell()
    # Apply transverse z load at the free corner.
    _setup_static_analysis(load_at_node=2, fz=1.0e3)

    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    # Build LayerSectionMetadata by hand (no apeGmsh session here).
    from apeGmsh.solvers._recorder_specs import (
        LayerSectionDef, LayerSectionMetadata,
        ResolvedRecorderRecord, ResolvedRecorderSpec,
    )
    sec_meta = LayerSectionMetadata(
        sections={
            7: LayerSectionDef(
                section_tag=7, section_name="LayeredShell_7",
                n_layers=3,
                thickness=np.array([0.10, 0.005, 0.10]),
                material_tags=np.array([1, 2, 1]),
            ),
        },
        element_to_section={ele_tag: 7},
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            ResolvedRecorderRecord(
                category="layers", name="slab_layers",
                components=("fiber_stress", "fiber_strain"),
                dt=None, n_steps=None,
                element_ids=np.array([ele_tag]),
                layer_section_metadata=sec_meta,
            ),
        ),
    )

    from apeGmsh.results.capture._domain import DomainCapture
    cap_path = tmp_path / "shell.h5"
    with DomainCapture(spec, cap_path, fem, ndm=3, ndf=6) as cap:
        cap.begin_stage("g", kind="static")
        for _ in range(2):
            assert ops.analyze(1) == 0
            cap.step(t=ops.getTime())
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(cap_path, fem=fem) as r:
        from apeGmsh.results.readers._protocol import ResultLevel
        sid = r.stages[0].id
        avail = r._reader.available_components(sid, ResultLevel.LAYERS)
        # PlateFiber returns 5 components per call (in-plane + transverse
        # shear), so capture surfaces them as fiber_stress_0..4.
        # If it ever turns out to return a different N, this assertion
        # will catch the divergence.
        if "fiber_stress" in avail:
            indexed = False
        else:
            indexed = True
            assert all(
                f"fiber_stress_{k}" in avail for k in range(5)
            ), f"available={avail}"

        s = r.stage(sid)
        comp = "fiber_stress" if not indexed else "fiber_stress_0"
        slab = s.elements.layers.get(component=comp)
        # Slab dimensions: 1 element × 4 surface GPs × 3 layers = 12
        # cells. 2 time steps.
        assert slab.values.shape == (2, 12)
        np.testing.assert_array_equal(
            slab.element_index, np.full(12, ele_tag),
        )
        np.testing.assert_array_equal(
            slab.gp_index,
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        )
        np.testing.assert_array_equal(
            slab.layer_index,
            [0, 1, 2] * 4,
        )
        # Sub-GP fixed at 0 in v1.
        assert (slab.sub_gp_index == 0).all()
        # Per-layer thickness pulled from metadata.
        np.testing.assert_allclose(
            slab.thickness, [0.10, 0.005, 0.10] * 4,
        )
        # Quaternion: unit quad in xy → identity (1, 0, 0, 0).
        np.testing.assert_allclose(
            slab.local_axes_quaternion,
            np.tile([1.0, 0.0, 0.0, 0.0], (12, 1)),
            atol=1e-12,
        )
        # Last step under load — at least some cells nonzero.
        assert np.any(np.abs(slab.values[-1]) > 1e-6)
