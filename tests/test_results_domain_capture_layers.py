"""Phase 11f — layered-shell DomainCapture, mocked ops.

Exercises the in-process capture path for ``category="layers"``
records without spinning up OpenSees. A fake ops module returns
deterministic ``eleType``, ``eleNodes``, ``nodeCoord``, and
per-layer ``ops.eleResponse(eid, "material", str(gp), "fiber",
str(layer), "stress"|"strain")`` values so the
LayerSectionMetadata + quaternion-from-geometry + per-layer probe
flow can be inspected on disk.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.capture._domain import DomainCapture
from apeGmsh.solvers._recorder_specs import (
    LayerSectionDef,
    LayerSectionMetadata,
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)


# =====================================================================
# Fake ops with shell + layered section support
# =====================================================================

class _FakeOpsLayers:
    """openseespy stand-in: shells + per-layer eleResponse."""

    def __init__(self) -> None:
        self.ele_class: dict[int, str] = {}
        self.ele_nodes: dict[int, list[int]] = {}
        self.node_coords: dict[int, np.ndarray] = {}
        # Per-layer responses keyed by (eid, gp, layer, "stress"/"strain")
        self.layer_response: dict[tuple[int, int, int, str], np.ndarray] = {}

    def eleType(self, eid: int) -> str:
        return self.ele_class[int(eid)]

    def eleNodes(self, eid: int) -> list[int]:
        return list(self.ele_nodes[int(eid)])

    def nodeCoord(self, node_tag: int) -> list[float]:
        return list(self.node_coords[int(node_tag)])

    def eleResponse(self, eid: int, *args) -> list[float]:
        # Expected pattern: ("material", str(gp), "fiber", str(layer), token)
        if (
            len(args) == 5
            and args[0] == "material"
            and args[2] == "fiber"
        ):
            gp = int(args[1])
            layer = int(args[3])
            token = str(args[4])
            return list(
                self.layer_response[(int(eid), gp, layer, token)],
            )
        raise KeyError(f"unhandled eleResponse args: {args}")

    def reactions(self) -> None: ...


# =====================================================================
# Mock fem
# =====================================================================

class _MockFem:
    def __init__(self, node_ids) -> None:
        ids = np.asarray(node_ids, dtype=np.int64)
        coords = np.zeros((ids.size, 3), dtype=np.float64)
        self.nodes = SimpleNamespace(ids=ids, coords=coords)
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
        n = group.create_group("nodes")
        n.create_dataset("ids", data=np.asarray(self.nodes.ids, dtype=np.int64))
        n.create_dataset(
            "coords", data=np.asarray(self.nodes.coords, dtype=np.float64),
        )
        group.create_group("elements")


def _spec_with(*records, snapshot_id) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id,
        records=tuple(records),
    )


# =====================================================================
# Single-element single-class layer capture
# =====================================================================

class TestSingleASDShellQ4:
    def test_three_layer_capture(self, tmp_path: Path) -> None:
        """One ASDShellQ4, 3 layers, 1-component-per-layer (scalar
        plane-stress probe). Verify shape, geometry, and component
        canonical name (``fiber_stress`` since N==1)."""
        fem = _MockFem([1, 2, 3, 4])

        # 3-layer section: thicknesses (10, 5, 10), mat tags (1, 2, 1).
        sec_meta = LayerSectionMetadata(
            sections={
                10: LayerSectionDef(
                    section_tag=10, section_name="Slab",
                    n_layers=3,
                    thickness=np.array([10.0, 5.0, 10.0]),
                    material_tags=np.array([1, 2, 1]),
                ),
            },
            element_to_section={100: 10},
        )

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="layers", name="slab_layers",
                components=("fiber_stress",),
                dt=None, n_steps=None,
                element_ids=np.array([100]),
                layer_section_metadata=sec_meta,
            ),
            snapshot_id=fem.snapshot_id,
        )

        ops = _FakeOpsLayers()
        ops.ele_class[100] = "ASDShellQ4"
        ops.ele_nodes[100] = [1, 2, 3, 4]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([1.0, 0.0, 0.0])
        ops.node_coords[3] = np.array([1.0, 1.0, 0.0])
        ops.node_coords[4] = np.array([0.0, 1.0, 0.0])
        # Quad_GL_2 → 4 surface GPs. Per (gp, layer) one scalar.
        # Encode: stress = 100*step + 10*gp + layer; strain unused.
        for step_idx in range(2):
            for gp in (1, 2, 3, 4):
                for layer in (1, 2, 3):
                    val = 100.0 * step_idx + 10.0 * gp + float(layer)
                    ops.layer_response[(100, gp, layer, "stress")] = (
                        np.array([val])
                    )

        path = tmp_path / "layers.h5"
        with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=ops) as cap:
            cap.begin_stage("static", kind="static")
            for step_idx in range(2):
                # Re-arm tables so step values reflect step_idx.
                for gp in (1, 2, 3, 4):
                    for layer in (1, 2, 3):
                        val = (
                            100.0 * step_idx + 10.0 * gp + float(layer)
                        )
                        ops.layer_response[(100, gp, layer, "stress")] = (
                            np.array([val])
                        )
                cap.step(t=float(step_idx))
            cap.end_stage()

        with Results.from_native(path, fem=fem) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.layers.get(component="fiber_stress")
            # 1 element × 4 GPs × 3 layers = 12 cells. 2 time steps.
            assert slab.values.shape == (2, 12)
            np.testing.assert_array_equal(
                slab.element_index, np.full(12, 100),
            )
            # GP order: 0,0,0,1,1,1,2,2,2,3,3,3 (gp slowest, layer fastest)
            np.testing.assert_array_equal(
                slab.gp_index,
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            )
            np.testing.assert_array_equal(
                slab.layer_index,
                [0, 1, 2] * 4,
            )
            # Sub-GP is always 0 in v1.
            assert (slab.sub_gp_index == 0).all()
            # Thickness picked up from metadata, repeated per gp.
            np.testing.assert_allclose(
                slab.thickness,
                [10.0, 5.0, 10.0] * 4,
            )
            # Quaternion (identity for unit quad in xy-plane).
            np.testing.assert_allclose(
                slab.local_axes_quaternion,
                np.tile([1.0, 0.0, 0.0, 0.0], (12, 1)),
                atol=1e-12,
            )
            # Step-1 stress = 1*100 + 10*(gp+1) + (layer+1) (the
            # capturer uses 1-based gp/layer in ops calls; the slab
            # stores 0-based indices).
            # Step 1, gp=0, layer=0 → ops gp=1, layer=1, val=100+10+1=111
            # Step 1, gp=3, layer=2 → ops gp=4, layer=3, val=100+40+3=143
            np.testing.assert_array_equal(
                slab.values[1, 0:3], [111.0, 112.0, 113.0],
            )
            np.testing.assert_array_equal(
                slab.values[1, 9:12], [141.0, 142.0, 143.0],
            )


# =====================================================================
# Multi-component-per-layer (N>1) — indexed canonical names
# =====================================================================

class TestMultiComponentPerLayer:
    def test_5_component_response_surfaces_indexed_canonicals(
        self, tmp_path: Path,
    ) -> None:
        fem = _MockFem([1, 2, 3, 4])

        sec_meta = LayerSectionMetadata(
            sections={
                7: LayerSectionDef(
                    section_tag=7, section_name="Slab2L",
                    n_layers=2,
                    thickness=np.array([0.05, 0.05]),
                    material_tags=np.array([3, 3]),
                ),
            },
            element_to_section={50: 7},
        )

        spec = _spec_with(
            ResolvedRecorderRecord(
                category="layers", name="multi",
                components=("fiber_stress",),
                dt=None, n_steps=None,
                element_ids=np.array([50]),
                layer_section_metadata=sec_meta,
            ),
            snapshot_id=fem.snapshot_id,
        )

        ops = _FakeOpsLayers()
        ops.ele_class[50] = "ASDShellQ4"
        ops.ele_nodes[50] = [1, 2, 3, 4]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([1.0, 0.0, 0.0])
        ops.node_coords[3] = np.array([1.0, 1.0, 0.0])
        ops.node_coords[4] = np.array([0.0, 1.0, 0.0])
        # 5-component plane-stress + transverse-shear vector per cell.
        for gp in (1, 2, 3, 4):
            for layer in (1, 2):
                ops.layer_response[(50, gp, layer, "stress")] = (
                    np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + 10.0 * (
                        gp - 1
                    ) + 0.1 * (layer - 1)
                )

        path = tmp_path / "multi.h5"
        with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

        with Results.from_native(path, fem=fem) as r:
            s = r.stage(r.stages[0].id)
            # N>1 → indexed canonicals.
            avail = r._reader.available_components(
                r.stages[0].id,
                __import__("apeGmsh.results.readers._protocol",
                           fromlist=["ResultLevel"]).ResultLevel.LAYERS,
            )
            assert sorted(avail) == [
                f"fiber_stress_{k}" for k in range(5)
            ]
            slab0 = s.elements.layers.get(component="fiber_stress_0")
            slab4 = s.elements.layers.get(component="fiber_stress_4")
            # 1 element × 4 gp × 2 layers = 8 cells.
            assert slab0.values.shape == (1, 8)
            assert slab4.values.shape == (1, 8)
            # Component-0 is the 1st entry of the 5-vector at each cell.
            # Component-4 is the 5th.
            assert slab0.values[0, 0] == 1.0
            assert slab4.values[0, 0] == 5.0


# =====================================================================
# Skipped: missing metadata for an element
# =====================================================================

class TestSkippedNoSection:
    def test_element_without_section_in_metadata_is_skipped(
        self, tmp_path: Path,
    ) -> None:
        fem = _MockFem([1, 2, 3, 4])
        sec_meta = LayerSectionMetadata(
            sections={
                1: LayerSectionDef(
                    section_tag=1, section_name="S",
                    n_layers=1,
                    thickness=np.array([0.1]),
                    material_tags=np.array([1]),
                ),
            },
            # eid=200 has a section, eid=201 doesn't.
            element_to_section={200: 1},
        )
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="layers", name="r",
                components=("fiber_stress",),
                dt=None, n_steps=None,
                element_ids=np.array([200, 201]),
                layer_section_metadata=sec_meta,
            ),
            snapshot_id=fem.snapshot_id,
        )

        ops = _FakeOpsLayers()
        ops.ele_class[200] = "ASDShellQ4"
        ops.ele_class[201] = "ASDShellQ4"
        ops.ele_nodes[200] = [1, 2, 3, 4]
        ops.node_coords[1] = np.array([0.0, 0.0, 0.0])
        ops.node_coords[2] = np.array([1.0, 0.0, 0.0])
        ops.node_coords[3] = np.array([1.0, 1.0, 0.0])
        ops.node_coords[4] = np.array([0.0, 1.0, 0.0])
        ops.layer_response[(200, 1, 1, "stress")] = np.array([1.0])
        ops.layer_response[(200, 2, 1, "stress")] = np.array([2.0])
        ops.layer_response[(200, 3, 1, "stress")] = np.array([3.0])
        ops.layer_response[(200, 4, 1, "stress")] = np.array([4.0])

        path = tmp_path / "skip.h5"
        with DomainCapture(spec, path, fem, ndm=3, ndf=6, ops=ops) as cap:
            cap.begin_stage("g", kind="static")
            cap.step(t=0.0)
            cap.end_stage()

            assert len(cap._layer_capturers) == 1
            skipped = cap._layer_capturers[0].skipped_elements
            assert len(skipped) == 1
            assert skipped[0][0] == 201
            assert "no LayeredShell" in skipped[0][1]


# =====================================================================
# Validation
# =====================================================================

class TestValidation:
    def test_no_recognised_components_raises(self, tmp_path: Path) -> None:
        fem = _MockFem([1])
        sec_meta = LayerSectionMetadata(
            sections={
                1: LayerSectionDef(
                    section_tag=1, section_name="S", n_layers=1,
                    thickness=np.array([0.1]),
                    material_tags=np.array([1]),
                ),
            },
            element_to_section={1: 1},
        )
        spec = _spec_with(
            ResolvedRecorderRecord(
                category="layers", name="bad",
                components=("displacement_x",),
                dt=None, n_steps=None,
                element_ids=np.array([1]),
                layer_section_metadata=sec_meta,
            ),
            snapshot_id=fem.snapshot_id,
        )
        ops = _FakeOpsLayers()
        with pytest.raises(ValueError, match="recognised components"):
            with DomainCapture(spec, tmp_path / "x.h5", fem, ops=ops):
                pass
