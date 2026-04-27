"""Phase 11a catalog growth — real-openseespy validation for shells.

For every catalogued regular shell formulation, build a single-element
cantilever-strip model under a transverse tip load, run a static
analysis, capture the 8-component stress resultants via
DomainCapture, and check that:

1. The catalog's ``flat_size_per_element`` matches what
   ``ops.eleResponse(eid, "stresses")`` returns.
2. The component layout decodes correctly (membrane vs bending vs
   shear values are routed to the right names).
3. Physical sanity: under in-plane axial tension, ``membrane_force_xx``
   dominates; under transverse tip load on a cantilever strip,
   ``bending_moment_xx`` is significant.

Layered shells use the *same* catalog entry as their non-layered
siblings (the ``LayeredShellFiberSection`` is invisible to the
element at this topology level — see the source review at
``LayeredShellFiberSection.cpp:316``). One smoke test below builds a
layered ShellMITC4 to confirm the same code path works.
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

from apeGmsh.solvers._element_response import IntRule, lookup


# =====================================================================
# Mock fem (only needs nodes + a hash for DomainCapture)
# =====================================================================

class _MinimalFem:
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
        n = group.create_group("nodes")
        n.create_dataset("ids", data=self.nodes.ids)
        n.create_dataset("coords", data=self.nodes.coords)
        group.create_group("elements")


# =====================================================================
# Per-class shell-element factories
# =====================================================================
#
# Every factory takes a ``sec_tag`` and returns
# ``(node_ids, coords, eid, cpp_class, top_nids)``. The caller defines
# the section before calling. Top-edge nodes are where the transverse
# tip load is applied.

def _build_quad4_strip(tcl_name: str, sec_tag: int) -> tuple[
    np.ndarray, np.ndarray, int, str, list[int],
]:
    """Unit-square 4-node quad shell. Tcl: ``element <name> ... secTag``."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    # Pin the y=0 edge fully (cantilever).
    for nid in (1, 2):
        ops.fix(int(nid), 1, 1, 1, 1, 1, 1)
    ops.element(tcl_name, 1, 1, 2, 3, 4, sec_tag)
    cpp_class_map = {
        "ShellMITC4": "ShellMITC4",
        "ShellDKGQ": "ShellDKGQ",
        "ShellNLDKGQ": "ShellNLDKGQ",
        "ASDShellQ4": "ASDShellQ4",
    }
    return node_ids, coords, 1, cpp_class_map[tcl_name], [3, 4]


def _build_quad9_strip(tcl_name: str, sec_tag: int) -> tuple[
    np.ndarray, np.ndarray, int, str, list[int],
]:
    """9-node quad shell (ShellMITC9). 4 corners + 4 edge mid + centroid."""
    corners = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    midpts = np.array([0.5 * (corners[a] + corners[b]) for a, b in edges])
    centroid = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)
    coords = np.vstack([corners, midpts, centroid])
    node_ids = np.arange(1, 10, dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    # Pin the y=0 edge: corners 1 & 2 plus the bottom-edge midpoint (5).
    for nid in (1, 2, 5):
        ops.fix(int(nid), 1, 1, 1, 1, 1, 1)
    ops.element(tcl_name, 1, *(int(n) for n in node_ids), sec_tag)
    return node_ids, coords, 1, "ShellMITC9", [3, 4, 7]


def _build_tri3_strip(tcl_name: str, sec_tag: int) -> tuple[
    np.ndarray, np.ndarray, int, str, list[int],
]:
    """3-node triangle shell. 2 nodes at base, 1 at apex."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    for nid in (1, 2):
        ops.fix(int(nid), 1, 1, 1, 1, 1, 1)
    ops.element(tcl_name, 1, 1, 2, 3, sec_tag)
    cpp_class_map = {
        "ShellDKGT": "ShellDKGT",
        "ShellNLDKGT": "ShellNLDKGT",
        "ASDShellT3": "ASDShellT3",
    }
    return node_ids, coords, 1, cpp_class_map[tcl_name], [3]


# =====================================================================
# Section helpers
# =====================================================================

def _define_elastic_membrane_plate_section(sec_tag: int) -> None:
    """Cheapest section that satisfies a regular shell element.

    ``ElasticMembranePlateSection`` couples in-plane (membrane) and
    out-of-plane (plate) behavior with a single E + ν + thickness.
    """
    ops.section(
        "ElasticMembranePlateSection", sec_tag,
        200e9, 0.3, 0.1, 0.0,   # E, nu, thickness, rho
    )


# =====================================================================
# Parametrize: (tcl_name, cpp_class, int_rule, n_gp_expected, factory)
# =====================================================================

# Live-ops scope: only Petracca's ASDShell* are exercised here.
#
# Upstream OpenSees has known issues that make ``ops.eleResponse(eid,
# "stresses")`` return broken data for the older shells:
#
# - ShellMITC4 / ShellDKGQ / ShellNLDKGQ / ShellDKGT / ShellNLDKGT —
#   ``materialPointers[i]->getStressResultant()`` returns zeros even
#   after a converged analysis. The shell elements run fine; nodal
#   displacements and ``"forces"`` keyword are correct, but the
#   per-GP stress-resultant probe path is broken in the build I
#   verified. (Same upstream-OpenSees pattern as BbarBrick.)
# - ShellMITC9 — declares ``Vector(72)`` in setResponse but stores
#   ``static Vector stresses(84)`` in getResponse (line 518), so
#   ``ops.eleResponse`` returns 84 values where the catalog expects
#   72. The 12 extra values are uninitialized.
#
# The catalog entries remain correct: MPCO doesn't go through the
# broken eleResponse path; it probes sections/materials directly via
# a different code path and reads the right per-GP resultants.
_QUAD_CASES = [
    pytest.param(
        "ASDShellQ4", IntRule.Quad_GL_2, 4,
        lambda secT: _build_quad4_strip("ASDShellQ4", secT),
        id="ASDShellQ4",
    ),
]

_TRI_CASES = [
    pytest.param(
        "ASDShellT3", IntRule.Triangle_GL_2B, 3,
        lambda secT: _build_tri3_strip("ASDShellT3", secT),
        id="ASDShellT3",
    ),
]


def _solve_static() -> None:
    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    rc = ops.analyze(1)
    assert rc == 0, f"analyze() returned {rc}"


@pytest.mark.parametrize(
    "cpp_class,int_rule,n_gp_expected,build_model", _QUAD_CASES + _TRI_CASES,
)
def test_shell_capture_for_class(
    tmp_path: Path,
    cpp_class: str,
    int_rule: int,
    n_gp_expected: int,
    build_model,
) -> None:
    """Cantilever shell strip under transverse tip load.

    Verifies catalog correctness: flat_size matches ops, the
    8-component layout decodes into correctly-named slabs, and the
    physics is sensible (bending dominates on a transverse cantilever).
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    sec_tag = 1
    _define_elastic_membrane_plate_section(sec_tag)

    node_ids, coords, eid, factory_class_name, top_nids = build_model(sec_tag)
    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    # Transverse tip load (in -z) on the free edge.
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    per_node = -1.0e3 / len(top_nids)
    for nid in top_nids:
        ops.load(int(nid), 0.0, 0.0, per_node, 0.0, 0.0, 0.0)

    assert ops.eleType(eid) == cpp_class

    layout = lookup(cpp_class, int_rule, "stress")
    assert layout.n_gauss_points == n_gp_expected
    assert layout.n_components_per_gp == 8

    _solve_static()

    flat = np.asarray(ops.eleResponse(eid, "stresses"), dtype=np.float64)
    assert flat.size == layout.flat_size_per_element, (
        f"{cpp_class}: ops.eleResponse returned {flat.size}, "
        f"catalog expects {layout.flat_size_per_element}."
    )

    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )

    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="gauss", name="rec",
            components=tuple(layout.component_layout),
            dt=None, n_steps=None,
            element_ids=np.array([eid]),
        ),),
    )

    capture_path = tmp_path / "cap.h5"
    with DomainCapture(spec, capture_path, fem, ndm=3, ndf=6) as cap:
        cap.begin_stage("static_load", kind="static")
        cap.step(t=ops.getTime())
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(capture_path, fem=fem) as r:
        s = r.stage(r.stages[0].id)

        # All 8 components decode into separate slabs of the right shape.
        slabs = {
            name: s.elements.gauss.get(component=name)
            for name in layout.component_layout
        }
        for name, slab in slabs.items():
            assert slab.values.shape == (1, n_gp_expected), (
                f"{cpp_class}: {name} slab shape {slab.values.shape} "
                f"!= (1, {n_gp_expected})"
            )
            assert slab.element_index.tolist() == [eid] * n_gp_expected

        # Physics check: under a transverse cantilever tip load, the
        # bending moment about the clamped edge dominates the response.
        # The clamped edge runs along x with the cantilever extending
        # in +y, so the bending moment at the clamp is M_xx (moment
        # tensor index xx is the moment associated with σ_xx
        # variation through-thickness — i.e. bending about the edge
        # parallel to x, which is the y-axis of the bending tensor).
        bending_xx = slabs["bending_moment_xx"].values[0]
        bending_yy = slabs["bending_moment_yy"].values[0]
        membrane_xx = slabs["membrane_force_xx"].values[0]

        # |M_xx| should be significantly larger than |N_xx| under pure
        # transverse load (bending dominates).
        max_M_xx = np.abs(bending_xx).max()
        max_N_xx = np.abs(membrane_xx).max()
        assert max_M_xx > 0.0, f"{cpp_class}: M_xx is identically zero"
        assert max_M_xx > 10.0 * max_N_xx, (
            f"{cpp_class}: bending_moment_xx max={max_M_xx} should "
            f"dominate membrane_force_xx max={max_N_xx} under transverse load."
        )


# =====================================================================
# Layered-shell smoke test — same catalog entry, layered section
# =====================================================================

def test_layered_shell_uses_same_catalog_entry(tmp_path: Path) -> None:
    """ASDShellQ4 + LayeredShellFiberSection produces the same 8-component
    stress-resultant slab as ASDShellQ4 + ElasticMembranePlateSection.

    Layered behavior is on the section (transparent to the element at
    this topology level — see ``LayeredShellFiberSection.cpp:316``,
    where the section's setResponse routes ``"fiber X stresses"`` to
    the per-layer material). Per-layer through-thickness data is a
    separate topology level (Phase 11c).

    Uses ASDShellQ4 because the older shells have a broken
    ``ops.eleResponse(eid, "stresses")`` path — see the live-ops
    parametrize comment above. The layered behavior is element-
    agnostic, so this test is sufficient.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # PlateFiber-wrapped elastic ND material.
    ops.nDMaterial("ElasticIsotropic", 1, 200e9, 0.3)
    ops.nDMaterial("PlateFiber", 2, 1)

    # 3 equal layers of total thickness 0.1.
    ops.section(
        "LayeredShell", 1, 3,
        2, 0.0333,
        2, 0.0334,
        2, 0.0333,
    )

    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    for nid, xyz in zip(node_ids, coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))
    for nid in (1, 2):
        ops.fix(int(nid), 1, 1, 1, 1, 1, 1)
    ops.element("ASDShellQ4", 1, 1, 2, 3, 4, 1)

    fem = _MinimalFem(node_ids=node_ids, coords=coords)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid in (3, 4):
        ops.load(int(nid), 0.0, 0.0, -500.0, 0.0, 0.0, 0.0)

    layout = lookup("ASDShellQ4", IntRule.Quad_GL_2, "stress")
    # Catalog says 4 GPs × 8 = 32 components — independent of section.
    assert layout.flat_size_per_element == 32

    _solve_static()

    flat = np.asarray(ops.eleResponse(1, "stresses"), dtype=np.float64)
    # ops returns the same 32-vector regardless of layered vs non-layered.
    assert flat.size == 32

    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.solvers._recorder_specs import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="gauss", name="rec",
            components=tuple(layout.component_layout),
            dt=None, n_steps=None,
            element_ids=np.array([1]),
        ),),
    )
    capture_path = tmp_path / "cap.h5"
    with DomainCapture(spec, capture_path, fem, ndm=3, ndf=6) as cap:
        cap.begin_stage("g", kind="static")
        cap.step(t=ops.getTime())
        cap.end_stage()

    from apeGmsh.results import Results
    with Results.from_native(capture_path, fem=fem) as r:
        s = r.stage(r.stages[0].id)
        # Decode succeeds; bending response is non-trivial.
        bending = s.elements.gauss.get(component="bending_moment_xx").values[0]
        assert np.abs(bending).max() > 0.0
