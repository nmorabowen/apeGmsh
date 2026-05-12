"""H5 fixture builders for the viewer-integration test bed.

Each builder constructs an :class:`H5Emitter`, drives it as if the
bridge had emitted a small representative model, and returns the
populated emitter. The companion test module ``test_fixtures`` writes
each fixture to a ``tmp_path`` and asserts the viewer-integration
contract (panels populated, counts match the sibling JSON).

Builders are also exported as importable functions so the viewer team
can call them in their own integration test bed without re-implementing
the construction logic.

Rationale for on-demand builds (rather than checked-in ``.h5`` blobs):
HDF5 files are binary and version-coupled to h5py; a repo-checked-in
blob would drift the moment we bump the schema. Building on demand
keeps the fixtures aligned with the current H5 emitter.
"""
from __future__ import annotations

from typing import Any, Callable

from apeGmsh.opensees._internal.tag_resolution import set_element_nodes
from apeGmsh.opensees.emitter.h5 import H5Emitter


__all__ = [
    "FIXTURE_BUILDERS",
    "FIXTURE_EXPECTATIONS",
    "build_arch_csys",
    "build_dome_spherical",
    "build_frame_3d",
    "build_incomplete",
    "build_minimal",
    "build_tank_cylindrical",
    "build_wrong_major",
]


def build_minimal() -> H5Emitter:
    """One column, one fiber section, one ground motion, no analysis."""
    e = H5Emitter(model_name="minimal_column")
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial(
        "Steel02", 1, 420.0e6, 200.0e9, 0.01, 20.0, 0.925, 0.15,
    )
    e.uniaxialMaterial(
        "Concrete02", 2, -30.0e6, -0.002, -25.0e6, -0.006,
        0.1, 2.5e6, 200.0e6,
    )
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.patch("rect", 2, 8, 8, -0.2, -0.2, 0.2, 0.2)
    for _ in range(8):
        e.fiber(0.18, 0.0, 0.001, 1)
    e.section_close()
    e.geomTransf("PDelta", 1, 1.0, 0.0, 0.0)
    e.beamIntegration("Lobatto", 1, 1, 5)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 1, 1, 2, 1, 1)
    e.timeSeries("Path", 1, "-dt", 0.01, "-filePath", "elcentro.txt")
    e.pattern_open("UniformExcitation", 1, 1, "-accel", 1)
    e.pattern_close()
    e.fix(1, 1, 1, 1, 1, 1, 1)
    return e


def build_frame_3d() -> H5Emitter:
    """Two-column / one-beam moment frame with two PGs and two patterns."""
    e = H5Emitter(model_name="frame_3d")
    e.model(ndm=3, ndf=6)
    # Materials: one elastic, one fiber-stack.
    e.uniaxialMaterial("Elastic", 1, 200.0e9)
    e.uniaxialMaterial("Steel02", 2, 420.0e6, 200.0e9, 0.01)
    # Beam section: simple elastic.
    e.section("Elastic", 1, 200.0e9, 0.01, 1.0e-4, 1.0e-4, 80.0e9, 5.0e-5)
    # Column section: fiber.
    e.section_open("Fiber", 2, "-GJ", 1.0e9)
    e.patch("rect", 1, 4, 4, -0.15, -0.15, 0.15, 0.15)
    e.section_close()
    e.geomTransf("PDelta", 1, 1.0, 0.0, 0.0)
    e.geomTransf("PDelta", 2, 0.0, 1.0, 0.0)
    e.beamIntegration("Lobatto", 1, 1, 5)
    e.beamIntegration("Lobatto", 2, 2, 3)
    # Beam elements (1-2 and 3-4).
    for tag, conn in ((1, (1, 2)), (2, (3, 4))):
        set_element_nodes(e, conn)
        e.element("forceBeamColumn", tag, *conn, 1, 1)
    # Column elements (2-3 and 4-5).
    for tag, conn in ((3, (2, 3)), (4, (4, 5))):
        set_element_nodes(e, conn)
        e.element("forceBeamColumn", tag, *conn, 2, 2)
    # Two patterns: gravity + lateral.
    e.timeSeries("Linear", 1, "-factor", 1.0)
    e.timeSeries("Linear", 2, "-factor", 1.0)
    e.pattern_open("Plain", 1, 1)
    e.load(2, 0.0, 0.0, -1.0e3, 0.0, 0.0, 0.0)
    e.load(4, 0.0, 0.0, -1.0e3, 0.0, 0.0, 0.0)
    e.pattern_close()
    e.pattern_open("Plain", 2, 2)
    e.load(3, 5.0e3, 0.0, 0.0, 0.0, 0.0, 0.0)
    e.pattern_close()
    e.fix(1, 1, 1, 1, 1, 1, 1)
    e.fix(5, 1, 1, 1, 1, 1, 1)
    e.mass(2, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0)
    e.mass(3, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0)
    # Recorder.
    e.recorder(
        "Node", "-file", "disp.out", "-node", 2, 3, "-dof", 1, "disp",
    )
    return e


def _build_csys_arch_like(model_name: str, n_ribs: int) -> H5Emitter:
    """Shared backbone for csys-bearing fixtures.

    Generates ``n_ribs`` ribs, each with a slightly different vecxz —
    emitting one ``geomTransf`` per distinct vecxz (post-fan-out shape
    the bridge would produce for cylindrical / spherical / arch CSs).
    """
    import math
    e = H5Emitter(model_name=model_name)
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial("Elastic", 1, 200.0e9)
    e.section("Elastic", 1, 200.0e9, 0.01, 1.0e-4, 1.0e-4, 80.0e9, 5.0e-5)
    e.beamIntegration("Lobatto", 1, 1, 3)
    # One geomTransf line per rib — distinct vecxz, sharing the same
    # type token. This mirrors the bridge's csys-driven fan-out.
    for i in range(n_ribs):
        theta = 2.0 * math.pi * i / n_ribs
        vx = math.cos(theta)
        vy = math.sin(theta)
        e.geomTransf("PDelta", i + 1, vx, vy, 0.0)
    # One rib element per geomTransf tag.
    for i in range(n_ribs):
        set_element_nodes(e, (i + 1, i + 2))
        e.element(
            "forceBeamColumn", i + 1,
            i + 1, i + 2,
            i + 1,  # transf tag
            1,      # integration tag
        )
    e.fix(1, 1, 1, 1, 1, 1, 1)
    return e


def build_arch_csys() -> H5Emitter:
    """Arch with cylindrical CS — six distinct vecxz."""
    return _build_csys_arch_like("arch_cylindrical", n_ribs=6)


def build_dome_spherical() -> H5Emitter:
    """Dome with spherical CS — eight distinct vecxz."""
    return _build_csys_arch_like("dome_spherical", n_ribs=8)


def build_tank_cylindrical() -> H5Emitter:
    """Tank with ring beams + vertical stiffeners.

    Two element groups (one beam type), three transforms (ring +
    two stiffener orientations).
    """
    e = H5Emitter(model_name="tank_cylindrical")
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial("Elastic", 1, 200.0e9)
    e.section("Elastic", 1, 200.0e9, 0.01, 1.0e-4, 1.0e-4, 80.0e9, 5.0e-5)
    e.beamIntegration("Lobatto", 1, 1, 3)
    # Ring beam transforms (radial vecxz).
    e.geomTransf("PDelta", 1, 1.0, 0.0, 0.0)
    e.geomTransf("PDelta", 2, 0.0, 1.0, 0.0)
    # Vertical stiffener transform (tangential vecxz).
    e.geomTransf("Linear", 3, 0.0, 0.0, 1.0)
    # Ring beam elements.
    for i, conn in enumerate([(1, 2), (2, 3), (3, 4)]):
        set_element_nodes(e, conn)
        e.element("forceBeamColumn", i + 1, *conn, 1, 1)
    # Stiffener elements.
    for i, conn in enumerate([(5, 6), (7, 8)]):
        set_element_nodes(e, conn)
        e.element("forceBeamColumn", 100 + i, *conn, 3, 1)
    e.fix(1, 1, 1, 1, 1, 1, 1)
    return e


def build_incomplete() -> H5Emitter:
    """``/meta`` + ``/elements`` only — viewer hides every enrichment panel."""
    e = H5Emitter(model_name="incomplete")
    e.model(ndm=3, ndf=6)
    set_element_nodes(e, (1, 2, 3, 4))
    e.element("FourNodeTetrahedron", 1, 1, 2, 3, 4, 1)
    return e


def build_wrong_major() -> H5Emitter:
    """Schema major v3 — the reference reader MUST refuse this file.

    Phase 8.4 made `2.x.y` the accepted major; this fixture probes the
    forward-major-mismatch path by stamping a v3 schema.
    """
    return H5Emitter(schema_version="3.0.0", model_name="wrong_major")


#: Fixture name → builder function. The test bed iterates this map.
FIXTURE_BUILDERS: dict[str, Callable[[], H5Emitter]] = {
    "minimal": build_minimal,
    "frame_3d": build_frame_3d,
    "arch_csys": build_arch_csys,
    "dome_spherical": build_dome_spherical,
    "tank_cylindrical": build_tank_cylindrical,
    "incomplete": build_incomplete,
    "wrong_major": build_wrong_major,
}


#: For each fixture, the expected viewer-integration outcomes (panels
#: populated, schema groups present, counts). Used as a contract test;
#: the viewer team treats this as their integration spec.
#:
#: Group paths reflect the Phase 8.4 zone reshuffle: `/meta` and
#: `/elements` are root-level; everything else the bridge writes lives
#: under `/opensees/`.
FIXTURE_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "minimal": {
        "should_validate": True,
        "expected_groups": [
            "meta", "opensees/materials/uniaxial", "opensees/sections",
            "opensees/transforms", "opensees/beam_integration",
            "elements", "opensees/time_series", "opensees/patterns",
            "opensees/bcs",
        ],
        "expected_absent_groups": [
            "opensees/analysis", "opensees/recorders",
        ],
        "material_count_uniaxial": 2,
        "section_count": 1,
        "element_type_count": 1,
        "pattern_count": 1,
    },
    "frame_3d": {
        "should_validate": True,
        "expected_groups": [
            "meta", "opensees/materials/uniaxial", "opensees/sections",
            "opensees/transforms", "opensees/beam_integration",
            "elements", "opensees/time_series", "opensees/patterns",
            "opensees/bcs", "opensees/recorders",
        ],
        "material_count_uniaxial": 2,
        "section_count": 2,
        "element_type_count": 1,
        "pattern_count": 2,
        "recorder_count": 1,
    },
    "arch_csys": {
        "should_validate": True,
        "expected_groups": [
            "meta", "opensees/transforms", "elements",
            "opensees/beam_integration",
        ],
        "transform_count": 6,
        "element_type_count": 1,
    },
    "dome_spherical": {
        "should_validate": True,
        "expected_groups": [
            "meta", "opensees/transforms", "elements",
            "opensees/beam_integration",
        ],
        "transform_count": 8,
        "element_type_count": 1,
    },
    "tank_cylindrical": {
        "should_validate": True,
        "expected_groups": [
            "meta", "opensees/transforms", "elements",
            "opensees/beam_integration",
        ],
        "transform_count": 3,
        "element_type_count": 1,
    },
    "incomplete": {
        "should_validate": True,
        "expected_groups": ["meta", "elements"],
        # No bridge content beyond /meta + /elements → /opensees is
        # never created.  One absent assertion covers every relocated
        # group at once.
        "expected_absent_groups": ["opensees"],
    },
    "wrong_major": {
        "should_open": False,
        "schema_version_starts_with": "3.",
    },
}
