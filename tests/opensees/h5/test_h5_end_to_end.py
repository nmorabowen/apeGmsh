"""End-to-end smoke test for ``apeSees.h5(path)``.

Drives the full bridge build pipeline through the
:class:`H5Emitter` and verifies the resulting HDF5 file is valid
per :func:`apeGmsh.opensees.emitter.h5_reader.open` /
:meth:`H5Model.validate`.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.section.fiber import FiberPoint

from tests.opensees.fixtures.fem_stub import make_two_node_beam


def test_apesees_h5_writes_valid_file(tmp_path: Path) -> None:
    """Build a minimal force-beam model end-to-end and confirm the file
    opens through the reference reader and validates clean."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )

    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(100e3, 0.0, 0.0))

    out = tmp_path / "smoke.h5"
    ops.h5(str(out))
    assert out.exists() and out.stat().st_size > 0

    with h5_reader.open(str(out)) as model:
        violations = model.validate()
        assert violations == [], violations
        # Required groups must be present.
        assert "meta" in model.handle
        # Phase 8.5: /elements is broker-owned and only emitted when
        # apeSees.h5 sees a real FEMData.  This test uses a hand-rolled
        # FEMStub (no `snapshot_id`, no PhysicalGroupSet), so the
        # broker zone is skipped — the file is bridge-only.
        # Material + section + transform + beamIntegration round-tripped.
        mats = model.materials()
        assert "uniaxial" in mats
        assert any(
            attrs.get("type") == "Steel02"
            for attrs in mats["uniaxial"].values()
        )
        sections = model.sections()
        assert sections, "no /sections group emitted"
        assert any(
            attrs.get("type") == "Fiber" for attrs in sections.values()
        )
        transforms = model.transforms()
        assert transforms
        # Pattern with series_ref resolved.
        patterns = model.patterns()
        assert patterns
        first_pattern = next(iter(patterns.values()))
        assert first_pattern.get("series_ref", "").startswith("/opensees/time_series/")
        # /meta/snapshot_id is always present (may be empty for stub
        # FEM snapshots that don't compute one).
        meta = model.meta()
        assert "snapshot_id" in meta
        # Phase 8.6: bridge fan-out captured the FEM element id (=1 for
        # the FEMStub's two-node beam) into
        # /opensees/element_meta/forceBeamColumn/fem_eids.
        em = model.element_meta_arrays("forceBeamColumn")
        assert list(em["fem_eids"]) == [1]
