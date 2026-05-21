"""Integration tests for ``apeSees.h5(path, cuts=, sweeps=)`` (v4-2).

Drives the full bridge build pipeline (per the existing end-to-end
smoke test) and confirms the new ``cuts=`` / ``sweeps=`` kwargs land
their content under ``/opensees/cuts/`` and ``/opensees/sweeps/`` —
read back identically through
:func:`apeGmsh.cuts._h5_io.read_cuts_and_sweeps`.

Also pins the schema-version bump (``2.4.0 → 2.5.0``): the producer
must stamp ``2.5.0`` regardless of whether cuts were attached, since
v4 changes the *file shape*, not the data.  (2.3.0 = Phase 9 commit 6
/ unified recorders; 2.4.0 = Phase 8.7 commit 2 / `/mesh_selections/`.)
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py

from apeGmsh.cuts import SectionCutDef, SectionSweepDef
from apeGmsh.cuts._h5_io import read_cuts_and_sweeps
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.section.fiber import FiberPoint

from tests.opensees.fixtures.fem_stub import make_two_node_beam


# --------------------------------------------------------------------- #
# Shared minimal apeSees builder — mirrors test_h5_end_to_end.py
# --------------------------------------------------------------------- #
def _build_minimal_ops() -> apeSees:
    """Build a minimal force-beam model.

    Identical shape to the existing end-to-end smoke test so the
    fixture stays small and the new ``cuts=`` / ``sweeps=`` wiring
    is what's actually under test.
    """
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
    return ops


# --------------------------------------------------------------------- #
# Schema bump
# --------------------------------------------------------------------- #
def test_apesees_h5_bumps_schema_version_to_2_5_0(tmp_path: Path) -> None:
    """v4 file shape requires the current opensees schema version stamp.

    The version reflects the schema the file is written *against*, not
    whether v4 content is actually present.  Phase 6 (ADR 0021) bumped
    to 2.6.0 for the additive ``/meta/lineage`` sub-group; Phase 7b
    (ADR 0022) bumped to 2.7.0 for the additive
    ``/opensees/constraints/`` group.
    """
    ops = _build_minimal_ops()
    out = tmp_path / "model.h5"
    ops.h5(str(out))

    with h5py.File(out, "r") as f:
        assert f["meta"].attrs["schema_version"] == "2.7.0"


# --------------------------------------------------------------------- #
# No-cuts path: backward-compatible
# --------------------------------------------------------------------- #
def test_apesees_h5_without_cuts_omits_groups(tmp_path: Path) -> None:
    """No ``cuts=`` / ``sweeps=`` kwarg → groups absent.

    The cuts writer is a no-op on empty input — file shape matches
    pre-v4 behaviour aside from the schema version stamp.
    """
    ops = _build_minimal_ops()
    out = tmp_path / "model.h5"
    ops.h5(str(out))

    with h5py.File(out, "r") as f:
        assert "opensees/cuts" not in f
        assert "opensees/sweeps" not in f


# --------------------------------------------------------------------- #
# Cuts kwarg threads through
# --------------------------------------------------------------------- #
def test_apesees_h5_with_cuts_writes_groups(tmp_path: Path) -> None:
    """``cuts=[cut, ...]`` lands at ``/opensees/cuts/cut_N``."""
    cut0 = SectionCutDef(
        plane_point=(0.0, 0.0, 0.5),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="cut zero",
    )
    cut1 = SectionCutDef(
        plane_point=(0.0, 0.0, 1.5),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        side="negative",
        label="cut one",
        bounding_polygon=(
            (-1.0, -1.0, 1.5),
            ( 1.0, -1.0, 1.5),
            ( 1.0,  1.0, 1.5),
            (-1.0,  1.0, 1.5),
        ),
    )
    ops = _build_minimal_ops()
    out = tmp_path / "model.h5"
    ops.h5(str(out), cuts=[cut0, cut1])

    cuts_out, sweeps_out = read_cuts_and_sweeps(out)
    assert sweeps_out == ()
    assert len(cuts_out) == 2
    assert cuts_out[0] == cut0
    assert cuts_out[1] == cut1


def test_apesees_h5_with_sweep_writes_groups(tmp_path: Path) -> None:
    """``sweeps=[sweep]`` lands at ``/opensees/sweeps/sweep_N``."""
    cut_a = SectionCutDef(
        plane_point=(0.0, 0.0, 0.25),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="story 1",
    )
    cut_b = SectionCutDef(
        plane_point=(0.0, 0.0, 0.75),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="story 2",
    )
    sweep = SectionSweepDef(cuts=(cut_a, cut_b))
    ops = _build_minimal_ops()
    out = tmp_path / "model.h5"
    ops.h5(str(out), sweeps=[sweep])

    cuts_out, sweeps_out = read_cuts_and_sweeps(out)
    assert cuts_out == ()
    assert len(sweeps_out) == 1
    restored = sweeps_out[0]
    assert len(restored) == 2
    assert restored[0] == cut_a
    assert restored[1] == cut_b


# --------------------------------------------------------------------- #
# File still validates through the reference reader
# --------------------------------------------------------------------- #
def test_apesees_h5_with_cuts_passes_reference_reader(
    tmp_path: Path,
) -> None:
    """The v4 file must still validate through h5_reader.open + .validate().

    Guards against the cuts append accidentally violating an existing
    schema invariant (material naming, section refs, pattern refs).
    """
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.5),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="reader test",
    )
    ops = _build_minimal_ops()
    out = tmp_path / "model.h5"
    ops.h5(str(out), cuts=[cut])

    with h5_reader.open(str(out)) as model:
        assert model.schema_version == "2.7.0"
        violations = model.validate()
        assert violations == [], violations
