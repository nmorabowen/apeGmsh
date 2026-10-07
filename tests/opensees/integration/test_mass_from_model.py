"""mass_from_model() — stream per-node masses from the snapshot (ADR 0065 Tier 2).

`ops.mass_from_model()` emits one `mass` line per `fem.nodes.masses` entry
without building a bridge MassRecord per node. These tests pin that it is
BYTE-IDENTICAL to the explicit per-node `ops.mass` loop it replaces, plus the
the overlap fail-loud guard, and the H5 archival emitter's skip-and-mark
contract (ADR 0112 amendment 5).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError


@pytest.fixture(scope="module")
def fem_with_masses():
    with apeGmsh(model_name="mfm", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 10, 10, 10, label="b")
        g.physical.add_volume("b", name="B")
        g.masses.volume("B", density=2400.0)
        g.mesh.sizing.set_global_size(3.0)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    assert len(fem.nodes.masses) > 0
    return fem


def _emit_mass_lines(fem, declare):
    ops = apeSees(fem, _artifacts=False)  # flat bridge; the archive is not this test's subject
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    declare(ops)
    fd, path = tempfile.mkstemp(suffix=".tcl")
    os.close(fd)
    try:
        ops.tcl(path)
        with open(path, encoding="utf-8") as f:
            # strip leading indent so partitioned per-rank blocks (mass lines
            # nested inside ``if {[getPID]==K} {`` ) compare like flat ones.
            return [ln.strip() for ln in f if ln.strip().startswith("mass ")]
    finally:
        os.remove(path)


@pytest.fixture(scope="module")
def fem_with_masses_partitioned():
    with apeGmsh(model_name="mfm_p", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 10, 10, 10, label="b")
        g.physical.add_volume("b", name="B")
        g.masses.volume("B", density=2400.0)
        g.mesh.sizing.set_global_size(2.5)
        g.mesh.generation.generate(dim=3)
        g.mesh.partitioning.partition(4)
        fem = g.mesh.queries.get_fem_data(dim=3)
    assert len(fem.nodes.masses) > 0
    return fem


def test_mass_from_model_byte_identical_to_explicit_loop(fem_with_masses):
    fem = fem_with_masses
    reference = _emit_mass_lines(
        fem,
        lambda ops: [
            ops.mass(nodes=[int(m.node_id)],
                     values=tuple(float(v) for v in m.mass))
            for m in fem.nodes.masses
        ],
    )
    streamed = _emit_mass_lines(fem, lambda ops: ops.mass_from_model())
    assert len(streamed) == len(fem.nodes.masses)
    assert streamed == reference


def test_mass_from_model_byte_identical_partitioned(fem_with_masses_partitioned):
    fem = fem_with_masses_partitioned
    reference = _emit_mass_lines(
        fem,
        lambda ops: [
            ops.mass(nodes=[int(m.node_id)],
                     values=tuple(float(v) for v in m.mass))
            for m in fem.nodes.masses
        ],
    )
    streamed = _emit_mass_lines(fem, lambda ops: ops.mass_from_model())
    # additive under MP → one line per node total (primary-owner gated)
    assert len(streamed) == len(fem.nodes.masses)
    assert streamed == reference


def test_mass_from_model_overlap_with_explicit_raises(fem_with_masses):
    fem = fem_with_masses
    shared = int(next(iter(fem.nodes.masses)).node_id)

    def declare(ops):
        ops.mass_from_model()
        ops.mass(nodes=[shared], values=(1.0, 1.0, 1.0))

    with pytest.raises(BridgeError, match="double-count|one mass channel"):
        _emit_mass_lines(fem, declare)


def test_mass_from_model_h5_emitter_skips_stream_and_marks(fem_with_masses):
    """The H5 archival emitter skips the redundant mass stream and marks the
    archive instead of raising (ADR 0112 amendment 5, #1304). The deck and
    replay equality oracle lives in tests/opensees/h5/test_mass_from_model_h5.py."""
    import h5py

    fem = fem_with_masses
    ops = apeSees(fem, _artifacts=False)  # flat bridge; the archive is not this test's subject
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    ops.mass_from_model()
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        ops.h5(path)
        with h5py.File(path, "r") as f:
            assert int(f["opensees/bcs"].attrs["mass_from_model"]) == 1
            assert "mass" not in f["opensees/bcs"]
    finally:
        os.remove(path)
