"""H5 round-trip for tagged ``damping`` objects (ADR 0053, D3b / schema 2.15.0).

D3a left ``H5Emitter.damping`` a no-op, so a model.h5 round-trip silently
dropped its damping objects. D3b persists them under ``/opensees/dampings/``
and replays them through ``OpenSeesModel._replay_into`` — an element-flag
``-damp`` attachment rides along because it lives in the element's own arg
tail. These tests cover the write, the read-back accessor, the re-emit
round-trip, and the lineage-hash fold.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION

from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


def _build_with_element_damp() -> apeSees:
    """One-column model whose beam carries an element-flag Uniform damping."""
    fem = build_simple_frame_fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    damp = ops.damping.uniform(
        ratio=0.03, freq_lower=0.5, freq_upper=10.0, name="soil_damp",
    )
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
        damp=damp,
    )
    return ops


def test_schema_stamped_2_15_0(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    _build_with_element_damp().h5(str(p))
    with h5py.File(str(p), "r") as f:
        assert f["meta"].attrs["opensees_schema_version"] == SCHEMA_VERSION
        assert SCHEMA_VERSION == "2.15.0"


def test_h5_writes_dampings_group(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    _build_with_element_damp().h5(str(p))
    with h5py.File(str(p), "r") as f:
        dampings = f["opensees"]["dampings"]
        (name,) = list(dampings)
        g = dampings[name]
        assert g.attrs["type"] == "Uniform"
        # params is a float64 attr (positional *args); zeta freq1 freq2 —
        # zeta is the physical ratio (OpenSees doubles internally).
        assert list(g.attrs["params"]) == [0.03, 0.5, 10.0]


def test_no_damping_writes_no_group(tmp_path: Path) -> None:
    fem = build_simple_frame_fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    p = tmp_path / "model.h5"
    ops.h5(str(p))
    with h5py.File(str(p), "r") as f:
        assert "dampings" not in f["opensees"]


def test_from_h5_reads_dampings_record(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    _build_with_element_damp().h5(str(p))
    om = OpenSeesModel.from_h5(str(p))
    (rec,) = om.dampings()
    assert rec.type_token == "Uniform"
    assert [float(x) for x in rec.args] == [0.03, 0.5, 10.0]


def test_round_trip_reemits_object_and_element_attach(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    _build_with_element_damp().h5(str(p))
    om = OpenSeesModel.from_h5(str(p))
    tcl = om.build("tcl")
    lines = [ln.strip() for ln in tcl.splitlines()]
    damp_line = next(ln for ln in lines if ln.startswith("damping Uniform"))
    damp_tag = damp_line.split()[2]
    # the object re-emits before the element, and the element keeps -damp
    ele_line = next(
        ln for ln in lines
        if ln.startswith("element elasticBeamColumn") and "-damp" in ln
    )
    assert ele_line.split()[ele_line.split().index("-damp") + 1] == damp_tag
    assert lines.index(damp_line) < lines.index(ele_line)


def test_dampings_fold_into_model_hash(tmp_path: Path) -> None:
    # Two models identical but for the damping object must hash differently
    # (damping is authored state, not a regenerable carve-out like regions).
    from apeGmsh.opensees._internal import lineage as _lin

    p_damp = tmp_path / "with.h5"
    _build_with_element_damp().h5(str(p_damp))

    fem = build_simple_frame_fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    p_plain = tmp_path / "without.h5"
    ops.h5(str(p_plain))

    with h5py.File(str(p_damp), "r") as f:
        h_damp = _lin.compute_model_hash("", f["opensees"])
    with h5py.File(str(p_plain), "r") as f:
        h_plain = _lin.compute_model_hash("", f["opensees"])
    assert h_damp != h_plain
