"""Integration tests for the ``/opensees/names`` alias sidecar (schema 2.13.0).

A name registered on the bridge (``ops.<family>.<Type>(..., name=...)``)
persists into ``/opensees/names`` so the read side can resolve it back
to ``(kind, tag)``.  Names are labels, not structure: they are excluded
from ``model_hash``, so relabelling never drifts lineage.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal._names_h5 import read_names
from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION

from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


def _build(*, named: bool) -> apeSees:
    """One-column model on a real FEMData; primitives named iff ``named``.

    Real FEMData (not the bridge stub) so ``OpenSeesModel.from_h5`` can
    reload the neutral zone.
    """
    from apeGmsh.opensees.section.fiber import FiberPoint

    fem = build_simple_frame_fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.Steel02(
        fy=420e6, E=200e9, b=0.01, name="rebar" if named else None
    )
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
        name="col_sec" if named else None,
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ts = ops.timeSeries.Linear(name="ramp" if named else None)
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="Cols", forces=(100e3, 0.0, 0.0, 0.0, 0.0, 0.0))
    return ops


# --------------------------------------------------------------------- #
# Persistence + read-back
# --------------------------------------------------------------------- #

def test_h5_writes_names_group(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    _build(named=True).h5(str(p))
    names = read_names(str(p))
    lut = {n: (k, t) for n, k, t in names}
    assert lut["rebar"][0] == "uniaxialMaterial"
    assert lut["ramp"][0] == "timeSeries"
    assert lut["col_sec"][0] == "section"


def test_no_names_writes_no_group(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    _build(named=False).h5(str(p))
    assert read_names(str(p)) == ()
    with h5py.File(str(p), "r") as f:
        assert "names" not in f["opensees"]


def test_schema_stamped_current(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    _build(named=True).h5(str(p))
    with h5py.File(str(p), "r") as f:
        assert f["meta"].attrs["opensees_schema_version"] == SCHEMA_VERSION
        assert SCHEMA_VERSION == "2.15.0"


def test_opensees_model_resolves_names(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    _build(named=True).h5(str(p))
    om = OpenSeesModel.from_h5(str(p))
    kind, tag = om.tag_for_name("rebar")
    assert kind == "uniaxialMaterial"
    assert om.name_for(tag, "uniaxialMaterial") == "rebar"
    assert om.tag_for_name("nonexistent") is None


# --------------------------------------------------------------------- #
# Lineage stability — relabel must not perturb model_hash
# --------------------------------------------------------------------- #

def _model_hash(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return str(f["meta"]["lineage"].attrs["model_hash"])


def test_names_excluded_from_model_hash(tmp_path: Path) -> None:
    named = tmp_path / "named.h5"
    bare = tmp_path / "bare.h5"
    _build(named=True).h5(str(named))
    _build(named=False).h5(str(bare))
    # Same structure, only the alias sidecar differs -> identical model_hash.
    assert _model_hash(named) == _model_hash(bare)


def test_to_h5_round_trips_names(tmp_path: Path) -> None:
    src = tmp_path / "src.h5"
    dst = tmp_path / "dst.h5"
    _build(named=True).h5(str(src))
    OpenSeesModel.from_h5(str(src)).to_h5(str(dst))
    assert dict(
        (n, (k, t)) for n, k, t in read_names(str(dst))
    ) == dict((n, (k, t)) for n, k, t in read_names(str(src)))
