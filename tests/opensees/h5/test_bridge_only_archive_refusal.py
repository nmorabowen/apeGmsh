"""A bridge-only archive refuses replay with a named error (#1797).

An ``apeSees`` built on ``FEMStub`` writes ``/meta`` and ``/opensees`` but no
neutral zone, so ``OpenSeesModel.from_h5`` has no mesh to replay. It used to
fail deep in the neutral-zone read with h5py's raw
``KeyError: "... object 'nodes' doesn't exist"``. It now raises
:class:`BridgeOnlyArchiveError` up front, naming the zones that are present.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import pytest

from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter.h5_reader import (
    BridgeOnlyArchiveError,
    MalformedH5Error,
)
from apeGmsh.opensees.section.fiber import FiberPoint
from tests.opensees.fixtures.fem_stub import make_two_node_beam
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_h5


def _bridge_only_archive(tmp_path: Path) -> Path:
    ops = apeSees(cast("object", make_two_node_beam()))
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    path = tmp_path / "bridge_only.h5"
    ops.h5(str(path))
    with h5py.File(path, "r") as f:
        assert "meta" in f and "opensees" in f and "nodes" not in f
    return path


def test_a_bridge_only_archive_raises_the_named_refusal(
    tmp_path: Path,
) -> None:
    path = _bridge_only_archive(tmp_path)
    with pytest.raises(BridgeOnlyArchiveError) as info:
        OpenSeesModel.from_h5(str(path))
    msg = str(info.value)
    assert "the archive carries no FEM (no neutral zone" in msg
    assert "cannot be replayed" in msg
    # It names the zones that are present.
    assert "Zones present:" in msg
    assert "/meta" in msg and "/opensees" in msg
    assert "doesn't exist" not in msg  # not h5py's raw text


def test_the_refusal_is_in_the_h5_error_family_not_a_key_error(
    tmp_path: Path,
) -> None:
    path = _bridge_only_archive(tmp_path)
    with pytest.raises(MalformedH5Error) as info:
        OpenSeesModel.from_h5(str(path))
    assert isinstance(info.value, ValueError)
    assert not isinstance(info.value, KeyError)


def test_the_refusal_comes_before_any_neutral_zone_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _bridge_only_archive(tmp_path)

    def _no_read(*_a: object, **_k: object) -> FEMData:
        raise AssertionError("the neutral zone was read before the guard")

    monkeypatch.setattr(FEMData, "from_h5", classmethod(_no_read))
    with pytest.raises(BridgeOnlyArchiveError):
        OpenSeesModel.from_h5(str(path))


def test_an_explicit_fem_root_names_its_own_zones(tmp_path: Path) -> None:
    flat = _bridge_only_archive(tmp_path)
    nested = tmp_path / "nested.h5"
    with h5py.File(flat, "r") as src, h5py.File(nested, "w") as dst:
        model = dst.create_group("model")
        for name in src:
            src.copy(src[name], model, name=name)
    with pytest.raises(BridgeOnlyArchiveError) as info:
        OpenSeesModel.from_h5(str(nested), fem_root="/model")
    msg = str(info.value)
    assert "no 'nodes' group under /model/" in msg
    assert "/model/meta" in msg and "/model/opensees" in msg


def test_an_archive_with_a_neutral_zone_still_replays(tmp_path: Path) -> None:
    path, fem = build_simple_frame_h5(tmp_path)
    model = OpenSeesModel.from_h5(str(path))
    assert len(model.fem.nodes.ids) == len(fem.nodes.ids)
