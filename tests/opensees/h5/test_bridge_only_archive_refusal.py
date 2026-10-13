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


# Other files without a neutral zone keep the error main raised: the guard
# fires only for a stamped /meta beside the /opensees bridge zone.


def _results_only(tmp_path: Path) -> Path:
    """A results-only file: a lineage-stub /meta (no schema_version)."""
    path = tmp_path / "results_only.h5"
    with h5py.File(path, "w") as f:
        f.create_group("meta").attrs["source"] = "lineage stub"
        f.create_group("results")
    return path


def _composed_without_nodes(tmp_path: Path) -> Path:
    """A composed results.h5 whose /model carries no ``nodes``: the root
    has a lineage-stub /meta and the verbatim /opensees zone, /model has
    the stamped bridge meta and nothing else."""
    flat = _bridge_only_archive(tmp_path)
    path = tmp_path / "composed_no_nodes.h5"
    with h5py.File(flat, "r") as src, h5py.File(path, "w") as dst:
        dst.create_group("meta").attrs["source"] = "lineage stub"
        src.copy(src["opensees"], dst, name="opensees")
        src.copy(src["meta"], dst.create_group("model"), name="meta")
    return path


def _main_error(path: Path, monkeypatch: pytest.MonkeyPatch) -> Exception:
    """What ``from_h5`` raised before this guard (main's behaviour)."""
    from apeGmsh.opensees import opensees_model

    with monkeypatch.context() as m:
        m.setattr(opensees_model, "_refuse_bridge_only", lambda *_a: None)
        with pytest.raises(Exception) as info:
            OpenSeesModel.from_h5(str(path))
    return info.value


@pytest.mark.parametrize("make", [_results_only, _composed_without_nodes],
                         ids=["results-only", "composed-model-no-nodes"])
def test_other_no_fem_files_keep_mains_error(
    make: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = make(tmp_path)  # type: ignore[operator]
    want = _main_error(path, monkeypatch)
    assert type(want) is MalformedH5Error
    assert "/meta/schema_version attribute is empty" in str(want)
    with pytest.raises(MalformedH5Error) as info:
        OpenSeesModel.from_h5(str(path))
    assert type(info.value) is type(want)  # not BridgeOnlyArchiveError
    assert str(info.value) == str(want)


def test_a_composed_file_with_the_bridge_zone_at_the_root_refuses(
    tmp_path: Path,
) -> None:
    """A composed results file keeps ``/opensees`` at the file root beside
    ``/model`` (ADR 0020). Built from a bridge-only archive, its ``/model``
    has a stamped ``meta`` and no ``nodes``; ``fem_root="/model"`` used to
    leak h5py's raw ``KeyError 'nodes'`` because the guard looked for the
    bridge zone under ``fem_root`` only (#1880)."""
    path = _composed_without_nodes(tmp_path)
    with h5py.File(path, "r") as f:
        assert "opensees" in f and "opensees" not in f["model"]
        assert "nodes" not in f["model"]
    with pytest.raises(BridgeOnlyArchiveError) as info:
        OpenSeesModel.from_h5(str(path), fem_root="/model")
    msg = str(info.value)
    assert "no 'nodes' group under /model/" in msg
    assert "Zones present: /model/meta." in msg
    assert "doesn't exist" not in msg  # not h5py's raw text
