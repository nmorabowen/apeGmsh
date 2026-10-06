"""``/meta/ndm`` is the model's spatial dimension, not the mesh dimension (#1291).

A line-only frame declared with ``ops.model(ndm=2, ndf=3)`` used to stamp
``/meta/ndm = 1`` (the highest element dimension), and every reader that
takes ``/meta/ndm`` as the ``ops.model`` dimension mis-read it:
``OpenSeesModel.from_h5(...).build()`` re-emitted ``model -ndm 1``.

The oracle is the ``ndm`` the caller passed to ``ops.model``: the file
must carry it back unchanged through every composer caller and reader.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from tests.fixtures.schema import NEUTRAL_CURRENT, OPENSEES_CURRENT
from tests.opensees.h5._opensees_model_fixtures import (
    PRE_NDM_FIX_STAMP,
    build_simple_frame_fem,
    build_simple_frame_h5,
)


def _write_frame(tmp_path: Path, *, ndm: int, ndf: int) -> Path:
    """One line element, declared in ``ndm`` dimensions, archived to H5."""
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    fem = build_simple_frame_fem(ndm=ndm)
    ops = apeSees(fem)
    ops.model(ndm=ndm, ndf=ndf)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    if ndm == 3:
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    else:
        transf = ops.geomTransf.Linear()
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )
    out = tmp_path / f"frame_{ndm}d.h5"
    ops.h5(str(out))
    return out


@pytest.mark.parametrize("ndm, ndf", [(2, 3), (3, 6)])
def test_apesees_h5_stamps_the_declared_ndm(
    tmp_path: Path, ndm: int, ndf: int,
) -> None:
    """A line-only frame carries the ``ops.model`` ndm, never the mesh dim."""
    out = _write_frame(tmp_path, ndm=ndm, ndf=ndf)
    with h5py.File(out, "r") as f:
        assert int(f["meta"].attrs["ndm"]) == ndm
        assert int(f["meta"].attrs["ndf"]) == ndf


def test_opensees_model_from_h5_reads_the_declared_ndm(tmp_path: Path) -> None:
    """The rebuilt deck declares the same ndm the archive was built with."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    out = _write_frame(tmp_path, ndm=2, ndf=3)
    om = OpenSeesModel.from_h5(out)
    assert om.ndm == 2
    model_lines = [
        line for line in om.build("tcl").splitlines()
        if line.startswith("model ")
    ]
    assert model_lines == ["model BasicBuilder -ndm 2 -ndf 3"]


def test_opensees_model_to_h5_round_trips_the_declared_ndm(
    tmp_path: Path,
) -> None:
    """``from_h5 -> to_h5`` re-stamps the ndm it read, not the mesh dim."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    src = _write_frame(tmp_path, ndm=2, ndf=3)
    dst = tmp_path / "rewritten.h5"
    OpenSeesModel.from_h5(src).to_h5(dst)
    with h5py.File(dst, "r") as f:
        assert int(f["meta"].attrs["ndm"]) == 2


def test_model_data_2d_frame_writes_and_reads_ndm(tmp_path: Path) -> None:
    """``ModelData(ndm=2)`` on a line-only fem writes, and reads back, 2.

    The read-back runs through both readers: ``OpenSeesModel.from_h5``
    (the replacement for the ``ModelData.ndm`` accessor, K19 #1506) and
    ``ModelData.from_h5`` (kept).
    """
    from apeGmsh.opensees.model_data import ModelData
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    fem = build_simple_frame_fem(ndm=2)
    out = tmp_path / "md.h5"
    ModelData(fem, ndm=2, ndf=3).write(str(out))
    with h5py.File(out, "r") as f:
        assert int(f["meta"].attrs["ndm"]) == 2
    assert OpenSeesModel.from_h5(str(out)).ndm == 2
    assert ModelData.from_h5(str(out)).ndm == 2


def test_opensees_model_from_h5_salvages_a_pre_fix_stamp(
    tmp_path: Path,
) -> None:
    """A file written before neutral 2.34.0 stamped the mesh dimension;
    the reader still recovers the spatial ndm from the transforms."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    out = _write_frame(tmp_path, ndm=3, ndf=6)
    with h5py.File(out, "r+") as f:
        f["meta"].attrs["ndm"] = 1
        f["meta"].attrs["neutral_schema_version"] = PRE_NDM_FIX_STAMP
    assert OpenSeesModel.from_h5(out).ndm == 3


def test_composed_results_forward_the_declared_ndm(tmp_path: Path) -> None:
    """A composed ``results.h5`` embeds the fem broker-only under
    ``/model/``; the sidecar's declared ndm is forwarded onto
    ``/model/meta`` alongside ``ndf``."""
    import numpy as np

    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from apeGmsh.results.writers import NativeWriter

    src = _write_frame(tmp_path, ndm=2, ndf=3)
    fem = build_simple_frame_fem(ndm=2)
    composed = tmp_path / "composed.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(composed) as w:
        w.open(fem=fem, model_h5_src=src)
        sid = w.begin_stage(name="g", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_x": np.zeros((1, node_ids.size))},
        )
        w.end_stage()

    with h5py.File(composed, "r") as f:
        assert int(f["model/meta"].attrs["ndm"]) == 2
        assert int(f["model/meta"].attrs["ndf"]) == 3
    om = OpenSeesModel.from_h5(
        composed, fem_root="/model", opensees_root="/opensees",
    )
    assert om.ndm == 2


def test_inv9_composed_results_do_not_launder_a_pre_fix_sidecar_stamp(
    tmp_path: Path,
) -> None:
    """ADR 0113 INV-9 (no laundering), on the one restamping path: the
    results twin.

    ``NativeWriter.write_model`` restamps the embedded ``/model/meta``
    with the CURRENT neutral version, and ``write_opensees_from``
    forwards the source's ``ndm`` onto it. A pre-2.34.0 source (here a
    3-D frame stamped ``ndm=1``, the mesh dimension) is below the
    ndm-trust version but inside the floor, so the raw attribute means
    something else than the new stamp says. The value forwarded must be
    the spatial ndm the source's OWN reader resolves through its shim
    (``read_spatial_ndm`` keyed on ``META_NDM_IS_SPATIAL_FROM``), never
    the raw attribute: a raw forward fails this test.
    """
    import numpy as np

    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from apeGmsh.results.writers import NativeWriter

    src, fem = build_simple_frame_h5(tmp_path)
    raw_stamp = 1
    with h5py.File(src, "r+") as f:
        f["meta"].attrs["ndm"] = raw_stamp
        f["meta"].attrs["neutral_schema_version"] = PRE_NDM_FIX_STAMP
    # The oracle: what the source's own reader resolves under the
    # source's own stamp (the shim lifts the mesh dimension to 3).
    salvaged = OpenSeesModel.from_h5(src).ndm
    assert salvaged == 3 and salvaged != raw_stamp

    composed = tmp_path / "composed_stale.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(composed) as w:
        w.open(fem=fem, model_h5_src=src)
        sid = w.begin_stage(name="g", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((1, node_ids.size))},
        )
        w.end_stage()

    with h5py.File(composed, "r") as f:
        meta = f["model/meta"].attrs
        # The twin IS restamped: the embedded zone carries the current
        # neutral version, under which a reader trusts /meta/ndm.
        assert str(meta["neutral_schema_version"]) == NEUTRAL_CURRENT
        assert int(meta["ndm"]) == salvaged
        assert int(meta["ndm"]) != raw_stamp
    om = OpenSeesModel.from_h5(
        composed, fem_root="/model", opensees_root="/opensees",
    )
    assert om.ndm == salvaged


def test_opensees_model_refuses_to_build_from_a_broker_only_file(
    tmp_path: Path,
) -> None:
    """``fem.to_h5`` declares no ndm (``0``); emitting a deck from it
    fails loud instead of writing ``model -ndm 0``."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    fem = build_simple_frame_fem()
    src = tmp_path / "broker_only.h5"
    fem.to_h5(str(src))
    om = OpenSeesModel.from_h5(src)
    assert om.ndm == 0
    with pytest.raises(ValueError, match="no declared ndm"):
        om.build("tcl")
    with pytest.raises(ValueError, match="no declared ndm"):
        om.to_h5(tmp_path / "rewritten.h5")


def test_domain_capture_from_h5_refuses_an_undeclared_ndm(
    tmp_path: Path,
) -> None:
    """``DomainCapture.from_h5`` never resolves a spec in zero dimensions."""
    from apeGmsh.results.capture import DomainCaptureSpec
    from apeGmsh.results.capture._domain import DomainCapture

    model_path = tmp_path / "broker_only_meta.h5"
    with h5py.File(model_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = OPENSEES_CURRENT
        meta.attrs["opensees_schema_version"] = OPENSEES_CURRENT
        meta.attrs["ndm"] = 0
        meta.attrs["ndf"] = 0
        meta.attrs["snapshot_id"] = "stub"
        meta.attrs["model_name"] = "stub"
    spec = DomainCaptureSpec()
    spec.nodes(components=["displacement"], ids=[1])
    with pytest.raises(RuntimeError, match="no declared ndm"):
        DomainCapture.from_h5(
            model_path, spec=spec, fem=build_simple_frame_fem(),
            output=tmp_path / "run.h5", ops=None,
        )


def test_compose_refuses_an_undeclared_ndm(tmp_path: Path) -> None:
    """The composer never stamps ``ndm=0`` on a bridge-written file."""
    from apeGmsh.opensees._internal.compose import _compose_model_h5
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    fem = build_simple_frame_fem()
    emitter = H5Emitter(model_name="m", snapshot_id=str(fem.snapshot_id))
    with pytest.raises(ValueError, match="ndm"):
        _compose_model_h5(
            fem, emitter, str(tmp_path / "m.h5"),
            model_name="m", ndm=0, ndf=6,
        )
