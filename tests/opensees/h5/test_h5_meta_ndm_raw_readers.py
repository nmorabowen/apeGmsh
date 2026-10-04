"""Three readers that took the raw ``/meta/ndm`` stamp now take the shim (#1368).

Before neutral 2.34.0 ``/meta/ndm`` was the *mesh* dimension: a 2-D
frame of lines read ``1``, a 3-D frame with a shell slab read ``2``.
:func:`~apeGmsh.opensees.emitter.h5_reader.read_spatial_ndm` salvages the
``ops.model`` ndm from such a file or refuses (#1300, #1358).  These
tests pin the three readers that bypassed it, each on a committed corpus
file whose declared ndm is known:

* ``opensees_2.20_frame2d.h5`` declared ``ops.model(ndm=2, ndf=3)`` and
  carries the pre-2.34.0 stamp ``1``;
* ``opensees_2.20.h5`` declared ``ops.model(ndm=3, ndf=6)`` and carries
  the pre-2.34.0 stamp ``1``.  Its deck (``opensees_2.20.tcl``) declares
  one ``geomTransf Linear 1 0.0 1.0 0.0`` that every beam-column uses.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.model_data import ModelData
from apeGmsh.results.capture import DomainCapture, DomainCaptureSpec

CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "schema_corpus"
FRAME2D = CORPUS / "opensees_2.20_frame2d.h5"
FRAME3D = CORPUS / "opensees_2.20.h5"

#: The declared ``ops.model`` ndm of each corpus file (its generator).
FRAME2D_DECLARED_NDM = 2
FRAME3D_DECLARED_NDM = 3

#: The single vecxz ``opensees_2.20.tcl`` declares for every beam-column.
FRAME3D_VECXZ = (0.0, 1.0, 0.0)


def _stamp(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["meta"].attrs["ndm"])


def test_the_corpus_files_carry_the_pre_2_34_mesh_stamp() -> None:
    """The precondition: both files carry the mesh dimension, not the
    declared ndm, so a reader of the raw stamp gets the wrong answer."""
    assert _stamp(FRAME2D) == 1 != FRAME2D_DECLARED_NDM
    assert _stamp(FRAME3D) == 1 != FRAME3D_DECLARED_NDM


# ---------------------------------------------------------------------------
# ModelData.from_h5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "declared"),
    [(FRAME2D, FRAME2D_DECLARED_NDM), (FRAME3D, FRAME3D_DECLARED_NDM)],
    ids=["frame2d", "frame3d"],
)
def test_model_data_from_h5_reads_the_declared_ndm(
    path: Path, declared: int,
) -> None:
    """The raw stamp ``1`` was refused (``ndm must be 2 or 3``); the shim
    resolves the declared ndm."""
    md = ModelData.from_h5(str(path))
    assert md.ndm == declared


# ---------------------------------------------------------------------------
# DomainCapture.from_h5
# ---------------------------------------------------------------------------


def test_domain_capture_from_h5_resolves_a_2d_frame_in_2d(
    tmp_path: Path,
) -> None:
    """The raw stamp resolved the spec in one dimension, so the shorthand
    ``displacement`` expanded to x only and y was never captured."""
    fem = FEMData.from_h5(str(FRAME2D))
    spec = DomainCaptureSpec()
    spec.nodes(components=["displacement"], ids=[int(fem.nodes.ids[0])])
    cap = DomainCapture.from_h5(
        FRAME2D, spec=spec, fem=fem, output=tmp_path / "run.h5",
        ops=object(),
    )
    assert cap._spec.ndm == FRAME2D_DECLARED_NDM
    assert cap._spec.records[0].components == (
        "displacement_x", "displacement_y",
    )


# ---------------------------------------------------------------------------
# H5Model.element_local_axes_vecxz
# ---------------------------------------------------------------------------


def _beam_fem_eids(path: Path) -> set[int]:
    with h5py.File(path, "r") as f:
        em = f["opensees/element_meta"]
        return {int(e) for t in em for e in em[t]["fem_eids"][()]}


def _vecxz(path: Path) -> dict[int, Any]:
    with h5_reader.open(str(path)) as model:
        return model.element_local_axes_vecxz()


@pytest.mark.parametrize("stamp", [1, 2])
def test_element_local_axes_vecxz_reads_every_3d_beam(
    tmp_path: Path, stamp: int,
) -> None:
    """A pre-2.34.0 3-D frame keeps every beam's vecxz under either mesh
    stamp.  ``2`` is what the same frame carried with a shell slab: the
    raw stamp read the 2-D slot vocabulary, took ``Iz`` for the transf tag
    and dropped every ``elasticBeamColumn`` from the orientation overlay.
    """
    path = tmp_path / "frame3d.h5"
    shutil.copyfile(FRAME3D, path)
    with h5py.File(path, "r+") as f:
        f["meta"].attrs["ndm"] = stamp
    got = _vecxz(path)
    assert set(got) == _beam_fem_eids(FRAME3D)
    for vec in got.values():
        np.testing.assert_allclose(vec, FRAME3D_VECXZ)


def test_element_local_axes_vecxz_of_the_2d_frame_is_empty() -> None:
    """A 2-D ``geomTransf`` carries no vecxz, so there is nothing to join."""
    assert _vecxz(FRAME2D) == {}


# ---------------------------------------------------------------------------
# A bridge-only file carries a per-zone opensees stamp and no neutral zone
# ---------------------------------------------------------------------------


def _bridge_only_frame(path: Path) -> None:
    """A real ``apeSees.h5`` write with no broker: ``H5Emitter._meta_attrs``
    stamps ``opensees_schema_version`` plus the envelope, no neutral key,
    no ``/nodes``, and ``/meta/ndm`` from the bridge's ``ops.model``."""
    from typing import cast

    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint
    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    ops = apeSees(cast("object", make_two_node_beam()))
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ops.h5(str(path))


def test_a_bridge_only_file_is_read_as_stamped(tmp_path: Path) -> None:
    """The envelope repeats the opensees version; borrowing it as a neutral
    version misfiled the file as pre-2.34.0 and demanded coordinates it
    does not carry (#1389)."""
    path = tmp_path / "bridge_only.h5"
    _bridge_only_frame(path)
    with h5py.File(path, "r") as f:
        attrs = f["meta"].attrs
        assert "opensees_schema_version" in attrs
        assert "neutral_schema_version" not in attrs
        assert "nodes" not in f

        def _no_coords() -> Any:
            raise AssertionError("a bridge-only stamp needs no coordinates")

        assert h5_reader.read_spatial_ndm(attrs, f, coords=_no_coords) == 3
    vecxz = _vecxz(path)
    assert set(vecxz) == {1}
    np.testing.assert_allclose(vecxz[1], [1.0, 0.0, 0.0])
