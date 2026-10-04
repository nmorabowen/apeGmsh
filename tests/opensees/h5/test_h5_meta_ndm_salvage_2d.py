"""The pre-2.34.0 ``/meta/ndm`` salvage reads a 2-D frame as 2-D (#1358).

A writer before neutral 2.34.0 stamped the mesh dimension in
``/meta/ndm``, so a line-only 2-D frame carries ``1``.  The salvage in
``h5_reader.read_spatial_ndm`` (#1300) lifted the stamp only on a 3-wide
``per_element_vecxz``; a 2-D ``geomTransf`` has no vecxz, its dataset is
``(N, 0)``, so the stamp stood and the rebuilt deck emitted
``model -ndm 1`` with every y coordinate dropped.

Oracles:

* the ``ndm`` the frame was declared with (``ops.model(ndm=2)``), and
  the node coordinates it was built from, which the rebuilt deck must
  carry unchanged;
* every 3-D writer writes a 3-wide vecxz (a 3-D ``geomTransf`` cannot
  be emitted without one), so a zero-width vecxz beside a 3-wide one, or
  beside a 3-D mesh stamp, is evidence in conflict, and the reader
  refuses;
* a resolved ``ndm`` that would drop a non-zero coordinate column is a
  refusal, never a silent truncation.

The pre-2.34.0 file is the current writer's output restamped the way
the old writer stamped it (``/meta/ndm`` = mesh dimension, neutral
version below ``META_NDM_IS_SPATIAL_FROM``), as #1300's salvage test
does: the transform layout (one group per ``geomTransf`` call, a
``per_element_vecxz`` row holding exactly the emitted vector) has not
changed since the H5 emitter first wrote it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from tests.fixtures.schema import NEUTRAL_CURRENT
from tests.opensees.h5._opensees_model_fixtures import PRE_NDM_FIX_STAMP

#: A portal in the x-y plane: two columns 3 m tall, a 4 m beam.
NODE_IDS = np.array([1, 2, 3, 4], dtype=np.int64)
NODE_COORDS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0],
    [4.0, 3.0, 0.0],
    [4.0, 0.0, 0.0],
], dtype=np.float64)
CONNECTIVITY = np.array([[1, 2], [2, 3], [4, 3]], dtype=np.int64)


def _portal_fem() -> FEMData:
    """A line-only 2-D portal frame; every element in PG ``"Frame"``."""
    n_el = CONNECTIVITY.shape[0]
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=n_el,
    )
    eids = np.arange(1, n_el + 1, dtype=np.int64)
    group = ElementGroup(
        element_type=line_info, ids=eids, connectivity=CONNECTIVITY,
    )
    pg = {(1, 100): {
        "name": "Frame",
        "node_ids": NODE_IDS,
        "node_coords": NODE_COORDS,
        "element_ids": eids,
    }}
    nodes = NodeComposite(
        node_ids=NODE_IDS, node_coords=NODE_COORDS,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: group}, physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=NODE_IDS.size, n_elems=n_el, bandwidth=1, types=[line_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


def _write_pre_fix_portal(tmp_path: Path, *, stamp: int = 1) -> Path:
    """Write the portal at ``ops.model(ndm=2, ndf=3)``, then stamp it the
    way a pre-2.34.0 writer did: ``/meta/ndm`` = ``stamp`` (the mesh
    dimension, 1 for a line mesh) under a neutral version below the fix."""
    from apeGmsh.opensees import apeSees

    ops = apeSees(_portal_fem())
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(
        pg="Frame", A=0.01, E=200e9, Iz=1e-4, transf=transf,
    )
    out = tmp_path / "portal_2d.h5"
    ops.h5(str(out))
    with h5py.File(out, "r+") as f:
        f["meta"].attrs["ndm"] = stamp
        f["meta"].attrs["neutral_schema_version"] = PRE_NDM_FIX_STAMP
    return out


def _deck_nodes(deck: str) -> dict[int, tuple[float, ...]]:
    """``{tag: coords}`` from the ``node`` lines of a Tcl deck."""
    out: dict[int, tuple[float, ...]] = {}
    for line in deck.splitlines():
        parts = line.split()
        if parts and parts[0] == "node":
            out[int(parts[1])] = tuple(float(v) for v in parts[2:])
    return out


# ---------------------------------------------------------------------------
# The 2-D signature resolves 2, and the deck keeps every y
# ---------------------------------------------------------------------------


def test_the_2d_writer_stores_a_zero_width_vecxz(tmp_path: Path) -> None:
    """The signature the salvage reads: a 2-D ``geomTransf`` has no vecxz,
    so its ``per_element_vecxz`` row is empty."""
    out = _write_pre_fix_portal(tmp_path)
    with h5py.File(out, "r") as f:
        shapes = {
            name: g["per_element_vecxz"].shape
            for name, g in f["opensees/transforms"].items()
        }
    assert shapes and all(s == (1, 0) for s in shapes.values()), shapes


def test_a_pre_fix_2d_frame_reads_ndm_2_and_keeps_y(tmp_path: Path) -> None:
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    out = _write_pre_fix_portal(tmp_path)
    om = OpenSeesModel.from_h5(out)
    assert om.ndm == 2
    deck = om.build("tcl")
    model_lines = [ln for ln in deck.splitlines() if ln.startswith("model ")]
    assert model_lines == ["model BasicBuilder -ndm 2 -ndf 3"]
    want = {int(n): (float(x), float(y)) for n, (x, y, _z) in zip(NODE_IDS, NODE_COORDS)}
    assert _deck_nodes(deck) == want


def test_a_composed_twin_forwards_the_salvaged_2(tmp_path: Path) -> None:
    """INV-9: ``write_opensees_from`` forwards what the source's own reader
    resolves (2), not the raw pre-fix stamp (1), under the new stamp."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from apeGmsh.results.writers import NativeWriter

    src = _write_pre_fix_portal(tmp_path)
    composed = tmp_path / "composed.h5"
    with NativeWriter(composed) as w:
        w.open(fem=_portal_fem(), model_h5_src=src)
        sid = w.begin_stage(name="g", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0", node_ids=NODE_IDS,
            components={"displacement_y": np.zeros((1, NODE_IDS.size))},
        )
        w.end_stage()
    with h5py.File(composed, "r") as f:
        assert int(f["model/meta"].attrs["ndm"]) == 2
    om = OpenSeesModel.from_h5(
        composed, fem_root="/model", opensees_root="/opensees",
    )
    assert om.ndm == 2


# ---------------------------------------------------------------------------
# Conflicting evidence refuses
# ---------------------------------------------------------------------------


def test_a_zero_width_vecxz_under_a_3d_mesh_stamp_refuses(tmp_path: Path) -> None:
    """A 3-D mesh stamp cannot belong to a 2-D model."""
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    out = _write_pre_fix_portal(tmp_path, stamp=3)
    with pytest.raises(MalformedH5Error, match="conflict"):
        OpenSeesModel.from_h5(out)


def test_a_zero_width_beside_a_3_wide_vecxz_refuses(tmp_path: Path) -> None:
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    out = _write_pre_fix_portal(tmp_path)
    with h5py.File(out, "r+") as f:
        g = f["opensees/transforms"].create_group("Linear_99")
        g.attrs["type"] = "Linear"
        g.attrs["tag"] = 99
        g.create_dataset("per_element_vecxz", data=np.array([[0.0, 0.0, 1.0]]))
        g.create_dataset("per_element_emitted_tag", data=np.array([99]))
    with pytest.raises(MalformedH5Error, match="conflict"):
        OpenSeesModel.from_h5(out)


def test_absent_evidence_never_drops_a_coordinate(tmp_path: Path) -> None:
    """With no transform to read, the stamp (1) stands only if it drops
    nothing; this frame's y column is not zero, so the reader refuses."""
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    out = _write_pre_fix_portal(tmp_path)
    with h5py.File(out, "r+") as f:
        del f["opensees/transforms"]
    with pytest.raises(MalformedH5Error, match="drop"):
        OpenSeesModel.from_h5(out)


# ---------------------------------------------------------------------------
# read_spatial_ndm on hand-built evidence
# ---------------------------------------------------------------------------


def _bridge_file(
    tmp_path: Path, *, widths: "tuple[int, ...]",
) -> Any:
    """An open h5py file whose ``/opensees/transforms`` holds one group per
    vecxz width in ``widths``."""
    f = h5py.File(tmp_path / "evidence.h5", "w")
    transforms = f.create_group("opensees").create_group("transforms")
    for i, w in enumerate(widths, start=1):
        g = transforms.create_group(f"Linear_{i}")
        g.create_dataset("per_element_vecxz", data=np.zeros((1, w)))
        g.create_dataset("per_element_emitted_tag", data=np.array([i]))
    return f


def _legacy_meta(stamp: int) -> dict[str, Any]:
    return {"ndm": stamp, "neutral_schema_version": PRE_NDM_FIX_STAMP}


@pytest.mark.parametrize(
    "stamp, widths, coords, want",
    [
        # #1358: the 2-D signature lifts a line-mesh stamp to 2.
        (1, (0,), NODE_COORDS, 2),
        (2, (0,), NODE_COORDS, 2),
        (1, (0, 0), NODE_COORDS, 2),
        # #1300: a 3-wide vecxz lifts the stamp to 3.
        (1, (3,), NODE_COORDS, 3),
        (2, (3, 3), NODE_COORDS, 3),
        # No transform: the stamp stands when it drops nothing.
        (1, (), np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), 1),
        (3, (), NODE_COORDS, 3),
        # Float noise far below the model's size is not a coordinate.
        (1, (), np.array([[0.0, 1e-17, 0.0], [5.0, 0.0, -1e-16]]), 1),
    ],
)
def test_read_spatial_ndm_salvage(
    tmp_path: Path, stamp: int, widths: "tuple[int, ...]",
    coords: np.ndarray, want: int,
) -> None:
    from apeGmsh.opensees.emitter.h5_reader import read_spatial_ndm

    with _bridge_file(tmp_path, widths=widths) as f:
        assert read_spatial_ndm(_legacy_meta(stamp), f, coords=coords) == want


@pytest.mark.parametrize(
    "stamp, widths, coords, match",
    [
        (3, (0,), NODE_COORDS, "conflict"),
        (1, (0, 3), NODE_COORDS, "conflict"),
        (1, (2,), NODE_COORDS, "width"),
        (1, (), NODE_COORDS, "drop"),
        # The 2-D signature, but the nodes use z: refuse, never truncate.
        (1, (0,), np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 1.0]]), "drop"),
    ],
)
def test_read_spatial_ndm_salvage_refuses(
    tmp_path: Path, stamp: int, widths: "tuple[int, ...]",
    coords: np.ndarray, match: str,
) -> None:
    from apeGmsh.opensees.emitter.h5_reader import (
        MalformedH5Error,
        read_spatial_ndm,
    )

    with _bridge_file(tmp_path, widths=widths) as f:
        with pytest.raises(MalformedH5Error, match=match):
            read_spatial_ndm(_legacy_meta(stamp), f, coords=coords)


def test_a_current_stamp_is_trusted_as_written(tmp_path: Path) -> None:
    """At or above ``META_NDM_IS_SPATIAL_FROM`` the declaration is read
    as-is; the salvage's evidence is not consulted."""
    from apeGmsh.opensees.emitter.h5_reader import read_spatial_ndm

    meta = {"ndm": 2, "neutral_schema_version": NEUTRAL_CURRENT}
    with _bridge_file(tmp_path, widths=(3,)) as f:
        assert read_spatial_ndm(meta, f, coords=NODE_COORDS) == 2
