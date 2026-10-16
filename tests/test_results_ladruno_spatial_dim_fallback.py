"""A ``.ladruno`` without ``INFO/SPATIAL_DIM`` is never read as 3-D (#2081).

The reader used to fall back to ``ndm = 3`` when the attribute was
absent. A 2-D truss in a frame (ndf 3) writes ``globalForce`` as
``P<node>_<dof>``, so its ``P1_3`` is the node-1 rotation slot; under the
fallback it decoded as ``nodal_resisting_force_z``. The pinned fork always
writes the attribute, so only foreign files lack it.

Now the dimension comes from the attribute, else from the width of
``MODEL/NODES/COORDINATES`` (the writer sizes it by ``ndm``), else it is
unknown. Then a ``nodal_resisting_*`` read over indexed ``P<a>_<b>``
columns raises ``SpatialDimUnknown`` naming the file, the element listing
leaves those names out with a warning, and ``opensees_model`` raises the
same. ``SpatialDimUnknown`` is not a ``MissingElementResults``, so a
partition set does not drop the rank that raises it. Axis-named beam
columns (``Px_1``) name their quantity and read at any dimension, and the
raw token reads stay as they were.

The oracle is constructive: each synthetic column holds
``_truth(e, n, q, t)`` for its canonical quantity ``q``, and a column with
no canonical quantity (the rotation slot) holds ``-999``. A read that takes
the rotation for force z returns ``-999``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno import LadrunoReader, SpatialDimUnknown
from apeGmsh.results.readers._ladruno_element_io import NodalForceNotRecorded
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._protocol import ResultLevel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ladruno"
STAGE = "MODEL_STAGE[1]"

_FORCES = ("nodal_resisting_force_x", "nodal_resisting_force_y",
           "nodal_resisting_force_z")
_MOMENTS = ("nodal_resisting_moment_x", "nodal_resisting_moment_y",
            "nodal_resisting_moment_z")
_Q = {c: i + 1 for i, c in enumerate(_FORCES + _MOMENTS)}


def _truth(e: int, n: int, q: str, t: int) -> float:
    return 1000.0 * e + 100.0 * n + _Q[q] + 0.125 * (t + 1)


def _node_dof(n_nodes: int, ndf: int, ndm: int):
    """Truss spelling ``P<node>_<dof>``; DOFs above ``ndm`` carry no
    canonical quantity."""
    return [(f"P{n}_{d}", n, _FORCES[d - 1] if d <= ndm else None)
            for n in range(1, n_nodes + 1) for d in range(1, ndf + 1)]


def _dof_node(n_nodes: int, ndf: int, ndm: int):
    """Brick spelling ``P<dof>_<node>``."""
    return [(f"P{d}_{n}", n, _FORCES[d - 1] if d <= ndm else None)
            for n in range(1, n_nodes + 1) for d in range(1, ndf + 1)]


# A 2-D truss in a frame model: ndf 3, so P1_3 / P2_3 are rotation slots.
_TRUSS_2D_NDF3 = ("12-Truss[1:0:0]", _node_dof(2, 3, 2), [1, 2])
# A 3-D 8-node brick: P3_<node> is force z.
_BRICK_3D = ("56-Brick[1:0:0]", _dof_node(8, 3, 3), [40])
# A 3-D elastic beam in the axis-named spelling Px_1, ..., Mz_2: each
# column names its quantity, so no decode of it consults ndm.
_BEAM_3D_AXIS = ("5-ElasticBeam3d[1:0:0]",
                 [(f"{'P' if '_force_' in c else 'M'}{c[-1]}_{n}", n, c)
                  for n in (1, 2) for c in _FORCES + _MOMENTS], [7])


def _write_bucket(on_e: "h5py.Group", key: str, layout, ids,
                  n_steps: int) -> None:
    grp = (on_e["globalForce"] if "globalForce" in on_e
           else on_e.create_group("globalForce")).create_group(key)
    data = np.zeros((n_steps, len(ids), len(layout)))
    for i, e in enumerate(ids):
        for col, (_, n, q) in enumerate(layout):
            for t in range(n_steps):
                data[t, i, col] = (_truth(e, n, q, t) if q is not None
                                   else -999.0)
    cm = grp.create_group("COLUMN_MAP")
    for name, value in (("LEVELS", 0), ("GAUSS_ID", -1), ("MULTIPLICITY", 1),
                        ("SECTION_TAG", -1), ("FIBER_ID", -1)):
        cm.create_dataset(name, data=np.array([[value]], dtype=np.int32))
    cm.create_dataset("NUM_COMP",
                      data=np.array([[len(layout)]], dtype=np.int32))
    cm.attrs["COMP_NAMES"] = np.bytes_(",".join(n for n, _, _ in layout))
    grp.create_dataset("DATA", data=data)
    grp.create_dataset("ID", data=np.array(ids, dtype=np.int32)[:, None])
    grp.create_dataset("STEP", data=np.arange(n_steps, dtype=np.int32))
    grp.create_dataset("TIME", data=0.25 * np.arange(1, n_steps + 1))


def _file(tmp_path: Path, source: str, buckets, *, strip_dim: bool = True,
          drop_model: bool = False) -> Path:
    """Copy a committed fixture, add ``globalForce`` buckets, and strip
    ``INFO/SPATIAL_DIM`` (and the stage ``MODEL`` when asked)."""
    dst = tmp_path / source
    shutil.copyfile(FIXTURES / source, dst)
    with h5py.File(dst, "r+") as f:
        on_e = f[f"{STAGE}/RESULTS/ON_ELEMENTS"]
        n_steps = int(next(iter(next(iter(on_e.values())).values()))
                      ["DATA"].shape[0])
        for key, layout, ids in buckets:
            _write_bucket(on_e, key, layout, ids, n_steps)
        if strip_dim:
            del f["INFO"].attrs["SPATIAL_DIM"]
        if drop_model:
            del f[f"{STAGE}/MODEL"]
    return dst


def _expected(ids, comp: str, n_nodes: int, n_steps: int = 4) -> np.ndarray:
    return np.array([[[_truth(e, n, comp, t) for n in range(1, n_nodes + 1)]
                      for e in ids] for t in range(n_steps)])


def _read(reader, comp: str):
    return reader.read_elements(reader.stages()[0].id, comp)


def _elements(reader) -> list[str]:
    return reader.available_components(
        reader.stages()[0].id, ResultLevel.ELEMENTS)


# ---------------------------------------------------------------------------
# The dimension is inferred from MODEL/NODES/COORDINATES
# ---------------------------------------------------------------------------


def test_2d_truss_rotation_slot_is_never_force_z(tmp_path: Path) -> None:
    path = _file(tmp_path, "truss2d.ladruno", [_TRUSS_2D_NDF3])
    with h5py.File(path, "r") as f:
        assert "SPATIAL_DIM" not in f["INFO"].attrs
        assert f[f"{STAGE}/MODEL/NODES/COORDINATES"].shape[1] == 2
    with LadrunoReader(path) as r:
        listed = _elements(r)
        assert "nodal_resisting_force_z" not in listed
        for comp in ("nodal_resisting_force_x", "nodal_resisting_force_y"):
            assert comp in listed
            np.testing.assert_array_equal(
                _read(r, comp).values, _expected([1, 2], comp, 2),
                err_msg=comp)
        with pytest.raises(NodalForceNotRecorded,
                           match="nodal_resisting_force_z"):
            _read(r, "nodal_resisting_force_z")
        model = r.opensees_model()
        assert model is not None and model.ndm == 2


def test_3d_brick_reads_force_z_from_the_coordinates(tmp_path: Path) -> None:
    path = _file(tmp_path, "beam3d.ladruno", [_BRICK_3D])
    with LadrunoReader(path) as r:
        listed = _elements(r)
        for comp in _FORCES:
            assert comp in listed
            np.testing.assert_array_equal(
                _read(r, comp).values, _expected([40], comp, 8, n_steps=1),
                err_msg=comp)
        model = r.opensees_model()
        assert model is not None and model.ndm == 3


def test_the_attribute_still_wins_over_the_coordinates(tmp_path: Path) -> None:
    # SPATIAL_DIM = 2 on a file whose coordinates are 3 wide: the attribute
    # is the writer's statement, so P3_<node> is left out.
    path = _file(tmp_path, "beam3d.ladruno", [_BRICK_3D], strip_dim=False)
    with h5py.File(path, "r+") as f:
        f["INFO"].attrs["SPATIAL_DIM"] = np.array([2], dtype=np.int32)
    with LadrunoReader(path) as r:
        assert "nodal_resisting_force_z" not in _elements(r)
        assert r.opensees_model().ndm == 2


# ---------------------------------------------------------------------------
# The dimension is unknown: refuse or warn, naming the file
# ---------------------------------------------------------------------------


def test_unknown_dimension_refuses_the_nodal_reads(tmp_path: Path) -> None:
    path = _file(tmp_path, "truss2d.ladruno", [_TRUSS_2D_NDF3],
                 drop_model=True)
    with LadrunoReader(path) as r:
        with pytest.warns(UserWarning, match=r"truss2d\.ladruno.*SPATIAL_DIM"):
            listed = _elements(r)
        assert not [c for c in listed if c.startswith("nodal_resisting_")]
        assert "globalForce" in listed and "basicForce" in listed
        for comp in _FORCES:
            with pytest.raises(SpatialDimUnknown,
                               match=r"truss2d\.ladruno.*SPATIAL_DIM"):
                _read(r, comp)


def test_unknown_dimension_leaves_token_reads_unchanged(tmp_path: Path) -> None:
    import warnings

    known = _file(tmp_path, "truss2d.ladruno", [_TRUSS_2D_NDF3],
                  strip_dim=False)
    with LadrunoReader(known) as r:
        want = {tok: _read(r, tok) for tok in ("globalForce", "basicForce")}
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    path = _file(unknown, "truss2d.ladruno", [_TRUSS_2D_NDF3],
                 drop_model=True)
    with LadrunoReader(path) as r, warnings.catch_warnings():
        warnings.simplefilter("error")
        for tok, slab in want.items():
            got = _read(r, tok)
            np.testing.assert_array_equal(got.values, slab.values)
            assert got.element_ids.tolist() == slab.element_ids.tolist()


def test_conflicting_coordinate_widths_refuse_the_model(tmp_path: Path) -> None:
    # Two stages whose MODEL coordinates disagree name no one dimension.
    path = _file(tmp_path, "truss2d.ladruno", [_TRUSS_2D_NDF3])
    with h5py.File(path, "r+") as f:
        f.copy(f[STAGE], "MODEL_STAGE[2]")
        nodes = f["MODEL_STAGE[2]/MODEL/NODES"]
        xy = nodes["COORDINATES"][...]
        del nodes["COORDINATES"]
        nodes.create_dataset(
            "COORDINATES", data=np.hstack([xy, np.zeros((len(xy), 1))]))
    with LadrunoReader(path) as r:
        with pytest.raises(SpatialDimUnknown, match=r"truss2d\.ladruno.*SPATIAL_DIM"):
            r.opensees_model()
        with pytest.raises(SpatialDimUnknown, match="SPATIAL_DIM"):
            _read(r, "nodal_resisting_force_z")


# ---------------------------------------------------------------------------
# Partition sets read each part's own dimension
# ---------------------------------------------------------------------------


def _parts(tmp_path: Path, *, drop_model: bool = False) -> list[Path]:
    return [
        _file(tmp_path, f"truss2d.part-{i}.ladruno",
              [("12-Truss[1:0:0]", _TRUSS_2D_NDF3[1], [i + 1])],
              drop_model=drop_model)
        for i in range(2)
    ]


def test_partitions_infer_2d_and_never_read_force_z(tmp_path: Path) -> None:
    with LadrunoMultiPartitionReader(_parts(tmp_path)) as m:
        listed = _elements(m)
        assert "nodal_resisting_force_z" not in listed
        for comp in ("nodal_resisting_force_x", "nodal_resisting_force_y"):
            assert comp in listed
            slab = _read(m, comp)
            order = np.argsort(slab.element_ids)
            assert slab.element_ids[order].tolist() == [1, 2]
            np.testing.assert_array_equal(
                slab.values[:, order], _expected([1, 2], comp, 2))
        with pytest.raises(NodalForceNotRecorded,
                           match="nodal_resisting_force_z"):
            _read(m, "nodal_resisting_force_z")


def test_partitions_of_unknown_dimension_refuse(tmp_path: Path) -> None:
    with LadrunoMultiPartitionReader(
            _parts(tmp_path, drop_model=True)) as m:
        with pytest.warns(UserWarning, match=r"truss2d\.part-\d\.ladruno"):
            listed = _elements(m)
        assert not [c for c in listed if c.startswith("nodal_resisting_")]
        with pytest.raises(SpatialDimUnknown, match="SPATIAL_DIM"):
            _read(m, "nodal_resisting_force_z")


# ---------------------------------------------------------------------------
# Review of #2087: axis-named columns, and a rank of unknown dimension
# ---------------------------------------------------------------------------


def test_unknown_dimension_reads_axis_named_beams(tmp_path: Path) -> None:
    import warnings

    path = _file(tmp_path, "truss2d.ladruno", [_BEAM_3D_AXIS],
                 drop_model=True)
    with LadrunoReader(path) as r, warnings.catch_warnings():
        warnings.simplefilter("error")
        listed = _elements(r)
        for comp in _FORCES + _MOMENTS:
            assert comp in listed
            np.testing.assert_array_equal(
                _read(r, comp).values, _expected([7], comp, 2),
                err_msg=comp)


def test_unknown_dimension_rank_fails_the_partition_read(
        tmp_path: Path) -> None:
    # Rank 0 infers 2-D from its MODEL; rank 1 has no MODEL. Rank 1's
    # elements are never dropped in silence: the set refuses, naming it.
    paths = [
        _file(tmp_path, f"truss2d.part-{i}.ladruno",
              [("12-Truss[1:0:0]", _TRUSS_2D_NDF3[1], [i + 1])],
              drop_model=(i == 1))
        for i in range(2)
    ]
    with LadrunoMultiPartitionReader(paths) as m:
        with pytest.raises(SpatialDimUnknown,
                           match=r"truss2d\.part-1\.ladruno.*SPATIAL_DIM"):
            _read(m, "nodal_resisting_force_x")
