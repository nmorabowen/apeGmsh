"""#2329: ``Results.from_ladruno(model_h5=)`` binds the archived FEMData.

A ``.ladruno``'s own ``MODEL`` group carries nodes and elements but no
physical groups or labels. With ``model_h5=`` the reader binds the FEMData
archived in the model, as :meth:`Results.from_mpco` does, so ``pg=`` and
``label=`` reads work; a model of another mesh warns
``ModelFemMismatchWarning`` naming the archive and binds the file's own
``MODEL`` group.

Oracle: the committed springs capture (``zl_springs.mpco``) read with
``from_mpco`` beside the same model. The ``.ladruno`` is that capture
converted (``convert_mpco``), so every group read must equal the MPCO read
at ``rtol=1e-12``.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

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
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.results import Results, convert_mpco
from apeGmsh.results._bind import ModelFemMismatchWarning
from apeGmsh.results.readers._ladruno_element_io import SpatialDimUnknown
from tests.test_results_mpco_to_ladruno import SPRINGS, _bridge_copy, _stages

_TRUSS = Path(__file__).resolve().parents[2] / "fixtures" / "ladruno"

_GROUPS = {"One": [100], "Three": [200]}
_LABEL = "springs.three"
_DISP = ("displacement_x", "displacement_y", "displacement_z")


def _fem(offset: float, tag_of: "dict[int, int] | None" = None) -> FEMData:
    """The springs capture's nodes and elements, two PGs and one label.

    ``offset`` shifts every node in ``x``: a non-zero one is another mesh
    with the same ids (the ``holds a different mesh`` mismatch).
    ``tag_of`` renumbers the elements (fem id -> the id written): a
    FEMData-only archive pairs no tags, so its element ids must be the
    capture's own.
    """
    eid_of = tag_of if tag_of is not None else {}
    with h5py.File(SPRINGS, "r") as f:
        model = f["MODEL_STAGE[1]/MODEL"]
        ids = np.asarray(model["NODES/ID"][...], dtype=np.int64).ravel()
        xyz = np.asarray(model["NODES/COORDINATES"][...], dtype=np.float64)
        rows = {int(r[0]): r[1:] for d in model["ELEMENTS"].values()
                for r in np.asarray(d[...], dtype=np.int64)}
    xyz = xyz + np.array([offset, 0.0, 0.0])

    def group(name: str, eids: "list[int]") -> dict:
        nids = np.unique(np.array([rows[e] for e in eids]))
        return {"name": name, "node_ids": nids,
                "node_coords": xyz[np.searchsorted(ids, nids)],
                "element_ids": np.asarray([eid_of.get(e, e) for e in eids],
                                          dtype=np.int64)}

    pgs = PhysicalGroupSet({(1, 100 + k): group(n, e)
                            for k, (n, e) in enumerate(_GROUPS.items())})
    labels = LabelSet({(1, 900): group(_LABEL, _GROUPS["Three"])})
    eids = [e for es in _GROUPS.values() for e in es]
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=len(eids))
    lines = ElementGroup(element_type=info,
                         ids=np.asarray([eid_of.get(e, e) for e in eids],
                                        dtype=np.int64),
                         connectivity=np.array([rows[e] for e in eids]))
    return FEMData(
        nodes=NodeComposite(node_ids=ids, node_coords=xyz, physical=pgs,
                            labels=labels),
        elements=ElementComposite(groups={1: lines}, physical=pgs,
                                  labels=labels),
        info=MeshInfo(n_nodes=ids.size, n_elems=len(eids), bandwidth=1,
                      types=[info]),
        model_name="ladruno_model_h5_groups",
    )


def _model(path: Path, *, offset: float = 0.0) -> Path:
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir

    ops = apeSees(_fem(offset))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1000.0)
    ops.element.ZeroLength(pg="One", mat_dirs=(ZeroLengthMatDir(material=mat, dof=1),))
    ops.element.ZeroLength(pg="Three", mat_dirs=tuple(
        ZeroLengthMatDir(material=mat, dof=d) for d in (1, 2, 3)))
    _stages(ops, [("load", "static")])
    ops.h5(str(path))
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Path]":
    tmp = tmp_path_factory.mktemp("ladruno_model_h5_groups")
    model = _model(tmp / "model.h5")
    other = _model(tmp / "other.h5", offset=10.0)
    mpco = _bridge_copy(SPRINGS, model, tmp / "src.mpco")
    (ladruno,) = convert_mpco(mpco, tmp / "src.ladruno", model_h5=model)
    # A FEMData-only archive (fem.to_h5: no /opensees zone, ndm 0) of the
    # same mesh, its elements numbered as the capture's (the raw-deck route).
    tag_of = {int(r.fem_eid): int(r.tag)
              for r in OpenSeesModel.from_h5(model).elements()}
    fem_only = tmp / "fem_only.h5"
    _fem(0.0, tag_of).to_h5(str(fem_only))
    return {"model": model, "other": other, "mpco": mpco, "ladruno": ladruno,
            "fem_only": fem_only}


def _read(res: Results, component: str, **sel: str) -> "tuple[np.ndarray, np.ndarray]":
    s = res.stage(res.stages[0].id).nodes.get(component=component, **sel)
    return np.asarray(s.node_ids), np.asarray(s.values)


@pytest.mark.parametrize("sel", [{"pg": "One"}, {"pg": "Three"},
                                 {"label": _LABEL}], ids=["pg-One", "pg-Three",
                                                          "label"])
def test_group_reads_equal_from_mpco(files, sel) -> None:
    """The archived groups bind: each read equals the MPCO read, silently."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = Results.from_mpco(str(files["mpco"]), model_h5=str(files["model"]))
        res = Results.from_ladruno(str(files["ladruno"]),
                                   model_h5=str(files["model"]))
    try:
        assert res.fem is not None
        assert sorted(res.fem.nodes.physical.names()) == sorted(_GROUPS)
        name, = sel.values()
        expect = (res.fem.nodes.labels if "label" in sel
                  else res.fem.nodes.physical).node_ids(name)
        for comp in _DISP:
            ids, vals = _read(res, comp, **sel)
            ref_ids, ref_vals = _read(ref, comp, **sel)
            np.testing.assert_array_equal(ids, ref_ids)
            np.testing.assert_array_equal(np.sort(ids),
                                          np.sort(np.asarray(expect, dtype=np.int64)))
            np.testing.assert_allclose(vals, ref_vals, rtol=1e-12, atol=0.0)
        # The values are not all zero: the comparison has something to compare.
        assert np.abs(_read(ref, "displacement_x", **sel)[1]).max() > 0.0
    finally:
        ref.close()
        res.close()


def test_a_model_of_another_mesh_warns_by_name_and_binds_the_file(files) -> None:
    """``model_h5=`` of another mesh: one warning naming it, no groups bound."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = Results.from_ladruno(str(files["ladruno"]),
                                   model_h5=str(files["other"]))
    try:
        mismatch = [w for w in caught
                    if issubclass(w.category, ModelFemMismatchWarning)]
        assert len(mismatch) == 1, [str(w.message) for w in caught]
        msg = str(mismatch[0].message)
        assert f"model_h5={str(files['other'])!r}" in msg
        assert "holds a different mesh" in msg
        with pytest.raises(KeyError, match="One"):
            _read(res, "displacement_x", pg="One")
    finally:
        res.close()


def test_fem_wins_over_model_h5(files) -> None:
    """``fem=`` is bound unchecked, as in ``from_mpco``."""
    mine = _fem(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = Results.from_ladruno(str(files["ladruno"]), fem=mine,
                                   model_h5=str(files["other"]))
    try:
        assert res.fem is mine
    finally:
        res.close()


def test_without_model_h5_the_file_model_group_binds(files) -> None:
    """Unchanged: no ``model_h5=``, the file's MODEL group, no groups."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = Results.from_ladruno(str(files["ladruno"]))
    try:
        assert res.fem is not None
        with h5py.File(SPRINGS, "r") as f:  # the converted file's nodes
            file_ids = np.asarray(f["MODEL_STAGE[1]/MODEL/NODES/ID"][...]).ravel()
        np.testing.assert_array_equal(np.sort(np.asarray(res.fem.nodes.ids)),
                                      np.sort(file_ids))
        assert res.fem.nodes.physical.names() == []
        with pytest.raises(KeyError, match="One"):
            _read(res, "displacement_x", pg="One")
    finally:
        res.close()


# ---------------------------------------------------------------------------
# A FEMData-only archive (fem.to_h5): no ndm, so the file's dimension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sel", [{"pg": "One"}, {"pg": "Three"},
                                 {"label": _LABEL}], ids=["pg-One", "pg-Three",
                                                          "label"])
def test_a_femdata_archive_binds_the_groups(files, sel) -> None:
    """``fem.to_h5`` records ndm 0; the file's dimension drives the check."""
    assert OpenSeesModel.from_h5(files["fem_only"]).ndm == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = Results.from_mpco(str(files["mpco"]), model_h5=str(files["model"]))
        res = Results.from_ladruno(str(files["ladruno"]),
                                   model_h5=str(files["fem_only"]))
    try:
        assert res.fem is not None
        assert sorted(res.fem.nodes.physical.names()) == sorted(_GROUPS)
        for comp in _DISP:
            ids, vals = _read(res, comp, **sel)
            ref_ids, ref_vals = _read(ref, comp, **sel)
            np.testing.assert_array_equal(ids, ref_ids)
            np.testing.assert_allclose(vals, ref_vals, rtol=1e-12, atol=0.0)
    finally:
        ref.close()
        res.close()


def _truss_archive(path: Path, z: float) -> Path:
    """The truss2d fixture's mesh (nodes 1-3 on x, elements 1-2) as a
    FEMData-only archive drawn on the plane ``z``, with PG ``Right``
    (element 2) and label ``truss.mid`` (node 2)."""
    ids = np.array([1, 2, 3], dtype=np.int64)
    xyz = np.array([[0.0, 0.0, z], [1.0, 0.0, z], [2.0, 0.0, z]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=2)
    lines = ElementGroup(element_type=info, ids=np.array([1, 2], dtype=np.int64),
                         connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64))
    pgs = PhysicalGroupSet({(1, 7): {
        "name": "Right", "node_ids": ids[1:], "node_coords": xyz[1:],
        "element_ids": np.array([2], dtype=np.int64)}})
    labels = LabelSet({(0, 8): {"name": "truss.mid", "node_ids": ids[1:2],
                                "node_coords": xyz[1:2]}})
    FEMData(
        nodes=NodeComposite(node_ids=ids, node_coords=xyz, physical=pgs,
                            labels=labels),
        elements=ElementComposite(groups={1: lines}, physical=pgs,
                                  labels=labels),
        info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=1, types=[info]),
        model_name="truss2d_parts",
    ).to_h5(str(path))
    return path


@pytest.mark.parametrize("sel, nodes", [({"pg": "Right"}, [2, 3]),
                                        ({"label": "truss.mid"}, [2])],
                         ids=["pg", "label"])
def test_a_partitioned_set_binds_a_femdata_archive_on_an_offset_plane(
    tmp_path, sel, nodes,
) -> None:
    """Two parts, a 2-D file, an archive at ``z = 5``: the file's dimension
    (2, from the part set) compares ``x, y`` only, so the archive binds; a
    3-D comparison would call it another mesh."""
    archive = _truss_archive(tmp_path / "fem_only.h5", z=5.0)
    parts = [str(_TRUSS / f"truss2d.part-{k}.ladruno") for k in (0, 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = Results.from_ladruno(parts, model_h5=str(archive))
        ref = Results.from_ladruno(parts)
    try:
        ids, vals = _read(res, "displacement_x", **sel)
        ref_ids, ref_vals = _read(ref, "displacement_x", ids=nodes)
        np.testing.assert_array_equal(ids, nodes)
        np.testing.assert_array_equal(ref_ids, nodes)
        np.testing.assert_allclose(vals, ref_vals, rtol=1e-12, atol=0.0)
        assert np.abs(ref_vals).max() > 0.0
    finally:
        res.close()
        ref.close()


def test_a_file_without_a_dimension_refuses_a_femdata_archive_by_name(
    tmp_path,
) -> None:
    """No dimension to compare over: refuse, naming the file and archive."""
    src = tmp_path / "truss2d.ladruno"
    shutil.copyfile(_TRUSS / "truss2d.ladruno", src)
    with h5py.File(src, "r+") as f:
        f["INFO"].attrs["SPATIAL_DIM"] = np.array([7], dtype=np.int32)
    archive = _truss_archive(tmp_path / "fem_only.h5", z=0.0)
    with pytest.raises(SpatialDimUnknown) as ei:
        Results.from_ladruno(str(src), model_h5=str(archive))
    msg = str(ei.value)
    assert str(src) in msg and f"model_h5={str(archive)!r}" in msg
    assert "records no ndm" in msg
