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
from apeGmsh.results import Results, convert_mpco
from apeGmsh.results._bind import ModelFemMismatchWarning
from tests.test_results_mpco_to_ladruno import SPRINGS, _bridge_copy, _stages

_GROUPS = {"One": [100], "Three": [200]}
_LABEL = "springs.three"
_DISP = ("displacement_x", "displacement_y", "displacement_z")


def _fem(offset: float) -> FEMData:
    """The springs capture's nodes and elements, two PGs and one label.

    ``offset`` shifts every node in ``x``: a non-zero one is another mesh
    with the same ids (the ``holds a different mesh`` mismatch).
    """
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
                "element_ids": np.asarray(eids, dtype=np.int64)}

    pgs = PhysicalGroupSet({(1, 100 + k): group(n, e)
                            for k, (n, e) in enumerate(_GROUPS.items())})
    labels = LabelSet({(1, 900): group(_LABEL, _GROUPS["Three"])})
    eids = [e for es in _GROUPS.values() for e in es]
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=len(eids))
    lines = ElementGroup(element_type=info, ids=np.asarray(eids, dtype=np.int64),
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
    return {"model": model, "other": other, "mpco": mpco, "ladruno": ladruno}


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
