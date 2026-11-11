"""An MPCO strain read never drops Tri31 elements in silence (#2893).

Tri31 answers no element-level strain: ``ops.eleResponse(eid,
"strains")`` is empty on it, so an MPCO ``-E strains`` record leaves its
elements out of every strain bucket. ``tests/fixtures/simplex_gp/tri.mpco``
is such a file: element 1 is a Tri31 with a ``stresses`` bucket and no
``strains`` one. Before the fix a whole-model ``strain_xx`` read returned
elements [2, 3, 4, 5] and a read of element 1 returned shape (1, 0), both
without a warning.

The oracle is the file itself: the read refuses exactly when a Tri31
element in scope has no row in any strain bucket, and a file whose Tri31
rows are recorded per Gauss point (``material.strain``) reads them back.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._mpco import MPCOReader, MPCOStrainNotRecorded
from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader
from apeGmsh.results.readers._tag_translation import ElementTagTranslator

TRI = Path(__file__).resolve().parents[1] / "fixtures" / "simplex_gp" / "tri.mpco"
STAGE = "MODEL_STAGE[1]"
ELEMENTS = f"{STAGE}/MODEL/ELEMENTS"
ON_ELEMENTS = f"{STAGE}/RESULTS/ON_ELEMENTS"
TRI31_MODEL = "33-Tri31[100:0]"
TRI31_BUCKET = "33-Tri31[100:0:0]"


def _read(reader, component: str, eids=None):
    with warnings.catch_warnings():
        # The readers' provenance warnings are not under test here; the
        # refusal is an exception, which this does not hide.
        warnings.simplefilter("ignore")
        return reader.read_gauss(
            reader.stages()[0].id, component,
            element_ids=None if eids is None else np.asarray(eids, np.int64),
        )


def _ids(slab) -> list[int]:
    return np.unique(np.asarray(slab.element_index)).tolist()


def _copy(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    shutil.copyfile(TRI, path)
    return path


# =====================================================================
# The card's four cases
# =====================================================================

def test_whole_model_strain_read_refuses_naming_tri31() -> None:
    with MPCOReader(TRI) as r:
        with pytest.raises(MPCOStrainNotRecorded) as info:
            _read(r, "strain_xx")
    msg = str(info.value)
    assert "Tri31 element(s) [1]" in msg
    assert "element_class_name='Tri31'" in msg


def test_tri31_element_ids_read_refuses() -> None:
    with MPCOReader(TRI) as r:
        with pytest.raises(MPCOStrainNotRecorded, match=r"Tri31 element\(s\) \[1\]"):
            _read(r, "strain_xx", [1])


def test_read_of_other_classes_only_is_silent() -> None:
    with MPCOReader(TRI) as r, warnings.catch_warnings():
        warnings.simplefilter("error")
        slab = r.read_gauss(
            r.stages()[0].id, "strain_xx",
            element_ids=np.array([2, 3, 4, 5], dtype=np.int64),
        )
    assert _ids(slab) == [2, 3, 4, 5]


def test_tri31_rows_under_material_strain_read_without_refusing(
    tmp_path: Path,
) -> None:
    """A recorder declared with ``element_class_name="Tri31"`` records
    per Gauss point under ``material.strain``; those rows read back."""
    path = _copy(tmp_path, "tri_material_strain.mpco")
    with h5py.File(path, "r+") as f:
        on = f[ON_ELEMENTS]
        dst = on.create_group("material.strain")
        f.copy(on["stresses"][TRI31_BUCKET], dst, name=TRI31_BUCKET)
        meta = dst[TRI31_BUCKET]["META"]
        meta["COMPONENTS"][0] = b"0.1.4.eta11,eta22,eta12"
        step = dst[TRI31_BUCKET]["DATA"]["STEP_0"]
        step[...] = np.array([[0.25, -0.5, 0.125]])
    with MPCOReader(path) as r:
        slab = _read(r, "strain_xx")
        assert _ids(slab) == [1, 2, 3, 4, 5]
        one = _read(r, "strain_yy", [1])
    assert np.asarray(one.values).tolist() == [[-0.5]]


# =====================================================================
# Seams
# =====================================================================

def test_stress_read_is_unaffected() -> None:
    with MPCOReader(TRI) as r:
        assert _ids(_read(r, "stress_xx")) == [1, 2, 3, 4, 5]


def test_component_tri31_cannot_answer_is_unaffected() -> None:
    """``curvature_xx`` routes through the strain token, but no Tri31
    record would hold it, so the read loses nothing."""
    with MPCOReader(TRI) as r:
        assert _ids(_read(r, "curvature_xx")) == []


def test_out_of_plane_strain_read_refuses() -> None:
    """``strain_zz`` is the plane layout's out-of-plane column, which a
    Tri31 record could hold."""
    with MPCOReader(TRI) as r:
        with pytest.raises(MPCOStrainNotRecorded):
            _read(r, "strain_zz")


def test_empty_element_ids_is_silent() -> None:
    with MPCOReader(TRI) as r:
        assert np.asarray(_read(r, "strain_xx", []).values).shape == (1, 0)


def test_tri31_beside_a_class_that_answers_strains_refuses() -> None:
    """Shared namespace: FourNodeQuad 5 has rows, Tri31 1 has none."""
    with MPCOReader(TRI) as r:
        with pytest.raises(MPCOStrainNotRecorded, match=r"\[1\]"):
            _read(r, "strain_xx", [1, 5])


def test_file_listing_no_elements_is_silent(tmp_path: Path) -> None:
    """A stage with no ``MODEL/ELEMENTS`` lists no Tri31, so none is in
    scope; the buckets still read."""
    path = _copy(tmp_path, "tri_no_listing.mpco")
    with h5py.File(path, "r+") as f:
        del f[ELEMENTS]
    with MPCOReader(path) as r:
        assert _ids(_read(r, "strain_xx")) == [2, 3, 4, 5]


def test_refusal_names_fem_ids_under_a_tag_map() -> None:
    fem_to_ops = {100 + t: t for t in range(1, 6)}
    with MPCOReader(TRI) as r:
        r.attach_tag_map(ElementTagTranslator(
            fem_to_ops, {o: f for f, o in fem_to_ops.items()},
        ))
        with pytest.raises(MPCOStrainNotRecorded, match=r"\[101\]"):
            _read(r, "strain_xx", [101])
        assert _ids(_read(r, "strain_xx", [102, 105])) == [102, 105]


def _keep_only(path: Path, keep: set[int]) -> None:
    """Drop every element listing and result bucket outside ``keep``."""
    with h5py.File(path, "r+") as f:
        for name in list(f[ELEMENTS]):
            if int(f[ELEMENTS][name][0, 0]) not in keep:
                del f[ELEMENTS][name]
        for token in f[ON_ELEMENTS].values():
            for name in list(token):
                if int(token[name]["ID"][0, 0]) not in keep:
                    del token[name]


def test_multi_partition_refuses_only_where_tri31_lives(
    tmp_path: Path,
) -> None:
    part0 = _copy(tmp_path, "tri.part-0.mpco")
    part1 = _copy(tmp_path, "tri.part-1.mpco")
    _keep_only(part0, {1, 5})
    _keep_only(part1, {2, 3, 4})
    with MPCOMultiPartitionReader([part0, part1]) as r:
        with pytest.raises(MPCOStrainNotRecorded, match=r"\[1\]"):
            _read(r, "strain_xx")
        assert _ids(_read(r, "strain_xx", [2, 3, 4, 5])) == [2, 3, 4, 5]
    with MPCOReader(part1) as r:
        assert _ids(_read(r, "strain_xx")) == [2, 3, 4]
