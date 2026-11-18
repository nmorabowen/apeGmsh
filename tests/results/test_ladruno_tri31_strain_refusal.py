""".ladruno strain read never drops Tri31 elements in silence (#2895).

The ``.ladruno`` side of #2893. Tri31 answers no element-level strain:
``ops.eleResponse(eid, "strains")`` is empty on it, so a ``strains``
record leaves its elements out of every strain bucket.
``tests/fixtures/simplex_gp/tri.ladruno`` is such a file: element 1 is a
Tri31 listed under ``MODEL/ELEMENTS/33-Tri31[100:0]`` with a ``stresses``
bucket and no ``strains`` one. Before the fix a whole-model ``strain_xx``
read returned elements [2, 3, 4, 5] and a read of element 1 returned
shape (1, 0), both without a warning.

The oracle is the file itself: the read refuses exactly when a Tri31
element in scope gets no row back, and a file whose Tri31 rows are
recorded per Gauss point (``material.strain``) reads them back.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._mpco import MPCOStrainNotRecorded
from apeGmsh.results.readers._tag_translation import ElementTagTranslator

TRI = (Path(__file__).resolve().parents[1] / "fixtures" / "simplex_gp"
       / "tri.ladruno")
STAGE = "MODEL_STAGE[1]"
ELEMENTS = f"{STAGE}/MODEL/ELEMENTS"
ON_ELEMENTS = f"{STAGE}/RESULTS/ON_ELEMENTS"
TRI31_BUCKET = "33-Tri31[100:0:0]"


def _read(reader, component: str, eids=None):
    with warnings.catch_warnings():
        # The reader's provenance warnings are not under test here; the
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
    with LadrunoReader(TRI) as r:
        with pytest.raises(MPCOStrainNotRecorded) as info:
            _read(r, "strain_xx")
    msg = str(info.value)
    assert "Tri31 element(s) [1]" in msg
    assert "element_class_name='Tri31'" in msg


def test_tri31_element_ids_read_refuses() -> None:
    with LadrunoReader(TRI) as r:
        with pytest.raises(MPCOStrainNotRecorded,
                           match=r"Tri31 element\(s\) \[1\]"):
            _read(r, "strain_xx", [1])


def test_read_of_other_classes_only_is_silent() -> None:
    with LadrunoReader(TRI) as r, warnings.catch_warnings():
        warnings.simplefilter("error")
        slab = r.read_gauss(
            r.stages()[0].id, "strain_xx",
            element_ids=np.array([2, 3, 4, 5], dtype=np.int64),
        )
    assert _ids(slab) == [2, 3, 4, 5]


def _with_material_strain(tmp_path: Path) -> Path:
    """``tri.ladruno`` plus a ``material.strain`` bucket for Tri31 1, as
    a recorder declared with ``element_class_name="Tri31"`` writes it."""
    path = _copy(tmp_path, "tri_material_strain.ladruno")
    with h5py.File(path, "r+") as f:
        on = f[ON_ELEMENTS]
        dst = on.create_group("material.strain")
        f.copy(on["stresses"][TRI31_BUCKET], dst, name=TRI31_BUCKET)
        bucket = dst[TRI31_BUCKET]
        bucket.attrs["DISPLAY_NAME"] = np.array([b"material.strain"])
        bucket["COLUMN_MAP"].attrs["COMP_NAMES"] = np.array(
            [b"eps11,eps22,eps12"])
        bucket["DATA"][...] = np.array([[[0.25, -0.5, 0.125]]])
    return path


def test_tri31_rows_under_material_strain_read_without_refusing(
    tmp_path: Path,
) -> None:
    path = _with_material_strain(tmp_path)
    with LadrunoReader(path) as r:
        slab = _read(r, "strain_xx")
        assert _ids(slab) == [1, 2, 3, 4, 5]
        one = _read(r, "strain_yy", [1])
    assert np.asarray(one.values).tolist() == [[-0.5]]


# =====================================================================
# Seams
# =====================================================================

def test_stress_read_is_unaffected() -> None:
    with LadrunoReader(TRI) as r:
        assert _ids(_read(r, "stress_xx")) == [1, 2, 3, 4, 5]


def test_component_tri31_cannot_answer_is_unaffected() -> None:
    """``curvature_xx`` routes through the strain token, but no Tri31
    record would hold it, so the read loses nothing."""
    with LadrunoReader(TRI) as r:
        assert _ids(_read(r, "curvature_xx")) == []


def test_out_of_plane_strain_read_refuses() -> None:
    """``strain_zz`` is the plane layout's out-of-plane column, which a
    Tri31 record could hold."""
    with LadrunoReader(TRI) as r:
        with pytest.raises(MPCOStrainNotRecorded):
            _read(r, "strain_zz")


def test_empty_element_ids_is_silent() -> None:
    with LadrunoReader(TRI) as r:
        assert np.asarray(_read(r, "strain_xx", []).values).shape == (1, 0)


def test_tri31_beside_a_class_that_answers_strains_refuses() -> None:
    """Shared namespace: FourNodeQuad 5 has rows, Tri31 1 has none."""
    with LadrunoReader(TRI) as r:
        with pytest.raises(MPCOStrainNotRecorded, match=r"\[1\]"):
            _read(r, "strain_xx", [1, 5])


def test_file_listing_no_elements_is_silent(tmp_path: Path) -> None:
    """A stage with no ``MODEL/ELEMENTS`` lists no Tri31, so none is in
    scope; the buckets still read."""
    path = _copy(tmp_path, "tri_no_listing.ladruno")
    with h5py.File(path, "r+") as f:
        del f[ELEMENTS]
    with LadrunoReader(path) as r:
        assert _ids(_read(r, "strain_xx")) == [2, 3, 4, 5]


@pytest.fixture
def unparseable_listing(tmp_path: Path) -> Path:
    """``tri.ladruno`` plus a listing whose name names no class,
    element 99."""
    path = _copy(tmp_path, "tri_weird_listing.ladruno")
    with h5py.File(path, "r+") as f:
        grp = f[ELEMENTS].create_group("weird_listing")
        grp.create_dataset(
            "CONNECTIVITY", data=np.array([[99, 1, 2, 3]], dtype=np.int32),
        )
    return path


def test_unparseable_listing_leaves_a_stress_read_silent(
    unparseable_listing: Path,
) -> None:
    with LadrunoReader(unparseable_listing) as r:
        assert _ids(_read(r, "stress_xx")) == [1, 2, 3, 4, 5]


def test_unparseable_listing_out_of_scope_leaves_strain_silent(
    unparseable_listing: Path,
) -> None:
    with LadrunoReader(unparseable_listing) as r:
        assert _ids(_read(r, "strain_xx", [2])) == [2]


def test_unparseable_listing_in_scope_refuses(
    unparseable_listing: Path,
) -> None:
    """Its class cannot be told, so its elements could be Tri31."""
    with LadrunoReader(unparseable_listing) as r:
        with pytest.raises(ValueError, match=r"names no class.*\[99\]"):
            _read(r, "strain_xx", [99])


def test_refusal_names_fem_ids_under_a_tag_map() -> None:
    fem_to_ops = {100 + t: t for t in range(1, 6)}
    with LadrunoReader(TRI) as r:
        r.attach_tag_map(ElementTagTranslator(
            fem_to_ops, {o: f for f, o in fem_to_ops.items()},
        ))
        with pytest.raises(MPCOStrainNotRecorded, match=r"\[101\]"):
            _read(r, "strain_xx", [101])
        assert _ids(_read(r, "strain_xx", [102, 105])) == [102, 105]


def test_mode_stage_refusal_keeps_its_precedence(tmp_path: Path) -> None:
    """A gauss read of a mode stage refuses on the stage kind, not on
    the Tri31 strain rows the host stage lacks."""
    path = _copy(tmp_path, "tri_mode.ladruno")
    with h5py.File(path, "r+") as f:
        grp = f[f"{STAGE}/RESULTS/ON_NODES"].create_group(
            "MODES_OF_VIBRATION(U)")
        grp.attrs["COMPONENTS"] = np.array([b"UX,UY"])
        grp.create_dataset("ID", data=np.array([[1]], dtype=np.int32))
        mode = grp.create_dataset("DATA/STEP_0/MODE_0",
                                  data=np.array([[1.0, 0.0]]))
        mode.attrs["LAMBDA"] = np.array([4.0])
    with LadrunoReader(path) as r:
        modes = [s for s in r.stages() if s.kind == "mode"]
        assert len(modes) == 1
        with pytest.raises(ValueError, match="is a mode stage") as info:
            r.read_gauss(modes[0].id, "strain_xx")
    assert not isinstance(info.value, MPCOStrainNotRecorded)


def _keep_only(path: Path, keep: set[int], *, part: int) -> None:
    """Drop every element listing and result bucket outside ``keep``,
    and stamp the file as partition ``part`` of two."""
    with h5py.File(path, "r+") as f:
        info = f["INFO"].attrs
        info["PARTITIONED"] = np.array([1], dtype=np.int32)
        info["NUM_PARTITIONS"] = np.array([2], dtype=np.int32)
        info["PARTITION_ID"] = np.array([part], dtype=np.int32)
        for name in list(f[ELEMENTS]):
            if int(f[ELEMENTS][name]["CONNECTIVITY"][0, 0]) not in keep:
                del f[ELEMENTS][name]
        for token in f[ON_ELEMENTS].values():
            for name in list(token):
                if int(token[name]["ID"][0, 0]) not in keep:
                    del token[name]


def test_multi_partition_refuses_only_where_tri31_lives(
    tmp_path: Path,
) -> None:
    part0 = _copy(tmp_path, "tri.part-0.ladruno")
    part1 = _copy(tmp_path, "tri.part-1.ladruno")
    _keep_only(part0, {1, 5}, part=0)
    _keep_only(part1, {2, 3, 4}, part=1)
    with LadrunoMultiPartitionReader([part0, part1]) as r:
        with pytest.raises(MPCOStrainNotRecorded, match=r"\[1\]"):
            _read(r, "strain_xx")
        assert _ids(_read(r, "strain_xx", [2, 3, 4, 5])) == [2, 3, 4, 5]
    with LadrunoReader(part1) as r:
        assert _ids(_read(r, "strain_xx")) == [2, 3, 4]


@pytest.fixture
def tri31_part_without_element_results(tmp_path: Path) -> "list[Path]":
    """Part 0 lists only Tri31 1 and records no element results at all
    (no ``ON_ELEMENTS``; its time axis is on a node record); part 1 holds
    elements 2 to 5. The partitioned reader drops a part that raises
    ``MissingElementResults`` (#2895 F1)."""
    part0 = _copy(tmp_path, "tri.part-0.ladruno")
    part1 = _copy(tmp_path, "tri.part-1.ladruno")
    _keep_only(part0, {1}, part=0)
    _keep_only(part1, {2, 3, 4, 5}, part=1)
    with h5py.File(part0, "r+") as f:
        del f[ON_ELEMENTS]
        disp = f[f"{STAGE}/RESULTS/ON_NODES"].create_group("DISPLACEMENT")
        disp.create_dataset("TIME", data=np.array([1.0]))
    return [part0, part1]


@pytest.mark.parametrize("eids", [None, [1], [1, 2]])
def test_tri31_in_a_part_without_element_results_refuses(
    tri31_part_without_element_results: "list[Path]", eids,
) -> None:
    with LadrunoMultiPartitionReader(tri31_part_without_element_results) as r:
        with pytest.raises(MPCOStrainNotRecorded,
                           match=r"Tri31 element\(s\) \[1\]"):
            _read(r, "strain_xx", eids)


def test_part_without_element_results_still_reads_other_classes(
    tri31_part_without_element_results: "list[Path]",
) -> None:
    with LadrunoMultiPartitionReader(tri31_part_without_element_results) as r:
        assert _ids(_read(r, "strain_xx", [2, 3, 4, 5])) == [2, 3, 4, 5]
        assert _ids(_read(r, "stress_xx")) == [2, 3, 4, 5]


def test_material_strain_bucket_without_the_component_reads_silently(
    tmp_path: Path,
) -> None:
    """#2895 F2: Tri31 1 is recorded per Gauss point under
    ``material.strain`` with its three plane columns, so ``strain_zz``
    finds no column for it. That is the answer a stress read gives for a
    missing column, not a drop: the element is recorded, and the read
    must not tell the user to record what they already recorded."""
    path = _with_material_strain(tmp_path)
    with LadrunoReader(path) as r, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slab = r.read_gauss(r.stages()[0].id, "strain_zz")
        stress = r.read_gauss(r.stages()[0].id, "stress_zz")
    assert _ids(slab) == _ids(stress) == []
