"""Element-id filters through ``ElementTagTranslator.read_translation`` (#2066).

An MPCO or ``.ladruno`` file keys element results by ops tag; the
results API speaks ``fem_eid``. The translator used to decide a whole
filter at once: a filter with one unpaired id was passed through
untranslated, so its paired ``fem_eid``s were read as ops tags. With
(fem, ops) = (2, 1), (3, 2), the filter ``[2, 999]`` read ops tag 2
(fem 3's row) and labelled it 2, and ``[3, 999]`` returned nothing.
A paired id whose ops tag the file did not record read back empty with
no warning.

The right answer, held here on the translator and through both readers,
single file and partition set:

* each paired ``fem_eid`` reads its own ops tag, whatever else the
  filter holds, and comes back labelled with its own ``fem_eid``;
* an id that is no element of the archive, or an element the files did
  not record, is not read and :class:`WarnElementFilterUnread` names it;
* an archive that does not describe the files (an unrelated stub
  ``model_h5=``) leaves both arrays untouched.

The value oracle is the reader's own untranslated read: each ops tag's
rows, compared exactly. The live two-element MPCO decks that pin the
same answers against statics are in ``test_results_mpco_nodal_real.py``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results.readers._tag_translation import (
    ElementTagTranslator,
    WarnElementFilterUnread,
)

# (fem, ops) = (2, 1), (3, 2): fem 2 is also the ops tag of fem 3, so a
# fem id read as an ops tag lands on another element's row.
_FEM_TO_OPS = {2: 1, 3: 2}
_OPS_TO_FEM = {o: f for f, o in _FEM_TO_OPS.items()}

_LADRUNO = Path(__file__).resolve().parent / "fixtures" / "ladruno"


def _translator(*, ops_only: "tuple[int, ...]" = ()) -> ElementTagTranslator:
    return ElementTagTranslator(_FEM_TO_OPS, _OPS_TO_FEM, ops_only)


def _silent(fn, *args):
    """Call ``fn`` and fail on any :class:`WarnElementFilterUnread`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnElementFilterUnread)
        return fn(*args)


# ── The translator ─────────────────────────────────────────────────────


class TestReadTranslation:

    @pytest.mark.parametrize("fem", [2, 3])
    def test_mixed_filter_translates_each_paired_id(self, fem: int) -> None:
        t = _translator()
        t.declare_recorded(np.array([1, 2]))
        with pytest.warns(WarnElementFilterUnread, match=r"\[999\]"):
            bucket, back = t.read_translation(np.array([fem, 999]))
        np.testing.assert_array_equal(bucket, [_FEM_TO_OPS[fem]])
        np.testing.assert_array_equal(back(bucket), [fem])

    def test_an_unpaired_id_is_never_read_as_an_ops_tag(self) -> None:
        """1 is fem 2's ops tag but no fem id: reading it as an ops tag
        is the mislabel this slice removes."""
        t = _translator()
        t.declare_recorded(np.array([1, 2]))
        with pytest.warns(WarnElementFilterUnread, match=r"\[1\] are no element"):
            bucket, _ = t.read_translation(np.array([1]))
        assert bucket.size == 0

    def test_a_paired_id_the_files_did_not_record_warns(self) -> None:
        t = _translator()
        t.declare_recorded(np.array([2]))  # fem 3 only
        with pytest.warns(WarnElementFilterUnread, match=r"\[2\] are elements"):
            bucket, _ = t.read_translation(np.array([2]))
        assert bucket.size == 0
        bucket, back = _silent(t.read_translation, np.array([3]))
        np.testing.assert_array_equal(bucket, [2])
        np.testing.assert_array_equal(back(bucket), [3])

    def test_a_sentinel_element_reads_by_its_ops_tag(self) -> None:
        """An element the archive emitted without a mesh id (an interface
        zeroLength) is addressed by its ops tag, beside paired ones."""
        t = _translator(ops_only=(7,))
        t.declare_recorded(np.array([1, 2, 7]))
        bucket, back = _silent(t.read_translation, np.array([7, 2]))
        np.testing.assert_array_equal(bucket, [7, 1])
        np.testing.assert_array_equal(back(np.array([1, 7])), [2, 7])
        none, back = _silent(t.read_translation, None)
        assert none is None
        # Unfiltered, a bucket mixing both kinds relabels id by id.
        np.testing.assert_array_equal(back(np.array([7, 2, 1])), [7, 3, 2])

    def test_an_unrelated_stub_archive_passes_both_arrays_through(self) -> None:
        """The stub ``model_h5=`` pattern: the file records ops tags the
        archive does not hold, so nothing translates and nothing warns."""
        t = _translator()
        t.declare_recorded(np.array([1, 5]))
        assert not t.describes_files()
        for ids in ([2], [1], [2, 999]):
            bucket, back = _silent(t.read_translation, np.array(ids))
            np.testing.assert_array_equal(bucket, ids)
            np.testing.assert_array_equal(back(np.array([1, 5])), [1, 5])
        _, back = _silent(t.read_translation, None)
        np.testing.assert_array_equal(back(np.array([1, 5])), [1, 5])

    def test_partitions_declare_into_one_union(self) -> None:
        """Each partition records part of the model; an element held by
        another partition is not unrecorded."""
        t = _translator()
        t.declare_recorded(np.array([1]))
        t.declare_recorded(np.array([2]))
        bucket, _ = _silent(t.read_translation, np.array([2, 3]))
        np.testing.assert_array_equal(bucket, [1, 2])

    def test_a_read_before_any_file_is_declared_raises(self) -> None:
        with pytest.raises(RuntimeError, match="declare_recorded"):
            _translator().read_translation(np.array([2]))

    def test_an_ops_tag_both_paired_and_unpaired_raises(self) -> None:
        with pytest.raises(ValueError, match="both paired"):
            _translator(ops_only=(1,))

    def test_from_model_keeps_sentinel_rows_as_ops_only(self) -> None:
        model = SimpleNamespace(elements=lambda: [
            SimpleNamespace(tag=1, fem_eid=2),
            SimpleNamespace(tag=9, fem_eid=-1),
        ])
        t = ElementTagTranslator.from_model(model)
        t.declare_recorded(np.array([1, 9]))
        assert t.describes_files()
        bucket, _ = _silent(t.read_translation, np.array([2, 9]))
        np.testing.assert_array_equal(bucket, [1, 9])


# ── Through the readers ────────────────────────────────────────────────


def _rows(slab) -> "dict[int, np.ndarray]":
    """``{element id: its value columns}`` of a station/Gauss slab."""
    index = np.asarray(slab.element_index)
    values = np.asarray(slab.values)
    return {int(e): values[:, index == e] for e in np.unique(index)}


def _ladruno_reader(*names: str):
    from apeGmsh.results.readers._ladruno import LadrunoReader
    from apeGmsh.results.readers._ladruno_multi import (
        LadrunoMultiPartitionReader,
    )
    paths = [_LADRUNO / n for n in names]
    if len(paths) == 1:
        return LadrunoReader(paths[0])
    return LadrunoMultiPartitionReader(paths)


def _ladruno_read(reader, ids):
    return reader.read_line_stations(
        reader.stages()[0].id, "axial_force", element_ids=ids,
    )


def _mpco_files(tmp_path: Path, parts: "list[list[int]]") -> "list[Path]":
    """One synthetic ``.mpco`` per entry of ``parts``, each recording the
    listed ops tags in one Gauss stress bucket; an element's value is
    ``ops_tag * 100 + step * 10 + component``."""
    from apeGmsh.opensees._response_catalog import IntRule, flatten, lookup
    from tests.test_results_mpco_element_mock import (
        _add_stress_bucket,
        _create_mpco_skeleton,
    )

    layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
    stem = "run.mpco" if len(parts) == 1 else "run.part-{}.mpco"
    paths = []
    for k, tags in enumerate(parts):
        path = tmp_path / stem.format(k)
        f, stage = _create_mpco_skeleton(
            path, node_ids=np.arange(1, 5, dtype=np.int64),
            node_coords=np.eye(4, 3), n_steps=2, dt=0.5,
        )
        try:
            ops = np.array(tags, dtype=np.int64)
            per_comp = {
                name: (ops.reshape(1, -1, 1) * 100.0
                       + np.arange(2.0).reshape(2, 1, 1) * 10.0 + c)
                for c, name in enumerate(layout.component_layout)
            }
            _add_stress_bucket(
                f[stage],
                bracket_key=f"{layout.class_tag}-FourNodeTetrahedron[300:0:0]",
                class_tag=layout.class_tag, int_rule=IntRule.Tet_GL_1,
                element_ids=ops, flat_data=flatten(per_comp, layout),
                n_gauss_points=1,
            )
        finally:
            f.close()
        paths.append(path)
    return paths


def _mpco_reader(paths: "list[Path]"):
    from apeGmsh.results.readers._mpco import MPCOReader
    from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader
    if len(paths) == 1:
        return MPCOReader(paths[0])
    return MPCOMultiPartitionReader(paths)


def _mpco_read(reader, ids):
    return reader.read_gauss(
        reader.stages()[0].id, "stress_xx", element_ids=ids,
    )


_LADRUNO_SETS = {
    "ladruno-single": ("truss2d.ladruno",),
    "ladruno-partitions": ("truss2d.part-0.ladruno", "truss2d.part-1.ladruno"),
}


@pytest.fixture(params=["ladruno-single", "ladruno-partitions",
                        "mpco-single", "mpco-partitions"])
def case(request, tmp_path: Path):
    """``(open_reader, read)`` for one format and file layout, recording
    ops tags 1 and 2."""
    name = request.param
    if name.startswith("ladruno"):
        return (lambda: _ladruno_reader(*_LADRUNO_SETS[name])), _ladruno_read
    parts = [[1, 2]] if name == "mpco-single" else [[1], [2]]
    paths = _mpco_files(tmp_path, parts)
    return (lambda: _mpco_reader(paths)), _mpco_read


def _oracle(case) -> "dict[int, np.ndarray]":
    """Each ops tag's rows from the untranslated reader."""
    open_reader, read = case
    reader = open_reader()
    try:
        rows = _rows(read(reader, None))
    finally:
        reader.close()
    assert sorted(rows) == [1, 2], rows
    return rows


def _translated(case):
    open_reader, read = case
    reader = open_reader()
    reader.attach_tag_map(_translator())
    return reader, read


class TestReaders:

    def test_unfiltered_read_relabels_every_row(self, case) -> None:
        oracle = _oracle(case)
        reader, read = _translated(case)
        try:
            got = _rows(_silent(read, reader, None))
        finally:
            reader.close()
        assert sorted(got) == [2, 3]
        for fem, ops in _FEM_TO_OPS.items():
            np.testing.assert_array_equal(got[fem], oracle[ops])

    @pytest.mark.parametrize("fem", [2, 3])
    def test_mixed_filter_returns_the_paired_element(self, case, fem) -> None:
        oracle = _oracle(case)
        reader, read = _translated(case)
        try:
            with pytest.warns(WarnElementFilterUnread, match=r"\[999\]"):
                got = _rows(read(reader, np.array([fem, 999])))
        finally:
            reader.close()
        assert sorted(got) == [fem]
        np.testing.assert_array_equal(got[fem], oracle[_FEM_TO_OPS[fem]])

    def test_a_filter_across_partitions_reads_both_silently(self, case) -> None:
        oracle = _oracle(case)
        reader, read = _translated(case)
        try:
            got = _rows(_silent(read, reader, np.array([3, 2])))
        finally:
            reader.close()
        assert sorted(got) == [2, 3]
        for fem, ops in _FEM_TO_OPS.items():
            np.testing.assert_array_equal(got[fem], oracle[ops])

    def test_an_unpaired_id_reads_nothing_and_warns(self, case) -> None:
        reader, read = _translated(case)
        try:
            with pytest.warns(WarnElementFilterUnread, match=r"\[1\] are no"):
                got = _rows(read(reader, np.array([1])))
        finally:
            reader.close()
        assert got == {}


# ── A lone unpaired id that equals an emitted ops tag (G18 shape) ──────
#
# The filter holds one id, no paired id beside it, and that id is the ops
# tag of another element (on G18's solid+frame model, ``ids=[12]`` names
# a never-emitted Base quad and returned strut fem 5's forces, ops tag
# 12). Each case pairs every recorded ops tag ``t`` with fem ``t + 100``,
# so ``[t]`` is no fem id; the untranslated reader proves ``t`` holds
# rows, and the translated read must return none of them and warn.

_RESULTS = Path(__file__).resolve().parent / "fixtures" / "results"
_OFFSET = 100


def _elements(reader, comp, ids):
    return reader.read_elements(reader.stages()[0].id, comp, element_ids=ids)


def _stations(reader, comp, ids):
    return reader.read_line_stations(
        reader.stages()[0].id, comp, element_ids=ids,
    )


def _gauss(reader, comp, ids):
    return reader.read_gauss(reader.stages()[0].id, comp, element_ids=ids)


def _slab_ids(slab) -> "list[int]":
    ids = getattr(slab, "element_ids", None)
    if ids is None:
        ids = slab.element_index
    return sorted({int(e) for e in np.asarray(ids)})


_COLLISIONS = {
    # name: (open the untranslated reader, read, component, ops tag)
    "ladruno-nodal-forces": (
        lambda tmp: _ladruno_reader("truss2d.ladruno"),
        _elements, "basicForce", 2),
    "ladruno-nodal-forces-partitions": (
        lambda tmp: _ladruno_reader(*_LADRUNO_SETS["ladruno-partitions"]),
        _elements, "basicForce", 2),
    "ladruno-line-stations": (
        lambda tmp: _ladruno_reader("truss2d.ladruno"),
        _stations, "axial_force", 1),
    "ladruno-gauss": (
        lambda tmp: _ladruno_reader("quad2d.ladruno"),
        _gauss, "stress_xx", 1),
    "mpco-nodal-forces": (
        lambda tmp: _mpco_reader([_RESULTS / "elasticFrame.mpco"]),
        _elements, "nodal_resisting_force_local_x", 5),
    "mpco-line-stations": (
        lambda tmp: _mpco_reader([_RESULTS / "elasticFrame.mpco"]),
        _stations, "axial_force", 5),
    "mpco-gauss": (
        lambda tmp: _mpco_reader(_mpco_files(tmp, [[1, 2]])),
        _gauss, "stress_xx", 2),
    "mpco-gauss-partitions": (
        lambda tmp: _mpco_reader(_mpco_files(tmp, [[1], [2]])),
        _gauss, "stress_xx", 2),
}


@pytest.mark.parametrize("name", sorted(_COLLISIONS))
def test_a_lone_unpaired_id_equal_to_an_ops_tag_reads_nothing(
    name: str, tmp_path: Path,
) -> None:
    open_reader, read, comp, ops = _COLLISIONS[name]
    reader = open_reader(tmp_path)
    try:
        own = read(reader, comp, np.array([ops]))
        assert _slab_ids(own) == [ops], "fixture: ops tag holds no rows"
        # Pair every element the file holds: its embedded snapshot plus
        # this component's rows (the synthetic MPCO lists no elements).
        recorded = set(_slab_ids(read(reader, comp, None)))
        embedded = reader.fem()
        if embedded is not None:
            for group in embedded.elements:
                recorded.update(int(e) for e in np.asarray(group.ids))
    finally:
        reader.close()
    pairing = {t + _OFFSET: t for t in recorded}
    translator = ElementTagTranslator(
        pairing, {o: f for f, o in pairing.items()},
    )
    reader = open_reader(tmp_path)
    reader.attach_tag_map(translator)
    try:
        with pytest.warns(
            WarnElementFilterUnread, match=rf"\[{ops}\] are no element",
        ):
            got = read(reader, comp, np.array([ops]))
        assert _slab_ids(got) == []
        # The element itself still reads under its own fem id.
        mine = _silent(read, reader, comp, np.array([ops + _OFFSET]))
        assert _slab_ids(mine) == [ops + _OFFSET]
        np.testing.assert_array_equal(
            np.asarray(mine.values), np.asarray(own.values),
        )
    finally:
        reader.close()


@pytest.mark.parametrize("fmt", ["ladruno", "mpco"])
def test_an_element_the_file_did_not_record_warns(fmt: str, tmp_path) -> None:
    """One file that records ops tag 1 only (fem 2); fem 3 is in the
    archive but not in the file."""
    if fmt == "ladruno":
        reader, read = _ladruno_reader("truss2d.part-0.ladruno"), _ladruno_read
    else:
        reader, read = _mpco_reader(_mpco_files(tmp_path, [[1]])), _mpco_read
    reader.attach_tag_map(_translator())
    try:
        with pytest.warns(WarnElementFilterUnread, match=r"\[3\] are elements"):
            got = _rows(read(reader, np.array([3])))
        assert got == {}
        assert sorted(_rows(_silent(read, reader, np.array([2])))) == [2]
    finally:
        reader.close()


# ── Partition sets open in linear time ─────────────────────────────────


def test_declaring_many_partitions_is_linear() -> None:
    """A union per declaration re-sorted the whole set for every
    partition: 96 partitions of 80k tags took 116 s (np240: ~12 min).
    Declaring is now an append, and the union is built once, on the
    first read. 200 partitions of 10k tags: well under a second each
    way here; the old code took tens of seconds."""
    import time

    n_parts, per_part = 200, 10_000
    pairing = {t + 1: t for t in range(n_parts * per_part)}
    t = ElementTagTranslator(pairing, {o: f for f, o in pairing.items()})
    parts = [
        np.arange(k * per_part, (k + 1) * per_part, dtype=np.int64)
        for k in range(n_parts)
    ]
    start = time.perf_counter()
    for part in parts:
        t.declare_recorded(part)
    declared = time.perf_counter() - start
    assert declared < 2.0, f"declaring {n_parts} partitions took {declared:.1f} s"
    start = time.perf_counter()
    assert t.describes_files()
    bucket, _ = _silent(t.read_translation, np.array([1, n_parts * per_part]))
    first_read = time.perf_counter() - start
    np.testing.assert_array_equal(bucket, [0, n_parts * per_part - 1])
    assert first_read < 10.0, f"the first read took {first_read:.1f} s"


# ── Results says reads are fem-keyed when they are ─────────────────────


def test_a_mixed_paired_and_sentinel_bucket_reads_fem_keyed(tmp_path) -> None:
    """One bucket records paired elements (ops 1, 2) and an element the
    archive emitted without a mesh id (ops 7, an interface zeroLength).
    The reads relabel the paired rows to fem ids, so
    ``_element_reads_fem_keyed`` must say so: the viewer builds its
    scene from the answer. Judged by the paired tags alone it said
    False, and the viewer joined fem-keyed rows on ops tags."""
    from apeGmsh.results import Results

    reader = _mpco_reader(_mpco_files(tmp_path, [[1, 2, 7]]))
    reader.attach_tag_map(ElementTagTranslator(
        {11: 1, 12: 2}, {1: 11, 2: 12}, ops_only=(7,),
    ))
    r = Results(reader, model=SimpleNamespace())
    try:
        assert r._element_reads_fem_keyed() is True
        gauss = r.stage(r.stages[0].id).elements.gauss
        slab = _silent(lambda: gauss.get(component="stress_xx"))
        got = _rows(slab)
        assert sorted(got) == [7, 11, 12]
        # value = ops_tag * 100 + step * 10 (component 0): each fem id
        # carries its own ops tag's rows.
        for label, ops in ((11, 1), (12, 2), (7, 7)):
            np.testing.assert_array_equal(
                got[label][:, 0], [ops * 100.0, ops * 100.0 + 10.0],
            )
    finally:
        reader.close()
