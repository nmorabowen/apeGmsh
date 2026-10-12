"""#1746: results 1.2.0 ``element_index_space`` stamp on transcoded files.

The recorder transcoder labels element rows with the spec's FEM ids. A
``.out`` file does not say which emit wrote it, so the transcoder stamps
``element_index_space = "fem"`` only when the rows are provably FEM ids
whatever the emit era: no element rows, or every recorded FEM element id
is its own OpenSees tag in the embedded model's pairing (with the
model's ``lineage.fem_hash`` equal to the spec's ``fem_snapshot_id``).

The model is a real bridge build of a 2-D portal with the beam declared
first. Under ``element_tags="sequential"`` the beam (FEM 3, 4) keeps
tags 3, 4 while the columns (FEM 1, 2, 5, 6) move to tags 5..8, so one
build gives both a provable and an unprovable record. No solver runs:
the ``.out`` files are written by hand in the layout the emit names.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.results.readers._native import NativeReader
from apeGmsh.results.schema import _native
from apeGmsh.results.spec._emit import emit_logical
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.transcoders._recorder import RecorderTranscoder
from apeGmsh.results.writers._native import NativeWriter

from tests.fixtures.schema import RESULTS_CURRENT, RESULTS_PRIOR_MINOR

H, L = 3.0, 5.0
BEAM = (3, 4)
COLS = (1, 2, 5, 6)
T = 3
GLOBAL_FORCE = ("nodal_resisting_force_x", "nodal_resisting_force_y",
                "nodal_resisting_moment_z")
_SCHEMA_DOC = Path(__file__).resolve().parents[1] / "architecture" / "h5-schema.md"


# ---------------------------------------------------------------------------
# Model: one FEMData, two builds (sequential and fem tags)
# ---------------------------------------------------------------------------


def _portal_fem(length: float) -> Any:
    with apeGmsh(model_name="b5d_portal", verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(0.0, H, 0.0)
        p2 = geo.add_point(length, H, 0.0)
        p3 = geo.add_point(length, 0.0, 0.0)
        c1 = geo.add_line(p0, p1)
        bm = geo.add_line(p1, p2)
        c2 = geo.add_line(p3, p2)
        for c in (c1, bm, c2):
            gmsh.model.mesh.setTransfiniteCurve(c, 3)
        g.physical.add(1, [c1, c2], name="Cols")
        g.physical.add(1, [bm], name="Beam")
        g.mesh.generation.generate(dim=1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    assert tuple(fem.elements.select(pg="Beam").ids) == BEAM
    assert tuple(fem.elements.select(pg="Cols").ids) == COLS
    return fem


def _build(fem: Any, mode: str, path: Path) -> Path:
    ops = apeSees(fem, default_orientation=None, element_tags=mode)
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    # Beam first: under sequential tags the columns' tags move.
    ops.element.elasticBeamColumn(pg="Beam", transf=tr,
                                  A=0.02, E=2.0e11, Iz=3.0e-4)
    ops.element.elasticBeamColumn(pg="Cols", transf=tr,
                                  A=0.02, E=2.0e11, Iz=3.0e-4)
    ops.h5(str(path))
    return path


@pytest.fixture(scope="module")
def portal(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    fem = _portal_fem(L)
    root = tmp_path_factory.mktemp("b5d_models")
    out: dict[str, Any] = {"femdata": fem}
    for mode in ("sequential", "fem"):
        out[mode] = _build(fem, mode, root / f"{mode}.model.h5")
    # The same ids and identity tags on a longer portal: another FEMData.
    other = _portal_fem(L + 1.0)
    assert other.snapshot_id != fem.snapshot_id
    out["other_fem"] = _build(other, "fem", root / "other_fem.model.h5")
    # Guard the premise, so no case below is vacuous.
    seq = {r.fem_eid: r.tag for r in OpenSeesModel.from_h5(out["sequential"]).elements()}
    fem_tags = {r.fem_eid: r.tag for r in OpenSeesModel.from_h5(out["fem"]).elements()}
    other_tags = {r.fem_eid: r.tag for r in OpenSeesModel.from_h5(out["other_fem"]).elements()}
    assert all(seq[e] == e for e in BEAM)
    assert all(seq[e] != e for e in COLS)
    assert all(fem_tags[e] == e for e in BEAM + COLS)
    assert other_tags == fem_tags
    return out


# ---------------------------------------------------------------------------
# Spec + hand-written .out files
# ---------------------------------------------------------------------------


def _node_record() -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="nodes", name="disp", components=("displacement_x",),
        dt=None, n_steps=None, node_ids=np.array([2, 3], dtype=np.int64),
    )


def _element_record(ids: tuple[int, ...]) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="elements", name="force", components=GLOBAL_FORCE,
        dt=None, n_steps=None, element_ids=np.asarray(ids, dtype=np.int64),
        element_class_name="ElasticBeam2d",
    )


def _write_outputs(spec: ResolvedRecorderSpec, out_dir: Path) -> None:
    """Write each record's ``.out`` file: a time column, then data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    time = np.arange(1, T + 1, dtype=np.float64)
    for rec in spec.records:
        for lr in emit_logical(rec, output_dir=str(out_dir)):
            if rec.category == "nodes":
                n_cols = len(lr.target_ids) * len(lr.dofs or ())
            else:
                n_cols = len(lr.target_ids) * 6   # 2 nodes x (Fx, Fy, Mz)
            data = np.arange(T * n_cols, dtype=np.float64).reshape(T, n_cols)
            np.savetxt(lr.file_path, np.column_stack([time, data]))


def _transcode(
    tmp_path: Path, portal: dict[str, Any], *records: ResolvedRecorderRecord,
    build: "str | None", snapshot_id: "str | None" = None,
) -> Path:
    fem = portal["femdata"]
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=snapshot_id or str(fem.snapshot_id), records=records,
    )
    out_dir = tmp_path / "out"
    _write_outputs(spec, out_dir)
    target = tmp_path / "results.h5"
    RecorderTranscoder(
        spec, out_dir, target, fem,
        model_h5_src=None if build is None else portal[build],
    ).run()
    return target


def _verdict(path: Path) -> tuple[str, "str | None", bool]:
    with NativeReader(path) as r:
        assert r.stages(), "the transcoded file must still read"
        version = str(r._h5.attrs[_native.ATTR_RESULTS_SCHEMA_VERSION])
        return version, r.element_index_space, r.element_index_verified


# ---------------------------------------------------------------------------
# Stamped: provable whatever the emit era
# ---------------------------------------------------------------------------


def test_node_only_file_is_stamped(tmp_path: Path, portal: dict) -> None:
    path = _transcode(tmp_path, portal, _node_record(), build="sequential")
    assert _verdict(path) == (RESULTS_CURRENT, "fem", True)


def test_identity_tag_element_file_is_stamped(tmp_path: Path, portal: dict) -> None:
    path = _transcode(tmp_path, portal, _node_record(),
                      _element_record(COLS), build="fem")
    assert _verdict(path) == (RESULTS_CURRENT, "fem", True)


def test_sequential_build_whose_recorded_ids_are_their_tags_is_stamped(
    tmp_path: Path, portal: dict,
) -> None:
    # The beam keeps tags 3, 4: a translated and an untranslated emit
    # wrote the same -ele list, so the rows are provably FEM ids.
    path = _transcode(tmp_path, portal, _element_record(BEAM),
                      build="sequential")
    assert _verdict(path) == (RESULTS_CURRENT, "fem", True)


# ---------------------------------------------------------------------------
# Not stamped: unverified, but the file reads
# ---------------------------------------------------------------------------


def test_sequential_tag_element_file_is_not_stamped(
    tmp_path: Path, portal: dict,
) -> None:
    # Columns 1, 2, 5, 6 are tags 5..8: only a translated emit records
    # them, and the files cannot say which emit ran. Fails if the stamp
    # is written unconditionally.
    path = _transcode(tmp_path, portal, _node_record(),
                      _element_record(COLS), build="sequential")
    assert _verdict(path) == (RESULTS_CURRENT, None, False)


def test_one_unprovable_id_withholds_the_stamp(tmp_path: Path, portal: dict) -> None:
    path = _transcode(tmp_path, portal, _element_record(BEAM + (1,)),
                      build="sequential")
    assert _verdict(path) == (RESULTS_CURRENT, None, False)


def test_no_model_is_not_stamped(tmp_path: Path, portal: dict) -> None:
    path = _transcode(tmp_path, portal, _node_record(), build=None)
    assert _verdict(path) == (RESULTS_CURRENT, None, False)


def test_model_from_another_fem_is_not_stamped(tmp_path: Path, portal: dict) -> None:
    # Identity pairing, but the spec was not resolved against the
    # FEMData the model was built from: nothing is proven.
    path = _transcode(tmp_path, portal, _element_record(COLS), build="fem",
                      snapshot_id="0" * 32)
    assert _verdict(path) == (RESULTS_CURRENT, None, False)


def test_prior_minor_file_reads_unverified(tmp_path: Path, portal: dict) -> None:
    # A 1.1.0 file has no stamp; even with one, the version alone never
    # verifies, and a stamp outside the value set never verifies.
    path = _transcode(tmp_path, portal, _node_record(), build="sequential")
    with h5py.File(path, "r+") as f:
        f.attrs[_native.ATTR_RESULTS_SCHEMA_VERSION] = RESULTS_PRIOR_MINOR
        del f.attrs[_native.ATTR_ELEMENT_INDEX_SPACE]
    assert _verdict(path) == (RESULTS_PRIOR_MINOR, None, False)
    with h5py.File(path, "r+") as f:
        f.attrs[_native.ATTR_ELEMENT_INDEX_SPACE] = "ops"
    assert _verdict(path) == (RESULTS_PRIOR_MINOR, "ops", False)


# ---------------------------------------------------------------------------
# The value set: code and doc cannot drift
# ---------------------------------------------------------------------------


def test_fem_is_the_only_value() -> None:
    assert _native.ELEMENT_INDEX_SPACES == ("fem",)


@pytest.mark.parametrize("value", ["ops", "FEM", "", "sequential"])
def test_writer_refuses_any_other_value(tmp_path: Path, value: str) -> None:
    with NativeWriter(tmp_path / "r.h5") as w:
        w.open()
        with pytest.raises(ValueError, match="element_index_space"):
            w.stamp_element_index_space(value)
    with h5py.File(tmp_path / "r.h5", "r") as f:
        assert _native.ATTR_ELEMENT_INDEX_SPACE not in f.attrs


def test_schema_doc_lists_exactly_the_value_set() -> None:
    text = _SCHEMA_DOC.read_text(encoding="utf-8")
    start = text.index("### `element_index_space`")
    section = text[start:text.index("\n### ", start + 1)]
    documented = tuple(re.findall(r'^\| `"([^"]*)"` \|', section, re.M))
    assert documented == _native.ELEMENT_INDEX_SPACES


# ---------------------------------------------------------------------------
# The cache keys on the model: one call's stamp is never served to another
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order", [("fem", "sequential"), ("sequential", "fem")],
    ids=["identity_then_sequential", "sequential_then_identity"],
)
def test_cached_stamp_follows_each_calls_model(
    tmp_path: Path, portal: dict, order: tuple[str, str],
) -> None:
    # Same spec, same recorder files, same cache_root; only model= differs.
    from apeGmsh.results import Results

    fem = portal["femdata"]
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=str(fem.snapshot_id),
        records=(_element_record(COLS),),
    )
    out_dir = tmp_path / "out"
    _write_outputs(spec, out_dir)
    cache_root = tmp_path / "cache"
    expected = {"fem": True, "sequential": False}
    for build in order:
        results = Results.from_recorders(
            spec, out_dir, fem=fem, cache_root=cache_root,
            model=OpenSeesModel.from_h5(portal[build]),
        )
        with NativeReader(results._path) as r:
            assert r.element_index_verified is expected[build], build


def test_cache_separates_models_without_model_hash_by_fem_hash(
    tmp_path: Path, portal: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A composed model carries lineage.model_hash = None. Two such models
    # with the same (fem_eid, tag) pairing but different FEMData must not
    # share a cached file: only the one built from the spec's FEMData
    # proves the stamp.
    import dataclasses

    from apeGmsh.results import Results

    original = OpenSeesModel.lineage
    monkeypatch.setattr(
        OpenSeesModel, "lineage",
        property(lambda self: dataclasses.replace(
            original.fget(self), model_hash=None)),
    )
    fem = portal["femdata"]
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=str(fem.snapshot_id),
        records=(_element_record(COLS),),
    )
    out_dir = tmp_path / "out"
    _write_outputs(spec, out_dir)
    cache_root = tmp_path / "cache"
    paths = []
    for build, verified in (("fem", True), ("other_fem", False)):
        model = OpenSeesModel.from_h5(portal[build])
        assert model.lineage.model_hash is None
        results = Results.from_recorders(
            spec, out_dir, fem=fem, cache_root=cache_root, model=model,
        )
        paths.append(results._path)
        with NativeReader(results._path) as r:
            assert r.element_index_verified is verified, build
    assert paths[0] != paths[1]
