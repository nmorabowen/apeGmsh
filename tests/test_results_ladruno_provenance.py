"""``stamp_provenance`` writes the ADR 0122 D3 pairing stamp (#2099).

The stamp goes into ``INFO/PROVENANCE`` of a closed ``.ladruno`` (fork
schema ``ladruno_schema_v1`` §7b.1, fork doc PR nmorabowen/OpenSees #992).
The oracles here are the contract itself:

* six lowercase attributes, each a scalar variable-length UTF-8 string;
* ``model_hash`` / ``fem_hash`` byte-equal to ``<model_h5>/meta/lineage``;
* every part of a partitioned set stamped;
* every refusal raises :class:`ProvenanceStampRefused` and leaves every
  part file byte-identical (a SHA-256 of each file before and after);
* nothing outside the six attributes changes, UPPERCASE attributes in
  ``PROVENANCE`` included;
* a re-stamp with identical values leaves the file byte-identical, and
  ``overwrite`` is the only way to re-pair a file;
* ``stage_names`` only when the count and kinds match, else a
  :class:`StageNamesOmittedWarning`;
* ``element_index_space="fem"`` only when every recorded element id is a
  FEM id the archive pairs with itself;
* ``opensees_backend`` / ``opensees_build_sha`` (§7b.1, fork doc PR
  nmorabowen/OpenSees #997) always written: ``"ladruno"`` on a recorder
  file, a build only when the caller passes it, ``"unknown"`` exactly where
  §7b.1 says so, and any stamped value (``"unknown"`` included) kept across
  a default re-stamp.

The stamp must never ask an OpenSees backend (#2208 F1):
:func:`_no_backend` makes every resolver entry point raise, for every test.

The part files are the committed ``truss2d.part-0/1`` fixtures, copied to
a temporary directory. They predate the recorder's empty
``INFO/PROVENANCE`` group, so :func:`_with_provenance` adds it, as the
current recorder writes it at open. The ``model.h5`` is written by the
real bridge from a synthetic two-bar FEMData. The bridge tags its two bars
2 and 3, so the copies are renumbered to match (:func:`_renumber`); the
one-file fixture keeps ids 1 and 2.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator

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
from apeGmsh.results.writers import (
    ProvenanceStampRefused,
    StageNamesOmittedWarning,
    stamp_provenance,
)
from apeGmsh.opensees.emitter import live as _live
from apeGmsh.results.readers._ladruno_provenance import PROVENANCE_ATTRS

FIXTURES = Path(__file__).parent / "fixtures" / "ladruno"
PROV = "INFO/PROVENANCE"
FORK_SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"
ENGINE = ("opensees_backend", "opensees_build_sha")
CONVERTED = "apeGmsh mpco->ladruno 9.9"


@pytest.fixture(autouse=True)
def _no_backend(monkeypatch: pytest.MonkeyPatch) -> "list[str]":
    """Every resolver entry point raises: the stamp may not bind a backend."""
    asked: list[str] = []

    def refuse(name: str):
        def call(*_a, **_k):
            asked.append(name)
            raise AssertionError(f"stamp_provenance asked the backend ({name})")
        return call

    for name in ("_get_ops", "get_ops", "get_backend_info",
                 "get_backend_build"):
        monkeypatch.setattr(_live, name, refuse(name))
    return asked


# ---------------------------------------------------------------------------
# model.h5: two bars on nodes 1-2-3, as the truss2d fixtures record them
# ---------------------------------------------------------------------------


def _truss_fem(element_ids: "tuple[int, int]" = (2, 3)) -> FEMData:
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=2)
    group = ElementGroup(
        element_type=info, ids=np.asarray(element_ids, dtype=np.int64),
        connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64),
    )
    pg = {
        (1, 1): {"name": "Bars", "node_ids": node_ids, "node_coords": coords,
                 "element_ids": group.ids},
        (0, 2): {"name": "Base", "node_ids": node_ids[:1],
                 "node_coords": coords[:1]},
    }
    nodes = NodeComposite(node_ids=node_ids, node_coords=coords,
                          physical=PhysicalGroupSet(pg), labels=LabelSet({}))
    elements = ElementComposite(groups={1: group},
                                physical=PhysicalGroupSet(pg),
                                labels=LabelSet({}))
    return FEMData(nodes=nodes, elements=elements,
                   info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=2,
                                 types=[info]))


def _write_model(
    path: Path,
    stages: "tuple[tuple[str, str], ...]" = (("load", "Static"),),
    *,
    element_ids: "tuple[int, int]" = (2, 3),
) -> Path:
    """Bridge-written ``model.h5``; ``stages`` are ``(name, analysis)``."""
    from apeGmsh.opensees import apeSees

    ops = apeSees(_truss_fem(element_ids), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=2.0e11)
    ops.element.Truss(pg="Bars", A=1e-4, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    for name, analysis in stages:
        with ops.stage(name=name) as s:
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.load(pg="Bars", forces=(1.0, 0.0))
            transient = analysis == "Transient"
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
                algorithm=ops.algorithm.Linear(),
                integrator=(ops.integrator.Newmark(gamma=0.5, beta=0.25)
                            if transient
                            else ops.integrator.LoadControl(dlam=0.25)),
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.RCM(),
                system=ops.system.BandGeneral(),
                analysis=(ops.analysis.Transient() if transient
                          else ops.analysis.Static()),
            )
            if transient:
                s.run(n_increments=4, dt=0.01)
            else:
                s.run(n_increments=4)
    ops.h5(str(path))
    return path


def _lineage(model_h5: Path) -> "dict[str, str]":
    with h5py.File(model_h5, "r") as f:
        return {k: str(f["meta/lineage"].attrs[k])
                for k in ("model_hash", "fem_hash")}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("provenance_models")
    _write_model(d / "one_static.h5")
    _write_model(d / "renamed.h5", (("gravity", "Static"),))
    _write_model(d / "two_stages.h5",
                 (("load", "Static"), ("more", "Static")))
    _write_model(d / "transient.h5", (("shake", "Transient"),))
    _write_model(d / "vanilla.h5", ())
    _write_model(d / "offset_ids.h5", element_ids=(11, 12))
    return d


# ---------------------------------------------------------------------------
# .ladruno part files
# ---------------------------------------------------------------------------


def _with_provenance(*paths: Path) -> None:
    """Add the empty ``INFO/PROVENANCE`` group today's recorder writes."""
    for p in paths:
        with h5py.File(p, "r+") as f:
            f["INFO"].create_group("PROVENANCE")


def _renumber(*paths: Path) -> None:
    """Element ids 1, 2 -> 2, 3: the ops tags the bridge gives the bars."""
    for p in paths:
        with h5py.File(p, "r+") as f:
            stage = f["MODEL_STAGE[1]"]
            for ds in (stage["MODEL/ELEMENTS/12-Truss[1:0]/CONNECTIVITY"],
                       stage["RESULTS/ON_ELEMENTS/basicForce/12-Truss[1:0:0]/ID"]):
                ids = ds[...]
                ids[..., 0] += 1
                ds[...] = ids


def _copy_parts(tmp_path: Path) -> "list[Path]":
    out = []
    for k in (0, 1):
        dst = tmp_path / f"truss2d.part-{k}.ladruno"
        shutil.copy(FIXTURES / f"truss2d.part-{k}.ladruno", dst)
        out.append(dst)
    _renumber(*out)
    return out


@pytest.fixture
def parts(tmp_path: Path) -> "list[Path]":
    out = _copy_parts(tmp_path)
    _with_provenance(*out)
    return out


def _sha(paths: "list[Path]") -> "list[str]":
    return [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def _snapshot(path: Path) -> "dict[str, object]":
    """Every attribute and dataset in the file, keyed by HDF5 path."""
    out: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: object) -> None:
            for key in obj.attrs:  # type: ignore[attr-defined]
                aid = obj.attrs.get_id(key)  # type: ignore[attr-defined]
                out[f"{name}@{key}"] = (str(aid.dtype), aid.shape,
                                        repr(obj.attrs[key]))  # type: ignore[attr-defined]
            if isinstance(obj, h5py.Dataset):
                out[name] = (str(obj.dtype), obj.shape, obj[...].tobytes())
        visit("/", f)
        f.visititems(visit)
    return out


def _provenance(path: Path) -> "dict[str, str]":
    with h5py.File(path, "r") as f:
        return {k: f[PROV].attrs[k] for k in f[PROV].attrs}


# ---------------------------------------------------------------------------
# The stamp
# ---------------------------------------------------------------------------


def test_every_part_carries_the_six_attributes(parts, model_dir) -> None:
    model = model_dir / "one_static.h5"
    out = stamp_provenance(parts[0], model, writer="apeGmsh test 0.0")
    assert out == parts
    lineage = _lineage(model)
    for part in parts:
        prov = _provenance(part)
        assert set(prov) == set(PROVENANCE_ATTRS)
        assert prov["model_hash"] == lineage["model_hash"]
        assert prov["fem_hash"] == lineage["fem_hash"]
        assert json.loads(prov["stage_names"]) == ["load"]
        assert prov["element_index_space"] == "fem"
        assert prov["unsupported"] == "[]"
        assert prov["writer"] == "apeGmsh test 0.0"


def test_attributes_are_scalar_vlen_utf8(parts, model_dir) -> None:
    stamp_provenance(parts[1], model_dir / "one_static.h5")
    for part in parts:
        with h5py.File(part, "r") as f:
            attrs = f[PROV].attrs
            for name in PROVENANCE_ATTRS:
                aid = attrs.get_id(name)
                info = h5py.check_string_dtype(aid.dtype)
                assert info is not None, name
                assert info.encoding == "utf-8" and info.length is None, name
                assert aid.shape == (), name


def test_hashes_are_byte_equal_to_meta_lineage(parts, model_dir) -> None:
    model = model_dir / "one_static.h5"
    stamp_provenance(parts[0], model)

    def raw(path: Path, group: str, name: str) -> bytes:
        with h5py.File(path, "r") as f:
            aid = f[group].attrs.get_id(name)
            assert h5py.check_string_dtype(aid.dtype).length is None
            value = f[group].attrs[name]
            return value.encode("utf-8") if isinstance(value, str) else bytes(value)

    for name in ("model_hash", "fem_hash"):
        want = raw(model, "meta/lineage", name)
        assert want  # a real hash, not an empty slot
        for part in parts:
            assert raw(part, PROV, name) == want


def test_default_writer_names_apegmsh_and_its_version(parts, model_dir) -> None:
    import apeGmsh

    stamp_provenance(parts[0], model_dir / "one_static.h5")
    assert _provenance(parts[0])["writer"] == f"apeGmsh {apeGmsh.__version__}"


def test_unsupported_is_a_compact_json_list(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "one_static.h5",
                     unsupported=("energy", "layered shells"))
    raw = _provenance(parts[0])["unsupported"]
    assert raw == '["energy","layered shells"]'


def test_nothing_else_in_the_file_changes(parts, model_dir) -> None:
    """Only the six lowercase names move; UPPERCASE ones stay byte-identical."""
    with h5py.File(parts[0], "r+") as f:
        f[PROV].attrs["RESULTS_HASH"] = np.bytes_(b"fork-reserved")
        f[PROV].attrs["MODEL_HASH"] = "fork-reserved-vlen"
    before = [_snapshot(p) for p in parts]
    stamp_provenance(parts[0], model_dir / "one_static.h5")
    for part, old in zip(parts, before):
        new = _snapshot(part)
        owned = {f"INFO/PROVENANCE@{n}" for n in PROVENANCE_ATTRS}
        assert set(new) - set(old) == owned
        assert {k: v for k, v in new.items() if k not in owned} == old


def test_single_file_is_stamped(tmp_path, model_dir) -> None:
    """The committed one-file fixture already carries an empty PROVENANCE."""
    path = tmp_path / "truss2d.ladruno"
    shutil.copy(FIXTURES / "truss2d.ladruno", path)
    assert stamp_provenance(path, model_dir / "one_static.h5") == [path]
    assert _provenance(path)["model_hash"] == \
        _lineage(model_dir / "one_static.h5")["model_hash"]


# ---------------------------------------------------------------------------
# Re-stamping
# ---------------------------------------------------------------------------


def test_restamp_with_identical_values_is_byte_identical(parts, model_dir) -> None:
    model = model_dir / "one_static.h5"
    stamp_provenance(parts[0], model)
    before = _sha(parts)
    stamp_provenance(parts[1], model)
    assert _sha(parts) == before


def test_restamp_same_model_rewrites_a_changed_attribute(parts, model_dir) -> None:
    model = model_dir / "one_static.h5"
    stamp_provenance(parts[0], model, writer="apeGmsh a")
    stamp_provenance(parts[0], model, writer="apeGmsh b")
    assert all(_provenance(p)["writer"] == "apeGmsh b" for p in parts)


def test_a_different_model_hash_needs_overwrite(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "one_static.h5")
    before = _sha(parts)
    other = model_dir / "offset_ids.h5"
    assert _lineage(other)["model_hash"] != \
        _lineage(model_dir / "one_static.h5")["model_hash"]
    with pytest.raises(ProvenanceStampRefused, match="already stamped"):
        stamp_provenance(parts[0], other)
    assert _sha(parts) == before

    stamp_provenance(parts[0], other, overwrite=True)
    for part in parts:
        prov = _provenance(part)
        assert prov["model_hash"] == _lineage(other)["model_hash"]
        # The re-pair drops an owned attribute the new stamp leaves out.
        assert "element_index_space" not in prov


# ---------------------------------------------------------------------------
# Refusals: raise before any write, every part byte-identical
# ---------------------------------------------------------------------------


def _refuses(parts: "list[Path]", model: Path, match: str, **kw) -> None:
    before = _sha(parts)
    with pytest.raises(ProvenanceStampRefused, match=match):
        stamp_provenance(parts[0], model, **kw)
    assert _sha(parts) == before


def test_refuses_a_part_without_provenance(tmp_path, model_dir) -> None:
    raw = _copy_parts(tmp_path)
    _with_provenance(raw[0])  # part 0 could be stamped; part 1 cannot
    _refuses(raw, model_dir / "one_static.h5", "no INFO/PROVENANCE")


def test_refuses_a_model_without_lineage(parts, model_dir, tmp_path) -> None:
    model = tmp_path / "no_lineage.h5"
    shutil.copy(model_dir / "one_static.h5", model)
    with h5py.File(model, "r+") as f:
        del f["meta/lineage"]
    _refuses(parts, model, "no /meta/lineage")


def test_refuses_parts_that_disagree_on_their_model(parts, model_dir) -> None:
    with h5py.File(parts[1], "r+") as f:
        xyz = f["MODEL_STAGE[1]/MODEL/NODES/COORDINATES"]
        xyz[0, 0] = 1.5  # the shared node 2 moves in part 1 only
    _refuses(parts, model_dir / "one_static.h5", "disagree on their MODEL")


def test_refuses_parts_from_different_runs(parts, model_dir) -> None:
    with h5py.File(parts[1], "r+") as f:
        f["INFO"].attrs["NUM_PARTITIONS"] = np.int32(3)
    _refuses(parts, model_dir / "one_static.h5", "NUM_PARTITIONS")


def test_refuses_a_foreign_file(parts, model_dir) -> None:
    with h5py.File(parts[1], "r+") as f:
        f["INFO"].attrs["GENERATOR"] = np.bytes_(b"MPCO")
    _refuses(parts, model_dir / "one_static.h5", "GENERATOR")


@pytest.mark.parametrize(("mode", "word"), [("r", "reading"), ("r+", "writing")])
def test_refuses_a_part_this_process_holds_open(parts, model_dir, mode, word) -> None:
    """In one process HDF5 shares the handle, so the lock cannot catch it."""
    before = _sha(parts)
    with h5py.File(parts[1], mode):
        with pytest.raises(ProvenanceStampRefused, match=f"open for {word}"):
            stamp_provenance(parts[0], model_dir / "one_static.h5")
    assert _sha(parts) == before


def test_refuses_a_part_a_results_reader_holds_open(parts, model_dir) -> None:
    from apeGmsh.results import Results

    with Results.from_ladruno(parts[0]):
        _refuses(parts, model_dir / "one_static.h5", "open for reading")


def test_refuses_a_part_another_process_holds_open(parts, model_dir) -> None:
    """HDF5 file locking: a reader in another process blocks the write."""
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, h5py; f = h5py.File(sys.argv[1], 'r'); "
         "print('open', flush=True); sys.stdin.read()", str(parts[1])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "open"
        _refuses(parts, model_dir / "one_static.h5", "will not open for writing")
    finally:
        assert holder.stdin is not None
        holder.stdin.close()
        holder.wait(timeout=30)


# ---------------------------------------------------------------------------
# stage_names: only on a count and kind match
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("model", "match"), [
    ("two_stages.h5", "1 MODEL_STAGE groups and the program 2 stages"),
    ("transient.h5", r"KINDs \['static'\] differ from the program's \['transient'\]"),
])
def test_stage_names_omitted_on_a_mismatch(parts, model_dir, model, match) -> None:
    with pytest.warns(StageNamesOmittedWarning, match=match) as record:
        stamp_provenance(parts[0], model_dir / model)
    assert len(record) == 2  # one per part
    for part in parts:
        prov = _provenance(part)
        assert "stage_names" not in prov
        assert prov["model_hash"] == _lineage(model_dir / model)["model_hash"]


def test_stage_names_absent_without_program_stages(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "vanilla.h5")  # silent under -W error
    assert all("stage_names" not in _provenance(p) for p in parts)


def test_stage_names_take_the_program_names(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "renamed.h5")
    assert json.loads(_provenance(parts[1])["stage_names"]) == ["gravity"]


def test_stage_names_override_replaces_the_names(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "one_static.h5",
                     stage_names=["präload"])
    raw = _provenance(parts[0])["stage_names"]
    assert raw == '["präload"]'  # compact UTF-8 JSON, not \\u escapes


def test_stage_names_override_with_the_wrong_count_is_omitted(parts, model_dir) -> None:
    with pytest.warns(StageNamesOmittedWarning, match="2 stage names for 1"):
        stamp_provenance(parts[0], model_dir / "one_static.h5",
                         stage_names=["a", "b"])
    assert all("stage_names" not in _provenance(p) for p in parts)


# ---------------------------------------------------------------------------
# element_index_space
# ---------------------------------------------------------------------------


def test_element_index_space_absent_when_ids_are_not_fem_ids(parts, model_dir) -> None:
    """FEM ids 11, 12 recorded under ops tags 2, 3: not FEM-keyed."""
    stamp_provenance(parts[0], model_dir / "offset_ids.h5")
    assert all("element_index_space" not in _provenance(p) for p in parts)


def test_element_index_space_absent_for_a_bridge_minted_element(
    parts, model_dir, tmp_path,
) -> None:
    """Element 3 carries ``fem_eid`` -1 in the archive: bridge-minted."""
    model = tmp_path / "minted.h5"
    shutil.copy(model_dir / "one_static.h5", model)
    with h5py.File(model, "r+") as f:
        meta = f["opensees/element_meta"]
        groups = [g for g in meta.values() if "fem_eids" in g]
        assert groups
        for g in groups:
            ids = np.asarray(g["ids"][...]).ravel()
            fem = np.asarray(g["fem_eids"][...]).copy()
            fem.reshape(-1)[ids == 3] = -1
            g["fem_eids"][...] = fem
    stamp_provenance(parts[0], model)
    assert "element_index_space" not in _provenance(parts[1])  # holds element 3
    assert _provenance(parts[0])["element_index_space"] == "fem"  # element 2 only


def test_inputs_are_type_checked(parts, model_dir) -> None:
    before = _sha(parts)
    with pytest.raises(TypeError, match="unsupported"):
        stamp_provenance(parts[0], model_dir / "one_static.h5",
                         unsupported="energy")
    with pytest.raises(TypeError, match="writer"):
        stamp_provenance(parts[0], model_dir / "one_static.h5", writer="")
    assert _sha(parts) == before


@pytest.fixture(autouse=True)
def _no_open_handles(tmp_path: Path) -> Iterator[None]:
    """Every test leaves no HDF5 handle open on its own files.

    Only files under this test's ``tmp_path`` count: earlier tests in the
    shared process may hold handles of their own.
    """
    yield
    import gc
    import os

    gc.collect()
    mine = os.path.normcase(str(tmp_path.resolve()))
    held = [
        os.fsdecode(fid.name)
        for fid in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)
        if os.path.normcase(os.path.realpath(os.fsdecode(fid.name))).startswith(mine)
    ]
    assert not held, held


# ---------------------------------------------------------------------------
# Review findings on #2111
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["FALSE", "false", "0", " False ",
                                   "BEST_EFFORT", "best_effort"])
def test_refuses_when_file_locking_is_disabled(parts, model_dir, monkeypatch,
                                               value) -> None:
    """Without the lock, another process's open handle is invisible."""
    monkeypatch.setenv("HDF5_USE_FILE_LOCKING", value)
    _refuses(parts, model_dir / "one_static.h5", "HDF5 file locking")


def test_file_locking_on_is_accepted(parts, model_dir, monkeypatch) -> None:
    monkeypatch.setenv("HDF5_USE_FILE_LOCKING", "TRUE")
    stamp_provenance(parts[0], model_dir / "one_static.h5")
    assert all("model_hash" in _provenance(p) for p in parts)


def test_refuses_an_incomplete_partition_set(parts, model_dir) -> None:
    """``part-1`` is missing; ``part-0`` declares NUM_PARTITIONS=2."""
    parts[1].unlink()
    _refuses(parts[:1], model_dir / "one_static.h5", "NUM_PARTITIONS=2")


def test_refuses_parts_whose_shared_element_differs(parts, model_dir) -> None:
    """Part 1 also lists element 2 (part 0's), with other nodes."""
    with h5py.File(parts[1], "r+") as f:
        conn = f["MODEL_STAGE[1]/MODEL/ELEMENTS/12-Truss[1:0]/CONNECTIVITY"]
        conn[...] = np.array([[2, 2, 3]], dtype=conn.dtype)
    _refuses(parts, model_dir / "one_static.h5", r"element \(12-Truss\[1:0\]\) 2")


def test_a_malformed_owned_attribute_needs_overwrite(parts, model_dir) -> None:
    """An int ``writer`` refuses; ``overwrite=True`` replaces it with a string."""
    with h5py.File(parts[0], "r+") as f:
        f[PROV].attrs["writer"] = np.int64(7)
    _refuses(parts, model_dir / "one_static.h5", r"not strings: \['writer'\]")
    stamp_provenance(parts[0], model_dir / "one_static.h5",
                     writer="apeGmsh fix", overwrite=True)
    with h5py.File(parts[0], "r") as f:
        aid = f[PROV].attrs.get_id("writer")
        assert h5py.check_string_dtype(aid.dtype).length is None
        assert f[PROV].attrs["writer"] == "apeGmsh fix"


@pytest.mark.parametrize("both", [False, True])
def test_refuses_a_nan_coordinate_on_a_shared_node(parts, model_dir, both) -> None:
    """NaN is never equal: one side or both, the shared node refuses."""
    targets = parts if both else parts[1:]
    for p in targets:
        with h5py.File(p, "r+") as f:
            xyz = f["MODEL_STAGE[1]/MODEL/NODES/COORDINATES"]
            ids = list(np.asarray(f["MODEL_STAGE[1]/MODEL/NODES/ID"][...]).ravel())
            arr = xyz[...]
            arr[ids.index(2), 1] = np.nan  # node 2 is the one both parts hold
            xyz[...] = arr
    _refuses(parts, model_dir / "one_static.h5", "disagree on their MODEL")


# ---------------------------------------------------------------------------
# The engine: opensees_backend and opensees_build_sha (§7b.1, #2196)
# ---------------------------------------------------------------------------


def _engine(path: Path) -> "tuple[str, str]":
    prov = _provenance(path)
    return prov["opensees_backend"], prov["opensees_build_sha"]


def _writer(path: Path) -> str:
    return _provenance(path)["writer"]


def test_a_default_stamp_is_ladruno_with_an_unknown_build(
        parts, model_dir, _no_backend) -> None:
    """The file does not carry its build, so a default stamp cannot know it."""
    stamp_provenance(parts[0], model_dir / "one_static.h5")
    assert [_engine(p) for p in parts] == [("ladruno", "unknown")] * 2
    assert _no_backend == []


def test_the_caller_passes_the_build(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "one_static.h5",
                     opensees_build_sha=FORK_SHA)
    assert [_engine(p) for p in parts] == [("ladruno", FORK_SHA)] * 2


def test_engine_attributes_are_scalar_vlen_utf8(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "one_static.h5",
                     opensees_build_sha=FORK_SHA)
    with h5py.File(parts[0], "r") as f:
        for name in ENGINE:
            aid = f[PROV].attrs.get_id(name)
            info = h5py.check_string_dtype(aid.dtype)
            assert info is not None and info.encoding == "utf-8"
            assert info.length is None and aid.shape == ()


def test_a_converted_file_is_unknown(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "one_static.h5", writer=CONVERTED)
    assert [_engine(p) for p in parts] == [("unknown", "unknown")] * 2


def test_a_converted_file_takes_an_explicit_stock_backend(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "one_static.h5", writer=CONVERTED,
                     opensees_backend="stock")
    assert [_engine(p) for p in parts] == [("stock", "unknown")] * 2


def test_three_stamps_keep_a_converted_file_and_its_engine(
        parts, model_dir) -> None:
    """#2208 F3: default re-stamps keep the mark and the stored engine."""
    model = model_dir / "one_static.h5"
    stamp_provenance(parts[0], model, writer=CONVERTED,
                     opensees_backend="stock")
    before = _sha(parts)
    stamp_provenance(parts[0], model)
    stamp_provenance(parts[1], model)
    assert _sha(parts) == before
    for p in parts:
        assert _writer(p) == CONVERTED
        assert _engine(p) == ("stock", "unknown")


def test_a_writer_that_drops_the_conversion_mark_refuses(
        parts, model_dir) -> None:
    """#2208 F3: ``stock`` never ends up on a file not marked as converted."""
    model = model_dir / "one_static.h5"
    stamp_provenance(parts[0], model, writer=CONVERTED,
                     opensees_backend="stock")
    before = _sha(parts)
    for overwrite in (False, True):
        with pytest.raises(ValueError, match="would unmark it"):
            stamp_provenance(parts[0], model, writer="apeGmsh 1.0",
                             overwrite=overwrite)
    assert _sha(parts) == before


def test_a_stamped_stock_on_an_unmarked_file_is_not_kept(
        parts, model_dir) -> None:
    """A hand-written ``stock`` without the mark falls back to the default."""
    model = model_dir / "one_static.h5"
    stamp_provenance(parts[0], model)
    for p in parts:
        with h5py.File(p, "r+") as f:
            f[PROV].attrs.create("opensees_backend", data="stock",
                                 dtype=h5py.string_dtype("utf-8"))
    stamp_provenance(parts[0], model)
    assert [_engine(p) for p in parts] == [("ladruno", "unknown")] * 2


def test_explicit_unknown_backend_writes_an_unknown_build(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "one_static.h5",
                     opensees_backend="unknown")
    assert [_engine(p) for p in parts] == [("unknown", "unknown")] * 2


@pytest.mark.parametrize(("kw", "match"), [
    ({"opensees_backend": "fork"}, "allows exactly"),
    ({"opensees_backend": "Ladruno"}, "allows exactly"),
    ({"opensees_build_sha": FORK_SHA[:12]}, "40 lowercase hex"),
    ({"opensees_build_sha": FORK_SHA.upper()}, "40 lowercase hex"),
    ({"opensees_build_sha": ""}, "40 lowercase hex"),
    ({"opensees_backend": "unknown", "opensees_build_sha": FORK_SHA},
     "needs opensees_backend='ladruno'"),
    ({"opensees_backend": "stock"}, "only a converted file can be stock"),
])
def test_bad_engine_values_refuse_before_any_write(
        parts, model_dir, kw, match) -> None:
    before = _sha(parts)
    with pytest.raises(ValueError, match=match):
        stamp_provenance(parts[0], model_dir / "one_static.h5", **kw)
    assert _sha(parts) == before


def test_a_default_restamp_keeps_a_stamped_build(parts, model_dir) -> None:
    """Idempotent: the hash the recording caller passed survives."""
    model = model_dir / "one_static.h5"
    stamp_provenance(parts[0], model, opensees_build_sha=FORK_SHA)
    before = _sha(parts)
    stamp_provenance(parts[1], model)
    assert _sha(parts) == before


def test_a_stamped_unknown_sticks_on_a_default_restamp(parts, model_dir) -> None:
    """#2208 F2: ``"unknown"`` is a stamped value like any other."""
    model = model_dir / "one_static.h5"
    stamp_provenance(parts[0], model, opensees_backend="unknown")
    before = _sha(parts)
    stamp_provenance(parts[0], model)
    assert _sha(parts) == before
    assert [_engine(p) for p in parts] == [("unknown", "unknown")] * 2


def test_explicit_value_replaces_a_stamped_build(parts, model_dir) -> None:
    model = model_dir / "one_static.h5"
    stamp_provenance(parts[0], model, opensees_build_sha=FORK_SHA)
    stamp_provenance(parts[0], model, opensees_build_sha=OTHER_SHA)
    assert [_engine(p) for p in parts] == [("ladruno", OTHER_SHA)] * 2
    stamp_provenance(parts[0], model, opensees_build_sha="unknown")
    assert [_engine(p) for p in parts] == [("ladruno", "unknown")] * 2


def test_overwrite_drops_the_stamped_engine(parts, model_dir) -> None:
    stamp_provenance(parts[0], model_dir / "one_static.h5",
                     opensees_build_sha=FORK_SHA)
    stamp_provenance(parts[0], model_dir / "offset_ids.h5", overwrite=True)
    assert [_engine(p) for p in parts] == [("ladruno", "unknown")] * 2


def test_a_malformed_stamped_build_becomes_unknown(parts, model_dir) -> None:
    """A string that is neither a 40-hex hash nor ``"unknown"`` is not kept."""
    stamp_provenance(parts[0], model_dir / "one_static.h5",
                     opensees_build_sha=FORK_SHA)
    for p in parts:
        with h5py.File(p, "r+") as f:
            f[PROV].attrs.create("opensees_build_sha", data="abc123",
                                 dtype=h5py.string_dtype("utf-8"))
    stamp_provenance(parts[0], model_dir / "one_static.h5")
    assert [_engine(p) for p in parts] == [("ladruno", "unknown")] * 2
