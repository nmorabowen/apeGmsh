"""#2120 — ``convert_mpco``: the explicit MPCO -> ``.ladruno`` converter.

ADR 0122 D5. Stock-lane oracles, on the committed ``elasticFrame.mpco``
and ``zl_springs.mpco``:

* **round trip** — every slab ``Results.from_mpco(src, model_h5=)``
  reads equals the slab ``Results.from_ladruno(converted, model_h5=)``
  reads, at every level, stage and component, except the spring level
  that ``unsupported`` declares;
* **declared loss** — ``INFO/PROVENANCE@unsupported`` lists exactly what
  the converted file lacks (``quadrature:<group>``, ``local_axes``,
  ``springs``), and nothing else;
* **KIND from the program** — each ``MODEL_STAGE`` takes its kind from the
  model archive's stage ``analysis`` record; a stage that holds only modes
  is ``eigen``;
* **refusals** — an existing target, a missing ``model_h5``, unpaired
  stages, a partition set with a gap or that disagrees, an unreadable or
  foreign source, and content the converter does not know: each leaves no
  target and no ``.partial`` staging, and the source bytes unchanged;
* **rollback** — a failure in the stamp or in the final move leaves no
  target and no staging.

The fork-side conformance oracle (the fork's own ``validate()`` and the
information-equality comparator against a fork-recorded ``.ladruno``) is
``tests/opensees/integration_ladruno/test_mpco_to_ladruno_oracle_live.py``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable

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
from apeGmsh.results import MPCOConversionRefused, Results, convert_mpco
from apeGmsh.results.readers._ladruno_from_mpco import StageKindRefused
from apeGmsh.results.writers import StageNamesOmittedWarning

_FIX = Path(__file__).parent / "fixtures" / "results"
FRAME = _FIX / "elasticFrame.mpco"
SPRINGS = _FIX / "zl_springs.mpco"
_STAGE = "MODEL_STAGE["


# ---------------------------------------------------------------------------
# Model archives that match the fixtures (ops tags == fem ids, nodes 1:1)
# ---------------------------------------------------------------------------


def _fem_from_mpco(path: Path, groups: "dict[str, list[int]]") -> FEMData:
    """A FEMData whose nodes and line elements are the capture's own."""
    with h5py.File(path, "r") as f:
        model = f["MODEL_STAGE[1]/MODEL"]
        ids = np.asarray(model["NODES/ID"][...], dtype=np.int64).ravel()
        xyz = np.asarray(model["NODES/COORDINATES"][...], dtype=np.float64)
        rows = {}
        for dset in model["ELEMENTS"].values():
            for row in np.asarray(dset[...], dtype=np.int64):
                rows[int(row[0])] = row[1:]
    if xyz.shape[1] == 2:
        xyz = np.column_stack([xyz, np.zeros(ids.size)])
    pgs, all_ids, all_conn = {}, [], []
    for k, (name, eids) in enumerate(groups.items()):
        conn = np.array([rows[e] for e in eids], dtype=np.int64)
        nids = np.unique(conn)
        pgs[(1, 100 + k)] = {
            "name": name, "node_ids": nids,
            "node_coords": xyz[np.searchsorted(ids, nids)],
            "element_ids": np.asarray(eids, dtype=np.int64),
        }
        all_ids.extend(eids)
        all_conn.append(conn)
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=len(all_ids))
    group = ElementGroup(element_type=info,
                         ids=np.asarray(all_ids, dtype=np.int64),
                         connectivity=np.concatenate(all_conn))
    return FEMData(
        nodes=NodeComposite(node_ids=ids, node_coords=xyz,
                            physical=PhysicalGroupSet(pgs),
                            labels=LabelSet({})),
        elements=ElementComposite(groups={1: group},
                                  physical=PhysicalGroupSet(pgs),
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=ids.size, n_elems=len(all_ids), bandwidth=1,
                      types=[info]),
        model_name="convert_mpco_fixture",
    )


def _stages(ops: Any, kinds: "list[tuple[str, str]]") -> None:
    for name, kind in kinds:
        if kind == "static":
            integ, ana = ops.integrator.LoadControl(dlam=0.1), ops.analysis.Static()
        else:
            integ = ops.integrator.Newmark(gamma=0.5, beta=0.25)
            ana = ops.analysis.Transient()
        with ops.stage(name=name) as s:
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-12, max_iter=10),
                algorithm=ops.algorithm.Linear(), integrator=integ,
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.RCM(), system=ops.system.BandGeneral(),
                analysis=ana,
            )
            s.run(n_increments=1)


def _frame_model(path: Path, kinds: "list[tuple[str, str]]") -> Path:
    from apeGmsh.opensees import apeSees

    ops = apeSees(_fem_from_mpco(FRAME, {"Frame": list(range(1, 12))}))
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))
    ops.element.elasticBeamColumn(pg="Frame", transf=t, A=1.0, E=1.0,
                                  Iz=1.0, Iy=1.0, G=1.0, J=1.0)
    _stages(ops, kinds)
    ops.h5(str(path))
    return path


def _springs_model(path: Path) -> Path:
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir

    ops = apeSees(_fem_from_mpco(SPRINGS, {"One": [100], "Three": [200]}))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1000.0)
    ops.element.ZeroLength(pg="One", mat_dirs=(ZeroLengthMatDir(material=mat, dof=1),))
    ops.element.ZeroLength(pg="Three", mat_dirs=tuple(
        ZeroLengthMatDir(material=mat, dof=d) for d in (1, 2, 3)))
    _stages(ops, [("load", "static")])
    ops.h5(str(path))
    return path


def _unstaged_springs_model(path: Path, kinds: "list[str]") -> Path:
    """The springs model with no ``ops.stage`` block and one ``analysis``
    call per entry of ``kinds`` (the implicit-stage rule, #1950)."""
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir

    ops = apeSees(_fem_from_mpco(SPRINGS, {"One": [100], "Three": [200]}))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1000.0)
    ops.element.ZeroLength(pg="One", mat_dirs=(ZeroLengthMatDir(material=mat, dof=1),))
    ops.element.ZeroLength(pg="Three", mat_dirs=tuple(
        ZeroLengthMatDir(material=mat, dof=d) for d in (1, 2, 3)))
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    for kind in kinds:
        if kind == "static":
            ops.integrator.LoadControl(dlam=0.1)
            ops.analysis.Static()
        else:
            ops.integrator.Newmark(gamma=0.5, beta=0.25)
            ops.analysis.Transient()
    ops.h5(str(path))
    return path


def _bridge_copy(src: Path, model: Path, dst: Path) -> Path:
    """``src`` with its element tags renumbered to the archive's ops tags.

    The bridge emits ops tags of its own (the fem ids are not the tags), so
    a copy of the committed capture carries the tags the archive pairs
    with its fem ids: element reads then relabel exactly, as on a real
    bridge run. Values, buckets and nodes are untouched.
    """
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    tag_of = {int(r.fem_eid): int(r.tag)
              for r in OpenSeesModel.from_h5(model).elements()}

    def remap(ids: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda e: tag_of[int(e)])(ids).astype(ids.dtype)

    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        for stage in (k for k in f if k.startswith(_STAGE)):
            for ds in f[f"{stage}/MODEL/ELEMENTS"].values():
                conn = ds[...]
                conn[:, 0] = remap(conn[:, 0])
                ds[...] = conn
            on_el = f[f"{stage}/RESULTS/ON_ELEMENTS"]
            for token in on_el.values():
                for bucket in token.values():
                    bucket["ID"][...] = remap(bucket["ID"][...])
    return dst


@pytest.fixture(scope="module")
def fx(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Path]":
    tmp = tmp_path_factory.mktemp("convert_mpco_models")
    frame = _frame_model(tmp / "frame.h5",
                         [("gravity", "static"), ("shake", "transient")])
    springs = _springs_model(tmp / "springs.h5")
    return {
        "frame": frame,
        "frame_1stage": _frame_model(tmp / "frame1.h5", [("only", "static")]),
        "springs": springs,
        "frame_src": _bridge_copy(FRAME, frame, tmp / "elasticFrame.mpco"),
        "springs_src": _bridge_copy(SPRINGS, springs, tmp / "zl_springs.mpco"),
        **{f"flat_{name}": _unstaged_springs_model(tmp / f"flat_{name}.h5", kinds)
           for name, kinds in _FLAT.items()},
    }


#: Unstaged springs programs, by their ``analysis`` calls.
_FLAT = {
    "static": ["static"],
    "transient": ["transient"],
    "two": ["static", "transient"],
    "none": [],
}


CASES = ("frame", "springs")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert(src: Path, dst: Path, model: Path, **kw: Any) -> list[Path]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return convert_mpco(src, dst, model_h5=model, **kw)


def _left_behind(directory: Path) -> list[str]:
    """Every ``.ladruno`` or ``.partial`` entry in ``directory``."""
    return sorted(p.name for p in directory.iterdir()
                  if p.suffix in (".ladruno", ".partial"))


def _prov(path: Path) -> "dict[str, str]":
    with h5py.File(path, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else str(v))
                for k, v in f["INFO/PROVENANCE"].attrs.items()}


# ---------------------------------------------------------------------------
# Round trip: the slabs from_mpco reads equal the slabs from_ladruno reads
# ---------------------------------------------------------------------------


_LEVELS: "dict[str, Callable[[Results], Any]]" = {
    "nodes": lambda r: r.nodes,
    "elements": lambda r: r.elements,
    "gauss": lambda r: r.elements.gauss,
    "line_stations": lambda r: r.elements.line_stations,
    "fibers": lambda r: r.elements.fibers,
    "layers": lambda r: r.elements.layers,
}


def _assert_slab_equal(a: Any, b: Any, where: str) -> None:
    assert type(a) is type(b), where
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y),
                                          err_msg=f"{where}.{f.name}")
        elif dataclasses.is_dataclass(x):
            _assert_slab_equal(x, y, f"{where}.{f.name}")
        else:
            assert x == y, f"{where}.{f.name}: {x!r} != {y!r}"


def _components(res: Results) -> "list[tuple[str, str, str]]":
    out = []
    for stage in res.stages:
        scoped = res.stage(stage.id)
        for level, get in _LEVELS.items():
            if stage.kind == "mode" and level != "nodes":
                continue
            out.extend((stage.id, level, name)
                       for name in get(scoped).available_components())
    return out


def _read(res: Results, key: "tuple[str, str, str]") -> Any:
    stage, level, name = key
    return _LEVELS[level](res.stage(stage)).get(component=name)


def _assert_token_is_the_source_bucket(
    res: Results, src: Path, stage_id: str, token: str,
) -> None:
    """A token read of the converted file equals the source's per-step data."""
    slab = res.stage(stage_id).elements.get(component=token)
    k = int(stage_id.rsplit("_", 1)[1])
    with h5py.File(src, "r") as f:
        stage = sorted((g for g in f if g.startswith(_STAGE)),
                       key=lambda g: int(g[len(_STAGE):-1]))[k]
        (bucket,) = f[f"{stage}/RESULTS/ON_ELEMENTS/{token}"].values()
        steps = sorted(bucket["DATA"].values(),
                       key=lambda d: int(d.attrs["STEP"][0]))
        want = np.stack([d[...] for d in steps])
    values = np.asarray(slab.values)
    assert values.shape == want.shape, (token, values.shape, want.shape)
    np.testing.assert_array_equal(values, want, err_msg=f"{stage_id} {token}")


@pytest.mark.parametrize("case", CASES)
def test_round_trip_slabs_are_equal(case, fx, tmp_path) -> None:
    src, model = fx[f"{case}_src"], fx[case]
    (dst,) = _convert(src, tmp_path / "out.ladruno", model)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = Results.from_mpco(str(src), model_h5=str(model))
        b = Results.from_ladruno(str(dst), model_h5=str(model))
        try:
            assert [(s.id, s.name, s.kind, s.n_steps) for s in a.stages] ==                 [(s.id, s.name, s.kind, s.n_steps) for s in b.stages]
            for s in a.stages:
                np.testing.assert_array_equal(a.stage(s.id).time,
                                              b.stage(s.id).time)
            keys_a, keys_b = set(_components(a)), set(_components(b))
            common = sorted(keys_a & keys_b)
            assert common, "nothing to read: the oracle compares no data"
            for key in common:
                _assert_slab_equal(_read(a, key), _read(b, key), str(key))
            # The readers name element-level columns differently, and only
            # there: MPCO decodes localForce into nodal_resisting_*_local_*,
            # while the .ladruno reader decodes globalForce only and lists
            # the other buckets by token. The bytes are the same: every
            # token read of the converted file equals the source bucket.
            only_a, only_b = keys_a - keys_b, keys_b - keys_a
            assert all(lv == "elements" and n.startswith("nodal_resisting_")
                       and "_local_" in n for _, lv, n in only_a), only_a
            assert all(lv == "elements" for _, lv, _ in only_b), only_b
            for stage, _, token in sorted(k for k in only_b
                                          if case == "frame"):
                _assert_token_is_the_source_bucket(
                    b, src, stage, token)
            extra = sorted(only_b)
            springs_a = [s.id for s in a.stages
                         if a.stage(s.id).elements.springs.available_components()]
            springs_b = [s.id for s in b.stages
                         if b.stage(s.id).elements.springs.available_components()]
        finally:
            a.close()
            b.close()
    # The spring level is the read the converted file loses: from_ladruno
    # has no spring level (the zero-length buckets are copied, and its
    # element level lists their raw columns instead). It is declared, and
    # only when the source holds springs.
    assert springs_b == []
    has_springs = bool(springs_a)
    assert has_springs == (case == "springs")
    assert ("springs" in json.loads(_prov(dst)["unsupported"])) == has_springs
    assert extra  # both fixtures list element tokens the MPCO reader decodes


# ---------------------------------------------------------------------------
# What the converted file says about itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_unsupported_lists_exactly_what_was_dropped(case, fx, tmp_path) -> None:
    src = fx[f"{case}_src"]
    (dst,) = _convert(src, tmp_path / "out.ladruno", fx[case])
    unsupported = json.loads(_prov(dst)["unsupported"])
    assert unsupported == sorted(set(unsupported))
    with h5py.File(src, "r") as s, h5py.File(dst, "r") as d:
        dropped = set()
        for stage in (k for k in s if k.startswith(_STAGE)):
            for gname in s[f"{stage}/MODEL/ELEMENTS"]:
                if "QUADRATURE" not in d[f"{stage}/MODEL/ELEMENTS/{gname}"]:
                    dropped.add(f"quadrature:{gname}")
            if "LOCAL_AXES" in s[f"{stage}/MODEL"] and                     len(s[f"{stage}/MODEL/LOCAL_AXES"]):
                assert "LOCAL_AXES" not in d[f"{stage}/MODEL"]
                dropped.add("local_axes")
        if case == "springs":
            dropped.add("springs")
    assert set(unsupported) == dropped
    assert dropped  # both fixtures lose something: the oracle bites


def test_local_axes_are_dropped_and_declared(fx, tmp_path) -> None:
    src = tmp_path / "axes.mpco"
    shutil.copy(fx["frame_src"], src)
    with h5py.File(src, "r+") as f:
        la = f["MODEL_STAGE[1]/MODEL"].require_group("LOCAL_AXES")
        la.create_dataset("ID", data=np.arange(1, 12, dtype=np.int32))
        la.create_dataset("LOCAL_AXES", data=np.tile([1.0, 0, 0, 0], (11, 1)))
    (dst,) = _convert(src, tmp_path / "out.ladruno", fx["frame"])
    assert "local_axes" in json.loads(_prov(dst)["unsupported"])
    with h5py.File(dst, "r") as d:
        assert "LOCAL_AXES" not in d["MODEL_STAGE[1]/MODEL"]


def test_info_kinds_and_provenance(fx, tmp_path) -> None:
    import apeGmsh

    (dst,) = _convert(fx["frame_src"], tmp_path / "out.ladruno", fx["frame"])
    prov = _prov(dst)
    assert prov["writer"] == f"apeGmsh mpco->ladruno {apeGmsh.__version__}"
    assert json.loads(prov["stage_names"]) == ["gravity", "shake"]
    with h5py.File(fx["frame"], "r") as m:
        lineage = m["meta/lineage"].attrs
        assert prov["model_hash"] == str(lineage["model_hash"])
        assert prov["fem_hash"] == str(lineage["fem_hash"])
    with h5py.File(dst, "r") as d, h5py.File(fx["frame_src"], "r") as s:
        info = d["INFO"].attrs
        assert info["GENERATOR"][0] == b"Ladruno"
        assert int(info["FORMAT_VERSION"][0]) == 1
        assert "FORMAT_MINOR_VERSION" not in info
        assert info["STORED_PRECISION"][0] == b"f64"
        assert [int(info[k][0]) for k in
                ("PARTITIONED", "PARTITION_ID", "NUM_PARTITIONS")] == [0, 0, 1]
        for name in ("SOLVER_NAME", "SOLVER_VERSION", "SPATIAL_DIM"):
            np.testing.assert_array_equal(info[name], s[f"INFO/{name}"][...])
        kinds = [d[g].attrs["KIND"][0] for g in ("MODEL_STAGE[1]", "MODEL_STAGE[2]")]
        assert kinds == [b"static", b"transient"]
        # Bucket names, tokens and IDs are verbatim; META is a COLUMN_MAP.
        b = "MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/localForce/5-ElasticBeam3d[1:0:0]"
        cmap = d[f"{b}/COLUMN_MAP"]
        assert cmap.attrs["COMP_NAMES"][0] == \
            b"N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2"
        assert cmap["LEVELS"][...].tolist() == [[0]]
        assert d[b].attrs["DISPLAY_NAME"][0] == b"localForce"
        np.testing.assert_array_equal(d[f"{b}/ID"][...], s[f"{b}/ID"][...])
        assert d[f"{b}/DATA"].shape == (10, 11, 12)
        np.testing.assert_array_equal(d[f"{b}/DATA"][3],
                                      s[f"{b}/DATA/STEP_3"][...])


def test_a_stage_with_only_modes_is_eigen(fx, tmp_path) -> None:
    """Modes recorded after the program's stages open their own stage."""
    src = tmp_path / "modes.mpco"
    shutil.copy(fx["frame_src"], src)
    with h5py.File(src, "r+") as f:
        f.copy(f["MODEL_STAGE[2]/MODEL"], f.create_group("MODEL_STAGE[3]"))
        stage = f["MODEL_STAGE[3]"]
        stage.attrs["STEP"] = np.array([20], dtype=np.int32)
        stage.attrs["TIME"] = np.array([2.0])
        grp = stage.create_group("RESULTS/ON_NODES/MODES_OF_VIBRATION(U)")
        grp.attrs["COMPONENTS"] = np.array([b"Ux,Uy,Uz"])
        grp.create_dataset("ID", data=np.arange(1, 13, dtype=np.int32)[:, None])
        step = grp.create_group("DATA/STEP_20")
        mode = step.create_dataset("MODE_0", data=np.ones((12, 3)))
        mode.attrs["LAMBDA"] = np.array([4.0])
    with pytest.warns(StageNamesOmittedWarning):
        (dst,) = convert_mpco(src, tmp_path / "out.ladruno",
                              model_h5=fx["frame"])
    with h5py.File(dst, "r") as d:
        kinds = [d[f"MODEL_STAGE[{k}]"].attrs["KIND"][0] for k in (1, 2, 3)]
        assert kinds == [b"static", b"transient", b"eigen"]
        assert "MODE_0" in d["MODEL_STAGE[3]/RESULTS/ON_NODES/"
                             "MODES_OF_VIBRATION(U)/DATA/STEP_20"]


# ---------------------------------------------------------------------------
# The implicit stage of an unstaged program (#1950)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["static", "transient"])
def test_an_unstaged_program_with_one_analysis_is_one_implicit_stage(
    fx, kind, tmp_path,
) -> None:
    """One MODEL_STAGE with step data and one ``analysis`` call: the stage
    takes the call's kind, and every slab reads as the source does."""
    model = fx[f"flat_{kind}"]
    src = _bridge_copy(SPRINGS, model, tmp_path / "src.mpco")
    (dst,) = _convert(src, tmp_path / "out.ladruno", model, _implicit_stage=True)
    with h5py.File(dst, "r") as d:
        stages = sorted(g for g in d if g.startswith(_STAGE))
        assert [d[g].attrs["KIND"][0] for g in stages] == [kind.encode()]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = Results.from_mpco(str(src), model_h5=str(model))
        b = Results.from_ladruno(str(dst), model_h5=str(model))
        try:
            assert [s.kind for s in b.stages] == [kind]
            assert [(s.id, s.n_steps) for s in a.stages] ==                 [(s.id, s.n_steps) for s in b.stages]
            common = sorted(set(_components(a)) & set(_components(b)))
            assert common, "nothing to read: the oracle compares no data"
            for key in common:
                _assert_slab_equal(_read(a, key), _read(b, key), str(key))
        finally:
            a.close()
            b.close()


@pytest.mark.parametrize("name, match", [
    ("two", "2 analysis calls"),
    ("none", "no analysis call"),
])
def test_an_unstaged_program_whose_kind_is_not_a_fact_refuses(
    fx, name, match, tmp_path,
) -> None:
    model = fx[f"flat_{name}"]
    src = _bridge_copy(SPRINGS, model, tmp_path / "src.mpco")
    _refuses(src, tmp_path / "out.ladruno", match, exc=StageKindRefused,
             model_h5=model, _implicit_stage=True)


def test_an_unknown_analysis_type_refuses(fx, tmp_path) -> None:
    """One call whose type names no kind: never defaulted to a kind."""
    model = Path(shutil.copy(fx["flat_static"], tmp_path / "bogus.h5"))
    with h5py.File(model, "r+") as f:
        f["opensees/analysis"].attrs["analysis"] = "Bogus"
    src = _bridge_copy(SPRINGS, model, tmp_path / "src.mpco")
    _refuses(src, tmp_path / "out.ladruno", "names no known kind",
             exc=StageKindRefused, model_h5=model, _implicit_stage=True)


def test_a_bare_convert_refuses_an_unstaged_program_as_before(fx, tmp_path) -> None:
    """Without the private flag (passed by read_deck_mpco alone) an
    unstaged program pairs with no stage: the in-process archive cannot be
    told from a deck's, so the public converter never guesses."""
    model = fx["flat_static"]
    src = _bridge_copy(SPRINGS, model, tmp_path / "src.mpco")
    _refuses(src, tmp_path / "out.ladruno", "declares 0 stage",
             exc=StageKindRefused, model_h5=model)


def test_an_unstaged_program_never_spans_two_mpco_stages(fx, tmp_path) -> None:
    """Two MODEL_STAGE groups with step data refuse even with one call."""
    model = fx["flat_static"]
    src = _bridge_copy(SPRINGS, model, tmp_path / "src.mpco")
    with h5py.File(src, "r+") as f:
        f.copy(f["MODEL_STAGE[1]"], "MODEL_STAGE[2]")
    _refuses(src, tmp_path / "out.ladruno", "one implicit stage at most",
             model_h5=model, _implicit_stage=True)


def test_groups_take_their_quadrature_from_the_source_or_the_catalog(
    fx, tmp_path,
) -> None:
    """A custom rule keeps its own ``GP_X``; a standard rule takes the
    catalog's parent coordinates; neither is then declared lost."""
    from apeGmsh.opensees._response_catalog import lookup

    src = tmp_path / "groups.mpco"
    shutil.copy(fx["frame_src"], src)
    gp_x = np.array([-1.0, -0.6546536707, 0.0, 0.6546536707, 1.0])
    with h5py.File(src, "r+") as f:
        for stage in ("MODEL_STAGE[1]", "MODEL_STAGE[2]"):
            els = f[f"{stage}/MODEL/ELEMENTS"]
            quad = els.create_dataset(
                "31-FourNodeQuad[201:0]",
                data=np.array([[90, 1, 2, 3, 4]], dtype=np.int32))
            quad.attrs["GEOMETRY"] = np.array([200], dtype=np.int32)
            quad.attrs["INTEGRATION_RULE"] = np.array([201], dtype=np.int32)
            beam = els.create_dataset(
                "73-ForceBeamColumn3d[1000:1]",
                data=np.array([[91, 1, 5]], dtype=np.int32))
            beam.attrs["CUSTOM_INTEGRATION_RULE"] = np.array([1], dtype=np.int32)
            beam.attrs["CUSTOM_INTEGRATION_RULE_DIMENSION"] = np.array(
                [1], dtype=np.int32)
            beam.attrs["GP_X"] = gp_x
    (dst,) = _convert(src, tmp_path / "out.ladruno", fx["frame"])
    with h5py.File(dst, "r") as d:
        g = d["MODEL_STAGE[1]/MODEL/ELEMENTS"]
        q, b = g["31-FourNodeQuad[201:0]"], g["73-ForceBeamColumn3d[1000:1]"]
        np.testing.assert_array_equal(
            q["QUADRATURE/GP_PARAM"][...],
            lookup("FourNodeQuad", 201, "stress").natural_coords)
        assert [q.attrs[k][0] for k in ("TOPOLOGY", "PARAM_DOMAIN", "FAMILY")]             == [b"quad", b"[-1,1]", b"lagrange"]
        assert [int(q.attrs[k][0]) for k in ("NUM_GP", "NDIR", "NUM_CTRL")]             == [4, 2, 4]
        np.testing.assert_array_equal(b["QUADRATURE/GP_PARAM"][...],
                                      gp_x[:, None])
        assert b.attrs["TOPOLOGY"][0] == b"line"
        np.testing.assert_array_equal(b.attrs["GP_X"], gp_x)  # source attrs kept
        e = g["5-ElasticBeam3d[1:0]"]
        assert "QUADRATURE" not in e
        assert [e.attrs[k][0] for k in ("TOPOLOGY", "FAMILY")] ==             [b"custom", b"custom"]
    assert json.loads(_prov(dst)["unsupported"]) == [
        "quadrature:5-ElasticBeam3d[1:0]"]


def test_meta_becomes_the_column_map(fx, tmp_path) -> None:
    """``META`` -> ``COLUMN_MAP`` the way the fork recorder derives it from
    the same header: ``LEVELS`` is the deepest path code of a block,
    ``SECTION_TAG`` / ``FIBER_ID`` are -1, ``COMP_NAMES`` one line per block."""
    src = tmp_path / "meta.mpco"
    shutil.copy(fx["frame_src"], src)
    blocks = "0.1.4.s11,s22;0.1.2.3.4.sigma11"
    with h5py.File(src, "r+") as f:
        steps = f["MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA"]
        tok = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"].create_group("odd")
        tok.attrs["DISPLAY_NAME"] = np.array([b"odd"])
        b = tok.create_group("31-FourNodeQuad[201:0:0]")
        b.attrs["NUM_COLUMNS"] = np.array([5], dtype=np.int32)
        meta = b.create_group("META")
        meta["MULTIPLICITY"] = np.array([[1], [3]], dtype=np.int32)
        meta["GAUSS_IDS"] = np.array([[0], [1]], dtype=np.int32)
        meta["NUM_COMPONENTS"] = np.array([[2], [1]], dtype=np.int32)
        meta["COMPONENTS"] = np.array([blocks.encode()])
        b["ID"] = np.array([[90]], dtype=np.int32)
        n_steps = len(steps)
        for name, step in steps.items():
            ds = b.create_dataset(f"DATA/{name}", data=np.full((1, 5), 7.0))
            ds.attrs["STEP"], ds.attrs["TIME"] = step.attrs["STEP"], step.attrs["TIME"]
    (dst,) = _convert(src, tmp_path / "out.ladruno", fx["frame"])
    with h5py.File(dst, "r") as d:
        out = d["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/odd/31-FourNodeQuad[201:0:0]"]
        cmap = out["COLUMN_MAP"]
        got = {k: cmap[k][...].ravel().tolist() for k in cmap}
        assert got == {"LEVELS": [4, 4], "GAUSS_ID": [0, 1],
                       "SECTION_TAG": [-1, -1], "FIBER_ID": [-1, -1],
                       "NUM_COMP": [2, 1], "MULTIPLICITY": [1, 3]}
        assert all(cmap[k].shape == (2, 1) for k in cmap)
        assert cmap.attrs["COMP_NAMES"][0] == b"s11,s22\nsigma11"
        assert out.attrs["DISPLAY_NAME"][0] == b"odd"
        assert int(out.attrs["NUM_COLUMNS"][0]) == 5
        assert out["DATA"].shape == (n_steps, 1, 5)


def test_groups_keep_the_source_order(fx, tmp_path) -> None:
    """A reader takes the first ``ON_NODES`` group naming a component
    (``REACTION_FORCE`` before ``RAYLEIGH_FORCE``): the order is information."""
    (dst,) = _convert(fx["frame_src"], tmp_path / "out.ladruno", fx["frame"])
    with h5py.File(dst, "r") as d, h5py.File(fx["frame_src"], "r") as s:
        for stage in ("MODEL_STAGE[1]", "MODEL_STAGE[2]"):
            for path in (f"{stage}/RESULTS/ON_NODES",
                         f"{stage}/RESULTS/ON_ELEMENTS"):
                kept = [n for n in s[path] if len(s[path][n])]
                assert list(d[path]) == kept, path


def test_a_large_node_result_streams_in_bounded_chunks(fx, tmp_path) -> None:
    """180,000 x 3: one (1024, n, c) chunk would be 4.4 GB, which HDF5
    refuses. Chunks follow the recorder's ~1 MiB rule, and the history is
    streamed a block of steps at a time (at most 32 MiB), never stacked."""
    import tracemalloc

    n = 180_000
    src = tmp_path / "big.mpco"
    shutil.copy(fx["frame_src"], src)
    with h5py.File(src, "r+") as f:
        on = f["MODEL_STAGE[1]/RESULTS/ON_NODES"]
        steps = on["DISPLACEMENT/DATA"]
        g = on.create_group("BIG_FIELD")
        g.attrs["COMPONENTS"] = np.array([b"a,b,c"])
        g["ID"] = np.arange(1, n + 1, dtype=np.int32)[:, None]
        for name, step in steps.items():
            k = float(step.attrs["STEP"][0])
            ds = g.create_dataset(f"DATA/{name}", data=np.full((n, 3), k))
            ds.attrs["STEP"], ds.attrs["TIME"] = step.attrs["STEP"], step.attrs["TIME"]
        n_t = len(steps)
    tracemalloc.start()
    try:
        (dst,) = _convert(src, tmp_path / "out.ladruno", fx["frame"])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    full = n_t * n * 3 * 8
    assert peak < (32 << 20) + (4 << 20) < full, (peak, full)
    with h5py.File(dst, "r") as d:
        out = d["MODEL_STAGE[1]/RESULTS/ON_NODES/BIG_FIELD"]
        assert out["DATA"].shape == (n_t, n, 3)
        assert np.prod(out["DATA"].chunks) * 8 <= 1 << 20
        np.testing.assert_array_equal(
            out["DATA"][:, 0, 0], out["STEP"][...].astype(float))
        assert out.attrs["COMPONENTS"][0] == b"a,b,c"


def test_overwrite_replaces_the_target(fx, tmp_path) -> None:
    dst = tmp_path / "out.ladruno"
    dst.write_bytes(b"old")
    (out,) = _convert(fx["frame_src"], dst, fx["frame"], overwrite=True)
    assert out == dst
    with h5py.File(dst, "r") as d:
        assert d["INFO"].attrs["GENERATOR"][0] == b"Ladruno"
    assert _left_behind(tmp_path) == ["out.ladruno"]


def test_a_partitioned_set_converts_part_for_part(fx, tmp_path) -> None:
    for i in (0, 1):
        shutil.copy(fx["frame_src"], tmp_path / f"run.part-{i}.mpco")
    out = _convert(tmp_path / "run.part-1.mpco", tmp_path / "res.ladruno",
                   fx["frame"])
    assert out == [tmp_path / "res.part-0.ladruno",
                   tmp_path / "res.part-1.ladruno"]
    for i, path in enumerate(out):
        with h5py.File(path, "r") as d:
            info = d["INFO"].attrs
            assert [int(info[k][0]) for k in
                    ("PARTITIONED", "PARTITION_ID", "NUM_PARTITIONS")] == [1, i, 2]
        assert _prov(path)["model_hash"] == _prov(out[0])["model_hash"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".partial") == []


# ---------------------------------------------------------------------------
# Refusals: no target, no staging, the source untouched
# ---------------------------------------------------------------------------


def _refuses(src: Path, dst: Path, match: str, *,
             exc: "type[Exception]" = MPCOConversionRefused, **kw: Any) -> None:
    before = _sha(src)
    listing = _left_behind(dst.parent)
    with pytest.raises(exc, match=match):
        convert_mpco(src, dst, **kw)
    assert _sha(src) == before
    assert _left_behind(dst.parent) == listing


def test_refuses_an_existing_target(fx, tmp_path) -> None:
    dst = tmp_path / "out.ladruno"
    dst.write_bytes(b"keep me")
    _refuses(fx["frame_src"], dst, "exists", model_h5=fx["frame"])
    assert dst.read_bytes() == b"keep me"


def test_refuses_without_model_h5(fx, tmp_path) -> None:
    _refuses(fx["frame_src"], tmp_path / "out.ladruno", "model_h5", model_h5=None)
    _refuses(fx["frame_src"], tmp_path / "out.ladruno", "no such model archive",
             model_h5=tmp_path / "missing.h5")
    with pytest.raises(TypeError):
        convert_mpco(fx["frame_src"], tmp_path / "out.ladruno")  # type: ignore[call-arg]


def test_refuses_unpaired_stages(fx, tmp_path) -> None:
    _refuses(fx["frame_src"], tmp_path / "out.ladruno", "2 MODEL_STAGE",
             model_h5=fx["frame_1stage"])


def test_refuses_the_archive_of_another_model(fx, tmp_path) -> None:
    """The check behind from_mpco's ModelFemMismatchWarning, as a refusal:
    zl_springs' nodes 1-4 exist in the frame archive, at other places."""
    _refuses(fx["springs_src"], tmp_path / "out.ladruno", "is not a run of",
             model_h5=fx["frame_1stage"])
    from apeGmsh.results._bind import ModelFemMismatchWarning

    with pytest.warns(ModelFemMismatchWarning):  # the same pair, from_mpco
        Results.from_mpco(str(fx["springs_src"]),
                          model_h5=str(fx["frame_1stage"])).close()


def test_refuses_a_partition_set_with_a_gap(fx, tmp_path) -> None:
    for i in (0, 2):
        shutil.copy(fx["frame_src"], tmp_path / f"run.part-{i}.mpco")
    _refuses(tmp_path / "run.part-0.mpco", tmp_path / "res.ladruno",
             "contiguous", model_h5=fx["frame"])


def test_refuses_parts_that_disagree(fx, tmp_path) -> None:
    for i in (0, 1):
        shutil.copy(fx["frame_src"], tmp_path / f"run.part-{i}.mpco")
    with h5py.File(tmp_path / "run.part-1.mpco", "r+") as f:
        del f["MODEL_STAGE[2]"]
    _refuses(tmp_path / "run.part-0.mpco", tmp_path / "res.ladruno",
             "1 MODEL_STAGE", model_h5=fx["frame"])


def test_refuses_stray_parts_of_an_older_set(fx, tmp_path) -> None:
    for i in (0, 1):
        shutil.copy(fx["frame_src"], tmp_path / f"run.part-{i}.mpco")
    stray = tmp_path / "res.part-2.ladruno"
    stray.write_bytes(b"stale")
    _refuses(tmp_path / "run.part-0.mpco", tmp_path / "res.ladruno",
             "earlier", model_h5=fx["frame"], overwrite=True)


def test_refuses_an_unreadable_source(fx, tmp_path) -> None:
    junk = tmp_path / "junk.mpco"
    junk.write_bytes(b"not hdf5")
    _refuses(junk, tmp_path / "out.ladruno", "cannot be read",
             model_h5=fx["frame"])


def test_refuses_a_ladruno_named_mpco(fx, tmp_path) -> None:
    (made,) = _convert(fx["frame_src"], tmp_path / "made.ladruno", fx["frame"])
    fake = tmp_path / "fake.mpco"
    shutil.copy(made, fake)
    _refuses(fake, tmp_path / "out.ladruno", "GENERATOR|cannot be read",
             model_h5=fx["frame"])


def test_refuses_content_it_does_not_know(fx, tmp_path) -> None:
    src = tmp_path / "odd.mpco"
    shutil.copy(fx["frame_src"], src)
    with h5py.File(src, "r+") as f:
        f["MODEL_STAGE[1]/MODEL"].create_group("SETS")
    _refuses(src, tmp_path / "out.ladruno", "SETS", model_h5=fx["frame"])


def test_a_malformed_bucket_found_while_writing_rolls_back(fx, tmp_path) -> None:
    src = tmp_path / "bad.mpco"
    shutil.copy(fx["frame_src"], src)
    with h5py.File(src, "r+") as f:
        del f["MODEL_STAGE[2]/RESULTS/ON_ELEMENTS/force/"
              "5-ElasticBeam3d[1:0:0]/META/COMPONENTS"]
    _refuses(src, tmp_path / "out.ladruno", "no COMPONENTS",
             model_h5=fx["frame"])


def test_refuses_a_left_over_staging_directory(fx, tmp_path) -> None:
    (tmp_path / "out.partial").mkdir()
    _refuses(fx["frame_src"], tmp_path / "out.ladruno", "staging",
             model_h5=fx["frame"])


def test_never_removes_a_staging_directory_that_holds_backups(fx, tmp_path) -> None:
    """Backups a rollback could not restore are the only copy of the old
    set: even ``overwrite=True`` refuses, names the directory, keeps them."""
    staging = tmp_path / "out.partial"
    staging.mkdir()
    backup = staging / "out.ladruno.old"
    backup.write_bytes(b"the only copy")
    for overwrite in (False, True):
        with pytest.raises(MPCOConversionRefused, match="out.partial") as e:
            convert_mpco(fx["frame_src"], tmp_path / "out.ladruno",
                         model_h5=fx["frame"], overwrite=overwrite)
        assert "never restored" in str(e.value)
        assert backup.read_bytes() == b"the only copy"
    assert not (tmp_path / "out.ladruno").exists()


# ---------------------------------------------------------------------------
# Rollback after the staging exists
# ---------------------------------------------------------------------------


def test_a_failed_stamp_rolls_back(fx, tmp_path, monkeypatch) -> None:
    from apeGmsh.results.readers import _ladruno_provenance

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("stamp failed")

    monkeypatch.setattr(_ladruno_provenance, "stamp_provenance", boom)
    before = _sha(fx["frame_src"])
    with pytest.raises(RuntimeError, match="stamp failed"):
        convert_mpco(fx["frame_src"], tmp_path / "out.ladruno", model_h5=fx["frame"])
    assert _left_behind(tmp_path) == []
    assert _sha(fx["frame_src"]) == before


def test_a_failed_move_removes_every_part(fx, tmp_path, monkeypatch) -> None:
    import apeGmsh._atomic_io as atomic

    for i in (0, 1):
        shutil.copy(fx["frame_src"], tmp_path / f"run.part-{i}.mpco")
    real = atomic.replace_with_retry
    calls = []

    def second_fails(src: Any, dst: Any) -> None:
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("disk gone")
        real(src, dst)

    monkeypatch.setattr(atomic, "replace_with_retry", second_fails)
    with pytest.raises(OSError, match="disk gone"):
        convert_mpco(tmp_path / "run.part-0.mpco", tmp_path / "res.ladruno",
                     model_h5=fx["frame"])
    assert len(calls) == 2
    assert _left_behind(tmp_path) == []


@pytest.mark.parametrize("fail_at", [2, 4], ids=["second-backup", "second-move"])
def test_overwrite_never_leaves_the_old_set_with_a_gap(
    fx, tmp_path, monkeypatch, fail_at,
) -> None:
    """Old parts move aside first; a failed backup or move restores them all."""
    import apeGmsh._atomic_io as atomic

    for i in (0, 1):
        shutil.copy(fx["frame_src"], tmp_path / f"run.part-{i}.mpco")
    src, dst = tmp_path / "run.part-0.mpco", tmp_path / "res.ladruno"
    old = convert_mpco(src, dst, model_h5=fx["frame"])
    before = {p: _sha(p) for p in old}
    real, calls = atomic.replace_with_retry, []

    def fails(a: Any, b: Any) -> None:
        calls.append(b)
        if len(calls) == fail_at:  # 1-2 back the old parts up, 3-4 move in
            raise OSError("disk gone")
        real(a, b)

    monkeypatch.setattr(atomic, "replace_with_retry", fails)
    with pytest.raises(OSError, match="disk gone"):
        convert_mpco(src, dst, model_h5=fx["frame"], overwrite=True)
    monkeypatch.undo()
    assert {p: _sha(p) for p in old} == before
    assert _left_behind(tmp_path) == sorted(p.name for p in old)


# ---------------------------------------------------------------------------
# META -> COLUMN_MAP block parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block, n, want", [
    ("0.N,M_1,M_2", 3, ([0], ["N", "M_1", "M_2"])),
    ("0.1.4.sigma11,sigma22,sigma12", 3, ([0, 1, 4], ["sigma11", "sigma22", "sigma12"])),
    ("0.1.2.3.4.sigma11", 1, ([0, 1, 2, 3, 4], ["sigma11"])),
    ("0.1.7", 1, ([0, 1], ["7"])),          # a numeric component name
    ("0.1.a.b,c", 2, ([0, 1], ["a.b", "c"])),  # a dotted component name
])
def test_meta_blocks_split_into_path_and_names(block, n, want) -> None:
    from apeGmsh.results.readers._ladruno_from_mpco import _split_block

    assert _split_block(block, n, "x") == want


def test_a_meta_block_that_does_not_split_refuses() -> None:
    from apeGmsh.results.readers._ladruno_from_mpco import _split_block

    with pytest.raises(MPCOConversionRefused, match="does not split"):
        _split_block("x.y", 1, "where")
