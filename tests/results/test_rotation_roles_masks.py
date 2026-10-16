"""#2029 / #2028: rotation masks key on the model's DOF roles, not on ndf.

A 2-D ``LadrunoUP`` vertex node and a 2-D frame node both have ndf 3, and
the emit's ``_node_layout(3, 2)`` lays both third DOFs in the ``rotation_z``
slot: (ux, uy, rz) for the frame, but (ux, uy, p) for the u-p node. A mask
keyed on ndf/ndm alone therefore returned the pore pressure as
``rotation_z``. The masks now also ask the model's DOF roles
(``_readback.RotationRoles``, the table the ``recorder.declare`` emit uses),
so the u-p node reads NaN and the frame node keeps its rotation, on all
three routes: ``.out`` (``mask_absent_rotations``), ``.ladruno`` and MPCO
(``RotationMask``), one file or partitioned.

The oracle is closed form. Every slot of every file holds
``_value(step, node, dof)``, non-zero and distinct, so a kept value must
equal it exactly and a masked one is NaN; which nodes are which comes from
the model's own physical groups. The model is a real bridge archive
(``apeSees(fem).h5``), so the per-node ndf and the element plan are the
bridge's own; no OpenSees process runs.

#2028: a partitioned reader forwards the ndf and roles to every part (it
used to forward nothing, so the parts stayed unmasked), and a model-less
partitioned read warns once, not once per part.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno import (
    LadrunoReader,
    attach_model_nodes_ndf as attach_ladruno,
)
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._mpco import (
    MPCOReader,
    attach_model_nodes_ndf as attach_mpco,
)
from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader
from apeGmsh.results.readers._mpco_nodal_io import AbsentDofWarning
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.transcoders import RecorderTranscoder

from tests.conftest import _open_model_from_h5

pytest.importorskip("gmsh")

TIME = (0.5, 1.0)
FORMATS = ["ladruno", "mpco"]


def _value(k: int, nid: int, dof: int) -> float:
    """Non-zero and distinct in every slot: a kept value is exact."""
    return 1.0 + 0.01 * nid + 0.001 * dof + 0.0001 * k


# ---------------------------------------------------------------------------
# The model: a 2-D LadrunoUP soil block beside a 2-D frame column
# ---------------------------------------------------------------------------


def _up_beside_frame(g: Any, tmp_path: Path) -> tuple[Any, Path, list[int], list[int]]:
    """``(fem, model_h5, up_nodes, frame_nodes)``; every node has ndf 3."""
    G = g.model.geometry
    block = G.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    col = G.add_line(G.add_point(3.0, 0.0, 0.0), G.add_point(3.0, 2.0, 0.0))
    g.model.sync()
    g.physical.add(2, [block], name="Soil")
    g.physical.add(1, [col], name="Col")
    g.mesh.structured.set_recombine("Soil", dim=2)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(2)
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    ops.element.LadrunoUP(
        pg="Soil", material=ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0),
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4),
    )
    ops.element.elasticBeamColumn(
        pg="Col", transf=ops.geomTransf.Linear(), A=0.01, E=200e9, Iz=1e-4)
    model_h5 = tmp_path / "model.h5"
    ops.h5(str(model_h5))
    up = sorted({int(n) for n in fem.nodes.select(pg="Soil").ids})
    frame = sorted({int(n) for n in fem.nodes.select(pg="Col").ids})
    with h5py.File(model_h5, "r") as f:
        grp = f["opensees/nodes_ndf"]
        ndf = {int(t): int(n) for t, n in zip(grp["tags"][...], grp["ndf"][...])}
    # The premise: ndf and ndm alone cannot tell the two nodes apart.
    assert {ndf[n] for n in up} == {3} and {ndf[n] for n in frame} == {3}
    return fem, model_h5, up, frame


@pytest.fixture
def model(g: Any, tmp_path: Path) -> tuple[Any, Path, list[int], list[int]]:
    return _up_beside_frame(g, tmp_path)


# ---------------------------------------------------------------------------
# .out route (RecorderTranscoder -> mask_absent_rotations)
# ---------------------------------------------------------------------------


def _transcode(fem: Any, tmp_path: Path, node_ids: list[int],
               model_h5: Path | None) -> Results:
    comps = ("displacement_x", "displacement_y", "rotation_z")
    dofs = [1, 2, 6]        # the reader's file convention: rz is DOF 6
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    rows = []
    for k, t in enumerate(TIME):
        cols = [t] + [_value(k, n, d) for n in node_ids for d in dofs]
        rows.append(" ".join(f"{v:.10g}" for v in cols))
    (out / "all_disp.out").write_text("\n".join(rows) + "\n", encoding="utf-8")
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        ResolvedRecorderRecord(
            category="nodes", name="all", components=comps,
            dt=None, n_steps=None, node_ids=np.array(node_ids),
        ),
    ))
    target = tmp_path / ("with_model.h5" if model_h5 else "no_model.h5")
    RecorderTranscoder(spec, out, target, fem, model_h5_src=model_h5).run()
    return Results.from_native(target, fem=fem,
                               model=_open_model_from_h5(target))


def _last(r: Results, comp: str) -> dict[int, float]:
    slab = r.nodes.get(component=comp)
    return {int(i): float(v) for i, v in zip(slab.node_ids,
                                             np.asarray(slab.values)[-1])}


def test_out_up_node_reads_nan_rotation_z_frame_keeps_it(
    model: tuple, tmp_path: Path,
) -> None:
    fem, model_h5, up, frame = model
    r = _transcode(fem, tmp_path, up + frame, model_h5)
    k = len(TIME) - 1
    rz = _last(r, "rotation_z")
    assert all(np.isnan(rz[n]) for n in up), {n: rz[n] for n in up}
    assert all(rz[n] == pytest.approx(_value(k, n, 6), rel=1e-12)
               for n in frame)
    for d, comp in ((1, "displacement_x"), (2, "displacement_y")):
        got = _last(r, comp)
        assert all(got[n] == pytest.approx(_value(k, n, d), rel=1e-12)
                   for n in up + frame), comp


# ---------------------------------------------------------------------------
# .ladruno / MPCO routes (RotationMask), one file and partitioned
# ---------------------------------------------------------------------------


def _write_file(path: Path, fmt: str, node_ids: list[int],
                up_conn: list[int]) -> Path:
    """A 2-D ``.ladruno`` or ``.mpco`` holding ``DISPLACEMENT`` (Ux, Uy)
    and ``ROTATION`` (Rz) for ``node_ids`` over two steps. Column ``Rz``
    holds ``_value(step, node, 6)``: what the recorder writes in that slot,
    a rotation at a frame node and a pore pressure at a u-p node.
    ``MODEL/ELEMENTS`` lists a ``LadrunoUP`` over ``up_conn``, so a
    model-less rotation read warns (a continuum class)."""
    ids = np.array(node_ids, dtype=np.int32).reshape(-1, 1)
    with h5py.File(path, "w") as f:
        if fmt == "ladruno":
            info = f.create_group("INFO")
            info.attrs["GENERATOR"] = np.array([b"Ladruno"])
            info.attrs["FORMAT_VERSION"] = np.array([1], dtype=np.int32)
            info.attrs["SPATIAL_DIM"] = np.array([2], dtype=np.int32)
        stage = f.create_group("MODEL_STAGE[1]")
        if fmt == "ladruno":
            stage.attrs["KIND"] = np.array([b"static"])
        nodes = stage.create_group("MODEL/NODES")
        nodes.create_dataset("ID", data=ids)
        nodes.create_dataset("COORDINATES", data=np.zeros((len(node_ids), 2)))
        conn = np.array([[100, *up_conn]], dtype=np.int32)
        key = "MODEL/ELEMENTS/33019-LadrunoUP[0:0]"
        if fmt == "ladruno":
            stage.create_group(key).create_dataset("CONNECTIVITY", data=conn)
        else:
            stage.create_dataset(key, data=conn)
        on_nodes = stage.create_group("RESULTS/ON_NODES")
        for name, labels, dofs in (("DISPLACEMENT", "Ux,Uy", (1, 2)),
                                   ("ROTATION", "Rz", (6,))):
            grp = on_nodes.create_group(name)
            grp.attrs["COMPONENTS"] = np.array([labels.encode()])
            grp.create_dataset("ID", data=ids)
            block = np.array([[[_value(k, n, d) for d in dofs]
                               for n in node_ids] for k in range(len(TIME))])
            if fmt == "ladruno":
                grp.create_dataset("DATA", data=block)
                grp.create_dataset("STEP", data=np.array([1, 2], dtype=np.int32))
                grp.create_dataset("TIME", data=np.array(TIME))
            else:
                for k, t in enumerate(TIME):
                    ds = grp.create_dataset(f"DATA/STEP_{k}", data=block[k])
                    ds.attrs["TIME"] = np.array([t])
    return path


def _open(fmt: str, paths: list[Path]) -> Any:
    if len(paths) == 1:
        return LadrunoReader(paths[0]) if fmt == "ladruno" else MPCOReader(paths[0])
    return (LadrunoMultiPartitionReader(paths) if fmt == "ladruno"
            else MPCOMultiPartitionReader(paths))


def _files(tmp_path: Path, fmt: str, up: list[int], frame: list[int],
           parts: int) -> list[Path]:
    """One file over every node, or two part files that split both the
    u-p nodes and the frame nodes, so each part holds both kinds."""
    nodes = up + frame
    if parts == 1:
        return [_write_file(tmp_path / f"r.{fmt}", fmt, nodes, up[:4])]
    halves = (up[: len(up) // 2] + frame[:1], up[len(up) // 2:] + frame[1:])
    return [
        _write_file(tmp_path / f"r.part-{i}.{fmt}", fmt, half,
                    [n for n in half if n in up][:4])
        for i, half in enumerate(halves)
    ]


def _read_last(reader: Any, comp: str) -> dict[int, float]:
    stage = reader.stages()[0].id
    slab = reader.read_nodes(stage, comp)
    return {int(i): float(v) for i, v in zip(slab.node_ids, slab.values[-1])}


@pytest.mark.parametrize("parts", [1, 2], ids=["one-file", "partitioned"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_reader_up_node_reads_nan_rotation_z_frame_keeps_it(
    model: tuple, tmp_path: Path, fmt: str, parts: int,
) -> None:
    """With the model attached through the production path
    (``attach_model_nodes_ndf``, as ``Results.from_*`` calls it), the u-p
    nodes read NaN ``rotation_z`` on every part, the frame nodes keep their
    rotation, and no warning fires (the model answers every node)."""
    _fem, model_h5, up, frame = model
    paths = _files(tmp_path, fmt, up, frame, parts)
    attach = attach_ladruno if fmt == "ladruno" else attach_mpco
    k = len(TIME) - 1
    with _open(fmt, paths) as r, warnings.catch_warnings():
        warnings.simplefilter("error", AbsentDofWarning)
        attach(r, model_h5, 2)
        rz = _read_last(r, "rotation_z")
        assert sorted(rz) == sorted(up + frame)
        assert all(np.isnan(rz[n]) for n in up), {n: rz[n] for n in up}
        assert all(rz[n] == _value(k, n, 6) for n in frame)
        ux = _read_last(r, "displacement_x")
        assert all(ux[n] == _value(k, n, 1) for n in up + frame)


@pytest.mark.parametrize("fmt", FORMATS)
def test_partitioned_read_without_a_model_warns_once(
    model: tuple, tmp_path: Path, fmt: str,
) -> None:
    """No model attached: every part holds a u-p element, and the read
    warns ``AbsentDofWarning`` once per partitioned reader, not once per
    part (#2028), however many rotation reads follow."""
    _fem, _model_h5, up, frame = model
    paths = _files(tmp_path, fmt, up, frame, 2)
    with _open(fmt, paths) as r, warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _read_last(r, "rotation_z")
        _read_last(r, "rotation_z")
    hits = [x for x in w if issubclass(x.category, AbsentDofWarning)]
    assert len(hits) == 1, [str(x.message) for x in hits]
    assert "model_h5=" in str(hits[0].message)


@pytest.mark.parametrize("fmt", FORMATS)
def test_an_unknown_reader_type_refuses_the_attach(
    model: tuple, fmt: str,
) -> None:
    """A reader that carries no rotation mask is refused by name, never
    left unmasked without a word (it used to return silently)."""
    _fem, model_h5, _up, _frame = model
    attach = attach_ladruno if fmt == "ladruno" else attach_mpco
    with pytest.raises(TypeError, match="object is none of"):
        attach(object(), model_h5, 2)
