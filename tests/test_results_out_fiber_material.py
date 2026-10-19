"""The ``.out`` route for fibres and per-Gauss-point material states (#1982).

Emit plan, refusals and the decode, without a solver. The values against a
closed form, on a live backend, are in
``tests/opensees/live/test_out_fiber_material_roundtrip_live.py``.

* A ``fibers`` record emits ``section fiberData2`` (every section) and its
  ``integrationPoints`` pair; a ``damage`` gauss record one ``material <gp>
  damage`` recorder per Gauss point of its class.
* The decode is held to files written here with a distinct value in every
  cell, so a transposed axis or an off-by-one point reads a wrong number.
* Fail closed: an unknown class, a non-fibre component, a missing translator
  (emit or read) and a ``layers`` record each raise by name.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._response_catalog import RESPONSE_CATALOG
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.section.fiber import FiberPoint
from apeGmsh.results import Results
from apeGmsh.results.spec._emit import emit_logical
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.transcoders import RecorderTranscoder

FIBRES = ((-0.3, -0.1), (-0.3, 0.1), (0.3, -0.1), (0.3, 0.1))
AREAS = (1.0e-2, 2.0e-2, 3.0e-2, 4.0e-2)
LOBATTO_3 = (-1.0, 0.0, 1.0)
TIMES = (0.5, 1.0)


class FakeOps:
    """Records ``recorder`` calls; returns sequential tags."""

    def __init__(self) -> None:
        self.recorder_calls: list[tuple] = []

    def recorder(self, *args: Any) -> int:
        self.recorder_calls.append(args)
        return len(self.recorder_calls)

    def remove(self, *args: Any) -> None:
        pass


def _fibres(ids: Any, cls: str | None = "ForceBeamColumn3d",
            comps: tuple[str, ...] = ("fiber_stress", "fiber_strain"),
            ) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="fibers", name="fib", components=comps, dt=None,
        n_steps=None, element_ids=np.asarray(ids), element_class_name=cls)


def _damage(ids: Any, cls: str | None = "Brick") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="gauss", name="dmg",
        components=("damage_tension", "damage_compression"), dt=None,
        n_steps=None, element_ids=np.asarray(ids), element_class_name=cls)


def _layers() -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="layers", name="shell_layers", components=("fiber_stress",),
        dt=None, n_steps=None, element_ids=np.asarray([1]))


# =====================================================================
# Models: a two-element fibre beam and a one-hex cube (h5 only, no solver)
# =====================================================================

@pytest.fixture(scope="module")
def beam(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("beam")
    with apeGmsh(model_name="out_fib_beam", verbose=False) as g:
        geo = g.model.geometry
        a, b = geo.add_point(0.0, 0.0, 0.0), geo.add_point(2.0, 0.0, 0.0)
        line = geo.add_line(a, b)
        g.model.sync()
        g.physical.add(1, [line], name="Beam")
        g.mesh.structured.set_transfinite_curve(line, 3)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=2.0e5)
    sec = ops.section.Fiber(fibers=tuple(
        FiberPoint(material=mat, y=y, z=z, area=area)
        for (y, z), area in zip(FIBRES, AREAS)), GJ=1.0e6)
    ops.element.forceBeamColumn(
        pg="Beam", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
        integration=ops.beamIntegration.Lobatto(section=sec, n_ip=3))
    model_h5 = tmp / "model.h5"
    ops.h5(str(model_h5))
    ids = sorted(int(e) for e in fem.elements.select(pg="Beam").ids)
    return {"fem": fem, "model_h5": model_h5, "ids": ids,
            "model": OpenSeesModel.from_h5(model_h5), "mat_tag": ops.tag_for(mat)}


@pytest.fixture(scope="module")
def cube(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("cube")
    with apeGmsh(model_name="out_dmg_cube", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="cube")
        g.physical.add(3, "cube", name="cube")
        g.mesh.structured.set_transfinite("cube", n=2)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=3)
    ops.element.stdBrick(pg="cube", material=ops.register(
        ElasticIsotropic(E=3.0e4, nu=0.2)))
    model_h5 = tmp / "model.h5"
    ops.h5(str(model_h5))
    ids = [int(e) for e in fem.elements.select(pg="cube").ids]
    return {"fem": fem, "model_h5": model_h5, "ids": ids}


def _spec(fem: Any, *records: ResolvedRecorderRecord) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=records)


def _write(path: str, rows: np.ndarray) -> None:
    """An ``.out`` file: the time column, then ``rows[t]``."""
    body = np.column_stack([np.asarray(TIMES)[: rows.shape[0]], rows])
    np.savetxt(path, body, fmt="%.17g")


def _transcode(m: dict[str, Any], spec: ResolvedRecorderSpec, out: Path,
               target: Path, *, model: bool = True) -> Path:
    return RecorderTranscoder(
        spec, out, target, m["fem"],
        model_h5_src=m["model_h5"] if model else None).run()


# =====================================================================
# Emit
# =====================================================================

def test_fibres_emit_fiberdata2_and_its_integration_points_pair() -> None:
    lrs = emit_logical(_fibres([4, 9]), output_dir="d", stage_id="s")
    assert [lr.response_tokens for lr in lrs] == [
        ("section", "fiberData2"), ("integrationPoints",)]
    assert [lr.file_path for lr in lrs] == [
        "d/s__fib_fibers.out", "d/s__fib_fibers_gpx.out"]
    assert all(lr.target_ids == (4, 9) and lr.target_kind == "ele"
               for lr in lrs)


@pytest.mark.parametrize("rec, match", [
    (_fibres([1], cls=None), "names element class None"),
    (_fibres([1], cls="ElasticBeam3d"), "'ElasticBeam3d'"),
    (_fibres([1], comps=("fiber_stress", "stress_xx")), "'stress_xx'"),
    (_fibres([1], comps=()), "not fibre responses"),
])
def test_fibre_record_refuses_what_it_cannot_read_back(
    rec: ResolvedRecorderRecord, match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        emit_logical(rec)


@pytest.mark.parametrize("cls, n_gp", [("Brick", 8), ("FourNodeQuad", 4)])
def test_damage_fans_out_one_material_recorder_per_gauss_point(
    cls: str, n_gp: int,
) -> None:
    lrs = emit_logical(_damage([3, 5], cls=cls), output_dir="d")
    assert [lr.response_tokens for lr in lrs] == [
        ("material", str(gp), "damage") for gp in range(1, n_gp + 1)]
    assert [lr.file_path for lr in lrs] == [
        f"d/dmg_gauss_material{gp}.out" for gp in range(1, n_gp + 1)]


@pytest.mark.parametrize("cls", [None, "Tri31"])
def test_damage_without_a_catalogued_class_refuses(cls: str | None) -> None:
    with pytest.raises(ValueError, match="per Gauss point"):
        emit_logical(_damage([3], cls=cls))


def test_stress_records_keep_the_element_level_token() -> None:
    rec = ResolvedRecorderRecord(
        category="gauss", name="g", components=("stress_xx",), dt=None,
        n_steps=None, element_ids=np.asarray([3]), element_class_name="Brick")
    assert [lr.response_tokens for lr in emit_logical(rec)] == [("stresses",)]


def test_catalog_gauss_order_is_brick_cpp_order() -> None:
    """``material <gp>`` follows Brick.cpp (``gp - 1 = 4 i + 2 j + k`` at
    ``(s_i, s_j, s_k)``, probe #1976); the decode puts file ``gp`` in
    catalog slot ``gp - 1``, so the catalog must list that order."""
    s = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
    want = [(s[(g // 4)], s[(g // 2) % 2], s[g % 2]) for g in range(8)]
    for (cls, _rule, tok), layout in RESPONSE_CATALOG.items():
        if cls == "Brick" and tok in ("damage", "stress"):
            np.testing.assert_allclose(layout.natural_coords, want, atol=1e-15)


# =====================================================================
# Fail closed: layers by name, and the translator is mandatory
# =====================================================================

def test_layers_refused_by_name_on_every_out_route(
    beam: dict[str, Any], tmp_path: Path,
) -> None:
    spec = _spec(beam["fem"], _layers())
    with pytest.raises(ValueError, match="'shell_layers'.*'layers'"):
        emit_logical(_layers())
    fake = FakeOps()
    with pytest.raises(ValueError, match="'shell_layers'.*'layers'"):
        with spec.emit_recorders(str(tmp_path), ops=fake, model=beam["model"]):
            pass
    assert fake.recorder_calls == []
    for line in (*spec.to_tcl_commands(model=beam["model"]),
                 *spec.to_python_commands(model=beam["model"])):
        assert "REFUSED" in line and "'shell_layers'" in line
        assert "category='layers'" in line
    with pytest.raises(ValueError, match="'shell_layers'.*'layers'"):
        _transcode(beam, spec, tmp_path, tmp_path / "r.h5")


def test_fibres_and_damage_refuse_without_the_translator(
    beam: dict[str, Any], cube: dict[str, Any], tmp_path: Path,
) -> None:
    for m, rec in ((beam, _fibres(beam["ids"])), (cube, _damage(cube["ids"]))):
        spec = _spec(m["fem"], rec)
        fake = FakeOps()
        with pytest.raises(ValueError, match="model="):
            with spec.emit_recorders(str(tmp_path), ops=fake):
                pass
        assert fake.recorder_calls == []          # refused before any recorder
        with pytest.raises(ValueError, match="model="):
            spec.to_tcl_commands()
        for lr in emit_logical(rec, output_dir=str(tmp_path)):
            Path(lr.file_path).write_text("0.5 0\n")
        with pytest.raises(ValueError, match="Pass the model"):
            _transcode(m, spec, tmp_path, tmp_path / "r.h5", model=False)


def test_fibres_emit_translated_ids(beam: dict[str, Any], tmp_path: Path) -> None:
    spec = _spec(beam["fem"], _fibres(beam["ids"]))
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake, model=beam["model"]) as live:
        live.begin_stage("s", kind="static")
        live.end_stage()
    tags = {int(r.fem_eid): int(r.tag) for r in beam["model"].elements()}
    want = tuple(tags[e] for e in beam["ids"])
    assert want != tuple(beam["ids"])            # the translation is exercised
    for call in fake.recorder_calls:
        ele = call.index("-ele")
        assert call[ele + 1: ele + 1 + len(want)] == want
    assert fake.recorder_calls[0][-2:] == ("section", "fiberData2")
    assert fake.recorder_calls[1][-1] == "integrationPoints"
    assert len(fake.recorder_calls) == 2


# =====================================================================
# Decode
# =====================================================================

def _fibre_files(beam: dict[str, Any], out: Path, *, skew: bool = False,
                 ) -> np.ndarray:
    """Write the fibre and gpx files; return the values ``(T, E, IP, F, 2)``.

    ``skew`` moves one fibre of the second element: its table then differs
    from the first's, as a record of two fibre sections would.
    """
    rec = _fibres(beam["ids"])
    fib_lr, gpx_lr = emit_logical(rec, output_dir=str(out))
    n_e, n_ip, n_f = len(beam["ids"]), 3, len(FIBRES)
    values = (1.0 + np.arange(len(TIMES) * n_e * n_ip * n_f * 2, dtype=float)
              ).reshape(len(TIMES), n_e, n_ip, n_f, 2) / 7.0
    geom = np.array([(y, z, a, beam["mat_tag"]) for (y, z), a in zip(FIBRES, AREAS)])
    geom = np.broadcast_to(geom, (n_e, n_ip, n_f, 4)).copy()
    if skew:
        geom[1, :, 0, 0] += 0.05
    rows = np.concatenate([np.broadcast_to(geom, values.shape[:-1] + (4,)), values],
                          axis=-1).reshape(len(TIMES), -1)
    _write(fib_lr.file_path, rows)
    # integrationPoints: physical x*L per element (length 1 each).
    _write(gpx_lr.file_path, np.tile([0.0, 0.5, 1.0] * n_e, (len(TIMES), 1)))
    return values


def test_fibre_decode_is_element_section_fibre_ordered(
    beam: dict[str, Any], tmp_path: Path,
) -> None:
    values = _fibre_files(beam, tmp_path)
    spec = _spec(beam["fem"], _fibres(beam["ids"]))
    target = _transcode(beam, spec, tmp_path, tmp_path / "r.h5")
    r = Results.from_native(target, fem=beam["fem"], model=beam["model"])
    n_e, n_ip, n_f = values.shape[1:4]
    for k, comp in enumerate(("fiber_stress", "fiber_strain")):
        slab = r.elements.fibers.get(component=comp)
        np.testing.assert_array_equal(slab.values, values[..., k].reshape(len(TIMES), -1))
        np.testing.assert_array_equal(slab.time, TIMES)
        np.testing.assert_array_equal(
            slab.element_index, np.repeat(beam["ids"], n_ip * n_f))
        np.testing.assert_array_equal(
            slab.gp_index, np.tile(np.repeat(np.arange(n_ip), n_f), n_e))
        np.testing.assert_array_equal(slab.y, np.tile([y for y, _ in FIBRES], n_e * n_ip))
        np.testing.assert_array_equal(slab.z, np.tile([z for _, z in FIBRES], n_e * n_ip))
        np.testing.assert_array_equal(slab.area, np.tile(AREAS, n_e * n_ip))
        np.testing.assert_array_equal(slab.material_tag, beam["mat_tag"])
        np.testing.assert_allclose(
            slab.station_natural_coord,
            np.tile(np.repeat(LOBATTO_3, n_f), n_e), atol=1e-15)


def test_fibre_decode_refuses_two_fibre_tables(
    beam: dict[str, Any], tmp_path: Path,
) -> None:
    """Mutation guard: the column count of two equal-count tables is the
    same as one's, so only the table check stops a mislabelled read."""
    _fibre_files(beam, tmp_path, skew=True)
    spec = _spec(beam["fem"], _fibres(beam["ids"]))
    with pytest.raises(ValueError, match="fibre tables"):
        _transcode(beam, spec, tmp_path, tmp_path / "r.h5")


def test_fibre_decode_refuses_a_partial_row(
    beam: dict[str, Any], tmp_path: Path,
) -> None:
    _fibre_files(beam, tmp_path)
    fib_lr, _gpx = emit_logical(_fibres(beam["ids"]), output_dir=str(tmp_path))
    rows = np.loadtxt(fib_lr.file_path)[:, 1:-6]
    _write(fib_lr.file_path, rows)                # one fibre short
    spec = _spec(beam["fem"], _fibres(beam["ids"]))
    with pytest.raises(ValueError, match="data columns"):
        _transcode(beam, spec, tmp_path, tmp_path / "r.h5")


def test_damage_decode_puts_file_gp_at_catalog_point_gp(
    cube: dict[str, Any], tmp_path: Path,
) -> None:
    rec = _damage(cube["ids"])
    lrs = emit_logical(rec, output_dir=str(tmp_path))
    per_gp = [(gp + 1.0) / 16.0 + np.arange(len(TIMES) * 2).reshape(len(TIMES), 2)
              for gp in range(len(lrs))]
    for lr, rows in zip(lrs, per_gp):
        _write(lr.file_path, rows)
    target = _transcode(cube, _spec(cube["fem"], rec), tmp_path, tmp_path / "r.h5")
    model = OpenSeesModel.from_h5(cube["model_h5"])
    r = Results.from_native(target, fem=cube["fem"], model=model)
    for k, comp in enumerate(("damage_tension", "damage_compression")):
        slab = r.elements.gauss.get(component=comp)
        np.testing.assert_array_equal(
            slab.values, np.stack([rows[:, k] for rows in per_gp], axis=1))
        np.testing.assert_array_equal(slab.element_index, cube["ids"] * 8)


def test_damage_decode_refuses_another_width(
    cube: dict[str, Any], tmp_path: Path,
) -> None:
    rec = _damage(cube["ids"])
    for lr in emit_logical(rec, output_dir=str(tmp_path)):
        _write(lr.file_path, np.ones((len(TIMES), 3)))    # a 3-value damage
    with pytest.raises(ValueError, match="data columns"):
        _transcode(cube, _spec(cube["fem"], rec), tmp_path, tmp_path / "r.h5")


# =====================================================================
# Integration-point counts: .out carries none per element
# =====================================================================

@pytest.fixture(scope="module")
def mixed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Two one-element force beams on one section, Lobatto 3 and Lobatto 5:
    8 sections in all, which divide evenly into 4 per element."""
    tmp = tmp_path_factory.mktemp("mixed")
    with apeGmsh(model_name="out_fib_mixed", verbose=False) as g:
        geo = g.model.geometry
        a, b, c = (geo.add_point(x, 0.0, 0.0) for x in (0.0, 1.0, 2.0))
        lines = [geo.add_line(a, b), geo.add_line(b, c)]
        g.model.sync()
        for line, pg in zip(lines, ("B1", "B2")):
            g.physical.add(1, [line], name=pg)
            g.mesh.structured.set_transfinite_curve(line, 2)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=2.0e5)
    sec = ops.section.Fiber(fibers=tuple(
        FiberPoint(material=mat, y=y, z=z, area=area)
        for (y, z), area in zip(FIBRES, AREAS)), GJ=1.0e6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    for pg, n_ip in (("B1", 3), ("B2", 5)):
        ops.element.forceBeamColumn(
            pg=pg, transf=tr,
            integration=ops.beamIntegration.Lobatto(section=sec, n_ip=n_ip))
    model_h5 = tmp / "model.h5"
    ops.h5(str(model_h5))
    ids = [int(fem.elements.select(pg=pg).ids[0]) for pg in ("B1", "B2")]
    return {"fem": fem, "model_h5": model_h5, "ids": ids,
            "mat_tag": ops.tag_for(mat)}


def _uniform_files(m: dict[str, Any], out: Path, gpx_row: list[float]) -> None:
    """A fibre file of one fibre table at every section (``len(gpx_row)``
    sections in all) and the given ``integrationPoints`` row."""
    fib_lr, gpx_lr = emit_logical(_fibres(m["ids"]), output_dir=str(out))
    table = [v for (y, z), a in zip(FIBRES, AREAS)
             for v in (y, z, a, m["mat_tag"], 1.0, 2.0)]
    _write(fib_lr.file_path, np.tile(table * len(gpx_row), (len(TIMES), 1)))
    _write(gpx_lr.file_path, np.tile(gpx_row, (len(TIMES), 1)))


def test_fibre_decode_refuses_mixed_integration_point_counts(
    mixed: dict[str, Any], tmp_path: Path,
) -> None:
    """3 + 5 Lobatto points split evenly into 4 + 4 with equal tables; only
    the model's counts tell the record mixes them."""
    _uniform_files(mixed, tmp_path,
                   [0.0, 0.5, 1.0, 0.0, 0.1727, 0.5, 0.8273, 1.0])
    spec = _spec(mixed["fem"], _fibres(mixed["ids"]))
    with pytest.raises(ValueError, match="integration-point counts"):
        _transcode(mixed, spec, tmp_path, tmp_path / "r.h5")


def test_fibre_decode_refuses_stations_out_of_order(
    beam: dict[str, Any], tmp_path: Path,
) -> None:
    """The second guard: within an element, stations increase strictly."""
    _uniform_files(beam, tmp_path, [0.0, 0.5, 1.0, 1.0, 0.5, 0.0])
    spec = _spec(beam["fem"], _fibres(beam["ids"]))
    with pytest.raises(ValueError, match="strictly increasing"):
        _transcode(beam, spec, tmp_path, tmp_path / "r.h5")


def test_fibre_decode_refuses_a_count_the_model_contradicts(
    beam: dict[str, Any], tmp_path: Path,
) -> None:
    """Both elements are Lobatto 3; a file of 2 stations each is refused."""
    _uniform_files(beam, tmp_path, [0.0, 1.0, 0.0, 1.0])
    spec = _spec(beam["fem"], _fibres(beam["ids"]))
    with pytest.raises(ValueError, match="the model gives 3"):
        _transcode(beam, spec, tmp_path, tmp_path / "r.h5")
