"""A ``.out`` line-stations record never reads one element's station as
another's (#2246, found by the Opus review of #2243).

The section-force file and its ``integrationPoints`` pair carry one column
block per element and no per-element count, so the transcoder slices them
into equal blocks. Two elements with 3 and 5 Lobatto points give 4 columns
each, and on ``main`` before #2246 the record read without error: element 1
gained element 2's first station, and element 2 lost it.

Two checks now hold it:

* with a bound model, each element's count comes from its
  ``beamIntegration``; unequal counts are refused, naming the elements, and
  equal ones prove the slice;
* without one, each element's stations must increase strictly within
  [-1, 1].

The Legendre case shows why the model check is needed beside the station
check: interior stations and a 12x length ratio make every mis-sliced block
increase within [-1, 1]. The long-hinge HingeRadau case shows why a proven
slice skips the order check: its legal stations are out of order. Uniform
records read exactly as before, and a ``from_recorders`` cache made by the
previous parser is not served.

The unit cases hand-write the files; the ``live`` case records the issue's
reproducer (two ``ForceBeamColumn2d``, one section, Lobatto 3 and 5) with
openseespy, and the record split by count is held to ``eleResponse``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.results import Results
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.transcoders._recorder import RecorderTranscoder

from tests.conftest import _open_model_from_h5

COMPS = ("axial_force", "bending_moment_z")
LOBATTO = {3: np.array([-1.0, 0.0, 1.0]),
           5: np.array([-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0])}
LEGENDRE = {2: np.array([-1.0, 1.0]) / np.sqrt(3.0),
            4: np.polynomial.legendre.leggauss(4)[0]}
UNORDERED = "do not increase strictly within"
TIMES = np.array([1.0, 2.0])

Rule = Callable[[apeSees, Any], Any]


def _lobatto(n: int) -> Rule:
    return lambda ops, sec: ops.beamIntegration.Lobatto(section=sec, n_ip=n)


def _legendre(n: int) -> Rule:
    return lambda ops, sec: ops.beamIntegration.Legendre(section=sec, n_ip=n)


def _hinge_radau(ops: apeSees, sec: Any, lp: float = 0.1) -> Any:
    return ops.beamIntegration.HingeRadau(
        section_i=sec, lp_i=lp, section_j=sec, lp_j=lp, section_interior=sec)


#: HingeRadau with lp_i = lp_j = 0.15 on L = 0.75, as OpenSees places its
#: stations (``eleResponse(tag, "integrationPoints")``): out of order.
HINGE_LONG = np.array([-1.0, 1.0 / 15.0, 0.2 * np.sqrt(3.0),
                       -0.2 * np.sqrt(3.0), -1.0 / 15.0, 1.0])
#: The ``PARSER_VERSION`` a cache written before this fix carries.
PREVIOUS_PARSER = "1.2"


def _two_beams(name: str, lengths: tuple[float, float]) -> tuple[Any, list[int]]:
    """Two collinear one-element lines along ``+x``, in groups B1 and B2."""
    with apeGmsh(model_name=name, verbose=False) as g:
        geo = g.model.geometry
        a = geo.add_point(0.0, 0.0, 0.0)
        b = geo.add_point(lengths[0], 0.0, 0.0)
        c = geo.add_point(lengths[0] + lengths[1], 0.0, 0.0)
        l1, l2 = geo.add_line(a, b), geo.add_line(b, c)
        g.model.sync()
        g.physical.add(1, [l1], name="B1")
        g.physical.add(1, [l2], name="B2")
        g.physical.add(0, [a], name="Base")
        g.physical.add(0, [c], name="Tip")
        g.mesh.structured.set_transfinite_curve(l1, 2)
        g.mesh.structured.set_transfinite_curve(l2, 2)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    eids = [int(fem.elements.select(pg=pg).ids[0]) for pg in ("B1", "B2")]
    return fem, eids


def _bridge(fem: Any, rules: tuple[Rule, Rule]) -> apeSees:
    """A 2-D ``forceBeamColumn`` per group, one elastic section, ``rules``."""
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    sec = ops.section.Elastic(E=2.0e5, A=1.0e-2, Iz=1.0e-4)
    tr = ops.geomTransf.Linear()
    for pg, rule in zip(("B1", "B2"), rules):
        ops.element.forceBeamColumn(pg=pg, transf=tr, integration=rule(ops, sec))
    return ops


def _model_h5(tmp: Path, fem: Any, rules: tuple[Rule, Rule]) -> Path:
    path = tmp / "model.h5"
    _bridge(fem, rules).h5(str(path))
    return path


def _record(name: str, eids: list[int]) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="line_stations", name=name, components=COMPS, dt=None,
        n_steps=None, element_ids=np.asarray(eids, dtype=np.int64),
        element_class_name="ForceBeamColumn2d")


def _value(t: int, e: int, s: int, k: int) -> float:
    return t * 1000.0 + e * 100.0 + s * 10.0 + k


def _write_files(tmp: Path, lengths: tuple[float, ...],
                 xis: tuple[np.ndarray, ...]) -> None:
    """The files OpenSees writes for one record over elements whose natural
    stations are ``xis``: element-major blocks, as many as each one has."""
    gpx = np.concatenate([(x + 1.0) * 0.5 * L for x, L in zip(xis, lengths)])
    sec = [[_value(t, e, s, k) for e, x in enumerate(xis)
            for s in range(x.size) for k in range(len(COMPS))]
           for t in range(TIMES.size)]
    np.savetxt(tmp / "beam_line_stations_gpx.out",
               np.column_stack([TIMES, np.tile(gpx, (TIMES.size, 1))]))
    np.savetxt(tmp / "beam_line_stations.out",
               np.column_stack([TIMES, np.asarray(sec)]))


def _transcode(tmp: Path, fem: Any, eids: list[int],
               model_h5: Path | None) -> Path:
    target = tmp / "out.h5"
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id,
                                records=(_record("beam", eids),))
    RecorderTranscoder(spec, output_dir=tmp, target_path=target, fem=fem,
                       stage_name="s", stage_kind="static",
                       model_h5_src=model_h5).run()
    return target


# =====================================================================
# Mixed counts: refused
# =====================================================================

def test_lobatto_3_and_5_refused_without_model(tmp_path: Path) -> None:
    """The issue's record: the station check catches it with no model."""
    lengths = (3.0, 5.0)
    fem, eids = _two_beams("mix35_nomodel", lengths)
    _write_files(tmp_path, lengths, (LOBATTO[3], LOBATTO[5]))
    with pytest.raises(ValueError, match=UNORDERED):
        _transcode(tmp_path, fem, eids, None)
    assert not (tmp_path / "out.h5").exists()


def test_lobatto_3_and_5_refused_by_name_with_model(tmp_path: Path) -> None:
    lengths = (3.0, 5.0)
    fem, eids = _two_beams("mix35_model", lengths)
    model_h5 = _model_h5(tmp_path, fem, (_lobatto(3), _lobatto(5)))
    _write_files(tmp_path, lengths, (LOBATTO[3], LOBATTO[5]))
    with pytest.raises(
        ValueError,
        match=(rf"different integration-point counts in model= \(3 points: "
               rf"FEM elements \[{eids[0]}\]; 5 points: FEM elements "
               rf"\[{eids[1]}\]\)")):
        _transcode(tmp_path, fem, eids, model_h5)


def test_legendre_mix_the_station_check_misses_is_refused_by_the_model(
        tmp_path: Path) -> None:
    """Legendre 2 on L = 1 beside Legendre 4 on L = 12: every equal block of
    3 increases within [-1, 1], so only the model's counts can refuse it."""
    lengths = (1.0, 12.0)
    gpx = np.concatenate([(LEGENDRE[2] + 1.0) * 0.5 * lengths[0],
                          (LEGENDRE[4] + 1.0) * 0.5 * lengths[1]])
    blocks = [2.0 * gpx[:3] / lengths[0] - 1.0, 2.0 * gpx[3:] / lengths[1] - 1.0]
    for xi in blocks:    # the precondition: the station check alone passes
        assert np.all(np.diff(xi) > 0.0) and np.all(np.abs(xi) <= 1.0)
    fem, eids = _two_beams("mix24_legendre", lengths)
    model_h5 = _model_h5(tmp_path, fem, (_legendre(2), _legendre(4)))
    _write_files(tmp_path, lengths, (LEGENDRE[2], LEGENDRE[4]))
    with pytest.raises(ValueError, match=r"2 points: FEM elements .*4 points"):
        _transcode(tmp_path, fem, eids, model_h5)


def test_file_that_does_not_match_the_model_is_refused(tmp_path: Path) -> None:
    lengths = (3.0, 5.0)
    fem, eids = _two_beams("mismatch", lengths)
    model_h5 = _model_h5(tmp_path, fem, (_lobatto(5), _lobatto(5)))
    _write_files(tmp_path, lengths, (LOBATTO[3], LOBATTO[3]))
    with pytest.raises(ValueError, match="model= gives its elements 5"):
        _transcode(tmp_path, fem, eids, model_h5)


def test_unordered_stations_refused_without_model(tmp_path: Path) -> None:
    """An unbound read of out-of-order stations is refused by their order."""
    lengths = (3.0, 5.0)
    fem, eids = _two_beams("unordered", lengths)
    _write_files(tmp_path, lengths, (LOBATTO[3], LOBATTO[3][::-1]))
    with pytest.raises(ValueError, match=rf"FEM element {eids[1]} .*{UNORDERED}"):
        _transcode(tmp_path, fem, eids, None)


# =====================================================================
# Equal counts: read exactly as before
# =====================================================================

@pytest.mark.parametrize("with_model", [False, True])
@pytest.mark.parametrize("n, rule, xi", [
    (3, _lobatto(3), LOBATTO[3]),
    (6, _hinge_radau, None),
])
def test_equal_counts_read_as_before(tmp_path: Path, with_model: bool, n: int,
                                     rule: Rule, xi: np.ndarray | None) -> None:
    lengths = (3.0, 5.0)
    if xi is None:    # HingeRadau's six stations, as OpenSees places them
        xi = np.array([-1.0, -0.8, -0.3, 0.3, 0.8, 1.0])
    fem, eids = _two_beams(f"uniform_{n}_{with_model}", lengths)
    model_h5 = _model_h5(tmp_path, fem, (rule, rule)) if with_model else None
    _write_files(tmp_path, lengths, (xi, xi))
    target = _transcode(tmp_path, fem, eids, model_h5)
    with Results.from_native(target, model=_open_model_from_h5(target)) as r:
        s = r.stage(r.stages[0].id)
        for k, comp in enumerate(COMPS):
            slab = s.elements.line_stations.get(component=comp)
            assert np.asarray(slab.element_index).tolist() == [
                e for e in eids for _s in range(n)]
            np.testing.assert_allclose(slab.station_natural_coord,
                                       np.tile(xi, 2), atol=1e-12)
            np.testing.assert_array_equal(slab.values, [
                [_value(t, e, s_, k) for e in range(2) for s_ in range(n)]
                for t in range(TIMES.size)])


@pytest.mark.parametrize("with_model", [True, False])
def test_long_hinge_stations_read_when_the_model_proves_the_slice(
        tmp_path: Path, with_model: bool) -> None:
    """Legal out-of-order stations: a model proves the slice and they read;
    unbound, they are refused by their order, never as mixed counts."""
    lengths = (0.75, 0.75)
    fem, eids = _two_beams(f"hinge_long_{with_model}", lengths)

    def rule(ops: apeSees, sec: Any) -> Any:
        return _hinge_radau(ops, sec, lp=0.15)

    model_h5 = _model_h5(tmp_path, fem, (rule, rule)) if with_model else None
    _write_files(tmp_path, lengths, (HINGE_LONG, HINGE_LONG))
    if not with_model:
        with pytest.raises(ValueError, match=rf"{UNORDERED}.*Pass model="):
            _transcode(tmp_path, fem, eids, model_h5)
        return
    target = _transcode(tmp_path, fem, eids, model_h5)
    with Results.from_native(target, model=_open_model_from_h5(target)) as r:
        slab = r.stage(r.stages[0].id).elements.line_stations.get(
            component="bending_moment_z")
        assert np.asarray(slab.element_index).tolist() == [
            e for e in eids for _s in range(6)]
        np.testing.assert_allclose(slab.station_natural_coord,
                                   np.tile(HINGE_LONG, 2), atol=1e-12)
        np.testing.assert_array_equal(slab.values, [
            [_value(t, e, s, 1) for e in range(2) for s in range(6)]
            for t in range(TIMES.size)])


def test_stale_cache_from_the_previous_parser_is_not_served(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``from_recorders`` keys its cache on ``PARSER_VERSION``: a transcode
    the previous parser cached (mis-sliced, no checks) is re-made, so the
    record is read again and refused."""
    lengths = (3.0, 5.0)
    fem, eids = _two_beams("stale_cache", lengths)
    model = OpenSeesModel.from_h5(
        _model_h5(tmp_path, fem, (_lobatto(3), _lobatto(5))))
    out = tmp_path / "out"
    out.mkdir()
    _write_files(out, lengths, (LOBATTO[3], LOBATTO[5]))
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id,
                                records=(_record("beam", eids),))

    def read() -> Results:
        return Results.from_recorders(spec, out, fem=fem, model=model,
                                      cache_root=tmp_path / "cache")

    with monkeypatch.context() as m:    # what the previous parser cached
        m.setattr("apeGmsh.results.schema._versions.PARSER_VERSION",
                  PREVIOUS_PARSER)
        m.setattr("apeGmsh.results.transcoders._recorder.require_line_stations",
                  lambda *a: None)
        with read() as r:
            slab = r.elements.line_stations.get(component="axial_force")
            assert np.asarray(slab.element_index).tolist() == [eids[0]] * 4 + [
                eids[1]] * 4, "precondition: the stale cache holds the mis-slice"
    with pytest.raises(ValueError, match="different integration-point counts"):
        read()


# =====================================================================
# Live: the issue's reproducer, recorded by openseespy
# =====================================================================

@pytest.mark.live
def test_live_lobatto_3_and_5_refused_and_split_matches_ele_response(
        tmp_path: Path) -> None:
    """A two-element cantilever, Lobatto 3 then Lobatto 5, one section.

    The mixed record is refused through ``Results.from_recorders`` (which
    requires ``model=``) and by a transcoder given no model. Split into one
    record per count, each station reads what ``eleResponse(tag, "section",
    k, "force")`` gives at every step.
    """
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    fem, eids = _two_beams("live_mix35", (3.0, 5.0))
    ops = _bridge(fem, (_lobatto(3), _lobatto(5)))
    ops.fix(pg="Base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Tip", forces=(10.0 / 3.0, 0.3 / 13.0, 1.5 / 7.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()
    model_h5 = tmp_path / "model.h5"
    ops.h5(str(model_h5))
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    model = OpenSeesModel.from_h5(model_h5)
    tag = {int(r.fem_eid): int(r.tag) for r in model.elements()}
    n_ip = dict(zip(eids, (3, 5)))

    mixed = _record("mixed", eids)
    split = tuple(_record(f"b{i}", [e]) for i, e in enumerate(eids))
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id,
                                records=(mixed, *split))
    out = tmp_path / "out"
    want: dict[int, list[list[float]]] = {e: [] for e in eids}
    with spec.emit_recorders(str(out), ops=emitter.ops, model=model) as live:
        live.begin_stage("s", kind="static")
        for _ in range(2):
            assert emitter.analyze(steps=1) == 0
            for e in eids:
                want[e].append([v for k in range(1, n_ip[e] + 1)
                                for v in emitter.ops.eleResponse(
                                    tag[e], "section", str(k), "force")])
        live.end_stage()

    def read(records: tuple, m: Any, cache: str) -> Results:
        return Results.from_recorders(
            ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=records),
            out, fem=fem, stage_id="s", model=m, cache_root=tmp_path / cache)

    with pytest.raises(ValueError, match="3 points: FEM elements"):
        read((mixed,), model, "c_model")
    with pytest.raises(ValueError, match=UNORDERED):    # no model: the transcoder alone
        RecorderTranscoder(
            ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(mixed,)),
            output_dir=out, target_path=tmp_path / "none.h5", fem=fem,
            stage_id="s").run()

    r = read(split, model, "c_split")
    for k, comp in enumerate(COMPS):
        slab = r.elements.line_stations.get(component=comp)
        idx = np.asarray(slab.element_index)
        for e in eids:
            got = np.asarray(slab.values)[:, idx == e]
            np.testing.assert_allclose(
                np.asarray(slab.station_natural_coord)[idx == e],
                LOBATTO[n_ip[e]], atol=1e-12)
            np.testing.assert_allclose(
                got, np.asarray(want[e])[:, k::len(COMPS)], rtol=1e-12,
                atol=1e-12 * np.abs(want[e]).max(), err_msg=f"{comp} e{e}")
