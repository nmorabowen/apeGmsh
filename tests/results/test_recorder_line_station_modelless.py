"""A direct ``RecorderTranscoder`` with no model and no pairing refuses a
``line_stations`` record by name (#2331, found in the #2330 review).

A ``.out`` section-force record and its ``integrationPoints`` pair carry one
column block per element and no per-element point count. Without a model
archive (``model_h5_src=``) or a live pairing (``pairing=``), nothing names a
count, and the transcoder used to divide the columns evenly by element. Two
``dispBeamColumn2d`` of lengths 1 and 13 with Legendre 2 and Legendre 4 give
6 columns, split 3 + 3: both blocks increase within [-1, 1], so the station
order check passes, and element 1 read element 2's first station as its own
third. #2330 closed the same record on ``emit_default`` over a FEMData
archive; this closes the direct call.

The refusal is unconditional: a uniform record is refused too, because no
input can prove that every element shares one rule. The oracle for the
mis-slice is the hand-written files themselves: element ``e``'s station
``s`` carries the value ``100 e + 10 s + k``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.results.readers._out_station_slicing import line_station_count
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.transcoders import RecorderTranscoder

COMPS = ("axial_force", "bending_moment_z")
LEGENDRE = {n: np.polynomial.legendre.leggauss(n)[0] for n in (2, 3, 4)}
TIMES = np.array([1.0, 2.0])
#: The issue's reproducer: lengths 1 and 13, Legendre 2 then Legendre 4.
LENGTHS = (1.0, 13.0)
NAMED = r"pass model_h5_src= .* or pairing="


def _two_beams(name: str) -> tuple[Any, list[int]]:
    """Two collinear one-element lines along ``+x`` of ``LENGTHS``."""
    with apeGmsh(model_name=name, verbose=False) as g:
        geo = g.model.geometry
        a = geo.add_point(0.0, 0.0, 0.0)
        b = geo.add_point(LENGTHS[0], 0.0, 0.0)
        c = geo.add_point(LENGTHS[0] + LENGTHS[1], 0.0, 0.0)
        l1, l2 = geo.add_line(a, b), geo.add_line(b, c)
        g.model.sync()
        g.physical.add(1, [l1], name="B1")
        g.physical.add(1, [l2], name="B2")
        g.mesh.structured.set_transfinite_curve(l1, 2)
        g.mesh.structured.set_transfinite_curve(l2, 2)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    eids = [int(fem.elements.select(pg=pg).ids[0]) for pg in ("B1", "B2")]
    return fem, eids


def _record(eids: list[int]) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="line_stations", name="beam", components=COMPS, dt=None,
        n_steps=None, element_ids=np.asarray(eids, dtype=np.int64),
        element_class_name="DispBeamColumn2d")


def _write_files(tmp: Path, xis: tuple[np.ndarray, ...]) -> None:
    """The files OpenSees writes for one record over elements whose natural
    stations are ``xis``: element-major blocks, as many as each one has."""
    gpx = np.concatenate([(x + 1.0) * 0.5 * L for x, L in zip(xis, LENGTHS)])
    sec = [[t * 1000.0 + e * 100.0 + s * 10.0 + k
            for e, x in enumerate(xis) for s in range(x.size)
            for k in range(len(COMPS))] for t in range(TIMES.size)]
    np.savetxt(tmp / "beam_line_stations_gpx.out",
               np.column_stack([TIMES, np.tile(gpx, (TIMES.size, 1))]))
    np.savetxt(tmp / "beam_line_stations.out",
               np.column_stack([TIMES, np.asarray(sec)]))


def _transcode(tmp: Path, fem: Any, eids: list[int]) -> Path:
    target = tmp / "out.h5"
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id,
                                records=(_record(eids),))
    RecorderTranscoder(spec, output_dir=tmp, target_path=target, fem=fem,
                       stage_name="s", stage_kind="static").run()
    return target


def test_the_2330_reproducer_passes_the_order_check() -> None:
    """The precondition: the equal 3 + 3 slice of the mixed record increases
    within [-1, 1] in both blocks, so only a count can refuse it."""
    gpx = np.concatenate([(LEGENDRE[2] + 1.0) * 0.5 * LENGTHS[0],
                          (LEGENDRE[4] + 1.0) * 0.5 * LENGTHS[1]])
    for block, L in ((gpx[:3], LENGTHS[0]), (gpx[3:], LENGTHS[1])):
        xi = 2.0 * block / L - 1.0
        assert np.all(np.diff(xi) > 0.0) and np.all(np.abs(xi) <= 1.0)


def test_mixed_counts_refused_by_name_without_model_or_pairing(
        tmp_path: Path) -> None:
    fem, eids = _two_beams("modelless_mix24")
    _write_files(tmp_path, (LEGENDRE[2], LEGENDRE[4]))
    with pytest.raises(ValueError, match=NAMED) as exc:
        _transcode(tmp_path, fem, eids)
    assert "'beam'" in str(exc.value)
    assert "DispBeamColumn2d" in str(exc.value)
    assert not (tmp_path / "out.h5").exists()


def test_uniform_counts_refused_too(tmp_path: Path) -> None:
    """Equal counts read correctly, but nothing without a model proves the
    counts equal, so the record is refused as the mixed one is."""
    fem, eids = _two_beams("modelless_uniform3")
    _write_files(tmp_path, (LEGENDRE[3], LEGENDRE[3]))
    with pytest.raises(ValueError, match=NAMED):
        _transcode(tmp_path, fem, eids)
    assert not (tmp_path / "out.h5").exists()


def test_refused_before_any_file_is_read(tmp_path: Path) -> None:
    """The refusal names the missing input, not a missing ``.out`` file."""
    fem, eids = _two_beams("modelless_nofiles")
    with pytest.raises(ValueError, match=NAMED):
        _transcode(tmp_path, fem, eids)


def test_line_station_count_refuses_no_model_by_type() -> None:
    """The shared slicer has no model-less branch left to divide by."""
    rec = _record([1, 2])
    with pytest.raises(TypeError, match="NoneType carries neither"):
        line_station_count(rec, 6, "beam_line_stations_gpx.out", None, "sid")
