"""cp1252 console smoke for the ADR 0100 D0 instrument (refuter lens 2).

Windows stdout is cp1252 whenever piped or redirected.  A single U+2248
in a printed f-string killed the bench at the first cell (rc=1) while
5,978 tests stayed green, because nothing exercised the CLI's print
path; ``--help`` was separately broken by a bare ``%`` in an argparse
help string (argparse %-formats help text), and the module docstring —
which argparse prints as its description — carried more non-cp1252
codepoints.  The contract these tests enforce: the module docstring and
every string the instrument prints stay ASCII-only.

Both tests force the child's stdio to cp1252 — on a UTF-8 console they
would prove nothing (the exact reason the breakage survived review).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tests" / "benchmarks" / "emit_throughput_profile.py"


def _cp1252_env() -> "dict[str, str]":
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    # Plain-python child: set PYTHONPATH explicitly — pytest's
    # ``pythonpath=["src"]`` applies only to THIS process, and relying
    # on it in a child is the editable-install trap's pytest cousin.
    env["PYTHONPATH"] = f"{_ROOT / 'src'}{os.pathsep}{_ROOT}"
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    return env


def test_help_renders_on_cp1252_stdout() -> None:
    """``--help`` formats the module docstring plus every ``help=``
    string through argparse (%-formatting included) and prints the lot;
    rc must be 0 on a cp1252 console."""
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True, env=_cp1252_env(), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    out = proc.stdout.decode("cp1252")
    assert "--rss-interval-ms" in out
    # The %%-escape must render as a literal percent, not TypeError.
    assert "~9% low" in out


_VERDICT_PREFIX = "gate verdict: "
_VERDICT_OK = "gate verdict: WITHIN ERROR BOUND"
_VERDICT_DISCARD = "gate verdict: *** DISCARDED - G0a error bound exceeded"


@pytest.mark.bench
def test_mem_cell_report_prints_on_cp1252_stdout(tmp_path: Path) -> None:
    """One tiny --mem cell end-to-end (~45 s): the whole
    ``report_instrumented`` print path must survive a cp1252 console —
    G2/G3 run exactly this pipeline with stdout redirected.

    This is a PRINT-PATH test, not a gate test.  Whether this tiny cell
    passes the G0a error bound is a measurement outcome of the machine
    it runs on — on the shared GitHub runner it discards — so asserting
    only the pass-branch text conflated "the report printed" with "this
    cell passed the gate" (the nightly Benchmarks job was red on that
    conflation from 2026-08-26).  The instrument now prints a verdict
    line in both branches under one prefix, and the assertions below
    require exactly one verdict, one of the two known spellings, and
    agreement with ``gate_status`` in the JSON record the campaign
    aggregator consumes — so the test still proves the verdict print
    path ran, and can no longer be satisfied by a half-printed report.
    """
    json_path = tmp_path / "smoke_cell.json"
    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPT),
            "--recipe", "box", "--sizes", "6", "--parts", "2",
            "--mem", "--stream", "--staged", "--repeats", "1",
            "--tm-frames", "4", "--json", str(json_path),
        ],
        capture_output=True, env=_cp1252_env(), timeout=600,
    )
    assert proc.returncode == 0, (
        proc.stderr.decode(errors="replace")[-2000:]
    )
    out = proc.stdout.decode("cp1252")
    for marker in (
        "per-term at anchor",
        "R8 at anchor",
        "G0a(conservative)",
        _VERDICT_PREFIX.strip(),
    ):
        assert marker in out, f"missing {marker!r} in report output"

    # One cell (one size x one repeat) => exactly one verdict, and it
    # must be one of the two spellings the instrument can print.
    n_verdicts = out.count(_VERDICT_PREFIX)
    n_ok = out.count(_VERDICT_OK)
    n_discard = out.count(_VERDICT_DISCARD)
    assert n_verdicts == 1, (
        f"expected exactly 1 verdict line for 1 cell, saw {n_verdicts}"
    )
    assert n_ok + n_discard == 1, (
        "the verdict line matched neither known spelling "
        f"(ok={n_ok}, discarded={n_discard}) - the instrument's verdict "
        "wording changed without this contract being updated"
    )

    # The printed branch must agree with the machine-readable record.
    records = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(records) == 1, f"expected 1 json record, got {len(records)}"
    expected = "ok" if n_ok else "discarded_error_bound"
    assert records[0].get("gate_status") == expected, (
        f"printed verdict says {expected!r} but the json record says "
        f"{records[0].get('gate_status')!r}"
    )


@pytest.mark.bench
def test_phases_cell_reports_every_rank_on_cp1252_stdout(
        tmp_path: Path) -> None:
    """One small ``--phases`` cell (#1766): the per-rank phase table and
    the element-phase attribution print on a cp1252 console, and the JSON
    record carries one main block per rank with an element phase, plus
    an attribution whose sites sum to the snapshot's growth.

    The cell is n=30 (~4 s of emit), not n=6, because of the phantom
    guard below (#1896): at n=6 a block's fixed overhead is ~100 B/node
    in EVERY phase, so fix_mass_regions reads 0.36x elements with the
    fix in place and 0.49x without it -- no margin.  At n=30 the fixed
    overhead is gone (1 B/node) and the two cases sit far apart.
    """
    json_path = tmp_path / "phases_cell.json"
    proc = subprocess.run(
        [
            sys.executable, str(_SCRIPT),
            "--recipe", "box", "--sizes", "30", "--parts", "2",
            "--stream", "--phases",
            "--json", str(json_path),
        ],
        capture_output=True, env=_cp1252_env(), timeout=600,
    )
    assert proc.returncode == 0, (
        proc.stderr.decode(errors="replace")[-2000:]
    )
    out = proc.stdout.decode("cp1252")
    for marker in ("per-rank phase peaks", "element-phase attribution",
                   "phase that sets the main-block peak",
                   "pre-block attribution (build() + global preamble",
                   "TRANSIENT sites", "RESIDENT sites"):
        assert marker in out, f"missing {marker!r} in report output"

    (rec,) = json.loads(json_path.read_text(encoding="utf-8"))
    # The pre-block window (#1920): captured, its two splits close, and
    # the snapshot sits near the banked counter peak (1.000 measured at
    # this cell with a 1 ms tick).
    pb = rec["pre_block"]
    assert pb["captured"] and pb["n_global_nodes"] == 30 ** 3, pb
    assert pb["held_b"] + pb["transient_b"] == pb["peak_snap_b"], pb
    assert pb["held_b"] + pb["late_b"] == pb["resident_b"], pb
    assert 0 < pb["resident_b"] <= pb["window_peak_b"], pb
    assert 0.8 < pb["coverage"] <= 1.01, pb
    assert pb["top_transient"] and pb["top_resident"], pb
    main = [b for b in rec["blocks"] if b["kind"] == "main"]
    assert sorted(b["rank"] for b in main) == [0, 1]
    for b in main:
        assert b["n_nodes"] > 0
        assert b["phases"]["elements"] > 0, b
        # The phantom guard (#1896).  The --phases wrapper around the
        # element call must drop its own reference to the materialised
        # rank bucket (``k.clear()``) before ``elem_exit``; production
        # frees it when the call returns.  Kept alive, the bucket is
        # billed to the NEXT phase.  Measured at this cell (box n=30,
        # 2 ranks, after #1901): with the fix fix_mass_regions is
        # 1 B/node against elements 107-108 B/node (0.01x); with
        # ``k.clear()`` removed it is 70 B/node (0.65x).  A 0.25x bound
        # leaves a wide margin on both sides.
        fmr = b["phases"]["fix_mass_regions"]
        assert fmr < 0.25 * b["phases"]["elements"], (
            f"rank {b['rank']}: fix_mass_regions {fmr} B is not much "
            f"smaller than elements {b['phases']['elements']} B -- the "
            "element bucket is being billed to the next phase", b)
        assert "contact_stress_pattern" in b["phases"], b
        assert "close" not in b["phases"], b
    (attr,) = rec["attribution"]
    assert attr["n_nodes"] == next(
        b["n_nodes"] for b in main if b["rank"] == 0)
    assert attr["rank"] == 0 and attr["top"]
    # The traceback groups partition the snapshot diff: their net sum is
    # the traced growth at the snapshot, up to the snapshot's own
    # bookkeeping (a few KiB).
    assert abs(attr["net_b"] - attr["growth_b"]) < 64 * 1024, attr


def test_phase_tracker_attributes_the_main_block_after_a_hoisted_one(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A rank with a hoisted gated block (ADR 0099 S5) opens twice: the
    hoisted block first, then its main block (#1896).  The attribution
    must come from the main block, against the main block's own floor
    snapshot, and be reported with the main block's node count and
    element peak.  Before the fix the floor snapshot was taken in the
    rank's first block only, so the main block's capture had none and
    the report dropped it without a word.  ``build_box`` has no gated
    elements, so only a direct drive of the tracker reaches this path.
    """
    import tracemalloc
    from types import SimpleNamespace

    from tests.benchmarks import emit_throughput_profile as prof

    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start(3)
    try:
        trk = prof.PhaseTracker(attr_rank=0, snap_dir=str(tmp_path))
        # Hoisted gated block: no owned-node loop, elements only.
        trk.open(0)
        trk.event("elem_enter")
        assert not trk._attr_wanted()  # no node count: never attributed
        trk.event("elem_exit")
        trk.close()
        # Main block of the same rank.
        trk.open(0)
        trk.cur["n_nodes"] = 4
        trk.event("nodes_exit")
        trk.event("elem_enter")
        assert trk._attr_wanted()
        held = [bytearray(64 * 1024)]  # the "element phase" growth
        trk.attr_capture(n_rows=3)
        trk.cur["attr_done"] = True
        trk.event("elem_exit")
        del held
        trk.event("mp_enter")
        trk.event("mp_exit")
        trk.close()
        result = {"blocks": trk.blocks, "attr": trk.attr}

        hoisted, main = trk.blocks
        assert (hoisted["kind"], main["kind"]) == ("hoisted", "main")
        assert set(hoisted["phases"]) == {"hoisted_nodes", "elements",
                                          "block_tail"}
        assert set(main["phases"]) == {"nodes", "elements",
                                       "fix_mass_regions", "constraints",
                                       "contact_stress_pattern"}
        (a,) = trk.attr
        assert a["block"] == main["idx"]
        assert a["floor_snap"] is not None
        assert a["floor_snap"] == main["floor_snap"]
        assert a["growth"] >= 64 * 1024

        args = SimpleNamespace(phase_top=5, attr_rank=0, parts=1,
                               recipe="box", staged=False, stream=False)
        rec = prof.report_phases(args, 1, 1, result)
    finally:
        if started:
            tracemalloc.stop()
    (ra,) = rec["attribution"]
    assert ra["n_nodes"] == 4
    assert ra["elem_peak_b"] == main["phases"]["elements"]
    assert "NONE captured" not in capsys.readouterr().out


def test_phase_tracker_refuses_a_capture_without_a_floor(
        tmp_path: Path) -> None:
    """An attribution capture in a block with no floor snapshot (a rank
    other than the attributed one) raises instead of diffing against
    some other block's floor."""
    import tracemalloc

    from tests.benchmarks import emit_throughput_profile as prof

    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start(1)
    try:
        trk = prof.PhaseTracker(attr_rank=0, snap_dir=str(tmp_path))
        trk.open(1)
        with pytest.raises(RuntimeError, match="baseline"):
            trk.attr_capture(n_rows=1)
    finally:
        if started:
            tracemalloc.stop()


_MIB = 1024 * 1024


def test_pre_block_tracker_splits_transient_held_and_late(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Drive ``PreBlockTracker`` by hand (no thread) through a known
    pre-block history (#1920), then check ``report_pre_block`` against
    it in bytes:

    * 1 MiB allocated before the peak snapshot and held to the open:
      HELD (resident at open AND in the peak snapshot);
    * 4 MiB in the peak snapshot, freed before the open: TRANSIENT;
    * an 8 MiB rise-and-fall AFTER the snapshot, inside ``build``:
      no snapshot sees it, but ``mark`` must bank it, so the window
      peak carries it and the coverage drops below 1 -- the report
      must not pass the snapshot off as the peak;
    * 2 MiB allocated in the preamble and held: LATE (resident, not in
      the peak snapshot).

    Each buffer is allocated on its own line, so each is its own site.
    """
    import tracemalloc
    from types import SimpleNamespace

    from tests.benchmarks import emit_throughput_profile as prof

    tol = 256 * 1024
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start(3)
    try:
        pre = prof.PreBlockTracker(str(tmp_path), poll_s=1.0, grow=1.05)
        base = tracemalloc.get_traced_memory()[0]
        held = bytearray(1 * _MIB)
        transient = bytearray(4 * _MIB)
        pre.tick()  # the peak snapshot: held + transient
        del transient
        spike = bytearray(8 * _MIB)
        del spike
        pre.mark("preamble")
        late = bytearray(2 * _MIB)
        pre.finish()
        n_gn = 1024
        args = SimpleNamespace(phase_top=5)
        rec = prof.report_pre_block(args, pre.result(), n_gn)
        del held, late
    finally:
        if started:
            tracemalloc.stop()

    assert rec["captured"] and rec["snaps"] == 1
    assert rec["peak_snap_sub"] == "build"
    assert abs(rec["transient_b"] - 4 * _MIB) < tol, rec
    assert abs(rec["held_b"] - (1 * _MIB + base)) < tol, rec
    assert abs(rec["late_b"] - 2 * _MIB) < tol, rec
    assert rec["held_b"] + rec["transient_b"] == rec["peak_snap_b"]
    assert rec["held_b"] + rec["late_b"] == rec["resident_b"]
    # mark() banked the unsnapshotted 8 MiB spike into ``build``.
    assert rec["peak_by_sub_b"]["build"] >= base + 9 * _MIB, rec
    assert rec["window_peak_b"] == rec["peak_by_sub_b"]["build"]
    assert rec["coverage"] < 0.6, rec
    top = rec["top_transient"][0]
    assert abs(top["bytes"] - 4 * _MIB) < tol, top
    assert top["b_per_gnode"] == top["bytes"] / n_gn
    assert top["at_open_b"] == 0, top
    late_site, held_site = rec["top_resident"][:2]
    assert abs(late_site["bytes"] - 2 * _MIB) < tol, late_site
    assert late_site["at_peak_b"] == 0, late_site
    assert abs(held_site["bytes"] - 1 * _MIB) < tol, held_site
    assert held_site["at_peak_b"] == held_site["bytes"], held_site
    assert "pre-block attribution (build() + global preamble" in (
        capsys.readouterr().out)


def test_pre_block_report_says_when_nothing_was_captured(
        capsys: pytest.CaptureFixture[str]) -> None:
    """No ``partition_open`` reached: the report says NOT CAPTURED
    instead of printing an empty table."""
    from types import SimpleNamespace

    from tests.benchmarks import emit_throughput_profile as prof

    pre = prof.PreBlockTracker(".", poll_s=1.0, grow=1.05)
    rec = prof.report_pre_block(SimpleNamespace(phase_top=5),
                                pre.result(), 10)
    assert rec == {"captured": False}
    assert "NOT CAPTURED" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["flat", "partitioned", "staged"])
def test_arch_orientation_fan_out_is_planned_once(mode: str) -> None:
    """Profile the ``arch_with_orientation_fan_out`` golden fixture's emit.

    K1-3d S3a moved the orientation fan-out's allocation loop out of the
    emit and into the build's tag plan (ADR 0114 D4, amended). The move
    must not double its cost: across two emits of one built model, the
    profile shows ``plan_transform_specs`` once (the plan is memoised per
    mode) and one ``vecxz`` computation per arch element, while
    ``emit_transform_specs`` still writes the lines on every emit.
    """
    import cProfile
    import pstats

    from apeGmsh.opensees.emitter.tcl import TclEmitter

    from tests.opensees.golden import builder as golden

    bm = golden.build_model(
        "arch_with_orientation_fan_out", mode, "tcl").build()
    n_arch = len(_oriented_element_ids(bm))
    assert n_arch >= 2

    prof = cProfile.Profile()
    prof.enable()
    for _ in range(2):
        bm.emit(TclEmitter())
    prof.disable()

    stats = pstats.Stats(prof).stats  # type: ignore[attr-defined]
    calls = {
        name: stat[1]     # primitive call count
        for (_file, _line, name), stat in stats.items()
        if name in ("plan_transform_specs", "emit_transform_specs",
                    "compute_vecxz_for_element")
    }
    assert calls.get("plan_transform_specs") == 1, calls
    assert calls.get("compute_vecxz_for_element") == n_arch, calls
    assert calls.get("emit_transform_specs") == 2, calls


def _oriented_element_ids(bm: "object") -> "list[int]":
    """The FEM ids of every element whose transform fans out."""
    from apeGmsh.opensees._internal.build.transforms import (
        _element_transf,
        is_orientation_transform,
    )
    from apeGmsh.opensees._internal.build.pg_expand import (
        expand_pg_to_elements,
    )

    out: "list[int]" = []
    for prim in bm.primitives:  # type: ignore[attr-defined]
        t = _element_transf(prim)
        if t is not None and is_orientation_transform(t):
            out += [int(eid) for eid, _ in
                    expand_pg_to_elements(bm.fem, prim.pg)]  # type: ignore[attr-defined]
    return out
