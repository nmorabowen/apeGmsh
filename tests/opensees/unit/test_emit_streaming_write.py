"""Streaming write-out (ADR 0065 Tier 1) + streaming sink (Tier 2).

Tier 1: ``TclEmitter.write_to`` / ``PyEmitter.write_to`` stream the
internal line buffer straight to a handle, avoiding both the
``list(self._lines)`` copy of :meth:`lines` and the deck-sized joined
string of ``"\\n".join(lines()) + "\\n"``. These tests lock two
properties:

* **byte-identity** — the streamed bytes equal the old join output; and
* **bounded write-time allocation** — the Python allocation incurred *by
  the write itself* is a small fraction of the join path's, independent of
  deck size (the actual memory win behind the ADR).

Tier 2 (ADR 0065 Decision §1–§5 / plan_emit_memory_columnar.md A1–A3):
``ops.tcl(path, stream=True)`` writes the deck through a live file sink
instead of accumulating the line buffer; ``per_rank=True`` live-routes
fragment files via ``partition_open``/``partition_close``. The
non-negotiable gates:

* **stream-vs-list byte-identity** — monolithic stream output equals
  list-mode ``write_to``; per-rank stream driver + every fragment equals
  ``_write_per_rank_tcl``'s files (flat / partitioned / staged+
  partitioned, incl. per-rank ``<seq>`` numbering);
* **O(1) emit peak** — stream-mode peak does not grow with element
  count (the line-buffer term vanishes);
* **fail-loud guards** — ``split=True``, ``ops.py(stream=True)``, and
  in-memory introspection (``lines()`` etc.) in stream mode; and
* **atomicity** — a mid-emit exception leaves no final deck and no
  ``.tmp`` litter.
"""
from __future__ import annotations

import tracemalloc
from pathlib import Path
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
    make_two_column_frame_partitioned,
)
from tests.opensees.integration.test_emit_partitioned_staged import (
    _make_4quad_2pg_2part_fem,
    _setup_partitioned_staged_ops,
)


class _NullSink:
    """A ``write``-only sink that discards output.

    Isolates the *Python-side* allocation of marshalling the deck to a
    handle from any OS/file-buffer cost, so tracemalloc sees only what the
    two strategies allocate.
    """

    def write(self, _s: str) -> None:  # noqa: D401 - trivial
        pass


def _fill_tcl(n: int) -> TclEmitter:
    em = TclEmitter()
    em.model(ndm=3, ndf=3)
    for i in range(1, n + 1):
        em.node(i, float(i), float(2 * i), float(3 * i))
    return em


def _fill_py(n: int) -> PyEmitter:
    em = PyEmitter()
    em.model(ndm=3, ndf=3)
    for i in range(1, n + 1):
        em.node(i, float(i), float(2 * i), float(3 * i))
    return em


@pytest.mark.parametrize("fill", [_fill_tcl, _fill_py])
def test_write_to_is_byte_identical_to_join(fill, tmp_path) -> None:
    em = fill(2_000)
    expected = "\n".join(em.lines()) + "\n"

    path = tmp_path / "deck.txt"
    with open(path, "w", encoding="utf-8") as f:
        em.write_to(f)

    assert path.read_text(encoding="utf-8") == expected


@pytest.mark.parametrize("fill", [_fill_tcl, _fill_py])
def test_write_to_allocates_far_less_than_join(fill) -> None:
    # A deck large enough that the join path's transient allocation
    # (list copy + one deck-sized string) is unambiguous against noise.
    em = fill(50_000)
    sink = _NullSink()

    tracemalloc.start()
    tracemalloc.reset_peak()
    sink.write("\n".join(em.lines()) + "\n")
    _, join_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    em.write_to(sink)
    _, stream_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The join path allocates a full list copy plus the joined deck string
    # (hundreds of KB at 50k nodes); streaming allocates ~nothing. Demand
    # at least a 10x reduction to stay robust across interpreters.
    assert stream_peak * 10 < join_peak, (
        f"streaming peak {stream_peak} not << join peak {join_peak}"
    )


# ===========================================================================
# ADR 0065 Tier 2 — dual-mode sink + live per-rank routing + atomic writes
# (plan_emit_memory_columnar.md A1–A3)
# ===========================================================================


def _make_flat_ops() -> apeSees:
    """Unpartitioned 2-column frame apeSees model."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    return ops


def _make_partitioned_ops() -> apeSees:
    """2-partition 2-column frame apeSees model."""
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    return ops


def _make_staged_partitioned_ops() -> apeSees:
    """Staged + partitioned fixture (SSI-2.C): stage-activated cimbra +
    initial_stress — multiple per-rank blocks per rank, so the stream
    route's per-rank ``<seq>`` numbering is actually exercised."""
    return _setup_partitioned_staged_ops(_make_4quad_2pg_2part_fem())


def _assert_no_temps_or_finals(root: Path) -> None:
    leftovers = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
    )
    assert not leftovers, f"expected no files, found {leftovers}"


# -- stream-vs-list byte-identity -------------------------------------------


@pytest.mark.parametrize("factory", [
    _make_flat_ops, _make_partitioned_ops, _make_staged_partitioned_ops,
], ids=["flat", "partitioned", "staged_partitioned"])
def test_stream_monolithic_byte_identical_to_list_mode(
    factory, tmp_path: Path,
) -> None:
    """``ops.tcl(stream=True)`` == list-mode ``write_to``, byte for
    byte, and leaves no ``.tmp`` behind."""
    list_path = tmp_path / "list.tcl"
    stream_path = tmp_path / "stream.tcl"
    factory().tcl(str(list_path))
    factory().tcl(str(stream_path), stream=True)

    assert stream_path.read_text(encoding="utf-8") == list_path.read_text(
        encoding="utf-8",
    )
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("factory", [
    _make_partitioned_ops, _make_staged_partitioned_ops,
], ids=["partitioned", "staged_partitioned"])
def test_stream_per_rank_byte_identical_to_post_hoc_writer(
    factory, tmp_path: Path,
) -> None:
    """Live-routed fragments + driver (stream mode) reproduce
    ``_write_per_rank_tcl``'s files byte-identically — same fragment
    names (per-rank ``<seq>`` numbering included), same contents, same
    driver."""
    list_dir = tmp_path / "list"
    stream_dir = tmp_path / "stream"
    list_dir.mkdir()
    stream_dir.mkdir()
    factory().tcl(str(list_dir / "main.tcl"), per_rank=True)
    factory().tcl(
        str(stream_dir / "main.tcl"), per_rank=True, stream=True,
    )

    assert (stream_dir / "main.tcl").read_text(encoding="utf-8") == (
        (list_dir / "main.tcl").read_text(encoding="utf-8")
    )
    list_frags = sorted(
        p.name for p in (list_dir / "ranks").glob("rank*.tcl")
    )
    stream_frags = sorted(
        p.name for p in (stream_dir / "ranks").glob("rank*.tcl")
    )
    assert list_frags, "oracle wrote no fragments — fixture degraded"
    assert stream_frags == list_frags
    for name in list_frags:
        assert (
            (stream_dir / "ranks" / name).read_text(encoding="utf-8")
            == (list_dir / "ranks" / name).read_text(encoding="utf-8")
        ), f"fragment {name} differs between stream and list mode"
    assert not list(tmp_path.rglob("*.tmp"))


def test_stream_per_rank_staged_seq_numbering(tmp_path: Path) -> None:
    """The staged fixture must produce a second block for a rank
    (``rank1_1.tcl``) — otherwise the seq-numbering leg of the parity
    test silently degrades to the base case."""
    _make_staged_partitioned_ops().tcl(
        str(tmp_path / "main.tcl"), per_rank=True, stream=True,
    )
    names = sorted(p.name for p in (tmp_path / "ranks").glob("rank*.tcl"))
    assert any(n.startswith("rank1_1") for n in names), (
        f"expected rank 1 to have a second (stage) fragment; got {names}"
    )


# -- O(1) emit peak ----------------------------------------------------------


def _tcl_stream_fill_peak(n: int, path: Path) -> int:
    """Peak traced allocation while streaming an n-node deck."""
    em = TclEmitter()
    em.stream_to(str(path))
    tracemalloc.start()
    tracemalloc.reset_peak()
    em.model(ndm=3, ndf=3)
    for i in range(1, n + 1):
        em.node(i, float(i), float(2 * i), float(3 * i))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    em.stream_finish()
    return peak


def test_stream_emit_peak_is_o1_in_element_count(tmp_path: Path) -> None:
    """Stream-mode emit peak is O(1) in deck size: 8x the lines must
    not meaningfully move the peak (the line-buffer term vanished),
    while list mode at the large size retains the whole buffer."""
    small_peak = _tcl_stream_fill_peak(5_000, tmp_path / "small.tcl")
    big_peak = _tcl_stream_fill_peak(40_000, tmp_path / "big.tcl")

    # O(1): allow generous constant slack (interpreter noise, file
    # buffer), but no size-proportional growth — 8x the lines would
    # cost ~8x in list mode.
    assert big_peak < small_peak * 2 + 64_000, (
        f"stream peak grew with deck size: {small_peak} -> {big_peak}"
    )

    # And the list-mode buffer at the same large size dwarfs it.
    em = _fill_tcl(40_000)
    list_resident = sum(len(ln) for ln in em.lines())
    assert big_peak * 10 < list_resident, (
        f"stream peak {big_peak} not << list-mode text {list_resident}"
    )


# -- guards ------------------------------------------------------------------


def test_stream_and_split_mutually_exclusive(tmp_path: Path) -> None:
    """split=True + stream=True fails loud before building (v1)."""
    ops = _make_flat_ops()
    with pytest.raises(ValueError, match="stream=True and split=True"):
        ops.tcl(str(tmp_path / "main.tcl"), split=True, stream=True)
    assert not (tmp_path / "main.tcl").exists()


def test_py_stream_fails_loud(tmp_path: Path) -> None:
    """ops.py(stream=True) is out of scope v1 — the HPC path is Tcl."""
    ops = _make_flat_ops()
    with pytest.raises(ValueError, match="stream=True is not supported"):
        ops.py(str(tmp_path / "main.py"), stream=True)
    assert not (tmp_path / "main.py").exists()


def test_stream_per_rank_requires_partitioned_model(
    tmp_path: Path,
) -> None:
    """Unpartitioned model + per_rank=True + stream=True fails loud
    like the list route, and cleans up its temps."""
    ops = _make_flat_ops()
    with pytest.raises(ValueError, match="per_rank=True requires"):
        ops.tcl(
            str(tmp_path / "main.tcl"), per_rank=True, stream=True,
        )
    _assert_no_temps_or_finals(tmp_path)


def test_stream_mode_introspection_fails_loud(tmp_path: Path) -> None:
    """lines()/line_buffer()/write_to()/partition_spans()/preamble()
    fail loud in stream mode — the deck is on disk, not in memory."""
    em = TclEmitter()
    em.stream_to(str(tmp_path / "deck.tcl"))
    em.node(1, 0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="stream mode"):
        em.lines()
    with pytest.raises(RuntimeError, match="stream mode"):
        em.line_buffer()
    with pytest.raises(RuntimeError, match="stream mode"):
        em.partition_spans()
    with pytest.raises(RuntimeError, match="stream mode"):
        em.write_to(_NullSink())
    with pytest.raises(RuntimeError, match="stream mode"):
        em.preamble("too late")
    em.stream_abort()
    _assert_no_temps_or_finals(tmp_path)


def test_preamble_before_stream_attach_flushes_first(
    tmp_path: Path,
) -> None:
    """Header lines present at stream-attach time (banner + preamble)
    flush to the sink before any body line (ADR 0065 Decision §2)."""
    em = TclEmitter()
    em.preamble("emitted before streaming")
    path = tmp_path / "deck.tcl"
    em.stream_to(str(path))
    em.model(ndm=3, ndf=3)
    em.stream_finish()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# emitted before streaming"
    assert lines[1].startswith("# auto-generated by apeGmsh.opensees")
    assert lines[2].startswith("model BasicBuilder")


# -- atomicity ---------------------------------------------------------------


@pytest.mark.parametrize("per_rank", [False, True], ids=["mono", "per_rank"])
def test_stream_mid_emit_exception_leaves_no_deck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, per_rank: bool,
) -> None:
    """A mid-emit exception leaves neither a final deck nor ``.tmp``
    litter (ADR 0065 Decision §4) — the error propagates unchanged."""
    def _boom(
        self: TclEmitter, ele_type: str, tag: int, *args: object,
    ) -> None:
        raise RuntimeError("mid-emit boom")

    monkeypatch.setattr(TclEmitter, "element", _boom)
    ops = _make_partitioned_ops()
    with pytest.raises(RuntimeError, match="mid-emit boom"):
        ops.tcl(
            str(tmp_path / "main.tcl"), per_rank=per_rank, stream=True,
        )
    _assert_no_temps_or_finals(tmp_path)
