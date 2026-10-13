"""R1b round-trip oracle (K2a, #1794): ``tcl() == from_h5(h5()).build('tcl')``.

For every model in :data:`CORPUS`, the deck the bridge writes and the deck
replayed from its own archive must be equal byte for byte:
``ops.tcl(path, flat=True)`` against
``OpenSeesModel.from_h5(ops.h5()).build("tcl")``. ``flat=True`` because a
partitioned archive replays a single-process deck (the documented degrade
that ``test_h5_commands_program.py`` already pins).

**The corpus is the replayable one.** These are the ``apeSees`` builders that
already live in ``tests/opensees/h5``. Each is imported read-only and called
as is. The golden emit corpus beside this file is built on ``FEMStub`` and
archives only the bridge zone, so it cannot be replayed (#1797). The ledger
header records that.

**Every difference is an assertion, not a skip.**
``roundtrip_oracle_ledger.txt`` holds one row per differing model:

- Class A: integral floats only (``1e6`` comes back as ``1000000``).
- Class B: the same lines in another order.
- Class C: a semantic difference (#1798).
- Class S: the model is not replayable, and replay raises the named error
  (#1797).

A/B/C rows pin the sha256 of the exact normalised diff, and the test also
re-derives the class. A waived model that starts to match fails, and so does
one that differs in a new way, which forces the ledger to shrink.
``N_WAIVED_ORACLE`` may only go down, and must reach 0 by 2026-12-11 (K2,
KC4).
"""
from __future__ import annotations

import collections
import difflib
import hashlib
import importlib
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from tests.opensees.golden import builder

_LEDGER = Path(__file__).with_name("roundtrip_oracle_ledger.txt")

#: ``N_WAIVED_ORACLE`` when the ledger opened (K2a, #1794). It may only go down.
_CEILING = 36
#: The K2 deadline: the ledger is empty by then, or KC4 applies.
DEADLINE = "2026-12-11"
_CLASSES = frozenset({"A", "B", "C", "S"})


# ---------------------------------------------------------------------------
# The replayable corpus: builders imported read-only from tests/opensees/h5
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    module: str
    function: str
    kwargs: tuple[tuple[str, Any], ...] = ()

    @property
    def id(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.kwargs)
        return f"{self.module}::{self.function}" + (f"[{args}]" if args else "")

    def build(self) -> apeSees:
        mod = importlib.import_module(f"tests.opensees.h5.{self.module}")
        factory: Callable[..., apeSees] = getattr(mod, self.function)
        return factory(**dict(self.kwargs))


def _m(module: str, function: str, **kwargs: Any) -> Model:
    return Model(module, function, tuple(kwargs.items()))


_B = (False, True)
CORPUS: tuple[Model, ...] = (
    *(_m("test_h5_commands_program", "_frame", rayleigh=r, modal=m)
      for r in _B for m in _B),
    *(_m("test_h5_commands_program", "_staged",
         partitioned=p, rayleigh=r, profile=pr)
      for p in _B for r in _B for pr in _B),
    _m("test_h5_commands_program", "_ledgered_frame"),
    _m("test_h5_apesees_cuts", "_build_minimal_ops"),
    _m("test_h5_builder_scope_replay", "build_mixed_ndf_bridge"),
    _m("test_h5_builder_scope_replay", "_staged_bridge", staged_pg="Rock"),
    _m("test_h5_builder_scope_replay", "_staged_bridge", staged_pg="Liner"),
    _m("test_h5_damping", "_build_with_element_damp"),
    *(_m("test_h5_decls", "_staged", named=n) for n in _B),
    *(_m("test_h5_decls", "_flat_frame", named=n) for n in _B),
    _m("test_h5_decls", "_param_stage_model"),
    *(_m("test_h5_names_sidecar", "_build", named=n) for n in _B),
    *(_m("test_h5_decl_params", f) for f in (
        "_params_frame", "_broad_staged", "_broad_frame",
        "_broad_partitioned")),
    *(_m("test_h5_initial_stress", "_build_frame", with_initial_stress=w)
      for w in _B),
    *(_m("test_h5_param_tags", f) for f in (
        "_update_then_stress_bridge", "_flat_quad_initial_stress_bridge",
        "_zero_some_bridge", "_absorb_and_stress_bridge")),
    _m("test_h5_partitioned_staged_capture", "_partitioned_bridge"),
    _m("test_h5_partitioned_staged_capture", "_flat_bridge"),
    _m("test_h5_partitions_roundtrip", "_build_bridge"),
    _m("test_h5_rc_layered_shell_roundtrip", "_bridge"),
    *(_m("test_h5_regions_round_trip", "_frame", rayleigh_on=ro, named=n)
      for ro in (None, "Cols") for n in _B),
    _m("test_h5_regions_round_trip", "_partitioned_staged_with_fan_out"),
    *(_m("test_h5_solve_stamp", "_frame_bridge", analysis=a) for a in _B),
    _m("test_h5_stages_reader", "_real_two_stage_bridge"),
    _m("test_h5_stages_reader", "_real_kitchen_sink_bridge"),
    _m("test_h5_stages_replay", "_no_param_two_stage_bridge"),
    _m("test_h5_stages_writer", "_build_two_stage_bridge"),
    _m("test_h5_stages_writer", "_build_kitchen_sink_bridge"),
    _m("test_opensees_model_build_targets", "_author_simple_frame_apesees"),
    *(_m("test_reinforce_deck_replay", "_reinforced_ops", bond_by_name=b)
      for b in _B),
)


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Waiver:
    model: str
    klass: str
    pin: str  # sha256[:16] of the normalised diff, or the S error text token
    reason: str


def _read_ledger(text: str | None = None) -> tuple[int, list[Waiver], str]:
    if text is None:
        text = _LEDGER.read_text(encoding="utf-8")
    n: int | None = None
    rows: list[Waiver] = []
    header: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            header.append(line)
            continue
        m = re.fullmatch(r"N_WAIVED_ORACLE\s*=\s*(\d+)", line)
        if m:
            n = int(m.group(1))
            continue
        row, sep, reason = line.partition("#")
        assert sep and reason.strip(), f"ledger row gives no reason: {raw!r}"
        parts = row.split()
        assert len(parts) == 3, f"malformed ledger row: {raw!r}"
        assert parts[1] in _CLASSES, f"unknown class in {raw!r}"
        rows.append(Waiver(parts[0], parts[1], parts[2], reason.strip()))
    assert n is not None, f"{_LEDGER.name} lacks N_WAIVED_ORACLE"
    return n, rows, "\n".join(header)


def _waivers() -> dict[str, Waiver]:
    _n, rows, _h = _read_ledger()
    return {w.model: w for w in rows}


# ---------------------------------------------------------------------------
# Emit, replay, diff
# ---------------------------------------------------------------------------

#: Every number, integral or not, so class A can be re-derived.
_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])")
#: Classes S rows assert: the replay error a bridge-only archive raises.
_S_ERRORS = {"KeyError:nodes": (KeyError, "object 'nodes' doesn't exist")}


def _canon(text: str, out_dir: Path) -> str:
    """The corpus normalisation: LF, scratch dir masked, stamp masked."""
    text = text.replace("\r\n", "\n")
    for form in {str(out_dir), out_dir.as_posix(), str(out_dir.resolve())}:
        text = text.replace(form, builder.OUT_TOKEN)
    return builder.STAMP_LINE.sub(
        lambda m: f"# apeGmsh {builder.VERSION_TOKEN}; backend {m['kind']}"
        + (f"; build {builder.BUILD_TOKEN}" if m["build"] else ""),
        text,
    )


def _emit(model: Model, out_dir: Path) -> tuple[str, Path]:
    """The bridge deck (canonical text) and the archive of the same model."""
    with warnings.catch_warnings():
        # Advisory emit warnings are their own tests' subject, not this one's.
        warnings.simplefilter("ignore")
        ops = model.build()
        deck = out_dir / "bridge.tcl"
        ops.tcl(str(deck), progress=False, flat=True)
        archive = out_dir / "model.h5"
        ops.h5(str(archive))
    return _canon(deck.read_bytes().decode("utf-8"), out_dir), archive


def _replay(archive: Path) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        replay = OpenSeesModel.from_h5(str(archive)).build("tcl")
    assert isinstance(replay, str)
    return _canon(replay, archive.parent)


def _emit_and_replay(model: Model, out_dir: Path) -> tuple[str, str]:
    bridge, archive = _emit(model, out_dir)
    return bridge, _replay(archive)


def _diff(bridge: str, replay: str) -> str:
    return "\n".join(difflib.unified_diff(
        bridge.splitlines(), replay.splitlines(),
        fromfile="bridge", tofile="replay", lineterm="", n=0,
    ))


#: A hunk header. Its line numbers move with anything above the hunk (a
#: gmsh release that meshes one more node shifts every later line), so the
#: pin hashes the hunk structure and the changed lines, never the offsets.
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)


def _pin(diff: str) -> str:
    body = _HUNK.sub("@@", diff)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _classify(bridge: str, replay: str) -> str:
    """A, B or C for two decks that differ (re-derives the ledger's claim)."""
    def norm(t: str) -> list[str]:
        return [_NUMBER.sub(lambda m: repr(float(m.group())), ln)
                for ln in t.splitlines()]
    nb, nr = norm(bridge), norm(replay)
    if nb == nr:
        return "A"
    if collections.Counter(nb) == collections.Counter(nr):
        return "B"
    return "C"


def _check(model: Model, waiver: Waiver | None, out_dir: Path) -> None:
    """The oracle for one model; raises AssertionError on any violation."""
    if waiver is not None and waiver.klass == "S":
        exc_type, text = _S_ERRORS[waiver.pin]
        _bridge, archive = _emit(model, out_dir)  # the bridge must emit
        with pytest.raises(exc_type, match=re.escape(text)):
            OpenSeesModel.from_h5(str(archive))
        return
    bridge, replay = _emit_and_replay(model, out_dir)
    diff = _diff(bridge, replay)
    if waiver is None:
        assert bridge == replay, (
            f"{model.id}: the replayed deck differs from the bridge deck "
            f"(class {_classify(bridge, replay)}, pin {_pin(diff)}); this is "
            f"a replay regression, not a new waiver:\n{diff}"
        )
        return
    assert bridge != replay, (
        f"{model.id} now round-trips byte for byte: delete its ledger row "
        "and lower N_WAIVED_ORACLE"
    )
    assert _pin(diff) == waiver.pin, (
        f"{model.id} differs in a new way (pin {_pin(diff)}, ledger "
        f"{waiver.pin}); a fix that changes the diff updates its row only "
        f"when the row shrinks to nothing:\n{diff}"
    )
    assert _classify(bridge, replay) == waiver.klass, (
        f"{model.id}: the ledger says class {waiver.klass}, the diff is "
        f"class {_classify(bridge, replay)}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", CORPUS, ids=[m.id for m in CORPUS])
def test_replay_matches_the_bridge_deck(model: Model, tmp_path: Path) -> None:
    _check(model, _waivers().get(model.id), tmp_path)


def test_pin_ignores_hunk_offsets() -> None:
    """The same diff body at another line offset has the same pin (#1799)."""
    want = "\n".join(
        ["head"] * 3
        + ["nDMaterial ElasticIsotropic 1 25000000000.0 0.2 2400.0", "tail"])
    got = want.replace("25000000000.0 0.2 2400.0", "25000000000 0.2 2400")
    shifted = "\n".join(["mesh"] * 40) + "\n"
    near, far = _diff(want, got), _diff(shifted + want, shifted + got)
    assert near != far  # the hunk headers differ ...
    assert _pin(near) == _pin(far)  # ... and the pin does not
    assert _pin(near) != _pin(_diff(want, want.replace("head", "hd", 1)))


def test_corpus_ids_are_unique() -> None:
    ids = [m.id for m in CORPUS]
    assert len(ids) == len(set(ids))


def test_ledger_only_shrinks() -> None:
    n, rows, _h = _read_ledger()
    assert n == len(rows), f"N_WAIVED_ORACLE = {n} but {len(rows)} rows"
    assert n <= _CEILING, "N_WAIVED_ORACLE may only go down"
    # Shrink-only: the PR that deletes a row lowers _CEILING with it, so the
    # committed ceiling never leaves room to add a row back.
    assert _CEILING == n, (
        f"_CEILING = {_CEILING} but N_WAIVED_ORACLE = {n}: lower _CEILING "
        "to match in the same PR")
    assert len({w.model for w in rows}) == len(rows), "duplicate ledger rows"


def test_every_row_names_a_corpus_model() -> None:
    corpus = {m.id for m in CORPUS}
    orphans = [w.model for w in _read_ledger()[1] if w.model not in corpus]
    assert not orphans, f"ledger rows for no corpus model: {orphans}"


def test_rows_cite_their_issue() -> None:
    for w in _read_ledger()[1]:
        if w.klass == "C":
            assert re.search(r"#1798 item [123]\b", w.reason), w
        if w.klass == "S":
            assert "#1797" in w.reason and w.pin in _S_ERRORS, w
        if w.klass in "ABC":
            assert re.fullmatch(r"[0-9a-f]{16}", w.pin), w


def test_ledger_header_names_the_deadline_and_the_stub_corpus() -> None:
    header = _read_ledger()[2]
    assert DEADLINE in header and "KC4" in header
    assert "#1797" in header and "bridge-only" in header


# Revert proofs, in-process: the ledger machinery catches a dropped row and a
# stale one (the PR body shows the same against the committed file).

_PASSING = next(m for m in CORPUS if m.id ==
                "test_h5_commands_program::_frame[rayleigh=False,modal=False]")
_WAIVED_A = next(m for m in CORPUS if m.id ==
                 "test_h5_decls::_flat_frame[named=False]")


def test_a_differing_model_with_no_row_fails(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="replay regression"):
        _check(_WAIVED_A, None, tmp_path)


def test_a_stale_row_fails(tmp_path: Path) -> None:
    bogus = Waiver(_PASSING.id, "A", "0" * 16, "bogus")
    with pytest.raises(AssertionError, match="now round-trips byte for byte"):
        _check(_PASSING, bogus, tmp_path)


def test_a_changed_diff_fails(tmp_path: Path) -> None:
    w = _waivers()[_WAIVED_A.id]
    moved = Waiver(w.model, w.klass, "f" * 16, w.reason)
    with pytest.raises(AssertionError, match="differs in a new way"):
        _check(_WAIVED_A, moved, tmp_path)
