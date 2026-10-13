"""K2b (#1795): ``/opensees/constraints/embeddedRebar`` (opensees 2.29.0).

The archive stores every ``g.reinforce`` ``LadrunoEmbeddedRebar`` tie the
build wrote: its planned tag and its arguments, the bond tag already
resolved. A replay of a 2.29 archive writes those rows after the fixes and
masses (the bridge's slot) and mints no tag; one of an older archive keeps
re-deriving the ties from the neutral zone, behind the version gate in
``compose._replay_into`` (the ``pre-2-29-reinforce-tie-replay`` waiver).

The oracle is the bridge's own deck. ``_reinforced_fixed_ops`` round-trips
byte for byte with and without a bond material. With ``rebar_cells`` the
bar's ``CorotTruss`` cells are numbered after the ties, so only the stored
tags reproduce the bridge's tie lines: the neutral-zone fallback numbers
the ties past the cells instead.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.build import constraints as constraints_mod
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5 import H5Emitter
from tests.fixtures.schema import OPENSEES_CURRENT, OPENSEES_EMBEDDED_REBAR_FROM
from tests.opensees.h5.test_reinforce_deck_replay import _reinforced_fixed_ops

_STORE = "opensees/constraints/embeddedRebar"
_TIE = "element LadrunoEmbeddedRebar "
_CELL = "element CorotTruss "


def _pre_embedded_rebar_version() -> str:
    """The minor just below ``OPENSEES_EMBEDDED_REBAR_FROM``."""
    major, minor, _patch = (int(p) for p in OPENSEES_EMBEDDED_REBAR_FROM.split("."))
    return f"{major}.{minor - 1}.0"


def downgrade_to_pre_embedded_rebar(path: Path) -> None:
    """Make ``path`` the archive a pre-2.29 writer would have left: no
    ``embeddedRebar`` store, its program runs linked to no store, and the
    opensees stamp one minor below ``EMBEDDED_REBAR_FROM``."""
    with h5py.File(path, "r+") as f:
        assert _STORE in f, "the archive stores no reinforce tie"
        del f[_STORE]
        if not len(f["opensees/constraints"]):
            del f["opensees/constraints"]
        prog = f["opensees/program"]
        stores = [str(s.decode() if isinstance(s, bytes) else s)
                  for s in prog.attrs["stores"]]
        st = stores.index("{scope}/constraints/embeddedRebar")
        rows = prog[()]
        hit = rows["store"] == st
        rows["store"][hit] = -1
        rows["row"][hit] = -1
        rows["store"][rows["store"] > st] -= 1
        stores.pop(st)
        prog[...] = rows
        prog.attrs.create(
            "stores", np.array(stores, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"))
        f["meta"].attrs["opensees_schema_version"] = _pre_embedded_rebar_version()


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., tuple[str, Path]]:
    """``(bridge deck, archive)`` per ``(bond_by_name, rebar_cells)``, built
    once per module; tests copy the archive before changing it."""
    cache: dict[tuple[bool, bool], tuple[str, Path]] = {}

    def get(*, bond: bool = False, cells: bool = False) -> tuple[str, Path]:
        key = (bond, cells)
        if key not in cache:
            d = tmp_path_factory.mktemp(f"k2b_{int(bond)}{int(cells)}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ops = _reinforced_fixed_ops(bond_by_name=bond, rebar_cells=cells)
                ops.tcl(str(d / "bridge.tcl"), progress=False, flat=True)
                ops.h5(str(d / "model.h5"))
            deck = (d / "bridge.tcl").read_text(encoding="utf-8")
            cache[key] = (_strip_stamp(deck), d / "model.h5")
        return cache[key]

    return get


def _strip_stamp(deck: str) -> str:
    """Drop the provenance header line, the one line a replay writes anew."""
    return "\n".join(
        ln for ln in deck.replace("\r\n", "\n").splitlines()
        if not ln.startswith("# apeGmsh "))


def _copy(src: Path, tmp_path: Path, name: str = "model.h5") -> Path:
    dst = tmp_path / name
    shutil.copyfile(src, dst)
    return dst


def _replay(path: Path) -> str:
    deck = OpenSeesModel.from_h5(str(path)).build("tcl")
    assert isinstance(deck, str)
    return _strip_stamp(deck)


def _rewrite(src: Path, tmp_path: Path, name: str = "rewritten.h5") -> Path:
    out = tmp_path / name
    with warnings.catch_warnings():
        # The rewrite drops the global equalDOF-family stores it never
        # replays (a deferred-feature warning); not this module's subject.
        warnings.simplefilter("ignore")
        OpenSeesModel.from_h5(str(src)).build("h5", out=str(out))
    return out


def _lines(deck: str, prefix: str) -> list[str]:
    return [ln for ln in deck.splitlines() if ln.startswith(prefix)]


def _tags(lines: list[str]) -> list[int]:
    return [int(ln.split()[2]) for ln in lines]


def _stored(path: Path) -> tuple[h5_reader.EmbeddedRebarRecordRO, ...]:
    with h5_reader.open(str(path)) as model:
        return model.embedded_rebars()


@pytest.fixture
def planted_fallback(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The neutral-zone tie fallback raises; returns one entry per call."""
    calls: list[int] = []

    def fail(*_a: Any, **_k: Any) -> None:
        calls.append(1)
        raise AssertionError("planted fallback: replay_reinforce_ties re-derived")

    monkeypatch.setattr(constraints_mod, "replay_reinforce_ties", fail)
    return calls


# -- the store ----------------------------------------------------------------


@pytest.mark.parametrize("bond", [False, True], ids=["perfect", "bond"])
def test_archive_stores_every_tie_the_build_wrote(built, bond) -> None:
    deck, archive = built(bond=bond)
    rows = _stored(archive)
    ties = _lines(deck, _TIE)
    assert len(ties) >= 2
    assert [r.ele_tag for r in rows] == _tags(ties)
    # The stored arguments are the deck's, token for token, with their types.
    for row, line in zip(rows, ties):
        assert line.split()[3:] == [str(a) for a in row.args]
        assert any(isinstance(a, str) for a in row.args)
        assert any(isinstance(a, float) for a in row.args)
        assert row.name == "cage1.L1"
    with h5py.File(archive, "r") as f:
        assert f["meta"].attrs["opensees_schema_version"] == OPENSEES_CURRENT
        if bond:
            # The bond arrives resolved: the tag, not the name.
            assert "bond1" not in [str(a) for r in rows for a in r.args]


def test_no_tie_writes_no_store(tmp_path: Path) -> None:
    from tests.opensees.h5.test_h5_decls import _flat_frame

    path = tmp_path / "frame.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _flat_frame(named=False).h5(str(path))
    with h5py.File(path, "r") as f:
        assert _STORE not in f
    assert _stored(path) == ()


def test_a_bad_argument_refuses_at_emit() -> None:
    e = H5Emitter()
    with pytest.raises(TypeError, match="embedded_rebar"):
        e.embedded_rebar(1, 2, True)
    with pytest.raises(TypeError, match="embedded_rebar"):
        e.embedded_rebar(1, 2, np.int64(3))
    with pytest.raises(ValueError, match="embedded_rebar"):
        e.embedded_rebar(1, 2**53)
    assert e._embedded_rebars == []


def test_a_tie_inside_a_stage_refuses() -> None:
    e = H5Emitter()
    e.stage_open("s1")
    with pytest.raises(RuntimeError, match="global-only"):
        e.embedded_rebar(1, 2, "-perfect", 1.5)
    assert e._embedded_rebars == []


def test_a_partition_replica_is_stored_once() -> None:
    e = H5Emitter()
    for rank in (0, 1):
        e.partition_open(rank)
        e.mp_constraint_comment("cage1")
        e.embedded_rebar(7, 3, "-perfect", 1.5)
        e.partition_close()
    assert [(r.ele_tag, r.args, r.name) for r in e._embedded_rebars] == [
        (7, (3, "-perfect", 1.5), "cage1")]


# -- replay of a 2.29 archive -------------------------------------------------


@pytest.mark.parametrize("bond", [False, True], ids=["perfect", "bond"])
def test_new_archive_replays_byte_for_byte(built, bond) -> None:
    deck, archive = built(bond=bond)
    replayed = _replay(archive)
    assert replayed == deck
    # The ties come after the fixes, as the bridge writes them.
    lines = replayed.splitlines()
    last_fix = max(i for i, ln in enumerate(lines) if ln.startswith("fix "))
    first_tie = min(i for i, ln in enumerate(lines) if ln.startswith(_TIE))
    assert last_fix < first_tie


def test_rebar_cells_replay_keeps_the_planned_tie_tags(built) -> None:
    """The tie tags are planned before the bar cells, so they sit below the
    highest element tag; only the stored tags reproduce them."""
    deck, archive = built(cells=True)
    replayed = _replay(archive)
    ties, cells = _lines(deck, _TIE), _lines(deck, _CELL)
    assert cells and max(_tags(ties)) < min(_tags(cells))
    assert _lines(replayed, _TIE) == ties
    # Every other line matches in order. The cells are the one difference:
    # the replay writes them with the other elements, the bridge after the
    # ties (a replay ordering of minted elements, outside K2b).
    assert sorted(_lines(replayed, _CELL)) == sorted(cells)

    def rest(text: str) -> list[str]:
        return [ln for ln in text.splitlines() if not ln.startswith(_CELL)]

    assert rest(replayed) == rest(deck)


def test_new_archive_never_takes_the_fallback(
    built, tmp_path: Path, planted_fallback: list[int],
) -> None:
    for kw in ({}, {"bond": True}, {"cells": True}):
        deck, archive = built(**kw)
        replayed = _replay(archive)
        assert _lines(replayed, _TIE) == _lines(deck, _TIE)
        # A rewrite stores the same rows, and its replay reads them.
        out = _rewrite(archive, tmp_path, f"rw_{len(kw)}_{sorted(kw)}.h5")
        assert _stored(out) == _stored(archive)
        assert _replay(out) == replayed
    assert planted_fallback == []


def test_stored_args_are_what_replays(built, tmp_path: Path) -> None:
    """A replay writes the stored arguments, never ones re-derived from the
    neutral zone: a changed stored value comes back changed."""
    deck, src = built()
    path = _copy(src, tmp_path)
    with h5py.File(path, "r+") as f:
        g = f[_STORE]
        kinds = g["arg_kinds"][()]
        strs = [s.decode() if isinstance(s, bytes) else str(s)
                for s in g["args_str"][()]]
        j = strs.index("-perfect") + 1
        assert kinds[j] == 1
        args = g["args"][()]
        args[j] = 2.25e8
        g["args"][...] = args
    ties = _lines(_replay(path), _TIE)
    assert "-perfect 225000000.0" in ties[0]
    assert ties[1:] == _lines(deck, _TIE)[1:]


def test_store_without_neutral_ties_refuses(built, tmp_path: Path) -> None:
    """A 2.29 archive whose store and neutral zone disagree refuses: here the
    store is gone but the stamp still says 2.29."""
    _deck, src = built()
    path = _copy(src, tmp_path)
    with h5py.File(path, "r+") as f:
        del f[_STORE]
    with pytest.raises(ValueError, match="stores none"):
        OpenSeesModel.from_h5(str(path)).build("tcl")


# -- an archive below 2.29.0 ------------------------------------------------


def test_pre_2_29_archive_replays_through_the_fallback(
    built, tmp_path: Path,
) -> None:
    for kw in ({}, {"bond": True}):
        deck, src = built(**kw)
        path = _copy(src, tmp_path, f"old_{len(kw)}.h5")
        downgrade_to_pre_embedded_rebar(path)
        assert _stored(path) == ()
        assert _replay(path) == deck


def test_pre_2_29_archive_reaches_the_planted_fallback(
    built, tmp_path: Path, planted_fallback: list[int],
) -> None:
    _deck, src = built()
    path = _copy(src, tmp_path)
    downgrade_to_pre_embedded_rebar(path)
    with pytest.raises(AssertionError, match="planted fallback"):
        OpenSeesModel.from_h5(str(path)).build("tcl")
    assert planted_fallback == [1]


def test_pre_2_29_cells_archive_retags_its_ties(built, tmp_path: Path) -> None:
    """What the store fixes: the fallback numbers the ties past every
    replayed element, the cells included, not as the bridge planned them."""
    deck, src = built(cells=True)
    path = _copy(src, tmp_path)
    downgrade_to_pre_embedded_rebar(path)
    replayed = _replay(path)
    ties = _tags(_lines(replayed, _TIE))
    top = max(
        t for t in _tags(_lines(replayed, "element ")) if t not in ties)
    assert ties == list(range(top + 1, top + 1 + len(ties)))
    assert ties != _tags(_lines(deck, _TIE))


def test_pre_2_29_rewrite_stores_the_replayed_ties(
    built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rewrite of an older archive stores the ties its replay derives, so
    the new file replays to the same deck without the fallback."""
    deck, src = built()
    path = _copy(src, tmp_path)
    downgrade_to_pre_embedded_rebar(path)
    out = _rewrite(path, tmp_path)
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs["opensees_schema_version"] == OPENSEES_CURRENT
    assert _stored(out) == _stored(src)
    with h5_reader.open(str(out)) as model:
        runs = [r for r in model.program() if r.method == "embedded_rebar"]
    assert runs and all(
        r.store == "{scope}/constraints/embeddedRebar" for r in runs)

    def fail(*_a: Any, **_k: Any) -> None:
        raise AssertionError("planted fallback: replay_reinforce_ties re-derived")

    monkeypatch.setattr(constraints_mod, "replay_reinforce_ties", fail)
    assert _replay(out) == deck
