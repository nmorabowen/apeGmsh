"""R1b round-trip oracle (K2a, #1794): ``tcl() == from_h5(h5()).build('tcl')``.

For every model in :data:`CORPUS`, the deck the bridge writes and the deck
replayed from its own archive must be equal byte for byte:
``ops.tcl(path, flat=True)`` against
``OpenSeesModel.from_h5(ops.h5()).build("tcl", flat=True)``. ``flat=True`` on
the bridge because a partitioned archive replays a single-process deck (the
documented degrade that ``test_h5_commands_program.py`` already pins), and on
the replay because a partitioned archive records the ADR 0027 INV-5
numberer/system auto-chain, which the bridge's flat emit never writes: the
replay's ``flat=True`` drops exactly those recorded lines (#1832), so the
two decks are the same mode.

**The corpus is the replayable one.** These are the ``apeSees`` builders that
already live in ``tests/opensees/h5``. Each is imported read-only and called
as is. The golden emit corpus beside this file is built on ``FEMStub`` and
archives only the bridge zone, so it cannot be replayed (#1797). The ledger
header records that.

**Every difference is an assertion, not a skip.**
``roundtrip_oracle_ledger.txt`` holds one row per differing model:

- Class A: integral floats only (``1e6`` comes back as ``1000000``).
- Class B: the same lines in another order, once every number is
  normalised (so float text may differ too), **and** the replayed deck
  still declares every tag before a line uses it (K2c, #1816).
- Class C: a semantic difference, cited to its open issue (``#NNNN``), or a
  reorder that uses a tag before its declaration, which aborts the deck
  (#1814). No row is class C today; :func:`test_a_c_row_is_enforced` keeps
  the class enforced on a synthetic C diff.
- No row is class A or B either: #2193 replays the bridge's emit order, and
  #2194 each slot's int/float type. The tests below keep both classes
  enforced on synthetic diffs, made by switching either fix off.
- Class S: the model is not replayable, and replay raises the named error
  (#1797). No row is S either since #2195 put a real-``FEMData`` twin of
  the last one (the golden cell ``two_column_frame_partitioned /
  staged_partitioned / recording``) in the corpus; the ``FEMStub`` model
  itself keeps the class enforced (:func:`test_an_s_row_is_enforced`).

The ledger is empty. Its machinery stays, each rule held by synthetic rows.

A/B/C rows pin the sha256 of the exact normalised diff. The class is never
taken on trust: the oracle derives it from the two decks and fails a row whose
label differs, so a B row is the mechanical proof that its diff is order and
float text only, in an order that still declares before it uses. There is no
override: a label must equal the derived class. A waived model that starts to
match fails, and so does one that differs in a new way, which forces the
ledger to shrink.

**Ledger transitions.** The ledger is compared row by row with the copy at
the merge base with ``origin/main`` (:func:`_ledger_transition_errors`). A row
may be deleted, or re-pinned to downgrade its class (C to A or B) when a fix
leaves only order or float text; the derived class is the evidence. A B row
may likewise re-pin to A when a fix leaves only float text (#1814): an A diff
is a strict subset of a B diff, and the class is still re-derived for every
row, so the A label is proven the same way. A row may
always escalate (A or B to C) when the oracle learns to see more, and then
cites the issue that tracks the bug. An S row whose model moves to a real
``FEMData`` re-pins to its derived class A or B, and its reason cites that
evidence as ``derived <class> (#1885): ...`` (maintainer ruling on #1797). Every C row cites an issue as ``#NNNN``
(openness is not checkable offline; the PR lists each one and its state).
Anything else (an added row, a re-pin that keeps its class) fails as a
replay regression.

**Declare-before-use (K2c).** Multiset equality cannot see order, and a
reorder can be fatal: a replay that writes ``uniaxialMaterial Parallel 3 1
2`` above ``Steel01 1`` aborts in OpenSees (#1814). :func:`_order_violations`
checks the order statically, from the bridge's own reference metadata, not
from the Tcl grammar: the archive's ``/opensees/decls`` give each
declaration's allocator tags, and its ``/opensees/decl_params`` give every
declaration it references (a :class:`DeclRef` at any depth, so a
``Parallel``'s components, a ``Fiber`` patch's material, an element's
``transf`` / ``integration`` / ``damp``). The only grammar the check reads
is where a declaring line carries its tag (:data:`_TAG_TOKEN`); an
allocator kind missing from that table raises.

What the static check **cannot** see, because no metadata names it: node
and element tags used by element connectivity, ``fix``, ``mass``,
``equationConstraint``, ``region -node/-ele``, loads, recorders and
``addToParameter``, and every reference inside a ``proc`` body. Those need
per-command grammar. ``test_roundtrip_oracle_live.py`` is the second net:
it runs the bridge and replay decks of every replayable corpus model in a
subprocess, and each replay must run wherever its bridge deck runs.

``N_WAIVED_ORACLE`` may only go down, and must reach 0 by 2026-12-11 (K2,
KC4).
"""
from __future__ import annotations

import collections
import collections.abc
import dataclasses
import difflib
import hashlib
import importlib
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.replay_types import (
    ReplaySlotTypeWarning, SlotTypes,
)
from apeGmsh.opensees.emitter.h5_reader import (
    BridgeOnlyArchiveError, DeclarationTable, DeclRef, DeclStruct,
)
from tests.opensees.golden import builder

_LEDGER = Path(__file__).with_name("roundtrip_oracle_ledger.txt")

#: ``N_WAIVED_ORACLE`` (33 when the ledger opened, K2a, #1794). It may only
#: go down: 32 once #2193 replayed ``_frame_bridge[analysis=True]`` exactly,
#: 1 once #2194 kept every slot's int/float type (the S row was left), 0
#: once #2195 replaced that row's ``FEMStub`` model with a real twin.
_CEILING = 0
#: The K2 deadline: the ledger is empty by then, or KC4 applies.
DEADLINE = "2026-12-11"
_CLASSES = frozenset({"A", "B", "C", "S"})
#: An issue citation: every C row, and every escalated row, carries one.
_ISSUE = re.compile(r"#\d{4,}\b")


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
    # The twin with a lumped mass (#2277 review): fix and mass lines in one
    # partitioned deck, so their flat sections cannot swap unseen.
    _m("test_h5_decl_params", "_two_column_frame_partitioned", mass=True),
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
    # The stages_writer bridges are these two (#1880): listed once.
    _m("test_h5_stages_replay", "_no_param_two_stage_bridge"),
    _m("test_opensees_model_build_targets", "_author_simple_frame_apesees"),
    *(_m("test_reinforce_deck_replay", "_reinforced_ops", bond_by_name=b)
      for b in _B),
    # K2b (#1795): reinforce ties replayed from the stored rows, after the
    # fixes; exact matches.
    *(_m("test_reinforce_deck_replay", "_reinforced_fixed_ops", bond_by_name=b)
      for b in _B),
    # ... and after the equationConstraint rows (B5-g's 9b'), as the bridge.
    _m("test_reinforce_deck_replay", "_reinforced_fixed_ops", equation=True),
    # ... and with the bar's CorotTruss cells, which replay after the ties,
    # in the bridge's slot (#1828).
    _m("test_reinforce_deck_replay", "_reinforced_fixed_ops", rebar_cells=True),
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
_S_ERRORS = {
    "BridgeOnlyArchiveError:no-neutral-zone": (
        BridgeOnlyArchiveError,
        "the archive carries no FEM (no neutral zone: no 'nodes' group"),
}


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


def _replay(archive: Path, *, flat: bool = True) -> str:
    """The replayed deck (canonical text). ``flat=False`` only for the
    synthetic C fixture: the deck as replay wrote it before #1832. A
    declaration whose slot types replay cannot recover fails it (#2194)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", ReplaySlotTypeWarning)
        replay = OpenSeesModel.from_h5(str(archive)).build("tcl", flat=flat)
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


# ---------------------------------------------------------------------------
# Declare-before-use (K2c, #1816)
# ---------------------------------------------------------------------------

#: Where a declaring deck line carries its tag, by allocator kind (the
#: ``kind`` of an ``/opensees/decls/tags`` run): ``<kind> <type> <tag>``
#: is token 2, ``region <tag>`` token 1. ``None``: the kind is allocated
#: but its deck line carries no tag (``system UmfPack``, ``recorder Node
#: ...``), so no line declares or references it by tag. An allocator kind
#: missing here raises: the check never guesses where a tag sits.
_TAG_TOKEN: dict[str, int | None] = {
    "uniaxialMaterial": 2, "nDMaterial": 2, "section": 2,
    "geomTransf": 2, "beamIntegration": 2, "timeSeries": 2,
    "pattern": 2, "damping": 2, "element": 2,
    "region": 1,
    "constraints": None, "numberer": None, "system": None, "test": None,
    "algorithm": None, "integrator": None, "analysis": None,
    "recorder": None,
}


def _definition_sites(deck: str) -> dict[tuple[str, int], int]:
    """``(kind, tag) -> index of the first top-level line declaring it``.

    Only unindented lines count: an indented line is inside a block (a
    ``Fiber`` section's patches, a pattern body, a ``proc``).

    A redeclared ``(kind, tag)`` keeps its *first* line, and that is exact
    for declare-before-use: a use has a declaration above it iff its
    nearest preceding one exists; that exists iff the first declaration is
    above the use; and the first line a tag is used on is above every
    later use. So comparing first declaration with first use gives the
    nearest-preceding verdict for every line. What the check does not
    model is a ``wipe``/``remove`` between two declarations, which would
    unbind the earlier one. That is safe today: the bridge's decks wipe
    once, in the header, and its ``remove`` lines name only elements, sp
    constraints, parameters and recorders, none of which a ``decl_params``
    reference names. ``test_a_redeclared_tag_binds_by_first_declaration``
    pins the redeclaration behaviour, and a real corpus row,
    ``_staged_bridge[staged_pg=Rock]``, re-declares ``damping``,
    ``timeSeries``, ``beamIntegration`` and ``geomTransf`` tag 1 after a
    ``model BasicBuilder -ndf 3`` switch."""
    sites: dict[tuple[str, int], int] = {}
    for i, line in enumerate(deck.splitlines()):
        tok = line.split()
        if not tok or line[0].isspace():
            continue
        idx = _TAG_TOKEN.get(tok[0])
        if idx is None or len(tok) <= idx or not re.fullmatch(r"\d+", tok[idx]):
            continue
        sites.setdefault((tok[0], int(tok[idx])), i)
    return sites


def _refs(value: Any) -> set[str]:
    """Every declaration key a ``decl_params`` value references, at any
    depth (a tuple of materials, a ``Fiber`` patch's material)."""
    if isinstance(value, DeclRef):
        return {value.key}
    if isinstance(value, DeclStruct):
        value = value.fields
    if isinstance(value, collections.abc.Mapping):
        value = tuple(value.values())
    if isinstance(value, (tuple, list)):
        return set().union(*(_refs(v) for v in value))
    return set()


def _order_violations(deck: str, table: DeclarationTable) -> list[str]:
    """Each use of a tag on a line above the line that declares it.

    For every declaration that references others (``decl_params``), each
    deck line declaring one of its tags must come after the lines that
    declare every tag of each referenced declaration. A referenced tag
    the deck never declares is a violation too. A declaration with
    references but no tagged deck line raises: the check cannot place its
    use, and it says so rather than pass it.
    """
    sites = _definition_sites(deck)
    tags: dict[int, list[tuple[str, int]]] = collections.defaultdict(list)
    for kind, first, count, decl in table.tags:
        if kind not in _TAG_TOKEN:
            raise KeyError(
                f"allocator kind {kind!r} has no _TAG_TOKEN entry: say "
                "where its deck line carries the tag")
        if _TAG_TOKEN[kind] is not None:
            tags[decl].extend((kind, t) for t in range(first, first + count))
    index = {d.key: i for i, d in enumerate(table.decls)}
    out: list[str] = []
    for i, decl in enumerate(table.decls):
        refs = set().union(*(_refs(p.params) for p in table.params.get(i, ())))
        if not refs:
            continue
        if not tags[i]:
            raise AssertionError(
                f"{decl.key} references {sorted(refs)} but declares no "
                "tagged deck line: the order check cannot place its use")
        for kind, tag in tags[i]:
            if (kind, tag) not in sites:
                raise AssertionError(
                    f"{decl.key}: no top-level deck line declares {kind} "
                    f"{tag}; _TAG_TOKEN and the deck disagree")
            use = sites[(kind, tag)]
            for ref in sorted(refs):
                for rkind, rtag in tags[index[ref]]:
                    at = sites.get((rkind, rtag))
                    if at is None or at > use:
                        where = ("never" if at is None
                                 else f"on line {at + 1}")
                        out.append(
                            f"line {use + 1} ({kind} {tag}) uses {rkind} "
                            f"{rtag}, declared {where}")
    return out


def _deck_order_violations(deck: str, archive: Path) -> list[str]:
    table = OpenSeesModel.from_h5(str(archive)).declarations
    assert table is not None, f"{archive} carries no /opensees/decls"
    return _order_violations(deck, table)


def _classify(bridge: str, replay: str, order_ok: bool = True) -> str:
    """A, B or C for two decks that differ (re-derives the ledger's claim).

    ``order_ok`` is the replay's declare-before-use verdict: a reorder
    that uses a tag before declaring it is C, never B (K2c, #1816).
    """
    def norm(t: str) -> list[str]:
        return [_NUMBER.sub(lambda m: repr(float(m.group())), ln)
                for ln in t.splitlines()]
    nb, nr = norm(bridge), norm(replay)
    if nb == nr:
        return "A"
    if collections.Counter(nb) == collections.Counter(nr) and order_ok:
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
    bridge, archive = _emit(model, out_dir)
    replay = _replay(archive)
    # The bridge deck runs, so it must pass the order check: a bridge
    # violation is a false positive of the check (or a bridge bug).
    bridge_bad = _deck_order_violations(bridge, archive)
    assert not bridge_bad, (
        f"{model.id}: the bridge deck fails declare-before-use:\n"
        + "\n".join(bridge_bad))
    order_bad = _deck_order_violations(replay, archive)
    klass = _classify(bridge, replay, not order_bad)
    diff = _diff(bridge, replay)
    if waiver is None:
        assert bridge == replay, (
            f"{model.id}: the replayed deck differs from the bridge deck "
            f"(class {klass}, pin {_pin(diff)}); this is "
            f"a replay regression, not a new waiver:\n{diff}"
        )
        return
    assert bridge != replay, (
        f"{model.id} now round-trips byte for byte: delete its ledger row "
        "and lower N_WAIVED_ORACLE"
    )
    assert klass == waiver.klass, (
        f"{model.id}: the ledger says class {waiver.klass}, the diff is "
        f"class {klass}"
        + (";\nthe replay uses tags before declaring them:\n"
           + "\n".join(order_bad) if order_bad else "")
        + f"\n{diff}"
    )
    assert _pin(diff) == waiver.pin, (
        f"{model.id} differs in a new way (pin {_pin(diff)}, ledger "
        f"{waiver.pin}); a fix that changes the diff deletes its row, or "
        f"re-pins it only to downgrade its class (C to A or B, B to A):\n{diff}"
    )


# ---------------------------------------------------------------------------
# The carve-out, enforced against the ledger at the merge base
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[3]
_LEDGER_REL = "tests/opensees/golden/roundtrip_oracle_ledger.txt"
#: Class changes that need no evidence beyond the derived class: a semantic
#: row downgraded once its fix leaves only order or float text, and an
#: order row once its fix leaves only float text (B to A, #1814).
_DOWNGRADES = frozenset({("C", "A"), ("C", "B"), ("B", "A")})
#: Escalations: the oracle learned to see a bug its old rule missed (K2c).
_ESCALATIONS = frozenset({("A", "C"), ("B", "C")})
#: The one same-class re-pin allowed (#1797): an S row moves from h5py's raw,
#: unnamed ``KeyError`` to the named refusal ``from_h5`` now raises up front.
#: The S rows' model and class stay; only the pinned error is renamed.
_S_RENAMES = frozenset({
    ("KeyError:nodes", "BridgeOnlyArchiveError:no-neutral-zone"),
})
#: S rows released by a real FEMData (#1885; maintainer ruling on #1797): a
#: model that moves off ``FEMStub`` replays, and its row re-pins S to the
#: class the oracle derives, A or B (never C: a semantic diff is a bug with
#: its own issue, not a release). Like C to A/B, the derived class is the
#: evidence, and :func:`_check` fails a label it does not derive. The row
#: also cites it: its reason opens ``derived <class> (#1885):`` and says
#: what the diff is (:data:`_DERIVED`).
_S_RELEASES = frozenset({("S", "A"), ("S", "B")})
#: The citation an S-released row carries; its class must be the row's.
_DERIVED = re.compile(r"^derived (?P<klass>[AB]) \(#1885\): \S")


def _cited_issues(reason: str) -> frozenset[str]:
    """The ``#NNNN`` issue numbers a row's reason cites."""
    return frozenset(_ISSUE.findall(reason))


def _ledger_transition_errors(
    base: dict[str, Waiver], head: dict[str, Waiver],
) -> list[str]:
    """Each change from ``base`` to ``head`` the ledger rules forbid.

    A row may stay as it is (its reason text may change), be deleted, be
    re-pinned from class C to A or B, or B to A (the oracle's derived class
    is the evidence), or escalate from A or B to C, citing the issue that tracks
    the bug. Anything else is a replay regression: an added row, or a re-pin
    that keeps its class. One exception (:data:`_S_RENAMES`, #1797): an S row
    may re-pin at S from the raw ``KeyError:nodes`` to the named
    ``BridgeOnlyArchiveError``. And an S row whose model moved to a real
    ``FEMData`` may re-pin to A or B when its reason cites that derived
    class (:data:`_S_RELEASES`, #1885).
    """
    errors: list[str] = []
    for model, w in head.items():
        b = base.get(model)
        if b is None:
            errors.append(f"{model}: added row (class {w.klass}); a replay "
                          "regression, not a new waiver")
            continue
        if (w.klass, w.pin) == (b.klass, b.pin):
            continue
        if (b.klass, w.klass) in _DOWNGRADES:
            continue
        if (b.klass, w.klass) == ("S", "S") and (b.pin, w.pin) in _S_RENAMES:
            continue
        if (b.klass, w.klass) in _S_RELEASES:
            cited = _DERIVED.match(w.reason)
            if cited is None or cited["klass"] != w.klass:
                errors.append(
                    f"{model}: released S -> {w.klass} without citing its "
                    f"derived evidence (a reason opening 'derived {w.klass} "
                    "(#1885): <what the diff is>')")
            continue
        if (b.klass, w.klass) in _ESCALATIONS:
            if not _cited_issues(w.reason):
                errors.append(
                    f"{model}: escalated {b.klass} -> C without citing the "
                    "issue that tracks it (#NNNN)")
            continue
        errors.append(
            f"{model}: re-pinned {b.klass} {b.pin} -> {w.klass} {w.pin}; "
            "a replay regression: a re-pin may only downgrade (C to A or B, "
            "B to A), or escalate A or B to C")
    return errors


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_REPO), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def _base_ledger_text() -> str | None:
    """The ledger at the merge base of ``HEAD`` and ``origin/main``.

    ``None`` when the base predates the ledger. The suite lane checks out
    the full history (``fetch-depth: 0``), so a missing ``git`` or
    ``origin/main`` fails here rather than skipping the check.
    """
    base = _git("merge-base", "HEAD", "refs/remotes/origin/main").strip()
    listed = _git("ls-tree", "--name-only", base, "--", _LEDGER_REL).strip()
    if not listed:
        return None
    return _git("show", f"{base}:{_LEDGER_REL}")


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


def test_ledger_changes_only_by_the_carve_out() -> None:
    """Against the merge base: rows are deleted, downgraded or escalated."""
    text = _base_ledger_text()
    if text is None:
        return  # the ledger is new on this branch: nothing to compare to
    base_n, base_rows, _bh = _read_ledger(text)
    n, rows, _h = _read_ledger()
    assert n <= base_n, f"N_WAIVED_ORACLE rose from {base_n} to {n}"
    base = {w.model: w for w in base_rows}
    head = {w.model: w for w in rows}
    errors = _ledger_transition_errors(base, head)
    assert not errors, "\n".join(errors)


_W = Waiver  # short alias for the carve-out cases below
_NZ = "BridgeOnlyArchiveError:no-neutral-zone"


@pytest.mark.parametrize(("base", "head", "ok"), [
    (_W("m", "C", "1" * 16, "#1798 item 1"), _W("m", "B", "2" * 16, "r"), True),
    (_W("m", "C", "1" * 16, "#1798 item 1"), _W("m", "A", "2" * 16, "r"), True),
    (_W("m", "B", "1" * 16, "r"), _W("m", "B", "1" * 16, "new reason"), True),
    (_W("m", "B", "1" * 16, "r"), _W("m", "C", "1" * 16, "#1814: x"), True),
    (_W("m", "A", "1" * 16, "r"), _W("m", "C", "2" * 16, "#1814: x"), True),
    (_W("m", "B", "1" * 16, "r"), _W("m", "C", "1" * 16, "order"), False),
    (_W("m", "C", "1" * 16, "#1798 item 1"), _W("m", "C", "2" * 16, "#1814"),
     False),
    (_W("m", "B", "1" * 16, "r"), _W("m", "B", "2" * 16, "r"), False),
    (_W("m", "B", "1" * 16, "r"), _W("m", "A", "2" * 16, "r"), True),
    (_W("m", "S", "KeyError:nodes", "r"), _W("m", "B", "2" * 16, "r"), False),
    # #1797: the one same-class re-pin, raw KeyError -> the named refusal.
    (_W("m", "S", "KeyError:nodes", "#1797"),
     _W("m", "S", "BridgeOnlyArchiveError:no-neutral-zone", "#1797"), True),
    # ... and no other S-to-S re-pin: not back, not to another text or type.
    (_W("m", "S", "BridgeOnlyArchiveError:no-neutral-zone", "#1797"),
     _W("m", "S", "KeyError:nodes", "#1797"), False),
    (_W("m", "S", "KeyError:nodes", "#1797"),
     _W("m", "S", "BridgeOnlyArchiveError:other", "#1797"), False),
    (_W("m", "S", "KeyError:nodes", "#1797"),
     _W("m", "S", "MalformedH5Error:no-neutral-zone", "#1797"), False),
    (_W("m", "S", "BridgeOnlyArchiveError:no-neutral-zone", "#1797"),
     _W("m", "S", "BridgeOnlyArchiveError:other", "#1797"), False),
    # The exception is S-only: a same-class re-pin of A, B or C still fails,
    # even with the S rename's own pins.
    (_W("m", "A", "1" * 16, "r"), _W("m", "A", "2" * 16, "r"), False),
    (_W("m", "A", "KeyError:nodes", "r"),
     _W("m", "A", "BridgeOnlyArchiveError:no-neutral-zone", "r"), False),
    (_W("m", "B", "KeyError:nodes", "r"),
     _W("m", "B", "BridgeOnlyArchiveError:no-neutral-zone", "r"), False),
    (_W("m", "C", "KeyError:nodes", "#1814"),
     _W("m", "C", "BridgeOnlyArchiveError:no-neutral-zone", "#1814"), False),
    # #1885: an S row released by a real FEMData re-pins to its derived
    # class A or B, citing it ...
    (_W("m", "S", _NZ, "#1797"),
     _W("m", "A", "2" * 16, "derived A (#1885): integral floats"), True),
    (_W("m", "S", _NZ, "#1797"),
     _W("m", "B", "2" * 16, "derived B (#1885): geomTransf moves"), True),
    # ... and is refused when the citation is absent, names another class,
    # another issue, or says nothing about the diff; never to C.
    (_W("m", "S", _NZ, "#1797"), _W("m", "A", "2" * 16, "#1885"), False),
    (_W("m", "S", _NZ, "#1797"),
     _W("m", "B", "2" * 16, "derived A (#1885): integral floats"), False),
    (_W("m", "S", _NZ, "#1797"),
     _W("m", "A", "2" * 16, "derived A (#1797): integral floats"), False),
    (_W("m", "S", _NZ, "#1797"), _W("m", "A", "2" * 16, "derived A (#1885):"),
     False),
    (_W("m", "S", _NZ, "#1797"),
     _W("m", "C", "2" * 16, "derived C (#1885): #1814 reorder"), False),
    # The citation releases S rows only: it does not license a same-class
    # re-pin (B -> A is already a plain downgrade, #1909).
    (_W("m", "B", "1" * 16, "r"),
     _W("m", "B", "2" * 16, "derived B (#1885): geomTransf moves"), False),
], ids=["C-to-B", "C-to-A", "reason-only", "B-to-C", "A-to-C",
        "B-to-C-uncited", "C-to-C", "B-to-B", "B-to-A", "S-to-B",
        "S-rename-1797", "S-rename-reversed", "S-other-text", "S-other-type",
        "S-named-to-other", "A-to-A", "A-rename-pins", "B-rename-pins",
        "C-rename-pins", "S-released-A", "S-released-B", "S-to-A-uncited",
        "S-to-B-cites-A", "S-to-A-other-issue", "S-to-A-no-diff-text",
        "S-to-C-cited", "B-to-B-cited"])
def test_the_ledger_allows_downgrades_and_cited_escalations(
    base: Waiver, head: Waiver, ok: bool,
) -> None:
    errors = _ledger_transition_errors({"m": base}, {"m": head})
    assert (not errors) is ok, errors


_HELD = next(m for m in CORPUS if m.id == "test_h5_decl_params::_broad_frame")


def _untyped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch #2194 off in-process: every slot keeps the reader's by-value
    type, so an integral float replays as an int, as before the fix."""
    monkeypatch.setattr(
        SlotTypes, "recover",
        lambda self, kind, tag, type_token, args, nodes=0: tuple(args))


def _a_row(model: Model, out_dir: Path) -> Waiver:
    """An A row pinned to ``model``'s diff with slot types off (the caller
    has applied :func:`_untyped`): the float text #2194 removed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge, archive = _emit(model, out_dir)
    replay = _replay(archive)
    assert bridge != replay and _classify(bridge, replay) == "A"
    return Waiver(model.id, "A", _pin(_diff(bridge, replay)), "synthetic")


def test_broad_frame_replays_exactly(tmp_path: Path) -> None:
    """#1814 fixed: replay declares ``_broad_frame``'s materials in the
    bridge's order, so the order check passes (it was C while ``Parallel 3``
    preceded ``Steel01 1``). #2193 fixed the rest of its order (it was B,
    then A), and #2194 its float text: the replay is the bridge deck."""
    bridge, archive = _emit(_HELD, tmp_path)
    replay = _replay(archive)
    assert _deck_order_violations(replay, archive) == []
    assert replay == bridge
    assert _HELD.id not in _waivers()


def test_a_label_must_equal_the_derived_class(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No override either way: a C label over a derived A fails (the
    interim override #1812 had is gone), as a B label over a C does. The
    A diff is synthetic (slot types off), since no ledger row is A."""
    model = next(m for m in CORPUS
                 if m.id == "test_h5_damping::_build_with_element_damp")
    _untyped(monkeypatch)
    w = _a_row(model, tmp_path / "pin")
    (tmp_path / "a").mkdir()
    _check(model, w, tmp_path / "a")
    raised = Waiver(w.model, "C", w.pin, "#1814: hand-held at C")
    (tmp_path / "c").mkdir()
    with pytest.raises(AssertionError,
                       match="the ledger says class C, the diff is class A"):
        _check(model, raised, tmp_path / "c")


# Revert proof (#2193): with the emit-index merge off, a cleared row fails.

#: The rows #2193 re-pinned B -> A and #2194 deleted: their replay follows
#: the program's order.
_REPINNED = (
    _HELD,
    next(m for m in CORPUS if m.id ==
         "test_h5_builder_scope_replay::build_mixed_ndf_bridge"),
    next(m for m in CORPUS if m.id ==
         "test_h5_decl_params::_broad_staged"),
)


@pytest.mark.parametrize("model", _REPINNED, ids=[m.id for m in _REPINNED])
def test_without_the_emit_index_merge_a_cleared_row_fails(
        model: Model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert proof: replayed in the fixed category order
    (``OpenSeesModel._replay_order`` off, as before #2193), each model's
    diff is B again, a replay regression; with the merge on it has no row
    and replays byte for byte."""
    assert model.id not in _waivers()
    (tmp_path / "on").mkdir()
    _check(model, None, tmp_path / "on")
    monkeypatch.setattr(OpenSeesModel, "_replay_order", lambda self: None)
    (tmp_path / "off").mkdir()
    with pytest.raises(AssertionError, match=r"class B, pin [0-9a-f]{16}"):
        _check(model, None, tmp_path / "off")


#: Revert proof (#2194): models whose rows #2194 deleted, covering the
#: stores :class:`SlotTypes` types: materials and sections (Steel02, -GJ),
#: elements (quad thickness, nDMaterial), damping and elasticBeamColumn.
_RETYPED = tuple(next(m for m in CORPUS if m.id == i) for i in (
    "test_h5_apesees_cuts::_build_minimal_ops",
    "test_h5_decls::_flat_frame[named=False]",
    "test_h5_damping::_build_with_element_damp",
))


@pytest.mark.parametrize("model", _RETYPED, ids=[m.id for m in _RETYPED])
def test_without_slot_types_a_deleted_row_differs(
        model: Model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert proof: with :meth:`SlotTypes.recover` off (as before #2194),
    each model's replay writes integral floats as ints again, a class A
    replay regression; with it on, the model has no row and matches."""
    assert model.id not in _waivers()
    (tmp_path / "on").mkdir()
    _check(model, None, tmp_path / "on")
    _untyped(monkeypatch)
    (tmp_path / "off").mkdir()
    with pytest.raises(AssertionError, match=r"class A, pin [0-9a-f]{16}"):
        _check(model, None, tmp_path / "off")


def test_without_the_emit_index_merge_the_deleted_row_differs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert proof for the deleted row: ``_frame_bridge[analysis=True]``
    replays byte for byte (no row), and with the merge off its ``system`` and
    ``analysis`` fall below the declarations again: a replay regression."""
    model = next(m for m in CORPUS if m.id ==
                 "test_h5_solve_stamp::_frame_bridge[analysis=True]")
    assert model.id not in _waivers()
    (tmp_path / "on").mkdir()
    _check(model, None, tmp_path / "on")
    monkeypatch.setattr(OpenSeesModel, "_replay_order", lambda self: None)
    (tmp_path / "off").mkdir()
    with pytest.raises(AssertionError, match=r"class B, pin [0-9a-f]{16}"):
        _check(model, None, tmp_path / "off")


def test_an_added_row_fails_the_carve_out() -> None:
    errors = _ledger_transition_errors(
        {}, {"m": _W("m", "B", "1" * 16, "r")})
    assert len(errors) == 1 and "added row" in errors[0]
    assert "replay regression" in errors[0]


def test_a_deleted_row_passes_the_carve_out() -> None:
    assert _ledger_transition_errors(
        {"m": _W("m", "C", "1" * 16, "r")}, {}) == []


def _orphans(rows: "list[Waiver]") -> list[str]:
    """The rows that name no :data:`CORPUS` model."""
    corpus = {m.id for m in CORPUS}
    return [w.model for w in rows if w.model not in corpus]


def test_every_row_names_a_corpus_model() -> None:
    orphans = _orphans(_read_ledger()[1])
    assert not orphans, f"ledger rows for no corpus model: {orphans}"


def test_an_orphan_row_is_reported() -> None:
    """The rule above, on a synthetic row now that the ledger is empty: an
    S row for the ``FEMStub`` model, which is not in the corpus, is an
    orphan; the same row for its twin is not."""
    stub = _W(_STUB.id, "S", _NZ, "#1797: synthetic")
    assert _orphans([stub]) == [_STUB.id]
    assert _orphans([dataclasses.replace(stub, model=_TWIN.id)]) == []


def _row_errors(w: Waiver) -> list[str]:
    """What a ledger row's citation and pin get wrong (empty: nothing)."""
    errors: list[str] = []
    if w.klass == "C":
        if not _cited_issues(w.reason):
            errors.append(f"C row cites no #NNNN: {w}")
        if "#1798" in w.reason and not re.search(
                r"#1798 item [123]\b", w.reason):
            errors.append(f"#1798 cited with no item: {w}")
    if w.klass == "S" and not ("#1797" in w.reason and w.pin in _S_ERRORS):
        errors.append(f"S row lacks #1797 or a known error pin: {w}")
    if w.klass in "ABC" and not re.fullmatch(r"[0-9a-f]{16}", w.pin):
        errors.append(f"pin is not sha256[:16]: {w}")
    return errors


@pytest.mark.parametrize(("row", "ok"), [
    (_W("m", "C", "1" * 16, "#1814: x"), True),
    (_W("m", "C", "1" * 16, "order"), False),
    (_W("m", "C", "1" * 16, "#1798 item 2"), True),
    (_W("m", "C", "1" * 16, "#1798"), False),
    (_W("m", "S", _NZ, "#1797"), True),
    (_W("m", "S", _NZ, "stub"), False),
    (_W("m", "S", "KeyError:other", "#1797"), False),
    (_W("m", "A", "1" * 16, "r"), True),
    (_W("m", "B", "nothex", "r"), False),
], ids=["C-cited", "C-uncited", "C-1798-item", "C-1798-no-item", "S-ok",
        "S-uncited", "S-unknown-error", "A-ok", "B-bad-pin"])
def test_the_row_rules_reject_an_uncited_row(row: Waiver, ok: bool) -> None:
    """The rules :func:`test_rows_cite_their_issue` applies, on synthetic
    rows: with no C row in the ledger (#1832), they still assert."""
    assert (not _row_errors(row)) is ok, _row_errors(row)


def test_rows_cite_their_issue() -> None:
    errors = [e for w in _read_ledger()[1] for e in _row_errors(w)]
    assert not errors, "\n".join(errors)


def test_ledger_header_names_the_deadline_and_the_stub_corpus() -> None:
    header = _read_ledger()[2]
    assert DEADLINE in header and "KC4" in header
    assert "#1797" in header and "bridge-only" in header


# Revert proofs, in-process: the ledger machinery catches a dropped row and a
# stale one (the PR body shows the same against the committed file).

_PASSING = next(m for m in CORPUS if m.id ==
                "test_h5_commands_program::_frame[rayleigh=False,modal=False]")
#: A model whose replay is float text away from the bridge deck once slot
#: types are off (:func:`_untyped`): the synthetic A diff.
_FLOAT_TEXT = next(m for m in CORPUS if m.id ==
                   "test_h5_decls::_flat_frame[named=False]")
#: The one corpus model whose archive records the ADR 0027 INV-5 auto-chain
#: (``*_runtime_fallback`` attrs). Its flat replay is the bridge deck (an A
#: row until #2194); its non-flat replay, what the oracle compared before
#: #1832, adds the auto-chain lines: a real semantic diff, the synthetic C
#: fixture.
_PARTITIONED = next(m for m in CORPUS if m.id ==
                    "test_h5_partitions_roundtrip::_build_bridge")


def _semantic(out_dir: Path) -> tuple[str, str, Waiver]:
    """The bridge deck, the non-flat replay of :data:`_PARTITIONED`, and a
    C row pinned to their diff."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge, archive = _emit(_PARTITIONED, out_dir)
    replay = _replay(archive, flat=False)
    row = Waiver(_PARTITIONED.id, "C", _pin(_diff(bridge, replay)),
                 "#1832: synthetic, the non-flat replay")
    return bridge, replay, row


def _unflat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the oracle's replay through ``build("tcl")`` (no ``flat``)."""
    real_replay = _replay
    monkeypatch.setattr(sys.modules[__name__], "_replay",
                        lambda archive: real_replay(archive, flat=False))


def test_a_differing_model_with_no_row_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _untyped(monkeypatch)
    with pytest.raises(AssertionError, match="replay regression"):
        _check(_FLOAT_TEXT, None, tmp_path)


def test_a_stale_row_fails(tmp_path: Path) -> None:
    bogus = Waiver(_PASSING.id, "A", "0" * 16, "bogus")
    with pytest.raises(AssertionError, match="now round-trips byte for byte"):
        _check(_PASSING, bogus, tmp_path)


#: The ADR 0027 INV-5 auto-chain as ``TclEmitter`` writes it: what the
#: non-flat replay of :data:`_PARTITIONED` adds and ``flat=True`` drops.
_AUTO_CHAIN = (
    "# ParallelPlain only exists in OpenSeesMP; fall back to RCM under "
    "single-process OpenSees.",
    "if {[catch {numberer ParallelPlain} _err]} { numberer RCM }",
    "# Mumps requires OpenSeesMP + MPI; fall back to UmfPack under "
    "single-process OpenSees.",
    "if {[catch {system Mumps} _err]} { system UmfPack }",
)


def test_the_non_flat_replay_is_a_semantic_diff(tmp_path: Path) -> None:
    """The synthetic C fixture is what #1832 removed: the non-flat replay
    is the flat one plus exactly the four auto-chain lines, in one run."""
    bridge, unflat, _row = _semantic(tmp_path / "unflat")
    (tmp_path / "flat").mkdir()
    _bridge, archive = _emit(_PARTITIONED, tmp_path / "flat")
    flat = _replay(archive)
    assert _classify(bridge, unflat) == "C"
    assert flat == bridge
    u, f = unflat.splitlines(), flat.splitlines()
    at = u.index(_AUTO_CHAIN[0])
    assert tuple(u[at:at + len(_AUTO_CHAIN)]) == _AUTO_CHAIN
    assert u[:at] + u[at + len(_AUTO_CHAIN):] == f


def test_a_c_row_is_enforced(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no C row in the ledger, a synthetic one: a C row pinned to the
    non-flat replay's diff passes the oracle, and the same row fails once
    its pin no longer matches the diff."""
    _bridge, _unflat_deck, row = _semantic(tmp_path / "pin")
    _unflat(monkeypatch)
    (tmp_path / "ok").mkdir()
    _check(_PARTITIONED, row, tmp_path / "ok")
    moved = dataclasses.replace(row, pin="f" * 16)
    (tmp_path / "moved").mkdir()
    with pytest.raises(AssertionError, match="differs in a new way"):
        _check(_PARTITIONED, moved, tmp_path / "moved")


def test_a_b_label_on_a_semantic_diff_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The class is derived: a C diff labelled B fails, even with its pin
    (on the synthetic C fixture, since no ledger row is C)."""
    _bridge, _unflat_deck, row = _semantic(tmp_path / "pin")
    _unflat(monkeypatch)
    fake = dataclasses.replace(row, klass="B")
    (tmp_path / "b").mkdir()
    with pytest.raises(AssertionError,
                       match="the ledger says class B, the diff is class C"):
        _check(_PARTITIONED, fake, tmp_path / "b")


def test_without_flat_the_partitioned_model_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert proof (#1832): replayed without ``flat=True``, ``_build_bridge``
    (no row since #2194) is a class C replay regression."""
    assert _PARTITIONED.id not in _waivers()
    (tmp_path / "flat").mkdir()
    _check(_PARTITIONED, None, tmp_path / "flat")
    _unflat(monkeypatch)
    (tmp_path / "unflat").mkdir()
    with pytest.raises(AssertionError, match=r"class C, pin [0-9a-f]{16}"):
        _check(_PARTITIONED, None, tmp_path / "unflat")


def test_a_changed_diff_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _untyped(monkeypatch)
    w = _a_row(_FLOAT_TEXT, tmp_path / "pin")
    moved = Waiver(w.model, w.klass, "f" * 16, w.reason)
    (tmp_path / "moved").mkdir()
    with pytest.raises(AssertionError, match="differs in a new way"):
        _check(_FLOAT_TEXT, moved, tmp_path / "moved")


# Declare-before-use revert proofs (K2c, #1816).

_DAMPED = next(m for m in CORPUS if m.id ==
               "test_h5_damping::_build_with_element_damp")


def _move(deck: str, line: str, above: str) -> str:
    """``deck`` with the line starting ``line`` moved just above the line
    starting ``above``."""
    lines = deck.splitlines()
    (moved,) = [ln for ln in lines if ln.startswith(line)]
    lines.remove(moved)
    at = next(i for i, ln in enumerate(lines) if ln.startswith(above))
    return "\n".join(lines[:at] + [moved] + lines[at:])


def test_a_fatal_reorder_is_class_c_and_a_benign_one_stays_b(
        tmp_path: Path) -> None:
    """The element uses ``damping 1`` and ``geomTransf 1``: above them it
    aborts the deck (C); swapping the two declarations is harmless (B)."""
    bridge, archive = _emit(_DAMPED, tmp_path)
    table = OpenSeesModel.from_h5(str(archive)).declarations
    assert table is not None
    assert _order_violations(bridge, table) == []

    fatal = _move(bridge, "element ", above="damping ")
    bad = _order_violations(fatal, table)
    assert bad == [
        "line 5 (element 2) uses damping 1, declared on line 6",
        "line 5 (element 2) uses geomTransf 1, declared on line 7",
    ]
    assert _classify(bridge, fatal, not bad) == "C"

    benign = _move(bridge, "geomTransf ", above="damping ")
    assert benign != bridge
    assert _order_violations(benign, table) == []
    assert _classify(bridge, benign, True) == "B"

    # A referenced tag the deck never declares is a violation too.
    dropped = "\n".join(
        ln for ln in bridge.splitlines() if not ln.startswith("damping "))
    assert _order_violations(dropped, table) == [
        "line 6 (element 2) uses damping 1, declared never"]


def test_a_redeclared_tag_binds_by_first_declaration(tmp_path: Path) -> None:
    """A synthetic redeclaration (#1836): a second ``damping 1`` below the
    element changes nothing, because the first is above it; a second
    ``element 2`` above the declarations fails at that first line."""
    bridge, archive = _emit(_DAMPED, tmp_path)
    table = OpenSeesModel.from_h5(str(archive)).declarations
    assert table is not None
    lines = bridge.splitlines()
    (damping,) = [ln for ln in lines if ln.startswith("damping ")]
    (element,) = [ln for ln in lines if ln.startswith("element ")]

    late = "\n".join([*lines, damping])
    assert _definition_sites(late)[("damping", 1)] == lines.index(damping)
    assert _order_violations(late, table) == []

    # The element's second line is legal; the first, above both of its
    # references, is reported. Line numbers are 1-based in the shifted deck.
    at = lines.index(damping)
    transf = next(i for i, ln in enumerate(lines) if ln.startswith("geomTransf "))
    assert at < transf
    early = "\n".join([*lines[:at], element, *lines[at:]])
    assert _definition_sites(early)[("element", 2)] == at
    assert _order_violations(early, table) == [
        f"line {at + 1} (element 2) uses damping 1, declared on line {at + 2}",
        f"line {at + 1} (element 2) uses geomTransf 1, declared on line "
        f"{transf + 2}",
    ]


def test_an_unknown_allocator_kind_raises(tmp_path: Path) -> None:
    _bridge, archive = _emit(_DAMPED, tmp_path)
    table = OpenSeesModel.from_h5(str(archive)).declarations
    assert table is not None
    odd = dataclasses.replace(table, tags=(*table.tags, ("frobnicator", 1, 1, 0)))
    with pytest.raises(KeyError, match="frobnicator"):
        _order_violations(_bridge, odd)


def test_without_the_order_check_a_fatal_reorder_fails_its_pin(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert proof: a replay with ``Parallel 3`` moved above ``Steel01 1``
    (the #1814 reorder, now synthetic since replay no longer makes it)
    is C under its pin; with declare-before-use disabled it reads as B
    again, and the C row fails."""
    real_replay = _replay

    def fatal_replay(archive: Path) -> str:
        return _move(real_replay(archive), "uniaxialMaterial Parallel 3 ",
                     above="uniaxialMaterial Steel01 1 ")

    monkeypatch.setattr(sys.modules[__name__], "_replay", fatal_replay)
    (tmp_path / "pin").mkdir()
    bridge, archive = _emit(_HELD, tmp_path / "pin")
    fatal = fatal_replay(archive)
    assert ("line 5 (uniaxialMaterial 3) uses uniaxialMaterial 1, "
            "declared on line 6") in _deck_order_violations(fatal, archive)
    row = Waiver(_HELD.id, "C", _pin(_diff(bridge, fatal)), "#1814: synthetic")
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    _check(_HELD, row, tmp_path / "on")
    monkeypatch.setattr(
        sys.modules[__name__], "_deck_order_violations", lambda *a: [])
    with pytest.raises(AssertionError,
                       match="the ledger says class C, the diff is class B"):
        _check(_HELD, row, tmp_path / "off")


# ---------------------------------------------------------------------------
# The last S row's twin (#2195): the golden cell on a real FEMData
# ---------------------------------------------------------------------------

#: The real-``FEMData`` twin of the golden cell, in the corpus with no row.
_TWIN = next(m for m in CORPUS if m.id == "test_h5_decl_params::_broad_partitioned")
#: The twin with a lumped mass on ``"Top"`` (the #2277 review finding).
_MASSED = next(m for m in CORPUS if m.id ==
               "test_h5_decl_params::_two_column_frame_partitioned[mass=True]")
#: The golden cell's own ``FEMStub`` model: bridge-only, so not in the corpus.
_STUB = _m("test_h5_decl_params", "_broad_partitioned_stub")
#: The cell the twin reproduces.
_CELL = ("two_column_frame_partitioned", "staged_partitioned", "recording")
#: The row this slice deleted; re-adding it must fail the carve-out.
_DELETED_S_ROW = (
    "test_h5_decl_params::_broad_partitioned S "
    "BridgeOnlyArchiveError:no-neutral-zone  # #1797: golden-corpus model "
    "(build_model), bridge-only by design")


def _twin_deck(build: Callable[[], apeSees], out_dir: Path) -> str:
    """``build()``'s non-flat Tcl deck as the golden cell renders it
    (``builder.render_deck``: ``ops.tcl(model.tcl)``, collected)."""
    out_dir.mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build().tcl(str(out_dir / "model.tcl"))
    return builder._collect(out_dir)


def _golden_without_stamp() -> str:
    """The cell's golden deck minus its one backend-stamp line: the golden
    pins ``OpenSeesTarget(mode="stock")``, the unpinned twin stamps nothing
    (T ruling on #2195; archiving the mode is #2278)."""
    path = builder.golden_paths(*_CELL)[0]
    lines = path.read_bytes().decode("utf-8").replace("\r\n", "\n").split("\n")
    stamps = [ln for ln in lines if builder.STAMP_LINE.match(ln)]
    assert stamps == [f"# apeGmsh {builder.VERSION_TOKEN}; backend stock"], stamps
    return "\n".join(ln for ln in lines if ln not in stamps)


def _twin_factory(**kwargs: Any) -> Callable[[], apeSees]:
    return lambda: _m("test_h5_decl_params", "_two_column_frame_partitioned",
                      **kwargs).build()


def test_the_twin_is_the_golden_cell(tmp_path: Path) -> None:
    """The twin's bridge deck is the golden cell's deck byte for byte, but
    for the stamp line the unpinned twin does not write."""
    deck = _twin_deck(_TWIN.build, tmp_path / "twin")
    assert not builder.STAMP_LINE.search(deck)
    assert deck == _golden_without_stamp()


#: Mutations of the twin: a dropped stage, and the two columns' ranks
#: swapped. Each changes the deck, so the golden equality is not vacuous.
_TWIN_MUTATIONS = {
    "drop-stage-push": {"stages": (("gravity", 10.0, 2),)},
    "swap-ranks": {"partitions": {
        1: {"node_ids": [3, 4], "element_ids": [2]},
        2: {"node_ids": [1, 2], "element_ids": [1]}}},
}


@pytest.mark.parametrize("mutation", list(_TWIN_MUTATIONS))
def test_a_mutated_twin_is_not_the_golden_cell(
        mutation: str, tmp_path: Path) -> None:
    deck = _twin_deck(_twin_factory(**_TWIN_MUTATIONS[mutation]), tmp_path / "m")
    assert builder.first_deck_mismatch(_golden_without_stamp(), deck) is not None


def test_a_twin_with_one_partition_is_refused(tmp_path: Path) -> None:
    """Dropping a partition leaves a serial deck that declares ``system
    Mumps``, which the bridge refuses: the partitions are load-bearing."""
    from apeGmsh.opensees._internal.build.errors import BridgeError

    one = {1: {"node_ids": [1, 2, 3, 4], "element_ids": [1, 2]}}
    with pytest.raises(BridgeError, match=r"`system Mumps` declared on a serial deck"):
        _twin_deck(_twin_factory(partitions=one), tmp_path / "m")


def test_the_twin_replays_exactly_without_a_row(tmp_path: Path) -> None:
    """The S row is gone: the twin and its massed variant replay byte for
    byte, and the replay is partitioned (its program opens rank blocks)."""
    assert _waivers() == {}
    for model in (_TWIN, _MASSED):
        assert model.id not in _waivers()
        out = tmp_path / model.function
        out.mkdir()
        _check(model, None, out)
        order = OpenSeesModel.from_h5(str(out / "model.h5"))._replay_order()
        assert order is not None and order.partitioned


@pytest.mark.parametrize("model", [_TWIN, _MASSED], ids=["twin", "massed"])
def test_without_the_flat_section_sort_the_twin_fails(
        model: Model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert proof for S3b (#2277) through the twin: with every unit in
    section 0, as before the fix, the partitioned replay is in emit order
    and differs from the bridge's flat deck, a replay regression."""
    from apeGmsh.opensees._internal import compose

    (tmp_path / "on").mkdir()
    _check(model, None, tmp_path / "on")
    monkeypatch.setattr(compose._ReplayOrder, "section", lambda self, kind, key: 0)
    (tmp_path / "off").mkdir()
    with pytest.raises(AssertionError, match="replay regression"):
        _check(model, None, tmp_path / "off")


def test_swapped_fix_and_mass_sections_fail(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The #2277 review finding: swap ``fixes`` and ``masses`` in
    ``_FLAT_SECTIONS`` and the massed twin replays its ``mass`` lines above
    its ``fix`` lines, a class B replay regression. The mass-free twin has
    no ``mass`` line, so the same swap leaves it exact: the mass is what
    makes the corpus see the swap."""
    from apeGmsh.opensees._internal import compose

    sections = compose._FLAT_SECTIONS
    fixes, masses = sections["fixes"], sections["masses"]
    assert fixes < masses
    monkeypatch.setitem(sections, "fixes", masses)
    monkeypatch.setitem(sections, "masses", fixes)
    (tmp_path / "twin").mkdir()
    _check(_TWIN, None, tmp_path / "twin")
    (tmp_path / "massed").mkdir()
    with pytest.raises(AssertionError, match="class B, pin"):
        _check(_MASSED, None, tmp_path / "massed")
    (tmp_path / "diff").mkdir()
    diff = _diff(*_emit_and_replay(_MASSED, tmp_path / "diff"))
    changed = [ln for ln in diff.splitlines()
               if ln[0] in "+-" and not ln.startswith(("+++", "---"))]
    assert changed and {ln[1:].split()[0] for ln in changed} == {"mass"}


def test_an_s_row_is_enforced(tmp_path: Path) -> None:
    """With no S row in the ledger, the stub model holds the class: an S
    row on it passes (the bridge emits, ``from_h5`` refuses by name); with
    no row its refusal fails the oracle; and an S row on the twin, which
    replays, fails too."""
    row = _W(_STUB.id, "S", _NZ, "#1797: synthetic, the FEMStub model")
    assert _row_errors(row) == []
    (tmp_path / "s").mkdir()
    _check(_STUB, row, tmp_path / "s")
    (tmp_path / "none").mkdir()
    with pytest.raises(BridgeOnlyArchiveError, match="no neutral zone"):
        _check(_STUB, None, tmp_path / "none")
    (tmp_path / "twin").mkdir()
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        _check(_TWIN, dataclasses.replace(row, model=_TWIN.id), tmp_path / "twin")


def test_re_adding_the_deleted_s_row_fails_the_carve_out() -> None:
    """The committed ledger with the deleted S row appended: its count
    exceeds :data:`_CEILING`, and against the committed ledger the row is
    an added one, refused as a replay regression."""
    text = _LEDGER.read_text(encoding="utf-8").replace(
        "N_WAIVED_ORACLE = 0", "N_WAIVED_ORACLE = 1")
    n, rows, _h = _read_ledger(f"{text}\n{_DELETED_S_ROW}\n")
    assert n == len(rows) == 1 and n > _CEILING
    errors = _ledger_transition_errors(
        {w.model: w for w in _read_ledger()[1]}, {w.model: w for w in rows})
    assert len(errors) == 1 and "added row (class S)" in errors[0]
