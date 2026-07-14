"""ADR 0074 adversarial-review fixes — H5-replay builder bracket (F7) and
per-element-TYPE live fork verification (F8).

Both are latent-corruption paths the review surfaced:

* F7: the compose / from_h5 replay re-emits element command lines after one
  global ``model(ndm, ndf)`` — without the builder-ndf bracket the forward
  emit applies, a mixed-envelope equal-order LadrunoUP archive replays into a
  deck the fork parser refuses (OPS_LadrunoUP.cpp:162, ndf != ndm+1).
* F8: the live fork-only element gate keyed verification on a single boolean,
  so a build knowing one fork element (LadrunoQuad) would wave an unknown one
  (LadrunoUP) through unverified. It must verify per element TYPE.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from apeGmsh.opensees._internal.compose import _replay_elements_bracketed
from apeGmsh.opensees.emitter.recording import RecordingEmitter


def _rec(type_token: str, tag: int, conn: tuple[int, ...]):
    """Minimal stand-in for a rehydrated ElementRecord."""
    return SimpleNamespace(
        type_token=type_token, tag=tag, connectivity=conn,
        fem_eid=tag, args=("matArgs",),
    )


# ============================================================================
# F7 — replay builder-ndf bracket
# ============================================================================

class TestReplayBracket:
    def test_equal_order_up_brackets_under_mixed_envelope(self) -> None:
        """Envelope ndf=3 (dry solids) + an equal-order LadrunoUP H8 element
        → the element must be wrapped in model(-ndf 4) then restored to 3."""
        e = RecordingEmitter()
        _replay_elements_bracketed(
            e,
            [_rec("stdBrick", 1, tuple(range(1, 9))),
             _rec("LadrunoUP", 2, tuple(range(9, 17)))],
            ndm=3, envelope_ndf=3,
        )
        kinds = [(c[0], c[2].get("ndf")) for c in e.calls]
        # stdBrick (no gate) → element; then model ndf=4, LadrunoUP, restore 3.
        assert ("model", 4) in kinds
        # bracket restored to the envelope after the gated run.
        assert kinds[-1] == ("model", 3)
        # the LadrunoUP element line was emitted between the open and restore.
        model4_i = next(i for i, k in enumerate(kinds) if k == ("model", 4))
        up_i = next(
            i for i, c in enumerate(e.calls)
            if c[0] == "element" and c[1][0] == "LadrunoUP"
        )
        assert model4_i < up_i < len(kinds) - 1

    def test_no_bracket_when_envelope_already_matches(self) -> None:
        """Pure-saturated envelope ndf=4 → equal-order LadrunoUP needs no
        bracket (no redundant model lines)."""
        e = RecordingEmitter()
        _replay_elements_bracketed(
            e, [_rec("LadrunoUP", 1, tuple(range(1, 9)))],
            ndm=3, envelope_ndf=4,
        )
        assert not any(c[0] == "model" for c in e.calls)
        assert [c[0] for c in e.calls] == ["element"]

    def test_contiguous_gated_run_is_coalesced(self) -> None:
        """Consecutive gated elements share ONE bracket open (not one model
        line per element)."""
        e = RecordingEmitter()
        _replay_elements_bracketed(
            e,
            [_rec("LadrunoUP", i, tuple(range(i, i + 8))) for i in (1, 9, 17)],
            ndm=3, envelope_ndf=3,
        )
        model_opens = [c for c in e.calls if c == ("model", (), {"ndm": 3, "ndf": 4})]
        assert len(model_opens) == 1          # one open for the whole run
        assert e.calls[-1] == ("model", (), {"ndm": 3, "ndf": 3})  # one restore

    def test_th_shape_bracket_is_harmless_and_restored(self) -> None:
        """A TH tet10 LadrunoUP under a mixed envelope still opens/closes the
        bracket (element_builder_ndf is class-keyed); harmless (TH skips the
        fork builder gate) and the envelope is restored."""
        e = RecordingEmitter()
        _replay_elements_bracketed(
            e, [_rec("LadrunoUP", 1, tuple(range(1, 11)))],
            ndm=3, envelope_ndf=3,
        )
        assert e.calls[0] == ("model", (), {"ndm": 3, "ndf": 4})
        assert e.calls[-1] == ("model", (), {"ndm": 3, "ndf": 3})


# ============================================================================
# F8 — per-TYPE live fork verification
# ============================================================================

class _FakeOps:
    """Openseespy stand-in: records element() calls; getEleTags returns every
    tag ever created. ``unknown`` element types are 'silently dropped' (not
    added), mimicking a build that lacks that element."""

    def __init__(self, unknown: set[str]):
        self._unknown = unknown
        self._tags: list[int] = []
        self.element_calls: list[str] = []

    def element(self, ele_type, tag, *args):
        self.element_calls.append(ele_type)
        if ele_type not in self._unknown:
            self._tags.append(int(tag))
        return 0

    def getEleTags(self):
        return list(self._tags)


def _live_emitter(fake_ops):
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    e = LiveOpsEmitter.__new__(LiveOpsEmitter)
    e._ops = fake_ops
    e._in_partition = False
    e._fork_verified_types = set()
    return e


def test_second_fork_type_is_still_verified() -> None:
    """A build that knows LadrunoQuad but NOT LadrunoUP: emitting LadrunoQuad
    first must NOT wave LadrunoUP through — the unknown second type raises the
    curated fork-required error instead of silently vanishing."""
    fake = _FakeOps(unknown={"LadrunoUP"})
    e = _live_emitter(fake)

    # First fork element (known) verifies fine.
    e.element("LadrunoQuad", 1, 1, 2, 3, 4)
    assert "LadrunoQuad" in e._fork_verified_types

    # Second fork element (unknown to this build) is STILL gated → raises.
    with pytest.raises(RuntimeError, match="requires the Ladruno fork"):
        e.element("LadrunoUP", 2, 1, 2, 3, 4)


def test_known_second_fork_type_passes_and_is_recorded() -> None:
    fake = _FakeOps(unknown=set())
    e = _live_emitter(fake)
    e.element("LadrunoQuad", 1, 1, 2, 3, 4)
    e.element("LadrunoUP", 2, 1, 2, 3, 4)
    assert {"LadrunoQuad", "LadrunoUP"} <= e._fork_verified_types


def test_verified_type_skips_regate_on_repeat() -> None:
    """Once a type is verified, subsequent same-type elements skip the
    getEleTags round-trip (fast path)."""
    fake = _FakeOps(unknown=set())
    e = _live_emitter(fake)
    e.element("LadrunoUP", 1, 1, 2, 3, 4)
    # A second LadrunoUP with a tag the fake would 'drop' if re-gated: since
    # the type is already verified, no getEleTags check runs, so no raise.
    fake._unknown.add("LadrunoUP")   # pretend the build 'forgets' it
    e.element("LadrunoUP", 2, 5, 6, 7, 8)   # must not raise (fast path)
    assert fake.element_calls == ["LadrunoUP", "LadrunoUP"]
