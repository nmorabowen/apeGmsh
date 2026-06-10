"""ADR 0055 P2.3 — staged replay to tcl/py (`_replay_staged_into`).

`OpenSeesModel.build("tcl"|"py")` now re-emits a staged archive's deck
(was fail-loud). The completeness proof is **deck-equality**: the deck
the bridge emits directly (`ops.tcl`) must equal the deck the archive
replays (`from_h5(...).build("tcl")`).

- For fixtures with NO freshly-allocated tags (no initial_stress /
  activate_absorbing / embedded), the decks are **exactly** equal
  (every node/element/region/pattern tag is stored and replayed
  verbatim — only parameter / absorbing-flip / embedded ele_tags
  diverge across round-trip, INV-5).
- For an initial-stress fixture, the only divergence is the freshly
  allocated `parameter` tag numbers; normalize those and the decks
  are equal. This also exercises the shared TagAllocator + per-stage
  hook-wrap.

Plus structural assertions (stage delimiters, owned topology inside
the block, HOLD-before-domainChange) and the build-target guards.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.h5.test_h5_stages_reader import (
    _chain,
    _real_kitchen_sink_bridge,
    _real_two_stage_bridge,
    build_two_quad_fem,
)


# ---------------------------------------------------------------------------
# Fixtures with NO freshly-allocated tags (exact deck-equality)
# ---------------------------------------------------------------------------


def _no_param_two_stage_bridge() -> apeSees:
    """2 stages, quad topology, stage fix + load pattern + recorder +
    chain + analyze. NO initial_stress / absorbing / embedded → every
    emitted tag is stored, so bridge deck == archive deck exactly."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))

    with ops.stage(name="construction") as s:
        s.activate(pgs=["Fill"])
        s.fix(pg="FillTop", dofs=(1, 1))
        s.analysis(**_chain(ops))
        s.run(n_increments=5)
    with ops.stage(name="loading") as s:
        ts = ops.timeSeries.Linear()
        with s.pattern(series=ts) as p:
            p.load(pg="Fill", forces=(10.0, 0.0))
        s.analysis(**_chain(ops))
        s.run(n_increments=3, dt=0.01)
    return ops


def _bridge_deck(ops: apeSees, tmp_path: Path) -> str:
    p = tmp_path / "bridge.tcl"
    ops.tcl(str(p))
    return p.read_text(encoding="utf-8")


def _archive_deck(ops: apeSees, tmp_path: Path) -> str:
    h5 = tmp_path / "model.h5"
    ops.h5(str(h5))
    return OpenSeesModel.from_h5(str(h5)).build("tcl")


def _canon_numbers(line: str) -> str:
    """Canonicalize numeric tokens to ``float`` form so the oracle
    compares VALUES, not int-vs-float formatting.

    The bridge emits a material/element param's original type
    (``1000000``); the archive re-emits it from a stored ``float64``
    (``1000000.0``). OpenSees/Tcl accept both (the one place it does
    NOT — analysis-chain ints via ``OPS_GetIntInput`` — is fixed by
    ``compose._int_recover``). Normalizing both streams to ``float()``
    keeps value differences visible (``50`` vs ``49``) while ignoring
    ``50`` vs ``50.0``."""
    # Split on tcl (space) AND py (comma / paren) delimiters uniformly,
    # so a numeric token canonicalizes whether it sits in
    # ``element quad 1 ...`` or ``ops.element('quad', 1, ...)``. Quoted
    # names with digits (``'Steel02'``) stay whole → float() fails →
    # kept verbatim.
    normalized = line.replace(",", " ").replace("(", " ").replace(")", " ")
    out = []
    for tok in normalized.split():
        try:
            out.append(repr(float(tok)))
        except ValueError:
            out.append(tok)
    return " ".join(out)


def _strip_comments(deck: str) -> list[str]:
    """Drop blank lines and pure-comment lines (timestamps / banners)
    so structural comparison is robust to header noise."""
    out = []
    for ln in deck.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(_canon_numbers(s))
    return out


# ---------------------------------------------------------------------------
# 1. Exact deck-equality — the completeness proof (no divergent tags)
# ---------------------------------------------------------------------------


def test_staged_deck_equality_no_divergent_tags(tmp_path: Path) -> None:
    ops = _no_param_two_stage_bridge()
    bridge = _strip_comments(_bridge_deck(ops, tmp_path))

    # Re-build a fresh identical bridge for the archive (ops.tcl and
    # ops.h5 on the SAME instance is fine, but keep them independent to
    # avoid any emit-once state coupling).
    ops2 = _no_param_two_stage_bridge()
    archive = _strip_comments(_archive_deck(ops2, tmp_path))

    assert archive == bridge


# ---------------------------------------------------------------------------
# 2. initial-stress fixture — equal after normalizing parameter tags
# ---------------------------------------------------------------------------


def _norm_param_tags(lines: list[str]) -> list[str]:
    """Canonicalize freshly-allocated parameter tags (initial_stress /
    activate_absorbing) so two decks that differ ONLY in those tag
    values compare equal. Replaces the tag in parameter/updateParameter/
    addToParameter/remove-parameter lines with a sequential alias."""
    alias: dict[str, str] = {}

    def sub(m: "re.Match[str]") -> str:
        raw = m.group(2)
        if raw not in alias:
            alias[raw] = f"P{len(alias)}"
        return m.group(1) + alias[raw]

    pat = re.compile(
        r"\b(parameter |updateParameter |addToParameter |"
        r"remove parameter )(\d+)"
    )
    return [pat.sub(sub, ln) for ln in lines]


def test_staged_deck_equality_with_initial_stress(tmp_path: Path) -> None:
    bridge = _norm_param_tags(_strip_comments(
        _bridge_deck(_real_two_stage_bridge(), tmp_path),
    ))
    archive = _norm_param_tags(_strip_comments(
        _archive_deck(_real_two_stage_bridge(), tmp_path),
    ))
    assert archive == bridge


def test_staged_deck_equality_kitchen_sink(tmp_path: Path) -> None:
    """Kitchen-sink exercises HOLD/support, rayleigh, removals,
    set_creep/reset, activate_absorbing (param tag) — equal after
    parameter-tag normalization."""
    bridge = _norm_param_tags(_strip_comments(
        _bridge_deck(_real_kitchen_sink_bridge(), tmp_path),
    ))
    archive = _norm_param_tags(_strip_comments(
        _archive_deck(_real_kitchen_sink_bridge(), tmp_path),
    ))
    assert archive == bridge


# ---------------------------------------------------------------------------
# 3. Structural assertions on the replayed deck
# ---------------------------------------------------------------------------


def test_staged_deck_structure(tmp_path: Path) -> None:
    deck = OpenSeesModel.from_h5(
        str(_write_archive(_real_two_stage_bridge(), tmp_path)),
    ).build("tcl")
    lines = deck.splitlines()

    # Two analyze loops (one per stage) and two wipeAnalysis (stage_close).
    assert sum(1 for ln in lines if re.search(r"\banalyze\b", ln)) >= 2
    assert sum(1 for ln in lines if "wipeAnalysis" in ln) == 2
    # loadConst between stages (stage_close emits loadConst -time 0.0).
    assert any("loadConst" in ln for ln in lines)
    # Owned Fill nodes (5, 6) emit AFTER the first analyze is NOT true;
    # they emit inside the construction stage block, BEFORE its analyze.
    # Assert node 5 appears and the in-situ chain is present.
    assert any(re.search(r"\bnode 5\b", ln) for ln in lines)


def test_hold_sp_precedes_domainchange(tmp_path: Path) -> None:
    """ADR 0052 HOLD lines (sp ... -const) must emit BEFORE the stage's
    domainChange, not after the chain (gate-1 slot-split fix)."""
    deck = OpenSeesModel.from_h5(
        str(_write_archive(_real_kitchen_sink_bridge(), tmp_path)),
    ).build("tcl")
    lines = [ln.strip() for ln in deck.splitlines()]
    sp_const = next(i for i, ln in enumerate(lines) if "-const" in ln)
    domain_change = next(
        i for i, ln in enumerate(lines)
        if "domainChange" in ln or "DomainChange" in ln
    )
    assert sp_const < domain_change


def _write_archive(ops: apeSees, tmp_path: Path) -> Path:
    h5 = tmp_path / f"arch_{id(ops)}.h5"
    ops.h5(str(h5))
    return h5


# ---------------------------------------------------------------------------
# 4. Build-target guards
# ---------------------------------------------------------------------------


def test_staged_build_tcl_and_py_no_longer_raise(tmp_path: Path) -> None:
    m = OpenSeesModel.from_h5(
        str(_write_archive(_real_two_stage_bridge(), tmp_path)),
    )
    assert isinstance(m.build("tcl"), str)
    assert isinstance(m.build("py"), str)


def test_staged_build_live_fails_loud(tmp_path: Path) -> None:
    m = OpenSeesModel.from_h5(
        str(_write_archive(_real_two_stage_bridge(), tmp_path)),
    )
    with pytest.raises(NotImplementedError, match="live"):
        m.build("live")


# ---------------------------------------------------------------------------
# 5. py deck also round-trips (the other text target)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Stage MP-constraint replay order + names (gate-2 must-fix)
# ---------------------------------------------------------------------------


def _replay_stage_mp(stage_ro) -> "list[tuple]":
    """Replay one StageRecordRO through a RecordingEmitter and return
    the MP-related call stream (name comments + constraint calls)."""
    from apeGmsh.opensees._internal.compose import _replay_staged_into
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    rec = RecordingEmitter()
    _replay_staged_into(rec, stages=(stage_ro,), ndm=3, ndf=6)
    mp = {
        "mp_constraint_comment", "rigidLink", "equalDOF",
        "rigidDiaphragm", "embeddedNode",
    }
    return [(c[0], c[1]) for c in rec.calls if c[0] in mp]


def test_stage_mp_kinematic_equaldof_replays_after_rigid_diaphragm() -> None:
    """The bridge emits a kinematic_coupling's equalDOF AFTER
    rigidDiaphragm (emit_stage_mp_constraints step 5). The merge-sort
    by emit_index must reproduce that: rigidLink → equalDOF(genuine) →
    rigidDiaphragm → equalDOF(kinematic) → embeddedNode, each preceded
    by its name comment."""
    from apeGmsh.opensees._internal.typed_records import (
        EmbeddedNodeRecord,
        EqualDOFRecord,
        RigidDiaphragmRecord,
        RigidLinkRecord,
        StageRecordRO,
    )

    st = StageRecordRO(
        name="claim",
        analyze_steps=1,
        domain_changed=True,
        rigid_links=(RigidLinkRecord(
            kind="beam", master=1, slave=2, name="rl",
        ),),
        rigid_link_seq=(1,),
        # genuine equalDOF (seq 2, before diaphragm) + kinematic
        # equalDOF (seq 4, AFTER diaphragm).
        equal_dofs=(
            EqualDOFRecord(master=1, slave=3, dofs=(1, 2), name="eq"),
            EqualDOFRecord(master=1, slave=4, dofs=(3,), name="kin"),
        ),
        equal_dof_seq=(2, 4),
        rigid_diaphragms=(RigidDiaphragmRecord(
            perp_dir=3, master=1, slaves=(5, 6), name="rd",
        ),),
        rigid_diaphragm_seq=(3,),
        embedded_nodes=(EmbeddedNodeRecord(
            ele_tag=100, cnode=7, args=(8, 9, 10),
            stiffness=1e18, stiffness_p=None,
            rotational=False, pressure=False, name="emb",
        ),),
        embedded_node_seq=(5,),
    )
    stream = _replay_stage_mp(st)
    # Drop the name comments to assert the constraint-call order.
    calls = [c for c in stream if c[0] != "mp_constraint_comment"]
    kinds = [c[0] for c in calls]
    assert kinds == [
        "rigidLink", "equalDOF", "rigidDiaphragm", "equalDOF", "embeddedNode",
    ]
    # The kinematic equalDOF (slave 4) is the one AFTER rigidDiaphragm.
    assert calls[1][1][1] == 3      # genuine equalDOF slave
    assert calls[3][1][1] == 4      # kinematic equalDOF slave
    # Every constraint is preceded by its name comment.
    names = [c[1][0] for c in stream if c[0] == "mp_constraint_comment"]
    assert names == ["rl", "eq", "rd", "kin", "emb"]


def test_stage_mp_fallback_fixed_order_without_seq() -> None:
    """Pre-P2.3 archives carry no emit_index → fixed kind order
    (rigid_links → equal_dofs → rigid_diaphragms → embedded_nodes)."""
    from apeGmsh.opensees._internal.typed_records import (
        EqualDOFRecord,
        RigidDiaphragmRecord,
        StageRecordRO,
    )

    st = StageRecordRO(
        name="legacy",
        analyze_steps=1,
        domain_changed=True,
        equal_dofs=(EqualDOFRecord(master=1, slave=3, dofs=(1,), name=""),),
        rigid_diaphragms=(RigidDiaphragmRecord(
            perp_dir=3, master=1, slaves=(5,), name="",
        ),),
        # NO *_seq → fallback.
    )
    kinds = [c[0] for c in _replay_stage_mp(st)]
    assert kinds == ["equalDOF", "rigidDiaphragm"]


def test_staged_py_deck_equality_no_divergent_tags(tmp_path: Path) -> None:
    def py_bridge(ops: apeSees) -> list[str]:
        p = tmp_path / f"b_{id(ops)}.py"
        ops.py(str(p))
        return _strip_comments(p.read_text(encoding="utf-8"))

    ops = _no_param_two_stage_bridge()
    bridge = py_bridge(ops)
    ops2 = _no_param_two_stage_bridge()
    h5 = tmp_path / "m.h5"
    ops2.h5(str(h5))
    archive = _strip_comments(OpenSeesModel.from_h5(str(h5)).build("py"))
    assert archive == bridge
