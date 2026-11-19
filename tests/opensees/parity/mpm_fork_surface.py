"""The fork's ``mpm`` command surface at ``FORK_PIN``, and what apeGmsh does with it.

The fork's parser (``OPS_LadrunoMPM.cpp``, function ``OPS_LadrunoMPM``)
dispatches on ``sub == "<name>"``. ``scripts/refresh_mpm_fork_surface.py``
reads that dispatch at ``FORK_PIN``'s ``sha`` and writes the sorted names, with
the sha and tag they were read at, to ``tests/fixtures/mpm_fork_surface/
surface.json``. The suite lane reads that snapshot offline.

Every name in it must be one of two things:

* the subcommand an :mod:`apeGmsh.opensees.mpm` member emits
  (``test_mpm_command_parity.py`` derives these by emitting each member); or
* a row of :data:`NOT_OFFERED` below, with the reason apeGmsh has no member.

A pin bump moves ``FORK_PIN`` and so fails :func:`pin_problems` until the
snapshot is refreshed; the refreshed snapshot then fails
:func:`surface_problems` by name for every new subcommand, until the bump
adds a member or a reasoned row.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SNAPSHOT = REPO / "tests" / "fixtures" / "mpm_fork_surface" / "surface.json"
FORK_PIN = REPO / "FORK_PIN"
REFRESH = "python scripts/refresh_mpm_fork_surface.py"
#: The keys a snapshot carries, and no others.
SNAPSHOT_KEYS = frozenset({"fork_tag", "fork_sha", "source", "function",
                           "refreshed_by", "subcommands"})

_SET = ("the particle set (ADR 0134 Contract 1) carries it per particle; "
        "apeGmsh never writes per-particle commands into a deck")
_MODE_A = ("mode A (implicit / quasiStatic, fork WP-232f) only; apeGmsh emits "
           "the explicit engine (mode F) and has no `mode` member")
_QUERY = ("a query that answers a value inside a running deck; apeGmsh "
          "declares the run and reads results from the .ladruno file "
          "(Results.particles, ADR 0134 D3)")

#: Fork ``mpm`` subcommands with no :mod:`apeGmsh.opensees.mpm` member, each
#: with the reason. A new fork subcommand gets a member or a row here.
NOT_OFFERED: dict[str, str] = {
    "particles": ("offered, but never as a member or a row: each deck "
                  "emitter writes the anchored `mpm particles -file` line "
                  "itself, right after `mpm grid`, from the particle set it "
                  "seeds beside the deck (ADR 0134 D1)"),
    "particle": f"one explicit particle: {_SET}",
    "velocity": f"a body's initial velocity: {_SET}",
    "stress": f"a body's initial stress: {_SET}",
    "particleVelocity": f"every particle's initial velocity as one flat list: {_SET}",
    "particleBodyForce": ("a per-particle body force reset between `mpm analyze` "
                          "calls (fork WP-232i, manufactured solutions): a "
                          "per-step driver loop, not a declaration"),
    "shift": ("translates a seeded body before the first step; apeGmsh "
              "seeds bodies from their physical groups where they stand"),
    "mode": ("selects mode A (implicit / quasiStatic, fork WP-232f); apeGmsh "
             "emits the explicit engine (mode F), the fork's default"),
    "pforce": ("mode A only, refused in mode F since fork WP-232k (#1109); "
               "apeGmsh emits mode F"),
    "condForce": ("mode A only, refused in mode F (a gridvel reaction query); "
                  "apeGmsh emits mode F"),
    "cut": f"the step-cut retry policy: {_MODE_A}",
    "emptyNodes": f"the empty-support policy: {_MODE_A}",
    "tagBase": f"the particle element tag bases: {_MODE_A}",
    "rayleigh": f"Rayleigh damping on the particle elements: {_MODE_A}",
    "gridvel": ("prescribed grid velocity (box / affine / translate / "
                "profile, setAffine; fork WP-232f, WP-232k): not adopted, a "
                "follow-up with no ADR decision yet"),
    "stage": ("the soil-model stage walk (fork WP-232k); a staged emit of a "
              "model with mpm declarations is refused (ADR 0134 Amendment 1), "
              "so there is no stage to walk"),
    "antiLocking": ("the cell-centre F-bar (fork WP-232j): not adopted, a "
                    "follow-up with no ADR decision yet"),
    "reset": ("rewinds particles and materials to before the first accepted "
              "step (a deck-driver control for retries); an apeGmsh deck runs "
              "one analysis from its declarations"),
    "criticalTimeStep": f"Δt_c: {_QUERY}",
    "particleData": f"a per-particle field list: {_QUERY}",
    "particleResponse": f"a material response per particle (WP-232k): {_QUERY}",
    "info": f"run counters and summaries: {_QUERY}",
    "debug": "the fork's unit-test hooks (shape functions, refusal injection)",
}


def read_pin(path: Path = FORK_PIN) -> dict[str, str]:
    """``FORK_PIN``'s ``key=value`` lines; ``#`` lines are comments."""
    pin: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or not line.strip():
            continue
        key, sep, val = line.partition("=")
        if not sep:
            raise ValueError(f"{path}: not a key=value line: {line!r}")
        pin[key.strip()] = val.strip()
    for key in ("tag", "sha"):
        if not pin.get(key):
            raise ValueError(f"{path} has no {key}= line")
    return pin


def load_snapshot(path: Path = SNAPSHOT) -> dict[str, Any]:
    """The committed snapshot, refused unless its shape is exactly the one
    the refresh script writes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != SNAPSHOT_KEYS:
        got = sorted(data) if isinstance(data, dict) else type(data).__name__
        raise ValueError(f"{path}: keys {got}, expected {sorted(SNAPSHOT_KEYS)}; "
                         f"regenerate it with `{REFRESH}`")
    subs = data["subcommands"]
    if (not isinstance(subs, list) or not subs
            or not all(isinstance(s, str) and s for s in subs)
            or subs != sorted(set(subs))):
        raise ValueError(f"{path}: `subcommands` must be a non-empty sorted list "
                         f"of distinct names; regenerate it with `{REFRESH}`")
    return data


def pin_problems(snapshot: Mapping[str, Any], pin: Mapping[str, str]) -> list[str]:
    """Each way the snapshot was not read at ``FORK_PIN``."""
    out = []
    for key in ("sha", "tag"):
        have, want = snapshot[f"fork_{key}"], pin[key]
        if have != want:
            out.append(f"the mpm fork-surface snapshot was read at {key} {have}, "
                       f"FORK_PIN names {want}: refresh it with `{REFRESH}`")
    return out


def surface_problems(
    fork: Iterable[str], members: Iterable[str], not_offered: Mapping[str, str],
) -> list[str]:
    """Each fork subcommand with no member and no row, each stale row, each
    member the fork does not accept, and each name that is both."""
    fork_set, member_set, rows = set(fork), set(members), set(not_offered)
    out = [f"fork mpm subcommand {s!r} is neither an apeGmsh.opensees.mpm "
           f"member nor a NOT_OFFERED row: add the member, or a row that says "
           f"why apeGmsh does not offer it"
           for s in sorted(fork_set - member_set - rows)]
    out += [f"NOT_OFFERED row {s!r} is stale: the fork's parser at the "
            f"snapshot has no such subcommand" for s in sorted(rows - fork_set)]
    out += [f"apeGmsh.opensees.mpm emits subcommand {s!r}, which the fork's "
            f"parser at the snapshot does not accept"
            for s in sorted(member_set - fork_set)]
    out += [f"{s!r} is both a member's subcommand and a NOT_OFFERED row"
            for s in sorted(member_set & rows)]
    out += [f"NOT_OFFERED row {s!r} gives no reason"
            for s in sorted(rows) if not str(not_offered[s]).strip()]
    return out
