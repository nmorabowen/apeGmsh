"""Build the schema corpus: one real file per schema minor, from git's frozen writers.

ADR 0113 D8. A zone's floor stands only where a real file from every minor
between the floor and the current minor opens through today's reader. The
old writers live in git, so this script checks each era out in a temporary
``git worktree`` and runs an era-stable generator under
``PYTHONPATH=<worktree>/src``:

* neutral zone, from ``NEUTRAL_SCHEMA_FLOOR``: the deterministic box of
  ``tests/fixtures/neutral_zone/_generate_fixtures.py`` via ``FEMData.to_h5``;
* opensees zone, from ``SCHEMA_FLOOR``: a one-bay elastic frame through
  ``apeSees(fem).h5()``.

The generator (``tests/fixtures/schema_corpus/_era_generator.py``) then
opens the file with the era's **own** reader and stores the semantic dump
(``_semantic_dump.py``) beside it, plus the era's own ``build("tcl")`` deck
for opensees files. ``MANIFEST.json`` names every minor from floor to
current with its commit, status ``ok`` or ``gap``, and the reason for a gap.
A gap is recorded, never filled with a file from another era (ADR 0113 D3).

**Which commit is a minor's era.** The minors come from the bump commits
(``git log --first-parent -G'^NEUTRAL_SCHEMA_VERSION' -- <writer>``, and
``-G'^SCHEMA_VERSION'`` for the opensees writer). A minor's file is written
by the *last* first-parent commit at that minor: the parent of the next
minor's bump commit, or the base commit for the current minor. So patch
bumps fold into their minor (2.26.1 writes the 2.26 file), and each file is
what that minor's writer looked like when it was last current. A minor no
first-parent commit ever stamped (a squash that bumped twice) is a gap.

Usage, from the repository root (needs gmsh, h5py and the project deps)::

    python scripts/build_schema_corpus.py --list            # the era plan
    python scripts/build_schema_corpus.py                   # build everything
    python scripts/build_schema_corpus.py --zone neutral --minor 2.10

**Variants.** The ADR 0113 D4 shim ledger names two semantic changes; each
has one real file from the last era *before* it, built by the generator's
``--variant`` at the era :data:`VARIANTS` fixes (the same frozen-writer
method, one more model). They sit beside the plain entries with a
``variant`` field and are not counted toward the one-file-per-minor rule.

**Below the floor.** An era whose file no reader opens is a ``below_floor``
gap: its file stays committed as evidence of the refusal, and a rebuild
keeps re-running it (the plan starts at the lowest minor already in the
manifest), so raising a floor never drops the files that justified it.

Frozen writers give frozen files: the corpus is committed and never rebuilt
in CI. A schema bump adds the outgoing minor's file by re-running this
script (or ``--zone Z --minor M`` for that one era) and committing the
result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests" / "fixtures" / "schema_corpus"
GENERATOR = CORPUS / "_era_generator.py"
MANIFEST = CORPUS / "MANIFEST.json"

#: zone -> (writer module, version constant, floor constant)
ZONES: dict[str, tuple[str, str, str]] = {
    "neutral": ("src/apeGmsh/mesh/_femdata_h5_io.py",
                "NEUTRAL_SCHEMA_VERSION", "NEUTRAL_SCHEMA_FLOOR"),
    "opensees": ("src/apeGmsh/opensees/emitter/h5.py",
                 "SCHEMA_VERSION", "SCHEMA_FLOOR"),
}

#: the /meta attrs that carry each zone's stamp (ADR 0023)
META_KEYS = {"neutral": "neutral_schema_version",
             "opensees": "opensees_schema_version"}

#: variant -> (zone, era minor, why). One real file per shim-ledger entry
#: (ADR 0113 D4), written by the last era before the semantic change.
VARIANTS: dict[str, tuple[str, str, str]] = {
    "sp_cases": (
        "neutral", "2.25",
        "prescribed displacements under two g.displacements.case names, "
        "written before 2.26.1 split /loads/sp/default into one group per "
        "case: the ledger says it reads as one `default` case (Q5)",
    ),
    "frame2d": (
        "opensees", "2.20",
        "the portal frame declared ops.model(ndm=2, ndf=3), written before "
        "neutral 2.34.0 made /meta/ndm the spatial dimension (#1291): the "
        "stamp is the mesh dimension 1, the file says 2-D through its "
        "(1, 0) vecxz and /meta/ndf = 3, and read_spatial_ndm (#1300) must "
        "recover 2 from that evidence (#1358)",
    ),
}

ERA_TIMEOUT_S = 900


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def _const(src: str, name: str) -> str | None:
    m = re.search(rf'^{name}\b[^=\n]*=\s*"([^"]+)"', src, re.M)
    return m.group(1) if m else None


def _version_at(sha: str, zone: str) -> str:
    path, const, _ = ZONES[zone]
    v = _const(_git("show", f"{sha}:{path}"), const)
    if v is None:
        raise RuntimeError(f"{sha[:10]}: {const} not found in {path}")
    return v


def tree_constant(zone: str, name: str) -> str:
    """A floor or version constant as the **working tree** holds it.

    The floors and current minors describe today's reader, the claim the
    corpus proves; the era history comes from ``base``, a commit on main.
    Reading them from the tree lets a PR that moves a floor build its
    corpus against a main ``base`` whose source still holds the old one.
    """
    path = ZONES[zone][0]
    v = _const((ROOT / path).read_text(encoding="utf-8"), name)
    if v is None:
        raise RuntimeError(f"{name} not found in the working tree's {path}")
    return v


def _minor(v: str) -> tuple[int, int]:
    major, minor, _patch = (int(p) for p in v.split("."))
    return major, minor


def _mstr(m: tuple[int, int]) -> str:
    return f"{m[0]}.{m[1]}"


@dataclass
class Era:
    zone: str
    minor: tuple[int, int]
    sha: str | None = None
    version: str | None = None
    gap_reason: str | None = None
    bumps: list[str] = field(default_factory=list)
    variant: str | None = None

    @property
    def stem(self) -> str:
        stem = f"{self.zone}_{_mstr(self.minor)}"
        return f"{stem}_{self.variant}" if self.variant else stem

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.zone, _mstr(self.minor), self.variant or "")


def check_current_is_written(
    zone: str, current: tuple[int, int], bumps: list[tuple[str, str]],
) -> None:
    """Refuse a current minor that no commit on ``base`` stamps.

    The current minor comes from the working tree and the history from
    ``base``. A bump PR run against a ``base`` that predates its bump
    commit would otherwise record its own minor as an ``unwritten`` gap,
    which INV-6 accepts, and land without the outgoing minor's file.
    """
    if any(_minor(v) == current for _sha, v in bumps):
        return
    raise RuntimeError(
        f"{zone}: the working tree is at {_mstr(current)}.x but no first-parent "
        f"commit on the base stamps it, so no writer of the current minor can be "
        f"checked out. Commit the bump and pass --base HEAD (the current minor's "
        f"file is written by the base commit); never record it as a gap"
    )


def plan(zone: str, base: str, *, start_minor: int | None = None) -> list[Era]:
    """Every minor from the zone's floor to its current minor, with its era commit.

    ``start_minor`` lower than the floor extends the plan downward: the
    builder passes the lowest minor the manifest already holds, so a
    below-floor evidence era is rebuilt rather than dropped.
    """
    path, const, floor_const = ZONES[zone]
    floor = _minor(tree_constant(zone, floor_const))
    current = _minor(tree_constant(zone, const))
    log = _git("log", "--first-parent", "--reverse", f"-G^{const}",
               "--format=%H", base, "--", path).split()
    bumps = [(sha, _version_at(sha, zone)) for sha in log]
    check_current_is_written(zone, current, bumps)

    first = floor[1] if start_minor is None else min(floor[1], start_minor)
    eras: list[Era] = []
    for minor in range(first, current[1] + 1):
        m = (floor[0], minor)
        at = [sha for sha, v in bumps if _minor(v) == m]
        era = Era(zone, m, bumps=at)
        if not at:
            prev = [(s, v) for s, v in bumps if _minor(v) < m]
            nxt = [(s, v) for s, v in bumps if _minor(v) > m]
            jump = (f"{nxt[0][0][:10]} bumped {prev[-1][1] if prev else '?'}"
                    f" -> {nxt[0][1]}") if nxt else "no later bump"
            era.gap_reason = (
                f"no first-parent commit on main ever stamped {_mstr(m)}.x "
                f"({jump}), so no writer of this minor exists to run"
            )
            eras.append(era)
            continue
        later = [sha for sha, v in bumps if _minor(v) > m]
        era.sha = _git("rev-parse", f"{later[0]}^1").strip() if later else base
        era.version = _version_at(era.sha, zone)
        if _minor(era.version) != m:
            raise RuntimeError(
                f"{zone} {_mstr(m)}: era commit {era.sha[:10]} stamps "
                f"{era.version}; the first-parent history is not linear here"
            )
        eras.append(era)
    return eras


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _stamps(h5_path: Path) -> dict[str, str]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        attrs = f["meta"].attrs
        return {zone: str(attrs[key]) for zone, key in META_KEYS.items()
                if key in attrs}


def _tail(text: str, era_root: str, n: int = 12) -> list[str]:
    lines = [ln.rstrip() for ln in text.replace(era_root, "<era>").splitlines()
             if ln.strip()]
    return lines[-n:]


def variant_era(variant: str, eras: list[Era]) -> Era:
    """The variant's era: the planned entry at the minor :data:`VARIANTS` fixes."""
    zone, minor, _why = VARIANTS[variant]
    for era in eras:
        if era.zone == zone and _mstr(era.minor) == minor:
            if era.sha is None:
                raise RuntimeError(f"variant {variant}: era {zone} {minor} is a gap")
            return Era(zone, era.minor, sha=era.sha, version=era.version,
                       bumps=era.bumps, variant=variant)
    raise RuntimeError(f"variant {variant}: {zone} {minor} is not in the plan")


def build_era(era: Era, python: str) -> dict:
    entry: dict = {"zone": era.zone, "minor": _mstr(era.minor)}
    if era.variant:
        entry.update(variant=era.variant, why=VARIANTS[era.variant][2])
    if era.gap_reason:
        entry.update(status="gap", reason=era.gap_reason)
        return entry
    assert era.sha is not None
    generator = f"tests/fixtures/schema_corpus/_era_generator.py --zone {era.zone}"
    if era.variant:
        generator += f" --variant {era.variant}"
    entry.update(
        sha=era.sha,
        date=_git("log", "-1", "--format=%cs", era.sha).strip(),
        version=era.version,
        bump_commits=era.bumps,
        generator=generator,
    )
    names = {"h5": f"{era.stem}.h5", "dump": f"{era.stem}.dump.json"}
    if era.zone == "opensees":
        names["tcl"] = f"{era.stem}.tcl"
    for name in names.values():
        (CORPUS / name).unlink(missing_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="apegmsh_corpus_"))
    wt = tmp / "era"
    stage = tmp / "out"
    stage.mkdir()
    try:
        _git("worktree", "add", "--detach", str(wt), era.sha)
        cmd = [python, str(GENERATOR), "--zone", era.zone,
               "--out", str(stage / names["h5"]),
               "--dump", str(stage / names["dump"]),
               "--expect-src", str(wt / "src")]
        if era.variant:
            cmd += ["--variant", era.variant]
        if "tcl" in names:
            cmd += ["--tcl", str(stage / names["tcl"])]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(wt / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True,
                                  text=True, timeout=ERA_TIMEOUT_S)
            failed = proc.returncode != 0
            out = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as exc:
            failed, out = True, f"timeout after {ERA_TIMEOUT_S}s: {exc}"
        if failed:
            tail = _tail(out, str(wt))
            entry.update(status="gap",
                         reason=f"the era's generator did not run: {tail[-1] if tail else '?'}",
                         stderr_tail=tail)
            return entry
        files = {}
        for kind, name in names.items():
            src = stage / name
            if not src.exists():
                raise RuntimeError(f"{era.stem}: generator exited 0 without {name}")
            shutil.copyfile(src, CORPUS / name)
            files[kind] = {"name": name, "sha256": _sha256(CORPUS / name)}
        entry.update(status="ok", files=files, stamps=_stamps(CORPUS / names["h5"]))
        return entry
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force",
                        str(wt)], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


def classify(entry: dict, floors: dict[str, str]) -> dict:
    """Name each entry's gap kind; idempotent.

    * ``unwritten``: no first-parent commit ever stamped the minor.
    * ``unrunnable``: a writer exists but the era-stable generator failed.
    * ``below_floor``: the era wrote a real file, but one of its zone stamps
      sits below that zone's floor, so today's readers refuse it by design.
      The file stays committed as evidence of the refusal.
    """
    if "files" not in entry:
        if entry["status"] == "gap":
            entry["gap_kind"] = "unrunnable" if "sha" in entry else "unwritten"
        return entry
    low = [
        f"{zone} {stamp} is below the {zone} floor {floors[zone]}"
        for zone, stamp in sorted(entry["stamps"].items())
        if zone in floors and _minor(stamp) < _minor(floors[zone])
    ]
    if low:
        entry.update(
            status="gap", gap_kind="below_floor",
            reason=("the era's writer ran, but " + "; ".join(low)
                    + ": today's readers refuse every file this era wrote"),
        )
    else:
        entry["status"] = "ok"
        entry.pop("gap_kind", None)
        entry.pop("reason", None)
    return entry


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zone", choices=(*ZONES, "all"), default="all")
    ap.add_argument("--minor", help="build only this minor, e.g. 2.10")
    ap.add_argument("--variant", choices=tuple(VARIANTS),
                    help="build only this variant (its zone and era are fixed)")
    ap.add_argument("--base", default=None,
                    help="the main commit the era history is read from, and the "
                         "writer of the current minors (default: merge-base of "
                         "HEAD and origin/main); floors and current minors come "
                         "from the working tree")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    ap.add_argument("--manifest-only", action="store_true",
                    help="re-classify the existing entries and rewrite MANIFEST.json")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)

    base = a.base or _git("merge-base", "HEAD", "origin/main").strip()
    base = _git("rev-parse", base).strip()
    zones = list(ZONES) if a.zone == "all" else [a.zone]

    manifest = _load_manifest()
    entries = {_key(e): e for e in manifest.get("entries", [])}
    floors, currents = {}, {}
    for zone, (_path, const, floor_const) in ZONES.items():
        floors[zone] = tree_constant(zone, floor_const)
        currents[zone] = tree_constant(zone, const)
    if a.manifest_only:
        _write_manifest(base, floors, currents, entries)
        return 0
    for zone in zones:
        held = [_minor(e["minor"] + ".0")[1] for e in entries.values()
                if e["zone"] == zone and "variant" not in e]
        eras = plan(zone, base, start_minor=min(held) if held else None)
        todo = [] if a.variant else [
            era for era in eras if not a.minor or _mstr(era.minor) == a.minor
        ]
        todo += [
            variant_era(name, eras) for name, (vzone, _m, _why) in VARIANTS.items()
            if vzone == zone and not a.minor and (a.variant in (None, name))
        ]
        for era in todo:
            if a.list:
                where = era.sha[:10] if era.sha else "-"
                print(f"{zone:9s} {_mstr(era.minor):6s} {era.variant or '':9s} "
                      f"{where:10s} {era.version or ''} {era.gap_reason or ''}")
                continue
            print(f"building {era.stem} ...", flush=True)
            e = classify(build_era(era, a.python), floors)
            print(f"  {e['status']}" + (f": {e['reason']}" if e["status"] == "gap" else ""),
                  flush=True)
            entries[era.key] = e
            _write_manifest(base, floors, currents, entries)
    return 0


def _key(entry: dict) -> tuple[str, str, str]:
    return (entry["zone"], entry["minor"], entry.get("variant", ""))


def _write_manifest(base: str, floors: dict, currents: dict, entries: dict) -> None:
    out = {
        "about": ("ADR 0113 D8 schema corpus. One entry per minor from each zone's "
                  "floor to its current minor; status 'gap' entries carry the reason "
                  "no real file exists. Built by scripts/build_schema_corpus.py."),
        "base": base,
        "floors": floors,
        "current": currents,
        "entries": sorted((classify(e, floors) for e in entries.values()),
                          key=lambda e: (e["zone"], _minor(e["minor"] + ".0"),
                                         e.get("variant", ""))),
    }
    MANIFEST.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
