#!/usr/bin/env python
"""Derive the installed skill copies from the canonical apeGmsh skill.

Single source of truth: ``skills/apegmsh/`` (SKILL.md + references/ + the
                        scripts/ ``*.py`` and ``*.json`` files).
Derived copy:           ``.claude/skills/apegmsh-helper/`` (what the Claude Code
                        harness actually loads in this repo).

The derived copy is a byte-for-byte mirror of the canonical *body* and
*references*; only the SKILL.md YAML front-matter (``name`` + ``description``)
is rewritten, because the installed skill triggers under a different name and
carries a trigger-tuned description.

Usage
-----
    python scripts/sync_skill.py            # write the derived copy + API index
    python scripts/sync_skill.py --check    # exit 1 if the derived copy is stale
                                            # Skill mirror is stdlib (CI lock-tests
                                            # pre-install). Live index drift runs
                                            # when apeGmsh/gmsh import (ADR 0096 S2).

The published ``anthropic-skills:apegmsh-helper`` plugin lives in a *separate*
marketplace repo and is NOT touched here — syncing it is a downstream release
step (copy the same files, keep that plugin's own front-matter).

The USER-LEVEL copy (``~/.claude/skills/apegmsh``, loaded outside this repo)
is maintained by ``scripts/refresh_user_skill.py`` from origin/main — run it
after merging any skill PR. (It was a junction into this working tree until
2026-07-18; see that script's docstring.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANONICAL = REPO / "skills" / "apegmsh"
DERIVED = REPO / ".claude" / "skills" / "apegmsh-helper"
SCRIPT_GLOBS = ("*.py", "*.json")       # skills/apegmsh/scripts/ files the mirror carries

# Front-matter for the DERIVED copy. The harness loads this skill under the
# name `apegmsh-helper` inside this repository, where most sessions change
# apeGmsh itself, so the description leads with what the skill is for and
# routes maintainer work to the AGENTS.md task guides. It is written as a YAML
# folded block (`>`), which holds any character, so a `: ` in the text cannot
# break the front matter the way the old plain scalar did.
DERIVED_NAME = "apegmsh-helper"
DERIVED_DESCRIPTION = (
    "How to use apeGmsh, the Gmsh wrapper for structural FEM: labelled geometry "
    "and meshes, a FEMData snapshot saved to model.h5, a typed OpenSees bridge "
    "(apeSees), and Results readers and viewers. Use it to write, run, debug or "
    "review a user model script that imports apeGmsh. "
    "TRIGGER: from apeGmsh import apeGmsh / Part / FEMData / Results; g.model, "
    "g.mesh, g.physical, labels and selections; loads, masses and constraints; "
    "Part and Assembly multi-part models; get_fem_data and model.h5; "
    "apeSees(fem), stages and the emitted Tcl or openseespy deck; reading "
    "OpenSees output with Results.from_*; the apeGmsh Studio CLI or MCP. "
    "SKIP: changing apeGmsh itself (its source, tests, ADRs, docs or this skill; "
    "use the AGENTS.md task guides: apegmsh-bridge-feature, "
    "apegmsh-viewer-results, apegmsh-adr-docs); the raw gmsh API or .geo files "
    "(gmsh-structural); OpenSees Tcl, openseespy or C++ work without apeGmsh "
    "(opensees-expert, opensees-ladruno); design-code checks."
)

#: The size cap of a skill description, the same one tests/test_skill_fences.py
#: enforces: a longer one is truncated or refused where the skill is uploaded.
DESCRIPTION_MAX = 1024


def folded_description(text: str, width: int = 78) -> str:
    """``text`` as a YAML ``description: >`` block, wrapped at ``width``.

    A folded block holds any character, and YAML folds its lines back with
    one space, so the loaded description equals ``text``. Raises when
    ``text`` is over ``DESCRIPTION_MAX`` characters or would not fold back
    to itself (a newline or a run of spaces).
    """
    if len(text) > DESCRIPTION_MAX:
        raise ValueError(
            f"the description is {len(text):,} characters, over {DESCRIPTION_MAX:,}: shorten it"
        )
    if text != " ".join(text.split()):
        raise ValueError("the description must be one line with single spaces")
    lines: list[str] = []
    line = ""
    for word in text.split(" "):
        if line and len(line) + 1 + len(word) > width - 2:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    lines.append(line)
    return "description: >\n" + "".join(f"  {chunk}\n" for chunk in lines)


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_block, body). Raises if no leading ``---`` block."""
    if not text.startswith("---"):
        raise ValueError("canonical SKILL.md has no YAML front-matter")
    end = text.index("\n---", 3)
    fm = text[: end + len("\n---")]
    body = text[end + len("\n---") :]
    return fm, body


def derived_skill_md() -> str:
    canonical = (CANONICAL / "SKILL.md").read_text(encoding="utf-8")
    _, body = split_frontmatter(canonical)
    fm = (
        "---\n"
        f"name: {DERIVED_NAME}\n"
        + folded_description(DERIVED_DESCRIPTION)
        + "---"
    )
    return fm + body


def planned_files() -> dict[Path, str]:
    """Map of derived-path -> intended content."""
    out: dict[Path, str] = {DERIVED / "SKILL.md": derived_skill_md()}
    ref_dir = CANONICAL / "references"
    if ref_dir.is_dir():
        for ref in sorted(ref_dir.glob("*.md")):
            out[DERIVED / "references" / ref.name] = ref.read_text(encoding="utf-8")
    script_dir = CANONICAL / "scripts"
    if script_dir.is_dir():
        # The lint and its data: R1 reads removed_symbols.json beside the lint.
        for pattern in SCRIPT_GLOBS:
            for script in sorted(script_dir.glob(pattern)):
                out[DERIVED / "scripts" / script.name] = script.read_text(encoding="utf-8")
    return out


def _check_index() -> int:
    """Live harvest vs committed JSON. Skip if apeGmsh/gmsh cannot import.

    lock-tests runs ``--check`` before pip install (stdlib skill mirror).
    Index freshness is gated after install via this same function and
    ``tests/studio/test_lookup.py``.
    """
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from apeGmsh.studio._index_build import committed_index_drift
    except ImportError:
        print("skip index drift (apeGmsh/gmsh not importable)")
        return 0
    idx = committed_index_drift()
    if idx:
        print(
            "API INDEX STALE — run `python -m apeGmsh.studio.lookup --build`:",
            file=sys.stderr,
        )
        for line in idx:
            print(f"  - {line}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify in-sync; non-zero exit if stale")
    args = ap.parse_args()

    if not (CANONICAL / "SKILL.md").exists():
        print(f"ERROR: canonical skill not found at {CANONICAL}", file=sys.stderr)
        return 2

    plan = planned_files()
    # Also flag derived reference files that no longer exist in canonical.
    existing_refs = set((DERIVED / "references").glob("*.md")) if (DERIVED / "references").is_dir() else set()
    existing_scripts = (
        {p for pattern in SCRIPT_GLOBS for p in (DERIVED / "scripts").glob(pattern)}
        if (DERIVED / "scripts").is_dir() else set()
    )
    stale = (existing_refs | existing_scripts) - set(plan)

    if args.check:
        drift = [p for p, c in plan.items() if not p.exists() or p.read_text(encoding="utf-8") != c]
        drift += list(stale)
        if drift:
            print("OUT OF SYNC — run `python scripts/sync_skill.py`:", file=sys.stderr)
            for p in drift:
                print(f"  - {p.relative_to(REPO)}", file=sys.stderr)
            return 1
        print("apegmsh-helper is in sync with the canonical skill.")
        return _check_index()

    for path, content in plan.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")
    for path in stale:
        path.unlink()
        print(f"removed stale {path.relative_to(REPO)}")
    print(f"\nDerived {len(plan)} file(s) into {DERIVED.relative_to(REPO)}.")

    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from apeGmsh.studio._index_build import write_index

    dest = write_index()
    print(f"wrote {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
