#!/usr/bin/env python
"""Derive the installed skill copies from the canonical apeGmsh skill.

Single source of truth: ``skills/apegmsh/`` (SKILL.md + references/).
Derived copy:           ``.claude/skills/apegmsh-helper/`` (what the Claude Code
                        harness actually loads in this repo).

The derived copy is a byte-for-byte mirror of the canonical *body* and
*references*; only the SKILL.md YAML front-matter (``name`` + ``description``)
is rewritten, because the installed skill triggers under a different name and
carries a trigger-tuned description.

Usage
-----
    python scripts/sync_skill.py            # write the derived copy
    python scripts/sync_skill.py --check    # exit 1 if the derived copy is stale
                                            # (use in CI to catch un-synced edits)

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

# Front-matter for the DERIVED copy. The harness loads this skill under the
# name `apegmsh-helper`; the description is trigger-tuned (kept verbatim from the
# installed copy, refreshed to name the v2.0 surfaces).
DERIVED_NAME = "apegmsh-helper"
DERIVED_DESCRIPTION = (
    "Use whenever the user is working with apeGmsh — the structural-FEM wrapper "
    "around Gmsh with OpenSees integration. Triggers on building FEM models from "
    "CAD/STEP imports, Part-based assembly workflows, composite-based "
    "geometry/mesh/constraint APIs (g.model, g.mesh, g.physical, g.constraints, "
    "etc.), the apeSees(fem) OpenSees bridge with typed primitives and automatic "
    "MP-constraint emission, staged analysis (ops.stage), loads/masses/constraints "
    "resolution into the FEMData broker, native model.h5 persistence "
    "(FEMData.to_h5/from_h5, save_to=/g.save(), apeGmsh.from_h5), model composition "
    "(g.compose), post-processing OpenSees output via Results "
    "(from_native/from_mpco/from_recorders) with the interactive and web viewers "
    "(results.viewer / results.show_web), and exporting models to OpenSees Tcl or "
    "openseespy scripts. Covers apeGmsh's own abstractions on top of Gmsh and "
    "OpenSees. For raw gmsh API questions see the gmsh-structural skill; for raw "
    "OpenSees analysis commands see opensees-expert; for FEM theory first "
    "principles see fem-mechanics-expert."
)


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
        f"description: {DERIVED_DESCRIPTION}\n"
        "---"
    )
    return fm + body


def planned_files() -> dict[Path, str]:
    """Map of derived-path -> intended content."""
    out: dict[Path, str] = {DERIVED / "SKILL.md": derived_skill_md()}
    ref_dir = CANONICAL / "references"
    if ref_dir.is_dir():
        for ref in sorted(ref_dir.glob("*.md")):
            out[DERIVED / "references" / ref.name] = ref.read_text(encoding="utf-8")
    return out


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
    stale = existing_refs - set(plan)

    if args.check:
        drift = [p for p, c in plan.items() if not p.exists() or p.read_text(encoding="utf-8") != c]
        drift += list(stale)
        if drift:
            print("OUT OF SYNC — run `python scripts/sync_skill.py`:", file=sys.stderr)
            for p in drift:
                print(f"  - {p.relative_to(REPO)}", file=sys.stderr)
            return 1
        print("apegmsh-helper is in sync with the canonical skill.")
        return 0

    for path, content in plan.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")
    for path in stale:
        path.unlink()
        print(f"removed stale {path.relative_to(REPO)}")
    print(f"\nDerived {len(plan)} file(s) into {DERIVED.relative_to(REPO)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
