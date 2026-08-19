"""Harvest SKILL.md catalogs with GitHub clones as the canonical home.

Policy (Studio projects):
  - Authoritative skills live under ``<github>/`` (= ~/Documents/Github/).
  - ``~/.claude/skills`` is legacy / mirror only (still scanned, demoted).
  - ``~/.cursor/skills-cursor`` stays local (Cursor product skills).

Skips: node_modules, .venv, venv, .git, and ``**/worktrees/**``.

    python APE/skills/harvest.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

HOME = Path.home()
GITHUB = HOME / "Documents" / "Github"
HERE = Path(__file__).resolve().parent

SKIP_DIR_NAMES = {
    "node_modules",
    ".venv",
    "venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    ".tox",
}

DOMAIN_ORDER = [
    "apeGmsh / meshing",
    "Contact / OpenSees contact",
    "OpenSees (expert / performance / explicit)",
    "FEM theory",
    "Steel design",
    "Concrete / RC",
    "Seismic (ASCE) / quake research",
    "Abaqus theory",
    "ETABS / Robot / Revit",
    "Kratos / AEM",
    "Post-process (STKO/MPCO)",
    "APE business (offers)",
    "Cursor product / agent ops",
    "Other",
]

# Studio recommended stack — access resolved after harvest when possible.
# Template ships this list EMPTY: any recommended skill stack is personal to
# whoever's SKILL.md trees this scans (not distributed with this template).
# Add your own rows here, e.g. ("1", "apegmsh", "why") — see APE/skills/README.md.
REC_SPECS: list[tuple[str, str, str]] = []


def _should_skip(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if parts & {n.lower() for n in SKIP_DIR_NAMES}:
        return True
    # git worktrees / cursor worktrees inside clones
    lowered = [p.lower() for p in path.parts]
    if "worktrees" in lowered:
        return True
    return False


def _portable(md: Path) -> tuple[str, str]:
    resolved = md.resolve()
    try:
        rel = resolved.relative_to(GITHUB)
        return "github", "<github>/" + rel.as_posix()
    except ValueError:
        pass
    try:
        rel = resolved.relative_to(HOME)
        bucket = "cursor-local" if ".cursor" in rel.parts[:3] else "claude-legacy"
        return bucket, "~/" + rel.as_posix()
    except ValueError:
        pass
    return "absolute", resolved.as_posix()


def parse_skill(md: Path) -> dict:
    text = md.read_text(encoding="utf-8", errors="replace")
    name = md.parent.name
    desc = ""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            fm = text[3:end]
            m = re.search(r"^name:\s*[\"']?([^\"'\n]+)", fm, re.M)
            if m:
                name = m.group(1).strip()
            m = re.search(
                r"^description:\s*>?\s*(.*?)(?=\n[a-zA-Z_][\w-]*:|\Z)",
                fm,
                re.S | re.M,
            )
            if m:
                desc = " ".join(
                    ln.strip().lstrip(">").strip()
                    for ln in m.group(1).splitlines()
                    if ln.strip() and not ln.strip().startswith("#")
                )
    if not desc:
        for line in text.splitlines():
            s = line.strip()
            if s and not s.startswith("#") and not s.startswith("<!--") and s != "---":
                desc = s[:500]
                break
    desc = desc.strip().strip('"').strip("'")
    if desc.startswith("- "):
        desc = desc[2:]
    bucket, access = _portable(md)
    return {
        "id": md.parent.name,
        "name": name,
        "description": desc[:500],
        "bucket": bucket,
        "access": access,
        "exists": md.is_file(),
    }


def domain(sid: str, desc: str) -> str:
    s = f"{sid} {desc}".lower()
    if sid in {
        "canvas", "create-rule", "create-skill", "create-hook",
        "create-subagent", "update-cursor-settings", "statusline", "loop",
        "automate", "autopilot", "sdk", "split-to-prs", "review-bugbot",
        "review-security", "review", "onboard", "rename-chat",
        "migrate-to-skills", "update-cli-config", "shell",
    }:
        return "Cursor product / agent ops"
    if "apegmsh" in sid or sid == "gmsh-structural":
        return "apeGmsh / meshing"
    if sid in {"opensees-contact"} or sid == "abaqus-theory-contact-loading":
        return "Contact / OpenSees contact"
    if sid in {"opensees-expert", "opensees-performance", "explicit-dynamics"}:
        return "OpenSees (expert / performance / explicit)"
    if sid == "fem-mechanics-expert":
        return "FEM theory"
    if any(x in sid for x in ("aisc", "aisi", "apesteel", "cold-formed")):
        return "Steel design"
    if any(x in sid for x in ("concrete", "aci", "prestress", "opensees-concrete")):
        return "Concrete / RC"
    if "asce" in sid or sid in {"seismic-detailing", "quake-research"}:
        return "Seismic (ASCE) / quake research"
    if "abaqus" in sid:
        return "Abaqus theory"
    if any(x in sid for x in ("etabs", "robot", "revit", "ape-drawing")):
        return "ETABS / Robot / Revit"
    if sid in {"kratos", "applied-element-method"}:
        return "Kratos / AEM"
    if "stko" in sid or "mpco" in s:
        return "Post-process (STKO/MPCO)"
    if "oferta" in sid or sid == "ape-server":
        return "APE business (offers)"
    return "Other"


def prefer_rows(rows: list[dict]) -> list[dict]:
    """One row per skill id. Prefer GitHub clones over ~/.claude mirrors."""
    rank = {
        "github": 0,
        "cursor-local": 1,
        "claude-legacy": 2,
        "absolute": 3,
    }

    def score(r: dict) -> tuple:
        b = rank.get(r["bucket"], 9)
        acc = r["access"]
        # Prefer dedicated *skills* repos and skills/ over nested .claude in libs
        dedicated = 0 if (
            "-skills" in acc.lower()
            or "/skills/" in acc.lower()
            or acc.lower().endswith("/skills/" + r["id"].lower() + "/skill.md")
        ) else 1
        # Prefer shorter / cleaner paths
        depth = acc.count("/")
        return (b, dedicated, depth, acc)

    best: dict[str, dict] = {}
    for r in rows:
        cur = best.get(r["id"])
        if cur is None or score(r) < score(cur):
            best[r["id"]] = r
    return sorted(best.values(), key=lambda x: x["id"].lower())


def iter_github_skills() -> list[Path]:
    if not GITHUB.is_dir():
        return []
    found: list[Path] = []
    for md in GITHUB.rglob("SKILL.md"):
        if _should_skip(md):
            continue
        found.append(md)
    return found


def iter_home_skills() -> list[Path]:
    roots = [
        HOME / ".claude" / "skills",
        HOME / ".cursor" / "skills-cursor",
    ]
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for md in root.rglob("SKILL.md"):
            if _should_skip(md):
                continue
            found.append(md)
    return found


def write_markdown(cat: dict, preferred: list[dict], dest: Path) -> None:
    by: dict[str, list[dict]] = defaultdict(list)
    for sk in preferred:
        by[domain(sk["id"], sk["description"])].append(sk)
    by_id = {s["id"]: s for s in preferred}

    n_gh = sum(1 for s in preferred if s["bucket"] == "github")
    n_legacy = sum(1 for s in preferred if s["bucket"] == "claude-legacy")
    n_cursor = sum(1 for s in preferred if s["bucket"] == "cursor-local")

    lines: list[str] = [
        "# Agent skills catalog",
        "",
        "Canonical skill home for Studio projects: **GitHub clones** under",
        "`<github>/` (= `~/Documents/Github/`). Personal `~/.claude/skills`",
        "copies are legacy mirrors (demoted if a GitHub copy exists).",
        "",
        "Recommended skill door for APE Studio habitats (FEM mechanics,",
        "wide generality). Trim or extend per habitat; keep GitHub-canonical paths.",
        "",
        "| Token | Expands to |",
        "|-------|------------|",
        "| `<github>/` | `~/Documents/Github/` |",
        "| `~/` | User home (legacy Claude / Cursor product skills) |",
        "",
        "## Localization policy",
        "",
        "1. Keep engineering skills in dedicated or library repos under `<github>/`.",
        "2. Agents **Read** `<github>/.../SKILL.md` — listing is not loaded context.",
        "3. Do not treat `~/.claude/skills` as source of truth once a GitHub copy exists.",
        "4. Cursor product skills may stay under `~/.cursor/skills-cursor/`.",
        "5. Harvest skips `node_modules`, `.venv`, and `**/worktrees/**`.",
        "",
        f"**This harvest:** {cat['count']} files; {len(preferred)} unique ids "
        f"(github={n_gh}, claude-legacy={n_legacy}, cursor-local={n_cursor}).",
        "Machine twin: `APE/skills/catalog.json`.",
        "",
        "Refresh:",
        "",
        "```text",
        "python APE/skills/harvest.py",
        "```",
        "",
    ]
    if REC_SPECS:
        lines += [
            "## Recommended first reads (template door)",
            "",
            "| Priority | Skill id | Portable access | Why |",
            "|----------|----------|-----------------|-----|",
        ]
        for pri, sid, why in REC_SPECS:
            acc = by_id.get(sid, {}).get("access", "(not harvested)")
            lines.append(f"| {pri} | `{sid}` | `{acc}` | {why} |")
        lines.append("")

    for dom in DOMAIN_ORDER:
        items = by.get(dom) or []
        if not items:
            continue
        lines.append(f"## {dom}")
        lines.append("")
        lines.append("| Id | Access (portable) | Bucket | What it does |")
        lines.append("|----|-------------------|--------|--------------|")
        for sk in sorted(items, key=lambda x: x["id"].lower()):
            desc = sk["description"].replace("|", "\\|")
            if len(desc) > 180:
                desc = desc[:177] + "..."
            lines.append(
                f"| `{sk['id']}` | `{sk['access']}` | `{sk['bucket']}` | {desc} |"
            )
        lines.append("")

    lines.append("## Scanned roots")
    lines.append("")
    for r in cat["scanned_roots"]:
        lines.append(f"- `{r}`")
    lines.append("")

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scanned = [
        f"<github>/ recursive SKILL.md "
        f"({'ok' if GITHUB.is_dir() else 'missing'})",
        "~/.claude/skills (legacy mirror)",
        "~/.cursor/skills-cursor (Cursor product)",
    ]
    rows: list[dict] = []
    seen: set[Path] = set()
    for md in iter_github_skills() + iter_home_skills():
        key = md.resolve()
        if key in seen:
            continue
        seen.add(key)
        rows.append(parse_skill(md))

    preferred = prefer_rows(rows)
    cat = {
        "schema": 2,
        "policy": "github-canonical",
        "note": (
            "Canonical skills under <github>/. ~/.claude/skills is legacy. "
            "Prefer github bucket when resolving a skill id."
        ),
        "scanned_roots": scanned,
        "count": len(rows),
        "unique_ids": len(preferred),
        "skills": rows,
        "preferred": preferred,
    }
    json_path = HERE / "catalog.json"
    json_path.write_text(json.dumps(cat, indent=2) + "\n", encoding="utf-8")
    write_markdown(cat, preferred, HERE / "catalog.md")
    n_gh = sum(1 for s in preferred if s["bucket"] == "github")
    print(
        f"wrote {json_path} ({len(rows)} files, {len(preferred)} unique, "
        f"{n_gh} github-canonical)"
    )
    print(f"wrote {HERE / 'catalog.md'}")


if __name__ == "__main__":
    main()
