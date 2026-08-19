"""Harvest ape* library repos under <github>/ into a portable catalog.

Writes:
  APE/libraries/catalog.json
  (human narrative stays in APE/libraries/catalog.md — edit roles there;
   this script refreshes on-disk status + README blurbs.)

    python APE/libraries/harvest.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HOME = Path.home()
GITHUB = HOME / "Documents" / "Github"
HERE = Path(__file__).resolve().parent

# Required infra only — these three are wired into the shipped tools/
# launchers and scripts/_habitat.py PYTHONPATH default, so they are genuine
# template dependencies, not personal recommendations.
#
# Any further design/access-point libraries (steel or concrete design
# modules, seismic ground-motion toolkits, BIM/analytical access points,
# etc.) are the template author's own personal stack — not distributed or
# required here. Add your own rows below if you have equivalents; see
# APE/libraries/README.md for agent guidance.
LIBRARIES = [
    {
        "id": "apeCAD",
        "layer": "user_agent_bridge",
        "priority": "now",
        "role": (
            "3D spatial scratchpad / intent document; bridge between user "
            "interaction and agent; Python document is the spatial language."
        ),
    },
    {
        "id": "apeSketch",
        "layer": "user_agent_bridge",
        "priority": "now",
        "role": (
            "Hand-ink / scheme bridge; drawings and sketches agents consume "
            "as basis for project development."
        ),
    },
    {
        "id": "apeGmsh",
        "layer": "fem_habitat",
        "priority": "now",
        "role": (
            "Gmsh FEM wrapper, FEMData, OpenSees bridge, Studio habitat "
            "(.apegmsh, MCP). Required — the whole habitat is built on it."
        ),
    },
]


def _readme_blurb(repo: Path) -> str:
    for name in ("README.md", "Readme.md", "readme.md"):
        p = repo / name
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            # first non-heading paragraph-ish
            paras: list[str] = []
            buf: list[str] = []
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("#"):
                    if buf:
                        break
                    continue
                if not s:
                    if buf:
                        paras.append(" ".join(buf))
                        break
                    continue
                if s.startswith("```") or s.startswith("|") or s.startswith(">"):
                    if buf:
                        paras.append(" ".join(buf))
                        break
                    continue
                buf.append(s)
            if buf and not paras:
                paras.append(" ".join(buf))
            blurb = paras[0] if paras else ""
            blurb = re.sub(r"\s+", " ", blurb).strip()
            return blurb[:400]
    return ""


def main() -> None:
    rows = []
    for spec in LIBRARIES:
        repo = GITHUB / spec["id"]
        on_disk = repo.is_dir()
        access = f"<github>/{spec['id']}"
        row = {
            **spec,
            "display_name": spec.get("display_name", spec["id"]),
            "access": access,
            "on_disk": on_disk,
            "readme_blurb": _readme_blurb(repo) if on_disk else "",
        }
        rows.append(row)
    cat = {
        "schema": 1,
        "note": (
            "Portable Studio library catalog. Human roles: APE/libraries/catalog.md. "
            "Paths use <github>/ = ~/Documents/Github/."
        ),
        "github_root_portable": "<github>/",
        "github_root_resolved": str(GITHUB),
        "count": len(rows),
        "on_disk": sum(1 for r in rows if r["on_disk"]),
        "libraries": rows,
    }
    dest = HERE / "catalog.json"
    dest.write_text(json.dumps(cat, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {dest} ({cat['on_disk']}/{cat['count']} on disk under {GITHUB})"
    )


if __name__ == "__main__":
    main()
