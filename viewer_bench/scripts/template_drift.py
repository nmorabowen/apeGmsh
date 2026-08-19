#!/usr/bin/env python3
"""Compare this habitat's APE/ and scripts/ trees against the INSTALLED
apeGmsh template — Lane B candidates (promotion-workflow v2, see
`APE/instructions/continuous-improvement.md`).

Self-contained: stdlib + `_habitat.py` only. The comparison target is the
template packaged with whatever apeGmsh is importable (PYTHONPATH), not
this file's own copy — that is the point (habitat vs installed template).
"""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path
from typing import Iterator, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _habitat import HABITAT, ensure_pythonpath  # noqa: E402

TEMPLATE_PACKAGE = "apeGmsh.studio.template"

# Mirrors apeGmsh.studio._init_habitat.DOTFILE_MAP / SKIP_ON_INIT. Kept as
# a local copy (not imported) so this script stays self-contained and
# still works against an older/newer apeGmsh than the one that wrote it.
DOTFILE_MAP = {
    "dot.gitignore": ".gitignore",
    "cursor/mcp.json": ".cursor/mcp.json",
}
SKIP_ON_INIT = {"README.md"}

# Lane B only promotes through these trees; models/, reports/, references/,
# postmortem/sessions/ etc. are habitat-owned content, never template copies.
COMPARE_PREFIXES = ("APE/", "scripts/")


def _iter_template_files() -> Iterator[Tuple[str, object]]:
    root = resources.files(TEMPLATE_PACKAGE)

    def walk(node, prefix: str) -> Iterator[Tuple[str, object]]:
        for child in sorted(node.iterdir(), key=lambda c: c.name):
            rel = f"{prefix}{child.name}"
            if child.is_dir():
                yield from walk(child, rel + "/")
            else:
                yield rel, child

    yield from walk(root, "")


def _target_rel(rel: str) -> str:
    return DOTFILE_MAP.get(rel, rel)


def _is_memory_content(rel: str) -> bool:
    return rel.startswith("APE/memory/") and not rel.endswith("README.md")


def main() -> int:
    ensure_pythonpath()
    try:
        template_files = list(_iter_template_files())
    except ModuleNotFoundError as exc:
        print(f"error: apeGmsh is not installed / importable: {exc}", file=sys.stderr)
        print(
            "  Fix PYTHONPATH / activate the apeGmsh venv (see "
            "APE/instructions/studio-mcp.md), then retry.",
            file=sys.stderr,
        )
        return 2

    template_map = {}
    for rel, node in template_files:
        habitat_rel = _target_rel(rel)
        if habitat_rel in SKIP_ON_INIT:
            continue
        if not habitat_rel.startswith(COMPARE_PREFIXES):
            continue
        template_map[habitat_rel] = node

    habitat_map = {}
    for prefix in COMPARE_PREFIXES:
        base = HABITAT / prefix
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                habitat_map[path.relative_to(HABITAT).as_posix()] = path

    template_rels = set(template_map)
    habitat_rels = set(habitat_map)
    template_only = sorted(template_rels - habitat_rels)
    habitat_only = sorted(habitat_rels - template_rels)
    common = sorted(template_rels & habitat_rels)

    modified = []
    memory_drift = []
    for rel in common:
        if template_map[rel].read_bytes() == habitat_map[rel].read_bytes():
            continue
        (memory_drift if _is_memory_content(rel) else modified).append(rel)

    print(f"habitat:  {HABITAT}")
    print(f"template: {TEMPLATE_PACKAGE} ({len(common)} files compared)")
    print()

    print(f"MODIFIED — habitat differs from template (Lane B candidates) ({len(modified)})")
    for rel in modified:
        print(f"  - {rel}")
    if not modified:
        print("  (none)")
    if memory_drift:
        print("  (memory — expected drift, not candidates)")
        for rel in memory_drift:
            print(f"    - {rel}")
    print()

    print(f"HABITAT-ONLY — not shipped by the template ({len(habitat_only)})")
    for rel in habitat_only:
        print(f"  - {rel}")
    if not habitat_only:
        print("  (none)")
    print()

    print(f"TEMPLATE-ONLY — not present in this habitat ({len(template_only)})")
    for rel in template_only:
        print(f"  - {rel}")
    if not template_only:
        print("  (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
