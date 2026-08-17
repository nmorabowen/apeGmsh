"""``python -m apeGmsh.studio init`` — stamp a fresh APE Studio habitat.

ADR 0095 Amendment 6, S7a. Ports ``ape-studio-template@bab594c``'s
``scripts/init_habitat.py`` (acceptance-tested there against a habitat
that already contained the whole template tree) into a **copier**: the
packaged ``apeGmsh.studio.template`` tree is copied into a target
directory, then personalized exactly as the original script did.

Uses :mod:`importlib.resources` to walk the packaged template rather than
``__file__`` / filesystem assumptions, so it works from an installed
wheel, not just an editable source checkout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

if TYPE_CHECKING:
    # Traversable's home moved from ``importlib.abc`` (3.9+) to
    # ``importlib.resources.abc`` (3.11+); this project floors 3.10, so
    # keep the precise type import type-checking-only rather than pick a
    # runtime import that breaks on one of the two floors.
    from importlib.resources.abc import Traversable

TEMPLATE_PACKAGE = "apeGmsh.studio.template"

HABITAT_PLACEHOLDER = "__HABITAT_NAME__"
MODEL_PLACEHOLDER = "__MODEL_ID__"
ROOT_PLACEHOLDER = "__HABITAT_ROOT__"

# Packaging hazard workaround (see the template's own docstring / the PR
# that introduced it): a literal nested ``.gitignore`` would apply its
# rules to this subtree of apeGmsh's own repo, and dot-prefixed entries
# are dropped by the glob-based package-data pattern. Stored under safe
# names in the package; restored to their real dotfile paths here.
DOTFILE_MAP = {
    "dot.gitignore": ".gitignore",
    "cursor/mcp.json": ".cursor/mcp.json",
}

# Shipped as package data (byte-identical, INV-21) for reference, but not
# written into initialized habitats: describes the rejected standalone
# template-repo workflow ("use this template on GitHub", "run
# scripts/init_habitat.py") that `studio init` supersedes.
SKIP_ON_INIT = {"README.md"}


def _iter_template_files() -> Iterator[tuple[str, Traversable]]:
    """Yield ``(relative_posix_path, Traversable)`` for every file in the
    packaged template tree, via ``importlib.resources``."""
    root = resources.files(TEMPLATE_PACKAGE)

    def walk(node: Traversable, prefix: str) -> Iterator[tuple[str, Traversable]]:
        for child in sorted(node.iterdir(), key=lambda c: c.name):
            rel = f"{prefix}{child.name}"
            if child.is_dir():
                yield from walk(child, rel + "/")
            else:
                yield rel, child

    yield from walk(root, "")


def _target_rel(rel: str) -> str:
    """Map a packaged-template relative path to its habitat path,
    undoing the dotfile renames."""
    return DOTFILE_MAP.get(rel, rel)


def _refusal_reason(target: Path) -> Optional[str]:
    """Return a reason to refuse, or ``None`` if the target is safe to
    initialize into. Read-only — never modifies ``target``."""
    if target.is_file():
        return f"target exists and is not a directory: {target}"
    if not target.is_dir():
        return None

    yaml_path = target / "ape.project.yaml"
    if yaml_path.is_file():
        text = yaml_path.read_text(encoding="utf-8")
        if HABITAT_PLACEHOLDER not in text:
            return (
                f"already initialized: {yaml_path} has no "
                f"{HABITAT_PLACEHOLDER} placeholder left; refusing to run "
                "twice (edit files by hand from here)"
            )

    for rel, _ in _iter_template_files():
        habitat_rel = _target_rel(rel)
        if habitat_rel in SKIP_ON_INIT:
            continue
        if (target / habitat_rel).exists():
            return (
                f"target is non-empty: {habitat_rel} already exists under "
                f"{target}"
            )
    return None


def _copy_template(target: Path) -> None:
    for rel, node in _iter_template_files():
        habitat_rel = _target_rel(rel)
        if habitat_rel in SKIP_ON_INIT:
            continue
        dest = target / habitat_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(node.read_bytes())


def _substitute_placeholders(target: Path, name: str, model: str) -> None:
    """Replace ``__HABITAT_NAME__`` / ``__MODEL_ID__`` in every copied
    text file (``ape.project.yaml``, ``APE/memory/brief.md`` today — but
    this is a blind sweep so a template file added later that carries
    either placeholder is covered too). ``.cursor/mcp.json`` carries the
    unrelated ``__HABITAT_ROOT__`` placeholder, handled separately by
    :func:`_align_mcp_json`."""
    for rel, _ in _iter_template_files():
        habitat_rel = _target_rel(rel)
        if habitat_rel in SKIP_ON_INIT or habitat_rel == ".cursor/mcp.json":
            continue
        dest = target / habitat_rel
        try:
            text = dest.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if HABITAT_PLACEHOLDER in text or MODEL_PLACEHOLDER in text:
            text = text.replace(HABITAT_PLACEHOLDER, name).replace(
                MODEL_PLACEHOLDER, model
            )
            dest.write_text(text, encoding="utf-8")
    print(f"OK: substituted {HABITAT_PLACEHOLDER}={name!r}, {MODEL_PLACEHOLDER}={model!r}")


def _align_mcp_json(target: Path) -> None:
    """Same mechanism as the template's own ``init_habitat.py`` /
    ``_habitat.py``: point ``.cursor/mcp.json``'s ``APEGMSH_STUDIO_ROOT``
    at this habitat's resolved path."""
    mcp_json = target / ".cursor" / "mcp.json"
    if not mcp_json.is_file():
        print("WARN: .cursor/mcp.json missing, skipping")
        return
    data = json.loads(mcp_json.read_text(encoding="utf-8"))
    resolved = str(target.resolve())
    changed = False
    for cfg in (data.get("mcpServers") or {}).values():
        env = cfg.get("env") or {}
        if env.get("APEGMSH_STUDIO_ROOT") == ROOT_PLACEHOLDER:
            env["APEGMSH_STUDIO_ROOT"] = resolved
            changed = True
    if changed:
        mcp_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"OK: .cursor/mcp.json APEGMSH_STUDIO_ROOT -> {resolved}")
    else:
        print(f"WARN: .cursor/mcp.json had no {ROOT_PLACEHOLDER} placeholder")


def _scaffold_model(target: Path, model: str) -> None:
    src = target / "models" / model / "src"
    cases = target / "models" / model / "cases"
    src.mkdir(parents=True, exist_ok=True)
    cases.mkdir(parents=True, exist_ok=True)
    readme = src / "README.md"
    if not readme.is_file():
        readme.write_text(
            f"# {model} — model source\n\n"
            "Geometry / labels / mesh / analysis scripts live here. "
            "See `models/README.md` for the cases contract.\n",
            encoding="utf-8",
        )
    print(f"OK: models/{model}/src/, models/{model}/cases/")


def _run_check_template(target: Path) -> int:
    return subprocess.call(
        [sys.executable, str(target / "scripts" / "check_template.py"), "--strict"],
        cwd=str(target),
    )


def run_init(name: str, model: str, root: Optional[str]) -> int:
    target = Path(root).resolve() if root else Path.cwd()

    reason = _refusal_reason(target)
    if reason is not None:
        print(f"error: {reason}", file=sys.stderr)
        return 1

    print("=== APE habitat INIT ===")
    print(f"root: {target}")

    target.mkdir(parents=True, exist_ok=True)
    _copy_template(target)
    _substitute_placeholders(target, name, model)
    _align_mcp_json(target)
    _scaffold_model(target, model)

    print()
    rc = _run_check_template(target)
    print(f"check_template.py --strict exit={rc}")
    if rc != 0:
        print("FAIL: template check did not pass after init.")
        return rc

    print()
    print("Next: open APE/README.md and follow session start")
    print("      (instructions -> memory/brief.md -> scripts/start.ps1).")
    print("=== INIT complete ===")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.studio init",
        description=(
            "Copy the packaged APE Studio habitat template into a target "
            "directory (default cwd), substitute __HABITAT_NAME__ / "
            "__MODEL_ID__, align .cursor/mcp.json, scaffold "
            "models/<model>/{src,cases}, and run the new habitat's own "
            "scripts/check_template.py --strict (ADR 0095 Amendment 6)."
        ),
    )
    parser.add_argument("--name", required=True, help="habitat name (ape.project.yaml)")
    parser.add_argument("--model", required=True, help="first model id (models/<model>/)")
    parser.add_argument(
        "--root",
        default=None,
        help="target directory to initialize (default: cwd)",
    )
    args = parser.parse_args(argv)
    return run_init(args.name, args.model, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
