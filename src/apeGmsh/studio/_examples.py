"""``python -m apeGmsh.studio example`` — the oracle-bearing example library.

ADR 0095 Amendment 7, S8a. Same ``importlib.resources`` mechanics and
raw-argv ``__main__`` integration pattern as ``init`` (Amendment 6, see
``apeGmsh.studio._init_habitat``): the packaged
``apeGmsh.studio.examples`` tree ships one directory per example
(script, ``manifest.json``, ``README.md``, ``verify.py``), reachable
from an installed wheel, not just an editable source checkout.

``list`` / ``show`` / ``copy`` — no MCP tool (same stance as ``init``).
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path
from typing import Iterator, Optional, Sequence

EXAMPLES_PACKAGE = "apeGmsh.studio.examples"


def _iter_example_names() -> Iterator[str]:
    """Yield packaged example directory names, sorted."""
    root = resources.files(EXAMPLES_PACKAGE)
    for child in sorted(root.iterdir(), key=lambda c: c.name):
        if child.is_dir():
            yield child.name


def _example_dir(name: str):
    root = resources.files(EXAMPLES_PACKAGE)
    node = root / name
    if not node.is_dir():
        raise KeyError(name)
    return node


def _load_manifest(name: str) -> dict:
    node = _example_dir(name)
    text = (node / "manifest.json").read_text(encoding="utf-8")
    return json.loads(text)


def _unknown_example_error(name: str) -> str:
    known = ", ".join(sorted(_iter_example_names()))
    return f"error: unknown example {name!r}; known examples: {known}"


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def run_list(tag: Optional[str]) -> int:
    for name in sorted(_iter_example_names()):
        manifest = _load_manifest(name)
        tags = manifest.get("tags", [])
        if tag is not None and tag not in tags:
            continue
        title = manifest.get("title", "")
        print(f"{name}  [{', '.join(tags)}]  {title}")
    return 0


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def run_show(name: str) -> int:
    try:
        manifest = _load_manifest(name)
    except (KeyError, FileNotFoundError):
        print(_unknown_example_error(name), file=sys.stderr)
        return 1

    print(f"=== {manifest.get('name', name)} ===")
    print(f"title:    {manifest.get('title', '')}")
    print(f"tags:     {', '.join(manifest.get('tags', []))}")
    print(f"requires: {', '.join(manifest.get('requires', []))}")
    print(f"provenance: {manifest.get('provenance', '')}")
    print(f"teaches:  {manifest.get('teaches', '')}")
    print("metrics:")
    for m in manifest.get("metrics", []):
        tol = m.get("tol_abs")
        tol_kind = "tol_abs"
        if tol is None:
            tol = m.get("tol_rel")
            tol_kind = "tol_rel"
        print(f"  - {m['name']}: expected={m['expected']} ({tol_kind}={tol}) [{m.get('units', '-')}]")

    node = _example_dir(name)
    readme = node / "README.md"
    if readme.is_file():
        print()
        print(readme.read_text(encoding="utf-8"))
    return 0


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def _copy_tree(node, dest: Path) -> None:
    dest.mkdir(parents=True)
    for child in sorted(node.iterdir(), key=lambda c: c.name):
        target = dest / child.name
        if child.is_dir():
            _copy_tree(child, target)
        else:
            target.write_bytes(child.read_bytes())


def run_copy(name: str, dest_dir: Optional[str]) -> int:
    try:
        node = _example_dir(name)
    except KeyError:
        print(_unknown_example_error(name), file=sys.stderr)
        return 1

    dest = Path(dest_dir).resolve() if dest_dir else (Path.cwd() / name)
    if dest.exists():
        print(f"error: destination already exists: {dest}", file=sys.stderr)
        return 1

    _copy_tree(node, dest)
    print(f"OK: copied {name!r} -> {dest}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.studio example",
        description=(
            "The oracle-bearing example library (ADR 0095 Amendment 7). "
            "'list' shows the packaged examples with tags; 'show' prints "
            "a manifest summary + README; 'copy' lands a runnable "
            "directory in the destination."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_list = sub.add_parser("list", help="List packaged examples.")
    p_list.add_argument("--tag", default=None, help="Filter by tag.")

    p_show = sub.add_parser("show", help="Show one example's manifest + README.")
    p_show.add_argument("name")

    p_copy = sub.add_parser("copy", help="Copy an example into a destination directory.")
    p_copy.add_argument("name")
    p_copy.add_argument(
        "--dest", default=None,
        help="Destination directory (default: cwd/<name>). Refuses to overwrite.",
    )

    args = parser.parse_args(argv)
    if args.subcommand == "list":
        return run_list(args.tag)
    if args.subcommand == "show":
        return run_show(args.name)
    if args.subcommand == "copy":
        return run_copy(args.name, args.dest)
    parser.print_usage(sys.stderr)  # pragma: no cover — argparse enforces choices
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
