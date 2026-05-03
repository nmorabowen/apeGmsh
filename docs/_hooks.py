"""MkDocs hooks — pull external content into the build.

The legacy content folders ``internal_docs/`` (guides, plans,
migration) and ``architecture/`` (design notes) live outside
``docs_dir``. Rather than duplicate or move those files, they are
registered as additional ``File`` objects during the ``on_files``
event. Content stays authoritative at its original location; the
site build sees it as if it were under ``docs/``.

Curated example notebooks from ``examples/EOS Examples/`` are
copied into ``docs/examples/notebooks/`` on every build so
mkdocs-jupyter can render them. Source of truth stays in
``examples/EOS Examples/``; the docs copy is git-ignored.

``docs/index.md`` and ``docs/changelog.md`` are thin
``pymdownx.snippets`` wrappers over ``README.md`` / ``CHANGELOG.md``,
so those two files are NOT registered here.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from mkdocs.structure.files import File

EXTERNAL_DIRS = ("internal_docs", "architecture")

# Curated notebooks surfaced under docs › Examples. The order here
# determines the gallery order; titles come from the .ipynb metadata
# but the ordering / curation is intentional.
CURATED_NOTEBOOKS: tuple[str, ...] = (
    "01_hello_plate.ipynb",
    "02_cantilever_beam_2D.ipynb",
    "04_portal_frame_2D.ipynb",
    "05_labels_and_pgs.ipynb",
    "10b_part_assembly.ipynb",
    "12_interface_tie.ipynb",
    "17_modal_analysis.ipynb",
    "19_pushover_elastoplastic.ipynb",
)

NOTEBOOKS_SOURCE = "examples/EOS Examples"
NOTEBOOKS_DEST = "docs/examples/notebooks"


def on_pre_build(config):
    """Copy curated notebooks into ``docs/examples/notebooks/``.

    Runs before ``on_files`` so mkdocs sees the .ipynb files as
    part of ``docs_dir`` and mkdocs-jupyter renders them. The
    destination is git-ignored so the source notebooks stay the
    single source of truth.
    """
    repo_root = Path(config["config_file_path"]).parent
    src_dir = repo_root / NOTEBOOKS_SOURCE
    dst_dir = repo_root / NOTEBOOKS_DEST

    if not src_dir.is_dir():
        return

    # Notebooks live under curriculum/<tier>/<stem>/<stem>.ipynb.
    _CURATED_TIERS = {
        "01_hello_plate":            "01-fundamentals",
        "02_cantilever_beam_2D":     "01-fundamentals",
        "04_portal_frame_2D":        "01-fundamentals",
        "05_labels_and_pgs":         "02-building-blocks",
        "10b_part_assembly":         "03-assemblies",
        "12_interface_tie":          "03-assemblies",
        "17_modal_analysis":         "05-analysis-types",
        "19_pushover_elastoplastic": "05-analysis-types",
    }
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in CURATED_NOTEBOOKS:
        stem = Path(name).stem
        tier = _CURATED_TIERS.get(stem)
        if tier is None:
            continue
        src = src_dir / "curriculum" / tier / stem / name
        if not src.is_file():
            continue
        dst = dst_dir / name
        # mtime-aware copy so incremental rebuilds are fast.
        if (
            not dst.is_file()
            or src.stat().st_mtime > dst.stat().st_mtime
        ):
            shutil.copy2(src, dst)


def on_files(files, config):
    repo_root = Path(config["config_file_path"]).parent

    for folder in EXTERNAL_DIRS:
        source = repo_root / folder
        if not source.is_dir():
            continue
        for md in sorted(source.glob("*.md")):
            rel = md.relative_to(repo_root).as_posix()
            files.append(
                File(
                    path=rel,
                    src_dir=str(repo_root),
                    dest_dir=config["site_dir"],
                    use_directory_urls=config["use_directory_urls"],
                )
            )
    return files
