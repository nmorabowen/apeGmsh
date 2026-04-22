"""MkDocs hooks — pull external content into the build.

The legacy content folders ``internal_docs/`` (guides, plans,
migration) and ``architecture/`` (design notes) live outside
``docs_dir``. Rather than duplicate or move those files, they are
registered as additional ``File`` objects during the ``on_files``
event. Content stays authoritative at its original location; the
site build sees it as if it were under ``docs/``.

``docs/index.md`` and ``docs/changelog.md`` are thin
``pymdownx.snippets`` wrappers over ``README.md`` / ``CHANGELOG.md``,
so those two files are NOT registered here.
"""
from __future__ import annotations

from pathlib import Path

from mkdocs.structure.files import File

EXTERNAL_DIRS = ("internal_docs", "architecture")


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
