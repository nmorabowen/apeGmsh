"""Cache layer for recorder transcoder output.

Phase 6 cache strategy
----------------------
Recorder output (``.out`` / ``.xml`` from a Tcl/Py-driven OpenSees
run) is parsed into a native HDF5 file. Subsequent
``Results.from_recorders(...)`` calls hash the inputs and skip the
re-parse if a cached HDF5 already exists.

Cache key inputs:
- Source files' ``(path, mtime, size)`` tuples
- ``parser_version`` from ``schema/_versions.py``
- ``fem_snapshot_id`` from the bound FEMData

Cache root resolution (in priority order):
1. Explicit ``cache_root=`` kwarg on ``from_recorders``.
2. ``APEGMSH_RESULTS_DIR`` environment variable.
3. ``<cwd>/results``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from ...solvers._recorder_specs import ResolvedRecorderSpec


_DEFAULT_CACHE_DIR_NAME = "results"
_ENV_OVERRIDE = "APEGMSH_RESULTS_DIR"


def resolve_cache_root(explicit: str | Path | None = None) -> Path:
    """Resolve the cache root directory per the documented precedence."""
    if explicit is not None:
        root = Path(explicit)
    elif os.environ.get(_ENV_OVERRIDE):
        root = Path(os.environ[_ENV_OVERRIDE])
    else:
        root = Path.cwd() / _DEFAULT_CACHE_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def compute_cache_key(
    source_files: Iterable[Path],
    *,
    parser_version: str,
    fem_snapshot_id: str,
) -> str:
    """Return a 16-char hex cache key from input fingerprints."""
    h = hashlib.blake2b(digest_size=16)
    h.update(parser_version.encode("utf-8"))
    h.update(b"|")
    h.update(fem_snapshot_id.encode("utf-8"))
    h.update(b"|")
    for f in sorted(source_files, key=lambda p: str(p)):
        try:
            stat = f.stat()
            h.update(str(f).encode("utf-8"))
            h.update(b"|")
            h.update(str(stat.st_mtime_ns).encode("utf-8"))
            h.update(b"|")
            h.update(str(stat.st_size).encode("utf-8"))
            h.update(b"\n")
        except FileNotFoundError:
            # Missing files are still part of the key — they invalidate
            # the cache when they appear later.
            h.update(str(f).encode("utf-8"))
            h.update(b"|missing\n")
    return h.hexdigest()


def cache_paths(
    cache_root: Path, key: str,
) -> tuple[Path, Path]:
    """Return ``(cached_h5_path, cached_manifest_path)`` for a key."""
    return (cache_root / f"{key}.h5", cache_root / f"{key}.manifest.h5")


def list_source_files(
    spec: "ResolvedRecorderSpec",
    output_dir: Path,
    *,
    file_format: str = "out",
    stage_id: str | None = None,
) -> list[Path]:
    """Enumerate the recorder output files emit_logical would target.

    ``stage_id`` matches the prefix used by live recorder emission;
    pass it when listing files for a specific stage.
    """
    from ...solvers._recorder_emit import emit_logical, _DEFERRED_CATEGORIES
    files: list[Path] = []
    for rec in spec.records:
        if rec.category in _DEFERRED_CATEGORIES:
            continue
        for lr in emit_logical(
            rec, output_dir=str(output_dir),
            file_format=file_format, stage_id=stage_id,
        ):
            files.append(Path(lr.file_path))
    return files
