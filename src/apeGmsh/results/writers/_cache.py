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
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ..spec._resolved import ResolvedRecorderSpec


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
    model_token: str | None = None,
) -> str:
    """Return a 16-char hex cache key from input fingerprints.

    ``model_token`` is :func:`model_cache_token` of the ``model=`` the
    transcode embeds and checks its element ids against. Two models
    built from one FEMData (``element_tags="fem"`` and
    ``"sequential"``) must not share a cached file: its ``/opensees``
    zone and its ``element_index_space`` stamp both come from the model.
    ``None`` (no model) keys apart from every model-bearing key.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(parser_version.encode("utf-8"))
    h.update(b"|")
    h.update(fem_snapshot_id.encode("utf-8"))
    h.update(b"|")
    if model_token is None:
        h.update(b"model:none|")
    else:
        h.update(b"model:" + model_token.encode("utf-8") + b"|")
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


def model_cache_token(model: Any) -> str:
    """Return a hex digest of what a cached transcode takes from ``model``.

    It covers the model's ``lineage.fem_hash`` and ``lineage.model_hash``
    (its ``/opensees`` content, when stamped) and its ``(fem_eid, tag)``
    element pairing. The stamp depends on the pairing and on the
    ``fem_hash``, so both are hashed directly: the token separates two
    models even when ``model_hash`` is ``None`` (a composed model).
    """
    lineage = model.lineage
    h = hashlib.blake2b(digest_size=16)
    h.update(str(lineage.fem_hash).encode("utf-8"))
    h.update(b"|")
    h.update(str(lineage.model_hash).encode("utf-8"))
    h.update(b"|")
    pairs = sorted((int(r.fem_eid), int(r.tag)) for r in model.elements())
    for fem_eid, tag in pairs:
        h.update(f"{fem_eid}:{tag};".encode("ascii"))
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
    from ..spec._emit import emit_logical, _DEFERRED_CATEGORIES
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
