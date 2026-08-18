"""Atomic text writes — one implementation, no package direction.

``atomic_write_text`` was born in ``studio/_paths.py`` under ADR 0095
INV-16 and is now needed by ``results.session``'s snapshot (ADR 0098
S5a) as well. ``results`` may not import ``studio`` — the direction is
forbidden, and a copy-paste would be two implementations of an
atomicity guarantee, which is the kind of thing that stays correct in
exactly one of them. So it lives here, at the package root, where both
sides are allowed to look. ``studio._paths`` re-exports it, so every
existing ``from ._paths import atomic_write_text`` keeps working.

Nothing here may import anything from ``apeGmsh`` — a leaf module by
construction.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write *text* via temp file + ``os.replace`` (ADR 0095 INV-16).

    Same-directory temp so replace is atomic on the destination volume.
    Readers may still see a missing file mid-replace; they must not see
    a truncated JSON body.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent),
        prefix=f".{dest.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        # Windows: a concurrent reader may briefly deny replace.
        last_err: OSError | None = None
        for attempt in range(20):
            try:
                os.replace(tmp_name, dest)
                last_err = None
                break
            except PermissionError as exc:
                last_err = exc
                time.sleep(0.01 * (attempt + 1))
        if last_err is not None:
            raise last_err
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return dest


__all__ = ["atomic_write_text"]
