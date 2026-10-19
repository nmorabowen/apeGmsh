"""The fork branch of the default results route (ADR 0122 D4, #1950).

:class:`~apeGmsh.results.live._default.LiveDefault` picks a route from the
bound build. On the Ladruno fork it records a ``.ladruno`` in process, and
everything that names that format lives here, behind the fork seam that the
fork-token ratchet exempts (``readers/_ladruno*``):

* the recorder call: the fork's Ladruno recorder takes MPCO's ``-N`` /
  ``-E`` / ``-T`` grammar, so its arguments are :func:`mpco_ops_args`' with
  the first token swapped;
* the target file and the partition parts a rerun would overwrite;
* the read: ``stamp_provenance`` (ADR 0122 D3), then
  ``Results.from_ladruno``. A stamp the file refuses is skipped with a
  warning (§7b.1 rule 3); the caller words and raises that warning;
* the deck route: ``convert_mpco`` of a deck's ``.mpco`` into the fork's
  format beside it, then the same read.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..Results import Results
from ..spec._emit import mpco_ops_args
from ._ladruno_from_mpco import convert_mpco
from ._ladruno_provenance import ProvenanceStampRefused, stamp_provenance

if TYPE_CHECKING:
    from ..spec._resolved import ResolvedRecorderRecord

__all__ = [
    "FORK_SUFFIX",
    "convert_and_open",
    "existing_fork_files",
    "fork_target_of",
    "fork_recorder_args",
    "open_fork_results",
    "stamp_fork_file",
]

#: The suffix of the file the fork's recorder writes.
FORK_SUFFIX = ".ladruno"
_TOKEN = "ladruno"
_PART = re.compile(r"^(?P<stem>.+?)\.part-\d+$")


def fork_recorder_args(
    records: "tuple[ResolvedRecorderRecord, ...]", target: Path,
) -> tuple:
    """``ops.recorder(*args)`` arguments that record ``records`` to ``target``.

    :func:`mpco_ops_args` with its leading ``"mpco"`` replaced by the
    fork recorder's token; the rest of the grammar is shared.
    """
    out_dir = str(target.parent) if str(target.parent) != "." else ""
    args = mpco_ops_args(records, output_dir=out_dir, filename=target.name)
    if not args or args[0] != "mpco":
        raise RuntimeError(
            f"mpco_ops_args returned {args[:1]!r} as its first token, not "
            "'mpco'; the fork recorder's arguments are built by swapping it."
        )
    return (_TOKEN, *args[1:])


def existing_fork_files(target: Path) -> "list[Path]":
    """``target`` and any ``<stem>.part-<N>`` file of it that exist."""
    found = [target] if target.exists() else []
    found += sorted(target.parent.glob(f"{target.stem}.part-*{FORK_SUFFIX}"))
    return found


def stamp_fork_file(
    target: Path, model_h5: Path, build: str = "unknown",
) -> Optional[str]:
    """Stamp ``target`` with ``model_h5``'s pairing; the refusal text, if any.

    ``build`` is the ``ladrunoBuild`` hash of the engine that ran the
    analysis in this process (``"unknown"`` when it answers none): the file
    does not carry it, so only this in-session caller can stamp it as
    ``opensees_build_sha`` (§7b.1, G10 #2208).

    ``None`` when the stamp was written (or already equal). A
    :class:`ProvenanceStampRefused` is returned as its message, so the
    caller can warn and read the file unstamped (§7b.1 rule 3); nothing
    else is caught.
    """
    try:
        stamp_provenance(target, model_h5, opensees_build_sha=build)
    except ProvenanceStampRefused as e:
        return str(e)
    return None


def open_fork_results(target: Path, model_h5: Path) -> Results:
    """Open the fork's file with the model archive the run was built from."""
    return Results.from_ladruno(str(target), model_h5=str(model_h5))


def fork_target_of(mpco: Path) -> Path:
    """The fork-format file a deck's ``.mpco`` converts to, beside it.

    ``run.mpco`` gives ``run.ladruno``; a part ``run.part-<N>.mpco`` names
    its set, ``run.ladruno``, which ``convert_mpco`` writes part for part.
    """
    stem = mpco.stem
    m = _PART.match(stem)
    return mpco.with_name(f"{m.group('stem') if m else stem}{FORK_SUFFIX}")


def convert_and_open(mpco: Path, model_h5: Path, *, overwrite: bool) -> Results:
    """Convert a deck's ``.mpco`` beside itself, then open the result.

    The one caller of ``convert_mpco``'s private ``_implicit_stage`` path,
    which refuses every unstaged program (ruling (iii) on #1950).
    :class:`~._ladruno_from_mpco.MPCOConversionRefused` propagates, before
    anything is written; the caller decides what a refusal reads instead.
    """
    targets = convert_mpco(mpco, fork_target_of(mpco), model_h5=model_h5,
                           overwrite=overwrite, _implicit_stage=True)
    paths: "str | list[str | Path]" = (
        str(targets[0]) if len(targets) == 1 else list(targets))
    return Results.from_ladruno(paths, model_h5=str(model_h5))
