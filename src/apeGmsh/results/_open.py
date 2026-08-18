"""Open a ``Results`` from a file path — the CLI doors' one opener.

Which reader answers for which extension is knowledge that belongs beside
the readers, not in a command-line module. It was living in
``viewers/__main__.py``, which flips at ADR 0098 S6, and the session
render door (S5c) needs the same dispatch without building on a module
scheduled to retire — so it lives here and ``viewers/__main__`` delegates.

``viewers/ui/_open_results.build_results`` is deliberately NOT folded in:
it sniffs the file's contents rather than its extension because the Qt
open dialog accepts whatever a human picked. Same words, different job.

Refusals are ``ValueError`` so each caller maps them to its own exit
code or error envelope; nothing here calls ``sys.exit``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from apeGmsh.results.Results import Results


def open_results(
    path: "str | Path", model_h5: "Optional[str | Path]" = None,
) -> "Results":
    """The right reader for this file.

    ``.mpco`` REQUIRES ``model_h5``: the file carries no ``/opensees/``
    zone, so the reader needs an explicit pointer (Phase 8 made this
    mandatory on :meth:`Results.from_mpco`).

    ``.ladruno`` is self-describing; a sibling archive, when supplied,
    binds a richer model. Routing it through the native branch would
    hand it to ``NativeReader`` and raise ``SchemaVersionError``.

    Native results normally carry their model in the same file
    (composed-file pattern, ``/opensees/`` at the root); ``model_h5``
    reads it from a sibling archive instead — for results whose embedded
    ``/model`` zone is not independently readable.
    """
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results import Results

    src = Path(path)
    model_path = None if model_h5 is None else Path(model_h5)
    suffix = src.suffix.lower()

    if suffix == ".mpco":
        if model_path is None:
            raise ValueError(
                "--model-h5 PATH is required for .mpco files "
                "(sibling model archive)."
            )
        return Results.from_mpco(src, model_h5=model_path)
    if suffix == ".ladruno":
        return Results.from_ladruno(src, model_h5=model_path)

    model_src = model_path if model_path is not None else src
    model = OpenSeesModel.from_h5(model_src)
    return Results.from_native(src, model=model, model_path=model_path)


__all__ = ["open_results"]
