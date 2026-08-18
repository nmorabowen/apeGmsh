"""``apeGmsh.results._open.open_results`` — the CLI doors' one opener.

Lifted out of ``viewers/__main__`` for ADR 0098 S5c: the session render
door needs the same extension dispatch without building on a module that
flips at S6. These are the rules that were only ever asserted indirectly,
through whichever CLI happened to call it — a mutation pass found the
``.mpco`` refusal untested from any direction.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.results._open import open_results


def test_mpco_without_its_model_archive_refuses(tmp_path: Path):
    """``.mpco`` carries no ``/opensees/`` zone, so the reader needs an
    explicit pointer (Phase 8 made this mandatory on
    ``Results.from_mpco``). Refusing here — rather than handing ``None``
    down — is what turns it into a sentence a caller can act on."""
    mpco = tmp_path / "run.mpco"
    mpco.write_bytes(b"not really mpco")

    with pytest.raises(ValueError, match="--model-h5"):
        open_results(mpco)


def test_the_refusal_is_a_ValueError_not_a_process_exit(tmp_path: Path):
    """It used to ``sys.exit(2)`` from inside the opener, which no
    library caller can catch and no error envelope can report. Each door
    maps it to its own exit code or JSON error instead."""
    mpco = tmp_path / "run.mpco"
    mpco.write_bytes(b"not really mpco")

    with pytest.raises(ValueError):
        open_results(mpco, None)


def test_the_viewers_cli_still_exits_2_on_that_refusal(tmp_path: Path):
    """The wrapper's contract is unchanged — it is the CLI's job to turn
    the refusal into an exit code, and it still does."""
    from apeGmsh.viewers.__main__ import _open_results

    mpco = tmp_path / "run.mpco"
    mpco.write_bytes(b"not really mpco")

    with pytest.raises(SystemExit) as excinfo:
        _open_results(mpco, None)

    assert excinfo.value.code == 2


def test_a_native_file_routes_to_the_native_reader(tmp_path: Path):
    """Dispatch is by extension; an unreadable native file fails in the
    READER, not in the dispatch — which is how we know it got there."""
    native = tmp_path / "out.h5"
    native.write_bytes(b"not really h5")

    with pytest.raises(Exception) as excinfo:
        open_results(native)

    assert "--model-h5" not in str(excinfo.value)
