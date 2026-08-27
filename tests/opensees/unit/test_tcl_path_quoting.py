"""Tcl token quoting — a bare Windows path is corrupted by the interpreter.

Tcl performs backslash substitution inside a bare word, so an unquoted
``C:\\Users\\nmora\\out.ladruno`` arrives at the recorder as ``C:Usersmora``
plus an embedded newline (``\\U`` and ``\\n`` are consumed). Brace-quoting
suppresses every substitution, so the token survives intact.

The failure this guards is SILENT, which is why it earns its own module:
a recorder that cannot open its file only WARNS, so the deck runs to
completion and exits 0 having written nothing. Measured on the fork —
``LadrunoRecorder error: cannot create file "C:Users``.

A path containing a space was always protected, because the whitespace
rule brace-quoted it for an unrelated reason. A path WITHOUT one was not,
and that is the majority of Windows paths.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.emitter.tcl import _fmt_value, _join

#: (token, must_be_braced, why)
_CASES = [
    (r"C:\Users\nmora\out.ladruno", True, "backslash escapes: \\U, \\n"),
    (r"C:\tmp\x.h5",               True, "\\t would become a TAB"),
    (r"C:\Users\nmora\My Libs\a",  True, "backslash AND space"),
    ("/c/users/nmora/out.ladruno", False, "posix path — nothing to protect"),
    ("out.ladruno",                False, "bare relative name"),
    ("-file",                      False, "an ordinary flag token"),
    ("with a space",               True,  "pre-existing whitespace rule"),
]


@pytest.mark.parametrize(
    "token,braced,why", _CASES, ids=[c[2] for c in _CASES],
)
def test_fmt_value_quotes_what_tcl_would_eat(token, braced, why) -> None:
    out = _fmt_value(token)
    assert (out == "{" + token + "}") is braced, why
    if not braced:
        assert out == token


@pytest.mark.parametrize(
    "token,braced,why", _CASES, ids=[c[2] for c in _CASES],
)
def test_join_agrees_with_fmt_value(token, braced, why) -> None:
    """The two must not drift.

    ``_join`` inlines the str branch for speed and its docstring promises
    output identical to :func:`_fmt_value`. A path quoted by one and not
    the other is exactly the silent-corruption case, and it would show up
    only on whichever emit sites happen to use the unfixed one.
    """
    assert _join(token) == _fmt_value(token)


def test_a_braced_path_round_trips_through_tcl_word_rules() -> None:
    """Brace-quoting is the fix precisely because braces suppress ALL
    substitution — the content between them is the literal path, so
    stripping the braces recovers exactly what was passed in."""
    p = r"C:\Users\nmora\AppData\Local\Temp\case\out.ladruno"
    emitted = _join("recorder", "ladruno", p)
    head, _, tail = emitted.partition(" ladruno ")
    assert head == "recorder"
    assert tail.startswith("{") and tail.endswith("}")
    assert tail[1:-1] == p


def test_ordinary_tokens_are_untouched() -> None:
    """Byte-identity for everything that needs no protection — the fix
    must not re-quote the millions of ordinary tokens in a bulk band."""
    assert _join("element", "quad", 1, 2, 3, 1.5) == "element quad 1 2 3 1.5"
    assert _join("node", 7, 0.0, -1.0) == "node 7 0.0 -1.0"
