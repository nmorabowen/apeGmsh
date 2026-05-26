"""Cross-rank partition-block diff helper for ADR 0027 INV-1 tests.

ADR 0027 INV-1 (locked phrasing):

    A cross-partition MP constraint emits **byte-identical text inside
    every owning rank's ``partition_open(rank)`` / ``partition_close()``
    block, *modulo foreign-node decl prefixes*.**

The intentional textual divergence is that **native-side** ``node(tag,
*xyz)`` decls omit ``-ndf`` (the model envelope ``ndf`` set via the
``model`` directive applies), while **foreign-side** decls force
``-ndf <broker_ndf>`` (per
``src/apeGmsh/opensees/_internal/build.py::_emit_node_with_broker_ndf``).
Everything else — the constraint declaration line, the
``mp_constraint_comment`` line, the argument order, the trailing
whitespace, the line terminator — must be byte-for-byte identical
across the owning ranks.

This module ships :func:`assert_partition_blocks_equivalent`, which
strips foreign-node decl lines from two blocks and compares what
remains.

Supported emit syntaxes:

* **Tcl** — ``node <tag> <x> <y> <z> -ndf <K>`` (foreign) /
  ``node <tag> <x> <y> <z>`` (native, never stripped).
* **Py** — ``ops.node(<tag>, <x>, <y>, <z>, '-ndf', <K>)`` (foreign) /
  ``ops.node(<tag>, <x>, <y>, <z>)`` (native, never stripped).

Native-decl lines are **not** stripped: they appear once per rank
(each rank natively owns disjoint nodes), so a caller comparing two
full rank-blocks with native decls will see them as differences. The
helper is intended for callers comparing the **constraint-emitting
portion** of each rank's block (constraint lines + their phantom /
foreign-node prologue + ``mp_constraint_comment`` lines), where the
only allowed difference is which foreign-node prologue precedes the
constraint on each rank.
"""
from __future__ import annotations

import difflib
import re

__all__ = [
    "FOREIGN_NODE_DECL_RE_TCL",
    "FOREIGN_NODE_DECL_RE_PY",
    "assert_partition_blocks_equivalent",
]


# ---------------------------------------------------------------------
# Foreign-decl recognition
# ---------------------------------------------------------------------
#
# Tcl form, post-emitter formatting:
#     node <tag> <x> <y> <z> -ndf <K>
# emitted by ``TclEmitter.node(..., ndf=K)`` → ``_join("node", tag,
# x, y, z, "-ndf", K)`` (single spaces, no leading whitespace beyond
# the partition indent ``_LineBuf`` adds).  The regex matches the
# whole line modulo leading whitespace.
FOREIGN_NODE_DECL_RE_TCL = re.compile(
    r"^\s*node\s+\S+\s+\S+\s+\S+\s+\S+\s+-ndf\s+\S+\s*$"
)

# Py form, post-emitter formatting:
#     ops.node(<tag>, <x>, <y>, <z>, '-ndf', <K>)
# emitted by ``PyEmitter.node(..., ndf=K)`` → ``_ops_call("node", tag,
# x, y, z, "-ndf", K)``.  Strings are repr'd with single quotes
# (``_fmt_value`` in ``emitter/py.py``); accept double quotes too in
# case future hand-written variants appear.
FOREIGN_NODE_DECL_RE_PY = re.compile(
    r"""^\s*ops\.node\(
        [^,]+,\s*    # tag
        [^,]+,\s*    # x
        [^,]+,\s*    # y
        [^,]+,\s*    # z
        ['"]-ndf['"]\s*,\s*
        [^,)]+\)\s*$
    """,
    re.VERBOSE,
)


def _strip_foreign_node_decls(block: str) -> list[str]:
    """Return ``block.splitlines()`` with foreign-node decl lines removed.

    A line is a foreign-node decl when it matches either the Tcl or
    the Py regex above.  Empty lines and other content pass through
    unchanged.
    """
    out: list[str] = []
    for line in block.splitlines():
        if FOREIGN_NODE_DECL_RE_TCL.match(line):
            continue
        if FOREIGN_NODE_DECL_RE_PY.match(line):
            continue
        out.append(line)
    return out


def assert_partition_blocks_equivalent(
    block_a: str,
    block_b: str,
    *,
    label: str = "",
) -> None:
    """Assert two per-rank emit blocks are byte-equivalent modulo
    foreign-node decl prefixes.

    Strips lines matching ``node(tag, *xyz, ndf=...)`` patterns (Tcl
    and Py syntax — see :data:`FOREIGN_NODE_DECL_RE_TCL` /
    :data:`FOREIGN_NODE_DECL_RE_PY`) from both blocks before
    comparing.  Those are the intentional textual divergence per
    ADR 0027 INV-1.  All remaining lines — the constraint declaration
    itself, ``mp_constraint_comment``, whitespace, line endings — must
    match byte-for-byte.

    Parameters
    ----------
    block_a, block_b:
        The emitted text for the corresponding portion of each rank's
        ``partition_open(K)`` / ``partition_close()`` block (Tcl or
        Py).  Typically the cross-partition-constraint portion plus
        its foreign-node prologue.
    label:
        Optional human-readable identifier included in the assertion
        message — useful when many block pairs are checked in one
        test.

    Raises
    ------
    AssertionError
        If the stripped portions differ.  The message includes a
        ``difflib`` unified diff so the divergence is easy to read.
    """
    stripped_a = _strip_foreign_node_decls(block_a)
    stripped_b = _strip_foreign_node_decls(block_b)
    if stripped_a == stripped_b:
        return
    diff = "\n".join(
        difflib.unified_diff(
            stripped_a,
            stripped_b,
            fromfile=f"{label} block_a" if label else "block_a",
            tofile=f"{label} block_b" if label else "block_b",
            lineterm="",
        )
    )
    prefix = f"[{label}] " if label else ""
    raise AssertionError(
        f"{prefix}ADR 0027 INV-1: partition blocks diverge beyond foreign-node "
        f"decl prefixes:\n{diff}"
    )
