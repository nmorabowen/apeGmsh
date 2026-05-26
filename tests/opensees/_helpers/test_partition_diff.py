"""Tests for :mod:`tests.opensees._helpers.partition_diff`.

The helper operates on plain text — it does not import openseespy or
any other runtime dependency, so these tests run cleanly on the
no-openseespy CI gate.
"""
from __future__ import annotations

import pytest

from tests.opensees._helpers.partition_diff import (
    FOREIGN_NODE_DECL_RE_PY,
    FOREIGN_NODE_DECL_RE_TCL,
    assert_partition_blocks_equivalent,
)


# ---------------------------------------------------------------------
# Stripping rules — Tcl + Py foreign-node decl recognition
# ---------------------------------------------------------------------


def test_strips_foreign_ndf_decl_from_tcl_blocks() -> None:
    """Two Tcl blocks identical except for a foreign ``-ndf`` on one
    node decl — helper accepts them as equivalent.
    """
    block_a = (
        "node 4 1.0 2.0 3.0 -ndf 6\n"
        "rigidLink beam 2 4\n"
    )
    block_b = (
        "node 2 0.0 0.0 0.0 -ndf 6\n"
        "rigidLink beam 2 4\n"
    )
    # Both foreign-decl lines get stripped; what remains is identical.
    assert_partition_blocks_equivalent(block_a, block_b)


def test_strips_foreign_ndf_decl_from_py_blocks() -> None:
    """Same as Tcl but for ``ops.node(..., '-ndf', K)`` syntax."""
    block_a = (
        "ops.node(4, 1.0, 2.0, 3.0, '-ndf', 6)\n"
        "ops.rigidLink('beam', 2, 4)\n"
    )
    block_b = (
        "ops.node(2, 0.0, 0.0, 0.0, '-ndf', 6)\n"
        "ops.rigidLink('beam', 2, 4)\n"
    )
    assert_partition_blocks_equivalent(block_a, block_b)


def test_accepts_identical_blocks() -> None:
    """Trivial case — no foreign decls anywhere; blocks are exactly
    equal; helper passes silently.
    """
    block = (
        "# floor diaphragm\n"
        "rigidDiaphragm 3 2 4 6 8\n"
    )
    assert_partition_blocks_equivalent(block, block)


def test_rejects_when_constraint_line_differs() -> None:
    """If the constraint line itself differs across blocks, helper
    raises with a unified diff naming the divergence.
    """
    block_a = (
        "node 4 1.0 2.0 3.0 -ndf 6\n"
        "rigidLink beam 2 4\n"
    )
    block_b = (
        "node 2 0.0 0.0 0.0 -ndf 6\n"
        "rigidLink beam 2 5\n"  # slave tag differs
    )
    with pytest.raises(AssertionError) as exc_info:
        assert_partition_blocks_equivalent(block_a, block_b)
    msg = str(exc_info.value)
    assert "INV-1" in msg, f"assertion msg should cite INV-1; got {msg!r}"
    assert "rigidLink beam 2 4" in msg, (
        f"unified diff should include the diverging rank-A line; got {msg!r}"
    )
    assert "rigidLink beam 2 5" in msg, (
        f"unified diff should include the diverging rank-B line; got {msg!r}"
    )


def test_rejects_when_non_node_line_differs() -> None:
    """A ``mp_constraint_comment`` (or any other non-foreign-decl line)
    differing across blocks must trigger a failure — the helper only
    strips foreign-node decls, nothing else.
    """
    block_a = (
        "# constraint: floor_3\n"
        "rigidDiaphragm 3 2 4\n"
    )
    block_b = (
        "# constraint: floor_4\n"  # comment differs
        "rigidDiaphragm 3 2 4\n"
    )
    with pytest.raises(AssertionError) as exc_info:
        assert_partition_blocks_equivalent(block_a, block_b)
    msg = str(exc_info.value)
    assert "INV-1" in msg
    assert "floor_3" in msg or "floor_4" in msg, (
        f"diff should expose the comment divergence; got {msg!r}"
    )


def test_label_appears_in_assertion_message() -> None:
    """The optional ``label`` is prepended to the failure message for
    easier identification in multi-pair tests.
    """
    with pytest.raises(AssertionError) as exc_info:
        assert_partition_blocks_equivalent(
            "rigidLink beam 2 4\n",
            "rigidLink beam 2 5\n",
            label="cross_link_test",
        )
    assert "cross_link_test" in str(exc_info.value)


# ---------------------------------------------------------------------
# Regex sanity — exact format produced by TclEmitter / PyEmitter
# ---------------------------------------------------------------------


def test_tcl_regex_matches_emitter_format() -> None:
    """Sanity check the Tcl regex against the exact format
    ``TclEmitter.node(..., ndf=K)`` produces (via ``_join``).
    """
    assert FOREIGN_NODE_DECL_RE_TCL.match("node 4 1.0 2.0 3.0 -ndf 6")
    # Indented (inside a partition_open block).
    assert FOREIGN_NODE_DECL_RE_TCL.match("    node 4 1.0 2.0 3.0 -ndf 6")
    # Native decl (no -ndf) must NOT match.
    assert not FOREIGN_NODE_DECL_RE_TCL.match("node 4 1.0 2.0 3.0")
    # Other commands must NOT match.
    assert not FOREIGN_NODE_DECL_RE_TCL.match("rigidLink beam 2 4")


def test_py_regex_matches_emitter_format() -> None:
    """Sanity check the Py regex against the exact format
    ``PyEmitter.node(..., ndf=K)`` produces (via ``_ops_call``).
    """
    assert FOREIGN_NODE_DECL_RE_PY.match(
        "ops.node(4, 1.0, 2.0, 3.0, '-ndf', 6)"
    )
    # Indented.
    assert FOREIGN_NODE_DECL_RE_PY.match(
        "    ops.node(4, 1.0, 2.0, 3.0, '-ndf', 6)"
    )
    # Double-quoted variant (defensive — emitter uses single quotes,
    # but we want robustness against hand-written input).
    assert FOREIGN_NODE_DECL_RE_PY.match(
        'ops.node(4, 1.0, 2.0, 3.0, "-ndf", 6)'
    )
    # Native decl must NOT match.
    assert not FOREIGN_NODE_DECL_RE_PY.match("ops.node(4, 1.0, 2.0, 3.0)")
    # Other ops calls must NOT match.
    assert not FOREIGN_NODE_DECL_RE_PY.match("ops.rigidLink('beam', 2, 4)")
