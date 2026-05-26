"""gmsh init refcount — nested sessions share a single gmsh runtime.

Covers the ``_gmsh_acquire`` / ``_gmsh_release`` module-level helpers in
``apeGmsh._session``.  These guard ``gmsh.initialize()`` /
``gmsh.finalize()`` so that nested sessions (e.g. a ``Part`` opened
inside an ``apeGmsh`` assembly session) don't tear down the shared
gmsh runtime out from under each other.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import Part, apeGmsh
from apeGmsh import _session as _session_mod


@pytest.fixture(autouse=True)
def _reset_refcount_and_gmsh():
    """Snapshot + restore the module refcount and finalize any stray
    gmsh runtime so each test starts from a clean baseline."""
    saved = _session_mod._GMSH_INIT_COUNT
    _session_mod._GMSH_INIT_COUNT = 0
    if gmsh.isInitialized():
        gmsh.finalize()
    try:
        yield
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
        _session_mod._GMSH_INIT_COUNT = saved


def test_single_session_init_and_finalize() -> None:
    """A solo apeGmsh session initializes gmsh on enter and finalizes
    on exit."""
    assert not gmsh.isInitialized()
    assert _session_mod._GMSH_INIT_COUNT == 0

    with apeGmsh(model_name="solo") as g:  # noqa: F841
        assert gmsh.isInitialized()
        assert _session_mod._GMSH_INIT_COUNT == 1

    assert not gmsh.isInitialized()
    assert _session_mod._GMSH_INIT_COUNT == 0


def test_nested_sessions_share_init() -> None:
    """A Part session nested inside an apeGmsh session must NOT
    finalize gmsh when the inner Part exits — the outer session is
    still using it."""
    with apeGmsh(model_name="outer") as g:  # noqa: F841
        assert gmsh.isInitialized()
        assert _session_mod._GMSH_INIT_COUNT == 1

        with Part("inner_part") as part:  # noqa: F841
            assert gmsh.isInitialized()
            assert _session_mod._GMSH_INIT_COUNT == 2

        # Inner Part exited — refcount drops but gmsh stays up
        assert gmsh.isInitialized(), (
            "inner Part.end() must NOT finalize gmsh while the outer "
            "session is still active"
        )
        assert _session_mod._GMSH_INIT_COUNT == 1

    # Outer session exited — refcount hits 0, gmsh torn down
    assert not gmsh.isInitialized()
    assert _session_mod._GMSH_INIT_COUNT == 0


def test_underflow_raises() -> None:
    """Releasing without a matching acquire is a lifecycle bug."""
    assert _session_mod._GMSH_INIT_COUNT == 0
    with pytest.raises(RuntimeError, match="session lifecycle bug"):
        _session_mod._gmsh_release()


def test_isInitialized_observation_at_each_lifecycle_point() -> None:
    """Trace ``gmsh.isInitialized()`` at every begin/end transition
    and confirm it tracks the refcount one-to-one."""
    trace: list[tuple[str, bool, int]] = []

    def snap(label: str) -> None:
        trace.append(
            (label, gmsh.isInitialized(), _session_mod._GMSH_INIT_COUNT)
        )

    snap("pre_outer")
    outer = apeGmsh(model_name="outer")
    outer.begin()
    snap("post_outer_begin")

    inner = Part("inner")
    inner.begin()
    snap("post_inner_begin")

    inner.end()
    snap("post_inner_end")

    outer.end()
    snap("post_outer_end")

    assert trace == [
        ("pre_outer", False, 0),
        ("post_outer_begin", True, 1),
        ("post_inner_begin", True, 2),
        ("post_inner_end", True, 1),
        ("post_outer_end", False, 0),
    ], trace
