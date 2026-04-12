"""Tests for infrastructure: _types, _logging, _session, _helpers."""
import io
import sys

import gmsh
import pytest


# ---------------------------------------------------------------------------
# SessionProtocol
# ---------------------------------------------------------------------------

def test_session_protocol_runtime_check(g):
    """apeGmsh session satisfies SessionProtocol at runtime."""
    from apeGmsh._types import SessionProtocol
    assert isinstance(g, SessionProtocol)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def test_session_begin_end():
    """begin() opens session, end() closes it."""
    from apeGmsh import apeGmsh
    s = apeGmsh(model_name="lifecycle_test")
    assert not s.is_active
    s.begin()
    assert s.is_active
    s.end()
    assert not s.is_active


def test_session_double_begin_raises():
    """Calling begin() twice raises RuntimeError."""
    from apeGmsh import apeGmsh
    s = apeGmsh(model_name="double_begin")
    s.begin()
    try:
        with pytest.raises(RuntimeError, match="already open"):
            s.begin()
    finally:
        s.end()


def test_session_context_manager():
    """with apeGmsh() as g: auto begin/end."""
    from apeGmsh import apeGmsh
    with apeGmsh(model_name="ctx_test") as s:
        assert s.is_active
    assert not s.is_active


# ---------------------------------------------------------------------------
# _HasLogging mixin
# ---------------------------------------------------------------------------

def test_has_logging_verbose_true(g_verbose, capsys):
    """_log() prints when _verbose=True."""
    g_verbose.model._log("hello from test")
    captured = capsys.readouterr()
    assert "[Model] hello from test" in captured.out


def test_has_logging_verbose_false(g, capsys):
    """_log() is silent when _verbose=False."""
    g.model._log("should not appear")
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# _helpers: resolve_dim, as_dimtags
# ---------------------------------------------------------------------------

def test_resolve_dim_finds_existing(gmsh_session):
    """resolve_dim returns the correct dimension for an existing entity."""
    from apeGmsh.core._helpers import resolve_dim
    tag = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    assert resolve_dim(tag, default_dim=2) == 3  # box is dim=3


def test_resolve_dim_returns_default(gmsh_session):
    """resolve_dim returns default_dim when tag not found."""
    from apeGmsh.core._helpers import resolve_dim
    assert resolve_dim(99999, default_dim=2) == 2


def test_as_dimtags_int(gmsh_session):
    """as_dimtags with a single int creates [(default_dim, tag)]."""
    from apeGmsh.core._helpers import as_dimtags
    tag = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    result = as_dimtags(tag, default_dim=3)
    assert result == [(3, tag)]


def test_as_dimtags_list(gmsh_session):
    """as_dimtags with a list resolves each tag."""
    from apeGmsh.core._helpers import as_dimtags
    t1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    t2 = gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    result = as_dimtags([t1, t2], default_dim=3)
    assert len(result) == 2
    assert all(d == 3 for d, _ in result)


# ---------------------------------------------------------------------------
# _temporary_tolerance context manager
# ---------------------------------------------------------------------------

def test_temporary_tolerance(gmsh_session):
    """Options overridden inside context, restored after."""
    from apeGmsh.core._model_queries import _temporary_tolerance

    original = gmsh.option.getNumber("Geometry.Tolerance")

    with _temporary_tolerance(0.123):
        inside = gmsh.option.getNumber("Geometry.Tolerance")
        assert abs(inside - 0.123) < 1e-12

    after = gmsh.option.getNumber("Geometry.Tolerance")
    assert abs(after - original) < 1e-12


def test_temporary_tolerance_none_is_noop(gmsh_session):
    """tolerance=None leaves options unchanged."""
    from apeGmsh.core._model_queries import _temporary_tolerance

    original = gmsh.option.getNumber("Geometry.Tolerance")

    with _temporary_tolerance(None):
        inside = gmsh.option.getNumber("Geometry.Tolerance")
        assert abs(inside - original) < 1e-12


# ---------------------------------------------------------------------------
# Fixture for verbose session
# ---------------------------------------------------------------------------

@pytest.fixture
def g_verbose():
    """Full apeGmsh session with verbose=True."""
    from apeGmsh import apeGmsh
    session = apeGmsh(model_name="test_verbose", verbose=True)
    session.begin()
    yield session
    session.end()
