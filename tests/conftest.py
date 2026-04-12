"""Shared test fixtures for apeGmsh test suite."""
import pytest
import gmsh


@pytest.fixture
def gmsh_session():
    """Bare Gmsh session for low-level tests (Labels, helpers)."""
    gmsh.initialize()
    gmsh.model.add("test")
    yield
    gmsh.finalize()


@pytest.fixture
def g():
    """Full apeGmsh session with all composites wired up."""
    from apeGmsh import apeGmsh
    session = apeGmsh(model_name="test", verbose=False)
    session.begin()
    yield session
    session.end()
