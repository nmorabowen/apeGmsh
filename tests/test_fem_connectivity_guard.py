"""Regression: FEM extraction must never emit element connectivity that
references a node absent from the mesh (the canonical symptom is node
tag ``0``).

Discovered while building a shell-on-solid model via the Part registry
(PR #474 collateral).  The original report blamed ``g.parts.add`` +
``renumber``, but the true trigger is a **global 3-D recombine**:
``g.mesh.structured.recombine()`` (Gmsh's ``mesh.recombine()``) is a
2-D surface operation.  Running it while 3-D tetrahedra exist deletes
interior nodes and leaves the tets that referenced them pointing at
node ``0`` — independent of Parts or renumber, and visible in the raw
Gmsh connectivity *before* any apeGmsh remap.

Passing that connectivity to OpenSees produces the opaque
``Domain::addElement - ... no Node 0 exists in the domain`` abort.
``_validate_connectivity`` (in ``mesh/_fem_extract.py``) now fails loud
at extraction instead, with a message naming the cause and the
transfinite remedy.

NOTE on test design: actually *running* the global 3-D recombine
corrupts Gmsh's process heap (glibc ``corrupted double-linked list`` ->
SIGABRT in a later test's teardown on Linux CI), so we never execute it
in-process.  The guard logic is unit-tested directly with synthetic
connectivity, and the source-level warning is tested with the native
``recombine`` call patched to a no-op.
"""
import gmsh
import numpy as np
import pytest

from apeGmsh.mesh._fem_extract import _validate_connectivity


# ---------------------------------------------------------------------------
# Guard logic — unit tests on synthetic connectivity (no Gmsh recombine).
# ---------------------------------------------------------------------------

def _group(gmsh_name: str, conn: list[list[int]]) -> dict:
    """Minimal raw-group dict in the shape _validate_connectivity reads."""
    return {'gmsh_name': gmsh_name, 'conn': np.asarray(conn, dtype=np.int64)}


def test_validate_connectivity_accepts_clean_mesh():
    node_tags = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    groups = {4: _group("Tetrahedron 4", [[1, 2, 3, 4], [2, 3, 4, 5]])}
    # Must not raise.
    _validate_connectivity(groups, node_tags)


def test_validate_connectivity_rejects_node_zero():
    """The canonical corruption symptom: a tet referencing node 0."""
    node_tags = np.array([1, 2, 3, 4], dtype=np.int64)
    groups = {4: _group("Tetrahedron 4", [[1, 2, 3, 4], [1, 2, 0, 4]])}
    with pytest.raises(ValueError) as exc:
        _validate_connectivity(groups, node_tags)
    msg = str(exc.value).lower()
    assert "node tag 0" in msg          # symptom called out by value
    assert "recombine" in msg           # cause + remedy in the message


def test_validate_connectivity_rejects_nonzero_dangling():
    """A reference to a real-looking-but-absent tag also fails loud."""
    node_tags = np.array([1, 2, 3, 4], dtype=np.int64)
    groups = {4: _group("Tetrahedron 4", [[1, 2, 3, 99]])}
    with pytest.raises(ValueError, match="99"):
        _validate_connectivity(groups, node_tags)


def test_validate_connectivity_empty_mesh_is_noop():
    _validate_connectivity({}, np.array([], dtype=np.int64))


# ---------------------------------------------------------------------------
# Clean path: two non-fragmented Parts extract valid connectivity.
# ---------------------------------------------------------------------------

def test_two_part_nonfragmented_extracts_valid_connectivity(g):
    """Two stacked Parts, NOT fragmented, tet mesh, renumbered — every
    connectivity tag is a real node (no tag 0, nothing dangling)."""
    # Footing (wide, flat) + wall (narrow, tall) sitting on its top —
    # two separate part instances, intentionally NOT fragment_all'd so
    # the interface stays non-conformal (each body keeps its own nodes).
    with g.parts.part("foot"):
        g.model.geometry.add_box(0, 0, 0, 2, 2, 0.5)
    with g.parts.part("wal"):
        g.model.geometry.add_box(0.5, 0.5, 0.5, 1, 1, 2)

    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber(dim=3, method="simple", base=1)

    fem = g.mesh.queries.get_fem_data(dim=3)

    ids = np.asarray([int(x) for x in fem.nodes.ids])
    assert ids.min() == 1                       # dense 1-based after renumber
    node_ids = set(int(x) for x in fem.nodes.ids)
    n_zero = n_dangling = 0
    for grp in fem.elements._groups.values():
        c = grp.connectivity
        n_zero += int((c == 0).sum())
        n_dangling += len(set(int(x) for x in c.ravel()) - node_ids)
    assert n_zero == 0, "connectivity contains node tag 0"
    assert n_dangling == 0, "connectivity references a non-existent node"


# ---------------------------------------------------------------------------
# Source-level warning: recombine() warns when 3-D elements are present.
# The native recombine() is patched to a no-op — actually running it
# corrupts the Gmsh heap (see module docstring).
# ---------------------------------------------------------------------------

def test_recombine_warns_when_3d_elements_present(g, monkeypatch):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(3)

    monkeypatch.setattr(gmsh.model.mesh, "recombine", lambda *a, **k: None)
    with pytest.warns(UserWarning, match="(?i)recombine.*3-D|3-D.*recombine"):
        g.mesh.structured.recombine()


def test_recombine_silent_on_pure_2d_mesh(g, monkeypatch):
    """No 3-D elements -> no warning (recombine is legitimate in 2-D)."""
    import warnings

    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="s")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(2)

    monkeypatch.setattr(gmsh.model.mesh, "recombine", lambda *a, **k: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any UserWarning -> failure
        g.mesh.structured.recombine()
