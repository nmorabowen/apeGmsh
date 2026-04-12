"""
Regression tests for the "unmeshed dimension" bug in
:mod:`apeGmsh.viewers.scene.mesh_scene`.

The scenario:  user imports a solid body (dim=3 entity exists) but
only meshes the surface (``generate(dim=2)``), then opens the mesh
viewer.  Before the fix, ``_collect_entity_cells`` crashed on
``if not elem_types:`` because Gmsh returns ``elem_types`` as a
numpy ndarray and ``bool()`` on an empty ndarray raises.

These tests stub out ``gmsh.model.mesh.getElements`` to return the
exact shape Gmsh returns in the wild — an ndarray, not a list — so
the ndarray-bool path is exercised even without a live Gmsh session.
"""
from __future__ import annotations

import sys
import types
import unittest

import numpy as np


def _install_fake_pyvista() -> None:
    """mesh_scene imports ``pyvista`` at the top — stub it."""
    if "pyvista" not in sys.modules:
        sys.modules["pyvista"] = types.ModuleType("pyvista")


def _fake_gmsh_with_elements(elements_by_dim_tag: dict):
    """Build a stub ``gmsh`` module that returns canned ``getElements``
    responses.

    Parameters
    ----------
    elements_by_dim_tag : dict
        ``{(dim, tag): (elem_types, elem_tags_list, elem_node_tags_list)}``
        where ``elem_types`` is an ndarray (matching real Gmsh's return
        type).  Missing entries yield an empty ndarray response.
    """
    fake = types.ModuleType("gmsh")
    fake_model = types.SimpleNamespace()
    fake_mesh = types.SimpleNamespace()

    def getElements(dim=-1, tag=-1):  # noqa: N802  (match Gmsh API)
        key = (dim, tag)
        if key in elements_by_dim_tag:
            return elements_by_dim_tag[key]
        # Default: empty (mimics an unmeshed entity)
        return (
            np.array([], dtype=np.int32),
            [],
            [],
        )

    def getElementProperties(etype):  # noqa: N802
        # type_name, dim, order, n_nodes_per_elem, local_coords, ...
        table = {
            1: ("Line 2",       1, 1, 2, [], 2),
            2: ("Triangle 3",   2, 1, 3, [], 3),
            3: ("Quadrangle 4", 2, 1, 4, [], 4),
            4: ("Tetrahedron 4", 3, 1, 4, [], 4),
            15: ("Point",       0, 1, 1, [], 1),
        }
        return table[int(etype)]

    fake_mesh.getElements = getElements
    fake_mesh.getElementProperties = getElementProperties
    fake_model.mesh = fake_mesh
    fake.model = fake_model
    return fake


class CollectEntityCellsEmptyArrayTests(unittest.TestCase):
    """The bug: ``if not elem_types`` with an empty ndarray → ValueError."""

    def setUp(self) -> None:
        _install_fake_pyvista()
        self._saved_gmsh = sys.modules.get("gmsh")
        # Install a fake gmsh BEFORE importing mesh_scene so the
        # module-level ``import gmsh`` picks up the stub.
        sys.modules["gmsh"] = _fake_gmsh_with_elements({})
        # Purge any previously-imported mesh_scene so it re-imports
        # with the fake gmsh.
        for mod_name in list(sys.modules):
            if mod_name.startswith("apeGmsh.viewers.scene"):
                del sys.modules[mod_name]

    def tearDown(self) -> None:
        # Purge all apeGmsh modules that may have cached the fake gmsh
        for mod_name in list(sys.modules):
            if mod_name.startswith("apeGmsh"):
                del sys.modules[mod_name]
        if self._saved_gmsh is None:
            sys.modules.pop("gmsh", None)
        else:
            sys.modules["gmsh"] = self._saved_gmsh

    def test_empty_ndarray_elem_types_returns_none_not_valueerror(self):
        """Unmeshed dim=3 volume: ``getElements`` returns empty ndarray.

        Before the fix this crashed with ``ValueError: The truth value
        of an empty array is ambiguous``.  After the fix it returns
        ``None`` cleanly so ``build_mesh_scene`` can skip the entity.
        """
        from apeGmsh.viewers.scene import mesh_scene

        tag_to_idx = np.full(100, -1, dtype=np.int64)
        tag_to_idx[:10] = np.arange(10)

        # Unmeshed volume — empty ndarray for elem_types
        result = mesh_scene._collect_entity_cells(
            dim=3,
            tag=42,
            tag_to_idx=tag_to_idx,
            elem_data_out={},
            elem_to_brep_out={},
        )
        self.assertIsNone(result)

    def test_single_type_entity_is_accepted(self):
        """Sanity check: a real entity with one element type still works.

        This is the case that used to succeed *by accident* because
        numpy scalar-coerces single-element arrays to bool.  It should
        keep working after the fix.
        """
        # One triangle (gmsh type 2) with nodes 1-2-3
        sys.modules["gmsh"] = _fake_gmsh_with_elements({
            (2, 5): (
                np.array([2], dtype=np.int32),
                [np.array([100], dtype=np.int64)],
                [np.array([1, 2, 3], dtype=np.int64)],
            ),
        })
        for mod_name in list(sys.modules):
            if mod_name.startswith("apeGmsh.viewers.scene"):
                del sys.modules[mod_name]
        from apeGmsh.viewers.scene import mesh_scene

        tag_to_idx = np.full(100, -1, dtype=np.int64)
        tag_to_idx[1] = 0
        tag_to_idx[2] = 1
        tag_to_idx[3] = 2

        result = mesh_scene._collect_entity_cells(
            dim=2,
            tag=5,
            tag_to_idx=tag_to_idx,
            elem_data_out={},
            elem_to_brep_out={},
        )
        self.assertIsNotNone(result)
        cell_parts, type_parts, brep_elem_tags, dom_type = result
        self.assertEqual(len(brep_elem_tags), 1)
        self.assertEqual(brep_elem_tags[0], 100)
        self.assertEqual(dom_type, "Triangle 3")

    def test_mixed_type_entity_with_multi_element_ndarray(self):
        """A surface with both triangles and quads returns a 2-element
        ndarray for elem_types.  Before the fix this would also crash
        on the bool check (numpy can't coerce len!=1 arrays)."""
        sys.modules["gmsh"] = _fake_gmsh_with_elements({
            (2, 7): (
                np.array([2, 3], dtype=np.int32),  # triangle + quad
                [
                    np.array([200, 201], dtype=np.int64),
                    np.array([210], dtype=np.int64),
                ],
                [
                    np.array([1, 2, 3, 2, 3, 4], dtype=np.int64),
                    np.array([1, 2, 3, 4], dtype=np.int64),
                ],
            ),
        })
        for mod_name in list(sys.modules):
            if mod_name.startswith("apeGmsh.viewers.scene"):
                del sys.modules[mod_name]
        from apeGmsh.viewers.scene import mesh_scene

        tag_to_idx = np.full(100, -1, dtype=np.int64)
        for i in range(1, 5):
            tag_to_idx[i] = i - 1

        # This used to crash with:
        #   ValueError: The truth value of an array with more than one
        #   element is ambiguous. Use a.any() or a.all()
        result = mesh_scene._collect_entity_cells(
            dim=2,
            tag=7,
            tag_to_idx=tag_to_idx,
            elem_data_out={},
            elem_to_brep_out={},
        )
        self.assertIsNotNone(result)
        _, _, brep_elem_tags, _ = result
        # 2 triangles + 1 quad
        self.assertEqual(len(brep_elem_tags), 3)


if __name__ == "__main__":
    unittest.main()
