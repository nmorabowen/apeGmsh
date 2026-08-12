"""ADR 0093 S3 — loud h5 guard for ``fem.elements.interfaces``.

There is no persisted representation for :class:`InterfaceRecord` yet
(no payload dtype, no ``/interfaces`` group, no compose support — that
lands in ADR 0093 S6). ``FEMData.to_h5`` must refuse loudly when
``fem.elements.interfaces`` is non-empty rather than silently write a
file that reads back with the interface records missing (the failure
class ADR 0093 S3 exists to kill, per the ``7c7883b4`` / ``746b513c``
history of compose silently dropping contact / embed-tie records
before their own persistence landed).

Built directly from a minimal in-memory FEMData (no gmsh session
needed) — lighter than the real-mesh fixture style of
``test_contact_h5_roundtrip.py`` since the guard fires before any real
mesh content is read.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import InterfaceRecord, NormalLaw
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite


def _minimal_fem(interfaces: list | None = None) -> FEMData:
    node_ids = np.array([1, 2], dtype=np.int64)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    elem_ids = np.array([10], dtype=np.int64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info,
        ids=elem_ids,
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
        interfaces=interfaces,
    )
    info = MeshInfo(n_nodes=2, n_elems=1, bandwidth=1, types=[line_info])
    return FEMData(nodes=nodes, elements=elements, info=info)


def _interface_record() -> InterfaceRecord:
    return InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        master_node=1,
        slave_node=2,
        backing_element=10,
        orient=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
        a_trib=0.5,
        normal_law=NormalLaw(kind="ent", k_per_area=1.0e6),
    )


def test_no_interfaces_saves_fine(tmp_path) -> None:
    """Sanity check: the guard doesn't fire on an ordinary model."""
    fem = _minimal_fem()
    fem.to_h5(str(tmp_path / "plain.h5"))


def test_interfaces_present_refuses_to_save(tmp_path) -> None:
    fem = _minimal_fem(interfaces=[_interface_record()])
    assert len(fem.elements.interfaces) == 1
    path = tmp_path / "should_not_exist.h5"
    with pytest.raises(NotImplementedError, match="ADR 0093"):
        fem.to_h5(str(path))
    # The refusal must fire BEFORE ``h5py.File(path, "w")`` truncates the
    # target — no file (not even a partial ``['meta']``-only stub) is
    # left behind.
    assert not path.exists()


def test_refused_save_does_not_corrupt_an_existing_file(tmp_path) -> None:
    """Probed failure mode: overwriting an existing ``model.h5`` with a
    FEMData that carries interfaces must leave the existing file
    untouched, not truncate it into a ``['meta']``-only stub that
    ``from_h5`` then fails on with a raw ``KeyError``.
    """
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5

    path = tmp_path / "model.h5"
    good_fem = _minimal_fem()
    good_fem.to_h5(str(path))

    bad_fem = _minimal_fem(interfaces=[_interface_record()])
    with pytest.raises(NotImplementedError, match="ADR 0093"):
        bad_fem.to_h5(str(path))

    # The file on disk must still be the ORIGINAL good save — readable,
    # not a truncated meta-only stub.
    reloaded = read_fem_h5(str(path))
    assert reloaded.nodes.ids.size == 2


def test_interfaces_present_refuses_neutral_zone_into_group(tmp_path) -> None:
    """The same guard covers the composed-results embed path
    (``write_neutral_zone_into_group``, ADR 0020)."""
    import h5py

    from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group

    fem = _minimal_fem(interfaces=[_interface_record()])
    path = tmp_path / "embedded.h5"
    with h5py.File(str(path), "w") as f:
        group = f.create_group("model")
        with pytest.raises(NotImplementedError, match="ADR 0093"):
            write_neutral_zone_into_group(fem, group)
