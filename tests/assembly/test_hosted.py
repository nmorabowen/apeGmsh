"""ADR 0120 D3: a hosted ``Assembly`` grafts instances onto a snapshot.

``Assembly(name, host=fem)`` + ``instance`` + couplings + ``fem()`` is the
public form of "graft this FEM into that one": the host keeps its ids and
its bare names, so declarations written for it keep working. Oracles, each
independent of the code under test:

* the host's node ids, coordinates, element ids and group names come back
  unchanged; the instance's are relocated and namespaced as in an unhosted
  assembly (``pier.Vol``);
* a bare port names a host group: ``equal_dof("Slab", "pier.bot")`` ties the
  9 co-located plate / block nodes (a 2x2 plate on a 2x2x2 block), every
  master a host node; ``embedded("pier.Vol", "Slab")`` ties the 9 plate
  nodes into the block;
* ``fem()`` is unpartitioned (the merge engine's host / module ranks are
  dropped) and cuts as one graph with ``repartition``;
* a path host reads the same snapshot as a ``FEMData`` host;
* the host passed in is not modified;
* refusals: reference nodes, ``partition_rank``, ``bridge()`` and ``h5()``
  on a hosted assembly, and a bare port that is neither a host name nor a
  reference node.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.assembly import Assembly, AssemblyError
from apeGmsh.mesh import FEMData
from tests.assembly.test_two_instances_one_tie import (
    block_fem,
    declare_block,
    plate_fem,
    write_instance,
)


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("hosted")
    plate = plate_fem(d)
    plate.to_h5(str(d / "host.h5"))
    return {
        "block": write_instance(d / "block.h5", block_fem(d), declare_block),
        "host_fem": plate,
        "host_h5": d / "host.h5",
        "dir": d,
    }


def _hosted(files, host=None) -> Assembly:
    asm = Assembly("hosted", host=files["host_fem"] if host is None else host)
    asm.instance("pier", files["block"])
    return asm


def test_the_host_keeps_its_ids_coords_and_bare_names(files):
    host = files["host_fem"]
    fem = _hosted(files).equal_dof("Slab", "pier.bot", dofs=[1, 2, 3]).fem()
    xyz = {int(n): c for n, c in zip(fem.nodes.ids, np.asarray(fem.nodes.coords))}
    for n, c in zip(host.nodes.ids, np.asarray(host.nodes.coords)):
        np.testing.assert_array_equal(xyz[int(n)], c)
    host_eids = {int(e) for g in host.elements for e in g.ids}
    assert host_eids <= {int(e) for g in fem.elements for e in g.ids}
    names = set(fem.elements.physical.names())
    assert "Slab" in names and "pier.Vol" in names
    np.testing.assert_array_equal(
        np.sort(fem.elements.select(pg="Slab").ids), np.sort(host.elements.select(pg="Slab").ids))
    assert len(fem.partitions) == 0


def test_a_bare_port_names_a_host_group(files):
    host_ids = {int(n) for n in files["host_fem"].nodes.ids}
    fem = _hosted(files).equal_dof("Slab", "pier.bot", dofs=[1, 2, 3]).fem()
    pairs = [r for r in fem.nodes.constraints if getattr(r, "kind", "") == "equal_dof"]
    assert len(pairs) == 9
    assert all(int(r.master_node) in host_ids for r in pairs)
    assert all(int(r.slave_node) not in host_ids for r in pairs)

    emb = _hosted(files).embedded("pier.Vol", "Slab", tolerance=1e-6,
                                  stiffness=1.0e9).fem()
    tied = [r for r in emb.elements.constraints if getattr(r, "kind", "") == "embedded"]
    assert sorted(int(r.slave_node) for r in tied) == sorted(
        int(n) for n in files["host_fem"].nodes.select(pg="Slab").ids)


def test_fem_is_unpartitioned_and_cuts_as_one_graph(files):
    fem = _hosted(files).equal_dof("Slab", "pier.bot", dofs=[1, 2, 3]).fem()
    two = fem.repartition(2)
    assert len(two.partitions) == 2
    held = set()
    for p in two.partitions:
        held |= {int(e) for e in p.element_ids}
    assert held == {int(e) for g in fem.elements for e in g.ids}


def test_a_path_host_is_the_snapshot_host(files):
    a = _hosted(files).fem()
    b = _hosted(files, host=files["host_h5"]).fem()
    np.testing.assert_array_equal(np.asarray(a.nodes.ids), np.asarray(b.nodes.ids))
    np.testing.assert_array_equal(np.asarray(a.nodes.coords), np.asarray(b.nodes.coords))


def test_the_host_is_not_modified(files):
    host = files["host_fem"]
    ids = np.asarray(host.nodes.ids).copy()
    n_cons = len(list(host.nodes.constraints))
    _hosted(files).equal_dof("Slab", "pier.bot", dofs=[1, 2, 3]).fem().repartition(2)
    np.testing.assert_array_equal(np.asarray(host.nodes.ids), ids)
    assert len(host.partitions) == 0
    assert len(list(host.nodes.constraints)) == n_cons


def test_the_bridge_builds_on_the_hosted_fem(files, tmp_path: Path):
    from apeGmsh.opensees import apeSees

    fem = _hosted(files).equal_dof("Slab", "pier.bot", dofs=[1, 2, 3]).fem()
    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=30e3, nu=0.2, h=0.5)
    ops.element.ShellMITC4(pg="Slab", section=sec)
    steel = ops.nDMaterial.ElasticIsotropic(E=200e3, nu=0.3)
    ops.element.stdBrick(pg="pier.Vol", material=steel)
    deck = tmp_path / "hosted.tcl"
    ops.tcl(str(deck))
    lines = deck.read_text().splitlines()
    assert sum(ln.startswith("element ShellMITC4") for ln in lines) == 4
    assert sum(ln.startswith("element stdBrick") for ln in lines) == 8
    assert sum(ln.startswith("equalDOF") for ln in lines) == 9


@pytest.mark.parametrize("call, match", [
    (lambda a: a.node("ref", (0.0, 0.0, 0.0)), "no reference nodes"),
    (lambda a: a.instance("p2", a._instances[0].source, partition_rank=0),
     "no partition_rank"),
    (lambda a: a.bridge(ndm=3, ndf=6), "built with fem"),
    (lambda a: a.h5("x.h5"), "no /assembly archive"),
    (lambda a: a.equal_dof("Nope", "pier.bot"), "names no assembly object"),
])
def test_refusals(files, call, match):
    asm = _hosted(files)
    with pytest.raises(AssemblyError, match=match):
        call(asm)


def test_a_bad_host_is_refused(files, tmp_path: Path):
    with pytest.raises(AssemblyError, match="no file"):
        Assembly("x", host=tmp_path / "missing.h5")
    with pytest.raises(AssemblyError, match="FEMData or a model.h5"):
        Assembly("x", host=3)  # type: ignore[arg-type]
    assert isinstance(FEMData.from_h5(str(files["host_h5"])), FEMData)
