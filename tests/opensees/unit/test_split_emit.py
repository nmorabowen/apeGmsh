"""Unit tests for split (mode A) emit — ADR 0043 slice 1.1.

Exercises ``apeSees(fem).tcl(path, split=True)`` /
``.py(path, split=True)``: per-module fragment files + a driver
that wires them, the byte-identity of the default single-file path,
and the fail-loud guards (non-composed / all-host / bad value).

The composed model is hand-built at the broker level (a ``FEMData``
with per-row ``module_label`` arrays, exactly what ``g.compose``
produces) so the test isolates the split seam from the compose
pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterfaceRecord,
    NormalLaw,
    TangentialLaw,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)

from tests.opensees.fixtures.fem_stub import make_two_module_frame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _composed_fem(
    *,
    node_module_label: "list[str] | None",
    elem_module_label: "list[str] | None",
    interfaces: "list[InterfaceRecord] | None" = None,
) -> FEMData:
    """A 6-node / 3-element FEM with per-row compose labels.

    Nodes 1..6, Line2 elements 10..12.  ``node_module_label`` /
    ``elem_module_label`` set the per-row provenance (``None`` →
    uncomposed broker, i.e. no module-label metadata at all).
    """
    node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    node_coords = np.array(
        [[float(i), 0.0, 0.0] for i in range(node_ids.size)],
        dtype=np.float64,
    )
    elem_ids = np.array([10, 11, 12], dtype=np.int64)
    conn = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=3,
    )
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )

    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
        module_label=(
            None if node_module_label is None
            else np.array(node_module_label, dtype=object)
        ),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
        module_label=(
            None if elem_module_label is None
            else {1: np.array(elem_module_label, dtype=object)}
        ),
        interfaces=interfaces,
    )
    info = MeshInfo(n_nodes=6, n_elems=3, bandwidth=1, types=[line_info])
    return FEMData(nodes=nodes, elements=elements, info=info)


def _bridge(fem: FEMData) -> apeSees:
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=3)
    return ops


# Host rows ("") + module A + module B.
_NODE_LABELS = ["", "", "A", "A", "B", "B"]
_ELEM_LABELS = ["", "A", "B"]


# ---------------------------------------------------------------------------
# Split — Tcl
# ---------------------------------------------------------------------------

def test_tcl_split_writes_fragments_and_driver(tmp_path: Path) -> None:
    fem = _composed_fem(
        node_module_label=_NODE_LABELS, elem_module_label=_ELEM_LABELS,
    )
    driver = tmp_path / "deck.tcl"
    _bridge(fem).tcl(str(driver), split=True)

    parts = tmp_path / "parts"
    assert (parts / "host.tcl").exists()
    assert (parts / "A.tcl").exists()
    assert (parts / "B.tcl").exists()

    # Module A fragment carries only A's nodes (3, 4).
    a_body = (parts / "A.tcl").read_text(encoding="utf-8")
    assert "node 3 " in a_body
    assert "node 4 " in a_body
    assert "node 1 " not in a_body
    assert "node 5 " not in a_body

    # Host fragment carries nodes 1, 2.
    host_body = (parts / "host.tcl").read_text(encoding="utf-8")
    assert "node 1 " in host_body
    assert "node 2 " in host_body


def test_tcl_driver_sources_fragments_and_holds_no_nodes(
    tmp_path: Path,
) -> None:
    fem = _composed_fem(
        node_module_label=_NODE_LABELS, elem_module_label=_ELEM_LABELS,
    )
    driver = tmp_path / "deck.tcl"
    _bridge(fem).tcl(str(driver), split=True)

    text = driver.read_text(encoding="utf-8")
    # The model directive stays in the driver.
    assert "model BasicBuilder" in text
    # One source line per module, none of the node bulk.
    assert text.count("parts host.tcl]") == 1
    assert text.count("parts A.tcl]") == 1
    assert text.count("parts B.tcl]") == 1
    assert "node " not in text


def test_tcl_fragments_partition_all_nodes_exactly_once(
    tmp_path: Path,
) -> None:
    fem = _composed_fem(
        node_module_label=_NODE_LABELS, elem_module_label=_ELEM_LABELS,
    )
    driver = tmp_path / "deck.tcl"
    _bridge(fem).tcl(str(driver), split=True)

    parts = tmp_path / "parts"
    all_node_lines: list[str] = []
    for frag in ("host.tcl", "A.tcl", "B.tcl"):
        for ln in (parts / frag).read_text(encoding="utf-8").splitlines():
            if ln.startswith("node "):
                all_node_lines.append(ln)
    tags = sorted(int(ln.split()[1]) for ln in all_node_lines)
    assert tags == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Split — Py
# ---------------------------------------------------------------------------

def test_py_split_fragments_expose_build_and_driver_calls(
    tmp_path: Path,
) -> None:
    fem = _composed_fem(
        node_module_label=_NODE_LABELS, elem_module_label=_ELEM_LABELS,
    )
    driver = tmp_path / "deck.py"
    _bridge(fem).py(str(driver), split=True)

    parts = tmp_path / "parts"
    a_body = (parts / "A.py").read_text(encoding="utf-8")
    assert "def build(ops):" in a_body
    assert "    ops.node(3," in a_body

    text = driver.read_text(encoding="utf-8")
    assert "import openseespy.opensees as ops" in text
    # Fragments are loaded by explicit file path (no sys.path mutation,
    # no bare-import name collisions) and their build(ops) is called.
    assert "_sys.path" not in text
    assert "'A.py').build(ops)" in text
    assert "'B.py').build(ops)" in text
    assert "ops.node(" not in text  # node bulk lives in fragments


# ---------------------------------------------------------------------------
# Element partitioning — a single PG fanned across two modules
# ---------------------------------------------------------------------------

def test_tcl_split_routes_elements_to_owning_module(tmp_path: Path) -> None:
    """``ops.element(pg="Cols")`` fans over eids 1 (module A) and 2
    (module B); each ``element`` line lands in its owning fragment."""
    fem = make_two_module_frame()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    driver = tmp_path / "deck.tcl"
    ops.tcl(str(driver), split=True)

    parts = tmp_path / "parts"
    a_body = (parts / "A.tcl").read_text(encoding="utf-8")
    b_body = (parts / "B.tcl").read_text(encoding="utf-8")

    # eid 1 (conn 1-2) lands in fragment A; eid 2 (conn 3-4) in B.
    # OpenSees element tags are allocator-assigned (not == eid), so
    # the connectivity is the stable identity to assert on.
    a_elems = [ln for ln in a_body.splitlines()
               if ln.startswith("element elasticBeamColumn")]
    b_elems = [ln for ln in b_body.splitlines()
               if ln.startswith("element elasticBeamColumn")]
    assert len(a_elems) == 1 and a_elems[0].split()[3:5] == ["1", "2"]
    assert len(b_elems) == 1 and b_elems[0].split()[3:5] == ["3", "4"]
    # Each fragment also carries its own nodes.
    assert "node 1 " in a_body and "node 2 " in a_body
    assert "node 3 " in b_body and "node 4 " in b_body
    # The driver holds neither node nor element bulk.
    text = driver.read_text(encoding="utf-8")
    assert "node " not in text
    assert "element " not in text
    # The geomTransf definition stays driver-side.
    assert "geomTransf Linear" in text


# ---------------------------------------------------------------------------
# Side-lists that land driver-side — g.constraints.interface (ADR 0093)
# ---------------------------------------------------------------------------

def _interface_rec(master: int, slave: int) -> InterfaceRecord:
    """An equal-ndf pair coupling module A's node to module B's — the
    cross-module coupling a composed split deck exists to express."""
    return InterfaceRecord(
        kind=ConstraintKind.INTERFACE,
        name="RockLiner",
        master_node=master,
        slave_node=slave,
        backing_element=10,
        orient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        a_trib=0.25,
        normal_law=NormalLaw(kind="ent", k_per_area=1.0e6),
        tangential_law=TangentialLaw(
            kind="epp", k_per_area=1.0e5, tau_b=250.0),
    )


def _interface_bridge(fem: FEMData) -> apeSees:
    # ndm=2: interface records only exist for 2D line masters (D2).
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    return ops


def test_split_emits_interfaces_into_the_driver(tmp_path: Path) -> None:
    """Regression (ADR 0093 S8 adversarial review, finding 5): the split
    path emitted every other side-list but never ``emit_interfaces``, so
    a model carrying ``g.constraints.interface()`` exported with
    ``split="parts"`` silently lost the whole interface."""
    fem = _composed_fem(
        node_module_label=_NODE_LABELS,
        elem_module_label=_ELEM_LABELS,
        interfaces=[_interface_rec(3, 5)],
    )
    driver = tmp_path / "deck.tcl"
    _interface_bridge(fem).tcl(str(driver), split=True)

    text = driver.read_text(encoding="utf-8")
    zl = [ln for ln in text.splitlines()
          if ln.startswith("element zeroLength ")]
    assert len(zl) == 1
    # INV-1 node order + the pair's own frame survive the split path.
    tok = zl[0].split()
    assert tok[3] == "3" and tok[4] == "5"
    i = tok.index("-orient")
    assert [float(v) for v in tok[i + 1:i + 7]] == [
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # Both tributary-scaled uniaxials ride along.
    assert "uniaxialMaterial ENT " in text
    assert "uniaxialMaterial ElasticPP " in text

    # The unit is cross-module, so it belongs driver-side — never in a
    # fragment, which is sourced before the other module's nodes exist.
    parts = tmp_path / "parts"
    for frag in ("host.tcl", "A.tcl", "B.tcl"):
        body = (parts / frag).read_text(encoding="utf-8")
        assert "zeroLength" not in body
        assert "uniaxialMaterial" not in body


def test_split_and_single_file_agree_on_the_interface_block(
    tmp_path: Path,
) -> None:
    """The split driver's interface lines are the same lines the
    single-file deck emits — same order, same tags."""
    def _iface_lines(deck: Path) -> "list[str]":
        return [
            ln for ln in deck.read_text(encoding="utf-8").splitlines()
            if ln.startswith(("element zeroLength ", "uniaxialMaterial "))
        ]

    def _fem() -> FEMData:
        return _composed_fem(
            node_module_label=_NODE_LABELS,
            elem_module_label=_ELEM_LABELS,
            interfaces=[_interface_rec(3, 5), _interface_rec(4, 6)],
        )

    single = tmp_path / "single.tcl"
    _interface_bridge(_fem()).tcl(str(single))
    split = tmp_path / "deck.tcl"
    _interface_bridge(_fem()).tcl(str(split), split=True)

    assert _iface_lines(split) == _iface_lines(single)


# ---------------------------------------------------------------------------
# Byte-identity of the default (single-file) path
# ---------------------------------------------------------------------------

def test_default_tcl_path_unchanged(tmp_path: Path) -> None:
    """``split=False`` writes one self-contained deck with all nodes
    and no fragment wiring — the pre-0043 behaviour."""
    fem = _composed_fem(
        node_module_label=_NODE_LABELS, elem_module_label=_ELEM_LABELS,
    )
    deck = tmp_path / "single.tcl"
    _bridge(fem).tcl(str(deck))

    text = deck.read_text(encoding="utf-8")
    assert "source " not in text
    for tag in range(1, 7):
        assert f"node {tag} " in text


def test_default_py_path_unchanged(tmp_path: Path) -> None:
    fem = _composed_fem(
        node_module_label=_NODE_LABELS, elem_module_label=_ELEM_LABELS,
    )
    deck = tmp_path / "single.py"
    _bridge(fem).py(str(deck))

    text = deck.read_text(encoding="utf-8")
    assert "build(ops)" not in text
    assert "ops.node(1," in text


# ---------------------------------------------------------------------------
# Fail-loud guards
# ---------------------------------------------------------------------------

def test_split_rejects_non_composed_model(tmp_path: Path) -> None:
    fem = _composed_fem(node_module_label=None, elem_module_label=None)
    with pytest.raises(BridgeError, match="composed model"):
        _bridge(fem).tcl(str(tmp_path / "deck.tcl"), split=True)


def test_split_rejects_all_host_model(tmp_path: Path) -> None:
    fem = _composed_fem(
        node_module_label=["", "", "", "", "", ""],
        elem_module_label=["", "", ""],
    )
    with pytest.raises(BridgeError, match="at least one composed source"):
        _bridge(fem).tcl(str(tmp_path / "deck.tcl"), split=True)


def test_split_rejects_partitioned_model(tmp_path: Path) -> None:
    fem = make_two_module_frame()
    fem.set_partitions([(0, [1, 2], [1]), (1, [3, 4], [2])])
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    with pytest.raises(BridgeError, match="partitioned"):
        ops.tcl(str(tmp_path / "deck.tcl"), split=True)


def test_split_rejects_element_label_node_mismatch(tmp_path: Path) -> None:
    """Finding B: an element whose module label disagrees with its
    connectivity nodes' module must fail loud, not silently route into
    the wrong (host-first) fragment referencing undefined nodes."""
    fem = make_two_module_frame()
    # eid 1 (conn nodes 1,2 — both module "A") deliberately mislabelled
    # as host ("") — the partial-metadata hazard.
    fem.elements._module_label_by_id = {1: "", 2: "B"}
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=t,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    with pytest.raises(BridgeError, match="module label"):
        ops.tcl(str(tmp_path / "deck.tcl"), split=True)


# ---------------------------------------------------------------------------
# fix / mass routing into the owning fragment
# ---------------------------------------------------------------------------

def test_tcl_split_routes_fix_to_owning_module(tmp_path: Path) -> None:
    """``fix(pg="Base")`` covers node 1 (module A) and node 3 (module
    B); each ``fix`` line lands in its owning fragment, not the driver."""
    fem = make_two_module_frame()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=t,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))  # Base = nodes 1, 3
    driver = tmp_path / "deck.tcl"
    ops.tcl(str(driver), split=True)

    parts = tmp_path / "parts"
    assert "fix 1 " in (parts / "A.tcl").read_text(encoding="utf-8")
    assert "fix 3 " in (parts / "B.tcl").read_text(encoding="utf-8")
    assert "fix " not in driver.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Py fragment executes — build(ops) reconstructs its module
# ---------------------------------------------------------------------------

class _RecordingOps:
    """Minimal ops stand-in that records ``ops.<method>(*args)`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        def _rec(*args, **_kwargs):
            self.calls.append((name, args))
        return _rec


def test_py_fragment_build_dispatches(tmp_path: Path) -> None:
    """Load a generated ``parts/<m>.py`` by file path (the driver's own
    importlib mechanism) and call ``build(ops)`` — it must reconstruct
    exactly that module's nodes/elements against the passed ops."""
    import importlib.util as ilu

    fem = make_two_module_frame()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=t,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.py(str(tmp_path / "deck.py"), split=True)

    frag = tmp_path / "parts" / "A.py"
    spec = ilu.spec_from_file_location("_test_frag_A", str(frag))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rec = _RecordingOps()
    mod.build(rec)
    node_tags = [a[0] for name, a in rec.calls if name == "node"]
    assert node_tags == [1, 2]  # module A owns nodes 1, 2
    assert any(name == "element" for name, _ in rec.calls)
