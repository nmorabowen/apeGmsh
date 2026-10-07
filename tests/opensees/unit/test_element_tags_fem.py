"""Locks for ``apeSees(fem, element_tags="fem")`` (ADR 0111 D2).

With ``"fem"`` every physical-group element keeps its FEM element id as its
OpenSees tag (a deck translated from STKO keeps STKO's ids), and every
element the bridge synthesises lands above the snapshot's largest element
id. The default, ``"sequential"``, must stay byte-identical to the golden
emit corpus (#1258). What is locked:

* the default and an explicit ``"sequential"`` reproduce every golden deck
  cell (flat, partitioned, staged, staged-partitioned, per-rank; tcl, py,
  recording);
* ``"fem"`` on the same cells changes nothing but element tags, and every
  element line carries a FEM id;
* unsorted FEM ids survive verbatim; node-pair springs, interface
  ``zeroLength`` (flat and partitioned) and MP-emitted coupling elements
  are numbered above the largest FEM id, even one that is not emitted;
* refusals: an unknown mode, a FEM id ``<= 0``, an element fanned out by
  two declarations;
* the audit: the set of ``"element"`` tag-allocation sites in the bridge is
  pinned, so a new site has to be checked against the reservation that
  ``BuiltModel.emit`` makes before any of them runs.
"""
from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path
from typing import Any, cast

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    allocate_element_tags,
    reserve_fem_element_tags,
)
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees._internal.types import Element
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir
from tests.opensees.fixtures import fem_stub
from tests.opensees.golden import builder

_DECK_CELLS = [
    c for c in builder.all_cells() if builder.applicability(*c) is None
]
_ELEMENT_LINE = re.compile(
    r"^\s*(?:element\s+\S+\s+(\d+)\b|ops\.element\('[^']+',\s*(\d+)\b)"
)


def _golden_text(fixture: str, mode: str, output: str) -> str:
    deck, _dump = builder.golden_paths(fixture, mode, output)
    return deck.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
            cell: tuple[str, str, str], mode: str) -> str:
    def _ops(fem: Any, **kw: Any) -> apeSees:
        return apeSees(fem, element_tags=mode, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(builder, "apeSees", _ops)
    return builder.render_deck(*cell, tmp_path / "cell")


def _element_tags(text: str) -> list[int]:
    out = []
    for line in text.split("\n"):
        m = _ELEMENT_LINE.match(line)
        if m:
            out.append(int(m.group(1) or m.group(2)))
    return out


# ---------------------------------------------------------------------------
# The default is unchanged: golden corpus
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", _DECK_CELLS, ids=lambda c: "/".join(c))
def test_explicit_sequential_reproduces_the_golden_deck(
    cell: tuple[str, str, str], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    got = _render(monkeypatch, tmp_path, cell, "sequential")
    bad = builder.first_deck_mismatch(_golden_text(*cell), got)
    assert bad is None, f"{'/'.join(cell)}: line {bad} differs from golden"


@pytest.mark.parametrize("cell", _DECK_CELLS, ids=lambda c: "/".join(c))
def test_fem_mode_changes_only_element_tags(
    cell: tuple[str, str, str], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture, mode, _output = cell
    want = _golden_text(*cell)
    got = _render(monkeypatch, tmp_path, cell, "fem")

    seq_tags, fem_tags = _element_tags(want), _element_tags(got)
    assert len(seq_tags) == len(fem_tags) > 0
    spec = builder.FIXTURES[fixture]
    stub = builder._fem_for(fixture, mode)
    fem_eids = {int(e) for e in stub.elements._pgs[spec.elem_pg].ids}
    assert set(fem_tags) == fem_eids, "every element line carries a FEM id"

    mapping: dict[int, int] = {}
    for s, f in zip(seq_tags, fem_tags):
        assert mapping.setdefault(s, f) == f, "one tag, one FEM id"
    assert len(set(mapping.values())) == len(mapping)

    # Every value is exact: the orientation-derived vecxz is
    # bit-reproducible (test_vecxz_determinism.py). Integers may differ
    # only by the element-tag mapping.
    num = r"-?\d+\.\d+(?:[eE][-+]?\d+)?|\d+|\D+"
    wl, gl = want.split("\n"), got.split("\n")
    assert len(wl) == len(gl)
    for i, (a, b) in enumerate(zip(wl, gl)):
        if a == b:
            continue
        ta, tb = re.findall(num, a), re.findall(num, b)
        assert len(ta) == len(tb), f"line {i + 1}: {a!r} vs {b!r}"
        for x, y in zip(ta, tb):
            if x == y:
                continue
            assert x.isdigit() and mapping.get(int(x)) == int(y), (
                f"line {i + 1}: {a!r} -> {b!r} changes more than an "
                f"element tag"
            )


# ---------------------------------------------------------------------------
# Real meshes
# ---------------------------------------------------------------------------

def _beam_deck(tmp_path: Path, mode: str | None,
               h5: Path | None = None) -> list[str]:
    """Three beams with unsorted FEM ids (30, 10, 20) + one node-pair
    spring; returns the deck's element lines (and writes ``h5`` if given)."""
    with apeGmsh(model_name="etags", verbose=False) as g:
        ent = gmsh.model.addDiscreteEntity(1)
        gmsh.model.mesh.addNodes(
            1, ent, [1, 2, 3, 4, 5],
            [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0])
        gmsh.model.mesh.addElementsByType(
            ent, 1, [30, 10, 20], [1, 2, 2, 3, 3, 4])
        gmsh.model.addPhysicalGroup(1, [ent], name="B")
        fem = g.mesh.queries.get_fem_data(dim=1)
        ops = apeSees(fem) if mode is None else apeSees(
            fem, element_tags=mode)  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        ops.element.elasticBeamColumn(
            pg="B", transf=tr, A=1.0, E=1.0, G=1.0, J=1.0, Iy=1.0, Iz=1.0)
        k = ops.uniaxialMaterial.ElasticMaterial(E=1.0)
        ops.element.ZeroLength(
            nodes=(4, 5), mat_dirs=(ZeroLengthMatDir(material=k, dof=1),))
        path = tmp_path / f"deck_{mode}.tcl"
        ops.tcl(str(path))
        if h5 is not None:
            ops.h5(str(h5))
    return [ln for ln in path.read_text().splitlines()
            if ln.startswith("element ")]


def test_unsorted_fem_ids_are_the_tags_and_springs_go_above(
    tmp_path: Path,
) -> None:
    lines = _beam_deck(tmp_path, "fem")
    assert [ln.split()[2] for ln in lines] == ["30", "10", "20", "31"]
    # Connectivity travels with its id: element 10 is nodes 2-3.
    assert lines[1].split()[3:5] == ["2", "3"]
    assert lines[3].startswith("element zeroLength 31 4 5 ")


def test_fem_tags_survive_the_h5_round_trip(tmp_path: Path) -> None:
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    h5 = tmp_path / "m.h5"
    _beam_deck(tmp_path, "fem", h5=h5)
    recs = OpenSeesModel.from_h5(str(h5)).elements()
    assert sorted((r.type_token, int(r.tag), int(r.fem_eid)) for r in recs) == [
        ("elasticBeamColumn", 10, 10), ("elasticBeamColumn", 20, 20),
        ("elasticBeamColumn", 30, 30), ("zeroLength", 31, -1)]


def test_default_still_numbers_sequentially(tmp_path: Path) -> None:
    default = _beam_deck(tmp_path, None)
    assert default == _beam_deck(tmp_path, "sequential")
    assert [ln.split()[2] for ln in default] == ["3", "4", "5", "6"]


_NORMAL = NormalLaw(kind="ent", k_per_area=1.0e9)
_TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)


def _curve_at_x(surface: int, x: float) -> int:
    for _dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < 1e-6 and abs(bb[3] - x) < 1e-6:
            return abs(tag)
    raise AssertionError(f"no boundary curve at x={x}")


def _interface_fem(partition: int) -> Any:
    with apeGmsh(model_name="etags_iface", verbose=False) as g:
        left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
        right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(2, left), (2, right)], n=4)
        g.mesh.generation.generate(2)
        g.physical.add(2, [left], name="rock")
        g.physical.add(2, [right], name="liner")
        g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
        g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
        g.constraints.interface(
            "face", "wire", normal=_NORMAL, tangential=_TANGENTIAL,
            thickness=0.5, name="RockLiner")
        if partition:
            g.mesh.partitioning.partition(partition)
        return g.mesh.queries.get_fem_data()


@pytest.mark.parametrize("partition", [0, 2], ids=["flat", "partitioned"])
def test_interface_zero_length_lands_above_every_fem_id(
    partition: int, tmp_path: Path,
) -> None:
    fem = _interface_fem(partition)
    ops = apeSees(fem, element_tags="fem", _artifacts=False)  # partitioned snapshot: no automatic model.h5
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in ("rock", "liner"):
        ops.element.FourNodeQuad(
            pg=pg, thickness=0.5, material=mat, plane_type="PlaneStrain")
    path = tmp_path / "iface.tcl"
    ops.tcl(str(path))
    lines = [ln.strip() for ln in path.read_text().splitlines()]

    quads = {int(ln.split()[2]) for ln in lines
             if ln.startswith("element quad ")}
    want = {int(e) for pg in ("rock", "liner")
            for e in fem.elements.select(pg=pg).ids}
    assert quads == want
    zl = [int(ln.split()[2]) for ln in lines
          if ln.startswith("element zeroLength ")]
    assert zl and min(zl) > int(np.max(fem.elements.ids))
    assert len(set(zl)) == len(zl)


def test_coupling_element_skips_a_fem_id_that_is_not_emitted(
    tmp_path: Path,
) -> None:
    with apeGmsh(model_name="etags_kc", verbose=False) as g:
        p0 = g.model.geometry.add_point(0, 0, 0)
        p1 = g.model.geometry.add_point(2, 0, 0)
        p2 = g.model.geometry.add_point(2, 1, 0)
        line = g.model.geometry.add_line(p0, p1)
        g.physical.add(1, [line], name="B")
        g.physical.add(0, [p1], name="M")
        g.physical.add(0, [p2], name="S")
        g.constraints.kinematic_coupling(
            "M", "S", master_point=(2, 0, 0), name="kc")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data()
    beams = {int(e) for e in fem.elements.select(pg="B").ids}
    top = int(np.max(fem.elements.ids))
    assert top not in beams, "fixture: the largest FEM id is not a beam"

    ops = apeSees(fem, element_tags="fem")
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="B", transf=tr, A=1.0, E=1.0, G=1.0, J=1.0, Iy=1.0, Iz=1.0)
    path = tmp_path / "kc.tcl"
    ops.tcl(str(path))
    lines = path.read_text().splitlines()
    assert {int(ln.split()[2]) for ln in lines
            if ln.startswith("element elasticBeamColumn ")} == beams
    kc = [int(ln.split()[2]) for ln in lines
          if ln.startswith("element LadrunoKinematicCoupling ")]
    assert kc == [top + 1]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_unknown_mode_is_refused() -> None:
    fem = fem_stub.make_two_column_frame()
    with pytest.raises(ValueError, match="element_tags must be one of"):
        apeSees(cast("object", fem), element_tags="gmsh")  # type: ignore[arg-type]


def _frame_ops(fem: Any, n_specs: int = 1) -> apeSees:
    ops = apeSees(fem, element_tags="fem")
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    for _ in range(n_specs):
        ops.element.elasticBeamColumn(
            pg="Cols", transf=tr,
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    return ops


def test_an_element_fanned_out_twice_is_refused(tmp_path: Path) -> None:
    ops = _frame_ops(fem_stub.make_two_column_frame(), n_specs=2)
    with pytest.raises(BridgeError, match="more than once"):
        ops.tcl(str(tmp_path / "dup.tcl"))


def test_a_non_positive_fem_id_is_refused(tmp_path: Path) -> None:
    fem = fem_stub.make_two_column_frame()
    fem.elements._pgs["Cols"] = fem_stub._ElementGroupView(
        ids=(0, 2), connectivity=((1, 2), (3, 4)))
    with pytest.raises(BridgeError, match="<= 0"):
        _frame_ops(fem).tcl(str(tmp_path / "zero.tcl"))


def test_allocation_without_the_reservation_is_refused() -> None:
    """The plan never hands out a FEM id the counter has not passed."""
    fem = fem_stub.make_two_column_frame()
    ops = _frame_ops(fem)
    elements = [p for p in ops._primitives if isinstance(p, Element)]
    with pytest.raises(BridgeError, match="not reserved"):
        allocate_element_tags(elements, fem, TagAllocator(),
                              element_tags="fem")
    tags = TagAllocator()
    assert reserve_fem_element_tags(elements, fem, tags) == 2
    plan = allocate_element_tags(elements, fem, tags, element_tags="fem")
    assert [t for _spec, rows in plan for _e, _c, t in rows] == [1, 2]
    assert tags.allocate("element") == 3


# ---------------------------------------------------------------------------
# The audit: every element-tag allocation site, pinned
# ---------------------------------------------------------------------------

#: ``(module, enclosing function)`` of every ``allocate("element")`` /
#: ``allocate_block("element", n)`` call in ``src/apeGmsh/opensees``. Each
#: runs inside the build's tag plan (``tag_plan.plan_tags``) AFTER
#: ``reserve_fem_element_tags``, so under ``element_tags="fem"`` it draws a
#: tag above every FEM id. A new site must be checked the same way before
#: it joins this set.
_AUDITED_ELEMENT_ALLOCATIONS = {
    ("_internal/build.py", "plan_mp_elements"),
    ("_internal/build.py", "plan_interface_tags"),
    ("_internal/build.py", "allocate_element_tags"),
}


def _element_allocation_sites() -> set[tuple[str, str]]:
    root = Path(__file__).resolve().parents[3] / "src" / "apeGmsh" / "opensees"
    found: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        with warnings.catch_warnings():     # a docstring's stray escape
            warnings.simplefilter("ignore")
            tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(root).as_posix()

        def visit(node: ast.AST, scope: str, rel: str = rel) -> None:
            for child in ast.iter_child_nodes(node):
                inner = scope
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    inner = child.name
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr in (
                        "allocate", "allocate_block", "allocate_for")
                    and any(isinstance(a, ast.Constant) and a.value == "element"
                            for a in child.args)
                ):
                    found.add((rel, scope))
                visit(child, inner, rel)

        visit(tree, "<module>")
    return found


def test_every_element_tag_allocation_site_is_audited() -> None:
    assert _element_allocation_sites() == _AUDITED_ELEMENT_ALLOCATIONS
