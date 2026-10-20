"""#2290 (C): the live pairing over a FEMData archive types the catalogued
element classes the bridge builds no element of, and narrows the
duplicate-connectivity refusal.

``LiveDomainFacts.pair_elements`` pairs a FEM element with the live element
of its own id by fact. Each class the pairing serves maps to its family's
bridge token (``_TOKEN_FOR_LIVE_CLASS``); a class not mapped refuses by
name. ``Twenty_Node_Brick`` holds its nodes in upstream's serendipity
order, so its ``eleNodes`` is compared with the FEM (Gmsh) connectivity in
that order. Two FEM elements on the same ordered nodes are told apart only
by class: a recorded element with twins pairs when every record of it
names a class, every twin is live, and no twin's live class is a named one.

The live domain here is the fake ``LiveOps`` of the #2251 tests; the stock
oracles (values and Gauss positions) are
``tests/opensees/live/test_live_unmapped_classes_live.py``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees._readback import (
    _CLASS_FOR_TOKEN,
    _ROTATIONAL_FAMILIES,
    _TRANSLATIONAL_ONLY,
)
from apeGmsh.opensees._recorder_translate import (
    _TOKEN_FOR_LIVE_CLASS,
    LiveDomainFacts,
)
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.results.live import DefaultRouteWarning
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.test_results_live_femdata_archive import LiveOps
from tests.test_results_live_femdata_elements import _tets_fem

# ---------------------------------------------------------------------------
# The class table
# ---------------------------------------------------------------------------

#: Each newly served class, its family's token and whether that family
#: establishes a rotation (a shell or a beam) or holds translations only
#: (a continuum). Spelled out here, not read from the table under test.
SERVED = {
    "BbarBrick": ("stdBrick", False),
    "SSPbrick": ("stdBrick", False),
    "Twenty_Node_Brick": ("stdBrick", False),
    "SSPquad": ("quad", False),
    "EightNodeQuad": ("quad", False),
    "ShellDKGT": ("ShellDKGQ", True),
    "ShellNLDKGQ": ("ShellDKGQ", True),
    "ShellNLDKGT": ("ShellDKGQ", True),
    "ForceBeamColumnCBDI2d": ("forceBeamColumn", True),
    "Truss2": ("Truss", False),
    "CorotTruss2": ("CorotTruss", False),
    "ModElasticBeam2d": ("elasticBeamColumn", True),
}

#: Catalogued classes the pairing still refuses, each with why: stock
#: cannot build the Bezier pair; the rest build but record nothing the
#: classic route reads yet (see the live file's UNMAPPED table).
REFUSED = ("BezierTri6", "BezierTet10", "ShellMITC9",
           "ElasticForceBeamColumn2d", "ElasticForceBeamColumn3d",
           "ForceBeamColumnWarping2d")


@pytest.mark.parametrize("cls", list(SERVED))
def test_a_served_class_takes_its_family_token(cls) -> None:
    token, rotational = SERVED[cls]
    assert _TOKEN_FOR_LIVE_CLASS[cls] == token
    family = _CLASS_FOR_TOKEN[token]
    assert issubclass(family, _ROTATIONAL_FAMILIES) is rotational
    assert issubclass(family, _TRANSLATIONAL_ONLY) is not rotational


def test_a_force_beam_variant_is_never_an_elastic_beam() -> None:
    """Its token gates an elastic beam's ``localForce`` stations: a force
    beam variant must take the force beam's, so that gate refuses it."""
    from apeGmsh.results.spec._emit import ELASTIC_BEAM_TYPE_TOKENS

    assert _TOKEN_FOR_LIVE_CLASS["ForceBeamColumnCBDI2d"] not in \
        ELASTIC_BEAM_TYPE_TOKENS


# ---------------------------------------------------------------------------
# Refused by name, before any recorder
# ---------------------------------------------------------------------------

_XYZ = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0),
        4: (0.0, 0.0, 1.0), 5: (1.0, 1.0, 1.0)}


def _spec(fem: FEMData, ids: "list[int]", cls: "str | None",
          category: str = "gauss",
          comps: "tuple[str, ...]" = ("stress_xx",)) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        ResolvedRecorderRecord(category=category, name="rec", components=comps,
                               dt=None, n_steps=None,
                               element_ids=np.asarray(ids, dtype=np.int64),
                               element_class_name=cls),))


@pytest.mark.parametrize("cls", REFUSED)
def test_an_unserved_class_refuses_by_name_and_records_nothing(
    cls, tmp_path,
) -> None:
    fem = _tets_fem()
    model_h5 = tmp_path / "fem.h5"
    fem.to_h5(str(model_h5))
    ops = LiveOps(ndm=3, ndf={n: 3 for n in _XYZ},
                  elements={1: ("FourNodeTetrahedron", (1, 2, 3, 4)),
                            2: (cls, (2, 3, 4, 5))}, coords=dict(_XYZ))
    assert cls not in _TOKEN_FOR_LIVE_CLASS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DefaultRouteWarning)
        with pytest.raises(ValueError, match=(
                rf"no bridge element token \('{cls}': FEM elements \[2\]\)")):
            with _spec(fem, [1, 2], None).emit_default(
                    tmp_path / "run", model_h5=model_h5, ops=ops):
                pass
    assert ops.recorder_calls[1:] == []                 # only the MPCO probe


# ---------------------------------------------------------------------------
# Twenty_Node_Brick: eleNodes in serendipity order
# ---------------------------------------------------------------------------

_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
#: Gmsh's hex20 edge order (slots 8-19).
_GMSH_EDGES = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6),
               (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)]
#: Upstream Twenty_Node_Brick's: lower ring, upper ring, verticals.
_SERENDIPITY_EDGES = [(0, 1), (1, 2), (2, 3), (0, 3), (4, 5), (5, 6),
                      (6, 7), (4, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
#: The FEM node of each Twenty_Node_Brick slot (FEM node k+1 is Gmsh slot k).
_HEX20_LIVE = tuple(list(range(1, 9)) + [_GMSH_EDGES.index(e) + 9
                                         for e in _SERENDIPITY_EDGES])


def _hex20(tmp: Path) -> "tuple[FEMData, Path, dict[int, tuple[float, ...]]]":
    xyz = np.vstack([_CORNERS, [0.5 * (_CORNERS[a] + _CORNERS[b])
                                for a, b in _GMSH_EDGES]])
    info = make_type_info(code=17, gmsh_name="Hexahedron 20", dim=3, order=2,
                          npe=20, count=1)
    group = ElementGroup(element_type=info, ids=np.array([9], dtype=np.int64),
                         connectivity=np.arange(1, 21, dtype=np.int64)[None])
    pg = PhysicalGroupSet({})
    fem = FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 21, dtype=np.int64),
                            node_coords=xyz, physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={17: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=20, n_elems=1, bandwidth=20, types=[info]),
        model_name="unmapped_hex20")
    path = tmp / "fem.h5"
    fem.to_h5(str(path))
    return fem, path, {k + 1: tuple(c) for k, c in enumerate(xyz)}


def _hex20_facts(tmp: Path, live: "tuple[int, ...]",
                 cls: str = "Twenty_Node_Brick") -> "tuple[LiveDomainFacts, LiveOps]":
    _, path, coords = _hex20(tmp)
    ops = LiveOps(ndm=3, ndf={n: 3 for n in coords},
                  elements={9: (cls, live)}, coords=coords)
    return LiveDomainFacts(ops, OpenSeesModel.from_h5(path)), ops


def test_a_twenty_node_brick_pairs_in_serendipity_order(tmp_path) -> None:
    assert _HEX20_LIVE[8:12] == (9, 12, 14, 10)        # the lower ring
    facts, _ = _hex20_facts(tmp_path, _HEX20_LIVE)
    facts.pair_elements([9], where="t")
    (rec,) = facts.elements()
    assert rec.type_token == "stdBrick" and rec.connectivity == _HEX20_LIVE
    assert facts.live_class(9) == "Twenty_Node_Brick"


def test_a_twenty_node_brick_in_gmsh_order_refuses_by_id(tmp_path) -> None:
    """Upstream's mid-edge slots are not Gmsh's: built straight from the
    Gmsh connectivity, its results would be another element's (on a real
    build it exits at ``setDomain`` on a non-positive Jacobian)."""
    facts, _ = _hex20_facts(tmp_path, tuple(range(1, 21)))
    with pytest.raises(ValueError, match=(
            r"FEM element 9 has nodes \[1, 2, .*, 20\] \(Twenty_Node_Brick "
            r"order \[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 10, 17, 19, 20, 18, "
            r"11, 13, 15, 16\]\), the live element 9 has \[1, 2, 3")):
        facts.pair_elements([9], where="t")
    assert facts.elements() == ()


def test_another_hex20_class_compares_in_gmsh_order(tmp_path) -> None:
    """Only Twenty_Node_Brick is reordered: a 20-node class with no order
    of its own compares the Gmsh connectivity as it is."""
    facts, _ = _hex20_facts(tmp_path, _HEX20_LIVE, cls="SomeHex20")
    with pytest.raises(ValueError, match=r"the live element 9 has \[1, 2, 3, "
                                         r"4, 5, 6, 7, 8, 9, 12, 14, 10"):
        facts.pair_elements([9], where="t")


# ---------------------------------------------------------------------------
# The duplicate-connectivity narrowing
# ---------------------------------------------------------------------------


def _twin_facts(tmp: Path, live: "dict[int, tuple[str, tuple[int, ...]]]",
                ) -> "tuple[LiveDomainFacts, LiveOps]":
    """FEM lines 5 and 6 on nodes [1, 2], line 7 on [2, 3]."""
    xyz = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=3)
    group = ElementGroup(element_type=info,
                         ids=np.array([5, 6, 7], dtype=np.int64),
                         connectivity=np.array([[1, 2], [1, 2], [2, 3]],
                                               dtype=np.int64))
    pg = PhysicalGroupSet({})
    fem = FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 4, dtype=np.int64),
                            node_coords=xyz, physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=3, n_elems=3, bandwidth=3, types=[info]),
        model_name="unmapped_twins")
    path = tmp / "fem.h5"
    fem.to_h5(str(path))
    ops = LiveOps(ndm=2, ndf={1: 3, 2: 3, 3: 3}, elements=live,
                  coords={1: (0.0, 0.0), 2: (3.0, 0.0), 3: (3.0, 2.0)})
    return LiveDomainFacts(ops, OpenSeesModel.from_h5(path)), ops


_BEAM = ("ElasticBeam2d", (1, 2))
_TRUSS = ("Truss", (1, 2))


def test_twins_of_another_live_class_pair_on_a_named_class(tmp_path) -> None:
    """The #2288 delta review's case: elastic beam 5 and truss 6 on [1, 2]."""
    facts, _ = _twin_facts(tmp_path, {5: _BEAM, 6: _TRUSS})
    facts.pair_elements([5, 6], where="t", named_classes={
        5: ("ElasticBeam2d",), 6: ("Truss",)})
    assert [(r.tag, r.type_token) for r in facts.elements()] == [
        (5, "elasticBeamColumn"), (6, "Truss")]


@pytest.mark.parametrize("live, named", [
    ({5: _BEAM, 6: ("ElasticBeam2d", (1, 2))}, {5: ("ElasticBeam2d",)}),
    ({5: _BEAM, 6: _TRUSS}, {5: (None,)}),
    ({5: _BEAM, 6: _TRUSS}, {5: ("ElasticBeam2d", None)}),
    ({5: _BEAM, 6: _TRUSS}, {}),
    ({5: _BEAM, 6: _TRUSS}, {6: ("Truss",)}),
    ({5: _BEAM}, {5: ("ElasticBeam2d",)}),
    ({5: _BEAM, 6: _TRUSS}, {5: ("ElasticBeam2d", "Truss")}),
], ids=["equal-class", "no-class", "one-record-names-none", "no-classes",
        "other-element-named", "twin-not-live", "twin-class-named"])
def test_twins_refuse_unless_the_classes_tell_them_apart(
    tmp_path, live, named,
) -> None:
    facts, ops = _twin_facts(tmp_path, live)
    with pytest.raises(ValueError, match=r"FEM element 5 shares nodes \[1, 2\] "
                                         r"with \[6\].*eleNodes cannot tell"):
        facts.pair_elements([5], where="t", named_classes=named)
    assert facts.elements() == () and "eleNodes" not in ops.asked


def test_a_twin_recorded_without_a_class_refuses_its_partner_does_not(
    tmp_path,
) -> None:
    """Each recorded element is judged by its own records: 5 names its
    class, 6 is recorded by a record that names none."""
    facts, _ = _twin_facts(tmp_path, {5: _BEAM, 6: _TRUSS})
    with pytest.raises(ValueError, match=r"FEM element 6 shares nodes \[1, 2\] "
                                         r"with \[5\]") as err:
        facts.pair_elements([5, 6], where="t", named_classes={
            5: ("ElasticBeam2d",), 6: (None,)})
    assert "FEM element 5 shares" not in str(err.value)


def test_no_record_classes_keeps_the_strict_refusal(tmp_path) -> None:
    """The call without ``named_classes`` refuses strictly, asking nothing
    of the twins; its message names each remedy."""
    facts, ops = _twin_facts(tmp_path, {5: _BEAM, 6: _TRUSS})
    with pytest.raises(ValueError, match=(
            r"FEM element 5 shares nodes \[1, 2\] with \[6\]\. eleNodes "
            r"cannot tell those elements apart, .* Drop them from the record, "
            r"name the record's element class where each twin is a live "
            r"element of another class, or record them through the bridge's "
            r"model archive \(ops\.h5\(\.\.\.\)\)\.$")):
        facts.pair_elements([5], where="t")
    assert "eleType" not in ops.asked and "eleNodes" not in ops.asked


def test_a_lone_element_needs_no_class(tmp_path) -> None:
    facts, _ = _twin_facts(tmp_path, {5: _BEAM, 6: _TRUSS,
                                      7: ("ElasticBeam2d", (2, 3))})
    facts.pair_elements([7], where="t")
    assert [r.tag for r in facts.elements()] == [7]

