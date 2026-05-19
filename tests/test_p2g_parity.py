"""P2-G behaviour proof — the v2 ``crossing_plane`` engine (frozen).

selection-unification-v2 **P2-G** (``docs/plans/selection-unification-v2.md``
§6 P2-G, §3 HT9, §6.1 STOP-1), carried through **P3-R** (§6.2 / §6.3
M-NOTE-G7-cascade disposition 2).  P2-G folded the legacy geometry
straddle surface — ``queries.select(on=/crossing=/not_on=/
not_crossing=)`` plus the 2-point ``queries.line`` primitive — into the
unified chain idiom as a single ``EntitySelection.crossing_plane(spec,
*, tol=, mode=)`` verb (``spec`` is the legacy ``_parse_primitive``
grammar: dict→axis plane, 2 pts→Line, 3 pts→plane, ``Plane``/``Line``
instance; ``mode`` ∈ {on, crossing, not_on, not_crossing}).

P3-R **deletes** the legacy ``queries.select`` / ``queries.line``
bodies, so the original legacy↔v2 *parity* half is no longer
expressible (the legacy call now raises ``AttributeError``).  Per the
adjudicated disposition (§6.3 M-NOTE-G7-cascade item 2 — this file is
**rewritten to v2-only, NOT deleted**) this file is converted to the
proof-file freeze pattern: P2-G already proved the v2 verb ≡ the legacy
engine, so the legacy half is dropped and the **v2-produced** ``(dim,
tag)`` set is pinned as an explicit literal (captured on the
PROD-correct tree).  It still uniquely pins v2-OWNED behaviour covered
nowhere else post-P3-R: the RETAINED ``EntitySelection.crossing_plane``
engine across every spec / mode / tol-boundary / multi-dim / empty /
chained scenario, plus the §6.1 STOP-1 point-family ``TypeError``
fail-loud (preserved verbatim).

The deterministic 1x1x1 box fixture makes every ``(dim, tag)`` an exact
literal: OCC tags the 6 faces ``(2,1)..(2,6)`` and 12 edges
``(1,1)..(1,12)`` deterministically; the seed is resolved by PG name
through the retained ``g.model.select(pg)`` host hook (apeGmsh is
verbose-by-name; never hard-coded raw tags — only the *frozen
post-resolution* result is a literal, the freeze pattern).

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy.
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh.core._selection import EntitySelection, Line


def _ts(ch) -> list:
    """Sorted ``[(int(dim), int(tag)), ...]`` from an ``EntitySelection``.

    The chain's identity is its ``_items`` atoms (entity family yields
    bare ``(dim, tag)``; no pair-view).  Frozen literals below are this
    same sorted int-pair list.
    """
    assert isinstance(ch, EntitySelection)
    return sorted((int(d), int(t)) for d, t in ch._items)


# =====================================================================
# Fixture — a unit box with a 6-face PG and all-12-edge PG.  Pattern
# mirrored from tests/test_geometry_chain.py::cube.
# =====================================================================

@pytest.fixture
def box(g):
    """1x1x1 box: ``box`` (dim-3 label) + ``Faces`` (6 dim-2) + ``Edges``
    (12 dim-1) physical groups.  Tags are resolved by PG name (apeGmsh
    is verbose-by-name; never hard-coded raw tags)."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
    g.model.sync()
    faces = g.model.queries.boundary("box", dim=3, oriented=False)
    g.physical.add_surface([int(t) for _d, t in faces], name="Faces")
    edges = list(dict.fromkeys(
        g.model.queries.boundary(faces, combined=False, oriented=False)
    ))
    g.physical.add_curve([int(t) for _d, t in edges], name="Edges")
    return g


def _seed_dimtags(g, pg: str) -> list:
    """Resolve a PG name to a concrete ``(dim, tag)`` list ONCE via the
    retained ``g.model.select(...)`` host hook (the v2 successor of the
    removed ``queries.select`` resolve-only path; ``EntitySelection``
    yields bare ``(dim, tag)`` atoms).  ``crossing_plane`` is then
    seeded from this *same* list so the assertion isolates the
    predicate, never name resolution."""
    return [(int(d), int(t)) for d, t in g.model.select(pg)._items]


# =====================================================================
# 1. Axis-aligned-plane dict spec — all four modes (frozen literals)
# =====================================================================

def test_axis_plane_dict_all_modes(box):
    g = box
    faces = _seed_dimtags(g, "Faces")          # 6 surfaces

    # frozen v2 result per (spec, mode) on the PROD-correct tree
    # (P2-G proved this ≡ the now-removed legacy engine).
    expected = {
        ("z", 0, "on"): [(2, 5)],
        ("z", 0, "crossing"): [],
        ("z", 0, "not_on"): [(2, 1), (2, 2), (2, 3), (2, 4), (2, 6)],
        ("z", 0, "not_crossing"): [(2, 1), (2, 2), (2, 3), (2, 4),
                                   (2, 5), (2, 6)],
        ("z", 1, "on"): [(2, 6)],
        ("z", 1, "crossing"): [],
        ("z", 1, "not_on"): [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
        ("z", 1, "not_crossing"): [(2, 1), (2, 2), (2, 3), (2, 4),
                                   (2, 5), (2, 6)],
        ("x", 0, "on"): [(2, 1)],
        ("x", 0, "crossing"): [],
        ("x", 0, "not_on"): [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)],
        ("x", 0, "not_crossing"): [(2, 1), (2, 2), (2, 3), (2, 4),
                                   (2, 5), (2, 6)],
        ("y", 1, "on"): [(2, 4)],
        ("y", 1, "crossing"): [],
        ("y", 1, "not_on"): [(2, 1), (2, 2), (2, 3), (2, 5), (2, 6)],
        ("y", 1, "not_crossing"): [(2, 1), (2, 2), (2, 3), (2, 4),
                                   (2, 5), (2, 6)],
    }
    for (axis, val, mode), exp in expected.items():
        new = g.model.select(faces).crossing_plane({axis: val}, mode=mode)
        assert isinstance(new, EntitySelection)
        assert _ts(new) == exp, (
            f"axis dict {{{axis!r}: {val}}} mode={mode}: "
            f"{_ts(new)} != frozen {exp}"
        )

    # sanity: the modes are not all-empty / all-everything (the proof
    # would be vacuous otherwise) — z=0 'on' is exactly 1 face,
    # 'crossing' z=0 is 0, 'not_on' z=0 is 5.
    assert len(g.model.select(faces).crossing_plane({"z": 0},
                                                    mode="on")) == 1
    assert len(g.model.select(faces).crossing_plane({"z": 0},
                                                    mode="crossing")) == 0
    assert len(g.model.select(faces).crossing_plane({"z": 0},
                                                    mode="not_on")) == 5


# =====================================================================
# 2. 3-point plane spec — on / crossing / not_* (frozen literals)
# =====================================================================

def test_three_point_plane_spec(box):
    g = box
    faces = _seed_dimtags(g, "Faces")

    # mid-height horizontal plane through 3 points (== {'z': 0.5})
    plane3 = [(0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5)]
    expected = {
        "on": [],
        "crossing": [(2, 1), (2, 2), (2, 3), (2, 4)],
        "not_on": [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)],
        "not_crossing": [(2, 5), (2, 6)],
    }
    for mode, exp in expected.items():
        new = g.model.select(faces).crossing_plane(plane3, mode=mode)
        assert _ts(new) == exp, (
            f"3-pt plane mode={mode}: {_ts(new)} != frozen {exp}"
        )
    # the mid plane straddles the 4 side faces (crossing), is 'on'
    # none, and 'not_crossing' the 2 horizontal caps.
    assert len(g.model.select(faces)
               .crossing_plane(plane3, mode="crossing")) == 4
    assert len(g.model.select(faces)
               .crossing_plane(plane3, mode="not_crossing")) == 2


# =====================================================================
# 3. 2-point Line spec (the queries.line path) + Line instance
# =====================================================================

def test_two_point_line_spec_and_line_instance(box):
    g = box
    edges = _seed_dimtags(g, "Edges")          # 12 curves

    # a 2-point spec → infinite Line (the legacy queries.line 2-point
    # path, folded into crossing_plane).
    line_spec = [(0, 0.5, 0), (1, 0.5, 0)]
    expected = {
        "on": [],
        "crossing": [(1, 2), (1, 4), (1, 6), (1, 8)],
        "not_on": [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                   (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12)],
        "not_crossing": [(1, 1), (1, 3), (1, 5), (1, 7), (1, 9),
                         (1, 10), (1, 11), (1, 12)],
    }
    for mode, exp in expected.items():
        new = g.model.select(edges).crossing_plane(line_spec, mode=mode)
        assert _ts(new) == exp, (
            f"2-pt line mode={mode}: {_ts(new)} != frozen {exp}"
        )

    # a ``Line`` *instance* passed straight through crossing_plane
    # (the legacy ``queries.line(...)`` constructor is removed; the
    # retained ``Line.through`` is its byte-identical replacement —
    # ``queries.line`` only ever called ``Line.through`` internally).
    line_obj = Line.through((0, 0.5, 0), (1, 0.5, 0))
    new = g.model.select(edges).crossing_plane(line_obj, mode="crossing")
    assert _ts(new) == [(1, 2), (1, 4), (1, 6), (1, 8)]
    # the mid-y line crosses the 4 edges running in the y-direction.
    assert len(new) == 4


# =====================================================================
# 4. tol boundary — a corner exactly `tol` off the plane (frozen)
# =====================================================================

def test_tol_boundary_parity(box):
    g = box
    faces = _seed_dimtags(g, "Faces")

    # Plane z = `eps`; the z=0 face's far bbox corners sit exactly `eps`
    # below it.  The 'on' test is `np.all(|sd| <= tol)`.  The frozen
    # results pin the exact `<=` boundary decision the engine makes for
    # tol at / just under / just over / far below `eps` — all four are
    # identical here because the straddle uses the bbox span, not the
    # razor-edge corner alone (the proof: the boundary behaviour is
    # stable and pinned, not asserted against a removed legacy engine).
    eps = 1e-6
    per_mode = {
        "on": [],
        "not_on": [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)],
        "crossing": [(2, 1), (2, 2), (2, 3), (2, 4)],
        "not_crossing": [(2, 5), (2, 6)],
    }
    for tol in (eps, eps * 0.999, eps * 1.001, 1e-9):
        for mode, exp in per_mode.items():
            new = g.model.select(faces).crossing_plane(
                {"z": eps}, mode=mode, tol=tol
            )
            assert _ts(new) == exp, (
                f"tol-boundary tol={tol!r} mode={mode}: "
                f"{_ts(new)} != frozen {exp}"
            )
    # default tol parity: crossing_plane defaults tol=1e-6 — the {'z':0}
    # 'on' result is exactly the z=0 face (2,5).
    assert _ts(
        g.model.select(faces).crossing_plane({"z": 0}, mode="on")
    ) == [(2, 5)]


# =====================================================================
# 5. Multi-dim seed — curves + surfaces + volume together (frozen)
# =====================================================================

def test_multi_dim_seed_parity(box):
    g = box
    # one mixed-dim seed: every face + every edge + the volume.
    mixed = (_seed_dimtags(g, "Faces")
             + _seed_dimtags(g, "Edges")
             + _seed_dimtags(g, "box"))
    assert {d for d, _ in mixed} == {1, 2, 3}

    plane3 = [(0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5)]
    expected = {
        ("z0", "on"): [(1, 4), (1, 8), (1, 9), (1, 11), (2, 5)],
        ("z0", "crossing"): [],
        ("z0", "not_on"): [(1, 1), (1, 2), (1, 3), (1, 5), (1, 6),
                           (1, 7), (1, 10), (1, 12), (2, 1), (2, 2),
                           (2, 3), (2, 4), (2, 6), (3, 1)],
        ("z0", "not_crossing"): [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                                 (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
                                 (1, 11), (1, 12), (2, 1), (2, 2),
                                 (2, 3), (2, 4), (2, 5), (2, 6), (3, 1)],
        ("p3", "on"): [],
        ("p3", "crossing"): [(1, 1), (1, 3), (1, 5), (1, 7), (2, 1),
                             (2, 2), (2, 3), (2, 4), (3, 1)],
        ("p3", "not_on"): [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                           (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
                           (1, 11), (1, 12), (2, 1), (2, 2), (2, 3),
                           (2, 4), (2, 5), (2, 6), (3, 1)],
        ("p3", "not_crossing"): [(1, 2), (1, 4), (1, 6), (1, 8), (1, 9),
                                 (1, 10), (1, 11), (1, 12), (2, 5),
                                 (2, 6)],
    }
    for spec, nm in (({"z": 0}, "z0"), (plane3, "p3")):
        for mode in ("on", "crossing", "not_on", "not_crossing"):
            new = g.model.select(mixed).crossing_plane(spec, mode=mode)
            assert _ts(new) == expected[(nm, mode)], (
                f"multi-dim {nm} mode={mode}: "
                f"{_ts(new)} != frozen {expected[(nm, mode)]}"
            )
    # the volume straddles z=0.5 (crossing) and is not 'on' it.
    cr = g.model.select(mixed).crossing_plane({"z": 0.5},
                                              mode="crossing")
    assert (3, 1) in _ts(cr)
    assert _ts(cr) == [(1, 1), (1, 3), (1, 5), (1, 7), (2, 1), (2, 2),
                       (2, 3), (2, 4), (3, 1)]


# =====================================================================
# 6. Empty result — predicate matches nothing
# =====================================================================

def test_empty_result_parity(box):
    g = box
    faces = _seed_dimtags(g, "Faces")

    # No face lies on / crosses z = 99 → an empty selection.
    new = g.model.select(faces).crossing_plane({"z": 99}, mode="on")
    assert _ts(new) == []
    assert len(new) == 0
    assert isinstance(new, EntitySelection)
    # nothing crosses a far plane either.
    assert _ts(
        g.model.select(faces).crossing_plane({"z": 99}, mode="crossing")
    ) == []


# =====================================================================
# 7. Chained refinement — crossing_plane after another spatial verb
# =====================================================================

def test_chained_refinement_parity(box):
    g = box
    faces = _seed_dimtags(g, "Faces")

    # chain on_plane (z=0 face) then crossing_plane(not_on x=0) → the
    # z=0 face (2,5) survives (it is not_on x=0).
    new = (g.model.select(faces)
           .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
           .crossing_plane({"x": 0}, mode="not_on"))
    assert _ts(new) == [(2, 5)]

    # crossing_plane refines (intersects with the chain's current
    # atoms) — start from not_on x=0.5 (4 sides + caps), then keep
    # those crossing z=0.5 → the 4 side faces.
    new2 = (g.model.select(faces)
            .crossing_plane({"x": 0.5}, mode="not_on")
            .crossing_plane({"z": 0.5}, mode="crossing"))
    assert _ts(new2) == [(2, 1), (2, 2), (2, 3), (2, 4)]
    # every result stays the entity terminal type (chainable)
    assert isinstance(new2, EntitySelection)


# =====================================================================
# 8. §6.1 STOP-1 — point family fails LOUD (the in_box(inclusive=)
#    →TypeError precedent, mirrored from test_geometry_chain.py).
#    PRESERVED VERBATIM through the P3-R rewrite — it pins v2-OWNED
#    behaviour (the point-family crossing_plane fail-loud) covered
#    nowhere else post-P3-R.
# =====================================================================

def test_point_family_crossing_plane_raises_typeerror(g):
    """``fem.nodes.select(...).crossing_plane(...)`` is inexpressible
    (a node id has no bbox to straddle) and MUST fail loud — exactly
    the ``EntitySelection.in_box(inclusive=)``→``TypeError`` precedent,
    never a silent empty selection."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.structured.set_transfinite_box("box", n=3)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    nodes = fem.nodes.select(pg="Body")
    assert nodes.FAMILY == "point"
    # the verb is REQUIRED (callable on the point chain) but its hook
    # raises loud — not a silent [] (§6.1 STOP-1).
    assert callable(getattr(type(nodes), "crossing_plane", None))
    with pytest.raises(TypeError, match="entity-family"):
        nodes.crossing_plane({"z": 0}, mode="crossing")
    with pytest.raises(TypeError, match="entity-family"):
        nodes.crossing_plane([(0, 0, 0), (1, 0, 0)], mode="on")

    # element level is equally loud.
    elems = fem.elements.select(pg="Body")
    assert elems.FAMILY == "point"
    with pytest.raises(TypeError, match="entity-family"):
        elems.crossing_plane({"z": 0.5}, mode="crossing")


# =====================================================================
# 9. Input validation is loud (the byte-unchanged _parse_primitive /
#    mode / tol guards on the retained engine).
# =====================================================================

def test_invalid_mode_is_loud(box):
    g = box
    faces = _seed_dimtags(g, "Faces")
    with pytest.raises(ValueError, match="mode="):
        g.model.select(faces).crossing_plane({"z": 0}, mode="sideways")
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        g.model.select(faces).crossing_plane({"z": 0}, tol=-1.0)
    # an unparseable spec raises through the legacy _parse_primitive
    # (1 point is neither a line nor a plane).
    with pytest.raises(ValueError, match="Cannot infer primitive"):
        g.model.select(faces).crossing_plane([(0, 0, 0)], mode="on")
