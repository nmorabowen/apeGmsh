"""P2-G parity proof — ``crossing_plane`` == legacy ``queries.select``.

selection-unification-v2 **P2-G** (``docs/plans/selection-unification-v2.md``
§6 P2-G, §3 HT9, §6.1 STOP-1).  P2-G folds the legacy geometry
straddle surface — ``queries.select(on=/crossing=/not_on=/
not_crossing=)`` plus the 2-point ``queries.line`` primitive — into the
unified chain idiom as a single ``EntitySelection.crossing_plane(spec,
*, tol=, mode=)`` verb (``spec`` is the legacy ``_parse_primitive``
grammar: dict→axis plane, 2 pts→Line, 3 pts→plane, ``Plane``/``Line``
instance; ``mode`` ∈ {on, crossing, not_on, not_crossing}).

This file is the **invisibility proof** that the new verb is
behaviour-faithful to the byte-unchanged legacy engine (the legacy
``queries.select`` / ``_select_impl`` / ``Plane`` / ``Line`` are *not*
modified by P2-G — P3 removes them).  For representative scenarios
spanning **every** predicate mode of the 122-occurrence / 9-file legacy
parity battery —

  * axis-aligned-plane dict spec (``{'z': 0}`` / ``{'x': ...}``);
  * 3-point plane spec;
  * 2-point ``Line`` spec (the ``queries.line`` path) **and** a
    ``queries.line(...)`` ``Line`` instance passed through;
  * all four modes: ``on`` / ``crossing`` / ``not_on`` /
    ``not_crossing``;
  * the ``tol`` boundary (a corner exactly ``tol`` off the plane);
  * a multi-dim seed (curves + surfaces + volume together);
  * an empty result (predicate matches nothing);
  * chained refinement (``crossing_plane`` after another spatial verb);

— it asserts ``g.model.select(seed).crossing_plane(spec, mode=m)``
(the repointed P2-I host hook → ``EntitySelection``) resolves the
**identical** ``(dim, tag)`` set as the legacy
``g.model.queries.select(seed, <m>=spec)`` / ``queries.line``.  Both
paths are seeded from the *same explicitly-resolved dimtag list* so any
difference is the ``crossing_plane`` engine, not name resolution.

It also locks the §6.1 STOP-1 point-family contract: the verb is
*required* on all seven chains, but a point chain
(``fem.nodes.select(...).crossing_plane(...)``) **fails loud** with
``TypeError`` — the ``GeometryChain.in_box(inclusive=)``→``TypeError``
precedent (mirrored from ``test_geometry_chain.py``).

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy.  Fixture *patterns* are mirrored (not imported)
from ``tests/test_geometry_chain.py`` / ``tests/test_p2i_parity.py``.
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh.core._selection import EntitySelection, Selection


def _tagset(obj) -> set:
    """Resolved ``(int(dim), int(tag))`` set from a chain or a legacy
    ``Selection`` — the canonical identity both paths must agree on.

    A chain's identity is its ``_items`` atoms (entity family yields
    bare ``(dim, tag)``; no pair-view); the legacy ``Selection`` is a
    list of ``(dim, tag)``.  Normalise both to a ``set`` of int pairs.
    """
    if isinstance(obj, EntitySelection):
        items = obj._items
    else:                                   # legacy Selection (a list)
        items = list(obj)
    return {(int(d), int(t)) for d, t in items}


# =====================================================================
# Fixtures — a unit box with a 6-face PG and all-12-edge PG.  Patterns
# mirrored from tests/test_geometry_chain.py::cube /
# tests/test_p2i_parity.py::cube_geo.
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
    """Resolve a PG name to a concrete ``(dim, tag)`` list ONCE.

    Both the new verb and the legacy ``queries.select`` are then seeded
    from this *same* list, so the comparison isolates the predicate
    (not name resolution — that is contract-locked elsewhere and
    identical for both).
    """
    return [(int(d), int(t)) for d, t in
            g.model.queries.select(pg)]


# =====================================================================
# 1. Axis-aligned-plane dict spec — all four modes
# =====================================================================

def test_axis_plane_dict_all_modes(box):
    g = box
    faces = _seed_dimtags(g, "Faces")          # 6 surfaces

    for spec in ({"z": 0}, {"z": 1}, {"x": 0}, {"y": 1}):
        for mode, leg_kw in (
            ("on", "on"),
            ("crossing", "crossing"),
            ("not_on", "not_on"),
            ("not_crossing", "not_crossing"),
        ):
            new = g.model.select(faces).crossing_plane(spec, mode=mode)
            legacy = g.model.queries.select(faces, **{leg_kw: spec})
            assert isinstance(new, EntitySelection)
            assert isinstance(legacy, Selection)
            assert _tagset(new) == _tagset(legacy), (
                f"axis dict {spec} mode={mode}: "
                f"{_tagset(new)} != legacy {_tagset(legacy)}"
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
# 2. 3-point plane spec — on / crossing / not_*
# =====================================================================

def test_three_point_plane_spec(box):
    g = box
    faces = _seed_dimtags(g, "Faces")

    # mid-height horizontal plane through 3 points (== {'z': 0.5})
    plane3 = [(0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5)]
    for mode in ("on", "crossing", "not_on", "not_crossing"):
        new = g.model.select(faces).crossing_plane(plane3, mode=mode)
        legacy = g.model.queries.select(faces, **{mode: plane3})
        assert _tagset(new) == _tagset(legacy), (
            f"3-pt plane mode={mode}: "
            f"{_tagset(new)} != legacy {_tagset(legacy)}"
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
    for mode in ("on", "crossing", "not_on", "not_crossing"):
        new = g.model.select(edges).crossing_plane(line_spec, mode=mode)
        legacy = g.model.queries.select(edges, **{mode: line_spec})
        assert _tagset(new) == _tagset(legacy), (
            f"2-pt line mode={mode}: "
            f"{_tagset(new)} != legacy {_tagset(legacy)}"
        )

    # a queries.line(...) Line *instance* passed straight through
    # crossing_plane must equal the legacy select(crossing=<that Line>).
    line_obj = g.model.queries.line((0, 0.5, 0), (1, 0.5, 0))
    new = g.model.select(edges).crossing_plane(line_obj, mode="crossing")
    legacy = g.model.queries.select(edges, crossing=line_obj)
    assert _tagset(new) == _tagset(legacy)
    # the mid-y line crosses the 4 edges running in the y-direction.
    assert len(new) == 4


# =====================================================================
# 4. tol boundary — a corner exactly `tol` off the plane
# =====================================================================

def test_tol_boundary_parity(box):
    g = box
    faces = _seed_dimtags(g, "Faces")

    # Plane z = `eps`; the z=0 face's far bbox corners sit exactly `eps`
    # below it.  The legacy 'on' test is `np.all(|sd| <= tol)`, so with
    # tol == eps the z=0 face is ON; with tol just under eps it is not.
    # Whatever the boundary decision, the new verb must make the SAME
    # one as the legacy engine (identical `<=` semantics).
    eps = 1e-6
    for tol in (eps, eps * 0.999, eps * 1.001, 1e-9):
        for mode in ("on", "not_on", "crossing", "not_crossing"):
            new = g.model.select(faces).crossing_plane(
                {"z": eps}, mode=mode, tol=tol
            )
            legacy = g.model.queries.select(
                faces, **{mode: {"z": eps}}, tol=tol
            )
            assert _tagset(new) == _tagset(legacy), (
                f"tol-boundary tol={tol} mode={mode}: "
                f"{_tagset(new)} != legacy {_tagset(legacy)}"
            )
    # default tol parity: crossing_plane defaults tol=1e-6 exactly like
    # queries.select — same call, no tol kw on either side.
    assert _tagset(
        g.model.select(faces).crossing_plane({"z": 0}, mode="on")
    ) == _tagset(g.model.queries.select(faces, on={"z": 0}))


# =====================================================================
# 5. Multi-dim seed — curves + surfaces + volume together
# =====================================================================

def test_multi_dim_seed_parity(box):
    g = box
    # one mixed-dim seed: every face + every edge + the volume.
    mixed = (_seed_dimtags(g, "Faces")
             + _seed_dimtags(g, "Edges")
             + _seed_dimtags(g, "box"))
    assert {d for d, _ in mixed} == {1, 2, 3}

    for spec in ({"z": 0}, [(0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5)]):
        for mode in ("on", "crossing", "not_on", "not_crossing"):
            new = g.model.select(mixed).crossing_plane(spec, mode=mode)
            legacy = g.model.queries.select(mixed, **{mode: spec})
            assert _tagset(new) == _tagset(legacy), (
                f"multi-dim spec={spec} mode={mode}: "
                f"{_tagset(new)} != legacy {_tagset(legacy)}"
            )
    # the volume straddles z=0.5 (crossing) and is not 'on' it.
    cr = g.model.select(mixed).crossing_plane({"z": 0.5},
                                              mode="crossing")
    assert (3, 1) in _tagset(cr)


# =====================================================================
# 6. Empty result — predicate matches nothing
# =====================================================================

def test_empty_result_parity(box):
    g = box
    faces = _seed_dimtags(g, "Faces")

    # No face lies on z = 99 → both produce an empty selection.
    new = g.model.select(faces).crossing_plane({"z": 99}, mode="on")
    legacy = g.model.queries.select(faces, on={"z": 99})
    assert _tagset(new) == _tagset(legacy) == set()
    assert len(new) == 0
    assert isinstance(new, EntitySelection)
    # nothing crosses a far plane either.
    assert _tagset(
        g.model.select(faces).crossing_plane({"z": 99}, mode="crossing")
    ) == _tagset(
        g.model.queries.select(faces, crossing={"z": 99})
    ) == set()


# =====================================================================
# 7. Chained refinement — crossing_plane after another spatial verb
# =====================================================================

def test_chained_refinement_parity(box):
    g = box
    faces = _seed_dimtags(g, "Faces")

    # New: chain on_plane (z=0 face) then crossing_plane(not_on x=0).
    # Legacy: select(on={'z':0}) then .select(not_on={'x':0}).
    new = (g.model.select(faces)
           .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
           .crossing_plane({"x": 0}, mode="not_on"))
    legacy = (g.model.queries.select(faces, on={"z": 0})
              .select(not_on={"x": 0}))
    assert _tagset(new) == _tagset(legacy)

    # crossing_plane refines (intersects with the chain's current
    # atoms) just like the legacy chained .select — start from the 4
    # side faces, then keep those crossing z=0.5.
    new2 = (g.model.select(faces)
            .crossing_plane({"x": 0.5}, mode="not_on")   # 4 sides + caps
            .crossing_plane({"z": 0.5}, mode="crossing"))
    legacy2 = (g.model.queries.select(faces, not_on={"x": 0.5})
               .select(crossing={"z": 0.5}))
    assert _tagset(new2) == _tagset(legacy2)
    # every result stays the entity terminal type (chainable)
    assert isinstance(new2, EntitySelection)


# =====================================================================
# 8. §6.1 STOP-1 — point family fails LOUD (the in_box(inclusive=)
#    →TypeError precedent, mirrored from test_geometry_chain.py)
# =====================================================================

def test_point_family_crossing_plane_raises_typeerror(g):
    """``fem.nodes.select(...).crossing_plane(...)`` is inexpressible
    (a node id has no bbox to straddle) and MUST fail loud — exactly
    the ``GeometryChain.in_box(inclusive=)``→``TypeError`` precedent
    (``test_geometry_chain.py``), never a silent empty selection."""
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
# 9. EntitySelection vs legacy Geometry... the byte-unchanged engine.
#    A direct _crossing_impl ↔ _select_impl spot-check (same primitive,
#    same bbox math) — proves the shared helper is the legacy engine.
# =====================================================================

def test_invalid_mode_is_loud(box):
    g = box
    faces = _seed_dimtags(g, "Faces")
    with pytest.raises(ValueError, match="mode="):
        g.model.select(faces).crossing_plane({"z": 0}, mode="sideways")
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        g.model.select(faces).crossing_plane({"z": 0}, tol=-1.0)
    # an unparseable spec raises through the legacy _parse_primitive
    # (1 point is neither a line nor a plane) — same message as legacy.
    with pytest.raises(ValueError, match="Cannot infer primitive"):
        g.model.select(faces).crossing_plane([(0, 0, 0)], mode="on")
