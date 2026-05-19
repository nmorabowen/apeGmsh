"""Lock: the GEOMETRY selection terminal yields plain Python ``int``.

Sibling guard to the FEM / mesh / result dtype parity already locked by
``test_fem_chain.py`` (``NodeResult`` object-dtype + ``.get()`` parity),
``test_selection_idiom.py`` and ``test_mesh_selection_chain.py``
(``tags`` / ``element_ids`` / ``connectivity`` ``.dtype == object``).

The geometry family is the ONE selection terminal whose Python-int-ness
is *inherited*, not enforced:

* ``Selection.tags()`` is ``[t for _, t in self]`` — **no ``int()``
  cast** (``core/_selection.py:450``).
* ``GeometryChain`` spatial hooks return the *original* atom unchanged
  (e.g. ``_spatial_box`` ``core/_selection.py:882``).

It is int-clean today only because ``resolve_to_dimtags`` ``int()``-wraps
every branch (``core/_helpers.py:197/211/219/259/271/289``) and gmsh
returns native ints. This pins that contract so a future resolver /
gmsh-binding change that leaks ``np.int64`` surfaces as a reviewed test
diff rather than silent drift into Tcl/JSON/H5 emit (``np.int64``
serialises differently than ``int``).

No openseespy (curated CI gate): pure apeGmsh + gmsh + numpy.
"""
from __future__ import annotations

import numpy as np

from apeGmsh.core._helpers import as_dimtags, resolve_to_dimtags


def _assert_python_int_dimtags(sel) -> None:
    """Every (dim, tag) and every ``.tags()`` entry is a *Python* int.

    Strict ``type(x) is int`` — ``isinstance(np.int64(1), int)`` is
    False on every platform, but the inverse trap (a numpy scalar that
    duck-types as int in arithmetic yet serialises differently) is
    exactly what this guards, so the strict identity check is the point.
    """
    assert len(sel) > 0, "fixture produced an empty selection"
    for d, t in sel:
        assert type(d) is int, f"dim is {type(d).__name__}, not int"
        assert type(t) is int, f"tag is {type(t).__name__}, not int"
    for t in sel.tags():
        assert type(t) is int, f"tags() entry is {type(t).__name__}, not int"


# =====================================================================
# Supported seed forms -> geometry terminal is Python int
# =====================================================================

def test_select_by_name_yields_python_int_tags(g):
    # Seed via the tiered name resolver (label/PG tier of
    # resolve_to_dimtags). g.model.select(...).result() -> legacy
    # Selection.
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
    g.physical.add_volume("box", name="Body")
    sel = g.model.select("Body", dim=3).result()
    _assert_python_int_dimtags(sel)


def test_select_by_int_tag_yields_python_int_tags(g):
    # Seed via a bare Python int tag (resolve_to_dimtags int branch:
    # core/_helpers.py:219 -> (resolve_dim(int(ref), ..), int(ref))).
    v = g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    sel = g.model.select(int(v), dim=3).result()
    _assert_python_int_dimtags(sel)


def test_select_by_dimtag_list_yields_python_int_tags(g):
    # Seed via a concrete dimtag list (the boundary-query output) ->
    # resolve_to_dimtags list recursion + dimtag passthrough
    # (core/_helpers.py:200/211).
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
    faces = g.model.queries.boundary("box", dim=2)
    sel = g.model.select(faces).result()
    _assert_python_int_dimtags(sel)


def test_predicate_select_yields_python_int_tags(g):
    # selection-unification-v2 P3-R: the legacy
    # ``g.model.queries.select(on=)`` entry point is removed; the v2
    # ``g.model.select(...).crossing_plane(spec, mode="on").result()``
    # materialises the SAME legacy ``Selection`` via the RETAINED
    # ``_select_impl`` (M-CORRECTION) — same terminal, same Python-int
    # dimtag contract.
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
    faces = g.model.queries.boundary("box", dim=2)
    sel = (
        g.model.select(faces)
        .crossing_plane({"z": 0}, mode="on")
        .result()
    )
    _assert_python_int_dimtags(sel)


def test_geometry_chain_spatial_refine_preserves_python_int(g):
    # GeometryChain._spatial_* returns the *original* atom unchanged
    # (core/_selection.py:882) — so a chained refine must not regress
    # the inherited Python-int contract either.
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
    faces = g.model.queries.boundary("box", dim=2)
    sel = (
        g.model.select(faces)
        .on_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), tol=1e-6)
        .result()
    )
    _assert_python_int_dimtags(sel)


# =====================================================================
# Numpy-int input parity (FLIPPED PIN — was the fail-loud asymmetry)
# =====================================================================
#
# Pre-flip this section asserted the geometry resolver REJECTED numpy
# tags (TypeError), framed as an intentional fail-loud divergence from
# the broker. Root cause was actually an unpropagated fix: the broker's
# NodeComposite._is_dimtag_tuple was widened to (int, np.integer) in
# commit 2e85abd ("5 library issues flagged during curriculum
# authoring", issue #4) but the six int-gates in core/_helpers.py were
# left bare `int`. They are now all widened via the shared `_INT`
# predicate, so the geometry/boolean resolver accepts numpy ints
# identically to the broker (the contract test_resolution_contract.py
# locks is name->entity precedence, NOT input dtype — unaffected).
#
# These pins assert numpy-input == Python-int-input PARITY (sibling to
# test_characterization_selection.py Item 4, but for the geometry
# family). A refactor that re-narrows any `_INT` gate to bare `int`
# must flip these back in the same commit.

def test_numpy_dimtag_tuple_to_geometry_select_parity(g):
    # resolve_to_dimtags dimtag-passthrough gate (_is_dimtag_tuple,
    # core/_helpers.py) now accepts (np.int64, np.int64).
    v = int(g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    np_sel = g.model.select((np.int64(3), np.int64(v))).result()
    py_sel = g.model.select((3, v)).result()
    _assert_python_int_dimtags(np_sel)
    assert list(np_sel) == list(py_sel)


def test_numpy_scalar_tag_to_geometry_select_parity(g):
    # resolve_to_dimtags bare-int gate now accepts a np.int64 scalar.
    v = int(g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    np_sel = g.model.select(np.int64(v), dim=3).result()
    py_sel = g.model.select(v, dim=3).result()
    _assert_python_int_dimtags(np_sel)
    assert list(np_sel) == list(py_sel)


def test_numpy_tag_list_to_geometry_select_parity(g):
    # resolve_to_dimtags list-recursion -> bare-int gate, numpy element.
    v = int(g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    np_sel = g.model.select([np.int64(v)], dim=3).result()
    py_sel = g.model.select([v], dim=3).result()
    _assert_python_int_dimtags(np_sel)
    assert list(np_sel) == list(py_sel)


def test_as_dimtags_numpy_parity(g):
    # Direct unit pin for the three `as_dimtags` gates (scalar / single
    # (dim,tag) / list element) — the geometry-builder normalisation
    # path, NOT reachable through g.model.select (which uses
    # resolve_to_dimtags). Needs a live gmsh model for resolve_dim.
    v = int(g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    assert as_dimtags(np.int64(v)) == as_dimtags(v)
    assert as_dimtags((np.int64(3), np.int64(v))) == [(np.int64(3), np.int64(v))]
    assert as_dimtags([np.int64(v)]) == as_dimtags([v])
    # resolve_to_dimtags parity sanity at the helper level too.
    assert (
        resolve_to_dimtags(np.int64(v), default_dim=3, session=g)
        == resolve_to_dimtags(v, default_dim=3, session=g)
    )
