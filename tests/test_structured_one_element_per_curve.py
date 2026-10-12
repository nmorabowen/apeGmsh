"""Tests for g.mesh.structured.one_element_per_curve() — R3-S3 (#1780).

The helper closes the truss reference script's "no one-element-per-member
switch" API gap: each selected curve is meshed with two nodes, one 1-D
element.  Oracles are counts read straight from gmsh after ``generate``:

* the Pratt truss of ``skills/apegmsh/references/model-scripts.md``
  (12 joints, 21 bars): mesh nodes == geometry points and 1-D elements
  == curves, both with the family labels and with no labels;
* a mixed frame + meshed slab: called with no labels, the slab's 2-D
  element count equals the count of the same model meshed without it;
* the fail-closed refusals.
"""
from __future__ import annotations

import dataclasses

import gmsh
import pytest

from apeGmsh import apeGmsh

# --- Pratt truss, restated from model-scripts.md --------------------------
N_PANELS = 6
PANEL = 4.0
DEPTH = 4.0
MEMBERS = {
    "bottom_chord": ["B0-B1", "B1-B2", "B2-B3", "B3-B4", "B4-B5", "B5-B6"],
    "top_chord": ["T1-T2", "T2-T3", "T3-T4", "T4-T5"],
    "end_posts": ["B0-T1", "T5-B6"],
    "verticals": ["B1-T1", "B2-T2", "B3-T3", "B4-T4", "B5-T5"],
    "diagonals": ["T1-B2", "T2-B3", "T4-B3", "T5-B4"],
}
N_JOINTS = (N_PANELS + 1) + (N_PANELS - 1)            # 12
N_BARS = sum(len(b) for b in MEMBERS.values())         # 21


def _build_truss(g) -> None:
    for i in range(N_PANELS + 1):
        g.model.geometry.add_point(i * PANEL, 0.0, 0.0, label=f"B{i}")
    for i in range(1, N_PANELS):
        g.model.geometry.add_point(i * PANEL, DEPTH, 0.0, label=f"T{i}")
    for bars in MEMBERS.values():
        for bar in bars:
            start, end = bar.split("-")
            g.model.geometry.add_line(start, end, label=bar)
    for family, bars in MEMBERS.items():
        g.physical.from_labels(bars, name=family)
    # A coarse global size that would still split every 4 m bar.
    g.mesh.sizing.set_global_size(1.0)


def _n_nodes() -> int:
    return len(gmsh.model.mesh.getNodes()[0])


def _n_elements(dim: int, tag: int = -1) -> int:
    _types, elem_tags, _nodes = gmsh.model.mesh.getElements(dim, tag)
    return int(sum(len(t) for t in elem_tags))


def _transfinite_directives(g) -> list[dict]:
    return [d for d in g.mesh._directives if d["kind"] == "transfinite_curve"]


# ---------------------------------------------------------------------------
# Truss oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("by_labels", [True, False], ids=["labels", "no_labels"])
def test_pratt_truss_one_element_per_bar(by_labels):
    with apeGmsh(model_name="r3s3_truss", verbose=False) as g:
        _build_truss(g)
        assert len(gmsh.model.getEntities(0)) == N_JOINTS
        assert len(gmsh.model.getEntities(1)) == N_BARS
        if by_labels:
            g.mesh.structured.one_element_per_curve(*MEMBERS)
        else:
            g.mesh.structured.one_element_per_curve()
        g.mesh.generation.generate(dim=1)

        assert _n_nodes() == N_JOINTS
        assert _n_elements(1) == N_BARS
        for _, c in gmsh.model.getEntities(1):
            assert _n_elements(1, c) == 1


def test_pratt_truss_without_helper_splits_bars():
    """Control: the oracle above is not vacuous at this mesh size."""
    with apeGmsh(model_name="r3s3_truss_ctrl", verbose=False) as g:
        _build_truss(g)
        g.mesh.generation.generate(dim=1)
        assert _n_nodes() > N_JOINTS
        assert _n_elements(1) > N_BARS


def test_labels_select_only_their_curves():
    with apeGmsh(model_name="r3s3_subset", verbose=False) as g:
        _build_truss(g)
        g.mesh.structured.one_element_per_curve("bottom_chord")
        g.mesh.generation.generate(dim=1)
        chord = set(g.physical.entities("bottom_chord", dim=1))
        for _, c in gmsh.model.getEntities(1):
            if c in chord:
                assert _n_elements(1, c) == 1
            else:
                assert _n_elements(1, c) > 1


def test_repeated_label_sets_each_curve_once():
    with apeGmsh(model_name="r3s3_dupe", verbose=False) as g:
        _build_truss(g)
        g.mesh.structured.one_element_per_curve("top_chord", "top_chord")
        tags = [d["tag"] for d in _transfinite_directives(g)]
        assert len(tags) == len(MEMBERS["top_chord"]) == len(set(tags))
        assert all(d["n_nodes"] == 2 for d in _transfinite_directives(g))


# ---------------------------------------------------------------------------
# Mixed frame + slab
# ---------------------------------------------------------------------------

def _build_frame_and_slab(g) -> int:
    """Four columns, one free beam, and a 6 x 4 slab at z = 3.  Returns the slab tag."""
    slab = g.model.geometry.add_rectangle(0.0, 0.0, 3.0, 6.0, 4.0, label="slab")
    for i, (x, y) in enumerate([(0, 0), (6, 0), (6, 4), (0, 4)]):
        g.model.geometry.add_point(x, y, 0.0, label=f"base{i}")
        g.model.geometry.add_point(x, y, 3.0, label=f"top{i}")
        g.model.geometry.add_line(f"base{i}", f"top{i}", label=f"col{i}")
    g.physical.from_labels([f"col{i}" for i in range(4)], name="columns")
    g.model.geometry.add_point(0.0, 0.0, 6.0, label="b0")
    g.model.geometry.add_point(6.0, 0.0, 6.0, label="b1")
    g.model.geometry.add_line("b0", "b1", label="beam")
    g.mesh.sizing.set_global_size(0.5)
    return slab


def test_no_labels_leaves_slab_mesh_unchanged():
    with apeGmsh(model_name="r3s3_mixed_ref", verbose=False) as g:
        slab = _build_frame_and_slab(g)
        g.mesh.generation.generate(dim=2)
        ref_slab = _n_elements(2, slab)
        ref_free = [_n_elements(1, c) for c in g.labels.entities("beam", dim=1)]

    with apeGmsh(model_name="r3s3_mixed", verbose=False) as g:
        slab = _build_frame_and_slab(g)
        g.mesh.structured.one_element_per_curve()
        g.mesh.generation.generate(dim=2)
        assert _n_elements(2, slab) == ref_slab
        # The four columns and the beam: one element each.
        for name in ["beam", *[f"col{i}" for i in range(4)]]:
            (c,) = g.labels.entities(name, dim=1)
            assert _n_elements(1, c) == 1
        # Slab edges were not touched.
        upward_bound = [
            c for _, c in gmsh.model.getEntities(1)
            if len(gmsh.model.getAdjacencies(1, c)[0])
        ]
        assert len(upward_bound) == 4
        assert all(_n_elements(1, c) > 1 for c in upward_bound)
    assert ref_slab > 0 and ref_free[0] > 1


# ---------------------------------------------------------------------------
# Refusals (fail closed, and nothing applied)
# ---------------------------------------------------------------------------

def test_unknown_label_raises_listing_labels():
    with apeGmsh(model_name="r3s3_unknown", verbose=False) as g:
        _build_truss(g)
        with pytest.raises(KeyError, match="Available labels") as exc:
            g.mesh.structured.one_element_per_curve("bottom_chord", "nope")
        assert "B0-B1" in str(exc.value)
        assert _transfinite_directives(g) == []


def test_label_with_no_curve_raises():
    with apeGmsh(model_name="r3s3_nocurve", verbose=False) as g:
        _build_frame_and_slab(g)
        with pytest.raises(ValueError, match="resolves to no curve"):
            g.mesh.structured.one_element_per_curve("columns", "slab")
        assert _transfinite_directives(g) == []


def test_surface_boundary_curve_is_refused():
    with apeGmsh(model_name="r3s3_edge", verbose=False) as g:
        slab = _build_frame_and_slab(g)
        _up, down = gmsh.model.getAdjacencies(2, slab)
        g.labels.add(1, [int(down[0])], name="slab_edge")
        with pytest.raises(ValueError, match="not free"):
            g.mesh.structured.one_element_per_curve("columns", "slab_edge")
        assert _transfinite_directives(g) == []


def test_curve_embedded_in_surface_is_refused():
    with apeGmsh(model_name="r3s3_embed", verbose=False) as g:
        slab = _build_frame_and_slab(g)
        g.model.geometry.add_point(1.0, 1.0, 3.0, label="e0")
        g.model.geometry.add_point(5.0, 1.0, 3.0, label="e1")
        g.model.geometry.add_line("e0", "e1", label="rib")
        (rib,) = g.labels.entities("rib", dim=1)
        gmsh.model.mesh.embed(1, [rib], 2, slab)
        with pytest.raises(ValueError, match="not free"):
            g.mesh.structured.one_element_per_curve("rib")
        # No labels: the embedded rib is not free, so it is skipped.
        g.mesh.structured.one_element_per_curve()
        tags = {d["tag"] for d in _transfinite_directives(g)}
        assert rib not in tags
        assert len(tags) == 5


def test_no_free_curve_raises():
    with apeGmsh(model_name="r3s3_nofree", verbose=False) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 2.0, 1.0, label="plate")
        with pytest.raises(ValueError, match="no free curve"):
            g.mesh.structured.one_element_per_curve()


def test_non_string_label_raises():
    with apeGmsh(model_name="r3s3_type", verbose=False) as g:
        _build_truss(g)
        (c,) = g.labels.entities("B0-B1", dim=1)
        with pytest.raises(TypeError, match="set_transfinite_curve"):
            g.mesh.structured.one_element_per_curve(c)
        assert _transfinite_directives(g) == []


# ---------------------------------------------------------------------------
# Non-conformal ties (review of #1783): a tied member is never coarsened
# ---------------------------------------------------------------------------

def _build_beam_with_bar(g) -> None:
    """The reviewer's reproducer: a 4 x 0.3 x 0.5 box with a bar inside it
    (not fragmented, so gmsh sees a free curve), plus one free post."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 4.0, 0.3, 0.5, label="beam")
    g.model.geometry.add_point(0.1, 0.15, 0.1, label="p0")
    g.model.geometry.add_point(3.9, 0.15, 0.1, label="p1")
    g.model.geometry.add_line("p0", "p1", label="bar")
    g.model.geometry.add_point(5.0, 0.0, 0.0, label="q0")
    g.model.geometry.add_point(5.0, 0.0, 3.0, label="q1")
    g.model.geometry.add_line("q0", "q1", label="post")
    # g.reinforce / g.embed resolve host and bars as physical groups.
    g.labels.promote_to_physical("beam")
    g.labels.promote_to_physical("bar")
    g.mesh.sizing.set_global_size(0.1)



@dataclasses.dataclass
class _FutureTieDef:
    """A definition kind the helper has never heard of, using the role
    naming convention (``*_label``)."""
    host_label: str
    follower_label: str
    name: str | None = None


@dataclasses.dataclass
class _OddlyNamedDef:
    """Another unknown kind whose curve reference is in an arbitrary field."""
    kind: str
    curve: str
    name: str | None = None


def _tie(g, source: str) -> None:
    if source == "constraints.embedded":
        g.constraints.embedded("beam", "bar")
    elif source == "constraints.tie":
        g.constraints.tie("beam", "bar")
    elif source == "constraints.tied_contact":
        g.constraints.tied_contact("beam", "bar")
    elif source == "constraints.bc":
        g.constraints.bc("bar", dofs=[1, 1, 1])
    elif source == "reinforce":
        g.reinforce.reinforce(
            host="beam", bars="bar", perfect=1.0e10, bar_diameter=0.02,
        )
    elif source == "embed":
        g.embed(host="beam", nodes="bar")
    elif source == "stub_label":
        g.constraints.constraint_defs.append(_FutureTieDef("beam", "bar"))
    else:
        g.constraints.constraint_defs.append(_OddlyNamedDef("future", "bar"))


def _curve_elements(g, label: str) -> int:
    (c,) = g.labels.entities(label, dim=1)
    return _n_elements(1, c)


#: source -> the "<DefClass> g.<composite>.<store>" its refusal names.
_TIE_WHERE = {
    "constraints.embedded": "EmbeddedDef g.constraints.constraint_defs[0]",
    "constraints.tie": "TieDef g.constraints.constraint_defs[0]",
    "constraints.tied_contact": "TiedContactDef g.constraints.constraint_defs[0]",
    "constraints.bc": "BCDef g.constraints._bc_defs[0]",
    "reinforce": "ReinforceDef g.reinforce.reinforce_defs[0]",
    "embed": "EmbedDef g.embed.embed_defs[0]",
    "stub_label": "_FutureTieDef g.constraints.constraint_defs[0]",
    "stub_any_field": "_OddlyNamedDef g.constraints.constraint_defs[0]",
}
_TIE_SOURCES = list(_TIE_WHERE)

# Tie definitions resolve only on a mesh (and the stubs never), so the
# session's exit autosave may warn; that is not what these tests check.
_ignore_autosave = pytest.mark.filterwarnings("ignore:autosave to")


@_ignore_autosave
@pytest.mark.parametrize("source", _TIE_SOURCES)
def test_no_labels_keeps_a_tied_bar_at_its_sizing_count(source):
    with apeGmsh(model_name="r3s3_bar_ref", verbose=False) as g:
        _build_beam_with_bar(g)
        g.mesh.generation.generate(dim=3)
        ref_bar = _curve_elements(g, "bar")
    assert ref_bar > 30  # 38 at size 0.1

    with apeGmsh(model_name="r3s3_bar", verbose=False) as g:
        _build_beam_with_bar(g)
        _tie(g, source)
        g.mesh.structured.one_element_per_curve()
        tags = {d["tag"] for d in _transfinite_directives(g)}
        assert tags == set(g.labels.entities("post", dim=1))
        g.mesh.generation.generate(dim=3)
        assert _curve_elements(g, "bar") == ref_bar
        assert _curve_elements(g, "post") == 1


@_ignore_autosave
@pytest.mark.parametrize("source", _TIE_SOURCES)
def test_named_tied_bar_is_refused_naming_the_def(source):
    with apeGmsh(model_name="r3s3_bar_named", verbose=False) as g:
        _build_beam_with_bar(g)
        _tie(g, source)
        with pytest.raises(ValueError, match="not free") as exc:
            g.mesh.structured.one_element_per_curve("post", "bar")
        assert _TIE_WHERE[source] in str(exc.value)
        assert _transfinite_directives(g) == []


@_ignore_autosave
def test_tie_with_unresolvable_member_fails_closed():
    with apeGmsh(model_name="r3s3_bar_bad", verbose=False) as g:
        _build_beam_with_bar(g)
        g.constraints.embedded("beam", "rebar")
        with pytest.raises(ValueError, match="cannot tell which curves"):
            g.mesh.structured.one_element_per_curve()
        assert _transfinite_directives(g) == []


@dataclasses.dataclass
class _OpaqueDef:
    """An unknown kind with no label or entity field at all."""
    kind: str
    stiffness: float


@_ignore_autosave
@pytest.mark.parametrize(
    "defn, why",
    [(_OpaqueDef("future", 1.0), "no label or entity field"),
     (object(), "not a dataclass")],
    ids=["no_reference_field", "not_a_dataclass"],
)
def test_def_whose_references_cannot_be_enumerated_is_refused(defn, why):
    with apeGmsh(model_name="r3s3_opaque", verbose=False) as g:
        _build_beam_with_bar(g)
        g.constraints.constraint_defs.append(defn)
        with pytest.raises(ValueError, match=why):
            g.mesh.structured.one_element_per_curve()
        assert _transfinite_directives(g) == []


@_ignore_autosave
def test_tie_declared_after_the_call_is_not_seen():
    """Pins the documented order rule: the helper reads the ties that
    exist when it is called, so call it after every tie is declared."""
    with apeGmsh(model_name="r3s3_order", verbose=False) as g:
        _build_beam_with_bar(g)
        g.mesh.structured.one_element_per_curve()
        g.constraints.embedded("beam", "bar")
        g.mesh.generation.generate(dim=3)
        assert _curve_elements(g, "bar") == 1
