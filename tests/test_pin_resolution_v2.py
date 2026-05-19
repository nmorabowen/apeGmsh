"""P0-C — name-resolution characterization battery (v2 plan).

These are **characterization pins**: every assertion locks the CURRENT
observed behavior of the name resolvers on this untouched worktree —
*including* the documented divergences and the one place where observed
reality contradicts the plan's hard-truth wording (see the module note
below). They exist so the deliberate behavior change a later phase
makes shows up as a *reviewed pin-flip diff*, never as silent drift.

Plan: ``docs/plans/selection-unification-v2.md`` §3 (HT2, HT3) + §6
(P0-C). Companion to ``tests/test_characterization_selection.py`` (v1
S0b) and the locked ``tests/test_resolution_contract.py`` /
``tests/test_target_resolution.py`` contract.

DO NOT "fix" a surprising assertion here. If a value looks wrong, that
is the point — it is reality on HEAD today. The owning phase flips the
pin in its own commit and the diff is the decision record.

> [!note] Observed reality vs HT wording (the lead must know)
> Two empirically-observed facts diverge from the plan's prose and are
> pinned to *observed reality* per the characterization cardinal rule:
>
> 1. HT2/§6 says the element path has "NO dim arg / returns the
>    unscoped union". The element *name-resolver*
>    (``_resolve_elem_ids``, FEMData.py:828-848) indeed does not
>    thread ``dim``; but the public ``fem.elements.get(dim=)``
>    parameter (FEMData.py:753) DOES exist and is honoured as a
>    *downstream group post-filter* (FEMData.py:807) — it is neither
>    rejected nor ignored: it silently empties the result when no
>    group matches the dim. ``test_pin_element_path_no_dim_scoping_
>    and_unconditional_part`` pins this exact observed behavior.
> 2. P0-C/§6 says the results single-selector dispatcher is
>    "exactly-one-selector; passing both raises". Observed reality:
>    passing ``pg=`` **and** ``label=`` together does NOT raise — it
>    returns their **sorted union** (results/_composites.py:312-335).
>    The exactly-one guard fires ONLY when ``ids=`` is combined with a
>    named selector (results/_composites.py:293-298).
>    ``test_pin_results_samename_label_pg_precedence`` pins both the
>    union (no-raise) and the ids+named raise.

No ``openseespy`` dependency (curated no-openseespy CI gate): pure
apeGmsh + gmsh + numpy. Every fixture is a **transfinite n=3**
structured cube → a deterministic 3x3x3 = 27-node lattice with 8 hex
cells (one box face = a 3x3 = 9-node grid), so every pinned count is an
exact integer, not "about N".
"""
from __future__ import annotations

import types

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh, Part
from apeGmsh.core.Labels import add_prefix
from apeGmsh.results._composites import (
    ElementResultsComposite,
    NodeResultsComposite,
)


# =====================================================================
# Fixtures — deterministic, raw-gmsh fabrication of the multi-dim /
# same-name states (mirrors tests/test_resolution_contract.py's
# _raw_multi_dim_pg / _raw_multi_dim_label idiom; transfinite for exact
# counts as in tests/test_characterization_selection.py's box_fem).
# =====================================================================


@pytest.fixture(scope="module")
def part_and_multidim_label_fem():
    """A transfinite ``Part('P1')`` whose nested volume also carries a
    (legitimate, Tier-1) multi-dim *label* ``reg`` at dim 3 (volume)
    and dim 2 (one face), fabricated via raw gmsh + ``add_prefix``
    exactly like ``test_resolution_contract._raw_multi_dim_label``.

    transfinite n=3 → 27 nodes / 8 hex; the dim-2 face slice is a
    3x3 = 9-node grid. ``Part`` nests its geometry label as
    ``'P1.body'`` (so transfinite targets that, not ``'blk'``).

    Follows the v1 S0b ``box_fem`` idiom: the session is closed inside
    a ``try/finally`` and only the **detached** fem is returned (no
    open gmsh session lives across the ``yield`` — gmsh is a process
    singleton and an open session across module fixtures double-
    finalizes).
    """
    blk = Part("blk")
    with blk:
        blk.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g = apeGmsh(model_name="pin_v2_part", verbose=False)
    g.begin()
    try:
        g.parts.add(blk, label="P1")
        g.model.sync()
        vol_tag = gmsh.model.getEntities(3)[0][1]
        face_tag = gmsh.model.getEntities(2)[0][1]
        lp = add_prefix("reg")
        tl3 = gmsh.model.addPhysicalGroup(3, [vol_tag])
        gmsh.model.setPhysicalName(3, tl3, lp)
        tl2 = gmsh.model.addPhysicalGroup(2, [face_tag])
        gmsh.model.setPhysicalName(2, tl2, lp)
        g.mesh.structured.set_transfinite_box("P1.body", n=3)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return fem


@pytest.fixture(scope="module")
def samename_label_pg_fem():
    """Two disjoint transfinite cubes: a LABEL ``Clash`` on volume 1
    and a same-named PG ``Clash`` on volume 2 (different, non-
    overlapping entity sets). The classic node-vs-element /
    label-vs-PG precedence scenario, the deterministic cousin of
    ``test_target_resolution._build_collision_scene``.

    Each cube transfinite n=3 → 27 nodes / 8 hex. The two cubes share
    no mesh nodes/elements, so label-set vs PG-set are exactly disjoint
    and equal-sized (27 nodes / 8 elements each). v1 S0b ``box_fem``
    idiom: detached fem returned, session closed in ``finally``.
    """
    g = apeGmsh(model_name="pin_v2_clash", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="v1")
        g.model.geometry.add_box(3, 0, 0, 1, 1, 1, label="v2")
        g.model.sync()
        g.labels.add(3, [1], "Clash")          # LABEL on volume 1
        g.physical.add(3, [2], name="Clash")   # PG on volume 2
        # so volume 1's elements are meshed + extracted into the fem
        g.physical.add_volume([1], name="keepv1")
        g.mesh.structured.set_transfinite_box("v1", n=3)
        g.mesh.structured.set_transfinite_box("v2", n=3)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    return fem


def _selection_mixin(comp_cls, fem):
    """A real results composite over a stub ``_r`` (no reader/openseespy).

    ``_SelectionMixin._resolve_node_ids`` / ``_resolve_element_ids``
    only read ``self._r._fem`` (results/_composites.py:304, :358) — they
    never touch the reader for pg/label resolution. So this exercises
    the *real* production resolver verbatim, with no results file. Same
    idiom as ``tests/test_characterization_selection._selection_mixin``.
    """
    comp = comp_cls.__new__(comp_cls)
    comp._r = types.SimpleNamespace(_fem=fem)
    return comp


# =====================================================================
# PIN 1 — broker silently merges a multi-dim PG; contract engines raise
# =====================================================================


def test_pin_broker_multidim_pg_silently_merges():
    """HT3 — broker ``_group_set._resolve`` SILENTLY MERGES a multi-dim
    PG while the contract engines RAISE for the same name. Pins BOTH
    sides of the divergence.

    Source: ``mesh/_group_set.py:146-180`` (broker ``_resolve``: a
    string matching several dims is merged via ``_merge_infos``, no
    raise) vs ``mesh/PhysicalGroups.py:398-404`` (``entities`` raises
    ValueError on a multi-dim PG name) and ``core/_resolution.py:122-130``
    (``_resolve_target`` raises ValueError). Mirrors
    ``test_resolution_contract.test_rule3_pg_multidim_raises_everywhere``
    which deliberately EXCLUDES the broker.

    KEPT-FOREVER invariant (HT3 / ratified R-v2-5): the resolvers stay
    semantically separate. This pin characterizes-and-keeps; do NOT
    silently "fix" the broker merge. (A deliberate future change here
    would be a same-commit reviewed pin-flip.)

    Inline single-session test (not a module fixture): the contract-
    engine side calls LIVE gmsh (``g.physical.entities`` /
    ``g.loads._resolve_target``) so it MUST run inside a still-open
    session — the exact reason v1 S0b's
    ``test_item4_np_int64_dimtag_resolves_same_as_python_int`` /
    ``test_item6_viz_*`` use an inline session queried before
    ``.end()``. Raw-gmsh multi-dim PG fabrication mirrors
    ``test_resolution_contract._raw_multi_dim_pg``; transfinite n=3 →
    27 nodes / 8 hex; the dim-2 face slice is a 3x3 = 9-node grid.
    """
    g = apeGmsh(model_name="pin_v2_pg", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
        g.model.sync()
        t3 = gmsh.model.addPhysicalGroup(3, [1])
        gmsh.model.setPhysicalName(3, t3, "Mixed")
        t2 = gmsh.model.addPhysicalGroup(2, [1])
        gmsh.model.setPhysicalName(2, t2, "Mixed")
        g.mesh.structured.set_transfinite_box("box", n=3)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)

        # Anchor: deterministic transfinite lattice (27 nodes, 8 hex).
        assert len(fem.nodes.ids) == 27

        # (a) BROKER side — silent merge, NO exception.
        res = fem.nodes.select(pg="Mixed")
        assert len(res.ids) == 27      # dim-3 (27) ∪ dim-2 face (9) = 27
        assert (
            len(np.asarray(fem.nodes.physical.node_ids("Mixed"))) == 27
        )
        # The merge is real: a dim-scoped read collapses to that one
        # slice (dim-2 face = 3x3 = 9-node grid; dim-3 = all 27).
        assert len(fem.nodes.select(pg="Mixed", dim=2).ids) == 9
        assert len(fem.nodes.select(pg="Mixed", dim=3).ids) == 27
        # The element broker likewise does NOT raise on the multi-dim
        # PG; it returns only the dim-3 hex groups (8 cells, no dim-2
        # elems extracted) — pinned so a future raise here is reviewed.
        eres = fem.elements.select(pg="Mixed").groups()
        assert sum(len(grp.ids) for grp in eres) == 8
        assert sorted({grp.element_type.dim for grp in eres}) == [3]

        # (b) CONTRACT-ENGINE side — same name RAISES ValueError.
        with pytest.raises(
            ValueError,
            match=r"multiple dimensions \[2, 3\].*not supported",
        ):
            g.physical.entities("Mixed")
        with pytest.raises(
            ValueError,
            match=r"multiple dimensions \[2, 3\].*not supported",
        ):
            g.loads._resolve_target("Mixed", source="pg")
    finally:
        g.end()


# =====================================================================
# PIN 2 — element path: name-resolver is NOT dim-scoped, dim= is a
# downstream post-filter, and the part branch is unconditional
# =====================================================================


def test_pin_element_path_no_dim_scoping_and_unconditional_part(
    part_and_multidim_label_fem,
):
    """HT2 — the three orthogonal node↔element resolver axes, runtime-
    proven on a multi-dim label + a Part.

    Source: node ``mesh/FEMData.py:456-496`` (``:478`` dim-threaded
    label, ``:489`` ``if dim is None and t in _part_node_map`` dim-
    gated part, ``:480/:485`` ``except KeyError``) vs element
    ``mesh/FEMData.py:851-880`` (``:868`` ``labels.element_ids(t)``
    with NO ``dim``, ``:875`` ``if t in _part_elem_map`` unconditional
    part, ``:869/:873`` ``except (KeyError, ValueError)``). Element
    ``get(dim=)`` itself is FEMData.py:753 → applied as the group
    post-filter at FEMData.py:807.

    KEPT-FOREVER invariant (HT2 / ratified R-v2-5): name-resolvers stay
    separate; this difference is load-bearing correctness.

    OBSERVED-REALITY NOTE (see module docstring §1): the plan's "element
    path has NO dim arg" is true of the *name-resolver* only — the
    public ``elements.get(dim=)`` exists and is honoured as a silent
    post-filter (empties the result; never raises/ignores). Pinned as
    observed.
    """
    fem = part_and_multidim_label_fem

    # --- axis 1: node path IS dim-scoped (different counts per dim) ---
    n2 = len(fem.nodes.select(label="reg", dim=2).ids)
    n3 = len(fem.nodes.select(label="reg", dim=3).ids)
    nN = len(fem.nodes.select(label="reg").ids)            # dim=None
    assert n2 == 9       # dim-2 face = 3x3 grid
    assert n3 == 27      # dim-3 volume = full lattice
    assert nN == 27      # dim=None UNIONs → 27 (matches dim-3 here)
    assert n2 != n3      # the scoping is real

    # --- element name-resolver is NOT dim-threaded: the unscoped
    #     union (label "reg") returns the dim-3 hex group. dim= is then
    #     a downstream group post-filter — dim=2 silently EMPTIES (no
    #     raise, no ignore); dim=3 keeps the 8 cells. ---
    e_union = fem.elements.select(label="reg").groups()
    assert sum(len(grp.ids) for grp in e_union) == 8
    assert sorted({grp.element_type.dim for grp in e_union}) == [3]

    e_d2 = fem.elements.select(label="reg", dim=2).groups()
    assert sum(len(grp.ids) for grp in e_d2) == 0       # EMPTIED
    assert list(e_d2) == []                              # GroupResult empty
    e_d3 = fem.elements.select(label="reg", dim=3).groups()
    assert sum(len(grp.ids) for grp in e_d3) == 8        # kept

    # --- axis 2 + 3: part branch dim-gating + exception breadth ---
    # node part branch is `if dim is None and t in _part_node_map`
    # (FEMData.py:489) → a dim-scoped part target falls through to the
    # final `raise KeyError` (FEMData.py:492).
    with pytest.raises(
        KeyError,
        match=r"No label, physical group, or part named 'P1' at dim=2",
    ):
        fem.nodes.select(target="P1", dim=2)
    # dim=None → the part branch fires; 27-node lattice.
    assert len(fem.nodes.select(target="P1").ids) == 27

    # element part branch is `if t in _part_elem_map` (FEMData.py:875)
    # — UNCONDITIONAL (no dim gate): it resolves the part with NO
    # exception, then dim= post-filters. target="P1" → 8 hex; with
    # dim=2 the post-filter silently EMPTIES (still NO exception —
    # contrast the node KeyError above).
    ep = fem.elements.select(target="P1").groups()
    assert sum(len(grp.ids) for grp in ep) == 8
    ep2 = fem.elements.select(target="P1", dim=2).groups()
    assert sum(len(grp.ids) for grp in ep2) == 0         # EMPTIED, no raise
    assert list(ep2) == []


# =====================================================================
# PIN 3 — element same-name LABEL/PG precedence (audit says UNPINNED)
# =====================================================================


def test_pin_element_samename_label_pg_precedence(
    samename_label_pg_fem,
):
    """Discover & pin the element-broker precedence when one name is
    BOTH a label and a PG on different entity sets — the audit flags
    element-side precedence as UNPINNED.

    Source: ``mesh/FEMData.py:861-876`` (``_resolve_one_elem_target``:
    tries ``labels.element_ids(t)`` FIRST, then
    ``physical.element_ids(t)``, then ``_part_elem_map``). Observed:
    **LABEL WINS** — identical to the node path (FEMData.py:477-486)
    and to the locked ``test_target_resolution`` /
    ``test_characterization_selection`` item-3 contract.

    KEPT-FOREVER invariant: the documented label→PG→part precedence
    (``test_resolution_contract.py`` Rule 2). Pinning it on the element
    side closes the audited gap; a future precedence change is a
    same-commit reviewed pin-flip.
    """
    fem = samename_label_pg_fem

    lab_e = sorted(
        int(x) for x in fem.elements.labels.element_ids("Clash"))
    pg_e = sorted(
        int(x) for x in fem.elements.physical.element_ids("Clash"))
    # Scenario integrity: disjoint, equal-sized (8 hex per cube).
    assert len(lab_e) == 8
    assert len(pg_e) == 8
    assert set(lab_e).isdisjoint(pg_e)

    got = sorted(
        int(x) for grp in fem.elements.select(target="Clash").groups() for x in grp.ids
    )
    assert got == lab_e        # LABEL wins
    assert got != pg_e         # did NOT fall through to the PG
    assert len(got) == 8

    # Node-path control: same precedence (label wins) — pinned so the
    # two paths cannot silently diverge on this case.
    lab_n = sorted(
        int(x) for x in fem.nodes.labels.node_ids("Clash"))
    pg_n = sorted(
        int(x) for x in fem.nodes.physical.node_ids("Clash"))
    got_n = sorted(int(x) for x in fem.nodes.select(target="Clash").ids)
    assert got_n == lab_n
    assert got_n != pg_n
    assert len(got_n) == 27


# =====================================================================
# PIN 4 — results single-selector dispatcher: same-name label/PG +
# the (observed) selector-combination rules
# =====================================================================


def test_pin_results_samename_label_pg_precedence(
    samename_label_pg_fem,
):
    """Pin the results ``_SelectionMixin`` dispatcher on a same-name
    label/PG: ``pg=`` vs ``label=`` resolve to DIFFERENT id sets, and
    pin the OBSERVED selector-combination rules.

    Source: ``results/_composites.py:277-387``
    (``_resolve_node_ids`` / ``_resolve_element_ids``). The fixture is
    the lightest viable Results surface: a real composite over a stub
    ``_r=SimpleNamespace(_fem=fem)`` — the resolver only reads
    ``self._r._fem`` (``:304``, ``:358``), never the reader/openseespy
    (same idiom as v1 S0b ``_selection_mixin``). No heavy results file
    is needed.

    FLIPS IN: P3 (spatial-dedup / removal phase touches the resolver
    family; any change here is a same-commit reviewed pin-flip).

    OBSERVED-REALITY NOTE (see module docstring §2): P0-C/§6 says
    "exactly-one-selector; passing both raises". Observed: ``pg=`` +
    ``label=`` together does NOT raise — it returns the **sorted
    union** (``:312-335``: no ``target=`` param, no precedence merge,
    ``np.unique`` of concatenated blocks). The exactly-one guard fires
    ONLY when ``ids=`` is combined with a named selector
    (``:293-298`` / ``:347-352``). Both pinned as observed.
    """
    fem = samename_label_pg_fem
    ncomp = _selection_mixin(NodeResultsComposite, fem)
    ecomp = _selection_mixin(ElementResultsComposite, fem)

    lab_n = sorted(
        int(x) for x in fem.nodes.labels.node_ids("Clash"))
    pg_n = sorted(
        int(x) for x in fem.nodes.physical.node_ids("Clash"))

    rp = ncomp._resolve_node_ids(
        pg="Clash", label=None, selection=None, ids=None)
    rl = ncomp._resolve_node_ids(
        pg=None, label="Clash", selection=None, ids=None)
    assert sorted(rp.tolist()) == pg_n      # pg= → strictly PG nodes
    assert sorted(rl.tolist()) == lab_n     # label= → strictly LABEL nodes
    assert len(rp) == 27
    assert len(rl) == 27
    assert set(rp.tolist()).isdisjoint(rl.tolist())   # DIFFERENT sets

    # OBSERVED: pg= AND label= together → NO raise; sorted union (no
    # target= precedence merge). 27 ∪ 27 disjoint = 54, sorted.
    rb = ncomp._resolve_node_ids(
        pg="Clash", label="Clash", selection=None, ids=None)
    assert len(rb) == 54
    assert rb.tolist() == np.unique(np.concatenate([rp, rl])).tolist()
    assert rb.tolist() == sorted(rb.tolist())

    # The exactly-one guard fires ONLY for ids= + a named selector.
    with pytest.raises(
        ValueError,
        match=r"Provide one of pg=, label=, selection=, or ids= "
              r"\(not multiple\)\.",
    ):
        ncomp._resolve_node_ids(
            pg="Clash", label=None, selection=None, ids=[1, 2])

    # Element dispatcher mirrors the node one.
    lab_e = sorted(
        int(x) for x in fem.elements.labels.element_ids("Clash"))
    pg_e = sorted(
        int(x) for x in fem.elements.physical.element_ids("Clash"))
    ep = ecomp._resolve_element_ids(
        pg="Clash", label=None, selection=None, ids=None)
    el = ecomp._resolve_element_ids(
        pg=None, label="Clash", selection=None, ids=None)
    assert sorted(ep.tolist()) == pg_e
    assert sorted(el.tolist()) == lab_e
    assert len(ep) == 8
    assert len(el) == 8
    assert set(ep.tolist()).isdisjoint(el.tolist())
    with pytest.raises(
        ValueError,
        match=r"Provide one of pg=, label=, selection=, or ids= "
              r"\(not multiple\)\.",
    ):
        ecomp._resolve_element_ids(
            pg=None, label="Clash", selection=None, ids=[1])
