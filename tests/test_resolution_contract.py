"""
The dimensional-reference resolution contract — locked across consumers.

apeGmsh has one well-posed operation behind many call sites: *resolve a
symbolic reference (label / PG / part / raw dimtags) to the entities of
the dimension a consumer requires, failing loud otherwise.*

Contract:
  1. raw (dim,tag)/list  -> verbatim (explicit intent, unscoped)
  2. string precedence    -> label (Tier 1) -> PG (Tier 2) -> part (Tier 3)
  3. PG                   -> single-dim; multi-dim => raise
  4. label                -> scoped to required D; union when D is None
  5. part                 -> spans dims; scoped to D
  6. NEVER silently truncate / silently return empty / bind all nodes
  7. consumer declares D; dim-agnostic node consumers use D=None

This file is the regression lock.  It deliberately spans the
``_helpers`` resolver, loads, masses and constraints so the contract
cannot silently diverge between them again.
"""
import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.core._helpers import resolve_to_tags, resolve_to_dimtags
from apeGmsh.core.Labels import add_prefix


# =====================================================================
# Helpers — raw-Gmsh fabrication of the "forbidden"/multi-dim states
# =====================================================================

def _raw_multi_dim_pg(name="Mixed"):
    """Box session carrying a (forbidden) multi-dim PG via raw gmsh."""
    g = apeGmsh(model_name="contract_pg", verbose=False)
    g.__enter__()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.model.sync()
    t3 = gmsh.model.addPhysicalGroup(3, [1]); gmsh.model.setPhysicalName(3, t3, name)
    t2 = gmsh.model.addPhysicalGroup(2, [1]); gmsh.model.setPhysicalName(2, t2, name)
    return g


def _raw_multi_dim_label(name="region"):
    """Box session with a legitimate multi-dim Tier-1 label (dims 2+3)."""
    g = apeGmsh(model_name="contract_lbl", verbose=False)
    g.__enter__()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.model.sync()
    lp = add_prefix(name)
    t3 = gmsh.model.addPhysicalGroup(3, [1]); gmsh.model.setPhysicalName(3, t3, lp)
    t2 = gmsh.model.addPhysicalGroup(2, [1]); gmsh.model.setPhysicalName(2, t2, lp)
    return g


# =====================================================================
# Rule 3 — a multi-dim PG raises in EVERY resolver (consistency lock)
# =====================================================================

def test_rule3_pg_multidim_raises_in_physical_entities():
    g = _raw_multi_dim_pg()
    try:
        with pytest.raises(ValueError, match="multiple dimensions|not supported"):
            g.physical.entities("Mixed")
    finally:
        g.__exit__(None, None, None)


@pytest.mark.parametrize("call", [
    lambda g: resolve_to_tags("Mixed", dim=None, session=g),
    lambda g: resolve_to_dimtags("Mixed", default_dim=3, session=g),
    lambda g: g.loads._resolve_target("Mixed", source="pg"),
    lambda g: g.masses._resolve_target("Mixed", source="pg"),
    lambda g: g.constraints._entities_for_label("Mixed"),
])
def test_rule3_pg_multidim_raises_everywhere(call):
    g = _raw_multi_dim_pg()
    try:
        with pytest.raises(ValueError, match="multiple dimensions|not supported"):
            call(g)
    finally:
        g.__exit__(None, None, None)


# =====================================================================
# Rule 4 — a multi-dim LABEL: scope to D, UNION at D=None (never raise)
# =====================================================================

def test_rule4_label_scoped_with_dim():
    g = _raw_multi_dim_label()
    try:
        assert set(resolve_to_tags("region", dim=2, session=g)) == {1}
        assert set(resolve_to_tags("region", dim=3, session=g)) == {1}
    finally:
        g.__exit__(None, None, None)


def test_rule4_label_union_at_dim_none_does_not_raise():
    """Phase B: _resolve_string must UNION a multi-dim label at
    dim=None (mirroring FEMData/_group_set), not raise the way a bare
    Labels.entities(dim=None) does."""
    g = _raw_multi_dim_label()
    try:
        tags = resolve_to_tags("region", dim=None, session=g)   # must not raise
        assert tags  # non-empty union
        dts = resolve_to_dimtags("region", default_dim=3, session=g)
        assert {d for d, _ in dts} == {2, 3}
    finally:
        g.__exit__(None, None, None)


def test_rule4_loads_scope_label_to_required_dim():
    g = _raw_multi_dim_label()
    try:
        s2 = g.loads._resolve_target("region", source="label", expected_dim=2)
        assert {d for d, _ in s2} == {2}
        with pytest.raises(ValueError, match="requires dim=1|dimension"):
            g.loads._resolve_target("region", source="label", expected_dim=1)
    finally:
        g.__exit__(None, None, None)


# =====================================================================
# Rule 2 + Tier 3 — _helpers precedence: label->PG->part; ms excluded
# =====================================================================

def test_tier3_part_label_resolves_through_helpers():
    """Phase C: a part label resolves through resolve_to_tags /
    resolve_to_dimtags (previously KeyError'd) — same precedence
    loads/FEMData use."""
    from apeGmsh import Part
    blk = Part("blk")
    with blk:
        blk.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    with apeGmsh(model_name="contract_part", verbose=False) as g:
        g.parts.add(blk, label="P1")
        dts = resolve_to_dimtags("P1", default_dim=3, session=g)
        assert dts                                  # part resolved
        assert resolve_to_tags("P1", dim=3, session=g)  # scoped to dim 3


def test_meshselection_name_is_not_a_helpers_tier():
    """Deliberate divergence: a mesh-selection name is a post-mesh
    node concept, not a geometry entity — resolve_to_tags must NOT
    resolve it (loads/masses handle ms via their own path)."""
    with apeGmsh(model_name="contract_ms", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        with pytest.raises(KeyError):
            resolve_to_tags("not_a_geometry_name", dim=3, session=g)


# =====================================================================
# Rule 6 — constraints FAIL LOUD, never silently bind/skip
# =====================================================================

def test_constraint_entities_for_label_unknown_raises_keyerror():
    """Was: return [] -> misleading 'no host elements' skip."""
    with apeGmsh(model_name="contract_c1", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        with pytest.raises(KeyError, match="resolved to neither|not found"):
            g.constraints._entities_for_label("does_not_exist")


def test_node_to_surface_master_multi_entity_raises():
    """Was: m_tags[0] silent first-match.  A master that resolves to
    >1 dim-0 entity must fail loud."""
    with apeGmsh(model_name="contract_c2", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        p1 = g.model.geometry.add_point(2, 0, 0)
        p2 = g.model.geometry.add_point(2, 1, 0)
        g.model.sync()
        g.physical.add_point([p1, p2], name="two_pts")
        g.physical.add_surface([1], name="aface")
        with pytest.raises(ValueError, match="exactly one reference point"):
            g.constraints.node_to_surface("two_pts", "aface")


def test_tie_wrong_dim_master_entities_fails_loud():
    """Headline lock: a face constraint given non-surface
    master_entities must RAISE, not silently vanish (return [])."""
    from apeGmsh import Part
    a = Part("a")
    with a:
        a.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="va")
    b = Part("b")
    with b:
        b.model.geometry.add_box(1, 0, 0, 1, 1, 1, label="vb")
    with apeGmsh(model_name="contract_tie", verbose=False) as g:
        g.parts.add(a, label="A")
        g.parts.add(b, label="B")
        g.parts.fragment_all()
        # dim-1 master_entities for a face constraint is wrong-dim.
        g.constraints.tie("A", "B", master_entities=[(1, 1)])
        g.mesh.sizing.set_global_size(0.6)
        g.mesh.generation.generate(3)
        with pytest.raises(ValueError, match="non-surface|requires dim=2|surface"):
            g.mesh.queries.get_fem_data(dim=3)


# =====================================================================
# Rule 6 (contributor-facing) — an undispatched constraint type must
# fail loud, never warn-and-skip (a silently dropped constraint is the
# cardinal sin). Mirrors LoadsComposite/MassesComposite._add_def.
# =====================================================================

def test_undispatched_constraintdef_rejected_at_add_def():
    """Declaration-time gate: a ConstraintDef whose type has no
    _DISPATCH entry is rejected at the factory/_add_def call, not
    silently dropped later at resolve()."""
    from dataclasses import dataclass
    from apeGmsh.core.constraints.defs import ConstraintDef

    @dataclass
    class _UndispatchedDef(ConstraintDef):
        pass

    with apeGmsh(model_name="contract_dispatch_add", verbose=False) as g:
        with pytest.raises(TypeError,
                           match="no _DISPATCH entry|silently dropped"):
            g.constraints._add_def(
                _UndispatchedDef(kind="dummy",
                                 master_label="A", slave_label="B"))


def test_undispatched_constraintdef_resolve_defense():
    """Defense-in-depth: even if a def reaches constraint_defs
    bypassing _add_def, resolve() raises rather than warn-and-skip."""
    from dataclasses import dataclass
    import numpy as np
    from apeGmsh.core.constraints.defs import ConstraintDef

    @dataclass
    class _UndispatchedDef(ConstraintDef):
        pass

    with apeGmsh(model_name="contract_dispatch_resolve", verbose=False) as g:
        g.constraints.constraint_defs.append(
            _UndispatchedDef(kind="dummy",
                             master_label="A", slave_label="B"))
        with pytest.raises(TypeError, match="No _DISPATCH entry"):
            g.constraints.resolve(
                node_tags=np.array([1], dtype=np.int64),
                node_coords=np.zeros((1, 3), dtype=float))
