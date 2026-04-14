"""Tests for _element_types: ElementTypeInfo, ElementGroup, GroupResult."""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.mesh._element_types import (
    ElementTypeInfo,
    ElementGroup,
    GroupResult,
    make_type_info,
    resolve_type_filter,
    _auto_alias,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _tet4_info(count: int = 10) -> ElementTypeInfo:
    return make_type_info(code=4, gmsh_name='Tetrahedron 4',
                          dim=3, order=1, npe=4, count=count)

def _hex8_info(count: int = 5) -> ElementTypeInfo:
    return make_type_info(code=5, gmsh_name='Hexahedron 8',
                          dim=3, order=1, npe=8, count=count)

def _tri3_info(count: int = 8) -> ElementTypeInfo:
    return make_type_info(code=2, gmsh_name='Triangle 3',
                          dim=2, order=1, npe=3, count=count)

def _make_group(info: ElementTypeInfo, n: int | None = None) -> ElementGroup:
    n = n or info.count
    ids = np.arange(1, n + 1, dtype=np.int64)
    conn = np.arange(n * info.npe, dtype=np.int64).reshape(n, info.npe) + 100
    return ElementGroup(element_type=info, ids=ids, connectivity=conn)


# =====================================================================
# ElementTypeInfo
# =====================================================================

class TestElementTypeInfo:

    def test_curated_alias(self):
        info = make_type_info(4, 'Tetrahedron 4', 3, 1, 4)
        assert info.name == 'tet4'

    def test_auto_alias_hex64(self):
        info = make_type_info(92, 'Hexahedron 64', 3, 3, 64)
        assert info.name == 'hex64'

    def test_auto_alias_unknown_shape(self):
        # A shape Gmsh knows but we don't have in _SHAPE_PREFIXES
        alias = _auto_alias('Serendipity 20', 20)
        assert alias == 'serendipity20'

    def test_auto_alias_from_prefix(self):
        alias = _auto_alias('Pyramid 14', 14)
        assert alias == 'pyramid14'

    def test_repr(self):
        info = _tet4_info()
        r = repr(info)
        assert 'tet4' in r
        assert 'code=4' in r

    def test_equality_by_code(self):
        a = make_type_info(4, 'Tetrahedron 4', 3, 1, 4, count=10)
        b = make_type_info(4, 'Tetrahedron 4', 3, 1, 4, count=99)
        assert a == b

    def test_hash(self):
        a = _tet4_info(10)
        b = _tet4_info(99)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


# =====================================================================
# ElementGroup
# =====================================================================

class TestElementGroup:

    def test_len(self):
        g = _make_group(_tet4_info(10))
        assert len(g) == 10

    def test_properties(self):
        g = _make_group(_tet4_info())
        assert g.type_name == 'tet4'
        assert g.type_code == 4
        assert g.dim == 3
        assert g.npe == 4

    def test_iteration(self):
        g = _make_group(_tet4_info(3))
        pairs = list(g)
        assert len(pairs) == 3
        eid, conn_row = pairs[0]
        assert isinstance(eid, int)
        assert len(conn_row) == 4

    def test_repr(self):
        g = _make_group(_tet4_info(5))
        r = repr(g)
        assert 'tet4' in r
        assert 'n=5' in r


# =====================================================================
# GroupResult
# =====================================================================

class TestGroupResult:

    def test_empty(self):
        r = GroupResult([])
        assert len(r) == 0
        assert r.n_elements == 0
        assert r.is_homogeneous
        assert list(r) == []
        assert len(r.ids) == 0
        assert not r  # falsy

    def test_single_group(self):
        g = _make_group(_tet4_info(5))
        r = GroupResult([g])
        assert len(r) == 1
        assert r.n_elements == 5
        assert r.is_homogeneous
        assert r.types[0].name == 'tet4'
        assert r.connectivity.shape == (5, 4)

    def test_mixed_groups(self):
        g1 = _make_group(_tet4_info(5))
        g2 = _make_group(_hex8_info(3))
        r = GroupResult([g1, g2])
        assert len(r) == 2
        assert r.n_elements == 8
        assert not r.is_homogeneous
        assert len(r.ids) == 8
        assert len(r.types) == 2

    def test_mixed_connectivity_raises(self):
        g1 = _make_group(_tet4_info(5))
        g2 = _make_group(_hex8_info(3))
        r = GroupResult([g1, g2])
        with pytest.raises(TypeError, match="element types present"):
            _ = r.connectivity

    def test_iteration(self):
        g1 = _make_group(_tet4_info(2))
        g2 = _make_group(_hex8_info(1))
        r = GroupResult([g1, g2])
        groups = list(r)
        assert len(groups) == 2
        assert groups[0].type_name == 'tet4'
        assert groups[1].type_name == 'hex8'

    # ── .get() chainable filter ─────────────────────────────

    def test_get_by_dim(self):
        g1 = _make_group(_tet4_info(5))    # dim=3
        g2 = _make_group(_tri3_info(3))    # dim=2
        r = GroupResult([g1, g2])
        r3 = r.get(dim=3)
        assert len(r3) == 1
        assert r3.types[0].name == 'tet4'

    def test_get_by_element_type_str(self):
        g1 = _make_group(_tet4_info(5))
        g2 = _make_group(_hex8_info(3))
        r = GroupResult([g1, g2])
        r_tet = r.get(element_type='tet4')
        assert len(r_tet) == 1
        assert r_tet.n_elements == 5

    def test_get_by_element_type_int(self):
        g1 = _make_group(_tet4_info(5))
        g2 = _make_group(_hex8_info(3))
        r = GroupResult([g1, g2])
        r_hex = r.get(element_type=5)
        assert len(r_hex) == 1
        assert r_hex.types[0].name == 'hex8'

    def test_get_contradictory_empty(self):
        g1 = _make_group(_tet4_info(5))
        r = GroupResult([g1])
        empty = r.get(dim=2)  # tet4 is dim=3
        assert len(empty) == 0
        assert empty.n_elements == 0

    def test_get_chained(self):
        g1 = _make_group(_tet4_info(5))
        g2 = _make_group(_hex8_info(3))
        g3 = _make_group(_tri3_info(4))
        r = GroupResult([g1, g2, g3])
        r2 = r.get(dim=3).get(element_type='tet4')
        assert r2.n_elements == 5

    # ── .resolve() ──────────────────────────────────────────

    def test_resolve_homogeneous(self):
        g = _make_group(_tet4_info(5))
        r = GroupResult([g])
        ids, conn = r.resolve()
        assert len(ids) == 5
        assert conn.shape == (5, 4)

    def test_resolve_with_type_filter(self):
        g1 = _make_group(_tet4_info(5))
        g2 = _make_group(_hex8_info(3))
        r = GroupResult([g1, g2])
        ids, conn = r.resolve(element_type='hex8')
        assert len(ids) == 3
        assert conn.shape == (3, 8)

    def test_resolve_mixed_raises(self):
        g1 = _make_group(_tet4_info(5))
        g2 = _make_group(_hex8_info(3))
        r = GroupResult([g1, g2])
        with pytest.raises(TypeError, match="resolve"):
            r.resolve()

    def test_resolve_empty(self):
        r = GroupResult([])
        ids, conn = r.resolve()
        assert len(ids) == 0

    def test_resolve_by_int_code(self):
        g1 = _make_group(_tet4_info(5))
        g2 = _make_group(_hex8_info(3))
        r = GroupResult([g1, g2])
        ids, conn = r.resolve(element_type=4)
        assert len(ids) == 5
        assert conn.shape == (5, 4)


# =====================================================================
# resolve_type_filter
# =====================================================================

class TestResolveTypeFilter:

    def test_int_code(self):
        groups = [_make_group(_tet4_info())]
        assert resolve_type_filter(4, groups) == {4}

    def test_alias_str(self):
        groups = [_make_group(_tet4_info()), _make_group(_hex8_info())]
        assert resolve_type_filter('hex8', groups) == {5}

    def test_gmsh_name(self):
        groups = [_make_group(_tet4_info())]
        assert resolve_type_filter('Tetrahedron 4', groups) == {4}

    def test_unknown_raises(self):
        groups = [_make_group(_tet4_info())]
        with pytest.raises(KeyError, match="Unknown"):
            resolve_type_filter('brick99', groups)
