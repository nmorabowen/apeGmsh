"""Unit tests for shell element primitives (Phase 2γ).

Covers the five shell elements:

* :class:`ShellMITC4` — 4-node
* :class:`ShellMITC3` — 3-node
* :class:`ShellDKGQ`  — 4-node
* :class:`ASDShellQ4` — 4-node, with optional flags
* :class:`ASDShellT3` — 3-node, with optional flags

Per the element fan-out contract in
:mod:`apeGmsh.opensees._internal.tag_resolution`, ``_emit`` reads the
section tag via :func:`resolve_tag` and the per-element node tags via
:func:`current_element_nodes`. Tests install both contexts on the
:class:`RecordingEmitter` manually with :func:`set_tag_resolver` and
:func:`set_element_nodes`.
"""
from __future__ import annotations

from typing import Callable

import pytest

from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import Primitive
from apeGmsh.opensees.element.shell import (
    ASDShellQ4,
    ASDShellT3,
    ShellDKGQ,
    ShellMITC3,
    ShellMITC4,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.section.plate import ElasticMembranePlateSection


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _resolver_from(tags: dict[int, int]) -> Callable[[Primitive], int]:
    """Return a callable that maps Primitive -> tag via id-keyed map."""
    def _resolve(prim: Primitive) -> int:
        return tags[id(prim)]
    return _resolve


def _section() -> ElasticMembranePlateSection:
    """A minimal plate section for shell tests."""
    return ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2)


def _prepare_emitter(
    section: ElasticMembranePlateSection,
    *,
    sec_tag: int,
    nodes: tuple[int, ...],
) -> RecordingEmitter:
    """Build a RecordingEmitter with the resolver + element-nodes context
    installed for ``section``."""
    e = RecordingEmitter()
    set_tag_resolver(e, _resolver_from({id(section): sec_tag}))
    set_element_nodes(e, nodes)
    return e


# ===========================================================================
# ShellMITC4
# ===========================================================================

class TestShellMITC4Construction:
    def test_construct(self) -> None:
        s = _section()
        ele = ShellMITC4(pg="Slab", section=s)
        assert ele.pg == "Slab"
        assert ele.section is s


class TestShellMITC4Emit:
    def test_emit_records_correct_call(self) -> None:
        s = _section()
        ele = ShellMITC4(pg="Slab", section=s)
        e = _prepare_emitter(s, sec_tag=7, nodes=(11, 12, 13, 14))
        ele._emit(e, tag=3)
        assert e.calls == [
            (
                "element",
                ("ShellMITC4", 3, 11, 12, 13, 14, 7),
                {},
            )
        ]

    def test_emit_without_element_nodes_raises(self) -> None:
        s = _section()
        ele = ShellMITC4(pg="Slab", section=s)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(s): 7}))
        with pytest.raises(RuntimeError, match="element-nodes context"):
            ele._emit(e, tag=3)

    def test_emit_without_resolver_raises(self) -> None:
        s = _section()
        ele = ShellMITC4(pg="Slab", section=s)
        e = RecordingEmitter()
        set_element_nodes(e, (11, 12, 13, 14))
        with pytest.raises(RuntimeError, match="tag resolver"):
            ele._emit(e, tag=3)

    @pytest.mark.parametrize("nodes", [(1, 2, 3), (1, 2, 3, 4, 5), (1,)])
    def test_emit_wrong_node_count_raises(
        self, nodes: tuple[int, ...]
    ) -> None:
        s = _section()
        ele = ShellMITC4(pg="Slab", section=s)
        e = _prepare_emitter(s, sec_tag=7, nodes=nodes)
        with pytest.raises(ValueError, match="expected 4 node tags"):
            ele._emit(e, tag=3)


class TestShellMITC4Misc:
    def test_dependencies_returns_section(self) -> None:
        s = _section()
        ele = ShellMITC4(pg="Slab", section=s)
        assert ele.dependencies() == (s,)

    def test_repr_includes_class_name(self) -> None:
        ele = ShellMITC4(pg="Slab", section=_section())
        assert "ShellMITC4" in repr(ele)


# ===========================================================================
# ShellMITC3
# ===========================================================================

class TestShellMITC3Construction:
    def test_construct(self) -> None:
        s = _section()
        ele = ShellMITC3(pg="Roof", section=s)
        assert ele.pg == "Roof"
        assert ele.section is s


class TestShellMITC3Emit:
    def test_emit_records_correct_call(self) -> None:
        s = _section()
        ele = ShellMITC3(pg="Roof", section=s)
        e = _prepare_emitter(s, sec_tag=4, nodes=(21, 22, 23))
        ele._emit(e, tag=9)
        assert e.calls == [
            (
                "element",
                ("ShellMITC3", 9, 21, 22, 23, 4),
                {},
            )
        ]

    def test_emit_without_element_nodes_raises(self) -> None:
        s = _section()
        ele = ShellMITC3(pg="Roof", section=s)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(s): 4}))
        with pytest.raises(RuntimeError, match="element-nodes context"):
            ele._emit(e, tag=9)

    @pytest.mark.parametrize("nodes", [(1, 2), (1, 2, 3, 4)])
    def test_emit_wrong_node_count_raises(
        self, nodes: tuple[int, ...]
    ) -> None:
        s = _section()
        ele = ShellMITC3(pg="Roof", section=s)
        e = _prepare_emitter(s, sec_tag=4, nodes=nodes)
        with pytest.raises(ValueError, match="expected 3 node tags"):
            ele._emit(e, tag=9)


class TestShellMITC3Misc:
    def test_dependencies_returns_section(self) -> None:
        s = _section()
        ele = ShellMITC3(pg="Roof", section=s)
        assert ele.dependencies() == (s,)

    def test_repr_includes_class_name(self) -> None:
        ele = ShellMITC3(pg="Roof", section=_section())
        assert "ShellMITC3" in repr(ele)


# ===========================================================================
# ShellDKGQ
# ===========================================================================

class TestShellDKGQConstruction:
    def test_construct(self) -> None:
        s = _section()
        ele = ShellDKGQ(pg="Wall", section=s)
        assert ele.pg == "Wall"
        assert ele.section is s


class TestShellDKGQEmit:
    def test_emit_records_correct_call(self) -> None:
        s = _section()
        ele = ShellDKGQ(pg="Wall", section=s)
        e = _prepare_emitter(s, sec_tag=2, nodes=(31, 32, 33, 34))
        ele._emit(e, tag=15)
        assert e.calls == [
            (
                "element",
                ("ShellDKGQ", 15, 31, 32, 33, 34, 2),
                {},
            )
        ]

    @pytest.mark.parametrize("nodes", [(1, 2, 3), (1, 2, 3, 4, 5)])
    def test_emit_wrong_node_count_raises(
        self, nodes: tuple[int, ...]
    ) -> None:
        s = _section()
        ele = ShellDKGQ(pg="Wall", section=s)
        e = _prepare_emitter(s, sec_tag=2, nodes=nodes)
        with pytest.raises(ValueError, match="expected 4 node tags"):
            ele._emit(e, tag=15)


class TestShellDKGQMisc:
    def test_dependencies_returns_section(self) -> None:
        s = _section()
        ele = ShellDKGQ(pg="Wall", section=s)
        assert ele.dependencies() == (s,)

    def test_repr_includes_class_name(self) -> None:
        ele = ShellDKGQ(pg="Wall", section=_section())
        assert "ShellDKGQ" in repr(ele)


# ===========================================================================
# ASDShellQ4
# ===========================================================================

class TestASDShellQ4Construction:
    def test_construct_minimum(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s)
        assert ele.pg == "Plate"
        assert ele.section is s
        assert ele.corotational is False
        assert ele.drilling_nt_alpha is None
        assert ele.local_cs is None

    def test_construct_full(self) -> None:
        s = _section()
        ele = ASDShellQ4(
            pg="Plate",
            section=s,
            corotational=True,
            drilling_nt_alpha=0.05,
            local_cs=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        assert ele.corotational is True
        assert ele.drilling_nt_alpha == 0.05
        assert ele.local_cs == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class TestASDShellQ4Validation:
    @pytest.mark.parametrize(
        "cs",
        [
            (1.0,),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 1.0),  # 5
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # 7
        ],
    )
    def test_local_cs_must_be_six_tuple(
        self, cs: tuple[float, ...]
    ) -> None:
        s = _section()
        with pytest.raises(ValueError, match="6-tuple"):
            ASDShellQ4(pg="Plate", section=s, local_cs=cs)


class TestASDShellQ4Emit:
    def test_emit_minimum(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s)
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8),
                {},
            )
        ]

    def test_emit_with_corotational(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s, corotational=True)
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8, "-corotational"),
                {},
            )
        ]

    def test_emit_with_drilling_nt(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s, drilling_nt_alpha=0.05)
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8,
                 "-drillingNT", 0.05),
                {},
            )
        ]

    def test_emit_with_local_cs(self) -> None:
        s = _section()
        cs = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        ele = ASDShellQ4(pg="Plate", section=s, local_cs=cs)
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8,
                 "-localCS", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                {},
            )
        ]

    def test_emit_with_all_flags(self) -> None:
        s = _section()
        cs = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        ele = ASDShellQ4(
            pg="Plate",
            section=s,
            corotational=True,
            drilling_nt_alpha=0.1,
            local_cs=cs,
        )
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8,
                 "-corotational",
                 "-drillingNT", 0.1,
                 "-localCS", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                {},
            )
        ]

    def test_emit_without_element_nodes_raises(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(s): 8}))
        with pytest.raises(RuntimeError, match="element-nodes context"):
            ele._emit(e, tag=20)

    @pytest.mark.parametrize("nodes", [(1, 2, 3), (1, 2, 3, 4, 5)])
    def test_emit_wrong_node_count_raises(
        self, nodes: tuple[int, ...]
    ) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s)
        e = _prepare_emitter(s, sec_tag=8, nodes=nodes)
        with pytest.raises(ValueError, match="expected 4 node tags"):
            ele._emit(e, tag=20)


class TestASDShellQ4Misc:
    def test_dependencies_returns_section(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s)
        assert ele.dependencies() == (s,)

    def test_repr_includes_class_name(self) -> None:
        ele = ASDShellQ4(pg="Plate", section=_section())
        assert "ASDShellQ4" in repr(ele)


# ===========================================================================
# ASDShellT3
# ===========================================================================

class TestASDShellT3Construction:
    def test_construct_minimum(self) -> None:
        s = _section()
        ele = ASDShellT3(pg="Tri", section=s)
        assert ele.pg == "Tri"
        assert ele.section is s
        assert ele.corotational is False
        assert ele.drilling_dof is None
        assert ele.local_cs is None

    def test_construct_full(self) -> None:
        s = _section()
        ele = ASDShellT3(
            pg="Tri",
            section=s,
            corotational=True,
            drilling_dof=6,
            local_cs=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        assert ele.corotational is True
        assert ele.drilling_dof == 6
        assert ele.local_cs == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class TestASDShellT3Validation:
    @pytest.mark.parametrize(
        "cs",
        [
            (1.0,),
            (1.0, 0.0, 0.0, 0.0, 1.0),  # 5
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # 7
        ],
    )
    def test_local_cs_must_be_six_tuple(
        self, cs: tuple[float, ...]
    ) -> None:
        s = _section()
        with pytest.raises(ValueError, match="6-tuple"):
            ASDShellT3(pg="Tri", section=s, local_cs=cs)


class TestASDShellT3Emit:
    def test_emit_minimum(self) -> None:
        s = _section()
        ele = ASDShellT3(pg="Tri", section=s)
        e = _prepare_emitter(s, sec_tag=5, nodes=(51, 52, 53))
        ele._emit(e, tag=22)
        assert e.calls == [
            (
                "element",
                ("ASDShellT3", 22, 51, 52, 53, 5),
                {},
            )
        ]

    def test_emit_with_corotational(self) -> None:
        s = _section()
        ele = ASDShellT3(pg="Tri", section=s, corotational=True)
        e = _prepare_emitter(s, sec_tag=5, nodes=(51, 52, 53))
        ele._emit(e, tag=22)
        assert e.calls == [
            (
                "element",
                ("ASDShellT3", 22, 51, 52, 53, 5, "-corotational"),
                {},
            )
        ]

    def test_emit_with_drilling_dof(self) -> None:
        s = _section()
        ele = ASDShellT3(pg="Tri", section=s, drilling_dof=6)
        e = _prepare_emitter(s, sec_tag=5, nodes=(51, 52, 53))
        ele._emit(e, tag=22)
        assert e.calls == [
            (
                "element",
                ("ASDShellT3", 22, 51, 52, 53, 5,
                 "-drillingDOF", 6),
                {},
            )
        ]

    def test_emit_with_local_cs(self) -> None:
        s = _section()
        cs = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        ele = ASDShellT3(pg="Tri", section=s, local_cs=cs)
        e = _prepare_emitter(s, sec_tag=5, nodes=(51, 52, 53))
        ele._emit(e, tag=22)
        assert e.calls == [
            (
                "element",
                ("ASDShellT3", 22, 51, 52, 53, 5,
                 "-localCS", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                {},
            )
        ]

    def test_emit_with_all_flags(self) -> None:
        s = _section()
        cs = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        ele = ASDShellT3(
            pg="Tri",
            section=s,
            corotational=True,
            drilling_dof=6,
            local_cs=cs,
        )
        e = _prepare_emitter(s, sec_tag=5, nodes=(51, 52, 53))
        ele._emit(e, tag=22)
        assert e.calls == [
            (
                "element",
                ("ASDShellT3", 22, 51, 52, 53, 5,
                 "-corotational",
                 "-drillingDOF", 6,
                 "-localCS", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                {},
            )
        ]

    def test_emit_without_element_nodes_raises(self) -> None:
        s = _section()
        ele = ASDShellT3(pg="Tri", section=s)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(s): 5}))
        with pytest.raises(RuntimeError, match="element-nodes context"):
            ele._emit(e, tag=22)

    @pytest.mark.parametrize("nodes", [(1, 2), (1, 2, 3, 4)])
    def test_emit_wrong_node_count_raises(
        self, nodes: tuple[int, ...]
    ) -> None:
        s = _section()
        ele = ASDShellT3(pg="Tri", section=s)
        e = _prepare_emitter(s, sec_tag=5, nodes=nodes)
        with pytest.raises(ValueError, match="expected 3 node tags"):
            ele._emit(e, tag=22)


class TestASDShellT3Misc:
    def test_dependencies_returns_section(self) -> None:
        s = _section()
        ele = ASDShellT3(pg="Tri", section=s)
        assert ele.dependencies() == (s,)

    def test_repr_includes_class_name(self) -> None:
        ele = ASDShellT3(pg="Tri", section=_section())
        assert "ASDShellT3" in repr(ele)
