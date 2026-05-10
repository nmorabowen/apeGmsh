"""Unit tests for the typed zero-length element primitives (Phase 2β).

For each class shipped by ``apeGmsh.opensees.element.zero_length``:

  * construction with valid parameters,
  * validation rejects bad inputs,
  * ``_emit`` records the correct OpenSees command on a
    :class:`RecordingEmitter` (with element-nodes and tag-resolver
    context installed),
  * ``_emit`` without context raises ``RuntimeError``,
  * ``dependencies`` returns the right materials / sections (with
    deduplication),
  * ``__repr__`` mentions the class name.

The contract gate ``test_element_zero_length_contract.py`` parametrizes
the family-wide checks; this file exercises per-class behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import (
    Primitive,
    Section,
    UniaxialMaterial,
)
from apeGmsh.opensees.element.zero_length import (
    ZeroLength,
    ZeroLengthMatDir,
    ZeroLengthSection,
)
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# ---------------------------------------------------------------------------
# Test-local UniaxialMaterial / Section — same shape as Phase 1C tests.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeMat(UniaxialMaterial):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.uniaxialMaterial("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeSection(Section):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.section("FakeSection", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


def _resolver_from(tags: dict[int, int]) -> object:
    """Return a callable that maps Primitive -> tag via id-keyed map."""
    def _resolve(prim: Primitive) -> int:
        return tags[id(prim)]
    return _resolve


# ===========================================================================
# ZeroLengthMatDir value object
# ===========================================================================

class TestZeroLengthMatDir:
    def test_construct(self) -> None:
        m = _FakeMat(name="spring")
        md = ZeroLengthMatDir(material=m, dof=1)
        assert md.material is m
        assert md.dof == 1

    def test_zero_dof_rejected(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="dof must be >= 1"):
            ZeroLengthMatDir(material=m, dof=0)

    def test_negative_dof_rejected(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="dof must be >= 1"):
            ZeroLengthMatDir(material=m, dof=-1)


# ===========================================================================
# ZeroLength
# ===========================================================================

class TestZeroLengthConstruction:
    def test_minimal_single_pair(self) -> None:
        m = _FakeMat(name="spring")
        zl = ZeroLength(
            pg="contacts",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
        )
        assert zl.pg == "contacts"
        assert len(zl.mat_dirs) == 1
        assert zl.orient is None
        assert zl.do_rayleigh is False

    def test_multiple_pairs(self) -> None:
        m1 = _FakeMat(name="x")
        m2 = _FakeMat(name="y")
        m3 = _FakeMat(name="z")
        zl = ZeroLength(
            pg="contacts",
            mat_dirs=(
                ZeroLengthMatDir(material=m1, dof=1),
                ZeroLengthMatDir(material=m2, dof=2),
                ZeroLengthMatDir(material=m3, dof=3),
            ),
        )
        assert len(zl.mat_dirs) == 3

    def test_with_orient_and_rayleigh(self) -> None:
        m = _FakeMat(name="x")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
            orient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            do_rayleigh=True,
        )
        assert zl.orient == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert zl.do_rayleigh is True

    def test_empty_mat_dirs_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="at least one .material, dof. pair"
        ):
            ZeroLength(pg="c", mat_dirs=())


class TestZeroLengthDependencies:
    def test_single_material(self) -> None:
        m = _FakeMat(name="x")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
        )
        assert zl.dependencies() == (m,)

    def test_distinct_materials(self) -> None:
        a = _FakeMat(name="A")
        b = _FakeMat(name="B")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(
                ZeroLengthMatDir(material=a, dof=1),
                ZeroLengthMatDir(material=b, dof=2),
            ),
        )
        assert zl.dependencies() == (a, b)

    def test_dedup_when_same_material_used_twice(self) -> None:
        # The same material on two different DOFs only appears once
        # in dependencies() — keep iteration order.
        a = _FakeMat(name="A")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(
                ZeroLengthMatDir(material=a, dof=1),
                ZeroLengthMatDir(material=a, dof=2),
                ZeroLengthMatDir(material=a, dof=3),
            ),
        )
        assert zl.dependencies() == (a,)

    def test_dedup_preserves_order_of_first_occurrence(self) -> None:
        a = _FakeMat(name="A")
        b = _FakeMat(name="B")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(
                ZeroLengthMatDir(material=a, dof=1),
                ZeroLengthMatDir(material=b, dof=2),
                ZeroLengthMatDir(material=a, dof=3),
            ),
        )
        assert zl.dependencies() == (a, b)


class TestZeroLengthEmit:
    def test_minimal_single_pair(self) -> None:
        m = _FakeMat(name="x")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 5}))
        set_element_nodes(e, (10, 20))
        zl._emit(e, tag=42)
        assert e.calls == [
            (
                "element",
                ("zeroLength", 42, 10, 20, "-mat", 5, "-dir", 1),
                {},
            ),
        ]

    def test_multiple_pairs(self) -> None:
        a = _FakeMat(name="A")
        b = _FakeMat(name="B")
        c = _FakeMat(name="C")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(
                ZeroLengthMatDir(material=a, dof=1),
                ZeroLengthMatDir(material=b, dof=2),
                ZeroLengthMatDir(material=c, dof=3),
            ),
        )
        e = RecordingEmitter()
        set_tag_resolver(
            e, _resolver_from({id(a): 11, id(b): 22, id(c): 33})
        )
        set_element_nodes(e, (1, 2))
        zl._emit(e, tag=7)
        assert e.calls == [
            (
                "element",
                (
                    "zeroLength", 7, 1, 2,
                    "-mat", 11, 22, 33,
                    "-dir", 1, 2, 3,
                ),
                {},
            ),
        ]

    def test_with_orient(self) -> None:
        m = _FakeMat(name="x")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
            orient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 5}))
        set_element_nodes(e, (1, 2))
        zl._emit(e, tag=3)
        assert e.calls[0] == (
            "element",
            (
                "zeroLength", 3, 1, 2,
                "-mat", 5,
                "-dir", 1,
                "-orient", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ),
            {},
        )

    def test_with_do_rayleigh(self) -> None:
        m = _FakeMat(name="x")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
            do_rayleigh=True,
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 5}))
        set_element_nodes(e, (1, 2))
        zl._emit(e, tag=3)
        assert e.calls[0] == (
            "element",
            (
                "zeroLength", 3, 1, 2,
                "-mat", 5, "-dir", 1,
                "-doRayleigh", 1,
            ),
            {},
        )

    def test_emit_without_nodes_raises(self) -> None:
        m = _FakeMat(name="x")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        with pytest.raises(RuntimeError, match="element-nodes"):
            zl._emit(e, tag=1)

    def test_emit_without_resolver_raises(self) -> None:
        m = _FakeMat(name="x")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
        )
        e = RecordingEmitter()
        set_element_nodes(e, (1, 2))
        with pytest.raises(RuntimeError, match="tag resolver"):
            zl._emit(e, tag=1)

    def test_emit_with_wrong_node_count_raises(self) -> None:
        m = _FakeMat(name="x")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        set_element_nodes(e, (1,))
        with pytest.raises(ValueError, match="expected 2 node tags"):
            zl._emit(e, tag=1)


class TestZeroLengthMisc:
    def test_repr_includes_class_name(self) -> None:
        m = _FakeMat(name="x")
        zl = ZeroLength(
            pg="c",
            mat_dirs=(ZeroLengthMatDir(material=m, dof=1),),
        )
        assert "ZeroLength" in repr(zl)


# ===========================================================================
# ZeroLengthSection
# ===========================================================================

class TestZeroLengthSectionConstruction:
    def test_minimal(self) -> None:
        s = _FakeSection(name="sec")
        zls = ZeroLengthSection(pg="c", section=s)
        assert zls.pg == "c"
        assert zls.section is s
        assert zls.orient is None
        assert zls.do_rayleigh is False

    def test_with_orient_and_rayleigh(self) -> None:
        s = _FakeSection(name="sec")
        zls = ZeroLengthSection(
            pg="c", section=s,
            orient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            do_rayleigh=True,
        )
        assert zls.orient == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert zls.do_rayleigh is True


class TestZeroLengthSectionEmit:
    def test_minimal(self) -> None:
        s = _FakeSection(name="sec")
        zls = ZeroLengthSection(pg="c", section=s)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(s): 5}))
        set_element_nodes(e, (10, 20))
        zls._emit(e, tag=42)
        assert e.calls == [
            ("element", ("zeroLengthSection", 42, 10, 20, 5), {}),
        ]

    def test_with_orient(self) -> None:
        s = _FakeSection(name="sec")
        zls = ZeroLengthSection(
            pg="c", section=s,
            orient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(s): 5}))
        set_element_nodes(e, (1, 2))
        zls._emit(e, tag=3)
        assert e.calls[0] == (
            "element",
            (
                "zeroLengthSection", 3, 1, 2, 5,
                "-orient", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ),
            {},
        )

    def test_with_do_rayleigh(self) -> None:
        s = _FakeSection(name="sec")
        zls = ZeroLengthSection(pg="c", section=s, do_rayleigh=True)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(s): 5}))
        set_element_nodes(e, (1, 2))
        zls._emit(e, tag=3)
        assert e.calls[0] == (
            "element",
            ("zeroLengthSection", 3, 1, 2, 5, "-doRayleigh", 1),
            {},
        )

    def test_emit_without_nodes_raises(self) -> None:
        s = _FakeSection(name="sec")
        zls = ZeroLengthSection(pg="c", section=s)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(s): 1}))
        with pytest.raises(RuntimeError, match="element-nodes"):
            zls._emit(e, tag=1)

    def test_emit_without_resolver_raises(self) -> None:
        s = _FakeSection(name="sec")
        zls = ZeroLengthSection(pg="c", section=s)
        e = RecordingEmitter()
        set_element_nodes(e, (1, 2))
        with pytest.raises(RuntimeError, match="tag resolver"):
            zls._emit(e, tag=1)

    def test_emit_with_wrong_node_count_raises(self) -> None:
        s = _FakeSection(name="sec")
        zls = ZeroLengthSection(pg="c", section=s)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(s): 1}))
        set_element_nodes(e, (1, 2, 3))
        with pytest.raises(ValueError, match="expected 2 node tags"):
            zls._emit(e, tag=1)


class TestZeroLengthSectionMisc:
    def test_dependencies_returns_section(self) -> None:
        s = _FakeSection(name="sec")
        assert ZeroLengthSection(pg="c", section=s).dependencies() == (s,)

    def test_repr_includes_class_name(self) -> None:
        s = _FakeSection(name="sec")
        assert "ZeroLengthSection" in repr(
            ZeroLengthSection(pg="c", section=s)
        )
