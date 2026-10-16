"""Unit tests for shell element primitives (Phase 2γ).

Covers the four shell elements, and the removal of ``ShellMITC3`` (#2092):

* :class:`ShellMITC4` — 4-node
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

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._element_capabilities import Unknown, element_capability
from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import Primitive
from apeGmsh.opensees.element.shell import (
    ASDShellQ4,
    ASDShellT3,
    ShellDKGQ,
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
# ShellMITC3: removed (#2092)
# ===========================================================================

_SHELL_MODULES = ("apeGmsh.opensees.element", "apeGmsh.opensees.element.shell")


_REPO = Path(__file__).resolve().parents[4]
_CHECK_QUIRKS = _REPO / "scripts" / "check_quirks.py"


def _names_the_replacements(message: str) -> None:
    assert "ShellMITC3 was removed" in message
    assert "#2092" in message
    assert "ASDShellT3" in message
    assert "ShellDKGT" in message


class TestShellMITC3Removed:
    """No OpenSees build has ``ShellMITC3`` (not the fork pin, not upstream),
    so the bridge dropped it. Every spelling fails at once. The bridge
    spelling ``ops.element.ShellMITC3``, the capability lookup and the
    readback token table name the 3-node shells that work. The module
    spellings (``from apeGmsh.opensees.element import ShellMITC3``) keep
    Python's plain errors, because a module ``__getattr__`` would blind the
    static resolvers to every typo there; the skill's removed-symbols
    registry and lint rule R1 name the replacement for those."""

    @pytest.mark.parametrize("module", _SHELL_MODULES)
    def test_attribute_read_fails_plainly(self, module: str) -> None:
        mod = importlib.import_module(module)
        with pytest.raises(AttributeError, match="has no attribute 'ShellMITC3'"):
            mod.ShellMITC3  # noqa: B018
        assert not hasattr(mod, "ShellMITC3")
        assert "ShellMITC3" not in mod.__all__
        # No module ``__getattr__``: the static resolvers stay exact.
        assert "__getattr__" not in vars(mod)

    def test_from_import_of_the_package_fails(self) -> None:
        with pytest.raises(ImportError, match="cannot import name 'ShellMITC3'"):
            from apeGmsh.opensees.element import ShellMITC3  # noqa: F401

    def test_from_import_of_the_module_fails(self) -> None:
        with pytest.raises(ImportError, match="cannot import name 'ShellMITC3'"):
            from apeGmsh.opensees.element.shell import ShellMITC3  # noqa: F401

    def test_bridge_namespace_names_the_replacements(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        with pytest.raises(AttributeError) as exc:
            ops.element.ShellMITC3(pg="Roof", section=_section())
        _names_the_replacements(str(exc.value))
        assert not hasattr(ops.element, "ShellMITC3")
        # Any other missing name keeps Python's own message.
        with pytest.raises(AttributeError, match="has no attribute 'ShellMITC5'"):
            ops.element.ShellMITC5  # noqa: B018
        assert ops.element.ASDShellT3 is not None

    def test_capability_lookup_names_the_replacements(self) -> None:
        with pytest.raises(ValueError) as exc:
            element_capability("ShellMITC3")
        _names_the_replacements(str(exc.value))
        # Every ``element_*`` helper routes through the same lookup.
        from apeGmsh.opensees._element_capabilities import element_class_ndf_ok

        with pytest.raises(ValueError) as exc:
            element_class_ndf_ok("ShellMITC3")
        _names_the_replacements(str(exc.value))
        # A name never in the registry is still Unknown, not refused.
        assert element_capability("NoSuchElement") is Unknown

    def test_readback_token_names_the_replacements(self) -> None:
        """A model.h5 keeps only the element token; one that names
        ShellMITC3 refuses by name rather than reading as an unknown class."""
        from apeGmsh.opensees._readback import (
            _CLASS_FOR_TOKEN,
            rotation_roles_from_tokens,
        )

        for lookup in (lambda: _CLASS_FOR_TOKEN.get("ShellMITC3"),
                       lambda: _CLASS_FOR_TOKEN["ShellMITC3"],
                       lambda: "ShellMITC3" in _CLASS_FOR_TOKEN,
                       lambda: rotation_roles_from_tokens([("ShellMITC3", (1, 2, 3))])):
            with pytest.raises(ValueError) as exc:
                lookup()
            _names_the_replacements(str(exc.value))
        # Live tokens and unknown tokens behave as before.
        assert _CLASS_FOR_TOKEN.get("ASDShellT3") is ASDShellT3
        assert _CLASS_FOR_TOKEN.get("NoSuchToken") is None
        roles = rotation_roles_from_tokens([("NoSuchToken", (1, 2, 3))])
        assert roles.model_blocker == "NoSuchToken"

    def test_stale_patch_target_still_sees_typos_in_the_shell_modules(self) -> None:
        """``check_quirks``'s ``stale-patch-target`` resolves a patch string
        statically; a module ``__getattr__`` would make every name resolve."""
        spec = importlib.util.spec_from_file_location("_check_quirks_2092", _CHECK_QUIRKS)
        assert spec is not None and spec.loader is not None
        quirks = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = quirks   # dataclasses resolve their module by name
        try:
            spec.loader.exec_module(quirks)
            for module in _SHELL_MODULES:
                assert quirks._target_resolves(f"{module}.ASDShellT3", _REPO)
                assert not quirks._target_resolves(f"{module}.NoSuchThing", _REPO)
                assert not quirks._target_resolves(f"{module}.ShellMITC3", _REPO)
            # The ``ops.element`` hook is a class ``__getattribute__``, not a
            # module or class ``__getattr__``: its module stays exact, and the
            # skill's fence resolver (which reads ``__getattr__``) too.
            ns = "apeGmsh.opensees._internal.ns.element"
            assert quirks._target_resolves(f"{ns}._ElementNS", _REPO)
            assert not quirks._target_resolves(f"{ns}.NoSuchThing", _REPO)
            from apeGmsh.opensees._internal.ns.element import _ElementNS

            assert not any("__getattr__" in vars(k) for k in _ElementNS.__mro__)
        finally:
            sys.modules.pop(spec.name, None)


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
        assert ele.no_eas is False
        assert ele.drilling_stab is None
        assert ele.drilling_nl is False
        assert ele.local_cs is None

    def test_construct_full(self) -> None:
        s = _section()
        ele = ASDShellQ4(
            pg="Plate",
            section=s,
            corotational=True,
            drilling_stab=0.05,
            local_cs=(1.0, 0.0, 0.0),
        )
        assert ele.corotational is True
        assert ele.drilling_stab == 0.05
        assert ele.local_cs == (1.0, 0.0, 0.0)


class TestASDShellQ4Validation:
    @pytest.mark.parametrize(
        "cs",
        [
            (1.0,),
            (1.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 1.0),  # 5
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # 7
        ],
    )
    def test_local_cs_must_be_three_tuple(
        self, cs: tuple[float, ...]
    ) -> None:
        s = _section()
        with pytest.raises(ValueError, match="3-tuple"):
            ASDShellQ4(pg="Plate", section=s, local_cs=cs)

    def test_legacy_six_tuple_is_refused_with_the_fix(self) -> None:
        # The 6-tuple emitted '-localCS', which the element never parsed.
        with pytest.raises(ValueError, match=r"local_cs=\(1\.0, 0\.0, 0\.0\)"):
            ASDShellQ4(pg="Plate", section=_section(),
                       local_cs=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    def test_zero_local_axis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="zero vector"):
            ASDShellQ4(pg="Plate", section=_section(), local_cs=(0, 0, 0))

    def test_drilling_nt_alpha_is_refused(self) -> None:
        # '-drillingNT' is not an ASDShellQ4 option; it was ignored.
        with pytest.raises(ValueError, match="drillingStab"):
            ASDShellQ4(pg="Plate", section=_section(), drilling_nt_alpha=0.05)

    @pytest.mark.parametrize("v", [-0.1, 1.5])
    def test_drilling_stab_range(self, v: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ASDShellQ4(pg="Plate", section=_section(), drilling_stab=v)

    def test_drilling_stab_and_nl_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            ASDShellQ4(pg="Plate", section=_section(),
                       drilling_stab=0.1, drilling_nl=True)


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

    def test_emit_with_drilling_stab(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s, drilling_stab=0.05)
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8,
                 "-drillingStab", 0.05),
                {},
            )
        ]

    def test_emit_with_no_eas(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s, no_eas=True)
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8, "-noeas"),
                {},
            )
        ]

    def test_emit_with_drilling_nl(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s, drilling_nl=True)
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8, "-drillingNL"),
                {},
            )
        ]

    def test_emit_with_local_cs(self) -> None:
        s = _section()
        ele = ASDShellQ4(pg="Plate", section=s, local_cs=(0, 1, 0))
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8,
                 "-local", 0.0, 1.0, 0.0),
                {},
            )
        ]

    def test_emit_with_all_flags(self) -> None:
        s = _section()
        ele = ASDShellQ4(
            pg="Plate",
            section=s,
            corotational=True,
            no_eas=True,
            drilling_stab=0.1,
            local_cs=(1.0, 0.0, 0.0),
        )
        e = _prepare_emitter(s, sec_tag=8, nodes=(41, 42, 43, 44))
        ele._emit(e, tag=20)
        assert e.calls == [
            (
                "element",
                ("ASDShellQ4", 20, 41, 42, 43, 44, 8,
                 "-corotational",
                 "-noeas",
                 "-drillingStab", 0.1,
                 "-local", 1.0, 0.0, 0.0),
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
        assert ele.drilling_nl is False
        assert ele.drilling_dof is None
        assert ele.local_cs is None

    def test_construct_full(self) -> None:
        s = _section()
        ele = ASDShellT3(
            pg="Tri",
            section=s,
            corotational=True,
            drilling_nl=True,
            local_cs=(1.0, 0.0, 0.0),
        )
        assert ele.corotational is True
        assert ele.drilling_nl is True
        assert ele.local_cs == (1.0, 0.0, 0.0)


class TestASDShellT3Validation:
    @pytest.mark.parametrize(
        "cs",
        [
            (1.0,),
            (1.0, 0.0, 0.0, 0.0, 1.0),  # 5
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),  # 6: the unparsed -localCS form
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),  # 7
        ],
    )
    def test_local_cs_must_be_three_tuple(
        self, cs: tuple[float, ...]
    ) -> None:
        s = _section()
        with pytest.raises(ValueError, match="local x axis"):
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

    def test_drilling_dof_is_refused(self) -> None:
        # '-drillingDOF' is not an ASDShellT3 option; it was ignored.
        with pytest.raises(ValueError, match="drilling_nl"):
            ASDShellT3(pg="Tri", section=_section(), drilling_dof=6)

    def test_emit_with_drilling_nl(self) -> None:
        s = _section()
        ele = ASDShellT3(pg="Tri", section=s, drilling_nl=True)
        e = _prepare_emitter(s, sec_tag=5, nodes=(51, 52, 53))
        ele._emit(e, tag=22)
        assert e.calls == [
            (
                "element",
                ("ASDShellT3", 22, 51, 52, 53, 5, "-drillingNL"),
                {},
            )
        ]

    def test_emit_with_local_cs(self) -> None:
        s = _section()
        ele = ASDShellT3(pg="Tri", section=s, local_cs=(1.0, 0.0, 0.0))
        e = _prepare_emitter(s, sec_tag=5, nodes=(51, 52, 53))
        ele._emit(e, tag=22)
        assert e.calls == [
            (
                "element",
                ("ASDShellT3", 22, 51, 52, 53, 5,
                 "-local", 1.0, 0.0, 0.0),
                {},
            )
        ]

    def test_emit_with_all_flags(self) -> None:
        s = _section()
        ele = ASDShellT3(
            pg="Tri",
            section=s,
            corotational=True,
            drilling_nl=True,
            local_cs=(1.0, 0.0, 0.0),
        )
        e = _prepare_emitter(s, sec_tag=5, nodes=(51, 52, 53))
        ele._emit(e, tag=22)
        assert e.calls == [
            (
                "element",
                ("ASDShellT3", 22, 51, 52, 53, 5,
                 "-corotational",
                 "-drillingNL",
                 "-local", 1.0, 0.0, 0.0),
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
