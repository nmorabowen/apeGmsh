"""Unit tests for ``apeGmsh.opensees.pattern.pattern``.

Phase 3A ships :class:`Plain` (the workhorse) and
:class:`UniformExcitation` (ground motion). Each class gets:

  * construction (defaults, explicit values).
  * validation (per-class invariants).
  * context-manager usage (``with p as scope:``).
  * recording API (``p.load(node=, forces=)`` / ``p.sp(node=, dof=,
    value=)``).
  * ``_emit`` records the right call sequence into a
    ``RecordingEmitter`` with a tag resolver installed.
  * ``_emit`` raises :class:`NotImplementedError` on ``pg=`` records
    (those defer to the Phase 4 build pipeline).
  * ``dependencies()`` returns ``(series,)``.
  * ``__repr__`` includes the class name.

Tests use ``RecordingEmitter`` only — no openseespy, no gmsh, no
subprocess.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees._internal.types import Primitive
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.pattern.pattern import (
    H5DRM,
    Plain,
    UniformExcitation,
    _LoadRecord,
    _SPRecord,
)
from apeGmsh.opensees.time_series.time_series import Linear


def _resolver_from(tags: dict[int, int]) -> object:
    """Return a callable that maps Primitive -> tag via id-keyed map."""
    def _resolve(prim: Primitive) -> int:
        return tags[id(prim)]
    return _resolve


def _make_ops() -> "apeSees":
    """Construct an apeSees with a stub FEMData (namespaces ignore it)."""
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


# ===========================================================================
# Plain
# ===========================================================================

class TestPlainConstruction:
    def test_construct_with_series(self) -> None:
        ts = Linear()
        p = Plain(series=ts)
        assert p.series is ts
        assert p.loads == ()
        assert p.sps == ()

    def test_repr_includes_class_name(self) -> None:
        p = Plain(series=Linear())
        assert "Plain" in repr(p)

    def test_dependencies_returns_series(self) -> None:
        ts = Linear()
        p = Plain(series=ts)
        assert p.dependencies() == (ts,)

    def test_series_field_is_frozen(self) -> None:
        # Cannot reassign ``series`` on a frozen-dataclass instance.
        p = Plain(series=Linear())
        with pytest.raises(Exception):
            p.series = Linear()  # type: ignore[misc]


class TestPlainContextManager:
    def test_with_block_returns_self(self) -> None:
        p = Plain(series=Linear())
        with p as scope:
            assert scope is p

    def test_can_record_inside_with_block(self) -> None:
        p = Plain(series=Linear())
        with p as scope:
            scope.load(node=10, forces=(100.0, 0.0, 0.0))
        assert len(p.loads) == 1


class TestPlainLoadRecording:
    def test_load_with_node_records(self) -> None:
        p = Plain(series=Linear())
        p.load(node=42, forces=(100.0, 0.0, -50.0))

        assert p.loads == (
            _LoadRecord(
                target_kind="node",
                target="42",
                forces=(100.0, 0.0, -50.0),
            ),
        )

    def test_load_with_pg_records(self) -> None:
        p = Plain(series=Linear())
        p.load(pg="RoofFloor", forces=(100e3, 0.0, 0.0))

        assert p.loads == (
            _LoadRecord(
                target_kind="pg",
                target="RoofFloor",
                forces=(100e3, 0.0, 0.0),
            ),
        )

    def test_load_neither_pg_nor_node_raises(self) -> None:
        p = Plain(series=Linear())
        with pytest.raises(
            ValueError, match="exactly one of pg= or node="
        ):
            p.load(forces=(1.0, 0.0))

    def test_load_both_pg_and_node_raises(self) -> None:
        p = Plain(series=Linear())
        with pytest.raises(
            ValueError, match="exactly one of pg= or node="
        ):
            p.load(pg="X", node=5, forces=(1.0, 0.0))

    def test_multiple_loads_accumulate(self) -> None:
        p = Plain(series=Linear())
        p.load(node=1, forces=(100.0, 0.0))
        p.load(node=2, forces=(200.0, 0.0))
        p.load(node=3, forces=(300.0, 0.0))
        assert len(p.loads) == 3
        assert p.loads[0].target == "1"
        assert p.loads[1].target == "2"
        assert p.loads[2].target == "3"


class TestPlainSPRecording:
    def test_sp_with_node_records(self) -> None:
        p = Plain(series=Linear())
        p.sp(node=7, dof=1, value=0.005)

        assert p.sps == (
            _SPRecord(
                target_kind="node", target="7", dof=1, value=0.005,
            ),
        )

    def test_sp_with_pg_records(self) -> None:
        p = Plain(series=Linear())
        p.sp(pg="Bearing", dof=2, value=0.01)

        assert p.sps == (
            _SPRecord(
                target_kind="pg", target="Bearing", dof=2, value=0.01,
            ),
        )

    def test_sp_neither_pg_nor_node_raises(self) -> None:
        p = Plain(series=Linear())
        with pytest.raises(
            ValueError, match="exactly one of pg= or node="
        ):
            p.sp(dof=1, value=0.0)

    def test_sp_both_pg_and_node_raises(self) -> None:
        p = Plain(series=Linear())
        with pytest.raises(
            ValueError, match="exactly one of pg= or node="
        ):
            p.sp(pg="X", node=5, dof=1, value=0.0)

    def test_loads_and_sps_independent(self) -> None:
        p = Plain(series=Linear())
        p.load(node=1, forces=(100.0,))
        p.sp(node=2, dof=1, value=0.005)
        assert len(p.loads) == 1
        assert len(p.sps) == 1


class TestPlainEmit:
    def test_emit_pattern_open_close_for_empty(self) -> None:
        ts = Linear()
        p = Plain(series=ts)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(ts): 17}))
        p._emit(e, tag=3)
        assert e.calls == [
            ("pattern_open", ("Plain", 3, 17), {}),
            ("pattern_close", (), {}),
        ]

    def test_emit_with_node_loads(self) -> None:
        ts = Linear()
        p = Plain(series=ts)
        p.load(node=1, forces=(100.0, 0.0, 0.0))
        p.load(node=2, forces=(50.0, 25.0, 0.0))
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(ts): 4}))
        p._emit(e, tag=2)
        assert e.calls == [
            ("pattern_open", ("Plain", 2, 4), {}),
            ("load", (1, 100.0, 0.0, 0.0), {}),
            ("load", (2, 50.0, 25.0, 0.0), {}),
            ("pattern_close", (), {}),
        ]

    def test_emit_with_node_sps(self) -> None:
        ts = Linear()
        p = Plain(series=ts)
        p.sp(node=5, dof=1, value=0.01)
        p.sp(node=6, dof=2, value=-0.005)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(ts): 1}))
        p._emit(e, tag=1)
        assert e.calls == [
            ("pattern_open", ("Plain", 1, 1), {}),
            ("sp", (5, 1, 0.01), {}),
            ("sp", (6, 2, -0.005), {}),
            ("pattern_close", (), {}),
        ]

    def test_emit_loads_then_sps(self) -> None:
        ts = Linear()
        p = Plain(series=ts)
        p.load(node=1, forces=(100.0, 0.0))
        p.sp(node=2, dof=1, value=0.001)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(ts): 1}))
        p._emit(e, tag=1)
        names = [c[0] for c in e.calls]
        assert names == ["pattern_open", "load", "sp", "pattern_close"]

    def test_emit_pg_load_raises_not_implemented(self) -> None:
        ts = Linear()
        p = Plain(series=ts)
        p.load(pg="RoofFloor", forces=(100e3, 0.0, 0.0))
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(ts): 1}))
        with pytest.raises(
            NotImplementedError, match="pg= load fan-out"
        ):
            p._emit(e, tag=1)

    def test_emit_pg_sp_raises_not_implemented(self) -> None:
        ts = Linear()
        p = Plain(series=ts)
        p.sp(pg="Bearing", dof=1, value=0.005)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(ts): 1}))
        with pytest.raises(
            NotImplementedError, match="pg= sp fan-out"
        ):
            p._emit(e, tag=1)

    def test_emit_without_resolver_raises(self) -> None:
        # ``_emit`` resolves the series's tag at emit time and raises
        # if the resolver has not been installed.
        p = Plain(series=Linear())
        e = RecordingEmitter()
        with pytest.raises(RuntimeError, match="tag resolver"):
            p._emit(e, tag=1)

    def test_emit_outside_with_block_works(self) -> None:
        # The ``with`` block is purely for textual visibility (ADR
        # 0005); it does not change any internal state. Recording API
        # calls + ``_emit`` work identically whether the user wraps
        # them in a ``with`` block or not.
        ts = Linear()
        p = Plain(series=ts)
        p.load(node=1, forces=(1.0, 0.0))
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(ts): 1}))
        p._emit(e, tag=1)
        names = [c[0] for c in e.calls]
        assert names == ["pattern_open", "load", "pattern_close"]


# ===========================================================================
# UniformExcitation
# ===========================================================================

class TestUniformExcitationConstruction:
    def test_construct_with_direction_and_series(self) -> None:
        ts = Linear()
        p = UniformExcitation(direction=1, series=ts)
        assert p.direction == 1
        assert p.series is ts

    @pytest.mark.parametrize("direction", [1, 2, 3, 4, 5, 6])
    def test_direction_in_range(self, direction: int) -> None:
        UniformExcitation(direction=direction, series=Linear())

    @pytest.mark.parametrize("direction", [0, 7, -1, 100])
    def test_direction_out_of_range_raises(
        self, direction: int
    ) -> None:
        with pytest.raises(
            ValueError, match="direction must be 1-6"
        ):
            UniformExcitation(direction=direction, series=Linear())

    def test_repr_includes_class_name(self) -> None:
        p = UniformExcitation(direction=1, series=Linear())
        assert "UniformExcitation" in repr(p)

    def test_dependencies_returns_series(self) -> None:
        ts = Linear()
        p = UniformExcitation(direction=2, series=ts)
        assert p.dependencies() == (ts,)


class TestUniformExcitationContextManager:
    def test_with_block_returns_self(self) -> None:
        p = UniformExcitation(direction=1, series=Linear())
        with p as scope:
            assert scope is p


class TestUniformExcitationEmit:
    def test_emit_records_pattern_open_with_dir_and_accel_flag(
        self,
    ) -> None:
        ts = Linear()
        p = UniformExcitation(direction=2, series=ts)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(ts): 9}))
        p._emit(e, tag=4)
        assert e.calls == [
            (
                "pattern_open",
                ("UniformExcitation", 4, 2, "-accel", 9),
                {},
            ),
            ("pattern_close", (), {}),
        ]

    def test_emit_without_resolver_raises(self) -> None:
        p = UniformExcitation(direction=1, series=Linear())
        e = RecordingEmitter()
        with pytest.raises(RuntimeError, match="tag resolver"):
            p._emit(e, tag=1)


# ===========================================================================
# Namespace integration — methods register with the bridge
# ===========================================================================

class TestPatternNamespace:
    def test_plain_namespace_constructs_and_registers(self) -> None:
        ops = _make_ops()
        ts = ops.timeSeries.Linear()
        p = ops.pattern.Plain(series=ts)
        assert isinstance(p, Plain)
        assert p.series is ts
        # The pattern tag-allocator counter is independent of the
        # timeSeries counter, so Plain is tag 1 in its own kind.
        assert ops.tag_for(p) == 1
        assert ops.tag_for(ts) == 1

    def test_uniform_excitation_namespace(self) -> None:
        ops = _make_ops()
        ts = ops.timeSeries.Path(file="elcentro.txt", dt=0.01, factor=9.81)
        p = ops.pattern.UniformExcitation(direction=1, series=ts)
        assert isinstance(p, UniformExcitation)
        assert p.direction == 1
        assert p.series is ts

    def test_distinct_patterns_get_distinct_tags(self) -> None:
        ops = _make_ops()
        ts1 = ops.timeSeries.Linear()
        ts2 = ops.timeSeries.Linear(factor=2.0)
        a = ops.pattern.Plain(series=ts1)
        b = ops.pattern.Plain(series=ts2)
        assert ops.tag_for(a) == 1
        assert ops.tag_for(b) == 2

    def test_namespace_with_block_records_loads(self) -> None:
        ops = _make_ops()
        ts = ops.timeSeries.Linear()
        with ops.pattern.Plain(series=ts) as p:
            p.load(node=1, forces=(100.0, 0.0, 0.0))
            p.load(node=2, forces=(50.0, 0.0, 0.0))
        assert len(p.loads) == 2


# ===========================================================================
# H5DRM (ADR 0066)
# ===========================================================================

# The canonical 18-arg emit tail for an identity frame: do_transform=1,
# T = row-major identity, x0 = 0.
_IDENTITY_TAIL = (
    1,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 0.0,
)


class TestH5DRMConstruction:
    def test_defaults(self) -> None:
        p = H5DRM(h5drm="motions.h5drm")
        assert p.h5drm == "motions.h5drm"
        assert p.factor == 1.0
        assert p.crd_scale == 1000.0          # km -> m
        assert p.distance_tolerance == 1.0
        assert p.transform is None            # identity
        assert p.x0 == (0.0, 0.0, 0.0)

    def test_explicit_values(self) -> None:
        T = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        p = H5DRM(
            h5drm="m.h5drm", factor=2.0, crd_scale=1.0,
            distance_tolerance=0.5, transform=T, x0=(10.0, 0.0, -5.0),
        )
        assert p.factor == 2.0
        assert p.crd_scale == 1.0
        assert p.distance_tolerance == 0.5
        assert p.transform == T
        assert p.x0 == (10.0, 0.0, -5.0)

    def test_transform_list_normalized_to_tuples(self) -> None:
        p = H5DRM(h5drm="m.h5drm",
                  transform=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert p.transform == (
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        )

    def test_repr_includes_class_name(self) -> None:
        assert "H5DRM" in repr(H5DRM(h5drm="m.h5drm"))

    def test_dependencies_empty(self) -> None:
        # Field-carrying — no TimeSeries dependency.
        assert H5DRM(h5drm="m.h5drm").dependencies() == ()

    def test_field_is_frozen(self) -> None:
        p = H5DRM(h5drm="m.h5drm")
        with pytest.raises(Exception):
            p.factor = 2.0  # type: ignore[misc]


class TestH5DRMValidation:
    def test_empty_path_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty path"):
            H5DRM(h5drm="")

    @pytest.mark.parametrize(
        "bad",
        [
            ((1, 0, 0), (0, 1, 0)),                       # 2 rows
            ((1, 0), (0, 1), (0, 0)),                     # 2 cols
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),  # 4 cols
        ],
    )
    def test_bad_transform_shape_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match="3x3"):
            H5DRM(h5drm="m.h5drm", transform=bad)  # type: ignore[arg-type]


class TestH5DRMContextManager:
    def test_with_block_returns_self(self) -> None:
        p = H5DRM(h5drm="m.h5drm")
        with p as scope:
            assert scope is p


class TestH5DRMEmit:
    def test_emit_identity_default(self) -> None:
        p = H5DRM(h5drm="motions.h5drm")
        e = RecordingEmitter()
        p._emit(e, tag=1)
        assert e.calls == [
            (
                "pattern_open",
                ("H5DRM", 1, "motions.h5drm", 1.0, 1000.0, 1.0)
                + _IDENTITY_TAIL,
                {},
            ),
            ("pattern_close", (), {}),
        ]

    def test_emit_custom_transform_and_x0(self) -> None:
        T = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        p = H5DRM(h5drm="m.h5drm", factor=2.0, crd_scale=1.0,
                  distance_tolerance=0.25, transform=T, x0=(5.0, 0.0, -1.0))
        e = RecordingEmitter()
        p._emit(e, tag=7)
        assert e.calls == [
            (
                "pattern_open",
                ("H5DRM", 7, "m.h5drm", 2.0, 1.0, 0.25,
                 1,
                 0.0, -1.0, 0.0,
                 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0,
                 5.0, 0.0, -1.0),
                {},
            ),
            ("pattern_close", (), {}),
        ]

    def test_emit_needs_no_resolver(self) -> None:
        # No TimeSeries dependency -> _emit must not require a tag resolver.
        H5DRM(h5drm="m.h5drm")._emit(RecordingEmitter(), tag=1)


class TestH5DRMNamespace:
    def test_namespace_constructs_and_registers(self) -> None:
        ops = _make_ops()
        p = ops.pattern.H5DRM(h5drm="motions.h5drm")
        assert isinstance(p, H5DRM)
        assert ops.tag_for(p) == 1

    def test_namespace_passes_through_kwargs(self) -> None:
        ops = _make_ops()
        p = ops.pattern.H5DRM(
            h5drm="m.h5drm", crd_scale=1.0, distance_tolerance=0.1,
            x0=(1.0, 2.0, 3.0),
        )
        assert p.crd_scale == 1.0
        assert p.distance_tolerance == 0.1
        assert p.x0 == (1.0, 2.0, 3.0)
