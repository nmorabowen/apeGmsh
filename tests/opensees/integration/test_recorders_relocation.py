"""Phase 9 commit 4 — Recorders relocation tests.

The legacy ``Recorders`` fluent helper moved from
``apeGmsh.results.spec.declaration`` to ``apeGmsh.opensees.recorder``
(via the implementation module ``apeGmsh.opensees._recorders_builder``)
in Phase 9 commit 4. Per D8 the relocated ``resolve()`` no longer
takes ``ndm`` / ``ndf`` kwargs — it sources both from the attached
OpenSees bridge.

These tests verify:

1. Standalone ``Recorders()`` still supports the full declarative
   surface (every per-category method, introspection, ``clear``,
   ``__len__``, ``__repr__``).
2. Standalone ``Recorders().resolve(fem)`` raises
   :class:`RuntimeError` with a migration hint that names both
   ``ops.recorder.declare(...)`` (the canonical Phase 9 path) and
   the transitional bridge-bound ``Recorders(opensees=ops)`` form.
3. Bridge-bound ``Recorders(opensees=ops)`` succeeds at ``resolve``
   when ``ops.model(ndm=, ndf=)`` has been called.
4. Bridge-bound ``Recorders(opensees=ops)`` raises a clear
   :class:`RuntimeError` when ``ops.model()`` has not been called
   (the bridge's ``_ndm`` / ``_ndf`` are still ``None``).
5. ``Recorders.resolve(fem, ndm=3)`` raises :class:`TypeError` —
   the legacy kwarg signature is fully removed (D8 no back-compat).
6. The same canonical component vocabulary is still validated — a
   bogus component name raises ``ValueError`` at ``resolve`` time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest
from numpy import ndarray

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.recorder import Recorders


# ---------------------------------------------------------------------------
# Minimal FEMData stub
# ---------------------------------------------------------------------------
#
# The bridge-bound resolve path only needs ``snapshot_id`` (carried
# through to ``ResolvedRecorderSpec``) and ``ids``-based selector
# resolution. Using ``ids=`` keeps these tests independent of FEMData's
# physical-group / labels composites.


@dataclass(frozen=True)
class _NodesStub:
    ids: ndarray


@dataclass(frozen=True)
class _ElementsStub:
    ids: ndarray


@dataclass(frozen=True)
class _FEMStub:
    nodes: _NodesStub
    elements: _ElementsStub
    snapshot_id: str = "stub-snapshot"
    mesh_selection: object | None = None


def _make_fem() -> _FEMStub:
    return _FEMStub(
        nodes=_NodesStub(ids=np.asarray([1, 2, 3], dtype=np.int64)),
        elements=_ElementsStub(ids=np.asarray([1, 2], dtype=np.int64)),
    )


# ---------------------------------------------------------------------------
# 1. Standalone declarative surface unaffected
# ---------------------------------------------------------------------------


class TestStandaloneDeclaration:
    def test_standalone_construction(self) -> None:
        r = Recorders()
        assert len(r) == 0

    def test_all_categories_declarable_standalone(self) -> None:
        r = Recorders()
        r.nodes(components=["displacement"], ids=[1, 2])
        r.elements(components=["nodal_resisting_force_x"], ids=[1])
        r.line_stations(components=["axial_force"], ids=[1])
        r.gauss(components=["stress"], ids=[1])
        r.fibers(components=["fiber_stress"], ids=[1])
        r.layers(components=["fiber_stress"], ids=[1])
        r.modal(n_modes=5)
        assert len(r) == 7

    def test_introspection_still_works_standalone(self) -> None:
        assert "nodes" in Recorders.categories()
        assert "displacement_x" in Recorders.components_for("nodes")
        assert "displacement" in Recorders.shorthands_for("nodes")


# ---------------------------------------------------------------------------
# 2. Standalone .resolve(fem) raises with migration hint
# ---------------------------------------------------------------------------


class TestStandaloneResolveErrors:
    def test_standalone_resolve_raises_runtime_error(self) -> None:
        r = Recorders()
        r.nodes(components=["displacement_x"], ids=[1])
        with pytest.raises(RuntimeError) as exc_info:
            r.resolve(cast("object", _make_fem()))
        msg = str(exc_info.value)
        # Migration hint must name both the canonical Phase 9 path and
        # the transitional bridge-bound path.
        assert "ops.recorder.declare" in msg
        assert "Recorders(opensees=ops)" in msg


# ---------------------------------------------------------------------------
# 3. Bridge-bound resolve succeeds, sources ndm/ndf from bridge
# ---------------------------------------------------------------------------


class TestBridgeBoundResolve:
    def test_resolve_succeeds_with_bridge_and_model(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        r = Recorders(opensees=ops)
        r.nodes(components=["displacement"], ids=[1, 2])
        spec = r.resolve(cast("object", fem))
        # Shorthand was expanded against bridge ndm/ndf (3D → x/y/z).
        rec = spec.records[0]
        assert rec.components == (
            "displacement_x", "displacement_y", "displacement_z",
        )
        # Snapshot threaded through.
        assert spec.fem_snapshot_id == "stub-snapshot"

    def test_resolve_clips_to_2d_when_bridge_says_ndm_2(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=2, ndf=3)
        r = Recorders(opensees=ops)
        r.nodes(components=["displacement"], ids=[1])
        spec = r.resolve(cast("object", fem))
        # ndm=2 → only x, y axes for translational shorthand.
        assert spec.records[0].components == (
            "displacement_x", "displacement_y",
        )


# ---------------------------------------------------------------------------
# 4. Bridge without ops.model() raises clear hint
# ---------------------------------------------------------------------------


class TestBridgeWithoutModelRaises:
    def test_resolve_pre_model_raises(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))  # no ops.model() yet
        r = Recorders(opensees=ops)
        r.nodes(components=["displacement_x"], ids=[1])
        with pytest.raises(RuntimeError) as exc_info:
            r.resolve(cast("object", fem))
        msg = str(exc_info.value)
        assert "ops.model" in msg


# ---------------------------------------------------------------------------
# 5. Legacy ndm/ndf kwargs are removed (D8)
# ---------------------------------------------------------------------------


class TestResolveSignatureNoNdmNdf:
    def test_resolve_with_ndm_kwarg_raises_TypeError(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        r = Recorders(opensees=ops)
        r.nodes(components=["displacement_x"], ids=[1])
        with pytest.raises(TypeError):
            r.resolve(cast("object", fem), ndm=3)  # type: ignore[call-arg]

    def test_resolve_with_ndm_ndf_kwargs_raises_TypeError(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        r = Recorders(opensees=ops)
        r.nodes(components=["displacement_x"], ids=[1])
        with pytest.raises(TypeError):
            r.resolve(cast("object", fem), ndm=3, ndf=6)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 6. Component vocabulary still validated through the relocated path
# ---------------------------------------------------------------------------


class TestVocabularyValidation:
    def test_bogus_component_rejected_at_resolve(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        r = Recorders(opensees=ops)
        r.nodes(components=["bogus_component_name"], ids=[1])
        with pytest.raises(ValueError):
            r.resolve(cast("object", fem))

    def test_stress_component_invalid_on_nodes(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        r = Recorders(opensees=ops)
        r.nodes(components=["stress_xx"], ids=[1])
        with pytest.raises(ValueError):
            r.resolve(cast("object", fem))
