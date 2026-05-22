"""Phase 9 commit 5 — DomainCaptureSpec declarative surface tests.

Covers the new results-side sibling of the bridge's recorder
declaration:

- Standalone declarative methods (nodes / elements / gauss /
  line_stations / fibers / layers / modal)
- Introspection (categories / components_for / shorthands_for)
- ``resolve(fem)`` — requires bridge attachment per Phase 9 D8
- ``_resolve_with_explicit_ndm_ndf`` — internal entry used by
  ``ops.domain_capture(...)`` and ``DomainCapture.from_h5(...)``
- ``ops.domain_capture(...)`` bridge entry
- ``DomainCapture.from_h5(...)`` file entry
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy import ndarray

from apeGmsh.opensees import apeSees
from apeGmsh.results.capture import (
    DomainCapture,
    DomainCaptureSpec,
    ResolvedDomainCaptureSpec,
)

from tests.fixtures.schema import OPENSEES_CURRENT


# ---------------------------------------------------------------------------
# Minimal FEMData stub
# ---------------------------------------------------------------------------


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
# Standalone declarative surface
# ---------------------------------------------------------------------------


class TestStandaloneDeclaration:
    def test_construction(self) -> None:
        spec = DomainCaptureSpec()
        assert len(spec) == 0

    def test_all_categories_declarable(self) -> None:
        spec = DomainCaptureSpec()
        spec.nodes(components=["displacement"], ids=[1, 2])
        spec.elements(components=["nodal_resisting_force_x"], ids=[1])
        spec.line_stations(components=["axial_force"], ids=[1])
        spec.gauss(components=["stress"], ids=[1])
        spec.fibers(components=["fiber_stress"], ids=[1])
        spec.layers(components=["fiber_stress"], ids=[1])
        spec.modal(n_modes=5)
        assert len(spec) == 7

    def test_chaining(self) -> None:
        spec = DomainCaptureSpec()
        out = (spec
               .nodes(components=["displacement"], ids=[1])
               .gauss(components=["stress"], ids=[1])
               .modal(n_modes=2))
        assert out is spec
        assert len(spec) == 3

    def test_clear(self) -> None:
        spec = DomainCaptureSpec()
        spec.nodes(components=["displacement"], ids=[1])
        spec.gauss(components=["stress"], ids=[1])
        assert len(spec) == 2
        spec.clear()
        assert len(spec) == 0


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_categories(self) -> None:
        assert set(DomainCaptureSpec.categories()) == {
            "nodes", "elements", "line_stations", "gauss",
            "fibers", "layers", "modal",
        }

    def test_components_for_nodes(self) -> None:
        comps = DomainCaptureSpec.components_for("nodes")
        assert "displacement_x" in comps
        assert "reaction_force_y" in comps

    def test_components_for_modal_empty(self) -> None:
        assert DomainCaptureSpec.components_for("modal") == ()

    def test_components_for_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            DomainCaptureSpec.components_for("bogus")

    def test_shorthands_for_nodes(self) -> None:
        sh = DomainCaptureSpec.shorthands_for("nodes")
        assert "displacement" in sh
        assert sh["displacement"] == (
            "displacement_x", "displacement_y", "displacement_z",
        )
        assert "reaction" in sh

    def test_shorthands_for_modal_empty(self) -> None:
        assert DomainCaptureSpec.shorthands_for("modal") == {}

    def test_where_does_routes_line_diagram(self) -> None:
        assert DomainCaptureSpec.where_does("axial_force") == (
            "line_stations",
        )
        assert DomainCaptureSpec.where_does("bending_moment_y") == (
            "line_stations",
        )

    def test_where_does_routes_nodal(self) -> None:
        assert DomainCaptureSpec.where_does("displacement_x") == ("nodes",)
        assert DomainCaptureSpec.where_does("reaction_force_z") == ("nodes",)

    def test_where_does_shared_categories(self) -> None:
        # FIBER components valid in both fibers and layers
        assert DomainCaptureSpec.where_does("fiber_stress") == (
            "fibers", "layers",
        )

    def test_where_does_state_variable_wildcard(self) -> None:
        # ``state_variable_*`` routes to the three element-level
        # material categories per _validate_components_for_category.
        assert DomainCaptureSpec.where_does("state_variable_42") == (
            "fibers", "gauss", "layers",
        )

    def test_where_does_unknown_returns_empty(self) -> None:
        assert DomainCaptureSpec.where_does("not_a_component") == ()


# ---------------------------------------------------------------------------
# resolve() — D8 enforcement
# ---------------------------------------------------------------------------


class TestResolveD8:
    def test_standalone_resolve_raises_with_migration_hint(self) -> None:
        spec = DomainCaptureSpec()
        spec.nodes(components=["displacement_x"], ids=[1])
        with pytest.raises(RuntimeError) as exc_info:
            spec.resolve(cast("object", _make_fem()))
        msg = str(exc_info.value)
        assert "DomainCaptureSpec(opensees=ops)" in msg
        assert "DomainCapture.from_h5" in msg

    def test_bridge_without_model_raises(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        spec = DomainCaptureSpec(opensees=ops)
        spec.nodes(components=["displacement_x"], ids=[1])
        with pytest.raises(RuntimeError) as exc_info:
            spec.resolve(cast("object", fem))
        assert "ops.model" in str(exc_info.value)

    def test_bridge_attached_resolve_succeeds(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        spec = DomainCaptureSpec(opensees=ops)
        spec.nodes(components=["displacement"], ids=[1, 2])
        resolved = spec.resolve(cast("object", fem))
        assert isinstance(resolved, ResolvedDomainCaptureSpec)
        assert resolved.ndm == 3
        assert resolved.ndf == 6
        rec = resolved.records[0]
        # Shorthand expanded against bridge ndm/ndf.
        assert rec.components == (
            "displacement_x", "displacement_y", "displacement_z",
        )

    def test_resolve_with_ndm_kwarg_raises_TypeError(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        spec = DomainCaptureSpec(opensees=ops)
        spec.nodes(components=["displacement_x"], ids=[1])
        with pytest.raises(TypeError):
            spec.resolve(cast("object", fem), ndm=3)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Vocabulary validation
# ---------------------------------------------------------------------------


class TestVocabularyValidation:
    def test_bogus_component_rejected_at_resolve(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        spec = DomainCaptureSpec(opensees=ops)
        spec.nodes(components=["totally_made_up"], ids=[1])
        with pytest.raises(ValueError):
            spec.resolve(cast("object", fem))

    def test_stress_on_nodes_rejected(self) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        spec = DomainCaptureSpec(opensees=ops)
        spec.nodes(components=["stress_xx"], ids=[1])
        with pytest.raises(ValueError):
            spec.resolve(cast("object", fem))


# ---------------------------------------------------------------------------
# ops.domain_capture(...) bridge entry
# ---------------------------------------------------------------------------


class TestBridgeEntry:
    def test_domain_capture_without_model_raises(
        self, tmp_path: Path,
    ) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))  # no ops.model() call
        spec = DomainCaptureSpec(opensees=ops)
        spec.nodes(components=["displacement_x"], ids=[1])
        with pytest.raises(RuntimeError) as exc_info:
            ops.domain_capture(spec, path=str(tmp_path / "run.h5"))
        assert "ops.model" in str(exc_info.value)

    def test_domain_capture_returns_DomainCapture(
        self, tmp_path: Path,
    ) -> None:
        fem = _make_fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        spec = DomainCaptureSpec(opensees=ops)
        spec.nodes(components=["displacement_x"], ids=[1])

        cap = ops.domain_capture(spec, path=str(tmp_path / "run.h5"))
        assert isinstance(cap, DomainCapture)


# ---------------------------------------------------------------------------
# DomainCapture.from_h5(...) file entry
# ---------------------------------------------------------------------------


class TestFromH5:
    def _write_minimal_model_h5(
        self, path: Path, *, ndm: int = 3, ndf: int = 6,
    ) -> None:
        import h5py
        # Per ADR 0023 fixture must be inside the two-version reader
        # window (2.7.x / 2.8.x).
        with h5py.File(path, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["schema_version"] = OPENSEES_CURRENT
            meta.attrs["ndm"] = ndm
            meta.attrs["ndf"] = ndf
            meta.attrs["snapshot_id"] = "stub-snapshot"
            meta.attrs["model_name"] = "stub"

    def test_from_h5_returns_DomainCapture(
        self, tmp_path: Path,
    ) -> None:
        model_path = tmp_path / "model.h5"
        self._write_minimal_model_h5(model_path)
        fem = _make_fem()
        spec = DomainCaptureSpec()
        spec.nodes(components=["displacement"], ids=[1])
        cap = DomainCapture.from_h5(
            model_path,
            spec=spec,
            fem=cast("object", fem),
            output=tmp_path / "run.h5",
        )
        assert isinstance(cap, DomainCapture)

    def test_from_h5_threads_ndm_ndf_through(
        self, tmp_path: Path,
    ) -> None:
        model_path = tmp_path / "model.h5"
        # Forge a 2D model header so from_h5 reads ndm=2 / ndf=2.
        self._write_minimal_model_h5(model_path, ndm=2, ndf=2)
        fem = _make_fem()
        spec = DomainCaptureSpec()
        spec.nodes(components=["displacement"], ids=[1])

        cap = DomainCapture.from_h5(
            model_path,
            spec=spec,
            fem=cast("object", fem),
            output=tmp_path / "run.h5",
        )
        # The resolved spec carried inside the cap captures ndm/ndf.
        assert cap._spec.ndm == 2
        assert cap._spec.ndf == 2
        # Shorthand "displacement" expanded to 2D x/y only.
        rec = cap._spec.records[0]
        assert rec.components == ("displacement_x", "displacement_y")
