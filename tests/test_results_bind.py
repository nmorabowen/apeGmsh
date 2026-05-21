"""Phase 8 (ADR 0020 INV-1) — bind / model= validation.

The bind contract: candidate FEMData and embedded snapshot must
share ``snapshot_id``.

Phase 8 prune (ADR 0021) — :class:`BindError` deleted; the
``model=`` kwarg on every :class:`Results` constructor is required
(missing supply raises :class:`TypeError`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5, _stub_model_h5_path


def _build_composed_results(g, tmp_path: Path) -> "tuple[Path, object, Path]":
    """Build a tiny mesh + Composed-file results.h5 (carries /opensees/).

    Returns ``(results_path, fem, model_path)`` where ``model_path``
    is the standalone bridge ``model.h5`` the composed file was built
    from (for tests that want to compare).
    """
    from apeGmsh.opensees import apeSees as _apeSees
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # Emit a standalone model.h5 so we have a sibling to compose from.
    # Need at least one material so the bridge emits a /opensees/ zone.
    model_path = tmp_path / "model.h5"
    ops = _apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    ops.h5(str(model_path))

    results_path = tmp_path / "grav.h5"
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(name="grav", kind="static",
                             time=np.array([0.0]))
        w.write_nodes(sid, "partition_0",
                      node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
                      components={"displacement_x":
                                   np.zeros((1, len(fem.nodes.ids)))})
        w.end_stage()
    return results_path, fem, model_path


@pytest.fixture
def grav_results_with_fem(g, tmp_path: Path):
    """Phase 8 fixture — composed-file results + the source fem."""
    return _build_composed_results(g, tmp_path)


def test_auto_bind_from_embedded(grav_results_with_fem) -> None:
    path, fem, _ = grav_results_with_fem
    model = OpenSeesModel.from_h5(path, fem_root="/model")
    with Results.from_native(path, model=model) as r:
        assert r.fem is not None
        assert r.fem.snapshot_id == fem.snapshot_id


def test_explicit_bind_matching_hash(grav_results_with_fem) -> None:
    path, fem, _ = grav_results_with_fem
    model = OpenSeesModel.from_h5(path, fem_root="/model")
    # Pass the same fem explicitly — should succeed and use it.
    with Results.from_native(path, fem=fem, model=model) as r:
        # The candidate fem (with full label/Part info) is preferred.
        assert r.fem is fem


def test_bind_after_construction(grav_results_with_fem) -> None:
    path, fem, _ = grav_results_with_fem
    model = OpenSeesModel.from_h5(path, fem_root="/model")
    with Results.from_native(path, model=model) as r:
        rebound = r.bind(fem)
        assert rebound.fem is fem
        # The original is unchanged (we returned a new instance).
        assert r.fem is not None
        assert r.fem is not fem


def test_bind_accepts_mismatched_fem(
    grav_results_with_fem, g, tmp_path: Path,
) -> None:
    """bind() no longer validates snapshot_id — it's on the user.

    Previously a different-mesh FEMData raised BindError; the check
    was removed because legitimate workflows (re-meshing, importing
    an mpco against a fresh fem) tripped it. The hash is still
    computed and stored, just not enforced.
    """
    path, _orig_fem, _ = grav_results_with_fem
    model = OpenSeesModel.from_h5(path, fem_root="/model")

    g.model.geometry.add_box(2, 0, 0, 1, 1, 1, label="box2")
    g.physical.add_volume("box2", name="Body2")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    other_fem = g.mesh.queries.get_fem_data(dim=3)

    with Results.from_native(path, model=model) as r:
        rebound = r.bind(other_fem)
        # Bind returned a Results bound to the candidate fem.
        assert rebound.fem is other_fem


def test_pg_query_works_after_bind(grav_results_with_fem, g) -> None:
    """PG selection resolves through the bound FEMData."""
    path, fem, _ = grav_results_with_fem
    model = OpenSeesModel.from_h5(path, fem_root="/model")
    with Results.from_native(path, fem=fem, model=model) as r:
        slab = r.nodes.get(pg="Body", component="displacement_x")
        # All nodes are in 'Body' — should match the full set.
        assert slab.node_ids.size == len(fem.nodes.ids)


# ---------------------------------------------------------------------------
# Phase 8 (ADR 0020 INV-1) — model= is required; missing raises TypeError
# ---------------------------------------------------------------------------


def test_from_native_without_model_raises_typeerror(
    grav_results_with_fem,
) -> None:
    """``Results.from_native(path)`` without ``model=`` raises TypeError."""
    path, _, _ = grav_results_with_fem
    with pytest.raises(TypeError, match="model= is required"):
        Results.from_native(path)


def test_from_recorders_without_model_raises_typeerror(tmp_path: Path) -> None:
    """``Results.from_recorders(...)`` without ``model=`` raises TypeError.

    The TypeError fires before any spec/file validation, so a
    minimal call signature suffices.
    """
    with pytest.raises(TypeError, match="model= is required"):
        Results.from_recorders(
            spec=None, output_dir=tmp_path, fem=None,
        )


def test_from_mpco_without_model_h5_raises_typeerror(tmp_path: Path) -> None:
    """``Results.from_mpco(path)`` without ``model_h5=`` raises TypeError."""
    with pytest.raises(TypeError, match="model_h5= is required"):
        Results.from_mpco(tmp_path / "fake.mpco")


# ---------------------------------------------------------------------------
# Phase 4 (ADR 0020) — lineage propagates from model.lineage onto Results
# ---------------------------------------------------------------------------

def test_lineage_propagates_from_model(tmp_path: Path) -> None:
    """``results.lineage`` carries the model's chain plus a results_hash.

    Phase 4 originally asserted identity-equality (``r.lineage is
    r.model.lineage``).  Phase 6 (ADR 0021) layers a
    :attr:`Lineage.results_hash` on top of the model layer, so the
    surface contract shifts from identity-equality to chain-forward:
    the model's ``fem_hash`` and ``model_hash`` propagate verbatim,
    and ``results_hash`` is the canonical-bytes derivation over
    ``/stages/...``.
    """
    from apeGmsh.results.writers import NativeWriter
    from tests.opensees.h5._opensees_model_fixtures import (
        build_simple_frame_h5,
    )

    model_path, fem = build_simple_frame_h5(tmp_path)
    results_path = tmp_path / "lineage_run.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(
            name="g", kind="static", time=np.array([0.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((1, node_ids.size))},
        )
        w.end_stage()

    model = OpenSeesModel.from_h5(results_path, fem_root="/model")
    with Results.from_native(results_path, model=model) as r:
        assert r.model is not None
        # Phase-6 contract: model's chain propagates verbatim.
        assert r.lineage.fem_hash == r.model.lineage.fem_hash
        assert r.lineage.model_hash == r.model.lineage.model_hash
        # Phase-6 adds the results layer.
        assert r.lineage.fem_hash != ""
        assert r.lineage.model_hash is not None
        assert r.lineage.results_hash is not None
        # No drift warnings on a freshly-written file.
        assert r.lineage.warnings == ()


def test_lineage_propagates_through_bind(grav_results_with_fem) -> None:
    """``r.bind(fem)`` preserves the chain (same reader, same lineage).

    Phase 6 (ADR 0021) — ``bind()`` swaps the FEMData in but doesn't
    perturb the underlying reader.  The lineage triple still reads
    off the same file; only the candidate FEMData identity changes.
    """
    path, fem, _ = grav_results_with_fem
    model = OpenSeesModel.from_h5(path, fem_root="/model")
    with Results.from_native(path, model=model) as r:
        lineage_before = r.lineage
        rebound = r.bind(fem)
        lineage_after = rebound.lineage
        assert lineage_before.fem_hash == lineage_after.fem_hash
        # No drift warnings on a freshly-written file.
        assert lineage_after.warnings == ()
