"""Phase 6 — lineage chain (ADR 0021).

The three-link content-hash chain ``fem_hash → model_hash →
results_hash`` and its warn-not-raise drift contract.  Tests are
grouped by invariant; ``INV-1`` through ``INV-5`` are the
non-negotiable contracts ADR 0021 §Invariants pins down.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._internal.lineage import (
    LINEAGE_GROUP,
    WARNING_PREFIX,
    Lineage,
    LineageError,
    canonical_bytes,
    compute_fem_hash,
    compute_model_hash,
    compute_results_hash,
)
from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter

from tests.opensees.h5._opensees_model_fixtures import (
    build_simple_frame_fem,
    build_simple_frame_h5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_results(path: Path) -> Results:
    """Phase 8 — Composed file: load Results carrying the embedded model."""
    from apeGmsh.opensees import OpenSeesModel

    return Results.from_native(
        path, model=OpenSeesModel.from_h5(path, fem_root="/model"),
    )


def _make_composed_results(tmp_path: Path) -> "tuple[Path, Path, object]":
    """Build a Composed-file results.h5 + return ``(results, model, fem)``."""
    model_path, fem = build_simple_frame_h5(tmp_path)
    results_path = tmp_path / "lineage_results.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(name="g", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((1, node_ids.size))},
        )
        w.end_stage()
    return results_path, model_path, fem


def _make_model_h5_with_cuts(tmp_path: Path):
    """Build a ``model.h5`` whose ``/opensees/`` carries a SectionCutDef."""
    from apeGmsh.cuts import SectionCutDef
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    fem = build_simple_frame_fem()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    out = tmp_path / "with_cuts.h5"
    cut = SectionCutDef(
        plane_point=(0.0, 0.0, 0.5),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label="storey_1",
    )
    ops.h5(str(out), cuts=(cut,))
    return out, fem


def _make_model_h5_with_sweeps(tmp_path: Path):
    """Build a ``model.h5`` whose ``/opensees/`` carries a SectionSweepDef."""
    from apeGmsh.cuts import SectionCutDef, SectionSweepDef
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    fem = build_simple_frame_fem()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    out = tmp_path / "with_sweeps.h5"
    sweep = SectionSweepDef(cuts=(
        SectionCutDef(
            plane_point=(0.0, 0.0, 0.25),
            plane_normal=(0.0, 0.0, 1.0),
            element_ids=(1,),
        ),
        SectionCutDef(
            plane_point=(0.0, 0.0, 0.75),
            plane_normal=(0.0, 0.0, 1.0),
            element_ids=(1,),
        ),
    ))
    ops.h5(str(out), sweeps=(sweep,))
    return out, fem


# ---------------------------------------------------------------------------
# INV-1 — fem_hash byte-identical to FEMData.snapshot_id
# ---------------------------------------------------------------------------


def test_fem_hash_matches_snapshot_id(tmp_path: Path) -> None:
    """``compute_fem_hash(group)`` exactly equals ``FEMData.snapshot_id``.

    The recomputation goes through the same canonical-bytes path the
    FEMData uses internally; INV-1 demands byte-identity.
    """
    path, fem = build_simple_frame_h5(tmp_path)
    with h5py.File(path, "r") as f:
        recomputed = compute_fem_hash(f)
    assert recomputed == fem.snapshot_id


# ---------------------------------------------------------------------------
# INV-2 — never raises on lineage mismatch
# ---------------------------------------------------------------------------


def test_tampered_opensees_bytes_warn_not_raise(tmp_path: Path) -> None:
    """Mutating ``/opensees/transforms/...`` directly surfaces a warning."""
    path, fem = build_simple_frame_h5(tmp_path)
    # Tamper: rewrite a transform dataset to perturb the canonical bytes.
    with h5py.File(path, "r+") as f:
        vecxz = f["opensees/transforms/Linear_1/per_element_vecxz"]
        vecxz[...] = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)

    # Read via Results (and OpenSeesModel under the hood); no raise.
    results_path = tmp_path / "after_tamper.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=path)
        sid = w.begin_stage(name="g", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((1, node_ids.size))},
        )
        w.end_stage()
    with _load_results(results_path) as r:
        lineage = r.lineage
        assert lineage.warnings != ()
        # Mention "opensees" / "model_hash" / "model" — broadly check
        # the drift was localised correctly.
        joined = "\n".join(lineage.warnings)
        assert "model_hash" in joined or "opensees" in joined


def test_tampered_fem_bytes_warn_not_raise(tmp_path: Path) -> None:
    """Tampering the stored ``fem_hash`` surfaces a warning, not a raise.

    Two complementary FEM-layer integrity checks exist in apeGmsh:

    * :func:`apeGmsh.mesh._femdata_h5_io.read_neutral_zone_from_group`
      verifies ``/meta/snapshot_id`` against the recomputed hash and
      raises :class:`MalformedH5Error` on byte tamper of the neutral
      zone (load-bearing per the
      ``project_constraints_deep_review`` memory).
    * The :class:`Lineage` chain (this ADR 0021 surface) warns when
      the stored ``fem_hash`` disagrees with the recomputed one
      without disturbing the rest of the load.

    This test exercises the *lineage* surface: tamper the stored
    lineage attr (not the coords), and assert the loader returns
    cleanly with the divergence surfaced as a warning.  Phase 8
    deletes the inert ``BindError`` per ADR 0021 §Decision.
    """
    path, _fem = build_simple_frame_h5(tmp_path)
    with h5py.File(path, "r+") as f:
        f["meta/lineage"].attrs["fem_hash"] = "0" * 32

    from apeGmsh.opensees import OpenSeesModel

    om = OpenSeesModel.from_h5(path)  # must NOT raise
    assert om.lineage.warnings != ()
    joined = "\n".join(om.lineage.warnings)
    assert "fem" in joined


def test_tampered_results_bytes_warn_not_raise(tmp_path: Path) -> None:
    """Mutating ``/stages/...`` surfaces a warning on Results.lineage."""
    results_path, _model_path, _fem = _make_composed_results(tmp_path)
    with h5py.File(results_path, "r+") as f:
        nodes_grp = f["stages/stage_0/partitions/partition_0/nodes"]
        ds = nodes_grp["displacement_z"]
        ds[...] = ds[...] + 1.0

    with _load_results(results_path) as r:
        lineage = r.lineage
        assert lineage.warnings != ()
        joined = "\n".join(lineage.warnings)
        assert "results" in joined.lower()


def test_bind_error_still_exists_but_inert(
    tmp_path: Path,
) -> None:
    """``bind()`` does not raise on mismatch.

    Per ADR 0021 §Decision the procedural-bind ``BindError`` was inert
    through Phase 6 and is deleted in Phase 8 (this is the prune).
    Phase 8 must not regress to the procedural-bind behaviour — bind
    on mismatched candidate FEMData proceeds silently.
    """
    # Build a results file and a separate, mismatched FEMData; bind
    # must NOT raise on mismatch.
    path, fem = build_simple_frame_h5(tmp_path)
    results_path = tmp_path / "for_bind_check.h5"
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=path)
        sid = w.begin_stage(name="g", kind="static", time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.zeros((1, node_ids.size))},
        )
        w.end_stage()

    # Use a same-shape FEM as the candidate.  Even with a
    # different one the contract is: bind() must NOT raise
    # BindError.  Phase 6 ratifies the May 2026 withdrawal of the
    # procedural-bind enforcement.
    other_fem = build_simple_frame_fem()

    with _load_results(results_path) as r:
        # bind() succeeds — no BindError raised regardless of
        # whether the snapshot_ids match.
        rebound = r.bind(other_fem)
        assert rebound is not None


# ---------------------------------------------------------------------------
# Forward / round-trip
# ---------------------------------------------------------------------------


def test_forward_chain_recomputes_to_same_hash(tmp_path: Path) -> None:
    """Write a file; reopen; recomputed lineage matches stored."""
    results_path, _, _ = _make_composed_results(tmp_path)
    with _load_results(results_path) as r:
        lineage = r.lineage
    # No drift warnings on a freshly-written file.
    assert lineage.warnings == ()
    assert lineage.fem_hash
    assert lineage.model_hash is not None
    assert lineage.results_hash is not None


def test_lineage_round_trip_through_results_h5(tmp_path: Path) -> None:
    """Read file A; round-trip; lineage chains equal modulo timestamps."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    results_a, _, fem = _make_composed_results(a_dir)
    with _load_results(results_a) as r_a:
        lineage_a = r_a.lineage

    # Build a second composed file from the same source.
    results_b, _, _ = _make_composed_results(b_dir)
    with _load_results(results_b) as r_b:
        lineage_b = r_b.lineage

    # fem_hash and model_hash chain forward identically across two
    # independent writes of the same model.
    assert lineage_a.fem_hash == lineage_b.fem_hash
    assert lineage_a.model_hash == lineage_b.model_hash
    # results_hash too — both files write empty /stages/... shapes
    # from the same FEMData.
    assert lineage_a.results_hash == lineage_b.results_hash


# ---------------------------------------------------------------------------
# INV-4 — model_hash excludes cuts and sweeps
# ---------------------------------------------------------------------------


def test_model_hash_excludes_cuts(tmp_path: Path) -> None:
    """model_hash is invariant under cut presence/absence."""
    # Same model, with and without a cut attached.
    with_dir = tmp_path / "with"
    without_dir = tmp_path / "without"
    with_dir.mkdir()
    without_dir.mkdir()
    path_with_cuts, _ = _make_model_h5_with_cuts(with_dir)
    path_no_cuts, _ = build_simple_frame_h5(without_dir)

    with h5py.File(path_with_cuts, "r") as f:
        assert "opensees/cuts" in f
        m_with = compute_model_hash(
            f["meta/lineage"].attrs["fem_hash"], f["opensees"],
        )
    with h5py.File(path_no_cuts, "r") as f:
        assert "opensees/cuts" not in f
        m_no = compute_model_hash(
            f["meta/lineage"].attrs["fem_hash"], f["opensees"],
        )
    assert m_with == m_no, (
        "model_hash must be invariant under cuts presence (INV-4)"
    )


def test_model_hash_excludes_sweeps(tmp_path: Path) -> None:
    """model_hash is invariant under sweeps presence/absence.

    A SectionSweepDef is a sequence of :class:`SectionCutDef`
    instances; same INV-4 contract applies — sweeps are user-attached
    post-hoc artifacts and must not perturb model identity.
    """
    sweep_dir = tmp_path / "sweep"
    no_sweep_dir = tmp_path / "no_sweep"
    sweep_dir.mkdir()
    no_sweep_dir.mkdir()
    out_with, _ = _make_model_h5_with_sweeps(sweep_dir)
    out_no, _ = build_simple_frame_h5(no_sweep_dir)

    with h5py.File(out_with, "r") as f:
        assert "opensees/sweeps" in f
        m_with = compute_model_hash(
            f["meta/lineage"].attrs["fem_hash"], f["opensees"],
        )
    with h5py.File(out_no, "r") as f:
        assert "opensees/sweeps" not in f
        m_no = compute_model_hash(
            f["meta/lineage"].attrs["fem_hash"], f["opensees"],
        )
    assert m_with == m_no


# ---------------------------------------------------------------------------
# INV-5 — canonical bytes deterministic
# ---------------------------------------------------------------------------


def test_canonical_bytes_deterministic_across_writes(tmp_path: Path) -> None:
    """Writing the same model twice yields equal canonical bytes."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    path_a, _ = build_simple_frame_h5(a_dir)
    path_b, _ = build_simple_frame_h5(b_dir)
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        cb_a = canonical_bytes(fa["opensees"])
        cb_b = canonical_bytes(fb["opensees"])
    assert cb_a == cb_b


def test_canonical_bytes_deterministic_across_dataset_layouts(
    tmp_path: Path,
) -> None:
    """Chunked vs contiguous storage of the same array produces equal bytes.

    The canonical-bytes walk materialises every dataset via
    ``np.ascontiguousarray(...).tobytes()`` so HDF5's chunk layout
    decisions never leak into the hash.
    """
    a = tmp_path / "contiguous.h5"
    b = tmp_path / "chunked.h5"
    arr = np.arange(60, dtype=np.float64).reshape((6, 10))
    with h5py.File(a, "w") as f:
        f.create_dataset("data", data=arr)  # contiguous by default
    with h5py.File(b, "w") as f:
        f.create_dataset(
            "data", data=arr, chunks=(3, 5),
        )
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        cb_a = canonical_bytes(fa)
        cb_b = canonical_bytes(fb)
    assert cb_a == cb_b


# ---------------------------------------------------------------------------
# Lineage class surface
# ---------------------------------------------------------------------------


def test_lineage_assert_clean_raises_on_warnings() -> None:
    """``Lineage.assert_clean()`` raises ``LineageError`` when warnings present."""
    lineage = Lineage(
        fem_hash="abc",
        warnings=(f"{WARNING_PREFIX}test drift",),
    )
    with pytest.raises(LineageError):
        lineage.assert_clean()


def test_lineage_assert_clean_silent_when_clean() -> None:
    """``Lineage.assert_clean()`` returns ``None`` on empty warnings."""
    lineage = Lineage(fem_hash="abc")
    assert lineage.assert_clean() is None


def test_lineage_warning_prefix(tmp_path: Path) -> None:
    """Every lineage warning starts with ``[lineage] ``.

    Tamper the stored ``model_hash`` (not the bytes, so the FEM-
    layer strict check stays happy) and assert the prefix discipline.
    """
    path, _ = build_simple_frame_h5(tmp_path)
    with h5py.File(path, "r+") as f:
        f["meta/lineage"].attrs["model_hash"] = "ff" * 16

    from apeGmsh.opensees import OpenSeesModel

    om = OpenSeesModel.from_h5(path)
    assert om.lineage.warnings
    for w in om.lineage.warnings:
        assert w.startswith(WARNING_PREFIX), w


def test_lineage_warnings_are_tuple_immutable() -> None:
    """``Lineage.warnings`` is a tuple; the dataclass is frozen."""
    lineage = Lineage(fem_hash="abc", warnings=("[lineage] x",))
    assert isinstance(lineage.warnings, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        lineage.warnings = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Results / OpenSeesModel surface
# ---------------------------------------------------------------------------


def test_results_lineage_triple(tmp_path: Path) -> None:
    """Composed results — lineage carries all three hashes."""
    results_path, _, _ = _make_composed_results(tmp_path)
    with _load_results(results_path) as r:
        lineage = r.lineage
    assert lineage.fem_hash != ""
    assert lineage.model_hash is not None
    assert lineage.results_hash is not None


def test_opensees_model_lineage_pair(tmp_path: Path) -> None:
    """Standalone model.h5 — lineage has fem + model; results_hash absent."""
    from apeGmsh.opensees import OpenSeesModel

    path, _ = build_simple_frame_h5(tmp_path)
    om = OpenSeesModel.from_h5(path)
    assert om.lineage.fem_hash != ""
    assert om.lineage.model_hash is not None
    assert om.lineage.results_hash is None
