"""``Results.from_fem`` — bind a bare :class:`FEMData` snapshot to a
results file without going through the ``apeSees`` bridge.

Covers the reported gap: a physical-group model where
``fem = g.mesh.queries.get_fem_data()`` drove a hand-written OpenSees
deck previously had no first-class route to ``Results`` / ``viewer``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.Results import _resolve_results_kind
from apeGmsh.results.writers import NativeWriter


# ---------------------------------------------------------------------
# kind resolution (pure)
# ---------------------------------------------------------------------


class TestResolveKind:
    def test_auto_detects_suffix(self) -> None:
        assert _resolve_results_kind("auto", "run.mpco") == "mpco"
        assert _resolve_results_kind("auto", "run.ladruno") == "ladruno"

    def test_explicit_passthrough(self) -> None:
        assert _resolve_results_kind("native", "run.h5") == "native"

    def test_auto_list_input_peeks_first(self) -> None:
        assert _resolve_results_kind(
            "auto", ["a.part-0.mpco", "a.part-1.mpco"]) == "mpco"

    def test_ambiguous_native_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot auto-detect"):
            _resolve_results_kind("auto", "run.h5")

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="not one of"):
            _resolve_results_kind("bogus", "run.mpco")


# ---------------------------------------------------------------------
# composed-fem guard (adversarial F1 — silent element mislabel)
# ---------------------------------------------------------------------


def test_composed_fem_refused(tmp_path: Path) -> None:
    """A composed fem carries no fem_eid<->ops-tag map in a neutral-only
    model.h5 — refuse rather than silently mislabel element results."""
    composed = SimpleNamespace(composed_from=("module",), snapshot_id="x")
    with pytest.raises(ValueError, match="composed"):
        Results.from_fem(composed, tmp_path / "run.mpco")


# ---------------------------------------------------------------------
# error messages point at the new route
# ---------------------------------------------------------------------


def test_from_mpco_missing_model_h5_names_from_fem() -> None:
    with pytest.raises(TypeError, match="from_fem"):
        Results.from_mpco("nonexistent.mpco")


def test_from_native_missing_model_names_from_fem() -> None:
    with pytest.raises(TypeError, match="from_fem"):
        Results.from_native("nonexistent.h5")


# ---------------------------------------------------------------------
# native round-trip — end-to-end plumbing (cache write + bind + read)
# ---------------------------------------------------------------------


def _native_results_no_model(g, tmp_path: Path):
    """A native results file with NO embedded /opensees zone — the
    bare-fem scenario from_fem targets."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    results = tmp_path / "grav.h5"
    with NativeWriter(results) as w:
        w.open(fem=fem)  # no model_h5_src => no bridge model in the file
        sid = w.begin_stage(name="grav", kind="static",
                            time=np.array([0.0]))
        w.write_nodes(
            sid, "partition_0",
            node_ids=np.asarray(fem.nodes.ids, dtype=np.int64),
            components={
                "displacement_x": np.zeros((1, len(fem.nodes.ids))),
            },
        )
        w.end_stage()
    return results, fem


def test_from_fem_native_roundtrip(g, tmp_path: Path) -> None:
    results, fem = _native_results_no_model(g, tmp_path)

    r = Results.from_fem(fem, results, kind="native", cache_root=tmp_path)

    # bound to the passed snapshot
    assert r.fem is not None
    assert r.fem.snapshot_id == fem.snapshot_id
    # a neutral-only model was materialised and bound
    assert r.model is not None and r.model.fem is not None
    # the cache file landed under <cache_root>/from_fem/
    cached = tmp_path / "from_fem" / f"{fem.snapshot_id}.model.h5"
    assert cached.exists()
    # nodal results read back through the bound reader
    slab = r.nodes.get(component="displacement_x")
    assert slab.values.shape[1] == len(fem.nodes.ids)


def test_from_fem_reuses_cached_model(g, tmp_path: Path) -> None:
    """A second from_fem for the same fem reuses the cached model.h5
    (keyed by snapshot_id) rather than rewriting it."""
    results, fem = _native_results_no_model(g, tmp_path)
    cached = tmp_path / "from_fem" / f"{fem.snapshot_id}.model.h5"

    Results.from_fem(fem, results, kind="native", cache_root=tmp_path)
    assert cached.exists()
    mtime = cached.stat().st_mtime_ns

    Results.from_fem(fem, results, kind="native", cache_root=tmp_path)
    assert cached.stat().st_mtime_ns == mtime  # not rewritten
