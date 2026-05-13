"""Phase 9 commit 6 — unified ``/opensees/recorders/`` archive (schema 2.3.0).

Schema 2.3.0 collapses the two recorder declaration systems
(``Node`` / ``Element`` / ``MPCO`` typed primitives and
``ops.recorder.declare(...)``-driven fan-out calls) into a single
``/opensees/recorders/`` group. Every record group carries a
``kind`` attr — ``"typed"`` or ``"declared"`` — and declared
records also expose the original declaration metadata
(``declaration_name``, ``record_name``, ``category``,
``components``, etc.) as attrs.

These tests exercise the emitter / reader round-trip end-to-end:

1. SCHEMA_VERSION bumped to 2.3.0.
2. Typed-primitive ``recorder(...)`` calls land as ``kind="typed"``.
3. ``recorder_declaration_begin`` / ``recorder_declaration_end``
   bracket fan-out calls; each fan-out lands as ``kind="declared"``
   plus the declaration metadata attrs.
4. The reader's :meth:`H5Model.recorders` returns rich per-record
   dicts; the kind attr is synthesized as ``"typed"`` for legacy
   2.0.0–2.2.0 archives.
5. End-to-end through the bridge: ``ops.recorder.Node(...)`` plus
   ``ops.recorder.declare(...)`` write side by side under the same
   group; both kinds round-trip cleanly.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.h5 import H5Emitter, SCHEMA_VERSION
from apeGmsh.opensees.emitter import h5_reader


# ---------------------------------------------------------------------------
# Schema-version bump
# ---------------------------------------------------------------------------


def test_schema_version_is_2_5_0() -> None:
    assert SCHEMA_VERSION == "2.5.0"


def test_schema_2_5_0_writes_to_meta(tmp_path: Path) -> None:
    e = H5Emitter()
    out = tmp_path / "x.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs["schema_version"] == "2.5.0"


# ---------------------------------------------------------------------------
# Typed primitive — kind="typed"
# ---------------------------------------------------------------------------


class TestTypedKindAttr:
    def test_typed_node_recorder_writes_kind_typed(
        self, tmp_path: Path,
    ) -> None:
        e = H5Emitter()
        e.recorder(
            "Node", "-file", "disp.out", "-node", 1, "-dof", 1, "disp",
        )
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            g = f["opensees/recorders/Node_0"]
            assert g.attrs["kind"] == "typed"
            assert g.attrs["type"] == "Node"
            assert g.attrs["file"] == "disp.out"
            # Declared-only attrs are absent.
            assert "declaration_name" not in g.attrs
            assert "category" not in g.attrs

    def test_typed_mpco_recorder_writes_kind_typed(
        self, tmp_path: Path,
    ) -> None:
        e = H5Emitter()
        e.recorder("mpco", "run.mpco", "-N", "displacement")
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            g = f["opensees/recorders/mpco_0"]
            assert g.attrs["kind"] == "typed"


# ---------------------------------------------------------------------------
# Declared record — kind="declared" + metadata attrs
# ---------------------------------------------------------------------------


class TestDeclaredKindAttr:
    def test_declared_record_carries_metadata_attrs(
        self, tmp_path: Path,
    ) -> None:
        e = H5Emitter()
        e.recorder_declaration_begin(
            declaration_name="default",
            record_name="top",
            category="nodes",
            components=("displacement_x", "displacement_y", "displacement_z"),
            pg=("Top",),
            dt=0.01,
            file_root="results/",
        )
        e.recorder(
            "Node",
            "-file", "results/default__top__disp.out",
            "-node", 2,
            "-dof", 1, 2, 3, "disp",
        )
        e.recorder_declaration_end()
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            g = f["opensees/recorders/Node_0"]
            assert g.attrs["kind"] == "declared"
            assert g.attrs["type"] == "Node"
            assert g.attrs["declaration_name"] == "default"
            assert g.attrs["record_name"] == "top"
            assert g.attrs["category"] == "nodes"
            comps = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in g.attrs["components"]
            ]
            assert comps == [
                "displacement_x", "displacement_y", "displacement_z",
            ]
            pgs = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in g.attrs["pg"]
            ]
            assert pgs == ["Top"]
            assert float(g.attrs["dt"]) == pytest.approx(0.01)
            assert g.attrs["file_root"] == "results/"

    def test_declared_record_multi_token_emits_multiple_groups(
        self, tmp_path: Path,
    ) -> None:
        """One declaration record fanning out to multiple ``recorder()``
        calls writes one ``/opensees/recorders/{name}_{i}`` group per
        call; each carries the same declaration metadata."""
        e = H5Emitter()
        e.recorder_declaration_begin(
            declaration_name="d",
            record_name="r",
            category="nodes",
            components=("displacement_x", "velocity_x"),
            ids=(1, 2),
        )
        e.recorder("Node", "-file", "a.out", "-node", 1, 2, "-dof", 1, "disp")
        e.recorder("Node", "-file", "b.out", "-node", 1, 2, "-dof", 1, "vel")
        e.recorder_declaration_end()
        out = tmp_path / "x.h5"
        e.write(str(out))
        with h5py.File(out, "r") as f:
            g0 = f["opensees/recorders/Node_0"]
            g1 = f["opensees/recorders/Node_1"]
            assert g0.attrs["kind"] == g1.attrs["kind"] == "declared"
            assert g0.attrs["declaration_name"] == g1.attrs["declaration_name"]
            assert g0.attrs["record_name"] == g1.attrs["record_name"]
            assert g0.attrs["file"] == "a.out"
            assert g1.attrs["file"] == "b.out"


# ---------------------------------------------------------------------------
# Stack discipline
# ---------------------------------------------------------------------------


class TestDeclarationContextStack:
    def test_nested_begin_raises(self) -> None:
        e = H5Emitter()
        e.recorder_declaration_begin(
            declaration_name="a",
            record_name=None,
            category="nodes",
            components=(),
        )
        with pytest.raises(RuntimeError, match="already open"):
            e.recorder_declaration_begin(
                declaration_name="b",
                record_name=None,
                category="nodes",
                components=(),
            )

    def test_end_without_begin_raises(self) -> None:
        e = H5Emitter()
        with pytest.raises(RuntimeError, match="no active"):
            e.recorder_declaration_end()


# ---------------------------------------------------------------------------
# Reader round-trip
# ---------------------------------------------------------------------------


class TestRecordersReader:
    def test_reader_synthesizes_kind_typed_for_legacy_archives(
        self, tmp_path: Path,
    ) -> None:
        """Legacy 2.0.0 / 2.1.0 / 2.2.0 archives wrote no ``kind`` attr.
        The reader synthesizes ``kind="typed"`` so callers can branch
        uniformly on the field."""
        out = tmp_path / "legacy.h5"
        with h5py.File(out, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["schema_version"] = "2.2.0"
            meta.attrs["snapshot_id"] = "x"
            meta.attrs["model_name"] = "x"
            meta.attrs["ndm"] = 3
            meta.attrs["ndf"] = 6
            ops = f.create_group("opensees")
            recs = ops.create_group("recorders")
            g = recs.create_group("Node_0")
            g.attrs["type"] = "Node"
            g.attrs["file"] = "out.out"
        with h5_reader.open(str(out)) as model:
            recorders = model.recorders()
            assert recorders["Node_0"]["kind"] == "typed"

    def test_reader_returns_both_kinds(
        self, tmp_path: Path,
    ) -> None:
        e = H5Emitter()
        e.recorder("Node", "-file", "raw.out", "-node", 1, "-dof", 1, "disp")
        e.recorder_declaration_begin(
            declaration_name="default",
            record_name="top",
            category="nodes",
            components=("displacement_x",),
            pg=("Top",),
        )
        e.recorder(
            "Node",
            "-file", "default__top__disp.out",
            "-node", 2, "-dof", 1, "disp",
        )
        e.recorder_declaration_end()
        out = tmp_path / "mixed.h5"
        e.write(str(out))
        with h5_reader.open(str(out)) as model:
            recorders = model.recorders()
            kinds = {name: r["kind"] for name, r in recorders.items()}
        # First call became Node_0 (typed); the declared fan-out became Node_1.
        assert kinds == {"Node_0": "typed", "Node_1": "declared"}


# ---------------------------------------------------------------------------
# End-to-end through the bridge build pipeline
# ---------------------------------------------------------------------------


class _NodesStub:
    def __init__(self, ids: tuple[int, ...]) -> None:
        import numpy as np
        self.ids = np.asarray(ids, dtype=np.int64)
        self.coords = np.zeros((len(ids), 3), dtype=np.float64)


class _ElementsStub:
    def __init__(self, ids: tuple[int, ...]) -> None:
        import numpy as np
        self.ids = np.asarray(ids, dtype=np.int64)


class _FEMStub:
    def __init__(self) -> None:
        self.nodes = _NodesStub((1, 2, 3))
        self.elements = _ElementsStub((1, 2))
        self.snapshot_id = "stub"
        self.mesh_selection = None


class TestBridgeIntegration:
    def test_declare_and_typed_coexist_in_archive(
        self, tmp_path: Path,
    ) -> None:
        fem = _FEMStub()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)

        # Typed primitive.
        ops.recorder.Node(
            file="typed.out",
            response="disp",
            nodes=(1,),
            dofs=(1, 2, 3),
        )
        # Declared (canonical Phase 9 path).
        ops.recorder.declare(
            nodes="displacement",
            ids=(2, 3),
            name="default",
            record_name="middle",
        )

        emitter = H5Emitter()
        built = ops.build()
        built.emit(emitter)
        out = tmp_path / "model.h5"
        emitter.write(str(out))

        with h5_reader.open(str(out)) as model:
            recorders = model.recorders()
        kinds = {r["kind"] for r in recorders.values()}
        assert kinds == {"typed", "declared"}
        # Find the declared record and verify metadata threading.
        declared = [r for r in recorders.values() if r["kind"] == "declared"]
        assert declared, "expected at least one declared record"
        d = declared[0]
        assert d["declaration_name"] == "default"
        assert d["record_name"] == "middle"
        assert d["category"] == "nodes"
        comps = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in d["components"]
        ]
        # ndm=3 ⇒ displacement → x, y, z.
        assert comps == ["displacement_x", "displacement_y", "displacement_z"]
