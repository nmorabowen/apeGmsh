"""Integration tests for ``ops.recorder.declare(...)`` end-to-end emit.

Phase 9 commit 3a wires the bridge to:

1. Construct a :class:`RecorderDeclaration` from the namespace method,
   capturing the bridge's ``ndm``/``ndf`` (D8) and expanding canonical
   shorthand at declaration time.
2. Fan out the declaration to one ``emitter.recorder(...)`` call per
   ``(ops_token, target_set)`` group at emit time via
   :func:`emit_recorder_spec`.

These tests cover the ``"nodes"`` category end-to-end through
:class:`RecordingEmitter`. Other categories raise
``NotImplementedError`` in commit 3a (covered by separate stub tests).

Equivalence test: declaring ``ops.recorder.declare(nodes="displacement",
pg="Top")`` produces the same nodal recorder vocabulary as
manually emitting ``ops.recorder.Node(file=..., response="disp",
nodes=(2,), dofs=(1, 2, 3))`` — modulo file path naming, the OpenSees
``recorder Node`` shape is equivalent.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.recorder import (
    RecorderDeclaration,
    RecorderRecord,
)

from apeGmsh.opensees._internal.build import BridgeError
from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_with_labels_and_selection,
    make_two_node_beam,
)


def _register_dummy_beam(ops, pg: str) -> None:
    """Register an elasticBeamColumn spec on ``pg`` so the bridge
    actually emits OpenSees elements at the PG's FEM eids.

    Element-level recorder tests need the bridge's
    ``fem_eid_to_ops_tag`` map to cover the targeted FEM eids — without
    a registered Element primitive the map is empty and
    :func:`_resolve_element_targets` raises ``BridgeError``.  This
    helper supplies the minimum geomTransf + elasticBeamColumn pair
    that the fan-out needs.

    Tag side-effects (relevant to assertions):
      * The spec itself consumes element-kind tag 1 via ``_register``.
      * Fan-out instances get tags 2, 3, ... per FEM eid in PG order.

    So FEM eid N maps to ops_tag (N + 1).
    """
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg=pg, transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )


# ---------------------------------------------------------------------------
# Namespace declare() — registration + shorthand expansion + D8 binding
# ---------------------------------------------------------------------------


class TestDeclareRegistration:
    def test_declare_returns_RecorderDeclaration(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        decl = ops.recorder.declare(nodes="displacement", pg="Top")
        assert isinstance(decl, RecorderDeclaration)

    def test_declare_registers_on_bridge(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        decl = ops.recorder.declare(nodes=("displacement_x",), pg="Top")
        # The bridge holds the declaration with an allocated tag
        assert ops.tag_for(decl) is not None

    def test_declare_captures_bridge_ndm_ndf(self) -> None:
        # Phase 9 D8: bridge state is the source of truth for ndm/ndf
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=2, ndf=3)
        decl = ops.recorder.declare(nodes="displacement", pg="Top")
        assert decl.ndm == 2
        assert decl.ndf == 3

    def test_declare_expands_shorthand_immediately(self) -> None:
        # "displacement" shorthand expands to per-axis canonicals at
        # declaration time, clipped by bridge's ndm.
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        decl = ops.recorder.declare(nodes="displacement", pg="Top")
        assert decl.records[0].components == (
            "displacement_x", "displacement_y", "displacement_z",
        )

    def test_declare_expands_shorthand_clipped_to_ndm_2d(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=2, ndf=3)
        decl = ops.recorder.declare(nodes="displacement", pg="Top")
        # 2D clips to (x, y) only
        assert decl.records[0].components == (
            "displacement_x", "displacement_y",
        )

    def test_declare_accepts_canonical_tuple(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        decl = ops.recorder.declare(
            nodes=("displacement_x", "reaction_force_y"),
            pg="Top",
        )
        assert decl.records[0].components == (
            "displacement_x", "reaction_force_y",
        )

    def test_declare_without_model_raises(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        # No ops.model() call → declare should raise
        with pytest.raises(RuntimeError, match="ops.model"):
            ops.recorder.declare(nodes="displacement", pg="Top")

    def test_declare_named(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        decl = ops.recorder.declare(
            nodes="displacement", pg="Top", name="roof_kinematics",
        )
        assert decl.name == "roof_kinematics"


# ---------------------------------------------------------------------------
# End-to-end emit through RecordingEmitter
# ---------------------------------------------------------------------------


class TestDeclareEmit:
    def test_nodes_pg_emit_produces_one_recorder_call(self) -> None:
        """A single 'displacement' declaration on a 3D model produces
        one ``recorder Node ... -dof 1 2 3 disp`` call."""
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement", pg="Top", dt=0.01, name="run1",
        )

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1, (
            f"expected 1 recorder call, got {len(recorder_calls)}: "
            f"{recorder_calls}"
        )

        name, args, _ = recorder_calls[0]
        assert name == "recorder"
        assert args[0] == "Node"
        # Args: "-file", <path>, "-dT", 0.01, "-time", "-node", 2,
        #       "-dof", 1, 2, 3, "disp"
        assert "-file" in args
        assert "-dT" in args
        # dt was 0.01
        dt_idx = args.index("-dT")
        assert args[dt_idx + 1] == 0.01
        assert "-time" in args
        assert "-node" in args
        node_idx = args.index("-node")
        # PG "Top" = node 2
        assert args[node_idx + 1] == 2
        assert "-dof" in args
        dof_idx = args.index("-dof")
        # Three DOFs (1, 2, 3) for displacement in 3D
        assert args[dof_idx + 1 : dof_idx + 4] == (1, 2, 3)
        # Last positional arg is the ops token
        assert args[-1] == "disp"

    def test_nodes_2d_emit_clips_to_two_dofs(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=2, ndf=3)
        ops.recorder.declare(nodes="displacement", pg="Top")

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        dof_idx = args.index("-dof")
        # 2D clips to DOFs (1, 2) — no z displacement
        assert args[dof_idx + 1 : dof_idx + 3] == (1, 2)

    def test_two_token_components_emit_two_calls(self) -> None:
        """``displacement_x + reaction_force_y`` route through different
        ops tokens (disp vs reaction) → two recorder calls."""
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes=("displacement_x", "reaction_force_y"),
            pg="Top",
        )

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 2

        ops_tokens = sorted(args[-1] for _, args, _ in recorder_calls)
        assert ops_tokens == ["disp", "reaction"]

    def test_same_token_components_collapse_to_one_call(self) -> None:
        """``displacement_x + displacement_y`` share the 'disp' token →
        one recorder call with two -dofs."""
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes=("displacement_x", "displacement_y"),
            pg="Top",
        )

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        assert args[-1] == "disp"
        dof_idx = args.index("-dof")
        assert args[dof_idx + 1 : dof_idx + 3] == (1, 2)

    def test_ids_selector_emit(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        # Explicit IDs instead of pg=
        ops.recorder.declare(nodes="displacement", ids=(1, 2))

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        node_idx = args.index("-node")
        assert args[node_idx + 1 : node_idx + 3] == (1, 2)


# ---------------------------------------------------------------------------
# Deferred categories (3b+) raise descriptive NotImplementedError
# ---------------------------------------------------------------------------


class TestDeclareElementsCategory:
    def test_elements_pg_emit(self) -> None:
        """``elements`` category emits one ``recorder Element ...
        globalForce`` line."""
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            elements=("nodal_resisting_force_x",),
            pg="Cols",
            dt=0.01,
        )

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        assert args[0] == "Element"
        assert "-ele" in args
        ele_idx = args.index("-ele")
        # PG "Cols" = FEM eid 1; spec consumed element-kind tag 1, so
        # the fan-out instance gets ops_tag 2.  The recorder must
        # target the OpenSees tag, not the FEM eid.
        assert args[ele_idx + 1] == 2
        # globalForce token at the end (matches the canonical's
        # routing in _response_catalog).
        assert args[-1] == "globalForce"

    def test_elements_local_frame_token(self) -> None:
        """``nodal_resisting_force_local_x`` routes through
        ``localForce`` instead of ``globalForce``."""
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            elements=("nodal_resisting_force_local_x",),
            pg="Cols",
        )

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        assert args[-1] == "localForce"

    def test_elements_pg_translates_fem_eids_to_ops_tags(self) -> None:
        """Regression: RecorderDeclaration's ``elements`` category
        resolves selectors to FEM eids, then translates through
        ``fem_eid_to_ops_tag`` so ``-ele`` carries the OpenSees tags
        the bridge actually emitted — not the raw FEM eids.

        Mirrors ``test_element_recorder_pg_resolves_to_explicit_element_ids``
        in ``test_full_emit_recording.py`` (the typed-``Element``
        recorder analogue).  Before this fix the RecorderDeclaration
        path passed raw FEM eids straight to ``-ele``, silently
        targeting the wrong elements whenever a spec consumed an
        allocator slot in ``_register``.
        """
        fem = make_two_column_frame_with_labels_and_selection()
        # PG "Cols" -> FEM eids (1, 2); spec consumes element-kind
        # tag 1; fan-out gives ops_tags (2, 3) for (eid 1, eid 2).
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            elements=("nodal_resisting_force_x",),
            ids=(1, 2),
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        ele_idx = args.index("-ele")
        # Critical: ops_tags (2, 3), NOT FEM eids (1, 2)
        assert args[ele_idx + 1 : ele_idx + 3] == (2, 3)

    def test_elements_missing_eid_raises_bridge_error(self) -> None:
        """Regression: targeting a FEM eid that no Element primitive
        emitted raises ``BridgeError`` with a helpful message.

        Before the fix this silently wrote a recorder line targeting
        an OpenSees tag that didn't exist; the output file would have
        nothing but the time column.  After the fix the user sees a
        loud error naming the missing eid.
        """
        fem = make_two_node_beam()  # FEM eid 1 in PG "Cols"
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        # NO Element primitive registered → fem_eid_to_ops_tag is empty.
        ops.recorder.declare(
            elements=("nodal_resisting_force_x",),
            pg="Cols",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        with pytest.raises(
            BridgeError, match="FEM element id 1.*no Element primitive",
        ):
            bm.emit(rec)


class TestDeclareGaussCategory:
    def test_gauss_stress_emit(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            gauss=("stress_xx", "stress_yy"),
            pg="Cols",
        )

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        assert args[0] == "Element"
        assert args[-1] == "stresses"

    def test_gauss_strain_emit(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            gauss=("strain_xx",),
            pg="Cols",
        )

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        assert args[-1] == "strains"

    def test_gauss_work_conjugate_mix_raises(self) -> None:
        """``stress_*`` + ``strain_*`` in one gauss record can't share
        a single ops.eleResponse call — must split."""
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            gauss=("stress_xx", "strain_yy"),
            pg="Cols",
            record_name="mixed",
        )

        bm = ops.build()
        rec = RecordingEmitter()
        with pytest.raises(ValueError, match="work-conjugate families"):
            bm.emit(rec)


class TestDeclareLineStationsCategory:
    def test_line_stations_emit(self) -> None:
        """``line_stations`` category emits two ``recorder Element``
        calls: one for ``section force`` (the canonical response) and
        a paired ``integrationPoints`` call writing to ``_gpx.out`` —
        the .out transcoder needs both to recover physical xi*L
        coordinates for section samples."""
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            line_stations=("axial_force", "bending_moment_y"),
            pg="Cols",
        )

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 2

        canonical = next(
            c for c in recorder_calls if "section" in c[1]
        )
        gpx = next(
            c for c in recorder_calls if "integrationPoints" in c[1]
        )

        _, c_args, _ = canonical
        assert c_args[0] == "Element"
        assert c_args[-2:] == ("section", "force")

        _, g_args, _ = gpx
        assert g_args[0] == "Element"
        assert g_args[-1] == "integrationPoints"
        # gpx file path mirrors the canonical path with _gpx suffix
        canonical_path = c_args[c_args.index("-file") + 1]
        gpx_path = g_args[g_args.index("-file") + 1]
        assert gpx_path == canonical_path.replace(".out", "_gpx.out")


class TestDeclareCrossCategory:
    def test_separate_declarations_per_category(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        # ids=(1,) for nodes is node tag 1; for elements it's FEM
        # eid 1 — which is translated through fem_eid_to_ops_tag
        # before landing in -ele.  See _register_dummy_beam for the
        # tag arithmetic (FEM eid 1 -> ops_tag 2).
        ops.recorder.declare(
            nodes=("displacement_x",),
            ids=(1,),
        )
        ops.recorder.declare(
            elements=("nodal_resisting_force_x",),
            ids=(1,),
            name="elem_decl",
        )

        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)

        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        # One Node call + one Element call
        kinds = sorted(args[0] for _, args, _ in recorder_calls)
        assert kinds == ["Element", "Node"]


class TestDeclareDeferredCategories:
    def test_fiber_record_emit_raises_with_domain_capture_hint(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)

        decl = RecorderDeclaration(
            records=(
                RecorderRecord(
                    category="fibers",
                    components=("fiber_stress",),
                    ids=(1,),
                ),
            ),
        )
        ops.register(decl)

        bm = ops.build()
        rec = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="DomainCapture"):
            bm.emit(rec)

    def test_layer_record_emit_raises_with_domain_capture_hint(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)

        decl = RecorderDeclaration(
            records=(
                RecorderRecord(
                    category="layers",
                    components=("fiber_stress",),
                    ids=(1,),
                ),
            ),
        )
        ops.register(decl)

        bm = ops.build()
        rec = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="DomainCapture"):
            bm.emit(rec)

    def test_modal_record_emit_raises_with_domain_capture_hint(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)

        decl = RecorderDeclaration(
            records=(
                RecorderRecord(category="modal", n_modes=5),
            ),
        )
        ops.register(decl)

        bm = ops.build()
        rec = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="DomainCapture"):
            bm.emit(rec)


# ---------------------------------------------------------------------------
# Phase 9 commit 3c — file_root kwarg
# ---------------------------------------------------------------------------


def _recorder_file_paths(rec: RecordingEmitter) -> list[str]:
    """Return the ``-file`` argument from every recorder call in order."""
    out: list[str] = []
    for name, args, _ in rec.calls:
        if name != "recorder":
            continue
        try:
            idx = args.index("-file")
        except ValueError:
            continue
        out.append(args[idx + 1])
    return out


class TestDeclareFileRoot:
    def test_default_file_root_is_dot(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(nodes="displacement_x", pg="Top")
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        paths = _recorder_file_paths(rec)
        assert all(p.startswith("./") for p in paths)

    def test_custom_file_root(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x", pg="Top", file_root="results",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        paths = _recorder_file_paths(rec)
        assert paths == [
            "results/default__default__disp.out",
        ]

    def test_file_root_with_trailing_slash_does_not_double_up(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x", pg="Top", file_root="results/",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        paths = _recorder_file_paths(rec)
        assert paths == ["results/default__default__disp.out"]

    def test_empty_file_root_yields_bare_filename(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x", pg="Top", file_root="",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        paths = _recorder_file_paths(rec)
        assert paths == ["default__default__disp.out"]

    def test_file_root_persisted_on_declaration(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        decl = ops.recorder.declare(
            nodes="displacement_x", pg="Top", file_root="out/",
        )
        assert decl.file_root == "out/"


# ---------------------------------------------------------------------------
# Phase 9 commit 3c — label= and selection= selectors
# ---------------------------------------------------------------------------


class TestDeclareLabelSelector:
    def test_nodes_label_resolves(self) -> None:
        fem = make_two_column_frame_with_labels_and_selection()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x", label="east_column",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        node_idx = args.index("-node")
        # east_column label = nodes 3, 4
        assert args[node_idx + 1 : node_idx + 3] == (3, 4)

    def test_elements_label_resolves(self) -> None:
        fem = make_two_column_frame_with_labels_and_selection()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            elements="nodal_resisting_force_x", label="east_column",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        ele_idx = args.index("-ele")
        # east_column label = FEM eid 2; spec consumed element-kind tag
        # 1, fan-out gives FEM eid 1 -> ops_tag 2 and FEM eid 2 -> 3.
        assert args[ele_idx + 1] == 3

    def test_unknown_label_raises_bridge_error(self) -> None:
        fem = make_two_column_frame_with_labels_and_selection()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x", label="missing_label",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        with pytest.raises(BridgeError, match="missing_label"):
            bm.emit(rec)


class TestDeclareSelectionSelector:
    def test_nodes_selection_resolves(self) -> None:
        fem = make_two_column_frame_with_labels_and_selection()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x", selection="upper_band",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        node_idx = args.index("-node")
        # upper_band selection = nodes 2, 4
        assert args[node_idx + 1 : node_idx + 3] == (2, 4)

    def test_elements_selection_resolves(self) -> None:
        fem = make_two_column_frame_with_labels_and_selection()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            elements="nodal_resisting_force_x", selection="upper_band",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        ele_idx = args.index("-ele")
        # upper_band selection = FEM eids (1, 2); spec consumed
        # element-kind tag 1 so FEM eid 1 -> ops_tag 2 and 2 -> 3.
        assert args[ele_idx + 1 : ele_idx + 3] == (2, 3)

    def test_selection_without_mesh_selection_raises(self) -> None:
        # make_two_node_beam has mesh_selection=None
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x", selection="anything",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        with pytest.raises(BridgeError, match="mesh_selection"):
            bm.emit(rec)

    def test_unknown_selection_raises_bridge_error(self) -> None:
        fem = make_two_column_frame_with_labels_and_selection()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x", selection="missing_set",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        with pytest.raises(BridgeError, match="missing_set"):
            bm.emit(rec)


class TestDeclareCombinedSelectors:
    def test_pg_plus_label_unions_and_dedups(self) -> None:
        """``pg="Top"`` (nodes 2, 4) + ``label="east_column"``
        (nodes 3, 4) → union is (2, 4, 3) — dedup keeps first-seen
        order from pg, label."""
        fem = make_two_column_frame_with_labels_and_selection()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x", pg="Top", label="east_column",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        node_idx = args.index("-node")
        # pg "Top" = (2, 4); label east_column = (3, 4); union with
        # dedup, first-seen order: (2, 4, 3)
        dof_idx = args.index("-dof")
        node_args = args[node_idx + 1 : dof_idx]
        assert node_args == (2, 4, 3)


# ---------------------------------------------------------------------------
# Phase 9 commit 3c — raw= per-category escape hatch
# ---------------------------------------------------------------------------


class TestDeclareRawNodes:
    def test_raw_only_emits_one_recorder(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(raw_nodes=("eigen",), pg="Top")
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        assert args[0] == "Node"
        assert args[-1] == "eigen"
        # Default dofs = all DOFs from bridge ndf=6
        dof_idx = args.index("-dof")
        assert args[dof_idx + 1 : dof_idx + 7] == (1, 2, 3, 4, 5, 6)
        # File path uses raw_<token> tag
        file_idx = args.index("-file")
        assert "raw_eigen" in args[file_idx + 1]

    def test_raw_alongside_canonical_emits_separate_recorders(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(
            nodes="displacement_x",
            raw_nodes=("eigen",),
            pg="Top",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 2
        # One canonical disp + one raw eigen
        last_tokens = sorted(args[-1] for _, args, _ in recorder_calls)
        assert last_tokens == ["disp", "eigen"]

    def test_raw_token_with_spaces_sanitized_in_filename(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.recorder.declare(raw_nodes=("eigen 1",), pg="Top")
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        # The OpenSees response token stays verbatim (with space)
        assert args[-1] == "eigen 1"
        # But the filename has the space replaced with underscore
        file_idx = args.index("-file")
        assert "raw_eigen_1" in args[file_idx + 1]
        assert " " not in args[file_idx + 1]


class TestDeclareRawElements:
    def test_raw_elements_emits_one_recorder(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            raw_elements=("customResponse",), pg="Cols",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        assert args[0] == "Element"
        assert args[-1] == "customResponse"

    def test_raw_gauss_emits_one_recorder(self) -> None:
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            raw_gauss=("customGauss",), pg="Cols",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 1
        _, args, _ = recorder_calls[0]
        assert args[0] == "Element"
        assert args[-1] == "customGauss"


# ---------------------------------------------------------------------------
# Phase 9 commit 3c — line_stations IP pairing edge cases
# ---------------------------------------------------------------------------


class TestDeclareLineStationsRawIp:
    def test_raw_line_stations_still_emits_gpx_pair(self) -> None:
        """A line_stations record with only raw tokens (no canonical
        components) still emits the paired ``integrationPoints``
        recorder — the .out transcoder needs GP positions regardless
        of which response token was used."""
        fem = make_two_node_beam()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        _register_dummy_beam(ops, pg="Cols")
        ops.recorder.declare(
            raw_line_stations=("section.shear",),
            pg="Cols",
        )
        bm = ops.build()
        rec = RecordingEmitter()
        bm.emit(rec)
        recorder_calls = [c for c in rec.calls if c[0] == "recorder"]
        assert len(recorder_calls) == 2

        last_tokens = sorted(args[-1] for _, args, _ in recorder_calls)
        assert last_tokens == ["integrationPoints", "section.shear"]
