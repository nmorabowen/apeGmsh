"""Unit tests for ``apeGmsh.opensees.recorder``.

Phase 3B ships three recorder primitives: Node, Element, MPCO. Each
class gets:

  * construction (defaults, explicit values)
  * validation (mutually-exclusive nodes/pg or elements/pg, dofs/
    response token requirements, dT/nsteps mutex on MPCO)
  * ``_emit`` records the right call into a ``RecordingEmitter``
    (with explicit nodes/elements; pg= path raises ``NotImplementedError``)
  * ``dependencies()`` returns ``()`` (recorders are leaves)
  * ``__repr__`` includes the class name

Tests use ``RecordingEmitter`` only — no openseespy, no gmsh, no
subprocess.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.recorder import MPCO, Element, Node


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TestNodeConstruction:
    def test_minimal_with_explicit_nodes(self) -> None:
        r = Node(
            file="disp.out",
            response="disp",
            nodes=(1, 2, 3),
            dofs=(1, 2),
        )
        assert r.file == "disp.out"
        assert r.response == "disp"
        assert r.nodes == (1, 2, 3)
        assert r.pg is None
        assert r.dofs == (1, 2)
        assert r.dT is None
        assert r.time_format == "step"

    def test_minimal_with_pg(self) -> None:
        r = Node(
            file="disp.out",
            response="disp",
            pg="Roof",
            dofs=(1,),
        )
        assert r.pg == "Roof"
        assert r.nodes is None

    def test_neither_nodes_nor_pg_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of nodes= or pg="):
            Node(file="x.out", response="disp", dofs=(1,))

    def test_both_nodes_and_pg_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of nodes= or pg="):
            Node(
                file="x.out",
                response="disp",
                nodes=(1, 2),
                pg="Roof",
                dofs=(1,),
            )

    def test_empty_dofs_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one dof required"):
            Node(file="x.out", response="disp", nodes=(1,), dofs=())

    def test_invalid_time_format_raises(self) -> None:
        with pytest.raises(
            ValueError, match="time_format must be 'step' or 'dt'"
        ):
            Node(
                file="x.out",
                response="disp",
                nodes=(1,),
                dofs=(1,),
                time_format="seconds",
            )


class TestNodeEmit:
    def test_explicit_nodes_basic(self) -> None:
        r = Node(
            file="disp.out",
            response="disp",
            nodes=(1, 2, 3),
            dofs=(1, 2),
        )
        e = RecordingEmitter()
        r._emit(e, tag=10)
        assert e.calls == [
            (
                "recorder",
                (
                    "Node",
                    "-file", "disp.out",
                    "-node", 1, 2, 3,
                    "-dof", 1, 2, "disp",
                ),
                {},
            )
        ]

    def test_dt_appears_when_set(self) -> None:
        r = Node(
            file="disp.out",
            response="disp",
            nodes=(5,),
            dofs=(1,),
            dT=0.01,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            (
                "recorder",
                (
                    "Node",
                    "-file", "disp.out",
                    "-dT", 0.01,
                    "-node", 5,
                    "-dof", 1, "disp",
                ),
                {},
            )
        ]

    def test_time_flag_appears_when_dt_format(self) -> None:
        r = Node(
            file="disp.out",
            response="disp",
            nodes=(5,),
            dofs=(1,),
            time_format="dt",
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        # -time appears (no value), between -dT (absent here) and -node.
        assert "-time" in e.calls[0][1]

    def test_step_format_omits_time_flag(self) -> None:
        r = Node(
            file="disp.out",
            response="disp",
            nodes=(5,),
            dofs=(1,),
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert "-time" not in e.calls[0][1]

    def test_full_optional_block(self) -> None:
        r = Node(
            file="reaction.out",
            response="reaction",
            nodes=(7, 8),
            dofs=(1, 2, 3),
            dT=0.05,
            time_format="dt",
        )
        e = RecordingEmitter()
        r._emit(e, tag=2)
        assert e.calls == [
            (
                "recorder",
                (
                    "Node",
                    "-file", "reaction.out",
                    "-dT", 0.05,
                    "-time",
                    "-node", 7, 8,
                    "-dof", 1, 2, 3, "reaction",
                ),
                {},
            )
        ]

    def test_pg_path_raises_not_implemented(self) -> None:
        r = Node(
            file="disp.out",
            response="disp",
            pg="Roof",
            dofs=(1,),
        )
        e = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="Phase 4 build"):
            r._emit(e, tag=1)


class TestNodeContract:
    def test_dependencies_is_empty(self) -> None:
        r = Node(file="x.out", response="disp", nodes=(1,), dofs=(1,))
        assert r.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        r = Node(file="x.out", response="disp", nodes=(1,), dofs=(1,))
        assert "Node" in repr(r)

    def test_is_frozen(self) -> None:
        r = Node(file="x.out", response="disp", nodes=(1,), dofs=(1,))
        with pytest.raises(Exception):  # FrozenInstanceError
            r.file = "y.out"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Element
# ---------------------------------------------------------------------------

class TestElementConstruction:
    def test_minimal_with_explicit_elements(self) -> None:
        r = Element(
            file="force.out",
            response=("globalForce",),
            elements=(1, 2, 3),
        )
        assert r.file == "force.out"
        assert r.response == ("globalForce",)
        assert r.elements == (1, 2, 3)
        assert r.pg is None
        assert r.dT is None
        assert r.time_format == "step"

    def test_minimal_with_pg(self) -> None:
        r = Element(
            file="force.out",
            response=("globalForce",),
            pg="Cols",
        )
        assert r.pg == "Cols"
        assert r.elements is None

    def test_multi_token_response(self) -> None:
        r = Element(
            file="sec.out",
            response=("section", "1", "force"),
            elements=(5,),
        )
        assert r.response == ("section", "1", "force")

    def test_neither_elements_nor_pg_raises(self) -> None:
        with pytest.raises(
            ValueError, match="exactly one of elements= or pg="
        ):
            Element(file="x.out", response=("globalForce",))

    def test_both_elements_and_pg_raises(self) -> None:
        with pytest.raises(
            ValueError, match="exactly one of elements= or pg="
        ):
            Element(
                file="x.out",
                response=("globalForce",),
                elements=(1,),
                pg="Cols",
            )

    def test_empty_response_raises(self) -> None:
        with pytest.raises(ValueError, match="response token required"):
            Element(file="x.out", response=(), elements=(1,))

    def test_invalid_time_format_raises(self) -> None:
        with pytest.raises(
            ValueError, match="time_format must be 'step' or 'dt'"
        ):
            Element(
                file="x.out",
                response=("globalForce",),
                elements=(1,),
                time_format="seconds",
            )


class TestElementEmit:
    def test_explicit_elements_single_token(self) -> None:
        r = Element(
            file="force.out",
            response=("globalForce",),
            elements=(1, 2, 3),
        )
        e = RecordingEmitter()
        r._emit(e, tag=20)
        assert e.calls == [
            (
                "recorder",
                (
                    "Element",
                    "-file", "force.out",
                    "-ele", 1, 2, 3,
                    "globalForce",
                ),
                {},
            )
        ]

    def test_explicit_elements_multi_token_response(self) -> None:
        r = Element(
            file="sec.out",
            response=("section", "1", "force"),
            elements=(5, 6),
        )
        e = RecordingEmitter()
        r._emit(e, tag=21)
        assert e.calls == [
            (
                "recorder",
                (
                    "Element",
                    "-file", "sec.out",
                    "-ele", 5, 6,
                    "section", "1", "force",
                ),
                {},
            )
        ]

    def test_dt_appears_when_set(self) -> None:
        r = Element(
            file="x.out",
            response=("globalForce",),
            elements=(1,),
            dT=0.02,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            (
                "recorder",
                (
                    "Element",
                    "-file", "x.out",
                    "-dT", 0.02,
                    "-ele", 1,
                    "globalForce",
                ),
                {},
            )
        ]

    def test_time_flag_appears_when_dt_format(self) -> None:
        r = Element(
            file="x.out",
            response=("globalForce",),
            elements=(1,),
            time_format="dt",
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert "-time" in e.calls[0][1]

    def test_step_format_omits_time_flag(self) -> None:
        r = Element(
            file="x.out",
            response=("globalForce",),
            elements=(1,),
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert "-time" not in e.calls[0][1]

    def test_full_optional_block(self) -> None:
        r = Element(
            file="stress.out",
            response=("stresses",),
            elements=(10, 11, 12),
            dT=0.1,
            time_format="dt",
        )
        e = RecordingEmitter()
        r._emit(e, tag=3)
        assert e.calls == [
            (
                "recorder",
                (
                    "Element",
                    "-file", "stress.out",
                    "-dT", 0.1,
                    "-time",
                    "-ele", 10, 11, 12,
                    "stresses",
                ),
                {},
            )
        ]

    def test_pg_path_raises_not_implemented(self) -> None:
        r = Element(
            file="x.out",
            response=("globalForce",),
            pg="Cols",
        )
        e = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="Phase 4 build"):
            r._emit(e, tag=1)


class TestElementContract:
    def test_dependencies_is_empty(self) -> None:
        r = Element(
            file="x.out", response=("globalForce",), elements=(1,),
        )
        assert r.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        r = Element(
            file="x.out", response=("globalForce",), elements=(1,),
        )
        assert "Element" in repr(r)

    def test_is_frozen(self) -> None:
        r = Element(
            file="x.out", response=("globalForce",), elements=(1,),
        )
        with pytest.raises(Exception):
            r.file = "y.out"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MPCO
# ---------------------------------------------------------------------------

class TestMPCOConstruction:
    def test_minimal_nodal_only(self) -> None:
        r = MPCO(file="run.mpco", nodal_responses=("displacement",))
        assert r.file == "run.mpco"
        assert r.nodal_responses == ("displacement",)
        assert r.elem_responses == ()
        assert r.dT is None
        assert r.nsteps is None

    def test_minimal_elem_only(self) -> None:
        r = MPCO(file="run.mpco", elem_responses=("stresses",))
        assert r.elem_responses == ("stresses",)
        assert r.nodal_responses == ()

    def test_both_nodal_and_elem_responses(self) -> None:
        r = MPCO(
            file="run.mpco",
            nodal_responses=("displacement", "reactionForce"),
            elem_responses=("stresses", "section.fiber.stress"),
        )
        assert len(r.nodal_responses) == 2
        assert len(r.elem_responses) == 2

    def test_empty_responses_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match="at least one of nodal_responses or elem_responses",
        ):
            MPCO(file="run.mpco")

    def test_dt_and_nsteps_both_set_raises(self) -> None:
        with pytest.raises(
            ValueError, match="supply only one of dT or nsteps"
        ):
            MPCO(
                file="run.mpco",
                nodal_responses=("displacement",),
                dT=0.01,
                nsteps=10,
            )


class TestMPCOEmit:
    def test_nodal_only(self) -> None:
        r = MPCO(file="run.mpco", nodal_responses=("displacement",))
        e = RecordingEmitter()
        r._emit(e, tag=30)
        assert e.calls == [
            (
                "recorder",
                ("mpco", "run.mpco", "-N", "displacement"),
                {},
            )
        ]

    def test_elem_only(self) -> None:
        r = MPCO(file="run.mpco", elem_responses=("stresses",))
        e = RecordingEmitter()
        r._emit(e, tag=31)
        assert e.calls == [
            (
                "recorder",
                ("mpco", "run.mpco", "-E", "stresses"),
                {},
            )
        ]

    def test_nodal_and_elem(self) -> None:
        r = MPCO(
            file="run.mpco",
            nodal_responses=("displacement", "reactionForce"),
            elem_responses=("stresses",),
        )
        e = RecordingEmitter()
        r._emit(e, tag=32)
        assert e.calls == [
            (
                "recorder",
                (
                    "mpco", "run.mpco",
                    "-N", "displacement", "reactionForce",
                    "-E", "stresses",
                ),
                {},
            )
        ]

    def test_dt_cadence(self) -> None:
        r = MPCO(
            file="run.mpco",
            nodal_responses=("displacement",),
            dT=0.05,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            (
                "recorder",
                (
                    "mpco", "run.mpco",
                    "-N", "displacement",
                    "-T", 0.05,
                ),
                {},
            )
        ]

    def test_nsteps_cadence(self) -> None:
        r = MPCO(
            file="run.mpco",
            nodal_responses=("displacement",),
            nsteps=10,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            (
                "recorder",
                (
                    "mpco", "run.mpco",
                    "-N", "displacement",
                    "-T", 10,
                ),
                {},
            )
        ]

    def test_no_cadence_omits_T_flag(self) -> None:
        r = MPCO(file="run.mpco", nodal_responses=("displacement",))
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert "-T" not in e.calls[0][1]


class TestMPCOContract:
    def test_dependencies_is_empty(self) -> None:
        r = MPCO(file="run.mpco", nodal_responses=("displacement",))
        assert r.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        r = MPCO(file="run.mpco", nodal_responses=("displacement",))
        assert "MPCO" in repr(r)

    def test_is_frozen(self) -> None:
        r = MPCO(file="run.mpco", nodal_responses=("displacement",))
        with pytest.raises(Exception):
            r.file = "y.mpco"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Namespace integration — namespace methods register with the bridge
# ---------------------------------------------------------------------------

def _make_ops() -> "apeSees":
    """Construct an apeSees with a stub FEMData (namespaces ignore it)."""
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


class TestRecorderNamespace:
    def test_node_namespace_constructs_and_registers(self) -> None:
        ops = _make_ops()
        r = ops.recorder.Node(
            file="disp.out",
            response="disp",
            nodes=(1, 2),
            dofs=(1, 2),
        )
        assert isinstance(r, Node)
        assert r.file == "disp.out"
        assert ops.tag_for(r) == 1

    def test_element_namespace_constructs_and_registers(self) -> None:
        ops = _make_ops()
        r = ops.recorder.Element(
            file="force.out",
            response=("globalForce",),
            elements=(1,),
        )
        assert isinstance(r, Element)
        assert ops.tag_for(r) == 1

    def test_mpco_namespace_constructs_and_registers(self) -> None:
        ops = _make_ops()
        r = ops.recorder.MPCO(
            file="run.mpco",
            nodal_responses=("displacement",),
        )
        assert isinstance(r, MPCO)
        assert ops.tag_for(r) == 1

    def test_distinct_recorders_get_distinct_tags(self) -> None:
        ops = _make_ops()
        a = ops.recorder.Node(
            file="a.out", response="disp", nodes=(1,), dofs=(1,),
        )
        b = ops.recorder.Element(
            file="b.out", response=("globalForce",), elements=(1,),
        )
        c = ops.recorder.MPCO(
            file="run.mpco", nodal_responses=("displacement",),
        )
        assert ops.tag_for(a) == 1
        assert ops.tag_for(b) == 2
        assert ops.tag_for(c) == 3

    def test_node_namespace_validates(self) -> None:
        ops = _make_ops()
        with pytest.raises(ValueError, match="exactly one of nodes= or pg="):
            ops.recorder.Node(file="x.out", response="disp", dofs=(1,))

    def test_element_namespace_validates(self) -> None:
        ops = _make_ops()
        with pytest.raises(
            ValueError, match="exactly one of elements= or pg="
        ):
            ops.recorder.Element(file="x.out", response=("globalForce",))

    def test_mpco_namespace_validates(self) -> None:
        ops = _make_ops()
        with pytest.raises(
            ValueError,
            match="at least one of nodal_responses or elem_responses",
        ):
            ops.recorder.MPCO(file="run.mpco")
