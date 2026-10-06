"""ADR 0018 amendment — ``ModelData`` recorder surface.

Hand-written decks declare recorders in canonical vocabulary via
:meth:`ModelData.recorders`, then either forward them into a live
``openseespy`` session (:meth:`attach_recorders`) or render them as
script lines (:meth:`recorder_commands`).

Covered here:

* Declaration storage + ``ndm``/``ndf`` binding.
* ``recorder_commands`` resolves PG selectors to **fem ids** (the
  ``tag == fem_eid`` identity ModelData relies on) for both node-level
  and element-level records, in py and tcl flavors.
* The rendered py lines never carry the ``ops.wipe()`` preamble that
  the whole-model ``PyEmitter`` seeds — pasting that into a live deck
  would erase the user's model.
* ``attach_recorders`` forwards exactly the same ``recorder`` calls
  into a session object, and touches nothing but ``.recorder`` and the
  read-only domain queries (no ``wipe`` / ``model`` / ``node``).
* The tag-correspondence warning (K19, #1506).  The oracle is the
  fake live domain itself: built from the fem (same tags, same
  coordinates, same element nodes), ``attach_recorders`` stays silent;
  built with other tags, other coordinates or other element nodes, it
  warns and names each id that is not the fem entity.
* Fail-loud + empty-state edges.
"""
from __future__ import annotations

import warnings
from typing import Any, cast

import pytest

from apeGmsh.opensees import ModelData
from apeGmsh.opensees.recorder import RecorderDeclaration

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
    make_two_node_beam,
)
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


# ---------------------------------------------------------------------------
# Declaration storage
# ---------------------------------------------------------------------------

def test_recorders_stores_declaration_with_bound_ndm_ndf() -> None:
    fem = make_two_node_beam()
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    decl = md.recorders(nodes="displacement", pg="Top")

    assert isinstance(decl, RecorderDeclaration)
    assert decl.ndm == 3 and decl.ndf == 6
    assert md._recorder_decls == [decl]
    # "displacement" shorthand expands against the bound ndf.
    assert decl.records[0].category == "nodes"
    assert decl.records[0].components == (
        "displacement_x", "displacement_y", "displacement_z",
    )


def test_recorders_accumulate() -> None:
    fem = make_two_column_frame()
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    md.recorders(nodes="displacement", pg="Base")
    md.recorders(line_stations="bending_moment_y", pg="Cols")
    assert len(md._recorder_decls) == 2


# ---------------------------------------------------------------------------
# recorder_commands — node level
# ---------------------------------------------------------------------------

def test_recorder_commands_py_resolves_pg_to_fem_node_ids() -> None:
    fem = make_two_column_frame()  # Base PG -> nodes 1, 3
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    md.recorders(nodes="displacement", pg="Base", file_root="out")

    lines = md.recorder_commands(target="py")

    # Banner first, then exactly one recorder line for disp.
    assert lines[0].startswith("# recorders assume")
    body = lines[1:]
    assert len(body) == 1
    line = body[0]
    assert line.startswith("ops.recorder('Node'")
    # tag == fem_eid: the recorder targets the Base PG's fem node ids.
    assert "'-node', 1, 3," in line
    assert "'-dof', 1, 2, 3," in line
    assert line.rstrip().endswith("'disp')")


def test_recorder_commands_py_has_no_wipe_preamble() -> None:
    """Pasting recorder lines into a live deck must not erase the model.

    The whole-model ``PyEmitter`` seeds ``ops.wipe()`` + an import
    header into its buffer; ``recorder_commands`` must strip it.
    """
    fem = make_two_node_beam()
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    md.recorders(nodes="displacement", pg="Top")

    lines = md.recorder_commands(target="py")
    joined = "\n".join(lines)
    assert "ops.wipe()" not in joined
    assert "import openseespy" not in joined


def test_recorder_commands_tcl() -> None:
    fem = make_two_column_frame()  # Base PG -> nodes 1, 3
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    md.recorders(nodes="displacement", pg="Base", file_root="out")

    lines = md.recorder_commands(target="tcl")
    assert lines[0].startswith("# recorders assume")
    body = lines[1:]
    assert len(body) == 1
    assert body[0].startswith("recorder Node -file out/")
    assert "-node 1 3 " in body[0]
    assert body[0].rstrip().endswith("disp")


# ---------------------------------------------------------------------------
# recorder_commands — element level
# ---------------------------------------------------------------------------

def test_recorder_commands_line_stations_resolves_to_fem_element_ids() -> None:
    fem = make_two_column_frame()  # Cols PG -> elements 1, 2
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    md.recorders(line_stations="bending_moment_y", pg="Cols")

    py = md.recorder_commands(target="py")[1:]
    # One Element recorder for the response + one paired
    # integrationPoints recorder (gpx file) for line stations.
    assert any(
        l.startswith("ops.recorder('Element'") and "'-ele', 1, 2," in l
        and l.rstrip().endswith("'section', 'force')")
        for l in py
    )
    assert any("integrationPoints" in l for l in py)


# ---------------------------------------------------------------------------
# attach_recorders — live forwarding
# ---------------------------------------------------------------------------

class _RecordingOps:
    """A fake live domain: captures ``recorder`` calls, answers the
    read-only queries, and raises if anything else is touched.

    ``nodes`` maps tag -> coordinates and ``eles`` tag -> node tags, as
    the user's hand-written ``ops.node`` / ``ops.element`` calls left
    them; ``n_procs`` is what ``getNP`` reports.  The default is the
    one-column frame's own nodes (``build_simple_frame_fem``), so the
    tags equal the fem ids.
    """

    def __init__(
        self,
        nodes: "dict[int, tuple[float, ...]] | None" = None,
        eles: "dict[int, tuple[int, ...]] | None" = None,
        n_procs: int = 1,
    ) -> None:
        self.n_procs = n_procs
        self.nodes = nodes if nodes is not None else dict(_COLUMN_NODES)
        self.eles = eles if eles is not None else {}
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def recorder(self, kind: str, *args: object) -> None:
        self.calls.append((kind, args))

    def getNodeTags(self) -> list[int]:
        return list(self.nodes)

    def nodeCoord(self, tag: int) -> list[float]:
        return list(self.nodes[tag])

    def getNP(self) -> int:
        return self.n_procs

    def getEleTags(self) -> list[int]:
        return list(self.eles)

    def eleNodes(self, tag: int) -> list[int]:
        return list(self.eles[tag])

    def __getattr__(self, name: str) -> object:  # pragma: no cover - guard
        raise AssertionError(
            f"attach_recorders touched ops.{name} — it must only call "
            f"ops.recorder(...) and read-only queries (no wipe / model / "
            f"node)."
        )


#: ``build_simple_frame_fem()``: one column, element 1 joins nodes 1 and 2,
#: 1.0 long (so the coordinate tolerance is 1e-3).
_COLUMN_NODES: "dict[int, tuple[float, ...]]" = {
    1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 1.0),
}


def _column_nodes_md() -> ModelData:
    """A displacement recorder on the column's nodes 1 and 2.

    The live check walks ``fem.elements`` for the mesh size, so these
    tests bind a real :class:`FEMData` (the fem stub's element composite
    does not iterate).
    """
    md = ModelData(build_simple_frame_fem(), ndm=3, ndf=6)
    md.recorders(nodes="displacement", pg="Cols", file_root="out")
    return md


def test_attach_recorders_forwards_only_recorder_calls() -> None:
    md = _column_nodes_md()

    ops = _RecordingOps()
    md.attach_recorders(ops)

    assert len(ops.calls) == 1
    kind, args = ops.calls[0]
    assert kind == "Node"
    # Same fem-id resolution as recorder_commands.
    assert "-node" in args
    node_pos = args.index("-node")
    assert args[node_pos + 1] == 1 and args[node_pos + 2] == 2


def test_attach_recorders_matches_command_rendering() -> None:
    """The live forward and the py rendering issue the same recorder."""
    md = _column_nodes_md()

    ops = _RecordingOps()
    md.attach_recorders(ops)

    # Build the equivalent py line from the captured call and compare to
    # the rendered command (modulo quoting / call syntax).
    kind, args = ops.calls[0]
    assert kind == "Node"
    rendered = md.recorder_commands(target="py")[1]
    for token in ("-file", "-node", 1, 2, "-dof", "disp"):
        assert (str(token) in rendered)


# ---------------------------------------------------------------------------
# attach_recorders — tag correspondence (K19, #1506)
# ---------------------------------------------------------------------------

def _attach(md: ModelData, ops: _RecordingOps) -> list[str]:
    """Attach, returning the messages of every warning raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        md.attach_recorders(ops)
    return [str(w.message) for w in caught]


def test_attach_is_silent_when_live_tags_are_the_fem_ids() -> None:
    md = _column_nodes_md()
    ops = _RecordingOps()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        md.attach_recorders(ops)
    assert len(ops.calls) == 1


def test_attach_warns_when_a_targeted_node_is_absent() -> None:
    """The deck numbered its nodes from 101: fem nodes 1 and 2 do not exist."""
    md = _column_nodes_md()
    ops = _RecordingOps(nodes={101: (0.0, 0.0, 0.0), 102: (0.0, 0.0, 1.0)})
    [msg] = _attach(md, ops)
    assert "2 of them" in msg
    assert "node 1: absent from the live domain" in msg
    assert "node 2: absent from the live domain" in msg
    # Still attached: the warning does not drop the user's recorders.
    assert len(ops.calls) == 1


def test_attach_warns_when_a_node_tag_names_another_point() -> None:
    """Same tags, other numbering: deck node 1 sits where fem node 2 is."""
    md = _column_nodes_md()
    ops = _RecordingOps(nodes={1: (0.0, 0.0, 1.0), 2: (0.0, 0.0, 0.0)})
    [msg] = _attach(md, ops)
    assert "2 of them" in msg
    assert "node 1: live coordinates (0.0, 0.0, 1.0)" in msg


def _rounded(nodes: "dict[int, tuple[float, ...]]") -> "dict[int, tuple[float, ...]]":
    """The deck typed every x with a 1e-7 rounding error."""
    return {t: (x + 1e-7, *rest) for t, (x, *rest) in nodes.items()}


@pytest.mark.parametrize(
    "nodes, expected",
    [
        # Same numbering, rounded coordinates: 1e-7 is far below the
        # tolerance (1e-3 of the 1.0 column length), so silent.
        (_rounded(_COLUMN_NODES), None),
        # Renumbered and rounded: deck node 1 is the fem node 2 point
        # (one column length away), so it still warns.
        (_rounded({1: (0.0, 0.0, 1.0), 2: (0.0, 0.0, 0.0)}),
         "node 1: live coordinates"),
    ],
    ids=["rounded_silent", "renumbered_warns"],
)
def test_attach_coordinate_tolerance_follows_the_mesh_size(
    nodes: "dict[int, tuple[float, ...]]", expected: "str | None",
) -> None:
    md = _column_nodes_md()
    msgs = _attach(md, _RecordingOps(nodes=nodes))
    if expected is None:
        assert msgs == []
    else:
        [msg] = msgs
        assert expected in msg


def test_attach_in_a_parallel_run_skips_absent_but_checks_present() -> None:
    """OpenSeesMP: tag lists are rank-local.  Node 2 lives on another
    rank (absent here, not reported); node 1 is on this rank at the
    wrong point (reported)."""
    md = _column_nodes_md()
    on_rank = {1: (0.0, 0.0, 1.0)}

    [msg] = _attach(md, _RecordingOps(nodes=on_rank, n_procs=2))
    assert "1 of them" in msg and "node 1: live coordinates" in msg
    assert "node 2" not in msg

    # The same deck run sequentially reports node 2 as absent too.
    [msg] = _attach(md, _RecordingOps(nodes=on_rank, n_procs=1))
    assert "node 2: absent from the live domain" in msg


def test_attach_in_a_parallel_run_is_silent_for_off_rank_elements() -> None:
    md = _frame_md()
    ops = _RecordingOps(nodes={}, eles={}, n_procs=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        md.attach_recorders(ops)
    assert len(ops.calls) == 2


def _frame_md() -> ModelData:
    fem = build_simple_frame_fem()  # element 1 joins nodes 1 (z=0), 2 (z=1)
    md = ModelData(fem, ndm=3, ndf=6)
    md.recorders(line_stations="bending_moment_y", pg="Cols", file_root="out")
    return md


_FRAME_NODES: "dict[int, tuple[float, ...]]" = {
    1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 1.0), 3: (5.0, 0.0, 0.0),
}


def test_attach_is_silent_when_the_element_is_the_fem_element() -> None:
    md = _frame_md()
    ops = _RecordingOps(nodes=_FRAME_NODES, eles={1: (1, 2)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        md.attach_recorders(ops)
    # One section-force recorder plus its integrationPoints pair.
    assert len(ops.calls) == 2
    for kind, args in ops.calls:
        assert kind == "Element" and args[args.index("-ele") + 1] == 1


@pytest.mark.parametrize(
    "eles, expected",
    [
        ({7: (1, 2)}, "element 1: absent from the live domain"),
        ({1: (2, 3)}, "element 1: live nodes (2, 3) vs fem nodes (1, 2)"),
    ],
    ids=["absent", "other_nodes"],
)
def test_attach_warns_when_the_element_is_not_the_fem_element(
    eles: "dict[int, tuple[int, ...]]", expected: str,
) -> None:
    md = _frame_md()
    ops = _RecordingOps(nodes=_FRAME_NODES, eles=eles)
    [msg] = _attach(md, ops)
    assert expected in msg
    assert "1 of them" in msg  # one element, counted once over both calls
    assert len(ops.calls) == 2


def test_attach_checks_the_nodes_of_a_matching_element() -> None:
    """Element 1 joins tags (1, 2) as the fem says, but the deck's node 2
    is not at the fem node 2 point: the element is another element."""
    md = _frame_md()
    nodes: "dict[int, Any]" = {**_FRAME_NODES, 2: (0.0, 0.0, 3.0)}
    ops = _RecordingOps(nodes=nodes, eles={1: (1, 2)})
    [msg] = _attach(md, ops)
    assert "node 2: live coordinates (0.0, 0.0, 3.0)" in msg
    assert "element 1" not in msg


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def test_recorder_commands_empty_when_nothing_declared() -> None:
    fem = make_two_node_beam()
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    assert md.recorder_commands(target="py") == []
    assert md.recorder_commands(target="tcl") == []


def test_recorder_commands_rejects_bad_target() -> None:
    fem = make_two_node_beam()
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    md.recorders(nodes="displacement", pg="Top")
    with pytest.raises(ValueError, match="must be 'py' or 'tcl'"):
        md.recorder_commands(target="json")  # type: ignore[arg-type]
