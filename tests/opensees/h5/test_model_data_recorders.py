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
  into a session object, and touches nothing but ``.recorder`` (no
  ``wipe`` / ``model``).
* Fail-loud + empty-state edges.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import ModelData
from apeGmsh.opensees.recorder import RecorderDeclaration

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
    make_two_node_beam,
)


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
    """Captures ``recorder`` calls; raises if anything else is touched."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def recorder(self, kind: str, *args: object) -> None:
        self.calls.append((kind, args))

    def __getattr__(self, name: str) -> object:  # pragma: no cover - guard
        raise AssertionError(
            f"attach_recorders touched ops.{name} — it must only call "
            f"ops.recorder(...) (no wipe / model / node)."
        )


def test_attach_recorders_forwards_only_recorder_calls() -> None:
    fem = make_two_column_frame()  # Base PG -> nodes 1, 3
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    md.recorders(nodes="displacement", pg="Base", file_root="out")

    ops = _RecordingOps()
    md.attach_recorders(ops)

    assert len(ops.calls) == 1
    kind, args = ops.calls[0]
    assert kind == "Node"
    # Same fem-id resolution as recorder_commands.
    assert "-node" in args
    node_pos = args.index("-node")
    assert args[node_pos + 1] == 1 and args[node_pos + 2] == 3


def test_attach_recorders_matches_command_rendering() -> None:
    """The live forward and the py rendering issue the same recorder."""
    fem = make_two_node_beam()  # Top PG -> node 2
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    md.recorders(nodes="displacement", pg="Top", file_root="out")

    ops = _RecordingOps()
    md.attach_recorders(ops)

    # Build the equivalent py line from the captured call and compare to
    # the rendered command (modulo quoting / call syntax).
    kind, args = ops.calls[0]
    assert kind == "Node"
    rendered = md.recorder_commands(target="py")[1]
    for token in ("-file", "-node", 2, "-dof", "disp"):
        assert (str(token) in rendered)


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
