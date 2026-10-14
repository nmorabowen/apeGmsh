"""Rebar cage cells: a gmsh-built cell family beside the FEMStub grid.

The grid in :mod:`builder` is closed over the ``make_*`` factories of
``fem_stub`` (``test_fixture_roster_is_every_make_factory``), and its five
modes vary the analysis shape of a stub model.  A ``g.rebar`` cage needs a
real gmsh session, and what it pins is the bar geometry and its coupling,
not the analysis shape.  So it is its own family: one fixture
(``rebar_column_cage``) x two placements x two outputs, compared through
the grid's normalise and tolerance helpers.  ``README.md`` documents the
model.

Nothing here writes into the repository: each render builds in a
caller-supplied scratch directory.  Only ``tests.opensees.golden.regen``
writes goldens.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from apeGmsh import apeGmsh
from apeGmsh.opensees import OpenSeesTarget, apeSees
from apeGmsh.rebar import ACI318_seismic, BarCatalog, BarLayout, TieLayout

from tests.opensees.golden import builder

FIXTURE = "rebar_column_cage"
PLACEMENTS: tuple[str, ...] = ("conformal", "embedded")
OUTPUTS: tuple[str, ...] = ("tcl", "py")
REBAR_CELLS_DIR = builder.GOLDEN_DIR / "rebar_cells"

#: The cage material: ``emit_elements=True`` resolves each bar's
#: ``CorotTruss`` material by this name at emit time.
BAR_MATERIAL = "rebar"

#: A node line in either deck: ``node <id> ...`` or ``ops.node(<id>, ...)``.
_NODE_LINE = re.compile(r"^(?:node |ops\.node\()(\d+)\b")
#: An element line in either deck, with its type and first two nodes.
_ELEMENT_LINE = re.compile(
    r"^(?:element (?P<type>\w+) \d+ (?P<i>\d+) (?P<j>\d+)\b"
    r"|ops\.element\('(?P<ptype>\w+)', \d+, (?P<pi>\d+), (?P<pj>\d+)\b)"
)

#: The first section of every rebar golden: not an emitted file, so its
#: header is bracketed (see :func:`_coupling_section`).
COUPLING_HEADER = "=== <coupling> ==="

#: ADR 0112 D1's artifact directory variable (``tests/conftest.py`` sets it
#: for the suite).
_ARTIFACT_DIR_ENV = "APEGMSH_ARTIFACT_DIR"


def cell_id(placement: str, output: str) -> str:
    return f"{FIXTURE}/{placement}/{output}"


def all_cells() -> list[tuple[str, str]]:
    return [(p, o) for p in PLACEMENTS for o in OUTPUTS]


def golden_path(placement: str, output: str) -> Path:
    return REBAR_CELLS_DIR / FIXTURE / placement / f"{output}.golden"


@contextmanager
def _artifacts_in(path: Path) -> Iterator[None]:
    """Route the automatic ``model.h5`` writes (ADR 0112 D1) into ``path``.

    Both the session's ``end()`` and the bridge's first emit write
    ``<model_name>.h5`` at ``$APEGMSH_ARTIFACT_DIR``, else beside the
    ``__main__`` script, which under ``regen`` is this package.  The
    variable is restored on exit, so the suite's own value survives.
    """
    previous = os.environ.get(_ARTIFACT_DIR_ENV)
    os.environ[_ARTIFACT_DIR_ENV] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ARTIFACT_DIR_ENV, None)
        else:
            os.environ[_ARTIFACT_DIR_ENV] = previous


def build_model(
    placement: str,
) -> tuple[apeSees, frozenset[int], frozenset[int]]:
    """Mesh the column; return the bridge model, bar nodes and host nodes.

    A 0.4 x 0.4 x 1.5 box, a 3x3 cage of #6 bars with #3 ties @ 0.25
    under ``ACI318_seismic`` (the X4-e harness, #1848).  ``conformal``
    embeds the bars into a tet host; ``embedded`` adds cross-ties and
    couples through ``LadrunoEmbeddedRebar`` into a transfinite tet host.
    Both auto-emit a ``CorotTruss`` per bar cell.  The bar nodes are every
    node on a bar physical group of the placement; the host nodes are every
    node the 3-D elements of ``Col`` connect.
    """
    if placement not in PLACEMENTS:
        raise KeyError(f"unknown rebar placement {placement!r}")
    embedded = placement == "embedded"
    with apeGmsh(model_name=f"rebar_{placement}") as g:
        g.model.geometry.add_box(0, 0, 0, 0.4, 0.4, 1.5, label="Col")
        g.physical.add_volume("Col", name="Col")
        g.rebar.use_standard(ACI318_seismic(BarCatalog(unit_length=0.0254)))
        cage = g.rebar.column(
            section=("rect", 0.4, 0.4), height=1.5, cover=0.04,
            longitudinal=BarLayout(
                n_x=3, n_y=3, db="#6", material=BAR_MATERIAL,
            ),
            ties=TieLayout(db="#3", spacing=0.25, material=BAR_MATERIAL),
            crossties=embedded,
        )
        if embedded:
            placed = g.rebar.place(
                cage, into="Col", coupling="embedded", perfect=1.0e8,
                emit_elements=True,
            )
            # The bars mesh on their own, so the host can be structured:
            # its nodes are algebraic, not a Delaunay outcome.
            g.mesh.structured.set_transfinite_box(
                "Col", size=0.2, recombine=False,
            )
        else:
            placed = g.rebar.place(
                cage, into="Col", coupling="conformal", emit_elements=True,
            )
        g.mesh.sizing.set_global_size(0.15)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    bar_nodes = frozenset(
        int(n) for m in placed.members
        for n in fem.nodes.select(pg=m.pg).ids
    )
    host_nodes = frozenset(
        int(n) for n in
        fem.elements.select(pg="Col", dim=3).connectivity.ravel()
    )
    # Pinned target, as in the grid: the stamp reads ``backend stock``.
    ops = apeSees(fem, opensees=OpenSeesTarget(mode="stock"))
    ops.model(ndm=3, ndf=3)
    ops.uniaxialMaterial.Steel02(
        fy=420e6, E=200e9, b=0.01, name=BAR_MATERIAL,
    )
    return ops, bar_nodes, host_nodes


def _bar_nodes_only(text: str, bar_nodes: frozenset[int]) -> str:
    """Drop every node line whose node lies on no bar.

    The conformal deck declares no host element, so a host-only node is
    referenced by nothing in it: it is gmsh's 3-D Delaunay output, and
    that differs between hosts (gmsh 4.15.0 on Windows against 4.15.2 on
    Linux CI kept nodes 1-434 identical and diverged after them).
    """
    lines = text.split("\n")
    # Fail closed: the filter is sound only while every element in the
    # deck is a bar CorotTruss on bar nodes.
    for line in lines:
        if not line.startswith(("element ", "ops.element(")):
            continue
        e = _ELEMENT_LINE.match(line)
        if e is None or (e["type"] or e["ptype"]) != "CorotTruss" or not {
            int(e["i"] or e["pi"]), int(e["j"] or e["pj"]),
        } <= bar_nodes:
            raise AssertionError(
                f"conformal deck: {line!r} is not a CorotTruss on bar "
                "nodes, so dropping host-only node lines would hide it"
            )
    return "\n".join(
        line for line in lines
        if (m := _NODE_LINE.match(line)) is None or int(m[1]) in bar_nodes
    )


def render_deck(placement: str, output: str, scratch: Path) -> str:
    """Emit one rebar cell's deck under ``scratch``; return its canonical text."""
    if output not in OUTPUTS:
        raise KeyError(f"unknown rebar output {output!r}")
    deck_dir, artifacts = scratch / "deck", scratch / "artifacts"
    deck_dir.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    deck = str(deck_dir / f"model.{output}")
    with _artifacts_in(artifacts):
        ops, bar_nodes, host_nodes = build_model(placement)
        if output == "py":
            ops.py(deck)
        else:
            ops.tcl(deck)
    text = builder._collect(deck_dir)
    if placement == "conformal":
        text = _bar_nodes_only(text, bar_nodes)
    return _coupling_section(bar_nodes, host_nodes) + text


def _coupling_section(
    bar_nodes: frozenset[int], host_nodes: frozenset[int],
) -> str:
    """The coupling, as counts the deck itself cannot show.

    No host element is emitted, so neither deck says whether the bars share
    nodes with the host mesh.  A conformal placement embeds every bar node
    in the host connectivity; an embedded one shares none.  Both counts
    are topology, not Delaunay coordinates, so they hold on every host.
    """
    return (
        f"{COUPLING_HEADER}\n"
        f"bar_nodes {len(bar_nodes)}\n"
        f"bar_nodes_in_host_connectivity {len(bar_nodes & host_nodes)}\n"
    )


def manifest() -> dict[str, object]:
    return {
        "fixture": FIXTURE,
        "placements": list(PLACEMENTS),
        "outputs": list(OUTPUTS),
        "cells": {
            cell_id(p, o): {
                "status": "golden",
                "deck": golden_path(p, o).relative_to(
                    builder.GOLDEN_DIR).as_posix(),
            }
            for p, o in all_cells()
        },
    }
