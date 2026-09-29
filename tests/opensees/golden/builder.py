"""Golden emit corpus builder: fixture x mode x output -> canonical text.

The grid is the 7 ``make_*`` factories of
:mod:`tests.opensees.fixtures.fem_stub` x the 5 emit modes x the 3 outputs
(105 cells).  Each cell is either ``golden`` (it emits a deck, pinned byte
for byte after EOL normalisation, plus a canonical dump of the same model's
``ops.h5()`` archive) or ``n/a`` with a named reason.  ``README.md`` beside
this file documents the model recipe, the recorder set and the masks.

Nothing here writes into the repository: :func:`render_cell` emits into a
caller-supplied scratch directory and returns text.  Only
``tests.opensees.golden.regen`` writes goldens.
"""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.transform import Spherical

from tests.opensees.fixtures import fem_stub
from tests.opensees.fixtures.fem_stub import FEMStub

GOLDEN_DIR = Path(__file__).resolve().parent
REPO_ROOT = GOLDEN_DIR.parents[2]
CELLS_DIR = GOLDEN_DIR / "cells"
MANIFEST_PATH = GOLDEN_DIR / "MANIFEST.json"
REGEN_COMMAND = "python -m tests.opensees.golden.regen"

MODES: tuple[str, ...] = (
    "flat", "partitioned", "staged", "staged_partitioned", "per_rank",
)
OUTPUTS: tuple[str, ...] = ("tcl", "py", "recording")
PARTITIONED_MODES: frozenset[str] = frozenset(
    {"partitioned", "staged_partitioned", "per_rank"},
)

# Placeholder for the scratch directory a cell emits into.  No deck is
# expected to embed it; the substitution only keeps a golden byte-stable
# if one ever does (the diff then shows ``<OUT>`` rather than a temp path).
OUT_TOKEN = "<OUT>"


# ---------------------------------------------------------------------------
# Fixtures: the 7 factories, each with the model recipe it is driven by
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureSpec:
    """One ``make_*`` factory plus the model recipe the corpus drives it with.

    ``partitions`` is the honest 2-rank split used to enter the
    partitioned modes when the factory returns an unpartitioned stub
    (``None`` for a factory that is already partitioned, or for one that
    cannot be split honestly: then ``unpartitionable`` names why).
    """

    factory: Callable[[], FEMStub]
    ndm: int
    ndf: int
    elem_pg: str
    fixed_pg: str
    loaded_pg: str
    partitions: "list[tuple[int, list[int], list[int]]] | None" = None
    unpartitionable: str | None = None
    arch: bool = False
    truss: bool = False
    labels_and_selection: bool = False


_FRAME_SPLIT = [(0, [1, 2], [1]), (1, [3, 4], [2])]

FIXTURES: dict[str, FixtureSpec] = {
    "two_node_beam": FixtureSpec(
        factory=fem_stub.make_two_node_beam,
        ndm=3, ndf=6, elem_pg="Cols", fixed_pg="Base", loaded_pg="Top",
        unpartitionable=(
            "one element: any 2-rank split leaves a rank that owns no "
            "element, which no mesh partitioner produces"
        ),
    ),
    "two_column_frame": FixtureSpec(
        factory=fem_stub.make_two_column_frame,
        ndm=3, ndf=6, elem_pg="Cols", fixed_pg="Base", loaded_pg="Top",
        partitions=_FRAME_SPLIT,
    ),
    "two_column_frame_with_labels_and_selection": FixtureSpec(
        factory=fem_stub.make_two_column_frame_with_labels_and_selection,
        ndm=3, ndf=6, elem_pg="Cols", fixed_pg="Base", loaded_pg="Top",
        partitions=_FRAME_SPLIT,
        labels_and_selection=True,
    ),
    "two_module_frame": FixtureSpec(
        factory=fem_stub.make_two_module_frame,
        ndm=3, ndf=6, elem_pg="Cols", fixed_pg="Base", loaded_pg="Top",
        # One rank per compose module (ADR 0038 "Rank model").
        partitions=_FRAME_SPLIT,
    ),
    "two_column_frame_partitioned": FixtureSpec(
        factory=fem_stub.make_two_column_frame_partitioned,
        ndm=3, ndf=6, elem_pg="Cols", fixed_pg="Base", loaded_pg="Top",
    ),
    "axial_chain_partitioned": FixtureSpec(
        factory=fem_stub.make_axial_chain_partitioned,
        ndm=2, ndf=2, elem_pg="Chain", fixed_pg="Base", loaded_pg="Masses",
        truss=True,
    ),
    "arch_with_orientation_fan_out": FixtureSpec(
        factory=fem_stub.make_arch_with_orientation_fan_out,
        ndm=3, ndf=6, elem_pg="Arch", fixed_pg="Springing",
        loaded_pg="Crown",
        # Shared node 3 on the rank boundary.
        partitions=[(0, [1, 2, 3], [1, 2]), (1, [3, 4], [3])],
        arch=True,
    ),
}


def _factory_is_partitioned(name: str) -> bool:
    return len(FIXTURES[name].factory().partitions) > 1


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


def applicability(fixture: str, mode: str, output: str) -> str | None:
    """``None`` when the cell is golden, else the ``n/a`` reason."""
    spec = FIXTURES[fixture]
    if mode not in MODES or output not in OUTPUTS:
        raise KeyError(f"unknown cell {fixture}/{mode}/{output}")
    if mode in PARTITIONED_MODES and spec.unpartitionable and not (
        _factory_is_partitioned(fixture)
    ):
        return spec.unpartitionable
    if mode == "per_rank" and output == "py":
        return (
            "apeSees.py has no per_rank layout; the per-rank fragment "
            "split (ADR 0061) is Tcl-only"
        )
    if mode in ("flat", "staged") and output == "py" and (
        _factory_is_partitioned(fixture)
    ):
        return (
            "apeSees.py has no flat= switch, so a partition-carrying "
            "fixture always takes the per-rank fan-out on the py route; "
            "the py partitioned cells pin that deck"
        )
    return None


def cell_id(fixture: str, mode: str, output: str) -> str:
    return f"{fixture}/{mode}/{output}"


def all_cells() -> list[tuple[str, str, str]]:
    return [
        (f, m, o) for f in FIXTURES for m in MODES for o in OUTPUTS
    ]


def golden_paths(fixture: str, mode: str, output: str) -> tuple[Path, Path]:
    """``(deck golden, h5 dump golden)`` for one cell."""
    base = CELLS_DIR / fixture / mode
    return base / f"{output}.golden", base / f"{output}.h5dump"


# ---------------------------------------------------------------------------
# Model recipe
# ---------------------------------------------------------------------------


def _fem_for(fixture: str, mode: str) -> FEMStub:
    spec = FIXTURES[fixture]
    fem = spec.factory()
    if mode in PARTITIONED_MODES and len(fem.partitions) <= 1:
        if spec.partitions is None:
            raise AssertionError(
                f"{fixture}: a partitioned mode reached a fixture with no "
                "partition recipe (applicability() should have said n/a)"
            )
        fem.set_partitions(spec.partitions)
    return fem


def _chain(ops: apeSees, parallel: bool) -> dict[str, object]:
    """A complete static chain; parallel numberer + system when partitioned.

    The partitioned modes declare ``ParallelRCM`` / ``Mumps`` so the pinned
    deck is the one OpenSeesMP can run (a serial pair there only warns
    ``OpenSeesAutoEmitWarning`` and keeps the unrunnable choice).
    """
    return {
        "test": ops.test.NormDispIncr(tol=1e-6, max_iter=10),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=0.5),
        "constraints": ops.constraints.Plain(),
        "numberer": (
            ops.numberer.ParallelRCM() if parallel else ops.numberer.RCM()
        ),
        "system": ops.system.Mumps() if parallel else ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def _load(spec: FixtureSpec, magnitude: float) -> tuple[float, ...]:
    forces = [0.0] * spec.ndf
    forces[0] = magnitude
    return tuple(forces)


def _declare_elements(ops: apeSees, spec: FixtureSpec) -> None:
    if spec.truss:
        ux = ops.uniaxialMaterial.ElasticMaterial(E=100.0)
        ops.element.Truss(pg=spec.elem_pg, A=1.0, material=ux)
        return
    if spec.arch:
        transf = ops.geomTransf.Linear(
            orientation=Spherical(origin=(0.0, 0.0, 0.0)),
        )
    else:
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg=spec.elem_pg, transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )


def _declare_fixes(ops: apeSees, spec: FixtureSpec) -> None:
    ops.fix(pg=spec.fixed_pg, dofs=(1,) * spec.ndf)
    if spec.truss:
        # Planar truss chain: restrain every transverse DOF, leaving the
        # axial DOFs free (the fixture's documented drive).
        ops.fix(pg=spec.loaded_pg, dofs=(0, 1))


def _declare_recorders(ops: apeSees, spec: FixtureSpec) -> None:
    """The smallest set that pins node, element and MPCO-region emission."""
    ops.recorder.Node(
        file="out/node_disp.out", response="disp",
        pg=spec.loaded_pg, dofs=(1, 2),
    )
    ops.recorder.Element(
        file="out/ele_force.out", response=("globalForce",),
        pg=spec.elem_pg,
    )
    ops.recorder.MPCO(
        file="out/run.mpco",
        nodal_responses=("displacement",),
        elem_responses=("force",),
        nodes_pg=spec.loaded_pg,
        elements_pg=spec.elem_pg,
    )
    if spec.labels_and_selection:
        # The only fixture carrying labels and a mesh_selection store:
        # pin the label= and selection= resolvers too.
        ops.recorder.declare(
            nodes="displacement_x", label="east_column", name="by_label",
        )
        ops.recorder.declare(
            nodes="displacement_x", selection="upper_band",
            name="by_selection",
        )


def build_model(fixture: str, mode: str, output: str) -> apeSees:
    """The ``apeSees`` model one cell emits."""
    spec = FIXTURES[fixture]
    fem = _fem_for(fixture, mode)
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=spec.ndm, ndf=spec.ndf)
    _declare_elements(ops, spec)
    _declare_fixes(ops, spec)
    if output == "recording":
        _declare_recorders(ops, spec)

    parallel = mode in PARTITIONED_MODES
    if mode in ("flat", "partitioned"):
        ts = ops.timeSeries.Linear()
        with ops.pattern.Plain(series=ts) as p:
            p.load(pg=spec.loaded_pg, forces=_load(spec, 10.0))
        _chain(ops, parallel)  # global chain: each primitive self-registers
        return ops

    # staged / staged_partitioned / per_rank: two stage-scoped patterns.
    with ops.stage(name="gravity") as s:
        ts1 = ops.timeSeries.Linear()
        with s.pattern(series=ts1) as p:
            p.load(pg=spec.loaded_pg, forces=_load(spec, 10.0))
        s.analysis(**_chain(ops, parallel))
        s.run(n_increments=2)
    with ops.stage(name="push") as s:
        ts2 = ops.timeSeries.Linear()
        with s.pattern(series=ts2) as p:
            p.load(pg=spec.loaded_pg, forces=_load(spec, -5.0))
        s.analysis(**_chain(ops, parallel))
        s.run(n_increments=1)
    return ops


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


_DUMP_MODULE: ModuleType | None = None


def _load_dump_module() -> ModuleType:
    """Load ``scripts/h5_canonical_dump.py`` without touching ``sys.modules``."""
    global _DUMP_MODULE
    if _DUMP_MODULE is None:
        path = REPO_ROOT / "scripts" / "h5_canonical_dump.py"
        mod_spec = importlib.util.spec_from_file_location(
            "_golden_h5_canonical_dump", path,
        )
        assert mod_spec is not None and mod_spec.loader is not None
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        _DUMP_MODULE = module
    return _DUMP_MODULE


# ---------------------------------------------------------------------------
# Deck comparison: exact text, float literals within a last-ulp tolerance
# ---------------------------------------------------------------------------

#: A float literal: digits with a ``.`` and/or an exponent, not glued to an
#: identifier (``rank0_0.tcl`` stays text).  Integers never match, so they
#: stay in the exactly-compared text.
FLOAT_TOKEN = re.compile(
    r"(?<![\w.])([-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?"
    r"|[-+]?\d+[eE][-+]?\d+)(?![\w.])"
)
#: Libm transcendentals (the orientation vecxz: sin/cos/sqrt) differ by the
#: last ulp across platforms (#1258: Linux dev vs CI runner).  1e-12 is
#: ~4500 ulp at 1.0 yet 1e6 below any modelling value a golden pins.
FLOAT_REL_TOL = 1e-12
FLOAT_ABS_TOL = 1e-15


def _floats_close(a: str, b: str) -> bool:
    x, y = float(a), float(b)
    return abs(x - y) <= max(FLOAT_REL_TOL * max(abs(x), abs(y)), FLOAT_ABS_TOL)


def lines_match(want: str, got: str) -> bool:
    """One line: non-float text exact, float literals within tolerance."""
    if want == got:
        return True
    w, g = FLOAT_TOKEN.split(want), FLOAT_TOKEN.split(got)
    if len(w) != len(g):
        return False
    # re.split with one group alternates text (even) / float (odd).
    return all(
        (a == b) if i % 2 == 0 else _floats_close(a, b)
        for i, (a, b) in enumerate(zip(w, g))
    )


def first_deck_mismatch(want: str, got: str) -> int | None:
    """0-based index of the first line that differs beyond tolerance."""
    wl, gl = want.split("\n"), got.split("\n")
    for i, (a, b) in enumerate(zip(wl, gl)):
        if not lines_match(a, b):
            return i
    if len(wl) != len(gl):
        return min(len(wl), len(gl))
    return None


def _normalise(text: str, out_dir: Path) -> str:
    text = text.replace("\r\n", "\n")
    for form in {str(out_dir), out_dir.as_posix(), str(out_dir.resolve())}:
        text = text.replace(form, OUT_TOKEN)
    return text


def _collect(out_dir: Path) -> str:
    """Concatenate every emitted file, sorted by relative POSIX path."""
    files = sorted(
        p for p in out_dir.rglob("*") if p.is_file()
    )
    if not files:
        raise AssertionError(f"cell emitted no files into {out_dir}")
    chunks = []
    for p in sorted(files, key=lambda q: q.relative_to(out_dir).as_posix()):
        rel = p.relative_to(out_dir).as_posix()
        body = p.read_bytes().decode("utf-8")
        chunks.append(f"=== {rel} ===\n{_normalise(body, out_dir)}")
    return "".join(chunks)


def render_deck(fixture: str, mode: str, output: str, out_dir: Path) -> str:
    """Emit one cell's deck into ``out_dir`` and return its canonical text."""
    reason = applicability(fixture, mode, output)
    if reason is not None:
        raise AssertionError(f"{cell_id(fixture, mode, output)} is n/a: {reason}")
    out_dir.mkdir(parents=True, exist_ok=False)
    ops = build_model(fixture, mode, output)
    ext = "py" if output == "py" else "tcl"
    deck = str(out_dir / f"model.{ext}")
    if output == "py":
        ops.py(deck)
    elif mode == "per_rank":
        ops.tcl(deck, per_rank=True)
    elif mode in ("flat", "staged"):
        ops.tcl(deck, flat=True)
    else:
        ops.tcl(deck)
    return _collect(out_dir)


def render_h5_dump(
    fixture: str, mode: str, output: str, out_dir: Path,
) -> str:
    """Write the cell model's ``ops.h5()`` and return its canonical dump."""
    out_dir.mkdir(parents=True, exist_ok=False)
    path = out_dir / "model.h5"
    build_model(fixture, mode, output).h5(str(path))
    return str(_load_dump_module().dump(path))


def manifest() -> dict[str, object]:
    """The MANIFEST content derived from the grid (golden or n/a per cell)."""
    cells: dict[str, object] = {}
    for f, m, o in all_cells():
        reason = applicability(f, m, o)
        if reason is None:
            deck, dump = golden_paths(f, m, o)
            cells[cell_id(f, m, o)] = {
                "status": "golden",
                "deck": deck.relative_to(GOLDEN_DIR).as_posix(),
                "h5dump": dump.relative_to(GOLDEN_DIR).as_posix(),
            }
        else:
            cells[cell_id(f, m, o)] = {"status": f"n/a: {reason}"}
    return {
        "fixtures": list(FIXTURES),
        "modes": list(MODES),
        "outputs": list(OUTPUTS),
        "regen": REGEN_COMMAND,
        "cells": cells,
    }


def manifest_text() -> str:
    return json.dumps(manifest(), indent=2, sort_keys=False) + "\n"
