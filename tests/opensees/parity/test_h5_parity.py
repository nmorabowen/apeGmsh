"""H5Emitter parity — verify the H5 archive carries the same logical
content as the Tcl / Py decks for the same model.

Tcl / Py are ordered text streams; H5 is a name-keyed group tree with
no canonical line order.  A literal line-by-line diff is meaningless
for H5.  Instead this file compares **bags** of typed records:

  * For each major model category (materials, sections, transforms,
    beam_integration, time_series, patterns, recorders, elements),
    extract a (type_token, tag) key set from each emitter.
  * Assert the key sets agree across all three.

Catches the class of bugs the line-stream parity sweep
(``test_emitter_parity_sweep.py``) cannot:

  * H5 state machine accidentally drops every section_close ⇒
    sections persist correctly from buffered state but the H5
    side under-reports if section_open / section_close pairing
    breaks.
  * A recorder fan-out branch in H5Emitter emits to Tcl/Py but not
    to H5 (or vice-versa).
  * A type_token / tag mistranscription on the H5 side.

This is **count-and-key parity**, not full payload parity.  Param
payloads (Steel02 E / fy / b values) are byte-identical across the
three emitters because they all share the same typed-record source,
but reconciling the H5 connectivity-prefix drop and the Tcl/Py
float-repr conventions adds noise for little marginal coverage —
that's the territory of ``test_femdata_from_h5.py`` and the
end-to-end smoke in ``test_h5_end_to_end.py``.

The line-stream parity for Tcl vs Py stays the province of
``test_emitter_parity_sweep.py``; this file is the missing leg.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.section.fiber import FiberPoint

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
    make_two_node_beam,
)


# ---------------------------------------------------------------------
# Fixtures — three model shapes covering the major H5 zones
# ---------------------------------------------------------------------

def _build_elastic_frame() -> apeSees:
    """Materials-light: model + transform + elasticBeamColumn + fix +
    pattern + analysis chain.  Exercises transforms, elements,
    time_series, patterns, analysis."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="Top", forces=(0.0, 0.0, -1000.0, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-6, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def _build_fiber_frame() -> apeSees:
    """Materials-heavy: Steel02 + Fiber section + Lobatto rule +
    forceBeamColumn.  Exercises materials, complex sections,
    beam_integration."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(100e3, 0.0, 0.0, 0.0, 0.0, 0.0))
    return ops


def _build_recorder_frame() -> apeSees:
    """Recorder-heavy: elasticBeamColumn + Node recorder + MPCO with
    explicit selectors (auto-emits a region line per ADR 0024).
    Exercises recorders, regions."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.recorder.Node(
        file="disp.out",
        response="disp",
        nodes=(2,),
        dofs=(1, 2, 3),
    )
    ops.recorder.MPCO(
        file="run.mpco",
        nodal_responses=("displacement",),
        elem_responses=("section.force",),
        nodes=(1, 2),
        elements=(1,),
    )
    return ops


# ---------------------------------------------------------------------
# Key-set extractors — one per emitter target
# ---------------------------------------------------------------------

def _h5_keys(model_h5: Path) -> dict[str, set[tuple]]:
    """Extract bag-of-keys from an h5 file via the typed H5Model.

    Returns one set per category; each set carries (type_token, tag)
    or category-specific keys.  Counts: ``len(keys["materials"])``
    etc.
    """
    keys: dict[str, set[tuple]] = {
        "materials": set(),
        "sections": set(),
        "transforms": set(),
        "beam_integration": set(),
        "time_series": set(),
        "patterns": set(),
        "recorders": set(),
        "elements": set(),
        "regions": set(),
    }
    with h5_reader.open(str(model_h5)) as model:
        for rec in model.materials():
            keys["materials"].add((rec.type_token, int(rec.tag)))
        for rec in model.sections():
            keys["sections"].add((rec.type_token, int(rec.tag)))
        for rec in model.transforms():
            keys["transforms"].add((rec.type_token, int(rec.tag)))
        for rec in model.beam_integration():
            keys["beam_integration"].add((rec.type_token, int(rec.tag)))
        for rec in model.time_series():
            keys["time_series"].add((rec.type_token, int(rec.tag)))
        for rec in model.patterns():
            keys["patterns"].add((rec.type_token, int(rec.tag)))
        for rec in model.recorders():
            keys["recorders"].add((rec.kind, _recorder_tag_h5(rec)))
        # Elements — walk every type_token group.
        f = model.handle
        if "opensees" in f and "element_meta" in f["opensees"]:
            for tok in f["opensees/element_meta"]:
                arrays = model.element_meta_arrays(tok)
                ids = arrays.get("ids", [])
                for tag in ids:
                    keys["elements"].add((tok, int(tag)))
        # Regions — walk the group directly (no public accessor, per
        # the ADR-0024 read-side rollback).
        if "opensees" in f and "regions" in f["opensees"]:
            for name in f["opensees/regions"]:
                g = f[f"opensees/regions/{name}"]
                tag = int(g.attrs.get("tag", 0))
                keys["regions"].add(("region", tag))
    return keys


def _tcl_keys(tcl: TclEmitter) -> dict[str, set[tuple]]:
    """Extract bag-of-keys from a Tcl deck by line classification.

    Each ``uniaxialMaterial`` / ``section`` / ``geomTransf`` /
    ``beamIntegration`` / ``timeSeries`` / ``pattern`` / ``recorder``
    / ``element`` / ``region`` line yields one key.
    """
    keys: dict[str, set[tuple]] = {
        "materials": set(),
        "sections": set(),
        "transforms": set(),
        "beam_integration": set(),
        "time_series": set(),
        "patterns": set(),
        "recorders": set(),
        "elements": set(),
        "regions": set(),
    }
    for raw in tcl.lines():
        ln = raw.strip()
        if not ln or ln.startswith("#") or ln in ("{", "}"):
            continue
        # Strip trailing { from open-block lines.
        if ln.endswith("{"):
            ln = ln[:-1].strip()
        toks = ln.split()
        if len(toks) < 3:
            continue
        verb = toks[0]
        if verb in ("uniaxialMaterial", "nDMaterial"):
            keys["materials"].add((toks[1], int(toks[2])))
        elif verb == "section":
            keys["sections"].add((toks[1], int(toks[2])))
        elif verb == "geomTransf":
            keys["transforms"].add((toks[1], int(toks[2])))
        elif verb == "beamIntegration":
            keys["beam_integration"].add((toks[1], int(toks[2])))
        elif verb == "timeSeries":
            keys["time_series"].add((toks[1], int(toks[2])))
        elif verb == "pattern":
            keys["patterns"].add((toks[1], int(toks[2])))
        elif verb == "recorder":
            keys["recorders"].add((toks[1], _recorder_tag_tcl(toks)))
        elif verb == "element":
            keys["elements"].add((toks[1], int(toks[2])))
        elif verb == "region":
            keys["regions"].add(("region", int(toks[1])))
    return keys


def _py_keys(py: PyEmitter) -> dict[str, set[tuple]]:
    """Extract bag-of-keys from a Py deck by line classification.

    Same scheme as ``_tcl_keys`` but parses ``ops.X('Type', 1, ...)``
    style calls.
    """
    keys: dict[str, set[tuple]] = {
        "materials": set(),
        "sections": set(),
        "transforms": set(),
        "beam_integration": set(),
        "time_series": set(),
        "patterns": set(),
        "recorders": set(),
        "elements": set(),
        "regions": set(),
    }
    for raw in py.lines():
        ln = raw.strip()
        if not ln or ln.startswith("#") or ln.startswith("import "):
            continue
        if not ln.startswith("ops."):
            continue
        verb, args = _parse_ops_call(ln)
        if verb is None:
            continue
        if verb in ("uniaxialMaterial", "nDMaterial"):
            keys["materials"].add((args[0], int(args[1])))
        elif verb == "section":
            keys["sections"].add((args[0], int(args[1])))
        elif verb == "geomTransf":
            keys["transforms"].add((args[0], int(args[1])))
        elif verb == "beamIntegration":
            keys["beam_integration"].add((args[0], int(args[1])))
        elif verb == "timeSeries":
            keys["time_series"].add((args[0], int(args[1])))
        elif verb == "pattern":
            keys["patterns"].add((args[0], int(args[1])))
        elif verb == "recorder":
            keys["recorders"].add((args[0], _recorder_tag_py(args)))
        elif verb == "element":
            keys["elements"].add((args[0], int(args[1])))
        elif verb == "region":
            keys["regions"].add(("region", int(args[0])))
    return keys


def _recorder_tag_h5(rec) -> int:  # type: ignore[no-untyped-def]
    """Derive a stable recorder key from a :class:`RecorderRecord`.

    Mirrors the Tcl/Py extractors: look for ``-file <path>`` in the
    args tail; fall back to ``args[0]`` for MPCO-style recorders
    whose file argument is positional.
    """
    args = list(rec.args)
    if "-file" in args:
        i = args.index("-file") + 1
        if i < len(args):
            return _file_hash(str(args[i]))
    # MPCO: ``args = ('run.mpco', '-N', 'displacement', ...)``.
    if args:
        return _file_hash(str(args[0]))
    return 0


def _recorder_tag_tcl(toks: list[str]) -> int:
    """The Tcl recorder line has no explicit tag — the bridge's tag
    is the position in the recorder list, which both Tcl and H5
    agree on.  Use the file-stem (``-file disp.out`` ⇒ ``disp``)
    plus the position as a stable surrogate key.

    Implementation: count recorder lines seen so far (1-indexed).
    Stored on the function via a module-level counter is too
    fragile across tests; instead use the file argument (every
    recorder in our fixtures supplies one) as the key.
    """
    # Look for -file <path>; fall back to MPCO's positional file arg.
    if "-file" in toks:
        i = toks.index("-file") + 1
        if i < len(toks):
            return _file_hash(toks[i])
    # MPCO: ``recorder mpco run.mpco -N ...`` — toks[2] is the file.
    if len(toks) >= 3:
        return _file_hash(toks[2])
    return 0


def _recorder_tag_py(args: list[str]) -> int:
    """Same as ``_recorder_tag_tcl`` but for a parsed ops.recorder
    call: ``ops.recorder('Node', '-file', 'disp.out', ...)`` ⇒
    ``args = ['Node', '-file', 'disp.out', ...]``."""
    if "-file" in args:
        i = args.index("-file") + 1
        if i < len(args):
            return _file_hash(args[i])
    # MPCO: ``ops.recorder('mpco', 'run.mpco', ...)``.
    if len(args) >= 2:
        return _file_hash(args[1])
    return 0


def _file_hash(filename: str) -> int:
    """A surrogate recorder key: hash of the file stem.

    Recorder primitives don't carry an OpenSees tag in their command
    line (Tcl/Py just write ``recorder Node -file ...``); the
    parity test needs a stable comparable key across the three
    emitters.  H5 writes recorders with synthesized names
    (``Node_0``, ``Node_1``, ...) — see ``H5Emitter._write_recorders``.
    Hashing the file basename gives a content-derived key that all
    three emitters can compute identically.
    """
    base = Path(filename.strip("'\"")).stem
    return hash(base) & 0xFFFF_FFFF


def _parse_ops_call(ln: str) -> tuple[str | None, list[str]]:
    """Parse ``ops.X('a', 1, 'b')`` into ``("X", ["a", "1", "b"])``.

    Returns ``(None, [])`` for non-call lines.  String args have
    their enclosing quotes stripped.  Doesn't handle nested
    parens; OpenSees vocabulary has none.
    """
    m = re.match(r"^ops\.([A-Za-z_][A-Za-z_0-9]*)\((.*)\)\s*$", ln)
    if not m:
        return None, []
    verb = m.group(1)
    body = m.group(2)
    args = [_strip_quotes(a.strip()) for a in body.split(",") if a.strip()]
    return verb, args


def _strip_quotes(token: str) -> str:
    if (
        len(token) >= 2
        and token[0] == token[-1]
        and token[0] in ("'", '"')
    ):
        return token[1:-1]
    return token


# ---------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------

def _drive_through_h5_tcl_py(
    ops: apeSees, tmp_path: Path,
) -> tuple[dict[str, set[tuple]], dict[str, set[tuple]], dict[str, set[tuple]]]:
    """Drive a fresh build through H5Emitter, TclEmitter, PyEmitter;
    return the extracted key bags for each."""
    h5_path = tmp_path / "model.h5"
    ops.h5(str(h5_path))
    h5_bag = _h5_keys(h5_path)

    tcl = TclEmitter()
    ops.build().emit(tcl)
    tcl_bag = _tcl_keys(tcl)

    py = PyEmitter()
    ops.build().emit(py)
    py_bag = _py_keys(py)

    return h5_bag, tcl_bag, py_bag


def _assert_bags_agree(
    h5_bag: dict[str, set[tuple]],
    tcl_bag: dict[str, set[tuple]],
    py_bag: dict[str, set[tuple]],
    *,
    expected_nonempty: tuple[str, ...] = (),
) -> None:
    """Assert key bags match across all three emitters.

    ``expected_nonempty`` is the list of categories the test author
    asserts MUST be non-empty for this fixture (sanity: if the
    fixture stops emitting recorders by accident, all three bags
    will be empty and parity passes vacuously).
    """
    categories = sorted(h5_bag.keys() | tcl_bag.keys() | py_bag.keys())
    for cat in categories:
        h5_set = h5_bag.get(cat, set())
        tcl_set = tcl_bag.get(cat, set())
        py_set = py_bag.get(cat, set())
        assert h5_set == tcl_set == py_set, (
            f"parity divergence on category {cat!r}\n"
            f"  H5  ({len(h5_set)}): {sorted(h5_set)}\n"
            f"  Tcl ({len(tcl_set)}): {sorted(tcl_set)}\n"
            f"  Py  ({len(py_set)}): {sorted(py_set)}\n"
        )
    for cat in expected_nonempty:
        assert h5_bag.get(cat), (
            f"sanity: fixture was expected to emit at least one "
            f"{cat!r} record but H5 bag is empty"
        )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_h5_parity_elastic_frame(tmp_path: Path) -> None:
    """elasticBeamColumn + Linear transform + Plain pattern + Static
    analysis: every category key set matches across H5 / Tcl / Py."""
    h5_bag, tcl_bag, py_bag = _drive_through_h5_tcl_py(
        _build_elastic_frame(), tmp_path,
    )
    _assert_bags_agree(
        h5_bag, tcl_bag, py_bag,
        expected_nonempty=(
            "transforms", "elements", "time_series", "patterns",
        ),
    )


def test_h5_parity_fiber_frame(tmp_path: Path) -> None:
    """Steel02 + Fiber section + Lobatto + forceBeamColumn: materials,
    complex sections, and beam_integration all round-trip
    consistently across H5 / Tcl / Py."""
    h5_bag, tcl_bag, py_bag = _drive_through_h5_tcl_py(
        _build_fiber_frame(), tmp_path,
    )
    _assert_bags_agree(
        h5_bag, tcl_bag, py_bag,
        expected_nonempty=(
            "materials", "sections", "transforms",
            "beam_integration", "elements", "time_series", "patterns",
        ),
    )


def test_h5_parity_recorder_frame(tmp_path: Path) -> None:
    """Node recorder + MPCO with explicit selectors (auto-emits a
    region line per ADR 0024): both recorders and the region land
    consistently across H5 / Tcl / Py."""
    h5_bag, tcl_bag, py_bag = _drive_through_h5_tcl_py(
        _build_recorder_frame(), tmp_path,
    )
    _assert_bags_agree(
        h5_bag, tcl_bag, py_bag,
        expected_nonempty=("recorders", "regions"),
    )
