"""ADR 0099 — what a ``model BasicBuilder`` re-issue actually destroys.

The builder-ndf bracket (``open_builder_ndf_bracket``) re-issues ``model
BasicBuilder`` around each gated element block.  That re-issue is not a
state tweak: ``specifyModelBuilder`` (fork
``SRC/modelbuilder/tcl/myCommands.cpp:85``) deletes the old builder, and
``~TclModelBuilder()`` (fork ``SRC/modelbuilder/tcl/TclModelBuilder.cpp:681``)
purges a set of *process-global* registries on the way out.  The ``Domain``
is a file-static that is reused, so nodes and elements survive — which is
why the "re-issuing model does not wipe the domain" premise read as safe.

This script is the executable form of the ADR's survival table.  It writes
one probe deck per declaration kind, runs each under the real OpenSees
binary, and reports whether the declaration survived.

Usage::

    python repros/repro4_builder_scoped_wipe.py [path\\to\\OpenSees.exe]
    python repros/repro4_builder_scoped_wipe.py --h5 [path\\to\\OpenSees.exe]
    python repros/repro4_builder_scoped_wipe.py --partitioned [path\\to\\OpenSees.exe]
    python repros/repro4_builder_scoped_wipe.py --split [path\\to\\OpenSees.exe]

The default arm is the survival table below.  ``--h5``, ``--partitioned``
and ``--split`` are the ORDERING arms: each builds a real mixed-ndf
apeGmsh model carrying a gated ``quad`` plus all four scoped kinds, emits
it down one path, and runs the result on the binary.  ``--h5`` covers the
archive replay (ADR 0099 S4a); ``--partitioned`` covers the per-rank
hoist (S5), running the single deck once per rank; ``--split`` covers the
file-per-module hoist (S6), running the fragment driver and matching its
displacements against the flat reference deck.  All need apeGmsh
importable; the default arm needs only the binary.

Measured on Ladruno build ``25a0647f``:

    node              SURVIVES
    uniaxialMaterial  SURVIVES
    nDMaterial        SURVIVES
    section           SURVIVES
    timeSeries        DESTROYED
    geomTransf        DESTROYED
    beamIntegration   DESTROYED
    damping           DESTROYED   <- and the command only WARNS

``damping`` is the dangerous row: ``region ... -damp`` does not return an
error, so a deck combining ``ops.damping.*`` with a gated element emits,
runs, converges, and reports an undamped answer.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_BIN = r"C:\Program Files\Ladruno\OpenSees\bin\OpenSees.exe"

PREAMBLE = """
model BasicBuilder -ndm 2 -ndf 3
node 1 0.0 0.0 ; node 2 1.0 0.0 ; node 3 2.0 0.0
fix 1 1 1 1
uniaxialMaterial Elastic 10 200.0
nDMaterial ElasticIsotropic 20 1000.0 0.3 0.0
section Elastic 30 200.0 1.0 1.0
geomTransf Linear 40
beamIntegration Lobatto 50 30 3
timeSeries Linear 60
damping Uniform 70 0.05 0.2 10.0
"""

# The bracket, exactly as apeGmsh emits it: open + envelope restore.
BRACKET = """
model BasicBuilder -ndm 2 -ndf 2
model BasicBuilder -ndm 2 -ndf 3
"""

#: The four kinds ``~TclModelBuilder()`` purges — the destroyed rows of
#: the table above, and what the ordering arms below check line
#: positions against.
SCOPED_KEYS = ("timeSeries", "geomTransf", "beamIntegration", "damping")

#: name -> (probe that re-uses the tag, marker proving it was NOT found)
PROBES: dict[str, tuple[str, str]] = {
    "node":             ("element truss 99 1 2 1.0 10",   "node 1 does not exist"),
    "uniaxialMaterial": ("element truss 98 1 2 1.0 10",   "none found with tag: 10"),
    "nDMaterial":       ("puts [nDMaterial ElasticIsotropic 21 1.0 0.1 0.0]", "__never__"),
    "section":          ("beamIntegration Lobatto 51 30 3", "none found with tag: 30"),
    "timeSeries":       ("pattern Plain 90 60 { load 3 1.0 0.0 0.0 }", "none found with tag: 60"),
    "geomTransf":       ("section Elastic 31 200.0 1.0 1.0\nbeamIntegration Lobatto 52 31 3\n"
                         "element dispBeamColumn 97 2 3 40 52", "none found with tag: 40"),
    "beamIntegration":  ("geomTransf Linear 41\nelement dispBeamColumn 96 2 3 41 50",
                         "none found with tag: 50"),
    "damping":          ("region 5 -node 1 2 -damp 70",    "none found with tag: 70"),
}


def run(binary: str, deck: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "probe.tcl"
        path.write_text(deck, encoding="utf-8")
        proc = subprocess.run(
            [binary, str(path)], capture_output=True, text=True, timeout=120,
        )
        return proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# The model side: a mixed-ndf deck carrying a gated quad + all four kinds
# ---------------------------------------------------------------------------

#: Appended to every deck this script runs. ``OpenSees.exe`` exits 0 even
#: on a Tcl error (a `catch`-less script just stops), so the ONLY honest
#: liveness signal is a marker printed from the last line.
MARKER = "APEGMSH_DECK_OK"

#: The bracket + envelope restore, as ``open_builder_ndf_bracket`` emits it.
BRACKET_OPEN = "model BasicBuilder -ndm 2 -ndf 2"


def _fem(partitioned: bool, composed: bool = False):
    """Two quads on ndf-2 nodes + two beams on ndf-3 nodes.

    The quad parser hard-gates on builder ``ndf == 2`` while the beams
    force an ``ndf=3`` envelope, so the quad block brackets — and the
    bracket is what purges the four registries.  When ``partitioned``,
    the quads land on rank 0 and the beams on rank 1: only ONE rank
    executes the bracket, which is why the partitioned failure is
    rank-local and therefore non-deterministic in ``np``.  When
    ``composed``, the same bodies are tagged as compose modules
    (``Soil`` / ``Frame``) so the model can emit ``split='parts'``.
    """
    import numpy as np
    from apeGmsh.mesh._element_types import ElementGroup, make_type_info
    from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import (
        ElementComposite, FEMData, MeshInfo, NodeComposite,
    )

    ids = np.arange(1, 10, dtype=np.int64)
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
         [0.0, 5.0, 0.0], [1.0, 5.0, 0.0], [2.0, 5.0, 0.0]],
        dtype=np.float64,
    )
    quad = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=2)
    line = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2)

    def sel(v):
        return np.array(v, dtype=np.int64)

    def xyz(v):
        return coords[[i - 1 for i in v]]

    pg = {
        (2, 201): {"name": "Rock", "node_ids": sel([1, 2, 3, 4, 5, 6]),
                   "node_coords": xyz([1, 2, 3, 4, 5, 6]),
                   "element_ids": sel([1, 2])},
        (1, 202): {"name": "Liner", "node_ids": sel([7, 8, 9]),
                   "node_coords": xyz([7, 8, 9]), "element_ids": sel([3, 4])},
        (0, 203): {"name": "Base", "node_ids": sel([1, 2, 3]),
                   "node_coords": xyz([1, 2, 3]), "element_ids": sel([])},
        (0, 204): {"name": "Anchor", "node_ids": sel([7]),
                   "node_coords": xyz([7]), "element_ids": sel([])},
        (0, 205): {"name": "Mid", "node_ids": sel([5]),
                   "node_coords": xyz([5]), "element_ids": sel([])},
        (0, 206): {"name": "Tip", "node_ids": sel([9]),
                   "node_coords": xyz([9]), "element_ids": sel([])},
    }
    node_parts = elem_parts = None
    if partitioned:
        node_parts = {1: {"node_ids": sel([1, 2, 3, 4, 5, 6])},
                      2: {"node_ids": sel([7, 8, 9])}}
        elem_parts = {1: {"element_ids": sel([1, 2])},
                      2: {"element_ids": sel([3, 4])}}
    node_modules = elem_modules = None
    if composed:
        node_modules = np.array(
            ["Soil"] * 6 + ["Frame"] * 3, dtype=object)
        elem_modules = {
            3: np.array(["Soil", "Soil"], dtype=object),
            1: np.array(["Frame", "Frame"], dtype=object),
        }
    nodes = NodeComposite(
        node_ids=ids, node_coords=coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        partitions=node_parts,
        module_label=node_modules,
    )
    elements = ElementComposite(
        groups={
            3: ElementGroup(
                element_type=quad, ids=sel([1, 2]),
                connectivity=np.array([[1, 2, 5, 4], [2, 3, 6, 5]],
                                      dtype=np.int64)),
            1: ElementGroup(
                element_type=line, ids=sel([3, 4]),
                connectivity=np.array([[7, 8], [8, 9]], dtype=np.int64)),
        },
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        partitions=elem_parts,
        module_label=elem_modules,
    )
    return FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=9, n_elems=4, bandwidth=4,
                      types=[quad, line]),
    )


def _bridge(partitioned: bool, composed: bool = False):
    """The gated quad, all four builder-scoped kinds, and a solvable chain.

    Damping rides the ungated beam, not the quad: ``quad(damp=...)`` is
    refused up front by INV-3 — its own bracket destroys the declaration
    it references, so no ordering can save it.
    """
    from apeGmsh.opensees.apesees import apeSees

    ops = apeSees(_fem(partitioned, composed), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2000.0, nu=0.25, rho=0.0)
    sec = ops.section.Elastic(E=2.0e5, A=0.01, Iz=1.0e-4)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    transf = ops.geomTransf.Linear()
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    damp = ops.damping.uniform(ratio=0.05, freq_lower=0.2, freq_upper=10.0)
    ops.element.dispBeamColumn(
        pg="Liner", transf=transf, integration=integ, damp=damp)
    ops.fix(pg="Base", dofs=(1, 1))
    ops.fix(pg="Anchor", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.load(pg="Mid", forces=(0.0, -10.0))
        pat.load(pg="Tip", forces=(0.0, -1.0, 0.0))
    ops.test.NormDispIncr(tol=1.0e-10, max_iter=25)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.constraints.Transformation()
    ops.analysis.Static()
    return ops


def _inv1_offenders(stream: "list[str]") -> "list[str]":
    """Builder-scoped declarations preceding the stream's LAST model line."""
    models = [i for i, ln in enumerate(stream)
              if ln.strip().startswith("model ")]
    if not models:
        return []
    return [ln.strip() for ln in stream[:models[-1]]
            if any(ln.strip().startswith(k) for k in SCOPED_KEYS)]


def _rank_streams(text: str) -> "dict[int, list[str]]":
    """Split a partitioned deck into the line stream each rank executes.

    Under OpenSeesMP every rank runs the WHOLE file, taking the global
    lines plus the bodies of its own ``if {[getPID] == K} { ... }`` guards.
    Reading the file as one text says "fine" for every rank that owns no
    gated element — which is exactly how this defect stayed hidden.
    """
    import re
    open_re = re.compile(r"^if \{\[getPID\] == (\d+)\} \{$")
    lines = text.splitlines()
    ranks = {int(m.group(1)) for ln in lines if (m := open_re.match(ln))}
    streams = {r: [] for r in ranks}
    cur = None
    for ln in lines:
        m = open_re.match(ln)
        if m is not None:
            cur = int(m.group(1))
            continue
        if cur is not None and ln == "}":
            cur = None
            continue
        for r in ranks:
            if cur is None or cur == r:
                streams[r].append(ln)
    return streams


def _run_deck(binary: str, deck: str, *, rank: int | None = None) -> str:
    """Run ``deck`` and return its combined output.

    ``rank`` simulates one OpenSeesMP rank under the SERIAL binary by
    pre-defining ``proc getPID`` before sourcing: the deck's own shim is
    guarded on ``[info commands getPID] == ""``, so it does not override.
    """
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "deck.tcl"
        path.write_text(deck + f'\nputs "{MARKER}"\n', encoding="utf-8")
        if rank is None:
            target = path
        else:
            target = Path(d) / "driver.tcl"
            target.write_text(
                f"proc getPID {{}} {{ return {rank} }}\n"
                f"source {{{path}}}\n",
                encoding="utf-8",
            )
        proc = subprocess.run(
            [binary, str(target)], capture_output=True, text=True, timeout=300,
        )
        return proc.stdout + proc.stderr


def _report(label: str, out: str, offenders: "list[str]") -> bool:
    ok = MARKER in out and not offenders
    print(f"  {label:<22} {'OK' if ok else 'FAILED'}")
    if offenders:
        print(f"      INV-1 offenders: {offenders}")
    if MARKER not in out:
        tail = [ln for ln in out.splitlines() if ln.strip()][-4:]
        for ln in tail:
            print(f"      | {ln}")
    return ok


def h5_arm(binary: str) -> int:
    """ADR 0099 S4a, end to end: archive the model, replay it, RUN it.

    The unit tests read the replayed deck's line order; this closes the
    loop the tests cannot — a deck whose ordering satisfies INV-1 on
    paper still has to build every object on the real binary.
    """
    from apeGmsh.opensees import OpenSeesModel

    print("--h5: build -> h5 -> from_h5 -> build('tcl') -> run")
    ok = True
    with tempfile.TemporaryDirectory() as d:
        archive = Path(d) / "model.h5"
        ops = _bridge(partitioned=False)
        ops.h5(str(archive))
        replayed = OpenSeesModel.from_h5(str(archive)).build("tcl")
        offenders = _inv1_offenders(replayed.splitlines())
        ok &= _report("replayed deck", _run_deck(binary, replayed), offenders)

        forward = Path(d) / "forward.tcl"
        ops.tcl(str(forward), progress=False, analyze_steps=1)
        text = forward.read_text(encoding="utf-8")
        ok &= _report(
            "forward (flat) deck", _run_deck(binary, text),
            _inv1_offenders(text.splitlines()),
        )
        if BRACKET_OPEN not in replayed:
            print("      WARNING: no bracket in the replayed deck — the "
                  "probe is vacuous")
            ok = False
    return 0 if ok else 1


def partitioned_arm(binary: str) -> int:
    """ADR 0099 S5: the same model on the partitioned path, per rank.

    Only rank 0 owns a gated element, so only rank 0 ever executes the
    bracket.  Pre-S5 that rank died at ``pattern Plain`` (or, with no
    pattern, ran to completion silently undamped) while rank 1 was
    already clean — the same model passing on some rank counts and
    failing on others.  Both ranks must now reach the marker.
    """
    print("--partitioned: one deck, one guard per rank, run rank by rank")
    ok = True
    with tempfile.TemporaryDirectory() as d:
        deck = Path(d) / "part.tcl"
        _bridge(partitioned=True).tcl(
            str(deck), progress=False, analyze_steps=1)
        text = deck.read_text(encoding="utf-8")
        streams = _rank_streams(text)
        gated = [r for r, s in streams.items()
                 if any(ln.strip() == BRACKET_OPEN for ln in s)]
        print(f"  ranks: {sorted(streams)}; executing the bracket: {gated}")
        if len(gated) == len(streams):
            print("      WARNING: every rank is gated — this fixture cannot "
                  "show the rank-local asymmetry")
            ok = False
        for rank in sorted(streams):
            ok &= _report(
                f"rank {rank}",
                _run_deck(binary, text, rank=rank),
                _inv1_offenders(streams[rank]),
            )
    return 0 if ok else 1


def _inline_sources(driver: Path) -> "list[str]":
    """The line stream Tcl executes for a split driver: the driver with
    each fragment ``source`` line replaced by the fragment's body.

    INV-1 is a property of THIS stream — the S6 hoist changes nothing
    but where the fragment bodies land in it.
    """
    import re
    src_re = re.compile(
        r"^source \[file join \[file dirname \[info script\]\] "
        r"parts (\S+)\]$"
    )
    parts = driver.parent / "parts"
    out: "list[str]" = []
    for ln in driver.read_text(encoding="utf-8").splitlines():
        m = src_re.match(ln)
        if m is not None:
            out.extend(
                (parts / m.group(1)).read_text(encoding="utf-8")
                .splitlines()
            )
        else:
            out.append(ln)
    return out


def split_arm(binary: str) -> int:
    """ADR 0099 S6: the same model, file-per-module fragments.

    The hoist moves the gated module's ``source`` line above the four
    declarations; the fragment itself carries its nodes along unchanged.
    Pre-S6 this emit was refused (INV-4); before the refusal landed the
    driver declared all four kinds above the sourced soil fragment,
    whose bracket destroyed them.  The driver must now run to the
    marker AND print the same displacements as the flat reference deck.
    """
    print("--split: composed model, driver + parts/*.tcl, vs flat reference")
    ok = True
    probe = 'puts "PROBE [nodeDisp 5] | [nodeDisp 9]"'
    with tempfile.TemporaryDirectory() as d:
        flat = Path(d) / "flat.tcl"
        _bridge(partitioned=False, composed=True).tcl(
            str(flat), progress=False, analyze_steps=1)
        flat_text = flat.read_text(encoding="utf-8") + probe + "\n"
        flat_out = _run_deck(binary, flat_text)
        ok &= _report(
            "flat reference", flat_out,
            _inv1_offenders(flat_text.splitlines()),
        )

        drv = Path(d) / "split" / "deck.tcl"
        drv.parent.mkdir()
        _bridge(partitioned=False, composed=True).tcl(
            str(drv), split=True, progress=False, analyze_steps=1)
        stream = _inline_sources(drv)
        if not any(ln.strip() == BRACKET_OPEN for ln in stream):
            print("      WARNING: no bracket in the split stream — the "
                  "probe is vacuous")
            ok = False
        with open(drv, "a", encoding="utf-8") as f:
            f.write(f'{probe}\nputs "{MARKER}"\n')
        proc = subprocess.run(
            [binary, str(drv)], capture_output=True, text=True, timeout=300,
        )
        out = proc.stdout + proc.stderr
        ok &= _report("split driver", out, _inv1_offenders(stream))

        flat_probe = [ln for ln in flat_out.splitlines()
                      if ln.startswith("PROBE")]
        split_probe = [ln for ln in out.splitlines()
                       if ln.startswith("PROBE")]
        match = bool(flat_probe) and flat_probe == split_probe
        print(f"  probe match vs flat: {'OK' if match else 'MISMATCH'}")
        print(f"      flat : {flat_probe}")
        print(f"      split: {split_probe}")
        ok &= match
    return 0 if ok else 1


def main() -> int:
    argv = sys.argv[1:]
    arm = argv.pop(0) if argv and argv[0].startswith("--") else None
    binary = argv[0] if argv else DEFAULT_BIN
    if not Path(binary).exists():
        print(f"OpenSees binary not found: {binary}")
        return 2
    if arm == "--h5":
        return h5_arm(binary)
    if arm == "--partitioned":
        return partitioned_arm(binary)
    if arm == "--split":
        return split_arm(binary)
    if arm is not None:
        print(f"unknown arm {arm!r}; expected --h5, --partitioned or "
              "--split")
        return 2

    print(f"{'declaration':<18} {'after a model re-issue'}")
    print("-" * 44)
    destroyed = []
    for name, (probe, marker) in PROBES.items():
        out = run(binary, PREAMBLE + BRACKET + probe + "\n")
        gone = marker in out
        print(f"{name:<18} {'DESTROYED' if gone else 'survives'}")
        if gone:
            destroyed.append(name)

    print()
    print("destroyed:", ", ".join(destroyed) or "(none)")
    print(
        "\nADR 0099 INV-1: every destroyed kind must be declared AFTER the\n"
        "last 'model BasicBuilder' line in an emitted deck."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
