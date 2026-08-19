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


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BIN
    if not Path(binary).exists():
        print(f"OpenSees binary not found: {binary}")
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
