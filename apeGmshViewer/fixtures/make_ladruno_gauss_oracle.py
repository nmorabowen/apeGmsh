"""Write ``ladruno_quad2d.oracle.json``: Python's reading of the fork's Gauss buckets (#2187).

    $env:PYTHONPATH = "<worktree>\\src"
    python apeGmshViewer/fixtures/make_ladruno_gauss_oracle.py

run from the repository root, with apeGmsh importable from ``src`` (the
editable install may point at another checkout: check the printed path). It
reads ``ladruno_quad2d.ladruno`` with apeGmsh's own ``.ladruno`` reader
(``LadrunoReader.read_gauss``) and records, for one component at one step,
each Gauss point's element id, natural coordinates and value. The app's test
(``test/results/ladrunoGauss.test.ts``) compares its reader against these
values: the expected array comes from Python, never from the app.

Python names the component by its canonical name (``stress_xx`` for the
fork's ``sigma11``); the app spells it as the file does (``stress/sigma11``,
ADR 0122 D7). Both name the same DATA columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import apeGmsh
from apeGmsh.results.readers._ladruno import LadrunoReader

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ladruno_quad2d.ladruno"
OUT = HERE / "ladruno_quad2d.oracle.json"

#: (the app's component, Python's canonical name, the step) for each recorded read.
READS = [("stress/sigma11", "stress_xx", 1), ("strain/eta12", "strain_xy", 0)]


def main() -> None:
    print("apeGmsh from", apeGmsh.__file__)
    reads = []
    with LadrunoReader(SOURCE) as r:
        stage = r.stages()[0].id
        for app_name, canonical, step in READS:
            slab = r.read_gauss(stage, canonical, time_slice=[step])
            assert slab.values.shape[0] == 1, slab.values.shape
            points = [
                {"element": int(e), "natural": [float(x) for x in nat], "value": float(v)}
                for e, nat, v in zip(slab.element_index, slab.natural_coords, slab.values[0])
            ]
            reads.append({"component": app_name, "python_component": canonical, "stage": stage, "step": step, "points": points})
    OUT.write_text(json.dumps({"source": SOURCE.name, "reader": "apeGmsh.results.readers._ladruno.LadrunoReader.read_gauss", "reads": reads}, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
