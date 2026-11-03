"""Write ``ladruno_quad2d.principal.oracle.json``: Python's principal frame of the fork's Gauss tensors.

    $env:PYTHONPATH = "<worktree>\\src"
    python tests/fixtures/ladruno/make_viewer_principal_oracle.py

run from the repository root, with apeGmsh importable from ``src`` (check the
printed path). The sibling of ``make_viewer_gauss_oracle.py``, outside
``apeGmshViewer/`` for the same reason (ADR 0112 D4). It reads
``apeGmshViewer/fixtures/ladruno_quad2d.ladruno`` with ``LadrunoReader.read_gauss``
and puts each family's Voigt columns through
``apeGmsh.results._derived.principal_frame`` (``plane=None``: a 2-D element's
zz is 0), the function Qt's PrincipalDirectionDiagram draws from. Per Gauss
point it records the element, natural coordinates, the principal values
(descending) and their unit vectors (sign-canonical). The app's test
(``test/results/ladrunoPrincipal.test.ts``) compares state/principal.ts against
them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import apeGmsh
from apeGmsh.results import _derived
from apeGmsh.results.readers._ladruno import LadrunoReader

FIXTURES = Path(__file__).resolve().parents[3] / "apeGmshViewer" / "fixtures"
SOURCE = FIXTURES / "ladruno_quad2d.ladruno"
OUT = FIXTURES / "ladruno_quad2d.principal.oracle.json"

#: (family, the step) for each recorded frame.
READS = [("stress", 1), ("strain", 1)]


def main() -> None:
    print("apeGmsh from", apeGmsh.__file__)
    frames = []
    with LadrunoReader(SOURCE) as r:
        stage = r.stages()[0].id
        for family, step in READS:
            columns = {}
            ref = None
            for suffix in ("xx", "yy", "xy"):
                slab = r.read_gauss(stage, f"{family}_{suffix}", time_slice=[step])
                columns[f"{family}_{suffix}"] = np.asarray(slab.values[0], dtype=np.float64)
                if ref is None:
                    ref = slab
                assert np.array_equal(slab.element_index, ref.element_index)
            values, vectors = _derived.principal_frame(columns, prefix=family)
            points = [
                {
                    "element": int(e),
                    "natural": [float(x) for x in nat],
                    "values": [float(x) for x in values[i]],
                    "vectors": [[float(x) for x in vectors[i][:, k]] for k in range(3)],
                }
                for i, (e, nat) in enumerate(zip(ref.element_index, ref.natural_coords))
            ]
            frames.append({"family": family, "stage": stage, "step": step, "points": points})
    OUT.write_text(json.dumps({"source": SOURCE.name, "function": "apeGmsh.results._derived.principal_frame", "frames": frames}, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
