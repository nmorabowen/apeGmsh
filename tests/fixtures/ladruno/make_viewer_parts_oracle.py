"""Write ``apeGmshViewer/fixtures/ladruno_parts2d.oracle.json``: Python's stitched read of the 2-part set (#2188).

    $env:PYTHONPATH = "<worktree>\\src"
    python tests/fixtures/ladruno/make_viewer_parts_oracle.py

run from the repository root, with apeGmsh importable from ``src`` (the
editable install may point at another checkout: check the printed path). It
lives here, not under ``apeGmshViewer/``, because no Python under the app may
import apeGmsh (ADR 0112 D4, ``scripts/check_app_wall.py``).

It opens ``ladruno_parts2d.part-<N>.ladruno`` with apeGmsh's own partitioned
reader (``LadrunoMultiPartitionReader``, which follows each result's
``PARTITION_REDUCTION``) and records, for every node and step, the stitched
displacement (``NONE``: one copy) and reaction (``SUM``: the copies added). The
app's test (``apeGmshViewer/test/results/ladrunoParts.test.ts``) compares its
stitch against these values: the expected arrays come from Python, never from
the app.

Python names a component by its canonical name (``reaction_force_x``); the app
spells it as the file does (``REACTION_FORCE/Px``, ADR 0122 D7). Both name the
same DATA columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import apeGmsh
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader, discover_partition_files

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "apeGmshViewer" / "fixtures"
SOURCE = FIXTURES / "ladruno_parts2d.part-0.ladruno"
OUT = FIXTURES / "ladruno_parts2d.oracle.json"

#: (Python's canonical name, the result group, its column) for each recorded read.
READS = [
    ("displacement_x", "DISPLACEMENT", 0),
    ("displacement_y", "DISPLACEMENT", 1),
    ("reaction_force_x", "REACTION_FORCE", 0),
    ("reaction_force_y", "REACTION_FORCE", 1),
]


def main() -> None:
    print("apeGmsh from", apeGmsh.__file__)
    parts = discover_partition_files(SOURCE)
    assert [p.name for p in parts] == ["ladruno_parts2d.part-0.ladruno", "ladruno_parts2d.part-1.ladruno"], parts
    reads = []
    with LadrunoMultiPartitionReader(parts) as r:
        stage = r.stages()[0].id
        for canonical, group, column in READS:
            slab = r.read_nodes(stage, canonical)
            reads.append({
                "python_component": canonical,
                "group": group,
                "column": column,
                "stage": stage,
                "node_ids": [int(n) for n in slab.node_ids],
                "time": [float(t) for t in slab.time],
                "values": [[float(v) for v in row] for row in slab.values],
            })
    OUT.write_text(json.dumps({
        "sources": [p.name for p in parts],
        "reader": "apeGmsh.results.readers._ladruno_multi.LadrunoMultiPartitionReader.read_nodes",
        "reads": reads,
    }, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
