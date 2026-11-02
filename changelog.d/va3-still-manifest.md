### ADDED — apeGmshViewer: every still ships with a numeric `manifest.json` (ADR 0128 D3, slice VA3, #2652)

`--render` now writes `manifest.json` in place of `render.json`, including for a spec refused whole. Each shot
has one entry: the shot as applied and the resolved camera; the files read with their sizes and lineage hashes;
the stage, step, time and pairing; for a contour its range, mode and history-scan state, its extrema (value, node or
element id, reference position) over every recorded value and over what is drawn, and its no-data count; for a
deformed shape its scale and the nodes drawn undeformed; and every banner line. Only `timing` varies between two
runs of one spec. The entry is built before the PNG is written, so a view the manifest cannot state is refused and
leaves no still. `tests/viewers/test_agent_manifest_oracle.py` holds the numbers against Python's `Results`.
