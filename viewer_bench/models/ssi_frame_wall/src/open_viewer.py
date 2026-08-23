"""Open the bench in the Results viewer, with panes already worth looking at.

`results.viewer()` boots one empty grey mesh view, which is a fair default
and a poor demo. This builds a three-pane session instead — a clipped
stress contour through the soil, the frame's moment diagram, and a roof
history plot — and shows that.

Run::

    python open_viewer.py                       # the canonical case
    python open_viewer.py --case c002_something
    python open_viewer.py --results /some/dir   # any results/ folder
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel
from apeGmsh.results.session import (
    Contour, Deform, Instant, Line, PlotSeries, PlotSource, Scope,
)

#: c001 is unreadable on current apeGmsh — its model.h5 carries
#: neutral schema 2.29.0 and the reader wants 2.30-2.31. The
#: payload is gitignored, so a schema bump silently strands
#: every case that is not re-run.
CANONICAL_CASE = "c005_mixed_dof_fixed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=CANONICAL_CASE)
    ap.add_argument("--results", type=Path, default=None,
                    help="a results/ folder holding model.h5 + run.h5")
    ap.add_argument("--render", type=Path, default=None,
                    help="render the mesh panes to this folder instead of "
                         "opening a window (ADR 0098 §10: same session, "
                         "different client)")
    args = ap.parse_args()

    results_dir = args.results or (
        Path(__file__).resolve().parents[1] / "cases" / args.case / "results")
    run_h5, model_h5 = results_dir / "run.h5", results_dir / "model.h5"
    if not run_h5.is_file():
        raise SystemExit(
            f"no run.h5 under {results_dir} — run the case first:\n"
            f"  python scripts/run_case.py --model ssi_frame_wall "
            f"--case {args.case} "
            f"--script models/ssi_frame_wall/src/build.py "
            f"--verify models/ssi_frame_wall/src/check_slots.py")

    results = Results.from_native(str(run_h5),
                                  model=OpenSeesModel.from_h5(str(model_h5)))
    dyn = next(s for s in results.stages if s.kind == "transient")
    peak = Instant(dyn.id, dyn.n_steps // 2)      # mid-shake, not the quiet end

    session = results.session()
    # The session boots LINKED with time=None, and under a link the session
    # instant wins — a per-pane `view.time` is stored but ignored (the
    # inspector says so out loud). So the instant goes on the SESSION.
    session.time = peak

    # Pane 1 — the ground, cut open so the raft and the stress bulb show.
    soil = session.panes[0]
    soil.scope = Scope("physical_groups", ("soil", "raft"))
    soil.contour = Contour("stress_zz")
    soil.deform = Deform("displacement", 200.0)
    soil.add_clip((0.0, 1.0, 0.0), offset=0.0)

    # Pane 2 — the structure, carrying its own moment diagram.
    frame = session.add_view(name="frame")
    frame.scope = Scope("physical_groups",
                        ("columns", "beams", "grade_beams", "wall", "slabs"))
    frame.line = Line("bending_moment_z")
    frame.deform = Deform("displacement", 200.0)

    # Pane 3 — the roof history the scrubber rides.
    roof = int(results.model.fem.nodes.select(pg="slabs").ids[-1])
    session.add_plot(kind="history", name="roof drift", series=(
        PlotSeries(PlotSource("node", roof), "displacement_x"),))

    print(f"[viewer] {run_h5}", flush=True)
    print(f"[viewer] {len(session.panes)} panes, linked at {peak}", flush=True)
    if args.render is not None:
        args.render.mkdir(parents=True, exist_ok=True)
        for pane, name in ((soil, "soil"), (frame, "frame")):
            out = args.render / f"{name}.png"
            session.render(str(out), pane.id, window_size=(1600, 900))
            print(f"[viewer] rendered {out}", flush=True)
        return
    print("[viewer] close the window to return", flush=True)
    session.show()


if __name__ == "__main__":
    main()
