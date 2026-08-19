"""Profile the ADR 0098 results-session window — INTERACTION, not boot.

Sibling of ``internal_docs/profiling/box_wave/profile_viewers.py``, which
covers ``model.viewer`` / ``mesh.viewer``. This one covers
``results.session().show()``: the pane docks, the reconciler pump and the
gestures that run while the window is open.

Boot is already understood and is mostly cold imports (first-touch
``pyvistaqt`` / ``pyvista.plotting`` ~0.84 s, two ``QtInteractor``
creations ~0.78 s, matplotlib font metrics ~0.5 s on a bench with a plot
pane). It does not grow with the model. Interaction does, on two axes at
once — model size AND pane count — which is what this measures.

How to run
----------
1. Use the venv that actually has gmsh + openseespy
   (``C:\\Users\\nmb\\venv\\opensees_env\\Scripts\\python.exe``).
2. From the repo root::

       python internal_docs/profiling/results_session/profile_session.py [stage]

   ``stage`` is ``replay`` | ``interact`` | ``both`` (default ``replay``).

   * ``replay`` drives a DETERMINISTIC gesture workload and prints
     per-gesture ms (mean / p50 / p95). These numbers are comparable
     across commits, which "I wiggled the mouse for ten seconds" is not.
   * ``interact`` opens the window and profiles whatever YOU do until you
     close it — the box_wave idiom, for chasing something you can feel
     but cannot script.

3. Inspect the dump with snakeviz::

       snakeviz internal_docs/profiling/results_session/replay.prof

Useful knobs
------------
``--panes N``      pane count (default 3). Interaction cost is per-pane
                   under the time link, so 1 vs 3 vs 6 is the experiment
                   that says whether panes or the model dominate.
``--repeat K``     iterations per gesture (default 20).
``--gestures ...`` subset of ``scrub orbit slot scope pane``.
``--results DIR``  any folder holding ``model.h5`` + ``run.h5``.
``--contour F``    force the first pane's contour field. A GAUSS field
                   pays a Gauss->node extrapolation per realize; a nodal
                   field skips it. That one choice moves ``scrub`` by
                   more than pane count does.
``--no-profile``   timings only, no cProfile — cProfile inflates wall
                   clock ~1.3x here, so trust these numbers for absolute
                   cost and the .prof for attribution.
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

HERE = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────
# Import shim — force the worktree's apeGmsh, not the editable install.
#
# The venv has a PEP-660 editable install of apeGmsh pinned to the main
# checkout. Its finder is *appended* to sys.meta_path, so it runs AFTER
# the default PathFinder; putting the worktree's src on sys.path[0] makes
# PathFinder resolve apeGmsh here first. Same shim, and the same hard
# guard in main(), as the box_wave harness — profiling the wrong tree is
# a silent way to measure nothing.
# ─────────────────────────────────────────────────────────────────────
_WORKTREE_SRC = HERE.parents[2] / "src"
sys.path.insert(0, str(_WORKTREE_SRC))

#: The ADR 0098 bench, when it is checked out beside the repo. It is not
#: tracked here, so it is only ever a DEFAULT — any results/ folder works.
_BENCH = (HERE.parents[2] / "viewer_bench" / "models" / "ssi_frame_wall"
          / "cases" / "c001_baseline_small" / "results")


# ─────────────────────────────────────────────────────────────────────
# Session under test
# ─────────────────────────────────────────────────────────────────────

def _group_names(results) -> "list[str]":
    """The physical-group names, from the same source the outline lists
    (``ViewerData.elements.physical``) — so a scope this harness sets is
    one a user could have set."""
    from apeGmsh.viewers.data import ViewerData

    return list(ViewerData.from_fem(results.fem).elements.physical.names())


def build_session(results_dir: Path, panes: int, contour: str = ''):
    """A representative session: contour+deform+clip, a line diagram, a
    history plot, then filler mesh views up to ``panes``.

    Deliberately the demanding shape rather than the default empty view —
    an empty pane realizes almost nothing and would flatter every number
    here.
    """
    from apeGmsh import Results
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results.session import (
        Contour, Deform, Instant, Line, PlotSeries, PlotSource,
    )

    run_h5, model_h5 = results_dir / "run.h5", results_dir / "model.h5"
    if not run_h5.is_file():
        raise SystemExit(
            f"[profile] no run.h5 under {results_dir} - pass --results at a "
            f"folder holding model.h5 + run.h5."
        )

    t = time.perf_counter()
    results = Results.from_native(
        str(run_h5), model=OpenSeesModel.from_h5(str(model_h5)))
    print(f"[profile] results open: {time.perf_counter() - t:.2f}s")

    dyn = next((s for s in results.stages if s.kind == "transient"),
               results.stages[-1])
    session = results.session()
    session.time = Instant(dyn.id, dyn.n_steps // 2)

    groups = _group_names(results)

    first = session.panes[0]
    first.deform = Deform("displacement", 200.0)
    # Gauss-sourced first: a Gauss contour pays the extrapolation a nodal
    # one skips entirely, and that difference is the whole scrub story.
    for field in ((contour,) if contour else ("stress_zz", "displacement_z")):
        try:
            first.contour = Contour(field)
            break
        except Exception:
            continue
    try:
        first.add_clip((0.0, 1.0, 0.0), offset=0.0)
    except Exception:
        pass

    if panes >= 2:
        second = session.add_view(name="frame")
        second.deform = Deform("displacement", 200.0)
        try:
            second.line = Line("bending_moment_z")
        except Exception:
            pass
    if panes >= 3:
        node = int(results.model.fem.nodes.ids[-1])
        session.add_plot(kind="history", name="history", series=(
            PlotSeries(PlotSource("node", node), "displacement_x"),))
    for k in range(4, panes + 1):
        session.add_view(name=f"view {k}")

    return results, session, dyn, groups


def open_window(session, out_dir: Path):
    """The real window, maximized, on a THROWAWAY QSettings scope.

    Profiling must not read or write the developer's saved arrangement:
    a restored layout would change what is being measured, and the
    profile run would then save whatever it produced back over it.
    """
    from qtpy import QtWidgets
    from qtpy.QtCore import QSettings

    from apeGmsh.viewers.session import SessionWindow
    from apeGmsh.viewers.session._window import SessionResultsWindow

    ini = str(out_dir / "profile_scope.ini")
    Path(ini).unlink(missing_ok=True)
    SessionResultsWindow._layout_settings = staticmethod(
        lambda: QSettings(ini, QSettings.Format.IniFormat))

    t = time.perf_counter()
    window = SessionWindow(session)
    print(f"[profile] SessionWindow(...): {time.perf_counter() - t:.2f}s")
    window.show(blocking=False)
    app = QtWidgets.QApplication.instance()
    for _ in range(30):
        app.processEvents()
    return app, window


# ─────────────────────────────────────────────────────────────────────
# The gesture workload
# ─────────────────────────────────────────────────────────────────────

def _settle(app, window) -> None:
    """Drive one gesture all the way to painted pixels.

    A session write only SCHEDULES a coalesced flush, so timing the write
    alone measures the scheduling and nothing else. Pumping first lets the
    real deferred path run; ``flush_now`` then collects any straggler and
    is a no-op for a pane that already flushed.
    """
    for _ in range(3):
        app.processEvents()
    for frame in window.host.pane_frames:
        frame.pane.reconciler.flush_now()
    app.processEvents()


def _timed(app, window, action) -> float:
    t = time.perf_counter()
    action()
    _settle(app, window)
    return (time.perf_counter() - t) * 1000.0


def gesture_scrub(app, window, ctx, repeat):
    """Step the session instant — what playback and a scrubber drag cost.

    Under the time link this is the ONE gesture every pane pays for, so
    it is the one that scales with pane count.
    """
    from apeGmsh.results.session import Instant

    session, dyn = window.session, ctx["stage"]
    n = max(2, dyn.n_steps)
    start = n // 4
    return [
        _timed(app, window, lambda i=i: setattr(
            session, "time", Instant(dyn.id, (start + i) % n)))
        for i in range(repeat)
    ]


def gesture_orbit(app, window, ctx, repeat):
    """Rotate the active pane's camera — the pure-GL cost, no realize."""
    frame = window.host.frame(window.host.active_pane_id)
    plotter = frame.plotter
    if plotter is None:
        return []

    def spin():
        # ``azimuth`` is a PROPERTY on this pyvista, not a method.
        plotter.camera.azimuth += 5.0
        plotter.render()

    return [_timed(app, window, spin) for _ in range(repeat)]


def gesture_slot(app, window, ctx, repeat):
    """Fill and clear one pane's contour slot — the §9 Add/clear loop.

    Criterion 12 says this costs ONE pane a realize; if it costs N, that
    shows up here as a per-pane slope.
    """
    from apeGmsh.results.session import Contour

    view = window.session.panes[0]
    if view.contour is None:
        return []
    quantity = view.contour.quantity
    out = []
    for i in range(repeat):
        out.append(_timed(app, window, lambda i=i: setattr(
            view, "contour", None if i % 2 else Contour(quantity))))
    view.contour = Contour(quantity)
    _settle(app, window)
    return out


def gesture_scope(app, window, ctx, repeat):
    """Flip one pane's scope — re-resolves the scoped element/node set."""
    from apeGmsh.results.session import Scope

    groups = ctx["groups"]
    if len(groups) < 2:
        return []
    view = window.session.panes[0]
    out = []
    for i in range(repeat):
        names = (groups[i % len(groups)],)
        out.append(_timed(app, window, lambda n=names: setattr(
            view, "scope", Scope("physical_groups", n))))
    view.scope = None
    _settle(app, window)
    return out


def gesture_pane(app, window, ctx, repeat):
    """Add a pane and close it again — Amendment 3's dock create/destroy
    plus a fresh GL context each time."""
    session = window.session
    out = []
    for i in range(repeat):
        def add_close():
            view = session.add_view(name="scratch")
            session.remove_pane(view.id)
        out.append(_timed(app, window, add_close))
    return out


GESTURES = {
    "scrub": gesture_scrub,
    "orbit": gesture_orbit,
    "slot": gesture_slot,
    "scope": gesture_scope,
    "pane": gesture_pane,
}


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def _report(rows: "dict[str, list[float]]", panes: int) -> None:
    import numpy as np

    # ASCII in PRINTED output: this runs on a cp1252 Windows console, and
    # a box-drawing character there is a UnicodeEncodeError that kills the
    # report after the whole workload has already been paid for.
    print(f"\n-- per-gesture cost, {panes} pane(s) (ms) --")
    print(f"  {'gesture':10s} {'n':>4s} {'mean':>9s} {'p50':>9s} "
          f"{'p95':>9s} {'max':>9s} {'total':>9s}")
    for name, samples in rows.items():
        if not samples:
            print(f"  {name:10s} {'-':>4s}   (not applicable to this model)")
            continue
        a = np.asarray(samples)
        print(f"  {name:10s} {a.size:4d} {a.mean():9.1f} "
              f"{np.percentile(a, 50):9.1f} {np.percentile(a, 95):9.1f} "
              f"{a.max():9.1f} {a.sum():9.1f}")


def _dump(profiler: cProfile.Profile, out: Path, label: str) -> None:
    profiler.disable()
    out.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(out))
    print(f"\n[profile] {label} -> {out}")
    print(f"\n-- Top 30 cumulative-time ({label}) --")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(30)
    print(f"\n-- Top 20 internal-time ({label}) --")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(20)


# ─────────────────────────────────────────────────────────────────────
# Drivers
# ─────────────────────────────────────────────────────────────────────

def run_replay(app, window, ctx, args) -> None:
    names = args.gestures or list(GESTURES)
    print(f"\n[profile] replay: {', '.join(names)} x{args.repeat}")
    profiler = None
    if not args.no_profile:
        profiler = cProfile.Profile()
        profiler.enable()
    rows = {name: GESTURES[name](app, window, ctx, args.repeat)
            for name in names}
    if profiler is not None:
        _dump(profiler, args.out / "replay.prof", "replay")
    _report(rows, len(window.session.panes))


def run_interact(app, window, args) -> None:
    print("\n[profile] interact: drive the window, then CLOSE it to end "
          "profiling ...")
    profiler = cProfile.Profile()
    profiler.enable()
    app.exec_()
    _dump(profiler, args.out / "interact.prof", "interact")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main(argv: "list[str]") -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", default="replay",
                   choices=("replay", "interact", "both"))
    p.add_argument("--results", type=Path, default=_BENCH,
                   help="a results/ folder holding model.h5 + run.h5")
    p.add_argument("--panes", type=int, default=3)
    p.add_argument("--contour", default="",
                   help="force the first pane's contour field; a Gauss "
                        "field (stress_zz) pays the Gauss->node "
                        "extrapolation, a nodal one (displacement_z) "
                        "does not")
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--gestures", nargs="*", choices=sorted(GESTURES))
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--out", type=Path, default=HERE)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    # Hard guard: confirm we're profiling the worktree code, not the
    # editable install. Abort loudly if the shim didn't take.
    import apeGmsh as _ape
    ape_path = Path(_ape.__file__).resolve()
    print(f"[profile] apeGmsh loaded from: {ape_path}")
    if str(_WORKTREE_SRC) not in str(ape_path):
        print(f"[profile] FATAL: expected apeGmsh under {_WORKTREE_SRC}, got "
              f"{ape_path}. The sys.path shim failed - aborting so we don't "
              f"profile the wrong code.", file=sys.stderr)
        return 2

    results, session, stage, groups = build_session(
        args.results, args.panes, args.contour)
    del results
    app, window = open_window(session, args.out)
    ctx = {"stage": stage, "groups": groups}
    print(f"[profile] {len(session.panes)} pane(s), stage {stage.id} "
          f"({stage.n_steps} steps)")

    try:
        if args.stage in ("replay", "both"):
            run_replay(app, window, ctx, args)
        if args.stage in ("interact", "both"):
            run_interact(app, window, args)
    finally:
        window.close()
        app.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
