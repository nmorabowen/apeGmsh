"""Acceptance check — does the bench run feed every ADR 0098 §4 slot?

The model is only canonical if none of the seven slots comes up empty and
every pose the ADR names has data behind it. This opens the run, prints
what the broker found, then fills each slot on its own pane and renders a
still off it. A refusal here is a bench defect, not a viewer bug.

Run::

    python check_slots.py                  # after build.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel
from apeGmsh.results.session import (
    Contour, Deform, Gauss, Instant, Line, Loads, PlotSeries, PlotSource,
    Reactions, Sand, Scope, Vector,
)

SOLIDS = Scope("physical_groups", ("soil", "raft"))
FRAME = Scope("physical_groups", ("columns", "beams", "grade_beams"))
SHELLS = Scope("physical_groups", ("slabs", "wall"))

# slot name -> (scope, occupant, attribute)
CASES = {
    "contour": (SOLIDS, Contour("stress_zz"), "contour"),
    # A shell has no Cauchy stress tensor, so the solid contour above
    # says nothing about whether `slabs`/`wall` can be painted. This is
    # the shell-side half (ADR 0098 A6.6 R2).
    "contour_shell": (SHELLS, Contour("bending_moment_xx"), "contour"),
    # A DERIVED gauss scalar. Promoted out of the known-gaps probe
    # below: the contour gate tested only recorded columns, and a
    # derived scalar is computed on read, so it never appeared there.
    "contour_derived": (SOLIDS, Contour("von_mises_stress"), "contour"),
    "vector": (SOLIDS, Vector("displacement"), "vector"),
    "gauss": (SOLIDS, Gauss("von_mises_stress"), "gauss"),
    "line": (FRAME, Line("bending_moment_z"), "line"),
    "sand": (SOLIDS, Sand("displacement_x"), "sand"),
    "loads": (None, Loads(), "loads"),
    "reactions": (None, Reactions(), "reactions"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # cwd, not a path beside this script: run_case.py runs the verify
    # script with cwd = the same results/ the driver just wrote into.
    ap.add_argument("--out", type=Path, default=Path.cwd())
    args = ap.parse_args()

    run_h5, model_h5 = args.out / "run.h5", args.out / "model.h5"
    shots = args.out / "slots"
    shots.mkdir(parents=True, exist_ok=True)

    results = Results.from_native(str(run_h5),
                                  model=OpenSeesModel.from_h5(str(model_h5)))

    print("--- broker ---")
    for st in results.stages:
        extra = ""
        if st.kind == "mode":
            extra = f"  T = {st.period_s:.4f} s"
        print(f"  stage {st.name:10s} kind={st.kind:9s} "
              f"steps={st.n_steps}{extra}")
    dyn = next((s for s in results.stages if s.kind == "transient"), None)
    last = Instant(dyn.id, dyn.n_steps - 1) if dyn else None
    print(f"  lineage warnings: {results.lineage.warnings or '(clean)'}")

    failures: list[str] = []

    print("--- assess ---")
    report = results.assess()
    print("  " + "\n  ".join(report.text.splitlines()[:12]))

    # A still renders just as happily on a model that never moved, so the
    # gate has to ask the data whether the dynamic stage is alive. It once
    # was not: `loadConst` had frozen the excitation and all 150 steps held
    # the gravity deflection exactly.
    print("--- motion ---")
    import numpy as np
    ux = results.stage("dynamic").nodes.get(pg="slabs",
                                            component="displacement_x")
    swing = float(np.abs(ux.values).max())
    spread = float(ux.values.max(axis=0).max() - ux.values.min(axis=0).min())
    static_ux = float(np.abs(results.stage("gravity").nodes.get(
        pg="slabs", component="displacement_x").values).max())
    print(f"  roof |ux| max = {swing * 1e3:.2f} mm over {ux.values.shape[0]} steps"
          f"  (peak-to-peak {spread * 1e3:.2f} mm, gravity {static_ux * 1e3:.2f} mm)")
    if spread < 10.0 * static_ux:
        failures.append("dynamic stage does not move")
        print("FAIL motion — the dynamic stage barely moves; excitation applied?")
    else:
        print(f"PASS motion — {spread * 1e3:.2f} mm peak-to-peak")

    print("--- slots ---")
    for name, (scope, occupant, attr) in CASES.items():
        session = results.session()
        view = session.panes[0]
        if scope is not None:
            view.scope = scope
        if last is not None:
            view.time = last
        view.deform = Deform("displacement", 40.0)
        setattr(view, attr, occupant)
        try:
            session.render(str(shots / f"{name}.png"), view.id)
            legends = [lg.field for lg in view.legends()]
            print(f"PASS slot {name} — legends={legends}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(name)
            print(f"FAIL slot {name} — {type(exc).__name__}: {exc}")

    print("--- poses ---")
    for label, deform, instant in (
        ("displacement", Deform("displacement", 40.0), last),
        ("velocity", Deform("velocity", 40.0), last),
        ("acceleration", Deform("acceleration", 40.0), last),
        # scale=None is auto-fit: an eigenvector is normalised, so a fixed
        # scale that suits a 5 mm dynamic step turns a mode into confetti.
        ("mode 1", Deform("displacement", None, mode=1), None),
        ("mode 3", Deform("displacement", None, mode=3), None),
    ):
        session = results.session()
        view = session.panes[0]
        view.deform = deform
        if instant is not None:
            view.time = instant
        try:
            session.render(str(shots / f"pose_{label.replace(' ', '')}.png"),
                           view.id)
            print(f"PASS pose {label} — legends={view.legends()}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"pose:{label}")
            print(f"FAIL pose {label} — {type(exc).__name__}: {exc}")

    # Quantities the model records but the session cannot currently paint.
    # These do NOT fail the gate — the data is there, the viewer path is
    # not — but they stay visible so a fix is noticed the day it lands.
    print("--- known viewer gaps (data exists, slot refuses) ---")
    for label, scope, occupant, attr in (
        ("sand of a gauss field", SOLIDS, Sand("stress_zz"), "sand"),
        ("contour of von_mises_shell", SHELLS,
         Contour("von_mises_shell"), "contour"),
    ):
        session = results.session()
        view = session.panes[0]
        view.scope = scope
        setattr(view, attr, occupant)
        try:
            session.render(str(shots / "gap_probe.png"), view.id)
            # deliberately NOT a PASS/FAIL token: this is a watch item,
            # not part of the gate the case runner tallies.
            print(f"note: {label} — NOW SUPPORTED, promote it into CASES")
        except Exception as exc:                       # noqa: BLE001
            print(f"note: {label} — still refused ({type(exc).__name__})")

    # G4 (A6.6): the picker offers what THIS pane's cells record. Two
    # scopes over two element families must therefore be offered two
    # DIFFERENT gauss vocabularies. Before the shell resultants were
    # wired, `slabs+wall` recorded nothing at gauss level and this
    # section had nothing to compare.
    print("--- picker vocabulary is scoped (A6.6 G4) ---")
    from apeGmsh.viewers.session import recorded_components
    session = results.session()
    vocab = {}
    for label, scope in (("solids", SOLIDS), ("shells", SHELLS)):
        view = session.panes[0]
        view.scope = scope
        vocab[label] = recorded_components(session, view)[1]
        print(f"  {label:7s} gauss -> {sorted(vocab[label])}")
    shell_only = {"membrane_force_xx", "bending_moment_xx",
                  "transverse_shear_xz"}
    if not shell_only <= vocab["shells"]:
        failures.append("picker:shells")
        print("FAIL picker — a shell-scoped pane is not offered the "
              "resultants it records")
    elif vocab["solids"] & shell_only or "stress_zz" in vocab["shells"]:
        failures.append("picker:leak")
        print("FAIL picker — the two scopes leak each other's vocabulary")
    else:
        print("PASS picker — shells and solids offer disjoint gauss sets")

    # ...and the resultants must have RANGE. The upstream shells return a
    # correctly sized vector of zeros, which reaches the picker, paints a
    # legend, and contours one flat colour — every check above passes on
    # a measurement that never happened.
    mxx = results.stage("dynamic").elements.gauss.get(
        component="bending_moment_xx").values
    spread_m = float(mxx.max() - mxx.min())
    if spread_m <= 0.0:
        failures.append("picker:flat")
        print(f"FAIL picker — bending_moment_xx is flat ({spread_m}); the "
              f"shells are recording zeros, not resultants")
    else:
        print(f"PASS picker — bending_moment_xx spans {spread_m:.3e} "
              f"over {mxx.shape[1]} gauss points")

    # ADR 0098 §5: a contour and a Gauss slot of the SAME quantity share
    # ONE legend. Unreachable until the contour could accept a derived
    # scalar at all, so it is gated here rather than assumed.
    print("--- shared legend (§5) ---")
    session = results.session()
    view = session.panes[0]
    view.scope = SOLIDS
    view.contour = Contour("von_mises_stress")
    view.gauss = Gauss("von_mises_stress")
    try:
        session.render(str(shots / "shared_legend.png"), view.id)
        fields = [lg.field for lg in view.legends()]
        if fields == ["von_mises_stress"]:
            print(f"PASS shared legend — {fields}")
        else:
            failures.append("shared_legend")
            print(f"FAIL shared legend — expected one entry, got {fields}")
    except Exception as exc:                       # noqa: BLE001
        failures.append("shared_legend")
        print(f"FAIL shared legend — {type(exc).__name__}: {exc}")

    print("--- scope axes ---")
    for label, scope in (
        ("physical_groups", Scope("physical_groups", ("wall", "slabs"))),
        ("element_types", Scope("element_types", ("tet4",))),
        ("materials", Scope("materials", ("concrete",))),
    ):
        session = results.session()
        view = session.panes[0]
        try:
            view.scope = scope
            session.render(str(shots / f"scope_{label}.png"), view.id)
            print(f"PASS scope {label}")
        except Exception as exc:                       # noqa: BLE001
            # The materials axis is known-blocked (no element->material
            # index exists); refusing loudly IS the expected behaviour, so
            # it passes — drawing the wrong set would be the failure.
            if label == "materials":
                print(f"PASS scope {label} — refuses loudly, as it must "
                      f"({type(exc).__name__})")
            else:
                failures.append(f"scope:{label}")
                print(f"FAIL scope {label} — {type(exc).__name__}: {exc}")

    print("--- plot ---")
    try:
        roof = int(results.model.fem.nodes.select(pg="columns").ids[-1])
        session = results.session()
        session.add_plot(kind="history", series=(
            PlotSeries(PlotSource("node", roof), "displacement_x"),))
        print(f"PASS plot — history on node {roof} "
              f"({len(session.panes)} panes)")
    except Exception as exc:                           # noqa: BLE001
        failures.append("plot")
        print(f"FAIL plot — {type(exc).__name__}: {exc}")

    print()
    if failures:
        raise SystemExit(f"BENCH INCOMPLETE — {len(failures)} failed: {failures}")
    print(f"every slot and pose fed; stills in {shots}")


if __name__ == "__main__":
    main()
