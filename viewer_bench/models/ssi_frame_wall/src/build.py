"""Build the ``ssi_frame_wall`` bench model — the canonical Results-viewer fixture.

A 3-D linear-elastic soil-structure model whose only purpose is to feed
every part of the ADR 0098 ``ResultsSession`` surface real data:

    soil block (tet4)  ──  raft (tet4, conforming)
                            │  embedded tie, rotational
                            └─ grade beams ── columns / beams (forceBeamColumn)
                                            ── shear wall / slabs (ASDShellQ4)

Three element families, two materials, nine physical groups, and records
for every slot in the §4 catalog, in two result sources: the recorder set
(stages gravity / dynamic, transcoded on open) and ``modes.h5`` (the
modes, from ``capture_live_modes``). See ``../README.md`` for the
slot-by-slot map.

Units are kN, m, s, t — so E is kPa, density t/m³, mass tonnes.

Run::

    python build.py                 # size="small", ~5 min, ~120 MB
    python build.py --size large    # heavier mesh for viewer performance work

Writes into the current directory: ``model.h5`` (the model archive), the
recorder set under ``out/`` (gravity and dynamic stages), ``modes.h5``
(the modes), ``bench.json`` (the size and stride the readers rebuild the
spec with) and the transcode cache under ``cache/``. ``open_results``
below opens both sources the way every bench consumer does. Under
``scripts/run_case.py`` that directory is the case's ``results/``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

import gmsh
import numpy as np
import openseespy.opensees as opspy

from apeGmsh import Results, apeGmsh
from apeGmsh._vocabulary import expand_shorthand
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.results.live import capture_live_modes
from apeGmsh.results.spec import ResolvedRecorderRecord, ResolvedRecorderSpec

# --------------------------------------------------------------- geometry
GRAV = 9.81
TOL = 1e-6

BOX_W, BOX_D = 30.0, 15.0          # soil block plan width / depth
RAFT_W, RAFT_T = 14.0, 0.8         # raft plan width / thickness
BAY, STOREY, N_STOREY = 6.0, 3.2, 3
GRID = (-BAY, 0.0, BAY)            # column grid lines, both plan directions
LEVELS = [STOREY * k for k in range(N_STOREY + 1)]   # 0 = grade beams

# -------------------------------------------------------------- materials
E_SOIL, NU_SOIL, RHO_SOIL = 320.0e3, 0.35, 1.9       # Vs ~ 250 m/s
E_CONC, NU_CONC, RHO_CONC = 25.0e6, 0.20, 2.4
G_CONC = E_CONC / (2.0 * (1.0 + NU_CONC))

COL_B = 0.50                       # square column
BEAM_B, BEAM_H = 0.30, 0.50
GRADE_B, GRADE_H = 0.40, 0.60
T_SLAB, T_WALL = 0.15, 0.25
SDL_SLAB = 2.0                     # superimposed dead load on slabs [kPa]

# --------------------------------------------------------------- analysis
N_GRAV = 10                        # gravity load increments
N_MODES = 6
DT = 0.01
DURATION = 6.0                     # includes the quiet lead
QUIET_LEAD = 0.5                   # flat lead-in so the histories start quiet
GM_SERIES_TAG = 9001               # raw-openseespy tags, clear of the bridge's
GM_PATTERN_TAG = 9002
PGA = 0.30 * GRAV                  # peak input acceleration [m/s²]

#: The recorder set's stages, in the order they run; every reader passes
#: these to ``Results.from_recorders(stages=...)``.
STAGES = (SimpleNamespace(name="gravity", kind="static"),
          SimpleNamespace(name="dynamic", kind="transient"))
OUT_DIR, CACHE_DIR, MODES_H5, META = "out", "cache", "modes.h5", "bench.json"

SIZES = {
    "small": dict(lc_shell=0.75, lc_soil_near=1.2, lc_soil_far=4.0, stride=4),
    "large": dict(lc_shell=0.40, lc_soil_near=0.60, lc_soil_far=2.0, stride=10),
}


def rect_section(b: float, h: float) -> dict:
    """Elastic section constants for a b×h rectangle (h is the strong axis)."""
    a_, b_ = max(b, h) / 2.0, min(b, h) / 2.0
    return dict(
        A=b * h,
        Iz=b * h ** 3 / 12.0,
        Iy=h * b ** 3 / 12.0,
        J=a_ * b_ ** 3 * (16.0 / 3.0 - 3.36 * b_ / a_ * (1.0 - b_ ** 4 / (12.0 * a_ ** 4))),
    )


def ground_motion() -> np.ndarray:
    """A deterministic three-frequency pulse, scaled to PGA, quiet at first.

    No data file and no licence question: the record is generated, and the
    leading quiet window gives the history plots a flat start.
    """
    t = np.arange(0.0, DURATION + DT, DT)
    tau = t - QUIET_LEAD - 1.6
    env = np.exp(-0.5 * (tau / 0.9) ** 2)
    a = env * (np.sin(2 * np.pi * 2.2 * tau)
               + 0.55 * np.sin(2 * np.pi * 4.7 * tau + 0.7)
               + 0.30 * np.sin(2 * np.pi * 7.9 * tau + 1.9))
    a[t < QUIET_LEAD] = 0.0
    return a * (PGA / np.abs(a).max())


def build_mesh(g, cfg: dict):
    """Geometry, physical groups, constraints and mesh. Returns the FEMData."""
    geo = g.model.geometry

    # --- soil and raft: one conforming solid pair (the raft is cast in) ---
    soil = geo.add_box(-BOX_W / 2, -BOX_W / 2, -BOX_D, BOX_W, BOX_W, BOX_D)
    raft = geo.add_box(-RAFT_W / 2, -RAFT_W / 2, -RAFT_T, RAFT_W, RAFT_W, RAFT_T)
    g.model.boolean.fragment([(3, soil)], [(3, raft)])
    g.model.sync()

    soil_tag = raft_tag = None
    for dim, tag in gmsh.model.getEntities(3):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if (abs(bb[0] + RAFT_W / 2) < TOL and abs(bb[2] + RAFT_T) < TOL
                and abs(bb[5]) < TOL):
            raft_tag = tag
        else:
            soil_tag = tag

    # --- superstructure: one conforming line/surface complex at z >= 0 ---
    pt = {(x, y, z): geo.add_point(x, y, z)
          for x in GRID for y in GRID for z in LEVELS}

    col = {}
    for x in GRID:
        for y in GRID:
            for k in range(N_STOREY):
                col[(x, y, k)] = geo.add_line(pt[(x, y, LEVELS[k])],
                                              pt[(x, y, LEVELS[k + 1])])

    bx, by = {}, {}                 # beams along x / along y, per level
    for k, z in enumerate(LEVELS):  # k = 0 is the grade-beam grillage
        for y in GRID:
            for i in range(len(GRID) - 1):
                bx[(i, y, k)] = geo.add_line(pt[(GRID[i], y, z)],
                                             pt[(GRID[i + 1], y, z)])
        for x in GRID:
            for j in range(len(GRID) - 1):
                by[(x, j, k)] = geo.add_line(pt[(x, GRID[j], z)],
                                             pt[(x, GRID[j + 1], z)])

    slab = {}
    for k in range(1, N_STOREY + 1):
        for i in range(len(GRID) - 1):
            for j in range(len(GRID) - 1):
                loop = geo.add_curve_loop([bx[(i, GRID[j], k)],
                                           by[(GRID[i + 1], j, k)],
                                           -bx[(i, GRID[j + 1], k)],
                                           -by[(GRID[i], j, k)]])
                slab[(i, j, k)] = geo.add_plane_surface(loop)

    # shear wall: the y = -BAY edge frame, x in [-BAY, 0], full height
    wall = {}
    for k in range(N_STOREY):
        loop = geo.add_curve_loop([bx[(0, -BAY, k)],
                                   col[(0.0, -BAY, k)],
                                   -bx[(0, -BAY, k + 1)],
                                   -col[(-BAY, -BAY, k)]])
        wall[k] = geo.add_plane_surface(loop)
    g.model.sync()

    # --- physical groups ------------------------------------------------
    g.physical.add(3, [soil_tag], name="soil")
    g.physical.add(3, [raft_tag], name="raft")
    g.physical.add(1, list(col.values()), name="columns")
    g.physical.add(1, [t for (_, _, k), t in bx.items() if k > 0]
                   + [t for (_, _, k), t in by.items() if k > 0], name="beams")
    g.physical.add(1, [t for (_, _, k), t in bx.items() if k == 0]
                   + [t for (_, _, k), t in by.items() if k == 0],
                   name="grade_beams")
    g.physical.add(2, list(slab.values()), name="slabs")
    g.physical.add(2, list(wall.values()), name="wall")

    # Support faces. Meshing a volume also meshes its boundary surfaces, so
    # the tri3 skin over the soil box exists whether or not these groups do
    # (get_fem_data collects the whole gmsh mesh, not just grouped entities).
    # Naming them is therefore strictly better than leaving ~2000 elements
    # unclaimed: they become scopeable, and they explain themselves.
    base_faces, side_faces = [], []
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] + BOX_D) < TOL and abs(bb[5] + BOX_D) < TOL:
            base_faces.append(tag)
        elif any(abs(bb[c] - s * BOX_W / 2) < TOL
                 and abs(bb[c + 3] - s * BOX_W / 2) < TOL
                 for c in (0, 1) for s in (-1, 1)):
            side_faces.append(tag)
    g.physical.add(2, base_faces, name="soil_base")
    g.physical.add(2, side_faces, name="soil_sides")

    # --- the mixed-DOF interface: structure base into the raft ----------
    # Every grade-beam node (which includes all nine column bases) is
    # constrained to the raft's displacement field. rotational=True so a
    # column base is fixed against rotation, not pinned.
    g.constraints.embedded(host_label="raft", embedded_label="grade_beams",
                           rotational=True, name="base_tie")

    # --- gravity, as consistent nodal loads on one case -----------------
    col_p, beam_p, grade_p = (rect_section(COL_B, COL_B),
                              rect_section(BEAM_B, BEAM_H),
                              rect_section(GRADE_B, GRADE_H))
    with g.loads.case("dead"):
        g.loads.gravity("raft", g=(0.0, 0.0, -GRAV), density=RHO_CONC)
        g.loads.surface.traction(
            "slabs", vector=(0.0, 0.0, -(RHO_CONC * GRAV * T_SLAB + SDL_SLAB)))
        g.loads.surface.traction(
            "wall", vector=(0.0, 0.0, -RHO_CONC * GRAV * T_WALL))
        for name, props in (("columns", col_p), ("beams", beam_p),
                            ("grade_beams", grade_p)):
            g.loads.line(name, magnitude=RHO_CONC * GRAV * props["A"],
                         direction=(0.0, 0.0, -1.0))

    # --- mesh -----------------------------------------------------------
    lc = cfg["lc_shell"]
    for t in col.values():
        g.mesh.structured.set_transfinite_curve(t, max(1, round(STOREY / lc)) + 1)
    for t in list(bx.values()) + list(by.values()):
        g.mesh.structured.set_transfinite_curve(t, max(1, round(BAY / lc)) + 1)
    for t in list(slab.values()) + list(wall.values()):
        g.mesh.structured.set_transfinite_surface(t)
        g.mesh.structured.set_recombine(t)

    raft_faces = [t for (_, t) in g.model.queries.boundary([(3, raft_tag)])]
    f_d = g.mesh.field.distance(surfaces=raft_faces)
    f_t = g.mesh.field.threshold(f_d, size_min=cfg["lc_soil_near"],
                                 size_max=cfg["lc_soil_far"],
                                 dist_min=2.0, dist_max=12.0)
    g.mesh.field.set_background(f_t)
    g.mesh.generation.generate(3)

    fem = g.mesh.queries.get_fem_data()
    return fem, support_nodes(fem)


def support_nodes(fem) -> dict:
    """Fixed base, rollers on the sides — as node ids, not as groups.

    The box's bottom edges belong to BOTH faces and OpenSees refuses a
    second SP on a DOF already constrained, so the rollers have to be the
    sides *minus* the base. That subtraction is why the supports resolve to
    ids here instead of going straight through ``ops.fix(pg=...)``.
    """
    base = [int(i) for i in fem.nodes.select(pg="soil_base").ids]
    seen = set(base)
    sides = [int(i) for i in fem.nodes.select(pg="soil_sides").ids
             if int(i) not in seen]
    return {"base": base, "sides": sides}


def declare_model(fem, bc: dict):
    """Materials, sections, elements, supports, patterns and the chains."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)        # solid nodes infer ndf=3 from their elements

    soil_mat = ops.nDMaterial.ElasticIsotropic(E=E_SOIL, nu=NU_SOIL, rho=RHO_SOIL)
    conc_mat = ops.nDMaterial.ElasticIsotropic(E=E_CONC, nu=NU_CONC, rho=RHO_CONC)
    ops.element.FourNodeTetrahedron(pg="soil", material=soil_mat)
    ops.element.FourNodeTetrahedron(pg="raft", material=conc_mat)

    # forceBeamColumn (not elasticBeamColumn): the `line` slot reads
    # line_stations, which only a sectioned beam produces. The section is
    # elastic, so the model stays linear.
    t_col = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    t_horiz = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    for name, (b, h), transf in (("columns", (COL_B, COL_B), t_col),
                                 ("beams", (BEAM_B, BEAM_H), t_horiz),
                                 ("grade_beams", (GRADE_B, GRADE_H), t_horiz)):
        p = rect_section(b, h)
        sec = ops.section.Elastic(E=E_CONC, G=G_CONC, **p)
        ops.element.forceBeamColumn(
            pg=name, transf=transf,
            integration=ops.beamIntegration.Lobatto(section=sec, n_ip=5))

    # ASDShellQ4, not ShellMITC4: the upstream shells return a correctly
    # sized vector of ZEROS from ops.eleResponse(eid, "stresses"), so a
    # bench built from them records shell resultants that contour one
    # flat colour. Measured side by side on a loaded 1-element plate —
    # identical tip displacement, max|σ| = 0 vs 2.0e4. See
    # _response_catalog.ZERO_GAUSS_PROBE_CLASSES.
    ops.element.ASDShellQ4(pg="slabs", section=ops.section.ElasticMembranePlateSection(
        E=E_CONC, nu=NU_CONC, h=T_SLAB, rho=RHO_CONC))
    ops.element.ASDShellQ4(pg="wall", section=ops.section.ElasticMembranePlateSection(
        E=E_CONC, nu=NU_CONC, h=T_WALL, rho=RHO_CONC))

    ops.fix(nodes=bc["base"], dofs=(1, 1, 1))
    ops.fix(nodes=bc["sides"], dofs=(1, 1, 0))   # rollers on the lateral faces

    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.from_model("dead")
    # NOTE: the earthquake pattern is deliberately NOT declared here. It is
    # created after gravity, in solve() — see the comment at `loadConst`.

    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    ops.algorithm.Linear()          # the model is linear: one solve per step
    ops.integrator.LoadControl(dlam=1.0 / N_GRAV)
    ops.analysis.Static()
    return ops


def _expand(*names: str) -> tuple[str, ...]:
    return tuple(c for n in names for c in expand_shorthand(n, ndm=3, ndf=6))


def make_spec(fem, *, dt: float | None = None) -> ResolvedRecorderSpec:
    """One recorder spec covering every slot in the ADR 0098 §4 catalog.

    ``dt`` is the recording cadence (the dynamic stride); ``None`` records
    every step. The readers rebuild the same spec from ``model.h5``'s
    FEMData, so the transcode cache key matches the build's.
    """
    bc = support_nodes(fem)

    def eids(*pgs: str) -> np.ndarray:
        return np.concatenate([
            np.asarray(fem.elements.select(pg=p).ids, dtype=np.int64)
            for p in pgs])

    def rec(category, name, comps, *, cls=None, **ids):
        return ResolvedRecorderRecord(
            category=category, name=name, components=comps, dt=dt,
            n_steps=None, element_class_name=cls, **ids)

    return ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            # contour / vector / sand / deform poses, plus the plot series
            rec("nodes", "motion",
                _expand("displacement", "velocity", "acceleration"),
                node_ids=np.asarray(fem.nodes.ids, dtype=np.int64)),
            # the reactions slot
            rec("nodes", "reactions", _expand("reaction_force"),
                node_ids=np.asarray(bc["base"] + bc["sides"], dtype=np.int64)),
            # the gauss slot (and the contour's unaveraged twin)
            rec("gauss", "solids", _expand("stress"),
                cls="FourNodeTetrahedron", element_ids=eids("soil", "raft")),
            # The shell half of the gauss slot. A shell has no Cauchy
            # stress tensor: `stresses` returns its eight RESULTANTS, so
            # `stress_*` is a solid-only vocabulary and this is the
            # shell-side answer. Its own record, not a second pg on the one
            # above: the two carry different component sets, and a gauss
            # record is one `eleResponse` layout.
            #
            # This is what makes the fixture cover the multi-family case
            # the picker law (ADR 0098 A6.6 G4) is about — `soil+raft` and
            # `slabs+wall` must offer DIFFERENT component lists.
            rec("gauss", "shells", _expand("shell_resultant"),
                cls="ASDShellQ4", element_ids=eids("slabs", "wall")),
            # the line slot
            rec("line_stations", "frame",
                ("axial_force", "shear_y", "shear_z",
                 "torsion", "bending_moment_y", "bending_moment_z"),
                cls="ForceBeamColumn3d",
                element_ids=eids("columns", "beams", "grade_beams")),
        ),
    )


def open_results(results_dir: Path) -> tuple[Results, Results]:
    """``(run, modes)``: the recorder set, transcoded on open, and the modes.

    The one opener every bench consumer uses. The first call after a build
    transcodes the recorder files into ``cache/``; later calls reuse it.
    """
    stride = json.loads((results_dir / META).read_text())["stride"]
    model_h5 = results_dir / "model.h5"
    model = OpenSeesModel.from_h5(str(model_h5))
    fem = model.fem
    run = Results.from_recorders(
        make_spec(fem, dt=stride * DT), results_dir / OUT_DIR, fem=fem,
        model=model, stages=STAGES, cache_root=results_dir / CACHE_DIR)
    modes = Results.from_native(str(results_dir / MODES_H5), model=model)
    return run, modes


def solve(ops, fem, results_dir: Path, cfg: dict) -> None:
    """Gravity → modes → dynamic: the stages on recorders, the modes on
    ``capture_live_modes``, then one warm transcode."""
    ops.run()                                   # push the model into openseespy

    model_h5 = results_dir / "model.h5"
    model = OpenSeesModel.from_h5(str(model_h5))
    stride, n_steps = cfg["stride"], int(round(DURATION / DT))
    (results_dir / META).write_text(json.dumps(
        {"size": cfg["name"], "stride": stride}))
    spec = make_spec(fem, dt=stride * DT)
    with spec.emit_recorders(str(results_dir / OUT_DIR), ops=opspy,
                             model=model) as live:
        live.begin_stage(STAGES[0].name, kind=STAGES[0].kind)
        for k in range(N_GRAV):
            if opspy.analyze(1) != 0:
                raise RuntimeError(f"gravity did not converge at increment {k}")
        live.end_stage()

        # Freeze gravity and restart the clock. `loadConst` freezes the load
        # factor of EVERY pattern in the domain, so an earthquake pattern
        # that already existed here would be frozen at its t=0 value — the
        # run would complete, 150 steps and all, with the model standing
        # perfectly still. Hence the excitation is born below, after this
        # line, through raw openseespy: the bridge emits its whole deck at
        # ops.run(), so it cannot declare a pattern "later" than this.
        opspy.loadConst("-time", 0.0)

        accel = ground_motion()
        opspy.timeSeries("Path", GM_SERIES_TAG, "-dt", DT,
                         "-values", *(float(a) for a in accel))
        opspy.pattern("UniformExcitation", GM_PATTERN_TAG, 1,
                      "-accel", GM_SERIES_TAG)

        # Rayleigh from the real modes, not from guessed frequencies.
        lam = opspy.eigen(N_MODES)
        w = [float(x) ** 0.5 for x in lam]
        w_i, w_j = w[0], w[min(2, len(w) - 1)]
        zeta = 0.05
        a0 = zeta * 2.0 * w_i * w_j / (w_i + w_j)
        a1 = zeta * 2.0 / (w_i + w_j)
        opspy.rayleigh(a0, 0.0, 0.0, a1)
        print(f"[bench] periods: {', '.join(f'{2 * np.pi / x:.3f}' for x in w)} s")
        print(f"[bench] rayleigh a0={a0:.5f} a1={a1:.6f}")

        # The modes, into their own file, before wipeAnalysis drops the
        # system the eigen solve ran on.
        capture_live_modes(str(model_h5), N_MODES,
                           str(results_dir / MODES_H5), ops=opspy)

        opspy.wipeAnalysis()
        opspy.constraints("Transformation")
        opspy.numberer("RCM")
        opspy.system("UmfPack")
        opspy.test("NormDispIncr", 1e-8, 10)
        opspy.algorithm("Linear")
        opspy.integrator("Newmark", 0.5, 0.25)
        opspy.analysis("Transient")

        live.begin_stage(STAGES[1].name, kind=STAGES[1].kind)
        for k in range(n_steps):
            if opspy.analyze(1, DT) != 0:
                raise RuntimeError(f"transient did not converge at step {k}")
        live.end_stage()

    # One transcode now, so every reader opens warm.
    run, modes = open_results(results_dir)
    run.close()
    modes.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=sorted(SIZES), default="small")
    # Default is the CURRENT directory, not a path beside this script: the
    # habitat case runner invokes `python build.py` with no arguments and
    # cwd set to the case's results/ folder, so "here" is the case.
    ap.add_argument("--out", type=Path, default=Path.cwd())
    args = ap.parse_args()

    cfg = dict(SIZES[args.size], name=args.size)
    args.out.mkdir(parents=True, exist_ok=True)
    model_h5 = args.out / "model.h5"

    with apeGmsh(model_name="ssi_frame_wall", verbose=False) as g:
        fem, bc = build_mesh(g, cfg)
        print(f"[bench] size={args.size}  nodes={fem.info.n_nodes}  "
              f"elements={fem.info.n_elems}")
        ops = declare_model(fem, bc)
        ops.h5(str(model_h5))
        solve(ops, fem, args.out, cfg)

    print(f"[bench] wrote {model_h5}")
    print(f"[bench] wrote {args.out / OUT_DIR} (recorder set) and "
          f"{args.out / MODES_H5}")
    print("[bench] open it with (from this folder):")
    print("    from build import open_results")
    print(f"    run, modes = open_results(Path(r'{args.out}'))")
    print("    run.viewer()        # gravity + dynamic; modes.viewer() for the modes")


if __name__ == "__main__":
    main()
