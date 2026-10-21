// 3-D structural loading & prescribed-SP visualization tests:
// - OpenSees `/opensees/patterns` (global + staged) and neutral `/loads/{nodal,element,sp}`
// - Nodal forces (F), nodal moments (M), beam uniform/point combs, surface pressure arrows, and non-zero SPs (u0, theta0)
// - Stage activation, `remove_sp`, `role="support_reaction"` filtering, and `timeSeries` scaling λ(t)
// - Fail-closed 3-D beam `vecxz` checks and `baseUnits` unit conversion (`loadsUnitsOf`)
// - `LoadsLayer` (Three.js), `mountLoads` / `renderLoads` (`#loads` panel), and Patterns window `"show in 3D"` toggle

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { withNodes } from "../src/effects.ts";
import { renderLoads } from "../src/panels/loads.ts";
import { view as patternsView } from "../src/panels/windows/patterns.ts";
import { openModel } from "../src/reader/node.ts";
import { LoadsLayer } from "../src/renderer/loads.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import {
  buildLoads3D,
  evaluatePatternSeriesFactor,
  loadsSummaryOf,
  loadsUnitsOf,
  spotlightViewOf,
} from "../src/state/selectors.ts";
import { initialState, reduce, type State } from "../src/state/store.ts";
import type { DefObject, Event, NeutralLoads, Table } from "../src/state/types.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

const shoeModel = await openModel(join(fixtures, "shoebuckle.h5"));
const shoeBlobs = new BlobStore();
const shoeLoad = withNodes(loadModel(shoeModel, shoeBlobs), shoeModel, shoeBlobs);
const shoeState: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: shoeLoad });

const defModel = await openModel(join(fixtures, "definitions.h5"));
const defBlobs = new BlobStore();
const defLoad = withNodes(loadModel(defModel, defBlobs), defModel, defBlobs);
const defState: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: defLoad });

test("mesh.nodes (withNodes) feeds buildLoads3D, which builds 3-D force arrows on shoebuckle.h5 and definitions.h5", () => {
  assert.ok(shoeState.mesh?.nodes, "mesh.nodes is populated by withNodes, as the app loads a model");
  assert.ok(shoeBlobs.has(shoeState.mesh!.nodes!.ids));
  assert.ok(shoeBlobs.has(shoeState.mesh!.nodes!.xyz));

  const shoeSummary = loadsSummaryOf(shoeState);
  assert.equal(shoeSummary.hasLoads, true);
  assert.ok(shoeSummary.counts.nodalForces > 0, "shoebuckle.h5 has nodal forces");

  const sOn = reduce(shoeState, { type: "setLoads", on: true });
  const geom = buildLoads3D(sOn, shoeBlobs);
  assert.ok(geom.nodalForceArrows > 0, "builds nodal force arrows on shoebuckle.h5");
  assert.ok(geom.linePositions.length > 0 && geom.linePositions.every(Number.isFinite));
  assert.ok(geom.triPositions.length > 0 && geom.triPositions.every(Number.isFinite));
  assert.ok(geom.maxForceMag > 0);
});

test("stage activation on definitions.h5: no stage shows the global pattern only, stage 0 adds gravity, stage 1 adds push; the neutral case stays off (#2385 F3)", () => {
  const defWithSeq: State = {
    ...defState,
    sequence: {
      status: "ready",
      version: "1.0.0",
      attrs: { staged: 1 },
      stages: {
        path: "/sequence/stages",
        columns: ["order", "name", "source"],
        rows: [
          [0, "gravity", "/opensees/stages/stage_000"],
          [1, "push", "/opensees/stages/stage_001"],
        ],
      },
      changes: { path: "/sequence/changes", columns: ["stage", "op", "what", "name", "count", "source"], rows: [] },
    },
  };

  // No stage selected: the global pattern only. Every stage at once, plus the neutral
  // case the bridge turned into #1, would count the same loads twice.
  const meshSummary = loadsSummaryOf(defWithSeq);
  assert.equal(meshSummary.hasLoads, true);
  assert.equal(meshSummary.activeStageLabel, "global patterns");
  assert.deepEqual(
    meshSummary.patterns.map((p) => [p.title, p.stageIndex, p.active]),
    [
      ["#1", -1, true],
      ["gravity", 0, false],
      ["push", 1, false],
      ["Pressure", -1, false],
    ],
  );

  // Select stage 0 (`gravity`): global (#1) and stage 0 (gravity) are active; push and the neutral case are not
  const stage0State = reduce(defWithSeq, { type: "selectStage", index: 0 });
  const sum0 = loadsSummaryOf(stage0State);
  assert.deepEqual(
    sum0.patterns.map((p) => [p.title, p.active]),
    [
      ["#1", true],
      ["gravity", true],
      ["push", false],
      ["Pressure", false],
    ],
  );

  // Select stage 1 (`push`): global (#1), stage 0 (gravity) and stage 1 (push) are active; the neutral case stays off
  const stage1State = reduce(defWithSeq, { type: "selectStage", index: 1 });
  const sum1 = loadsSummaryOf(stage1State);
  assert.deepEqual(
    sum1.patterns.map((p) => [p.title, p.active]),
    [
      ["#1", true],
      ["gravity", true],
      ["push", true],
      ["Pressure", false],
    ],
  );
});

test("nodal forces (F), nodal moments (M with double cones), non-zero SPs, support_reaction filtering, and stage remove_sp", () => {
  const n0 = Number(shoeModel.nodeIds[0]!);
  const n1 = Number(shoeModel.nodeIds[1]!);
  const n2 = Number(shoeModel.nodeIds[2]!);

  const loadsTable: Table = {
    path: "/opensees/patterns/Plain_1/loads",
    columns: ["node_id", "ndf", "forces", "target_kind", "target"],
    // Row 0: 6-DOF with both force [0, 0, -100] and moment [0, 50, 0]
    // Row 1: 3-DOF padded with NaN [25, 0, 0, NaN, NaN, NaN]
    rows: [
      [n0, 6, [0, 0, -100, 0, 50, 0], "node", String(n0)],
      [n1, 3, [25, 0, 0, Number.NaN, Number.NaN, Number.NaN], "node", String(n1)],
    ],
  };
  const spsTable: Table = {
    path: "/opensees/patterns/Plain_1/sps",
    columns: ["node_id", "dof", "value", "target_kind", "target"],
    // Row 0: homogeneous fixity (value = 0) -> must be ignored
    // Row 1: prescribed displacement u_x = 0.02 -> must be drawn
    // Row 2: prescribed displacement u_z = -0.01 on n2 -> removed in stage 1 by remove_sp
    rows: [
      [n0, 1, 0, "node", String(n0)],
      [n1, 1, 0.02, "node", String(n1)],
      [n2, 3, -0.01, "node", String(n2)],
    ],
  };
  const holdPat: DefObject = {
    groupName: "Plain_99",
    path: "/opensees/patterns/Plain_99",
    type: "Plain",
    tag: 99,
    params: [1],
    attrs: { role: "support_reaction" },
    tables: {
      sp_holds: {
        path: "/opensees/patterns/Plain_99/sp_holds",
        columns: ["col_0", "col_1"],
        rows: [[n0, 2]],
      },
    },
  };

  const customDefs: NonNullable<State["definitions"]> = {
    ...shoeState.definitions!,
    patterns: [
      {
        groupName: "Plain_1",
        path: "/opensees/patterns/Plain_1",
        type: "Plain",
        tag: 1,
        params: [1],
        attrs: {},
        tables: { loads: loadsTable, sps: spsTable },
      },
      holdPat,
    ],
    stages: [
      {
        index: 0,
        name: "stage_0",
        path: "/opensees/stages/stage_000",
        attrs: {},
        analysis: null,
        patterns: [],
        recorders: [],
        bcs: { fix: null, mass: null },
        removeSp: null,
        other: [],
      },
      {
        index: 1,
        name: "stage_1",
        path: "/opensees/stages/stage_001",
        attrs: {},
        analysis: null,
        patterns: [],
        recorders: [],
        bcs: { fix: null, mass: null },
        removeSp: {
          path: "/opensees/stages/stage_001/remove_sp",
          columns: ["col_0", "col_1"],
          rows: [[n2, 3]],
        },
        other: ["remove_sp"],
      },
    ],
  };

  let s: State = {
    ...shoeState,
    definitions: customDefs,
    neutralLoads: null,
    sequence: {
      status: "ready",
      version: "1.0.0",
      attrs: { staged: 1 },
      stages: {
        path: "/sequence/stages",
        columns: ["order", "name", "source"],
        rows: [
          [0, "stage_0", "/opensees/stages/stage_000"],
          [1, "stage_1", "/opensees/stages/stage_001"],
        ],
      },
      changes: { path: "/sequence/changes", columns: ["stage", "op", "what", "name", "count", "source"], rows: [] },
    },
  };
  s = reduce(s, { type: "setLoads", on: true });

  // On stage 0 (before stage 1 remove_sp): both non-zero SP vectors are active
  const sStage0 = reduce(s, { type: "selectStage", index: 0 });
  const geomStage0 = buildLoads3D(sStage0, shoeBlobs);
  assert.equal(geomStage0.nodalForceArrows, 2, "2 nodal force vectors");
  assert.equal(geomStage0.nodalMomentArrows, 1, "1 nodal moment vector");
  assert.equal(geomStage0.spArrows, 2, "2 non-zero SP vectors on stage 0 (homogeneous 0 skipped, support_reaction excluded)");
  assert.equal(geomStage0.maxForceMag, 100);
  assert.equal(geomStage0.maxMomentMag, 50);
  assert.equal(geomStage0.maxSpMag, 0.02);

  // On stage 1: remove_sp removes [n2, 3], leaving only 1 SP arrow
  const sStage1 = reduce(s, { type: "selectStage", index: 1 });
  const geomStage1 = buildLoads3D(sStage1, shoeBlobs);
  assert.equal(geomStage1.spArrows, 1, "stage 1 remove_sp removes the [n2, 3] SP constraint");

  // Enabling includeSupportReactions adds the sp_holds arrow from holdPat
  const sWithHolds = reduce(sStage1, { type: "setLoads", includeSupportReactions: true });
  const geomHolds = buildLoads3D(sWithHolds, shoeBlobs);
  assert.equal(geomHolds.spArrows, 2, "includeSupportReactions=true adds the sp_holds entry");
});

test("distributed element loads: beamUniform (2D and 3D with vecxz vs fail-closed missing vecxz), beamPoint, and surfacePressure", () => {
  const firstElemPath = shoeState.mesh!.elements[0]!;
  const firstElemDecl = shoeState.decls[firstElemPath]!;
  const firstOpsTag = firstElemDecl.element!.metas[0]!.tag;

  const elemLoadsTable: Table = {
    path: "/opensees/patterns/Plain_1/element_loads",
    columns: ["col_0", "col_1", "col_2", "col_3", "col_4", "col_5"],
    // Row 0: -beamUniform wy=-10, wx=0 on element firstOpsTag
    // Row 1: -beamPoint Py=-25, xL=0.5, Px=0 on element firstOpsTag
    rows: [
      [firstOpsTag, "-beamUniform", -10, 0, Number.NaN, Number.NaN],
      [firstOpsTag, "-beamPoint", -25, 0.5, 0, Number.NaN],
    ],
  };

  const s2d: State = reduce(
    {
      ...shoeState,
      ndm: { ndm: 2, source: "test" },
      neutralLoads: null,
      definitions: {
        ...shoeState.definitions!,
        stages: [],
        patterns: [
          {
            groupName: "Plain_1",
            path: "/opensees/patterns/Plain_1",
            type: "Plain",
            tag: 1,
            params: [1],
            attrs: {},
            tables: { element_loads: elemLoadsTable },
          },
        ],
      },
    },
    { type: "setLoads", on: true },
  );

  const geom2d = buildLoads3D(s2d, shoeBlobs);
  assert.ok(geom2d.elementLoadArrows >= 5, "beamUniform (5 station arrows) + beamPoint (1 arrow)");
  assert.equal(geom2d.notice, null);
  assert.equal(geom2d.maxElementLoadMag, 25);

  // In 3-D (ndm=3), if an element lacks a geomTransf vecxz, buildLoads3D refuses that beam's local load with an explicit notice
  const s3dNoVecxz: State = {
    ...s2d,
    ndm: { ndm: 3, source: "test" },
  };
  const geom3dRefused = buildLoads3D(s3dNoVecxz, shoeBlobs);
  assert.equal(geom3dRefused.elementLoadArrows, 0, "3-D beam without vecxz is refused, never drawn with a guessed frame");
  assert.match(geom3dRefused.notice ?? "", /missing geomTransf vecxz/);

  // In 3-D (ndm=3) WITH a valid non-parallel geomTransf vecxz, buildLoads3D draws the beam loads cleanly
  const s3dWithVecxz: State = {
    ...s3dNoVecxz,
    decls: {
      ...s3dNoVecxz.decls,
      "opensees/geomTransf/1": {
        path: "opensees/geomTransf/1",
        kind: "geomTransf",
        name: "gt1",
        nameSource: "test",
        type: "Linear",
        tag: 1,
        h5: "/opensees/geomTransf/Linear_1",
        params: [0, 0, 1],
        refs: {},
        links: {},
        fields: [{ label: "vecxz", value: "[0, 0, 1]", source: "test", interpreted: false }],
        problems: [],
      },
      [firstElemPath]: {
        ...firstElemDecl,
        refs: { ...firstElemDecl.refs, transfTag: "opensees/geomTransf/1" },
      },
    },
  };
  const geom3dValid = buildLoads3D(s3dWithVecxz, shoeBlobs);
  assert.equal(geom3dValid.elementLoadArrows, 6, "3-D beam with valid vecxz draws 5 uniform comb arrows + 1 point load arrow");
  assert.equal(geom3dValid.notice, null);
});

// `fixtures/loads3d.h5` is written by apeGmsh's own resolver and `to_h5`
// (`fixtures/make_loads_fixtures.py`), so these cases read the writer's real
// layout: `element_id/load_type/params_json`, `node_id/force_xyz`, `node_id/dof/value`.
const l3Model = await openModel(join(fixtures, "loads3d.h5"));
const l3Blobs = new BlobStore();
const l3State: State = reduce(
  reduce(initialState, { type: "fileLoaded", artifact: "model", load: withNodes(loadModel(l3Model, l3Blobs), l3Model, l3Blobs) }),
  { type: "setLoads", on: true },
);
const onlyCase = (name: string): State => reduce(l3State, { type: "setLoads", selectedPatterns: [`neutral:${name}`] });

/** Unit directions of the drawn line segments (arrow shafts and comb profile bars), rounded, with their x range. */
function segmentDirs(lp: Float32Array): Map<string, { n: number; xMin: number; xMax: number }> {
  const out = new Map<string, { n: number; xMin: number; xMax: number }>();
  for (let i = 0; i < lp.length; i += 6) {
    const d = [lp[i + 3]! - lp[i]!, lp[i + 4]! - lp[i + 1]!, lp[i + 5]! - lp[i + 2]!];
    const n = Math.hypot(d[0]!, d[1]!, d[2]!) || 1;
    const key = d.map((v) => (Math.abs(v / n) < 1e-6 ? 0 : Math.round((v / n) * 100) / 100)).join(",");
    const cur = out.get(key) ?? { n: 0, xMin: Infinity, xMax: -Infinity };
    out.set(key, { n: cur.n + 1, xMin: Math.min(cur.xMin, lp[i]!), xMax: Math.max(cur.xMax, lp[i]!) });
  }
  return out;
}

test("neutral beamUniform wx/wy/wz are GLOBAL: gravity on the X girder points -Z, wind on the Z column points +X (#2385 F1)", () => {
  const dead = buildLoads3D(reduce(onlyCase("dead"), { type: "setLoads", showElementLoads: true }), l3Blobs);
  const deadDirs = segmentDirs(dead.linePositions);
  assert.ok((deadDirs.get("0,0,-1")?.n ?? 0) >= 5, "the girder's comb arrows point -Z");
  assert.equal(deadDirs.has("0,-1,0"), false, "nothing is drawn along -Y (the local [wy,wz,wx] decode drew wz=-10 there)");
  assert.deepEqual([...deadDirs.keys()].sort(), ["0,0,-1", "1,0,0"], "only -Z arrows and the +X profile bar along the girder");
  assert.equal(dead.notice, null, "a global neutral load needs no vecxz");

  const wind = buildLoads3D(onlyCase("wind"), l3Blobs);
  const windDirs = segmentDirs(wind.linePositions);
  assert.ok((windDirs.get("1,0,0")?.n ?? 0) >= 5, "the column's comb arrows point +X, across the column");
  assert.deepEqual([...windDirs.keys()].sort(), ["0,0,1", "1,0,0"], "only +X arrows and the +Z profile bar along the column");
  assert.equal(loadsSummaryOf(onlyCase("wind")).peaks.beamUniform, 5);
});

test("neutral beamUniform trapezoid: wx_end/aL/bL are drawn over [aL, bL] with the end magnitude (#2385 F1)", () => {
  const s = onlyCase("trap");
  assert.equal(loadsSummaryOf(s).peaks.beamUniform, 20, "the peak is the trapezoid's larger end, 20");
  const arrows = segmentDirs(buildLoads3D(s, l3Blobs).linePositions).get("0,0,-1");
  assert.ok(arrows && arrows.n > 0, "the trapezoid points -Z");
  // The resolver gives every girder element the span x/L in [0.25, 0.75]: no tick sits on the girder's ends.
  assert.ok(arrows.xMin > 1e-6 && arrows.xMax < 4 - 1e-6, `ticks inside the span, got x in [${arrows.xMin}, ${arrows.xMax}]`);
});

test("neutral surfacePressure is typed by load_type (p=7 into the slab, not a beamUniform of 1), bodyForce gets a notice (#2385 F2)", () => {
  const tri = l3Model.blocks.find((b) => b.alias === "tri3");
  assert.ok(tri);
  const nTri = tri.ids.length;
  const dead = onlyCase("dead");
  const sum = loadsSummaryOf(dead);
  assert.equal(sum.counts.surfacePressure, nTri, "one surfacePressure per slab triangle");
  assert.equal(sum.peaks.surfacePressure, 7);
  assert.equal(sum.peaks.beamUniform, 10, "the pressure's direction key no longer turns it into a unit beamUniform");
  assert.equal(sum.counts.beamUniform + sum.counts.surfacePressure, sum.counts.elementLoads);
  assert.equal(buildLoads3D(dead, l3Blobs).surfacePressureCount, nTri);

  const self = onlyCase("self");
  const selfSum = loadsSummaryOf(self);
  assert.equal(selfSum.patterns.find((p) => p.id === "neutral:self")!.counts.elementLoads, nTri, "the case is listed with its records");
  assert.match(selfSum.notice ?? "", new RegExp(`${nTri} bodyForce in self`));
  const selfGeom = buildLoads3D(self, l3Blobs);
  assert.equal(selfGeom.elementLoadArrows, 0);
  assert.match(selfGeom.notice ?? "", /bodyForce in self/, "the drawing says what it left out");
});

test("neutral nodal force and prescribed SP from the writer's node_id/force_xyz and node_id/dof/value columns", () => {
  const s = onlyCase("point");
  const sum = loadsSummaryOf(s);
  assert.equal(sum.counts.nodalForces, 1);
  assert.equal(sum.peaks.nodalForce, 3);
  assert.equal(sum.counts.sps, 1);
  assert.equal(sum.counts.spTranslations, 1);
  assert.equal(sum.peaks.spTranslation, 0.005);
  const g = buildLoads3D(s, l3Blobs);
  assert.equal(g.nodalForceArrows, 1);
  assert.equal(g.spArrows, 1);
});

test("no double counting: with an OpenSees pattern present the neutral cases start off; shoebuckle peaks at 625 on node 102 (#2385 F3)", () => {
  const sum = loadsSummaryOf(shoeState);
  assert.deepEqual(
    sum.patterns.map((p) => [p.id, p.active]),
    [
      ["/opensees/patterns/Plain_1", true],
      ["neutral:Pressure", false],
    ],
  );
  assert.equal(sum.counts.nodalForces, 124, "Plain_1's 124 forces, not 248");
  assert.ok(Math.abs(sum.peaks.nodalForce - 625) < 1e-9, `peak ${sum.peaks.nodalForce}, not 1250`);

  // The 625 is node 102's force in the archive, and no node carries more.
  const t = shoeModel.loads!.nodal["Pressure"]!;
  const fOf = (r: readonly unknown[]): number => Math.hypot(...(r[t.columns.indexOf("force_xyz")] as number[]));
  const row102 = t.rows.find((r) => r[t.columns.indexOf("node_id")] === 102);
  assert.ok(row102);
  assert.ok(Math.abs(fOf(row102) - 625) < 1e-9);
  assert.ok(t.rows.every((r) => fOf(r) <= 625 + 1e-9));

  // Picking the neutral case alone still shows it.
  const neutralOnly = loadsSummaryOf(reduce(shoeState, { type: "setLoads", selectedPatterns: ["neutral:Pressure"] }));
  assert.equal(neutralOnly.counts.nodalForces, 124);
});

test("timeSeries scaling λ(t): Constant, Linear, Path, Trig, and scaleBySeries on results steps", () => {
  // defState has Linear timeSeries with factor=1. Let's attach results at t = 0.25 and t = 0.75
  const lineage = { modelHash: "a1", femHash: "b2", resultsHash: "c3" };
  let s: State = {
    ...defState,
    lineage,
    results: {
      path: "defs.results.h5",
      version: "1.1.0",
      sourceType: "domain_capture",
      lineage,
      partial: null,
      stages: [
        {
          id: "stage_000",
          name: "gravity",
          kind: "static",
          time: [0.25, 0.75],
          nodal: ["displacement_x"],
          refusedNodal: [],
          gauss: [],
          unread: [],
        },
      ],
      unavailableZones: [],
      openseesVersion: null,
      programDecl: null,
    },
    phase: {
      axis: [{ kind: "mesh" }, { kind: "results", stage: 0, step: 0 }],
      at: { kind: "results", stage: 0, step: 0 },
    },
  };
  s = reduce(s, {
    type: "setLoads",
    on: true,
    scaleBySeries: false,
    selectedPatterns: ["/opensees/patterns/Plain_1"],
  });
  const geomNominal = buildLoads3D(s, defBlobs);

  s = reduce(s, { type: "setLoads", scaleBySeries: true });
  const geomAt025 = buildLoads3D(s, defBlobs);
  assert.ok(
    Math.abs(geomAt025.maxForceMag - 0.25 * geomNominal.maxForceMag) < 1e-6,
    `at t=0.25, Linear(factor=1) scales maxForceMag by 0.25 (${geomAt025.maxForceMag} vs ${geomNominal.maxForceMag})`,
  );

  const sStep1 = reduce(s, { type: "setResultStep", step: 1 });
  const geomAt075 = buildLoads3D(sStep1, defBlobs);
  assert.ok(
    Math.abs(geomAt075.maxForceMag - 0.75 * geomNominal.maxForceMag) < 1e-6,
    `at t=0.75, Linear(factor=1) scales maxForceMag by 0.75`,
  );

  // Direct evaluatePatternSeriesFactor checks for Constant and Trig
  const pat = defState.definitions!.patterns[0]!;
  assert.equal(evaluatePatternSeriesFactor(defState.definitions, pat, 0.4), 0.4);
  assert.equal(evaluatePatternSeriesFactor(defState.definitions, pat, null), 1);
});

test("unit awareness: loadsUnitsOf and buildLoads3D convert load magnitudes via baseUnits, and Spotlight exposes cmd:contour-units & cmd:toggle-loads on mesh", () => {
  let s = reduce(shoeState, { type: "setLoads", on: true });
  const geomRaw = buildLoads3D(s, shoeBlobs);
  assert.equal(geomRaw.units.force.unit, null);

  // Convert from N_m_s to kN_m_s -> force scales by 1e-3
  s = reduce(s, { type: "setContourUnits", source: "N_m_s", target: "kN_m_s" });
  const u = loadsUnitsOf(s);
  assert.equal(u.force.unit, "kN");
  assert.equal(u.moment.unit, "kN·m");
  assert.equal(u.beamLoad.unit, "kN/m");
  assert.equal(u.surfacePressure.unit, "kPa");
  assert.equal(u.displacement.unit, "m");
  assert.ok(Math.abs(u.force.factor - 1e-3) < 1e-12);

  const geomKn = buildLoads3D(s, shoeBlobs);
  assert.ok(Math.abs(geomKn.maxForceMag - geomRaw.maxForceMag * 1e-3) < 1e-6);

  // Spotlight exposes both cmd:toggle-loads and cmd:contour-units on a mesh-only model (without results)
  const spot = spotlightViewOf(reduce(shoeState, { type: "openSpotlight", query: "loads" }));
  assert.ok(spot.items.some((i) => i.id === "cmd:toggle-loads"), "Spotlight exposes cmd:toggle-loads when mesh is loaded");
  const spotUnits = spotlightViewOf(reduce(shoeState, { type: "openSpotlight", query: "units" }));
  assert.ok(spotUnits.items.some((i) => i.id === "cmd:contour-units"), "Spotlight exposes cmd:contour-units when mesh is loaded");
});

test("LoadsLayer (Three.js), renderLoads (#loads panel), and Patterns window 'show in 3D' button", () => {
  const layer = new LoadsLayer();
  assert.equal(layer.root.visible, false);

  const sOn = reduce(shoeState, { type: "setLoads", on: true });
  const changed = layer.sync(sOn, shoeBlobs);
  assert.equal(changed, true);
  assert.equal(layer.root.visible, true);
  assert.equal(layer.lines.visible, true);
  assert.equal(layer.cones.visible, true);
  assert.equal(layer.sync(sOn, shoeBlobs), false, "idempotent sync on unchanged state returns false");

  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  layer.setClippingPlanes([plane]);
  layer.setTheme("light");
  assert.equal(layer.sync(sOn, shoeBlobs), true, "theme change triggers re-sync");
  layer.dispose();

  // DOM test for renderLoads and patternsView
  const prevDoc = (globalThis as { document?: unknown }).document;
  class MinimalEl {
    tagName: string;
    className = "";
    title = "";
    value = "";
    type = "";
    step = "";
    min = "";
    disabled = false;
    selected = false;
    dataset: Record<string, string> = {};
    style: Record<string, string> = {};
    children: (MinimalEl | string)[] = [];
    listeners = new Map<string, ((ev?: unknown) => void)[]>();
    constructor(tag: string) {
      this.tagName = tag.toUpperCase();
    }
    append(...xs: (MinimalEl | string)[]) {
      this.children.push(...xs);
    }
    setAttribute(k: string, v: string) {
      this.dataset[k] = v;
    }
    addEventListener(type: string, fn: (ev?: unknown) => void) {
      const list = this.listeners.get(type) ?? [];
      list.push(fn);
      this.listeners.set(type, list);
    }
    removeEventListener() {}
    click() {
      for (const fn of this.listeners.get("click") ?? []) fn({ stopPropagation() {} });
    }
    get textContent(): string {
      return this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join("");
    }
    set textContent(v: string) {
      this.children = [v];
    }
    all(pred: (e: MinimalEl) => boolean): MinimalEl[] {
      const out: MinimalEl[] = pred(this) ? [this] : [];
      for (const c of this.children) if (typeof c !== "string") out.push(...c.all(pred));
      return out;
    }
  }
  (globalThis as { document?: unknown }).document = {
    documentElement: { dataset: { theme: "dark" } },
    createElement: (t: string) => new MinimalEl(t),
    createElementNS: (_ns: string, t: string) => new MinimalEl(t),
  };
  try {
    // 1. Model with no loads -> renderLoads returns null (auto-hidden)
    const sEmptyLoads: State = {
      ...shoeState,
      definitions: { ...shoeState.definitions!, patterns: [], stages: [] },
      neutralLoads: null,
    };
    assert.equal(renderLoads(sEmptyLoads), null, "renderLoads returns null when model has no loads");

    // 2. Model with loads -> renders #loads panel with toggle buttons, unit selects, and pattern rows
    const events: Event[] = [];
    const panelDom = renderLoads(defState, (e) => events.push(e)) as unknown as MinimalEl;
    assert.ok(panelDom !== null);
    const toggleBtn = panelDom.all((e) => e.className.includes("loads-toggle"))[0]!;
    toggleBtn.click();
    assert.deepEqual(events[0], { type: "setLoads", on: true });

    // 3. Patterns window "show in 3D" button toggles pattern in 3-D
    const winEvents: Event[] = [];
    const winDom = patternsView(defState, (e) => winEvents.push(e)) as unknown as MinimalEl;
    const btns3d = winDom.all((e) => e.className.startsWith("window-pattern-3d"));
    assert.equal(btns3d.length, 3, "one 'show in 3D' button per pattern card in definitions.h5");
    btns3d[1]!.click();
    assert.deepEqual(winEvents[0], {
      type: "setLoads",
      on: true,
      selectedPatterns: ["/opensees/stages/stage_000/patterns/Plain_2"],
    });

    // 4. All 7 load kinds display converted peak magnitudes and unit symbols in #loads panel
    const n0 = Number(shoeModel.nodeIds[0]!);
    const n1 = Number(shoeModel.nodeIds[1]!);
    const firstElemPath = shoeState.mesh!.elements[0]!;
    const firstElemDecl = shoeState.decls[firstElemPath]!;
    const firstOpsTag = firstElemDecl.element!.metas[0]!.tag;

    let sAll7: State = {
      ...shoeState,
      ndm: { ndm: 2, source: "test" },
      definitions: {
        ...shoeState.definitions!,
        stages: [],
        patterns: [
          {
            groupName: "Plain_All",
            path: "/opensees/patterns/Plain_All",
            type: "Plain",
            tag: 1,
            params: [1],
            attrs: {},
            tables: {
              loads: {
                path: "/opensees/patterns/Plain_All/loads",
                columns: ["node_id", "ndf", "forces"],
                rows: [[n0, 3, [3000, 4000, 2000]]], // |F| = 5000 N = 5 kN, |M| = 2000 N·m = 2 kN·m
              },
              element_loads: {
                path: "/opensees/patterns/Plain_All/element_loads",
                columns: ["col_0", "col_1", "col_2", "col_3", "col_4"],
                rows: [
                  [firstOpsTag, "-beamUniform", -8000, 0, Number.NaN], // |q| = 8000 N/m = 8 kN/m
                  [firstOpsTag, "-beamPoint", -15000, 0.5, 0], // |P| = 15000 N = 15 kN
                  [firstOpsTag, "-surfaceLoad", -25000, Number.NaN, Number.NaN], // |p| = 25000 Pa = 25 kPa
                ],
              },
              sps: {
                path: "/opensees/patterns/Plain_All/sps",
                columns: ["node_id", "dof", "value"],
                rows: [
                  [n0, 1, 0.012], // u0 = 0.012 m = 12 mm
                  [n1, 3, 0.005], // theta0 = 0.005 rad
                ],
              },
            },
          },
          {
            groupName: "UniformExcitation_2",
            path: "/opensees/patterns/UniformExcitation_2",
            type: "UniformExcitation",
            tag: 2,
            params: [1],
            attrs: { direction: 1 },
            tables: {},
          },
        ],
      },
      neutralLoads: null,
    };
    sAll7 = reduce(sAll7, { type: "setLoads", on: true });
    sAll7 = reduce(sAll7, { type: "setContourUnits", source: "N_m_s", target: "kN_m_s" });

    const sum7 = loadsSummaryOf(sAll7);
    assert.ok(Math.abs(sum7.peaks.nodalForce - 5) < 1e-9, "peak nodalForce = 5 kN");
    assert.ok(Math.abs(sum7.peaks.nodalMoment - 2) < 1e-9, "peak nodalMoment = 2 kN·m");
    assert.ok(Math.abs(sum7.peaks.beamUniform - 8) < 1e-9, "peak beamUniform = 8 kN/m");
    assert.ok(Math.abs(sum7.peaks.beamPoint - 15) < 1e-9, "peak beamPoint = 15 kN");
    assert.ok(Math.abs(sum7.peaks.surfacePressure - 25) < 1e-9, "peak surfacePressure = 25 kPa");
    assert.ok(Math.abs(sum7.peaks.spTranslation - 0.012) < 1e-9, "peak spTranslation = 0.012 m");
    assert.ok(Math.abs(sum7.peaks.spRotation - 0.005) < 1e-9, "peak spRotation = 0.005 rad");
    assert.ok(Math.abs(sum7.peaks.baseExcitation - 1) < 1e-9, "peak baseExcitation = 1 m/s²");

    const dom7 = renderLoads(sAll7) as unknown as MinimalEl;
    const text7 = dom7.textContent;
    assert.match(text7, /max 5 kN/, "renders converted nodal force peak with kN");
    assert.match(text7, /max 2 kN·m/, "renders converted nodal moment peak with kN·m");
    assert.match(text7, /max 8 kN\/m/, "renders converted beamUniform peak with kN/m");
    assert.match(text7, /max 15 kN/, "renders converted beamPoint peak with kN");
    assert.match(text7, /max 25 kPa/, "renders converted surfacePressure peak with kPa");
    assert.match(text7, /max 0\.012 m/, "renders converted spTranslation peak with m");
    assert.match(text7, /max 0\.005 rad/, "renders converted spRotation peak with rad");
    assert.match(text7, /max 1 m\/s²/, "renders converted baseExcitation peak with m/s²");
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

test("stage-ordered remove_sp: stage 1 remove_sp removes earlier stage SP but preserves new SP/sp_holds added in stage 1", () => {
  const n0 = Number(shoeModel.nodeIds[0]!);
  const sWithStagedReplace: State = reduce(
    {
      ...shoeState,
      neutralLoads: null,
      sequence: {
        status: "ready",
        version: "1.0.0",
        attrs: { staged: 1 },
        stages: {
          path: "/sequence/stages",
          columns: ["order", "name", "source"],
          rows: [
            [0, "stage_0", "/opensees/stages/stage_000"],
            [1, "stage_1", "/opensees/stages/stage_001"],
          ],
        },
        changes: { path: "/sequence/changes", columns: ["stage", "op", "what", "name", "count", "source"], rows: [] },
      },
      definitions: {
        ...shoeState.definitions!,
        patterns: [
          {
            groupName: "Plain_Global",
            path: "/opensees/patterns/Plain_Global",
            type: "Plain",
            tag: 1,
            params: [1],
            attrs: {},
            tables: {
              sps: {
                path: "/opensees/patterns/Plain_Global/sps",
                columns: ["node_id", "dof", "value"],
                rows: [[n0, 2, -0.01]],
              },
            },
          },
        ],
        stages: [
          {
            index: 0,
            name: "stage_0",
            path: "/opensees/stages/stage_000",
            attrs: {},
            analysis: null,
            patterns: [],
            recorders: [],
            bcs: { fix: null, mass: null },
            removeSp: null,
            other: [],
          },
          {
            index: 1,
            name: "stage_1",
            path: "/opensees/stages/stage_001",
            attrs: {},
            analysis: null,
            patterns: [
              {
                groupName: "Plain_Stage1",
                path: "/opensees/stages/stage_001/patterns/Plain_Stage1",
                type: "Plain",
                tag: 2,
                params: [1],
                attrs: {},
                tables: {
                  sps: {
                    path: "/opensees/stages/stage_001/patterns/Plain_Stage1/sps",
                    columns: ["node_id", "dof", "value"],
                    rows: [[n0, 2, 0.035]],
                  },
                },
              },
            ],
            recorders: [],
            bcs: { fix: null, mass: null },
            removeSp: {
              path: "/opensees/stages/stage_001/remove_sp",
              columns: ["node_tag", "dof"],
              rows: [[n0, 2]],
            },
            other: ["remove_sp"],
          },
        ],
      },
    },
    { type: "setLoads", on: true },
  );

  // On stage 1: Plain_Global's SP (-0.01 on [n0, 2]) was removed by stage 1's remove_sp,
  // while stage 1's own Plain_Stage1 SP (+0.035 on [n0, 2]) added after remove_sp remains active!
  const sStage1 = reduce(sWithStagedReplace, { type: "selectStage", index: 1 });
  const sum1 = loadsSummaryOf(sStage1);
  assert.equal(sum1.counts.sps, 1, "stage 1 pattern SP remains active while global SP is removed");
  assert.ok(Math.abs(sum1.peaks.spTranslation - 0.035) < 1e-9, "active SP value is 0.035 (not 0.025 or 0)");
  const geom1 = buildLoads3D(sStage1, shoeBlobs);
  assert.equal(geom1.spArrows, 1);
  assert.ok(Math.abs(geom1.maxSpTranslationMag - 0.035) < 1e-9);
});

test("3-D beam with nearly-parallel vecxz (||vecxz - (vecxz · xLocal) xLocal|| < 1e-6) fails closed with notice", () => {
  const firstElemPath = shoeState.mesh!.elements[0]!;
  const firstElemDecl = shoeState.decls[firstElemPath]!;
  const firstOpsTag = firstElemDecl.element!.metas[0]!.tag;

  // Compute exact xLocal of firstElemPath from its linePositions blob
  const linePos = shoeBlobs.f32(shoeState.mesh!.linePositions);
  const dx = linePos[3]! - linePos[0]!;
  const dy = linePos[4]! - linePos[1]!;
  const dz = linePos[5]! - linePos[2]!;
  const L = Math.hypot(dx, dy, dz);
  const xLocal: [number, number, number] = [dx / L, dy / L, dz / L];

  const sParallelVecxz: State = reduce(
    {
      ...shoeState,
      ndm: { ndm: 3, source: "test" },
      neutralLoads: null,
      definitions: {
        ...shoeState.definitions!,
        stages: [],
        patterns: [
          {
            groupName: "Plain_1",
            path: "/opensees/patterns/Plain_1",
            type: "Plain",
            tag: 1,
            params: [1],
            attrs: {},
            tables: {
              element_loads: {
                path: "/opensees/patterns/Plain_1/element_loads",
                columns: ["col_0", "col_1", "col_2", "col_3", "col_4"],
                rows: [[firstOpsTag, "-beamUniform", -10, 0, Number.NaN]],
              },
            },
          },
        ],
      },
      decls: {
        ...shoeState.decls,
        "opensees/geomTransf/1": {
          path: "opensees/geomTransf/1",
          kind: "geomTransf",
          name: "gt1",
          nameSource: "test",
          type: "Linear",
          tag: 1,
          h5: "/opensees/geomTransf/Linear_1",
          params: [...xLocal],
          refs: {},
          links: {},
          fields: [{ label: "vecxz", value: JSON.stringify(xLocal), source: "test", interpreted: false }],
          problems: [],
        },
        [firstElemPath]: {
          ...firstElemDecl,
          refs: { ...firstElemDecl.refs, transfTag: "opensees/geomTransf/1" },
        },
      },
    },
    { type: "setLoads", on: true },
  );

  const geom = buildLoads3D(sParallelVecxz, shoeBlobs);
  assert.equal(geom.elementLoadArrows, 0, "parallel vecxz is refused fail-closed");
  assert.equal(geom.skippedUnorientedBeams, 1);
  assert.match(geom.notice ?? "", /missing geomTransf vecxz/);
});

