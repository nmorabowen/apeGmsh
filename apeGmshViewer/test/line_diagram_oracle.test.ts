// Oracle & contract tests for Escalón 2:
// - reader/ladruno.ts (LOCAL_FORCE_STATION_SIGNS, quaternionToZRec)
// - reader/read.ts (transformVecxzOf, modelTransformVecxz)
// - state/lineDiagram.ts (beamLocalAxes, defaultVecxz, screenExtrusionDirection,
//   lineDiagramFieldsOf, defaultFlipSignFor, joinLadrunoElements,
//   declTransformVecxz, elementTransformVecxzFromState,
//   buildLineDiagram3D, projectHoverNeedle, buildBeamChains)

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import type { ModelFile, OpsObject } from "../src/model/types.ts";
import { LOCAL_FORCE_STATION_SIGNS, openLadruno, quaternionToZRec } from "../src/reader/ladruno.ts";
import { openModel } from "../src/reader/node.ts";
import { modelTransformVecxz, transformVecxzOf } from "../src/reader/read.ts";
import type { ResultsH5 } from "../src/reader/results.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import {
  beamLocalAxes,
  buildBeamChains,
  buildLineDiagram3D,
  declTransformVecxz,
  defaultFlipSignFor,
  defaultVecxz,
  elementTransformVecxzFromState,
  joinLadrunoElements,
  lineDiagramFieldsOf,
  localAxisForComponent,
  projectHoverNeedle,
  requires3dFrame,
  screenExtrusionDirection,
} from "../src/state/lineDiagram.ts";
import type { Decl, DeclPath, LineStationComponent } from "../src/state/types.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;

const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const closeVec = (a: readonly number[], b: readonly number[], tol = 1e-9) =>
  a.length === b.length && a.every((v, i) => close(v, b[i]!, tol));

// -- 1. Local axes parity with Python compute_beam_local_axes ------------------

test("beamLocalAxes matches Python compute_beam_local_axes for 2D and 3D chords", () => {
  // 2D horizontal beam along +X: x = (1, 0, 0), defaultVecxz = (0, 0, 1) -> y = (0, 1, 0), z = (0, 0, 1)
  const f2dH = beamLocalAxes([1, 0, 0], defaultVecxz([1, 0, 0]))!;
  assert.ok(closeVec(f2dH.xLocal, [1, 0, 0]));
  assert.ok(closeVec(f2dH.yLocal, [0, 1, 0]));
  assert.ok(closeVec(f2dH.zLocal, [0, 0, 1]));

  // 2D vertical column along +Y: x = (0, 1, 0), defaultVecxz = (0, 0, 1) -> y = (-1, 0, 0), z = (0, 0, 1)
  const f2dV = beamLocalAxes([0, 1, 0], defaultVecxz([0, 1, 0]))!;
  assert.ok(closeVec(f2dV.xLocal, [0, 1, 0]));
  assert.ok(closeVec(f2dV.yLocal, [-1, 0, 0]));
  assert.ok(closeVec(f2dV.zLocal, [0, 0, 1]));

  // 3D vertical column along +Z with defaultVecxz -> vecxz = (1, 0, 0)
  // y = (1,0,0) x (0,0,1) = (0, -1, 0); z = (1, 0, 0)
  const vDefZ = defaultVecxz([0, 0, 1]);
  assert.ok(closeVec(vDefZ, [1, 0, 0]));
  const f3dZ = beamLocalAxes([0, 0, 1], vDefZ)!;
  assert.ok(closeVec(f3dZ.xLocal, [0, 0, 1]));
  assert.ok(closeVec(f3dZ.yLocal, [0, -1, 0]));
  assert.ok(closeVec(f3dZ.zLocal, [1, 0, 0]));

  // 3D inclined chord (0.6, 0.8, 0) with custom vecxz = (0, 0, 1):
  // x = (0.6, 0.8, 0), y = (0,0,1) x (0.6,0.8,0) = (-0.8, 0.6, 0), z = (0, 0, 1)
  const f3dInc = beamLocalAxes([0.6, 0.8, 0], [0, 0, 1])!;
  assert.ok(closeVec(f3dInc.xLocal, [0.6, 0.8, 0]));
  assert.ok(closeVec(f3dInc.yLocal, [-0.8, 0.6, 0]));
  assert.ok(closeVec(f3dInc.zLocal, [0, 0, 1]));

  // Deformed chord re-orthogonalization against recorded zRec = (0, 0, 1):
  const s = Math.hypot(1, 0.1);
  const fDef = beamLocalAxes([1 / s, 0, 0.1 / s], [0, 0, 1])!;
  assert.ok(closeVec(fDef.yLocal, [0, 1, 0]));
  assert.ok(closeVec(fDef.xLocal, [1 / s, 0, 0.1 / s]));
  assert.ok(closeVec(fDef.zLocal, [-0.1 / s, 0, 1 / s]));
});

test("quaternionToZRec extracts row 2 of R(q) matching Python LocalAxes.matrices and axes_from_quaternion", () => {
  const c = Math.SQRT1_2;
  // 90 deg rotation about Y: q = [w=cos(45), x=0, y=sin(45), z=0]
  // Row 2 of R(q) (global -> local): [2(xz - yw), 2(yz + xw), 1 - 2(x^2 + y^2)] = [-1, 0, 0]
  assert.ok(closeVec(quaternionToZRec(c, 0, c, 0), [-1, 0, 0]));
  // 90 deg rotation about X: q = [w=cos(45), x=sin(45), y=0, z=0] -> [0, +1, 0]
  assert.ok(closeVec(quaternionToZRec(c, c, 0, 0), [0, 1, 0]));
  // General unit quaternion q = [0.5, 0.5, 0.5, 0.5]:
  // 2(xz - yw) = 0, 2(yz + xw) = 1, 1 - 2(x^2 + y^2) = 0 -> [0, 1, 0]
  assert.ok(closeVec(quaternionToZRec(0.5, 0.5, 0.5, 0.5), [0, 1, 0]));
});

// -- 2. model.h5 per_element_vecxz & per_element_emitted_tag -------------------

test("read.ts and lineDiagram.ts extract per_element_vecxz and per_element_emitted_tag from model.h5 and State.decls (including >8 rows and secTag vs transfTag disambiguation)", () => {
  const transfObj: OpsObject = {
    path: "/opensees/transforms/Linear_1",
    family: "geomTransf",
    groupName: "Linear_1",
    type: "LinearTransform",
    tag: 1,
    params: [],
    attrs: {},
    tables: {
      per_element_emitted_tag: {
        path: "/opensees/transforms/Linear_1/per_element_emitted_tag",
        columns: ["value"],
        rows: [[101], [102]],
      },
      per_element_vecxz: {
        path: "/opensees/transforms/Linear_1/per_element_vecxz",
        columns: ["[0]", "[1]", "[2]"],
        rows: [
          [0, 0, 1],
          [-1, 0, 0],
        ],
      },
    },
  };

  const single = transformVecxzOf(transfObj);
  assert.equal(single.length, 2);
  assert.deepEqual(single[0], { tag: 101, vecxz: [0, 0, 1] });
  assert.deepEqual(single[1], { tag: 102, vecxz: [-1, 0, 0] });

  const map = modelTransformVecxz({
    opensees: { objects: [transfObj] },
  } as unknown as ModelFile);
  assert.equal(map.size, 2);
  assert.ok(closeVec(map.get(101)!, [0, 0, 1]));
  assert.ok(closeVec(map.get(102)!, [-1, 0, 0]));

  // State.decls extraction + elementTransformVecxzFromState:
  // Suppose `elasticBeamColumn` has `args = [secTag=101, transfTag=102]`.
  // Even though 101 is ALSO a transform tag in `per_element_emitted_tag`,
  // `elementTransformVecxzFromState` must decode slot 1 (`transfTag=102`), NOT slot 0 (`secTag=101`)!
  const transfDecl: Decl = {
    path: "opensees/transforms/Linear_1",
    kind: "geomTransf",
    type: "LinearTransform",
    name: "Linear_1",
    nameSource: "/opensees/names",
    h5: "/opensees/transforms/Linear_1",
    tag: 101,
    params: [],
    fields: [
      { label: "per_element_vecxz", value: "[[0, 0, 1], [-1, 0, 0]]", source: "/opensees/transforms/Linear_1/per_element_vecxz", interpreted: false },
      { label: "per_element_emitted_tag", value: "[101, 102]", source: "/opensees/transforms/Linear_1/per_element_emitted_tag", interpreted: false },
    ],
    refs: {},
    links: {},
    problems: [],
  };
  const elemDecl: Decl = {
    path: "mesh/element/1",
    kind: "element",
    type: "elasticBeamColumn",
    name: "Beam_1",
    nameSource: "/elements/line2",
    h5: "/elements/line2",
    tag: 1,
    params: [],
    fields: [],
    refs: { secTag: "opensees/section/Sec_1", transfTag: "opensees/transforms/Linear_1" },
    links: {},
    problems: [],
    element: {
      femId: 1,
      cell: { alias: "line2", block: 0, row: 0 },
      inlineNodes: [],
      groups: [],
      metas: [{ type: "elasticBeamColumn", tag: 10, h5: "/opensees/element_meta/elasticBeamColumn", row: 0, args: [101, 102] }],
    },
  };
  const decls: Record<DeclPath, Decl> = {
    "opensees/transforms/Linear_1": transfDecl,
    "mesh/element/1": elemDecl,
  };
  const pairs = declTransformVecxz(transfDecl);
  assert.equal(pairs.length, 2);
  assert.ok(closeVec(pairs[0]!.vecxz, [0, 0, 1]));
  assert.ok(closeVec(pairs[1]!.vecxz, [-1, 0, 0]));

  const resolved = elementTransformVecxzFromState(decls, elemDecl, map);
  assert.ok(resolved !== null);
  // Decodes slot 1 (`transfTag = 102`) -> [-1, 0, 0], never colliding with `secTag = 101` at slot 0!
  assert.ok(closeVec(resolved, [-1, 0, 0]));

  // Verify >8 rows on Decl.vecxz (where `formatTable` truncates `decl.fields` after 8 rows)
  const twelveVecs = Array.from({ length: 12 }, (_, i) => ({
    tag: 200 + i,
    vecxz: [0, Math.cos(i * 0.1), Math.sin(i * 0.1)] as [number, number, number],
  }));
  const largeTransfDecl: Decl = {
    ...transfDecl,
    tag: 200,
    fields: [
      { label: "per_element_vecxz", value: "12 rows ([0], [1], [2]): ... (+4 rows)", source: "/opensees/transforms/Linear_1/per_element_vecxz", interpreted: false },
    ],
    vecxz: twelveVecs,
  };
  const beamRow11: Decl = {
    ...elemDecl,
    path: "mesh/element/11",
    refs: { transfTag: "opensees/transforms/Linear_1" },
    element: {
      ...elemDecl.element!,
      femId: 11,
      metas: [{ type: "elasticBeamColumn", tag: 21, h5: "/opensees/element_meta/elasticBeamColumn", row: 11, args: [5, 211] }],
    },
  };
  const resolvedRow11 = elementTransformVecxzFromState(
    { "opensees/transforms/Linear_1": largeTransfDecl, "mesh/element/11": beamRow11 },
    beamRow11,
  );
  assert.ok(resolvedRow11 !== null);
  assert.ok(closeVec(resolvedRow11, twelveVecs[11]!.vecxz));
});

// -- 3. Sign & picker metadata parity ------------------------------------------

test("picker metadata, defaultFlipSignFor, localAxisForComponent, and LOCAL_FORCE_STATION_SIGNS follow ADR 0074/0076", () => {
  const fields = lineDiagramFieldsOf(["N", "Vy", "Vz", "T", "My", "Mz", "eps", "kappaZ"]);
  assert.deepEqual(fields.map((f) => f.name), ["section_force", "section_deformation"]);
  assert.deepEqual(fields[0]!.components.map((c) => c.key), ["N", "Vy", "Vz", "T", "My", "Mz"]);
  assert.deepEqual(fields[1]!.components.map((c) => c.key), ["eps", "kappaZ"]);

  // Bending moments and curvatures default to tension-side convention (flipSign = true); others false
  assert.equal(defaultFlipSignFor("My"), true);
  assert.equal(defaultFlipSignFor("Mz"), true);
  assert.equal(defaultFlipSignFor("kappaY"), true);
  assert.equal(defaultFlipSignFor("kappaZ"), true);
  assert.equal(defaultFlipSignFor("N"), false);
  assert.equal(defaultFlipSignFor("Vy"), false);
  assert.equal(defaultFlipSignFor("Vz"), false);
  assert.equal(defaultFlipSignFor("T"), false);
  assert.equal(defaultFlipSignFor("eps"), false);
  assert.equal(defaultFlipSignFor("gammaY"), false);
  assert.equal(defaultFlipSignFor("gammaZ"), false);
  assert.equal(defaultFlipSignFor("theta"), false);

  // Local axis mapping: Vy & Mz & T -> yLocal, Vz & My & N -> zLocal
  assert.equal(localAxisForComponent("Vy"), "y");
  assert.equal(localAxisForComponent("Mz"), "y");
  assert.equal(localAxisForComponent("Vz"), "z");
  assert.equal(localAxisForComponent("My"), "z");
  assert.equal(localAxisForComponent("N"), "z");
  assert.equal(localAxisForComponent("T"), "y");

  assert.equal(requires3dFrame("N"), false);
  assert.equal(requires3dFrame("T"), false);
  assert.equal(requires3dFrame("Vy"), true);
  assert.equal(requires3dFrame("Mz"), true);

  // OpenSees localForce nodal equilibrium -> internal section force sign table
  assert.deepEqual(LOCAL_FORCE_STATION_SIGNS, {
    N: [-1, 1],
    Vy: [1, -1],
    Vz: [-1, 1],
    T: [-1, 1],
    My: [-1, 1],
    Mz: [-1, 1],
  });

  // Screen extrusion direction: normalize(cross(xLocal, cameraViewDir)), fallback for colinear viewDir
  const dir = screenExtrusionDirection([1, 0, 0], [0, 0, -1]);
  // (1, 0, 0) x (0, 0, -1) = (0, 1, 0)
  assert.ok(closeVec(dir, [0, 1, 0]));
  const fallback = screenExtrusionDirection([0, 0, 1], [0, 0, 1]);
  assert.ok(close(Math.hypot(fallback[0], fallback[1], fallback[2]), 1));
  assert.ok(close(fallback[2], 0));
});

// -- 4. 3D geometry, single-station expansion, zero-crossing, fail-closed ------

function makeTwoBeamScene(): {
  decls: Record<DeclPath, Decl>;
  meshElements: DeclPath[];
  lineElement: Int32Array;
  drawnLinePositions: Float32Array;
  join: { ok: true; space: "ops_tag"; byLadrunoId: Map<number, DeclPath>; byDeclPath: Map<DeclPath, number> };
} {
  const transfDecl: Decl = {
    path: "opensees/transforms/Tr1",
    kind: "geomTransf",
    type: "LinearTransform",
    name: "Tr1",
    nameSource: "/opensees/names",
    h5: "/opensees/transforms/Tr1",
    tag: 1,
    params: [0, 0, 1],
    fields: [],
    refs: {},
    links: {},
    problems: [],
  };
  const e1: Decl = {
    path: "mesh/element/1",
    kind: "element",
    type: "ElasticBeamColumn3d",
    name: "Beam1",
    nameSource: "/elements/line2",
    h5: "/elements/line2",
    tag: 10,
    params: [],
    fields: [],
    refs: { transfTag: "opensees/transforms/Tr1" }, // framed!
    links: {},
    problems: [],
    element: {
      femId: 1,
      cell: { alias: "line2", block: 0, row: 0 },
      inlineNodes: [],
      groups: [],
      metas: [{ type: "ElasticBeamColumn3d", tag: 10, h5: "/opensees/element_meta/ElasticBeamColumn3d", row: 0, args: [] }],
    },
  };
  const e2: Decl = {
    path: "mesh/element/2",
    kind: "element",
    type: "ElasticBeamColumn3d",
    name: "Beam2",
    nameSource: "/elements/line2",
    h5: "/elements/line2",
    tag: 20,
    params: [],
    fields: [],
    refs: {}, // NO transfTag and NO LOCAL_AXES -> unframed in 3D!
    links: {},
    problems: [],
    element: {
      femId: 2,
      cell: { alias: "line2", block: 0, row: 1 },
      inlineNodes: [],
      groups: [],
      metas: [{ type: "ElasticBeamColumn3d", tag: 20, h5: "/opensees/element_meta/ElasticBeamColumn3d", row: 1, args: [] }],
    },
  };
  return {
    decls: {
      "opensees/transforms/Tr1": transfDecl,
      "mesh/element/1": e1,
      "mesh/element/2": e2,
    },
    meshElements: ["mesh/element/1", "mesh/element/2"],
    lineElement: new Int32Array([0, 1]),
    // Beam 1: (0,0,0) -> (2,0,0); Beam 2: (2,0,0) -> (4,0,0)
    drawnLinePositions: new Float32Array([0, 0, 0, 2, 0, 0, 2, 0, 0, 4, 0, 0]),
    join: {
      ok: true,
      space: "ops_tag",
      byLadrunoId: new Map<number, DeclPath>([
        [10, "mesh/element/1"],
        [20, "mesh/element/2"],
      ]),
      byDeclPath: new Map<DeclPath, number>([
        ["mesh/element/1", 10],
        ["mesh/element/2", 20],
      ]),
    },
  };
}

test("buildLineDiagram3D expands single-station basicForce to [-1, +1], splits zero crossings without degenerate triangles, and fails closed on unframed 3D beams", () => {
  const scene = makeTwoBeamScene();

  // Case A: Single-station N (axialForce/basicForce at xi=0):
  // N does not require a 3D frame (uses defaultVecxz when unframed), so both elements draw.
  const geomN = buildLineDiagram3D({
    ...scene,
    stepData: {
      component: "N",
      elementIndex: new Float64Array([10, 20]),
      stationNaturalCoord: new Float64Array([0, 0]),
      values: new Float64Array([50, 100]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomN.skippedUnorientedBeams, 0);
  assert.equal(geomN.beams.length, 2);
  // Each single-station element expands to 2 stations [-1, +1] -> 1 quad = 2 triangles = 18 floats in triPositions
  assert.deepEqual(Array.from(geomN.beams[0]!.xi), [-1, 1]);
  assert.deepEqual(Array.from(geomN.beams[0]!.values), [50, 50]);
  assert.equal(geomN.triPositions.length, 2 * 2 * 9);

  // Case B: Mz with zero crossing on elem 10 ([-100, +100] across [-1, +1]), while elem 20 is unframed in 3D:
  const geomMz = buildLineDiagram3D({
    ...scene,
    stepData: {
      component: "Mz",
      elementIndex: new Float64Array([10, 10, 20, 20]),
      stationNaturalCoord: new Float64Array([-1, 1, -1, 1]),
      values: new Float64Array([-100, 100, 50, 50]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
    clim: [-100, 100],
  });
  // Elem 20 is refused (skippedUnorientedBeams = 1) because Mz requires a 3D frame and elem 20 has neither LOCAL_AXES nor vecxz!
  assert.equal(geomMz.skippedUnorientedBeams, 1);
  assert.deepEqual(geomMz.skippedBeamDecls, ["mesh/element/2"]);
  assert.equal(geomMz.beams.length, 1);
  // Elem 10 has a sign change (-100 to +100), so its bow-tie quad splits at pZero=(1,0,0) into 2 non-degenerate triangles = 18 floats!
  assert.equal(geomMz.triPositions.length, 2 * 9);
  // Default flipSign=true for Mz: station 0 has rawValue = -100, signMult = -1, extruded along +yLocal = (0, 1, 0)
  // autoScale = 0.15 * L_char / v_peak = 0.15 * 2 / 100 = 0.003; offset = (-1)*(-100)*0.003 = +0.3
  assert.ok(closeVec(Array.from(geomMz.profilePositions.slice(0, 6)), [0, 0.3, 0, 2, -0.3, 0], 1e-6));
  // Both zero-crossing triangles have strictly positive area:
  // Tri 0: (0,0,0) -> (1,0,0) -> (0,0.3,0); Tri 1: (1,0,0) -> (2,0,0) -> (2,-0.3,0)
  assert.ok(closeVec(Array.from(geomMz.triPositions.slice(0, 9)), [0, 0, 0, 1, 0, 0, 0, 0.3, 0], 1e-6));
  assert.ok(closeVec(Array.from(geomMz.triPositions.slice(9, 18)), [1, 0, 0, 2, 0, 0, 2, -0.3, 0], 1e-6));
});

// -- 5. 3D hover needle projection ---------------------------------------------

test("projectHoverNeedle linearly interpolates station values and tips at x/L in {0, 0.25, 0.5, 1.0}", () => {
  const scene = makeTwoBeamScene();
  const geom = buildLineDiagram3D({
    ...scene,
    stepData: {
      component: "Vy",
      elementIndex: new Float64Array([10, 10]),
      stationNaturalCoord: new Float64Array([-1, 1]),
      values: new Float64Array([10, 50]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  const beam = geom.beams[0]!; // L = 2 along +X, yLocal = (0, 1, 0)

  for (const [frac, expectedVal] of [
    [0, 10],
    [0.25, 20],
    [0.5, 30],
    [1.0, 50],
  ] as const) {
    const x = frac * 2;
    const hit = projectHoverNeedle(beam, [x, 0.5, -0.2], 0.1, false);
    assert.ok(close(hit.xRel, frac));
    assert.ok(close(hit.value, expectedVal));
    assert.ok(closeVec(hit.pBase, [x, 0, 0]));
    assert.ok(closeVec(hit.pTop, [x, expectedVal * 0.1, 0]));
  }
});

// -- 6. Multi-beam chains & cumulative x ---------------------------------------

test("buildBeamChains orders colinear, L-corner, and T-junction chains, reverses backward elements, and separates disconnected spans", () => {
  // Span 0: node 1 -> node 2, L = 2, stations at xi=[-1, 1] with values [10, 20]
  // Span 1: defined backwards! node 3 -> node 2, L = 3, stations at xi=[-1, 1] with values [99, 20]
  //   When reversed in chain (node 2 -> node 3), stations flip to [20, 99] at x = [2, 5]!
  // Span 2: disconnected span node 10 -> node 11, L = 2, values [5, 15]
  const chains = buildBeamChains([
    {
      decl: "mesh/element/1",
      elementId: 1,
      nodeStart: 1,
      nodeEnd: 2,
      length: 2,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([10, 20]),
    },
    {
      decl: "mesh/element/2",
      elementId: 2,
      nodeStart: 3,
      nodeEnd: 2,
      length: 3,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([99, 20]),
    },
    {
      decl: "mesh/element/3",
      elementId: 3,
      nodeStart: 10,
      nodeEnd: 11,
      length: 2,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([5, 15]),
    },
  ]);

  assert.equal(chains.length, 2);
  const c0 = chains[0]!;
  assert.ok(close(c0.totalLength, 5));
  assert.deepEqual(c0.joints.map((j) => j.nodeId), [1, 2, 3]);
  assert.deepEqual(c0.joints.map((j) => j.x), [0, 2, 5]);
  assert.equal(c0.spans[0]!.reversed, false);
  assert.equal(c0.spans[1]!.reversed, true);
  assert.deepEqual(c0.spans.flatMap((sp) => sp.stations.map((st) => st.x)), [0, 2, 2, 5]);
  assert.deepEqual(c0.spans.flatMap((sp) => sp.stations.map((st) => st.value)), [10, 20, 20, 99]);

  const c1 = chains[1]!;
  assert.ok(close(c1.totalLength, 2));
  assert.deepEqual(c1.joints.map((j) => j.nodeId), [10, 11]);
  assert.deepEqual(c1.spans.flatMap((sp) => sp.stations.map((st) => st.x)), [0, 2]);
  assert.deepEqual(c1.spans.flatMap((sp) => sp.stations.map((st) => st.value)), [5, 15]);

  // T-junction at node 2: (1->2), (2->3), and branch (2->4)
  const tChains = buildBeamChains([
    {
      decl: "mesh/element/1",
      elementId: 1,
      nodeStart: 1,
      nodeEnd: 2,
      length: 2,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([10, 20]),
    },
    {
      decl: "mesh/element/2",
      elementId: 2,
      nodeStart: 2,
      nodeEnd: 3,
      length: 3,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([20, 30]),
    },
    {
      decl: "mesh/element/3",
      elementId: 3,
      nodeStart: 2,
      nodeEnd: 4,
      length: 4,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([25, 35]),
    },
  ]);
  assert.equal(tChains.length, 2);
  assert.equal(tChains[0]!.contiguous, false);
  assert.deepEqual(tChains[0]!.joints.map((j) => j.nodeId), [1, 2, 3]);
  assert.deepEqual(tChains[1]!.joints.map((j) => j.nodeId), [2, 4]);
});

// -- 7. Element-ID join (ops_tag vs femId & fem_hash-only refusal) -------------

test("joinLadrunoElements joins by emittedTags when model_hash matches, or femIds when element_index_space=='fem', and refuses fem_hash-only without 'fem'", () => {
  const p = join(mkdtempSync(join(tmpdir(), "agv-join-")), "join.ladruno");
  const w = new h5wasm.File(p, "w");
  try {
    const info = w.create_group("INFO");
    info.create_attribute("GENERATOR", "Ladruno");
    info.create_attribute("FORMAT_VERSION", 1);
    const prov = info.create_group("PROVENANCE");
    prov.create_attribute("model_hash", "abcd");
    prov.create_attribute("fem_hash", "1234");
  } finally {
    w.close();
  }

  const decls: Record<DeclPath, Decl> = {
    "mesh/element/11": {
      path: "mesh/element/11",
      kind: "element",
      type: "ElasticBeamColumn2d",
      name: "Beam11",
      nameSource: "/elements/line2",
      h5: "/elements/line2",
      tag: 501,
      params: [],
      fields: [],
      refs: {},
      links: {},
      problems: [],
      element: {
        femId: 11,
        cell: { alias: "line2", block: 0, row: 0 },
        inlineNodes: [],
        groups: [],
        metas: [{ type: "ElasticBeamColumn2d", tag: 501, h5: "/opensees/element_meta/ElasticBeamColumn2d", row: 0, args: [] }],
      },
    },
    "mesh/element/12": {
      path: "mesh/element/12",
      kind: "element",
      type: "ElasticBeamColumn2d",
      name: "Beam12",
      nameSource: "/elements/line2",
      h5: "/elements/line2",
      tag: 502,
      params: [],
      fields: [],
      refs: {},
      links: {},
      problems: [],
      element: {
        femId: 12,
        cell: { alias: "line2", block: 0, row: 1 },
        inlineNodes: [],
        groups: [],
        metas: [{ type: "ElasticBeamColumn2d", tag: 502, h5: "/opensees/element_meta/ElasticBeamColumn2d", row: 1, args: [] }],
      },
    },
  };

  const r = openLadruno(H5, p);
  try {
    // 1. model_hash matches -> joins by ops_tag (501 -> mesh/element/11, 502 -> mesh/element/12)
    const joinOps = joinLadrunoElements(
      decls,
      { state: "paired", by: "model_hash" },
      r.index.provenance,
    );
    assert.equal(joinOps.ok, true);
    if (joinOps.ok) {
      assert.equal(joinOps.space, "ops_tag");
      assert.equal(joinOps.byLadrunoId.get(501), "mesh/element/11");
      assert.equal(joinOps.byLadrunoId.get(502), "mesh/element/12");
    }

    // 2. fem_hash-only WITHOUT element_index_space=="fem" -> refused!
    const joinRefused = joinLadrunoElements(
      decls,
      { state: "paired", by: "fem_hash" },
      r.index.provenance,
    );
    assert.equal(joinRefused.ok, false);
    if (!joinRefused.ok) {
      assert.match(joinRefused.reason, /element_index_space is not "fem"/);
    }
  } finally {
    r.close();
  }

  // 3. Stamp element_index_space = "fem" -> fem_hash-only joins by femIds (11 -> mesh/element/11, 12 -> mesh/element/12)
  const w2 = new h5wasm.File(p, "a");
  try {
    (w2.get("INFO/PROVENANCE") as h5wasm.Group).create_attribute("element_index_space", "fem");
  } finally {
    w2.close();
  }
  const r2 = openLadruno(H5, p);
  try {
    const joinFem = joinLadrunoElements(
      decls,
      { state: "paired", by: "fem_hash" },
      r2.index.provenance,
    );
    assert.equal(joinFem.ok, true);
    if (joinFem.ok) {
      assert.equal(joinFem.space, "fem");
      assert.equal(joinFem.byLadrunoId.get(11), "mesh/element/11");
      assert.equal(joinFem.byLadrunoId.get(12), "mesh/element/12");
    }
  } finally {
    r2.close();
  }
});

test("multi-segment 1-D elements span their full end-to-end chord in buildLineDiagram3D and buildSelectedBeamChainsFromState", () => {
  const scene = makeTwoBeamScene();
  // Suppose element 1 (`mesh/element/1`, meshElements index 0) is subdivided into 2 line segments:
  // seg 0: (0,0,0) -> (1,0,0) (nodes 1 -> 5)
  // seg 1: (1,0,0) -> (2,0,0) (nodes 5 -> 2)
  // And element 2 (`mesh/element/2`, meshElements index 1) is seg 2: (2,0,0) -> (4,0,0) (nodes 2 -> 3)
  const geom = buildLineDiagram3D({
    ...scene,
    lineElement: new Int32Array([0, 0, 1]),
    drawnLinePositions: new Float32Array([
      0, 0, 0, 1, 0, 0,
      1, 0, 0, 2, 0, 0,
      2, 0, 0, 4, 0, 0,
    ]),
    stepData: {
      component: "N",
      elementIndex: new Float64Array([10, 10, 20, 20]),
      stationNaturalCoord: new Float64Array([-1, 1, -1, 1]),
      values: new Float64Array([10, 20, 20, 40]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geom.beams.length, 2);
  // Beam 0 spans the full chord (0,0,0) -> (2,0,0) of length 2, not just the first half (0,0,0) -> (1,0,0)!
  assert.ok(close(geom.beams[0]!.length, 2));
  assert.ok(closeVec(geom.beams[0]!.pStart, [0, 0, 0]));
  assert.ok(closeVec(geom.beams[0]!.pEnd, [2, 0, 0]));
  assert.equal(geom.beams[0]!.segmentIndex, 0);
  assert.equal(geom.beams[0]!.endSegmentIndex, 1);
});

test("F1b: buildLineDiagram3D preserves step_local_axes provenance from stepData and never lets modelLocalAxesZRec override it", () => {
  const scene = makeTwoBeamScene();
  // Element 10 along +X: stepData has localAxesZRec = [0, 1, 0] with provenance "step_local_axes",
  // while opts.modelLocalAxesZRec has [0, 0, 1].
  // Priority 1 ("step_local_axes") must win over Priority 2 (opts.modelLocalAxesZRec)!
  const geom = buildLineDiagram3D({
    ...scene,
    stepData: {
      component: "Mz",
      elementIndex: new Float64Array([10, 10]),
      stationNaturalCoord: new Float64Array([-1, 1]),
      values: new Float64Array([50, 50]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: new Float64Array([0, 1, 0, 0, 1, 0]),
      localAxesSources: ["step_local_axes", "step_local_axes"],
    },
    modelLocalAxesZRec: new Map([[10, [0, 0, 1] as const]]),
    ndm: 3,
    scale: 1,
  });
  assert.equal(geom.beams.length, 1);
  const b0 = geom.beams[0]!;
  assert.equal(b0.zRecSource, "step_local_axes");
  // With xLocal = [1, 0, 0] and zRec = [0, 1, 0], yLocal = zLocal x xLocal = [0, 1, 0] x [1, 0, 0] = [0, 0, -1]
  assert.ok(closeVec(b0.extrudeDir, [0, 0, -1]));
});

test("F8: buildBeamChains prefers colinear continuation (dot > 0.95) at degree >= 3 junctions over lower elementId T-junction stubs", () => {
  // Straight girder along +X: node 1 (0,0,0) -> node 2 (3,0,0) (elementId = 10)
  // T-junction perpendicular stub at node 2: node 2 (3,0,0) -> node 4 (3,2,0) (elementId = 2, lower elementId & listed before right girder!)
  // Continuing girder along +X: node 2 (3,0,0) -> node 3 (6,0,0) (elementId = 20, higher elementId!)
  const chains = buildBeamChains([
    {
      decl: "mesh/element/10",
      elementId: 10,
      nodeStart: 1,
      nodeEnd: 2,
      length: 3,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([10, 20]),
      pStart: [0, 0, 0],
      pEnd: [3, 0, 0],
    },
    {
      decl: "mesh/element/2",
      elementId: 2,
      nodeStart: 2,
      nodeEnd: 4,
      length: 2,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([99, 99]),
      pStart: [3, 0, 0],
      pEnd: [3, 2, 0],
    },
    {
      decl: "mesh/element/20",
      elementId: 20,
      nodeStart: 2,
      nodeEnd: 3,
      length: 3,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([20, 30]),
      pStart: [3, 0, 0],
      pEnd: [6, 0, 0],
    },
  ]);

  assert.equal(chains.length, 2);
  // Main chain follows the colinear girder 1 -> 2 -> 3 (elements [10, 20]), NOT hijacked by stub element 2!
  assert.deepEqual(chains[0]!.spans.map((sp) => sp.elementId), [10, 20]);
  assert.deepEqual(chains[0]!.joints.map((j) => j.nodeId), [1, 2, 3]);
  assert.ok(close(chains[0]!.totalLength, 6));
  // Secondary chain has the T-junction stub 2 -> 4 (element [2])
  assert.deepEqual(chains[1]!.spans.map((sp) => sp.elementId), [2]);
  assert.deepEqual(chains[1]!.joints.map((j) => j.nodeId), [2, 4]);

  // Multi-span girder (1 -> 2 -> 3 -> 4) with an inward-directed stub (5 -> 3, elementId = 1, listed first!)
  // and outward-directed girder span (3 -> 2) so stub endpoint 5 is the ONLY directedStart if not walking degree-2 paths:
  const multiSpanChains = buildBeamChains([
    {
      decl: "mesh/element/1",
      elementId: 1,
      nodeStart: 5,
      nodeEnd: 3,
      length: 2,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([99, 99]),
      pStart: [6, 2, 0],
      pEnd: [6, 0, 0],
    },
    {
      decl: "mesh/element/10",
      elementId: 10,
      nodeStart: 2,
      nodeEnd: 1,
      length: 3,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([20, 10]),
      pStart: [3, 0, 0],
      pEnd: [0, 0, 0],
    },
    {
      decl: "mesh/element/20",
      elementId: 20,
      nodeStart: 3,
      nodeEnd: 2,
      length: 3,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([30, 20]),
      pStart: [6, 0, 0],
      pEnd: [3, 0, 0],
    },
    {
      decl: "mesh/element/30",
      elementId: 30,
      nodeStart: 3,
      nodeEnd: 4,
      length: 3,
      xi: new Float64Array([-1, 1]),
      values: new Float64Array([30, 40]),
      pStart: [6, 0, 0],
      pEnd: [9, 0, 0],
    },
  ]);
  assert.equal(multiSpanChains.length, 2);
  assert.deepEqual(multiSpanChains[0]!.spans.map((sp) => sp.elementId), [10, 20, 30]);
  assert.deepEqual(multiSpanChains[0]!.joints.map((j) => j.nodeId), [1, 2, 3, 4]);
  assert.deepEqual(multiSpanChains[1]!.spans.map((sp) => sp.elementId), [1]);
});

test("buildLineDiagram3D extrudes My on the 3-D tension side when flipSign=true (+zLocal for hogging My>0, -zLocal for sagging My<0)", () => {
  const scene = makeTwoBeamScene();
  // Beam 10 along +X with vecxz = [0, 0, 1] -> zLocal = [0, 0, +1].
  // Under downward gravity (-Z): support hogging has My > 0 (+100 at xi=-1, tension on +Z top fiber),
  // midspan sagging has My < 0 (-50 at xi=+1, tension on -Z bottom fiber).
  const geomMy = buildLineDiagram3D({
    ...scene,
    stepData: {
      component: "My",
      elementIndex: new Float64Array([10, 10]),
      stationNaturalCoord: new Float64Array([-1, 1]),
      values: new Float64Array([100, -50]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
    clim: [-50, 100],
  });
  assert.equal(geomMy.beams.length, 1);
  // autoScale = 0.15 * 2 / 100 = 0.003:
  // At xi=-1 (x=0, My=+100 hogging): top point is at z = +0.3 (UP, top tension side)
  // At xi=+1 (x=2, My=-50 sagging): top point is at z = -0.15 (DOWN, bottom tension side)
  assert.ok(closeVec(Array.from(geomMy.profilePositions.slice(0, 6)), [0, 0, 0.3, 2, 0, -0.15], 1e-6));
  const needle = projectHoverNeedle(geomMy.beams[0]!, [2, 0, 0], geomMy.effectiveScale, true);
  assert.ok(closeVec(needle.pTop, [2, 0, -0.15], 1e-6));
});

// -- 8. OpenSees Golden Oracle suite (fixtures/line_oracle_*.{h5,ladruno,golden.json}) --

interface GoldenStationEntry {
  xi: number[];
  values: number[][];
  closed_form: number[][];
}

interface GoldenBucketEntry {
  stations: Record<string, GoldenStationEntry>;
}

interface GoldenElementEntry {
  fem_id: number;
  ops_tag: number;
  nodes: [number, number];
  coords_i: [number, number, number];
  coords_j: [number, number, number];
  length: number;
  frame: {
    x: [number, number, number];
    y: [number, number, number];
    z: [number, number, number];
  };
  frame_quaternion_wxyz: [number, number, number, number];
  buckets: Record<string, GoldenBucketEntry>;
}

interface GoldenOracleFile {
  fixture: string;
  ndm: 2 | 3;
  responses: string[];
  elements: GoldenElementEntry[];
}

const COMP_TO_CANON: Record<LineStationComponent, string> = {
  N: "axial_force",
  Vy: "shear_y",
  Vz: "shear_z",
  T: "torsion",
  My: "bending_moment_y",
  Mz: "bending_moment_z",
  eps: "axial_strain",
  gammaY: "shear_strain_y",
  gammaZ: "shear_strain_z",
  theta: "torsion_rate",
  kappaY: "curvature_y",
  kappaZ: "curvature_z",
};

const BUCKET_PRIORITY = ["section.force", "section.deformation", "localForce", "basicForce", "axialForce"] as const;

function winningBucketStation(el: GoldenElementEntry, canon: string): GoldenStationEntry | null {
  for (const b of BUCKET_PRIORITY) {
    const st = el.buckets[b]?.stations[canon];
    if (st) return st;
  }
  return null;
}

for (const fixtureName of [
  "line_oracle_ss_udl",
  "line_oracle_cantilever3d",
  "line_oracle_column3d",
  "line_oracle_fbc_lobatto",
  "line_oracle_basic_only",
] as const) {
  test(`OpenSees golden oracle (${fixtureName}): readLineStations, joinLadrunoElements, beamLocalAxes, and buildLineDiagram3D match OpenSees 3.7.1`, async () => {
    const base = new URL(`../fixtures/${fixtureName}`, import.meta.url);
    const modelPath = fileURLToPath(new URL(`${fixtureName}.h5`, base));
    const ladrunoPath = fileURLToPath(new URL(`${fixtureName}.ladruno`, base));
    const goldenPath = fileURLToPath(new URL(`${fixtureName}.golden.json`, base));
    const golden = JSON.parse(readFileSync(goldenPath, "utf8")) as GoldenOracleFile;

    const model = await openModel(modelPath);
    const blobs = new BlobStore();
    const loaded = loadModel(model, blobs);
    const r = openLadruno(H5, ladrunoPath);
    try {
      const stageId = r.index.stages[0]!.id;
      const join = joinLadrunoElements(loaded.decls, { state: "paired", by: "model_hash" }, r.index.provenance);
      assert.equal(join.ok, true, `${fixtureName}: joinLadrunoElements must succeed`);
      if (!join.ok) return;
      assert.equal(join.space, "ops_tag");

      for (const el of golden.elements) {
        const declPath = join.byLadrunoId.get(el.ops_tag);
        assert.ok(declPath !== undefined, `${fixtureName}: ops_tag ${el.ops_tag} joined`);
        assert.equal(loaded.decls[declPath!]?.element?.femId, el.fem_id);
        const [qw, qx, qy, qz] = el.frame_quaternion_wxyz;
        assert.ok(
          closeVec(quaternionToZRec(qw, qx, qy, qz), el.frame.z, 1e-6),
          `${fixtureName} ops_tag ${el.ops_tag}: quaternionToZRec matches golden frame.z`,
        );
      }

      const avail = r.availableLineStations(stageId);
      assert.ok(avail.length > 0, `${fixtureName}: has availableLineStations`);

      const lineElement = blobs.i32(loaded.mesh.lineElement);
      const drawnLinePositions = blobs.f32(loaded.mesh.linePositions);
      const lineNodes = blobs.get(loaded.mesh.vertexNodes.line) as Float64Array;

      for (const comp of avail) {
        const canon = COMP_TO_CANON[comp];
        const slab = r.readLineStations(stageId, comp);

        for (let stepIdx = 0; stepIdx < slab.nSteps; stepIdx++) {
          const stepData = r.readLineStationStep(stageId, comp, stepIdx);
          // Check raw station read against golden
          for (const el of golden.elements) {
            const expected = winningBucketStation(el, canon);
            assert.ok(expected !== null, `${fixtureName} ${comp}: golden station entry for ops_tag ${el.ops_tag}`);
            const idxs: number[] = [];
            for (let i = 0; i < stepData.elementIndex.length; i++) {
              if (stepData.elementIndex[i] === el.ops_tag) idxs.push(i);
            }
            assert.equal(idxs.length, expected.xi.length, `${fixtureName} ${comp} step ${stepIdx} ops_tag ${el.ops_tag} station count`);
            const actualXi = idxs.map((i) => stepData.stationNaturalCoord[i]!);
            const actualVals = idxs.map((i) => stepData.values[i]!);
            assert.ok(closeVec(actualXi, expected.xi, 1e-9), `${fixtureName} ${comp} xi`);
            assert.ok(closeVec(actualVals, expected.values[stepIdx]!, 1e-9), `${fixtureName} ${comp} step ${stepIdx} values`);
          }

          // Check 3-D diagram build (frame xLocal, extrudeDir, expanded stations, zero skipped beams)
          const geom = buildLineDiagram3D({
            decls: loaded.decls,
            meshElements: loaded.mesh.elements,
            lineElement,
            drawnLinePositions,
            lineNodes,
            stepData,
            join,
            ndm: golden.ndm,
            scale: 1,
          });
          assert.equal(geom.skippedUnorientedBeams, 0, `${fixtureName} ${comp}: 0 unoriented beams`);
          assert.equal(geom.notice, null, `${fixtureName} ${comp}: no error/skip notice`);
          assert.equal(geom.beams.length, golden.elements.length, `${fixtureName} ${comp}: all beams drawn`);

          for (const el of golden.elements) {
            const beam = geom.beams.find((b) => b.elementId === el.ops_tag);
            assert.ok(beam !== undefined, `${fixtureName} ${comp}: beam ops_tag ${el.ops_tag} present`);
            assert.ok(closeVec(beam.pStart, el.coords_i, 1e-6), `${fixtureName} ${comp} ops_tag ${el.ops_tag} pStart`);
            assert.ok(closeVec(beam.pEnd, el.coords_j, 1e-6), `${fixtureName} ${comp} ops_tag ${el.ops_tag} pEnd`);
            assert.ok(close(beam.length, el.length, 1e-6), `${fixtureName} ${comp} ops_tag ${el.ops_tag} length`);
            assert.ok(closeVec(beam.xLocal, el.frame.x, 1e-6), `${fixtureName} ${comp} ops_tag ${el.ops_tag} xLocal`);
            const expectedExtrude: readonly [number, number, number] =
              localAxisForComponent(comp) === "y"
                ? el.frame.y
                : comp === "My" || comp === "kappaY"
                  ? [-el.frame.z[0], -el.frame.z[1], -el.frame.z[2]]
                  : el.frame.z;
            assert.ok(
              closeVec(beam.extrudeDir, expectedExtrude, 1e-6),
              `${fixtureName} ${comp} ops_tag ${el.ops_tag} extrudeDir (${beam.extrudeDir.join(",")} vs ${expectedExtrude.join(",")})`,
            );

            const expected = winningBucketStation(el, canon)!;
            if (expected.xi.length === 1) {
              // Single-station basicForce N/T expands to [-1, +1]
              assert.ok(closeVec(Array.from(beam.xi), [-1, 1], 1e-9));
              const v = expected.values[stepIdx]![0]!;
              assert.ok(closeVec(Array.from(beam.values), [v, v], 1e-9));
            } else {
              assert.ok(closeVec(Array.from(beam.xi), expected.xi, 1e-9));
              assert.ok(closeVec(Array.from(beam.values), expected.values[stepIdx]!, 1e-9));
            }
          }

          // On line_oracle_ss_udl at step 1 for Mz: verify both Fwd (ops_tag 4) and Rev (ops_tag 6)
          // extrude midspan sagging moment downward (global Y < 0) under default flipSign=true!
          if (fixtureName === "line_oracle_ss_udl" && comp === "Mz" && stepIdx === 1) {
            const bFwd = geom.beams.find((b) => b.elementId === 4)!;
            const bRev = geom.beams.find((b) => b.elementId === 6)!;
            const hitFwd = projectHoverNeedle(bFwd, [3, 0, 0], geom.effectiveScale, true);
            const hitRev = projectHoverNeedle(bRev, [3, 0, 0], geom.effectiveScale, true);
            assert.ok(hitFwd.pTop[1] < -0.1, `Fwd midspan sagging Mz extrudes toward -Y (${hitFwd.pTop[1]})`);
            assert.ok(closeVec(hitFwd.pTop, hitRev.pTop, 1e-6), "Fwd and Rev meet at identical 3-D midspan tip");
          }
        }
      }
    } finally {
      r.close();
    }
  });
}

