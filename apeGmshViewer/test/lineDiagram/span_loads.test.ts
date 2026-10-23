// Unit and OpenSees oracle tests for Slice E2 (`src/state/spanLoads.ts`):
// - `parseEleLoadRow` & `parseEleLoadBeamUniform` (2-D 1/2/4/6-arg & 3-D 2/3/5/8-arg)
// - `evaluateTimeSeriesFactor` (`Constant`, `Linear`, `Path`, `Trig`)
// - `reconstructBeamElementDiagram` & `findZeroShearRoots` (exact breakpoints,
//   parabolic/cubic moments, ruled Vz/My sign convention)
// - `checkBeamSpanClosure` (node-j equilibrium closure & fallback)
// - `resolveStageSpanLoads`, `reconstructSpanDiagramsForStep`, and
//   `reconstructSpanDiagramsFromState` against committed OpenSees oracle
//   fixtures (`line_oracle_ss_udl`, `line_oracle_cantilever3d`,
//   `line_oracle_column3d`, `line_oracle_fbc_legendre`, `line_oracle_basic_only`).

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openLadruno, type LadrunoLineStationStep, type LineStationComponent } from "../../src/reader/ladruno.ts";
import { openModel } from "../../src/reader/node.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { joinLadrunoElements } from "../../src/state/lineDiagram.ts";
import { loadModel } from "../../src/state/load.ts";
import {
  BEAM_FORCE_COMPONENTS,
  checkBeamSpanClosure,
  evaluateTimeSeriesFactor,
  extractElementLoadRows,
  findZeroShearRoots,
  integrateSegmentLoadDouble,
  integrateSegmentLoadSingle,
  isBeamForceComponent,
  parseEleLoadBeamUniform,
  parseEleLoadRow,
  reconstructBeamElementDiagram,
  reconstructSpanDiagramsForStep,
  reconstructSpanDiagramsFromState,
  resolveElementGeomTransfType,
  resolveStageSpanLoads,
  scaleBeamSpanLoad,
  unwrapSpatialNdm,
} from "../../src/state/spanLoads.ts";
import type { Decl, DeclParams, DeclPath, Definitions, DefObject, SequenceZone } from "../../src/state/types.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;

const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

function interpAtXi(xi: Float64Array, values: Float64Array, targetXi: number): number {
  for (let i = 0; i < xi.length; i++) {
    if (Math.abs(xi[i]! - targetXi) <= 1e-11) return values[i]!;
  }
  for (let i = 0; i + 1 < xi.length; i++) {
    const x0 = xi[i]!;
    const x1 = xi[i + 1]!;
    if (targetXi >= x0 && targetXi <= x1) {
      const t = (targetXi - x0) / (x1 - x0);
      return values[i]! + t * (values[i + 1]! - values[i]!);
    }
  }
  throw new Error(`targetXi=${targetXi} outside [${xi[0]}, ${xi[xi.length - 1]}]`);
}

// -- 1. Parsing `-beamUniform` & unsupported `eleLoad` rows -------------------

test("parseEleLoadBeamUniform and parseEleLoadRow parse all 2-D and 3-D -beamUniform variants and reject unsupported loads", () => {
  assert.deepEqual(BEAM_FORCE_COMPONENTS, ["N", "Vy", "Vz", "T", "My", "Mz"]);
  assert.equal(isBeamForceComponent("Mz"), true);
  assert.equal(isBeamForceComponent("kappaZ"), false);
  assert.equal(unwrapSpatialNdm(2), 2);
  assert.equal(unwrapSpatialNdm({ ndm: 3, source: "meta" }), 3);
  assert.equal(unwrapSpatialNdm(null), null);

  // 2-D 1-value (wy only, as emitted by line_oracle_ss_udl) and 2-value (wy, wx) with trailing empty strings
  const r2d1 = parseEleLoadBeamUniform(["-ele", "1", "2", "3", "4", "-type", "-beamUniform", "-10", "", ""], 2);
  assert.deepEqual(r2d1.eleTags, [1, 2, 3, 4]);
  assert.equal(r2d1.load.aL, 0);
  assert.equal(r2d1.load.bL, 1);
  assert.equal(r2d1.load.wya, -10);
  assert.equal(r2d1.load.wyb, -10);
  assert.equal(r2d1.load.wxa, 0);
  assert.equal(r2d1.load.wxb, 0);

  const r2d2 = parseEleLoadBeamUniform(["-ele", "1", "2", "-type", "-beamUniform", "-12.5", "3.0", ""], 2);
  assert.deepEqual(r2d2.eleTags, [1, 2]);
  assert.equal(r2d2.load.wya, -12.5);
  assert.equal(r2d2.load.wyb, -12.5);
  assert.equal(r2d2.load.wxa, 3.0);
  assert.equal(r2d2.load.wxb, 3.0);

  // 2-D 4-value partial uniform with -range
  const r2d4 = parseEleLoadBeamUniform(["-range", "10", "12", "-type", "-beamUniform", "-8.0", "1.5", "0.25", "0.75"], 2);
  assert.deepEqual(r2d4.eleTags, [10, 11, 12]);
  assert.equal(r2d4.load.aL, 0.25);
  assert.equal(r2d4.load.bL, 0.75);
  assert.equal(r2d4.load.wya, -8.0);
  assert.equal(r2d4.load.wyb, -8.0);
  assert.equal(r2d4.load.wxa, 1.5);
  assert.equal(r2d4.load.wxb, 1.5);

  // 2-D 6-value trapezoidal (wya, wxa, aL, bL, wyb, wxb)
  const r2d6 = parseEleLoadBeamUniform(["-ele", "5", "-type", "-beamUniform", "-4.0", "1.0", "0.2", "0.8", "-12.0", "3.0"], 2);
  assert.deepEqual(r2d6.eleTags, [5]);
  assert.equal(r2d6.load.aL, 0.2);
  assert.equal(r2d6.load.bL, 0.8);
  assert.equal(r2d6.load.wya, -4.0);
  assert.equal(r2d6.load.wyb, -12.0);
  assert.equal(r2d6.load.wxa, 1.0);
  assert.equal(r2d6.load.wxb, 3.0);

  // 3-D 2-value (wy, wz), 3-value (wy, wz, wx), 5-value (wy, wz, wx, aL, bL)
  const r3d2 = parseEleLoadBeamUniform(["-ele", "7", "-type", "-beamUniform", "-10.0", "-5.0"], 3);
  assert.equal(r3d2.load.wya, -10.0);
  assert.equal(r3d2.load.wza, -5.0);
  assert.equal(r3d2.load.wxa, 0.0);

  const r3d3 = parseEleLoadBeamUniform(["-ele", "7", "-type", "-beamUniform", "-10.0", "-5.0", "2.0"], 3);
  assert.equal(r3d3.load.wya, -10.0);
  assert.equal(r3d3.load.wza, -5.0);
  assert.equal(r3d3.load.wxa, 2.0);

  const r3d5 = parseEleLoadBeamUniform(["-eleRange", "3", "4", "-type", "-beamUniform", "-10.0", "-5.0", "2.0", "0.1", "0.9"], 3);
  assert.deepEqual(r3d5.eleTags, [3, 4]);
  assert.equal(r3d5.load.aL, 0.1);
  assert.equal(r3d5.load.bL, 0.9);
  assert.equal(r3d5.load.wya, -10.0);
  assert.equal(r3d5.load.wza, -5.0);
  assert.equal(r3d5.load.wxa, 2.0);

  // 3-D 8-value canonical order: (wya, wza, wxa, aL, bL, wyb, wzb, wxb)
  const r3d8Canon = parseEleLoadBeamUniform(
    ["-ele", "7", "-type", "-beamUniform", "-2.0", "-3.0", "0.5", "0.1", "0.9", "-6.0", "-9.0", "1.5"],
    3,
  );
  assert.equal(r3d8Canon.tokenOrder, "canonical");
  assert.equal(r3d8Canon.load.aL, 0.1);
  assert.equal(r3d8Canon.load.bL, 0.9);
  assert.equal(r3d8Canon.load.wya, -2.0);
  assert.equal(r3d8Canon.load.wyb, -6.0);
  assert.equal(r3d8Canon.load.wza, -3.0);
  assert.equal(r3d8Canon.load.wzb, -9.0);
  assert.equal(r3d8Canon.load.wxa, 0.5);
  assert.equal(r3d8Canon.load.wxb, 1.5);

  // 3-D 8-value Tcl C++ order: (Wya, Wzb, aL, bL, Wyb, Wza, Wxa, Wxb) where aL=0.1, bL=0.9 are at indices [2, 3]
  const r3d8Tcl = parseEleLoadBeamUniform(
    ["-ele", "7", "-type", "-beamUniform", "-2.0", "-9.0", "0.1", "0.9", "-6.0", "-3.0", "0.5", "1.5"],
    3,
  );
  assert.equal(r3d8Tcl.tokenOrder, "tcl_cpp_3d8");
  assert.equal(r3d8Tcl.load.aL, 0.1);
  assert.equal(r3d8Tcl.load.bL, 0.9);
  assert.equal(r3d8Tcl.load.wya, -2.0);
  assert.equal(r3d8Tcl.load.wyb, -6.0);
  assert.equal(r3d8Tcl.load.wza, -3.0);
  assert.equal(r3d8Tcl.load.wzb, -9.0);
  assert.equal(r3d8Tcl.load.wxa, 0.5);
  assert.equal(r3d8Tcl.load.wxb, 1.5);

  // Unsupported load types (-beamPoint, -beamThermal, -selfWeight) and invalid segment bounds
  const ptLoad = parseEleLoadRow(["-ele", "1", "-type", "-beamPoint", "-10.0", "0.5"], 2);
  assert.equal(ptLoad.kind, "unsupported_load");
  if (ptLoad.kind === "unsupported_load") {
    assert.equal(ptLoad.loadType, "-beamPoint");
    assert.deepEqual(ptLoad.eleTags, [1]);
    assert.match(ptLoad.reason, /-beamPoint/);
  }
  assert.throws(() => parseEleLoadBeamUniform(["-ele", "1", "-type", "-beamPoint", "-10.0", "0.5"], 2), RangeError);
  assert.throws(
    () => parseEleLoadBeamUniform(["-ele", "1", "-type", "-beamUniform", "-10.0", "0.0", "0.8", "0.2"], 2),
    RangeError,
  );
});

test("extractElementLoadRows handles both flat 2-D rows and h5wasm nested 2-D string dataset rows", () => {
  const flatTable = {
    path: "/opensees/patterns/Plain_1/element_loads",
    columns: ["[0]", "[1]", "[2]", "[3]", "[4]", "[5]"],
    rows: [
      ["-ele", "1", "-type", "-beamUniform", "-10", ""],
      ["-ele", "2", "-type", "-beamUniform", "-20", ""],
    ],
  };
  const nestedTable = {
    path: "/opensees/patterns/Plain_1/element_loads",
    columns: ["[0]", "[1]", "[2]", "[3]", "[4]", "[5]"],
    rows: [
      [
        ["-ele", "1", "-type", "-beamUniform", "-10", ""],
        ["-ele", "2", "-type", "-beamUniform", "-20", ""],
      ],
    ],
  };
  assert.deepEqual(extractElementLoadRows(flatTable), [
    ["-ele", "1", "-type", "-beamUniform", "-10"],
    ["-ele", "2", "-type", "-beamUniform", "-20"],
  ]);
  assert.deepEqual(extractElementLoadRows(nestedTable), [
    ["-ele", "1", "-type", "-beamUniform", "-10"],
    ["-ele", "2", "-type", "-beamUniform", "-20"],
  ]);
});

// -- 2. Closed-form analytical verification of `reconstructBeamElementDiagram` -

test("reconstructBeamElementDiagram reproduces exact parabolic Mz(x) on full-span simply-supported UDL (L=6, w=-10 -> Mz(L/2)=+45; w=-10000 -> Mz(L/2)=+45000)", () => {
  // Case A: L = 6 m, w_y = -10 kN/m -> Vy(0) = +30 kN, Mz(0) = 0, Mz(3) = +45 kN*m, Vy(6) = -30 kN, Mz(6) = 0
  const L = 6.0;
  const w = -10.0;
  const diag = reconstructBeamElementDiagram({
    elementId: 1,
    length: L,
    nodalReactionsI: { VyI: 30.0, MzI: 0.0 },
    loads: [{ aL: 0.0, bL: 1.0, wya: w, wyb: w, wza: 0, wzb: 0, wxa: 0, wxb: 0 }],
    nSub: 16,
  });

  // Zero-shear root at x = 3.0 (xi = 0.0) must be present as an exact grid point
  const midIdx = diag.xi.findIndex((xi) => Math.abs(xi - 0.0) <= 1e-12);
  assert.ok(midIdx >= 0, "midspan zero-shear root xi = 0 is in diagram.xi");
  assert.ok(close(diag.Vy[midIdx]!, 0.0, 1e-12));
  assert.ok(close(diag.Mz[midIdx]!, 45.0, 1e-12));

  // Every point matches exact analytical Vy(x) = 30 - 10x and Mz(x) = 30x - 5x^2
  for (let i = 0; i < diag.x.length; i++) {
    const x = diag.x[i]!;
    assert.ok(close(diag.Vy[i]!, 30.0 + w * x, 1e-11));
    assert.ok(close(diag.Mz[i]!, 30.0 * x + 0.5 * w * x * x, 1e-11));
  }
  const closure = checkBeamSpanClosure(diag, { VyJ: -30.0, MzJ: 0.0 });
  assert.equal(closure.ok, true);

  // Case B: L = 6 m, w_y = -10000 N/m -> Mz(L/2) = +45000 N*m (plan acceptance criterion)
  const diag45k = reconstructBeamElementDiagram({
    elementId: 1,
    length: 6.0,
    endForcesI: { Vy0: 30000.0, Mz0: 0.0 },
    loads: [{ aL: 0.0, bL: 1.0, wya: -10000.0, wyb: -10000.0, wza: 0, wzb: 0, wxa: 0, wxb: 0 }],
    nSub: 16,
  });
  assert.ok(close(interpAtXi(diag45k.xi, diag45k.Mz, 0.0), 45000.0, 1e-11));
  assert.equal(checkBeamSpanClosure(diag45k, { VyJ: -30000.0, MzJ: 0.0 }).ok, true);
});

test("reconstructBeamElementDiagram matches closed-form partial uniform, triangular/trapezoidal, 3-D Vz/My, and axial loads", () => {
  // 1. Partial uniform load on [0.25, 0.75] of L = 8 m, w_y = -12 kN/m
  const L1 = 8.0;
  const w1 = -12.0;
  const R1 = 0.5 * Math.abs(w1) * (0.5 * L1); // 24 kN
  const diagPart = reconstructBeamElementDiagram({
    elementId: 10,
    length: L1,
    endForcesI: { Vy0: R1, Mz0: 0.0 },
    loads: [{ aL: 0.25, bL: 0.75, wya: w1, wyb: w1, wza: 0, wzb: 0, wxa: 0, wxb: 0 }],
    nSub: 8,
  });
  // Breakpoints x = 2 (xi = -0.5), x = 4 (xi = 0, zero shear), x = 6 (xi = +0.5) must be exact
  assert.ok(diagPart.x.some((x) => Math.abs(x - 2.0) <= 1e-12));
  assert.ok(diagPart.x.some((x) => Math.abs(x - 4.0) <= 1e-12));
  assert.ok(diagPart.x.some((x) => Math.abs(x - 6.0) <= 1e-12));
  for (let i = 0; i < diagPart.x.length; i++) {
    const x = diagPart.x[i]!;
    const vyEx = x <= 2.0 ? R1 : x <= 6.0 ? R1 + w1 * (x - 2.0) : R1 + w1 * 4.0;
    const mzEx =
      x <= 2.0
        ? R1 * x
        : x <= 6.0
          ? R1 * x + 0.5 * w1 * (x - 2.0) * (x - 2.0)
          : R1 * x + w1 * 4.0 * (x - 4.0);
    assert.ok(close(diagPart.Vy[i]!, vyEx, 1e-11));
    assert.ok(close(diagPart.Mz[i]!, mzEx, 1e-11));
  }
  assert.ok(close(Math.max(...diagPart.Mz), 72.0, 1e-12));

  // 2. Triangular load wya = 0, wyb = -18 kN/m on L = 6 m -> zero shear at x* = L / sqrt(3)
  const L2 = 6.0;
  const wPeak = -18.0;
  const Vy0Tri = (Math.abs(wPeak) * L2) / 6.0; // 18 kN
  const diagTri = reconstructBeamElementDiagram({
    elementId: 20,
    length: L2,
    endForcesI: { Vy0: Vy0Tri, Mz0: 0.0 },
    loads: [{ aL: 0.0, bL: 1.0, wya: 0.0, wyb: wPeak, wza: 0, wzb: 0, wxa: 0, wxb: 0 }],
    nSub: 12,
  });
  const xStar = L2 / Math.sqrt(3.0);
  const mzMaxExpected = (Math.abs(wPeak) * L2 * L2) / (9.0 * Math.sqrt(3.0));
  const rootIdx = diagTri.x.findIndex((x) => Math.abs(x - xStar) <= 1e-9);
  assert.ok(rootIdx >= 0, `exact zero-shear root x*=${xStar} present in grid`);
  assert.ok(Math.abs(diagTri.Vy[rootIdx]!) <= 1e-11);
  assert.ok(close(diagTri.Mz[rootIdx]!, mzMaxExpected, 1e-11));
  assert.equal(checkBeamSpanClosure(diagTri, { VyJ: -36.0, MzJ: 0.0 }).ok, true);

  // 3. 3-D transverse z-load (ruled Slice S #2490 sign convention: dVz/dx = -wz, dMy/dx = +Vz)
  // Simply-supported 3-D beam L = 4 m with downward wz = -12 kN/m -> Vz_I = +24, Vz(0) = -24, Vz(4) = +24, My(2) = -24
  const L3 = 4.0;
  const wz = -12.0;
  const diag3d = reconstructBeamElementDiagram({
    elementId: 30,
    length: L3,
    nodalReactionsI: { VzI: 24.0, MyI: 0.0 },
    loads: [{ aL: 0.0, bL: 1.0, wya: 0, wyb: 0, wza: wz, wzb: wz, wxa: 0, wxb: 0 }],
    nSub: 8,
  });
  assert.ok(close(diag3d.Vz[0]!, -24.0, 1e-12));
  assert.ok(close(interpAtXi(diag3d.xi, diag3d.Vz, 0.0), 0.0, 1e-12));
  assert.ok(close(diag3d.Vz[diag3d.Vz.length - 1]!, +24.0, 1e-12));
  assert.ok(close(interpAtXi(diag3d.xi, diag3d.My, 0.0), -24.0, 1e-12));
  assert.equal(checkBeamSpanClosure(diag3d, { VzJ: +24.0, MyJ: 0.0 }).ok, true);

  // 4. Distributed trapezoidal axial load wxa = 6, wxb = 18 on L = 5 m -> N(x) = N0 - (6x + 1.2 x^2)
  const L4 = 5.0;
  const diagAxial = reconstructBeamElementDiagram({
    elementId: 40,
    length: L4,
    endForcesI: { N0: 60.0 },
    loads: [{ aL: 0.0, bL: 1.0, wya: 0, wyb: 0, wza: 0, wzb: 0, wxa: 6.0, wxb: 18.0 }],
    nSub: 8,
  });
  for (let i = 0; i < diagAxial.x.length; i++) {
    const x = diagAxial.x[i]!;
    assert.ok(close(diagAxial.N[i]!, 60.0 - (6.0 * x + 1.2 * x * x), 1e-11));
  }
  assert.equal(checkBeamSpanClosure(diagAxial, { NJ: 0.0 }).ok, true);

  // Also exercise direct single/double integral & root-finder helpers on edge cases
  assert.equal(integrateSegmentLoadSingle(0.5, 4.0, 0.5, 0.8, -10, -10), 0.0);
  assert.equal(integrateSegmentLoadDouble(0.5, 4.0, 0.5, 0.8, -10, -10), 0.0);
  const scaled = scaleBeamSpanLoad({ aL: 0.1, bL: 0.9, wya: -10, wyb: -20, wza: -4, wzb: -8, wxa: 2, wxb: 6 }, 0.5);
  assert.deepEqual(scaled, { aL: 0.1, bL: 0.9, wya: -5, wyb: -10, wza: -2, wzb: -4, wxa: 1, wxb: 3 });
  assert.deepEqual(findZeroShearRoots(10.0, 4.0, [], "y", 1), []);
});

// -- 3. Time-series factor `\lambda(t)` evaluation ----------------------------

test("evaluateTimeSeriesFactor evaluates Constant, Linear, Path (dt, explicit time, archived table), and Trig series", () => {
  // Constant
  const cDecl: DeclParams = {
    row: 0,
    type: "Constant",
    params: { factor: 2.5 },
    paramsNames: null,
    refs: { transf: [], integration: [], section: [] },
  };
  assert.deepEqual(evaluateTimeSeriesFactor(cDecl, 0.0), { ok: true, factor: 2.5 });
  assert.deepEqual(evaluateTimeSeriesFactor(cDecl, 99.0), { ok: true, factor: 2.5 });

  // Linear (with t_start)
  const lDecl: DeclParams = {
    row: 1,
    type: "Linear",
    params: { factor: 3.0, t_start: 1.0 },
    paramsNames: null,
    refs: { transf: [], integration: [], section: [] },
  };
  assert.deepEqual(evaluateTimeSeriesFactor(lDecl, 0.5), { ok: true, factor: 0.0 });
  assert.deepEqual(evaluateTimeSeriesFactor(lDecl, 2.0), { ok: true, factor: 6.0 });

  // Path with dt, start_time, prepend_zero, use_last
  const pDtDecl: DeclParams = {
    row: 2,
    type: "Path",
    params: {
      values: [10.0, 20.0],
      dt: 0.5,
      start_time: 1.0,
      prepend_zero: true,
      use_last: true,
      factor: 2.0,
    },
    paramsNames: null,
    refs: { transf: [], integration: [], section: [] },
  };
  // Knots at t = [1.0, 1.5, 2.0] with values = [0, 10, 20] * 2 = [0, 20, 40]
  assert.deepEqual(evaluateTimeSeriesFactor(pDtDecl, 0.5), { ok: true, factor: 0.0 });
  assert.deepEqual(evaluateTimeSeriesFactor(pDtDecl, 1.25), { ok: true, factor: 10.0 });
  assert.deepEqual(evaluateTimeSeriesFactor(pDtDecl, 1.75), { ok: true, factor: 30.0 });
  assert.deepEqual(evaluateTimeSeriesFactor(pDtDecl, 5.0), { ok: true, factor: 40.0 });

  // Path with explicit time array and use_last = false
  const pTimeDecl: DeclParams = {
    row: 3,
    type: "Path",
    params: {
      time: [0.0, 1.0, 3.0],
      values: [0.0, 1.0, 5.0],
      factor: 1.5,
      use_last: false,
    },
    paramsNames: null,
    refs: { transf: [], integration: [], section: [] },
  };
  assert.deepEqual(evaluateTimeSeriesFactor(pTimeDecl, 2.0), { ok: true, factor: 1.5 * 3.0 });
  assert.deepEqual(evaluateTimeSeriesFactor(pTimeDecl, 4.0), { ok: true, factor: 0.0 });

  // Path from DefObject archived tables
  const pDefObj: DefObject = {
    path: "/opensees/time_series/Path_1",
    groupName: "Path_1",
    type: "Path",
    tag: 1,
    params: ["-dt", 0.5, "-factor", 2.0],
    attrs: {},
    tables: {
      values: { path: "/opensees/time_series/Path_1/values", columns: ["value"], rows: [[0.0], [1.0], [2.0]] },
    },
  };
  assert.deepEqual(evaluateTimeSeriesFactor(pDefObj, 0.75), { ok: true, factor: 3.0 });

  // Path referencing external file without inline values is refused
  const pExtDecl: DeclParams = {
    row: 4,
    type: "Path",
    params: { file_path: "ground_motion.txt" },
    paramsNames: null,
    refs: { transf: [], integration: [], section: [] },
  };
  const extRes = evaluateTimeSeriesFactor(pExtDecl, 1.0);
  assert.equal(extRes.ok, false);
  if (!extRes.ok) assert.match(extRes.reason, /external file/);

  // Trig series
  const trigDecl: DeclParams = {
    row: 5,
    type: "Trig",
    params: { t_start: 0.0, t_finish: 4.0, period: 2.0, factor: 10.0, shift: 0.0 },
    paramsNames: null,
    refs: { transf: [], integration: [], section: [] },
  };
  const trigAtQuarter = evaluateTimeSeriesFactor(trigDecl, 0.5);
  assert.equal(trigAtQuarter.ok, true);
  if (trigAtQuarter.ok) assert.ok(close(trigAtQuarter.factor, 10.0, 1e-12));
  assert.deepEqual(evaluateTimeSeriesFactor(trigDecl, 5.0), { ok: true, factor: 0.0 });
});

// -- 4. OpenSees Oracle Fixture Verification (`line_oracle_ss_udl` & others) ---

test("OpenSees oracle line_oracle_ss_udl: reconstructs exact parabolic Mz(x) and linear Vy(x) at t=0.5 and t=1.0 across all 4 elements", async () => {
  const modelPath = fileURLToPath(new URL("../../fixtures/line_oracle_ss_udl.h5", import.meta.url));
  const ladrunoPath = fileURLToPath(new URL("../../fixtures/line_oracle_ss_udl.ladruno", import.meta.url));

  const model = await openModel(modelPath);
  const ndm = model.meta["ndm"] as 2 | 3;
  const blobs = new BlobStore();
  const loaded = loadModel(model, blobs);
  const r = openLadruno(H5, ladrunoPath);
  try {
    const stage = r.index.stages[0]!;
    const join = joinLadrunoElements(loaded.decls, { state: "paired", by: "model_hash" }, r.index.provenance);
    assert.equal(join.ok, true);
    if (!join.ok) return;

    // 6 m simply-supported beam discretized into 4 elements of L = 1.5 m each:
    // - ops_tag 3 ([0, 1.5]) and ops_tag 4 ([1.5, 3.0]): forward (+X), wy = -10.0 * t
    // - ops_tag 5 ([6.0, 4.5]) and ops_tag 6 ([4.5, 3.0]): reversed (-X), wy = +10.0 * t
    const elementLengths = new Map<number, number>([
      [3, 1.5],
      [4, 1.5],
      [5, 1.5],
      [6, 1.5],
    ]);

    for (let stepIdx = 0; stepIdx < stage.time.length; stepIdx++) {
      const time = stage.time[stepIdx]!; // 0.5 at step 0, 1.0 at step 1
      const wFwd = -10.0 * time; // -5 at step 0, -10 at step 1
      const wRev = +10.0 * time; // +5 at step 0, +10 at step 1

      const stepByComponent: Partial<Record<LineStationComponent, LadrunoLineStationStep>> = {
        N: r.readLineStationStep(stage.id, "N", stepIdx),
        Vy: r.readLineStationStep(stage.id, "Vy", stepIdx),
        Mz: r.readLineStationStep(stage.id, "Mz", stepIdx),
      };

      // Verify stage load resolution
      const stageLoads = resolveStageSpanLoads({
        definitions: loaded.definitions,
        sequence: loaded.sequence,
        ndm,
        resultsStage: stage,
        stageIndex: 0,
        time,
      });
      assert.equal(stageLoads.ok, true);
      if (!stageLoads.ok) return;
      for (const fwdTag of [3, 4]) {
        const lds = stageLoads.loadsByElementTag.get(fwdTag);
        assert.ok(lds && lds.length === 1);
        assert.ok(close(lds[0]!.wya, wFwd, 1e-12));
        assert.ok(close(lds[0]!.wyb, wFwd, 1e-12));
      }
      for (const revTag of [5, 6]) {
        const lds = stageLoads.loadsByElementTag.get(revTag);
        assert.ok(lds && lds.length === 1);
        assert.ok(close(lds[0]!.wya, wRev, 1e-12));
        assert.ok(close(lds[0]!.wyb, wRev, 1e-12));
      }

      // Reconstruct Mz across all 4 elements via reconstructSpanDiagramsFromState
      const mzMap = reconstructSpanDiagramsFromState(
        {
          decls: loaded.decls,
          definitions: loaded.definitions,
          sequence: loaded.sequence,
          ndm,
          results: {
            path: ladrunoPath,
            stages: r.index.stages,
          },
          phase: {
            axis: [{ kind: "results", stage: 0, step: stepIdx }],
            at: { kind: "results", stage: 0, step: stepIdx },
          },
        },
        stepByComponent,
        elementLengths,
        { state: "paired", by: "model_hash" },
        "Mz",
        { provenance: r.index.provenance },
      );

      for (const elId of [3, 4, 5, 6]) {
        const rec = mzMap.get(elId);
        assert.ok(rec !== undefined, `element ${elId} present in reconstruction map`);
        assert.equal(rec.mode, "exact", `element ${elId} uses exact reconstruction at t=${time}`);
      }

      // Check Element 3 ([0, 1.5], forward):
      // Mz(x) = 30*t*x - 5*t*x^2 -> at xi = 0 (x = 0.75): 19.6875 * t (vs linear interp 16.875 * t)
      const e3 = mzMap.get(3)!;
      assert.equal(e3.mode, "exact");
      if (e3.mode === "exact") {
        assert.ok(close(interpAtXi(e3.xi, e3.values, -1.0), 0.0, 1e-9));
        assert.ok(close(interpAtXi(e3.xi, e3.values, 0.0), 19.6875 * time, 1e-6));
        assert.ok(close(interpAtXi(e3.xi, e3.values, 1.0), 33.75 * time, 1e-6));
      }

      // Check Element 4 ([1.5, 3.0], forward, reaching mid-span of 6 m beam at xi = +1):
      // Mz(x_local) = 33.75*t + 15*t*x - 5*t*x^2 -> at xi = 0 (x = 0.75): 42.1875 * t (vs linear interp 39.375 * t);
      // at xi = +1 (x = 1.5, mid-span): Mz = 45.0 * t and Vy = 0.0!
      const e4 = mzMap.get(4)!;
      assert.equal(e4.mode, "exact");
      if (e4.mode === "exact") {
        assert.ok(close(interpAtXi(e4.xi, e4.values, -1.0), 33.75 * time, 1e-6));
        assert.ok(close(interpAtXi(e4.xi, e4.values, 0.0), 42.1875 * time, 1e-6));
        assert.ok(close(interpAtXi(e4.xi, e4.values, 1.0), 45.0 * time, 1e-6));
        assert.ok(close(interpAtXi(e4.diagram.xi, e4.diagram.Vy, 1.0), 0.0, 1e-6));
      }

      // Check Reversed Elements 5 ([6.0, 4.5]) and 6 ([4.5, 3.0]):
      const e5 = mzMap.get(5)!;
      const e6 = mzMap.get(6)!;
      assert.equal(e5.mode, "exact");
      assert.equal(e6.mode, "exact");
      if (e5.mode === "exact" && e6.mode === "exact") {
        assert.ok(close(interpAtXi(e5.xi, e5.values, 0.0), -19.6875 * time, 1e-6));
        assert.ok(close(interpAtXi(e6.xi, e6.values, 0.0), -42.1875 * time, 1e-6));
        assert.ok(close(interpAtXi(e6.xi, e6.values, 1.0), -45.0 * time, 1e-6));
        assert.ok(close(interpAtXi(e6.diagram.xi, e6.diagram.Vy, 1.0), 0.0, 1e-6));
      }
    }
  } finally {
    r.close();
  }
});

test("OpenSees 3-D and FBC/basic fixtures: cantilever3d and column3d reconstruct exact diagrams; fbc_legendre and basic_only (1-station N/T) fall back cleanly", async () => {
  // 1. line_oracle_cantilever3d (3-D cantilever, 2 elements [2, 3] of L = 2 m each): exact mode for all 6 force components
  for (const [fixtureName, elemLengths] of [
    ["line_oracle_cantilever3d", new Map([[2, 2.0], [3, 2.0]])],
    ["line_oracle_column3d", new Map([[2, 1.5], [3, 1.5]])],
  ] as const) {
    const model = await openModel(fileURLToPath(new URL(`../../fixtures/${fixtureName}.h5`, import.meta.url)));
    const ndm = model.meta["ndm"] as 2 | 3;
    const blobs = new BlobStore();
    const loaded = loadModel(model, blobs);
    const r = openLadruno(H5, fileURLToPath(new URL(`../../fixtures/${fixtureName}.ladruno`, import.meta.url)));
    try {
      const stage = r.index.stages[0]!;
      const join = joinLadrunoElements(loaded.decls, { state: "paired", by: "model_hash" }, r.index.provenance);
      const stepByComponent: Partial<Record<LineStationComponent, LadrunoLineStationStep>> = {};
      for (const c of BEAM_FORCE_COMPONENTS) {
        stepByComponent[c] = r.readLineStationStep(stage.id, c, 0);
      }
      for (const c of BEAM_FORCE_COMPONENTS) {
        const res = reconstructSpanDiagramsForStep({
          component: c,
          stepByComponent,
          elementLengths: elemLengths,
          join,
          decls: loaded.decls,
          definitions: loaded.definitions,
          sequence: loaded.sequence,
          ndm,
          resultsStage: stage,
          time: stage.time[0]!,
        });
        for (const eid of elemLengths.keys()) {
          const eRec = res.get(eid)!;
          assert.equal(eRec.mode, "exact", `${fixtureName} element ${eid} ${c} is exact`);
        }
      }
    } finally {
      r.close();
    }
  }

  // 2a. line_oracle_fbc_lobatto (forceBeamColumn ops_tag 2 with 4-point Lobatto stations including xi = +/-1):
  // Reconstructs exact parabolic Mz(x) = -24 + 11*x - x^2 at t=1.0 matching interior Lobatto points xi = +/-1/sqrt(5)
  {
    const model = await openModel(fileURLToPath(new URL("../../fixtures/line_oracle_fbc_lobatto.h5", import.meta.url)));
    const ndm = model.meta["ndm"] as 2 | 3;
    const blobs = new BlobStore();
    const loaded = loadModel(model, blobs);
    const r = openLadruno(H5, fileURLToPath(new URL("../../fixtures/line_oracle_fbc_lobatto.ladruno", import.meta.url)));
    try {
      const stage = r.index.stages[0]!;
      const join = joinLadrunoElements(loaded.decls, { state: "paired", by: "model_hash" }, r.index.provenance);
      const stepByComponent: Partial<Record<LineStationComponent, LadrunoLineStationStep>> = {
        N: r.readLineStationStep(stage.id, "N", 1),
        Vy: r.readLineStationStep(stage.id, "Vy", 1),
        Mz: r.readLineStationStep(stage.id, "Mz", 1),
      };
      const res = reconstructSpanDiagramsForStep({
        component: "Mz",
        stepByComponent,
        elementLengths: new Map([[2, 3.0]]),
        join,
        decls: loaded.decls,
        definitions: loaded.definitions,
        sequence: loaded.sequence,
        ndm,
        resultsStage: stage,
        time: stage.time[1]!,
      });
      const e2 = res.get(2)!;
      assert.equal(e2.mode, "exact");
      if (e2.mode === "exact") {
        const xiLob1 = -1.0 / Math.sqrt(5.0);
        const xiLob2 = +1.0 / Math.sqrt(5.0);
        assert.ok(close(interpAtXi(e2.xi, e2.values, -1.0), -24.0, 1e-9));
        assert.ok(close(interpAtXi(e2.xi, e2.values, xiLob1), -15.566563145999492, 1e-6));
        assert.ok(close(interpAtXi(e2.xi, e2.values, xiLob2), -4.833436854000505, 1e-6));
        assert.ok(close(interpAtXi(e2.xi, e2.values, +1.0), 0.0, 1e-9));
      }
    } finally {
      r.close();
    }
  }

  // 2b. line_oracle_fbc_legendre (forceBeamColumn ops_tag 2 with Gauss-Legendre interior stations [-0.7746, 0, +0.7746]):
  // Stations do not include element ends xi = +/-1, so it must fall back cleanly to mode: "interpolated"
  {
    const model = await openModel(fileURLToPath(new URL("../../fixtures/line_oracle_fbc_legendre.h5", import.meta.url)));
    const ndm = model.meta["ndm"] as 2 | 3;
    const blobs = new BlobStore();
    const loaded = loadModel(model, blobs);
    const r = openLadruno(H5, fileURLToPath(new URL("../../fixtures/line_oracle_fbc_legendre.ladruno", import.meta.url)));
    try {
      const stage = r.index.stages[0]!;
      const join = joinLadrunoElements(loaded.decls, { state: "paired", by: "model_hash" }, r.index.provenance);
      const stepByComponent: Partial<Record<LineStationComponent, LadrunoLineStationStep>> = {
        Vy: r.readLineStationStep(stage.id, "Vy", 0),
        Mz: r.readLineStationStep(stage.id, "Mz", 0),
      };
      const res = reconstructSpanDiagramsForStep({
        component: "Mz",
        stepByComponent,
        elementLengths: new Map([[2, 3.0]]),
        join,
        decls: loaded.decls,
        definitions: loaded.definitions,
        sequence: loaded.sequence,
        ndm,
        resultsStage: stage,
        time: stage.time[0]!,
      });
      const e2 = res.get(2)!;
      assert.equal(e2.mode, "interpolated");
      if (e2.mode === "interpolated") {
        assert.match(e2.reason, /do not include element ends/);
        assert.equal(e2.xi.length, 3);
      }
    } finally {
      r.close();
    }
  }

  // 3. line_oracle_basic_only (ops_tag 2, 3): basicForce records 1-station N/T (at xi = 0) and 2-station Mz/My without shear Vy/Vz;
  // both fall back cleanly to "interpolated" with explicit diagnostic reasons
  {
    const model = await openModel(fileURLToPath(new URL("../../fixtures/line_oracle_basic_only.h5", import.meta.url)));
    const ndm = model.meta["ndm"] as 2 | 3;
    const blobs = new BlobStore();
    const loaded = loadModel(model, blobs);
    const r = openLadruno(H5, fileURLToPath(new URL("../../fixtures/line_oracle_basic_only.ladruno", import.meta.url)));
    try {
      const stage = r.index.stages[0]!;
      const join = joinLadrunoElements(loaded.decls, { state: "paired", by: "model_hash" }, r.index.provenance);
      const avail = r.availableLineStations(stage.id);
      const stepByComponent: Partial<Record<LineStationComponent, LadrunoLineStationStep>> = {};
      for (const c of avail) {
        stepByComponent[c] = r.readLineStationStep(stage.id, c, 0);
      }
      const resN = reconstructSpanDiagramsForStep({
        component: "N",
        stepByComponent,
        elementLengths: new Map([[2, 2.0], [3, 2.0]]),
        join,
        decls: loaded.decls,
        definitions: loaded.definitions,
        sequence: loaded.sequence,
        ndm,
        resultsStage: stage,
        time: stage.time[0]!,
      });
      const n2 = resN.get(2)!;
      assert.equal(n2.mode, "interpolated");
      if (n2.mode === "interpolated") {
        assert.match(n2.reason, /has 1 recorded station\(s\)/);
      }

      const resMz = reconstructSpanDiagramsForStep({
        component: "Mz",
        stepByComponent,
        elementLengths: new Map([[2, 2.0], [3, 2.0]]),
        join,
        decls: loaded.decls,
        definitions: loaded.definitions,
        sequence: loaded.sequence,
        ndm,
        resultsStage: stage,
        time: stage.time[0]!,
      });
      const mz2 = resMz.get(2)!;
      assert.equal(mz2.mode, "interpolated");
      if (mz2.mode === "interpolated") {
        assert.match(mz2.reason, /requires companion shear Vy end stations/);
      }
    } finally {
      r.close();
    }
  }
});

// -- 5. Precondition & Closure Fallback Edge Cases ----------------------------

function makeSyntheticBeamSetup(opts?: {
  transfType?: string;
  integrator?: string;
  analysisType?: string;
  loadRow?: string[];
  programLoadConst?: boolean;
  stageSetTime?: number;
  sequenceRemovePattern?: boolean;
}) {
  const transfDecl: Decl = {
    path: "opensees/geomTransf/tr1",
    kind: "geomTransf",
    type: opts?.transfType ?? "Linear",
    name: "tr1",
    nameSource: "/opensees/names",
    h5: "/opensees/transforms/Linear_1",
    tag: 1,
    params: [],
    refs: {},
    links: {},
    problems: [],
    fields: [],
  };
  const elemDecl: Decl = {
    path: "mesh/element/1",
    kind: "element",
    type: "elasticBeamColumn",
    name: "1",
    nameSource: "/elements/line2",
    h5: "/elements/line2",
    tag: null,
    params: [],
    refs: { transfTag: "opensees/geomTransf/tr1" },
    links: {},
    problems: [],
    fields: [],
    element: {
      femId: 1,
      cell: { alias: "line2", block: 0, row: 0 },
      inlineNodes: [],
      groups: [],
      metas: [{ type: "elasticBeamColumn", tag: 1, h5: "/opensees/element_meta/elasticBeamColumn", row: 0, args: [1, 1] }],
    },
  };
  const decls: Record<DeclPath, Decl> = {
    "opensees/geomTransf/tr1": transfDecl,
    "mesh/element/1": elemDecl,
  };

  const loadRow = opts?.loadRow ?? ["-ele", "1", "-type", "-beamUniform", "-10", "0"];
  const definitions: Definitions = {
    decls: null,
    analysis: {
      path: "/opensees/analysis",
      attrs: {
        analysis: opts?.analysisType ?? "Static",
        integrator: opts?.integrator ?? "LoadControl",
      },
    },
    timeSeries: [
      {
        path: "/opensees/time_series/Linear_1",
        groupName: "Linear_1",
        type: "Linear",
        tag: 1,
        params: [1.0],
        attrs: {},
        tables: {},
      },
    ],
    patterns: [
      {
        path: "/opensees/patterns/Plain_1",
        groupName: "Plain_1",
        type: "Plain",
        tag: 1,
        params: [1],
        attrs: { series_ref: "/opensees/time_series/Linear_1" },
        tables: {
          element_loads: {
            path: "/opensees/patterns/Plain_1/element_loads",
            columns: ["[0]", "[1]", "[2]", "[3]", "[4]", "[5]"],
            rows: [loadRow],
          },
        },
      },
    ],
    recorders: [],
    bcs: { fix: null, mass: null },
    stages:
      opts?.stageSetTime !== undefined
        ? [
            {
              index: 0,
              path: "/opensees/stages/stage_000",
              name: "s0",
              attrs: { name: "s0", set_time: opts.stageSetTime },
              analysis: null,
              patterns: [],
              recorders: [],
              bcs: { fix: null, mass: null },
              other: [],
            },
          ]
        : [],
    embeddedRebar: null,
    program: opts?.programLoadConst ? { loadConst: true } : null,
  };

  const sequence: SequenceZone = opts?.sequenceRemovePattern
    ? {
        status: "ready",
        version: "1.0.0",
        attrs: {},
        stages: { path: "/sequence/stages", columns: ["order", "name"], rows: [[0, "s0"]] },
        changes: {
          path: "/sequence/changes",
          columns: ["stage", "op", "what", "name", "count", "source"],
          rows: [[0, "remove", "pattern", "Plain_1", 1, ""]],
        },
      }
    : { status: "absent" };

  const join = joinLadrunoElements(decls, { state: "paired", by: "model_hash" }, null);

  return { decls, elemDecl, definitions, sequence, join };
}

function makeStep(component: LineStationComponent, xi: number[], values: number[]): LadrunoLineStationStep {
  return {
    component,
    time: 1.0,
    values: new Float64Array(values),
    elementIndex: new Float64Array(xi.map(() => 1)),
    stationNaturalCoord: new Float64Array(xi),
    localAxesQuaternion: null,
    localAxesZRec: null,
  };
}

test("reconstructSpanDiagramsForStep falls back to interpolated mode with clear reason on Corotational transform, -beamPoint load, closure failure, non-LoadControl integrator, loadConst, and section_deformation", () => {
  // 1. Happy path with Linear / PDelta transform (L = 6, w = -10 -> Vy = [+30, -30], Mz = [0, 0], Mz(mid) = +45)
  for (const transfType of ["Linear", "PDelta"] as const) {
    const setup = makeSyntheticBeamSetup({ transfType });
    assert.equal(resolveElementGeomTransfType(setup.decls, setup.elemDecl, setup.definitions), transfType);
    const res = reconstructSpanDiagramsForStep({
      component: "Mz",
      stepByComponent: {
        Vy: makeStep("Vy", [-1, 1], [30, -30]),
        Mz: makeStep("Mz", [-1, 1], [0, 0]),
      },
      elementLengths: new Map([[1, 6.0]]),
      join: setup.join,
      decls: setup.decls,
      definitions: setup.definitions,
      sequence: setup.sequence,
      ndm: 2,
      time: 1.0,
    });
    const e1 = res.get(1)!;
    assert.equal(e1.mode, "exact");
    if (e1.mode === "exact") {
      assert.ok(close(interpAtXi(e1.xi, e1.values, 0.0), 45.0, 1e-11));
    }
  }

  // 2. Corotational transform -> refused with reason naming Corotational
  {
    const setup = makeSyntheticBeamSetup({ transfType: "Corotational" });
    const res = reconstructSpanDiagramsForStep({
      component: "Mz",
      stepByComponent: {
        Vy: makeStep("Vy", [-1, 1], [30, -30]),
        Mz: makeStep("Mz", [-1, 1], [0, 0]),
      },
      elementLengths: new Map([[1, 6.0]]),
      join: setup.join,
      decls: setup.decls,
      definitions: setup.definitions,
      sequence: setup.sequence,
      ndm: 2,
      time: 1.0,
    });
    const e1 = res.get(1)!;
    assert.equal(e1.mode, "interpolated");
    if (e1.mode === "interpolated") assert.match(e1.reason, /Corotational/);
  }

  // 3. -beamPoint load on element 1 -> refused for element 1 with reason naming -beamPoint
  {
    const setup = makeSyntheticBeamSetup({
      loadRow: ["-ele", "1", "-type", "-beamPoint", "-20", "0.5"],
    });
    const res = reconstructSpanDiagramsForStep({
      component: "Mz",
      stepByComponent: {
        Vy: makeStep("Vy", [-1, 1], [10, -10]),
        Mz: makeStep("Mz", [-1, 1], [0, 0]),
      },
      elementLengths: new Map([[1, 6.0]]),
      join: setup.join,
      decls: setup.decls,
      definitions: setup.definitions,
      sequence: setup.sequence,
      ndm: 2,
      time: 1.0,
    });
    const e1 = res.get(1)!;
    assert.equal(e1.mode, "interpolated");
    if (e1.mode === "interpolated") assert.match(e1.reason, /-beamPoint/);
  }

  // 4. End-j equilibrium closure failure (e.g. unarchived mid-span point load P = -20 kN where pattern has no loads, so Vy=[+10, -10], Mz=[0, 0]):
  // Integrating w_y = 0 from Vy(0) = +10 gives Vy(L) = +10 != -10 and Mz(L) = +60 != 0!
  {
    const setup = makeSyntheticBeamSetup({
      loadRow: ["-ele", "99", "-type", "-beamUniform", "-10", "0"],
    });
    const res = reconstructSpanDiagramsForStep({
      component: "Mz",
      stepByComponent: {
        Vy: makeStep("Vy", [-1, 1], [10, -10]),
        Mz: makeStep("Mz", [-1, 1], [0, 0]),
      },
      elementLengths: new Map([[1, 6.0]]),
      join: setup.join,
      decls: setup.decls,
      definitions: setup.definitions,
      sequence: setup.sequence,
      ndm: 2,
      time: 1.0,
    });
    const e1 = res.get(1)!;
    assert.equal(e1.mode, "interpolated");
    if (e1.mode === "interpolated") assert.match(e1.reason, /equilibrium closure failed/);
  }

  // 5. Non-LoadControl integrator (DisplacementControl), program loadConst, stage set_time, /sequence remove pattern
  for (const [override, regex] of [
    [{ integrator: "DisplacementControl" }, /DisplacementControl/],
    [{ analysisType: "Transient" }, /Transient/],
    [{ programLoadConst: true }, /loadConst/],
    [{ stageSetTime: 0 }, /set_time/],
    [{ sequenceRemovePattern: true }, /remove pattern/],
  ] as const) {
    const setup = makeSyntheticBeamSetup(override);
    const res = reconstructSpanDiagramsForStep({
      component: "Mz",
      stepByComponent: {
        Vy: makeStep("Vy", [-1, 1], [30, -30]),
        Mz: makeStep("Mz", [-1, 1], [0, 0]),
      },
      elementLengths: new Map([[1, 6.0]]),
      join: setup.join,
      decls: setup.decls,
      definitions: setup.definitions,
      sequence: setup.sequence,
      ndm: 2,
      time: 1.0,
    });
    const e1 = res.get(1)!;
    assert.equal(e1.mode, "interpolated");
    if (e1.mode === "interpolated") assert.match(e1.reason, regex);
  }

  // 6. Section deformation component (`kappaZ`) always returns interpolated
  {
    const setup = makeSyntheticBeamSetup();
    const res = reconstructSpanDiagramsForStep({
      component: "kappaZ",
      stepByComponent: {
        kappaZ: makeStep("kappaZ", [-1, 1], [0.001, -0.001]),
      },
      elementLengths: new Map([[1, 6.0]]),
      join: setup.join,
      decls: setup.decls,
      definitions: setup.definitions,
      sequence: setup.sequence,
      ndm: 2,
      time: 1.0,
    });
    const e1 = res.get(1)!;
    assert.equal(e1.mode, "interpolated");
    if (e1.mode === "interpolated") assert.match(e1.reason, /section_deformation/);
  }
});
