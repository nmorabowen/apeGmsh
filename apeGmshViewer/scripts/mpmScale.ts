// Synthetic `.ladruno` particle files for the MEASUREMENTS row of ADR 0134 D5
// (chain M, M2-S4 #2983): 10^5 and 10^6 particles, read by the points layer.
//
//   node scripts/mpmScale.ts <particles> <outputs> <out.ladruno>
//
// The file is the fork's MPM fixture (tests/fixtures/mpm/particles_v1m2.ladruno)
// grown, so it keeps every attribute the fork writes (INFO, the stage's KIND,
// MODEL/PARTICLES and its GRID and BODIES, ON_PARTICLES and each field group's
// attributes), with the datasets rewritten at size:
// - a 2-D block of `particles` particles, rows of ceil(sqrt(particles)), one in four
//   in body 1 ("block", whose `STATE_eqPlasticStrain` is NaN, as the fixture's);
// - `outputs` outputs, every particle in every one, falling under gravity, so a
//   scrub moves every point and the colour by POSITION/y changes each output;
// - POSITION (float64, as the contract requires) and STATE_eqPlasticStrain (float64),
//   chunked one output per chunk, as the fork writes them.
// The file is checked by reading it back through reader/ladruno.ts before the
// script ends; a file the reader refuses is never left as a measure file.

import { rmSync } from "node:fs";
import * as h5wasm from "h5wasm/node";
import { createH5, openH5, type H5Creatable, type H5Openable } from "../src/reader/h5open.ts";
import { openLadruno } from "../src/reader/ladruno.ts";
import type { ResultsH5 } from "../src/reader/results.ts";

const [nArg, tArg, outArg] = process.argv.slice(2);
const want = Number(nArg);
const T = Number(tArg);
if (!Number.isInteger(want) || want < 4 || !Number.isInteger(T) || T < 1 || !outArg) {
  console.error("node scripts/mpmScale.ts <particles >= 4> <outputs >= 1> <out.ladruno>");
  process.exit(2);
}
const out = outArg;
const FIXTURE = new URL("../../tests/fixtures/mpm/particles_v1m2.ladruno", import.meta.url);
const STAGE = "MODEL_STAGE[1]";

await h5wasm.ready;
const side = Math.ceil(Math.sqrt(want));
const N = want;
const h = 2 / side;

function attrValue(a: h5wasm.Attribute): unknown {
  const v = a.value as unknown;
  const one = Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView)) ? ((v as ArrayLike<unknown>).length === 1 ? (v as ArrayLike<unknown>)[0] : v) : v;
  return typeof one === "string" ? one.replace(/\0+$/, "") : one;
}
function copyAttrs(from: h5wasm.Group | h5wasm.Dataset, to: h5wasm.Group | h5wasm.Dataset): void {
  for (const [k, a] of Object.entries(from.attrs)) {
    const v = attrValue(a);
    if (typeof v === "string") to.create_attribute(k, v);
    else to.create_attribute(k, a.value as never, a.shape, a.dtype as never);
  }
}
/** A group copied whole (attributes and small datasets): GRID and BODIES. */
function copyGroup(from: h5wasm.Group, to: h5wasm.Group): void {
  copyAttrs(from, to);
  for (const k of from.keys()) {
    const item = from.get(k);
    if (item instanceof h5wasm.Group) copyGroup(item, to.create_group(k));
    else if (item instanceof h5wasm.Dataset) {
      const strings = item.metadata.type === 3;
      const value = item.value as unknown;
      const ds = to.create_dataset({
        name: k,
        data: (strings ? (value as string[]).map((x) => x.replace(/\0+$/, "")) : value) as never,
        shape: item.shape!,
        ...(strings ? {} : { dtype: item.dtype as never }),
      });
      copyAttrs(item, ds);
    }
  }
}

// Through reader/h5open.ts (lesson #2260): exact file ids; the measure file is created exclusively, never over another.
const src = openH5(h5wasm as unknown as H5Openable<h5wasm.File>, FIXTURE.pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const f = createH5(h5wasm as unknown as H5Creatable<h5wasm.File>, out);
const t0 = performance.now();
try {
  copyAttrs(src.get("INFO") as h5wasm.Group, f.create_group("INFO"));
  const stage = f.create_group(STAGE);
  copyAttrs(src.get(STAGE) as h5wasm.Group, stage);
  const srcModel = src.get(`${STAGE}/MODEL/PARTICLES`) as h5wasm.Group;
  const model = stage.create_group("MODEL").create_group("PARTICLES");
  copyAttrs(srcModel, model);
  copyGroup(srcModel.get("GRID") as h5wasm.Group, model.create_group("GRID"));
  copyGroup(srcModel.get("BODIES") as h5wasm.Group, model.create_group("BODIES"));
  // The registry: particle i at row i; one in four in body 1 (material 1), the rest body 0 (material 2).
  const ids = new BigInt64Array(N);
  const body = new Int32Array(N);
  const mat = new Int32Array(N);
  const mass = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    ids[i] = BigInt(i);
    body[i] = i % 4 === 0 ? 1 : 0;
    mat[i] = body[i] === 1 ? 1 : 2;
    mass[i] = h * h * 2;
  }
  model.create_dataset({ name: "ID", data: ids, shape: [N], dtype: "<q" });
  model.create_dataset({ name: "BODY_ID", data: body, shape: [N], dtype: "<i" });
  model.create_dataset({ name: "MATERIAL_TAG", data: mat, shape: [N], dtype: "<i" });
  model.create_dataset({ name: "MASS", data: mass, shape: [N], dtype: "<d" });

  const srcOn = src.get(`${STAGE}/RESULTS/ON_PARTICLES`) as h5wasm.Group;
  const on = stage.create_group("RESULTS").create_group("ON_PARTICLES");
  copyAttrs(srcOn, on);
  const S = N * T;
  on.create_dataset({ name: "STEP", data: Int32Array.from({ length: T }, (_, k) => 10 * k), shape: [T], dtype: "<i" });
  on.create_dataset({ name: "TIME", data: Float64Array.from({ length: T }, (_, k) => 1e-4 * k), shape: [T], dtype: "<d" });
  on.create_dataset({ name: "OFFSETS", data: BigInt64Array.from({ length: T + 1 }, (_, k) => BigInt(k * N)), shape: [T + 1], dtype: "<q" });
  const id = on.create_dataset({ name: "ID", data: new BigInt64Array(0), shape: [0], maxshape: [null], chunks: [N], dtype: "<q" });
  const pos = on.create_group("POSITION");
  copyAttrs(srcOn.get("POSITION") as h5wasm.Group, pos);
  const posData = pos.create_dataset({ name: "DATA", data: new Float64Array(0), shape: [0, 2], maxshape: [null, 2], chunks: [N, 2], dtype: "<d" });
  const eq = on.create_group("STATE_eqPlasticStrain");
  copyAttrs(srcOn.get("STATE_eqPlasticStrain") as h5wasm.Group, eq);
  const eqData = eq.create_dataset({ name: "DATA", data: new Float64Array(0), shape: [0, 1], maxshape: [null, 1], chunks: [N, 1], dtype: "<d" });
  // One output at a time, appended as the fork's writer does (never the whole history in memory).
  const p = new Float64Array(N * 2);
  const e = new Float64Array(N);
  for (let k = 0; k < T; k++) {
    const fall = 0.5 * 9.81 * (1e-4 * k * 20) ** 2;
    for (let i = 0; i < N; i++) {
      const x = (i % side) * h + h / 2;
      const y = Math.floor(i / side) * h + h / 2;
      // Columns fall by their height above the base, so the block shears as it drops.
      p[2 * i] = x;
      p[2 * i + 1] = y - fall * (y / 2);
      e[i] = body[i] === 1 ? NaN : 1e-4 * k * (1 + x);
    }
    const a = k * N;
    id.resize([a + N]);
    id.write_slice([[a, a + N]], ids);
    posData.resize([a + N, 2]);
    posData.write_slice([[a, a + N], [0, 2]], p);
    eqData.resize([a + N, 1]);
    eqData.write_slice([[a, a + N], [0, 1]], e);
  }
  if (S !== N * T) throw new Error("row count");
} finally {
  f.close();
  src.close();
}

// Read it back through the app's reader: a file it refuses is no measure file.
const rf = openLadruno(h5wasm as unknown as ResultsH5, out);
try {
  const p = rf.particles;
  const s = p?.stages[0];
  if (!p || p.refused.length || !s || s.nOutputs !== T || s.offsets[T] !== N * T) {
    throw new Error(`${out}: the reader does not read it back as ${N} particles in ${T} outputs (${JSON.stringify(p?.refused ?? "no particles")})`);
  }
  const o = p.readOutput(s.stageId, T - 1, "STATE_eqPlasticStrain");
  if (o.ids.length !== N || !Number.isNaN(o.values[0]!)) throw new Error(`${out}: output ${T - 1} does not read back as written`);
} catch (err) {
  rf.close();
  rmSync(out, { force: true });
  throw err;
}
rf.close();
console.log(`${out}: ${N} particles (rows of ${side}), ${T} outputs, written and read back in ${Math.round(performance.now() - t0)} ms`);
