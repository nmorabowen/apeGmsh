// The viewer's supported scale (#2362 ruling, slice #2373): every open route refuses a
// stage past 2^24 nodes or 2^24 elements, and buildMesh a mesh past the buffers' triangle
// or outline-edge count, by name, before state/ or mesh/ code can reach a V8 cap.
//
// The counts are stubbed, never written: `stubH5` is the h5wasm module seen through a
// wrapper that reports a different shape (and, where the reader counts rows it read, a
// different value) for named datasets of a committed fixture. Everything else is the
// real file. The oracles name the right answer without the app:
// - the limits are the ruling's numbers, 2^24 = 16,777,216, and V8's array limit of 2^27
//   entries (measured, ceiling.ts) over 9 floats per triangle and 6 per outline edge;
// - each refusal's message is written out here in full: the file, the count, the limit
//   and the cause;
// - the mesh's triangle and edge counts are the closed form of a box of hexes;
// - exactly at a limit opens, one past it is refused.

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { Effects, type Bridge } from "../../src/effects.ts";
import { ResultsHost } from "../../src/main/results.ts";
import { buildMesh } from "../../src/mesh/build.ts";
import type { ModelFile } from "../../src/model/types.ts";
import {
  assertWithinCeiling,
  CeilingError,
  datasetRows,
  EDGE_LIMIT,
  ELEMENT_LIMIT,
  JS_ARRAY_LIMIT,
  ladrunoStageElements,
  nativeStageElements,
  NODE_LIMIT,
  TRIANGLE_LIMIT,
} from "../../src/reader/ceiling.ts";
import { openLadruno } from "../../src/reader/ladruno.ts";
import { openLadrunoSet } from "../../src/reader/ladrunoParts.ts";
import { readModel, type H5Module } from "../../src/reader/read.ts";
import { openResults, type ResultsH5 } from "../../src/reader/results.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { Store } from "../../src/state/store.ts";

await h5wasm.ready;

const fixture = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const TRUSS = fixture("ladruno_truss2d.ladruno");
const MODES = fixture("ladruno_modes.ladruno");
const QUAD = fixture("ladruno_quad2d.ladruno");
const FRAME = fixture("ladruno_frame3d.ladruno");
const PART0 = fixture("ladruno_parts2d.part-0.ladruno");
const SOLID_RESULTS = fixture("elem_solid.results.h5");
const SOLID_MODEL = fixture("elem_solid.h5");

const N = 2 ** 24; // the ruling's limit, written out
const RULING = "ruling on #2362";
const nodesPast = (where: string, scope: string, count: string) =>
  `${where}: ${scope} has ${count} nodes; the viewer supports at most 16,777,216 (2^24) nodes per stage (JavaScript Map/Set limit, ${RULING})`;
const elementsPast = (where: string, scope: string, count: string) =>
  `${where}: ${scope} has ${count} elements; the viewer supports at most 16,777,216 (2^24) elements per stage (JavaScript Map/Set limit, ${RULING})`;

/** A refusal by the ceiling, with exactly `message`: a CeilingError, never a RangeError. */
function refused(run: () => unknown, message: string): void {
  assert.throws(run, (err: unknown) => {
    assert.ok(err instanceof CeilingError, `a CeilingError, not ${String(err)}`);
    assert.ok(!(err instanceof RangeError), "not a RangeError");
    assert.equal(err.message, message);
    return true;
  });
}

// ---- the stub: real fixtures, stubbed counts ---------------------------------

interface Override {
  /** the shape the dataset reports, from its real one */
  shape?: (real: number[]) => number[];
  /** the rows a read returns (only where the reader counts rows it read) */
  value?: () => unknown;
}
type Overrides = (file: string, path: string) => Override | undefined;

const baseName = (p: string) => p.split(/[\\/]/).pop()!;

/** h5wasm, with `over(<file base name>, <dataset path>)` replacing a dataset's shape or value. */
function stubH5(over: Overrides) {
  const wrap = (x: unknown, file: string): unknown =>
    x instanceof h5wasm.Dataset ? new SDataset(x, file) : x instanceof h5wasm.Group ? new SGroup(x, file) : x;
  class SGroup {
    protected readonly real: h5wasm.Group;
    protected readonly file: string;
    constructor(real: h5wasm.Group, file: string) {
      this.real = real;
      this.file = file;
    }
    get path(): string {
      return this.real.path;
    }
    get attrs(): unknown {
      return this.real.attrs;
    }
    keys(): string[] {
      return this.real.keys();
    }
    get(name: string): unknown {
      return wrap(this.real.get(name), this.file);
    }
  }
  class SDataset {
    private readonly real: h5wasm.Dataset;
    private readonly file: string;
    constructor(real: h5wasm.Dataset, file: string) {
      this.real = real;
      this.file = file;
    }
    private get o(): Override | undefined {
      return over(this.file, this.real.path);
    }
    get path(): string {
      return this.real.path;
    }
    get attrs(): unknown {
      return this.real.attrs;
    }
    get shape(): number[] | null {
      const o = this.o;
      return o?.shape ? o.shape(this.real.shape ?? []) : this.real.shape;
    }
    get metadata(): unknown {
      return this.real.metadata;
    }
    get dtype(): unknown {
      return this.real.dtype;
    }
    get value(): unknown {
      const o = this.o;
      return o?.value ? o.value() : this.real.value;
    }
    get json_value(): unknown {
      return this.real.json_value;
    }
    slice(ranges: Parameters<h5wasm.Dataset["slice"]>[0]): unknown {
      if (this.o) throw new Error(`stub: ${this.real.path} has a stubbed count, and an open reads no slice of it`);
      return this.real.slice(ranges);
    }
  }
  class SFile extends SGroup {
    constructor(filename: string, mode: "r" = "r") {
      super(new h5wasm.File(filename, mode), baseName(filename));
    }
    get file_id(): bigint {
      return (this.real as h5wasm.File).file_id;
    }
    get filename(): string {
      return (this.real as h5wasm.File).filename;
    }
    close(): void {
      (this.real as h5wasm.File).close();
    }
  }
  return { File: SFile, FS: h5wasm.FS, Group: SGroup, Dataset: SDataset };
}
const asResults = (m: ReturnType<typeof stubH5>) => m as unknown as ResultsH5;
const asModel = (m: ReturnType<typeof stubH5>) => m as unknown as H5Module;

/** `n` distinct ascending ids from `first`, as the fork writes them (int32). */
const ids32 = (n: number, first = 1) => {
  const a = new Int32Array(n);
  for (let i = 0; i < n; i++) a[i] = first + i;
  return a;
};

// ---- the module itself -------------------------------------------------------

test("the limits are the ruling's: 2^24 nodes and elements per stage; triangles and edges as V8's 2^27-entry array holds", () => {
  assert.equal(NODE_LIMIT, 16_777_216);
  assert.equal(ELEMENT_LIMIT, 16_777_216);
  assert.equal(JS_ARRAY_LIMIT, 134_217_728);
  // 9 floats per triangle, 6 per outline edge: the largest count whose buffer fits.
  assert.equal(TRIANGLE_LIMIT, 14_913_080);
  assert.ok(9 * TRIANGLE_LIMIT <= JS_ARRAY_LIMIT && 9 * (TRIANGLE_LIMIT + 1) > JS_ARRAY_LIMIT);
  assert.equal(EDGE_LIMIT, 22_369_621);
  assert.ok(6 * EDGE_LIMIT <= JS_ARRAY_LIMIT && 6 * (EDGE_LIMIT + 1) > JS_ARRAY_LIMIT);
});

test("assertWithinCeiling: 0 and the limit pass, limit + 1 is a named CeilingError, a non-count raises", () => {
  assertWithinCeiling("m.h5", "the model", "nodes", 0);
  assertWithinCeiling("m.h5", "the model", "nodes", N);
  refused(() => assertWithinCeiling("m.h5", 'stage "s"', "nodes", 17_000_000), nodesPast("m.h5", 'stage "s"', "17,000,000"));
  refused(() => assertWithinCeiling("m.h5", "the model", "elements", N + 1), elementsPast("m.h5", "the model", "16,777,217"));
  refused(
    () => assertWithinCeiling("m.h5", "the mesh", "triangles", 14_913_081),
    `m.h5: the mesh has 14,913,081 triangles; the viewer supports at most 14,913,080 triangles (JavaScript array limit of 2^27 entries, 9 per triangle, ${RULING})`,
  );
  for (const bad of [Number.NaN, -1, 1.5, Infinity]) {
    assert.throws(() => assertWithinCeiling("m.h5", "the model", "nodes", bad), TypeError, `${bad} is not a count`);
  }
});

test("counts: a dataset's rows from its shape; a .ladruno stage holds its largest token, a native one its larger category", () => {
  assert.equal(datasetRows(null), 0);
  assert.equal(datasetRows([]), 1);
  assert.equal(datasetRows([7, 1]), 7);
  // stress and strain list the same 6 + 4 elements: 10, not 20.
  const quad = (token: string, bucket: string, n: number) => ({ path: `/MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/${token}/${bucket}`, nElements: n });
  assert.equal(ladrunoStageElements([quad("stress", "31-a", 6), quad("stress", "31-b", 4), quad("strain", "31-a", 6), quad("strain", "31-b", 4)]), 10);
  assert.equal(ladrunoStageElements([]), 0);
  // A bucket listed as a Gauss bucket and as a line-station bucket counts once.
  assert.equal(ladrunoStageElements([quad("basicForce", "12-a", 5), quad("basicForce", "12-a", 5), quad("basicForce", "12-b", 2)]), 7);
  assert.equal(nativeStageElements([{ nElements: 3 }, { nElements: 4 }], [{ nElements: 5 }]), 7);
  assert.equal(nativeStageElements([], [{ nElements: 5 }, { nElements: 1 }]), 6);
});

// ---- .ladruno ----------------------------------------------------------------

const TRUSS_DISP = "/MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT";
const nodalRows = (group: string, n: number): Overrides => (_file, path) =>
  path === `${group}/ID` ? { shape: () => [n, 1] } : path === `${group}/DATA` ? { shape: (r) => [r[0]!, n, r[2]!] } : undefined;

test(".ladruno nodes: a stage of 2^24 nodes opens; 2^24 + 1 is refused by name, from the ID shape", () => {
  const at = openLadruno(asResults(stubH5(nodalRows(TRUSS_DISP, N))), TRUSS);
  assert.deepEqual(at.index.stages.map((s) => s.id), ["stage_0"]);
  at.close();
  refused(() => openLadruno(asResults(stubH5(nodalRows(TRUSS_DISP, N + 1))), TRUSS), nodesPast(TRUSS, 'stage "MODEL_STAGE[1]"', "16,777,217"));
});

const MODES_U = "/MODEL_STAGE[1]/RESULTS/ON_NODES/MODES_OF_VIBRATION(U)";
const modeRows = (n: number): Overrides => (_file, path) =>
  path === `${MODES_U}/ID` ? { shape: () => [n, 1] } : path.startsWith(`${MODES_U}/DATA/STEP_0/MODE_`) ? { shape: (r) => [n, r[1]!] } : undefined;

test(".ladruno mode stage: a mode of 2^24 nodes opens; 2^24 + 1 is refused by name, naming the mode stage", () => {
  const at = openLadruno(asResults(stubH5(modeRows(N))), MODES);
  assert.deepEqual(at.index.stages.map((s) => s.kind), ["static", "mode", "mode"]);
  at.close();
  refused(() => openLadruno(asResults(stubH5(modeRows(N + 1))), MODES), nodesPast(MODES, 'stage "mode_1"', "16,777,217"));
});

const STRESS = "/MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/stress/31-FourNodeQuad[201:0:0]";
/** The stress token's one bucket holds `n` elements (strain keeps its 6): the reader counts the ID rows it reads. */
const gaussRows = (n: number): Overrides => {
  let rows: Int32Array | null = null;
  return (_file, path) =>
    path === `${STRESS}/ID` ? { shape: () => [n, 1], value: () => (rows ??= ids32(n)) } : path === `${STRESS}/DATA` ? { shape: (r) => [r[0]!, n, r[2]!] } : undefined;
};

test(".ladruno elements: a token of 2^24 elements opens; 2^24 + 1 is refused by name (stress and strain counted once)", () => {
  const at = openLadruno(asResults(stubH5(gaussRows(N))), QUAD);
  const g = at.index.stages[0]!.gauss;
  assert.deepEqual(g.map((x) => [x.id, x.nElements]), [["stress/31-FourNodeQuad[201:0:0]", N], ["strain/31-FourNodeQuad[201:0:0]", 6]]);
  at.close();
  refused(() => openLadruno(asResults(stubH5(gaussRows(N + 1))), QUAD), elementsPast(QUAD, 'stage "MODEL_STAGE[1]"', "16,777,217"));
});

const SECTION_FORCE = "/MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/section.force/20-ForceBeamColumn3d[0]";
/** The frame's one section.force bucket (its 10 beams) holds `n` elements: counted from the ID shape, never read at open. */
const lineRows = (n: number): Overrides => (_file, path) =>
  path === `${SECTION_FORCE}/ID` ? { shape: () => [n] } : path === `${SECTION_FORCE}/DATA` ? { shape: (r) => [r[0]!, n, r[2]!] } : undefined;

test(".ladruno line stations: a section.force token of 2^24 beams opens; 2^24 + 1 is refused by name, from the ID shape", () => {
  openLadruno(asResults(stubH5(lineRows(N))), FRAME).close();
  refused(() => openLadruno(asResults(stubH5(lineRows(N + 1))), FRAME), elementsPast(FRAME, 'stage "MODEL_STAGE[1]"', "16,777,217"));
});

/** Each part of the 2-part set lists half the nodes, sharing one interface node: `total` in the union. */
const partRows = (total: number): Overrides => {
  const half = Math.ceil((total + 1) / 2);
  const rows: Record<string, Int32Array> = {};
  const idsOf = (file: string) => (rows[file] ??= file.includes("part-0") ? ids32(half, 1) : ids32(total + 1 - half, half));
  return (file, path) => {
    const m = /^\/MODEL_STAGE\[1\]\/RESULTS\/ON_NODES\/(DISPLACEMENT|REACTION_FORCE)\/(ID|DATA)$/.exec(path);
    if (!m) return undefined;
    const n = idsOf(file).length;
    return m[2] === "ID" ? { shape: () => [n, 1], value: () => idsOf(file) } : { shape: (r) => [r[0]!, n, r[2]!] };
  };
};

test(".ladruno part set: a union of 2^24 nodes opens; 2^24 + 1 is refused by name, though each part is under the limit", () => {
  const list = (dir: string) => readdirSync(dir);
  const at = openLadrunoSet(asResults(stubH5(partRows(N))), PART0, list);
  assert.equal(at.index.stages[0]!.nodes!.ids.length, N);
  at.close();
  refused(() => openLadrunoSet(asResults(stubH5(partRows(N + 1))), PART0, list), nodesPast(PART0, 'stage "MODEL_STAGE[1]" (the union of 2 parts)', "16,777,217"));
});

// ---- native results.h5 -------------------------------------------------------

const PART = "/stages/stage_0/partitions/partition_0";
const nativeNodes = (n: number): Overrides => (_file, path) => (path === `${PART}/nodes/_ids` ? { shape: () => [n] } : undefined);
const HEX = `${PART}/elements/gauss_points/hex_G8`;
const nativeElements = (n: number): Overrides => (_file, path) =>
  path === `${HEX}/_element_index` ? { shape: () => [n] } : path === `${HEX}/stress_xx` || path === `${HEX}/stress_yy` ? { shape: (r) => [r[0]!, n, r[2]!] } : undefined;
const solidStage = (() => {
  const rf = openResults(h5wasm as unknown as ResultsH5, SOLID_RESULTS);
  const s = rf.index.stages[0]!.name;
  rf.close();
  return `stage "${s}"`;
})();

test("native results nodes: a stage of 2^24 _ids opens; 2^24 + 1 is refused by name, from the shape", () => {
  openResults(asResults(stubH5(nativeNodes(N))), SOLID_RESULTS).close();
  refused(() => openResults(asResults(stubH5(nativeNodes(N + 1))), SOLID_RESULTS), nodesPast(SOLID_RESULTS, solidStage, "16,777,217"));
});

test("native results elements: a stage of 2^24 gauss elements opens; 2^24 + 1 is refused by name, from the shape", () => {
  const at = openResults(asResults(stubH5(nativeElements(N))), SOLID_RESULTS);
  assert.equal(at.index.stages[0]!.gauss[0]!.nElements, N);
  at.close();
  refused(() => openResults(asResults(stubH5(nativeElements(N + 1))), SOLID_RESULTS), elementsPast(SOLID_RESULTS, solidStage, "16,777,217"));
});

// ---- native model.h5 ---------------------------------------------------------

const modelRows = (dataset: string, n: number): Overrides => (_file, path) => (path === dataset ? { shape: () => [n] } : undefined);

test("native model: 2^24 nodes and 2^24 elements open; 2^24 + 1 of either is refused by name, from the shapes", () => {
  readModel(asModel(stubH5(modelRows("/nodes/ids", N))), SOLID_MODEL, 0);
  readModel(asModel(stubH5(modelRows("/elements/hex8/ids", N))), SOLID_MODEL, 0);
  refused(() => readModel(asModel(stubH5(modelRows("/nodes/ids", N + 1))), SOLID_MODEL, 0), nodesPast(SOLID_MODEL, "the model", "16,777,217"));
  refused(() => readModel(asModel(stubH5(modelRows("/elements/hex8/ids", N + 1))), SOLID_MODEL, 0), elementsPast(SOLID_MODEL, "the model", "16,777,217"));
});

// ---- the mesh buffers --------------------------------------------------------

/** An a x b x c box of unit hexes, gmsh corner order, node and cell ids from 1. */
function boxModel(a: number, b: number, c: number): ModelFile {
  const node = (i: number, j: number, k: number) => 1 + i + (a + 1) * (j + (b + 1) * k);
  const coords: number[] = [];
  for (let k = 0; k <= c; k++) for (let j = 0; j <= b; j++) for (let i = 0; i <= a; i++) coords.push(i, j, k);
  const conn: number[] = [];
  for (let k = 0; k < c; k++) {
    for (let j = 0; j < b; j++) {
      for (let i = 0; i < a; i++) {
        for (const [di, dj, dk] of [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]] as const) conn.push(node(i + di, j + dj, k + dk));
      }
    }
  }
  const nNodes = (a + 1) * (b + 1) * (c + 1);
  return {
    path: "box.h5",
    sizeBytes: 0,
    neutralVersion: "2.35.0",
    meta: {},
    nodeIds: Float64Array.from({ length: nNodes }, (_, i) => 1 + i),
    nodeCoords: Float64Array.from(coords),
    blocks: [{ alias: "hexa8", code: 5, dim: 3, npe: 8, ids: Float64Array.from({ length: a * b * c }, (_, e) => 1 + e), connectivity: Float64Array.from(conn) }],
    physicalGroups: [],
    labels: [],
    opensees: null,
    sequence: { status: "absent" },
    warnings: [],
    readMs: 0,
  };
}

test("buildMesh: a mesh of exactly the triangle (or edge) limit builds; one past it is refused by name before the buffer is filled", () => {
  const [a, b, c] = [3, 2, 2];
  // The box's boundary: F = 2(ab + bc + ca) quads, two triangles each, and V + F - 2 outline edges (Euler).
  const F = 2 * (a * b + b * c + c * a);
  const V = (a + 1) * (b + 1) * (c + 1) - (a - 1) * (b - 1) * (c - 1);
  const [tris, edges] = [2 * F, V + F - 2];
  const model = boxModel(a, b, c);
  const full = buildMesh(model);
  assert.equal(full.triRefs.length, tris);
  assert.equal(full.edgePositions.length, 6 * edges);
  assert.deepEqual(buildMesh(model, { triangles: tris, edges }), full, "at both limits: the same buffers");
  refused(
    () => buildMesh(model, { triangles: tris - 1, edges }),
    `box.h5: the mesh has ${tris} triangles; the viewer supports at most ${tris - 1} triangles (JavaScript array limit of 2^27 entries, 9 per triangle, ${RULING})`,
  );
  refused(
    () => buildMesh(model, { triangles: tris, edges: edges - 1 }),
    `box.h5: the mesh has ${edges} outline edges; the viewer supports at most ${edges - 1} outline edges (JavaScript array limit of 2^27 entries, 6 per edge, ${RULING})`,
  );
});

// ---- before state/: no Map or Set is filled first ----------------------------

/**
 * Map and Set stubbed to throw a RangeError (V8's own class) past a few entries, as V8's
 * do past 2^24, restored in a finally. A reader that filled a Map or a Set by the node or
 * element count before refusing would throw that RangeError instead of the CeilingError.
 */
function underStubbedCollections<T>(run: () => T): T {
  const CAP = 64;
  const [RealMap, RealSet] = [globalThis.Map, globalThis.Set];
  class CappedMap<K, V> extends RealMap<K, V> {
    override set(k: K, v: V): this {
      if (!this.has(k) && this.size >= CAP) throw new RangeError("Map maximum size exceeded (stub)");
      return super.set(k, v);
    }
  }
  class CappedSet<V> extends RealSet<V> {
    override add(v: V): this {
      if (!this.has(v) && this.size >= CAP) throw new RangeError("Set maximum size exceeded (stub)");
      return super.add(v);
    }
  }
  globalThis.Map = CappedMap as MapConstructor;
  globalThis.Set = CappedSet as SetConstructor;
  try {
    assert.throws(() => new Set(Array.from({ length: CAP + 1 }, (_, i) => i)), RangeError, "the stub bites");
    return run();
  } finally {
    globalThis.Map = RealMap;
    globalThis.Set = RealSet;
  }
}

test("with Map and Set stubbed to throw, every over-ceiling open is the named CeilingError, never a RangeError", () => {
  const list = (dir: string) => readdirSync(dir);
  underStubbedCollections(() => {
    refused(() => openLadruno(asResults(stubH5(nodalRows(TRUSS_DISP, N + 1))), TRUSS), nodesPast(TRUSS, 'stage "MODEL_STAGE[1]"', "16,777,217"));
    refused(() => openLadruno(asResults(stubH5(gaussRows(N + 1))), QUAD), elementsPast(QUAD, 'stage "MODEL_STAGE[1]"', "16,777,217"));
    refused(() => openLadruno(asResults(stubH5(lineRows(N + 1))), FRAME), elementsPast(FRAME, 'stage "MODEL_STAGE[1]"', "16,777,217"));
    refused(() => openLadrunoSet(asResults(stubH5(partRows(N + 1))), PART0, list), nodesPast(PART0, 'stage "MODEL_STAGE[1]" (the union of 2 parts)', "16,777,217"));
    refused(() => openResults(asResults(stubH5(nativeNodes(N + 1))), SOLID_RESULTS), nodesPast(SOLID_RESULTS, solidStage, "16,777,217"));
    refused(() => openResults(asResults(stubH5(nativeElements(N + 1))), SOLID_RESULTS), elementsPast(SOLID_RESULTS, solidStage, "16,777,217"));
    refused(() => readModel(asModel(stubH5(modelRows("/nodes/ids", N + 1))), SOLID_MODEL, 0), nodesPast(SOLID_MODEL, "the model", "16,777,217"));
    refused(() => readModel(asModel(stubH5(modelRows("/elements/hex8/ids", N + 1))), SOLID_MODEL, 0), elementsPast(SOLID_MODEL, "the model", "16,777,217"));
  });
});

// ---- the open-failure path ---------------------------------------------------

test("the refusal reaches the user: ResultsHost answers it, the store shows it, and the loaded model stays", async () => {
  const model = readModel(h5wasm as unknown as H5Module, SOLID_MODEL, 0);
  const host = new ResultsHost(asResults(stubH5(nativeNodes(N + 1))), h5wasm.ready);
  const bridge = {
    openModel: async () => ({ ok: true as const, model }),
    openResults: (path: string) => host.open(path),
  } as unknown as Bridge;
  const store = new Store();
  const effects = new Effects(store, new BlobStore(), bridge);
  assert.equal(await effects.open(SOLID_MODEL), true);
  const mesh = store.get().mesh;
  assert.ok(mesh, "the model loaded");

  assert.equal(await effects.openResults(SOLID_RESULTS), false, "the open is refused");
  const s = store.get();
  assert.equal(s.artifacts.results?.status, "failed");
  assert.equal(s.artifacts.results?.error, nodesPast(SOLID_RESULTS, solidStage, "16,777,217"), "main's answer carries the named message to the store");
  assert.equal(s.results, null);
  assert.equal(s.artifacts.model?.status, "ready", "the model is intact");
  assert.equal(s.mesh, mesh, "and so is its mesh");
  host.close();
});
