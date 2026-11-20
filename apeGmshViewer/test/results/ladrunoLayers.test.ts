// Layered-shell buckets from a fork-written .ladruno: reader/ladrunoLayers.ts through reader/ladruno.ts.
//
// ladruno_shells3d.ladruno (fixtures/make_ladruno_fixtures.py --shells, fork build 2b2116b9 of
// nmorabowen/OpenSees#1133): a wall in the x-z plane of 3 x 2 ASDShellQ4 (1000 mm), `-local 1 0 0`,
// LayeredShell sections of ASDConcrete3D (through PlateFromPlaneStress) and PlateRebar layers:
// - elements 1, 2, 4, 5: [25 C, 1 R, 1 R, 50 C, 50 C, 50 C, 50 C, 1 R, 1 R, 25 C] mm (6 concrete layers);
// - elements 3, 6:       [25 C, 1 R, 1 R, 73 C, 73 C, 1 R, 1 R, 25 C] mm (4 concrete layers);
// pushed in plane at the top, with a little out of plane, until it cracks, with
// `-E section.fiber.crackPattern section.fiber.damage`. Rebar does not answer either token, so each
// bucket's COLUMN_MAP is a run per Gauss point (cover, rebar, core, rebar, cover) and the two
// sections land in two header buckets [201:0:0] and [201:0:1]: the layout of the research pushover
// (CRACK_PLOTS_SPEC.md §1.2). The fork build writes MODEL/LOCAL_AXES for the shells (#1133);
// ladruno_shells3d.noframes.ladruno is the same file without it.
//
// The oracle for values is the spec's own reading of the bucket, not the reader's code: DATA[t, e] is
// (Gauss point, responding layer, component) row-major, `col = ((g * L) + k) * C + c`.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openLadruno, type LadrunoFile } from "../../src/reader/ladruno.ts";
import { isLayerRun } from "../../src/reader/ladrunoLayers.ts";
import type { LayerGroupInfo, ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const fixture = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const SHELLS = fixture("ladruno_shells3d.ladruno");
const NOFRAMES = fixture("ladruno_shells3d.noframes.ladruno");
const STAGE_PATH = "/MODEL_STAGE[1]";
const CRACK6 = "section.fiber.crackPattern/203-ASDShellQ4[201:0:0]";
const CRACK4 = "section.fiber.crackPattern/203-ASDShellQ4[201:0:1]";
const DAMAGE6 = "section.fiber.damage/203-ASDShellQ4[201:0:0]";
const CRACK_LABELS = [1, 2, 3].flatMap((k) => ["x", "y", "z"].map((a) => `C${k}${a}`));
const R = 1 / Math.sqrt(3);

function withFile<T>(path: string, body: (r: LadrunoFile) => T): T {
  const r = openLadruno(H5, path);
  try {
    return body(r);
  } finally {
    r.close();
  }
}

function edited(from: string, edit: (f: h5wasm.File) => void): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-ladruno-layers-")), "edited.ladruno");
  copyFileSync(from, p);
  const f = new h5wasm.File(p, "a");
  try {
    edit(f);
  } finally {
    f.close();
  }
  return p;
}

const stageOf = (r: LadrunoFile) => r.index.stages[0]!;
const groupOf = (r: LadrunoFile, id: string): LayerGroupInfo => {
  const g = stageOf(r).layers?.find((x) => x.id === id);
  assert.ok(g, `no layer group ${id}; have ${(stageOf(r).layers ?? []).map((x) => x.id).join(", ")}`);
  return g;
};

/** DATA[step] of a bucket, raw: (E, width). */
function rawStep(path: string, bucketPath: string, step: number): { values: Float64Array; E: number; width: number } {
  const f = new h5wasm.File(path, "r");
  try {
    const ds = f.get(`${bucketPath}/DATA`) as h5wasm.Dataset;
    const [, E, width] = ds.shape as [number, number, number];
    const v = ds.slice([[step, step + 1], [], []]) as Float64Array | Float32Array;
    return { values: Float64Array.from(v), E, width };
  } finally {
    f.close();
  }
}

test("isLayerRun: a Gauss point owning several blocks, or a repeated block, is a layer run; one block per Gauss point is not", () => {
  assert.equal(isLayerRun([0, 0, 0, 1, 1, 1], [1, 2, 1, 1, 2, 1]), true);
  assert.equal(isLayerRun([0, 1, 2, 3], [4, 4, 4, 4]), true);
  assert.equal(isLayerRun([0, 1, 2, 3], [1, 1, 1, 1]), false);
  assert.equal(isLayerRun([0, 1, 2, 3], null), false);
});

test("a fork-written layered-shell .ladruno indexes each header bucket as a layer group, read as the file names it", () => {
  withFile(SHELLS, (r) => {
    const st = stageOf(r);
    const ids = (st.layers ?? []).map((g) => g.id).sort();
    assert.deepEqual(ids, [CRACK6, CRACK4, DAMAGE6, "section.fiber.damage/203-ASDShellQ4[201:0:1]"].sort());
    // Read here, so neither a Gauss group nor unread.
    assert.deepEqual(st.gauss, []);
    assert.ok(!st.unread.some((u) => u.includes("section.fiber")), st.unread.join(" | "));
    const c6 = groupOf(r, CRACK6);
    assert.deepEqual([c6.nElements, c6.nGauss, c6.nLayers, c6.classTag, c6.refused.length], [4, 4, 6, 203, 0]);
    assert.deepEqual(c6.components, CRACK_LABELS.map((l) => `section.fiber.crackPattern/${l}`));
    assert.deepEqual(c6.naturalCoords, [[-R, -R], [R, -R], [R, R], [-R, R]]);
    const c4 = groupOf(r, CRACK4);
    assert.deepEqual([c4.nElements, c4.nLayers], [2, 4]);
    assert.deepEqual(groupOf(r, DAMAGE6).components.map((c) => c.split("/")[1]), ["d+", "d-"]);
  });
});

test("oracle: readLayerStep is DATA read as (Gauss point, layer, component), col = ((g * L) + k) * C + c, every step and group", () => {
  withFile(SHELLS, (r) => {
    const st = stageOf(r);
    for (const g of st.layers!) {
      const C = g.components.length;
      for (const step of [0, st.nSteps - 1]) {
        const got = r.readLayerStep(st.id, g.id, g.components, null, step);
        const raw = rawStep(SHELLS, g.path, step);
        assert.equal(raw.width, g.nGauss * g.nLayers * C, g.id);
        assert.equal(got.length, raw.E * raw.width, g.id);
        for (let e = 0; e < raw.E; e++)
          for (let gp = 0; gp < g.nGauss; gp++)
            for (let k = 0; k < g.nLayers; k++)
              for (let c = 0; c < C; c++) {
                const at = ((e * g.nGauss + gp) * g.nLayers + k) * C + c;
                assert.equal(got[at], raw.values[e * raw.width + (gp * g.nLayers + k) * C + c], `${g.id} step ${step} e${e} g${gp} k${k} c${c}`);
              }
      }
    }
    // The last step has cracked: the reader returns the file's crack vectors, not zeros.
    const last = r.readLayerStep(st.id, CRACK6, st.layers!.find((g) => g.id === CRACK6)!.components, null, st.nSteps - 1);
    assert.ok(last.some((x) => Math.abs(x) > 0.05), "the fixture's last step holds open cracks");
  });
});

test("readLayerStep takes the layers and components asked, in the order asked; a bad one is an argument error", () => {
  withFile(SHELLS, (r) => {
    const st = stageOf(r);
    const g = groupOf(r, CRACK6);
    const step = st.nSteps - 1;
    const all = r.readLayerStep(st.id, CRACK6, g.components, null, step);
    const comps = [g.components[1]!, g.components[0]!];
    const sub = r.readLayerStep(st.id, CRACK6, comps, [5, 0], step);
    assert.equal(sub.length, g.nElements * g.nGauss * 2 * 2);
    let o = 0;
    for (let e = 0; e < g.nElements; e++)
      for (let gp = 0; gp < g.nGauss; gp++)
        for (const k of [5, 0])
          for (const c of [1, 0]) assert.equal(sub[o++], all[((e * g.nGauss + gp) * g.nLayers + k) * 9 + c]);
    assert.throws(() => r.readLayerStep(st.id, CRACK6, ["section.fiber.crackPattern/C4x"], null, step), /has no component/);
    assert.throws(() => r.readLayerStep(st.id, CRACK6, comps, [6], step), /layers 0\.\.5, not 6/);
    assert.throws(() => r.readLayerStep(st.id, CRACK6, comps, null, st.nSteps), /step/i);
  });
});

test("geometry: each concrete layer's z and thickness, and the section thickness, from the run definition", () => {
  // Layers centred through the thickness, the first on the -e3 side; rebar layers are counted in the
  // thickness and skipped among the responding layers.
  const centres = (t: number[]) => {
    const total = t.reduce((a, x) => a + x, 0);
    let at = -total / 2;
    return t.map((x) => ((at += x), at - x / 2));
  };
  const six = [25, 1, 1, 50, 50, 50, 50, 1, 1, 25];
  const four = [25, 1, 1, 73, 73, 1, 1, 25];
  const concrete6 = [0, 3, 4, 5, 6, 9];
  const concrete4 = [0, 3, 4, 7];
  withFile(SHELLS, (r) => {
    for (const [id, layers, idx] of [
      [CRACK6, six, concrete6],
      [CRACK4, four, concrete4],
    ] as const) {
      const g = groupOf(r, id);
      assert.equal(g.geometry.state, "present", JSON.stringify(g.geometry));
      if (g.geometry.state !== "present") continue;
      const z = centres([...layers]);
      for (let e = 0; e < g.nElements; e++) {
        assert.ok(Math.abs(g.geometry.thickness[e]! - layers.reduce((a, x) => a + x, 0)) < 1e-9);
        idx.forEach((sl, k) => {
          assert.ok(Math.abs(g.geometry.state === "present" ? g.geometry.z[e * g.nLayers + k]! - z[sl]! : 1) < 1e-9, `${id} e${e} k${k}`);
          assert.ok(Math.abs(g.geometry.state === "present" ? g.geometry.t[e * g.nLayers + k]! - layers[sl]! : 1) < 1e-9);
        });
      }
    }
  });
});

test("frames: #1133's MODEL/LOCAL_AXES gives every shell of a layer group its material frame (e1 = x, |e2_z| > 0.99); an older file gives none", () => {
  withFile(SHELLS, (r) => {
    const st = stageOf(r);
    const g = groupOf(r, CRACK6);
    const ids = r.layerElementIds(st.id, g.id);
    assert.equal(ids.mapping, "fem_eid");
    const la = r.readLocalAxes(st.id, { elementIds: ids.mapping === "fem_eid" ? ids.ids : [] });
    assert.deepEqual(Array.from(la.hasRecorded), [1, 1, 1, 1]);
    for (let e = 0; e < g.nElements; e++) {
      // FRAME takes global to element-local: the rotation matrix's rows are e1, e2, e3.
      const [w, x, y, z] = la.quaternions.subarray(4 * e, 4 * e + 4) as unknown as number[];
      const e1 = [1 - 2 * (y! * y! + z! * z!), 2 * (x! * y! + w! * z!), 2 * (x! * z! - w! * y!)];
      const e2 = [2 * (x! * y! - w! * z!), 1 - 2 * (x! * x! + z! * z!), 2 * (y! * z! + w! * x!)];
      assert.ok(Math.abs(e1[0]! - 1) < 1e-9, `e1 = ${e1}`);
      assert.ok(Math.abs(e2[2]!) > 0.99, `e2 = ${e2}`);
    }
  });
  withFile(NOFRAMES, (r) => {
    const st = stageOf(r);
    const ids = r.layerElementIds(st.id, CRACK6);
    const la = r.readLocalAxes(st.id, { elementIds: ids.mapping === "fem_eid" ? ids.ids : [] });
    assert.deepEqual(Array.from(la.hasRecorded), [0, 0, 0, 0]);
  });
});

test("element rows are FEM ids only under element_index_space = \"fem\"; otherwise unverified, with the reason", () => {
  const p = edited(SHELLS, (f) => {
    const prov = f.get("INFO/PROVENANCE") as unknown as { delete_attribute(n: string): number };
    prov.delete_attribute("element_index_space");
  });
  withFile(p, (r) => {
    const ids = r.layerElementIds(stageOf(r).id, CRACK6);
    assert.equal(ids.mapping, "unverified");
    assert.match(ids.mapping === "unverified" ? ids.reason : "", /element_index_space/);
  });
});

test("mutation: NUM_COLUMNS that disagrees with the column map refuses the bucket by name, never read", () => {
  const p = edited(SHELLS, (f) => {
    const b = f.get(`${STAGE_PATH}/RESULTS/ON_ELEMENTS/${CRACK6}`) as unknown as { delete_attribute(n: string): number; create_attribute(n: string, v: number): void };
    b.delete_attribute("NUM_COLUMNS");
    b.create_attribute("NUM_COLUMNS", 215);
  });
  withFile(p, (r) => {
    const g = groupOf(r, CRACK6);
    assert.match(g.refused[0]?.reason ?? "", /NUM_COLUMNS = 215.*sums MULTIPLICITY \* NUM_COMP to 216/);
    assert.throws(() => r.readLayerStep(stageOf(r).id, CRACK6, ["section.fiber.crackPattern/C1x"], null, 0), /refused/);
    // The other buckets still read.
    assert.equal(groupOf(r, CRACK4).refused.length, 0);
  });
});

test("mutation: a section that does not cover the run leaves the group's geometry absent with the reason; its values still read", () => {
  const p = edited(SHELLS, (f) => {
    // Element 1 moves to the 8-layer section: its rows in the 10-layer one name a missing element 99,
    // and the 8-layer one's first four rows become element 1's Gauss points 0..3. Its 10-layer run no longer fits.
    const sa = f.get(`${STAGE_PATH}/MODEL/SECTION_ASSIGNMENTS`) as h5wasm.Group;
    const asg = (n: string) => sa.get(`${n}/ASSIGNMENT`) as h5wasm.Dataset;
    const layers = (n: string) => ((sa.get(`${n}/FIBER_DATA`) as h5wasm.Dataset).shape ?? [])[0];
    const eight = sa.keys().find((n) => layers(n) === 8)!;
    const ten = sa.keys().find((n) => layers(n) === 10)!;
    const rows = Int32Array.from(asg(ten).value as Int32Array);
    for (let i = 0; i < rows.length; i += 2) if (rows[i] === 1) rows[i] = 99;
    asg(ten).write_slice([[0, rows.length / 2]], rows);
    asg(eight).write_slice([[0, 4]], new Int32Array([1, 0, 1, 1, 1, 2, 1, 3]));
  });
  withFile(p, (r) => {
    const g = groupOf(r, CRACK6);
    assert.equal(g.geometry.state, "absent");
    assert.match(g.geometry.state === "absent" ? g.geometry.reason : "", /has 8 layers and the column map's run covers 10 \(element 1\)/);
    assert.equal(r.readLayerStep(stageOf(r).id, CRACK6, g.components, null, 0).length, g.nElements * g.nGauss * g.nLayers * 9);
  });
});
