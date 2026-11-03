// The Display card's BC toggle (state/loads.ts `buildSupports3D`, renderer/loads.ts
// `SupportsLayer`): at each support, an arrow into the node per restrained translation
// (the SP bar at its tip) and a half circle with no head per restrained rotation, from
// the model alone (no results). DOFs by ndm: 2-D 1, 2 = x, y and 3 = rotation about z;
// otherwise 1..3 = x, y, z and 4..6 the rotations.

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type * as THREE from "three";
import { withNodes } from "../src/effects.ts";
import { openModel } from "../src/reader/node.ts";
import { SupportsLayer } from "../src/renderer/loads.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import { ARROW, ARROW_STRIDE, BAR, buildSupports3D } from "../src/state/loads.ts";
import { initialState, reduce, type State } from "../src/state/store.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
async function load(name: string) {
  const m = await openModel(join(fixtures, name));
  const blobs = new BlobStore();
  const s: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: withNodes(loadModel(m, blobs), m, blobs) });
  return { s: reduce(s, { type: "setSupports", on: true }), blobs };
}
const def = await load("definitions.h5");
const frame = await load("loads3d.h5");

/** `s` with only these global fix rows (no stages, no patterns, no neutral SPs). */
function onlyFix(s: State, rows: unknown[][]): State {
  return {
    ...s,
    definitions: { ...s.definitions!, patterns: [], stages: [], bcs: { fix: { path: "/opensees/bcs/fix", columns: ["target_kind", "target", "dofs"], rows: rows as never }, mass: null } },
    neutralLoads: null,
  };
}
const axisOf = (buf: Float32Array, i: number) => [0, 1, 2].map((c) => buf[i * ARROW_STRIDE + ARROW.dir + c]).join(",");

test("2-D: one barred arrow into the node per restrained translation, along its axis; the fixture's own fix rows", () => {
  assert.equal(def.s.ndm.ndm, null, "ndm unread in a test load: the builder reads the flat mesh as 2-D");
  const fixRows = [def.s.definitions!.bcs!.fix!, ...def.s.definitions!.stages.map((st) => st.bcs.fix)].flatMap((t) => (t ? t.rows : []));
  const translations = fixRows.reduce((n, r) => n + (r[2] as number[]).slice(0, 2).filter((v) => v !== 0).length, 0);
  const g = buildSupports3D(def.s, def.blobs);
  assert.ok(translations > 0, "definitions.h5 fixes translations");
  assert.equal(g.arrows.length / ARROW_STRIDE, translations, "one arrow per restrained x or y");
  for (let i = 0; i < g.arrows.length / ARROW_STRIDE; i++) {
    assert.ok(["1,0,0", "0,1,0"].includes(axisOf(g.arrows, i)), "along x or y");
    assert.equal(g.arrows[i * ARROW_STRIDE + ARROW.bar], BAR.sp, "the SP bar at the tip: a restraint, not a load");
  }
});

test("a restrained rotation is a half circle about its axis: z in 2-D, x, y or z in 3-D", () => {
  const free = (s: State, blobs: BlobStore) => String((blobs.get(s.mesh!.nodes!.ids) as Float64Array)[0]);
  const flat = buildSupports3D(onlyFix(def.s, [["node", free(def.s, def.blobs), [0, 0, 1]]]), def.blobs);
  assert.deepEqual([flat.arrows.length, flat.arcs.length / ARROW_STRIDE, axisOf(flat.arcs, 0)], [0, 1, "0,0,1"]);
  const node = free(frame.s, frame.blobs);
  const space = buildSupports3D(onlyFix(frame.s, [["node", node, [1, 1, 1, 1, 0, 0]]]), frame.blobs);
  assert.deepEqual(
    [Array.from({ length: space.arrows.length / ARROW_STRIDE }, (_, i) => axisOf(space.arrows, i)), Array.from({ length: space.arcs.length / ARROW_STRIDE }, (_, i) => axisOf(space.arcs, i))],
    [["1,0,0", "0,1,0", "0,0,1"], ["1,0,0"]],
    "3 translations and the rotation about x",
  );
  assert.equal(space.nodes, 1);
});

test("the layer: off by default, on with the BC switch, hidden on the geometry phase; its arcs have no heads", () => {
  const layer = new SupportsLayer();
  const off = { ...def.s, supports: { ...def.s.supports, on: false } };
  assert.equal(layer.sync(off, def.blobs), false, "off by default draws nothing");
  const rot = onlyFix(def.s, [["node", String((def.blobs.get(def.s.mesh!.nodes!.ids) as Float64Array)[0]), [1, 0, 1]]]);
  assert.equal(layer.sync(rot, def.blobs), true);
  assert.equal(layer.root.visible, true);
  assert.equal((layer.cones.geometry as THREE.InstancedBufferGeometry).instanceCount, 1, "one translation arrow");
  assert.equal((layer.arcs.lines.geometry as THREE.InstancedBufferGeometry).instanceCount, 1, "one rotation arc");
  assert.equal(layer.arcs.heads.visible, false, "a restraint has no turning sense: no head");
  const geometry = { ...rot, phase: { ...rot.phase, at: { kind: "geometry" as const } } };
  layer.sync(geometry, def.blobs);
  assert.equal(layer.root.visible, false, "nothing on the geometry phase");
  layer.dispose();
});

test("setSupports refuses a non-boolean switch and a scale at or below 0, and an unchanged value changes nothing", () => {
  assert.equal(initialState.supports.on, false);
  for (const bad of [{ on: "yes" }, { scale: 0 }, { scale: -2 }]) {
    assert.throws(() => reduce(def.s, { type: "setSupports", ...(bad as object) } as never), RangeError, JSON.stringify(bad));
  }
  assert.equal(reduce(def.s, { type: "setSupports", on: true }), def.s);
});
