// Reader + chain on the committed fixture. The expected values are closed
// forms of the generator's parameters (examples/shoebuckle_arch.py, named
// in fixtures/README.md), not values copied from this app's own output.

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import * as h5wasm from "h5wasm/node";
import { resolveChain, walk, type ElementRef } from "../src/chain/resolve.ts";
import { buildMesh, colourGroups, IdLookup } from "../src/mesh/build.ts";
import { openModel } from "../src/reader/node.ts";

const FIXTURE = fileURLToPath(new URL("../fixtures/shoebuckle.h5", import.meta.url));

// examples/shoebuckle_arch.py parameters.
const SPAN = 5.9, H_COL = 2.2, RISE = 2.2, ELEM = 0.1;
const HT = 0.16, BF = 0.15, TF = 0.012, TW = 0.008;
const R = (SPAN ** 2 + 4 * RISE ** 2) / (8 * RISE);
const SWEEP = Math.atan2(R - RISE, -SPAN / 2) - Math.atan2(R - RISE, SPAN / 2);
const N_COL = Math.round(H_COL / ELEM) + 1; // nodes per column
const N_ARCH = Math.round((R * SWEEP) / ELEM) + 1; // nodes on the arch

const model = await openModel(FIXTURE);

test("closed-form counts: nodes and elements", () => {
  assert.equal(N_COL, 23);
  assert.equal(N_ARCH, 80);
  // the two springing nodes are shared by a column and the arch
  assert.equal(model.nodeIds.length, 2 * N_COL + N_ARCH - 2);
  const line2 = model.blocks.find((b) => b.alias === "line2");
  assert.ok(line2);
  assert.equal(line2.code, 1);
  assert.equal(line2.ids.length, 2 * (N_COL - 1) + (N_ARCH - 1));
});

test("a file written by main's current writer opens with no neutral and no opensees banner", () => {
  assert.equal(model.neutralVersion, "2.35.0");
  assert.equal(model.opensees?.version, "2.22.0");
  assert.deepEqual(model.warnings, []);
});

test("neutral 2.34: /meta/ndm is the declared spatial dimension (ops.model(ndm=2)), read as a plain attribute", () => {
  // Before 2.34 the writer stamped the highest element dimension (1 for this
  // line-only frame); the app never branched on it (ADR 0113 D7), so the
  // value comes through as the example declares it.
  assert.equal(model.meta["ndm"], 2);
  assert.equal(model.meta["ndf"], 3);
});

test("neutral 2.35: the fixture carries the `source` column on /loads/nodal (#1338), which the reader never opens", () => {
  // The app does not read /loads, so the column is ignored safely: the
  // warnings above are empty on this very file. This checks the file really
  // is a 2.35 one, so the ignore is proven on the column and not on its absence.
  const f = new h5wasm.File(FIXTURE, "r");
  try {
    const ds = f.get("loads/nodal/Pressure") as h5wasm.Dataset;
    const members = ds.dtype as unknown as [string, unknown][];
    const payload = members.find(([name]) => name === "payload")?.[1] as [string, unknown][] | undefined;
    assert.ok(payload, "the nodal load rows have a payload member");
    assert.deepEqual(payload.map(([name]) => name), ["node_id", "force_xyz", "moment_xyz", "name", "basis", "source"]);
  } finally {
    f.close();
  }
});

test("bounding box: crown at H_COL + RISE, span SPAN", () => {
  let xmax = -Infinity, ymax = -Infinity;
  for (let i = 0; i < model.nodeCoords.length; i += 3) {
    xmax = Math.max(xmax, model.nodeCoords[i]!);
    ymax = Math.max(ymax, model.nodeCoords[i + 1]!);
  }
  assert.ok(Math.abs(xmax - SPAN) < 1e-9);
  assert.ok(Math.abs(ymax - (H_COL + RISE)) < 1e-3);
});

test("physical groups and the most-specific colour group", () => {
  const count = (n: string) => model.physicalGroups.find((g) => g.name === n)?.elementIds.length;
  assert.equal(count("LeftColumn"), N_COL - 1);
  assert.equal(count("RightColumn"), N_COL - 1);
  assert.equal(count("Arch"), N_ARCH - 1);
  assert.equal(count("Frame"), 2 * (N_COL - 1) + (N_ARCH - 1));
  assert.equal(count("Base"), 0); // dim-0 group: nodes only
  const { byElement, order } = colourGroups(model);
  const arch = model.physicalGroups.find((g) => g.name === "Arch")!;
  for (const e of arch.elementIds) assert.equal(order[byElement.get(e)!], "Arch");
  const mesh = buildMesh(model);
  assert.equal(mesh.lineRefs.length, 2 * (N_COL - 1) + (N_ARCH - 1));
  assert.equal(mesh.triRefs.length, 0);
  assert.deepEqual(
    mesh.legend.map((l) => [l.name, l.elements]),
    [["Arch", N_ARCH - 1], ["LeftColumn", N_COL - 1], ["RightColumn", N_COL - 1]],
  );
});

/** The Map that #2359 replaced, built as the old code built it: the oracle IdLookup must answer like. */
function referenceMap(keys: ArrayLike<number>, repeated: "first" | "last"): Map<number, number> {
  const m = new Map<number, number>();
  for (let i = 0; i < keys.length; i++) if (repeated === "last" || !m.has(keys[i]!)) m.set(keys[i]!, 10 * i + 1);
  return m;
}

test("#2359: IdLookup answers as the Map it replaced: get, has, size and first-listed iteration, NaN and -0 included", () => {
  const lists = [
    [],
    [3, 5, 9],
    [9, 3, 5, 3, 9, 1],
    [NaN, 2, -0, 0, NaN, 2, -1e300, 4e15],
    [7, 7, 7],
    Array.from(model.nodeIds).reverse(),
    Array.from({ length: 300 }, (_, i) => (i * 37) % 101),
  ];
  for (const keys of lists) {
    for (const repeated of ["first", "last"] as const) {
      const ref = referenceMap(keys, repeated);
      const got = IdLookup.of(keys, (i) => 10 * i + 1, repeated);
      const what = `${repeated} of [${keys.slice(0, 8).join(", ")}]`;
      assert.equal(got.size, ref.size, what);
      assert.deepEqual([...got.entries()], [...ref.entries()], `${what}: entries in first-listed order`);
      assert.deepEqual([...got.keys()], [...ref.keys()], what);
      assert.deepEqual([...got.values()], [...ref.values()], what);
      for (const k of [...keys, NaN, -0, 0, 4, 1e9, -1]) {
        assert.equal(got.get(k), ref.get(k), `${what}: get(${k})`);
        assert.equal(got.has(k), ref.has(k), `${what}: has(${k})`);
      }
    }
  }
  const ro = IdLookup.of([1], (i) => i, "first");
  assert.throws(() => ro.set(1, 2), TypeError);
  assert.throws(() => ro.delete(1), TypeError);
  assert.throws(() => ro.clear(), TypeError);
});

test("#2359: colourGroups on the fixture equals the Map it replaced, entry for entry and in order", () => {
  const { byElement, order } = colourGroups(model);
  const groups = model.physicalGroups.filter((g) => g.elementIds.length > 0).sort((a, b) => a.elementIds.length - b.elementIds.length || a.name.localeCompare(b.name));
  assert.deepEqual(order, groups.map((g) => g.name));
  const ref = new Map<number, number>();
  groups.forEach((g, gi) => g.elementIds.forEach((e) => ref.has(e) || ref.set(e, gi)));
  assert.deepEqual([...byElement.entries()], [...ref.entries()]);
});

function archElementRef(): ElementRef {
  const blockIndex = model.blocks.findIndex((b) => b.alias === "line2");
  const arch = model.physicalGroups.find((g) => g.name === "Arch")!;
  const row = Array.from(model.blocks[blockIndex]!.ids).indexOf(arch.elementIds[10]!);
  return { kind: "fem", blockIndex, row };
}

test("beam definition chain: element -> geomTransf -> beamIntegration -> section -> material", () => {
  const chain = resolveChain(model, archElementRef());
  assert.deepEqual(chain.problems, []);
  const flat = walk(chain.root).map(({ node, depth }) => [depth, node.role, node.type]);
  assert.deepEqual(flat, [
    [0, "element", "dispBeamColumn"],
    [1, "geomTransf", "Linear"],
    [1, "beamIntegration", "Legendre"],
    [2, "section", "Fiber"],
    [3, "uniaxialMaterial", "ASDSteel1D"],
  ]);
});

test("each link records its source and whether it was interpreted", () => {
  const chain = resolveChain(model, archElementRef());
  const [transf, integ] = chain.root.children;
  assert.equal(transf!.via!.label, "transfTag");
  assert.equal(transf!.via!.interpreted, true);
  assert.match(transf!.via!.source, /^\/opensees\/element_meta\/dispBeamColumn\/args\[\d+\]\[0\]$/);
  assert.equal(integ!.via!.label, "integrationTag");
  assert.equal(integ!.via!.interpreted, true);
  const section = integ!.children[0]!;
  assert.equal(section.via!.interpreted, true);
  assert.equal(section.via!.source, "/opensees/beam_integration/Legendre_1@params[0]");
  const mat = section.children[0]!;
  assert.equal(mat.via!.label, "material_ref");
  assert.equal(mat.via!.interpreted, false);
  assert.equal(mat.via!.value, "/opensees/materials/uniaxial/ASDSteel1D_1 (3 rows)");
});

test("parameters come through verbatim (closed form from the generator)", () => {
  const chain = resolveChain(model, archElementRef());
  const integ = chain.root.children[1]!;
  const val = (n: typeof integ, label: string) => n.fields.find((f) => f.label === label)?.value;
  assert.equal(val(integ, "params[1]"), "2"); // n_ip=2
  const section = integ.children[0]!;
  assert.match(val(section, "patches")!, /^3 rows \(kind, material_ref, ny, nz, coords\)$/);
  const mat = section.children[0]!;
  assert.deepEqual(
    [0, 1, 2, 3, 4].map((i) => val(mat, `params[${i}]`)),
    [String(200e9), String(375e6), String(480e6), "0.2", "-fracture"],
  );
  // patch geometry: top flange (y from HT/2-TF to HT/2, z from -BF/2 to BF/2), web, bottom flange
  const obj = model.opensees!.objects.find((o) => o.path === "/opensees/sections/Fiber_1")!;
  const rows = obj.tables["patches"]!.rows;
  const close = (a: unknown, b: number) => assert.ok(Math.abs(Number(a) - b) < 1e-12, `${a} != ${b}`);
  const coords = rows.map((r) => r[4] as number[]);
  close(coords[0]![0], HT / 2 - TF); close(coords[0]![2], HT / 2); close(coords[0]![3], BF / 2);
  close(coords[1]![1], -TW / 2); close(coords[1]![3], TW / 2);
  close(coords[2]![0], -HT / 2);
  assert.deepEqual(rows.map((r) => [r[2], r[3]]), [[1, 20], [16, 1], [1, 20]]);
});
