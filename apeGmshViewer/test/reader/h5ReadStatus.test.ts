// #3010: every reader opens through reader/h5open.ts, which makes a read HDF5
// refused throw `H5ReadError`. These tests make each dataset a reader touches
// refuse in turn, and check the reader never returns as if nothing happened.
//
// The refusal is injected under the check: this file replaces h5wasm's read
// binding before the first open, so `checkedReads` wraps the faulty one.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { readGeometry } from "../../src/reader/geometry.ts";
import { H5ReadError, openH5, type H5Openable } from "../../src/reader/h5open.ts";
import { LadrunoFile, openLadruno } from "../../src/reader/ladruno.ts";
import { openModel } from "../../src/reader/node.ts";
import { readProvenance } from "../../src/reader/provenance.ts";
import { openResults, type ResultsFile, type ResultsH5 } from "../../src/reader/results.ts";
import { resultsFingerprint, type FpH5 } from "../../src/reader/rf1.ts";
import { readUserFile, type UserH5 } from "../../src/reader/userFile.ts";
import { readWorkspaceFile, type WorkspaceH5 } from "../../src/reader/workspace.ts";

await h5wasm.ready;
const here = (p: string): string => fileURLToPath(new URL(p, import.meta.url));
const fixture = (name: string): string => here(`../../fixtures/${name}`);
const H5 = h5wasm as unknown as ResultsH5;

type Read = (fileId: bigint, path: string, ...rest: unknown[]) => number;
const mod = h5wasm.Module as unknown as { get_dataset_data: Read };
const real = mod.get_dataset_data;
let failing: string | null = null;
let seen: Set<string> | null = null;
mod.get_dataset_data = (fileId, path, ...rest) => {
  seen?.add(path);
  return path === failing ? -1 : real(fileId, path, ...rest);
};

/** Everything a results file's reads can reach: the first and last step of each stage's nodal and Gauss data. */
function readAll(rf: ResultsFile | LadrunoFile): void {
  try {
    for (const s of rf.index.stages) {
      if (s.nodes?.components.length) rf.readNodalStep(s.id, s.nodes.components, s.nSteps - 1);
      for (const g of s.gauss) if (g.components.length) rf.readGaussStep(s.id, g.id, g.components[0]!, s.nSteps - 1);
    }
  } finally {
    rf.close();
  }
}

const list = (dir: string): string[] => readdirSync(dir);

const READERS: Record<string, () => unknown> = {
  "read.ts (model.h5)": () => openModel(fixture("definitions.h5")),
  "read.ts (element zones)": () => openModel(fixture("elem_solid.h5")),
  "geometry.ts": () => readGeometry(H5 as never, fixture("zones.geometry.h5")),
  "provenance.ts": () => readProvenance(H5 as never, fixture("bridge_provenance.h5")),
  "results.ts": () => readAll(openResults(H5, fixture("elem_solid.results.h5"))),
  "ladruno.ts": () => readAll(openLadruno(H5, fixture("ladruno_frame3d.ladruno"))),
  "ladruno.ts, ladrunoElements.ts": () => readAll(openLadruno(H5, fixture("ladruno_quad2d.ladruno"))),
  "ladruno.ts (modes)": () => readAll(openLadruno(H5, fixture("ladruno_modes.ladruno"))),
  "rf1.ts": () => resultsFingerprint(H5 as unknown as FpH5, fixture("ladruno_frame3d.ladruno"), list),
  "userFile.ts": () => readUserFile(H5 as unknown as UserH5, here("./fixtures/entities/rows.user.h5"), null),
  "workspace.ts": () => readWorkspaceFile(H5 as unknown as WorkspaceH5, here("../fixtures/vu9a/frame.workspace.h5")),
};

async function outcome(run: () => unknown): Promise<string> {
  try {
    return JSON.stringify(await run(), (_k, v: unknown) => (typeof v === "bigint" ? String(v) : ArrayBuffer.isView(v) ? "<array>" : v)) ?? "";
  } catch (err) {
    return `threw: ${err instanceof Error ? err.message : String(err)}`;
  }
}

for (const [name, run] of Object.entries(READERS)) {
  test(`${name}: a refused read of any dataset it touches is refused by name`, async () => {
    seen = new Set();
    failing = null;
    await run();
    const paths = [...seen];
    seen = null;
    assert.ok(paths.length > 0, "the reader read no dataset, so this proves nothing");
    const silent: string[] = [];
    for (const p of paths) {
      failing = p;
      const out = await outcome(run);
      failing = null;
      if (!out.includes(p) || !out.includes("HDF5 refused to read")) silent.push(`${p} -> ${out.slice(0, 90)}`);
    }
    assert.deepEqual(silent, []);
  });
}

test("every way h5wasm reads a dataset goes through the check", () => {
  const f = openH5(h5wasm as unknown as H5Openable<h5wasm.File>, fixture("elem_solid.h5"));
  try {
    const ds = f.get("nodes/ids") as h5wasm.Dataset;
    failing = ds.path;
    assert.throws(() => ds.value, H5ReadError);
    assert.throws(() => ds.json_value, H5ReadError);
    assert.throws(() => ds.slice([[0, 1]]), H5ReadError);
    assert.throws(() => ds.to_array(), H5ReadError);
    failing = null;
    assert.ok((ds.value as ArrayLike<unknown>).length > 0, "the same dataset still reads once HDF5 accepts it");
  } finally {
    failing = null;
    f.close();
  }
});

test("a refused read of variable-length strings throws and leaves the module working", () => {
  const dir = mkdtempSync(join(tmpdir(), "h5-read-status-"));
  const path = join(dir, "s.h5");
  const w = new h5wasm.File(path, "w");
  w.create_dataset({ name: "names", data: ["one", "two", "three"] });
  w.close();
  const f = openH5(h5wasm as unknown as H5Openable<h5wasm.File>, path);
  try {
    const ds = f.get("names") as h5wasm.Dataset;
    failing = ds.path;
    assert.throws(() => ds.value, H5ReadError);
    assert.throws(() => ds.slice([[0, 2]]), H5ReadError);
    failing = null;
    assert.deepEqual(ds.value, ["one", "two", "three"]);
  } finally {
    failing = null;
    f.close();
  }
});

test("a really damaged chunk index is refused, never read as other numbers", () => {
  const src = fixture("ladruno_truss2d.ladruno");
  const copy = join(mkdtempSync(join(tmpdir(), "h5-read-damage-")), "damaged.ladruno");
  const everyStep = (path: string): number[][] => {
    const rf = openLadruno(H5, path);
    try {
      return rf.index.stages.flatMap((s) => Array.from({ length: s.nSteps }, (_, k) => rf.readNodalStep(s.id, s.nodes!.components, k).flatMap((c) => Array.from(c))));
    } finally {
      rf.close();
    }
  };
  const clean = everyStep(src);
  const bytes = readFileSync(src);
  let refused = 0, nodes = 0;
  for (let at = bytes.indexOf("TREE"); at >= 0; at = bytes.indexOf("TREE", at + 4)) {
    if (bytes[at + 4] !== 1) continue; // node type 1 is a chunk index; 0 is a group's, which no dataset read touches
    nodes++;
    copyFileSync(src, copy);
    const b = readFileSync(copy);
    b.fill(0, at, at + 4);
    writeFileSync(copy, b);
    try {
      assert.deepEqual(everyStep(copy), clean, `chunk index at byte ${at}: read as other numbers with no error`);
    } catch (err) {
      if (!(err instanceof H5ReadError)) throw err;
      refused++;
    }
  }
  assert.ok(nodes > 0 && refused > 0, `${refused} of ${nodes} damaged chunk indexes were refused`);
});
