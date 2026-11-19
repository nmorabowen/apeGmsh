// A damaged group index is a refusal. h5wasm's `Group.keys()` drops H5Literate2's status, so a
// group whose B-tree does not read used to list short with no error, and a stage lost its
// results. reader/h5open.ts makes that listing throw, for every reader at once.
//
// The damage is real: the 4-byte `TREE` signature of a version-1 B-tree node of type 0 (a group
// index; type 1 indexes chunks) is zeroed in a copy of a fixture.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openH5, type H5Openable } from "../../src/reader/h5open.ts";
import { openLadruno } from "../../src/reader/ladruno.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const fixture = (name: string): string => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const FRAME = fixture("ladruno_frame3d.ladruno");
const TRUSS = fixture("ladruno_truss2d.ladruno");
const H5 = h5wasm as unknown as ResultsH5;
const Openable = h5wasm as unknown as H5Openable<h5wasm.File>;
const dir = mkdtempSync(join(tmpdir(), "apegmsh-h5-list-"));
after(() => rmSync(dir, { recursive: true, force: true }));

const clean = readFileSync(FRAME);

/** The offset of every group-index B-tree node in `bytes`. */
function groupNodes(bytes: Buffer): number[] {
  const out: number[] = [];
  for (let at = bytes.indexOf("TREE"); at >= 0; at = bytes.indexOf("TREE", at + 1)) if (bytes[at + 4] === 0) out.push(at);
  return out;
}

/** A copy of the frame fixture with the group-index node at `offset` damaged. */
function damaged(offset: number): string {
  const bytes = Buffer.from(clean);
  bytes.fill(0, offset, offset + 4);
  const path = join(dir, `damaged-${offset}.ladruno`);
  writeFileSync(path, bytes);
  return path;
}

/** A copy of `from` in the test directory. */
function copy(from: string, name: string): string {
  const to = join(dir, name);
  copyFileSync(from, to);
  return to;
}

/** Every link below `group`, by listing each group. */
function walk(group: h5wasm.Group, out: string[] = []): string[] {
  for (const name of group.keys()) {
    const child = group.get(name);
    out.push(`${group.path}/${name}`);
    if (child instanceof h5wasm.Group) walk(child, out);
  }
  return out;
}

const nodes = groupNodes(clean);
const stagesOf = (path: string): string => {
  const f = openLadruno(H5, path);
  try {
    return JSON.stringify(f.index.stages);
  } finally {
    f.close();
  }
};

test("the fixture has group-index nodes to damage", () => {
  assert.ok(nodes.length > 10, `only ${nodes.length} group-index nodes`);
});

test("a damaged group index never changes what openLadruno reports", () => {
  const expected = stagesOf(FRAME);
  let refused = 0;
  for (const offset of nodes) {
    let stages: string;
    try {
      stages = stagesOf(damaged(offset));
    } catch (err) {
      assert.match((err as Error).message, /HDF5 could not list this group/, `node at ${offset}`);
      refused++;
      continue;
    }
    assert.equal(stages, expected, `node at ${offset}: the index changed with no error`);
  }
  assert.ok(refused > 0, "no damaged node was refused");
});

test("listing every group of a file with any damaged group index throws, naming the group", () => {
  for (const offset of nodes) {
    const f = openH5(Openable, damaged(offset));
    try {
      assert.throws(() => walk(f), /^Error: \/.*: HDF5 could not list this group/, `node at ${offset}`);
    } finally {
      f.close();
    }
  }
});

test("a damaged INFO index leaves the results readable and says provenance is unknown, not absent", () => {
  const offset = nodes.find((at) => {
    const f = openH5(Openable, damaged(at));
    try {
      (f.get("INFO") as h5wasm.Group).keys();
      return false;
    } catch (err) {
      return /^\/INFO: /.test((err as Error).message);
    } finally {
      f.close();
    }
  });
  assert.notEqual(offset, undefined, "no group-index node belongs to /INFO");
  const f = openLadruno(H5, damaged(offset!));
  try {
    assert.equal(JSON.stringify(f.index.stages), stagesOf(FRAME));
    assert.equal(f.index.provenance.state, "absent");
    assert.match((f.index.provenance as { reason: string }).reason, /is unknown: \/INFO: HDF5 could not list this group/);
  } finally {
    f.close();
  }
});

test("a refused listing holds nothing open: the path reads its new contents after a replace in place", () => {
  const path = damaged(nodes[nodes.length - 1]!);
  const f = openH5(Openable, path);
  assert.throws(() => walk(f), /could not list this group/);
  f.close();

  writeFileSync(path, readFileSync(TRUSS));
  const again = openH5(Openable, path);
  const truth = openH5(Openable, copy(TRUSS, "truth.ladruno"));
  try {
    assert.deepEqual(walk(again), walk(truth));
  } finally {
    again.close();
    truth.close();
  }
});

test("sound groups list as before: empty groups, dangling soft links and external links are not damage", () => {
  const path = join(dir, "links.h5");
  const w = new h5wasm.File(path, "w");
  w.create_group("empty");
  w.create_soft_link("/nowhere", "dangling");
  w.create_external_link("missing.h5", "/x", "external");
  w.close();

  const f = openH5(Openable, path);
  try {
    assert.deepEqual(f.keys().sort(), ["dangling", "empty", "external"]);
    assert.deepEqual((f.get("empty") as h5wasm.Group).keys(), []);
    assert.equal(f.get("absent"), null);
  } finally {
    f.close();
  }
});
