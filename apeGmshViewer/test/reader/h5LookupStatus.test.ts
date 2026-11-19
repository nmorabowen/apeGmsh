// A lookup through a damaged group index is a refusal. h5wasm's `Group.get` asks `Module.get_type`,
// whose C++ ignores both HDF5 statuses, so a lookup that could not read the index came back as
// type 0: an existing dataset, and a name that does not exist, both looked like groups.
// reader/h5open.ts makes that lookup throw, for every reader at once.
//
// The damage is real: one signature of a group's index (its B-tree node, symbol-table node or local
// heap) is zeroed in a copy of a fixture.

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openH5, type H5Openable } from "../../src/reader/h5open.ts";

await h5wasm.ready;
const fixture = (name: string): string => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const FRAME = fixture("ladruno_frame3d.ladruno");
const TRUSS = fixture("ladruno_truss2d.ladruno");
const Openable = h5wasm as unknown as H5Openable<h5wasm.File>;
const dir = mkdtempSync(join(tmpdir(), "apegmsh-h5-lookup-"));
after(() => rmSync(dir, { recursive: true, force: true }));

const clean = readFileSync(FRAME);
const REFUSED = /^Error: HDF5 could not look up \/.* \(the index of a group on this path does not read\)$/;

/** Every offset of `signature` in `bytes`; a B-tree node counts only as a group index (type 0). */
function signatures(bytes: Buffer, signature: string): number[] {
  const out: number[] = [];
  for (let at = bytes.indexOf(signature); at >= 0; at = bytes.indexOf(signature, at + 1)) {
    if (signature !== "TREE" || bytes[at + 4] === 0) out.push(at);
  }
  return out;
}

/** A copy of the frame fixture with the signatures at `offsets` zeroed. */
function damaged(offsets: number[], name: string): string {
  const bytes = Buffer.from(clean);
  for (const at of offsets) bytes.fill(0, at, at + 4);
  const path = join(dir, `${name}.ladruno`);
  writeFileSync(path, bytes);
  return path;
}

/** What `get` answers, as a name a comparison can read. */
const kind = (entity: unknown): string => (entity === null ? "null" : (entity as object).constructor.name);

/** The paths to look up in the frame fixture, each with its answer in the undamaged file. */
function truth(): Map<string, string> {
  const out = new Map<string, string>();
  const f = openH5(Openable, FRAME);
  try {
    const walk = (group: h5wasm.Group): void => {
      const at = group.path === "/" ? "" : group.path;
      out.set(`${at}/__absent__`, "null");
      for (const name of group.keys()) {
        const child = group.get(name);
        out.set(`${at}/${name}`, kind(child));
        if (child instanceof h5wasm.Group) walk(child);
      }
    };
    walk(f);
  } finally {
    f.close();
  }
  return out;
}

const expected = truth();

/** Look every path up in `path`: nothing may differ from the clean file; returns how many were refused. */
function sweep(path: string, label: string): number {
  const f = openH5(Openable, path);
  let refused = 0;
  try {
    for (const [p, want] of expected) {
      let got: string;
      try {
        got = kind(f.get(p));
      } catch (err) {
        assert.match(String(err), REFUSED, `${label}: ${p}`);
        assert.ok(String(err).includes(p), `${label}: the refusal names ${p}`);
        refused++;
        continue;
      }
      assert.equal(got, want, `${label}: ${p} changed with no error`);
    }
  } finally {
    f.close();
  }
  return refused;
}

for (const signature of ["TREE", "SNOD", "HEAP"]) {
  test(`a damaged ${signature} never changes what a lookup reports: it answers as the clean file or refuses`, () => {
    const offsets = signatures(clean, signature);
    assert.ok(offsets.length > 0, `the fixture has no ${signature}`);
    let refused = 0;
    for (const at of offsets) refused += sweep(damaged([at], `${signature}-${at}`), `${signature} at ${at}`);
    assert.ok(refused > 0, `no lookup was refused with a ${signature} damaged`);
  });
}

test("with every group index damaged, a dataset is not a group and an absent name is not a group", () => {
  const path = damaged(signatures(clean, "TREE"), "all-trees");
  const dataset = [...expected].find(([, want]) => want === "Dataset")![0];
  const f = openH5(Openable, path);
  try {
    assert.throws(() => f.get(dataset), REFUSED);
    assert.throws(() => f.get("/__absent__"), REFUSED);
  } finally {
    f.close();
  }
});

test("a refused lookup holds nothing open: the path reads its new contents after a replace in place", () => {
  const path = damaged(signatures(clean, "TREE"), "replaced");
  const f = openH5(Openable, path);
  assert.throws(() => f.get("/INFO"), REFUSED);
  f.close();

  writeFileSync(path, readFileSync(TRUSS));
  const again = openH5(Openable, path);
  try {
    assert.equal(kind(again.get("/INFO")), "Group");
  } finally {
    again.close();
  }
});

test("lookups HDF5 reports an error on in a sound file keep their answers", () => {
  const path = join(dir, "links.h5");
  const w = new h5wasm.File(path, "w");
  w.create_group("grp");
  w.create_dataset({ name: "ds", data: [1, 2, 3] });
  w.create_soft_link("/nowhere", "dangling");
  w.create_soft_link("/ds", "soft_ds");
  w.create_soft_link("/loop", "loop");
  w.create_external_link("missing.h5", "/x", "ext_missing");
  w.close();

  // h5wasm's answers, odd ones included: a path that does not resolve past its first failure reads
  // as type 0. They are today's behaviour for a sound file, not damage, so they do not change here.
  const answers: Record<string, string> = {
    "/ds": "Dataset",
    "/grp": "Group",
    "/absent": "null",
    "/absent/x": "null",
    "/ds/x": "Group",
    "/ds/x/y": "Group",
    "/dangling": "BrokenSoftLink",
    "/dangling/x": "Group",
    "/soft_ds": "Dataset",
    "/soft_ds/x": "Group",
    "/loop": "Group",
    "/loop/x": "Group",
    "/ext_missing": "ExternalLink",
    "/ext_missing/x": "Group",
  };
  const f = openH5(Openable, path);
  try {
    for (const [p, want] of Object.entries(answers)) assert.equal(kind(f.get(p)), want, p);
  } finally {
    f.close();
  }
});
