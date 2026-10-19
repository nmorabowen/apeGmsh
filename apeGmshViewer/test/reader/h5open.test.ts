// reader/h5open.ts against the rounded-inode alias of lesson #2260.
//
// h5wasm's Node build reports a file's id to HDF5 as BigInt(fs.fstatSync(fd).ino),
// and HDF5's sec2 driver dedupes open files by (dev, ino). Two NTFS ids that
// differ only below the double's resolution (the real pair: 13792273863324032 and
// 13792273863324033, fixtures elem_mixed_unstamped.results.h5 and elem_solid.h5)
// round to one number, so opening B while A was open returned A.
//
// The collision is made deterministic on any filesystem: fs.fstatSync is stubbed
// to report those two ids for two scratch files, as a double for the plain call
// and exactly for {bigint: true}. The oracle is each file's own root keys.
//
// The tests run in declaration order in one process, and the first openH5 call
// fixes h5wasm's FS for the rest of the process, so the control (raw h5wasm,
// aliasing) comes first.

import assert from "node:assert/strict";
import fs, { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import * as h5wasm from "h5wasm/node";
import { openH5, type H5FS, type H5Openable } from "../../src/reader/h5open.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as H5Openable<h5wasm.File>;

// The real colliding NTFS ids of lesson #2260: different as bigints, one double.
const ID_A = 13792273863324032n;
const ID_B = 13792273863324033n;
assert.notEqual(ID_A, ID_B);
assert.equal(Number(ID_A), Number(ID_B));

/** A scratch HDF5 file whose only root member is the group `name`. */
function scratch(name: string): string {
  const p = join(mkdtempSync(join(tmpdir(), "agv-h5open-")), `${name}.h5`);
  const f = new h5wasm.File(p, "w");
  f.create_group(name);
  f.close();
  return p;
}
const A = scratch("a_only");
const B = scratch("b_only");

/** Make fs.fstatSync report ID_A for A and ID_B for B (a double unless {bigint: true}). */
function collide(t: TestContext): void {
  const real = fs.fstatSync.bind(fs);
  const fake = new Map<bigint, bigint>([
    [fs.statSync(A, { bigint: true }).ino, ID_A],
    [fs.statSync(B, { bigint: true }).ino, ID_B],
  ]);
  t.mock.method(fs, "fstatSync", (fd: number, opts?: { bigint?: boolean }) => {
    const id = fake.get(real(fd, { bigint: true }).ino);
    const st = real(fd, opts) as { ino: number | bigint };
    if (id !== undefined) st.ino = opts?.bigint ? id : Number(id);
    return st;
  });
}

test("control: raw h5wasm opens B as A when their ids round to one double", (t) => {
  collide(t);
  const a = new h5wasm.File(A, "r");
  try {
    const b = new h5wasm.File(B, "r");
    try {
      assert.deepEqual(a.keys(), ["a_only"]);
      assert.deepEqual(b.keys(), ["a_only"], "the stub must reproduce the alias, or the next test proves nothing");
    } finally {
      b.close();
    }
  } finally {
    a.close();
  }
});

test("openH5 reads B's own data while A is open, under the same colliding ids", (t) => {
  collide(t);
  const a = openH5(H5, A);
  try {
    const b = openH5(H5, B);
    try {
      assert.deepEqual(a.keys(), ["a_only"]);
      assert.deepEqual(b.keys(), ["b_only"]);
    } finally {
      b.close();
    }
  } finally {
    a.close();
  }
});

test("openH5 refuses before h5wasm is ready", () => {
  const notReady = { File: H5.File, FS: null } as H5Openable<h5wasm.File>;
  assert.throws(() => openH5(notReady, A), /h5wasm is not ready/);
});

test("openH5 refuses an h5wasm whose FS lacks the members the fix needs", () => {
  const bare = { File: H5.File, FS: {} as H5FS } as H5Openable<h5wasm.File>;
  assert.throws(() => openH5(bare, A), /cannot be installed; refusing to open HDF5 files/);
});

test("the installed fstat refuses a stream with no host descriptor", () => {
  const fsLike: H5FS = { fstat: () => ({ ino: 1 }), getStreamChecked: () => ({}) };
  let opened = false;
  class Probe {
    constructor() {
      opened = true;
      fsLike.fstat(3);
    }
  }
  const mod = { File: Probe, FS: fsLike } as unknown as H5Openable<Probe>;
  assert.throws(() => openH5(mod, A), /has no host file descriptor/);
  assert.ok(opened);
});
