// The group-index check in reader/h5open.ts reads HDF5's error stack as it passes through
// h5wasm's `FS.write` to the host's stderr. On Linux, `node --test` gives each test file a
// non-blocking stderr pipe, so a burst of diagnostics can fill it and the host refuses a
// write (EAGAIN). libc then drops the rest of that fprintf, the frame `in H5O_protect()` went
// missing from the stack, and a torn stage read as a damaged group index: "HDF5 could not look
// up /MODEL_STAGE[1]" instead of the named refusal (the app-tests flake of 2026-11-19, run
// 44563525106). This file refuses every other stderr write, below the check, and the reader
// must still give the named refusal.
//
// It must be its own file: the check binds `FS.write` at the module's first open, so the
// refusing write is installed here before any open in this process.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { lookup3 } from "../../src/reader/h5open.ts";
import { LadrunoError, openLadruno } from "../../src/reader/ladruno.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const KILLED = fileURLToPath(new URL("../../../tests/fixtures/ladruno_killed/killed_swmr.ladruno", import.meta.url));
/** MODEL_STAGE[1]'s object header in KILLED (ladrunoKilledRun.test.ts asserts the same address). */
const OHDR_STAGE_1 = 45736;
const tmp = mkdtempSync(join(tmpdir(), "h5-stderr-refused-"));
after(() => rmSync(tmp, { recursive: true, force: true }));

type Write = (stream: { nfd?: unknown }, ...rest: unknown[]) => number;
const fs = h5wasm.FS as unknown as { write: Write };
const write = fs.write.bind(fs);
let refused = 0;
let n = 0;
fs.write = (stream, ...rest) => {
  if (stream.nfd === 2 && n++ % 2 === 1) {
    refused++;
    throw Object.assign(new Error("EAGAIN"), { name: "ErrnoError", errno: 6 });
  }
  return write(stream, ...rest);
};

test("a torn earlier stage is refused by name while the host refuses HDF5's diagnostics", () => {
  const path = join(tmp, "torn_first.ladruno");
  copyFileSync(KILLED, path);
  const b = readFileSync(path);
  b[11] = 0x00;
  b.writeBigUInt64LE(BigInt(b.length), 28);
  b.writeUInt32LE(lookup3(b.subarray(0, 44)), 44);
  assert.equal(b.subarray(OHDR_STAGE_1, OHDR_STAGE_1 + 4).toString("latin1"), "OHDR");
  b[OHDR_STAGE_1] = b[OHDR_STAGE_1]! ^ 0xff;
  writeFileSync(path, b);

  assert.throws(() => openLadruno(H5, path), (e: unknown) => e instanceof LadrunoError && /MODEL_STAGE\[1\] does not read .*corruption/.test(e.message));
  assert.ok(refused > 0, "the host refused no diagnostic, so this test proves nothing");
});
