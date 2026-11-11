// A combination save in a process of its own (test/main/combinationsSave.test.ts,
// and the D7 fixtures of test/fixtures/vu9b/README.md). It runs main/combinations.ts
// `saveCombination` on `<dir>/plate.user.h5` (model `<dir>/plate.h5`), the function
// behind the `combinations:save` door.
//
//   node test/main/combinationsDriver.ts --dir=<dir> --op=add|revert [--kill-at=<step>] [--loop]
//
// `--kill-at` names a step of the save (`copying`, `copied`, `appending`,
// `appended`, `verified`, `before-replace`) where the process kills itself,
// hard. `--block=<bytes>` makes the copy take several blocks, so `copying` lands
// inside it. `--loop` saves and reverts until its parent kills it. Otherwise it
// prints the answer as JSON on one line.

import { join } from "node:path";
import * as h5wasm from "h5wasm/node";
import { combinationToken, nodeCombinationDeps, nodeCombinationDisk, saveCombination, type CombinationDeps, type CombinationH5 } from "../../src/main/combinations.ts";
import { listDir } from "../../src/main/pairing.ts";
import { readRevision, type UserH5 } from "../../src/reader/userFile.ts";

const arg = (k: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.slice(k.length + 3);
const dir = arg("dir");
if (!dir) throw new Error("combinationsDriver needs --dir=<dir>");
await h5wasm.ready;
const H5 = h5wasm as unknown as CombinationH5;
const killAt = arg("kill-at");
const block = arg("block");
const base = nodeCombinationDeps(H5, h5wasm.ready, "apeGmshViewer driver", listDir);
const deps: CombinationDeps = {
  ...base,
  disk: block
    ? {
        ...nodeCombinationDisk,
        // The same copy in small blocks, so a kill lands inside it.
        copy: (from, to, chunk) => copyInBlocks(from, to, Number(block), chunk),
      }
    : nodeCombinationDisk,
  step: (name: string) => {
    if (name === killAt) process.kill(process.pid, "SIGKILL");
  },
};
const file = join(dir, "plate.user.h5");
const target = { user: file, model: join(dir, "plate.h5"), results: null };
const expect = () => combinationToken(file, readRevision(H5 as unknown as UserH5, file));
const ROW = { name: "SRSS Ex modes 1-2", rule: "srss", envelope: "", fraction: null, damping: null, terms: [1, 2].map((k) => ({ kind: "spectrum", source: "Ex (synthetic)", step: -1, mode: k, factor: 1 })) };

if (process.argv.includes("--loop")) {
  process.stdout.write("LOOPING\n");
  for (let i = 0; ; i++) {
    const edit = i % 2 ? { op: "revert", expect: expect() } : { op: "add", row: { ...ROW, name: `churn ${process.pid} ${i}` }, expect: expect() };
    const r = await saveCombination(deps, target, edit);
    if (!r.ok) throw new Error(`save ${i} refused: ${r.reason}: ${r.detail}`);
  }
}
const op = arg("op") ?? "add";
// `--preset=` saves another row than the default one (the cross-read fixture, test/fixtures/vu9b/README.md).
const PRESETS: Record<string, unknown> = {
  cqc: { name: "CQC Ex modes 1-3 3%", rule: "cqc", envelope: "", fraction: null, damping: 0.03, terms: [1, 2, 3].map((k) => ({ kind: "spectrum", source: "Ex (synthetic)", step: -1, mode: k, factor: 1 })) },
  linear: { name: "1.3 SRSS Ex", rule: "linear", envelope: "", fraction: null, damping: null, terms: [{ kind: "combination", source: "SRSS Ex", step: -1, mode: 0, factor: 1.3 }] },
};
const preset = arg("preset");
if (preset !== undefined && !Object.hasOwn(PRESETS, preset)) throw new Error(`no preset ${preset}`);
const row = preset !== undefined ? PRESETS[preset] : { ...ROW, name: arg("name") ?? ROW.name };
const r = await saveCombination(deps, target, op === "revert" ? { op, expect: expect() } : { op: "add", row, expect: expect() });
process.stdout.write(`${killAt ? "NOT KILLED " : ""}${JSON.stringify(r.ok ? { ok: true, op: r.op, generation: r.generation, revision: r.revision } : r)}\n`);

function copyInBlocks(from: string, to: string, size: number, chunk: () => void): void {
  const fs = process.getBuiltinModule("node:fs");
  const src = fs.openSync(from, "r");
  const dst = fs.openSync(to, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL);
  try {
    const buf = Buffer.alloc(size);
    for (;;) {
      const n = fs.readSync(src, buf, 0, size, null);
      if (n === 0) break;
      fs.writeSync(dst, buf, 0, n);
      chunk();
    }
  } finally {
    fs.closeSync(dst);
    fs.closeSync(src);
  }
}
