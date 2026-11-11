// A save that dies (test/main/workspaceSave.test.ts): it runs main/workspace.ts
// `saveWorkspace` on `<dir>/frame.h5`'s workspace and kills its own process,
// hard, at the step `--kill-at=` names (`partial-written`, `verified`,
// `before-replace`), or (`--loop`) saves until its parent kills it.
//
//   node test/main/workspaceKillDriver.ts --dir=<dir> --kill-at=<step>
//   node test/main/workspaceKillDriver.ts --dir=<dir> --loop

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import * as h5wasm from "h5wasm/node";
import { nodeDeps, saveWorkspace } from "../../src/main/workspace.ts";
import type { WorkspaceH5 } from "../../src/reader/workspace.ts";
import type { OpenSet } from "../../src/main/pairing.ts";

const arg = (k: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.slice(k.length + 3);
const dir = arg("dir");
if (!dir) throw new Error("workspaceKillDriver needs --dir=<dir>");
await h5wasm.ready;
const killAt = arg("kill-at");
const deps = {
  ...nodeDeps(h5wasm as unknown as WorkspaceH5, h5wasm.ready, "kill driver"),
  step: (name: string) => {
    if (name === killAt) process.kill(process.pid, "SIGKILL");
  },
};
/** The token of the file as it is now, as a read hands it to the page. */
const expect = () => {
  const ws = join(dir, "frame.workspace.h5");
  const bytes = existsSync(ws) ? createHash("sha256").update(readFileSync(ws)).digest("hex") : "absent";
  const path = resolve(ws);
  return createHash("sha256").update(`${process.platform === "win32" || process.platform === "darwin" ? path.toLowerCase() : path}\0${bytes}`).digest("hex");
};
const set: OpenSet = { model: join(dir, "frame.h5"), geometry: null, results: null, user: null, workspace: null };
if (process.argv.includes("--loop")) {
  process.stdout.write("LOOPING\n");
  for (let i = 0; ; i++) {
    // A name of this process's own: a set an earlier killed loop left behind never collides with it.
    const name = `churn ${process.pid} ${i >> 1}`;
    const edit = i % 2 ? { op: "delete", table: "sets", name } : { op: "create", name, kind: "nodes", ids: [1, 2, 3, 11, 12, 13] };
    const r = await saveWorkspace(deps, set, { ...edit, expect: expect() });
    if (!r.ok) throw new Error(`save ${i} refused: ${r.reason}: ${r.detail}`);
  }
}
const r = await saveWorkspace(deps, set, { op: "create", name: "killed", kind: "nodes", ids: [21, 22], expect: expect() });
process.stdout.write(`NOT KILLED ${JSON.stringify(r)}\n`);
