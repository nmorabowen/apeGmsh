// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts,
// opens `<dir>/set/frame.h5` the way a double-click does (`--file=`), and calls
// its `workspace:read` and `workspace:save` doors with a fixed battery of
// requests (test/main/workspaceDoors.test.ts, VU9a #2803). It runs in a child
// process, because main.ts watches the set it opens; it prints one JSON line
// and exits.
//
//   node test/main/workspaceDoorsDriver.ts --file=<dir>/set/frame.h5 --dir=<dir> [--mode=render ...]
//   node test/main/workspaceDoorsDriver.ts --file=<A>/frame.h5 --dir=<dir> --switch-to=<B>/frame.h5
//
// `--switch-to` reads A's workspace, then opens B as a drop does (the window
// set changes before the page hears of it), and sends an edit made on A's
// read: main must refuse it `stale`, whatever B's file holds.

import { registerHooks } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});

const root = process.argv.find((a) => a.startsWith("--dir="))?.slice("--dir=".length);
if (!root) throw new Error("workspaceDoorsDriver needs --dir=<root>");

const { handlers, listeners } = (await import(STUB)) as typeof import("./electronStub.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const call = (channel: string, ...args: unknown[]) => {
  const handle = handlers.get(channel);
  if (!handle) throw new Error(`main.ts registered no ${channel}`);
  return handle({}, ...args);
};
const switchTo = process.argv.find((a) => a.startsWith("--switch-to="))?.slice("--switch-to=".length);
if (switchTo) {
  const readA = (await call("workspace:read")) as { token?: string; file?: string };
  listeners.get("viewer:dropped")!({ returnValue: undefined }, switchTo);
  const opened = await call("viewer:requestOpen", switchTo);
  const edits = [
    { op: "create", name: "made on A", kind: "nodes", ids: [31, 32, 33], expect: readA.token },
    { op: "delete", table: "sets", name: "roof", expect: readA.token },
  ];
  const answers = [];
  for (const e of edits) answers.push(await call("workspace:save", e));
  const readB = await call("workspace:read");
  process.stdout.write(`WORKSPACE_DOORS ${JSON.stringify({ readA, opened, answers, readB })}\n`);
  process.exit(0);
}

const elsewhere = join(root, "elsewhere", "frame.workspace.h5");
// The edit is made on the set's first read, as the page makes it: its token.
const first = (await call("workspace:read")) as { token: string };
const good = { op: "create", name: "roof", kind: "nodes", ids: [31, 32, 33], expect: first.token };

/** [channel, args] by name, in order. */
const battery: [string, string, unknown[]][] = [
  ["read:path", "workspace:read", [elsewhere]],
  ["read:object", "workspace:read", [{ path: elsewhere }]],
  ["save:pathKey", "workspace:save", [{ ...good, path: elsewhere }]],
  ["save:fileKey", "workspace:save", [{ ...good, file: elsewhere }]],
  ["save:string", "workspace:save", [elsewhere]],
  ["save:second", "workspace:save", [good, elsewhere]],
  ["save:none", "workspace:save", []],
  ["save:bigint", "workspace:save", [{ ...good, ids: [31n] }]],
  ["save:staleToken", "workspace:save", [{ ...good, expect: "0".repeat(64) }]],
  ["read:before", "workspace:read", []],
  ["save:good", "workspace:save", [good]],
  ["read:after", "workspace:read", []],
  // The same edit again on the token of the first read: the file the page read is no longer what is there.
  ["save:afterOwnSave", "workspace:save", [{ ...good, name: "roof again" }]],
];

const out: Record<string, unknown> = {};
for (const [name, channel, args] of battery) {
  const handle = handlers.get(channel);
  if (!handle) throw new Error(`main.ts registered no ${channel}`);
  try {
    out[name] = { resolved: await handle({}, ...args) };
  } catch (err) {
    out[name] = { rejected: err instanceof Error ? err.message : String(err) };
  }
}
process.stdout.write(`WORKSPACE_DOORS ${JSON.stringify(out)}\n`);
process.exit(0);
