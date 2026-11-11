// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts,
// opens `<dir>/set/frame.h5` the way a double-click does (`--file=`), and calls
// its `workspace:read` and `workspace:save` doors with a fixed battery of
// requests (test/main/workspaceDoors.test.ts, VU9a #2803). It runs in a child
// process, because main.ts watches the set it opens; it prints one JSON line
// and exits.
//
//   node test/main/workspaceDoorsDriver.ts --file=<dir>/set/frame.h5 --dir=<dir> [--mode=render ...]

import { registerHooks } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});

const root = process.argv.find((a) => a.startsWith("--dir="))?.slice("--dir=".length);
if (!root) throw new Error("workspaceDoorsDriver needs --dir=<root>");

const { handlers } = (await import(STUB)) as typeof import("./electronStub.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const elsewhere = join(root, "elsewhere", "frame.workspace.h5");
const good = { op: "create", name: "roof", kind: "nodes", ids: [31, 32, 33] };

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
  ["read:before", "workspace:read", []],
  ["save:good", "workspace:save", [good]],
  ["read:after", "workspace:read", []],
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
