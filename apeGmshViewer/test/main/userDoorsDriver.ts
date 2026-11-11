// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts, and
// calls its `user:read`, `user:readCase` and (VU16) `user:readResult` handlers
// with a fixed battery of requests over the set in `--dir=`
// (test/main/userDoors.test.ts, VU11 #2750).
// It runs in a child process, because main.ts watches the set it opens and
// keeps the process alive; it prints one JSON line and exits.
//
//   node test/main/userDoorsDriver.ts [--file=<dir>/s.h5] --dir=<root>
//
// `<root>` holds `set/` (the set `--file=` opens), `outside/` and `elsewhere/`.

import { registerHooks } from "node:module";
import { join, sep } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});

const root = process.argv.find((a) => a.startsWith("--dir="))?.slice("--dir=".length);
if (!root) throw new Error("userDoorsDriver needs --dir=<root>");

const { handlers, listeners } = (await import(STUB)) as typeof import("./electronStub.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const set = join(root, "set");
const paired = join(set, "s.user.h5");
const model = join(set, "s.h5");
const CASE = { case: "NEC-15 Ex (synthetic nodal)", target: "nodes", label: "DISPLACEMENT" };
// VU16: footfall's ratio map, W3's state-free user result (a set without it refuses it by name)
const RESULT = { name: "footfall", component: "footfall_ratio" };

/** The paths a renderer could send, each with the set's model and results. */
const PATHS: Record<string, string> = {
  paired,
  sibling: join(set, "t.user.h5"),
  // unnormalised on purpose: `..` out of the set's folder, onto a readable user file
  traversal: `${set}${sep}..${sep}outside${sep}s.user.h5`,
  elsewhere: join(root, "elsewhere", "s.user.h5"),
};

const circular: Record<string, unknown> = { path: 7 };
circular.self = circular;

/** [handler, request] by name. */
const battery: Record<string, [string, unknown]> = {};
for (const [name, path] of Object.entries(PATHS)) {
  battery[`read:${name}`] = ["user:read", { path, model, results: null }];
  battery[`readCase:${name}`] = ["user:readCase", { path, ...CASE }];
  battery[`readResult:${name}`] = ["user:readResult", { path, ...RESULT }];
}
battery["readResult:bigint"] = ["user:readResult", { ...RESULT, path: 5n }];
battery["readResult:circular"] = ["user:readResult", circular];
battery["readResult:otherComponent"] = ["user:readResult", { path: paired, name: "footfall", component: "footfall_nope" }];
battery["read:otherModel"] = ["user:read", { path: paired, model: join(set, "t.h5"), results: null }];
battery["read:otherResults"] = ["user:read", { path: paired, model, results: join(set, "s.results.h5") }];
battery["read:bigint"] = ["user:read", { path: 5n, model, results: null }];
battery["read:circular"] = ["user:read", circular];
battery["readCase:bigint"] = ["user:readCase", { ...CASE, path: 5n }];
battery["readCase:circular"] = ["user:readCase", circular];

const out: Record<string, unknown> = {};
for (const [name, [channel, req]] of Object.entries(battery)) {
  const handle = handlers.get(channel);
  if (!handle) throw new Error(`main.ts registered no ${channel}`);
  try {
    // `user:readCase` answers `data`, `user:readResult` answers `values` (and the ids it is joined by).
    const r = (await handle({}, req)) as { ok: boolean; error?: string; reason?: string; detail?: string; value?: { data?: Float64Array; values?: Float64Array; ids?: Float64Array } };
    const v = r.value;
    out[name] = {
      resolved: { ok: r.ok, error: r.error, reason: r.reason, detail: r.detail, values: v?.data?.length ?? v?.values?.length, ...(v?.values ? { column: { ids: [...v.ids!], values: [...v.values].map((x) => (Number.isNaN(x) ? null : x)) } } : {}) },
    };
  } catch (err) {
    out[name] = { rejected: err instanceof Error ? err.message : String(err) };
  }
}
// After the battery, in order: steer the set through `model:open` (the
// renderer's own door), and read through the user doors after each step.
const call = async (channel: string, ...args: unknown[]) => {
  try {
    const r = (await handlers.get(channel)!({}, ...args)) as { ok: boolean; error?: string; reason?: string; detail?: string; value?: { data: Float64Array } } | null;
    return { resolved: r === null ? null : { ok: r.ok, error: r.error, reason: r.reason, detail: r.detail, values: r.value?.data.length } };
  } catch (err) {
    return { rejected: err instanceof Error ? err.message : String(err) };
  }
};
const userOf = async () => {
  const s = (await handlers.get("viewer:currentSet")!({})) as { user: string | null; model: string | null } | null;
  return s === null ? null : { user: s.user, model: s.model };
};
const elsewhere = PATHS.elsewhere!;
const elsewhereModel = join(root, "elsewhere", "s.h5");
if (process.argv.includes("--steer")) {
  // VU12 (#2753, option A): main opens only what the user dropped, so each steer is a drop (preload.ts reports it on `viewer:dropped`).
  const drop = (path: string) => listeners.get("viewer:dropped")!({ returnValue: undefined }, path);
  out["steer:1 readCase"] = await call("user:readCase", { path: elsewhere, ...CASE });
  drop(elsewhere);
  out["steer:2 model:open user file"] = await call("model:open", elsewhere);
  out["steer:3 currentSet"] = await userOf();
  out["steer:4 readCase"] = await call("user:readCase", { path: elsewhere, ...CASE });
  out["steer:4 read"] = await call("user:read", { path: elsewhere, model, results: null });
  drop(elsewhereModel);
  out["steer:5 model:open model"] = await call("model:open", elsewhereModel);
  out["steer:6 currentSet"] = await userOf();
  out["steer:7 readCase"] = await call("user:readCase", { path: elsewhere, ...CASE });
  // a model still being written (truncated): its open fails, and its set is the one watched
  drop(join(root, "bad", "b.h5"));
  out["steer:8 model:open truncated"] = await call("model:open", join(root, "bad", "b.h5"));
  out["steer:9 currentSet"] = await userOf();
  out["steer:10 readCase"] = await call("user:readCase", { path: elsewhere, ...CASE });
}
process.stdout.write(`USER_DOORS ${JSON.stringify(out)}\n`);
process.exit(0);
