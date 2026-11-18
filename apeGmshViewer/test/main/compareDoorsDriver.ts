// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts,
// opened in view mode on a variants manifest (`--file=<study>.variants.json`),
// and calls VU19's compare doors (`compare:model`, `compare:userRead`,
// `compare:userReadResult`) and the window set's user doors with a fixed
// battery (test/main/compareDoors.test.ts, VU19 #2861). It runs in a child
// process, because main.ts reads its mode and file from argv; it prints one
// JSON line and exits.
//
//   node test/main/compareDoorsDriver.ts --dir=<root> --file=<root>/study/study.variants.json
//
// `<root>/study` holds the VU10 pair with a `.user.h5` beside each model (test/fixtures/vu19).

import { renameSync } from "node:fs";
import { registerHooks } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});

const opt = (name: string) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const root = opt("dir");
if (!root) throw new Error("compareDoorsDriver needs --dir=");

const { appEvents, handlers, pageEvents, ready } =(await import(STUB)) as typeof import("./electronStub.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const study = join(root, "study");
const UNCRACKED = { model: join(study, "plate_uncracked.h5"), results: join(study, "plate_uncracked.ladruno"), user: join(study, "plate_uncracked.user.h5") };
const CRACKED = { model: join(study, "cracked", "plate_cracked.h5"), results: join(study, "cracked", "plate_cracked.ladruno"), user: join(study, "cracked", "plate_cracked.user.h5") };

const out: Record<string, unknown> = {};
const event = { sender: { isDestroyed: () => false, send: () => undefined } };

async function call(channel: string, ...args: unknown[]): Promise<unknown> {
  const handle = handlers.get(channel);
  if (!handle) throw new Error(`main.ts registered no ${channel}`);
  try {
    return { resolved: await handle(event, ...args) };
  } catch (err) {
    return { rejected: err instanceof Error ? err.message : String(err) };
  }
}

/** A `user:readResult`-shaped answer with its column as plain arrays, so it survives JSON. */
function plain(a: unknown): unknown {
  return JSON.parse(JSON.stringify(a, (_k, v: unknown) => (v instanceof Float64Array ? Array.from(v) : typeof v === "bigint" ? String(v) : v)));
}

const all = async (tag: string) => {
  out[`${tag}:model`] = plain(await call("compare:model", {}));
  out[`${tag}:userRead`] = plain(await call("compare:userRead", {}));
  out[`${tag}:userReadResult`] = plain(await call("compare:userReadResult", { name: "footfall", component: "ratio" }));
};

// 1. No compare set: every compare door refuses.
await all("none");
// 2. The compare set open: its model, its own user file judged against its own model, and a case read.
out["compare"] = plain(await call("variants:compare", 1, "Cracked"));
await all("open");
out["open:drift"] = plain(await call("compare:userReadResult", { name: "drift", component: "d" }));
// A compare door names no path and no row: a request that carries one is refused, whatever it names.
out["path:model"] = plain(await call("compare:model", { path: CRACKED.model }));
out["path:userRead"] = plain(await call("compare:userRead", { path: CRACKED.user }));
out["path:userReadResult"] = plain(await call("compare:userReadResult", { path: CRACKED.user, name: "footfall", component: "ratio" }));
out["row:userRead"] = plain(await call("compare:userRead", { index: 0, label: "Uncracked" }));
// 3. The shared namespace, one way: the window doors read the window set's file, and refuse the compare set's.
out["window:userRead own"] = plain(await call("user:read", { path: UNCRACKED.user, model: UNCRACKED.model, results: UNCRACKED.results }));
out["window:userReadResult own"] = plain(await call("user:readResult", { path: UNCRACKED.user, name: "footfall", component: "ratio" }));
out["window:userRead compare"] = plain(await call("user:read", { path: CRACKED.user, model: CRACKED.model, results: CRACKED.results }));
out["window:userReadResult compare"] = plain(await call("user:readResult", { path: CRACKED.user, name: "footfall", component: "ratio" }));
out["window:set"] = plain(await call("viewer:currentSet"));
// 4. The other way: with no .user.h5 in the compare set, the compare read names the compare set's
// missing file, never the window set's (which is still there).
renameSync(CRACKED.user, `${CRACKED.user}.away`);
out["gone:userRead"] = plain(await call("compare:userRead", {}));
out["gone:userReadResult"] = plain(await call("compare:userReadResult", { name: "footfall", component: "ratio" }));
renameSync(`${CRACKED.user}.away`, CRACKED.user);
// 4b. The compare row's own model opened as the window set another way than a switch (a second
// instance's argv; a drop, an offer and macOS open-file take the same openPath) empties the slot (review F1).
const second = appEvents.get("second-instance") ?? [];
if (second.length === 0) throw new Error("main.ts listens for no second-instance");
for (const fn of second) fn({}, [process.execPath, "main.mjs", `--file=${CRACKED.model}`], root, null);
out["second:set"] = plain(await call("viewer:currentSet"));
out["second:study"] = plain(await call("variants:current"));
await all("second");
await call("variants:switch", 0, "Uncracked");
await call("variants:compare", 1, "Cracked");
// 5. compare:close: every compare door refuses again.
await call("compare:close");
await all("closed");
// 6. A switch onto the compare row empties the slot.
await call("variants:compare", 1, "Cracked");
out["switch"] = plain(await call("variants:switch", 1, "Cracked"));
await all("switched");
// 7. A page load empties it too (the reload rule of VU10/VU14).
await call("variants:switch", 0, "Uncracked");
out["reopen"] = plain(await call("variants:compare", 1, "Cracked"));
await all("before reload");
ready();
await new Promise((r) => setImmediate(r));
for (const fn of pageEvents.get("did-start-loading") ?? []) fn({});
await all("reloaded");

process.stdout.write(`COMPARE_DOORS ${JSON.stringify(out)}\n`);
process.exit(0);
