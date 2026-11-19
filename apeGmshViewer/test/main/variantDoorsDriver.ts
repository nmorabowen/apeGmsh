// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts,
// opened in view mode on a variants manifest (`--file=<study>.variants.json`,
// as a double-click or argv gives it), and calls its variant doors and the
// window set's doors with a fixed battery (test/main/variantDoors.test.ts,
// VU10 #2694). It runs in a child process, because main.ts reads its mode and
// file from argv; it prints one JSON line and exits.
//
//   node test/main/variantDoorsDriver.ts --dir=<root> --file=<root>/study/study.variants.json
//
// `<root>/study` holds the VU10 pair: plate_uncracked.h5 and its .ladruno,
// cracked/plate_cracked.h5 and its .ladruno, and the manifest.

import { registerHooks } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});

const opt = (name: string) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const root = opt("dir");
if (!root) throw new Error("variantDoorsDriver needs --dir=");

const { handlers, pageEvents, ready } = (await import(STUB)) as typeof import("./electronStub.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const study = join(root, "study");
const UNCRACKED = { model: join(study, "plate_uncracked.h5"), results: join(study, "plate_uncracked.ladruno"), user: join(study, "plate_uncracked.user.h5") };
const CRACKED = { model: join(study, "cracked", "plate_cracked.h5"), results: join(study, "cracked", "plate_cracked.ladruno"), user: join(study, "cracked", "plate_cracked.user.h5") };

const out: Record<string, unknown> = {};
const event = { sender: { isDestroyed: () => false, send: () => undefined } };

async function call(channel: string, ...args: unknown[]): Promise<Record<string, unknown>> {
  const handle = handlers.get(channel);
  if (!handle) throw new Error(`main.ts registered no ${channel}`);
  return ((await handle(event, ...args)) ?? {}) as Record<string, unknown>;
}

/** What a door answered, without its arrays: ok, the reason, and the few facts the test judges. */
function seen(r: Record<string, unknown>): Record<string, unknown> {
  const c = r["compare"] as { index: number; label: string; results: string; femHash: string | null; stages: unknown[] } | undefined;
  const nodal = r["nodal"] as Record<string, Float64Array> | undefined;
  return {
    ok: r["ok"],
    reason: r["reason"] ?? r["error"] ?? null,
    ...(c ? { compare: { index: c.index, label: c.label, results: c.results, femHash: c.femHash, stages: c.stages.length } } : {}),
    ...(nodal ? { uy: Array.from(nodal["DISPLACEMENT/Uy"] ?? []) } : {}),
  };
}

const windowModel = async () => ((await call("viewer:currentSet")) as { model?: string | null })?.model ?? null;
const view = async () => (await call("variants:current")) as unknown;

// 1. The manifest opened the study, and its first row as the window set.
out["open:model"] = await windowModel();
out["open:study"] = await view();
// 2. The compare door: a row by index and label opens read-only; the window set does not move.
out["compare:mismatch"] = seen(await call("variants:compare", 1, "Uncracked"));
out["compare:path"] = seen(await call("variants:compare", CRACKED.model, "Cracked"));
out["compare:window"] = seen(await call("variants:compare", 0, "Uncracked"));
out["compare:ok"] = seen(await call("variants:compare", 1, "Cracked"));
out["compare:model"] = await windowModel();
out["compare:study"] = await view();
// The window doors still read only the window set: the compare set's files are refused there.
out["window:results compare"] = seen(await call("results:open", CRACKED.results));
out["window:user compare"] = seen(await call("user:read", { path: CRACKED.user, model: CRACKED.model, results: CRACKED.results }));
out["window:results own"] = seen(await call("results:open", UNCRACKED.results));
// The compare host is not the window's: closing the window's results leaves the compare set readable.
await call("results:close");
out["compare:step after results:close"] = seen(await call("compare:readStep", { stage: "stage_0", step: 2, seq: 7 }));
// A compare step names no path: a request that carries one is refused, whatever it names.
out["compare:step with path"] = seen(await call("compare:readStep", { path: CRACKED.results, stage: "stage_0", step: 2, seq: 8 }));
out["compare:step with window path"] = seen(await call("compare:readStep", { path: UNCRACKED.results, stage: "stage_0", step: 2, seq: 9 }));
// 3. The switch: the compare row becomes the window set, and the compare slot empties.
out["switch:mismatch"] = seen(await call("variants:switch", 1, "Uncracked"));
out["switch:ok"] = seen(await call("variants:switch", 1, "Cracked"));
out["switch:model"] = await windowModel();
out["switch:study"] = await view();
out["switch:compare step"] = seen(await call("compare:readStep", { stage: "stage_0", step: 2, seq: 10 }));
// 4. At most one compare set, and it closes.
out["compare2:ok"] = seen(await call("variants:compare", 0, "Uncracked"));
out["compare2:step"] = seen(await call("compare:readStep", { stage: "stage_0", step: 2, seq: 11 }));
await call("compare:close");
out["close:step"] = seen(await call("compare:readStep", { stage: "stage_0", step: 2, seq: 12 }));
out["close:study"] = await view();
// 5. A page load empties the compare slot: the new page starts with no compare set (#2817 review 4).
out["reload:compare"] = seen(await call("variants:compare", 0, "Uncracked"));
out["reload:step before"] = seen(await call("compare:readStep", { stage: "stage_0", step: 2, seq: 13 }));
ready(); // main.ts makes its window once the app is ready
await new Promise((r) => setImmediate(r));
const loads = pageEvents.get("did-start-loading") ?? [];
if (loads.length === 0) throw new Error("main.ts listens for no did-start-loading");
for (const fn of loads) fn({});
out["reload:step after"] = seen(await call("compare:readStep", { stage: "stage_0", step: 2, seq: 14 }));
out["reload:study"] = await view();

process.stdout.write(`VARIANT_DOORS ${JSON.stringify(out)}\n`);
process.exit(0);
