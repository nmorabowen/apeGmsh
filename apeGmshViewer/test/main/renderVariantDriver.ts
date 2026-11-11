// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts,
// in render mode on a spec over the VU10 pair (`--mode=render --spec=<spec>
// --out=<out>`), and calls `render:variant` and the window set's doors with a
// fixed battery (test/main/renderVariant.test.ts, VU18 #2856). It runs in a
// child process, because main.ts reads its mode and files from argv; it prints
// one JSON line and exits.
//
//   node test/main/renderVariantDriver.ts --dir=<root> --mode=render --spec=<spec> --out=<out> [--battery=no-variants]
//
// `<root>/study` holds the VU10 pair and its manifest, which the battery
// rewrites between calls, as a writer between two shots would.

import { readFileSync, writeFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});

const opt = (name: string) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const root = opt("dir");
if (!root) throw new Error("renderVariantDriver needs --dir=");

const { handlers } = (await import(STUB)) as typeof import("./electronStub.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const study = join(root, "study");
const manifest = join(study, "study.variants.json");
const UNCRACKED = { model: join(study, "plate_uncracked.h5"), results: join(study, "plate_uncracked.ladruno") };
const CRACKED = { model: join(study, "cracked", "plate_cracked.h5"), results: join(study, "cracked", "plate_cracked.ladruno") };

const out: Record<string, unknown> = {};
const event = { sender: { isDestroyed: () => false, send: () => undefined } };

async function call(channel: string, ...args: unknown[]): Promise<Record<string, unknown>> {
  const handle = handlers.get(channel);
  if (!handle) throw new Error(`main.ts registered no ${channel}`);
  return ((await handle(event, ...args)) ?? {}) as Record<string, unknown>;
}

/** What a door answered: ok, the reason, and the set it named. */
function seen(r: Record<string, unknown>): Record<string, unknown> {
  return { ok: r["ok"], reason: r["reason"] ?? r["error"] ?? null, ...(r["set"] !== undefined ? { set: r["set"], row: r["row"] } : {}) };
}

/** Which results file the window's door reads now: each of the two is asked, and only the drawn set's opens. */
async function doors(): Promise<Record<string, unknown>> {
  const a = seen(await call("results:open", UNCRACKED.results));
  const b = seen(await call("results:open", CRACKED.results));
  await call("results:close");
  return { uncracked: a.ok, cracked: b.ok, reason: a.ok ? b.reason : a.reason };
}

function rewrite(edit: (rows: { label: string; path: string; params: object }[]) => void): void {
  const m = JSON.parse(readFileSync(manifest, "utf8")) as { variants: { label: string; path: string; params: object }[] };
  edit(m.variants);
  writeFileSync(manifest, JSON.stringify(m, null, 2));
}

if (opt("battery") === "no-variants") {
  out["row"] = seen(await call("render:variant", { index: 1, label: "Cracked" }));
  out["doors"] = await doors();
} else {
  // 1. The spec's set is drawn first.
  out["job"] = await call("render:job");
  out["doors:start"] = await doors();
  // 2. A compare slot holding row 1, then a shot that draws row 1: the slot empties (the drawn set is never its compare set).
  out["compare:1"] = seen(await call("variants:compare", 1, "Cracked"));
  out["row:1"] = seen(await call("render:variant", { index: 1, label: "Cracked" }));
  out["compare:step after row:1"] = seen(await call("compare:readStep", { stage: "stage_0", step: 2, seq: 1 }));
  out["doors:row 1"] = await doors();
  out["model:row 1"] = seen(await call("model:open", CRACKED.model));
  out["model:spec set"] = seen(await call("model:open", UNCRACKED.model));
  // 3. Degenerate names: refused, and the drawn set stays where it was.
  out["row:case"] = seen(await call("render:variant", { index: 1, label: "cracked" }));
  out["row:out of range"] = seen(await call("render:variant", { index: 5, label: "Cracked" }));
  out["row:negative"] = seen(await call("render:variant", { index: -1, label: "Cracked" }));
  out["row:empty label"] = seen(await call("render:variant", { index: 1, label: "" }));
  out["row:path"] = seen(await call("render:variant", CRACKED.model));
  out["row:path key"] = seen(await call("render:variant", { index: 1, label: "Cracked", path: CRACKED.model }));
  out["doors:after refusals"] = await doors();
  // 4. Reload (D9.3): the manifest rewritten between two shots is read again.
  rewrite((rows) => (rows[1]!.label = "Cracked x2"));
  out["reload:old label"] = seen(await call("render:variant", { index: 1, label: "Cracked" }));
  out["reload:new label"] = seen(await call("render:variant", { index: 1, label: "Cracked x2" }));
  // 5. D8: a row whose path leaves the study is refused by the read, with its code.
  rewrite((rows) => (rows[1]!.path = "../outside/plate_cracked.h5"));
  out["confine:row"] = seen(await call("render:variant", { index: 1, label: "Cracked x2" }));
  out["doors:after confine"] = await doors();
  // 6. Back to the spec's set.
  out["row:null"] = seen(await call("render:variant", null));
  out["doors:spec set"] = await doors();
  // 7. The page's switch door stays shut in render mode.
  out["switch"] = seen(await call("variants:switch", 0, "Uncracked"));
}

process.stdout.write(`RENDER_VARIANT ${JSON.stringify(out)}\n`);
process.exit(0);
