// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts and main's loader
// client to ./heldLoader.ts, and runs two `model:open`s that overlap: the first is held until the
// second has finished (test/main/openOrder.test.ts; architecture review 2026-11-24, finding 2).
// It runs in a child process, as the other door drivers do; it prints one JSON line and exits.
//
//   node test/main/openOrderDriver.ts --dir=<root>
//
// `<root>` holds a/s.h5 and b/s.h5 (models) and b/s.ladruno (b's results), and c/s.results.h5,
// a file that is no model, so its model:open fails.

import { registerHooks } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
const HELD = new URL("./heldLoader.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => {
    if (specifier === "electron") return { url: STUB, shortCircuit: true };
    // main.ts's own import only; heldLoader.ts imports the real client by its relative path from test/.
    if (specifier === "./loader/client.ts" && context.parentURL?.endsWith("/src/main/main.ts")) return { url: HELD, shortCircuit: true };
    return next(specifier, context);
  },
});

const root = process.argv.find((a) => a.startsWith("--dir="))?.slice(6);
if (!root) throw new Error("openOrderDriver needs --dir=");

const { handlers, listeners } = (await import(STUB)) as typeof import("./electronStub.ts");
const { hold, release } = (await import(HELD)) as typeof import("./heldLoader.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const event = { sender: { isDestroyed: () => false, send: () => undefined } };
const drop = (path: string) => listeners.get("viewer:dropped")!({ returnValue: undefined }, path);
const open = (path: string) => handlers.get("model:open")!(event, path) as Promise<{ ok: boolean; error?: string }>;
const modelOf = async () => ((await handlers.get("viewer:currentSet")!(event)) as { model: string | null } | null)?.model ?? null;

const A = join(root, "a", "s.h5");
const B = join(root, "b", "s.h5");
const C = join(root, "c", "s.results.h5");
const out: Record<string, unknown> = {};

// 1. A large model dropped first, a small one second: B finishes, then A.
hold(A);
drop(A);
const first = open(A);
drop(B);
const second = await open(B);
out["1 second ok"] = second.ok;
out["1 after second"] = await modelOf();
release(A);
out["1 first ok"] = (await first).ok;
out["1 after first"] = await modelOf();
// the doors pair against the set main follows: B's results open, by B's path
const r = (await handlers.get("results:open")!(event, join(root, "b", "s.ladruno"))) as { ok: boolean; error?: string };
out["1 results:open b"] = { ok: r.ok, error: r.error };

// 2. An older open that fails after a newer one finished leaves the set alone too.
hold(C);
drop(C);
const failing = open(C);
drop(A);
out["2 newer ok"] = (await open(A)).ok;
release(C);
out["2 older ok"] = (await failing).ok;
out["2 after older"] = await modelOf();

process.stdout.write(`OPEN_ORDER ${JSON.stringify(out)}\n`);
process.exit(0);
