// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts (whose
// `utilityProcess.fork` forks src/main/loader/process.ts as a Node child), and calls
// `model:open` and `geometry:open` on the set of its `--file=` (test/main/loader.test.ts,
// ADR 0137 D2, #3058). It prints one JSON line and exits.
//
//   node test/main/loaderDriver.ts --case=<good|unreadable> --mode=describe --file=<dir>/s.h5 --out=<dir>/out

import { registerHooks } from "node:module";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});

const opt = (name: string) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const which = opt("case");
const file = opt("file");
if (!which || !file) throw new Error("loaderDriver needs --case= and --file=");
const model = resolve(file);

const { forked, handlers } = (await import(STUB)) as typeof import("./electronStub.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const event = { sender: { isDestroyed: () => false, send: () => undefined } };
type Opened = { ok: boolean; error?: string; model?: { nodeIds: unknown; blocks: unknown[] }; provenance?: { ok: boolean }; lineage?: { ok: boolean }; stageElements?: { ok: boolean }; geometry?: unknown };
async function call(channel: string, ...args: unknown[]): Promise<Opened> {
  const handle = handlers.get(channel);
  if (!handle) throw new Error(`main.ts registered no ${channel}`);
  return (await handle(event, ...args)) as Opened;
}
/** What the test judges of a `model:open` answer; the model stays here. */
const seen = (r: Opened) => ({
  ok: r.ok,
  error: r.error,
  nodeIds: r.model ? (r.model.nodeIds as object).constructor.name : null,
  nodes: r.model ? (r.model.nodeIds as ArrayLike<number>).length : null,
  blocks: r.model ? r.model.blocks.length : null,
  zones: r.ok ? [r.provenance?.ok, r.lineage?.ok, r.stageElements?.ok] : null,
});

const out: Record<string, unknown> = {};
if (which === "unreadable") {
  out["open"] = seen(await call("model:open", model));
} else {
  out["open"] = seen(await call("model:open", model));
  out["forksAfterOpen"] = forked.length;
  out["entry"] = forked[0]?.modulePath ?? null;
  const g = await call("geometry:open", model.replace(/\.h5$/, ".geometry.h5"));
  out["geometry"] = { ok: g.ok, error: g.error, drawn: g.geometry !== undefined && g.geometry !== null };
  out["forksAfterGeometry"] = forked.length;
  // The loader stops between reads: the next read starts another.
  forked.at(-1)!.kill();
  await forked.at(-1)!.exited;
  out["afterStop"] = seen(await call("model:open", model));
  out["forksAfterStop"] = forked.length;
  // The loader stops during a read: that read is refused with why, and the next read starts another.
  const during = call("model:open", model);
  forked.at(-1)!.kill();
  out["stoppedDuring"] = seen(await during);
  out["afterStoppedDuring"] = seen(await call("model:open", model));
  out["forksAtEnd"] = forked.length;
}
process.stdout.write(`LOADER ${JSON.stringify(out)}\n`);
process.exit(0);
