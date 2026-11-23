// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts (whose
// `utilityProcess.fork` forks src/main/loader/process.ts as a Node child), and calls
// `model:open` and `geometry:open` on the set of its `--file=`, or (`--case=results`) the
// `results:*` doors (test/main/loader.test.ts, ADR 0137 D2, #3058). It prints one JSON line and exits.
//
//   node test/main/loaderDriver.ts --case=<good|unreadable> --mode=describe --file=<dir>/s.h5 --out=<dir>/out
//   node test/main/loaderDriver.ts --case=results --mode=capture --file=<dir>/s.h5 --out=<dir>/out.png

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

/** What main sent the asking page: the scans' progress (`results:scanProgress`). */
const progress: { scanId: number; done: number; total: number }[] = [];
const event = {
  sender: {
    isDestroyed: () => false,
    send: (channel: string, p: { scanId: number; done: number; total: number }) => void (channel === "results:scanProgress" && progress.push(p)),
  },
};
/** Wait until `cond`, or fail after 20 s. */
async function until(cond: () => boolean, what: string): Promise<void> {
  const t = Date.now();
  while (!cond()) {
    if (Date.now() - t > 20_000) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 10));
  }
}
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
if (which === "results") {
  // A capture paces a scan at one step per chunk, 2 s apart (main.ts `results:scan`), so a cancel lands mid-scan.
  const results = model.replace(/\.h5$/, ".results.h5");
  type Answer = { ok: boolean; error?: string; meta?: { stages: unknown[] }; nodal?: Record<string, unknown>; cancelled?: boolean };
  const said = (a: Answer) => ({ ok: a.ok, error: a.error, stages: a.meta?.stages.length, nodal: a.nodal ? Object.keys(a.nodal).sort() : undefined, cancelled: a.cancelled });
  const step = (seq: number) => call("results:readStep", { path: results, stage: "stage_2", step: 1, seq }) as Promise<Answer>;
  const scan = (scanId: number) => call("results:scan", { path: results, stage: "stage_2", components: ["displacement_x"], magnitude: false, scanId }) as Promise<Answer>;
  out["open"] = said((await call("results:open", results)) as Answer);
  out["step"] = said(await step(1));
  // A cancel reaches the loader while its scan runs.
  const cancelled = scan(7);
  await until(() => progress.some((p) => p.scanId === 7), "scan 7's first progress");
  await call("results:scanCancel", 7);
  out["cancelled"] = said(await cancelled);
  out["progress"] = progress.filter((p) => p.scanId === 7);
  // A step read is served while a scan runs.
  const running = scan(8);
  await until(() => progress.some((p) => p.scanId === 8), "scan 8's first progress");
  out["stepDuringScan"] = said(await step(2));
  // The loader stops during the scan: the scan is refused with why, and the next step read finds its file opened again.
  const forks = forked.length;
  forked.at(-1)!.kill();
  out["stoppedScan"] = said(await running);
  out["stepAfterStop"] = said(await step(3));
  out["newForks"] = forked.length - forks;
} else if (which === "unreadable") {
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
