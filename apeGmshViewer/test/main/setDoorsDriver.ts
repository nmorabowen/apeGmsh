// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts, and
// calls its `results:*` and `geometry:open` handlers with a fixed battery of
// requests over the sets in `--dir=` (test/main/setDoors.test.ts, VU12 #2753).
// It runs in a child process, because main.ts reads its mode from argv and
// watches the set it opens; it prints one JSON line and exits.
//
//   node test/main/setDoorsDriver.ts --case=<view|none|render|capture> --dir=<root> [main's own --mode= --file= --spec= --out=]
//
// `<root>` holds `set/` (the set the window opens), `outside/` and `elsewhere/`.

import { registerHooks } from "node:module";
import { join, sep } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});

const opt = (name: string) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const root = opt("dir");
const which = opt("case");
if (!root || !which) throw new Error("setDoorsDriver needs --case= and --dir=");

const { handlers } = (await import(STUB)) as typeof import("./electronStub.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const set = join(root, "set");
const elsewhere = join(root, "elsewhere");
/** Readable files a renderer could name, none of them the set's: a sibling, a `..` path out of the set's folder (unnormalised on purpose), an absolute path elsewhere. */
const RESULTS = {
  ladruno: join(set, "s.ladruno"),
  native: join(set, "s.results.h5"),
  traversal: `${set}${sep}..${sep}outside${sep}s.ladruno`,
  elsewhere: join(elsewhere, "s.ladruno"),
};
const GEOMETRY = {
  paired: join(set, "s.geometry.h5"),
  sibling: join(set, "t.geometry.h5"),
  traversal: `${set}${sep}..${sep}outside${sep}s.geometry.h5`,
  elsewhere: join(elsewhere, "s.geometry.h5"),
};

/** The parts of an answer the test judges; the arrays stay in this process. */
type Seen = { ok: boolean; error?: string; seq?: number; scanId?: number; stages?: number; geometry?: boolean };
const out: Record<string, { resolved: Seen } | { rejected: string }> = {};
const event = { sender: { isDestroyed: () => false, send: () => undefined } };

async function call(name: string, channel: string, ...args: unknown[]): Promise<Seen | null> {
  const handle = handlers.get(channel);
  if (!handle) throw new Error(`main.ts registered no ${channel}`);
  try {
    const r = (await handle(event, ...args)) as Seen & { meta?: { stages: unknown[] }; geometry?: unknown; reason?: string };
    const seen: Seen = { ok: r.ok, error: r.error ?? r.reason, seq: r.seq, scanId: r.scanId, stages: r.meta?.stages.length, geometry: r.geometry === undefined ? undefined : r.geometry !== null };
    out[name] = { resolved: seen };
    return seen;
  } catch (err) {
    out[name] = { rejected: err instanceof Error ? err.message : String(err) };
    return null;
  }
}

/** The first stage and nodal component of `path`, read by the host after a `results:open` (a step and a scan need them). */
let probe: { stage: string; component: string } | null = null;
async function openFor(name: string, path: string): Promise<void> {
  const r = (await handlers.get("results:open")!(event, path)) as { ok: boolean; error?: string; meta?: { stages: { id: string; nodal: string[] }[] } };
  out[name] = { resolved: { ok: r.ok, error: r.error, stages: r.meta?.stages.length } };
  const st = r.meta?.stages.find((s) => s.nodal.length > 0);
  if (st) probe = { stage: st.id, component: st.nodal[0]! };
}

const step = (path: unknown, seq: number) => ({ path, stage: probe?.stage ?? "?", step: 0, seq });
const scan = (path: unknown, scanId: number) => ({ path, stage: probe?.stage ?? "?", components: [probe?.component ?? "?"], magnitude: false, scanId });

const circular: Record<string, unknown> = { path: 7 };
circular.self = circular;

/** Refused opens first, so the host still indexes the set's own file for the step and scan battery. */
async function battery(own: string, refused: Record<string, string>, geometryOwn: string, scanOwn = true): Promise<void> {
  for (const [k, p] of Object.entries(refused)) await call(`results:open ${k}`, "results:open", p);
  for (const [k, p] of Object.entries(GEOMETRY)) if (p !== geometryOwn) await call(`geometry:open ${k}`, "geometry:open", p);
  await call("geometry:open own", "geometry:open", geometryOwn);
  await openFor("results:open own", own);
  let n = 0;
  for (const [k, p] of Object.entries(refused)) {
    await call(`results:readStep ${k}`, "results:readStep", step(p, ++n));
    await call(`results:scan ${k}`, "results:scan", scan(p, n));
  }
  await call("results:readStep own", "results:readStep", step(own, 100));
  if (scanOwn) await call("results:scan own", "results:scan", scan(own, 100));
}

if (which === "view") {
  // opened by `--file=set/s.h5`, as a double-click opens it: its results slot is the `.ladruno`
  const { ladruno, ...refused } = RESULTS;
  await battery(ladruno, refused, GEOMETRY.paired);
  // requests a JSON log line cannot encode: each must resolve with the shape it needed
  for (const door of ["results:open", "geometry:open"]) {
    await call(`${door} bigint`, door, 5n);
    await call(`${door} circular`, door, circular);
  }
  await call("results:readStep bigint", "results:readStep", step(5n, 1));
  await call("results:readStep circular", "results:readStep", circular);
  await call("results:scan bigint", "results:scan", scan(5n, 1));
  await call("results:scan circular", "results:scan", circular);
  // A drop moves the set: the old set's results, still indexed by the host, are refused; the new set's read.
  await call("drop:1 requestOpen", "viewer:requestOpen", join(elsewhere, "s.h5"));
  await call("drop:2 readStep old", "results:readStep", step(ladruno, 200));
  await call("drop:2 scan old", "results:scan", scan(ladruno, 200));
  await call("drop:3 open old", "results:open", ladruno);
  await call("drop:3 geometry old", "geometry:open", GEOMETRY.paired);
  await call("drop:4 open new", "results:open", RESULTS.elsewhere);
  await call("drop:4 readStep new", "results:readStep", step(RESULTS.elsewhere, 201));
  await call("drop:4 geometry new", "geometry:open", GEOMETRY.elsewhere);
  // #1310 F2 (VU11's rule): a results file read as a model ends the set, so nothing is open for these doors.
  await call("model:1 open a results file", "model:open", RESULTS.elsewhere);
  await call("model:2 open new", "results:open", RESULTS.elsewhere);
  await call("model:2 geometry new", "geometry:open", GEOMETRY.elsewhere);
} else if (which === "none") {
  await call("results:open ladruno", "results:open", RESULTS.ladruno);
  await call("geometry:open paired", "geometry:open", GEOMETRY.paired);
} else if (which === "render") {
  // the spec names the native `.results.h5`, so it is the job's results file and the `.ladruno` beside it is not
  const { native, ...refused } = RESULTS;
  await battery(native, refused, GEOMETRY.paired);
} else if (which === "capture") {
  // a capture paces its scan 2 s a step, so only the refusals are scanned here
  const { ladruno, ...refused } = RESULTS;
  await battery(ladruno, refused, GEOMETRY.paired, false);
} else throw new Error(`setDoorsDriver: unknown --case=${which}`);

process.stdout.write(`SET_DOORS ${JSON.stringify(out)}\n`);
process.exit(0);
