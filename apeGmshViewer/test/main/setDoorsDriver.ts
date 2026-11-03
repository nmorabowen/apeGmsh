// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts, and
// calls its `results:*` and `geometry:open` handlers with a fixed battery of
// requests over the sets in `--dir=` (test/main/setDoors.test.ts, VU12 #2753).
// It runs in a child process, because main.ts reads its mode from argv and
// watches the set it opens; it prints one JSON line and exits.
//
//   node test/main/setDoorsDriver.ts --case=<view|reload|stale|none|render|capture|describe|measure|source|source-none> --dir=<root> [main's own --mode= --file= --spec= --out=]
//
// `<root>` holds `set/` (the set the window opens), `outside/` and `elsewhere/`;
// the `source` cases (VU13 #2760) also `src-set/`, `lib/` and the editor's log.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
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

const { appEvents, commitFrame, handlers, listeners, pageEvent, pageEvents, pageSent, ready } = (await import(STUB)) as typeof import("./electronStub.ts");
type StubFrame = import("./electronStub.ts").StubFrame;
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
const out: Record<string, { resolved: Seen } | { rejected: string } | { onScreen: string | null } | { sent: number }> = {};
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
  // the open doors' error lines are total too (they open nothing for a path that is not a string)
  await call("viewer:requestOpen circular", "viewer:requestOpen", circular);
  await call("viewer:offerOpen bigint", "viewer:offerOpen", 5n, []);
  // offerOpen opens only what main offered, or the set an offer left (its Back).
  const modelOf = async () => ((await handlers.get("viewer:currentSet")!(event)) as { model: string | null } | null)?.model ?? null;
  const outside = join(root, "outside", "s.ladruno");
  await call("offer:1 un-offered", "viewer:offerOpen", outside, [outside]);
  out["offer:1 model"] = { onScreen: await modelOf() };
  // a second instance's `--present` of a spec over the elsewhere set (main.ts `heardFromAgent`; its ack pipe has no listener)
  for (const fn of appEvents.get("second-instance") ?? []) fn({}, [], root, { kind: "present", spec: join(root, "present.json"), ack: join(root, "no-such-pipe") });
  await call("offer:2 empty needs", "viewer:offerOpen", RESULTS.elsewhere, []);
  await call("offer:3 un-offered need", "viewer:offerOpen", RESULTS.elsewhere, [join(elsewhere, "s.h5"), outside]);
  out["offer:3 model"] = { onScreen: await modelOf() };
  await call("offer:4 offered", "viewer:offerOpen", RESULTS.elsewhere, [join(elsewhere, "s.h5"), RESULTS.elsewhere]);
  out["offer:4 model"] = { onScreen: await modelOf() };
  await call("offer:5 back", "viewer:offerOpen", ladruno, [join(set, "s.h5"), ladruno]);
  out["offer:5 model"] = { onScreen: await modelOf() };
  // VU12, owner ruling "option A": the page names no path to open. A free string is refused and the set stays;
  // the argv set's own paths (main sent them) still open.
  const loose = join(root, "outside", "s.h5");
  await call("gesture:1 requestOpen free", "viewer:requestOpen", loose);
  out["gesture:1 model"] = { onScreen: await modelOf() };
  await call("gesture:2 model:open free", "model:open", loose);
  out["gesture:2 model"] = { onScreen: await modelOf() };
  await call("gesture:3 requestOpen argv", "viewer:requestOpen", join(set, "s.h5"));
  await call("gesture:3 model:open argv", "model:open", join(set, "s.h5"));
  out["gesture:3 model"] = { onScreen: await modelOf() };
  /** A drop as preload.ts reports it: Electron's path for the File, sent synchronously on `viewer:dropped`. */
  const drop = (path: string) => listeners.get("viewer:dropped")!({ returnValue: undefined }, path);
  // A drop moves the set: the old set's results, still indexed by the host, are refused; the new set's read.
  drop(join(elsewhere, "s.h5"));
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
  // a model the user dropped onto a page without requestOpen (effects.ts `openDropped`'s fallback) opens, and the set follows it
  drop(loose);
  await call("model:3 model:open dropped", "model:open", loose);
  out["model:3 model"] = { onScreen: await modelOf() };
  // VU13 (#2760): only the latest drop is a gesture. elsewhere/s.h5 was dropped (drop:1), then outside/s.h5:
  // a gesture-free model:open of the replaced drop is refused.
  await call("drop:5 model:open replaced drop", "model:open", join(elsewhere, "s.h5"));
  out["drop:5 model"] = { onScreen: await modelOf() };
  // An unpaired drop (a model read from a file that is no set's model) ends the set, and its re-read rests on the
  // drop alone: the drop is not spent by its first open.
  const unpaired = join(root, "outside", "loose.user.h5");
  drop(unpaired);
  await call("drop:6 model:open unpaired drop", "model:open", unpaired);
  out["drop:6 model"] = { onScreen: await modelOf() };
  await call("drop:7 model:open unpaired drop again", "model:open", unpaired);
  // VU13: an offer that falls off the page's list (state/reduce.ts MAX_OFFERS) is gone, with the set it left.
  const { MAX_OFFERS } = await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "state", "reduce.ts")).href);
  const present = () => {
    for (const fn of appEvents.get("second-instance") ?? []) fn({}, [], root, { kind: "present", spec: join(root, "flood.json"), ack: join(root, "no-such-pipe") });
  };
  for (let i = 1; i < MAX_OFFERS; i++) present();
  const offered = [join(elsewhere, "s.h5"), RESULTS.elsewhere];
  await call("prune:1 still listed", "viewer:offerOpen", RESULTS.elsewhere, offered);
  present();
  await call("prune:2 fallen off", "viewer:offerOpen", RESULTS.elsewhere, offered);
  out["prune:2 model"] = { onScreen: await modelOf() };
  await call("prune:3 back to the set it left", "viewer:offerOpen", ladruno, [join(set, "s.h5"), ladruno]);
} else if (which === "reload" || which === "stale") {
  // VU14 (#2771): a page load starts an empty offer list, so an offer the old page listed is no longer openable;
  // an offer still waiting to be sent (the new page will list it) is. main.ts makes its window once the app is ready.
  ready();
  await new Promise((r) => setImmediate(r));
  const deliver = (name: string, details: object = {}) => {
    const fns = pageEvents.get(name) ?? [];
    if (fns.length === 0) throw new Error(`main.ts listens for no ${name}`);
    for (const fn of fns) fn(details);
  };
  /** A navigation starts (Electron's `did-start-navigation` details); a reload while loading starts no `did-start-loading`. */
  const navigate = (isMainFrame: boolean, isSameDocument: boolean) => deliver("did-start-navigation", { isMainFrame, isSameDocument });
  // VU15 (#2776): a load starts (`did-start-loading`) and commits its page in a frame (`did-navigate`);
  // only that frame's offersReady counts. Each load here commits a new frame, as Chromium may.
  let frames = 0;
  let current: StubFrame = { processId: 1, routingId: 0 };
  const load = () => deliver("did-start-loading");
  const commit = (f: StubFrame = { processId: 1, routingId: ++frames }) => {
    current = f;
    commitFrame(f);
    deliver("did-navigate");
    return f;
  };
  const listening = (e: object = pageEvent(current)) => listeners.get("viewer:offersReady")!(e);
  const present = (spec: string) => {
    for (const fn of appEvents.get("second-instance") ?? []) fn({}, [], root, { kind: "present", spec: join(root, spec), ack: join(root, "no-such-pipe") });
  };
  const offersSent = () => pageSent.filter((s) => s.channel === "viewer:offers").length;
  const modelOf = async () => ((await handlers.get("viewer:currentSet")!(event)) as { model: string | null } | null)?.model ?? null;
  const there = [join(elsewhere, "s.h5"), RESULTS.elsewhere];
  if (which === "stale") {
    // Each late or foreign offersReady is refused, and the offer it would have released stays waiting:
    // nothing is sent until the current load's own page listens.
    const old = (load(), commit());
    present("present.json"); // the page has not listened yet: the offer waits
    load(); // the page reloads ...
    listening(pageEvent(old)); // ... and the page being replaced says it listens, before the new one commits
    out["stale:1 replaced page, mid-load"] = { sent: offersSent() };
    commit({ processId: 2, routingId: 7 }); // the new page commits in another process
    listening(pageEvent(old)); // and the old page's message arrives after the commit
    out["stale:2 replaced page, after commit"] = { sent: offersSent() };
    listening(pageEvent({ processId: 2, routingId: 8 })); // another frame of the new process (not the window's page)
    out["stale:3 another frame"] = { sent: offersSent() };
    listening(pageEvent(current, { send: () => undefined })); // another webContents with the frame's ids
    out["stale:4 another webContents"] = { sent: offersSent() };
    listening({}); // no sender at all
    out["stale:5 no sender"] = { sent: offersSent() };
    present("present.json"); // nothing above re-armed the page: a new offer waits too
    out["stale:6 a later offer"] = { sent: offersSent() };
    listening(); // the current load's page listens: both waiting batches go
    out["stale:7 current page"] = { sent: offersSent() };
    // A load whose page commits in the same frame (Chromium may keep it): until the commit, its offersReady is the old page's.
    load();
    present("flood.json");
    listening(pageEvent(current));
    out["stale:9 same frame, mid-load"] = { sent: offersSent() };
    commit(current);
    listening();
    out["stale:10 same frame, committed"] = { sent: offersSent() };
    // VU15 review: navigations that replace no page leave it armed: an in-page one, and one in a subframe.
    navigate(true, true);
    present("present.json");
    out["stale:11 in-page navigation"] = { sent: offersSent() };
    navigate(false, false);
    present("present.json");
    out["stale:12 subframe navigation"] = { sent: offersSent() };
    // A reload while the page is still loading: no did-start-loading, only the main frame's new navigation.
    const loading = current;
    navigate(true, false);
    present("present.json");
    out["stale:13 reload during load"] = { sent: offersSent() };
    listening(pageEvent(loading)); // the page being replaced, still named by its frame until the new one commits
    out["stale:14 reload during load, old page"] = { sent: offersSent() };
    commit();
    listening();
    out["stale:15 reload during load, new page"] = { sent: offersSent() };
    process.stdout.write(`SET_DOORS ${JSON.stringify(out)}\n`);
    process.exit(0);
  }
  load();
  commit();
  listening();
  present("present.json"); // the page listens: sent at once, and listed
  out["reload:0 sent"] = { sent: offersSent() };
  await call("reload:1 listed offer", "viewer:offerOpen", RESULTS.elsewhere, there);
  out["reload:1 model"] = { onScreen: await modelOf() };
  load(); // the page reloads: its list starts empty
  await call("reload:2 offer the old page listed", "viewer:offerOpen", RESULTS.elsewhere, there);
  await call("reload:3 the set that offer left", "viewer:offerOpen", RESULTS.ladruno, [join(set, "s.h5"), RESULTS.ladruno]);
  out["reload:3 model"] = { onScreen: await modelOf() };
  // An offer that arrives while the page loads waits for it, and survives a second load before the page listens.
  const outside = join(root, "outside");
  present("flood.json");
  load();
  out["reload:4 sent"] = { sent: offersSent() };
  commit();
  listening();
  out["reload:5 sent"] = { sent: offersSent() };
  await call("reload:5 waiting offer", "viewer:offerOpen", join(outside, "s.ladruno"), [join(outside, "s.h5"), join(outside, "s.ladruno")]);
  out["reload:5 model"] = { onScreen: await modelOf() };
} else if (which === "source" || which === "source-none") {
  // VU13 (#2760): go-to-source opens only a source file the set's model names in its /provenance.
  // The editor is a recorder (APEGMSH_EDITOR, ./source.test.ts): each launch appends its argv to editor.log.
  const log = join(root, "editor.log");
  const rec = join(root, "rec.mjs");
  writeFileSync(rec, `import { appendFileSync } from "node:fs";\nappendFileSync(${JSON.stringify(log)}, JSON.stringify(process.argv.slice(2)) + "\\n");\n`);
  process.env["APEGMSH_EDITOR"] = `"${process.execPath}" "${rec}" {line} {file}`;
  const posix = (p: string) => p.replaceAll("\\", "/");
  const srcSet = join(root, "src-set");
  await call("source:1 relative row", "source:goto", posix(join(srcSet, "build.py")), 3);
  await call("source:2 absolute row outside the folder", "source:goto", posix(join(root, "lib", "helper.py")), 4);
  await call("source:3 stranger in the folder", "source:goto", posix(join(srcSet, "stranger.py")), 1);
  await call("source:4 stranger elsewhere", "source:goto", join(root, "stranger.py"), 1);
  await call("source:5 bigint path", "source:goto", 5n, 1);
  await call("source:6 circular path", "source:goto", circular, 1);
  await call("source:7 bigint line", "source:goto", posix(join(srcSet, "build.py")), 5n);
  // the launches are detached: wait for the ones that opened (none with no set)
  const launches = which === "source" ? 2 : 0;
  const end = Date.now() + 10000;
  const lines = () => (existsSync(log) ? readFileSync(log, "utf8").split("\n").filter(Boolean).length : 0);
  while (lines() < launches && Date.now() < end) await new Promise((r) => setTimeout(r, 50));
  await new Promise((r) => setTimeout(r, 300)); // and any launch that should not have happened
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
} else if (which === "describe" || which === "measure") {
  const { ladruno, ...refused } = RESULTS;
  await battery(ladruno, refused, GEOMETRY.paired);
} else throw new Error(`setDoorsDriver: unknown --case=${which}`);

process.stdout.write(`SET_DOORS ${JSON.stringify(out)}\n`);
process.exit(0);
