// ADR 0128 D5, D6: an agent's `--present` and `--snapshot` (VH3, #2868: moved
// out of ../main.ts unchanged). The sending side, `reachWindow`, runs at
// main's top level before any window; the window's side, `heardFromAgent`,
// is what main.ts's `second-instance` handler hands an agent's call to.
// `registerReach` registers `viewer:offersReady` and `viewer:snapshotReply`,
// each once. The modes and their arguments are described at the top of
// ../main.ts.
//
// INV-5: an agent's call lists or reads; nothing here opens a set or moves
// the window, and test/offers.test.ts reads this file to hold that.
//
// What it needs of main.ts comes through `registerReach` alone, so this
// module never imports main.ts. main.ts's file doors read the listed offers
// (`listedOffers`), and its page events call `offersPageLoading` and
// `pageCommitted`.

import { app, ipcMain, type BrowserWindow, type IpcMainEvent, type WebContents } from "electron";
import { randomUUID } from "node:crypto";
import { connect, createServer } from "node:net";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { resolveSpecFiles } from "../../spec/files.ts";
import { outRefusal } from "../../spec/job.ts";
import { MANIFEST_FILE, refusedManifest } from "../../spec/manifest.ts";
import { presentDocument, SNAPSHOT_PNG, SNAPSHOT_VIEW, type OfferPayload, type PresentedDocument } from "../../spec/snapshot.ts";
import type { OfferIn } from "../../state/types.ts";
import { arg, mode, out } from "../args.ts";
import { FOLD_CASE } from "../pairing.ts";
import { jobFs, stamped, writeManifestLast, type RunManifest } from "./out.ts";

/** What `--present` and `--snapshot` hand the running window (the lock's additional data); `ack` is the sender's private pipe. */
export type Reach = { kind: "present"; spec: string; ack: string } | { kind: "snapshot"; out: string; ack: string };
/** The window's answer on that pipe. */
type Ack = { ok: true } | { ok: false; code: number; reason: string };

/** What the window's side takes from main.ts, handed over once (`registerReach`). */
export interface ReachHost {
  /** The app's window, or null before it is made and after it closes (main.ts's `win`). */
  window: () => BrowserWindow | null;
  /** main.ts's log line of whatever the IPC carried. */
  shown: (req: unknown) => string;
}

let reachHost: ReachHost | null = null;

/** The registered host; an agent's call heard before `registerReach` ran is a bug in main.ts, and says so. */
function host(): ReachHost {
  if (reachHost === null) throw new Error("reach: registerReach has not run");
  return reachHost;
}

/** A spec file as offers, or as refused rows; never a throw (it runs in the human's window). */
function presentOffers(specPath: string): PresentedDocument {
  return presentDocument(() => readFileSync(specPath, "utf8"), (spec) => resolveSpecFiles(spec, dirname(specPath), existsSync), FOLD_CASE);
}

/** How long the sender waits for the window to acknowledge: a window's main process answers at once, so silence means no window took it. */
const REACH_ACK_MS = 5000;

/** The sender's end of the acknowledgement, listening before the window is asked. */
function ackPipe(): Promise<{ name: string; answer: Promise<Ack | null> }> {
  const id = `apegmsh-viewer-${randomUUID()}`;
  const name = process.platform === "win32" ? `\\\\.\\pipe\\${id}` : join(tmpdir(), `${id}.sock`);
  return new Promise((listening, failed) => {
    let settle: (a: Ack | null) => void = () => {};
    const answer = new Promise<Ack | null>((r) => (settle = r));
    setTimeout(() => settle(null), REACH_ACK_MS).unref();
    const server = createServer((socket) => {
      let text = "";
      socket.setEncoding("utf8");
      socket.on("data", (d) => (text += String(d)));
      socket.on("end", () => settle(JSON.parse(text) as Ack));
      socket.on("error", () => settle(null));
    });
    server.on("error", failed);
    server.listen(name, () => listening({ name, answer }));
  });
}

/** Tell the sender what became of its call. A sender that has gone is no matter for the human's window. */
function sendAck(pipe: string, ack: Ack): void {
  const c = connect(pipe, () => c.end(JSON.stringify(ack)));
  c.on("error", (err) => process.stderr.write(`apeGmshViewer: acknowledging an agent's call: ${err.message}\n`));
}

/** How long `--snapshot` waits for the window's manifest. */
const SNAPSHOT_WAIT_MS = 60000;

/** Wait for the manifest the running window writes last; exit with what it says. */
function awaitSnapshot(dir: string): Promise<never> {
  const path = join(dir, MANIFEST_FILE);
  const t = Date.now();
  return new Promise<never>(() => {
    const timer = setInterval(() => {
      if (!existsSync(path)) {
        if (Date.now() - t < SNAPSHOT_WAIT_MS) return;
        clearInterval(timer);
        process.stderr.write(`apeGmshViewer: snapshot: the window did not answer within ${SNAPSHOT_WAIT_MS / 1000} s\n`);
        app.exit(1);
        return;
      }
      clearInterval(timer);
      const m = JSON.parse(readFileSync(path, "utf8")) as { refused?: { reason: string } };
      if (m.refused) process.stderr.write(`apeGmshViewer: snapshot refused: ${m.refused.reason}\n`);
      else process.stdout.write(`apeGmshViewer: snapshot: ${path}\n`);
      app.exit(m.refused ? 1 : 0);
    }, 50);
  });
}

/**
 * ADR 0128 D5, D6 startup: check what is asked here (exit 2, nothing sent),
 * then reach the running window through the lock. Holding the lock means no
 * window is running: let go of it, say so, exit 1, and open nothing. Denied the
 * lock proves nothing (another agent call may hold it for a moment): only the
 * window's acknowledgement is an exit 0. The module's top level awaits this,
 * and Electron holds `ready` until it ends, so Chromium never starts here and
 * nothing is written beside the human's profile (INV-1).
 */
export async function reachWindow(): Promise<void> {
  const refuse = (code: number, message: string) => {
    process.stderr.write(`apeGmshViewer: ${message}\n`);
    app.exit(code);
  };
  try {
    let what: { kind: "present"; spec: string } | { kind: "snapshot"; out: string };
    if (mode === "present") {
      const spec = arg("spec");
      if (!spec) return refuse(2, "--mode=present needs --spec");
      const specPath = resolve(spec);
      const p = presentOffers(specPath);
      if (p.refused) return refuse(2, `the spec ${specPath} is refused: ${p.refused.key}: ${p.refused.reason}`);
      what = { kind: "present", spec: specPath };
    } else {
      if (!out) return refuse(2, "--mode=snapshot needs --out");
      const busy = outRefusal(resolve(out), jobFs);
      if (busy) return refuse(2, `--out refused: ${busy}`);
      what = { kind: "snapshot", out: resolve(out) };
    }
    const ack = await ackPipe();
    if (app.requestSingleInstanceLock({ ...what, ack: ack.name })) {
      app.releaseSingleInstanceLock();
      return refuse(1, "no apeGmshViewer window is running; none was opened. Ask the human to open one.");
    }
    const said = await ack.answer;
    if (said === null) return refuse(1, `no apeGmshViewer window acknowledged within ${REACH_ACK_MS / 1000} s; none was opened. Ask the human to open one.`);
    if (!said.ok) return refuse(said.code, said.reason);
    if (what.kind === "present") {
      process.stdout.write(`apeGmshViewer: present: the window took ${what.spec}\n`);
      app.exit(0);
      return;
    }
    await awaitSnapshot(what.out);
  } catch (err) {
    refuse(2, `${mode} failed to start: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ---------------------------------------------------------------------------
// ADR 0128 D5, D6 in the running window. Offers wait until the page listens;
// a page load never follows from one (INV-5). A snapshot captures the page as
// it is, then asks the page for the spec and the manifest entry of that view;
// main writes every file, into the `--out` the caller named (INV-1), the
// manifest last.
// ---------------------------------------------------------------------------

let offersReady = false;
/** One offer row: the files `viewer:offerOpen` may open for it, and each set the window left when it opened. */
export type OfferRow = { files: string[]; left: string[][] };
/** The batches the page has not heard yet, each with its rows of `offerRows` (VU14: a page load keeps only these). */
let offersWaiting: { offers: OfferIn[]; rows: OfferRow[] }[] = [];
/**
 * VU12 (#2753): the files `viewer:offerOpen` may open, all chosen by main, one
 * row per offer row main sent in `viewer:offers`: the offer's model and results
 * (none for a refused row), and each set the window left when that offer
 * opened (its Back reopens that set). VU13 (#2760): the rows mirror the page's
 * list (state/reduce.ts `offerReceived`: newest first, at most `MAX_OFFERS`),
 * so an offer that falls off the page's list is gone here too, and its files
 * and the sets it left are no longer offerable. VU14 (#2771): a page load
 * starts an empty list, so it keeps only the rows of batches still waiting to
 * be sent (`offersPageLoading`).
 */
let offerRows: OfferRow[] = [];
/** state/reduce.ts `MAX_OFFERS`, the page's longest offer list (test/main/setDoors.test.ts holds the two equal). */
const PAGE_OFFERS = 8;
let snapshotSeq = 0;
const snapshotReplies = new Map<number, (reply: SnapshotReply) => void>();
type SnapshotReply = { ok: true; view: unknown; manifest: RunManifest } | { ok: false; reason: string };
/** How long the page has to answer a snapshot. */
const SNAPSHOT_REPLY_MS = 10000;

/** The offer rows listed now, for main.ts's `viewer:offerOpen`; a row's `left` grows when its offer opens. */
export function listedOffers(): readonly OfferRow[] {
  return offerRows;
}

/** Nothing an agent sends may raise a modal in the human's window: every failure here goes to stderr and to the sender. */
export function heardFromAgent(reach: Reach): void {
  try {
    const h = host();
    if (reach.kind === "present") {
      const p = presentOffers(reach.spec);
      const rows = p.offers.map((o): OfferRow => {
        if (o.refused !== undefined) return { files: [], left: [] }; // a refused row is listed, never applied, and carries no files
        const { model, results } = (o.payload as OfferPayload).files;
        return { files: [model, ...(results === null ? [] : [results])], left: [] };
      });
      offerRows = [...rows, ...offerRows].slice(0, PAGE_OFFERS);
      const win = h.window();
      if (win && offersReady) win.webContents.send("viewer:offers", p.offers);
      else offersWaiting.push({ offers: p.offers, rows });
      // The sender checked this file a moment ago. Changed since, it is listed as refused, and the sender hears why.
      sendAck(reach.ack, p.refused ? { ok: false, code: 2, reason: `the spec ${reach.spec} is refused: ${p.refused.key}: ${p.refused.reason}` } : { ok: true });
      return;
    }
    if (reach.kind === "snapshot") {
      sendAck(reach.ack, { ok: true });
      void writeSnapshot(reach.out).catch((err: unknown) => {
        const reason = err instanceof Error ? err.message : String(err);
        process.stderr.write(`apeGmshViewer: snapshot: ${reason}\n`);
        try {
          writeManifestLast(reach.out, refusedManifest("--snapshot", "snapshot", reason, { version: app.getVersion(), version_reason: null }, new Date().toISOString()));
        } catch {
          // No folder to write in: the sender's wait ends non-zero.
        }
      });
      return;
    }
    const unknown: never = reach;
    process.stderr.write(`apeGmshViewer: an unknown agent call reached the window: ${h.shown(unknown)}\n`);
  } catch (err) {
    process.stderr.write(`apeGmshViewer: an agent's call failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
  }
}

/**
 * VU15 (#2776): the page that may say it listens for offers, as Electron names
 * a sender: the window's main frame as the current load committed it
 * (`did-navigate`). A load (`did-start-loading`, or a new document's
 * `did-start-navigation` in the main frame) clears it until the new page
 * commits, because until then the page on screen is the one being replaced;
 * a page that commits in a new frame has a new process or frame id. So a late
 * `viewer:offersReady` from a page being replaced re-arms nothing.
 */
let pageFrame: { processId: number; routingId: number } | null = null;

export function pageCommitted(contents: WebContents): void {
  const f = contents.mainFrame;
  pageFrame = { processId: f.processId, routingId: f.routingId };
}

/** Whether `e` comes from the current load of the window's page (VU15). */
function fromCurrentPage(e: IpcMainEvent): boolean {
  const win = host().window();
  return win !== null && e.sender === win.webContents && pageFrame !== null && e.processId === pageFrame.processId && e.frameId === pageFrame.routingId;
}

/** A page load, as main.ts's `pageLoading` sees one: the page is not listening, and every offer row it will not be sent goes (VU14). */
export function offersPageLoading(): void {
  offersReady = false;
  pageFrame = null;
  const waiting = new Set(offersWaiting.flatMap((b) => b.rows));
  offerRows = offerRows.filter((r) => waiting.has(r));
}

async function writeSnapshot(dir: string): Promise<void> {
  mkdirSync(dir, { recursive: true });
  const refuse = (reason: string) =>
    writeManifestLast(dir, refusedManifest("--snapshot", "snapshot", reason, { version: app.getVersion(), version_reason: null }, new Date().toISOString()));
  const win = host().window();
  if (!win || !offersReady) return refuse("the window's page is not ready");
  const contents = win.webContents;
  const img = await contents.capturePage();
  const size = img.getSize();
  if (img.isEmpty() || size.width === 0) return refuse("the window could not be captured (it may be minimized); it was left as it is");
  const id = ++snapshotSeq;
  const reply = await new Promise<SnapshotReply>((done) => {
    const timer = setTimeout(() => {
      snapshotReplies.delete(id);
      done({ ok: false, reason: `the page did not answer within ${SNAPSHOT_REPLY_MS / 1000} s` });
    }, SNAPSHOT_REPLY_MS);
    snapshotReplies.set(id, (r) => {
      clearTimeout(timer);
      done(r);
    });
    contents.send("viewer:snapshot", { id, image: [size.width, size.height] });
  });
  if (!reply.ok) return refuse(reply.reason);
  writeFileSync(join(dir, SNAPSHOT_PNG), img.toPNG());
  writeFileSync(join(dir, SNAPSHOT_VIEW), JSON.stringify(reply.view, null, 2));
  writeManifestLast(dir, JSON.stringify(stamped(reply.manifest), null, 2));
}

/** Register `viewer:offersReady` and `viewer:snapshotReply`, each once. */
export function registerReach(h: ReachHost): void {
  if (reachHost !== null) throw new Error("reach: registerReach runs once");
  reachHost = h;
  const { shown } = h;

  ipcMain.on("viewer:offersReady", (e: IpcMainEvent) => {
    if (!fromCurrentPage(e)) {
      process.stderr.write(`apeGmshViewer: viewer:offersReady from a page that is not the window's current load (process ${shown(e.processId)}, frame ${shown(e.frameId)}) is ignored\n`);
      return;
    }
    offersReady = true;
    for (const { offers } of offersWaiting) h.window()?.webContents.send("viewer:offers", offers);
    offersWaiting = [];
  });

  ipcMain.handle("viewer:snapshotReply", (_e, id: unknown, reply: unknown) => {
    const done = typeof id === "number" ? snapshotReplies.get(id) : undefined;
    if (!done) throw new Error(`viewer:snapshotReply: no snapshot ${shown(id)} is waiting`);
    snapshotReplies.delete(id as number);
    done(reply as SnapshotReply);
  });
}
