// The open set of the view window (src/main/session.ts), with a fake disk
// and a fake watcher that records whether it was closed.

import assert from "node:assert/strict";
import { join, resolve } from "node:path";
import { test } from "node:test";
import type { OpenSet } from "../../src/main/pairing.ts";
import { OpenSession, type SessionWatcher } from "../../src/main/session.ts";
import type { WatchSink } from "../../src/main/watch.ts";

const dir = resolve("/runs");
const p = (name: string) => join(dir, name);

function harness(...files: string[]) {
  const disk = new Set(files.map(p));
  const watchers: { opened: string; closed: boolean; sink: WatchSink }[] = [];
  const log: string[] = [];
  const session = new OpenSession({
    exists: (path) => disk.has(path),
    watch: (opened, sink): SessionWatcher => {
      const w = { opened, closed: false, sink, set: { model: null, geometry: null, results: null } as OpenSet, close: () => (w.closed = true) };
      watchers.push(w);
      return w;
    },
    deliver: (channel, payload) => log.push(`${channel} ${JSON.stringify(payload)}`),
    fail: (m) => log.push(`fail ${m}`),
    note: (m) => log.push(`note ${m}`),
  });
  return { session, watchers, log };
}

test("open pairs the file, watches its set and, when asked, tells the renderer", () => {
  const { session, watchers, log } = harness("a.h5", "a.geometry.h5");
  assert.ok(session.open(p("a.h5"), true));
  assert.deepEqual(session.set, { model: p("a.h5"), geometry: p("a.geometry.h5"), results: null });
  assert.equal(watchers.length, 1);
  assert.equal(log.length, 1);
  assert.match(log[0]!, /^open /);
});

test("a file that cannot be paired fails loudly and keeps the open set", () => {
  const { session, watchers, log } = harness("a.h5");
  session.open(p("a.h5"), false);
  assert.equal(session.open(p("b.hdf5"), true), false);
  assert.match(log.join("\n"), /fail apeGmshViewer opens <stem>\.h5/);
  assert.equal(session.set?.model, p("a.h5"));
  assert.equal(watchers[0]!.closed, false);
});

// Fable, #1310 finding 2: the renderer reads b.hdf5 itself while a.h5 is
// open. The pairing error was swallowed and the watcher stayed on a's set, so
// a rewrite of a.h5 reloaded the page onto a.
test("a renderer-opened file that cannot be paired ends the old set", () => {
  const { session, watchers, log } = harness("a.h5", "b.hdf5");
  session.open(p("a.h5"), false);
  session.rendererOpened(p("b.hdf5"));
  assert.equal(session.set, null);
  assert.equal(watchers[0]!.closed, true);
  assert.match(log.join("\n"), /note not watching .*b\.hdf5/);
  // A late report from the closed watcher reaches nobody.
  watchers[0]!.sink.changed(p("a.h5"));
  watchers[0]!.sink.reopened({ model: p("a.h5"), geometry: null, results: null });
  assert.equal(log.filter((l) => !l.startsWith("note")).length, 0);
});

test("a renderer-opened model of another stem becomes the open set", () => {
  const { session, watchers } = harness("a.h5", "b.h5", "b.geometry.h5");
  session.open(p("a.h5"), false);
  session.rendererOpened(p("b.h5"));
  assert.deepEqual(session.set, { model: p("b.h5"), geometry: p("b.geometry.h5"), results: null });
  assert.equal(watchers[0]!.closed, true);
  assert.equal(watchers.length, 2);
});

test("the renderer re-reading the open model changes nothing", () => {
  const { session, watchers } = harness("a.h5");
  session.open(p("a.h5"), false);
  session.rendererOpened(p("a.h5"));
  assert.equal(watchers.length, 1);
  assert.equal(watchers[0]!.closed, false);
});

test("a reopen from the live watcher updates the set and tells the renderer", () => {
  const { session, watchers, log } = harness("a.h5");
  session.open(p("a.h5"), false);
  const next = { model: p("a.h5"), geometry: p("a.geometry.h5"), results: null };
  watchers[0]!.sink.reopened(next);
  assert.deepEqual(session.set, next);
  assert.deepEqual(log, [`open ${JSON.stringify(next)}`]);
});
