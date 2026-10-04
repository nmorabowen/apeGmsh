// Watching an open set (src/main/watch.ts) on a real directory: a change is
// reported once, only after the writer has stopped for 300 ms; a sibling that
// appears or vanishes changes the set.

import assert from "node:assert/strict";
import { appendFileSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { OpenSet } from "../../src/main/pairing.ts";
import { SetWatcher } from "../../src/main/watch.ts";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Heard = { at: number; changed?: string; reopened?: OpenSet; error?: string };

function harness(opened: string) {
  const heard: Heard[] = [];
  const watcher = new SetWatcher(opened, {
    changed: (path) => heard.push({ at: Date.now(), changed: path }),
    reopened: (set) => heard.push({ at: Date.now(), reopened: set }),
    error: (message) => heard.push({ at: Date.now(), error: message }),
  }).start();
  return { heard, watcher };
}

async function until(cond: () => boolean, ms = 4000): Promise<void> {
  const end = Date.now() + ms;
  while (!cond()) {
    if (Date.now() > end) throw new Error("timed out");
    await sleep(25);
  }
}

test("a slow rewrite is reported once, at least 300 ms after the last write", async () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-watch-"));
  const model = join(dir, "m.h5");
  writeFileSync(model, "v1");
  const { heard, watcher } = harness(model);
  try {
    assert.deepEqual(watcher.set, { model, geometry: null, results: null });
    await sleep(100);
    // A writer that truncates, then writes in chunks 100 ms apart for 600 ms.
    writeFileSync(model, "");
    let lastWrite = Date.now();
    for (let i = 0; i < 6; i++) {
      await sleep(100);
      appendFileSync(model, `chunk ${i};`);
      lastWrite = Date.now();
    }
    await until(() => heard.length > 0);
    await sleep(600); // nothing more may arrive
    assert.equal(heard.length, 1, JSON.stringify(heard));
    assert.equal(heard[0]!.changed, model);
    assert.ok(heard[0]!.at - lastWrite >= 300, `reported ${heard[0]!.at - lastWrite} ms after the last write`);
  } finally {
    watcher.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a sibling that appears, then vanishes, changes the set", async () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-watch-"));
  const model = join(dir, "m.h5");
  const geometry = join(dir, "m.geometry.h5");
  writeFileSync(model, "model");
  writeFileSync(join(dir, "other.geometry.h5"), "x");
  const { heard, watcher } = harness(model);
  try {
    await sleep(100);
    writeFileSync(geometry, "geometry");
    await until(() => heard.length > 0);
    assert.deepEqual(heard[0]!.reopened, { model, geometry, results: null });
    assert.deepEqual(watcher.set, { model, geometry, results: null });

    rmSync(geometry);
    await until(() => heard.length > 1);
    assert.deepEqual(heard[1]!.reopened, { model, geometry: null, results: null });

    // A file of another stem is not watched.
    writeFileSync(join(dir, "other.geometry.h5"), "changed");
    await sleep(700);
    assert.equal(heard.length, 2, JSON.stringify(heard));
  } finally {
    watcher.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a rewritten geometry of an unchanged set is a fileChanged, not a reopen", async () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-watch-"));
  const model = join(dir, "m.h5");
  const geometry = join(dir, "m.geometry.h5");
  writeFileSync(model, "model");
  writeFileSync(geometry, "g1");
  const { heard, watcher } = harness(geometry); // opened from the geometry
  try {
    assert.deepEqual(watcher.set, { model, geometry, results: null });
    await sleep(100);
    writeFileSync(geometry, "g2, longer");
    await until(() => heard.length > 0);
    await sleep(500);
    assert.deepEqual(
      heard.map((h) => h.changed),
      [geometry],
    );
  } finally {
    watcher.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

// Fable, #1310 finding 1: the watcher compared the lower-case candidate with
// the opened spelling, so a rewrite of a file opened as `M.RESULTS.H5` was
// never reported.
test("a file opened under an upper-case suffix is reported when rewritten", async () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-watch-"));
  const model = join(dir, "M.h5");
  const results = join(dir, "M.RESULTS.H5");
  writeFileSync(model, "model");
  writeFileSync(results, "r1");
  const { heard, watcher } = harness(results);
  try {
    assert.deepEqual(watcher.set, { model, geometry: null, results });
    await sleep(100);
    writeFileSync(results, "r2, longer");
    await until(() => heard.length > 0);
    await sleep(500);
    assert.deepEqual(heard.map((h) => h.changed ?? h), [results]);
  } finally {
    watcher.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
