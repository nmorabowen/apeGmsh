// Architecture review 2026-11-24, finding 2: two `model:open`s that overlap. The page keeps only
// its latest read (effects.ts `openToken`), so main's open set must follow the latest open, not
// the one that finished last. main.ts runs in a child Node process with `electron` stubbed and
// its loader's model reads held until the test releases them (openOrderDriver.ts, heldLoader.ts).

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";

const FIX = fileURLToPath(new URL("../../fixtures/", import.meta.url));
const DRIVER = fileURLToPath(new URL("./openOrderDriver.ts", import.meta.url));

const root = mkdtempSync(join(tmpdir(), "open-order-"));
after(() => rmSync(root, { recursive: true, force: true }));
for (const d of ["a", "b", "c"]) mkdirSync(join(root, d));
copyFileSync(join(FIX, "ladruno_frame3d.h5"), join(root, "a", "s.h5"));
copyFileSync(join(FIX, "ladruno_frame3d.h5"), join(root, "b", "s.h5"));
copyFileSync(join(FIX, "ladruno_frame3d.ladruno"), join(root, "b", "s.ladruno"));
copyFileSync(join(FIX, "contour_staged.results.h5"), join(root, "c", "s.results.h5"));

const stdout = execFileSync(process.execPath, [DRIVER, `--dir=${root}`, "--mode=view"], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
const line = stdout.split(/\r?\n/).find((l) => l.startsWith("OPEN_ORDER "));
const seen = JSON.parse(line?.slice("OPEN_ORDER ".length) ?? "{}") as Record<string, unknown>;

test("F2: an older model:open that finishes after a newer one leaves main's set on the newer model", () => {
  assert.ok(line, `the driver printed no answers:\n${stdout}`);
  assert.equal(seen["1 second ok"], true);
  assert.equal(seen["1 after second"], join(root, "b", "s.h5"));
  assert.equal(seen["1 first ok"], true, "the older read still answers; the page drops it");
  assert.equal(seen["1 after first"], join(root, "b", "s.h5"), "main follows the last-started open, not the last-finished one");
  assert.deepEqual(seen["1 results:open b"], { ok: true }, "the doors pair against the model on screen");
});

test("F2: an older model:open that fails after a newer one finished leaves the set alone too", () => {
  assert.equal(seen["2 newer ok"], true);
  assert.equal(seen["2 older ok"], false);
  assert.equal(seen["2 after older"], join(root, "a", "s.h5"));
});
