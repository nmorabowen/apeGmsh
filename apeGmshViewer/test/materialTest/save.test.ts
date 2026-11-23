// ADR 0135 D7: main's CSV save. The page sends text and a suggested name; the path
// is only ever what the save dialog returns. The dialog is injected, so this runs
// in Node.

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { MAX_CSV_BYTES, saveCsv } from "../../src/main/materialTest.ts";

const dir = mkdtempSync(join(tmpdir(), "apegmsh-csv-test-"));
after(() => rmSync(dir, { recursive: true, force: true }));

/** A dialog that records what it was offered and answers `path`. */
function dialog(path: string | null) {
  const offered: string[] = [];
  return { offered, choose: async (suggested: string) => (offered.push(suggested), path) };
}

test("the text is written where the dialog says, and the answer names the path", async () => {
  const d = dialog(join(dir, "steel-run1.csv"));
  const a = await saveCsv("step,strain\n0,0\n", "steel-run1.csv", d.choose);
  assert.deepEqual(a, { ok: true, path: join(dir, "steel-run1.csv") });
  assert.deepEqual(d.offered, ["steel-run1.csv"]);
  assert.equal(readFileSync(join(dir, "steel-run1.csv"), "utf8"), "step,strain\n0,0\n");
});

test("a malformed request is refused before any dialog opens", async () => {
  const d = dialog(join(dir, "never.csv"));
  assert.deepEqual(await saveCsv(42, "a.csv", d.choose), { ok: false, error: "there is no CSV text to save" });
  assert.deepEqual(await saveCsv("", "a.csv", d.choose), { ok: false, error: "there is no CSV text to save" });
  const big = "x".repeat(MAX_CSV_BYTES + 1);
  assert.match((await saveCsv(big, "a.csv", d.choose) as { error: string }).error, /larger than 64 MB/);
  assert.deepEqual(d.offered, [], "no dialog was shown");
});

test("a suggested name that is not a plain .csv name is replaced, never passed to the dialog", async () => {
  for (const name of ["../../evil.csv", "C:\\x\\y.csv", "run.exe", 7, "a b.csv", `${"n".repeat(81)}.csv`]) {
    const d = dialog(null);
    await saveCsv("x", name, d.choose);
    assert.deepEqual(d.offered, ["material-test.csv"], String(name));
  }
});

test("a cancelled dialog, a relative path and a failed write each say so", async () => {
  assert.deepEqual(await saveCsv("x", "a.csv", dialog(null).choose), { ok: false, cancelled: true, error: "cancelled" });
  assert.match((await saveCsv("x", "a.csv", dialog("a.csv").choose) as { error: string }).error, /not absolute/);
  const failing = async () => {
    throw new Error("EACCES: permission denied");
  };
  assert.deepEqual(await saveCsv("x", "a.csv", dialog(join(dir, "b.csv")).choose, failing), { ok: false, error: "EACCES: permission denied" });
});
