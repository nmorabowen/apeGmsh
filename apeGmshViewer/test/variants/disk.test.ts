// VU10 (#2694) review of #2817: the manifest reader and main's study on a real
// disk, each against its oracle.
//
// - Codes (review 5): Python's `read_variants` (#2806) is canonical. For every
//   case of test/fixtures/vu10/codes.golden.json (the schema corpus and one case
//   per code it does not reach, written by scripts/make_viewer_variants_fixture.py
//   with Python's verdicts), the app's reader gives the same rows, the same
//   refused rows with the same codes, or the same file code. One difference is
//   ADR 0132 D2's own: a same-major newer minor, which Python refuses (`newer`),
//   opens in the app read-only with a banner.
// - 8.3 names (review 2): a row naming `SLABUS~1.H5`, the short name of
//   `slab.user.h5`, is refused as `bad-path`, as Python refuses it: the
//   resolved name is judged. Windows only, and only where the volume makes 8.3 names.
// - A junction after the read (review 3): `Study.row` resolves and confines the
//   row again when it is opened, so a folder swapped for a junction to a folder
//   outside the study is refused, never followed; so is a set file of the row.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, renameSync, rmSync, statSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { Study } from "../../src/main/session.ts";
import { confinedToStudy, readVariants, resolveRow, type VariantsFs } from "../../src/reader/variants.ts";

const FOLD = process.platform === "win32" || process.platform === "darwin";
/** The disk as main reads it (main.ts `variantsFs`). */
const realFs: VariantsFs = {
  readFile: (p) => readFileSync(p, "utf8"),
  isFile: (p) => {
    try {
      return statSync(p).isFile();
    } catch {
      return false;
    }
  },
  realpath: (p) => realpathSync.native(p),
  fold: FOLD,
};

const roots: string[] = [];
/** Junctions made here, removed before their trees so a removal never follows one into its target. */
const links: string[] = [];
after(() => {
  for (const l of links) {
    try {
      unlinkSync(l);
    } catch {
      // already gone
    }
  }
  for (const r of roots) rmSync(r, { recursive: true, force: true });
});
function tempRoot(tag: string): string {
  const r = mkdtempSync(join(tmpdir(), `vu10-${tag}-`));
  roots.push(r);
  return r;
}
function lay(root: string, files: readonly string[]): void {
  for (const f of files) {
    mkdirSync(dirname(join(root, f)), { recursive: true });
    writeFileSync(join(root, f), "");
  }
}

type Verdict = { file: string } | { rows: number[]; refused: [number, string][] };
const GOLDEN = JSON.parse(readFileSync(fileURLToPath(new URL("../fixtures/vu10/codes.golden.json", import.meta.url)), "utf8")) as {
  files: string[];
  cases: { name: string; text: string; python: Verdict }[];
};

test("codes: the app's reader gives Python's verdict on every golden case (the newer minor opens read-only, by ADR 0132 D2)", () => {
  assert.ok(GOLDEN.cases.length >= 30);
  for (const c of GOLDEN.cases) {
    const root = tempRoot("codes");
    lay(root, GOLDEN.files);
    const m = join(root, "s.variants.json");
    writeFileSync(m, c.text);
    const r = readVariants(m, realFs);
    if ("file" in c.python && c.python.file === "newer") {
      assert.ok(r.ok && r.newer !== null, `${c.name}: a same-major newer minor opens read-only with a banner`);
      continue;
    }
    const got: Verdict = r.ok ? { rows: r.rows.map((x) => x.index), refused: r.refused.map((x) => [x.index, x.code]) } : { file: r.code };
    assert.deepEqual(got, c.python, c.name);
  }
});

/** The 8.3 short name of `path` (cmd's `%~s`), or null where it has none. */
function shortName(path: string): string | null {
  const r = spawnSync("cmd.exe", ["/d", "/c", `for %I in ("${path}") do @echo %~sI`], { encoding: "utf8", windowsVerbatimArguments: true });
  if (r.status !== 0) throw new Error(`cmd could not name the short name of ${path}: ${r.stderr}`);
  const out = r.stdout.trim();
  const short = basename(out);
  return short.toLowerCase() === basename(path).toLowerCase() ? null : short;
}

const SHORT = process.platform !== "win32" ? "8.3 names are Windows' alone" : null;

test("8.3 names: a row naming the short name of slab.user.h5 is refused as bad-path, as Python refuses it", { skip: SHORT ?? false }, (t) => {
  const root = tempRoot("short");
  lay(root, ["slab.h5", "slab.user.h5"]);
  const short = shortName(join(root, "slab.user.h5"));
  if (short === null) {
    t.skip("this volume makes no 8.3 names (fsutil 8dot3name), so there is no short name to name");
    return;
  }
  assert.match(short, /~\d\.H5$/i);
  const m = join(root, "s.variants.json");
  writeFileSync(m, JSON.stringify({ format: "apegmsh-variants", version: "1.0.0", variants: [{ label: "A", path: "slab.h5", params: {} }, { label: "S", path: short, params: {} }] }));
  const r = readVariants(m, realFs);
  assert.ok(r.ok);
  assert.deepEqual(r.rows.map((x) => x.index), [0]);
  assert.deepEqual(r.refused.map((x) => [x.index, x.code]), [[1, "bad-path"]]);
  assert.match(r.refused[0]!.reason, /resolves to slab\.user\.h5, not a model file/);
});

test("a folder swapped for a junction after the read is refused when the row is opened, and a set file outside the study is refused", () => {
  const root = tempRoot("junction");
  const study = join(root, "study");
  lay(study, ["a.h5", "sub/b.h5"]);
  lay(root, ["outside/b.h5", "outside/b.ladruno"]);
  const m = join(study, "s.variants.json");
  writeFileSync(m, JSON.stringify({ format: "apegmsh-variants", version: "1.0.0", variants: [{ label: "A", path: "a.h5", params: {} }, { label: "B", path: "sub/b.h5", params: {} }] }));
  const s = new Study({
    read: (p) => readVariants(p, realFs),
    watch: () => ({ close: () => undefined }),
    note: () => undefined,
    recheck: (manifest, row) => resolveRow(manifest, row.path, realFs),
    confined: (manifest, path) => confinedToStudy(manifest, path, realFs),
  });
  const r = s.open(m);
  assert.ok(r.ok && r.rows.length === 2, JSON.stringify(r));
  const before = s.row(1, "B");
  assert.ok(before.ok, JSON.stringify(before));
  // The swap: sub/ moves aside and a junction to the outside folder takes its name.
  renameSync(join(study, "sub"), join(study, "sub.old"));
  const junction = join(study, "sub");
  symlinkSync(join(root, "outside"), junction, "junction");
  links.push(junction);
  const after = s.row(1, "B");
  assert.equal(after.ok, false, "the row must be refused, not opened through the junction");
  assert.match(!after.ok ? after.reason : "", /no longer opens \(escapes\)/);
  assert.ok(s.row(0, "A").ok, "a row the swap did not touch still opens");
  // A set file of a row (the paired results) is confined too.
  assert.match(s.confine(join(junction, "b.ladruno")) ?? "", /outside the manifest's folder/);
  assert.equal(s.confine(join(study, "a.h5")), null);
});
