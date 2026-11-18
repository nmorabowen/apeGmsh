// VU14 (#2771): a /provenance row is the model's to name, so a hostile model can
// name another machine (`//host/share/x.py`, `\\host\share`). On Windows a
// lookup of such a path connects to the host, and can send it the user's NTLM
// credentials. Neither main's zone read (zones.ts `openProvenance`, which fills
// `present`) nor go-to-source (source.ts `goToSource`) may look one up.
//
// The oracle is the file system itself: every `node:fs` function is wrapped
// (and the ESM bindings re-synced, `syncBuiltinESMExports`), so any call that
// names the remote host is recorded, and refused with ENOENT instead of reaching
// the network. A local row in the same zone is still looked up (`present`
// true), so the wrapper is proved to see the calls the code makes.

import assert from "node:assert/strict";
import fs, { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import * as h5wasm from "h5wasm/node";
import { goToSource, isRemotePath, type SourceEnv } from "../../src/main/source.ts";
import { openProvenance } from "../../src/main/zones.ts";
import { tree, writeTree } from "../zones/tree.ts";

const HOST = "vu14-remote.invalid";
const root = mkdtempSync(join(tmpdir(), "vu14-remote-"));
const posix = (p: string) => p.replaceAll("\\", "/");
const local = join(root, "local.py");
writeFileSync(local, "print('hi')\n");

/** Every fs call whose arguments name HOST, in order. */
const reached: string[] = [];
const names = (a: unknown) => (typeof a === "string" || Buffer.isBuffer(a) || a instanceof URL) && String(a).includes(HOST);
const restore: (() => void)[] = [];

function wrapAll(target: Record<string, unknown>, label: string): void {
  for (const key of Object.keys(target)) {
    const fn = target[key];
    if (typeof fn !== "function" || !/^[a-z]/.test(key)) continue; // classes (Stats, ReadStream) are not calls
    target[key] = function (this: unknown, ...args: unknown[]) {
      if (args.some(names)) {
        reached.push(`${label}.${key}(${args.filter(names).map(String).join(", ")})`);
        throw Object.assign(new Error(`ENOENT: vu14 stub, ${key}`), { code: "ENOENT" });
      }
      return (fn as (...a: unknown[]) => unknown).apply(this, args);
    };
    restore.push(() => (target[key] = fn));
  }
}

before(() => {
  wrapAll(fs as unknown as Record<string, unknown>, "fs");
  wrapAll(fs.promises as unknown as Record<string, unknown>, "fs.promises");
  syncBuiltinESMExports();
});
after(() => {
  for (const r of restore) r();
  syncBuiltinESMExports();
  rmSync(root, { recursive: true, force: true });
});

/** A model whose /provenance has `base_dir` and the rows `paths`. */
async function model(name: string, baseDir: string, paths: string[]): Promise<string> {
  await h5wasm.ready;
  const path = join(root, `${name}.h5`);
  writeTree(h5wasm, path, tree({
    meta: { attrs: { provenance_schema_version: "1.0.0" } },
    provenance: {
      attrs: { base_dir: baseDir },
      files: { path: paths, sha256: paths.map(() => ""), kind: paths.map(() => "script") },
      sites: { file: new Int32Array([0]), line: new Int32Array([1]), function: ["<module>"] },
      records: { path: ["opensees/section/A"], site: new Int32Array([0]), script: new Int32Array([0]), seq: new Int32Array([0]) },
    },
  }));
  return path;
}

test("isRemotePath: two leading separators, either way round, name a host; a local path does not", () => {
  for (const p of [`//${HOST}/s/a.py`, `\\\\${HOST}\\s\\a.py`, `\\\\?\\UNC\\${HOST}\\s\\a.py`, `\\\\.\\pipe\\x`, `/\\${HOST}/s`, `\\/${HOST}/s`]) {
    assert.equal(isRemotePath(p), true, p);
  }
  for (const p of ["C:/runs/a.py", "C:\\runs\\a.py", "/home/u/a.py", "a.py", posix(local)]) assert.equal(isRemotePath(p), false, p);
});

test("openProvenance looks up no row that names another machine; a local row in the same zone is still looked up", async () => {
  const cases: [string, string, string[], boolean[]][] = [
    // an absolute row naming a host, beside a local one
    ["abs", posix(root), [`//${HOST}/share/a.py`, posix(local), "<string>"], [false, true, false]],
    // a relative row under a @base_dir naming a host (the reader refuses a backslash @base_dir)
    ["base", `//${HOST}/share`, ["build.py", posix(local)], [false, true]],
  ];
  for (const [name, base, paths, present] of cases) {
    const a = await openProvenance(await model(name, base, paths));
    assert.ok(a.ok, JSON.stringify(a));
    assert.deepEqual(a.zone!.present, present, `${name}: a remote row is not present, the local row is`);
  }
  assert.deepEqual(reached, [], "no fs call named the remote host");
});

test("goToSource refuses a row naming another machine as unavailable, before any fs call or editor", async () => {
  reached.length = 0;
  const env: SourceEnv = { editor: `"${process.execPath}" -e 0 {file}`, path: "", pathext: ".exe", comspec: undefined, platform: process.platform };
  // A backslash path is absolute only on Windows; elsewhere goToSource refuses it as relative, before any fs call.
  const rows = [`//${HOST}/share/a.py`, ...(process.platform === "win32" ? [`\\\\${HOST}\\share\\a.py`, `\\\\?\\UNC\\${HOST}\\share\\a.py`] : [])];
  for (const row of rows) {
    const r = await goToSource(row, 3, { files: [row, posix(local)] }, env);
    assert.deepEqual(r, { ok: false, reason: `unavailable: ${row} names another machine, and goToSource opens only a local file` });
  }
  assert.deepEqual(reached, [], "no fs call named the remote host");
});
