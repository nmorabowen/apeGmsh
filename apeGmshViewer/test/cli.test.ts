// The `apegmsh-viewer` command (scripts/cli.mjs): what it hands Electron, and
// what it refuses before it builds or starts anything. A stub stands in for the
// Electron binary (the electron package's ELECTRON_OVERRIDE_DIST_PATH), so no
// window opens; the stub is a shell script, so those tests skip on Windows.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, realpathSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const app = fileURLToPath(new URL("..", import.meta.url));
const cli = join(app, "scripts", "cli.mjs");

function folder(): string {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "agv-cli-")));
  writeFileSync(join(dir, "a model.h5"), "");
  mkdirSync(join(dir, "sub"));
  mkdirSync(join(dir, "electron-stub"));
  writeFileSync(join(dir, "electron-stub", "electron"), '#!/bin/sh\nprintf "argv:%s\\n" "$@"\nexit "${STUB_EXIT:-0}"\n', { mode: 0o755 });
  return dir;
}

function run(cwd: string, args: string[], o: { script?: string; stub?: string; exit?: number } = {}) {
  const env = { ...process.env, ELECTRON_OVERRIDE_DIST_PATH: o.stub ?? join(cwd, "electron-stub"), STUB_EXIT: String(o.exit ?? 0) };
  const r = spawnSync(process.execPath, [o.script ?? cli, ...args], { cwd, env, encoding: "utf8" });
  const argv = r.stdout.split("\n").filter((l) => l.startsWith("argv:")).map((l) => l.slice("argv:".length));
  return { status: r.status, out: r.stdout + r.stderr, argv };
}

const windows = process.platform === "win32" && "the Electron stub is a shell script";

test("a relative path means the caller's folder", { skip: windows }, () => {
  const dir = folder();
  const r = run(dir, ["a model.h5"]);
  assert.equal(r.status, 0, r.out);
  assert.ok(r.argv.includes("--mode=view"), r.out);
  assert.ok(r.argv.includes(`--file=${join(dir, "a model.h5")}`), r.out);

  const up = run(join(dir, "sub"), ["../a model.h5"], { stub: join(dir, "electron-stub") });
  assert.ok(up.argv.includes(`--file=${join(dir, "a model.h5")}`), up.out);
});

test("it runs through a link on PATH, as npm link and ln -s install it", { skip: windows }, () => {
  const dir = folder();
  const link = join(dir, "apegmsh-viewer");
  symlinkSync(cli, link);
  const r = run(dir, ["a model.h5"], { script: link });
  assert.equal(r.status, 0, r.out);
  assert.ok(r.argv.includes(`--file=${join(dir, "a model.h5")}`), r.out);
});

test("without a file it opens the app on none, and it exits with the app's code", { skip: windows }, () => {
  const dir = folder();
  const r = run(dir, [], { exit: 3 });
  assert.equal(r.status, 3, r.out);
  assert.ok(r.argv.includes("--mode=view"), r.out);
  assert.ok(!r.argv.some((a) => a.startsWith("--file=")), r.out);
});

for (const [what, args, why] of [
  ["a missing file", ["nope.h5"], /no such file: nope\.h5/],
  ["a folder", ["sub"], /not a file: sub/],
  ["two files", ["a model.h5", "a model.h5"], /one file at a time/],
  ["an unknown flag", ["--render", "a model.h5"], /unknown flag --render/],
] as const) {
  test(`${what} is refused with exit 2, and nothing starts`, () => {
    const r = run(folder(), [...args]);
    assert.equal(r.status, 2, r.out);
    assert.match(r.out, why);
    assert.match(r.out, /usage: apegmsh-viewer/);
    assert.deepEqual(r.argv, []);
  });
}

test("--help prints the usage and starts nothing", () => {
  const r = run(folder(), ["--help"]);
  assert.equal(r.status, 0, r.out);
  assert.match(r.out, /usage: apegmsh-viewer \[file\]/);
  assert.deepEqual(r.argv, []);
});

test("package.json names the command", async () => {
  const pkg = (await import("../package.json", { with: { type: "json" } })).default;
  assert.deepEqual(pkg.bin, { "apegmsh-viewer": "scripts/cli.mjs" });
});
