// Go to source (src/main/source.ts). The editor choice is tested on an
// environment built here (never this machine's PATH), and the launch is
// proved end to end by a fake editor that records the argv it received.

import assert from "node:assert/strict";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { FOLD_CASE } from "../../src/main/pairing.ts";
import { editorLaunch, goToSource, parseTemplate, provenanceSources, which, type SourceEnv, type SourceScope } from "../../src/main/source.ts";

const win = process.platform === "win32";
const root = mkdtempSync(join(tmpdir(), "agv-source-"));
after(() => rmSync(root, { recursive: true, force: true }));

/** A directory holding the named (empty) executables, for PATH lookups. */
function binDir(name: string, ...files: string[]): string {
  const d = join(root, name);
  mkdirSync(d, { recursive: true });
  for (const f of files) writeFileSync(join(d, f), "");
  return d;
}

const envOf = (over: Partial<SourceEnv>): SourceEnv => ({
  editor: undefined,
  path: "",
  pathext: ".exe;.cmd",
  comspec: "C:\\Windows\\system32\\cmd.exe",
  platform: process.platform,
  ...over,
});

// The script the editor opens: spaces and `&` in the path on purpose.
const scriptDir = join(root, "a b&c");
mkdirSync(scriptDir);
const script = join(scriptDir, "frame one.py");
writeFileSync(script, "print('hi')\n");

test("parseTemplate: whitespace splits, double quotes group", () => {
  assert.deepEqual(parseTemplate(`"C:\\Program Files\\ed.exe"  -n{line} {file}`), [
    "C:\\Program Files\\ed.exe",
    "-n{line}",
    "{file}",
  ]);
  assert.deepEqual(parseTemplate(`ed ""`), ["ed", ""]);
  assert.throws(() => parseTemplate(`"ed -g {file}`), /unclosed quote/);
});

test("which: PATH and PATHEXT", () => {
  const d = binDir("which", win ? "tool.cmd" : "tool");
  const env = envOf({ path: ["", join(root, "nowhere"), `"${d}"`].join(win ? ";" : ":") });
  assert.equal(which("tool", env), join(d, win ? "tool.cmd" : "tool"));
  assert.equal(which("absent", env), null);
  assert.equal(which(join(d, win ? "tool.cmd" : "tool"), env), join(d, win ? "tool.cmd" : "tool"));
});

test("APEGMSH_EDITOR: placeholders filled per argument; a broken template is the answer", () => {
  const d = binDir("tmpl", win ? "ed.exe" : "ed");
  const env = (editor: string) => envOf({ editor, path: d, platform: win ? "win32" : "linux" });
  const l = editorLaunch(script, 12, env(`ed --line {line} {file}`));
  assert.deepEqual(l, { command: join(d, win ? "ed.exe" : "ed"), args: ["--line", "12", script], verbatim: false });
  // `$&` in a path is not a replacement pattern.
  assert.deepEqual(editorLaunch("/x/$&.py", 3, env(`ed {file}:{line}`)).args, ["/x/$&.py:3"]);
  assert.throws(() => editorLaunch(script, 1, env(`ed {path}`)), /unknown placeholder \{path\}/);
  assert.throws(() => editorLaunch(script, 1, env(`ed -g {line}`)), /must pass \{file\}/);
  // Set but not found: no fallback to VS Code or the OS editor.
  const withCode = binDir("tmpl-code", win ? "code.cmd" : "code");
  assert.throws(
    () => editorLaunch(script, 1, envOf({ editor: "nosuch {file}", path: withCode })),
    /APEGMSH_EDITOR: nosuch not found/,
  );
});

test("VS Code on PATH: code -g file:line, through cmd.exe for code.cmd", () => {
  const d = binDir("vscode", "code.cmd", "code");
  const w = editorLaunch(script, 7, envOf({ path: d, platform: "win32", pathext: ".cmd" }));
  assert.deepEqual(w, {
    command: "C:\\Windows\\system32\\cmd.exe",
    args: ["/d", "/s", "/c", `""${join(d, "code.cmd")}" "-g" "${script}:7""`],
    verbatim: true,
  });
  const l = editorLaunch(script, 7, envOf({ path: d, platform: "linux" }));
  assert.deepEqual(l, { command: join(d, "code"), args: ["-g", `${script}:7`], verbatim: false });
  // cmd.exe cannot quote `"` or `%`: refused, not mangled.
  assert.throws(
    () => editorLaunch(join(root, "100%.py"), 1, envOf({ path: d, platform: "win32", pathext: ".cmd" })),
    /cannot pass .* through cmd\.exe/,
  );
});

test("no code on PATH: the OS text editor, never the default opener", () => {
  const w = binDir("os-win", "notepad.exe");
  assert.deepEqual(editorLaunch(script, 7, envOf({ path: w, platform: "win32" })).args, [script]);
  assert.equal(editorLaunch(script, 7, envOf({ path: w, platform: "win32" })).command, join(w, "notepad.exe"));
  const m = binDir("os-mac", "open");
  assert.deepEqual(editorLaunch(script, 7, envOf({ path: m, platform: "darwin" })).args, ["-t", script]);
  assert.throws(() => editorLaunch(script, 7, envOf({ path: "", platform: "linux" })), /no editor found/);
});

/** The scope main hands the door for a set whose /provenance names `files`. */
const scopeOf = (...files: string[]): SourceScope => ({ files });

test("goToSource refuses a missing file, a relative path and a bad line", async () => {
  const env = envOf({});
  const scope = scopeOf(join(root, "gone.py"), script, scriptDir);
  assert.deepEqual(await goToSource(join(root, "gone.py"), 3, scope, env), { ok: false, reason: `missing: ${join(root, "gone.py")}` });
  assert.match(((await goToSource("frame.py", 3, scope, env)) as { reason: string }).reason, /absolute file path/);
  assert.match(((await goToSource(script, 0, scope, env)) as { reason: string }).reason, /line number >= 1/);
  assert.match(((await goToSource(script, 2.5, scope, env)) as { reason: string }).reason, /line number >= 1/);
  assert.match(((await goToSource(scriptDir, 1, scope, env)) as { reason: string }).reason, /missing/);
});

// VU13 (#2760), option A: the page names no file to open. A readable script the
// set's /provenance does not name is refused before any editor is chosen: the
// env below names a working editor (a recorder), so without the check it would launch.
test("goToSource opens only a file the set's /provenance names; a readable stranger is refused", async () => {
  const rec = recorder("scope");
  const stranger = join(scriptDir, "stranger.py");
  writeFileSync(stranger, "print('not in the set')\n");
  const env = envOf({ editor: `"${process.execPath}" "${join(rec.dir, "rec.mjs")}" {line} {file}` });
  assert.deepEqual(await goToSource(stranger, 3, scopeOf(script), env), {
    ok: false,
    reason: `goToSource opens only a source file the open set's /provenance names; refused ${stranger}`,
  });
  assert.deepEqual(await goToSource(script, 3, { reason: "no set is open" }, env), {
    ok: false,
    reason: `goToSource opens only a source file the open set's /provenance names, and no set is open; refused ${script}`,
  });
  assert.equal(existsSync(join(rec.dir, "argv.json")), false, "a refused request launched no editor");
  // The path the scope holds is the one opened (case-folded where the file system folds).
  const asked = FOLD_CASE ? script.toUpperCase() : script;
  assert.deepEqual(await goToSource(asked, 5, scopeOf(script), env), { ok: true });
  assert.deepEqual(await rec.argv(), ["5", script]);
});

test("goToSource's refusals are total: a BigInt or a circular request resolves with the reason", async () => {
  const circular: Record<string, unknown> = { file: 7 };
  circular.self = circular;
  const env = envOf({});
  const scope = scopeOf(script);
  assert.deepEqual(await goToSource(5n, 3, scope, env), { ok: false, reason: "goToSource needs an absolute file path; got 5n" });
  assert.match(((await goToSource(circular, 3, scope, env)) as { reason: string }).reason, /^goToSource needs an absolute file path; got .*\[Circular \*1\]/);
  assert.deepEqual(await goToSource(script, 5n, scope, env), { ok: false, reason: "goToSource needs a line number >= 1; got 5n" });
  assert.match(((await goToSource(script, circular, scope, env)) as { reason: string }).reason, /^goToSource needs a line number >= 1; got .*\[Circular \*1\]/);
});

test("provenanceSources: a relative row joins @base_dir, an absolute row (outside the set's folder) is kept, a pseudo-file names none", () => {
  const zone = { baseDir: "C:/runs/frame/", files: { path: ["build.py", "/opt/helpers/frames.py", "<string>", "D:/lib/x.py"], sha256: [], kind: [] } };
  assert.deepEqual(provenanceSources(zone), ["C:/runs/frame/build.py", "/opt/helpers/frames.py", "D:/lib/x.py"]);
});

/** A fake editor named `name` that writes the argv it received to argv.json. */
function recorder(name: string): { dir: string; argv: () => Promise<string[]> } {
  const dir = join(root, `rec-${name}`);
  mkdirSync(dir);
  const out = join(dir, "argv.json");
  writeFileSync(
    join(dir, "rec.mjs"),
    `import { writeFileSync } from "node:fs";\nwriteFileSync(${JSON.stringify(out)}, JSON.stringify(process.argv.slice(2)));\n`,
  );
  if (win) {
    writeFileSync(join(dir, `${name}.cmd`), `@"${process.execPath}" "%~dp0rec.mjs" %*\r\n`);
  } else {
    writeFileSync(join(dir, name), `#!/bin/sh\nexec "${process.execPath}" "$(dirname "$0")/rec.mjs" "$@"\n`);
    chmodSync(join(dir, name), 0o755);
  }
  const argv = async () => {
    const end = Date.now() + 10000;
    while (!existsSync(out)) {
      if (Date.now() > end) throw new Error(`the fake editor never wrote ${out}`);
      await new Promise((r) => setTimeout(r, 50));
    }
    await new Promise((r) => setTimeout(r, 100));
    return JSON.parse(readFileSync(out, "utf8")) as string[];
  };
  return { dir, argv };
}

test("live: `code` on PATH receives -g and file:line intact (spaces, &)", async () => {
  const rec = recorder("code");
  const res = await goToSource(script, 42, scopeOf(script), envOf({ path: rec.dir, pathext: ".cmd", comspec: process.env["ComSpec"] }));
  assert.deepEqual(res, { ok: true });
  assert.deepEqual(await rec.argv(), ["-g", `${script}:42`]);
});

test("live: an APEGMSH_EDITOR template receives its filled arguments", async () => {
  const rec = recorder("tmpl");
  const editor = `"${process.execPath}" "${join(rec.dir, "rec.mjs")}" --at {line} {file}`;
  assert.deepEqual(await goToSource(script, 9, scopeOf(script), envOf({ editor })), { ok: true });
  assert.deepEqual(await rec.argv(), ["--at", "9", script]);
});
