// VU9a (#2803; owner ruling option A, #2753): the workspace doors take no path
// from the page. main.ts runs in a child Node process with `electron` stubbed
// (workspaceDoorsDriver.ts), opens a set the way a double-click does, and
// answers a battery. Every refused request names a path to a folder the test
// watches: nothing may appear there, and the set's own workspace must not
// appear until the one well-formed save. In an agent mode the save door
// refuses outright (ADR 0128 INV-1 as ADR 0132 narrowed it).

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import type { WorkspaceIn, WorkspaceSaveIn } from "../../src/state/types/workspace.ts";

const FRAME = fileURLToPath(new URL("../fixtures/vu9a/frame.h5", import.meta.url));
const DRIVER = fileURLToPath(new URL("./workspaceDoorsDriver.ts", import.meta.url));

const root = mkdtempSync(join(tmpdir(), "vu9a-doors-"));
after(() => rmSync(root, { recursive: true, force: true }));
for (const d of ["set", "elsewhere"]) mkdirSync(join(root, d));
copyFileSync(FRAME, join(root, "set", "frame.h5"));
copyFileSync(FRAME, join(root, "elsewhere", "frame.h5"));

type Answer = { resolved: WorkspaceIn & WorkspaceSaveIn } | { rejected: string };

function run(extra: string[] = []): Record<string, Answer> {
  const args = [DRIVER, `--file=${join(root, "set", "frame.h5")}`, `--dir=${root}`, ...extra];
  const stdout = execFileSync(process.execPath, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("WORKSPACE_DOORS "));
  assert.ok(line, `the driver printed no answers:\n${stdout}`);
  return JSON.parse(line.slice("WORKSPACE_DOORS ".length)) as Record<string, Answer>;
}

const answers = run();
const resolved = (name: string) => {
  const a = answers[name];
  assert.ok(a && "resolved" in a, `${name} did not resolve: ${JSON.stringify(a)}`);
  return a.resolved;
};

test("workspace:read takes no argument: a path (or any argument) from the page is refused, never read", () => {
  for (const name of ["read:path", "read:object"]) {
    const r = resolved(name);
    assert.equal(r.status, "none");
    assert.match((r as { reason: string }).reason, /^workspace:read takes no argument/);
  }
});

test("workspace:save refuses any edit that names a path, a second argument, or no edit; nothing is written anywhere", () => {
  for (const name of ["save:pathKey", "save:fileKey", "save:string", "save:second", "save:none"]) {
    const r = resolved(name);
    assert.equal(r.ok, false, name);
    assert.equal((r as { reason?: string }).reason, "bad-request", name);
    assert.match((r as { detail?: string }).detail ?? "", /never a path/, name);
  }
  assert.equal((resolved("save:bigint") as { reason?: string }).reason, "bad-members");
  assert.deepEqual(readdirSync(join(root, "elsewhere")), ["frame.h5"], "no file appeared in the folder the requests named");
  assert.equal(resolved("read:before").status, "absent", "the refused saves wrote no workspace into the set either");
});

test("a save whose token is not the window set's file now refuses stale, naming main's file; nothing is written", () => {
  const stale = resolved("save:staleToken") as { ok: boolean; reason?: string; file?: string };
  assert.equal(stale.ok, false);
  assert.equal(stale.reason, "stale");
  assert.equal(stale.file?.toLowerCase(), join(root, "set", "frame.workspace.h5").toLowerCase());
  const again = resolved("save:afterOwnSave") as { ok: boolean; reason?: string };
  assert.equal(again.reason, "stale", "an edit made on the read before a save is refused once the file changed");
});

test("the one well-formed save writes the set's own <stem>.workspace.h5, at the path main chose", () => {
  const r = resolved("save:good");
  assert.equal(r.ok, true, JSON.stringify(r));
  const file = join(root, "set", "frame.workspace.h5");
  assert.ok(existsSync(file));
  const read = resolved("read:after");
  assert.equal(read.status, "ready");
  assert.equal((read as { file: string }).file.toLowerCase(), file.toLowerCase());
  assert.deepEqual(readdirSync(join(root, "set")).sort(), ["frame.h5", "frame.workspace.h5"], "no partial left beside it");
});

test("an agent mode reads the workspace and its save door refuses outright", () => {
  const out = join(root, "measure.json");
  const agent = run(["--mode=measure", `--out=${out}`]);
  const r = (agent["save:good"] as { resolved: WorkspaceSaveIn }).resolved;
  assert.equal(r.ok, false);
  assert.equal((r as { reason: string }).reason, "read-only");
  assert.equal(((agent["read:after"] as { resolved: WorkspaceIn }).resolved).status, "ready", "it still reads the file the window saved");
});
