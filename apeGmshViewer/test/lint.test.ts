// `npm run lint` (dependency-cruiser) over a copy of src/ with one planted
// violation per rule fails and names each rule; the real tree passes.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { cpSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const app = fileURLToPath(new URL("..", import.meta.url));
const depcruise = join(app, "node_modules", "dependency-cruiser", "bin", "dependency-cruiser.mjs");

function cruise(cwd: string): { status: number | null; out: string } {
  const r = spawnSync(process.execPath, [depcruise, "--config", join(app, ".dependency-cruiser.cjs"), "src"], { cwd, encoding: "utf8" });
  return { status: r.status, out: r.stdout + r.stderr };
}

test("dependency-cruiser passes the tree as committed", () => {
  const r = cruise(app);
  assert.equal(r.status, 0, r.out);
});

test("dependency-cruiser fails on a planted violation of each panel rule", () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-lint-"));
  cpSync(join(app, "src"), join(dir, "src"), { recursive: true });
  cpSync(join(app, "tsconfig.json"), join(dir, "tsconfig.json"));
  writeFileSync(
    join(dir, "src", "panels", "planted.ts"),
    [
      'import { mountLegend } from "./legend.ts";',
      'import { Viewport } from "../renderer/viewport.ts";',
      'import { readModel } from "../reader/read.ts";',
      'import { BlobStore } from "../state/blobs.ts";',
      "export const planted = [mountLegend, Viewport, readModel, BlobStore];",
      "",
    ].join("\n"),
  );
  const r = cruise(dir);
  assert.notEqual(r.status, 0, "a planted violation must fail the run");
  for (const rule of ["no-panel-to-panel", "no-panel-to-render", "no-panel-to-reader", "no-panel-to-blobs"]) {
    assert.match(r.out, new RegExp(`${rule}: src/panels/planted.ts`), rule);
  }
});
