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

const PLANTED = [
  ["./legend.ts", "mountLegend"],
  ["../renderer/viewport.ts", "Viewport"],
  ["../reader/read.ts", "readModel"],
  ["../state/blobs.ts", "BlobStore"],
  ["../effects.ts", "Effects"],
  ["../state/load.ts", "loadModel"],
  ["../state/reduce.ts", "reduce"],
] as const;

test("dependency-cruiser fails on each planted import outside the panel allow-list, and passes an allowed one", () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-lint-"));
  cpSync(join(app, "src"), join(dir, "src"), { recursive: true });
  cpSync(join(app, "tsconfig.json"), join(dir, "tsconfig.json"));
  writeFileSync(
    join(dir, "src", "panels", "planted.ts"),
    [
      ...PLANTED.map(([spec, name]) => `import { ${name} } from "${spec}";`),
      'import type { State } from "../state/store.ts";',
      'import { el } from "../ui/dom.ts";',
      `export const planted: unknown[] = [${PLANTED.map(([, name]) => name).join(", ")}, el, null as unknown as State];`,
      "",
    ].join("\n"),
  );
  const r = cruise(dir);
  assert.notEqual(r.status, 0, "a planted violation must fail the run");
  for (const [spec] of PLANTED) {
    const target = spec.replace(/^\.\.\//, "src/").replace(/^\.\//, "src/panels/");
    assert.match(r.out, new RegExp(`panel-allow-list: src/panels/planted.ts → ${target.replace(/[.]/g, "\\.")}`), spec);
  }
  assert.doesNotMatch(r.out, /src\/state\/store\.ts|src\/ui\/dom\.ts/, "allowed imports are not reported");
  assert.match(r.out, new RegExp(`${PLANTED.length} dependency violations`));
});

test("dependency-cruiser fails on a ui/ module that re-exports what a panel may not reach", () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-lint-ui-"));
  cpSync(join(app, "src"), join(dir, "src"), { recursive: true });
  cpSync(join(app, "tsconfig.json"), join(dir, "tsconfig.json"));
  writeFileSync(
    join(dir, "src", "ui", "leak.ts"),
    [
      'export { Effects } from "../effects.ts";',
      'export { BlobStore } from "../state/blobs.ts";',
      'export type { State } from "../state/types.ts";',
      'export { el } from "./dom.ts";',
      "",
    ].join("\n"),
  );
  writeFileSync(join(dir, "src", "panels", "launder.ts"), 'import { Effects, BlobStore } from "../ui/leak.ts";\nexport const x = [Effects, BlobStore];\n');
  const r = cruise(dir);
  assert.notEqual(r.status, 0);
  assert.match(r.out, /ui-allow-list: src\/ui\/leak\.ts → src\/effects\.ts/);
  assert.match(r.out, /ui-allow-list: src\/ui\/leak\.ts → src\/state\/blobs\.ts/);
  assert.doesNotMatch(r.out, /leak\.ts → src\/state\/types\.ts|leak\.ts → src\/ui\/dom\.ts/);
  assert.match(r.out, /2 dependency violations/);
});
