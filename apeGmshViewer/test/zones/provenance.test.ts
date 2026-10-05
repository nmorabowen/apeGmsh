// The /provenance reader (src/reader/provenance.ts).
//
// Oracles:
// - fixtures/zones.h5, written by h5py to h5-schema.md: each record points at
//   the line of examples/shoebuckle_arch.py that declares it, and the file's
//   sha256 is the script's real digest, so both are checked against the script
//   itself, not against values copied from the reader;
// - fail-closed cases on synthetic files written here with h5wasm.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { readProvenance, sourceOf, type SourceSite } from "../../src/reader/provenance.ts";
import { SchemaError, type H5Module } from "../../src/reader/read.ts";
import { tree, writeTree, type Tree } from "./tree.ts";

await h5wasm.ready;
const h5 = h5wasm as unknown as H5Module;
const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, "..", "..", "fixtures");
const repo = join(here, "..", "..", "..");
const tmp = mkdtempSync(join(tmpdir(), "agv-provenance-"));
after(() => rmSync(tmp, { recursive: true, force: true }));

const script = join(repo, "examples", "shoebuckle_arch.py");
const scriptLines = readFileSync(script, "utf8").split(/\r?\n/);
// The fixture hashes the script with LF line ends; see make_zone_fixtures.py.
const scriptSha = createHash("sha256").update(readFileSync(script, "utf8").replace(/\r\n/g, "\n")).digest("hex");
const site = (r: ReturnType<typeof sourceOf>): SourceSite => {
  assert.ok(r.ok, JSON.stringify(r));
  return (r as { site: SourceSite }).site;
};

test("the fixture's records point at the lines that declare the beam chain", () => {
  const p = readProvenance(h5, join(fixtures, "zones.h5"))!;
  assert.ok(p);
  assert.equal(p.version, "1.0.0");
  assert.equal(p.baseDir, "/work/apeGmsh");
  assert.deepEqual(p.warnings, []);
  // The lines are valid for the script the capture hashed. When the example
  // is edited, regenerate: python apeGmshViewer/fixtures/make_zone_fixtures.py
  assert.equal(
    p.files.sha256[0],
    scriptSha,
    "examples/shoebuckle_arch.py changed since fixtures/zones.h5 was made; re-run fixtures/make_zone_fixtures.py",
  );
  const expect: Record<string, RegExp> = {
    "opensees/uniaxialMaterial/Steel": /ops\.uniaxialMaterial\.ASDSteel1D\(/,
    "opensees/section/W_section": /ops\.section\.Fiber\(/,
    "opensees/geomTransf/#1": /ops\.geomTransf\.Linear\(/,
    "opensees/beamIntegration/#1": /ops\.beamIntegration\.Legendre\(/,
    "opensees/element/#1": /ops\.element\.dispBeamColumn\(/,
  };
  for (const [decl, code] of Object.entries(expect)) {
    const s = site(sourceOf(p, decl));
    assert.equal(s.file, "/work/apeGmsh/examples/shoebuckle_arch.py", decl);
    assert.equal(s.function, "declare_model");
    assert.equal(s.kind, "script");
    assert.match(scriptLines[s.line - 1]!, code, `${decl} -> line ${s.line}`);
  }
  // The script's outermost line, and the sha256 the capture recorded.
  const r = sourceOf(p, "opensees/element/#1") as { script: SourceSite; seq: number };
  assert.match(scriptLines[r.script.line - 1]!, /^\s+main\(\)/);
  assert.equal(r.seq, 4);
  assert.equal(r.script.sha256, scriptSha);
});

test("an absolute file path is kept; a record without a script frame has script null", () => {
  const p = readProvenance(h5, join(fixtures, "zones.h5"))!;
  const fix = sourceOf(p, "opensees/fix/#1") as { site: SourceSite; script: SourceSite };
  assert.equal(fix.site.file, "/opt/helpers/frames.py");
  assert.equal(fix.site.kind, "module");
  assert.equal(fix.site.line, 12);
  assert.equal(fix.script.line, 451);
  const arch = sourceOf(p, "mesh/physical_group/Arch") as { site: SourceSite; script: SourceSite | null };
  assert.equal(arch.script, null);
  assert.equal(arch.site.line, 451);
});

test("an unknown declaration path says so; it is not a silent null", () => {
  const p = readProvenance(h5, join(fixtures, "zones.h5"))!;
  assert.deepEqual(sourceOf(p, "opensees/section/#2"), { ok: false, reason: "no provenance record for opensees/section/#2" });
});

test("a file without the zone is ignored", () => {
  // main's writer stamps /provenance into every model.h5 (#1378), so the
  // zone-less model is written here, as a file from before the zone existed.
  const bare = join(tmp, "no-provenance.h5");
  writeTree(h5wasm, bare, tree({ meta: { attrs: { neutral_schema_version: "2.33.0" } }, nodes: { ids: new Int32Array([1]) } }));
  assert.equal(readProvenance(h5, bare), null);
  assert.equal(readProvenance(h5, join(fixtures, "zones.geometry.h5")), null);
});

// --- synthetic files -------------------------------------------------------

function valid(): Tree {
  return tree({
    meta: { attrs: { provenance_schema_version: "1.0.0" } },
    provenance: {
      attrs: { base_dir: "C:/runs/frame" },
      files: { path: ["model.py"], sha256: ["a".repeat(64)], kind: ["script"] },
      sites: { file: new Int32Array([0, 0]), line: new Int32Array([10, 40]), function: ["build", "<module>"] },
      records: {
        path: ["opensees/section/Cols", "opensees/section/#1"],
        site: new Int32Array([0, -1]),
        script: new Int32Array([1, 1]),
        seq: new Int32Array([0, 1]),
      },
    },
  });
}

let n = 0;
function write(mutate: (t: Tree) => void): string {
  const t = valid();
  mutate(t);
  const path = join(tmp, `p-${n++}.h5`);
  writeTree(h5wasm, path, t);
  return path;
}

test("the synthetic file is valid; a Windows base_dir joins with /", () => {
  const p = readProvenance(h5, write(() => {}))!;
  assert.equal(site(sourceOf(p, "opensees/section/Cols")).file, "C:/runs/frame/model.py");
  const unnamed = sourceOf(p, "opensees/section/#1") as { site: SourceSite | null; script: SourceSite };
  assert.equal(unnamed.site, null);
  assert.equal(unnamed.script.line, 40);
});

const refuse = (mutate: (t: Tree) => void, pattern: RegExp) => () => {
  assert.throws(() => readProvenance(h5, write(mutate)), (e: unknown) => e instanceof SchemaError && pattern.test(e.message));
};
const rec = (t: Tree) => t.provenance!["records"] as Tree;

test("another major is refused", refuse((t) => (t.meta!.attrs!["provenance_schema_version"] = "2.1.0"), /provenance_schema_version 2\.1\.0: this app reads major 1 only/));
test("the key without the group is refused", refuse((t) => delete t.provenance, /provenance_schema_version is 1\.0\.0 but \/provenance is missing/));
test("the group without the key is refused", refuse((t) => delete t.meta!.attrs!["provenance_schema_version"], /\/provenance is present but/));
test("a duplicate declaration path is refused", refuse((t) => (rec(t)["path"] = ["a/b/c", "a/b/c"]), /records\/path has "a\/b\/c" twice/));
test("a site row out of range is refused", refuse((t) => (rec(t)["site"] = new Int32Array([0, 2])), /records\/site\[1\] = 2 is neither -1 nor a row of sites/));
test("columns of different lengths are refused", refuse((t) => (rec(t)["seq"] = new Int32Array([0])), /records: column lengths differ/));
test("a 0 line is refused", refuse((t) => ((t.provenance!["sites"] as Tree)["line"] = new Int32Array([0, 40])), /sites\/line\[0\] = 0; lines are 1-based/));
test("an unknown file kind is refused", refuse((t) => ((t.provenance!["files"] as Tree)["kind"] = ["notebook"]), /files\/kind\[0\] = "notebook"/));
test("a malformed sha256 is refused", refuse((t) => ((t.provenance!["files"] as Tree)["sha256"] = ["xyz"]), /sha256\[0\] is not a hex sha256/));
// Strict where h5-schema.md implies an invariant (Fable on #1316):
test("a backslash path is refused (paths are POSIX)", refuse((t) => ((t.provenance!["files"] as Tree)["path"] = ["sub\\model.py"]), /files\/path\[0\] = "sub\\\\model\.py" has a backslash; paths are POSIX/));
test("a backslash base_dir is refused", refuse((t) => (t.provenance!.attrs!["base_dir"] = "C:\\runs\\frame"), /@base_dir = .* has a backslash/));
test("a duplicate seq is refused (one capture order per declaration)", refuse((t) => (rec(t)["seq"] = new Int32Array([3, 3])), /records\/seq has 3 twice/));
test("a newer same-major minor opens with exactly one banner (ADR 0113 D7)", () => {
  const p = readProvenance(h5, write((t) => (t.meta!.attrs!["provenance_schema_version"] = "1.3.0")))!;
  assert.equal(p.version, "1.3.0");
  assert.deepEqual(p.warnings, ["provenance_schema_version 1.3.0 is newer than this app (1.0.x): the file opens, and what that apeGmsh added is not shown"]);
});
test("an int64 column is refused", refuse((t) => (rec(t)["seq"] = new BigInt64Array([0n, 1n])), /records\/seq is int64/));
