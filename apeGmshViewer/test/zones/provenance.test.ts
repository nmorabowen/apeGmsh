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
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openProvenance } from "../../src/main/zones.ts";
import type { ModelFile } from "../../src/model/types.ts";
import { openModel } from "../../src/reader/node.ts";
import { isRecorded, readProvenance, sourceOf, type ProvenanceZone, type SourceSite } from "../../src/reader/provenance.ts";
import { SchemaError, type H5Module } from "../../src/reader/read.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import { sourceFor, sourcesOf } from "../../src/state/selectors.ts";
import { tree, writeTree, type Tree } from "./tree.ts";

await h5wasm.ready;
const h5 = h5wasm as unknown as H5Module;
const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, "..", "..", "fixtures");
const repo = join(here, "..", "..", "..");
const tmp = mkdtempSync(join(tmpdir(), "agv-provenance-"));
after(() => rmSync(tmp, { recursive: true, force: true }));
const zonesModel: ModelFile = await openModel(join(fixtures, "zones.h5"));

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

// --- sha256 "" (#1435 item 1; orchestrator ruling on #1444) -----------------
// h5-schema.md, "Files": sha256 is "" for a pseudo-file (`<string>`, `<stdin>`,
// a notebook cell) and for a source the writer could not read (`_file_row`'s
// `except OSError`), so "" is valid for any path. Such a source opens only
// when main finds its path on disk; otherwise go-to-source is off with
// "source not recorded". Only a non-empty non-hex value is malformed.

const files = (t: Tree) => t.provenance!["files"] as Tree;
const oneFile = (path: string, sha: string) => (t: Tree) => {
  files(t)["path"] = [path];
  files(t)["sha256"] = [sha];
};
/** A /provenance zone as main answers it (with `present` filled from the disk). */
const viaMain = async (path: string): Promise<ProvenanceZone> => {
  const answer = await openProvenance(path);
  assert.ok(answer.ok, JSON.stringify(answer).slice(0, 300));
  return (answer as { zone: ProvenanceZone }).zone;
};
const stateWith = (zone: ProvenanceZone) =>
  reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(zonesModel, new BlobStore(), zone) });

for (const pseudo of ["<string>", "<stdin>", "<ipython-input-3-9f2c41ab>"]) {
  test(`a pseudo-file ${pseudo} with an empty sha256 opens, keeps its name, and is not recorded`, async () => {
    const p = await viaMain(write(oneFile(pseudo, "")));
    assert.deepEqual(p.files.sha256, [""]);
    const s = site(sourceOf(p, "opensees/section/Cols"));
    assert.equal(s.file, pseudo, "never joined to @base_dir");
    assert.equal(s.recorded, false);
    assert.deepEqual(sourceFor(stateWith(p), "opensees/section/Cols"), { ok: false, reason: `source not recorded (${pseudo})` });
  });
}

// (a) ipykernel 7 names a cell <tmp>/ipykernel_<pid>/<hash>.py and never
// writes it, so the writer records that absolute path with no digest.
const IPYKERNEL_CELL = "C:/Users/someone/AppData/Local/Temp/ipykernel_4242/2093384912.py";
test("(a) an ipykernel-style temp path with an empty sha256 opens; its go-to-source is off: source not recorded", async () => {
  const path = write(oneFile(IPYKERNEL_CELL, ""));
  assert.equal(readProvenance(h5, path)!.files.sha256[0], "", "the reader takes it");
  const p = await viaMain(path);
  assert.deepEqual(p.present, [false]);
  const s = site(sourceOf(p, "opensees/section/Cols"));
  assert.equal(s.file, IPYKERNEL_CELL);
  assert.equal(s.recorded, false);
  const state = stateWith(p);
  const reason = `source not recorded (${IPYKERNEL_CELL})`;
  assert.deepEqual(sourceFor(state, "opensees/section/Cols"), { ok: false, reason });
  const row = sourcesOf(state).find((r) => r.key === "opensees/section/Cols")!;
  assert.equal(row.label, null, "the button is disabled");
  assert.equal(row.off, reason, "and the row says why");
  // The model itself loads with no malformed warning.
  assert.deepEqual(state.artifacts.model!.zones["provenance"], { status: "ready", version: "1.0.0" });
});

test("an empty sha256 whose path is on disk still opens (no edit check: no digest to compare)", async () => {
  const cell = join(tmp, "cell.py");
  writeFileSync(cell, "x = 1\n");
  const posix = cell.replace(/\\/g, "/");
  const p = await viaMain(write(oneFile(posix, "")));
  assert.deepEqual(p.present, [true]);
  const w = sourceFor(stateWith(p), "opensees/section/Cols");
  assert.ok(w.ok, JSON.stringify(w));
  assert.equal(w.source.file, posix);
  assert.equal(w.source.sha256, "");
  // Read without main, the disk is unknown: not recorded, never guessed.
  assert.equal(site(sourceOf(readProvenance(h5, write(oneFile(posix, "")))!, "opensees/section/Cols")).recorded, false);
});

// (b) Only a non-empty value that is not 64 lower-case hex digits is malformed.
for (const [path, sha] of [["model.py", "xyz"], ["<string>", "xyz"], [IPYKERNEL_CELL, "a".repeat(63)], ["model.py", "A".repeat(64)]] as const) {
  test(`(b) sha256 ${JSON.stringify(sha.length > 8 ? `${sha.slice(0, 4)}…(${sha.length})` : sha)} for ${path} is refused`, refuse(oneFile(path, sha), /^\/provenance\/files\/sha256\[0\] is not a hex sha256$/));
}

test("a mixed table: a hashed file, a pseudo-file and an unreadable file all open; only the hashed one is recorded", async () => {
  const p = await viaMain(write((t) => {
    files(t)["path"] = ["model.py", "<string>", "/opt/helpers/frames.py"];
    files(t)["sha256"] = ["a".repeat(64), "", ""];
    files(t)["kind"] = ["module", "script", "module"];
  }));
  assert.deepEqual(p.files.sha256, ["a".repeat(64), "", ""]);
  assert.deepEqual([0, 1, 2].map((i) => isRecorded(p, i)), [true, false, false]);
  assert.equal(site(sourceOf(p, "opensees/section/Cols")).file, "C:/runs/frame/model.py");
});

test("shoebuckle.h5, written by the writer from `python -c`, opens: <string> has no digest", async () => {
  const path = join(fixtures, "shoebuckle.h5");
  const p = readProvenance(h5, path)!;
  assert.ok(p, "the regenerated fixture carries /provenance");
  assert.equal(p.version, "1.1.0");
  const at = p.files.path.indexOf("<string>");
  assert.ok(at >= 0, "the -c command is the script, recorded as <string>");
  assert.equal(p.files.sha256[at], "");
  assert.equal(p.files.kind[at], "script");
  // Every other file is real and hashed.
  p.files.path.forEach((f, i) => {
    if (i !== at) assert.match(p.files.sha256[i]!, /^[0-9a-f]{64}$/, f);
  });
  // A record's script frame is the `-c` line, kept as <string>.
  const withScript = p.records.path.find((_, i) => p.records.script[i]! >= 0 && p.sites.file[p.records.script[i]!] === at)!;
  const r = sourceOf(p, withScript) as { script: SourceSite; site: SourceSite };
  assert.equal(r.script.file, "<string>");
  assert.equal(r.script.recorded, false, "a pseudo-file never opens");
  assert.match(r.site.file, /\/examples\/shoebuckle_arch\.py$/);
  assert.equal(r.site.recorded, true, "a hashed file opens");
  // The main process answers it as read, not as malformed.
  const answer = await openProvenance(path);
  assert.equal(answer.ok, true, JSON.stringify(answer).slice(0, 300));
});
// Strict where h5-schema.md implies an invariant (Fable on #1316):
test("a backslash path is refused (paths are POSIX)", refuse((t) => ((t.provenance!["files"] as Tree)["path"] = ["sub\\model.py"]), /files\/path\[0\] = "sub\\\\model\.py" has a backslash; paths are POSIX/));
test("a backslash base_dir is refused", refuse((t) => (t.provenance!.attrs!["base_dir"] = "C:\\runs\\frame"), /@base_dir = .* has a backslash/));
test("a duplicate seq is refused (one capture order per declaration)", refuse((t) => (rec(t)["seq"] = new Int32Array([3, 3])), /records\/seq has 3 twice/));
test("a newer same-major minor opens with exactly one banner (ADR 0113 D7)", () => {
  const p = readProvenance(h5, write((t) => {
    t.meta!.attrs!["provenance_schema_version"] = "1.3.0";
    rec(t)["origin"] = ["user", "user"]; // required from 1.1.0
  }))!;
  assert.equal(p.version, "1.3.0");
  assert.deepEqual(p.warnings, ["provenance_schema_version 1.3.0 is newer than this app (1.1.x): the file opens, and what that apeGmsh added is not shown"]);
});

// --- 1.1.0: records/origin (#1378; V2g, #1425) ------------------------------

test("the 1.0.0 fixture reads every record as origin user", () => {
  const p = readProvenance(h5, join(fixtures, "zones.h5"))!;
  assert.equal(p.version, "1.0.0");
  assert.equal(p.records.origin.length, 7);
  assert.deepEqual([...new Set(p.records.origin)], ["user"]);
  assert.equal((sourceOf(p, "opensees/fix/#1") as { origin: string }).origin, "user");
});

const v110 = (t: Tree) => {
  t.meta!.attrs!["provenance_schema_version"] = "1.1.0";
  rec(t)["origin"] = ["user", "synthesised"];
};
test("a 1.1.0 file reads its origin column, with no banner", () => {
  const p = readProvenance(h5, write(v110))!;
  assert.deepEqual(p.warnings, []);
  assert.deepEqual(p.records.origin, ["user", "synthesised"]);
  assert.equal((sourceOf(p, "opensees/section/#1") as { origin: string }).origin, "synthesised");
});
test("a 1.1.0 file without origin is malformed", refuse((t) => {
  v110(t);
  delete rec(t)["origin"];
}, /^\/provenance\/records\/origin is missing \(required from provenance_schema_version 1\.1\.0; this file is 1\.1\.0\)$/));
test("a 1.2.0 file without origin is malformed too (the column stays required)", refuse((t) => {
  v110(t);
  t.meta!.attrs!["provenance_schema_version"] = "1.2.0";
  delete rec(t)["origin"];
}, /records\/origin is missing/));
test("an unknown origin is refused", refuse((t) => {
  v110(t);
  rec(t)["origin"] = ["user", "generated"];
}, /records\/origin\[1\] = "generated"; expected user or synthesised/));
test("an origin column of another length is refused", refuse((t) => {
  v110(t);
  rec(t)["origin"] = ["user"];
}, /records: column lengths differ/));
// The Python reader reads the column when a pre-1.1 file has one
// (_femdata_h5_io._read_provenance); so does the app.
test("a 1.0.0 file that carries the column is read as written", () => {
  const p = readProvenance(h5, write((t) => (rec(t)["origin"] = ["synthesised", "user"])))!;
  assert.deepEqual(p.records.origin, ["synthesised", "user"]);
  assert.equal(p.originColumn, true);
  assert.equal(readProvenance(h5, write(() => {}))!.originColumn, false);
});

// The fixture main's own writer produced (fixtures/README.md, "bridge_provenance.h5").
const readme = readFileSync(join(fixtures, "README.md"), "utf8").replace(/\r\n/g, "\n");
const bridgeScript = readme.match(/## `bridge_provenance\.h5`[\s\S]*?```python\n([\s\S]*?)```/)![1]!;
const bridgeLines = bridgeScript.split("\n");
const lineOf = (code: RegExp) => {
  const hits = bridgeLines.flatMap((l, i) => (code.test(l) ? [i + 1] : []));
  assert.equal(hits.length, 1, `${code} matches ${hits.length} lines of the README script`);
  return hits[0]!;
};

test("main's writer: the 1.1.0 file opens with no banner and marks the synthesised objects", () => {
  const p = readProvenance(h5, join(fixtures, "bridge_provenance.h5"))!;
  assert.equal(p.version, "1.1.0");
  assert.deepEqual(p.warnings, [], "no 'newer than this app' banner for /provenance 1.1.0");
  assert.equal(p.files.sha256[0], createHash("sha256").update(bridgeScript).digest("hex"), "the fixture was made by the README's script");
  const synthesised = p.records.path.filter((_, i) => p.records.origin[i] === "synthesised");
  assert.deepEqual(synthesised, ["opensees/timeSeries/support:gravity/hold", "opensees/pattern/support:gravity"]);
  assert.equal(p.records.origin.filter((o) => o === "user").length, p.records.path.length - 2);
  // Both point at the user's verb call, the `s.support(` line.
  const support = lineOf(/^\s+s\.support\(/);
  for (const key of synthesised) {
    const s = site(sourceOf(p, key));
    assert.equal(s.line, support, key);
    assert.equal(s.file, `${p.baseDir}/bridge_model.py`);
    assert.equal(s.kind, "script");
  }
  // A user declaration keeps its own line.
  assert.equal(site(sourceOf(p, "opensees/nDMaterial/concrete")).line, lineOf(/ops\.nDMaterial\.ElasticIsotropic\(/));
});
test("an int64 column is refused", refuse((t) => (rec(t)["seq"] = new BigInt64Array([0n, 1n])), /records\/seq is int64/));
