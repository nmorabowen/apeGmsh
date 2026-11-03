// reader/userFile.ts (ADR 0130 slice VU3a): the app reads `<stem>.user.h5`.
//
// The oracle is the layout of h5-schema.md "`<stem>.user.h5`" and the Python
// reader it describes (src/apeGmsh/results/user/_reader.py, _staleness.py and
// opensees/_internal/_combination_rules.py): each test writes a file as the
// schema lays it out, with h5wasm, and names the Python verdict it expects. A
// cross-language probe ran the same reader over files Python's own writer made
// (the PR body says how); this file holds the rules in CI.
//
// The staleness tests pair the user file with a copy of the repository's
// truss2d.ladruno, whose rf1 is a cross-language golden (test/reader/rf1.test.ts),
// or (VU7a, the `/modes` row rule) with fixtures/ladruno_modes.ladruno, whose two
// eigenvalues test/reader/userModes.test.ts holds as Python reads them.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { readUserFile, readUserSet, staleness, UserFileRefused, type UserH5, type UserIndex } from "../../src/reader/userFile.ts";
import { judge, nameProblem, type Combination } from "../../src/reader/userRules.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const TRUSS = fileURLToPath(new URL("../../../tests/fixtures/ladruno/truss2d.ladruno", import.meta.url));
const TRUSS_RF1 = "rf1:76a404478cbb9ef919c2d79a676fbe1a0a300712819b344edd9a43e7289cfcc3";
const MODES = fileURLToPath(new URL("../../fixtures/ladruno_modes.ladruno", import.meta.url));
/** fixtures/ladruno_modes.ladruno's LAMBDA of modes 1 and 2 on stage_0 STEP_0, as Python's `recorded_lambdas` reads them. */
const LAMBDAS = [381.9660112501052, 2618.0339887498953];
const FEM = "fedcba9876543210fedcba9876543210";
const MODEL = "0123456789abcdef0123456789abcdef";
const list = (d: string) => readdirSync(d);

type G = h5wasm.Group;
const at = (f: G, path: string) => f.get(path) as G;
const str = (g: G, name: string, data: string[]) => g.create_dataset({ name, data, dtype: "S" });
const i4 = (g: G, name: string, data: number[], shape?: number[]) => g.create_dataset({ name, data: Int32Array.from(data), shape: shape ?? [data.length] });
const f8 = (g: G, name: string, data: number[], shape?: number[]) => g.create_dataset({ name, data: Float64Array.from(data), shape: shape ?? [data.length] });

interface Row {
  name: string;
  rule: string;
  terms: [string, string, number, number, number][];
  envelope?: string;
  fraction?: number;
  damping?: number;
}

/** One `g<N>` (h5-schema.md `/combinations`). */
function generation(g: G, rows: Row[]): void {
  str(g, "name", rows.map((r) => r.name));
  str(g, "rule", rows.map((r) => r.rule));
  str(g, "envelope", rows.map((r) => r.envelope ?? ""));
  f8(g, "fraction", rows.map((r) => r.fraction ?? NaN));
  f8(g, "damping", rows.map((r) => r.damping ?? NaN));
  const offsets = [0];
  for (const r of rows) offsets.push(offsets.at(-1)! + r.terms.length);
  i4(g, "term_offsets", offsets);
  const t = g.create_group("terms") as G;
  const terms = rows.flatMap((r) => r.terms);
  str(t, "kind", terms.map((x) => x[0]));
  str(t, "source", terms.map((x) => x[1]));
  i4(t, "step", terms.map((x) => x[2]));
  i4(t, "mode", terms.map((x) => x[3]));
  f8(t, "factor", terms.map((x) => x[4]));
}

const ROWS: Row[] = [
  { name: "1.2D+1.6L", rule: "linear", terms: [["stage", "D", -1, 0, 1.2], ["stage", "L", -1, 0, 1.6]] },
  { name: "spec", rule: "srss", terms: [["spectrum", "c", -1, 0, 1]] },
  { name: "via spec", rule: "linear", terms: [["combination", "spec", -1, 0, 1], ["stage", "D", -1, 0, 1]] },
  { name: "uses model", rule: "linear", terms: [["combination", "M1", -1, 0, 2]] },
];

/** A holder of `values` (h5-schema.md `/results/r<k>/RESULTS/ON_*`), DATA shaped (s, n, c). */
function holder(g: G, s: number, n: number, c: number, target: string): void {
  f8(g, "DATA", Array.from({ length: s * n * c }, (_, k) => k / 7), [s, n, c]);
  i4(g, "STEP", Array.from({ length: s }, (_, k) => k));
  f8(g, "TIME", Array.from({ length: s }, (_, k) => k));
  if (target === "nodes" || target === "elements") i4(g, "ID", Array.from({ length: n }, (_, k) => k + 1), [n, 1]);
  if (target === "entities") {
    str(g, "ENTITY_KIND", Array.from({ length: n }, () => "cut"));
    i4(g, "ENTITY_ROW", Array.from({ length: n }, (_, k) => k));
  }
  if (target !== "elements") g.create_attribute("COMPONENTS", Array.from({ length: c }, (_, k) => `c${k}`).join(","));
}

interface EntryOpts {
  name: string;
  target: "nodes" | "elements" | "entities" | "domain";
  basis?: "results" | "model";
  states?: [string, string, number, number][];
}

function entry(results: G, link: string, o: EntryOpts): G {
  const g = results.create_group(link) as G;
  g.create_attribute("name", o.name);
  g.create_attribute("KIND", "static");
  g.create_attribute("target", o.target);
  g.create_attribute("basis", o.basis ?? "model");
  g.create_attribute("per_state", Int8Array.of(o.states ? 1 : 0));
  const s = o.states?.length ?? 1;
  if (o.states) {
    const st = g.create_group("STATES") as G;
    str(st, "kind", o.states.map((x) => x[0]));
    str(st, "source", o.states.map((x) => x[1]));
    i4(st, "step", o.states.map((x) => x[2]));
    i4(st, "mode", o.states.map((x) => x[3]));
  }
  const on = { nodes: "ON_NODES", elements: "ON_ELEMENTS", entities: "ON_ENTITIES", domain: "ON_DOMAIN" }[o.target];
  const values = (g.create_group("RESULTS") as G).create_group(on) as G;
  const v = values.create_group("values") as G;
  if (o.target === "elements") {
    const b = v.create_group("204-ASDShellT3[102:0:0]") as G;
    holder(b, s, 2, 6, "elements");
    const cmap = b.create_group("COLUMN_MAP") as G;
    cmap.create_attribute("COMP_NAMES", "mxx,myy,mxy");
    i4(cmap, "MULTIPLICITY", [2], [1, 1]);
    i4(cmap, "GAUSS_ID", [0], [1, 1]);
    const placed = ((g.create_group("MODEL") as G).create_group("ELEMENTS") as G).create_group("204-ASDShellT3[102:0:0]") as G;
    placed.create_attribute("TOPOLOGY", "tri");
    placed.create_attribute("PARAM_DOMAIN", "bary");
    placed.create_attribute("FAMILY", "lagrange");
    placed.create_attribute("ORDER", Int32Array.of(1, 1));
    placed.create_attribute("NUM_GP", Int32Array.of(2));
    f8(placed.create_group("QUADRATURE") as G, "GP_PARAM", [0.2, 0.3, 0.1, 0.6], [2, 2]);
  } else {
    holder(v, s, o.target === "domain" ? 1 : 3, 2, o.target);
  }
  return g;
}

/** A two-mode `/modes` table (h5-schema.md `/modes`). */
function modesTable(z: G, drop: string | null = null, lam: number[] = [4, 9], step = 1): void {
  z.create_attribute("components", "MX,MY");
  const cols: [string, (g: G, n: string) => unknown][] = [
    ["host", (g, n) => str(g, n, ["stage_0", "stage_0"])],
    ["kind", (g, n) => str(g, n, ["eigen", "eigen"])],
    ["step", (g, n) => i4(g, n, [step, step])],
    ["mode", (g, n) => i4(g, n, [1, 2])],
    ["lambda", (g, n) => f8(g, n, lam)],
    ["period", (g, n) => f8(g, n, lam.map((x) => (2 * Math.PI) / Math.sqrt(Math.abs(x) || 1)))],
    ["generalized_mass", (g, n) => f8(g, n, [1, 1])],
    ["participation", (g, n) => f8(g, n, [0.9, 0.1, 0.1, 0.8], [2, 2])],
    ["mass_ratio", (g, n) => f8(g, n, [0.7, 0.0, 0.0, 0.6], [2, 2])],
    ["cumulative_ratio", (g, n) => f8(g, n, [0.7, 0.0, 0.7, 0.6], [2, 2])],
  ];
  for (const [name, write] of cols) if (name !== drop) write(z, name);
}

/** One result of a spectrum case: `(on, label, ids or null, components)`. */
type Quantity = [string, string, number[] | null, string[]];

function quantities(root: G, s: number, qs: Quantity[]): void {
  const results = root.create_group("RESULTS") as G;
  const families = new Map<string, G>();
  for (const [on, label, ids, comps] of qs) {
    if (!families.has(on)) families.set(on, results.create_group(on) as G);
    const h = families.get(on)!.create_group(label) as G;
    const n = ids?.length ?? 1;
    f8(h, "DATA", Array.from({ length: s * n * comps.length }, (_, k) => k / 3), [s, n, comps.length]);
    i4(h, "STEP", Array.from({ length: s }, (_, k) => k));
    f8(h, "TIME", Array.from({ length: s }, (_, k) => k));
    if (ids) i4(h, "ID", ids, [n, 1]);
    h.create_attribute("COMPONENTS", comps.join(","));
  }
}

interface CaseOpts {
  name?: string;
  kind?: string;
  direction?: number;
  modes?: number[];
  damping?: number[];
  periods?: number[];
  results?: Quantity[];
  reference?: Quantity[];
  scale?: number;
  /** the gamma and lambda columns as given (their length is judged by the case's rules) */
  gamma?: number[];
  lambda?: number[];
  /** the mode column as int64, which a case refuses */
  modeI8?: boolean;
}

const BASE_SHEAR: Quantity = ["ON_DOMAIN", "base_shear", null, ["Vx", "Vy"]];
const DISP: Quantity = ["ON_NODES", "DISPLACEMENT", [1, 2, 3], ["Ux", "Uy"]];

/** `/spectra/<link>` as h5-schema.md lays it out (and Python's `write_spectra` writes it). */
function spectrumCase(z: G, link: string, o: CaseOpts = {}): G {
  const g = z.create_group(link) as G;
  g.create_attribute("name", o.name ?? "Ex");
  g.create_attribute("KIND", o.kind ?? "static");
  g.create_attribute("direction", Int32Array.of(o.direction ?? 1));
  g.create_attribute("scale", Float64Array.of(o.scale ?? 9.81));
  const sp = g.create_group("spectrum") as G;
  f8(sp, "T", o.periods ?? [0, 0.5, 1, 2]);
  f8(sp, "Sa", [0.4, 1, 0.6, 0.3]);
  const modes = o.modes ?? [1, 2];
  const m = g.create_group("modes") as G;
  if (o.modeI8) m.create_dataset({ name: "mode", data: BigInt64Array.from(modes.map(BigInt)), shape: [modes.length] });
  else i4(m, "mode", modes);
  str(m, "kind", modes.map(() => "eigen"));
  f8(m, "lambda", o.lambda ?? modes.map((k) => LAMBDAS[k - 1] ?? 100 * k));
  f8(m, "sa_period", modes.map(() => 0));
  f8(m, "gamma", o.gamma ?? modes.map((k) => 1 / k));
  f8(m, "sa", modes.map(() => 5));
  f8(m, "damping", o.damping ?? modes.map(() => 0.05));
  const results = o.results ?? [BASE_SHEAR, DISP];
  quantities(g, modes.length, results);
  quantities(g.create_group("reference") as G, 1, o.reference ?? results);
  return g;
}

interface UserOpts {
  resultsHash?: string | null;
  femHash?: string;
  /** a version per zone key, or null to leave the key out */
  versions?: Record<string, string | null>;
  rows?: Row[];
  /** writes the `/modes` group (default: a valid two-mode table) */
  modes?: (z: G) => void;
  /** writes the `/spectra` group (default: one valid case, `Ex`) */
  spectra?: (z: G) => void;
  edit?: (f: h5wasm.File) => void;
}

const dir = () => mkdtempSync(join(tmpdir(), "agv-user-"));

/** A `<stem>.user.h5` laid out as h5-schema.md says, with every zone this slice reads. */
function writeUser(path: string, o: UserOpts = {}): string {
  const f = new h5wasm.File(path, "w");
  try {
    const meta = f.create_group("meta") as G;
    const versions: Record<string, string | null> = {
      user_meta_schema_version: "1.0.0",
      combinations_schema_version: "1.0.0",
      user_results_schema_version: "1.0.0",
      modes_schema_version: "1.0.0",
      spectra_schema_version: "1.0.0",
      ...o.versions,
    };
    for (const [k, v] of Object.entries(versions)) if (v !== null) meta.create_attribute(k, v);
    meta.create_attribute("fem_hash", o.femHash ?? FEM);
    meta.create_attribute("model_hash", MODEL);
    if (o.resultsHash !== null) meta.create_attribute("results_hash", o.resultsHash ?? TRUSS_RF1);
    meta.create_attribute("writer", "test");
    meta.create_attribute("revision", Int32Array.of(3));
    const c = f.create_group("combinations") as G;
    c.create_attribute("current", Int32Array.of(1));
    generation(c.create_group("g0") as G, [ROWS[0]!]); // a backup generation, not in force
    generation(c.create_group("g1") as G, o.rows ?? ROWS);
    const r = f.create_group("results") as G;
    entry(r, "r0", { name: "Delta_after (L/480)", target: "nodes", states: [["stage", "gravity", -1, 0], ["combination", "1.2D+1.6L", -1, 0]] });
    entry(r, "r1", { name: "wood_armer", target: "elements" });
    entry(r, "r2", { name: "pier P1 base", target: "entities" });
    entry(r, "r10", { name: "energy", target: "domain", basis: "results" });
    (o.modes ?? ((z: G) => modesTable(z)))(f.create_group("modes") as G);
    (o.spectra ?? ((z: G) => spectrumCase(z, "s0")))(f.create_group("spectra") as G);
    o.edit?.(f);
  } finally {
    f.close();
  }
  return path;
}

/** A model.h5 holding only what the pairing reads, and optionally `/opensees/combinations/g0`. */
function writeModel(path: string, o: { fem?: string; model?: string | null; rows?: Row[] } = {}): string {
  const f = new h5wasm.File(path, "w");
  try {
    const meta = f.create_group("meta") as G;
    meta.create_attribute("opensees_schema_version", "2.39.0");
    const lineage = meta.create_group("lineage") as G;
    lineage.create_attribute("fem_hash", o.fem ?? FEM);
    if (o.model !== null) lineage.create_attribute("model_hash", o.model ?? MODEL);
    if (o.rows) generation(((f.create_group("opensees") as G).create_group("combinations") as G).create_group("g0") as G, o.rows);
  } finally {
    f.close();
  }
  return path;
}

const M1: Row = { name: "M1", rule: "linear", terms: [["stage", "D", -1, 0, 1]] };

/** The set: user file, model (with M1) and a copy of truss2d.ladruno. */
function set(o: UserOpts = {}, model: Parameters<typeof writeModel>[1] = { rows: [M1] }) {
  const d = dir();
  const results = join(d, "run.ladruno");
  copyFileSync(TRUSS, results);
  return { user: writeUser(join(d, "run.user.h5"), o), model: writeModel(join(d, "run.h5"), model), results, dir: d };
}

function ok<T>(z: { ok: true; value: T } | { ok: false; reason: string }): T {
  assert.ok(z.ok, z.ok ? "" : z.reason);
  return z.value;
}

// ---- the file and its zones ----------------------------------------------

test("a set's user file reads: /meta, the generation in force, the /results index", () => {
  const s = set();
  const u = readUserFile(H5, s.user, s.model);
  assert.equal(u.femHash, FEM);
  assert.equal(u.modelHash, MODEL);
  assert.equal(u.resultsHash, TRUSS_RF1);
  assert.equal(u.revision, 3);
  assert.equal(u.writer, "test");
  assert.deepEqual(u.zones, ["combinations", "modes", "results", "spectra"]);
  assert.deepEqual(u.warnings, []);
  const c = ok(u.combinations);
  assert.equal(c.generation, 1);
  assert.deepEqual(c.rows.map((r) => r.name), ["1.2D+1.6L", "spec", "via spec", "uses model"]);
  assert.deepEqual(c.modelRows.map((r) => r.name), ["M1"]);
  assert.deepEqual(c.refused, []);
  assert.deepEqual(c.rows[0], { name: "1.2D+1.6L", rule: "linear", envelope: "", fraction: null, damping: null, terms: [
    { kind: "stage", source: "D", step: -1, mode: 0, factor: 1.2 },
    { kind: "stage", source: "L", step: -1, mode: 0, factor: 1.6 },
  ] });
  const r = ok(u.results);
  // Links sort by number (r10 after r2), as Python's.
  assert.deepEqual(r.entries.map((e) => [e.name, e.target, e.basis, e.perState, e.link]), [
    ["Delta_after (L/480)", "nodes", "model", true, "r0"],
    ["wood_armer", "elements", "model", false, "r1"],
    ["energy", "domain", "results", false, "r10"],
  ]);
  assert.deepEqual(r.entries[0]!.states, [
    { kind: "stage", source: "gravity", step: -1, mode: 0 },
    { kind: "combination", source: "1.2D+1.6L", step: -1, mode: 0 },
  ]);
  // The file holds no /entities, so an ON_ENTITIES entry names a row it does not hold (Python's `_bound_entries`).
  assert.deepEqual(r.refused.map((x) => [x.name, x.reason, x.row, x.detail]), [["pier P1 base", "unresolved", 2, "user result 'pier P1 base' names cut row 0, and /entities/cuts holds 0"]]);
});

test("the index reads no /results or /spectra DATA value (the first-frame criterion: only shapes)", (t: TestContext) => {
  const s = set();
  const read: string[] = [];
  const proto = h5wasm.Dataset.prototype;
  const value = Object.getOwnPropertyDescriptor(proto, "value")!.get!;
  t.mock.getter(proto, "value", function (this: h5wasm.Dataset) {
    read.push(this.path);
    return value.call(this);
  });
  const slice = proto.slice;
  t.mock.method(proto, "slice", function (this: h5wasm.Dataset, ranges: Parameters<typeof slice>[0]) {
    read.push(this.path);
    return slice.call(this, ranges);
  });
  const r = readUserSet(H5, s.user, s.model, s.results, list);
  assert.ok(r.ok, r.ok ? "" : r.error);
  assert.ok(read.some((p) => p.startsWith("/combinations/")), "the probe sees reads (the mock is live)");
  assert.deepEqual(read.filter((p) => /^\/results\/.*\/DATA$/.test(p)), []);
  // /modes is a small table read whole, as Python's staleness reads it; a /spectra case is read but for its DATA.
  assert.ok(read.some((p) => p === "/spectra/s0/modes/lambda"), "the case's modes table is read");
  assert.deepEqual(read.filter((p) => /^\/spectra\/.*\/DATA$/.test(p)), []);
});

test("a bad /meta refuses the file, naming why (Python's UserFileRefused)", () => {
  const d = dir();
  const cases: [string, UserOpts, RegExp][] = [
    ["noversion", { versions: { user_meta_schema_version: null } }, /no \/meta@user_meta_schema_version: it is not a \.user\.h5/],
    ["major", { versions: { user_meta_schema_version: "2.0.0" } }, /user_meta_schema_version 2\.0\.0: this app reads major 1 only/],
    ["nofem", { femHash: "" }, /\/meta@fem_hash is required/],
  ];
  for (const [name, o, why] of cases) assert.throws(() => readUserFile(H5, writeUser(join(d, `${name}.user.h5`), o), null), (e: unknown) => e instanceof UserFileRefused && why.test((e as Error).message), name);
  const bare = new h5wasm.File(join(d, "bare.user.h5"), "w");
  bare.create_group("results");
  bare.close();
  assert.throws(() => readUserFile(H5, join(d, "bare.user.h5"), null), /has no \/meta group/);
  assert.throws(() => readUserFile(H5, join(d, "absent.user.h5"), null), /no such file/);
  // A model.h5 opened as a user file is refused, never read as one.
  assert.throws(() => readUserFile(H5, writeModel(join(d, "x.user.h5")), null), /no \/meta@user_meta_schema_version/);
  const r = readUserSet(H5, join(d, "absent.user.h5"), null, null, list);
  assert.ok(!r.ok && /no such file/.test(r.error), "readUserSet answers, it does not raise");
});

test("a zone outside its range, or a key and a group that disagree, refuses that zone alone", () => {
  const d = dir();
  const major = readUserFile(H5, writeUser(join(d, "a.user.h5"), { versions: { combinations_schema_version: "2.0.0" } }), null);
  assert.ok(!major.combinations.ok && /combinations_schema_version 2\.0\.0: this app reads major 1 only/.test(major.combinations.reason));
  assert.ok(major.results.ok, "the other zones still read");
  const nokey = readUserFile(H5, writeUser(join(d, "b.user.h5"), { versions: { user_results_schema_version: null } }), null);
  assert.ok(!nokey.results.ok && /\/results: \/meta has no version key for zone 'user_results'/.test(nokey.results.reason));
  const nozone = readUserFile(H5, writeUser(join(d, "c.user.h5"), { edit: (f) => at(f, "meta").create_attribute("combinations_schema_version_x", "1") }), null);
  assert.ok(nozone.combinations.ok);
  const stamped = new h5wasm.File(join(d, "e.user.h5"), "w");
  const m = stamped.create_group("meta") as G;
  for (const [k, v] of [["user_meta_schema_version", "1.0.0"], ["user_results_schema_version", "1.0.0"], ["fem_hash", FEM]]) m.create_attribute(k!, v!);
  m.create_attribute("revision", Int32Array.of(0));
  stamped.close();
  const ghost = readUserFile(H5, join(d, "e.user.h5"), null);
  assert.ok(!ghost.results.ok && /stamps zone 'user_results' but the file has no \/results/.test(ghost.results.reason));
  assert.deepEqual(ok(ghost.combinations), { generation: null, rows: [], modelRows: [], refused: [] }, "an absent zone is empty, not refused");
  // A newer minor opens, with one banner.
  const newer = readUserFile(H5, writeUser(join(d, "f.user.h5"), { versions: { combinations_schema_version: "1.4.0" } }), null);
  assert.ok(newer.combinations.ok);
  assert.equal(newer.warnings.length, 1);
  assert.match(newer.warnings[0]!, /combinations_schema_version 1\.4\.0 is newer than this app/);
});

test("malformed columns refuse /combinations; @current must name a generation", () => {
  const d = dir();
  const wrongType = readUserFile(H5, writeUser(join(d, "a.user.h5"), {
    edit: (f) => {
      const g = at(f, "combinations").create_group("g7") as G;
      generation(g, ROWS);
      at(f, "combinations").delete_attribute("current");
      at(f, "combinations").create_attribute("current", Int32Array.of(9));
    },
  }), null);
  assert.ok(!wrongType.combinations.ok && /@current names g9, which it does not hold/.test(wrongType.combinations.reason));
  const offsets = readUserFile(H5, writeUser(join(d, "b.user.h5"), {
    edit: (f) => {
      const c = at(f, "combinations");
      c.delete_attribute("current");
      c.create_attribute("current", Int32Array.of(2));
      const g = c.create_group("g2") as G;
      str(g, "name", ["a"]);
      str(g, "rule", ["linear"]);
      str(g, "envelope", [""]);
      f8(g, "fraction", [NaN]);
      f8(g, "damping", [NaN]);
      i4(g, "term_offsets", [0, 2]);
      const t = g.create_group("terms") as G;
      str(t, "kind", ["stage"]);
      str(t, "source", ["D"]);
      i4(t, "step", [-1]);
      i4(t, "mode", [0]);
      f8(t, "factor", [1]);
    },
  }), null);
  assert.ok(!offsets.combinations.ok && /term_offsets does not partition 1 terms/.test(offsets.combinations.reason));
});

// ---- rows refuse alone (A3), as Python's judge ----------------------------

test("a bad row refuses alone with Python's reason; a clash with model.h5 refuses both rows", () => {
  const rows: Row[] = [
    { name: "ok", rule: "linear", terms: [["stage", "D", -1, 0, 1]] },
    { name: "dup", rule: "linear", terms: [["stage", "D", -1, 0, 1]] },
    { name: "dup", rule: "linear", terms: [["stage", "L", -1, 0, 1]] },
    { name: "median", rule: "median", terms: [["stage", "D", -1, 0, 1]] },
    { name: "a", rule: "linear", terms: [["combination", "b", -1, 0, 1]] },
    { name: "b", rule: "linear", terms: [["combination", "a", -1, 0, 1]] },
    { name: "reaches cycle", rule: "linear", terms: [["combination", "a", -1, 0, 1]] },
    { name: "dangling", rule: "linear", terms: [["combination", "nope", -1, 0, 1]] },
    { name: "via dangling", rule: "linear", terms: [["combination", "dangling", -1, 0, 1]] },
    { name: "M1", rule: "linear", terms: [["stage", "D", -1, 0, 1]] },
    { name: " padded", rule: "linear", terms: [["stage", "D", -1, 0, 1]] },
    { name: "cqc modes", rule: "cqc", terms: [["mode", "eig", -1, 1, 1], ["mode", "eig", -1, 2, 1]] },
    { name: "bad step", rule: "linear", terms: [["combination", "ok", 3, 0, 1]] },
  ];
  const s = set({ rows });
  const c = ok(readUserFile(H5, s.user, s.model).combinations);
  assert.deepEqual(c.rows.map((r) => r.name), ["ok"]);
  assert.deepEqual(c.modelRows, [], "M1 clashes, so the model's row goes too");
  assert.deepEqual(c.refused.map((x) => [x.name, x.reason, x.origin, x.row]), [
    ["dup", "duplicate", "user", 1],
    ["dup", "duplicate", "user", 2],
    ["median", "unknown-rule", "user", 3],
    ["a", "cycle", "user", 4],
    ["b", "cycle", "user", 5],
    ["reaches cycle", "cycle", "user", 6],
    ["dangling", "unresolved", "user", 7],
    ["via dangling", "unresolved", "user", 8],
    ["M1", "clash", "user", 9],
    [" padded", "empty-name", "user", 10],
    ["cqc modes", "bad-param", "user", 11],
    ["bad step", "bad-term", "user", 12],
    ["M1", "clash", "model", 0],
  ]);
});

test("a model.h5 row never sees a .user.h5 row, and cannot name a case", () => {
  const user: Combination = { name: "u", rule: "linear", terms: [{ kind: "stage", source: "D", step: -1, mode: 0, factor: 1 }], envelope: "", fraction: null, damping: null };
  const model = (terms: Combination["terms"]): Combination => ({ name: "m", rule: "linear", terms, envelope: "", fraction: null, damping: null });
  const v1 = judge([["user", user], ["model", model([{ kind: "combination", source: "u", step: -1, mode: 0, factor: 1 }])]]);
  assert.deepEqual([...v1], [[1, ["unresolved", 'combination "u" does not stand']]]);
  const v2 = judge([["model", model([{ kind: "spectrum", source: "c", step: -1, mode: 0, factor: 1 }])]]);
  assert.equal(v2.get(0)![0], "bad-term");
});

test("the name rule is A2's: non-empty NFC, no control character, no outer whitespace", () => {
  assert.equal(nameProblem("Delta_after (L/480)"), null);
  assert.equal(nameProblem("Δ réal"), null);
  assert.equal(nameProblem(""), "empty");
  assert.equal(nameProblem("é"), "not NFC");
  assert.equal(nameProblem("a\tb"), "has a control character");
  assert.equal(nameProblem("a\u0085"), "has a control character");
  assert.equal(nameProblem("a "), "has leading or trailing whitespace");
  assert.equal(nameProblem("　a"), "has leading or trailing whitespace");
  assert.equal(nameProblem("\ud800"), "not valid UTF-8");
  assert.equal(nameProblem(3), "not a string: 3");
});

// ---- /results entries refuse alone ----------------------------------------

test("a bad /results entry refuses alone; two entries of one name both refuse", () => {
  const s = set({
    edit: (f) => {
      const r = at(f, "results");
      entry(r, "r3", { name: "energy", target: "domain" }); // a second "energy"
      const bad = entry(r, "r4", { name: "bad target", target: "nodes" });
      bad.delete_attribute("target");
      bad.create_attribute("target", "bogus");
      const term = entry(r, "r5", { name: "bad state", target: "nodes", states: [["stage", "D", 4, 1]] });
      void term;
      const shape = entry(r, "r6", { name: "bad shape", target: "nodes", states: [["stage", "D", -1, 0], ["stage", "L", -1, 0]] });
      shape.delete_attribute("per_state");
      shape.create_attribute("per_state", Int8Array.of(0)); // DATA axis 0 is 2, not 1
      const noplace = entry(r, "r7", { name: "unplaced", target: "elements" });
      (at(noplace, "MODEL/ELEMENTS/204-ASDShellT3[102:0:0]")).delete_attribute("FAMILY");
    },
  });
  const r = ok(readUserFile(H5, s.user, s.model).results);
  assert.deepEqual(r.entries.map((e) => e.name), ["Delta_after (L/480)", "wood_armer"]);
  assert.deepEqual(r.refused.map((x) => [x.name, x.reason]).sort(), [
    ["bad shape", "bad-param"],
    ["bad state", "bad-term"],
    ["bad target", "bad-param"],
    ["energy", "duplicate"],
    ["energy", "duplicate"],
    ["pier P1 base", "unresolved"],
    ["unplaced", "bad-param"],
  ]);
  assert.match(r.refused.find((x) => x.name === "bad shape")!.detail, /has 2 rows on axis 0, expected 1/);
  assert.match(r.refused.find((x) => x.name === "unplaced")!.detail, /has no @FAMILY/);
});

// ---- staleness (A6) --------------------------------------------------------

function verdict(u: UserIndex, model: string | null, results: string | null) {
  const st = staleness(H5, u, model, results, list);
  return {
    zones: Object.fromEntries(Object.entries(st.zones).map(([k, v]) => [k, v.stale])),
    results: Object.fromEntries(Object.entries(st.results).map(([k, v]) => [k, v.stale])),
    combinations: st.combinations,
    st,
  };
}

test("a matching set is current: rf1 equals the stored results_hash, the model hashes agree", () => {
  const s = set();
  const v = verdict(readUserFile(H5, s.user, s.model), s.model, s.results);
  assert.deepEqual(v.zones, { meta: false, modes: false, spectra: false, results: false, combinations: false });
  assert.deepEqual(v.results, { "Delta_after (L/480)": false, wood_armer: false, energy: false });
  assert.deepEqual(v.combinations, []);
});

test("a re-run stales only the @basis=results entries; a combination is never stale itself", () => {
  const s = set();
  const f = new h5wasm.File(s.results, "a");
  const data = f.get("MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA") as h5wasm.Dataset;
  const t = data.shape![0]!;
  const last = data.slice([[t - 1, t]]) as Float64Array;
  data.write_slice([[t - 1, t]], last.map((x) => x + 1));
  f.close();
  const v = verdict(readUserFile(H5, s.user, s.model), s.model, s.results);
  assert.deepEqual(v.results, { "Delta_after (L/480)": false, wood_armer: false, energy: true });
  assert.equal(v.zones["results"], true);
  assert.equal(v.zones["combinations"], false);
  assert.match(v.st.results["energy"]!.reason, /^results_hash differs: rf1:76a4.* here, rf1:[0-9a-f]{64} on disk$/);
});

test("another model stales the model zones, the @basis=model entries, and the combinations whose source is a stale case", () => {
  const s = set();
  writeModel(s.model, { fem: "00000000000000000000000000000000", rows: [M1] });
  const v = verdict(readUserFile(H5, s.user, s.model), s.model, s.results);
  assert.deepEqual(v.zones, { meta: false, modes: true, spectra: true, results: true, combinations: false });
  assert.deepEqual(v.results, { "Delta_after (L/480)": true, wood_armer: true, energy: false });
  // "spec" reads a spectrum case; "via spec" reaches it.
  assert.deepEqual(v.combinations, ["spec", "via spec"]);
  assert.match(v.st.zones["modes"]!.reason, /^fem_hash differs/);
});

test("model_hash compares only when both sides have it", () => {
  const s = set({}, { model: null, rows: [M1] });
  assert.equal(verdict(readUserFile(H5, s.user, s.model), s.model, s.results).zones["modes"], false);
});

test("what cannot be compared is stale: no files, no results_hash, a results file rf1 refuses", () => {
  const s = set();
  const u = readUserFile(H5, s.user, s.model);
  const none = verdict(u, null, null);
  assert.deepEqual(none.zones, { meta: false, modes: true, spectra: true, results: true, combinations: false });
  assert.equal(none.st.zones["modes"]!.reason, "no model file to compare with");
  assert.equal(none.st.results["energy"]!.reason, "no results file to compare with");
  const unhashed = readUserFile(H5, writeUser(join(s.dir, "x.user.h5"), { resultsHash: null }), s.model);
  assert.equal(verdict(unhashed, s.model, s.results).st.results["energy"]!.reason, "this file has no results_hash");
  const notHdf5 = join(s.dir, "broken.ladruno");
  copyFileSync(s.model, notHdf5);
  const v = verdict(u, s.model, notHdf5);
  assert.equal(v.results["energy"], true);
  assert.match(v.st.results["energy"]!.reason, /^the results file cannot be fingerprinted: /);
  const noModel = verdict(u, join(s.dir, "absent.h5"), s.results);
  assert.match(noModel.st.zones["modes"]!.reason, /^the model hashes cannot be read: no such model archive/);
});

test("a refused zone reports stale with the refusal, and the listing still answers", () => {
  const s = set({ versions: { combinations_schema_version: "9.0.0" } });
  const v = verdict(readUserFile(H5, s.user, s.model), s.model, s.results);
  assert.equal(v.zones["combinations"], true);
  assert.match(v.st.zones["combinations"]!.reason, /^refused: \/combinations: combinations_schema_version 9\.0\.0/);
  assert.equal(v.zones["results"], false);
});

test("a model whose /opensees/combinations cannot be read refuses the combinations zone", () => {
  const s = set();
  const broken = new h5wasm.File(s.model, "w");
  const meta = broken.create_group("meta") as G;
  (meta.create_group("lineage") as G).create_attribute("fem_hash", FEM);
  ((broken.create_group("opensees") as G).create_group("combinations") as G).create_group("g9");
  broken.close();
  const u = readUserFile(H5, s.user, s.model);
  assert.ok(!u.combinations.ok && /\/opensees\/combinations: the archive has no opensees_schema_version/.test(u.combinations.reason));
});

// Review of #2729, finding 2: Python's staleness reads /modes (`_mode_rows` ->
// `UserFile.modes()`), so a zone that refuses there is stale with the refusal.
test("/modes is checked as a zone: a missing key, another major, a missing column or a bad table stales it", () => {
  const good = set();
  const u = readUserFile(H5, good.user, good.model);
  const table = ok(u.modes!);
  assert.deepEqual(table.components, ["MX", "MY"]);
  assert.deepEqual(table.rows.map((r) => [r.host, r.kind, r.step, r.mode, r.lambda, r.generalizedMass]), [["stage_0", "eigen", 1, 1, 4, 1], ["stage_0", "eigen", 1, 2, 9, 1]]);
  // The (M, C) tables are row-major: row i holds one value per component.
  assert.deepEqual(table.rows.map((r) => [r.participation, r.massRatio, r.cumulativeRatio]), [[[0.9, 0.1], [0.7, 0], [0.7, 0]], [[0.1, 0.8], [0, 0.6], [0.7, 0.6]]]);
  const cases: [string, UserOpts, RegExp][] = [
    ["key gone", { versions: { modes_schema_version: null } }, /^refused: \/modes: \/meta has no version key for zone 'modes'$/],
    ["major 2", { versions: { modes_schema_version: "2.0.0" } }, /^refused: \/modes: modes_schema_version 2\.0\.0: this app reads major 1 only$/],
    ["column gone", { modes: (z) => modesTable(z, "period") }, /^refused: \/modes: it has no period column$/],
    ["lambda <= 0", { modes: (z) => modesTable(z, null, [4, -1]) }, /^refused: \/modes: every lambda must be finite and > 0/],
  ];
  for (const [name, o, why] of cases) {
    const s = set(o);
    const v = verdict(readUserFile(H5, s.user, s.model), s.model, s.results);
    assert.equal(v.zones["modes"], true, name);
    assert.match(v.st.zones["modes"]!.reason, why, name);
    assert.equal(v.zones["spectra"], false, `${name}: the other model zones keep the model's state`);
  }
});

// Review of #2729, finding 4: ADR 0113 D7, a newer minor of the model's opensees zone opens with one banner.
test("a model.h5 whose opensees zone is newer than this app gives one banner, naming the model", () => {
  const s = set();
  const f = new h5wasm.File(s.model, "a");
  at(f, "meta").delete_attribute("opensees_schema_version");
  at(f, "meta").create_attribute("opensees_schema_version", "2.99.0");
  f.close();
  const u = readUserFile(H5, s.user, s.model);
  assert.ok(u.combinations.ok);
  assert.equal(u.warnings.length, 1, JSON.stringify(u.warnings));
  assert.ok(u.warnings[0]!.startsWith(`${s.model} (read for /opensees/combinations): opensees_schema_version 2.99.0 is newer than this app`), u.warnings[0]);
});

// ---- /modes rows: A6's lambda rule (VU7a) ----------------------------------

/** A set whose results file is fixtures/ladruno_modes.ladruno, with a `/modes` table over its two modes. */
function modalSet(lam: number[], step = 0) {
  const s = set({ modes: (z) => modesTable(z, null, lam, step) });
  copyFileSync(MODES, s.results);
  return s;
}

test("a /modes row is current when its lambda equals the .ladruno's LAMBDA, and stale past 1e-9 relative", () => {
  const cur = modalSet(LAMBDAS);
  const st = staleness(H5, readUserFile(H5, cur.user, cur.model), cur.model, cur.results, list);
  assert.equal(st.zones["modes"]!.stale, false);
  assert.deepEqual(st.modes, [{ stale: false, reason: "" }, { stale: false, reason: "" }]);
  // step -1 names the only set (STEP_0).
  const only = modalSet(LAMBDAS, -1);
  assert.deepEqual(staleness(H5, readUserFile(H5, only.user, only.model), only.model, only.results, list).modes.map((v) => v.stale), [false, false]);
  const moved = modalSet([LAMBDAS[0]!, LAMBDAS[1]! * (1 + 2e-9)]);
  const v = staleness(H5, readUserFile(H5, moved.user, moved.model), moved.model, moved.results, list).modes;
  assert.deepEqual(v[0], { stale: false, reason: "" });
  assert.equal(v[1]!.stale, true);
  assert.match(v[1]!.reason, /^lambda 2618\.03399\d* here, 2618\.0339887498953 in the results file \(more than 1e-09 relative\)$/);
});

test("a /modes row with no proof is stale: no results file, no such set, a stale zone, a modal record that does not read", () => {
  const s = modalSet(LAMBDAS);
  const u = readUserFile(H5, s.user, s.model);
  assert.deepEqual(staleness(H5, u, s.model, null, list).modes, [0, 1].map(() => ({ stale: true, reason: "no results file to compare lambda with" })));
  // The truss records no mode.
  copyFileSync(TRUSS, s.results);
  assert.deepEqual(staleness(H5, u, s.model, s.results, list).modes.map((v) => v.reason), [0, 1].map(() => "the results file records no modes on stage_0"));
  const step = modalSet(LAMBDAS, 1);
  assert.deepEqual(staleness(H5, readUserFile(H5, step.user, step.model), step.model, step.results, list).modes.map((v) => v.reason), [0, 1].map(() => "the results file records no STEP_1 on stage_0"));
  // A row of a stale zone is stale with the zone's reason.
  writeModel(s.model, { fem: "00000000000000000000000000000000", rows: [M1] });
  copyFileSync(MODES, s.results);
  const remodel = staleness(H5, readUserFile(H5, s.user, s.model), s.model, s.results, list);
  assert.equal(remodel.modes.length, 2);
  for (const v of remodel.modes) assert.match(v.reason, /^fem_hash differs/);
  // A results file that is not HDF5: the modes cannot be read, so every row is stale, and staleness still answers.
  const broken = join(s.dir, "broken.ladruno");
  copyFileSync(s.model, broken);
  writeModel(s.model, { rows: [M1] });
  const v = staleness(H5, readUserFile(H5, s.user, s.model), s.model, broken, list).modes;
  assert.deepEqual(v.map((x) => x.stale), [true, true]);
  // A refused table has no rows to judge.
  const refused = set({ modes: (z) => modesTable(z, "period") });
  assert.deepEqual(staleness(H5, readUserFile(H5, refused.user, refused.model), refused.model, refused.results, list).modes, []);
});

// ---- /spectra: the cases (VU7a) --------------------------------------------

test("a /spectra case reads: name, direction, scale, the spectrum, the modes table, its results by shape", () => {
  const s = set();
  const sp = ok(readUserFile(H5, s.user, s.model).spectra!);
  assert.deepEqual(sp.refused, []);
  assert.equal(sp.cases.length, 1);
  const c = sp.cases[0]!;
  assert.deepEqual([c.name, c.link, c.direction, c.scale], ["Ex", "s0", 1, 9.81]);
  assert.deepEqual([c.periods, c.sa], [[0, 0.5, 1, 2], [0.4, 1, 0.6, 0.3]]);
  assert.deepEqual(c.modes, [
    { mode: 1, kind: "eigen", lambda: LAMBDAS[0], saPeriod: 0, gamma: 1, sa: 5, damping: 0.05 },
    { mode: 2, kind: "eigen", lambda: LAMBDAS[1], saPeriod: 0, gamma: 0.5, sa: 5, damping: 0.05 },
  ]);
  assert.deepEqual(c.results.map((q) => [q.target, q.label, q.components, q.shape, q.ids]), [
    ["domain", "base_shear", ["Vx", "Vy"], [2, 1, 2], null],
    ["nodes", "DISPLACEMENT", ["Ux", "Uy"], [2, 3, 2], [1, 2, 3]],
  ]);
  assert.deepEqual(c.reference.map((q) => q.shape), [[1, 1, 2], [1, 3, 2]]);
  // A /spectra group with no version key refuses the zone; an empty one holds no case.
  assert.equal(readUserFile(H5, writeUser(join(s.dir, "n.user.h5"), { versions: { spectra_schema_version: null } }), s.model).spectra!.ok, false);
  assert.deepEqual(ok(readUserFile(H5, writeUser(join(s.dir, "e.user.h5"), { spectra: () => undefined }), s.model).spectra!), { cases: [], refused: [] });
});

test("a bad /spectra case refuses alone, with Python's reason; duplicate names refuse both; the zone still opens", () => {
  const s = set({
    spectra: (z) => {
      spectrumCase(z, "s0", { name: "Ex" });
      spectrumCase(z, "s1", { name: "Ey", direction: 2 });
      spectrumCase(z, "s2", { name: "elements", results: [["ON_ELEMENTS", "force", [1], ["N"]]] });
      spectrumCase(z, "s3", { name: "Ey" });
      spectrumCase(z, "s4", { name: " padded" });
      spectrumCase(z, "s5", { name: "dynamic", kind: "transient" });
      spectrumCase(z, "s6", { name: "wet", damping: [0.05, 1.5] });
      spectrumCase(z, "s7", { name: "unmatched", reference: [BASE_SHEAR] });
      spectrumCase(z, "s8", { name: "repeat", modes: [1, 1] });
      spectrumCase(z, "s9", { name: "flat", periods: [0, 1, 1, 2] });
      spectrumCase(z, "s10", { name: "down", direction: 0 });
      z.create_group("notacase");
    },
  });
  const sp = ok(readUserFile(H5, s.user, s.model).spectra!);
  assert.deepEqual(sp.cases.map((c) => c.name), ["Ex"]);
  const by = (name: string) => sp.refused.filter((r) => r.name === name);
  assert.deepEqual(sp.refused.map((r) => [r.name, r.reason, r.row]).sort((a, b) => Number(a[2]) - Number(b[2])), [
    ["Ey", "duplicate", 1],
    ["elements", "bad-param", 2],
    ["Ey", "duplicate", 3],
    [" padded", "empty-name", 4],
    ["dynamic", "bad-param", 5],
    ["wet", "bad-param", 6],
    ["unmatched", "bad-param", 7],
    ["repeat", "bad-param", 8],
    ["flat", "bad-param", 9],
    ["down", "bad-param", 10],
  ]);
  assert.equal(by("elements")[0]!.detail, "/spectra/s2/RESULTS/ON_ELEMENTS: the family ON_ELEMENTS is not in spectra 1.0.0, whose cases hold ['ON_DOMAIN', 'ON_NODES'] only");
  assert.equal(by("dynamic")[0]!.detail, "/spectra/s5@KIND is 'transient', not 'static'");
  assert.match(by("wet")[0]!.detail, /damping must be NaN or lie in \(0, 1\)$/);
  assert.match(by("unmatched")[0]!.detail, /reference\/RESULTS must hold the results of RESULTS, with their ids and columns$/);
  assert.match(by("repeat")[0]!.detail, /modes\/mode must be distinct, 1-based and fit i4$/);
  assert.match(by("flat")[0]!.detail, /spectrum T must be non-negative and strictly increasing$/);
  assert.match(by("down")[0]!.detail, /direction 0 is 1-based$/);
  assert.equal(by("Ey")[0]!.detail, "2 cases are named 'Ey'");
});

test("/spectra is versioned as its zone: another major refuses it, a newer minor opens with one banner, and it stales by the model", () => {
  const major = set({ versions: { spectra_schema_version: "2.0.0" } });
  const u = readUserFile(H5, major.user, major.model);
  assert.ok(!u.spectra!.ok && /spectra_schema_version 2\.0\.0: this app reads major 1 only/.test(u.spectra!.reason));
  assert.ok(u.modes!.ok, "the other zones still read");
  const newer = set({ versions: { spectra_schema_version: "1.3.0" } });
  const n = readUserFile(H5, newer.user, newer.model);
  assert.ok(n.spectra!.ok);
  assert.deepEqual(n.warnings.filter((w) => /spectra_schema_version 1\.3\.0 is newer than this app/.test(w)).length, 1);
  const s = set();
  writeModel(s.model, { fem: "00000000000000000000000000000000", rows: [M1] });
  assert.match(staleness(H5, readUserFile(H5, s.user, s.model), s.model, s.results, list).zones["spectra"]!.reason, /^fem_hash differs/);
});

// #2744 review, F3: a refusal's detail is Python's text, and Python's first fault is the one named.
test("a refused case's detail is Python's, word for word, and names the fault Python meets first", () => {
  const s = set({
    spectra: (z) => {
      spectrumCase(z, "s0", { name: "both", scale: 0, gamma: [1] });
      spectrumCase(z, "s1", { name: "short", lambda: [LAMBDAS[0]!] });
      spectrumCase(z, "s2", { name: "i8mode", modeI8: true });
      spectrumCase(z, "s3", { name: "it's", scale: -1 });
    },
  });
  const sp = ok(readUserFile(H5, s.user, s.model).spectra!);
  assert.deepEqual(sp.refused.map((r) => [r.name, r.detail]), [
    // Python's `case_problem` checks the scale before the modes columns' lengths.
    ["both", "/spectra/s0: scale 0.0 must be finite and > 0"],
    ["short", "/spectra/s1: modes/lambda has shape (1,), expected (2,)"],
    ["i8mode", "/spectra/s2/modes/mode is int64 (2,), expected i4"],
    ["it's", "/spectra/s3: scale -1.0 must be finite and > 0"],
  ]);
});
