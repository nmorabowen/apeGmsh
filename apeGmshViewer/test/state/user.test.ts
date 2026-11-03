// ADR 0130 slice VU3b (#2688): the set's `.user.h5` in the store, the phase
// bar's state families and the contour picker's User Results group.
//
// The oracles: the A6 staleness table (h5-schema.md "`<stem>.user.h5`",
// "Staleness"), which reader/userFile.ts computes and VU3a cross-checked
// against Python's `staleness()`; main/pairing.ts `candidates` for where a
// set's user file is; and D1's family and group tables. Each file here is
// written with h5wasm as the schema lays it out, read by the reader main
// runs, and then taken by the reducer, so the whole path from file to panel
// is under test.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { candidates } from "../../src/main/pairing.ts";
import { barClearPx, familiesOf, familyMarks, mountPhase, userBadgeOf } from "../../src/panels/phase.ts";
import { modalTableOf, mountModalTable } from "../../src/panels/modalTable.ts";
import { bannerTopBelowScrubber, mountBanner } from "../../src/panels/banner.ts";
import { userResultsGroup } from "../../src/panels/legend.ts";
import { readUserSet, type UserH5, type UserRead } from "../../src/reader/userFile.ts";
import { initialState, NO_USER_FILE, reduce, userEventOf, userInfoOf, userPathOf, userReadable } from "../../src/state/reduce.ts";
import type { Store } from "../../src/state/store.ts";
import type { ArtifactInfo, Event, ResultsInfo, ResultsStage, State, UserReadIn } from "../../src/state/types.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const TRUSS = fileURLToPath(new URL("../../../tests/fixtures/ladruno/truss2d.ladruno", import.meta.url));
const MODES = fileURLToPath(new URL("../../fixtures/ladruno_modes.ladruno", import.meta.url));
/** fixtures/ladruno_modes.ladruno's LAMBDA of modes 1 and 2 (stage_0, STEP_0), as Python's `recorded_lambdas` reads them. */
const LAMBDAS = [381.9660112501052, 2618.0339887498953];
/** truss2d.ladruno's rf1, a cross-language golden (test/reader/rf1.test.ts). */
const TRUSS_RF1 = "rf1:76a404478cbb9ef919c2d79a676fbe1a0a300712819b344edd9a43e7289cfcc3";
const FEM = "fedcba9876543210fedcba9876543210";
const MODEL = "0123456789abcdef0123456789abcdef";
const list = (d: string) => readdirSync(d);

// The reader's answer type is assignable to the store's: the two cannot drift apart (a compile error here, under `npm run typecheck`).
const tie: (r: UserRead) => UserReadIn = (r) => r;
void tie;

type G = h5wasm.Group;
const str = (g: G, name: string, data: string[]) => g.create_dataset({ name, data, dtype: "S" });
const i4 = (g: G, name: string, data: number[], shape?: number[]) => g.create_dataset({ name, data: Int32Array.from(data), shape: shape ?? [data.length] });
const f8 = (g: G, name: string, data: number[], shape?: number[]) => g.create_dataset({ name, data: Float64Array.from(data), shape: shape ?? [data.length] });

type TermRow = [string, string, number, number, number];
interface Row {
  name: string;
  rule: string;
  terms: TermRow[];
}

/** One `g<N>` of `/combinations` (h5-schema.md). */
function generation(g: G, rows: Row[]): void {
  str(g, "name", rows.map((r) => r.name));
  str(g, "rule", rows.map((r) => r.rule));
  str(g, "envelope", rows.map(() => ""));
  f8(g, "fraction", rows.map(() => NaN));
  f8(g, "damping", rows.map(() => NaN));
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
  { name: "E", rule: "cqc", terms: [["spectrum", "Ex", -1, 0, 1]] },
  { name: "D+E", rule: "linear", terms: [["combination", "E", -1, 0, 1], ["stage", "D", -1, 0, 1]] },
  { name: "dup", rule: "linear", terms: [["stage", "D", -1, 0, 1]] },
  { name: "dup", rule: "linear", terms: [["stage", "L", -1, 0, 1]] },
];

/** A nodal or domain user result (h5-schema.md `/results/r<k>`), `n` rows of 2 components. */
function entry(results: G, link: string, name: string, target: "nodes" | "domain", basis: "results" | "model", states: [string, string, number, number][] | null): void {
  const g = results.create_group(link) as G;
  g.create_attribute("name", name);
  g.create_attribute("KIND", "static");
  g.create_attribute("target", target);
  g.create_attribute("basis", basis);
  g.create_attribute("per_state", Int8Array.of(states ? 1 : 0));
  const s = states?.length ?? 1;
  if (states) {
    const st = g.create_group("STATES") as G;
    str(st, "kind", states.map((x) => x[0]));
    str(st, "source", states.map((x) => x[1]));
    i4(st, "step", states.map((x) => x[2]));
    i4(st, "mode", states.map((x) => x[3]));
  }
  const on = target === "nodes" ? "ON_NODES" : "ON_DOMAIN";
  const v = ((g.create_group("RESULTS") as G).create_group(on) as G).create_group("values") as G;
  const n = target === "domain" ? 1 : 3;
  f8(v, "DATA", Array.from({ length: s * n * 2 }, (_, k) => k), [s, n, 2]);
  i4(v, "STEP", Array.from({ length: s }, (_, k) => k));
  f8(v, "TIME", Array.from({ length: s }, (_, k) => k));
  if (target === "nodes") i4(v, "ID", [1, 2, 3], [3, 1]);
  v.create_attribute("COMPONENTS", "c0,c1");
}

interface SetOpts {
  /** the user file's `/meta@results_hash` (default: the truss's rf1, current) */
  resultsHash?: string;
  /** the model.h5's `model_hash` (default: the user file's, current) */
  modelHash?: string;
  /** another root member, which this app does not know */
  extraZone?: string;
  /** a `/modes` table (h5-schema.md `/modes`) with these lambdas, one row per mode of `stage_0` */
  modes?: number[];
  /** the results file: the truss (default, no mode) or fixtures/ladruno_modes.ladruno */
  results?: "truss" | "modes";
  /** the `/spectra` cases, by name (default: one, `Ex`); [] writes an empty zone */
  cases?: string[];
}

/** A `/modes` table over MX, MY (the layout test/reader/userFile.test.ts writes). */
function modesTable(z: G, lam: number[]): void {
  const n = lam.length;
  const k = Array.from({ length: n }, (_, i) => i);
  z.create_attribute("components", "MX,MY");
  str(z, "host", k.map(() => "stage_0"));
  str(z, "kind", k.map(() => "eigen"));
  i4(z, "step", k.map(() => 0));
  i4(z, "mode", k.map((i) => i + 1));
  f8(z, "lambda", lam);
  f8(z, "period", lam.map((x) => (2 * Math.PI) / Math.sqrt(x)));
  f8(z, "generalized_mass", k.map(() => 1));
  f8(z, "participation", k.flatMap((i) => [1 / (i + 1), 0.1]), [n, 2]);
  f8(z, "mass_ratio", k.flatMap((i) => [0.6 / (i + 1), 0.05]), [n, 2]);
  f8(z, "cumulative_ratio", k.flatMap((i) => [0.6 + 0.2 * i, 0.05 * (i + 1)]), [n, 2]);
}

/** One `/spectra/s<k>` case over modes 1 and 2, a domain base shear (h5-schema.md `/spectra`). */
function spectrumCase(z: G, link: string, name: string, direction: number): void {
  const g = z.create_group(link) as G;
  g.create_attribute("name", name);
  g.create_attribute("KIND", "static");
  g.create_attribute("direction", Int32Array.of(direction));
  g.create_attribute("scale", Float64Array.of(1));
  const sp = g.create_group("spectrum") as G;
  f8(sp, "T", [0.1, 0.5, 1, 2]);
  f8(sp, "Sa", [0.8, 1, 0.6, 0.3]);
  const m = g.create_group("modes") as G;
  i4(m, "mode", [1, 2]);
  str(m, "kind", ["eigen", "eigen"]);
  f8(m, "lambda", LAMBDAS);
  f8(m, "sa_period", [0, 0]);
  f8(m, "gamma", [1, 0.5]);
  f8(m, "sa", [0.9, 0.9]);
  f8(m, "damping", [0.05, 0.05]);
  for (const [root, s] of [[g, 2], [g.create_group("reference") as G, 1]] as const) {
    const h = ((root.create_group("RESULTS") as G).create_group("ON_DOMAIN") as G).create_group("base_shear") as G;
    f8(h, "DATA", Array.from({ length: s }, (_, k) => 100 - k), [s, 1, 1]);
    i4(h, "STEP", Array.from({ length: s }, (_, k) => k));
    f8(h, "TIME", Array.from({ length: s }, (_, k) => k));
    h.create_attribute("COMPONENTS", "Vx");
  }
}

/** A set: `run.h5` (lineage only, with one `/opensees/combinations` row), a copy of truss2d.ladruno, and `run.user.h5`. */
function writeSet(o: SetOpts = {}): { user: string; model: string; results: string } {
  const d = mkdtempSync(join(tmpdir(), "agv-vu3b-"));
  const results = join(d, "run.ladruno");
  copyFileSync(o.results === "modes" ? MODES : TRUSS, results);
  const model = join(d, "run.h5");
  const m = new h5wasm.File(model, "w");
  try {
    const meta = m.create_group("meta") as G;
    meta.create_attribute("opensees_schema_version", "2.39.0");
    const lineage = meta.create_group("lineage") as G;
    lineage.create_attribute("fem_hash", FEM);
    lineage.create_attribute("model_hash", o.modelHash ?? MODEL);
    generation(((m.create_group("opensees") as G).create_group("combinations") as G).create_group("g0") as G, [{ name: "M1", rule: "linear", terms: [["stage", "D", -1, 0, 1]] }]);
  } finally {
    m.close();
  }
  const user = join(d, "run.user.h5");
  const f = new h5wasm.File(user, "w");
  try {
    const meta = f.create_group("meta") as G;
    for (const k of ["user_meta", "combinations", "user_results", "modes", "spectra"]) meta.create_attribute(`${k}_schema_version`, "1.0.0");
    meta.create_attribute("fem_hash", FEM);
    meta.create_attribute("model_hash", MODEL);
    meta.create_attribute("results_hash", o.resultsHash ?? TRUSS_RF1);
    meta.create_attribute("writer", "test");
    meta.create_attribute("revision", Int32Array.of(1));
    const c = f.create_group("combinations") as G;
    c.create_attribute("current", Int32Array.of(0));
    generation(c.create_group("g0") as G, ROWS);
    const r = f.create_group("results") as G;
    entry(r, "r0", "Delta_after (L/480)", "nodes", "model", [["stage", "gravity", -1, 0], ["combination", "1.2D+1.6L", -1, 0]]);
    entry(r, "r1", "base shear", "domain", "results", null);
    const spectra = f.create_group("spectra") as G;
    (o.cases ?? ["Ex"]).forEach((name, k) => spectrumCase(spectra, `s${k}`, name, k + 1));
    if (o.extraZone) f.create_group(o.extraZone);
    if (o.modes) modesTable(f.create_group("modes") as G, o.modes);
  } finally {
    f.close();
  }
  return { user, model, results };
}

/** The store after the set's user file was read the way renderer/app.ts reads it. */
function readInto(s: State, set: { user: string; model: string | null; results: string | null }): State {
  const read = readUserSet(H5, set.user, set.model, set.results, list);
  return reduce(reduce(s, { type: "userOpened", path: set.user }), userEventOf(set.user, read));
}

const ready = (s: State) => {
  assert.equal(s.user?.status, "ready", s.user?.status === "failed" ? s.user.error : "");
  return (s.user as Extract<NonNullable<State["user"]>, { status: "ready" }>).info;
};

// ---- where the user file is, and when it is read ----------------------------

test("userPathOf is main/pairing.ts candidates' user slot for every file the app opens", () => {
  for (const p of ["C:/w/run.h5", "C:/w/run.ladruno", "C:/w/run.results.h5", "C:/w/run.part-3.ladruno", "C:/w/RUN.H5", "C:/w/a.b.h5", "/w/run.geometry.h5"]) {
    const isModel = /(?<!\.(geometry|results|user))\.h5$/i.test(p);
    const set = isModel ? { model: p, results: null } : { model: null, results: p };
    assert.equal(userPathOf(set), candidates(p).user, p);
  }
  assert.equal(userPathOf({ model: "C:/w/run.h5", results: "C:/w/other.ladruno" }), "C:/w/run.user.h5", "the model's stem wins");
  assert.equal(userPathOf({ model: null, results: null }), null);
  assert.throws(() => userPathOf({ model: "C:/w/run.txt", results: null }), /is not a <stem>\.h5/);
});

test("no file there is no user file (userClosed), pinned to the reader's own refusal", () => {
  const d = mkdtempSync(join(tmpdir(), "agv-vu3b-"));
  const missing = join(d, "run.user.h5");
  const read = readUserSet(H5, missing, null, null, list);
  assert.deepEqual(read, { ok: false, error: NO_USER_FILE(missing) });
  assert.deepEqual(userEventOf(missing, read), { type: "userClosed" });
  // Any other refusal is shown, never taken as absence.
  const other: UserReadIn = { ok: false, error: `${missing} is not a readable HDF5 file: x` };
  assert.deepEqual(userEventOf(missing, other), { type: "userRead", path: missing, read: other });
  assert.deepEqual(userEventOf(missing, { ok: false, error: NO_USER_FILE("C:/elsewhere.user.h5") }).type, "userRead", "another path's absence is not this one's");
});

const artifact = (path: string, status: ArtifactInfo["status"]): ArtifactInfo => ({
  path, status, error: null, stale: false, sizeBytes: 0, readMs: 0, zones: {}, warnings: [], counts: { nodes: 0, cells: 0, opsOnly: 0 }, sessionId: null,
});

test("userReadable: only after the model's first read has landed, so the first frame never waits on the user file", () => {
  const withModel = (a: ArtifactInfo | null): State => ({ ...initialState, artifacts: { ...initialState.artifacts, model: a } });
  assert.equal(userReadable(withModel(null), "m.h5"), false, "not opened yet");
  assert.equal(userReadable(withModel(artifact("m.h5", "opened")), "m.h5"), false, "being read");
  assert.equal(userReadable(withModel(artifact("m.h5", "ready")), "m.h5"), true);
  assert.equal(userReadable(withModel(artifact("m.h5", "failed")), "m.h5"), true, "a failed read has landed too");
  assert.equal(userReadable(withModel(artifact("old.h5", "ready")), "m.h5"), false, "the last set's model is not this one");
  assert.equal(userReadable(initialState, null), true, "a set with no model reads at once");
});

// ---- the reducer ----------------------------------------------------------

test("a current user file: the combinations of both files, the user results, /spectra, nothing stale", () => {
  const set = writeSet();
  const info = ready(readInto(initialState, set));
  assert.deepEqual(info.warnings, []);
  assert.ok(info.combinations?.ok);
  assert.deepEqual(info.combinations.items.map((x) => [x.name, x.rule, x.origin, x.terms, x.stale]), [
    ["1.2D+1.6L", "linear", "user", 2, false],
    ["E", "cqc", "user", 1, false],
    ["D+E", "linear", "user", 2, false],
    ["M1", "linear", "model", 1, false],
  ]);
  assert.deepEqual(info.combinations.refused.map((x) => [x.name, x.reason]), [["dup", "duplicate"], ["dup", "duplicate"]]);
  assert.ok(info.results?.ok);
  assert.deepEqual(info.results.items.map((x) => [x.name, x.target, x.basis, x.perState, x.stale.stale]), [
    ["Delta_after (L/480)", "nodes", "model", true, false],
    ["base shear", "domain", "results", false, false],
  ]);
  assert.deepEqual(info.spectra, {
    ok: true,
    items: [{ name: "Ex", direction: 1, scale: 1, modes: [1, 2], damping: 0.05, results: ["domain/base_shear"] }],
    refused: [],
    stale: { stale: false, reason: "" },
  });
  assert.equal(info.modes, null, "no /modes zone");
});

test("a stale user file (A6): a results-basis entry on another run, and everything model-based on another model", () => {
  // Another run: the results_hash no longer matches the .ladruno.
  const rerun = ready(readInto(initialState, writeSet({ resultsHash: `rf1:${"0".repeat(64)}` })));
  assert.ok(rerun.results?.ok);
  const byName = Object.fromEntries(rerun.results.items.map((x) => [x.name, x.stale]));
  assert.equal(byName["base shear"]!.stale, true, "basis results: stale on another run");
  assert.match(byName["base shear"]!.reason, /results_hash/);
  assert.equal(byName["Delta_after (L/480)"]!.stale, false, "basis model: a re-run does not stale it");
  assert.equal(rerun.spectra?.stale.stale, false, "/spectra never derives from a .ladruno");
  // Another model: every model-based zone and entry is stale, and a combination over a spectrum case with it.
  const remodel = ready(readInto(initialState, writeSet({ modelHash: "ffffffffffffffffffffffffffffffff" })));
  assert.equal(remodel.spectra?.stale.stale, true);
  assert.ok(remodel.results?.ok && remodel.combinations?.ok);
  assert.equal(remodel.results.items.find((x) => x.name === "Delta_after (L/480)")!.stale.stale, true);
  assert.deepEqual(remodel.combinations.items.filter((x) => x.stale).map((x) => x.name), ["E", "D+E"], "a stale source stales the combination, and through it the next");
});

test("a root member this app does not know is named in a warning, never dropped silently", () => {
  const info = ready(readInto(initialState, writeSet({ extraZone: "footfall" })));
  assert.deepEqual(info.warnings, ["run.user.h5 holds /footfall, a zone this app does not know; it is not read"]);
  const s = readInto(initialState, writeSet({ extraZone: "footfall" }));
  assert.deepEqual(userBadgeOf(s), { text: "user file: 1 warning", title: info.warnings[0], refused: false });
});

const IDX: UserReadIn & { ok: true } = {
  ok: true,
  user: {
    path: "u.user.h5",
    zones: ["results"],
    warnings: [],
    combinations: { ok: true, value: { rows: [], modelRows: [], refused: [] } },
    results: { ok: true, value: { entries: [{ name: "constructor", target: "nodes", basis: "model", perState: false, linear: true }], refused: [] } },
    modes: null,
    spectra: null,
  },
  staleness: { zones: {}, results: {}, combinations: [], modes: [] },
};

test("userInfoOf fails closed: an unknown target refuses the answer; a missing verdict is stale, `constructor` included", () => {
  const info = userInfoOf(IDX.user, IDX.staleness);
  assert.deepEqual(info.results?.ok && info.results.items[0]!.stale, { stale: true, reason: "the reader gave no staleness verdict for it" });
  assert.equal(info.combinations, null, "no row in either file: no family");
  const odd = { ...IDX.user, results: { ok: true as const, value: { entries: [{ ...IDX.user.results.ok ? IDX.user.results.value.entries[0]! : (null as never), target: "planets" }], refused: [] } } };
  assert.throws(() => userInfoOf(odd, IDX.staleness), (e: unknown) => e instanceof RangeError && /target of constructor "planets"/.test((e as Error).message));
});

test("the user slot: opened, read, re-read, failed, closed; it survives the model's reset", () => {
  let s = reduce(initialState, { type: "userOpened", path: "u.user.h5" });
  assert.deepEqual(s.user, { status: "reading", path: "u.user.h5" });
  assert.throws(() => reduce(s, { type: "userRead", path: "other.user.h5", read: IDX }), /is not the open \.user\.h5/);
  assert.throws(() => reduce(s, { type: "userRead", path: "u.user.h5", read: { ...IDX, user: { ...IDX.user, path: "x.user.h5" } } }), /the answer is for x\.user\.h5/);
  s = reduce(s, { type: "userRead", path: "u.user.h5", read: IDX });
  assert.equal(s.user?.status, "ready");
  // A re-read after a change is pending: the verdict from before it is not shown as current (#2737 review, F1).
  const again = reduce(s, { type: "userOpened", path: "u.user.h5" });
  assert.deepEqual(again.user, { status: "reading", path: "u.user.h5" });
  assert.equal(reduce(again, { type: "userOpened", path: "u.user.h5" }), again, "already reading: a no-op");
  assert.deepEqual(familiesOf(again), [], "nothing of the file is listed while it is read");
  assert.deepEqual(userBadgeOf(again), { text: "reading user file", title: "u.user.h5 is being read", refused: false });
  assert.equal(reduce(s, { type: "userOpened", path: "v.user.h5" }).user?.status, "reading", "another file starts over");
  assert.throws(() => reduce(s, { type: "userOpened", path: "" }), RangeError);
  const failed = reduce(s, { type: "userRead", path: "u.user.h5", read: { ok: false, error: "u.user.h5 has no /meta group" } });
  assert.deepEqual(failed.user, { status: "failed", path: "u.user.h5", error: "u.user.h5 has no /meta group" });
  assert.deepEqual(userBadgeOf(failed), { text: "user file refused", title: "u.user.h5: u.user.h5 has no /meta group", refused: true });
  // The model's failure and close reset the model's state, not the set's user file.
  assert.equal(reduce(s, { type: "fileFailed", artifact: "model", path: "m.h5", error: "x" }).user, s.user);
  assert.equal(reduce({ ...s, artifacts: { ...s.artifacts, model: artifact("m.h5", "ready") } }, { type: "fileClosed", artifact: "model" }).user, s.user);
  assert.equal(reduce(s, { type: "userClosed" }).user, null);
  assert.equal(reduce(initialState, { type: "userClosed" }), initialState, "closing nothing is a no-op");
});

// ---- the phase bar's families (D1) ------------------------------------------

const stage = (id: string, name: string, kind: ResultsStage["kind"]): ResultsStage => ({ id, name, kind, time: [0, 1], nodal: [], refusedNodal: [], gauss: [], unread: [] });

/** A results file of two stages and two modes, paired: the axis holds all four. */
function withResults(s: State): State {
  const results: ResultsInfo = {
    path: "C:/w/run.ladruno",
    version: "1.0.0",
    sourceType: "ladruno",
    lineage: null,
    partial: null,
    stages: [stage("stage_0", "gravity", "static"), stage("stage_1", "", "transient"), stage("stage_2_mode_1", "mode_1", "mode"), stage("stage_2_mode_2", "mode_2", "mode")],
    unavailableZones: [],
    openseesVersion: null,
    programDecl: null,
  };
  return { ...s, results, phase: { axis: [{ kind: "mesh" }, ...results.stages.map((_, i) => ({ kind: "results" as const, stage: i, step: 0 }))], at: { kind: "mesh" } } };
}

test("the phase bar lists Stages, Modes, Spectra and User Combinations, each with its count and what is stale", () => {
  const s = withResults(readInto(initialState, writeSet({ modelHash: "ffffffffffffffffffffffffffffffff" })));
  const f = familiesOf(s);
  assert.deepEqual(f.map((x) => [x.key, x.stages, x.count, x.stale, x.refused, x.drawable]), [
    ["stages", [0, 1], 2, 0, 0, true],
    ["modes", [2, 3], 2, 0, 0, true],
    ["spectra", [], 1, 1, 0, false],
    ["combinations", [], 4, 2, 2, false],
  ]);
  assert.deepEqual(f.map((x) => familyMarks(x)), [
    { count: " 2", stale: "", refused: "" },
    { count: " 2", stale: "", refused: "" },
    { count: " 1", stale: " · stale", refused: "" },
    { count: " 4", stale: " · 2 stale", refused: " · 2 refused" },
  ]);
  assert.match(f[0]!.title, /^Stages of run\.ladruno: gravity, stage_1 \(unnamed\)$/);
  assert.match(f[2]!.title, /\/spectra is stale: /);
  assert.deepEqual(f[2]!.title.split("\n").slice(0, 2), [
    "Spectra, listed; this app does not combine or draw a case yet:",
    "Ex: direction 1, scale 1, 2 modes, damping 0.05; domain/base_shear",
  ]);
  const combos = f[3]!.title.split("\n");
  assert.equal(combos[0], "User Combinations, listed; this app does not evaluate them yet:");
  assert.ok(combos.includes("E: cqc, 1 term (stale: a source of it is stale)"));
  assert.ok(combos.includes("M1: linear, 1 term, from model.h5"));
  assert.ok(combos.some((l) => l.startsWith("dup: refused, duplicate: ")));
  // With no user file, only the recorded families; with no paired results, only the user file's.
  assert.deepEqual(familiesOf(withResults(initialState)).map((x) => x.key), ["stages", "modes"]);
  assert.deepEqual(familiesOf(readInto(initialState, writeSet())).map((x) => x.key), ["spectra", "combinations"]);
});

test("a stage that records no step is listed and never opened (San Ramon's MODEL_STAGE[1] before its modes)", () => {
  const s = withResults(initialState);
  const results = { ...s.results!, stages: [{ ...stage("stage_0", "MODEL_STAGE[1]", "static"), time: [] }, ...s.results!.stages.slice(2)] };
  const f = familiesOf({ ...s, results, phase: { axis: [{ kind: "mesh" }, ...results.stages.map((_, i) => ({ kind: "results" as const, stage: i, step: 0 }))], at: { kind: "mesh" } } });
  assert.deepEqual(f.map((x) => [x.key, x.stages, x.count, x.drawable]), [
    ["stages", [], 1, false],
    ["modes", [1, 2], 2, true],
  ]);
  assert.equal(f[0]!.title, "Stages of run.ladruno: MODEL_STAGE[1]\nMODEL_STAGE[1] records no step");
});

test("each /modes row carries A6's verdict against the .ladruno; the Modes button counts the stale rows (replaces #2737's 'rows not judged')", () => {
  // Lambdas equal to the run's: every row current.
  const cur = withResults(readInto(initialState, writeSet({ modes: LAMBDAS, results: "modes" })));
  const info = ready(cur);
  assert.ok(info.modes?.ok);
  assert.deepEqual(info.modes.rows.map((r) => [r.mode, r.verdict]), [[1, { stale: false, reason: "" }], [2, { stale: false, reason: "" }]]);
  const m = familiesOf(cur).find((f) => f.key === "modes")!;
  assert.deepEqual([m.stale, m.staleAll, m.refused, familyMarks(m).stale], [0, false, 0, ""]);
  assert.match(m.title, /the modal table of run\.user\.h5: 2 rows, every row current/);
  // One lambda moved past 1e-9 relative: that row alone is stale, and the button counts it.
  const one = withResults(readInto(initialState, writeSet({ modes: [LAMBDAS[0]!, LAMBDAS[1]! * 1.5], results: "modes" })));
  const f1 = familiesOf(one).find((f) => f.key === "modes")!;
  assert.deepEqual([f1.stale, f1.staleAll, familyMarks(f1).stale], [1, false, " · 1 stale"]);
  assert.match(f1.title, /mode 2 of stage_0 is stale: lambda 3927\.05\d* here, 2618\.0339887498953 in the results file/);
  // Lambdas 1.5x a run's (#2737 review, F2): Python's staleness() stales every row, with the zone current.
  const all = withResults(readInto(initialState, writeSet({ modes: LAMBDAS.map((x) => x * 1.5), results: "modes" })));
  assert.deepEqual(ready(all).modes?.stale, { stale: false, reason: "" }, "the zone itself is current");
  const fa = familiesOf(all).find((f) => f.key === "modes")!;
  assert.deepEqual([fa.stale, fa.staleAll, familyMarks(fa).stale], [2, true, " · stale"]);
  // The truss records no mode: no row can be proven current.
  const truss = ready(readInto(initialState, writeSet({ modes: LAMBDAS })));
  assert.ok(truss.modes?.ok);
  assert.deepEqual(truss.modes.rows.map((r) => r.verdict.reason), ["the results file records no modes on stage_0", "the results file records no modes on stage_0"]);
  // No /modes zone: nothing to judge, and the family reads plainly.
  const plain = familiesOf(withResults(readInto(initialState, writeSet()))).find((f) => f.key === "modes")!;
  assert.deepEqual([plain.stale, familyMarks(plain).stale], [0, ""]);
});

test("userInfoOf fails closed on the row verdicts: a count that disagrees with the table is no verdict for any row", () => {
  const set = writeSet({ modes: LAMBDAS, results: "modes" });
  const read = readUserSet(H5, set.user, set.model, set.results, list);
  assert.ok(read.ok);
  const short = userInfoOf(read.user, { ...read.staleness, modes: read.staleness.modes.slice(0, 1) });
  assert.ok(short.modes?.ok);
  assert.deepEqual(short.modes.rows.map((r) => r.verdict), [0, 1].map(() => ({ stale: true, reason: "the reader gave no staleness verdict for it" })));
  // A spectrum target this app does not know refuses the answer.
  assert.ok(read.user.spectra?.ok);
  const odd = { ...read.user, spectra: { ok: true as const, value: { ...read.user.spectra.value, cases: [{ ...read.user.spectra.value.cases[0]!, results: [{ label: "x", target: "elements", components: ["a"] }] }] } } };
  assert.throws(() => userInfoOf(odd, read.staleness), (e: unknown) => e instanceof RangeError && /the target of Ex x "elements"/.test((e as Error).message));
});

test("the Spectra family lists its cases, a refused case, and a refused zone", () => {
  const two = familiesOf(readInto(initialState, writeSet({ cases: ["Ex", "Ey"] }))).find((f) => f.key === "spectra")!;
  assert.deepEqual([two.count, two.stale, two.refused, two.drawable, familyMarks(two)], [2, 0, 0, false, { count: " 2", stale: "", refused: "" }]);
  assert.ok(two.title.split("\n").includes("Ey: direction 2, scale 1, 2 modes, damping 0.05; domain/base_shear"));
  const dup = familiesOf(readInto(initialState, writeSet({ cases: ["Ex", "Ex", "Ey"] }))).find((f) => f.key === "spectra")!;
  assert.deepEqual([dup.count, dup.refused, familyMarks(dup).refused], [1, 2, " · 2 refused"]);
  assert.ok(dup.title.split("\n").includes("Ex: refused, duplicate: 2 cases are named 'Ex'"));
  const empty = familiesOf(readInto(initialState, writeSet({ cases: [] }))).find((f) => f.key === "spectra")!;
  assert.deepEqual([empty.count, familyMarks(empty).count], [0, " 0"]);
});

test("a refused /combinations zone, and a stale or refused modal table, show on their families", () => {
  const base = withResults(initialState);
  const user = (info: Partial<NonNullable<ReturnType<typeof userInfoOf>>>): State => ({
    ...base,
    user: { status: "ready", path: "C:/w/run.user.h5", info: { warnings: [], combinations: null, table: null, results: null, spectra: null, modes: null, ...info } },
  });
  const refused = familiesOf(user({ combinations: { ok: false, reason: "refused: /combinations@current names no generation" } }));
  assert.deepEqual(refused.map((x) => [x.key, x.count, x.refused]).at(-1), ["combinations", 0, 1]);
  assert.deepEqual(familyMarks(refused.at(-1)!), { count: " 0", stale: "", refused: " · refused" });
  const zoneStale = { stale: true, reason: "the model hashes differ" };
  const row = (mode: number) => ({ host: "stage_0", kind: "eigen", step: 0, mode, lambda: 4, period: 1, generalizedMass: 1, participation: [1], massRatio: [0.5], cumulativeRatio: [0.5], verdict: zoneStale });
  const stale = familiesOf(user({ modes: { ok: true, components: ["MX"], rows: [row(1), row(2)], stale: zoneStale } }))[1]!;
  assert.deepEqual([stale.key, stale.stale, stale.staleAll, stale.refused, familyMarks(stale).stale], ["modes", 2, true, 0, " · stale"]);
  assert.match(stale.title, /the modal table of run\.user\.h5 is stale: the model hashes differ/);
  const bad = familiesOf(user({ modes: { ok: false, reason: "refused: /modes has no lambda", stale: { stale: true, reason: "the zone is refused" } } }))[1]!;
  assert.deepEqual([bad.refused, familyMarks(bad).refused, familyMarks(bad).stale], [1, " · refused", ""]);
  assert.match(bad.title, /the modal table of run\.user\.h5 is refused: refused: \/modes has no lambda/);
  const spectra = familiesOf(user({ spectra: { ok: false, reason: "refused: /spectra: spectra_schema_version 2.0.0: this app reads major 1 only", stale: { stale: false, reason: "" } } })).at(-1)!;
  assert.deepEqual([spectra.key, spectra.count, spectra.refused, familyMarks(spectra).refused], ["spectra", 0, 1, " · refused"]);
});

test("the contour picker's User Results group: each entry, what it is, and what is stale", () => {
  const g = userResultsGroup(readInto(initialState, writeSet({ resultsHash: `rf1:${"0".repeat(64)}` })));
  assert.ok(g);
  assert.equal(g.label, "User Results");
  assert.deepEqual(g.rows.map((r) => [r.text, r.stale, r.refused]), [
    ["Delta_after (L/480) (nodal, per state)", false, false],
    ["base shear (whole model, state-free, stale)", true, false],
  ]);
  assert.match(g.rows[1]!.title, /on the results basis; stale: .*results_hash/);
  assert.equal(g.note, "Listed only: this app does not draw a user result yet.");
  assert.equal(userResultsGroup(initialState), null);
  assert.equal(userResultsGroup(reduce(initialState, { type: "userOpened", path: "u.user.h5" })), null, "nothing listed while reading");
});

// ---- the phase bar, mounted ---------------------------------------------------

test("the capped bar leaves clear what #header does: 684 px, or 216 px at 1000 px wide or less and with the rail collapsed", () => {
  // style.css: `#header { max-width: calc(100vw - 684px) }`, `@media (max-width: 1000px)` and `body[data-rail-collapsed="true"]` give 216.
  assert.equal(barClearPx(1600, false), 684);
  assert.equal(barClearPx(1001, false), 684);
  assert.equal(barClearPx(1000, false), 216);
  assert.equal(barClearPx(1600, true), 216);
});

test("the banner sits below a wrapped scrubber bar by the bar's published height, and at 106 px under a one-line bar", () => {
  // 58 px (the scrubber row's top) + the bar's height + 10 px; a one-line bar (38 px or less) leaves the max at 106 px, as before.
  assert.equal(bannerTopBelowScrubber(), "max(106px, calc(58px + var(--phase-bar-h, 0px) + 10px))");
});

class FakeEl {
  tagName: string;
  className = "";
  title = "";
  value = "";
  type = "";
  min = "";
  max = "";
  hidden = false;
  disabled = false;
  selected = false;
  dataset: Record<string, string | undefined> = {};
  style: Record<string, string> = {};
  children: (FakeEl | string)[] = [];
  listeners = new Map<string, (() => void)[]>();
  constructor(tag: string) {
    this.tagName = tag.toUpperCase();
  }
  append(...xs: (FakeEl | string)[]) {
    this.children.push(...xs);
  }
  replaceChildren(...xs: (FakeEl | string)[]) {
    this.children = [...xs];
  }
  addEventListener(type: string, fn: () => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  removeEventListener(type: string, fn: () => void) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((f) => f !== fn));
  }
  click() {
    for (const fn of this.listeners.get("click") ?? []) fn();
  }
  get textContent(): string {
    return this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join("");
  }
  set textContent(v: string) {
    this.children = [v];
  }
  prepend(...xs: (FakeEl | string)[]) {
    this.children.unshift(...xs);
  }
  /** The rail holds no #offers in these tests. */
  querySelector(): null {
    return null;
  }
  removed = false;
  remove() {
    this.removed = true;
  }
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = pred(this) ? [this] : [];
    for (const c of this.children) if (typeof c !== "string") out.push(...c.all(pred));
    return out;
  }
}

test("mounted, the bar shows each family as a button; the user families are disabled, and Modes opens the first mode", () => {
  const prevDoc = (globalThis as { document?: unknown }).document;
  const bar = Object.assign(new FakeEl("div"), { getBoundingClientRect: () => ({ height: 76.4 }) });
  // The root's custom properties, as style.css reads them (`--phase-bar-h` moves #left-rail below a wrapped bar).
  const root = new Map<string, string>();
  (globalThis as { document?: unknown }).document = {
    createElement: (t: string) => new FakeEl(t),
    getElementById: (id: string) => (id === "phase" ? bar : null),
    documentElement: { style: { setProperty: (k: string, v: string) => root.set(k, v), removeProperty: (k: string) => root.delete(k) } },
  };
  let state = withResults(readInto(initialState, writeSet({ modelHash: "ffffffffffffffffffffffffffffffff" })));
  const listeners: ((s: State, prev: State) => void)[] = [];
  const dispatched: Event[] = [];
  const store = {
    get: () => state,
    dispatch: (e: Event) => {
      dispatched.push(e);
      const prev = state;
      // The axis here has no mesh behind it, so the reducer would rebuild it empty: a phase move is applied as the reducer's setPhase leaves it.
      if (e.type !== "setPhase") throw new Error(`the phase bar dispatched ${e.type}`);
      state = { ...state, phase: { ...state.phase, at: e.at } };
      for (const l of listeners) l(state, prev);
    },
    subscribe: (l: (s: State, prev: State) => void) => {
      listeners.push(l);
      return () => listeners.splice(listeners.indexOf(l), 1);
    },
  } as unknown as Store;
  const off = mountPhase(store);
  try {
    assert.equal(bar.hidden, false);
    const families = () => bar.all((e) => e.dataset["family"] !== undefined);
    assert.deepEqual(families().map((b) => [b.textContent, b.disabled]), [
      ["Stages 2", false],
      ["Modes 2", false],
      ["Spectra 1 · stale", true],
      ["User Combinations 4 · 2 stale · 2 refused", true],
    ]);
    // With results on the axis the bar sits on the scrubber row, so its height is published already.
    assert.equal(root.get("--phase-bar-h"), "77px");
    families()[1]!.click();
    assert.deepEqual(dispatched.at(-1), { type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
    assert.match(families()[1]!.className, /active/);
    // The stage picker lists the family shown: the two modes only.
    const pick = bar.all((e) => e.className === "results-stage")[0]!;
    assert.deepEqual(pick.children.map((o) => (o as FakeEl).textContent), ["mode_1", "mode_2"]);
    assert.equal(pick.title, "results stage (Modes)");
    // F3 (#2737 review): on the scrubber row the bar stops where #header does, clear of the ViewCube, and its buttons keep one line.
    assert.match(bar.style["cssText"] ?? "", /max-width: calc\(100vw - 684px\); flex-wrap: wrap;/);
    // The scrubber keeps its least width on a line, else it wraps to a line of its own.
    const box = bar.all((e) => e.className === "scrubber")[0]!;
    assert.match(box.style["cssText"] ?? "", /flex: 1 1 0; min-width: 340px;/);
    // The drawn height is published for #left-rail on the scrubber row, rounded up.
    assert.equal(root.get("--phase-bar-h"), "77px");
    assert.doesNotMatch(bar.style["cssText"] ?? "", /z-index/, "the bar is not stacked over the left rail");
    assert.deepEqual([families()[3]!.style["flex"], families()[3]!.style["whiteSpace"]], ["0 0 auto", "nowrap"]);
    // A disabled family dispatches nothing.
    const n = dispatched.length;
    families()[3]!.click();
    assert.equal(dispatched.length, n);
  } finally {
    off();
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

// ---- the banner's published place, and what clears a wrapped bar (#2737 delta review 2) ----

/** A state that shows the banner (one model warning) on the scrubber row. */
function bannerState(): State {
  return { ...withResults(initialState), artifacts: { ...initialState.artifacts, model: { ...artifact("C:/w/run.h5", "ready"), warnings: ["a warning"] } } };
}

function bannerEnv(height: () => number) {
  const prevDoc = (globalThis as { document?: unknown }).document;
  const banner = Object.assign(new FakeEl("div"), { getBoundingClientRect: () => ({ height: height() }) });
  const root = new Map<string, string>();
  (globalThis as { document?: unknown }).document = {
    createElement: (t: string) => new FakeEl(t),
    getElementById: (id: string) => (id === "banner" ? banner : null),
    documentElement: { style: { setProperty: (k: string, v: string) => root.set(k, v), removeProperty: (k: string) => root.delete(k) } },
  };
  let state = bannerState();
  const listeners: ((s: State, prev: State) => void)[] = [];
  const store = {
    get: () => state,
    dispatch: () => {},
    subscribe: (l: (s: State, prev: State) => void) => {
      listeners.push(l);
      return () => listeners.splice(listeners.indexOf(l), 1);
    },
  } as unknown as Store;
  const set = (s: State) => {
    const prev = state;
    state = s;
    for (const l of [...listeners]) l(state, prev);
  };
  return { banner, root, store, set, restore: () => ((globalThis as { document?: unknown }).document = prevDoc) };
}

test("F2: the banner publishes its closed height only; the open list overlays the rail and leaves it where it was", () => {
  let h = 54.2;
  const e = bannerEnv(() => h);
  const off = mountBanner(e.store);
  try {
    assert.equal(e.banner.hidden, false);
    assert.equal(e.root.get("--banner-h"), "55px");
    assert.equal(e.root.get("--banner-top"), bannerTopBelowScrubber());
    const summary = () => e.banner.all((x) => x.className.startsWith("banner-summary"))[0]!;
    h = 512; // open: max-height 100vh - 180px
    summary().click();
    assert.equal(e.banner.dataset["open"], "true");
    assert.equal(e.root.get("--banner-h"), "55px", "the open height is not published");
    h = 54.2;
    summary().click();
    assert.equal(e.banner.dataset["open"], undefined);
    assert.equal(e.root.get("--banner-h"), "55px");
  } finally {
    off();
    e.restore();
  }
});

test("F3: the banner's variables are removed when it hides and when it unmounts, as the bar's are", () => {
  const e = bannerEnv(() => 54);
  const off = mountBanner(e.store);
  try {
    assert.ok(e.root.has("--banner-h") && e.root.has("--banner-top"));
    e.set({ ...initialState, artifacts: { ...initialState.artifacts, model: artifact("C:/w/run.h5", "ready") } });
    assert.equal(e.banner.hidden, true);
    assert.deepEqual([e.root.has("--banner-top"), e.root.has("--banner-h")], [false, false], "hidden");
    e.set(bannerState());
    assert.ok(e.root.has("--banner-h"));
  } finally {
    off();
    e.restore();
  }
  assert.deepEqual([e.root.has("--banner-top"), e.root.has("--banner-h")], [false, false], "after unmount");
});

/** One `top:` of style.css, evaluated under a published bar height (px): its `max(...)`/`calc(...)` of px terms. */
function cssTop(selector: string, barPx: number): number {
  const css = readFileSync(fileURLToPath(new URL("../../src/renderer/style.css", import.meta.url)), "utf8");
  const at = css.indexOf(`${selector} {`);
  assert.ok(at >= 0, `style.css has no ${selector} rule`);
  const rule = css.slice(at, css.indexOf("}", at));
  const m = /(?:^|[\s;])top:\s*([^;]+);/.exec(rule);
  assert.ok(m, `${selector} has no top`);
  const expr = m[1]!.replace(/var\(--phase-bar-h, 0px\)/g, `${barPx}px`);
  assert.match(expr, /^[\d\s+.()a-z,-]+$/, `an expression this test can evaluate: ${expr}`);
  const js = expr.replace(/(\d+(?:\.\d+)?)px/g, "$1").replace(/max\(/g, "Math.max(").replace(/calc\(/g, "(");
  return Function(`return ${js};`)() as number;
}

test("F1: #windows and #left-rail sit below a wrapped bar, and where they were under a one-line bar", () => {
  // A one-line bar draws 33 px (y 58 to 90 in the 1600 px D7 still).
  assert.equal(cssTop("#windows", 33), 100);
  assert.equal(cssTop("#left-rail", 33), 102);
  assert.equal(cssTop("#windows", 0), 100, "no bar published");
  // A two-line bar (1280 px) and a three-line bar (1100 px): 6 px below the bar's bottom (58 px + its height).
  for (const h of [77, 116]) {
    assert.equal(cssTop("#windows", h), 58 + h + 6);
    assert.equal(cssTop("#left-rail", h), 58 + h + 6);
  }
});

// ---- the modal table panel (VU7a) ---------------------------------------------

/** A run of the modal fixture's two modes, shown at mode `shown` (1-based), or at the mesh (null). */
function atMode(s: State, shown: number | null): State {
  const mode = (k: number): ResultsStage => ({ ...stage(`stage_0_mode_${k}`, `mode_${k}`, "mode"), time: [0], mode: { index: k, eigenvalue: LAMBDAS[k - 1]!, frequencyHz: 0, periodS: 0 } });
  const results: ResultsInfo = { path: "C:/w/run.ladruno", version: "1.0.0", sourceType: "ladruno", lineage: null, partial: null, stages: [mode(1), mode(2)], unavailableZones: [], openseesVersion: null, programDecl: null };
  const axis = [{ kind: "mesh" as const }, ...results.stages.map((_, i) => ({ kind: "results" as const, stage: i, step: 0 }))];
  return { ...s, results, phase: { axis, at: shown === null ? { kind: "mesh" } : { kind: "results", stage: shown - 1, step: 0 } } };
}

test("the modal table: shown beside a mode, each row with its period, ratios and A6 verdict; the shown mode is marked", () => {
  const cur = readInto(initialState, writeSet({ modes: [LAMBDAS[0]!, LAMBDAS[1]! * 1.5], results: "modes" }));
  assert.equal(modalTableOf(atMode(cur, null)), null, "not beside the mesh");
  assert.equal(modalTableOf(atMode(initialState, 1)), null, "no user file");
  const v = modalTableOf(atMode(cur, 1))!;
  assert.deepEqual([v.file, v.components, v.status, v.note, v.bad, v.missing], ["run.user.h5", ["MX", "MY"], "2 rows · 1 stale", null, false, null]);
  // T = 2 pi / sqrt(lambda); the ratios are fractions, drawn in percent.
  assert.deepEqual(v.lines.map((l) => [l.mode, l.period, l.ratios, l.cumulative, l.verdict, l.shown]), [
    [1, "0.3215", ["60.0", "5.0"], ["60.0", "5.0"], "current", true],
    [2, "0.1003", ["30.0", "5.0"], ["80.0", "10.0"], "stale", false],
  ]);
  // Mode 2's row holds another lambda than the stage shown: it is not that mode's row, so none is marked (#2744 review, F1).
  const two = modalTableOf(atMode(cur, 2))!;
  assert.deepEqual([two.lines.map((l) => l.shown), two.missing], [[false, false], 2]);
  assert.match(v.lines[1]!.title, /^mode 2 \(eigen\) of stage_0, STEP_0\nlambda 3927, period 0\.1003 s, generalised mass 1\nparticipation factor: MX 0\.5, MY 0\.1\nstale: lambda 3927\.05\d* here, 2618\.0339887498953 in the results file/);
  // A mode beyond the table is shown without its row, not refused (D6).
  const short = modalTableOf(atMode(readInto(initialState, writeSet({ modes: [LAMBDAS[0]!], results: "modes" })), 2))!;
  assert.deepEqual([short.missing, short.status, short.lines.map((l) => l.shown)], [2, "1 row · current", [false]]);
});

test("the modal table says a stale or refused zone, whole", () => {
  const remodel = modalTableOf(atMode(readInto(initialState, writeSet({ modes: LAMBDAS, results: "modes", modelHash: "ffffffffffffffffffffffffffffffff" })), 1))!;
  assert.deepEqual([remodel.status, remodel.bad], ["2 rows · stale", true]);
  assert.match(remodel.note!, /^the modal table of run\.user\.h5 is stale: model_hash differs/);
  const base = atMode(initialState, 1);
  const refused = modalTableOf({ ...base, user: { status: "ready", path: "C:/w/run.user.h5", info: { warnings: [], combinations: null, table: null, results: null, spectra: null, modes: { ok: false, reason: "refused: /modes: it has no period column", stale: { stale: true, reason: "x" } } } } })!;
  assert.deepEqual([refused.status, refused.lines, refused.bad, refused.note], ["refused", [], true, "the modal table of run.user.h5 is refused: refused: /modes: it has no period column"]);
});

test("mounted, the modal table shows while a mode is shown, one row per mode, and leaves the rail when unmounted", () => {
  const prevDoc = (globalThis as { document?: unknown }).document;
  const rail = new FakeEl("div");
  (globalThis as { document?: unknown }).document = {
    createElement: (t: string) => new FakeEl(t),
    getElementById: (id: string) => (id === "rail" ? rail : null),
  };
  let state = atMode(readInto(initialState, writeSet({ modes: [LAMBDAS[0]!, LAMBDAS[1]! * 1.5], results: "modes" })), null);
  const listeners: ((s: State, prev: State) => void)[] = [];
  const store = {
    get: () => state,
    dispatch: () => {
      throw new Error("the modal table dispatches nothing");
    },
    subscribe: (l: (s: State, prev: State) => void) => {
      listeners.push(l);
      return () => listeners.splice(listeners.indexOf(l), 1);
    },
  } as unknown as Store;
  const set = (next: State) => {
    const prev = state;
    state = next;
    for (const l of listeners) l(state, prev);
  };
  const off = mountModalTable(store);
  try {
    const panel = rail.children[0] as FakeEl & { id?: string };
    assert.equal(panel.id, "modal-table");
    assert.equal(panel.hidden, true, "hidden beside the mesh");
    set(atMode(state, 1));
    assert.equal(panel.hidden, false);
    const rows = panel.all((e) => e.tagName === "TR" && e.dataset["mode"] !== undefined);
    assert.deepEqual(rows.map((r) => r.children.map((c) => (c as FakeEl).textContent)), [
      ["1", "0.3215", "60.0", "5.0", "60.0", "5.0", "current"],
      ["2", "0.1003", "30.0", "5.0", "80.0", "10.0", "stale"],
    ]);
    assert.deepEqual(rows.map((r) => r.className), ["modal-row shown", "modal-row"]);
    assert.equal(panel.all((e) => e.className === "modal-status")[0]!.textContent, "2 rows · 1 stale");
    set(atMode(state, null));
    assert.equal(panel.hidden, true);
  } finally {
    off();
    (globalThis as { document?: unknown }).document = prevDoc;
  }
  assert.equal((rail.children[0] as FakeEl).removed, true);
});

// ---- which row is the mode shown (#2744 review, F1 and F4) ---------------------

const modeRow = (host: string, step: number, mode: number, lambda: number) => ({
  host, kind: "eigen", step, mode, lambda, period: (2 * Math.PI) / Math.sqrt(lambda), generalizedMass: 1,
  participation: [1], massRatio: [0.5], cumulativeRatio: [0.5], verdict: { stale: false, reason: "" },
});
const withTable = (s: State, rows: ReturnType<typeof modeRow>[]): State => ({
  ...s,
  user: { status: "ready", path: "C:/w/run.user.h5", info: { warnings: [], combinations: null, table: null, results: null, spectra: null, modes: { ok: true, components: ["MX"], rows, stale: { stale: false, reason: "" } } } },
});
/** `atMode`, with the shown stage's LAMBDA replaced: another modal set's mode of the same number. */
const atOtherSet = (s: State, k: number, eigenvalue: number): State => {
  const t = atMode(s, k);
  const stages = t.results!.stages.map((x, i) => (i === k - 1 ? { ...x, mode: { ...x.mode!, eigenvalue } } : x));
  return { ...t, results: { ...t.results!, stages } };
};

test("a row is the mode shown only when its lambda is the shown stage's LAMBDA: another set's mode 1 marks no row", () => {
  // The table holds stage_0's modes; the viewport shows stage_2's mode 1 (lambda 9999). The review's repro.
  const s = withTable(initialState, [modeRow("stage_0", 0, 1, LAMBDAS[0]!), modeRow("stage_0", 0, 2, LAMBDAS[1]!)]);
  const other = modalTableOf(atOtherSet(s, 1, 9999))!;
  assert.deepEqual([other.lines.map((l) => l.shown), other.missing], [[false, false], 1]);
  // Inside 1e-9 relative it is that row; past it, it is not.
  assert.deepEqual(modalTableOf(atOtherSet(s, 1, LAMBDAS[0]! * (1 + 5e-10)))!.lines.map((l) => l.shown), [true, false]);
  assert.deepEqual(modalTableOf(atOtherSet(s, 1, LAMBDAS[0]! * (1 + 2e-9)))!.lines.map((l) => l.shown), [false, false]);
});

test("two rows that are both the mode shown (two sets, one lambda) mark neither; the mode is not reported missing", () => {
  const s = withTable(initialState, [modeRow("stage_0", 0, 1, LAMBDAS[0]!), modeRow("stage_1", 0, 1, LAMBDAS[0]!), modeRow("stage_0", 0, 2, LAMBDAS[1]!)]);
  const v = modalTableOf(atMode(s, 1))!;
  assert.deepEqual([v.lines.map((l) => l.shown), v.missing], [[false, false, false], null]);
});

test("mounted, the modal table redraws when the user file or the results change under the same phase", () => {
  const prevDoc = (globalThis as { document?: unknown }).document;
  const rail = new FakeEl("div");
  (globalThis as { document?: unknown }).document = { createElement: (t: string) => new FakeEl(t), getElementById: (id: string) => (id === "rail" ? rail : null) };
  let state = atMode(withTable(initialState, [modeRow("stage_0", 0, 1, LAMBDAS[0]!), modeRow("stage_0", 0, 2, LAMBDAS[1]!)]), 1);
  const listeners: ((s: State, prev: State) => void)[] = [];
  const store = { get: () => state, dispatch: () => undefined, subscribe: (l: (s: State, prev: State) => void) => (listeners.push(l), () => undefined) } as unknown as Store;
  const set = (next: State) => {
    const prev = state;
    state = next;
    for (const l of listeners) l(state, prev);
  };
  const off = mountModalTable(store);
  try {
    const panel = rail.children[0] as FakeEl;
    const shown = () => panel.all((e) => e.tagName === "TR" && e.dataset["mode"] !== undefined).map((r) => r.className);
    const status = () => panel.all((e) => e.className === "modal-status")[0]!.textContent;
    assert.deepEqual([shown(), status()], [["modal-row shown", "modal-row"], "2 rows · current"]);
    // A re-read of the user file, the phase object untouched: row 2 is now stale.
    const phase = state.phase;
    const reread = withTable(state, [modeRow("stage_0", 0, 1, LAMBDAS[0]!), { ...modeRow("stage_0", 0, 2, LAMBDAS[1]!), verdict: { stale: true, reason: "x" } }]);
    set({ ...reread, phase });
    assert.equal(status(), "2 rows · 1 stale");
    // Another results file, the phase and the user file untouched: its mode 1 is another set's, so no row is marked.
    const other = atOtherSet(state, 1, 9999);
    set({ ...other, phase: state.phase, user: state.user });
    assert.deepEqual(shown(), ["modal-row", "modal-row"]);
  } finally {
    off();
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});
