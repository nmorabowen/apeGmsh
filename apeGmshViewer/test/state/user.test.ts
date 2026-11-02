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
import { copyFileSync, mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { candidates } from "../../src/main/pairing.ts";
import { familiesOf, familyMarks, mountPhase, userBadgeOf } from "../../src/panels/phase.ts";
import { userResultsGroup } from "../../src/panels/legend.ts";
import { readUserSet, type UserH5, type UserRead } from "../../src/reader/userFile.ts";
import { initialState, NO_USER_FILE, reduce, userEventOf, userInfoOf, userPathOf, userReadable } from "../../src/state/reduce.ts";
import type { Store } from "../../src/state/store.ts";
import type { ArtifactInfo, Event, ResultsInfo, ResultsStage, State, UserReadIn } from "../../src/state/types.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const TRUSS = fileURLToPath(new URL("../../../tests/fixtures/ladruno/truss2d.ladruno", import.meta.url));
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
}

/** A set: `run.h5` (lineage only, with one `/opensees/combinations` row), a copy of truss2d.ladruno, and `run.user.h5`. */
function writeSet(o: SetOpts = {}): { user: string; model: string; results: string } {
  const d = mkdtempSync(join(tmpdir(), "agv-vu3b-"));
  const results = join(d, "run.ladruno");
  copyFileSync(TRUSS, results);
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
    f.create_group("spectra"); // listed by name only (VU7)
    if (o.extraZone) f.create_group(o.extraZone);
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
  assert.deepEqual(info.spectra, { stale: { stale: false, reason: "" } });
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
  },
  staleness: { zones: {}, results: {}, combinations: [] },
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
  assert.equal(reduce(s, { type: "userOpened", path: "u.user.h5" }), s, "a re-read keeps what was read until the reply");
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
    ["spectra", [], null, 1, 0, false],
    ["combinations", [], 4, 2, 2, false],
  ]);
  assert.deepEqual(f.map((x) => familyMarks(x)), [
    { count: " 2", stale: "", refused: "" },
    { count: " 2", stale: "", refused: "" },
    { count: "", stale: " · stale", refused: "" },
    { count: " 4", stale: " · 2 stale", refused: " · 2 refused" },
  ]);
  assert.match(f[0]!.title, /^Stages of run\.ladruno: gravity, stage_1 \(unnamed\)$/);
  assert.match(f[2]!.title, /\/spectra is stale: /);
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

test("a refused /combinations zone, and a stale or refused modal table, show on their families", () => {
  const base = withResults(initialState);
  const user = (info: Partial<NonNullable<ReturnType<typeof userInfoOf>>>): State => ({
    ...base,
    user: { status: "ready", path: "C:/w/run.user.h5", info: { warnings: [], combinations: null, results: null, spectra: null, modes: null, ...info } },
  });
  const refused = familiesOf(user({ combinations: { ok: false, reason: "refused: /combinations@current names no generation" } }));
  assert.deepEqual(refused.map((x) => [x.key, x.count, x.refused]).at(-1), ["combinations", 0, 1]);
  assert.deepEqual(familyMarks(refused.at(-1)!), { count: " 0", stale: "", refused: " · refused" });
  const stale = familiesOf(user({ modes: { ok: true, rows: 2, stale: { stale: true, reason: "the model hashes differ" } } }))[1]!;
  assert.deepEqual([stale.key, stale.stale, stale.refused, familyMarks(stale).stale], ["modes", 1, 0, " · stale"]);
  assert.match(stale.title, /the modal table of run\.user\.h5 is stale: the model hashes differ/);
  const bad = familiesOf(user({ modes: { ok: false, reason: "refused: /modes has no lambda", stale: { stale: true, reason: "the zone is refused" } } }))[1]!;
  assert.deepEqual([bad.stale, bad.refused, familyMarks(bad).refused], [1, 1, " · refused"]);
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
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = pred(this) ? [this] : [];
    for (const c of this.children) if (typeof c !== "string") out.push(...c.all(pred));
    return out;
  }
}

test("mounted, the bar shows each family as a button; the user families are disabled, and Modes opens the first mode", () => {
  const prevDoc = (globalThis as { document?: unknown }).document;
  const bar = new FakeEl("div");
  (globalThis as { document?: unknown }).document = {
    createElement: (t: string) => new FakeEl(t),
    getElementById: (id: string) => (id === "phase" ? bar : null),
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
      ["Spectra · stale", true],
      ["User Combinations 4 · 2 stale · 2 refused", true],
    ]);
    families()[1]!.click();
    assert.deepEqual(dispatched.at(-1), { type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
    assert.match(families()[1]!.className, /active/);
    // The stage picker lists the family shown: the two modes only.
    const pick = bar.all((e) => e.className === "results-stage")[0]!;
    assert.deepEqual(pick.children.map((o) => (o as FakeEl).textContent), ["mode_1", "mode_2"]);
    assert.equal(pick.title, "results stage (Modes)");
    // A disabled family dispatches nothing.
    const n = dispatched.length;
    families()[3]!.click();
    assert.equal(dispatched.length, n);
  } finally {
    off();
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});
