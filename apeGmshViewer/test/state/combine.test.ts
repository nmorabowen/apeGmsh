// VU9b (#2693, ADR 0130 A3, A4, A7.3): the authoring panel's judgement of a draft,
// before anything reaches main.
//
// The set is the VU9b fixture (test/fixtures/vu9b: the plate, a synthetic /spectra
// case and three rows Python wrote). A draft is checked by the readers' rules
// beside the file's rows, then by VU4's evaluator through main's `user:readCase`
// (here the same reader calls, without Electron). The oracles:
//   - Python: the preview of "SRSS Ex modes 1-2" (a srss of two increments) is the
//     range of the values Python's evaluator gives on the app-written file
//     (test/fixtures/vu9b/app_written.golden.json), over the same 91 nodes;
//   - the readers' rules (A3): a duplicate, a directional row with no fraction, a
//     cycle and a padded name are refused with their reasons, and never offered;
//   - a stage term stands but is not evaluated here (the app combines /spectra only);
//   - Save is offered only on a standing draft checked against the read on screen,
//     with that read's token; Revert only once after a save (A7.3).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { checkCombinationDraft, type Bridge } from "../../src/effects.ts";
import { checkLine, combineLines } from "../../src/panels/combine.ts";
import { combinationToken } from "../../src/main/combinations.ts";
import { readUserFile, readUserSet, type UserH5 } from "../../src/reader/userFile.ts";
import { readCaseValues } from "../../src/reader/userSpectra.ts";
import { initialState, reduce, userEventOf } from "../../src/state/reduce.ts";
import { combineCannotSave, combineEntryOf, draftJudgedOf, draftKeyOf, EMPTY_DRAFT } from "../../src/state/selectors.ts";
import type { CombinationDraft, CombineState, DraftCheck, State, UserReadIn } from "../../src/state/types.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const FIX = (p: string) => fileURLToPath(new URL(`../fixtures/vu9b/${p}`, import.meta.url));
const MODEL = FIX("plate.h5");
const CASE = "Ex (synthetic)";
const GOLDEN = JSON.parse(readFileSync(FIX("app_written.golden.json"), "utf8")) as { values: Record<string, Record<string, { ids: number[]; values: number[] }>> };

/** The state of a window that read `user` (main's `user:read`, with its token unless `token` is null). */
function withUser(user: string, token: string | null = "auto"): State {
  const read = readUserSet(H5, user, MODEL, null, () => []);
  assert.ok(read.ok, JSON.stringify(read));
  const answer: UserReadIn = token === null ? read : { ...read, token: token === "auto" ? combinationToken(user, read.user.revision) : token };
  return reduce(reduce(initialState, { type: "userOpened", path: user }), userEventOf(user, answer));
}
const S = withUser(FIX("plate.user.h5"));
const SAVED = withUser(FIX("app_written/plate.user.h5"));

const bridge: Pick<Bridge, "readUserCase"> = {
  readUserCase: async (req) => readCaseValues(H5, req.path, readUserFile(H5, req.path, null).spectra, req.case, req.target, req.label),
};
const draft = (over: Partial<CombinationDraft>): CombinationDraft => ({ ...EMPTY_DRAFT, ...over });
const SRSS12 = draft({ name: "SRSS Ex modes 1-2", rule: "srss", terms: [1, 2].map((mode) => ({ kind: "spectrum", source: CASE, step: -1, mode, factor: 1 })) });
const panel = (over: Partial<CombineState> = {}): CombineState => ({ draft: EMPTY_DRAFT, check: null, saving: null, notice: null, last: null, ...over });

test("a read carries main's save token and the generations; the phase bar offers + Combination only with a user file", () => {
  const u = S.user;
  assert.ok(u?.status === "ready");
  assert.equal(u.token, combinationToken(FIX("plate.user.h5"), 1));
  assert.deepEqual(u.info.generations, { current: 0, held: [0] });
  assert.ok(SAVED.user?.status === "ready" && SAVED.user.info.generations?.current === 3);
  assert.notEqual(combineEntryOf(S), null);
  assert.equal(combineEntryOf(initialState), null);
});

test("the preview of a valid draft is VU4's value, and its range is Python's over the same 91 nodes", async () => {
  const check = await checkCombinationDraft(bridge, S, SRSS12);
  assert.ok(check !== null && check.status === "ok", JSON.stringify(check));
  assert.equal(check.key, draftKeyOf(S, SRSS12));
  const p = check.preview!;
  assert.equal(p.label, "DISPLACEMENT/Ux");
  const py = GOLDEN.values["SRSS Ex modes 1-2"]!["displacement_x"]!;
  assert.equal(p.nodes, py.ids.length);
  assert.equal(p.signed, false, "srss is unsigned (A3)");
  const finite = py.values.filter(Number.isFinite);
  assert.ok(Math.abs(p.range!.min - Math.min(...finite)) <= 1e-12 * Math.abs(Math.min(...finite)) + 1e-300);
  assert.ok(Math.abs(p.range!.max - Math.max(...finite)) <= 1e-12 * Math.abs(Math.max(...finite)));
  assert.match(checkLine(check, check.key).text, /^stands: DISPLACEMENT\/Ux over 91 nodes, .*unsigned: contour only/);
});

test("a refused descriptor is never offered: duplicate, directional with no fraction, a cycle, a padded name, a whole case in a linear", async () => {
  const cases: [CombinationDraft, string][] = [
    [{ ...SRSS12, name: "SRSS Ex" }, "duplicate"],
    [{ ...SRSS12, rule: "directional" }, "bad-param"],
    [draft({ name: "loop", terms: [{ kind: "combination", source: "loop", step: -1, mode: 0, factor: 1 }] }), "cycle"],
    [{ ...SRSS12, name: "SRSS " }, "empty-name"],
    [draft({ name: "lin", terms: [{ kind: "spectrum", source: CASE, step: -1, mode: 0, factor: 1 }] }), "bad-term"],
  ];
  for (const [d, reason] of cases) {
    const check = await checkCombinationDraft(bridge, S, d);
    assert.ok(check?.status === "refused", `${d.name}: ${JSON.stringify(check)}`);
    assert.equal(check.reason, reason, `${d.name}: ${check.detail}`);
    assert.match(combineCannotSave(S, panel({ draft: d, check }), "add") ?? "", /^the draft is refused/);
  }
});

test("the evaluator's own refusal blocks the save too: a mode the case does not have", async () => {
  const d = { ...SRSS12, terms: [{ kind: "spectrum", source: CASE, step: -1, mode: 9, factor: 1 }] };
  const check = await checkCombinationDraft(bridge, S, d);
  assert.ok(check?.status === "refused" && check.reason === "unresolved", JSON.stringify(check));
});

test("a stage term stands but is not evaluated here, and may be saved", async () => {
  const d = draft({ name: "1.2D + 1.6L (app)", terms: [{ kind: "stage", source: "D", step: -1, mode: 0, factor: 1.2 }, { kind: "stage", source: "L", step: -1, mode: 0, factor: 1.6 }] });
  const check = await checkCombinationDraft(bridge, S, d);
  assert.ok(check?.status === "ok" && check.preview === null, JSON.stringify(check));
  assert.match(check.note ?? "", /not evaluated here: a stage or mode term is read from the results file/);
  assert.equal(combineCannotSave(S, panel({ draft: d, check }), "add"), null);
});

test("Save is offered only on a standing draft checked against the read on screen, with a token; Revert once after a save", async () => {
  const check = (await checkCombinationDraft(bridge, S, SRSS12)) as DraftCheck;
  assert.equal(combineCannotSave(S, panel({ draft: SRSS12, check }), "add"), null);
  // A check of another draft, or of another read of the file, is not this draft's.
  assert.equal(combineCannotSave(S, panel({ draft: { ...SRSS12, name: "other" }, check }), "add"), "the draft is being checked");
  const reread = withUser(FIX("plate.user.h5"), "another read");
  assert.equal(combineCannotSave(reread, panel({ draft: SRSS12, check }), "add"), "the draft is being checked");
  assert.equal(combineCannotSave(S, panel({ draft: SRSS12, check, saving: "add" }), "add"), "a save is running");
  assert.match(combineCannotSave(withUser(FIX("plate.user.h5"), null), panel({ draft: SRSS12, check }), "add")!, /only the human's window/);
  assert.match(combineCannotSave(initialState, panel(), "add")!, /no \.user\.h5/);
});

test("Revert is offered only right after a save made in this window, while the file is as that save left it, once", () => {
  const u = SAVED.user;
  assert.ok(u?.status === "ready");
  // A fresh open: g3 is the newest and in force, and still nothing to revert (no save was made here).
  assert.match(combineCannotSave(SAVED, panel(), "revert")!, /none was made on this file/);
  assert.match(combineCannotSave(S, panel(), "revert")!, /none was made on this file/);
  const last = { op: "add" as const, name: "1.3 SRSS Ex", generation: 3, file: u.path, target: 2, token: u.token ?? null };
  assert.equal(combineCannotSave(SAVED, panel({ last }), "revert"), null, "right after this window's save");
  assert.match(combineCannotSave(SAVED, panel({ last: { ...last, op: "revert", target: null } }), "revert")!, /reverted already/);
  assert.match(combineCannotSave(SAVED, panel({ last: { ...last, token: "an earlier read" } }), "revert")!, /changed since the last save/);
  assert.match(combineCannotSave(SAVED, panel({ last: { ...last, generation: 2 } }), "revert")!, /changed since the last save/);
  assert.match(combineCannotSave(SAVED, panel({ last: { ...last, target: null } }), "revert")!, /created \/combinations/);
  assert.match(combineCannotSave(SAVED, panel({ last: { ...last, file: "elsewhere.user.h5" } }), "revert")!, /none was made on this file/);
});

test("the panel's lines: the file, the generation in force, the check, and a held refusal in its own words", async () => {
  const check = (await checkCombinationDraft(bridge, S, SRSS12)) as DraftCheck;
  const l = combineLines(S, panel({ draft: SRSS12, check }))!;
  assert.equal(l.title, "New combination · plate.user.h5");
  assert.equal(l.generation, "g0 in force (3 rows); 1 generation kept");
  assert.equal(l.canSave, true);
  assert.equal(l.canRevert, false);
  const held = combineLines(S, panel({ notice: { reason: "held", detail: "x is open in another program", file: "x", held: true } }))!;
  assert.deepEqual(held.notice, { text: "x is open in another program", held: true });
  assert.equal(combineLines(initialState, panel()), null);
  assert.deepEqual(checkLine(null, null), { text: "no user file is read", tone: "muted" });
  const j = draftJudgedOf(S, SRSS12);
  assert.ok(j?.ok && j.cases[0] === CASE && j.quantity?.label === "DISPLACEMENT");
});
