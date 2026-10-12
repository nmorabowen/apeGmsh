// Stage activation for a results picture (V4b, #1728; X3 cluster 3: it
// applies to every results picture of a staged model).
//
// Which elements exist while a results stage is shown comes from the paired
// model's `/sequence` zone (V3a): its element change rows, `activate element`
// (a stage's `owned_element_ids`: the elements its `s.activate(pgs=)` brings
// in) and `remove element` (a stage's `remove_element`). The rows give a
// count and the archive path they were read from (`source`); the tags are
// that dataset, which main reads (src/main/zones.ts, `StageElements`).
//
//   active(K) = everything - (owned by any stage - owned by stages <= K)
//                          - removed by stages <= K
//
// as Python's `viewers/data/_stage_activation.py::build_stage_activation`
// has it (read for its semantics, never imported): an element a later stage
// activates is hidden until then; a removal hides the element in the stage
// that removes it, and after. `activate group` rows are the same activation
// by physical-group name (`owned_element_ids` is its resolved list), so they
// add nothing; `activate absorbing`, `set material_stage` and every other
// pair change no element's existence. An op or what outside the /sequence
// 1.0.0 vocabulary, or any element row this code cannot apply, refuses the
// activation (`refused`, with the reason; never skipped):
// it might remove an element this code cannot see.
//
// A results stage pairs with a `/sequence` stage by name: h5-schema.md calls
// that a viewer convention, not a stored reference, so anything but one
// exact match is a typed state that draws every element and says why.
// Pure: tested in Node (test/results/activation.test.ts).

import type { Definitions, SequenceZone, StageElements } from "./types.ts";

export type Activation =
  /** a unique name match: the tags of the elements that do not exist at that stage */
  | { state: "filtered"; row: number; name: string; inactiveTags: ReadonlySet<number> }
  /** the model has no stages: every element exists, which is correct, not a fallback */
  | { state: "unstaged" }
  /** two or more `/sequence` stages match, or the results stage has no name */
  | { state: "ambiguous"; reason: string }
  /** no `/sequence`, no stage of that name, or change rows that cannot be read without guessing */
  | { state: "absent"; reason: string }
  /** the `/sequence` element rows cannot be applied without guessing (an op this code does not know, a source or count that disagrees, a read error) */
  | { state: "refused"; reason: string }
  /**
   * the results pair with this model by fem_hash only: the mesh matches, the
   * OpenSees program is not proven to be the one that ran, so no `/sequence`
   * stage is joined to them (selectors.activationNow decides it)
   */
  | { state: "unbound"; reason: string };

/** The `/sequence/changes` vocabulary (h5-schema.md, `/sequence`, "Changes"). */
const OPS: ReadonlySet<string> = new Set(["add", "remove", "activate", "set"]);
const WHATS: ReadonlySet<string> = new Set([
  "pattern", "recorder", "fix", "mass", "element", "group", "region",
  "rayleigh", "initial_stress", "absorbing", "material_stage", "constraint",
]);
/** The element rows and the stage dataset each one's source must name. */
const ELEMENT_SOURCE: Readonly<Record<string, string>> = { activate: "owned_element_ids", remove: "remove_element" };

function columns(t: { columns: string[] }, want: readonly string[], where: string): number[] | string {
  const at = want.map((c) => t.columns.indexOf(c));
  const missing = want.filter((_, i) => at[i] === -1);
  return missing.length ? `${where} has no ${missing.map((c) => `'${c}'`).join(", ")} column` : at;
}

/**
 * The activation of the results stage named `name` on the model whose
 * `/sequence` is `seq` (`stageElements` the tags its element rows name;
 * `definitions` the model's /opensees data, which tells an unstaged model
 * from one that predates `/sequence`).
 */
export function activationOf(
  seq: SequenceZone,
  stageElements: StageElements,
  definitions: Definitions | null,
  name: string,
): Activation {
  if (seq.status !== "ready") {
    // No /sequence to pair by name. A model whose /opensees has no stages is
    // unstaged all the same: there is nothing to activate.
    if (definitions && definitions.stages.length === 0) return { state: "unstaged" };
    const why = seq.status === "refused" ? `its /sequence ${seq.version} was not read (${seq.reason})` : "it has no /sequence zone";
    return { state: "absent", reason: `stage activation is not applied: ${why}, so its stages cannot be paired by name` };
  }
  const staged = seq.attrs["staged"];
  if (staged === 0 || seq.stages.rows.length === 0) return { state: "unstaged" };
  if (staged !== 1) return { state: "absent", reason: `stage activation is not applied: /sequence@staged is ${JSON.stringify(staged)}, not 0 or 1` };
  if (name === "") {
    return { state: "ambiguous", reason: "the results stage has no name, so it pairs with no /sequence stage (never with the first one)" };
  }
  const sc = columns(seq.stages, ["order", "name"], "/sequence/stages");
  if (typeof sc === "string") return { state: "absent", reason: `stage activation is not applied: ${sc}` };
  const [oc, nc] = sc as [number, number];
  const hits = seq.stages.rows.flatMap((r, i) => (r[nc] === name ? [i] : []));
  if (hits.length === 0) {
    return { state: "absent", reason: `no /sequence stage is named ${JSON.stringify(name)}, so the results stage pairs with none` };
  }
  if (hits.length > 1) {
    return { state: "ambiguous", reason: `${hits.length} /sequence stages are named ${JSON.stringify(name)} (rows ${hits.join(", ")}), so the results stage pairs with none of them` };
  }
  const row = hits[0]!;
  const order = seq.stages.rows[row]![oc];
  if (typeof order !== "number" || !Number.isInteger(order)) {
    return { state: "absent", reason: `stage activation is not applied: /sequence/stages order[${row}] is ${JSON.stringify(order)}, not an integer` };
  }
  const tags = inactiveAt(seq, stageElements, order);
  if (typeof tags === "string") return { state: "refused", reason: tags };
  return { state: "filtered", row, name, inactiveTags: tags };
}

/** The tags inactive at stage order `k`, or why they cannot be known. */
function inactiveAt(seq: Extract<SequenceZone, { status: "ready" }>, stageElements: StageElements, k: number): Set<number> | string {
  const cc = columns(seq.changes, ["stage", "op", "what", "count", "source"], "/sequence/changes");
  if (typeof cc === "string") return cc;
  const [sc, opc, wc, cnt, src] = cc as [number, number, number, number, number];
  // The rows of each element source, and the counts they claim.
  const claimed = new Map<string, { stage: number; op: string; count: number }>();
  for (const [i, r] of seq.changes.rows.entries()) {
    const op = r[opc], what = r[wc], stage = r[sc], count = r[cnt], source = r[src];
    if (typeof op !== "string" || typeof what !== "string" || !OPS.has(op) || !WHATS.has(what)) {
      return `/sequence/changes row ${i} is (${JSON.stringify(op)}, ${JSON.stringify(what)}), outside the /sequence 1.0.0 vocabulary: it may change which elements exist`;
    }
    if (what !== "element") continue;
    const dataset = ELEMENT_SOURCE[op];
    if (!dataset) return `/sequence/changes row ${i} is (${op}, element), which this app does not know how to apply`;
    if (typeof stage !== "number" || typeof count !== "number" || typeof source !== "string") return `/sequence/changes row ${i} has a malformed stage, count or source`;
    const m = /^\/opensees\/stages\/stage_(\d+)\/([a-z_]+)$/.exec(source);
    if (!m || Number(m[1]) !== stage || m[2] !== dataset) {
      return `/sequence/changes row ${i} (${op} element, stage ${stage}) names ${JSON.stringify(source)}, not /opensees/stages/stage_${String(stage).padStart(3, "0")}/${dataset}`;
    }
    const prev = claimed.get(source);
    claimed.set(source, { stage, op, count: (prev?.count ?? 0) + count });
  }
  if (claimed.size === 0) return new Set();
  if (stageElements.status === "error") return stageElements.reason;
  if (stageElements.status === "none") return "the element tags of the /sequence element rows were not read";
  const owned = new Set<number>();
  const ownedSoFar = new Set<number>();
  const removed = new Set<number>();
  for (const [source, c] of claimed) {
    const tags = stageElements.sets[source];
    if (!tags) return `${source} was not read`;
    if (tags.length !== c.count) return `${source} holds ${tags.length} tags, and its /sequence rows count ${c.count}`;
    for (const t of tags) {
      if (c.op === "activate") {
        owned.add(t);
        if (c.stage <= k) ownedSoFar.add(t);
      } else if (c.stage <= k) removed.add(t);
    }
  }
  const out = new Set<number>(removed);
  for (const t of owned) if (!ownedSoFar.has(t)) out.add(t);
  return out;
}

/** The banner sentence for an activation that is not `filtered`/`unstaged`, or a note for `filtered`. */
export function activationNotice(a: Activation, resultsStage: string): string | null {
  switch (a.state) {
    case "filtered":
    case "unstaged":
      return null;
    case "ambiguous":
      return `Stage activation ambiguous for results stage ${JSON.stringify(resultsStage)}: ${a.reason}. Every element is drawn.`;
    case "absent":
      return `Stage activation absent for results stage ${JSON.stringify(resultsStage)}: ${a.reason}. Every element is drawn.`;
    case "refused":
      return `Stage activation refused for results stage ${JSON.stringify(resultsStage)}: ${a.reason}. Every element is drawn.`;
    case "unbound":
      return `Stage activation unbound for results stage ${JSON.stringify(resultsStage)}: ${a.reason}. Every element is drawn.`;
    default: {
      const unknown: never = a;
      throw new Error(`activationNotice: unknown activation ${JSON.stringify(unknown)}`);
    }
  }
}
