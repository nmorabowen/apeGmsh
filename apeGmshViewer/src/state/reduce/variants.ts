// The variants slice of the reducer (state/reduce.ts delegates here): VU10's
// study, the switch to another variant, the compare set and the difference.
// It writes `variants` only.

import type { VariantsEvent } from "../types/events/variants.ts";
import type { CompareSlotIn, CompareUserState, UserReadIn, VariantsIn, VariantsState } from "../types.ts";
import { NO_USER_FILE, userInfoOf } from "./user.ts";

/** VU10: no study, no compare set, no difference. VU19: no compare model, user file or compare user result. */
export const initialVariants: VariantsState = {
  study: null,
  switching: null,
  notice: null,
  compare: null,
  difference: false,
  read: null,
  compareModel: null,
  compareUser: null,
  shown: null,
  shownRead: null,
};

/** VU19: what goes with the compare set: its model, its user file and the compare user result drawn. */
const NO_COMPARE_READS = { compareModel: null, compareUser: null, shown: null, shownRead: null } as const;

/** What a variants event reads of the rest of the state; asked only by the case that needs it. */
export interface VariantsInputs {
  /** The key of the difference the screen asks for now, or null when it asks for none (`differenceRead`). */
  differenceKey: () => string | null;
  /** VU19: the key of the compare user result the screen asks for now, or null when it asks for none (`compareUserResultRead`). */
  compareUserKey: () => string | null;
}

/** Whether `slot` is the open compare set: a compare answer for any other slot (a closed or replaced one) is dropped. */
function isOpenSlot(v: VariantsState, slot: CompareSlotIn | null): boolean {
  const c = v.compare;
  return slot !== null && c?.status === "open" && c.index === slot.index && c.label === slot.label;
}

/** The compare user file state `compare:userRead`'s answer makes; a malformed answer raises, so it is never shown. */
function compareUserOf(path: string | null, read: UserReadIn): CompareUserState {
  if (read === null || typeof read !== "object" || typeof read.ok !== "boolean") throw new RangeError(`compareUserRead: ${JSON.stringify(read)} is not a user read`);
  if (!read.ok) {
    if (typeof read.error !== "string") throw new RangeError(`compareUserRead: error ${JSON.stringify(read.error)} is not a string`);
    // No file beside the compare set's model is no failure: the set has no .user.h5.
    if (path !== null && read.error === NO_USER_FILE(path)) return { status: "none", path };
    return { status: "failed", path, error: read.error };
  }
  if (path === null || read.user.path !== path) throw new Error(`compareUserRead: the answer is for ${read.user.path}, not ${String(path)}`);
  return { status: "ready", path, info: userInfoOf(read.user, read.staleness) };
}

/** The study's row `index` when its label is `label` (the panel offers only those); a stale or unknown row is a bug. */
function studyRow(v: VariantsState, index: number, label: string, who: string): VariantsIn["rows"][number] {
  const study = v.study;
  if (!study) throw new Error(`${who}: no variants manifest is open`);
  const row = study.rows.find((r) => r.index === index);
  if (!row || row.label !== label) throw new Error(`${who}: the study has no variant row ${index} labelled ${JSON.stringify(label)}`);
  return row;
}

/**
 * VU10: main's study replaces the page's. With none, the compare set and the
 * difference go; a compare set main no longer holds (its row left the
 * manifest, or main closed it) goes too, so the page never shows a slot main
 * has emptied. A switch in flight stays until main answers it.
 */
function withStudy(v: VariantsState, study: VariantsIn | null): VariantsState {
  if (study !== null && (typeof study !== "object" || !Array.isArray(study.rows) || !Array.isArray(study.refused) || typeof study.manifest !== "string")) {
    throw new RangeError(`variantsRead: ${JSON.stringify(study)} is not a study`);
  }
  const c = v.compare;
  const kept = c !== null && study !== null && (c.status === "opening" || (study.compare !== null && study.compare.index === c.index && study.compare.label === c.label));
  // VU19: a compare set main no longer holds takes its model, its user file and the user result drawn from it along.
  return kept ? { ...v, study } : { ...v, study, compare: null, difference: false, read: null, ...NO_COMPARE_READS };
}

/** The variants after `e`: the same object when nothing changed. */
export function reduceVariants(v: VariantsState, e: VariantsEvent, inputs: VariantsInputs): VariantsState {
  switch (e.type) {
    case "variantsRead":
      return withStudy(v, e.study);
    case "variantSwitch": {
      const row = studyRow(v, e.index, e.label, "variantSwitch");
      if (v.study!.active === row.index) return v;
      return { ...v, switching: { index: row.index, label: row.label }, notice: null };
    }
    case "variantSwitched":
      if (v.switching === null) return v;
      return { ...v, switching: null, notice: e.reason };
    case "variantCompare": {
      if (e.row === null) return v.compare === null ? v : { ...v, compare: null, difference: false, read: null, ...NO_COMPARE_READS };
      const row = studyRow(v, e.row.index, e.row.label, "variantCompare");
      // INV-6: the compare set is never the window set.
      if (v.study!.active === row.index) throw new Error(`variantCompare: variant ${JSON.stringify(row.label)} is the window set; a compare set is another row`);
      return { ...v, compare: { index: row.index, label: row.label, status: "opening" }, difference: false, read: null, notice: null, ...NO_COMPARE_READS };
    }
    case "compareOpened": {
      const c = v.compare;
      const r = e.result;
      // Only the compare asked for now: a late answer for another row is dropped.
      if (c?.status !== "opening" || c.index !== r.index || c.label !== r.label) return v;
      if ("reason" in r) return { ...v, compare: { index: r.index, label: r.label, status: "failed", reason: r.reason }, notice: r.reason };
      // VU19: the runner then reads the compare set's model and user file through the compare doors.
      return { ...v, compare: { status: "open", ...r }, compareModel: { status: "reading" }, compareUser: { status: "reading" } };
    }
    case "setDifference": {
      if (e.on === v.difference) return v;
      // A spec turns it on as it asks for the compare set (spec/compile.ts); the difference then waits for the set.
      const st = v.compare?.status;
      if (e.on && st !== "open" && st !== "opening") throw new Error("setDifference: no compare set is open, so there is nothing to subtract");
      // VU19: one contour at a time: the difference replaces a compare user result drawn.
      return { ...v, difference: e.on, read: null, ...(e.on ? { shown: null, shownRead: null } : {}) };
    }
    case "compareModelRead": {
      const a = e.answer;
      if (a === null || typeof a !== "object" || typeof a.ok !== "boolean") throw new RangeError(`compareModelRead: ${JSON.stringify(a)} is not a compare model answer`);
      // Only for the compare set open now: a late answer for a closed or replaced slot is dropped.
      if (!isOpenSlot(v, a.slot)) return v;
      if (!a.ok) return { ...v, compareModel: { status: "failed", reason: String(a.reason) } };
      if (typeof a.femHash !== "string" || a.femHash === "" || typeof a.modelHash !== "string" || !Number.isInteger(a.nodes) || !Number.isInteger(a.elements)) {
        throw new RangeError(`compareModelRead: ${JSON.stringify(a)} names no fem_hash, model_hash and counts`);
      }
      return { ...v, compareModel: { status: "ready", femHash: a.femHash, modelHash: a.modelHash, nodes: a.nodes, elements: a.elements } };
    }
    case "compareUserRead": {
      const a = e.answer;
      if (a === null || typeof a !== "object" || (a.path !== null && typeof a.path !== "string")) throw new RangeError(`compareUserRead: ${JSON.stringify(a)} is not a compare user answer`);
      if (!isOpenSlot(v, a.slot)) return v;
      return { ...v, compareUser: compareUserOf(a.path, a.read), shown: null, shownRead: null };
    }
    case "showCompareUser": {
      const p = e.pick;
      if (p === null) return v.shown === null ? v : { ...v, shown: null, shownRead: null };
      if (typeof p.name !== "string" || typeof p.component !== "string") throw new RangeError(`showCompareUser: ${JSON.stringify(p)} is not a user result and a component`);
      // The panel offers a compare user result only once the compare set's file is read; another pick is a bug.
      if (v.compare?.status !== "open" || v.compareUser?.status !== "ready") throw new Error("showCompareUser: no compare set's user file is read, so there is no compare user result to draw");
      if (v.shown?.name === p.name && v.shown.component === p.component) return v;
      return { ...v, shown: { name: p.name, component: p.component }, shownRead: null, difference: false, read: null };
    }
    case "compareUserResultRead": {
      const r = e.read;
      if (r === null || typeof r !== "object" || typeof r.key !== "string" || !["loading", "ready", "refused", "failed"].includes(r.status)) {
        throw new RangeError(`compareUserResultRead: ${JSON.stringify(r)} is not a user result read`);
      }
      if (r.status === "refused" && r.reason !== "refused" && r.reason !== "stale") throw new RangeError(`compareUserResultRead: refusal reason ${JSON.stringify(r.reason)} is not refused or stale`);
      // Only for the compare user result shown now: a late answer for another is dropped.
      if (inputs.compareUserKey() !== r.key) return v;
      return { ...v, shownRead: r };
    }
    case "differenceRead": {
      const r = e.read;
      if (r === null || typeof r !== "object" || typeof r.key !== "string" || !["loading", "ready", "refused"].includes(r.status)) {
        throw new RangeError(`differenceRead: ${JSON.stringify(r)} is not a difference read`);
      }
      // Only for the difference shown now: a late answer for another key is dropped.
      if (inputs.differenceKey() !== r.key) return v;
      return { ...v, read: r };
    }
    default: {
      const unknown: never = e;
      throw new Error(`reduceVariants: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
