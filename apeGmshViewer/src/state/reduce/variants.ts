// The variants slice of the reducer (state/reduce.ts delegates here): VU10's
// study, the switch to another variant, the compare set and the difference.
// It writes `variants` only.

import type { VariantsEvent } from "../types/events/variants.ts";
import type { VariantsIn, VariantsState } from "../types.ts";

/** VU10: no study, no compare set, no difference. */
export const initialVariants: VariantsState = { study: null, switching: null, notice: null, compare: null, difference: false, read: null };

/** What a variants event reads of the rest of the state; asked only by the case that needs it. */
export interface VariantsInputs {
  /** The key of the difference the screen asks for now, or null when it asks for none (`differenceRead`). */
  differenceKey: () => string | null;
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
  return { ...v, study, compare: kept ? c : null, difference: kept && v.difference, read: kept ? v.read : null };
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
      if (e.row === null) return v.compare === null ? v : { ...v, compare: null, difference: false, read: null };
      const row = studyRow(v, e.row.index, e.row.label, "variantCompare");
      // INV-6: the compare set is never the window set.
      if (v.study!.active === row.index) throw new Error(`variantCompare: variant ${JSON.stringify(row.label)} is the window set; a compare set is another row`);
      return { ...v, compare: { index: row.index, label: row.label, status: "opening" }, difference: false, read: null, notice: null };
    }
    case "compareOpened": {
      const c = v.compare;
      const r = e.result;
      // Only the compare asked for now: a late answer for another row is dropped.
      if (c?.status !== "opening" || c.index !== r.index || c.label !== r.label) return v;
      if ("reason" in r) return { ...v, compare: { index: r.index, label: r.label, status: "failed", reason: r.reason }, notice: r.reason };
      return { ...v, compare: { status: "open", ...r } };
    }
    case "setDifference": {
      if (e.on === v.difference) return v;
      // A spec turns it on as it asks for the compare set (spec/compile.ts); the difference then waits for the set.
      const st = v.compare?.status;
      if (e.on && st !== "open" && st !== "opening") throw new Error("setDifference: no compare set is open, so there is nothing to subtract");
      return { ...v, difference: e.on, read: null };
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
