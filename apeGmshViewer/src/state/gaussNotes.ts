// What the Gauss row on the Display card states: why it cannot draw, and what
// it leaves out. Apart from state/gaussPoints.ts, which reads the selectors, so
// the selectors can re-export these for the panels without an import cycle.

import type { GaussGroup, State } from "./types.ts";

/**
 * What the Gauss row states while on: groups the file cannot place (counted by
 * reason, never skipped silently), and that solid surfaces hide interior points.
 * Element-level refusals (a type with no shape functions) are counted by the
 * layout itself and are not listed here.
 */
export function gaussNotesOf(s: State): string[] {
  const at = s.phase.at;
  const stage = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
  if (!stage) return [];
  const out: string[] = [];
  const m = s.gaussPoints.measured;
  if (s.gaussPoints.on && m && m.stage === stage.id) {
    // What the viewport placed: every refusal it counted, by reason.
    if (m.count === 0) out.push("No Gauss point or beam station placed on this stage.");
    for (const r of m.refused) out.push(`${r.elements.toLocaleString()} ${r.elements === 1 ? "element" : "elements"} not placed: ${r.reason}.`);
    out.push(...m.notes);
    if (s.mesh && s.mesh.counts.solidCells > 0 && s.visibility.opacity >= 1) out.push("Points inside solids are hidden by the surfaces: set them to ghost or off.");
    return out;
  }
  const groups = stage.gauss;
  // Counted by group: two groups (stresses, strains) often cover the same elements, so element counts would double.
  const of = (pick: (g: GaussGroup) => boolean) => groups.filter(pick).length;
  const unverified = of((g) => !g.elementIds);
  const noCoords = of((g) => Boolean(g.elementIds) && !g.naturalCoords);
  const n = groups.length;
  if (unverified) out.push(`${unverified} of ${n} Gauss groups not placed: their element ids are not stamped as the model's.`);
  if (noCoords) out.push(`${noCoords} of ${n} Gauss groups not placed: the file gives no usable Gauss point coordinates for them.`);
  if (s.mesh && s.mesh.counts.solidCells > 0 && s.visibility.opacity >= 1) out.push("Points inside solids are hidden by the surfaces: set them to ghost or off.");
  if (!s.mesh?.nodes) out.push("This model was read without its node table, so no point can be placed.");
  return out;
}

/**
 * Why the Gauss row can do nothing, or null when it can: no results stage
 * shown, or a stage with neither Gauss groups nor beam stations.
 */
export function gaussUnavailable(s: State): string | null {
  const at = s.phase.at;
  if (at?.kind !== "results") return "Gauss points come from a results stage: pick one in the phase bar";
  const stage = s.results?.stages[at.stage];
  if (!stage) return "no results stage";
  if (!stage.gauss.length && !stage.lineStations?.length) return "this stage records no Gauss points or beam stations";
  return null;
}
