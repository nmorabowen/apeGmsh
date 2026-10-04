// Group colours (R2 on #1283, revised on #1308 round 3 and by the
// maintainer's ruling on #1330): the office palette (theme/tokens.ts, from
// apeGraphStyle v0.1.0) adapted to the dark theme, with the slots assigned
// so that groups that are neighbours in the view contrast.
//
// The sixteen slots are the eight office colours (ring 0) and the same
// hues one lightness step away (ring 1, a striped legend chip as the second
// cue). `assignSlots` colours the group-adjacency graph greedily: the group
// with most neighbours first (ties by legend order), each taking the slot
// that maximises the smallest colour-vision-deficiency-simulated OKLab
// distance (protanopia and deuteranopia) to its already-coloured
// neighbours, never a slot in the same hue family as a neighbour when
// another is free, and never office orange beside dark gold (12 degrees
// apart, one hue to a protanope). A group with no coloured neighbour takes
// the lowest free slot. The assignment is a pure function of the
// adjacency, so the same file gives the same colours.

import { DARK, DARK_MAIN, DARK_MAIN_RING2, hexToRgb, oklabDistance, simulateCvd, srgbToOklch, type RGB } from "../theme/tokens.ts";

/** Elements in no physical group, and OpenSees-only elements. */
export const NO_GROUP: RGB = hexToRgb(DARK.noGroup);
export const OPS_ONLY: RGB = hexToRgb(DARK.opsOnly);

/** The rings of the group palette: the office colours, then the same hues one lightness step away. */
export const RINGS: readonly (readonly RGB[])[] = [DARK_MAIN.map(hexToRgb), DARK_MAIN_RING2.map(hexToRgb)];
/** Office colours per ring. */
export const RING_SIZE = DARK_MAIN.length;
/** Slots: ring 0 then ring 1. */
export const SLOTS: readonly RGB[] = RINGS.flat();
/** Office indices that may never colour adjacent groups, the office orange (1) beside the dark gold (4). */
export const NEVER_ADJACENT: readonly [number, number][] = [[1, 4]];
/** What two adjacent groups must keep: this much CVD-simulated OKLab distance, and a lightness step or a hue gap. */
export const ADJACENT_MIN_DISTANCE = 0.05;
export const ADJACENT_MIN_DL = 0.1;

/** The colour and ring of slot `s`. */
export function slotColour(s: number): { color: RGB; ring: number } {
  if (!Number.isInteger(s) || s < 0 || s >= SLOTS.length) throw new RangeError(`slotColour: ${s}`);
  return { color: SLOTS[s]!, ring: Math.floor(s / RING_SIZE) };
}

/** Office hue family of a slot. */
export const hueOf = (s: number): number => s % RING_SIZE;

/** The smaller of the protanopia- and deuteranopia-simulated OKLab distances between two slots. */
export function cvdDistance(a: number, b: number): number {
  return Math.min(
    ...(["protanopia", "deuteranopia"] as const).map((k) => oklabDistance(simulateCvd(k, SLOTS[a]!), simulateCvd(k, SLOTS[b]!))),
  );
}

const DIST: number[][] = SLOTS.map((_, a) => SLOTS.map((__, b) => cvdDistance(a, b)));
const LIGHT: number[] = SLOTS.map((c) => srgbToOklch(c).L);

const forbidden = (a: number, b: number): boolean =>
  NEVER_ADJACENT.some(([x, y]) => (hueOf(a) === x && hueOf(b) === y) || (hueOf(a) === y && hueOf(b) === x));

/** Whether two slots may colour adjacent groups: far enough as a protanope and a deuteranope see them, by lightness or by a hue gap, and not the office orange beside the dark gold. */
export function slotsContrast(a: number, b: number): boolean {
  if (forbidden(a, b)) return false;
  return DIST[a]![b]! >= ADJACENT_MIN_DISTANCE && (Math.abs(LIGHT[a]! - LIGHT[b]!) >= ADJACENT_MIN_DL - 0.012 || hueOf(a) !== hueOf(b));
}

/**
 * Assign a slot to each of `n` groups so that adjacent groups contrast.
 * `adjacency` lists pairs of group indices. Returns the slot of each group.
 *
 * Greedy, largest degree first (ties by index): among the slots that
 * contrast with every already-coloured neighbour, an unused slot first (so
 * the legend stays varied), then the largest smallest CVD distance to those
 * neighbours, then the lowest slot. When no slot contrasts with them all
 * (more neighbours than the palette can separate), the one with the largest
 * smallest distance is taken anyway; the stripe cue past the office eight
 * still tells its ring.
 */
export function assignSlots(n: number, adjacency: readonly (readonly [number, number])[]): number[] {
  const neighbours: Set<number>[] = Array.from({ length: n }, () => new Set());
  for (const [a, b] of adjacency) {
    if (a === b || a < 0 || b < 0 || a >= n || b >= n) throw new RangeError(`assignSlots: pair ${a},${b} of ${n} groups`);
    neighbours[a]!.add(b);
    neighbours[b]!.add(a);
  }
  const order = [...Array(n).keys()].sort((x, y) => neighbours[y]!.size - neighbours[x]!.size || x - y);
  const slot = new Array<number>(n).fill(-1);
  const used = new Set<number>();
  for (const g of order) {
    const coloured = [...neighbours[g]!].filter((h) => slot[h]! >= 0).map((h) => slot[h]!);
    const minDist = (s: number) => (coloured.length ? Math.min(...coloured.map((c) => DIST[s]![c]!)) : 1);
    const key = (s: number): [number, number, number, number] => [
      coloured.every((c) => slotsContrast(s, c)) ? 1 : 0,
      used.has(s) ? 0 : 1,
      minDist(s),
      -s,
    ];
    const better = (p: number[], q: number[]) => p.findIndex((v, i) => v !== q[i]) >= 0 && p[p.findIndex((v, i) => v !== q[i])]! > q[p.findIndex((v, i) => v !== q[i])]!;
    let best = 0;
    for (let s = 1; s < SLOTS.length; s++) if (better(key(s), key(best))) best = s;
    slot[g] = best;
    used.add(best);
  }
  return slot;
}
