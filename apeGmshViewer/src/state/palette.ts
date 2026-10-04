// Group colours (R2 on #1283, revised on #1308 round 3): the office palette
// (theme/tokens.ts, from apeGraphStyle v0.1.0), in its order, adapted to
// the dark theme.
//
// Legend entry i takes office colour i % 8. The first eight alternate in
// lightness (every legend neighbour steps by at least 0.10 in OKLCH L) and
// never rely on a red/green difference alone (test/tokens.test.ts simulates
// protanopia and checks every pair). Past eight groups the hue repeats one
// lightness step away and the legend chip carries a stripe: the second cue,
// as apeGraphStyle's `redundant_cycle` adds a line style. Past sixteen the
// cycle repeats and the chip takes a second stripe; the maintainer's models
// have eleven groups.

import { DARK, DARK_MAIN, DARK_MAIN_RING2, hexToRgb, type RGB } from "../theme/tokens.ts";

/** Elements in no physical group, and OpenSees-only elements. */
export const NO_GROUP: RGB = hexToRgb(DARK.noGroup);
export const OPS_ONLY: RGB = hexToRgb(DARK.opsOnly);

/** The rings of the group palette: the office colours, then the same hues one lightness step away. */
export const RINGS: readonly (readonly RGB[])[] = [DARK_MAIN.map(hexToRgb), DARK_MAIN_RING2.map(hexToRgb)];

/** Office colours per ring. */
export const RING_SIZE = DARK_MAIN.length;

/** The colour of legend entry `i`, and its ring (0: the office colour; 1+: a repeat, with a stripe cue). */
export function paletteEntry(i: number): { color: RGB; ring: number } {
  if (!Number.isInteger(i) || i < 0) throw new RangeError(`paletteEntry: ${i}`);
  const ring = Math.floor(i / RING_SIZE);
  return { color: RINGS[Math.min(ring, RINGS.length - 1)]![i % RING_SIZE]!, ring };
}

/** The colours of `n` legend entries, in legend order. */
export function paletteFor(n: number): RGB[] {
  return Array.from({ length: n }, (_, i) => paletteEntry(i).color);
}
