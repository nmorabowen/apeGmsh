// Group colours (R2 on #1283): generated, not picked from a list, so that
// no two groups share a hue family and groups next to each other in the
// legend contrast.
//
// `n` groups take `n` hues evenly spaced on the wheel (for n <= 12 that is
// at least 30 degrees apart, one hue family each). The legend order is a
// greedy farthest-first walk over those hues, so consecutive entries are
// at least floor(n / 2) - 1 steps apart. Hues 30 degrees apart still read
// as one family side by side (two pinks on the V1 gate), so lightness
// alternates between neighbouring hue slots: two slots less than 45 degrees
// apart never share a tone. Beyond 12 groups a hue family has to repeat (a
// second ring of the 12 hues); a hue, its repeat and both their neighbours
// are then all within 45 degrees, which takes four tones: tone = 2 * (hue
// parity) + ring. That holds up to 24 groups; a third ring (25 or more)
// repeats a tone. Colours are built in OKLCH, so no hue reads darker or
// duller than its tone says, and clipped to sRGB by reducing chroma only.

/** Hue families: the wheel in 30-degree sectors. */
export const HUE_FAMILIES = 12;
/** The first hue (a blue, as the old fixed palette started). */
const HUE0 = 250;
/** Tones: mid, deep, pale, dark; any two differ in lightness by at least 0.14. Neighbouring hue slots take different tones. */
export const TONES: readonly { L: number; C: number }[] = [
  { L: 0.74, C: 0.15 },
  { L: 0.52, C: 0.13 },
  { L: 0.88, C: 0.11 },
  { L: 0.36, C: 0.1 },
];
/** The most groups for which no two hues under SAME_FAMILY_DEG apart share a tone. */
export const TONE_GUARANTEE = 2 * HUE_FAMILIES;
/** Two hues closer than this read as one family and must differ in tone. */
export const SAME_FAMILY_DEG = 45;

export const NO_GROUP: readonly [number, number, number] = [0.62, 0.64, 0.68];
export const OPS_ONLY: readonly [number, number, number] = [0.93, 0.93, 0.93];

const circular = (a: number, b: number, n: number) => Math.min(Math.abs(a - b), n - Math.abs(a - b));

/**
 * The order in which `n` evenly spaced hue slots are handed to the legend:
 * each next slot is the unused one farthest from the previous entry, ties
 * broken by distance from the entry before that, then by index.
 */
export function hueOrder(n: number): number[] {
  if (!Number.isInteger(n) || n < 0) throw new RangeError(`hueOrder: ${n}`);
  const used = new Array<boolean>(n).fill(false);
  const out: number[] = [];
  for (let k = 0; k < n; k++) {
    let best = -1, bestKey = [-1, -1];
    for (let s = 0; s < n; s++) {
      if (used[s]) continue;
      const key = out.length === 0 ? [n, n] : [circular(s, out[out.length - 1]!, n), out.length > 1 ? circular(s, out[out.length - 2]!, n) : 0];
      if (key[0]! > bestKey[0]! || (key[0] === bestKey[0] && key[1]! > bestKey[1]!)) {
        best = s;
        bestKey = key;
      }
    }
    used[best] = true;
    out.push(best);
  }
  return out;
}

/**
 * Hue in degrees and tone of legend entry `i` of `n`. Up to 12 groups the
 * tone cycles over the hue slots with a period of 2 (even n) or 3 (odd n), so
 * that slot k and slot k + 1, and the last slot and slot 0, never share one.
 * Beyond 12 the 12 hues repeat in rings and the tone is 2 * (hue parity) +
 * ring: the four hues within 45 degrees of each other (a hue, its neighbour,
 * and both repeats) get four tones. A third ring (n > 24) wraps.
 */
export function hueOf(i: number, n: number): { hue: number; tone: number } {
  const slot = hueOrder(n)[i];
  if (slot === undefined) throw new RangeError(`hueOf: entry ${i} of ${n}`);
  const families = Math.min(n, HUE_FAMILIES);
  const family = slot % families, ring = Math.floor(slot / families);
  const hue = (HUE0 + family * (360 / families)) % 360;
  if (n <= HUE_FAMILIES) return { hue, tone: family % (families % 2 === 0 ? 2 : 3) };
  return { hue, tone: (2 * (family % 2) + ring) % TONES.length };
}

/** OKLCH (L in 0..1, C, h in degrees) to sRGB in 0..1, reducing chroma until it fits the gamut. */
export function oklchToSrgb(L: number, C: number, h: number): [number, number, number] {
  for (let c = C; c >= 0; c -= 0.005) {
    const rgb = linear(L, c, h);
    if (rgb.every((v) => v >= -1e-6 && v <= 1 + 1e-6)) return rgb.map((v) => Math.round(gamma(Math.min(1, Math.max(0, v))) * 1000) / 1000) as [number, number, number];
  }
  return linear(L, 0, h).map((v) => gamma(Math.min(1, Math.max(0, v)))) as [number, number, number];
}

function linear(L: number, C: number, h: number): [number, number, number] {
  const a = C * Math.cos((h * Math.PI) / 180), b = C * Math.sin((h * Math.PI) / 180);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}

const gamma = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** The colours of `n` legend entries, in legend order. */
export function paletteFor(n: number): (readonly [number, number, number])[] {
  const out: (readonly [number, number, number])[] = [];
  for (let i = 0; i < n; i++) {
    const { hue, tone } = hueOf(i, n);
    const t = TONES[tone]!;
    out.push(oklchToSrgb(t.L, t.C, hue));
  }
  return out;
}
