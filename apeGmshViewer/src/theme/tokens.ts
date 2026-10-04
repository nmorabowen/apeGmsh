// The one theme-token module: every colour and font the app shows comes
// from here (test/tokens.test.ts greps the rest of src/ for colour literals).
//
// Source: APE Ingeniería house style, apeGraphStyle `__init__.py` v0.1.0
// (ape-workflow/ape-libraries/apeGraphStyle, 2026-10-03): `main_colors`,
// `colors` and `band_shades` are copied verbatim below, never imported
// across repositories. Its rules carried over: the palette never relies on
// a red/green difference alone (protanopia-safe), neighbouring entries
// differ in lightness as well as hue, and past the eight colours a second
// cue is added (`redundant_cycle` adds a line style; here a lightness step
// and a striped legend chip).
//
// The app's default theme is dark (ADR 0112 D7 stills). The office colours
// are made for white paper, so `DARK` carries dark-adapted variants: the
// same hue order, the same protanopia rule, each relit in OKLCH to a
// lightness found by search (DARK_L1, DARK_L2 below) so that every group
// colour clears WCAG 3:1 against the viewport background and the rules
// under DARK_L1 hold (test/tokens.test.ts reports the ratios). A light
// theme with the palette as-is comes later.

export type Hex = `#${string}`;
export type RGB = readonly [number, number, number];

/** apeGraphStyle `main_colors`, in order: consecutive entries alternate warm/cool and dark/light. */
export const OFFICE_MAIN: readonly Hex[] = [
  "#0B5394", // deep blue
  "#E69F00", // orange (Okabe-Ito)
  "#37474F", // slate
  "#56B4E9", // sky blue
  "#B08900", // dark gold
  "#7B5EA7", // purple
  "#8C8C8C", // grey
  "#004D5C", // teal, dark
];

/** apeGraphStyle `colors`: the things drawn again and again. */
export const OFFICE = {
  primary: "#0B5394",
  accent: "#E69F00",
  ink: "#1F2933",
  muted: "#6B7280",
  grid: "#E4E7EB",
  concrete: "#CFD8DC",
  concrete_edge: "#546E7A",
  beam: "#90A4AE",
  wall: "#37474F",
  column: "#0B5394",
  void: "#FFFFFF",
  demand: "#0B5394",
  capacity: "#37474F",
  warn: "#E69F00",
  ok: "#56B4E9",
} as const satisfies Record<string, Hex>;

/** apeGraphStyle `band_shades`, light to dark, all neutral. */
export const OFFICE_BANDS: readonly Hex[] = ["#ECEFF1", "#CFD8DC", "#B0BEC5", "#90A4AE", "#78909C"];

/** Structural roles the colour-by-role mode knows, with the office colour of each. */
export const ROLES = ["column", "wall", "beam", "concrete", "void"] as const;
export type Role = (typeof ROLES)[number];

export const FONT_FAMILY = "Archivo Narrow";
/** The bundled files (SIL OFL 1.1, `fonts/OFL.txt`), relative to the page. */
export const FONT_FILES: readonly { file: string; weight: number; style: "normal" | "italic" }[] = [
  { file: "fonts/ArchivoNarrow-Regular.ttf", weight: 400, style: "normal" },
  { file: "fonts/ArchivoNarrow-Italic.ttf", weight: 400, style: "italic" },
  { file: "fonts/ArchivoNarrow-Medium.ttf", weight: 500, style: "normal" },
  { file: "fonts/ArchivoNarrow-SemiBold.ttf", weight: 600, style: "normal" },
  { file: "fonts/ArchivoNarrow-Bold.ttf", weight: 700, style: "normal" },
];

// ---- colour maths ------------------------------------------------------------

export function hexToRgb(hex: string): [number, number, number] {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) throw new RangeError(`not a #rrggbb colour: ${hex}`);
  const v = parseInt(m[1]!, 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

export function rgbToHex(rgb: RGB): Hex {
  const c = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, "0");
  return `#${c(rgb[0])}${c(rgb[1])}${c(rgb[2])}`;
}

/** sRGB (0..1) as a three.js hex number. */
export const toHexNumber = (hex: string): number => parseInt(hex.slice(1), 16);

const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toGamma = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** WCAG 2 relative luminance of an sRGB colour. */
export function luminance(rgb: RGB): number {
  const [r, g, b] = rgb.map(toLinear) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG 2 contrast ratio between two sRGB colours (1 to 21). */
export function contrastRatio(a: RGB, b: RGB): number {
  const la = luminance(a), lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/** sRGB (0..1) to OKLCH {L, C, h in degrees}. */
export function srgbToOklch(rgb: RGB): { L: number; C: number; h: number } {
  const [r, g, b] = rgb.map(toLinear) as [number, number, number];
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  const L = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
  const A = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  const B = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  const h = ((Math.atan2(B, A) * 180) / Math.PI + 360) % 360;
  return { L, C: Math.hypot(A, B), h };
}

function oklchToLinear(L: number, C: number, h: number): [number, number, number] {
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

/** OKLCH to sRGB (0..1), reducing chroma until the colour is inside the gamut. */
export function oklchToSrgb(L: number, C: number, h: number): [number, number, number] {
  for (let c = C; c >= 0; c -= 0.005) {
    const rgb = oklchToLinear(L, c, h);
    if (rgb.every((v) => v >= -1e-6 && v <= 1 + 1e-6)) return rgb.map((v) => Math.round(toGamma(Math.min(1, Math.max(0, v))) * 1000) / 1000) as [number, number, number];
  }
  return oklchToLinear(L, 0, h).map((v) => toGamma(Math.min(1, Math.max(0, v)))) as [number, number, number];
}

/** The same hue and chroma at another OKLCH lightness. */
export function relight(hex: string, L: number): Hex {
  const { C, h } = srgbToOklch(hexToRgb(hex));
  return rgbToHex(oklchToSrgb(L, C, h));
}

export type Cvd = "protanopia" | "deuteranopia" | "tritanopia";

/** Machado, Oliveira and Fernandes 2009, severity 1.0, in linear sRGB. */
const CVD_MATRIX: Readonly<Record<Cvd, readonly (readonly [number, number, number])[]>> = {
  protanopia: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  deuteranopia: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
  tritanopia: [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]],
};

/** What a person with `kind` colour-vision deficiency sees of a colour (sRGB in, sRGB out). */
export function simulateCvd(kind: Cvd, rgb: RGB): [number, number, number] {
  const [r, g, b] = rgb.map(toLinear) as [number, number, number];
  return CVD_MATRIX[kind].map((row) => toGamma(Math.min(1, Math.max(0, row[0] * r + row[1] * g + row[2] * b)))) as [number, number, number];
}

/** Protanopia simulation: what a protanope sees of a colour. */
export const protanopia = (rgb: RGB): [number, number, number] => simulateCvd("protanopia", rgb);

/** Perceptual distance in OKLab (0 = identical; about 0.02 is just noticeable). */
export function oklabDistance(a: RGB, b: RGB): number {
  const p = srgbToOklch(a), q = srgbToOklch(b);
  const pa = p.C * Math.cos((p.h * Math.PI) / 180), pb = p.C * Math.sin((p.h * Math.PI) / 180);
  const qa = q.C * Math.cos((q.h * Math.PI) / 180), qb = q.C * Math.sin((q.h * Math.PI) / 180);
  return Math.hypot(p.L - q.L, pa - qa, pb - qb);
}

// ---- the dark theme ----------------------------------------------------------

/**
 * OKLCH lightness of each dark-adapted group colour, per office index,
 * found by a constrained search (levels 0.50 to 0.92 in steps of 0.02)
 * against the rules test/tokens.test.ts holds: every level clears WCAG
 * 3:1 on the background; legend neighbours step by at least 0.10; a
 * repeated hue (the second ring, slots 8 to 15, taken when no free office
 * colour contrasts with a group's neighbours; state/palette.ts assigns the
 * slots from the view's adjacency) is at least 0.10 from its first
 * appearance; every pair of the sixteen is at least 0.05 OKLab apart
 * as a protanope and as a deuteranope sees it (both lose red/green, and
 * blue, sky, teal, slate and purple then fall into one family, so
 * lightness is what tells them apart); and within the office eight a pair
 * closer than 0.08 owes its separation to lightness (>= 0.10 L), not to a
 * chroma crumb. Tritanopia is reported, not enforced: blue/yellow is the
 * office palette's main axis.
 */
export const DARK_L1: readonly number[] = [0.64, 0.82, 0.52, 0.82, 0.62, 0.92, 0.74, 0.62];
export const DARK_L2: readonly number[] = [0.52, 0.7, 0.86, 0.58, 0.88, 0.76, 0.58, 0.78];

/** The office `main_colors` adapted to the dark background: same hue order, alternating lightness. */
export const DARK_MAIN: readonly Hex[] = OFFICE_MAIN.map((hex, i) => relight(hex, DARK_L1[i]!));
/** The second ring (slots 8 to 15): the same hues one lightness step away (found by search against the rules above; the legend adds a striped chip to a group that takes one). */
export const DARK_MAIN_RING2: readonly Hex[] = OFFICE_MAIN.map((hex, i) => relight(hex, DARK_L2[i]!));

/** The role colours adapted to the dark background. */
export const DARK_ROLE: Readonly<Record<Role, Hex>> = {
  column: relight(OFFICE.column, 0.7),
  wall: relight(OFFICE.wall, 0.62),
  beam: relight(OFFICE.beam, 0.8),
  concrete: relight(OFFICE.concrete, 0.74),
  void: "#F4F6F8",
};

export const DARK = {
  /** viewport and page background (the gradient runs bg1 to bg0) */
  bg0: "#121418",
  bg1: "#1B1E24",
  /** the main window's background before the page paints (main.ts) */
  window: "#16181D",
  panel: "rgba(30, 33, 40, 0.92)",
  line: "rgba(255, 255, 255, 0.08)",
  faint: "rgba(255, 255, 255, 0.03)",
  hover: "rgba(255, 255, 255, 0.08)",
  dashed: "rgba(255, 255, 255, 0.12)",
  dropBorder: "rgba(255, 255, 255, 0.18)",
  /** text tokens clear WCAG 4.5:1 on bg0 and on bg1 (the gradient's light end) */
  text: "#E6E8EC",
  muted: "#8B919C",
  source: "#8A9099",
  /** the office accent: selection, pins, role chips */
  accent: OFFICE.accent,
  accentInk: "#1B1E24",
  accentWash: "rgba(230, 159, 0, 0.16)",
  /** interpreted-value tag: the office warn colour */
  interp: OFFICE.warn,
  interpWash: "rgba(70, 40, 20, 0.92)",
  interpText: "#FFD7A8",
  error: "#FF6B6B",
  errorWash: "rgba(255, 107, 107, 0.1)",
  errorLine: "rgba(255, 107, 107, 0.4)",
  /** elements in no physical group, and OpenSees-only elements */
  noGroup: "#9EA3AD",
  opsOnly: "#EDEDED",
  /** colour-by-role: elements whose file carries no structural role (the office concrete, muted) */
  unassigned: relight(OFFICE.concrete, 0.6),
  /** the outline edges of drawn faces, in colour-by-group and in colour-by-role */
  edge: "#0B0C0F",
  edgeRole: relight(OFFICE.concrete_edge, 0.42),
  /** the geometry layer (V2f): curves, faint surfaces, points */
  geometryCurve: "#D8DDE6",
  geometrySurface: "#7D8796",
  geometryPoint: "#FFFFFF",
  /** the three.js lights */
  skyLight: "#DFE6F0",
  groundLight: "#30343C",
  keyLight: "#FFFFFF",
  /**
   * selection core and halo (round 3 item 3, on #1318's selection.ts): the
   * core is the office accent, the halo stays white at its fixed pixel width.
   */
  selection: OFFICE.accent,
  halo: "#FFFFFF",
} as const;

/** The CSS custom properties the stylesheet reads; `applyTheme` sets them on the document. */
export function cssVariables(): Record<string, string> {
  return {
    "--bg-0": DARK.bg0,
    "--bg-1": DARK.bg1,
    "--panel": DARK.panel,
    "--line": DARK.line,
    "--faint": DARK.faint,
    "--hover": DARK.hover,
    "--dashed": DARK.dashed,
    "--drop-border": DARK.dropBorder,
    "--text": DARK.text,
    "--muted": DARK.muted,
    "--source": DARK.source,
    "--accent": DARK.accent,
    "--accent-ink": DARK.accentInk,
    "--accent-wash": DARK.accentWash,
    "--interp": DARK.interp,
    "--interp-wash": DARK.interpWash,
    "--interp-text": DARK.interpText,
    "--error": DARK.error,
    "--error-wash": DARK.errorWash,
    "--error-line": DARK.errorLine,
    "--font": `"${FONT_FAMILY}", "Arial Narrow", "Segoe UI", system-ui, sans-serif`,
    "--mono": '"Cascadia Mono", "Consolas", ui-monospace, monospace',
  };
}

/** Set the theme on a document: the custom properties and the bundled font faces. */
export function applyTheme(doc: Document): void {
  for (const [k, v] of Object.entries(cssVariables())) doc.documentElement.style.setProperty(k, v);
  if (doc.getElementById("theme-fonts")) return;
  const style = doc.createElement("style");
  style.id = "theme-fonts";
  style.textContent = FONT_FILES.map(
    (f) => `@font-face { font-family: "${FONT_FAMILY}"; src: url("${f.file}") format("truetype"); font-weight: ${f.weight}; font-style: ${f.style}; font-display: swap; }`,
  ).join("\n");
  doc.head.append(style);
}
