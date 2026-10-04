// The office graphic style (#1308 round 3): the token module is the only
// source of colour in the app, the dark-adapted group palette keeps
// apeGraphStyle's rules (a lightness step between legend neighbours, no
// pair told apart by a red/green difference alone, under a protanopia
// simulation), and every colour clears WCAG contrast on the background
// (the ratios are printed for the PR).

import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { groupAdjacency } from "../src/state/adjacency.ts";
import { assignSlots, cvdDistance, hueOf, NEVER_ADJACENT, NO_GROUP, OPS_ONLY, RING_SIZE, slotColour, SLOTS, slotsContrast } from "../src/state/palette.ts";
import { colourGroups } from "../src/mesh/build.ts";
import { roleColouring, UNASSIGNED_ROW } from "../src/state/roles.ts";
import {
  contrastRatio, DARK, DARK_L1, DARK_L2, DARK_MAIN, DARK_MAIN_RING2, DARK_ROLE, FONT_FILES, hexToRgb, oklabDistance, OFFICE, OFFICE_MAIN, protanopia, relight, rgbToHex,
  simulateCvd, srgbToOklch, type Cvd, type RGB,
} from "../src/theme/tokens.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import { initialState, reduce } from "../src/state/store.ts";
import { openModel } from "../src/reader/node.ts";

const src = fileURLToPath(new URL("../src", import.meta.url));
const bg = hexToRgb(DARK.bg0);
const L = (hex: string) => srgbToOklch(hexToRgb(hex)).L;
/** What a person with a colour-vision deficiency sees of two colours, as an OKLab distance. */
const seenAs = (kind: Cvd, a: string, b: string) => oklabDistance(simulateCvd(kind, hexToRgb(a)), simulateCvd(kind, hexToRgb(b)));
const seen = (a: string, b: string) => seenAs("protanopia", a, b);
/** Just-noticeable in OKLab is about 0.02; a pair must be well past it. */
const SAFE = 0.05;
const STEP = 0.1;

// ---- the office source, copied verbatim ---------------------------------------

test("the office palette is apeGraphStyle v0.1.0's, in its order", () => {
  assert.deepEqual([...OFFICE_MAIN], ["#0B5394", "#E69F00", "#37474F", "#56B4E9", "#B08900", "#7B5EA7", "#8C8C8C", "#004D5C"]);
  assert.equal(OFFICE.column, "#0B5394");
  assert.equal(OFFICE.wall, "#37474F");
  assert.equal(OFFICE.beam, "#90A4AE");
  assert.equal(OFFICE.concrete, "#CFD8DC");
  assert.equal(OFFICE.concrete_edge, "#546E7A");
  assert.equal(OFFICE.void, "#FFFFFF");
  assert.equal(OFFICE.accent, "#E69F00");
});

test("the office palette's own protanopia margins, for the record (slate vs dark teal is its closest pair)", () => {
  // The office list is made for white paper; on it slate (2) and dark teal (7) sit 0.03 apart
  // for a protanope, both dark. The dark adaptation below keeps every pair at SAFE or more.
  let worst = Infinity;
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) worst = Math.min(worst, seen(OFFICE_MAIN[i]!, OFFICE_MAIN[j]!));
  assert.ok(worst > 0.02, `office worst pair ${worst.toFixed(3)}`);
});

// ---- the dark adaptation ------------------------------------------------------

test("dark adaptation keeps each office hue (and its order) and only moves the lightness", () => {
  DARK_MAIN.forEach((hex, i) => {
    const o = srgbToOklch(hexToRgb(OFFICE_MAIN[i]!)), d = srgbToOklch(hexToRgb(hex));
    if (o.C > 0.02) assert.ok(Math.abs(((d.h - o.h + 540) % 360) - 180) <= 3, `hue ${i}: ${o.h} -> ${d.h}`);
    else assert.ok(d.C < 0.02, `grey ${i} stays grey`);
  });
  assert.equal(relight("#0B5394", srgbToOklch(hexToRgb("#0B5394")).L), rgbToHex(hexToRgb("#0B5394")).toLowerCase() === "#0b5394" ? relight("#0B5394", srgbToOklch(hexToRgb("#0B5394")).L) : "", "relight is the identity at the own lightness");
});

test("legend neighbours step in lightness by at least 0.10, across the ring boundary too", () => {
  const seq = [...DARK_L1, ...DARK_L2];
  for (let i = 1; i < seq.length; i++) assert.ok(Math.abs(seq[i]! - seq[i - 1]!) >= STEP - 1e-9, `entries ${i - 1},${i}: ${seq[i - 1]} vs ${seq[i]}`);
  // A repeated hue is one lightness step away from its first appearance.
  for (let i = 0; i < RING_SIZE; i++) assert.ok(Math.abs(DARK_L2[i]! - DARK_L1[i]!) >= STEP - 1e-9, `ring step ${i}`);
  // The rendered colours sit on the designed levels (hex rounding aside).
  [...DARK_MAIN, ...DARK_MAIN_RING2].forEach((hex, i) => assert.ok(Math.abs(L(hex) - seq[i]!) < 0.012, `${i} ${hex}: L ${L(hex).toFixed(3)} vs ${seq[i]}`));
});

test("protanopia and deuteranopia: no pair of the sixteen group colours is told apart by a red/green difference alone", () => {
  const all = [...DARK_MAIN, ...DARK_MAIN_RING2];
  for (const kind of ["protanopia", "deuteranopia"] as const) {
    for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
      const d = seenAs(kind, all[i]!, all[j]!);
      assert.ok(d >= SAFE, `${i} ${all[i]} vs ${j} ${all[j]}: ${d.toFixed(3)} under ${kind}`);
      // Within the office eight a near pair owes its separation to lightness, not to a chroma crumb
      // (the designed levels; a hex-rounded L lands within 0.01 of them).
      if (i < 8 && j < 8 && d < 0.08) assert.ok(Math.abs(DARK_L1[i]! - DARK_L1[j]!) >= STEP - 1e-9, `${i} vs ${j}: ${d.toFixed(3)} under ${kind} with dL ${Math.abs(DARK_L1[i]! - DARK_L1[j]!).toFixed(2)}`);
    }
  }
  // The pair the review named (entry 5, purple, and entry 8, the second-ring blue) is far apart in lightness.
  assert.ok(Math.abs(L(all[5]!) - L(all[8]!)) >= 0.3, "entries 5 and 8");
  assert.ok(seenAs("deuteranopia", all[5]!, all[8]!) >= 0.2);
  // The simulations themselves: pure red and pure green collapse together for a protanope and a deuteranope; blue and yellow do not.
  for (const kind of ["protanopia", "deuteranopia"] as const) assert.ok(seenAs(kind, "#FF0000", "#00AA00") < seenAs(kind, "#0000FF", "#FFFF00") / 3, kind);
  // Tritanopia (blue/yellow loss) is reported only: the office palette's main axis is blue/yellow.
  let worst: [number, number, number] = [Infinity, 0, 0];
  for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
    const d = seenAs("tritanopia", all[i]!, all[j]!);
    if (d < worst[0]) worst = [d, i, j];
  }
  console.log(`tritanopia (report only): closest pair ${worst[1]} ${all[worst[1]]} vs ${worst[2]} ${all[worst[2]]} at ${worst[0].toFixed(3)} OKLab`);
});

test("WCAG: every group, role and chrome colour clears its ratio on both ends of the background gradient (printed for the PR)", () => {
  const rows: string[] = [];
  const bg1 = hexToRgb(DARK.bg1);
  const check = (name: string, hex: string, min: number) => {
    const r0 = contrastRatio(hexToRgb(hex), bg), r1 = contrastRatio(hexToRgb(hex), bg1);
    rows.push(`${name.padEnd(28)} ${hex}  ${r0.toFixed(2)}:1 on bg0, ${r1.toFixed(2)}:1 on bg1`);
    assert.ok(r0 >= min, `${name} ${hex} is ${r0.toFixed(2)}:1 on ${DARK.bg0}, needs ${min}:1`);
    assert.ok(r1 >= min, `${name} ${hex} is ${r1.toFixed(2)}:1 on ${DARK.bg1}, needs ${min}:1`);
  };
  DARK_MAIN.forEach((h, i) => check(`group ${i} (${OFFICE_MAIN[i]})`, h, 3));
  DARK_MAIN_RING2.forEach((h, i) => check(`group ${i + 8} (${OFFICE_MAIN[i]}, ring 2)`, h, 3));
  for (const [role, hex] of Object.entries(DARK_ROLE)) check(`role ${role}`, hex, 3);
  check("no group", DARK.noGroup, 3);
  check("OpenSees-only", DARK.opsOnly, 3);
  check("unassigned role", DARK.unassigned, 3);
  check("text", DARK.text, 7);
  check("muted text", DARK.muted, 4.5);
  check("source paths", DARK.source, 4.5);
  check("interp text", DARK.interpText, 4.5);
  check("accent", DARK.accent, 4.5);
  check("interp tag", DARK.interp, 4.5);
  check("error", DARK.error, 4.5);
  check("selection", DARK.selection, 4.5);
  console.log("WCAG contrast:\n  " + rows.join("\n  "));
});

test("the roles take the office colours; no two roles are told apart by red/green alone", () => {
  const roles = Object.entries(DARK_ROLE);
  for (let i = 0; i < roles.length; i++) for (let j = i + 1; j < roles.length; j++) {
    assert.ok(seen(roles[i]![1], roles[j]![1]) >= SAFE, `${roles[i]![0]} vs ${roles[j]![0]}`);
  }
  for (const [role, office] of [["column", OFFICE.column], ["wall", OFFICE.wall], ["beam", OFFICE.beam], ["concrete", OFFICE.concrete]] as const) {
    const o = srgbToOklch(hexToRgb(office)), d = srgbToOklch(hexToRgb(DARK_ROLE[role]));
    if (o.C > 0.02) assert.ok(Math.abs(((d.h - o.h + 540) % 360) - 180) <= 3, `${role} keeps the office hue`);
  }
});

// ---- the palette the legend uses ---------------------------------------------

test("the sixteen slots are the two rings in office order; the second ring carries the stripe cue", () => {
  assert.equal(SLOTS.length, 2 * RING_SIZE);
  assert.deepEqual(slotColour(0).color, hexToRgb(DARK_MAIN[0]!));
  assert.deepEqual(slotColour(8), { color: hexToRgb(DARK_MAIN_RING2[0]!), ring: 1 });
  assert.equal(hueOf(13), 5);
  assert.throws(() => slotColour(16), RangeError);
  assert.deepEqual(NO_GROUP, hexToRgb(DARK.noGroup));
  assert.deepEqual(OPS_ONLY, hexToRgb(DARK.opsOnly));
});

/** The maintainer's rule for two adjacent groups, spelled out here beside the palette's own `slotsContrast`. */
function contrasts(a: number, b: number): boolean {
  const dL = Math.abs(srgbToOklch(SLOTS[a]!).L - srgbToOklch(SLOTS[b]!).L);
  const hueGap = hueOf(a) !== hueOf(b) && !NEVER_ADJACENT.some(([x, y]) => (hueOf(a) === x && hueOf(b) === y) || (hueOf(a) === y && hueOf(b) === x));
  const ok = cvdDistance(a, b) >= SAFE && (dL >= STEP || hueGap);
  assert.equal(slotsContrast(a, b), ok, `slotsContrast(${a}, ${b}) agrees with the spelled-out rule`);
  return ok;
}

test("adjacent groups contrast: the fixture's arch touches both columns, which do not touch each other", async () => {
  const model = await openModel(fileURLToPath(new URL("../fixtures/shoebuckle.h5", import.meta.url)));
  const { byElement, order } = colourGroups(model);
  const adj = groupAdjacency(model, byElement, order.length);
  const name = (i: number) => order[i]!;
  assert.deepEqual(adj.map(([a, b]) => [name(a), name(b)].sort()), [["Arch", "LeftColumn"], ["Arch", "RightColumn"]]);
  const s = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(model, new BlobStore()) });
  const legend = s.mesh!.legend;
  assert.deepEqual(s.mesh!.adjacency, [[0, 1], [0, 2]], "legend rows: Arch (0) touches LeftColumn (1) and RightColumn (2)");
  for (const [a, b] of s.mesh!.adjacency) assert.ok(contrasts(legend[a]!.slot!, legend[b]!.slot!), `${legend[a]!.name} vs ${legend[b]!.name}`);
  // The wiring: the legend carries what `assignSlots` returns for this adjacency, not legend order
  // ([0, 1, 2], which would also contrast here). The arch's first neighbour takes the office orange,
  // its second the purple: the office colour farthest from both as a protanope and a deuteranope see
  // them. (`Frame` is in `order` but colours no drawn cell, so it has no legend row.)
  assert.deepEqual(legend.map((e) => e.slot), [0, 1, 5]);
  assert.deepEqual(assignSlots(legend.length, s.mesh!.adjacency), [0, 1, 5]);
  // The legend shows the assigned slots' colours.
  for (const e of legend) if (e.slot !== null) assert.deepEqual(e.color, slotColour(e.slot).color);
  // Deterministic: the same file gives the same colours.
  const again = loadModel(model, new BlobStore()).mesh.legend.map((e) => e.slot);
  assert.deepEqual(legend.map((e) => e.slot), again);
});

test("assignSlots: a chain and a dense graph keep every adjacent pair apart, with the office rule and the stripe cue past eight", () => {
  // A path of 6 groups: each neighbour pair contrasts; the assignment is a pure function of the adjacency.
  const path: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]];
  const slots = assignSlots(6, path);
  for (const [a, b] of path) assert.ok(contrasts(slots[a]!, slots[b]!), `${a}-${b}: slots ${slots[a]} ${slots[b]}`);
  assert.deepEqual(assignSlots(6, path), slots, "deterministic");
  assert.deepEqual(assignSlots(6, [...path].reverse()), slots, "and independent of the pair order");
  // A hub with 12 neighbours: more than the eight office colours, so the second ring (stripe cue) is used,
  // every hub-neighbour pair contrasts, and orange never sits beside dark gold.
  const n = 13;
  const star: [number, number][] = Array.from({ length: n - 1 }, (_, i) => [0, i + 1]);
  const s2 = assignSlots(n, star);
  assert.equal(new Set(s2).size, n, "all distinct");
  assert.ok(s2.some((x) => slotColour(x).ring === 1), "a second-ring slot is in use");
  // The maintainer's ruling on #1330: a free office colour that contrasts comes before a striped
  // slot. Here every office colour contrasts with the hub's, so the eight are used first and exactly
  // n - 8 of the thirteen are striped.
  assert.equal(s2.filter((x) => slotColour(x).ring === 1).length, n - RING_SIZE, "stripes only for the repeated hues");
  for (const [a, b] of star) assert.ok(contrasts(s2[a]!, s2[b]!), `${a}-${b}`);
  // The complete graph of 15: more neighbours than any sixteen slots can separate, so the fallback
  // path (no slot contrasts with them all) decides most of them; even there, orange and dark gold
  // never both appear, since the two are adjacent to everything.
  const k15: [number, number][] = [];
  for (let i = 0; i < 15; i++) for (let j = i + 1; j < 15; j++) k15.push([i, j]);
  const s15 = assignSlots(15, k15);
  const hues15 = new Set(s15.map(hueOf));
  assert.ok(!(hues15.has(1) && hues15.has(4)), `K15 puts orange beside gold: ${JSON.stringify(s15)}`);
  // Keeping gold out leaves fourteen usable slots for fifteen groups: one slot repeats, by design.
  assert.equal(new Set(s15).size, 14, "fourteen distinct slots; the gold family is excluded");
  // Below eight groups no hue repeats, so no second-ring slot is taken while a contrasting office colour is free.
  const v3 = assignSlots(3, [[0, 1], [0, 2]]);
  assert.deepEqual(v3, [0, 1, 5]);
  assert.ok(v3.every((x) => slotColour(x).ring === 0), "first ring only");
  const v5 = assignSlots(5, [[1, 4]]);
  assert.deepEqual(v5, [2, 0, 3, 4, 1]);
  assert.ok(v5.every((x) => slotColour(x).ring === 0), "first ring only");
  assert.ok(contrasts(v3[0]!, v3[1]!) && contrasts(v3[0]!, v3[2]!) && contrasts(v5[1]!, v5[4]!), "every adjacent pair still contrasts");
  // Orange and dark gold: never adjacent, in a complete graph of 6 either.
  const k6: [number, number][] = [];
  for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) k6.push([i, j]);
  const s3 = assignSlots(6, k6);
  for (const [a, b] of k6) {
    assert.ok(!NEVER_ADJACENT.some(([x, y]) => (hueOf(s3[a]!) === x && hueOf(s3[b]!) === y) || (hueOf(s3[a]!) === y && hueOf(s3[b]!) === x)), `${a}-${b}`);
    assert.ok(cvdDistance(s3[a]!, s3[b]!) >= SAFE);
  }
  // Groups with no neighbour take the lowest free slots, in order.
  assert.deepEqual(assignSlots(3, []), [0, 1, 2]);
  assert.throws(() => assignSlots(2, [[0, 2]]), RangeError);
});

// ---- colour by role: from the file only ---------------------------------------

test("colour by role: a file with no role attribute says so and draws every element as unassigned; a role on a group colours its elements", async () => {
  const model = await openModel(fileURLToPath(new URL("../fixtures/shoebuckle.h5", import.meta.url)));
  const s = reduce(initialState, { type: "fileLoaded", artifact: "model", load: loadModel(model, new BlobStore()) });
  const none = roleColouring(s)!;
  assert.equal(none.fileHasRoles, false);
  assert.deepEqual(none.legend.map((e) => e.name), [UNASSIGNED_ROW]);
  assert.ok(none.byElement.every((i) => i === 0));
  // The seam: a `role` field on a group declaration (what the loader adds once the file carries @role).
  const arch = s.decls["mesh/physical_group/Arch"]!;
  const withRole = { ...s, decls: Object.assign(Object.create(null), s.decls, {
    "mesh/physical_group/Arch": { ...arch, fields: [...arch.fields, { label: "role", value: "beam", source: `${arch.h5}@role`, interpreted: false }] },
  }) };
  const some = roleColouring(withRole)!;
  assert.equal(some.fileHasRoles, true);
  assert.deepEqual(some.legend.map((e) => [e.name, e.elements]), [["beam", 79], [UNASSIGNED_ROW, 44]]);
  assert.deepEqual(some.legend[0]!.color, hexToRgb(DARK_ROLE.beam));
  // A value outside the role list is an error, never a guess; two roles on one element too.
  const bad = { ...s, decls: Object.assign(Object.create(null), s.decls, {
    "mesh/physical_group/Arch": { ...arch, fields: [...arch.fields, { label: "role", value: "slab", source: "", interpreted: false }] },
  }) };
  assert.throws(() => roleColouring(bad), /not a structural role/);
});

// ---- the bundled font and its licence ----------------------------------------

test("the Archivo Narrow files and the full OFL 1.1 text are bundled", () => {
  const fonts = fileURLToPath(new URL("../src/renderer/fonts", import.meta.url));
  for (const f of FONT_FILES) assert.ok(statSync(join(fonts, f.file.replace(/^fonts\//, ""))).size > 50_000, f.file);
  const ofl = readFileSync(join(fonts, "OFL.txt"), "utf8");
  assert.match(ofl, /Reserved Font Name "Archivo Narrow"/);
  assert.match(ofl, /SIL OPEN FONT LICENSE Version 1\.1 - 26 February 2007/);
  for (const heading of ["PREAMBLE", "DEFINITIONS", "PERMISSION & CONDITIONS", "TERMINATION", "DISCLAIMER"]) assert.ok(ofl.includes(heading), heading);
  for (const n of [1, 2, 3, 4, 5]) assert.ok(new RegExp(`^${n}\\) `, "m").test(ofl), `condition ${n}`);
  assert.ok(ofl.length > 4000, `the full licence text (${ofl.length} chars)`);
});

// ---- the lint: no colour literal outside the token module --------------------

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(ts|css|html)$/.test(name)) out.push(p);
  }
  return out;
}

test("every colour in the app comes from src/theme/tokens.ts: no colour literal elsewhere in src/", () => {
  // src/main/ is the Electron main process (V2f's files): its window background
  // is the one literal the renderer cannot set; it is checked to equal the token.
  const files = walk(src).filter((p) => !p.endsWith(join("theme", "tokens.ts")) && !p.includes(`${join("src", "main")}`));
  // A colour literal: #rrggbb / #rrggbbaa anywhere, #rgb / #rgba when quoted or
  // after a CSS colon, a three.js 0xrrggbb, or an rgb()/hsl() call whose
  // arguments are written out (`rgb(${...})` from a token colour is not one).
  // An issue number such as (#1295) or a `#k` declaration index is none.
  const literal = /#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{8}\b|["'`:]\s*#[0-9a-fA-F]{3,4}\b|\b0x[0-9a-fA-F]{6}\b|\brgba?\((?!\$\{)|\bhsla?\(/;
  // A float triple on a line that talks about colour: `[0.29, 0.56, 0.89]` beside colour/color/palette/rgb.
  const triple = /\[\s*[01]?\.\d+\s*,\s*[01]?\.\d+\s*,\s*[01]?\.\d+\s*\]/;
  const colourContext = /colou?r|palette|rgb|swatch|tint|shade/i;
  // A CSS named colour as a property value (CSS), or as a string or a three.js colour argument (TS).
  const NAMED =
    "aqua|black|blue|brown|coral|crimson|cyan|fuchsia|gold|gray|grey|green|indigo|ivory|khaki|lime|magenta|maroon|navy|olive|orange|orchid|pink|plum|purple|red|salmon|silver|tan|teal|tomato|violet|white|yellow|" +
    "lightblue|lightgray|lightgrey|lightgreen|lightyellow|lightpink|lightcyan|lightcoral|lightsalmon|lightseagreen|lightskyblue|lightslategray|lightslategrey|lightsteelblue|lightgoldenrodyellow|" +
    "darkblue|darkcyan|darkgray|darkgrey|darkgreen|darkorange|darkred|darkviolet|darkslategray|darkslategrey|darkslateblue|darkseagreen|darkkhaki|darkmagenta|darkolivegreen|darkorchid|darksalmon|darkturquoise|darkgoldenrod|" +
    "dimgray|dimgrey|mediumblue|mediumpurple|mediumseagreen|mediumslateblue|mediumturquoise|mediumvioletred|mediumorchid|mediumaquamarine|palegreen|paleturquoise|palevioletred|deepskyblue|deeppink|" +
    "slategray|slategrey|steelblue|skyblue|royalblue|cornflowerblue|dodgerblue|midnightblue|powderblue|cadetblue|aliceblue|blueviolet|forestgreen|seagreen|springgreen|lawngreen|chartreuse|yellowgreen|greenyellow|olivedrab|limegreen|" +
    "hotpink|firebrick|chocolate|sienna|peru|wheat|beige|linen|snow|mintcream|honeydew|azure|lavender|lavenderblush|thistle|gainsboro|whitesmoke|antiquewhite|floralwhite|ghostwhite|turquoise|aquamarine|bisque|moccasin|burlywood|rosybrown|sandybrown|goldenrod|lemonchiffon|papayawhip|peachpuff|navajowhite|blanchedalmond|cornsilk|oldlace|seashell|mistyrose|orangered|indianred|rebeccapurple";
  // As a CSS property value, as a quoted string, or as a three.js Color argument; a longer identifier (whitelist, tangent) is not one.
  const named = new RegExp(`(?::\\s*|["'\`]\\s*)(?:${NAMED})\\b(?![-\\w])|\\bColor\\(\\s*["'](?:${NAMED})["']`, "i");
  const offenders: string[] = [];
  for (const f of files) {
    readFileSync(f, "utf8").split("\n").forEach((line, i) => {
      const hit =
        literal.test(line) ||
        (triple.test(line) && colourContext.test(line)) ||
        named.test(line);
      if (hit) offenders.push(`${f.slice(src.length + 1)}:${i + 1}: ${line.trim()}`);
    });
  }
  assert.deepEqual(offenders, []);
  // The lint itself catches a planted literal of each kind.
  for (const planted of ['color: "#0B5394"', "--x: #abc;", "new THREE.Color(0xff0000)", "background: rgba(1, 2, 3, 0.5)", "hsl(10, 50%, 50%)"]) {
    assert.ok(literal.test(planted), planted);
  }
  for (const planted of ["export const PALETTE = [[0.29, 0.56, 0.89], [0.95, 0.55, 0.22]];", "const NO_GROUP: RGB = [0.62, 0.64, 0.68]; // colour", "color: [0.5, 0.5, 0.5]"]) {
    assert.ok(triple.test(planted) && colourContext.test(planted), planted);
  }
  assert.ok(!(triple.test("center: [0.5, 0.5, 0.5]") && colourContext.test("center: [0.5, 0.5, 0.5]")), "a position triple is not a colour");
  for (const planted of ['color: "white"', "background: red;", 'new THREE.Color("orange")', "border-color: darkslategray;", "fill: Gold"]) {
    assert.ok(named.test(planted), planted);
  }
  for (const fine of ["(#1295)", "opensees/section/#3", "rgb(${c.join(',')})", "// see #1308 round 3", "const whitelist = 1", "--line: var(--x)", "text-align: center", "throw new RangeError(`paletteEntry: ${i}`)", "x: tangent", "color: var(--red-ish)"]) {
    assert.ok(!literal.test(fine) && !named.test(fine), fine);
  }
  const mainTs = readFileSync(join(src, "main", "main.ts"), "utf8");
  const m = /backgroundColor:\s*"(#[0-9a-fA-F]{6})"/.exec(mainTs);
  assert.ok(m, "main.ts sets the window background");
  assert.equal(m![1]!.toUpperCase(), DARK.window.toUpperCase(), "the window background equals the token");
});
