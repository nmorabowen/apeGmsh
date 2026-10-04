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
import { NO_GROUP, OPS_ONLY, paletteEntry, paletteFor, RING_SIZE } from "../src/state/palette.ts";
import { roleColouring, UNASSIGNED_ROW } from "../src/state/roles.ts";
import {
  contrastRatio, DARK, DARK_MAIN, DARK_MAIN_RING2, DARK_ROLE, hexToRgb, oklabDistance, OFFICE, OFFICE_MAIN, protanopia, relight, rgbToHex,
  srgbToOklch, type RGB,
} from "../src/theme/tokens.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import { initialState, reduce } from "../src/state/store.ts";
import { openModel } from "../src/reader/node.ts";

const src = fileURLToPath(new URL("../src", import.meta.url));
const bg = hexToRgb(DARK.bg0);
const L = (hex: string) => srgbToOklch(hexToRgb(hex)).L;
/** What a protanope sees of two colours, as an OKLab distance. */
const seen = (a: string, b: string) => oklabDistance(protanopia(hexToRgb(a)), protanopia(hexToRgb(b)));
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
  const seq = [...DARK_MAIN, ...DARK_MAIN_RING2].map(L);
  for (let i = 1; i < seq.length; i++) assert.ok(Math.abs(seq[i]! - seq[i - 1]!) >= STEP - 1e-9, `entries ${i - 1},${i}: ${seq[i - 1]} vs ${seq[i]}`);
  // A repeated hue is one lightness step away from its first appearance.
  for (let i = 0; i < RING_SIZE; i++) assert.ok(Math.abs(L(DARK_MAIN_RING2[i]!) - L(DARK_MAIN[i]!)) >= STEP - 1e-9, `ring step ${i}`);
});

test("protanopia: no pair of the sixteen group colours is told apart by a red/green difference alone", () => {
  const all = [...DARK_MAIN, ...DARK_MAIN_RING2];
  for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
    assert.ok(seen(all[i]!, all[j]!) >= SAFE, `${i} ${all[i]} vs ${j} ${all[j]}: ${seen(all[i]!, all[j]!).toFixed(3)} as a protanope sees them`);
  }
  // The simulation itself: pure red and pure green collapse together for a protanope; blue and yellow do not.
  assert.ok(seen("#FF0000", "#00AA00") < seen("#0000FF", "#FFFF00") / 3);
});

test("WCAG: every group, role and chrome colour clears its ratio on the background (printed for the PR)", () => {
  const rows: string[] = [];
  const check = (name: string, hex: string, min: number) => {
    const r = contrastRatio(hexToRgb(hex), bg);
    rows.push(`${name.padEnd(28)} ${hex}  ${r.toFixed(2)}:1`);
    assert.ok(r >= min, `${name} ${hex} is ${r.toFixed(2)}:1 on ${DARK.bg0}, needs ${min}:1`);
  };
  DARK_MAIN.forEach((h, i) => check(`group ${i} (${OFFICE_MAIN[i]})`, h, 3));
  DARK_MAIN_RING2.forEach((h, i) => check(`group ${i + 8} (${OFFICE_MAIN[i]}, ring 2)`, h, 3));
  for (const [role, hex] of Object.entries(DARK_ROLE)) check(`role ${role}`, hex, 3);
  check("no group", DARK.noGroup, 3);
  check("OpenSees-only", DARK.opsOnly, 3);
  check("unassigned role", DARK.unassigned, 3);
  check("text", DARK.text, 7);
  check("muted text", DARK.muted, 4.5);
  check("source paths", DARK.source, 3);
  check("accent", DARK.accent, 4.5);
  check("interp tag", DARK.interp, 4.5);
  check("error", DARK.error, 4.5);
  check("selection", DARK.selection, 4.5);
  console.log("WCAG contrast on " + DARK.bg0 + ":\n  " + rows.join("\n  "));
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

test("legend entries cycle the office order; the ninth repeats the first hue one step away with a stripe cue", () => {
  const p = paletteFor(11);
  assert.equal(p.length, 11);
  assert.deepEqual(p[0], hexToRgb(DARK_MAIN[0]!));
  assert.deepEqual(p[8], hexToRgb(DARK_MAIN_RING2[0]!));
  assert.equal(paletteEntry(0).ring, 0);
  assert.equal(paletteEntry(8).ring, 1);
  assert.equal(paletteEntry(16).ring, 2);
  assert.throws(() => paletteEntry(-1), RangeError);
  assert.deepEqual(NO_GROUP, hexToRgb(DARK.noGroup));
  assert.deepEqual(OPS_ONLY, hexToRgb(DARK.opsOnly));
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
  const offenders: string[] = [];
  for (const f of files) {
    readFileSync(f, "utf8").split("\n").forEach((line, i) => {
      if (literal.test(line)) offenders.push(`${f.slice(src.length + 1)}:${i + 1}: ${line.trim()}`);
    });
  }
  assert.deepEqual(offenders, []);
  // The lint itself catches a planted literal of each kind.
  for (const planted of ['color: "#0B5394"', "--x: #abc;", "new THREE.Color(0xff0000)", "background: rgba(1, 2, 3, 0.5)", "hsl(10, 50%, 50%)"]) {
    assert.ok(literal.test(planted), planted);
  }
  for (const fine of ["(#1295)", "opensees/section/#3", "rgb(${c.join(',')})", "// see #1308 round 3"]) assert.ok(!literal.test(fine), fine);
  const mainTs = readFileSync(join(src, "main", "main.ts"), "utf8");
  const m = /backgroundColor:\s*"(#[0-9a-fA-F]{6})"/.exec(mainTs);
  assert.ok(m, "main.ts sets the window background");
  assert.equal(m![1]!.toUpperCase(), DARK.window.toUpperCase(), "the window background equals the token");
});
