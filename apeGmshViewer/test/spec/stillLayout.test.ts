// VF3 (#2959): where an app-chrome still (ADR 0128 D9) puts the colour scales.
// The still draws the banner's whole list, and the plane line (VF2, #2952) made
// it taller, so the scales under it reached about 60 px into the mesh of
// extrap_plane's 2-D still. A fit frames the mesh about the viewport's centre,
// leaving the bottom margin as wide as the top one, and no banner grows into
// the bottom: there the scales stay clear of the mesh. Measured on the D7
// stills (1600 x 1000, top camera, von Mises at step 3): the banner spans
// 106-282 px, the contour scale card is 132 px tall, the mesh spans rows
// 361-638, and the Windows menu's top is at about 952 px.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const CSS = readFileSync(new URL("../../src/renderer/style.css", import.meta.url), "utf8");

/** A property's text in the first style.css rule whose selector list names `selector`; null when that rule does not set it. */
function cssValue(selector: string, prop: string): string | null {
  const rule = new RegExp(`(?:^|[},])\\s*(?:[^{}]*,\\s*)?${selector.replace(/[[\]()~:"=]/g, "\\$&")}\\s*(?:,[^{}]*)?\\{([^}]*)\\}`, "m").exec(CSS);
  assert.ok(rule, `style.css has no rule for ${selector}`);
  const body = rule[1]!.replace(/\/\*[\s\S]*?\*\//g, "");
  const m = new RegExp(`(?:^|[\\s;])${prop}:\\s*([^;]+);`).exec(body);
  return m ? m[1]!.trim() : null;
}

/** A px length, or a failure naming the text. */
function px(v: string | null, what: string): number {
  const m = v === null ? null : /^(\d+(?:\.\d+)?)px$/.exec(v);
  assert.ok(m, `${what} is a px length: ${v}`);
  return Number(m[1]);
}

/** The banner rule's `top` with the banner's published variables: the window's place for the scales. */
function topUnderBanner(bannerTop: number, bannerH: number, vh: number): number {
  const expr = cssValue("#banner:not([hidden]) ~ #scale-stack", "top")!
    .replace(/var\(--banner-top, 58px\)/g, `${bannerTop}px`)
    .replace(/var\(--banner-h, 0px\)/g, `${bannerH}px`)
    .replace(/(\d+(?:\.\d+)?)vh/g, (_, n: string) => `${(Number(n) * vh) / 100}px`);
  assert.match(expr, /^[\d\s+.()a-z,-]+$/, `an expression this test can evaluate: ${expr}`);
  const js = expr.replace(/(\d+(?:\.\d+)?)px/g, "$1").replace(/max\(/g, "Math.max(").replace(/min\(/g, "Math.min(").replace(/calc\(/g, "(");
  return Function(`return ${js};`)() as number;
}

const STILL = 'body[data-still-chrome="app"] #scale-stack';
const D7 = { vh: 1000, bannerTop: 106, bannerH: 176, scaleH: 132, meshTop: 361, meshBottom: 638, menuTop: 952 };

test("VF3: in an app-chrome still the scales sit at the foot of their box, whatever the banner's height", () => {
  assert.equal(cssValue(STILL, "justify-content"), "flex-end", "packed at the box's foot");
  const foot = px(cssValue(STILL, "bottom"), "the still's bottom");
  assert.equal(foot, 64);
  // Nothing more specific moves the foot back: the banner rule sets the box's top only.
  assert.equal(cssValue("#banner:not([hidden]) ~ #scale-stack", "bottom"), null);
  assert.equal(cssValue("#banner:not([hidden]) ~ #scale-stack", "justify-content"), null);
  // The D7 still: the scale card spans 804-936 px, below the mesh (rows 361-638) and above the Windows menu (about 952 px).
  const top = D7.vh - foot - D7.scaleH;
  assert.ok(top > D7.meshBottom, `the scales (from ${top} px) clear the mesh's bottom row ${D7.meshBottom}`);
  assert.ok(D7.vh - foot < D7.menuTop, `the scales end (${D7.vh - foot} px) above the Windows menu (${D7.menuTop} px)`);
  // Under the banner (the window's rule) they started at 290 px and covered the mesh's top 61 px: the overlap VF2 left (#1283).
  const under = topUnderBanner(D7.bannerTop, D7.bannerH, D7.vh);
  assert.equal(under, 290);
  assert.equal(under + D7.scaleH - D7.meshTop, 61);
});

test("VF3: the foot clears the Windows menu, which sits 16 px off the bottom and is about 32 px tall", () => {
  const menuBottom = px(cssValue("#windows-menu", "bottom"), "the menu's bottom");
  assert.equal(menuBottom, 16);
  assert.ok(px(cssValue(STILL, "bottom"), "the still's bottom") >= menuBottom + 32 + 8, "8 px over the menu, as the stack's gap");
  // The open model windows end at the same height (#windows), so the scales line up with them.
  assert.equal(cssValue("#windows", "bottom"), cssValue(STILL, "bottom"));
});

test("VF3: the window keeps its layout: the scales under the banner, the box 16 px off the bottom, packed at its top", () => {
  assert.equal(px(cssValue("#scale-stack", "bottom"), "the window's bottom"), 16);
  assert.equal(cssValue("#scale-stack", "justify-content"), null);
  // Every rule that sets the stack's foot or packing names the app-chrome still, save the window's own `#scale-stack` box.
  const rules = [...CSS.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/([^{}]*)\{([^}]*)\}/g)]
    .filter((m) => m[1]!.includes("#scale-stack") && /(?:^|[\s;])(?:bottom|justify-content):/.test(m[2]!))
    .map((m) => m[1]!.trim());
  assert.deepEqual(rules, ["#scale-stack", STILL]);
});
