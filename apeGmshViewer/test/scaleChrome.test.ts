// The contour scale's stored preference, the banner's closed line, and the
// page structure the left rail and the scale rely on.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { bannerSummary } from "../src/panels/banner.ts";
import { signRampCss } from "../src/state/lineDiagram.ts";
import { parseSavedScale } from "../src/ui/scaleChrome.ts";

test("parseSavedScale: reads what the app writes", () => {
  assert.deepEqual(parseSavedScale('{"orient":"vertical","at":{"left":10,"top":20}}'), { orient: "vertical", at: { left: 10, top: 20 } });
  assert.deepEqual(parseSavedScale('{"orient":"horizontal","at":null}'), { orient: "horizontal", at: null });
});

test("parseSavedScale: anything else is ignored, so the default place holds", () => {
  for (const raw of [null, "", "not json", "null", "3", "{}", '{"orient":"diagonal","at":null}', '{"orient":"vertical"}', '{"orient":"vertical","at":{"left":"1","top":2}}', '{"orient":"vertical","at":{"left":1}}', '{"orient":"vertical","at":{"left":null,"top":2}}']) {
    assert.equal(parseSavedScale(raw), null, String(raw));
  }
});

test("signRampCss: the colours change at zero, not at the bar's midpoint", () => {
  assert.match(signRampCss("dark"), / 50%, /);
  assert.match(signRampCss("dark", { min: -300, max: 100 }), / 75%, /);
  assert.match(signRampCss("dark", { min: 2, max: 4 }), / 0%, #\w+ 0%, /, "all positive");
  assert.match(signRampCss("dark", { min: -4, max: -2 }), / 100%, /, "all negative");
  assert.match(signRampCss("dark", { min: 3, max: 3 }), / 50%, /, "constant");
});

test("bannerSummary: one message is shown as itself, several as counts of each kind", () => {
  assert.equal(bannerSummary(0, 0, 1, "the one warning"), "the one warning");
  assert.equal(bannerSummary(1, 0, 0, "the one refusal"), "the one refusal");
  assert.equal(bannerSummary(0, 0, 9, "first"), "9 warnings");
  assert.equal(bannerSummary(2, 1, 9, "first"), "2 refused · 1 notice · 9 warnings");
});

test("the scale is a sibling after the banner and the rails, so it paints over them and clears a banner", () => {
  const html = readFileSync(new URL("../src/renderer/index.html", import.meta.url), "utf8");
  const at = (id: string) => html.indexOf(`id="${id}"`);
  assert.ok(at("banner") > 0 && at("banner") < at("viewport-scale"));
  assert.ok(at("rail") < at("scale-stack"), "a drag across the right rail must stay on top of it");
  assert.ok(at("scale-stack") < at("viewport-scale") && at("viewport-scale") < at("line-diagram-scale"), "the two scales stack, contour first");
  const css = readFileSync(new URL("../src/renderer/style.css", import.meta.url), "utf8");
  assert.match(css, /body\[data-still-chrome="model"\] #scale-stack \{ display: none; \}/);
});

/**
 * A px property (`top:`, `bottom:`, `max-height:`) of the first style.css rule
 * whose selector list names `selector`, evaluated with the banner's published
 * variables (panels/banner.ts) and a viewport `vh` px tall: its
 * `min(...)`/`max(...)`/`calc(...)` of px and vh terms, as
 * test/state/user.test.ts's cssTop. Null when the rule does not set it.
 */
function cssPx(selector: string, prop: string, at: { bannerTop?: number; bannerH?: number; vh: number }): number | null {
  const css = readFileSync(new URL("../src/renderer/style.css", import.meta.url), "utf8");
  const rule = new RegExp(`(?:^|[},])\\s*(?:[^{}]*,\\s*)?${selector.replace(/[[\]()~:]/g, "\\$&")}\\s*(?:,[^{}]*)?\\{([^}]*)\\}`, "m").exec(css);
  assert.ok(rule, `style.css has no rule for ${selector}`);
  const body = rule[1]!.replace(/\/\*[\s\S]*?\*\//g, "");
  const m = new RegExp(`(?:^|[\\s;])${prop}:\\s*([^;]+);`).exec(body);
  if (!m) return null;
  let expr = m[1]!.trim();
  if (at.bannerTop !== undefined) expr = expr.replace(/var\(--banner-top, 58px\)/g, `${at.bannerTop}px`);
  if (at.bannerH !== undefined) expr = expr.replace(/var\(--banner-h, 0px\)/g, `${at.bannerH}px`);
  expr = expr.replace(/(\d+(?:\.\d+)?)vh/g, (_, n: string) => `${(Number(n) * at.vh) / 100}px`);
  assert.match(expr, /^[\d\s+.()a-z,-]+$/, `an expression this test can evaluate: ${expr}`);
  assert.doesNotMatch(expr, /var\(/, `a variable this test does not set: ${expr}`);
  const js = expr.replace(/(\d+(?:\.\d+)?)px/g, "$1").replace(/max\(/g, "Math.max(").replace(/min\(/g, "Math.min(").replace(/calc\(/g, "(");
  return Function(`return ${js};`)() as number;
}

const SCALE = "#banner:not([hidden]) ~ #scale-stack";
const RAIL = "#banner:not([hidden]) ~ #left-rail";

test("#2924 item 7: the colour scales sit 8 px below a shown banner however tall it is drawn, as the left rail does", () => {
  const top = (bannerTop: number, bannerH: number, vh = 1000) => cssPx(SCALE, "top", { bannerTop, bannerH, vh });
  // A one-line banner (about 30 px at 58 px, or at 106 px under a one-line scrubber bar): 168 px, as before.
  assert.equal(top(58, 30), 168);
  assert.equal(top(106, 30), 168);
  // An app-chrome still draws the whole list (#2921 still 2: 2 notices, 136 px): the scale clears the banner's bottom by 8 px.
  for (const [t, h] of [[58, 136], [58, 240], [106, 240], [106, 380]] as const) {
    assert.equal(top(t, h), t + h + 8, `banner at ${t} px, ${h} px tall`);
    assert.equal(top(t, h), cssPx(RAIL, "top", { bannerTop: t, bannerH: h, vh: 1000 }), "the scale and the rail clear the banner alike");
  }
});

test("#2944 review: a banner at its height cap never pushes the colour scales off the bottom", () => {
  // The stack is bounded below as the left rail is.
  assert.equal(cssPx("#scale-stack", "bottom", { vh: 1000 }), 16);
  assert.equal(cssPx("#left-rail", "bottom", { vh: 1000 }), 16);
  // The banner's own cap, as style.css draws it in a still or open.
  const capped = 'body[data-still-chrome="app"] #banner';
  for (const vh of [600, 800, 1000, 1400]) {
    const cap = cssPx(capped, "max-height", { vh });
    assert.equal(cap, vh - 180, `the banner's cap at ${vh} px`);
    for (const bannerTop of [58, 106]) {
      const t = cssPx(SCALE, "top", { bannerTop, bannerH: cap!, vh })!;
      // It starts by half the height (never above 168 px), painting over the banner's tail, and leaves the
      // horizontal contour scale (131 px drawn in the D7 still) room above the bottom bound.
      assert.equal(t, Math.max(168, vh / 2), `${vh} px viewport, banner at ${bannerTop} px`);
      assert.ok(t + 131 <= vh - 16, `${vh} px: the scale's ${t} + 131 px fits above the bottom bound`);
    }
  }
});
