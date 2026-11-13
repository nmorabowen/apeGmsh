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
 * The `top:` of the first style.css rule whose selector list names `selector`,
 * evaluated with the banner's published variables (panels/banner.ts) in px:
 * its `max(...)`/`calc(...)` of px terms, as test/state/user.test.ts's cssTop.
 */
function topUnderBanner(selector: string, bannerTop: number, bannerH: number): number {
  const css = readFileSync(new URL("../src/renderer/style.css", import.meta.url), "utf8");
  const rule = new RegExp(`(?:^|[},])\\s*(?:[^{}]*,\\s*)?${selector.replace(/[[\]()~:]/g, "\\$&")}\\s*(?:,[^{}]*)?\\{([^}]*)\\}`, "m").exec(css);
  assert.ok(rule, `style.css has no rule for ${selector}`);
  const body = rule[1]!.replace(/\/\*[\s\S]*?\*\//g, "");
  const m = /(?:^|[\s;])top:\s*([^;]+);/.exec(body);
  assert.ok(m, `${selector} has no top`);
  const expr = m[1]!.trim().replace(/var\(--banner-top, 58px\)/g, `${bannerTop}px`).replace(/var\(--banner-h, 0px\)/g, `${bannerH}px`);
  assert.match(expr, /^[\d\s+.()a-z,-]+$/, `an expression this test can evaluate: ${expr}`);
  const js = expr.replace(/(\d+(?:\.\d+)?)px/g, "$1").replace(/max\(/g, "Math.max(").replace(/calc\(/g, "(");
  return Function(`return ${js};`)() as number;
}

test("#2924 item 7: the colour scales sit 8 px below a shown banner however tall it is drawn, as the left rail does", () => {
  const scale = "#banner:not([hidden]) ~ #scale-stack";
  const rail = "#banner:not([hidden]) ~ #left-rail";
  // A one-line banner (about 30 px at 58 px, or at 106 px under a one-line scrubber bar): 168 px, as before.
  assert.equal(topUnderBanner(scale, 58, 30), 168);
  assert.equal(topUnderBanner(scale, 106, 30), 168);
  // An app-chrome still draws the whole list (#2921 still 2): the scale clears the banner's bottom by 8 px.
  for (const [top, h] of [[58, 240], [106, 240], [58, 600]] as const) {
    assert.equal(topUnderBanner(scale, top, h), top + h + 8, `banner at ${top} px, ${h} px tall`);
    assert.equal(topUnderBanner(scale, top, h), topUnderBanner(rail, top, h), "the scale and the rail clear the banner alike");
  }
});
