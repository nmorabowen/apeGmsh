// The contour scale's stored preference, the banner's closed line, and the
// page structure the left rail and the scale rely on.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { bannerSummary } from "../src/panels/banner.ts";
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
  assert.ok(at("rail") < at("viewport-scale"), "a drag across the right rail must stay on top of it");
  const css = readFileSync(new URL("../src/renderer/style.css", import.meta.url), "utf8");
  assert.match(css, /#banner:not\(\[hidden\]\) ~ #viewport-scale:not\(\[data-moved\]\) \{ top: 168px; \}/);
  assert.match(css, /body\[data-still-chrome="model"\] #viewport-scale \{ display: none; \}/);
});
