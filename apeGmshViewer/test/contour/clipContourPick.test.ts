// #3079: the capture's clip stills (renderer/harness/sections.ts `captureClip`) colour the deformed model by a
// nodal contour they pick themselves. They used to take whatever pick the sequence before left: captureElements
// leaves its element field, which an unverified file refuses (the yoo_slab_N56_75 run), and a refused contour
// never shows a step, so `clip-11-deformed-contour-step1` waited 15 s and failed with "the step did not show".

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { pickNodalContour } from "../../src/renderer/harness/sections.ts";
import { contourFieldsOf, contourView } from "../../src/state/selectors.ts";
import { opened, until } from "./harness.ts";

const fx = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));

test("the clip stills pick a nodal contour of their own, so an element field refused earlier does not hold them", async () => {
  const h = await opened(fx("elem_mixed_unstamped.h5"), fx("elem_mixed_unstamped.results.h5"));
  try {
    // What captureElements leaves on an unverified file: colour by contour, its first element field, refused.
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
    h.store.dispatch({ type: "setColourBy", by: "contour" });
    h.store.dispatch({ type: "setContourField", field: "stress", component: "stress_xx", source: "element" });
    await until(() => contourView(h.store.get()).state === "refused", "the inherited element pick to be refused");

    // The clip sequence's contour setup.
    assert.equal(pickNodalContour(h.store), true);
    const nf = contourFieldsOf(h.store.get())!.fields[0]!;
    await until(() => {
      const v = contourView(h.store.get());
      return v.state === "ready" && v.stepMeasured;
    }, "the nodal contour to measure the step on screen");
    const v = contourView(h.store.get());
    assert.ok(v.state === "ready");
    assert.equal(v.field, nf.name);
    assert.equal(v.component.key, nf.components[0]!.key);
  } finally {
    h.effects.dispose();
  }
});

test("captureClip makes that pick right after colouring by contour", () => {
  const src = readFileSync(fileURLToPath(new URL("../../src/renderer/harness/sections.ts", import.meta.url)), "utf8");
  const body = src.slice(src.indexOf("export async function captureClip"), src.indexOf('await still("clip-11-deformed-contour-step1")'));
  assert.match(body, /setColourBy", by: "contour" \}\);\s*pickNodalContour\(store\);/);
});
