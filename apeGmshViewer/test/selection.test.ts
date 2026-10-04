// The selection halo's pure parts (R1 follow-up on #1308), against closed
// forms: the pulse envelope, the bounds `F` frames, and the fit distance.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fitDistance, HALO_PX, pulseAt, PULSE_MS, PULSE_SCALE, selectionBounds } from "../src/renderer/selection.ts";

const close = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);

test("halo: a fixed width of a few device pixels, and the pulse swells to PULSE_SCALE and settles", () => {
  assert.equal(HALO_PX, 3);
  assert.equal(pulseAt(0), 1, "at the select, the resting width");
  close(pulseAt(PULSE_MS / 2), PULSE_SCALE, 1e-12, "the middle of the pulse");
  close(pulseAt(PULSE_MS - 1e-9), 1, 1e-6, "the end of the pulse");
  assert.equal(pulseAt(PULSE_MS), 1);
  assert.equal(pulseAt(PULSE_MS * 10), 1);
  assert.equal(pulseAt(Infinity), 1, "no pulse running");
  assert.equal(pulseAt(NaN), 1);
  for (let t = 0; t < PULSE_MS; t += 25) assert.ok(pulseAt(t) >= 1 && pulseAt(t) <= PULSE_SCALE, `t=${t}`);
});

test("selection bounds: the box of the selected primitives, half its diagonal as the radius", () => {
  // Two segments and one triangle; only the triangle and the first segment are selected.
  const lp = new Float32Array([0, 0, 0, 1, 0, 0, 100, 100, 100, 101, 101, 101]);
  const tp = new Float32Array([0, 0, 0, 2, 0, 0, 0, 3, 0]);
  const b = selectionBounds(lp, tp, [0], [0], 1e-3)!;
  assert.deepEqual(b.center, [1, 1.5, 0]);
  close(b.radius, Math.hypot(2, 3, 0) / 2, 1e-12, "radius");
  // Only the far segment: a different box.
  assert.deepEqual(selectionBounds(lp, tp, [1], [], 1e-3)!.center, [100.5, 100.5, 100.5]);
  // Nothing drawn for the selection: null. A degenerate selection keeps the floor radius.
  assert.equal(selectionBounds(lp, tp, [], [], 1e-3), null);
  assert.equal(selectionBounds(new Float32Array([5, 5, 5, 5, 5, 5]), tp, [0], [], 0.25)!.radius, 0.25);
});

test("fit distance: the sphere touches the narrower half-angle of the view, with 8 % of margin", () => {
  const fov = 35;
  const half = (fov / 2) * (Math.PI / 180);
  // A wide free area: the vertical half-angle is the narrower one.
  close(fitDistance(10, fov, 1600, 900), (10 / Math.sin(half)) * 1.08, 1e-9, "wide");
  // A tall, narrow free area: the horizontal half-angle limits.
  const halfH = Math.atan(Math.tan(half) * (300 / 900));
  close(fitDistance(10, fov, 300, 900), (10 / Math.sin(halfH)) * 1.08, 1e-9, "narrow");
  // Framing a selection ten times smaller brings the camera ten times closer.
  close(fitDistance(1, fov, 1600, 900) * 10, fitDistance(10, fov, 1600, 900), 1e-9, "linear in the radius");
});
