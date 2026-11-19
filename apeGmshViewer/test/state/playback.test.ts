import assert from "node:assert/strict";
import { test } from "node:test";
import { clockStep, playMove, type PlayClock } from "../../src/state/playback.ts";

const clock = (speed: PlayClock["speed"], startStep = 0): PlayClock => ({ startStep, startMs: 1000, speed });

test("the clock runs at 10 steps per second times the speed", () => {
  assert.equal(clockStep(clock(1), 2000), 10);
  assert.equal(clockStep(clock(2), 2000), 20);
  assert.equal(clockStep(clock(4), 1500), 20);
  assert.equal(clockStep(clock(1, 7), 1000), 7);
});

test("playback asks for nothing new until the step asked for has arrived", () => {
  assert.deepEqual(playMove(clock(16), 9000, 3, 50, false), { kind: "wait" });
});

test("a slow model skips to where the clock is, so it always moves", () => {
  assert.deepEqual(playMove(clock(1), 2000, 1, 50, true), { kind: "go", step: 10 });
});

test("a fast model waits for the clock", () => {
  assert.deepEqual(playMove(clock(1), 1050, 0, 50, true), { kind: "wait" });
});

test("the last step is always shown, then playback ends", () => {
  assert.deepEqual(playMove(clock(16), 60_000, 40, 50, true), { kind: "go", step: 49 });
  assert.deepEqual(playMove(clock(16), 60_000, 49, 50, false), { kind: "wait" });
  assert.deepEqual(playMove(clock(16), 60_000, 49, 50, true), { kind: "end" });
});
