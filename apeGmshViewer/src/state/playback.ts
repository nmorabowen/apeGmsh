// Playback's clock, without a window: which step to ask for next.
//
// Playback follows the time since play was pressed, and asks for a new step
// only once the step it asked for last has arrived. A model that cannot keep
// up therefore skips steps and still moves. Asking on a fixed timer would
// replace every request before it finished (the newest request wins), and on
// a slow model nothing would ever be drawn.

/** Steps per second at speed x1. */
export const PLAY_STEPS_PER_SECOND = 10;

/** The speeds the step bar offers. */
export const PLAY_SPEEDS = [1, 2, 4, 8, 16] as const;
export type PlaySpeed = (typeof PLAY_SPEEDS)[number];

export interface PlayClock {
  /** the step and the time (ms) the clock counts from */
  startStep: number;
  startMs: number;
  speed: PlaySpeed;
}

/** The step the clock is at, past the end of the stage if it ran out. */
export function clockStep(c: PlayClock, nowMs: number): number {
  return c.startStep + Math.floor((Math.max(0, nowMs - c.startMs) * PLAY_STEPS_PER_SECOND * c.speed) / 1000);
}

export type PlayMove = { kind: "wait" } | { kind: "go"; step: number } | { kind: "end" };

/**
 * What playback does now. `settled`: the step shown has arrived, or failed
 * (a failed step is passed over, and says so in the readout).
 */
export function playMove(c: PlayClock, nowMs: number, step: number, steps: number, settled: boolean): PlayMove {
  if (!settled) return { kind: "wait" };
  if (step >= steps - 1) return { kind: "end" };
  const target = Math.min(steps - 1, clockStep(c, nowMs));
  return target > step ? { kind: "go", step: target } : { kind: "wait" };
}
