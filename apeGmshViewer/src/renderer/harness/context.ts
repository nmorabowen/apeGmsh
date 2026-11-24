// The measure and capture harness (VH1, #2865): what it reads from the renderer's composition root
// (renderer/app.ts builds these five and hands them in). Nothing under harness/ imports app.ts.

import type { Bridge, Effects } from "../../effects.ts";
import type { BlobStore } from "../../state/blobs.ts";
import type { Store } from "../../state/store.ts";
import type { Viewport } from "../viewport.ts";

/** The five the harness drives (and, in a capture, the baseline): the page's store, BlobStore, effects, viewport and main's bridge. */
export interface HarnessContext {
  readonly store: Store;
  readonly blobs: BlobStore;
  readonly effects: Effects;
  readonly viewport: Viewport;
  readonly bridge: Bridge;
  /** VU9b: the combination authoring panel (panels/combine.ts), for its stills; absent before it is mounted */
  readonly combine?: { open(): void; receive(answer: import("../../state/types.ts").CombinationSaveIn): void };
  /** The capture's starting view, which `resetView` (./reset.ts) returns to before each step; runCapture records it. */
  readonly baseline?: ViewBaseline;
}

/** What a capture step starts from: the model main opened, the camera it opened on, and the capture window's size. */
export interface ViewBaseline {
  readonly model: string;
  readonly camera: import("../../state/types.ts").CameraPose;
  readonly width: number;
  readonly height: number;
}

/** What main's `config()` answers. */
export type HarnessConfig = Awaited<ReturnType<Bridge["config"]>>;

export const nextFrame = () => new Promise<number>((r) => requestAnimationFrame(r));
/** Resolves once the frame after the current DOM change has been painted. */
export const painted = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));

/** What a wait was for: a name (the step and what it waited on), or a function that writes the whole message at the timeout. */
export type WaitWhat = string | (() => string);

/** The error a wait throws when its time runs out: `what` names the step, so the run stops there instead of capturing a wrong still. */
export class WaitTimeout extends Error {
  constructor(what: WaitWhat, ms: number) {
    super(typeof what === "function" ? what() : `${what}: not met after ${ms} ms`);
    this.name = "WaitTimeout";
  }
}

/** Poll (every 20 ms, at most `ms`) until `ok` holds; false on timeout. Only for a caller that handles the timeout itself. */
export async function poll(ok: () => boolean, ms = 15000): Promise<boolean> {
  for (let t = 0; t < ms; t += 20) {
    if (ok()) return true;
    await new Promise((r) => setTimeout(r, 20));
  }
  return ok();
}

/**
 * Poll the store (every 20 ms, at most `ms`) until `ok` holds; throws a `WaitTimeout` naming `what` on timeout. A
 * capture step that timed out and shot anyway handed a wrong still to D7 review (the 2026-11-24 review, finding 3).
 */
export async function until(what: WaitWhat, ok: () => boolean, ms = 15000): Promise<void> {
  if (!(await poll(ok, ms))) throw new WaitTimeout(what, ms);
}

/**
 * Poll (every 20 ms) until `ok` holds, for as long as `progress` keeps changing: throws a `WaitTimeout` naming `what`
 * once it has not changed for `stallMs`, or after `maxMs` in all. For work whose length grows with the model (a
 * history scan reads every step), where a fixed total wait passes on a small fixture and fails on a real building that
 * is still making progress.
 */
export async function untilSteady(what: WaitWhat, ok: () => boolean, progress: () => unknown, stallMs = 30000, maxMs = 600000): Promise<void> {
  let last = progress(), since = 0;
  for (let t = 0; t < maxMs && since < stallMs; t += 20, since += 20) {
    if (ok()) return;
    await new Promise((r) => setTimeout(r, 20));
    const now = progress();
    if (now !== last) {
      last = now;
      since = 0;
    }
  }
  if (!ok()) throw new WaitTimeout(what, since >= stallMs ? stallMs : maxMs);
}
