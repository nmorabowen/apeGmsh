// The node markers' size: its bounds, and the size chosen when they turn on.
// Pure, so the rule is tested without a window.

export const NODE_SIZE_MIN = 1;
export const NODE_SIZE_MAX = 24;

/** Segments measured for the typical element size; enough for a stable median at any mesh size. */
const SAMPLE = 5000;
/** A marker this share of the typical on-screen node spacing reads as a dot on a line, not a blanket. */
const SPACING_SHARE = 0.2;
/** Below 3 px a circle reads as noise; above 10 px it hides the element it sits on. */
const SMART_MIN = 3;
const SMART_MAX = 10;

/**
 * The marker size in pixels for a model framed in the viewport: a share of the
 * median element edge as it appears at that frame, within readable bounds.
 * `segments` is 6 floats per segment. It assumes the fit view rather than the
 * current zoom, so the same model gets the same size every time.
 */
export function smartNodeSize(segments: Float32Array, radius: number, viewportPx: number): number {
  const count = Math.floor(segments.length / 6);
  if (count === 0 || !(radius > 0) || !(viewportPx > 0)) return SMART_MAX;
  const step = Math.max(1, Math.floor(count / SAMPLE));
  const lengths: number[] = [];
  for (let i = 0; i < count; i += step) {
    const o = i * 6;
    const length = Math.hypot(segments[o + 3]! - segments[o]!, segments[o + 4]! - segments[o + 1]!, segments[o + 5]! - segments[o + 2]!);
    if (length > 0) lengths.push(length);
  }
  if (!lengths.length) return SMART_MAX;
  lengths.sort((a, b) => a - b);
  const median = lengths[lengths.length >> 1]!;
  const spacingPx = (median / (2 * radius)) * viewportPx;
  return Math.max(SMART_MIN, Math.min(SMART_MAX, Math.round(spacingPx * SPACING_SHARE)));
}
