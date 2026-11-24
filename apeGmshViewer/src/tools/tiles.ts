// The tiled tool window's layout and tile set (ADR 0137 D12, H4), pure, so the tests drive it.
//
// - The live tiles split the window as a balanced binary tree that alternates side by side and
//   one above the other (the tool-host spike's layout). A splitter between two subtrees moves its
//   ratio; the tree is rebuilt, ratios even, when the live tiles change.
// - D12: at most `LIVE_TILES` tiles are live (laid out, updated each frame). A tile added beyond
//   them waits paused, as a tab; bringing it in pauses the live tile shown longest ago. A paused
//   tile gets no update, and draws the newest context when it is live again.

/** D12: eight visible live tiles per tool window is the tested limit. */
export const LIVE_TILES = 8;

export type SplitNode = { leaf: number } | { dir: "h" | "v"; ratio: number; a: SplitNode; b: SplitNode };

/** The balanced tree over tiles `ids`, its first split side by side ("h"), alternating below. */
export function splitTree(ids: readonly number[], dir: "h" | "v" = "h"): SplitNode | null {
  if (ids.length === 0) return null;
  if (ids.length === 1) return { leaf: ids[0]! };
  const mid = Math.ceil(ids.length / 2);
  const next = dir === "h" ? "v" : "h";
  return { dir, ratio: mid / ids.length, a: splitTree(ids.slice(0, mid), next)!, b: splitTree(ids.slice(mid), next)! };
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A splitter: where it is drawn, its direction, and the split node it moves. */
export interface Bar {
  node: Extract<SplitNode, { dir: "h" | "v" }>;
  rect: Rect;
  /** the rectangle the node splits, which a drag measures against */
  span: Rect;
}

/** Each tile's rectangle and each splitter's, for `tree` laid out in `r` with `gap` pixels between tiles. */
export function layoutTree(tree: SplitNode | null, r: Rect, gap = 4): { tiles: Map<number, Rect>; bars: Bar[] } {
  const tiles = new Map<number, Rect>();
  const bars: Bar[] = [];
  const go = (n: SplitNode, q: Rect): void => {
    if ("leaf" in n) {
      tiles.set(n.leaf, q);
      return;
    }
    const side = n.dir === "h";
    const cut = Math.round((side ? q.w : q.h) * n.ratio);
    const half = gap / 2;
    bars.push({ node: n, span: q, rect: side ? { x: q.x + cut - half, y: q.y, w: gap, h: q.h } : { x: q.x, y: q.y + cut - half, w: q.w, h: gap } });
    go(n.a, side ? { ...q, w: cut - half } : { ...q, h: cut - half });
    go(n.b, side ? { ...q, x: q.x + cut + half, w: q.w - cut - half } : { ...q, y: q.y + cut + half, h: q.h - cut - half });
  };
  if (tree) go(tree, r);
  return { tiles, bars };
}

/** A splitter dragged to (`px`, `py`): its node's new ratio, kept between 10% and 90% of its span. */
export function dragRatio(bar: Bar, px: number, py: number): number {
  const s = bar.span;
  const t = bar.node.dir === "h" ? (px - s.x) / s.w : (py - s.y) / s.h;
  return Math.min(0.9, Math.max(0.1, t));
}

export interface TileEntry {
  id: number;
  /** the registered tool it mounts */
  tool: string;
}

/** The tiles of one window: which are live (laid out, updated) and which wait paused, as tabs. */
export class TileSet {
  private next = 1;
  private readonly all: TileEntry[] = [];
  /** live tile ids, in layout order */
  private liveIds: number[] = [];
  /** when each live tile was last brought in or added (a counter), to pause the one shown longest ago */
  private readonly shownAt = new Map<number, number>();
  private clock = 0;

  /** Add a tile of `tool`: live when there is room, else paused. Returns its id. */
  add(tool: string): number {
    const id = this.next++;
    this.all.push({ id, tool });
    if (this.liveIds.length < LIVE_TILES) this.makeLive(id);
    return id;
  }

  /** Close tile `id`; a paused tile, if any, takes its place. */
  remove(id: number): void {
    const at = this.all.findIndex((t) => t.id === id);
    if (at < 0) return;
    this.all.splice(at, 1);
    if (!this.liveIds.includes(id)) return;
    this.liveIds = this.liveIds.filter((x) => x !== id);
    this.shownAt.delete(id);
    const waiting = this.paused()[0];
    if (waiting) this.makeLive(waiting.id);
  }

  /** Bring paused tile `id` in; when the window is full, the live tile shown longest ago is paused. Returns the tile paused, or null. */
  bringIn(id: number): number | null {
    if (!this.all.some((t) => t.id === id) || this.liveIds.includes(id)) return null;
    let paused: number | null = null;
    if (this.liveIds.length >= LIVE_TILES) {
      paused = [...this.liveIds].sort((a, b) => this.shownAt.get(a)! - this.shownAt.get(b)!)[0]!;
      this.liveIds = this.liveIds.map((x) => (x === paused ? id : x));
      this.shownAt.delete(paused);
      this.shownAt.set(id, ++this.clock);
    } else this.makeLive(id);
    return paused;
  }

  live(): TileEntry[] {
    return this.liveIds.map((id) => this.all.find((t) => t.id === id)!);
  }

  paused(): TileEntry[] {
    return this.all.filter((t) => !this.liveIds.includes(t.id));
  }

  isLive(id: number): boolean {
    return this.liveIds.includes(id);
  }

  private makeLive(id: number): void {
    this.liveIds.push(id);
    this.shownAt.set(id, ++this.clock);
  }
}
