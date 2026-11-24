// ADR 0137 D12 (H4): the tiled tool window's layout and tile set.
//
// - The split tree is balanced and alternates side by side, then one above the other.
// - The laid-out tiles fill the window without overlap, with a splitter between each split's halves.
// - A splitter drag moves its ratio, held between 10% and 90%.
// - At most LIVE_TILES tiles are live; the ninth waits paused; bringing it in pauses the live tile
//   shown longest ago; closing a live tile lets the first paused one in.

import assert from "node:assert/strict";
import { test } from "node:test";
import { dragRatio, layoutTree, LIVE_TILES, splitTree, TileSet, type Rect } from "../../src/tools/tiles.ts";

test("the tree is balanced and alternates side by side, then one above the other", () => {
  assert.deepEqual(splitTree([]), null);
  assert.deepEqual(splitTree([7]), { leaf: 7 });
  const four = splitTree([1, 2, 3, 4])!;
  assert.ok("dir" in four && four.dir === "h");
  assert.ok("dir" in four.a && four.a.dir === "v" && "dir" in four.b && four.b.dir === "v");
  const three = splitTree([1, 2, 3])!;
  assert.ok("dir" in three && "dir" in three.a && "leaf" in three.b, "two on the left, one on the right");
  assert.equal(three.ratio, 2 / 3, "the split follows the count");
});

const overlap = (a: Rect, b: Rect) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

test("eight tiles fill the window without overlap, with a splitter per split", () => {
  const ids = Array.from({ length: LIVE_TILES }, (_, i) => i + 1);
  const W = 1200, H = 800, gap = 4;
  const { tiles, bars } = layoutTree(splitTree(ids), { x: 0, y: 0, w: W, h: H }, gap);
  assert.equal(tiles.size, 8);
  assert.equal(bars.length, 7, "n tiles, n - 1 splits");
  const rects = [...tiles.values()];
  for (let i = 0; i < rects.length; i++) for (let j = i + 1; j < rects.length; j++) assert.ok(!overlap(rects[i]!, rects[j]!), `tiles ${i} and ${j} overlap`);
  for (const r of rects) assert.ok(r.x >= 0 && r.y >= 0 && r.x + r.w <= W && r.y + r.h <= H && r.w > 0 && r.h > 0);
  const area = rects.reduce((s, r) => s + r.w * r.h, 0) + bars.reduce((s, b) => s + b.rect.w * b.rect.h, 0);
  assert.ok(Math.abs(area - W * H) < W * H * 0.02, "tiles and splitters cover the window");
});

test("a splitter drag moves its ratio, between 10% and 90% of its span", () => {
  const { bars } = layoutTree(splitTree([1, 2]), { x: 100, y: 0, w: 1000, h: 500 });
  const bar = bars[0]!;
  assert.equal(dragRatio(bar, 400, 0), 0.3);
  assert.equal(dragRatio(bar, 100, 0), 0.1);
  assert.equal(dragRatio(bar, 1100, 0), 0.9);
  bar.node.ratio = dragRatio(bar, 850, 0);
  const after = layoutTree({ ...bar.node }, { x: 100, y: 0, w: 1000, h: 500 });
  assert.ok(after.tiles.get(1)!.w > after.tiles.get(2)!.w * 2.5, "the left tile grew");
});

test("at most eight tiles are live; the ninth waits paused", () => {
  const set = new TileSet();
  const ids = Array.from({ length: 9 }, (_, i) => set.add(i % 2 ? "history" : "step"));
  assert.equal(set.live().length, LIVE_TILES);
  assert.deepEqual(set.paused().map((t) => t.id), [ids[8]]);
  assert.equal(set.isLive(ids[8]!), false);
});

test("bringing a paused tile in pauses the live tile shown longest ago", () => {
  const set = new TileSet();
  const ids = Array.from({ length: 10 }, () => set.add("step"));
  // Live: 1..8 (added in order); paused: 9, 10.
  assert.equal(set.bringIn(ids[8]!), ids[0], "tile 1 was shown longest ago");
  assert.equal(set.isLive(ids[8]!), true);
  assert.equal(set.isLive(ids[0]!), false);
  assert.equal(set.bringIn(ids[9]!), ids[1], "then tile 2");
  assert.equal(set.bringIn(ids[0]!), ids[2], "tile 1 comes back in place of tile 3");
  assert.equal(set.bringIn(ids[0]!), null, "a live tile is not brought in again");
  assert.equal(set.live().length, LIVE_TILES);
});

test("closing a live tile lets the first paused tile in; closing a paused one changes nothing live", () => {
  const set = new TileSet();
  const ids = Array.from({ length: 10 }, () => set.add("step"));
  set.remove(ids[9]!);
  assert.equal(set.live().length, LIVE_TILES);
  assert.deepEqual(set.paused().map((t) => t.id), [ids[8]]);
  set.remove(ids[3]!);
  assert.equal(set.isLive(ids[8]!), true, "the paused tile came in");
  assert.deepEqual(set.paused(), []);
  set.remove(ids[0]!);
  assert.equal(set.live().length, 7);
});
