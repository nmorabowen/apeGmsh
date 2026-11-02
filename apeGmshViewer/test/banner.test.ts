// The banner's summary line: the open or closed arrow is drawn by style.css
// (`.banner-summary::before`), so an app-chrome still can draw it open. A text
// arrow here would draw two.

import assert from "node:assert/strict";
import { test } from "node:test";
import { mountBanner } from "../src/panels/banner.ts";
import { Store } from "../src/state/store.ts";

class FakeEl {
  className = "";
  title = "";
  type = "";
  hidden = false;
  textContent = "";
  dataset: Record<string, string | undefined> = {};
  style: Record<string, string> = {};
  children: FakeEl[] = [];
  listeners: (() => void)[] = [];
  replaceChildren(...xs: FakeEl[]) {
    this.children = xs;
  }
  addEventListener(_: string, fn: () => void) {
    this.listeners.push(fn);
  }
  removeEventListener(_: string, fn: () => void) {
    this.listeners = this.listeners.filter((f) => f !== fn);
  }
}

test("the banner's summary carries no arrow of its own, closed or open", () => {
  const real = new Store().get();
  const s = { ...real, artifacts: { ...real.artifacts, model: { path: "m.h5", status: "ready", error: null, stale: false, sizeBytes: 0, readMs: 0, zones: {}, warnings: ["first", "second"] } } } as typeof real;
  const fake = { get: () => s, subscribe: () => () => {}, dispatch: () => {} };
  const banner = new FakeEl();
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = { getElementById: () => banner, createElement: () => new FakeEl() };
  try {
    const unmount = mountBanner(fake as never);
    const summary = () => banner.children[0]!;
    assert.equal(banner.hidden, false, "two warnings show the banner");
    assert.equal(summary().className, "banner-summary");
    assert.match(summary().textContent, /^2 warnings/, "closed: the count, with no arrow in front");
    for (const fn of summary().listeners) fn();
    assert.equal(banner.dataset["open"], "true");
    assert.match(summary().textContent, /^2 warnings/, "open: the count, with no arrow in front");
    assert.doesNotMatch(summary().textContent, /[▸▾]/);
    unmount();
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});
