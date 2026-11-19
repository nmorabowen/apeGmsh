// #2838: the sources panel builds its rows a page at a time, every row stays
// reachable in capture order, and a redraw keeps the pages already shown.
//
// Oracle: a /provenance zone of literal rows, on fixtures/zones.h5's model.

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { mountSources, SOURCES_PAGE } from "../../src/panels/sources.ts";
import { openModel } from "../../src/reader/node.ts";
import type { ProvenanceZone } from "../../src/reader/provenance.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { Store } from "../../src/state/store.ts";

class FakeEl {
  children: FakeEl[] = [];
  parent: FakeEl | null = null;
  className = "";
  title = "";
  hidden = false;
  disabled = false;
  textContent = "";
  scrollTop = 0;
  clientHeight = 100;
  scrollHeight = 10000;
  private listeners = new Map<string, Set<() => void>>();
  append(...xs: (FakeEl | string)[]): void {
    for (const x of xs) {
      if (typeof x === "string") continue;
      x.parent = this;
      this.children.push(x);
    }
  }
  replaceChildren(...xs: FakeEl[]): void {
    this.children = [];
    this.append(...xs);
  }
  before(...xs: FakeEl[]): void {
    const siblings = this.parent!.children;
    for (const x of xs) x.parent = this.parent;
    siblings.splice(siblings.indexOf(this), 0, ...xs);
  }
  remove(): void {
    this.parent!.children = this.parent!.children.filter((c) => c !== this);
  }
  addEventListener(t: string, fn: () => void): void {
    if (!this.listeners.has(t)) this.listeners.set(t, new Set());
    this.listeners.get(t)!.add(fn);
  }
  removeEventListener(t: string, fn: () => void): void {
    this.listeners.get(t)?.delete(fn);
  }
  fire(t: string): void {
    for (const f of [...(this.listeners.get(t) ?? [])]) f();
  }
}

const panel = new FakeEl();
(globalThis as { document?: unknown }).document = {
  createElement: () => new FakeEl(),
  getElementById: (id: string) => (id === "sources" ? panel : null),
};

const zoneOf = (n: number): ProvenanceZone => ({
  version: "1.1.0",
  baseDir: "C:/runs",
  files: { path: ["model.py"], sha256: ["a".repeat(64)], kind: ["script"] },
  sites: { file: Int32Array.of(0), line: Int32Array.of(7), function: ["<module>"] },
  records: {
    path: Array.from({ length: n }, (_, i) => `opensees/section/s${i}`),
    site: new Int32Array(n),
    script: new Int32Array(n),
    seq: Int32Array.from({ length: n }, (_, i) => i),
    origin: Array.from({ length: n }, () => "user" as const),
  },
  originColumn: true,
  present: [true],
  warnings: [],
});

const model = await openModel(join(dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures", "zones.h5"));
const loadOf = (n: number) => ({ type: "fileLoaded", artifact: "model", load: loadModel(model, new BlobStore(), zoneOf(n)) }) as const;
const rows = () => panel.children.filter((c) => c.className.startsWith("source-row"));
const more = () => panel.children.find((c) => c.className === "sources-more");
const keyOf = (row: FakeEl) => row.children[0]!.children.map((c) => c.textContent).join("");

test("the sources rows arrive a page at a time, in order, and all can be reached", () => {
  const store = new Store();
  store.dispatch(loadOf(2 * SOURCES_PAGE + 5));
  const unmount = mountSources(store);
  try {
    assert.equal(rows().length, SOURCES_PAGE);
    assert.match(more()!.textContent, new RegExp(`show ${SOURCES_PAGE} more \\(${SOURCES_PAGE + 5} left\\)`));

    panel.scrollTop = 0;
    panel.fire("scroll");
    assert.equal(rows().length, SOURCES_PAGE, "scrolling far from the end adds nothing");
    panel.scrollTop = panel.scrollHeight - panel.clientHeight - 10;
    panel.fire("scroll");
    assert.equal(rows().length, 2 * SOURCES_PAGE, "scrolling near the end adds a page");

    more()!.fire("click");
    assert.equal(rows().length, 2 * SOURCES_PAGE + 5);
    assert.equal(more(), undefined, "nothing left: no button");
    panel.fire("scroll");
    assert.deepEqual(rows().map(keyOf), zoneOf(2 * SOURCES_PAGE + 5).records.path);
  } finally {
    unmount();
  }
});

test("a redraw keeps the pages shown, and a new listing starts again", () => {
  const store = new Store();
  store.dispatch(loadOf(3 * SOURCES_PAGE));
  const unmount = mountSources(store);
  try {
    more()!.fire("click");
    assert.equal(rows().length, 2 * SOURCES_PAGE);
    const last = rows().at(-1)!;
    last.children.find((c) => c.className === "source")!.fire("click");
    assert.notEqual(rows().at(-1), last, "the request redrew the panel");
    assert.equal(rows().length, 2 * SOURCES_PAGE);

    store.dispatch(loadOf(SOURCES_PAGE + 1));
    assert.equal(rows().length, SOURCES_PAGE);
  } finally {
    unmount();
  }
});

test("a listing that fits one page has no button", () => {
  const store = new Store();
  store.dispatch(loadOf(3));
  const unmount = mountSources(store);
  try {
    assert.equal(rows().length, 3);
    assert.equal(more(), undefined);
  } finally {
    unmount();
  }
});
