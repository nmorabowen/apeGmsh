// #2836: the loads summary is parsed once per model and stage, not once per store event.

import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { withNodes } from "../src/effects.ts";
import { openModel } from "../src/reader/node.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { loadModel } from "../src/state/load.ts";
import { LOADS_SUMMARY_INPUTS } from "../src/state/loads.ts";
import { loadsSummaryOf } from "../src/state/selectors.ts";
import { initialState, reduce, type State } from "../src/state/store.ts";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
const model = await openModel(join(fixtures, "definitions.h5"));
const blobs = new BlobStore();
const base: State = reduce(initialState, { type: "fileLoaded", artifact: "model", load: withNodes(loadModel(model, blobs), model, blobs) });

/** A copy of the model's tables, so no memo can answer for it: the summary reads everything it needs. */
const unmemoised = (s: State): State => ({ ...s, definitions: structuredClone(s.definitions), loads: { ...s.loads } });

test("the summary reads only the parts of the state its memos are built on", () => {
  const read = new Set<string>();
  const spy = new Proxy(unmemoised(base), {
    get(t, k, r) {
      if (typeof k === "string") read.add(k);
      return Reflect.get(t, k, r);
    },
  });
  assert.ok(loadsSummaryOf(spy).hasLoads, "the fixture has loads");
  const allowed = new Set<string>(LOADS_SUMMARY_INPUTS);
  assert.deepEqual([...read].filter((k) => !allowed.has(k)), []);
});

test("a state change the summary does not read reuses the parsed patterns", () => {
  const a = loadsSummaryOf(base);
  const b = loadsSummaryOf({ ...base, hover: "/x", visibility: { ...base.visibility } } as State);
  assert.equal(b.patterns, a.patterns);
});

test("a memoised summary equals one computed from nothing, after every input that changes it", () => {
  const changes: ((s: State) => State)[] = [
    (s) => reduce(s, { type: "setLoads", includeSupportReactions: !s.loads.includeSupportReactions }),
    (s) => reduce(s, { type: "setLoads", showNodalForces: !s.loads.showNodalForces }),
    (s) => reduce(s, { type: "setLoads", scaleBySeries: !s.loads.scaleBySeries }),
    (s) => reduce(s, { type: "setLoads", selectedPatterns: [] }),
    (s) => ({ ...s, ndm: { ...s.ndm, ndm: s.ndm.ndm === 2 ? 3 : 2 } }) as State,
  ];
  for (const change of changes) {
    loadsSummaryOf(base);
    const s = change(base);
    assert.deepEqual(loadsSummaryOf(s), loadsSummaryOf(unmemoised(s)));
  }
});
