// reader/ladruno.ts against the fork's MPM fixtures, before the particle reader lands (M2-S0).
//
// The fixtures are tests/fixtures/mpm/*.ladruno, byte-identical copies from the fork (see that
// directory's README). ADR 0134 D3 "Before M2" pins today's behaviour: the reader opens both
// without a layout error and reports no particles. The TypeScript particle slice of M2 replaces
// the "no particles" assertions with the new contract; the "opens" ones stay.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openLadruno, type LadrunoFile } from "../../src/reader/ladruno.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const repo = (rel: string) => fileURLToPath(new URL(`../../../${rel}`, import.meta.url));
const FIXTURES = ["particles_v1m2.ladruno", "particles_mixed_v1m2.ladruno"] as const;

function withFile(name: string, body: (f: LadrunoFile) => void): void {
  const f = openLadruno(H5, repo(`tests/fixtures/mpm/${name}`));
  try {
    body(f);
  } finally {
    f.close();
  }
}

for (const name of FIXTURES) {
  test(`${name}: opens without a layout error`, () => {
    withFile(name, (f) => {
      assert.equal(f.index.generator, "Ladruno");
      assert.equal(f.index.formatVersion, 1);
      assert.equal(f.index.formatMinorVersion, 2);
      assert.ok(!f.index.warnings.some((w) => /layout/i.test(w)), f.index.warnings.join("\n"));
    });
  });

  test(`${name}: reports no particles`, () => {
    withFile(name, (f) => {
      assert.ok(!/particle/i.test(JSON.stringify(Object.keys(f.index))), "index exposes a particle member");
      assert.ok(!/particle/i.test(JSON.stringify(f.index.stages.map((s) => Object.keys(s)))), "a stage exposes particles");
      // The only trace is the "not read" notice for the ON_PARTICLES bucket.
      const noUnread = JSON.stringify(f.index.stages.map(({ unread: _u, ...rest }) => rest));
      assert.ok(!/particle/i.test(noUnread), "a stage exposes a particle result");
      assert.ok(f.index.stages.every((s) => s.unread.every((u) => !/particle/i.test(u) || u.startsWith("ON_PARTICLES (not a RESULTS bucket"))));
    });
  });
}
