// reader/ladruno.ts against the fork's MPM fixtures (M2-S0, then M2-S3 #2982).
//
// The fixtures are tests/fixtures/mpm/*.ladruno, byte-identical copies from the fork (see that
// directory's README). ADR 0134 D3 "Before M2" pinned the reader opening both without a layout
// error and reporting no particles. M2-S3 (the TypeScript particle reader) replaced the "no
// particles" pin with "After M2": the particles are read (`LadrunoFile.particles`; their values
// are checked in test/reader/ladrunoParticles.test.ts), and the FE stage index carries no
// particle member, only the ON_PARTICLES bucket listed as PARTICLES_UNREAD. The "opens" pin stays.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { openLadruno, type LadrunoFile } from "../../src/reader/ladruno.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { PARTICLES_UNREAD } from "../../src/state/results.ts";

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

// M2-S7 (#2991): the fork's torn file (M2-S6 #2990) reads the 3 outputs its reference
// `committed_outputs` counts (tests/results/test_mpm_fixtures_baseline.py pins that count),
// with a torn-tail warning; and the torn-stage fixtures (tests/fixtures/mpm/
// make_torn_stage_fixtures.py, closed, so opened by step 1) read by the stage rule of 232a §2.5.
test("particles_torn_v1m2.ladruno: 3 committed outputs, the fork's committed_outputs, with a torn-tail warning", () => {
  withFile("particles_torn_v1m2.ladruno", (f) => {
    assert.equal(f.opened.step, 1);
    assert.deepEqual(f.particles!.refused, []);
    assert.equal(f.particles!.stages.length, 1);
    assert.equal(f.particles!.stages[0]!.nOutputs, 3);
    assert.ok(f.index.warnings.some((w) => /ON_PARTICLES: 3 outputs read; a torn tail/.test(w)), f.index.warnings.join("\n"));
  });
});

for (const [name, stages] of [
  ["particles_torn_last_stage_v1m2.ladruno", { read: [["stage_0", 3]], skipped: ["stage_1"], refused: [] }],
  ["particles_torn_early_stage_v1m2.ladruno", { read: [["stage_1", 3]], skipped: [], refused: ["stage_0"] }],
  ["particles_two_stages_v1m2.ladruno", { read: [["stage_0", 3], ["stage_1", 3]], skipped: [], refused: [] }],
] as const) {
  test(`${name}: opens without a layout error, and the stage rule reads, skips and refuses as pinned`, () => {
    withFile(name, (f) => {
      assert.equal(f.opened.step, 1);
      assert.equal(f.index.formatMinorVersion, 2);
      assert.ok(!f.index.warnings.some((w) => /layout/i.test(w)), f.index.warnings.join("\n"));
      const p = f.particles!;
      assert.deepEqual(p.stages.map((s) => [s.stageId, s.nOutputs]), stages.read);
      assert.deepEqual(p.skipped.map((s) => s.stageId), stages.skipped);
      assert.deepEqual(p.refused.map((s) => s.stageId), stages.refused);
    });
  });
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

  test(`${name}: reads its particles apart from the FE index (ADR 0134 D3 "After M2")`, () => {
    withFile(name, (f) => {
      assert.equal(f.particles?.stages.length, 1, "one particle stage");
      assert.deepEqual(f.particles.refused, []);
      assert.ok(!/particle/i.test(JSON.stringify(Object.keys(f.index))), "the FE index exposes no particle member");
      assert.ok(!/particle/i.test(JSON.stringify(f.index.stages.map((s) => Object.keys(s)))), "an FE stage exposes no particle member");
      const noUnread = JSON.stringify(f.index.stages.map(({ unread: _u, ...rest }) => rest));
      assert.ok(!/particle/i.test(noUnread), "an FE stage exposes no particle result");
      // The one trace in the FE index: the particle stage's ON_PARTICLES bucket, listed as read apart.
      const traces = f.index.stages.flatMap((s) => s.unread.filter((u) => /particle/i.test(u)).map((u) => [s.id, u]));
      assert.deepEqual(traces, [[f.particles.stages[0]!.stageId, PARTICLES_UNREAD]]);
    });
  });
}
