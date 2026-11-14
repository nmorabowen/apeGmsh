// reader/particleSet.ts against the fork's Contract-1 fixture (ADR 0134 D1; M2-S3 #2982).
//
// The oracle is tests/fixtures/mpm/particle_set_v1.h5 as h5py reads it, and the
// fork's contract §4: 2-D plane strain, t = 1 m, kN_m_s, a 4 x 4 grid of 0.5 m cells
// with GIMP, the soil (8 particles, tag 2, density 2.0) and the block (4 particles,
// tag 1, density 7.85), seeded 2 x 2 per cell. The refusals are fixture copies this
// test edits with h5wasm in a temp folder.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { ParticleContractError } from "../../src/reader/ladrunoParticles.ts";
import { openParticleSet } from "../../src/reader/particleSet.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;
const SET = fileURLToPath(new URL("../../../tests/fixtures/mpm/particle_set_v1.h5", import.meta.url));
const tmp = mkdtempSync(join(tmpdir(), "particle-set-"));
after(() => rmSync(tmp, { recursive: true, force: true }));

test("particle_set_v1.h5 reads 12 particles in 2 bodies, as the fixture holds them", () => {
  const s = openParticleSet(H5, SET);
  assert.equal(s.setId, "3f6c1d2e-8a4b-4c5d-9e7f-0a1b2c3d4e5f");
  assert.equal(s.generator, "apeGmsh");
  assert.equal(s.ndm, 2);
  assert.equal(s.count, 12);
  assert.equal(s.thickness, 1);
  assert.deepEqual(s.units, { length: "m", force: "kN", time: "s", system: "kN_m_s" });
  assert.deepEqual(s.grid, { origin: [0, 0], cellSize: [0.5, 0.5], numCells: [4, 4], basis: "gimp" });
  assert.deepEqual(s.bodies, [{ id: 0, name: "soil" }, { id: 1, name: "block" }]);
  assert.deepEqual(Array.from(s.ids), Array.from({ length: 12 }, (_, i) => i));
  assert.deepEqual(Array.from(s.bodyIds), [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]);
  assert.deepEqual(Array.from(s.materialTags), [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]);
  assert.equal(s.massSource, "density");
  assert.deepEqual(Array.from(s.mass), [...Array(8).fill(2), ...Array(4).fill(7.85)]);
  assert.deepEqual(Array.from(s.volume), Array(12).fill(0.0625));
  assert.deepEqual(Array.from(s.gimpHalfLength), Array(24).fill(0.125));
  assert.deepEqual(
    Array.from(s.positions),
    [0.625, 0.625, 0.875, 0.625, 1.125, 0.625, 1.375, 0.625, 0.625, 0.875, 0.875, 0.875, 1.125, 0.875, 1.375, 0.875, 1.125, 1.125, 1.375, 1.125, 1.125, 1.375, 1.375, 1.375],
  );
  assert.deepEqual(s.provenance, { apegmsh_commit: "unknown", apegmsh_version: "fixture-wp232a", session_id: "9b2e7c41-5d3a-4f6e-8c1b-2a3d4e5f6a7b" });
});

/** A copy of the fixture, changed by `edit` with h5wasm. */
function edited(name: string, edit: (f: h5wasm.File) => void): string {
  const path = join(tmp, name);
  copyFileSync(SET, path);
  const f = new h5wasm.File(path, "a");
  try {
    edit(f);
  } finally {
    f.close();
  }
  return path;
}

const refusedWith = (code: string, text: RegExp) => (e: unknown) => e instanceof ParticleContractError && e.code === code && text.test(e.message);

test("a FORMAT_VERSION other than 1 is refused by name (C1-VER): the exact integer, no floor", () => {
  const p = edited("v2.particles.h5", (f) => {
    f.delete_attribute("FORMAT_VERSION");
    f.create_attribute("FORMAT_VERSION", new Int32Array([2]), []);
  });
  assert.throws(() => openParticleSet(H5, p), refusedWith("C1-VER", /FORMAT_VERSION is 2; this app reads 1/));
});

test("another FORMAT is refused by name (C1-FMT)", () => {
  const p = edited("fmt.particles.h5", (f) => {
    f.delete_attribute("FORMAT");
    f.create_attribute("FORMAT", "something-else");
  });
  assert.throws(() => openParticleSet(H5, p), refusedWith("C1-FMT", /FORMAT is "something-else"/));
});

test("a dataset the contract does not name refuses the set (C1-UNKNOWN), never dropped", () => {
  const p = edited("unknown.particles.h5", (f) => {
    (f.get("PARTICLES") as h5wasm.Group).create_dataset({ name: "PORE_PRESSURE", data: new Float64Array(12) });
  });
  assert.throws(() => openParticleSet(H5, p), refusedWith("C1-UNKNOWN", /PARTICLES\/PORE_PRESSURE is not a Contract-1 dataset/));
});

test("a BODY_ID outside BODIES is refused (C1-BODYREF)", () => {
  const p = edited("bodyref.particles.h5", (f) => {
    (f.get("PARTICLES/BODY_ID") as h5wasm.Dataset).write_slice([[0, 1]], new Int32Array([7]));
  });
  assert.throws(() => openParticleSet(H5, p), refusedWith("C1-BODYREF", /BODY_ID\[0\] = 7 is not in BODIES\/ID/));
});
