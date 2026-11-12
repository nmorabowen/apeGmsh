// G3 (VA6c, #2765): the auto deform scale a spec with no `deform.scale` takes.
// Two oracles:
//
// - the closed form of Python's rule (`_auto_deform_scale` in
//   src/apeGmsh/viewers/session/_results_stills.py, `_DEFORM_FRACTION` 0.12 in
//   _realize.py, `model_diagonal` in results/_geometry.py), restated: the
//   largest finite |u| component becomes 0.12 of the model's bounding-box
//   diagonal, and the scale is 1 when either is zero;
// - Python's own numbers on the committed fixtures (PY_AUTO below), compared
//   with the scale the app computes from what its Effects read (the step's
//   nodal rows and every model node), at every step of every stage.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { AUTO_DEFORM_FRACTION, autoDeformScale, maxAbsDisplacement, modelDiagonal } from "../../src/state/deform.ts";
import { deformView } from "../../src/state/selectors.ts";
import { opened, STAGED, STAGED_RESULTS, until } from "../contour/harness.ts";

const DEFORM_2D = fileURLToPath(new URL("../../fixtures/deform_2d.h5", import.meta.url));
const DEFORM_2D_RESULTS = fileURLToPath(new URL("../../fixtures/deform_2d.ladruno", import.meta.url));

/**
 * Python's `_auto_deform_scale(results, stage_id, step)` on the fixtures, by
 * stage index and step. Written 2026-11-12 from main ea08f26e3 with
 * `Results.from_native(<set>.results.h5, model=OpenSeesModel.from_h5(<set>.h5))`
 * and `Results.from_ladruno(deform_2d.ladruno, model_h5=deform_2d.h5)`.
 */
const PY_AUTO: Record<string, { model: string; results: string; scales: number[][] }> = {
  contour_staged: {
    model: STAGED,
    results: STAGED_RESULTS,
    scales: [
      [42.719779866435815, 21.359889933217907, 14.23992662214527],
      [14.23992662214527, 7.119963311072635],
      [18.98656882952703, 9.493284414763515, 6.3288562765090095, 4.746642207381758],
    ],
  },
  // A 2-D model (ndm 2) read from a `.ladruno`: z is no degree of freedom, so no z row.
  deform_2d: { model: DEFORM_2D, results: DEFORM_2D_RESULTS, scales: [[20, 10, 6.666666666666667]] },
};

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b));

test("the auto scale's closed form: 0.12 of the diagonal over the largest finite |u| component", () => {
  assert.equal(AUTO_DEFORM_FRACTION, 0.12, "Python's _DEFORM_FRACTION");
  // A 3 x 4 x 12 box: diagonal 13.
  const xyz = new Float64Array([0, 0, 0, 3, 4, 12, 1, 1, 1]);
  assert.equal(modelDiagonal(xyz), 13);
  // The largest component, by absolute value, over x, y and z; NaN and Infinity are skipped (Python keeps only finite values).
  const rows = [new Float64Array([0.1, -0.2, NaN]), new Float64Array([0.05, 0, Infinity]), new Float64Array([-0.39, 0.3, 0])] as const;
  assert.equal(maxAbsDisplacement(rows), 0.39);
  assert.ok(close(autoDeformScale(rows, xyz), (0.12 * 13) / 0.39));
  // A 2-D model has no z row.
  assert.equal(maxAbsDisplacement([new Float64Array([0.5]), new Float64Array([-2]), null]), 2);
});

test("the auto scale is 1 when no displacement is finite or non-zero, and the diagonal is 1 for a point or no node", () => {
  const xyz = new Float64Array([0, 0, 0, 3, 4, 12]);
  assert.equal(autoDeformScale([new Float64Array(3), new Float64Array(3), new Float64Array(3)], xyz), 1, "a zero displacement");
  assert.equal(autoDeformScale([new Float64Array([NaN]), new Float64Array([NaN]), null], xyz), 1, "no finite displacement");
  // Python's model_diagonal: 1 for no node and for a box that is a point, so the scale is then 0.12 / max |u|.
  assert.equal(modelDiagonal(new Float64Array(0)), 1);
  assert.equal(modelDiagonal(new Float64Array([2, 2, 2, 2, 2, 2])), 1);
  assert.ok(close(autoDeformScale([new Float64Array([0.06]), new Float64Array([0]), null], new Float64Array([5, 5, 5])), 2));
  assert.throws(() => modelDiagonal(new Float64Array(4)), /4 coordinates are not 3 per node/);
});

test("the app's auto scale equals Python's _auto_deform_scale at every step of every stage of the fixtures", async () => {
  for (const [name, set] of Object.entries(PY_AUTO)) {
    const h = await opened(set.model, set.results);
    try {
      h.store.dispatch({ type: "setDeform", on: true });
      for (const [stage, scales] of set.scales.entries()) {
        assert.equal(h.store.get().results!.stages[stage]!.time.length, scales.length, `${name} stage ${stage}: the oracle covers every step`);
        for (const [k, want] of scales.entries()) {
          h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
          if (k !== 0) h.store.dispatch({ type: "setResultStep", step: k });
          await until(() => {
            const v = deformView(h.store.get());
            return (v.state === "ready" || v.state === "partial") && h.effects.currentDisplacement()?.key === v.keys.disp;
          }, `${name} stage ${stage} step ${k}`);
          const mesh = h.store.get().mesh!;
          assert.ok(mesh.nodes, `${name}: the model's nodes are read`);
          const got = autoDeformScale(h.effects.currentDisplacement()!.nodal!.rows, h.blobs.get(mesh.nodes.xyz));
          assert.ok(close(got, want), `${name} stage ${stage} step ${k}: the app's ${got}, Python's ${want}`);
        }
      }
    } finally {
      h.effects.dispose();
    }
  }
});
