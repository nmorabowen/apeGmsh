// A results set with nodal reactions, for the ADR 0094 `reactions` view: no
// committed fixture records them in a pairable file (the only one,
// ladruno_parts2d, is a partitioned run the app refuses). This copies
// fixtures/contour_staged.h5 and its results into a folder and adds
// `reaction_force_{x,y,z}` to every stage's nodes, in closed form:
//
//   reaction_force_c[k, j] = (stage + 1) (k + 1) (c + 1) (j mod 4)
//
// for step k, the j-th node of the stage's `_ids`, and c = 0, 1, 2 for x, y, z.
// Adding datasets leaves `/meta/lineage` alone, so the copy pairs with the
// model as the original does.

import { copyFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";

const FIXTURES = fileURLToPath(new URL("../../fixtures/", import.meta.url));

export const REACTION_COMPONENTS = ["reaction_force_x", "reaction_force_y", "reaction_force_z"] as const;

/** The closed form at step k of stage `stage`, node position j in `_ids`, component c. */
export const reactionAt = (stage: number, k: number, j: number, c: number): number => (stage + 1) * (k + 1) * (c + 1) * (j % 4);

/** Write the set into `dir`; returns the model and results paths. */
export async function makeReactionSet(dir: string): Promise<{ model: string; results: string }> {
  await h5wasm.ready;
  mkdirSync(dir, { recursive: true });
  const model = join(dir, "reactions.h5");
  const results = join(dir, "reactions.results.h5");
  copyFileSync(join(FIXTURES, "contour_staged.h5"), model);
  copyFileSync(join(FIXTURES, "contour_staged.results.h5"), results);
  const f = new h5wasm.File(results, "a");
  try {
    const stages = f.get("stages") as h5wasm.Group;
    for (const [s, id] of stages.keys().sort().entries()) {
      const nodes = f.get(`stages/${id}/partitions/partition_0/nodes`) as h5wasm.Group;
      const ids = (nodes.get("_ids") as h5wasm.Dataset).value as ArrayLike<number>;
      const T = ((f.get(`stages/${id}/time`) as h5wasm.Dataset).shape ?? [0])[0]!;
      const N = ids.length;
      for (const [c, name] of REACTION_COMPONENTS.entries()) {
        const data = new Float64Array(T * N);
        for (let k = 0; k < T; k++) for (let j = 0; j < N; j++) data[k * N + j] = reactionAt(s, k, j, c);
        nodes.create_dataset({ name, data, shape: [T, N], dtype: "<d" });
      }
    }
  } finally {
    f.close();
  }
  return { model, results };
}
