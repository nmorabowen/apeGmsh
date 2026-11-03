// A results set with nodal reactions, for the ADR 0094 `reactions` view: no
// committed fixture records them in a pairable file (the only one,
// ladruno_parts2d, is a partitioned run the app refuses). This copies
// fixtures/contour_staged.h5 and its results into a folder and adds
// `reaction_force_{x,y,z}` to every stage's nodes. A reaction exists only at
// a held node: the nodes of the model's support holds
// (`/opensees/stages/stage_000/patterns/Plain_1/sp_holds`, rows of node and
// dof: the `Col` group, held in x, y and z), and is NaN (not recorded) at
// every other node. At a held node n, in closed form:
//
//   reaction_force_c[k] = (stage + 1) (k + 1) (c + 1) (1 + n mod 4)
//
// for step k and c = 0, 1, 2 for x, y, z. Adding datasets leaves
// `/meta/lineage` alone, so the copy pairs with the model as the original does.

import { copyFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";

const FIXTURES = fileURLToPath(new URL("../../fixtures/", import.meta.url));

export const REACTION_COMPONENTS = ["reaction_force_x", "reaction_force_y", "reaction_force_z"] as const;
export const HOLDS = "opensees/stages/stage_000/patterns/Plain_1/sp_holds";

/** The closed form at step k of stage `stage`, held node `node`, component c. */
export const reactionAt = (stage: number, k: number, node: number, c: number): number => (stage + 1) * (k + 1) * (c + 1) * (1 + (node % 4));

/** The held nodes of the model's support holds (the first column of `sp_holds`). */
export function heldNodes(model: string): Set<number> {
  const f = new h5wasm.File(model, "r");
  try {
    const d = f.get(HOLDS) as h5wasm.Dataset;
    if (d.shape?.length !== 2 || d.shape[1] !== 2) throw new Error(`${model}: ${HOLDS} is not (rows, 2): ${JSON.stringify(d.shape)}`);
    const v = Array.from(d.value as ArrayLike<number | bigint>, Number);
    const out = new Set<number>();
    for (let i = 0; i < v.length; i += 2) out.add(v[i]!);
    return out;
  } finally {
    f.close();
  }
}

/**
 * Write the set into `dir`; returns the model and results paths and the held nodes. `free`, when given, is
 * the value written at every other node (a free node's residual or a constraint's force) instead of NaN.
 */
export async function makeReactionSet(dir: string, free?: (node: number) => number): Promise<{ model: string; results: string; held: Set<number> }> {
  await h5wasm.ready;
  mkdirSync(dir, { recursive: true });
  const model = join(dir, "reactions.h5");
  const results = join(dir, "reactions.results.h5");
  copyFileSync(join(FIXTURES, "contour_staged.h5"), model);
  copyFileSync(join(FIXTURES, "contour_staged.results.h5"), results);
  const held = heldNodes(model);
  if (held.size === 0) throw new Error(`${model}: no held node`);
  const f = new h5wasm.File(results, "a");
  try {
    const stages = f.get("stages") as h5wasm.Group;
    for (const [s, id] of stages.keys().sort().entries()) {
      const nodes = f.get(`stages/${id}/partitions/partition_0/nodes`) as h5wasm.Group;
      const ids = Array.from((nodes.get("_ids") as h5wasm.Dataset).value as ArrayLike<number | bigint>, Number);
      const T = ((f.get(`stages/${id}/time`) as h5wasm.Dataset).shape ?? [0])[0]!;
      const N = ids.length;
      for (const [c, name] of REACTION_COMPONENTS.entries()) {
        const data = new Float64Array(T * N);
        for (let k = 0; k < T; k++) for (let j = 0; j < N; j++) data[k * N + j] = held.has(ids[j]!) ? reactionAt(s, k, ids[j]!, c) : free ? free(ids[j]!) : NaN;
        nodes.create_dataset({ name, data, shape: [T, N], dtype: "<d" });
      }
    }
  } finally {
    f.close();
  }
  return { model, results, held };
}
