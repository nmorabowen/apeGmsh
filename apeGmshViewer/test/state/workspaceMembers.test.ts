// VU9a (#2803; the D7 still question on #2890): in Nodes mode a click selects
// the clicked cell (the viewport highlights the whole cell), and a save takes
// the node of that cell nearest the click. This drives the real Effects on a
// real model (the gmsh plate of test/fixtures/vu9a): three picks, two
// triangles and one boundary line cell as in the capture still, and checks
// (1) the preview the Sets panel lists, and (2) the edit sent to main's
// `workspace:save`, are the same node tags: nodes, never element ids.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { Effects, withNodes, type Bridge } from "../../src/effects.ts";
import { openModel } from "../../src/reader/node.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { Store } from "../../src/state/store.ts";
import type { Pick, WorkspaceEdit } from "../../src/state/types.ts";

const PLATE = fileURLToPath(new URL("../fixtures/vu9a/plate/remeshed.h5", import.meta.url));

test("Nodes: a save takes, per clicked cell, its node nearest the click; the panel's preview lists exactly those node tags", async () => {
  const model = await openModel(PLATE);
  const blobs = new BlobStore();
  const sent: WorkspaceEdit[] = [];
  const bridge = {
    workspaceSave: async (edit: WorkspaceEdit) => {
      sent.push(edit);
      return { ok: false, reason: "held", detail: "probe", file: "plate.workspace.h5", held: true };
    },
  } as unknown as Bridge;
  const store = new Store();
  const effects = new Effects(store, blobs, bridge);
  try {
    store.dispatch({ type: "fileLoaded", artifact: "model", load: withNodes(loadModel(model, blobs), model, blobs) });
    store.dispatch({ type: "workspaceRead", read: { status: "absent", file: "plate.workspace.h5", token: "t-absent" } });
    const s = store.get();
    const ids = blobs.get(s.mesh!.nodes!.ids) as Float64Array;
    const xyz = blobs.get(s.mesh!.nodes!.xyz) as Float64Array;
    const at = (tag: number) => {
      const i = ids.indexOf(tag);
      return [xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!] as const;
    };
    const cellNodes = (decl: string): number[] => {
      const c = s.decls[decl]!.element!.cell!;
      const b = s.blocks[c.block]!;
      return Array.from((blobs.get(b.connectivity) as ArrayLike<number>) as Int32Array).slice(c.row * b.npe, (c.row + 1) * b.npe);
    };
    // Two triangles and one line cell, as the capture still clicks them; each click lands a little off its cell's second node, toward the cell's middle.
    const drawn = s.mesh!.elements;
    const line = drawn.find((d) => s.decls[d]!.element?.cell && s.blocks[s.decls[d]!.element!.cell!.block]!.npe === 2)!;
    const tris = drawn.filter((d) => s.decls[d]!.element?.cell && s.blocks[s.decls[d]!.element!.cell!.block]!.npe === 3).slice(0, 2);
    const want: number[] = [];
    for (const decl of [...tris, line]) {
      const nodes = cellNodes(decl);
      const target = nodes[1]!;
      const mid = [0, 1, 2].map((k) => nodes.reduce((a, n) => a + at(n)[k]!, 0) / nodes.length);
      const p = at(target).map((v, k) => v + 0.2 * (mid[k]! - v)) as [number, number, number];
      const pick: Pick = { decl, at: p, nodes };
      store.dispatch({ type: "selectAdd", pick });
      if (!want.includes(target)) want.push(target);
    }
    await new Promise((r) => setTimeout(r, 0));
    const preview = store.get().workspace.preview;
    assert.ok(preview && "ids" in preview, JSON.stringify(preview));
    assert.equal(preview.kind, "nodes");
    assert.deepEqual([...preview.ids], want, "the preview lists node tags, the node of each clicked cell nearest the click");
    const femIds = new Set([...tris, line].map((d) => s.decls[d]!.element!.femId));
    assert.ok(!preview.ids.every((t) => femIds.has(t)), "not the clicked cells' element ids");
    store.dispatch({ type: "workspaceSave", edit: { op: "create", name: "picked nodes" } });
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(sent.length, 1);
    assert.deepEqual(sent[0], { op: "create", name: "picked nodes", kind: "nodes", ids: want, expect: "t-absent" }, "what is sent is what the panel listed");
    // Elements: the same selection previews the cells' OpenSees tags; this plate has no /opensees zone, so it says why.
    store.dispatch({ type: "setWorkspaceKind", kind: "elements" });
    await new Promise((r) => setTimeout(r, 0));
    const el = store.get().workspace.preview;
    assert.ok(el && "reason" in el && el.kind === "elements", JSON.stringify(el));
    assert.match(el.reason, /OpenSees element/);
  } finally {
    effects.dispose();
  }
});
