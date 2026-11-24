// #3130: buildFrameExtrusion finds an OpenSees-only line's declaration by its
// key (`opsRowPath(type, row)`, load.ts `pathOfRef`). It used to scan every
// decl for each such line: a mat on 19 476 inline zeroLength springs (about
// 62 000 decls) never finished loading. The scan also took the first decl of
// any OpenSees-only type with the same row number.

import assert from "node:assert/strict";
import { test } from "node:test";
import type { ElementRef } from "../src/chain/resolve.ts";
import type { ElementMeta, Param } from "../src/model/types.ts";
import { opsRowPath } from "../src/state/decls.ts";
import { buildFrameExtrusion } from "../src/state/frameExtrusion.ts";
import type { Decl } from "../src/state/types.ts";

const BASE = { nameSource: "test", refs: {}, links: {}, problems: [], fields: [] };

/** An OpenSees-only element row's decl, as load.ts `declarationsOf` keys it. */
function opsDecl(meta: ElementMeta, row: number): Decl {
  const path = opsRowPath(meta.type, row);
  const args = meta.args[row] ?? [];
  return {
    ...BASE,
    path,
    h5: meta.path,
    kind: "element",
    type: meta.type,
    tag: null,
    name: `element ${meta.ids[row]}`,
    params: args,
    element: { femId: null, cell: null, inlineNodes: meta.inlineConnectivity![row]!, groups: [], metas: [{ type: meta.type, tag: meta.ids[row]!, h5: meta.path, row, args }] },
  };
}

function meta(type: string, rows: number, args: (row: number) => Param[], nodes: (row: number) => [number, number]): ElementMeta {
  return {
    type,
    path: `/opensees/element_meta/${type}`,
    ids: Float64Array.from({ length: rows }, (_, r) => 1 + r),
    femEids: new Float64Array(rows).fill(-1),
    args: Array.from({ length: rows }, (_, r) => args(r)),
    inlineConnectivity: Array.from({ length: rows }, (_, r) => nodes(r)),
  };
}

test("#3130: an OpenSees-only line is joined to its own type's row, not another type's row of the same number", () => {
  // Row 0 of both types; the zeroLength decl comes first, as a scan would meet it.
  const springs = meta("zeroLength", 1, () => ["-mat", 1, "-dir", 1], () => [1, 2]);
  const truss = meta("Truss", 1, () => [0.01, 1], () => [3, 4]);
  const decls: Record<string, Decl> = { [opsRowPath("zeroLength", 0)]: opsDecl(springs, 0), [opsRowPath("Truss", 0)]: opsDecl(truss, 0) };
  const ext = buildFrameExtrusion({
    linePositions: new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0]),
    lineNodes: new Float64Array([1, 2, 3, 4]),
    lineElementIds: new Int32Array([-1, -1]),
    lineRefs: [
      { kind: "ops", metaIndex: 0, row: 0 },
      { kind: "ops", metaIndex: 1, row: 0 },
    ],
    decls,
    definitions: null,
    elementMeta: [springs, truss],
    ndm: 3,
  });
  assert.deepEqual([...ext.extrudedSegmentMask], [0, 1], "the truss (segment 1) is extruded from its own area; the spring is not");
  assert.ok(ext.frameTriSegments.length > 0 && [...ext.frameTriSegments].every((s) => s === 1));
});

/** `n` inline springs and their `2n` materials, as the model's decls: what a per-line scan walked through. */
function springModel(n: number): { springs: ElementMeta; decls: Record<string, Decl>; lineRefs: ElementRef[] } {
  const springs = meta("zeroLength", n, () => ["-mat", 1, "-dir", 1, 2, 3], (r) => [2 * r + 1, 2 * r + 2]);
  const decls: Record<string, Decl> = {};
  for (let r = 0; r < n; r++) decls[opsRowPath("zeroLength", r)] = opsDecl(springs, r);
  for (let k = 0; k < 2 * n; k++) {
    const path = `opensees/uniaxialMaterial/Elastic_${k}`;
    decls[path] = { ...BASE, path, h5: `/opensees/materials/uniaxial/Elastic_${k}`, kind: "uniaxialMaterial", type: "Elastic", tag: k + 1, name: path, params: [1e5] };
  }
  const lineRefs: ElementRef[] = Array.from({ length: n }, (_, r) => ({ kind: "ops", metaIndex: 0, row: r }));
  return { springs, decls, lineRefs };
}

const extrude = (n: number, springs: ElementMeta, decls: Record<string, Decl>, lineRefs: ElementRef[]) =>
  buildFrameExtrusion({
    linePositions: new Float32Array(6 * n),
    lineNodes: Float64Array.from({ length: 2 * n }, (_, i) => i + 1),
    lineElementIds: new Int32Array(n).fill(-1),
    lineRefs,
    decls,
    definitions: null,
    elementMeta: [springs],
    ndm: 3,
  });

test("#3130: the decls are enumerated a fixed number of times, however many OpenSees-only lines there are", () => {
  // A whole-table walk (Object.values, for...in) asks the proxy for its keys; a keyed lookup does not.
  const walks = (n: number): number => {
    const { springs, decls, lineRefs } = springModel(n);
    let count = 0;
    const counted = new Proxy(decls, {
      ownKeys(target) {
        count++;
        return Reflect.ownKeys(target);
      },
    });
    extrude(n, springs, counted, lineRefs);
    return count;
  };
  const few = walks(10);
  const many = walks(400);
  assert.equal(many, few, `the decls were walked ${many} times for 400 lines and ${few} times for 10: a walk per line`);
});

test("#3130: 20 000 inline zeroLength springs beside 40 000 other decls extrude within seconds (a keyed lookup, not a scan per line)", () => {
  const n = 20_000;
  const { springs, decls, lineRefs } = springModel(n);
  const t0 = performance.now();
  const ext = extrude(n, springs, decls, lineRefs);
  const ms = performance.now() - t0;
  assert.equal(ext.extrudedSegmentMask.length, n);
  // The keyed lookup takes tens of ms; the scan did not finish in 5 minutes on this input. The bound leaves room for a slow runner.
  assert.ok(ms < 5_000, `buildFrameExtrusion took ${ms.toFixed(0)} ms for ${n} inline springs`);
});
