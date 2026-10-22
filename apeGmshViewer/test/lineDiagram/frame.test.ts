// #2473 (line-diagram program H0b): the 3-D beam frame fails closed.
// - P0-3: an element type not in ELEMENT_SYNTAX yields no transform-tag candidate,
//   and the "across all transforms" fallback (#2142) answers only when exactly
//   one vecxz matches the decoded tag.
// - P0-4: a vecxz parallel to the beam is never swapped for the default in 3-D;
//   the beam is skipped and counted in the frame-skip notice.

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  beamLocalAxes,
  buildLineDiagram3D,
  elementTransformVecxzFromState,
  type BuildLineDiagram3DOptions,
} from "../../src/state/lineDiagram.ts";
import type { Decl, DeclPath } from "../../src/state/types.ts";
import type { Param } from "../../src/model/types.ts";

type Vec3 = [number, number, number];

function transfDecl(name: string, pairs: { tag: number; vecxz: Vec3 }[]): Decl {
  return {
    path: `opensees/transforms/${name}`,
    kind: "geomTransf",
    type: "Linear",
    name,
    nameSource: "/opensees/names",
    h5: `/opensees/transforms/${name}`,
    tag: pairs[0]!.tag,
    params: [],
    fields: [],
    refs: {},
    links: {},
    problems: [],
    vecxz: pairs,
  };
}

function beamDecl(type: string, args: Param[], refs: Record<string, DeclPath> = {}): Decl {
  return {
    path: "mesh/element/1",
    kind: "element",
    type,
    name: "Beam1",
    nameSource: "/elements/line2",
    h5: "/elements/line2",
    tag: 10,
    params: [],
    fields: [],
    refs,
    links: {},
    problems: [],
    element: {
      femId: 1,
      cell: { alias: "line2", block: 0, row: 0 },
      inlineNodes: [],
      groups: [],
      metas: [{ type, tag: 10, h5: `/opensees/element_meta/${type}`, row: 0, args }],
    },
  };
}

/** One beam (ops tag 10) from `p0` to `p1`, Mz stations at both ends. */
function scene(
  decls: Record<DeclPath, Decl>,
  p0: Vec3,
  p1: Vec3,
  component: "Mz" | "N" = "Mz",
  ndm: 2 | 3 = 3,
  byTag?: ReadonlyMap<number, Vec3>,
): BuildLineDiagram3DOptions {
  return {
    decls,
    meshElements: ["mesh/element/1"],
    lineElement: new Int32Array([0]),
    drawnLinePositions: new Float32Array([...p0, ...p1]),
    join: {
      ok: true,
      space: "ops_tag",
      byLadrunoId: new Map<number, DeclPath>([[10, "mesh/element/1"]]),
      byDeclPath: new Map<DeclPath, number>([["mesh/element/1", 10]]),
    },
    stepData: {
      component,
      elementIndex: new Float64Array([10, 10]),
      stationNaturalCoord: new Float64Array([-1, 1]),
      values: new Float64Array([-5, 5]),
      time: 1,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    modelTransformVecxzByTag: byTag,
    ndm,
    scale: 1,
  };
}

// -- P0-3 ---------------------------------------------------------------------

test("P0-3: an undecoded element type whose node tag equals a transform tag is skipped, never framed by the collision", () => {
  // `MysteryBeam3d` is not in ELEMENT_SYNTAX. Its args hold node tag 2 (and 1);
  // the model has transforms 1 and 2. Neither integer is known to be a transform tag.
  const t1 = transfDecl("Linear_1", [{ tag: 1, vecxz: [0, 0, 1] }]);
  const t2 = transfDecl("Linear_2", [{ tag: 2, vecxz: [0, 1, 0] }]);
  const beam = beamDecl("MysteryBeam3d", [1, 2, 0.01, 2e11]);
  const decls: Record<DeclPath, Decl> = { [t1.path]: t1, [t2.path]: t2, [beam.path]: beam };
  const byTag = new Map<number, Vec3>([
    [1, [0, 0, 1]],
    [2, [0, 1, 0]],
  ]);

  assert.equal(elementTransformVecxzFromState(decls, beam), null);
  assert.equal(elementTransformVecxzFromState(decls, beam, byTag), null);

  const geom = buildLineDiagram3D(scene(decls, [0, 0, 0], [4, 0, 0], "Mz", 3, byTag));
  assert.equal(geom.beams.length, 0);
  assert.equal(geom.skippedUnorientedBeams, 1);
  assert.deepEqual(geom.skippedBeamDecls, ["mesh/element/1"]);
  assert.match(geom.notice ?? "", /^1 3-D beam skipped: Mz in local plane requires LOCAL_AXES or geomTransf vecxz/);
});

test("P0-3: a decoded transfTag without refs.transfTag still resolves on a fan-out transform (#2142)", () => {
  // elasticBeamColumn 3-D form `A E G J Iy Iz transfTag`: slot 6 is the transform.
  const fan = transfDecl("Linear_1", [
    { tag: 5, vecxz: [0, 0, 1] },
    { tag: 6, vecxz: [0, 1, 0] },
  ]);
  const other = transfDecl("Linear_2", [{ tag: 7, vecxz: [1, 0, 0] }]);
  const beam = beamDecl("elasticBeamColumn", [0.01, 2e11, 8e10, 1e-4, 1e-4, 2e-4, 6]);
  const decls: Record<DeclPath, Decl> = { [fan.path]: fan, [other.path]: other, [beam.path]: beam };
  assert.deepEqual(elementTransformVecxzFromState(decls, beam), [0, 1, 0]);

  const geom = buildLineDiagram3D(scene(decls, [0, 0, 0], [4, 0, 0]));
  assert.equal(geom.skippedUnorientedBeams, 0);
  assert.equal(geom.beams.length, 1);
  assert.equal(geom.beams[0]!.zRecSource, "model_transform");
});

test("P0-3: two transforms claiming the decoded tag with different vecxz are ambiguous, and the beam is skipped", () => {
  const a = transfDecl("Linear_1", [{ tag: 6, vecxz: [0, 0, 1] }]);
  const b = transfDecl("Linear_2", [{ tag: 6, vecxz: [0, 1, 0] }]);
  const beam = beamDecl("elasticBeamColumn", [0.01, 2e11, 8e10, 1e-4, 1e-4, 2e-4, 6]);
  const decls: Record<DeclPath, Decl> = { [a.path]: a, [b.path]: b, [beam.path]: beam };
  assert.equal(elementTransformVecxzFromState(decls, beam), null);

  const geom = buildLineDiagram3D(scene(decls, [0, 0, 0], [4, 0, 0]));
  assert.equal(geom.beams.length, 0);
  assert.deepEqual(geom.skippedBeamDecls, ["mesh/element/1"]);
});

// -- P0-4 ---------------------------------------------------------------------

test("P0-4: beamLocalAxes returns null for a vecxz parallel to the chord", () => {
  assert.equal(beamLocalAxes([0, 0, 1], [0, 0, 1]), null);
  assert.equal(beamLocalAxes([0, 0, 1], [0, 0, -3]), null);
  assert.equal(beamLocalAxes([1, 0, 0], [0, 0, 0]), null);
  assert.equal(beamLocalAxes([1, 0, 0], [Number.NaN, 0, 1]), null);
  assert.notEqual(beamLocalAxes([0, 0, 1], [1, 0, 0]), null);
});

test("P0-4: a vertical 3-D beam with vecxz=(0,0,1) is skipped for Mz and counted in the notice", () => {
  const t = transfDecl("Linear_1", [{ tag: 1, vecxz: [0, 0, 1] }]);
  const beam = beamDecl("elasticBeamColumn", [0.01, 2e11, 8e10, 1e-4, 1e-4, 2e-4, 1], { transfTag: t.path });
  const decls: Record<DeclPath, Decl> = { [t.path]: t, [beam.path]: beam };
  const geom = buildLineDiagram3D(scene(decls, [0, 0, 0], [0, 0, 3]));
  assert.equal(geom.beams.length, 0);
  assert.equal(geom.skippedUnorientedBeams, 1);
  assert.deepEqual(geom.skippedBeamDecls, ["mesh/element/1"]);
  assert.match(geom.notice ?? "", /^1 3-D beam skipped: Mz in local plane requires LOCAL_AXES or geomTransf vecxz/);
});

test("P0-4: the same degenerate frame still draws N (no 3-D frame needed), on the default vecxz", () => {
  const t = transfDecl("Linear_1", [{ tag: 1, vecxz: [0, 0, 1] }]);
  const beam = beamDecl("elasticBeamColumn", [0.01, 2e11, 8e10, 1e-4, 1e-4, 2e-4, 1], { transfTag: t.path });
  const decls: Record<DeclPath, Decl> = { [t.path]: t, [beam.path]: beam };
  const geom = buildLineDiagram3D(scene(decls, [0, 0, 0], [0, 0, 3], "N"));
  assert.equal(geom.skippedUnorientedBeams, 0);
  assert.equal(geom.beams.length, 1);
  assert.equal(geom.beams[0]!.zRecSource, "default_vecxz");
  assert.equal(geom.notice, null);
});
