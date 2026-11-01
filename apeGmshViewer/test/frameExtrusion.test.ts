import assert from "node:assert/strict";
import { test } from "node:test";
import type { ModelFile, OpsObject, SectionProfile } from "../src/model/types.ts";
import { sectionProfileOf, transformJntOffsetOf } from "../src/reader/read.ts";
import { basePositions, deformFull } from "../src/renderer/deform.ts";
import { BlobStore } from "../src/state/blobs.ts";
import { vertexDisplacements } from "../src/state/deform.ts";
import { buildFrameExtrusion, triangulateSectionProfile } from "../src/state/frameExtrusion.ts";
import { buildLineDiagram3D } from "../src/state/lineDiagram.ts";
import { loadModel } from "../src/state/load.ts";
import { initialState, reduce } from "../src/state/store.ts";
import type { Decl } from "../src/state/types.ts";

test("transformJntOffsetOf extracts 2-D and 3-D -jntOffset vectors from geomTransf objects", () => {
  const tr2d: OpsObject = {
    family: "geomTransf",
    path: "/opensees/transforms/Linear_1",
    groupName: "Linear_1",
    type: "Linear",
    tag: 1,
    params: ["-jntOffset", 0.1, -0.2, -0.15, 0.25],
    attrs: {},
    tables: {},
  };
  assert.deepEqual(transformJntOffsetOf(tr2d), [
    {
      tag: 1,
      dI: [0.1, -0.2, 0],
      dJ: [-0.15, 0.25, 0],
    },
  ]);

  const tr3d: OpsObject = {
    family: "geomTransf",
    path: "/opensees/transforms/PDelta_2",
    groupName: "PDelta_2",
    type: "PDelta",
    tag: 2,
    params: [0, 0, 1],
    attrs: {},
    tables: {
      jnt_offset: {
        path: "/opensees/transforms/PDelta_2/jnt_offset",
        columns: ["dXi", "dYi", "dZi", "dXj", "dYj", "dZj"],
        rows: [[0.05, 0.1, 0.2, -0.05, -0.1, -0.2]],
      },
      per_element_emitted_tag: {
        path: "/opensees/transforms/PDelta_2/per_element_emitted_tag",
        columns: ["v0"],
        rows: [[2]],
      },
    },
  };
  assert.deepEqual(transformJntOffsetOf(tr3d), [
    {
      tag: 2,
      dI: [0.05, 0.1, 0.2],
      dJ: [-0.05, -0.1, -0.2],
    },
  ]);

  const trZero: OpsObject = {
    family: "geomTransf",
    path: "/opensees/transforms/Corotational_3",
    groupName: "Corotational_3",
    type: "Corotational",
    tag: 3,
    params: [0, 0, 1, "-jntOffset", 0, 0, 0, 0, 0, 0],
    attrs: {},
    tables: {},
  };
  assert.deepEqual(transformJntOffsetOf(trZero), [], "all-zero -jntOffset is ignored");
});

test("sectionProfileOf resolves ComputedSection outline_yz, multi-patch W/HSS/circ fibers, RCSection2d, ElasticTube, and Elastic", () => {
  // Tier 1A: ComputedSection outline_yz (exterior + hole)
  const compSec: OpsObject = {
    family: "section",
    path: "/opensees/sections/Elastic_10",
    groupName: "Elastic_10",
    type: "Elastic",
    tag: 10,
    params: [2e11, 0.04, 1e-4, 1e-4, 8e10, 2e-5],
    attrs: {},
    tables: {},
  };
  const outline: readonly (readonly [number, number][])[] = [
    [
      [-0.1, -0.2],
      [0.1, -0.2],
      [0.1, 0.2],
      [-0.1, 0.2],
    ],
    [
      [-0.05, -0.1],
      [-0.05, 0.1],
      [0.05, 0.1],
      [0.05, -0.1],
    ],
  ];
  const pComp = sectionProfileOf(compSec, outline)!;
  assert.equal(pComp.source, "computed");
  assert.equal(pComp.loops.length, 2);
  assert.equal(pComp.loops[0]!.length, 4);
  assert.equal(pComp.loops[1]!.length, 4);

  // Tier 1B: 3-patch W-shape (bottom flange, web, top flange) -> 12-vertex I-beam outline
  // Uses the real model.h5 /opensees/sections/{name}/patches schema: ["kind", "material_ref", "ny", "nz", "coords"]
  const wFiber: OpsObject = {
    family: "section",
    path: "/opensees/sections/Fiber_11",
    groupName: "Fiber_11",
    type: "Fiber",
    tag: 11,
    params: ["-GJ", 1e5],
    attrs: {},
    tables: {
      patches: {
        path: "/opensees/sections/Fiber_11/patches",
        columns: ["kind", "material_ref", "ny", "nz", "coords"],
        rows: [
          ["rect", "/opensees/materials/Steel01_1", 2, 8, [-0.15, -0.1, -0.13, 0.1, 0, 0, 0, 0]],
          ["rect", "/opensees/materials/Steel01_1", 8, 2, [-0.13, -0.01, 0.13, 0.01, 0, 0, 0, 0]],
          ["rect", "/opensees/materials/Steel01_1", 2, 8, [0.13, -0.1, 0.15, 0.1, 0, 0, 0, 0]],
        ],
      },
    },
  };
  const pW = sectionProfileOf(wFiber)!;
  assert.equal(pW.source, "patches");
  assert.equal(pW.loops.length, 1, "contiguous W-shape patches merge into 1 exterior loop");
  assert.equal(pW.loops[0]!.length, 12, "merged W-shape has 12 collinear-simplified vertices");

  // Tier 1B: 4-patch hollow rectangular HSS -> 1 outer loop + 1 inner hole loop
  const hssFiber: OpsObject = {
    family: "section",
    path: "/opensees/sections/Fiber_12",
    groupName: "Fiber_12",
    type: "Fiber",
    tag: 12,
    params: ["-GJ", 1e5],
    attrs: {},
    tables: {
      patches: {
        path: "/opensees/sections/Fiber_12/patches",
        columns: ["kind", "material_ref", "ny", "nz", "coords"],
        rows: [
          ["rect", "/opensees/materials/Steel01_1", 2, 8, [-0.1, -0.1, -0.08, 0.1, 0, 0, 0, 0]],
          ["rect", "/opensees/materials/Steel01_1", 2, 8, [0.08, -0.1, 0.1, 0.1, 0, 0, 0, 0]],
          ["rect", "/opensees/materials/Steel01_1", 8, 2, [-0.08, -0.1, 0.08, -0.08, 0, 0, 0, 0]],
          ["rect", "/opensees/materials/Steel01_1", 8, 2, [-0.08, 0.08, 0.08, 0.1, 0, 0, 0, 0]],
        ],
      },
    },
  };
  const pHss = sectionProfileOf(hssFiber)!;
  assert.equal(pHss.source, "patches");
  assert.equal(pHss.loops.length, 2, "4-wall HSS merges into outer loop + inner hole loop");
  assert.equal(pHss.loops[0]!.length, 4);
  assert.equal(pHss.loops[1]!.length, 4);
  const hssTri = triangulateSectionProfile(pHss);
  assert.ok(hssTri.triangles.length >= 8, "hollow HSS triangulates with hole bridged");

  // Tier 1B: circ sector patch (0..180 deg) and quadr patch
  const circSectorSec: OpsObject = {
    family: "section",
    path: "/opensees/sections/Fiber_12b",
    groupName: "Fiber_12b",
    type: "Fiber",
    tag: 120,
    params: [],
    attrs: {},
    tables: {
      patches: {
        path: "/opensees/sections/Fiber_12b/patches",
        columns: ["kind", "material_ref", "ny", "nz", "coords"],
        rows: [["circ", "/opensees/materials/Steel01_1", 8, 2, [0, 0, 0.05, 0.15, 0, 180, 0, 0]]],
      },
    },
  };
  const pSector = sectionProfileOf(circSectorSec)!;
  assert.equal(pSector.source, "patches");
  assert.equal(pSector.loops.length, 1, "half-annulus sector forms 1 closed loop");
  assert.equal(pSector.axisymmetric, undefined, "partial sector is not axisymmetric");

  // Multi-island section with a hole in the second island
  const twoBoxHss = triangulateSectionProfile({
    source: "computed",
    loops: [
      [
        [-0.3, -0.1],
        [-0.1, -0.1],
        [-0.1, 0.1],
        [-0.3, 0.1],
      ],
      [
        [0.1, -0.1],
        [0.3, -0.1],
        [0.3, 0.1],
        [0.1, 0.1],
      ],
      [
        [0.15, -0.05],
        [0.15, 0.05],
        [0.25, 0.05],
        [0.25, -0.05],
      ],
    ],
  });
  assert.ok(twoBoxHss.triangles.length >= 10, "multi-island section bridges hole into second island");

  // Tier 2: RCSection2d, WFSection2d, ElasticTube
  const rc2d: OpsObject = {
    family: "section",
    path: "/opensees/sections/RCSection2d_13",
    groupName: "RCSection2d_13",
    type: "RCSection2d",
    tag: 13,
    params: [1, 2, 3, 0.6, 0.3, 0.04, 0.002, 0.002, 0.001, 4, 4, 2],
    attrs: {},
    tables: {},
  };
  const pRc = sectionProfileOf(rc2d)!;
  assert.equal(pRc.source, "parametric");
  assert.deepEqual(pRc.loops[0], [
    [-0.3, -0.15],
    [0.3, -0.15],
    [0.3, 0.15],
    [-0.3, 0.15],
  ]);

  const wf2d: OpsObject = {
    family: "section",
    path: "/opensees/sections/WFSection2d_14",
    groupName: "WFSection2d_14",
    type: "WFSection2d",
    tag: 14,
    params: [1, 0.4, 0.01, 0.2, 0.02, 8, 4],
    attrs: {},
    tables: {},
  };
  const pWf = sectionProfileOf(wf2d)!;
  assert.equal(pWf.source, "parametric");
  assert.equal(pWf.loops[0]!.length, 12);

  const tube: OpsObject = {
    family: "section",
    path: "/opensees/sections/ElasticTube_15",
    groupName: "ElasticTube_15",
    type: "ElasticTube",
    tag: 15,
    params: [2e11, 0.3, 0.01, 0.3],
    attrs: {},
    tables: {},
  };
  const pTube = sectionProfileOf(tube)!;
  assert.equal(pTube.source, "parametric");
  assert.equal(pTube.loops.length, 2, "ElasticTube has outer and inner circular loops");

  // Tier 3: Elastic / ElasticShear 3-D section with Iz = b*d^3/12, Iy = d*b^3/12
  const d = 0.6, b = 0.3;
  const A = b * d;
  const Iz = (b * d ** 3) / 12;
  const Iy = (d * b ** 3) / 12;
  const elastic3d: OpsObject = {
    family: "section",
    path: "/opensees/sections/ElasticShear_16",
    groupName: "ElasticShear_16",
    type: "ElasticShear",
    tag: 16,
    params: [2e11, A, Iz, Iy, 8e10, 1e-4, 0.833, 0.833],
    attrs: {},
    tables: {},
  };
  const pEl = sectionProfileOf(elastic3d)!;
  assert.equal(pEl.source, "elastic");
  assert.ok(Math.abs(pEl.loops[0]![2]![0] - d / 2) < 1e-6);
  assert.ok(Math.abs(pEl.loops[0]![2]![1] - b / 2) < 1e-6);
});

test("buildFrameExtrusion handles Tier 2 InsertionPoint decl_params bbox, -endZones, P0-2 reversed node alignment, and 3-D fail-closed rule", () => {
  // Build a synthetic ModelFile with:
  // - Element 101: 3-D dispBeamColumn with Elastic section 10, geomTransf 1 carrying InsertionPoint (depth=0.5, width=0.25) in decl_params, and reversed inlineConnectivity [1, 2] vs cell [2, 1]
  // - Element 102: 3-D LadrunoDispBeamColumn with -endZones 0.4 0.6 and transform -jntOffset (10-token params)
  // - Element 103: 3-D dispBeamColumn referencing a transform WITHOUT vecxz (must fail closed: not extruded, increments extrusionSkippedUnoriented)
  const rawMat1: OpsObject = {
    family: "uniaxialMaterial",
    path: "/opensees/materials/Elastic_1",
    groupName: "Elastic_1",
    type: "Elastic",
    tag: 1,
    params: [3e10],
    attrs: {},
    tables: {},
  };
  const rawSec10: OpsObject = {
    family: "section",
    path: "/opensees/sections/Elastic_10",
    groupName: "Elastic_10",
    type: "Elastic",
    tag: 10,
    // Non-axisymmetric Tier 3 elastic section (Iz = 4e-5 != Iy = 1e-5 -> hy != bz)
    params: [3e10, 0.01, 4e-5, 1e-5, 1.2e10, 1e-5],
    attrs: {},
    tables: {},
  };
  const rawTr1: OpsObject = {
    family: "geomTransf",
    path: "/opensees/transforms/Linear_1",
    groupName: "Linear_1",
    type: "Linear",
    tag: 1,
    // 10-element 3-D geomTransf params with inline "-jntOffset"
    params: [0, 0, 1, "-jntOffset", 0.2, 0, -0.25, -0.3, 0, -0.25],
    attrs: {},
    tables: {},
  };
  const rawTrNoVec: OpsObject = {
    family: "geomTransf",
    path: "/opensees/transforms/Linear_2",
    groupName: "Linear_2",
    type: "Linear",
    tag: 2,
    params: [],
    attrs: {},
    tables: {},
  };
  const rawInt1: OpsObject = {
    family: "beamIntegration",
    path: "/opensees/beam_integration/Lobatto_1",
    groupName: "Lobatto_1",
    type: "Lobatto",
    tag: 1,
    params: [10, 5],
    attrs: {},
    tables: {},
  };

  const model: ModelFile = {
    path: "synthetic.h5",
    sizeBytes: 1024,
    neutralVersion: "1.0.0",
    meta: { ndm: 3, ndf: 6 },
    nodeIds: new Float64Array([1, 2, 3, 4]),
    nodeCoords: new Float64Array([
      0, 0, 0, // node 1
      4, 0, 0, // node 2
      0, 2, 0, // node 3
      4, 2, 0, // node 4
    ]),
    blocks: [
      {
        alias: "line2",
        code: 1,
        dim: 1,
        npe: 2,
        ids: new Float64Array([101, 102, 103]),
        // Cell 0 has connectivity [2, 1] while OpenSees element 101 has inlineConnectivity [1, 2] (P0-2 reversed!)
        connectivity: new Float64Array([2, 1, 3, 4, 1, 3]),
      },
    ],
    physicalGroups: [
      {
        name: "Beams",
        dim: 1,
        tag: 1,
        path: "/physical_groups/element_side/Beams",
        elementIds: new Float64Array([101, 102, 103]),
      },
    ],
    labels: [],
    opensees: {
      version: "2.34.0",
      objects: [rawMat1, rawSec10, rawTr1, rawTrNoVec, rawInt1],
      elementMeta: [
        {
          type: "dispBeamColumn",
          path: "/opensees/element_meta/dispBeamColumn",
          ids: new Float64Array([101, 103]),
          femEids: new Float64Array([101, 103]),
          args: [
            [1, 1],
            [2, 1],
          ],
          inlineConnectivity: [
            [1, 2],
            [1, 3],
          ],
        },
        {
          type: "LadrunoDispBeamColumn",
          path: "/opensees/element_meta/LadrunoDispBeamColumn",
          ids: new Float64Array([102]),
          femEids: new Float64Array([102]),
          args: [[1, 1, "-endZones", 0.4, 0.6]],
          inlineConnectivity: null,
        },
      ],
      names: [],
      definitions: {
        decls: {
          hasParams: true,
          records: [
            {
              row: 0,
              key: "opensees/geomTransf/Linear_1",
              family: "geomTransf",
              name: "Linear_1",
              synth: false,
              params: [
                {
                  row: 0,
                  type: "Linear",
                  params: {
                    jnt_offset: {
                      $struct: "InsertionPoint",
                      fields: {
                        cardinal_point: 8,
                        depth: 0.5,
                        width: 0.25,
                        bbox: null,
                        axis_2: "y",
                        mirror_2: false,
                        mirror_3: false,
                      },
                    },
                  },
                  paramsNames: null,
                  refs: { transf: [], integration: [], section: [] },
                },
              ],
            },
            {
              row: 1,
              key: "opensees/uniaxialMaterial/Elastic_1",
              family: "uniaxialMaterial",
              name: "Elastic_1",
              synth: false,
              params: [],
            },
            {
              row: 2,
              key: "opensees/section/Elastic_10",
              family: "section",
              name: "Elastic_10",
              synth: false,
              params: [],
            },
            {
              row: 3,
              key: "opensees/geomTransf/Linear_2",
              family: "geomTransf",
              name: "Linear_2",
              synth: false,
              params: [],
            },
            {
              row: 4,
              key: "opensees/beamIntegration/Lobatto_1",
              family: "beamIntegration",
              name: "Lobatto_1",
              synth: false,
              params: [],
            },
          ],
          tags: [
            { kind: "geomTransf", first: 1, count: 1, decl: 0 },
            { kind: "uniaxialMaterial", first: 1, count: 1, decl: 1 },
            { kind: "section", first: 10, count: 1, decl: 2 },
            { kind: "geomTransf", first: 2, count: 1, decl: 3 },
            { kind: "beamIntegration", first: 1, count: 1, decl: 4 },
          ],
          rows: {},
        },
        analysis: null,
        timeSeries: [],
        patterns: [],
        recorders: [],
        bcs: { fix: null, mass: null },
        stages: [],
        embeddedRebar: null,
        program: null,
      },
    },
    sequence: { status: "absent" },
    warnings: [],
    readMs: 1,
  };

  const blobs = new BlobStore();
  const load = loadModel(model, blobs);
  const s = reduce(initialState, { type: "fileLoaded", artifact: "model", load });

  assert.equal(s.visibility.showOffsets, true, "fileLoaded auto-enables showOffsets when model has joint offsets");
  assert.equal(s.visibility.extrudeFrames, false, "extrudeFrames starts false");

  const fe = s.mesh?.frameExtrusion;
  assert.ok(fe, "frameExtrusion blobs are built");
  assert.equal(fe.hasExtrudableFrames, true);
  assert.equal(fe.hasJointOffsets, true);
  assert.equal(fe.extrusionSkippedUnoriented, 1, "segment 2 (missing 3-D vecxz) is counted in extrusionSkippedUnoriented");

  const extMask = blobs.get(fe.extrudedSegmentMask) as Uint8Array;
  assert.deepEqual(
    Array.from(extMask),
    [1, 1, 0],
    "segments 0 and 1 are extruded; segment 2 (no vecxz on 3-D transform 2) fails closed",
  );

  // Check P0-2 reversed node alignment on segment 0 (cell [2, 1], OpenSees inlineNodes [1, 2]):
  // OpenSees I = node 1 (at x=0) has dI = (0.2, 0, -0.25) -> pIPrime = (0.2, 0, -0.25)
  // OpenSees J = node 2 (at x=4) has dJ = (-0.3, 0, -0.25) -> pJPrime = (3.7, 0, -0.25)
  // Since cell 0 has start=node 2 (x=4) and end=node 1 (x=0):
  // clearLinePositions for seg 0 starts at pJPrime = (3.7, 0, -0.25) and ends at pIPrime = (0.2, 0, -0.25)!
  const clearLines = blobs.f32(fe.clearLinePositions);
  assert.ok(Math.abs(clearLines[0]! - 3.7) < 1e-5, `seg 0 start X is 3.7 (got ${clearLines[0]})`);
  assert.ok(Math.abs(clearLines[2]! - -0.25) < 1e-5, `seg 0 start Z is -0.25 (got ${clearLines[2]})`);
  assert.ok(Math.abs(clearLines[3]! - 0.2) < 1e-5, `seg 0 end X is 0.2 (got ${clearLines[3]})`);
  assert.ok(Math.abs(clearLines[5]! - -0.25) < 1e-5, `seg 0 end Z is -0.25 (got ${clearLines[5]})`);

  // Check segment 1 (Element 102, nodes 3=(0,2,0) -> 4=(4,2,0)):
  // Transform 1 has dI = (0.2, 0, -0.25), dJ = (-0.3, 0, -0.25) AND -endZones 0.4, 0.6 along +X
  // Total offset at I (node 3): x = 0 + 0.2 + 0.4 = 0.6, y = 2, z = -0.25
  // Total offset at J (node 4): x = 4 - 0.3 - 0.6 = 3.1, y = 2, z = -0.25
  assert.ok(Math.abs(clearLines[6]! - 0.6) < 1e-5, `seg 1 start X is 0.6 (got ${clearLines[6]})`);
  assert.ok(Math.abs(clearLines[8]! - -0.25) < 1e-5, `seg 1 start Z is -0.25 (got ${clearLines[8]})`);
  assert.ok(Math.abs(clearLines[9]! - 3.1) < 1e-5, `seg 1 end X is 3.1 (got ${clearLines[9]})`);
  assert.ok(Math.abs(clearLines[11]! - -0.25) < 1e-5, `seg 1 end Z is -0.25 (got ${clearLines[11]})`);

  // Verify buildLineDiagram3D shifts stations to the clear span [pIPrime, pJPrime]
  const diag = buildLineDiagram3D({
    decls: s.decls,
    ndm: 3,
    meshElements: s.mesh!.elements,
    lineElement: blobs.get(s.mesh!.lineElement) as Int32Array,
    drawnLinePositions: blobs.f32(s.mesh!.linePositions),
    lineNodes: blobs.get(s.mesh!.vertexNodes.line) as Float64Array,
    join: {
      ok: true,
      space: "fem",
      byLadrunoId: new Map([[102, "mesh/element/102"]]),
      byDeclPath: new Map([["mesh/element/102", 102]]),
    },
    stepData: {
      component: "Vy",
      elementIndex: new Float64Array([102, 102]),
      stationNaturalCoord: new Float64Array([-1, 1]),
      values: new Float64Array([10, 10]),
      time: 1,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    scale: 1,
    flipSign: false,
    cmap: "sign",
  });
  assert.equal(diag.beams.length, 1);
  // Span endpoints in beams[0] start at x = 0.6, z = -0.25 and end at x = 3.1, z = -0.25
  assert.ok(Math.abs(diag.beams[0]!.pStart[0] - 0.6) < 1e-5);
  assert.ok(Math.abs(diag.beams[0]!.pStart[2] - -0.25) < 1e-5);
  assert.ok(Math.abs(diag.beams[0]!.pEnd[0] - 3.1) < 1e-5);
  assert.ok(Math.abs(diag.beams[0]!.pEnd[2] - -0.25) < 1e-5);

  // Verify toggleExtrudeFrames and toggleShowOffsets reducer transitions
  const s1 = reduce(s, { type: "toggleExtrudeFrames" });
  assert.equal(s1.visibility.extrudeFrames, true);
  const s2 = reduce(s1, { type: "toggleShowOffsets" });
  assert.equal(s2.visibility.showOffsets, false);

  // Verify deformFull translates both rigid offset links and prism vertices by the exact same nodal displacement
  const base = basePositions(s.mesh!, blobs);
  const col = new Map<number, number>([
    [1, 0],
    [2, 1],
    [3, 2],
    [4, 3],
  ]);
  const rows: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array([0, 0, 0, 0]),
    new Float64Array([0, 0, 0, 0]),
    new Float64Array([-0.1, -0.2, -0.3, -0.4]),
  ];
  const vd = (ref: typeof fe.triNodes) => vertexDisplacements(blobs.get(ref) as Float64Array, col, rows);
  const def = deformFull(
    base,
    {
      key: "test-disp",
      line: vd(s.mesh!.vertexNodes.line),
      tri: vd(s.mesh!.vertexNodes.tri),
      edge: vd(s.mesh!.vertexNodes.edge),
      frameTri: vd(fe.triNodes),
      frameEdge: vd(fe.edgeNodes),
      offsetLine: vd(fe.offsetLineNodes),
      offsetJoint: vd(fe.offsetJointNodes),
    },
    2,
    "test-disp|scale 2",
  );
  assert.ok(def.frameTri && base.frameTri);
  assert.ok(def.offsetLine && base.offsetLine);
  // First offsetLine is at node 1 (moves by 2 * -0.1 = -0.2 in Z at both pI and pIPrime)
  assert.ok(Math.abs(def.offsetLine[2]! - (base.offsetLine[2]! - 0.2)) < 1e-5);
  assert.ok(Math.abs(def.offsetLine[5]! - (base.offsetLine[5]! - 0.2)) < 1e-5);
  // First prism vertex is at node 1 (moves by the exact same 2 * -0.1 = -0.2 in Z)
  assert.ok(Math.abs(def.frameTri[2]! - (base.frameTri[2]! - 0.2)) < 1e-5);
});

test("frameExtrusion: subdivided collinear beam omits internal partition caps, deduplicates interior offsets, and stays watertight when deformed", () => {
  const rectProfile: SectionProfile = {
    loops: [
      [
        [-0.25, -0.15],
        [0.25, -0.15],
        [0.25, 0.15],
        [-0.25, 0.15],
      ],
    ],
    source: "parametric",
  };
  const baseDecl = {
    nameSource: "test",
    h5: "/test",
    refs: {},
    links: {},
    problems: [],
    fields: [],
  };
  const decls: Record<string, Decl> = {
    "opensees/geomTransf/Linear_1": {
      ...baseDecl,
      path: "opensees/geomTransf/Linear_1",
      kind: "geomTransf",
      type: "Linear",
      tag: 1,
      name: "GirderTransf",
      params: [0, -1, 0, "-jntOffset", 0, 0, -0.25, 0, 0, -0.25],
      vecxz: [{ tag: 1, vecxz: [0, -1, 0] }],
      jntOffset: [{ tag: 1, dI: [0, 0, -0.25], dJ: [0, 0, -0.25] }],
    },
    "opensees/section/Elastic_10": {
      ...baseDecl,
      path: "opensees/section/Elastic_10",
      kind: "section",
      type: "Elastic",
      tag: 10,
      name: "Sec10",
      params: [200e9, 0.15, 0.003125, 0.001125, 80e9, 0.002],
      sectionProfile: rectProfile,
    },
    "opensees/beamIntegration/Lobatto_1": {
      ...baseDecl,
      path: "opensees/beamIntegration/Lobatto_1",
      kind: "beamIntegration",
      type: "Lobatto",
      tag: 1,
      name: "Int1",
      params: [10, 4],
    },
  };
  for (let i = 0; i < 3; i++) {
    const eid = 201 + i;
    decls[`mesh/element/${eid}`] = {
      ...baseDecl,
      path: `mesh/element/${eid}`,
      kind: "element",
      type: "forceBeamColumn",
      tag: eid,
      name: `element ${eid}`,
      params: [1, 1],
      element: {
        femId: eid,
        cell: { block: 0, row: i, alias: "line2" },
        inlineNodes: [i + 1, i + 2],
        metas: [{ type: "forceBeamColumn", tag: eid, args: [1, 1], h5: "/opensees/element_meta/forceBeamColumn", row: i }],
        groups: ["Girder"],
      },
    };
  }

  const ext = buildFrameExtrusion({
    linePositions: new Float32Array([
      0, 0, 4, 2, 0, 4, // seg 0: node 1 -> 2
      2, 0, 4, 4, 0, 4, // seg 1: node 2 -> 3
      4, 0, 4, 6, 0, 4, // seg 2: node 3 -> 4
    ]),
    lineNodes: new Float64Array([1, 2, 2, 3, 3, 4]),
    lineElementIds: new Int32Array([201, 202, 203]),
    lineRefs: [
      { kind: "fem", blockIndex: 0, row: 0 },
      { kind: "fem", blockIndex: 0, row: 1 },
      { kind: "fem", blockIndex: 0, row: 2 },
    ],
    decls,
    ndm: 3,
  });

  // 4 rectangular side faces (8 tris) per segment; end caps (2 tris each) ONLY at outer ends (seg 0 I-end and seg 2 J-end)
  const trisSeg0 = Array.from(ext.frameTriSegments).filter((s) => s === 0).length;
  const trisSeg1 = Array.from(ext.frameTriSegments).filter((s) => s === 1).length;
  const trisSeg2 = Array.from(ext.frameTriSegments).filter((s) => s === 2).length;
  assert.equal(trisSeg0, 10, "outer left segment has 8 side tris + 2 I-cap tris");
  assert.equal(trisSeg1, 8, "interior middle segment omits both internal partition caps (8 side tris only)");
  assert.equal(trisSeg2, 10, "outer right segment has 8 side tris + 2 J-cap tris");

  // Offset links at shared interior nodes 2 and 3 are deduplicated: 4 unique nodes -> 4 offset lines
  assert.equal(ext.offsetLineSegments.length, 4, "4 unique offset links for 4 nodes along the subdivided girder");

  // Verify watertightness at shared interior nodes under deformed displacements
  const col = new Map<number, number>([
    [1, 0],
    [2, 1],
    [3, 2],
    [4, 3],
  ]);
  const rows: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array([0.05, 0.05, 0.05, 0.05]),
    new Float64Array([0.01, 0.02, 0.02, 0.01]),
    new Float64Array([0.0, -0.12, -0.15, -0.02]),
  ];
  const triDisp = vertexDisplacements(ext.frameTriNodes, col, rows);
  const defTri = new Float32Array(ext.frameTriPositions.length);
  for (let i = 0; i < ext.frameTriNodes.length; i++) {
    defTri[3 * i] = ext.frameTriPositions[3 * i]! + 10 * triDisp.d[3 * i]!;
    defTri[3 * i + 1] = ext.frameTriPositions[3 * i + 1]! + 10 * triDisp.d[3 * i + 1]!;
    defTri[3 * i + 2] = ext.frameTriPositions[3 * i + 2]! + 10 * triDisp.d[3 * i + 2]!;
  }

  // Collect all deformed vertex positions tagged with interior node 2 on seg 0 vs seg 1
  const coordsFor = (seg: number, nodeId: number): string[] => {
    const set = new Set<string>();
    for (let t = 0; t < ext.frameTriSegments.length; t++) {
      if (ext.frameTriSegments[t] !== seg) continue;
      for (let v = 0; v < 3; v++) {
        const idx = 3 * t + v;
        if (ext.frameTriNodes[idx] === nodeId) {
          set.add(
            `${defTri[3 * idx]!.toFixed(5)},${defTri[3 * idx + 1]!.toFixed(5)},${defTri[3 * idx + 2]!.toFixed(5)}`,
          );
        }
      }
    }
    return [...set].sort();
  };
  assert.deepEqual(
    coordsFor(0, 2),
    coordsFor(1, 2),
    "seg 0 J-end and seg 1 I-end share identical deformed 3-D cross-section vertices at interior node 2",
  );
  assert.deepEqual(
    coordsFor(1, 3),
    coordsFor(2, 3),
    "seg 1 J-end and seg 2 I-end share identical deformed 3-D cross-section vertices at interior node 3",
  );
});
