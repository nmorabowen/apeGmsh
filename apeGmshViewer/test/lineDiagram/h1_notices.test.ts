// H1 + P0-2 tests:
// - P0-2 element vs. cell orientation guard (forward, reversed swap, contradiction skip + notice)
// - Every buildLineDiagram3D skip reason & zero-beams-drawn notice (3-D frame, orientation mismatch,
//   non-finite/out-of-range station xi/values, zero-length chord, unjoined station rows, deformed reference frame)
// - Unreadable bucket warnings across section.force, section.deformation, basicForce, axialForce, localForce
// - End-to-end Effects isolation when readLineStationStep throws (nodal contour & deformed shape stay ready,
//   lineDiagram.notice surfaces refusal)
// - LineDiagramLayer.sync hides diagram when s.deform.on is true and deformed positions are missing/mismatched
// - Reducer notice on setPhase / resultsLoaded when selected component is absent in the new stage

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import * as THREE from "three";
import { openLadruno } from "../../src/reader/ladruno.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { LineDiagramLayer } from "../../src/renderer/lineDiagram.ts";
import {
  buildLineDiagram3D,
  buildSelectedBeamChainsFromState,
  lineDiagramKeysOf,
} from "../../src/state/lineDiagram.ts";
import { reduce } from "../../src/state/reduce.ts";
import { contourView } from "../../src/state/selectors.ts";
import type { Decl, DeclPath, ResultsInfo, ResultsStage, State } from "../../src/state/types.ts";
import { opened, until } from "../contour/harness.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as ResultsH5;

const FRAME3D_MODEL = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.h5", import.meta.url));
const FRAME3D_LADRUNO = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.ladruno", import.meta.url));
const LOBATTO_MODEL = fileURLToPath(new URL("../../fixtures/line_oracle_fbc_lobatto.h5", import.meta.url));
const LOBATTO_LADRUNO = fileURLToPath(new URL("../../fixtures/line_oracle_fbc_lobatto.ladruno", import.meta.url));

const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const closeVec = (a: readonly number[], b: readonly number[], tol = 1e-9) =>
  a.length === b.length && a.every((v, i) => close(v, b[i]!, tol));

function makeSceneWithNodes(inlineNodes1: number[], inlineNodes2: number[]) {
  const transfDecl: Decl = {
    path: "opensees/transforms/Tr1",
    kind: "geomTransf",
    type: "LinearTransform",
    name: "Tr1",
    nameSource: "/opensees/names",
    h5: "/opensees/transforms/Tr1",
    tag: 1,
    params: [0, 0, 1],
    fields: [],
    refs: {},
    links: {},
    problems: [],
  };
  const e1: Decl = {
    path: "mesh/element/1",
    kind: "element",
    type: "ElasticBeamColumn3d",
    name: "Beam1",
    nameSource: "/elements/line2",
    h5: "/elements/line2",
    tag: 10,
    params: [],
    fields: [],
    refs: { transfTag: "opensees/transforms/Tr1" },
    links: {},
    problems: [],
    element: {
      femId: 1,
      cell: { alias: "line2", block: 0, row: 0 },
      inlineNodes: inlineNodes1,
      groups: [],
      metas: [{ type: "ElasticBeamColumn3d", tag: 10, h5: "/opensees/element_meta/ElasticBeamColumn3d", row: 0, args: [] }],
    },
  };
  const e2: Decl = {
    path: "mesh/element/2",
    kind: "element",
    type: "ElasticBeamColumn3d",
    name: "Beam2",
    nameSource: "/elements/line2",
    h5: "/elements/line2",
    tag: 20,
    params: [],
    fields: [],
    refs: { transfTag: "opensees/transforms/Tr1" },
    links: {},
    problems: [],
    element: {
      femId: 2,
      cell: { alias: "line2", block: 0, row: 1 },
      inlineNodes: inlineNodes2,
      groups: [],
      metas: [{ type: "ElasticBeamColumn3d", tag: 20, h5: "/opensees/element_meta/ElasticBeamColumn3d", row: 1, args: [] }],
    },
  };
  return {
    decls: {
      "opensees/transforms/Tr1": transfDecl,
      "mesh/element/1": e1,
      "mesh/element/2": e2,
    } as Record<DeclPath, Decl>,
    meshElements: ["mesh/element/1", "mesh/element/2"] as DeclPath[],
    lineElement: new Int32Array([0, 1]),
    // Cell 0 drawn from node 1 (0,0,0) -> node 2 (2,0,0); Cell 1 drawn from node 2 (2,0,0) -> node 3 (4,0,0)
    drawnLinePositions: new Float32Array([0, 0, 0, 2, 0, 0, 2, 0, 0, 4, 0, 0]),
    lineNodes: new Float64Array([1, 2, 2, 3]),
    join: {
      ok: true as const,
      space: "ops_tag" as const,
      byLadrunoId: new Map<number, DeclPath>([
        [10, "mesh/element/1"],
        [20, "mesh/element/2"],
      ]),
      byDeclPath: new Map<DeclPath, number>([
        ["mesh/element/1", 10],
        ["mesh/element/2", 20],
      ]),
    },
  };
}

// -- 1. P0-2: element vs. cell orientation guard -------------------------------

test("P0-2: buildLineDiagram3D checks element connectivity (iNode, jNode) against cell corners (nStart, nEnd), flips reversed chords, and refuses contradicting connectivity", () => {
  // Element 1 has forward connectivity [1, 2] matching cell [1, 2].
  // Element 2 has REVERSED OpenSees connectivity [3, 2] while its mesh cell is [2, 3]!
  const sceneRev = makeSceneWithNodes([1, 2], [3, 2]);
  const geomRev = buildLineDiagram3D({
    ...sceneRev,
    stepData: {
      component: "N",
      elementIndex: new Float64Array([10, 10, 20, 20]),
      stationNaturalCoord: new Float64Array([-1, 1, -1, 1]),
      values: new Float64Array([10, 20, 100, 200]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });

  assert.equal(geomRev.skippedOrientationMismatchBeams, 0);
  assert.equal(geomRev.notice, null);
  assert.equal(geomRev.beams.length, 2);

  // Beam 1 (forward [1, 2]): pStart = (0,0,0), pEnd = (2,0,0), xLocal = (+1,0,0), nodes 1 -> 2
  const b1 = geomRev.beams[0]!;
  assert.ok(closeVec(b1.pStart, [0, 0, 0]));
  assert.ok(closeVec(b1.pEnd, [2, 0, 0]));
  assert.ok(closeVec(b1.xLocal, [1, 0, 0]));
  assert.equal(b1.nodeStart, 1);
  assert.equal(b1.nodeEnd, 2);

  // Beam 2 (reversed [3, 2] vs cell [2, 3]): pStart flipped to (4,0,0) (node 3 = iNode) and pEnd to (2,0,0) (node 2 = jNode), xLocal = (-1,0,0)!
  const b2 = geomRev.beams[1]!;
  assert.ok(closeVec(b2.pStart, [4, 0, 0]));
  assert.ok(closeVec(b2.pEnd, [2, 0, 0]));
  assert.ok(closeVec(b2.xLocal, [-1, 0, 0]));
  assert.equal(b2.nodeStart, 3);
  assert.equal(b2.nodeEnd, 2);
  // Station 0 (xi=-1, value=100) riser base sits at pStart = (4,0,0), station 1 (xi=+1, value=200) sits at pEnd = (2,0,0):
  // Beam 2 risers start at offset 2 stations * 6 floats = 12 floats:
  assert.ok(closeVec(Array.from(geomRev.riserPositions.slice(12, 15)), [4, 0, 0]));
  assert.ok(closeVec(Array.from(geomRev.riserPositions.slice(18, 21)), [2, 0, 0]));

  // buildSelectedBeamChainsFromState uses the oriented (nodeStart, nodeEnd) from geomRev.beams
  // so the chain walks node 1 -> node 2 -> node 3 and reverses Beam 2's stations [100, 200] to [200, 100]:
  const fakeState = {
    selection: { decls: ["mesh/element/1", "mesh/element/2"] as DeclPath[] },
  };
  const chains = buildSelectedBeamChainsFromState(
    fakeState as Pick<State, "selection">,
    geomRev,
    sceneRev.lineNodes,
  );
  assert.equal(chains.length, 1);
  assert.deepEqual(chains[0]!.joints.map((j) => j.nodeId), [1, 2, 3]);
  assert.equal(chains[0]!.spans[0]!.reversed, false);
  assert.equal(chains[0]!.spans[1]!.reversed, true);
  assert.deepEqual(chains[0]!.spans[1]!.stations.map((st) => st.value), [200, 100]);

  // Contradiction: Element 2 has inlineNodes [2, 99] while cell has [2, 3] -> skipped with notice!
  const sceneBad = makeSceneWithNodes([1, 2], [2, 99]);
  const geomBad = buildLineDiagram3D({
    ...sceneBad,
    stepData: {
      component: "N",
      elementIndex: new Float64Array([10, 10, 20, 20]),
      stationNaturalCoord: new Float64Array([-1, 1, -1, 1]),
      values: new Float64Array([10, 20, 100, 200]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomBad.skippedOrientationMismatchBeams, 1);
  assert.equal(geomBad.beams.length, 1);
  assert.match(geomBad.notice ?? "", /1 beam skipped: element connectivity \(iNode, jNode\) contradicts mesh cell corner nodes\./);

  // Explicit elementNodesById map also triggers P0-2 reversal or contradiction when inlineNodes is empty:
  const sceneMap = makeSceneWithNodes([], []);
  const geomMapRev = buildLineDiagram3D({
    ...sceneMap,
    elementNodesById: new Map([
      [10, [2, 1] as const],
      [20, [2, 99] as const],
    ]),
    stepData: {
      component: "N",
      elementIndex: new Float64Array([10, 10, 20, 20]),
      stationNaturalCoord: new Float64Array([-1, 1, -1, 1]),
      values: new Float64Array([10, 20, 100, 200]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomMapRev.beams.length, 1);
  assert.ok(closeVec(geomMapRev.beams[0]!.pStart, [2, 0, 0]));
  assert.ok(closeVec(geomMapRev.beams[0]!.pEnd, [0, 0, 0]));
  assert.equal(geomMapRev.skippedOrientationMismatchBeams, 1);
});

// -- 2. H1: Every skip reason & zero-beams-drawn notice in buildLineDiagram3D --

test("H1: buildLineDiagram3D reports zero-beams-drawn, non-finite/out-of-range stations, zero-length chords, unjoined station rows, and deformed reference frame", () => {
  const scene = makeSceneWithNodes([1, 2], [2, 3]);

  // 1. Zero beams drawn when stepData has no matching element rows:
  const geomEmpty = buildLineDiagram3D({
    ...scene,
    stepData: {
      component: "N",
      elementIndex: new Float64Array([]),
      stationNaturalCoord: new Float64Array([]),
      values: new Float64Array([]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomEmpty.beams.length, 0);
  assert.equal(geomEmpty.notice, "No line-station row joins a drawn 1-D beam.");

  // 2. Unjoined station rows (element 999 not in 1-D mesh):
  //    a) Partial unjoined (2 rows for beam 10 + 2 rows for unknown element 999):
  const geomPartialUnjoined = buildLineDiagram3D({
    ...scene,
    stepData: {
      component: "N",
      elementIndex: new Float64Array([10, 10, 999, 999]),
      stationNaturalCoord: new Float64Array([-1, 1, -1, 1]),
      values: new Float64Array([10, 20, 30, 40]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomPartialUnjoined.beams.length, 1);
  assert.equal(
    geomPartialUnjoined.notice,
    "2 of 4 station rows unjoined (no matching 1-D beam in mesh).",
  );

  //    b) Total unjoined (all rows belong to unknown element 999):
  const geomAllUnjoined = buildLineDiagram3D({
    ...scene,
    stepData: {
      component: "N",
      elementIndex: new Float64Array([999, 999]),
      stationNaturalCoord: new Float64Array([-1, 1]),
      values: new Float64Array([30, 40]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomAllUnjoined.beams.length, 0);
  assert.equal(
    geomAllUnjoined.notice,
    "No line-station row joins a drawn 1-D beam (2 of 2 station rows unjoined).",
  );

  // 3. Non-finite station value (NaN) or out-of-range xi (xi = 1.5 > 1):
  const geomBadStations = buildLineDiagram3D({
    ...scene,
    stepData: {
      component: "N",
      elementIndex: new Float64Array([10, 10, 20, 20]),
      stationNaturalCoord: new Float64Array([-1, 1.5, -1, 1]),
      values: new Float64Array([10, 20, 30, Number.NaN]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomBadStations.beams.length, 0);
  assert.match(
    geomBadStations.notice ?? "",
    /2 beams skipped: non-finite station value or natural coordinate xi outside \[-1, 1\]\./,
  );

  // 4. Zero-length chord:
  const geomZeroChord = buildLineDiagram3D({
    ...scene,
    // Beam 1 has coincident endpoints (0,0,0) -> (0,0,0); Beam 2 has (2,0,0) -> (4,0,0)
    drawnLinePositions: new Float32Array([0, 0, 0, 0, 0, 0, 2, 0, 0, 4, 0, 0]),
    stepData: {
      component: "N",
      elementIndex: new Float64Array([10, 10, 20, 20]),
      stationNaturalCoord: new Float64Array([-1, 1, -1, 1]),
      values: new Float64Array([10, 20, 30, 40]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomZeroChord.beams.length, 1);
  assert.match(geomZeroChord.notice ?? "", /1 beam skipped: zero or non-finite chord length\./);

  // 5. Deformed baseline with reference local frame vs per-step LOCAL_AXES:
  const geomDefRef = buildLineDiagram3D({
    ...scene,
    deformed: true,
    stepData: {
      component: "Mz",
      elementIndex: new Float64Array([10, 10]),
      stationNaturalCoord: new Float64Array([-1, 1]),
      values: new Float64Array([10, 20]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: null,
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomDefRef.beams.length, 1);
  assert.equal(geomDefRef.beams[0]!.zRecSource, "model_transform");
  assert.match(
    geomDefRef.notice ?? "",
    /Deformed baseline uses reference local frame \(no per-step RESULTS\/ON_ELEMENTS\/LOCAL_AXES recorded\)\./,
  );

  // When step_local_axes is present, no reference-frame notice is emitted on deformed baseline:
  const geomDefStep = buildLineDiagram3D({
    ...scene,
    deformed: true,
    stepData: {
      component: "Mz",
      elementIndex: new Float64Array([10, 10]),
      stationNaturalCoord: new Float64Array([-1, 1]),
      values: new Float64Array([10, 20]),
      time: 1.0,
      localAxesQuaternion: null,
      localAxesZRec: new Float64Array([0, 0, 1, 0, 0, 1]),
      localAxesSources: ["step_local_axes", "step_local_axes"],
    },
    ndm: 3,
    scale: 1,
  });
  assert.equal(geomDefStep.beams.length, 1);
  assert.equal(geomDefStep.notice, null);
});

// -- 3. H1: Unreadable bucket warnings across all 5 line-station tokens --------

test("H1: availableLineStations and readLineStations emit unreadable bucket warnings for section.force, section.deformation, basicForce, axialForce, and localForce", () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-h1-buckets-"));
  const p = join(dir, "corrupt_buckets.ladruno");
  copyFileSync(LOBATTO_LADRUNO, p);

  const f = new h5wasm.File(p, "a");
  try {
    const onElems = f.get("MODEL_STAGE[1]/RESULTS/ON_ELEMENTS") as h5wasm.Group;
    // Add one unreadable bucket (missing DATA or COLUMN_MAP) under each of the 5 tokens:
    for (const token of ["section.force", "section.deformation", "basicForce", "axialForce", "localForce"]) {
      const tg = onElems.keys().includes(token)
        ? (onElems.get(token) as h5wasm.Group)
        : onElems.create_group(token);
      const bad = tg.create_group(`99-CorruptBucket_${token.replace(".", "_")}`);
      bad.create_dataset({ name: "ID", data: new Int32Array([99]), shape: [1] });
      // Intentionally omit DATA and COLUMN_MAP so the bucket is unreadable!
    }
  } finally {
    f.close();
  }

  const r = openLadruno(H5, p);
  try {
    const stageId = r.index.stages[0]!.id;
    const openWarnings: string[] = [];
    const avail = r.availableLineStations(stageId, openWarnings);
    assert.ok(avail.includes("Mz"), "valid buckets still expose Mz");

    for (const token of ["section.force", "section.deformation", "basicForce", "axialForce", "localForce"]) {
      const bucketName = `99-CorruptBucket_${token.replace(".", "_")}`;
      assert.ok(
        openWarnings.some((w) => w.includes(`${token} bucket "${bucketName}"`) && w.includes("is unreadable:")),
        `expected availableLineStations warning for ${token} bucket "${bucketName}", got: ${openWarnings.join(" | ")}`,
      );
    }

    const readWarnings: string[] = [];
    const slab = r.readLineStations(stageId, "Mz", { warningsOut: readWarnings });
    assert.ok(slab.nStationsTotal > 0);
    // Pass 1 checks section.force (for Mz) and Pass 2 checks localForce & basicForce (for Mz):
    for (const token of ["section.force", "basicForce", "localForce"]) {
      const bucketName = `99-CorruptBucket_${token.replace(".", "_")}`;
      assert.ok(
        readWarnings.some((w) => w.includes(`${token} bucket "${bucketName}"`) && w.includes("is unreadable:")),
        `expected readLineStations(Mz) warning for ${token} bucket "${bucketName}", got: ${readWarnings.join(" | ")}`,
      );
    }

    const epsWarnings: string[] = [];
    const epsSlab = r.readLineStations(stageId, "eps", { warningsOut: epsWarnings });
    assert.equal(epsSlab.nStationsTotal, 0);
    assert.ok(
      epsWarnings.some((w) => w.includes(`section.deformation bucket "99-CorruptBucket_section_deformation"`)),
      "readLineStations(eps) warns on unreadable section.deformation bucket",
    );

    const nWarnings: string[] = [];
    r.readLineStations(stageId, "N", { warningsOut: nWarnings });
    assert.ok(
      nWarnings.some((w) => w.includes(`axialForce bucket "99-CorruptBucket_axialForce"`)),
      "readLineStations(N) warns on unreadable axialForce bucket",
    );
  } finally {
    r.close();
  }
});

// -- 4. H1: Effects isolates step line-diagram refusal from nodal contour/deform --

test("H1: Effects keeps nodal contour and deformed shape ready when readLineStationStep throws, surfacing lineDiagramError on s.lineDiagram.notice", async () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-h1-effects-"));
  const badLadruno = join(dir, "bad_lobatto.ladruno");
  copyFileSync(LOBATTO_LADRUNO, badLadruno);
  const f = new h5wasm.File(badLadruno, "a");
  try {
    const onElems = f.get("MODEL_STAGE[1]/RESULTS/ON_ELEMENTS") as h5wasm.Group;
    const stepLa = onElems.create_group("LOCAL_AXES").create_group("73-ForceBeamColumn2d[1000:1:0]");
    stepLa.create_dataset({ name: "ID", data: new Int32Array([2]), shape: [1] });
    // Malformed step LOCAL_AXES (all-zero quaternion row at step 1) throws in readLineStationStep(step=1)
    stepLa.create_dataset({
      name: "DATA",
      data: new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]),
      shape: [2, 1, 4],
    });
  } finally {
    f.close();
  }

  const h = await opened(LOBATTO_MODEL, badLadruno);
  try {
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 1 } });
    h.store.dispatch({ type: "setColourBy", by: "contour" });
    h.store.dispatch({ type: "setDeform", on: true });
    h.store.dispatch({ type: "setLineDiagram", on: true });

    await until(() => {
      const cur = h.store.get();
      return (
        contourView(cur).state === "ready" &&
        h.effects.currentDisplacement() !== null &&
        cur.lineDiagram.notice !== null
      );
    }, "nodal contour + deform ready and lineDiagram.notice populated with refusal");

    const s = h.store.get();
    assert.equal(contourView(s).state, "ready", "nodal contour succeeds despite line-diagram step refusal");
    assert.ok(h.effects.currentDisplacement() !== null, "deformed shape succeeds despite line-diagram step refusal");
    assert.equal(h.effects.currentLineDiagram(), null, "currentLineDiagram is null when step read failed");
    assert.equal(s.lineDiagram.stepRange?.min, null);
    assert.equal(s.lineDiagram.stepRange?.max, null);
    assert.match(s.lineDiagram.notice ?? "", /holds at step 1 a DATA row that is not a frame/);
  } finally {
    h.effects.dispose();
  }
});

// -- 5. H1: LineDiagramLayer hides when s.deform.on is true and deform is missing/mismatched --

test("H1: LineDiagramLayer.sync hides the diagram when s.deform.on is true and deformed positions are missing, stale, or wrong length", async () => {
  const h = await opened(FRAME3D_MODEL, FRAME3D_LADRUNO);
  try {
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 2 } });
    h.store.dispatch({ type: "setLineDiagram", on: true });

    await until(() => {
      const cur = h.store.get();
      const ldKeys = lineDiagramKeysOf(cur);
      return ldKeys !== null && h.effects.currentLineDiagram()?.key === ldKeys.step;
    }, "lineDiagram step loaded");

    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
    camera.position.set(4, -12, 3);
    camera.lookAt(4, 0, 3);
    camera.updateMatrixWorld();

    const layer = new LineDiagramLayer(camera);
    const stepData = h.effects.currentLineDiagram()!;

    // Undeformed: layer is visible
    assert.equal(layer.sync(h.store.get(), h.blobs, stepData, null), true);
    assert.equal(layer.root.visible, true);

    // Enable s.deform.on:
    h.store.dispatch({ type: "setDeform", on: true });
    await until(() => h.effects.currentDisplacement() !== null, "displacement ready");
    const sDef = h.store.get();
    const validDisp = h.effects.currentDisplacement()!;

    // 1. Passing null deform while s.deform.on is true hides the layer:
    assert.equal(layer.sync(sDef, h.blobs, stepData, null), true);
    assert.equal(layer.root.visible, false);

    // 2. Passing valid deform shows the layer:
    assert.equal(layer.sync(sDef, h.blobs, stepData, validDisp), true);
    assert.equal(layer.root.visible, true);

    // 3. Passing stale key hides the layer:
    const rawLineLen = h.blobs.f32(sDef.mesh!.linePositions).length;
    assert.equal(
      layer.sync(sDef, h.blobs, stepData, { key: "stale-key", line: new Float32Array(rawLineLen) }),
      true,
    );
    assert.equal(layer.root.visible, false);

    // 4. Passing wrong-length array hides the layer:
    assert.equal(layer.sync(sDef, h.blobs, stepData, validDisp), true);
    assert.equal(layer.root.visible, true);
    assert.equal(
      layer.sync(sDef, h.blobs, stepData, { dispKey: validDisp.key, line: new Float32Array(3) }),
      true,
    );
    assert.equal(layer.root.visible, false);

    layer.dispose();
  } finally {
    h.effects.dispose();
  }
});

// -- 6. H1: Reducer emits notice when selected component is absent in new stage --

test("H1: reduce emits a notice on setPhase and resultsLoaded when a selected component is absent in the stage's lineStations", async () => {
  const h = await opened(FRAME3D_MODEL, FRAME3D_LADRUNO);
  const baseState = h.store.get();
  h.effects.dispose();

  const stage0: ResultsStage = {
    id: "MODEL_STAGE[1]",
    name: "stage_0",
    kind: "static",
    time: [0, 1],
    nodal: [],
    refusedNodal: [],
    gauss: [],
    unread: [],
    elementIndexSpace: "ops_tag",
    lineStations: ["N", "Mz", "kappaZ"],
  };
  const stage1: ResultsStage = {
    id: "MODEL_STAGE[2]",
    name: "stage_1",
    kind: "static",
    time: [0, 1],
    nodal: [],
    refusedNodal: [],
    gauss: [],
    unread: [],
    elementIndexSpace: "ops_tag",
    lineStations: ["N", "Vy"],
  };
  const stage2Empty: ResultsStage = {
    id: "MODEL_STAGE[3]",
    name: "stage_2",
    kind: "static",
    time: [0, 1],
    nodal: [],
    refusedNodal: [],
    gauss: [],
    unread: [],
    elementIndexSpace: "ops_tag",
    lineStations: [],
  };
  const results: ResultsInfo = {
    ...baseState.results!,
    stages: [stage0, stage1, stage2Empty],
  };

  let s = reduce(baseState, { type: "fileOpened", artifact: "results", path: results.path });
  s = reduce(s, {
    type: "resultsLoaded",
    info: results,
    warnings: [],
    sizeBytes: 100,
    readMs: 1,
  });
  s = reduce(s, { type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  s = reduce(s, { type: "setLineDiagram", on: true });
  s = reduce(s, { type: "setLineDiagramComponent", component: "kappaZ" });
  assert.equal(s.lineDiagram.component, "kappaZ");
  assert.equal(s.lineDiagram.notice, null);

  // 1. Switching to stage 1 (which only has ["N", "Vy"], lacking "kappaZ") auto-switches to "Vy" and sets notice:
  s = reduce(s, { type: "setPhase", at: { kind: "results", stage: 1, step: 0 } });
  assert.equal(s.lineDiagram.component, "Vy");
  assert.equal(
    s.lineDiagram.notice,
    "Stage stage_1 has no kappaZ line stations; switched to Vy.",
  );

  // 2. Switching to stage 2 (which has no lineStations at all) sets component to null and emits notice:
  s = reduce(s, { type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  assert.equal(s.lineDiagram.component, null);
  assert.equal(
    s.lineDiagram.notice,
    "Stage stage_2 has no Vy line stations; the line-diagram pick was cleared.",
  );

  // 3. Switching back to mesh phase clears notice:
  s = reduce(s, { type: "setPhase", at: { kind: "mesh" } });
  assert.equal(s.lineDiagram.notice, null);

  // 4. Reloading results when a component ("Mz") was selected on stage 0, but the new file's stage 0 only has ["N"]:
  s = reduce(s, { type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  assert.equal(s.lineDiagram.component, "Mz");
  const resultsOnlyN: ResultsInfo = {
    ...results,
    stages: [{ ...stage0, lineStations: ["N"] }],
  };
  s = reduce(s, { type: "fileOpened", artifact: "results", path: results.path });
  s = reduce(s, {
    type: "resultsLoaded",
    info: resultsOnlyN,
    warnings: [],
    sizeBytes: 100,
    readMs: 1,
  });
  assert.equal(s.lineDiagram.component, "N");
  assert.equal(
    s.lineDiagram.notice,
    "ladruno_frame3d.ladruno has no Mz line stations; switched to N.",
  );
});
