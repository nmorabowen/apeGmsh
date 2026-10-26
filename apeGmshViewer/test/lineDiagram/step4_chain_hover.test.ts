// 2-D multi-beam chain plot and 3-D/2-D hover readout tests:
// - Selection-driven `buildSelectedBeamChainsFromState` & `buildBeamChains`
//   (single span, multi-span colinear merge, reversed span orientation,
//   disjoint chains, T-junction colinear tie-break, empty/non-beam selection)
// - Gauss-interior station handling (`xiMin`, `xiMax`, `startEndKnown`,
//   `endEndKnown`, unextrapolated polyline/polygon, dashed unknown-end gaps)
// - `projectHoverNeedle`, `computeLineDiagramHover`, `pickChainPlotAtS`,
//   `formatHoverReadout`, and `withinStationSpan`
// - End-to-end `Effects` + `reduce` + `LineDiagramLayer.hoverNeedle` on
//   `fixtures/ladruno_frame3d.{h5,ladruno}`

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { LineDiagramLayer } from "../../src/renderer/lineDiagram.ts";
import {
  buildBeamChainPlotModel,
  buildBeamChains,
  buildSelectedBeamChainsFromState,
  computeLineDiagramHover,
  formatHoverReadout,
  lineDiagramKeysOf,
  pickChainPlotAtS,
  projectHoverNeedle,
} from "../../src/state/selectors.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import type { LineDiagramBeamSummary, State } from "../../src/state/types.ts";
import { opened, until } from "../contour/harness.ts";

const FRAME3D_MODEL = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.h5", import.meta.url));
const FRAME3D_LADRUNO = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.ladruno", import.meta.url));

test("buildSelectedBeamChainsFromState returns [] when selection is empty or selects only non-beam declarations", () => {
  const beams: LineDiagramBeamSummary[] = [
    {
      decl: "opensees/element/elasticBeamColumn/1",
      elementId: 1,
      nodeStart: 1,
      nodeEnd: 2,
      pStart: [0, 0, 0],
      pEnd: [4, 0, 0],
      length: 4,
      xLocal: [1, 0, 0],
      extrudeDir: [0, 0, 1],
      xi: [-1, 0, 1],
      values: [-10, 5, -10],
      autoScale: 0.05,
    },
  ];

  assert.deepEqual(
    buildSelectedBeamChainsFromState({ selection: { decls: [], picks: [] } }, { beams }),
    [],
  );
  assert.deepEqual(
    buildSelectedBeamChainsFromState(
      { selection: { decls: ["opensees/element/ShellMITC4/100"], picks: [] } },
      { beams },
    ),
    [],
  );
});

test("buildBeamChains merges connected spans, handles reversed span orientation, splits disjoint chains, and breaks T-junction ties by colinearity", () => {
  // Two connected spans where the second span is oriented reversed (node 3 -> node 2):
  const chains = buildBeamChains([
    {
      decl: "opensees/element/beam/1",
      elementId: 1,
      nodeStart: 1,
      nodeEnd: 2,
      length: 4,
      xi: [-1, 0, 1],
      values: [10, 20, 30],
      pStart: [0, 0, 0],
      pEnd: [4, 0, 0],
      xLocal: [1, 0, 0],
    },
    {
      decl: "opensees/element/beam/2",
      elementId: 2,
      nodeStart: 3,
      nodeEnd: 2,
      length: 6,
      xi: [-1, 1],
      values: [100, 40],
      pStart: [10, 0, 0],
      pEnd: [4, 0, 0],
      xLocal: [-1, 0, 0],
    },
  ]);

  assert.equal(chains.length, 1);
  const chain = chains[0]!;
  assert.equal(chain.contiguous, true);
  assert.equal(chain.totalLength, 10);
  assert.deepEqual(
    chain.joints.map((j) => ({ nodeId: j.nodeId, x: j.x })),
    [
      { nodeId: 1, x: 0 },
      { nodeId: 2, x: 4 },
      { nodeId: 3, x: 10 },
    ],
  );
  assert.equal(chain.spans.length, 2);
  assert.equal(chain.spans[0]!.reversed, false);
  assert.equal(chain.spans[1]!.reversed, true);
  assert.equal(chain.spans[1]!.nodeStart, 2);
  assert.equal(chain.spans[1]!.nodeEnd, 3);
  // Because span 2 was reversed (3 -> 2 walked as 2 -> 3), xi = +1 (value 40) is at x = 4 and xi = -1 (value 100) is at x = 10:
  assert.deepEqual(
    chain.spans[1]!.stations.map((st) => ({ x: st.x, xi: st.xi, value: st.value })),
    [
      { x: 4, xi: 1, value: 40 },
      { x: 10, xi: -1, value: 100 },
    ],
  );

  // Disjoint spans produce separate chains:
  const disjoint = buildBeamChains([
    {
      decl: "opensees/element/beam/1",
      elementId: 1,
      nodeStart: 1,
      nodeEnd: 2,
      length: 3,
      xi: [-1, 1],
      values: [0, 10],
    },
    {
      decl: "opensees/element/beam/9",
      elementId: 9,
      nodeStart: 10,
      nodeEnd: 11,
      length: 5,
      xi: [-1, 1],
      values: [-5, 5],
    },
  ]);
  assert.equal(disjoint.length, 2);
  assert.equal(disjoint[0]!.contiguous, false);
  assert.equal(disjoint[1]!.contiguous, false);

  // T-junction at node 2: girder E1 (1->2 along +X), column E2 (2->5 along +Z), girder E3 (2->3 along +X):
  const tJunction = buildBeamChains([
    {
      decl: "opensees/element/beam/1",
      elementId: 1,
      nodeStart: 1,
      nodeEnd: 2,
      length: 4,
      xi: [-1, 1],
      values: [0, 10],
      xLocal: [1, 0, 0],
    },
    {
      decl: "opensees/element/beam/2",
      elementId: 2,
      nodeStart: 2,
      nodeEnd: 5,
      length: 3,
      xi: [-1, 1],
      values: [5, 5],
      xLocal: [0, 0, 1],
    },
    {
      decl: "opensees/element/beam/3",
      elementId: 3,
      nodeStart: 2,
      nodeEnd: 3,
      length: 4,
      xi: [-1, 1],
      values: [10, 0],
      xLocal: [1, 0, 0],
    },
  ]);
  // First two spans in the walked chain must be the colinear girders E1 -> E3:
  assert.equal(tJunction[0]!.spans[0]!.elementId, 1);
  assert.equal(tJunction[0]!.spans[1]!.elementId, 3);
});

test("Gauss-interior stations are never extrapolated to element ends in buildBeamChains or buildBeamChainPlotModel", () => {
  const g = 1 / Math.sqrt(3);
  const chains = buildBeamChains([
    {
      decl: "opensees/element/beam/7",
      elementId: 7,
      nodeStart: 10,
      nodeEnd: 20,
      length: 6,
      xi: [-g, g],
      values: [-18, 12],
    },
  ]);
  assert.equal(chains.length, 1);
  const span = chains[0]!.spans[0]!;
  assert.equal(span.startEndKnown, false);
  assert.equal(span.endEndKnown, false);
  assert.ok(Math.abs(span.xiMin - -g) < 1e-6);
  assert.ok(Math.abs(span.xiMax - g) < 1e-6);
  // Stations remain at their true Gauss positions inside (0, 6), never clamped to 0 or 6:
  assert.ok(span.stations[0]!.x > 1.2 && span.stations[0]!.x < 1.3);
  assert.ok(span.stations[1]!.x > 4.7 && span.stations[1]!.x < 4.8);

  const plot = buildBeamChainPlotModel(chains[0]!, 0, 1, {
    flipSign: false,
    fill: true,
    component: "Mz",
  });
  assert.equal(plot.hasUnknownEnds, true);
  assert.equal(plot.unknownEndBadge, "ends unrecorded (Gauss stations)");
  const spPlot = plot.spans[0]!;
  assert.equal(spPlot.startEndKnown, false);
  assert.equal(spPlot.endEndKnown, false);
  assert.ok(spPlot.measuredX0 > spPlot.x0, "polyline/polygon starts at first Gauss station, not at joint x0");
  assert.ok(spPlot.measuredX1 < spPlot.x1, "polyline/polygon ends at last Gauss station, not at joint x1");
  assert.ok(spPlot.unknownStartGap !== null);
  assert.ok(spPlot.unknownEndGap !== null);
  assert.equal(spPlot.unknownStartGap.x0, spPlot.x0);
  assert.equal(spPlot.unknownStartGap.x1, spPlot.measuredX0);
  assert.equal(spPlot.unknownEndGap.x0, spPlot.measuredX1);
  assert.equal(spPlot.unknownEndGap.x1, spPlot.x1);

  // Flipping sign mirrors station y-coordinates across zeroY:
  const plotFlipped = buildBeamChainPlotModel(chains[0]!, 0, 1, {
    flipSign: true,
    fill: true,
    component: "Mz",
  });
  const dyUnflipped = spPlot.stationDots[0]!.y - plot.zeroY;
  const dyFlipped = plotFlipped.spans[0]!.stationDots[0]!.y - plotFlipped.zeroY;
  assert.ok(Math.abs(dyUnflipped + dyFlipped) < 1e-2);
});

test("projectHoverNeedle, computeLineDiagramHover, and pickChainPlotAtS project 3-D and 2-D hover and flag withinStationSpan", () => {
  const g = 0.5773502691896257;
  const beam: LineDiagramBeamSummary = {
    decl: "opensees/element/beam/5",
    elementId: 5,
    nodeStart: 1,
    nodeEnd: 2,
    pStart: [0, 0, 0],
    pEnd: [4, 0, 0],
    length: 4,
    xLocal: [1, 0, 0],
    extrudeDir: [0, 0, 1],
    xi: [-g, g],
    values: [-20, 20],
    autoScale: 0.05,
  };

  // Midspan hover (xi = 0, inside [-g, +g]):
  const mid = projectHoverNeedle(beam, [2, 0.5, -0.2], 0.05, true);
  assert.ok(Math.abs(mid.xRel - 0.5) < 1e-9);
  assert.ok(Math.abs(mid.xi) < 1e-9);
  assert.ok(Math.abs(mid.value) < 1e-9);
  assert.equal(mid.withinStationSpan, true);

  // Near start end (x = 0.1 -> xi = -0.95, outside [-g, +g]):
  const endHover = projectHoverNeedle(beam, [0.1, 0, 0], 0.05, false);
  assert.ok(endHover.xi < -g);
  assert.equal(endHover.withinStationSpan, false);
  assert.equal(endHover.value, -20);

  let s: State = {
    ...initialState,
    decls: {
      [beam.decl]: {
        path: beam.decl,
        kind: "element",
        type: "elasticBeamColumn",
        name: "B5",
        nameSource: "/opensees/names",
        h5: "/elements/line2",
        tag: 5,
        params: [],
        fields: [],
        refs: {},
        links: {},
        problems: [],
      },
    },
    lineDiagram: {
      ...initialState.lineDiagram,
      on: true,
      component: "Mz",
      drawnBeams: [beam],
    },
  };
  s = reduce(s, { type: "select", pick: { decl: beam.decl, at: [2, 0, 0], nodes: [1, 2] } });
  assert.equal(s.lineDiagram.chains.length, 1);

  // Pick from 2-D plot at s = 3.0 (xi = +0.5):
  const pick2D = pickChainPlotAtS(s.lineDiagram.chains[0]!, 3.0, s.lineDiagram.drawnBeams);
  assert.ok(pick2D !== null);
  s = reduce(s, { type: "hover", at: pick2D });
  assert.ok(s.lineDiagram.hover !== null);
  assert.equal(s.lineDiagram.hover.elementId, 5);
  assert.equal(s.lineDiagram.hover.chainIndex, 0);
  assert.ok(Math.abs(s.lineDiagram.hover.s - 3.0) < 1e-6);
  assert.ok(Math.abs(s.lineDiagram.hover.xi - 0.5) < 1e-6);
  assert.equal(s.lineDiagram.hover.withinStationSpan, true);
  assert.match(formatHoverReadout(s.lineDiagram.hover, "Mz"), /E5 · ξ = 0\.500 · s = 3 · Mz \(bending_moment_z\)/);

  const roClamped = computeLineDiagramHover(
    {
      hover: { decl: beam.decl, at: [0.05, 0, 0], nodes: [1, 2] },
      lineDiagram: s.lineDiagram,
    },
    [beam],
  );
  assert.ok(roClamped !== null);
  assert.equal(roClamped.withinStationSpan, false);
  assert.match(formatHoverReadout(roClamped, "Mz"), /\[clamped to station end\]/);
});

test("ladruno_frame3d end-to-end: selecting connected girders builds continuous 2-span chain and hovering drives LineDiagramLayer.hoverNeedle", async () => {
  const h = await opened(FRAME3D_MODEL, FRAME3D_LADRUNO);
  try {
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 2 } });
    h.store.dispatch({ type: "setLineDiagram", on: true });

    await until(() => {
      const cur = h.store.get();
      const keys = lineDiagramKeysOf(cur);
      return (
        keys !== null &&
        h.effects.currentLineDiagram()?.key === keys.step &&
        cur.lineDiagram.stepRange?.key === keys.step &&
        cur.lineDiagram.drawnBeams.length === 10
      );
    }, "ladruno_frame3d drawnBeams populated");

    let s = h.store.get();
    assert.equal(s.lineDiagram.drawnBeams.length, 10);
    assert.equal(s.lineDiagram.chains.length, 0, "no chain when nothing is selected");

    // Find two connected beams sharing a node:
    const b0 = s.lineDiagram.drawnBeams[0]!;
    const connected = s.lineDiagram.drawnBeams.find(
      (b) =>
        b.decl !== b0.decl &&
        (b.nodeStart === b0.nodeEnd ||
          b.nodeEnd === b0.nodeEnd ||
          b.nodeStart === b0.nodeStart ||
          b.nodeEnd === b0.nodeStart),
    )!;
    assert.ok(connected !== undefined, "found connected beam sharing a node with b0");

    h.store.dispatch({ type: "selectDecls", decls: [b0.decl, connected.decl], add: false });
    s = h.store.get();
    assert.equal(s.lineDiagram.chains.length, 1);
    assert.equal(s.lineDiagram.chains[0]!.spans.length, 2);
    assert.ok(
      Math.abs(s.lineDiagram.chains[0]!.totalLength - (b0.length + connected.length)) < 1e-6,
    );

    // Hover midpoint of b0 and sync LineDiagramLayer:
    const midPt: [number, number, number] = [
      0.5 * (b0.pStart[0] + b0.pEnd[0]),
      0.5 * (b0.pStart[1] + b0.pEnd[1]),
      0.5 * (b0.pStart[2] + b0.pEnd[2]),
    ];
    h.store.dispatch({ type: "hover", at: { decl: b0.decl, at: midPt, nodes: [b0.nodeStart, b0.nodeEnd] } });
    s = h.store.get();
    assert.ok(s.lineDiagram.hover !== null);
    assert.equal(s.lineDiagram.hover.decl, b0.decl);
    assert.ok(Math.abs(s.lineDiagram.hover.xi) < 1e-6);

    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
    camera.position.set(4, -12, 3);
    camera.lookAt(4, 0, 3);
    camera.updateMatrixWorld();

    const layer = new LineDiagramLayer(camera);
    const stepData = h.effects.currentLineDiagram();
    assert.equal(layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement()), true);
    assert.equal(layer.hoverNeedle.visible, true);
    const needlePos = layer.hoverNeedle.geometry.getAttribute("position") as THREE.BufferAttribute;
    assert.equal(needlePos.count, 2, "hoverNeedle has 2 vertices (base -> top)");

    // Clearing hover hides hoverNeedle without rebuilding the ribbon when nothing else changed:
    h.store.dispatch({ type: "hover", at: null });
    s = h.store.get();
    assert.equal(s.lineDiagram.hover, null);
    assert.equal(layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement()), true);
    assert.equal(layer.hoverNeedle.visible, false);

    layer.dispose();
  } finally {
    h.effects.dispose();
  }
});
