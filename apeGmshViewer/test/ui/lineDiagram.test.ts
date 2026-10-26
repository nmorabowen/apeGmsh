// Step 4 UI tests (`src/panels/lineDiagram.ts`):
// - `.line-diagram-chain-empty` note when no 1-D beam is selected
// - Single-span and multi-span 2-D `.line-diagram-chain-svg` plots (joints, span labels, fill polygon, extrema)
// - Multiple disconnected chains rendered as stacked `.line-diagram-chain-card` elements
// - Gauss-Legendre interior stations (`[-0.5774, +0.5774]`): `.line-diagram-chain-unknown-ends` badge and
//   `.line-diagram-chain-unknown-start` / `.line-diagram-chain-unknown-finish` dashed gap markers (never clamped to joints)
// - `flipSign` toggle inverts the 2-D SVG curve around `zeroY`
// - 2-D SVG `pointermove` / `pointerleave` dispatches `{ type: "hover", at }` and renders
//   `.line-diagram-hover-readout` + `.line-diagram-chain-hover-needle`

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { mountLineDiagram, renderLineDiagram } from "../../src/panels/lineDiagram.ts";
import { lineDiagramKeysOf } from "../../src/state/lineDiagram.ts";
import type { Event, LineDiagramBeamSummary } from "../../src/state/types.ts";
import { opened, until } from "../contour/harness.ts";

const LEGENDRE_MODEL = fileURLToPath(new URL("../../fixtures/line_oracle_fbc_legendre.h5", import.meta.url));
const LEGENDRE_LADRUNO = fileURLToPath(new URL("../../fixtures/line_oracle_fbc_legendre.ladruno", import.meta.url));
const FRAME3D_MODEL = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.h5", import.meta.url));
const FRAME3D_LADRUNO = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.ladruno", import.meta.url));

class FakeEl {
  readonly tagName: string;
  className = "";
  textContent = "";
  title = "";
  disabled = false;
  hidden = false;
  type = "";
  step = "";
  min = "";
  value = "";
  label = "";
  selected = false;
  dataset: Record<string, string | undefined> = {};
  style: Record<string, string> = {};
  children: FakeEl[] = [];
  parentElement: FakeEl | null = null;
  private attrs = new Map<string, string>();
  private listeners = new Map<string, ((ev?: unknown) => void)[]>();

  constructor(tagName: string) {
    this.tagName = tagName;
  }

  setAttribute(k: string, v: string): void {
    this.attrs.set(k, v);
    if (k === "class") this.className = v;
  }

  getAttribute(k: string): string | null {
    return this.attrs.get(k) ?? null;
  }

  append(...nodes: (FakeEl | string)[]): void {
    for (const n of nodes) {
      if (typeof n === "string") {
        this.textContent += n;
      } else {
        n.parentElement = this;
        this.children.push(n);
      }
    }
  }

  replaceChildren(...nodes: FakeEl[]): void {
    for (const c of this.children) c.parentElement = null;
    this.children = [];
    this.textContent = "";
    this.append(...nodes);
  }

  addEventListener(ev: string, fn: (e?: unknown) => void): void {
    const arr = this.listeners.get(ev) ?? [];
    arr.push(fn);
    this.listeners.set(ev, arr);
  }

  removeEventListener(ev: string, fn: (e?: unknown) => void): void {
    const arr = this.listeners.get(ev) ?? [];
    this.listeners.set(
      ev,
      arr.filter((f) => f !== fn),
    );
  }

  trigger(ev: string, payload?: unknown): void {
    for (const fn of this.listeners.get(ev) ?? []) fn(payload);
  }

  hasClass(cls: string): boolean {
    return this.className.split(/\s+/).includes(cls);
  }

  find(cls: string): FakeEl | undefined {
    for (const c of this.children) {
      if (c.hasClass(cls)) return c;
      const found = c.find(cls);
      if (found) return found;
    }
    return undefined;
  }

  findAll(cls: string): FakeEl[] {
    const out: FakeEl[] = [];
    for (const c of this.children) {
      if (c.hasClass(cls)) out.push(c);
      out.push(...c.findAll(cls));
    }
    return out;
  }
}

function withFakeDom<T>(fn: (aside: FakeEl) => Promise<T> | T): Promise<T> {
  const prevDoc = (globalThis as { document?: unknown }).document;
  const aside = new FakeEl("aside");
  const docEl = { dataset: { theme: "dark" } as Record<string, string | undefined> };
  (globalThis as { document?: unknown }).document = {
    documentElement: docEl,
    createElement: (tag: string) => new FakeEl(tag),
    createElementNS: (_ns: string, tag: string) => new FakeEl(tag),
    getElementById: (id: string) => (id === "line-diagram" ? aside : null),
  };
  return Promise.resolve()
    .then(() => fn(aside))
    .finally(() => {
      (globalThis as { document?: unknown }).document = prevDoc;
    });
}

test("Step 4 UI: renderLineDiagram renders empty note, single-span and multi-span 2-D SVG chain plots, flipSign inversion, and interactive 2-D/3-D hover readout", async () => {
  await withFakeDom(async (aside) => {
    const h = await opened(FRAME3D_MODEL, FRAME3D_LADRUNO);
    const unmount = mountLineDiagram(h.store);
    try {
      h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 2 } });
      h.store.dispatch({ type: "setLineDiagram", on: true });

      await until(() => {
        const cur = h.store.get();
        const ldKeys = lineDiagramKeysOf(cur);
        return (
          ldKeys !== null &&
          cur.lineDiagram.stepRange?.key === ldKeys.step &&
          cur.lineDiagram.drawnBeams.length === 10
        );
      }, "Frame3d lineDiagram stepRange + drawnBeams ready");

      // 1. No beam selected -> renders `.line-diagram-chain-empty` note and 0 cards:
      assert.equal(aside.findAll("line-diagram-chain-card").length, 0);
      const emptyNote = aside.find("line-diagram-chain-empty");
      assert.ok(emptyNote !== undefined);
      assert.match(emptyNote.textContent, /Select a 1-D beam/);

      // 2. Select two connected beams -> 1 merged `.line-diagram-chain-card` with 2 spans and 3 joints:
      const beams = h.store.get().lineDiagram.drawnBeams;
      const b0 = beams[0]!;
      const b1 = beams.find(
        (b) =>
          b.decl !== b0.decl &&
          (b.nodeStart === b0.nodeEnd ||
            b.nodeEnd === b0.nodeEnd ||
            b.nodeStart === b0.nodeStart ||
            b.nodeEnd === b0.nodeStart),
      )!;
      assert.ok(b1 !== undefined);

      h.store.dispatch({
        type: "selectDecls",
        decls: [b0.decl, b1.decl],
        add: false,
      });

      const cards = aside.findAll("line-diagram-chain-card");
      assert.equal(cards.length, 1);
      const card = cards[0]!;
      assert.equal(card.dataset["spanCount"], "2");
      assert.match(card.find("line-diagram-chain-title")?.textContent ?? "", /2 spans/);
      assert.equal(card.find("line-diagram-chain-unknown-ends"), undefined);

      const svg = card.find("line-diagram-chain-svg");
      assert.ok(svg !== undefined);
      assert.equal(svg.getAttribute("data-flip-sign"), "true");

      // 3 joints: 2 boundary + 1 internal:
      const joints = svg.findAll("line-diagram-chain-joint");
      assert.equal(joints.length, 3);
      assert.equal(svg.findAll("internal").length, 1);
      assert.equal(svg.findAll("boundary").length, 2);
      assert.equal(svg.findAll("line-diagram-chain-joint-label").length, 3);

      // 2 spans with fill polygons, curves, and extrema markers:
      assert.equal(svg.findAll("line-diagram-chain-span-label").length, 2);
      assert.equal(svg.findAll("line-diagram-chain-fill").length, 2);
      const curvesBeforeFlip = svg.findAll("line-diagram-chain-curve");
      assert.equal(curvesBeforeFlip.length, 2);
      const ptsBefore = curvesBeforeFlip[0]!.getAttribute("points")!;
      assert.ok(ptsBefore.length > 0);
      assert.equal(curvesBeforeFlip[0]!.getAttribute("data-start-end-known"), "true");
      assert.equal(curvesBeforeFlip[0]!.getAttribute("data-end-end-known"), "true");
      assert.ok(svg.findAll("line-diagram-chain-extrema").length >= 2);

      // 3. Toggling fill off removes `.line-diagram-chain-fill` polygons:
      h.store.dispatch({ type: "setLineDiagramFill", fill: false });
      assert.equal(aside.find("line-diagram-chain-svg")!.findAll("line-diagram-chain-fill").length, 0);
      h.store.dispatch({ type: "setLineDiagramFill", fill: true });
      assert.equal(aside.find("line-diagram-chain-svg")!.findAll("line-diagram-chain-fill").length, 2);

      // 4. Clicking `.line-diagram-flip` flips the 2-D SVG curve y-coordinates:
      const flipBtn = aside.find("line-diagram-flip")!;
      flipBtn.trigger("click");
      const svgAfterFlip = aside.find("line-diagram-chain-svg")!;
      assert.equal(svgAfterFlip.getAttribute("data-flip-sign"), "false");
      const ptsAfter = svgAfterFlip.findAll("line-diagram-chain-curve")[0]!.getAttribute("points")!;
      assert.notEqual(ptsAfter, ptsBefore);

      // 5. Pointer move on `.line-diagram-chain-svg` dispatches `hover` and updates `.line-diagram-hover-readout` + `.line-diagram-chain-hover-needle` in place (preserving SVG DOM node identity):
      assert.equal(aside.find("line-diagram-hover-readout"), undefined);
      svgAfterFlip.trigger("pointermove", { offsetX: 80 });
      assert.equal(aside.find("line-diagram-chain-svg"), svgAfterFlip);
      const readout = aside.find("line-diagram-hover-readout");
      assert.ok(readout !== undefined, "pointermove on chain SVG populates hover readout");
      assert.match(readout.textContent, /E\d+ · ξ =/);
      assert.match(readout.textContent, /Mz \(bending_moment_z\) =/);
      const needle = aside.find("line-diagram-chain-hover-needle");
      assert.ok(needle !== undefined, "hover needle rendered in chain SVG");
      assert.ok(aside.find("line-diagram-chain-hover-dot") !== undefined);

      // Pointer leave clears hover readout and needle in place:
      aside.find("line-diagram-chain-svg")!.trigger("pointerleave");
      assert.equal(aside.find("line-diagram-chain-svg"), svgAfterFlip);
      assert.equal(aside.find("line-diagram-hover-readout"), undefined);
      assert.equal(aside.find("line-diagram-chain-hover-needle"), undefined);

      // 6. Selecting two disconnected beams renders two stacked `.line-diagram-chain-card` plots:
      const bDisjoint = beams.find(
        (b) =>
          b.nodeStart !== b0.nodeStart &&
          b.nodeStart !== b0.nodeEnd &&
          b.nodeEnd !== b0.nodeStart &&
          b.nodeEnd !== b0.nodeEnd,
      )!;
      assert.ok(bDisjoint !== undefined);
      h.store.dispatch({
        type: "selectDecls",
        decls: [b0.decl, bDisjoint.decl],
        add: false,
      });
      const twoCards = aside.findAll("line-diagram-chain-card");
      assert.equal(twoCards.length, 2);
      assert.match(twoCards[0]!.find("line-diagram-chain-title")?.textContent ?? "", /^Chain 1:/);
      assert.match(twoCards[1]!.find("line-diagram-chain-title")?.textContent ?? "", /^Chain 2:/);
    } finally {
      unmount();
      h.effects.dispose();
    }
  });
});

test("Step 4 UI: Gauss-Legendre interior stations render unknown-end dashed gap markers and badge without clamping to joints", async () => {
  await withFakeDom(async () => {
    const h = await opened(LEGENDRE_MODEL, LEGENDRE_LADRUNO);
    try {
      h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 1 } });
      h.store.dispatch({ type: "setLineDiagram", on: true });

      await until(() => {
        const cur = h.store.get();
        const ldKeys = lineDiagramKeysOf(cur);
        return (
          ldKeys !== null &&
          cur.lineDiagram.stepRange?.key === ldKeys.step &&
          cur.lineDiagram.drawnBeams.length === 1
        );
      }, "Gauss-Legendre lineDiagram stepRange + drawnBeams ready");

      const baseBeam = h.store.get().lineDiagram.drawnBeams[0]!;
      const g = Math.sqrt(3 / 5);
      const gaussBeam: LineDiagramBeamSummary = {
        ...baseBeam,
        xi: [-g, 0, g],
        values: [-20.39516, -9.75, -1.80484],
      };
      h.store.dispatch({
        type: "lineDiagramStepRange",
        key: lineDiagramKeysOf(h.store.get())!.step,
        min: -20.39516,
        max: -1.80484,
        notice: null,
        drawnBeams: [gaussBeam],
      });
      h.store.dispatch({
        type: "selectDecls",
        decls: [gaussBeam.decl],
        add: false,
      });

      const dispatched: Event[] = [];
      const unlisten: (() => void)[] = [];
      let dom = renderLineDiagram(
        h.store.get(),
        (e) => {
          dispatched.push(e);
          h.store.dispatch(e);
        },
        unlisten,
      ) as unknown as FakeEl | null;
      assert.ok(dom !== null);

      // Unknown-end badge + dashed start/finish gap lines + hollow dots:
      const badge = dom.find("line-diagram-chain-unknown-ends");
      assert.ok(badge !== undefined);
      assert.equal(badge.textContent, "ends unrecorded (Gauss stations)");

      const curve = dom.find("line-diagram-chain-curve")!;
      assert.equal(curve.getAttribute("data-start-end-known"), "false");
      assert.equal(curve.getAttribute("data-end-end-known"), "false");

      const unknownStart = dom.find("line-diagram-chain-unknown-start");
      const unknownFinish = dom.find("line-diagram-chain-unknown-finish");
      assert.ok(unknownStart !== undefined);
      assert.ok(unknownFinish !== undefined);
      assert.equal(unknownStart.getAttribute("stroke-dasharray"), "3,3");
      assert.equal(unknownFinish.getAttribute("stroke-dasharray"), "3,3");
      assert.equal(dom.findAll("line-diagram-chain-unknown-end-dot").length, 2);

      // Hover near the start joint (offsetX = padLeft = 26 -> u = 0 -> xi = -1 < -0.7746) marks `[clamped to station end]`:
      const svg = dom.find("line-diagram-chain-svg")!;
      svg.trigger("pointermove", { offsetX: 26 });
      assert.equal(dispatched.at(-1)?.type, "hover");
      for (const u of unlisten.splice(0)) u();

      dom = renderLineDiagram(h.store.get(), (e) => h.store.dispatch(e), unlisten) as unknown as FakeEl | null;
      assert.ok(dom !== null);
      const readout = dom.find("line-diagram-hover-readout");
      assert.ok(readout !== undefined);
      assert.equal(readout.dataset["withinStationSpan"], "false");
      assert.match(readout.textContent, /\[clamped to station end\]/);

      for (const u of unlisten) u();
    } finally {
      h.effects.dispose();
    }
  });
});
