// The overlay slice of the reducer (state/reduce.ts delegates here): the
// supports, principal-stress, crack, reaction, spring, vector and load overlays.
// Each event writes one of the seven and reads nothing else of the state.

import type { OverlaysEvent } from "../types/events/overlays.ts";
import { CMAP_NAMES } from "../contour.ts";
import { NODE_SIZE_MAX, NODE_SIZE_MIN } from "../nodes.ts";
import type { CracksState, HingesState, LoadsState, PrincipalState, ReactionsState, SpringsState, State, SupportsState, VectorsState } from "../types.ts";

/** The seven overlay settings, each its own field of the state. */
export type Overlays = Pick<State, "supports" | "principal" | "cracks" | "reactions" | "springs" | "hinges" | "vectors" | "loads">;

export const initialSupports: SupportsState = { on: false, scale: 1 };

export const initialPrincipal: PrincipalState = { on: false, field: null, show: [true, true, true], head: "axis", cmap: "RdBu_r", scale: 1, headSize: 10, thin: true };
export const initialCracks: CracksState = { on: false, minOpening: 0, cmap: "turbo", scale: 1, face: "both" };
export const initialReactions: ReactionsState = { on: false, scale: 1, headSize: 12, colorBy: "mono", thin: true, component: "resultant", allNodes: false, forces: true, moments: true };
export const initialHinges: HingesState = { on: false, colorBy: "rotation", size: null, auto: true, element: null, end: "i", axis: "z", history: null, notice: null };
export const initialSprings: SpringsState = { on: false, scale: 1, headSize: 12, colorBy: "mono", thin: true, component: "resultant", notice: null };
export const initialVectors: VectorsState = { on: false, field: null, scale: 1, headSize: 12, colorBy: "mono", thin: true, component: "resultant" };

export const initialLoads: LoadsState = {
  on: false,
  scale: 1,
  showNodalForces: true,
  showNodalMoments: true,
  showElementLoads: true,
  showSp: true,
  includeSupportReactions: false,
  scaleBySeries: false,
  headSize: 12,
  thin: true,
  colorBy: "mono",
  selectedPatterns: null,
};

/** The overlays after `e`: the same object when nothing changed. */
export function reduceOverlays(o: Overlays, e: OverlaysEvent): Overlays {
  switch (e.type) {
    case "setSupports": {
      if (e.on !== undefined && typeof e.on !== "boolean") throw new RangeError(`setSupports: on ${JSON.stringify(e.on)}`);
      if (e.scale !== undefined && (typeof e.scale !== "number" || !Number.isFinite(e.scale) || e.scale <= 0)) {
        throw new RangeError(`setSupports: the scale must be a finite number above 0; got ${JSON.stringify(e.scale)}`);
      }
      const on = e.on ?? o.supports.on;
      const scale = e.scale ?? o.supports.scale;
      return on === o.supports.on && scale === o.supports.scale ? o : { ...o, supports: { on, scale } };
    }
    case "setPrincipal": {
      for (const k of ["on", "thin"] as const) {
        if (e[k] !== undefined && typeof e[k] !== "boolean") throw new RangeError(`setPrincipal: ${k} ${JSON.stringify(e[k])}`);
      }
      for (const k of ["scale", "headSize"] as const) {
        const v = e[k];
        if (v !== undefined && (typeof v !== "number" || !Number.isFinite(v) || v <= 0)) {
          throw new RangeError(`setPrincipal: the ${k} must be a finite number above 0; got ${JSON.stringify(v)}`);
        }
      }
      if (e.field !== undefined && e.field !== null && typeof e.field !== "string") throw new RangeError(`setPrincipal: field ${JSON.stringify(e.field)}`);
      if (e.show !== undefined && !(Array.isArray(e.show) && e.show.length === 3 && e.show.every((b) => typeof b === "boolean"))) {
        throw new RangeError(`setPrincipal: show must be three booleans (p1, p2, p3); got ${JSON.stringify(e.show)}`);
      }
      if (e.head !== undefined && e.head !== "axis" && e.head !== "arrow") throw new RangeError(`setPrincipal: head ${JSON.stringify(e.head)}`);
      if (e.cmap !== undefined && !(CMAP_NAMES as readonly string[]).includes(e.cmap)) throw new RangeError(`setPrincipal: cmap ${JSON.stringify(e.cmap)}`);
      const cur = o.principal;
      const show = e.show && e.show.some((b, i) => b !== cur.show[i]) ? ([e.show[0], e.show[1], e.show[2]] as const) : cur.show;
      const next: PrincipalState = {
        on: e.on ?? cur.on,
        field: e.field !== undefined ? e.field : cur.field,
        show,
        head: e.head ?? cur.head,
        cmap: e.cmap ?? cur.cmap,
        scale: e.scale ?? cur.scale,
        headSize: e.headSize ?? cur.headSize,
        thin: e.thin ?? cur.thin,
      };
      return (Object.keys(next) as (keyof PrincipalState)[]).every((k) => next[k] === cur[k]) ? o : { ...o, principal: next };
    }
    case "setCracks": {
      if (e.on !== undefined && typeof e.on !== "boolean") throw new RangeError(`setCracks: on ${JSON.stringify(e.on)}`);
      if (e.minOpening !== undefined && (typeof e.minOpening !== "number" || !Number.isFinite(e.minOpening) || e.minOpening < 0)) {
        throw new RangeError(`setCracks: the minimum opening must be a finite number of 0 or more; got ${JSON.stringify(e.minOpening)}`);
      }
      if (e.scale !== undefined && (typeof e.scale !== "number" || !Number.isFinite(e.scale) || e.scale <= 0)) {
        throw new RangeError(`setCracks: the scale must be a finite number above 0; got ${JSON.stringify(e.scale)}`);
      }
      if (e.cmap !== undefined && !(CMAP_NAMES as readonly string[]).includes(e.cmap)) throw new RangeError(`setCracks: cmap ${JSON.stringify(e.cmap)}`);
      if (e.face !== undefined && !["both", "plus", "minus"].includes(e.face)) throw new RangeError(`setCracks: face ${JSON.stringify(e.face)}`);
      const cur = o.cracks;
      const next: CracksState = { on: e.on ?? cur.on, minOpening: e.minOpening ?? cur.minOpening, cmap: e.cmap ?? cur.cmap, scale: e.scale ?? cur.scale, face: e.face ?? cur.face };
      return (Object.keys(next) as (keyof CracksState)[]).every((k) => next[k] === cur[k]) ? o : { ...o, cracks: next };
    }
    case "setReactions": {
      for (const k of ["on", "thin", "allNodes", "forces", "moments"] as const) {
        if (e[k] !== undefined && typeof e[k] !== "boolean") throw new RangeError(`setReactions: ${k} ${JSON.stringify(e[k])}`);
      }
      for (const k of ["scale", "headSize"] as const) {
        const v = e[k];
        if (v !== undefined && (typeof v !== "number" || !Number.isFinite(v) || v <= 0)) {
          throw new RangeError(`setReactions: the ${k} must be a finite number above 0; got ${JSON.stringify(v)}`);
        }
      }
      if (e.colorBy !== undefined && !["mono", "type", "magnitude"].includes(e.colorBy)) throw new RangeError(`setReactions: colorBy ${JSON.stringify(e.colorBy)}`);
      if (e.component !== undefined && !["resultant", "xyz", "x", "y", "z"].includes(e.component)) {
        throw new RangeError(`setReactions: component ${JSON.stringify(e.component)}`);
      }
      const cur = o.reactions;
      const pos = (v: number | undefined, was: number) => (v !== undefined && Number.isFinite(v) && v > 0 ? v : was);
      const next: ReactionsState = {
        on: e.on ?? cur.on,
        scale: pos(e.scale, cur.scale),
        headSize: pos(e.headSize, cur.headSize),
        colorBy: e.colorBy ?? cur.colorBy,
        thin: e.thin ?? cur.thin,
        component: e.component ?? cur.component,
        allNodes: e.allNodes ?? cur.allNodes,
        forces: e.forces ?? cur.forces,
        moments: e.moments ?? cur.moments,
      };
      return (Object.keys(next) as (keyof ReactionsState)[]).every((k) => next[k] === cur[k]) ? o : { ...o, reactions: next };
    }
    case "setSprings": {
      for (const k of ["on", "thin"] as const) {
        if (e[k] !== undefined && typeof e[k] !== "boolean") throw new RangeError(`setSprings: ${k} ${JSON.stringify(e[k])}`);
      }
      for (const k of ["scale", "headSize"] as const) {
        const v = e[k];
        if (v !== undefined && (typeof v !== "number" || !Number.isFinite(v) || v <= 0)) {
          throw new RangeError(`setSprings: the ${k} must be a finite number above 0; got ${JSON.stringify(v)}`);
        }
      }
      if (e.colorBy !== undefined && !["mono", "magnitude"].includes(e.colorBy)) throw new RangeError(`setSprings: colorBy ${JSON.stringify(e.colorBy)}`);
      if (e.component !== undefined && !["resultant", "xyz", "x", "y", "z"].includes(e.component)) {
        throw new RangeError(`setSprings: component ${JSON.stringify(e.component)}`);
      }
      if (e.notice !== undefined && e.notice !== null && typeof e.notice !== "string") throw new RangeError(`setSprings: notice ${JSON.stringify(e.notice)}`);
      const cur = o.springs;
      const next: SpringsState = {
        on: e.on ?? cur.on,
        scale: e.scale ?? cur.scale,
        headSize: e.headSize ?? cur.headSize,
        colorBy: e.colorBy ?? cur.colorBy,
        thin: e.thin ?? cur.thin,
        component: e.component ?? cur.component,
        notice: e.notice !== undefined ? e.notice : cur.notice,
      };
      return (Object.keys(next) as (keyof SpringsState)[]).every((k) => next[k] === cur[k]) ? o : { ...o, springs: next };
    }
    case "setHinges": {
      if (e.on !== undefined && typeof e.on !== "boolean") throw new RangeError(`setHinges: on ${JSON.stringify(e.on)}`);
      if (e.colorBy !== undefined && !["rotation", "moment", "capacity"].includes(e.colorBy)) throw new RangeError(`setHinges: colorBy ${JSON.stringify(e.colorBy)}`);
      if (e.size !== undefined && e.size !== null && (typeof e.size !== "number" || !Number.isFinite(e.size) || e.size <= 0)) throw new RangeError(`setHinges: the size must be null or a finite number above 0; got ${JSON.stringify(e.size)}`);
      if (e.auto !== undefined && typeof e.auto !== "boolean") throw new RangeError(`setHinges: auto ${JSON.stringify(e.auto)}`);
      if (e.end !== undefined && e.end !== "i" && e.end !== "j") throw new RangeError(`setHinges: end ${JSON.stringify(e.end)}`);
      if (e.axis !== undefined && e.axis !== "z" && e.axis !== "y") throw new RangeError(`setHinges: axis ${JSON.stringify(e.axis)}`);
      if (e.notice !== undefined && e.notice !== null && typeof e.notice !== "string") throw new RangeError(`setHinges: notice ${JSON.stringify(e.notice)}`);
      const cur = o.hinges;
      const on = e.on ?? cur.on;
      // As the node markers: null chooses again, a number is the user's unless the effects say `auto`, and switching
      // on chooses again unless the user has set a size.
      let size = cur.size, auto = cur.auto;
      if (e.size === null) [size, auto] = [null, true];
      else if (e.size !== undefined) [size, auto] = [Math.max(NODE_SIZE_MIN, Math.min(NODE_SIZE_MAX, e.size)), e.auto === true];
      else if (on && !cur.on && cur.auto) size = null;
      const next: HingesState = {
        on,
        colorBy: e.colorBy ?? cur.colorBy,
        size,
        auto,
        element: e.element !== undefined ? e.element : cur.element,
        end: e.end ?? cur.end,
        axis: e.axis ?? cur.axis,
        history: e.history !== undefined ? e.history : cur.history,
        notice: e.notice !== undefined ? e.notice : cur.notice,
      };
      return (Object.keys(next) as (keyof HingesState)[]).every((k) => next[k] === cur[k]) ? o : { ...o, hinges: next };
    }
    case "setVectors": {
      for (const k of ["on", "thin"] as const) {
        if (e[k] !== undefined && typeof e[k] !== "boolean") throw new RangeError(`setVectors: ${k} ${JSON.stringify(e[k])}`);
      }
      for (const k of ["scale", "headSize"] as const) {
        const v = e[k];
        if (v !== undefined && (typeof v !== "number" || !Number.isFinite(v) || v <= 0)) {
          throw new RangeError(`setVectors: the ${k} must be a finite number above 0; got ${JSON.stringify(v)}`);
        }
      }
      if (e.field !== undefined && e.field !== null && (typeof e.field !== "string" || e.field === "")) throw new RangeError(`setVectors: field ${JSON.stringify(e.field)}`);
      if (e.colorBy !== undefined && !["mono", "type", "magnitude"].includes(e.colorBy)) throw new RangeError(`setVectors: colorBy ${JSON.stringify(e.colorBy)}`);
      if (e.component !== undefined && !["resultant", "xyz", "x", "y", "z"].includes(e.component)) {
        throw new RangeError(`setVectors: component ${JSON.stringify(e.component)}`);
      }
      const cur = o.vectors;
      const next: VectorsState = {
        on: e.on ?? cur.on,
        field: e.field !== undefined ? e.field : cur.field,
        scale: e.scale ?? cur.scale,
        headSize: e.headSize ?? cur.headSize,
        colorBy: e.colorBy ?? cur.colorBy,
        thin: e.thin ?? cur.thin,
        component: e.component ?? cur.component,
      };
      return (Object.keys(next) as (keyof VectorsState)[]).every((k) => next[k] === cur[k]) ? o : { ...o, vectors: next };
    }
    case "setLoads": {
      if (e.on !== undefined && typeof e.on !== "boolean") {
        throw new RangeError(`setLoads: on ${JSON.stringify(e.on)}`);
      }
      if (e.scale !== undefined && (typeof e.scale !== "number" || !Number.isFinite(e.scale) || e.scale <= 0)) {
        throw new RangeError(`setLoads: the scale must be a finite number above 0; got ${JSON.stringify(e.scale)}`);
      }
      const cur = o.loads;
      let on = e.on ?? cur.on;
      const scale = e.scale ?? cur.scale;
      const showNodalForces = e.showNodalForces ?? cur.showNodalForces;
      const showNodalMoments = e.showNodalMoments ?? cur.showNodalMoments;
      const showElementLoads = e.showElementLoads ?? cur.showElementLoads;
      const showSp = e.showSp ?? cur.showSp;
      const includeSupportReactions = e.includeSupportReactions ?? cur.includeSupportReactions;
      const scaleBySeries = e.scaleBySeries ?? cur.scaleBySeries;
      const headSize = e.headSize ?? cur.headSize;
      const thin = e.thin ?? cur.thin;
      const colorBy = e.colorBy ?? cur.colorBy;
      let selectedPatterns = cur.selectedPatterns;
      if (e.selectedPatterns !== undefined) {
        selectedPatterns = e.selectedPatterns === null ? null : [...new Set(e.selectedPatterns)];
      } else if (e.togglePattern !== undefined) {
        const key = e.togglePattern;
        if (!cur.on || cur.selectedPatterns === null) {
          on = e.on ?? true;
          selectedPatterns = [key];
        } else if (cur.selectedPatterns.includes(key)) {
          const rest = cur.selectedPatterns.filter((p) => p !== key);
          if (rest.length === 0) {
            on = e.on ?? false;
            selectedPatterns = null;
          } else {
            selectedPatterns = rest;
          }
        } else {
          selectedPatterns = [...cur.selectedPatterns, key];
        }
      }
      const samePatterns =
        selectedPatterns === cur.selectedPatterns ||
        (selectedPatterns !== null &&
          cur.selectedPatterns !== null &&
          selectedPatterns.length === cur.selectedPatterns.length &&
          selectedPatterns.every((p, i) => p === cur.selectedPatterns![i]));
      if (
        on === cur.on &&
        scale === cur.scale &&
        showNodalForces === cur.showNodalForces &&
        showNodalMoments === cur.showNodalMoments &&
        showElementLoads === cur.showElementLoads &&
        showSp === cur.showSp &&
        includeSupportReactions === cur.includeSupportReactions &&
        scaleBySeries === cur.scaleBySeries &&
        headSize === cur.headSize &&
        thin === cur.thin &&
        colorBy === cur.colorBy &&
        samePatterns
      ) {
        return o;
      }
      return {
        ...o,
        loads: {
          on,
          scale,
          showNodalForces,
          showNodalMoments,
          showElementLoads,
          showSp,
          includeSupportReactions,
          scaleBySeries,
          headSize,
          thin,
          colorBy,
          selectedPatterns,
        },
      };
    }
    default: {
      const unknown: never = e;
      throw new Error(`reduceOverlays: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
