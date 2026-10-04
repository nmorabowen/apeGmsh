// The pure reducer: `reduce(state, event)` returns the next state and never
// mutates its arguments (test/store.test.ts freezes the input and checks).
// The event union is exhaustive: the `never` check at the end of the switch
// fails to compile when an event is added without a case, and an event the
// code does not know raises at run time.

import type { ArtifactInfo, ArtifactKind, DeclPath, Event, PhaseKey, State } from "./types.ts";

/** A record with no prototype: a key named `constructor` or `__proto__` is just a key. */
export const emptyRecord = <T>(): Record<string, T> => Object.create(null) as Record<string, T>;

export const initialState: State = {
  artifacts: { geometry: null, model: null, results: null },
  decls: emptyRecord(),
  names: emptyRecord(),
  blocks: [],
  mesh: null,
  phase: { axis: [], at: null },
  selection: { decls: [], picks: [] },
  hover: null,
  visibility: { hidden: [], edges: true, opacity: 1 },
  inspector: { pinned: [] },
  windows: { open: [] },
  overrides: {},
};

const NO_SELECTION: State["selection"] = { decls: [], picks: [] };

function opened(path: string): ArtifactInfo {
  return {
    path,
    status: "opened",
    error: null,
    stale: false,
    sizeBytes: 0,
    readMs: 0,
    zones: {},
    warnings: [],
    counts: { nodes: 0, cells: 0, opsOnly: 0 },
  };
}

function samePhase(a: PhaseKey, b: PhaseKey): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "stage" && b.kind === "stage") return a.index === b.index;
  if (a.kind === "results" && b.kind === "results") return a.step === b.step;
  return true;
}

function artifactAt(s: State, path: string): ArtifactKind | null {
  for (const k of ["geometry", "model", "results"] as const) if (s.artifacts[k]?.path === path) return k;
  return null;
}

/** The decl paths of every colour group (physical groups), the unit `isolate` and `showAll` work on. */
function groupPaths(s: State): DeclPath[] {
  return Object.values(s.decls).filter((d) => d.kind === "physical_group").map((d) => d.path);
}

const uniq = (a: readonly DeclPath[]): DeclPath[] => [...new Set(a)];

export function reduce(s: State, e: Event): State {
  switch (e.type) {
    case "fileOpened": {
      const prev = s.artifacts[e.artifact];
      // The same path again (a reload after fileChanged) keeps what was read until the new read lands.
      const info: ArtifactInfo = prev && prev.path === e.path ? { ...prev, stale: false, error: null } : opened(e.path);
      return { ...s, artifacts: { ...s.artifacts, [e.artifact]: info } };
    }
    case "fileLoaded": {
      const { decls, names, blocks, mesh } = e.load;
      // A fileChanged that arrived while this read ran keeps the artifact
      // stale, so the effects read it once more.
      const prev = s.artifacts.model;
      const info: ArtifactInfo = { ...e.load.info, stale: prev !== null && prev.path === e.load.info.path && prev.stale };
      return {
        ...s,
        artifacts: { ...s.artifacts, model: info },
        decls,
        names,
        blocks,
        mesh,
        phase: { axis: [{ kind: "mesh" }], at: { kind: "mesh" } },
        selection: NO_SELECTION,
        hover: null,
        visibility: { ...s.visibility, hidden: [] },
        inspector: { pinned: [] },
      };
    }
    case "fileFailed": {
      const prev = s.artifacts[e.artifact];
      // A fileChanged that arrived during the failed read keeps the artifact
      // stale (the read may have hit a half-written file); the effects retry.
      const stale = prev !== null && prev.path === e.path && prev.stale;
      const info: ArtifactInfo = { ...(prev && prev.path === e.path ? prev : opened(e.path)), status: "failed", error: e.error, stale };
      if (e.artifact !== "model") return { ...s, artifacts: { ...s.artifacts, [e.artifact]: info } };
      // The model is gone with its derivations; the failure is shown instead.
      return {
        ...initialState,
        artifacts: { ...s.artifacts, model: info },
        visibility: s.visibility,
        windows: s.windows,
      };
    }
    case "fileClosed": {
      if (!s.artifacts[e.artifact]) return s;
      if (e.artifact !== "model") return { ...s, artifacts: { ...s.artifacts, [e.artifact]: null } };
      return { ...initialState, artifacts: { ...s.artifacts, model: null }, visibility: s.visibility, windows: s.windows };
    }
    case "fileChanged": {
      const k = artifactAt(s, e.path);
      if (!k) return s;
      return { ...s, artifacts: { ...s.artifacts, [k]: { ...s.artifacts[k]!, stale: true } } };
    }
    case "zoneRefused": {
      const prev = s.artifacts[e.artifact];
      if (!prev) throw new Error(`zoneRefused for ${e.artifact} before fileOpened`);
      const zones = {
        ...prev.zones,
        [e.zone]: { status: "refused" as const, version: e.version, accepted: e.accepted, newer: e.newer, reason: e.reason },
      };
      return { ...s, artifacts: { ...s.artifacts, [e.artifact]: { ...prev, zones } } };
    }
    case "select":
      if (!(e.pick.decl in s.decls)) throw new Error(`select: ${e.pick.decl} is not a declaration of the loaded model`);
      return { ...s, selection: { decls: [e.pick.decl], picks: [e.pick] } };
    case "selectAdd": {
      if (!(e.pick.decl in s.decls)) throw new Error(`selectAdd: ${e.pick.decl} is not a declaration of the loaded model`);
      if (s.selection.decls.includes(e.pick.decl)) return s;
      return { ...s, selection: { decls: [...s.selection.decls, e.pick.decl], picks: [...s.selection.picks, e.pick] } };
    }
    case "clearSelection":
      return s.selection.decls.length === 0 ? s : { ...s, selection: NO_SELECTION };
    case "hover":
      return { ...s, hover: e.at };
    case "setHidden": {
      const hidden = e.hidden
        ? uniq([...s.visibility.hidden, ...e.decls])
        : s.visibility.hidden.filter((d) => !e.decls.includes(d));
      return { ...s, visibility: { ...s.visibility, hidden } };
    }
    case "isolate": {
      const keep = new Set(e.decls);
      return { ...s, visibility: { ...s.visibility, hidden: groupPaths(s).filter((p) => !keep.has(p)) } };
    }
    case "showAll":
      return { ...s, visibility: { ...s.visibility, hidden: [] } };
    case "setEdges":
      return { ...s, visibility: { ...s.visibility, edges: e.on } };
    case "setOpacity":
      if (!Number.isFinite(e.value)) throw new RangeError(`setOpacity: ${e.value} is not a number`);
      return { ...s, visibility: { ...s.visibility, opacity: Math.min(1, Math.max(0, e.value)) } };
    case "setPhase":
      if (!s.phase.axis.some((k) => samePhase(k, e.at))) throw new Error(`setPhase: ${JSON.stringify(e.at)} is not on the phase axis`);
      return { ...s, phase: { ...s.phase, at: e.at } };
    case "setResultStep": {
      if (!s.phase.axis.some((k) => k.kind === "results")) throw new Error("setResultStep: the phase axis has no results");
      return { ...s, phase: { ...s.phase, at: { kind: "results", step: e.step } } };
    }
    case "inspectorPin":
      if (!(e.decl in s.decls)) throw new Error(`inspectorPin: ${e.decl} is not a declaration of the loaded model`);
      if (s.inspector.pinned.includes(e.decl)) return s;
      return { ...s, inspector: { pinned: [...s.inspector.pinned, e.decl] } };
    case "inspectorUnpin":
      return { ...s, inspector: { pinned: s.inspector.pinned.filter((d) => d !== e.decl) } };
    case "openWindow":
      if (s.windows.open.includes(e.window)) return s;
      return { ...s, windows: { open: [...s.windows.open, e.window] } };
    case "closeWindow":
      return { ...s, windows: { open: s.windows.open.filter((w) => w !== e.window) } };
    default: {
      const unknown: never = e;
      throw new Error(`reduce: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
