// The user-file slice of the reducer (state/reduce.ts delegates here): the
// set's `.user.h5` (ADR 0130), a User Combination's read and a user result's
// read. It writes `user`, `combination` and `userResult` only. A new user
// file leaves a User Combination shown when the file no longer lists it, and
// that is the phase's; reduce.ts applies it after this returns.

import type { UserEvent } from "../types/events/user.ts";
import {
  USER_ENTITY_KINDS,
  USER_RESULT_BASES,
  USER_RESULT_TARGETS,
  USER_ZONES,
  type Event,
  type StaleMark,
  type State,
  type UserCombination,
  type UserCombinationIn,
  type UserIndexIn,
  type UserInfo,
  type UserReadIn,
  type UserRefusal,
  type UserRefusalIn,
  type UserStalenessIn,
} from "../types.ts";

/** The `.user.h5` and what was read from it, each its own field of the state. */
export type UserSlice = Pick<State, "user" | "combination" | "userResult">;

/** What a user-file event reads of the rest of the state; each is asked only by the case that needs it. */
export interface UserInputs {
  /** The key of the User Combination read the screen asks for now (`combinationRead`). */
  combinationKey: () => string | undefined;
  /** The key of the user result read the screen asks for now (`userResultRead`). */
  userResultKey: () => string | undefined;
}

/** Whatever has no staleness verdict is stale (A6: nothing is drawn as current on no proof). */
const NO_VERDICT: StaleMark = { stale: true, reason: "the reader gave no staleness verdict for it" };

function oneOf<T extends string>(v: string, allowed: readonly T[], who: string): T {
  if (!(allowed as readonly string[]).includes(v)) throw new RangeError(`userRead: ${who} ${JSON.stringify(v)} is not one of ${allowed.join(", ")}`);
  return v as T;
}

const refusalsOf = (rows: readonly UserRefusalIn[]): UserRefusal[] => rows.map((r) => ({ name: r.name, reason: r.reason, detail: r.detail }));

/** VU16: a user result's columns, copied; anything but a list of names raises, so the caller refuses the answer. */
function componentsOf(v: unknown, name: string): string[] {
  if (!Array.isArray(v) || !v.every((c) => typeof c === "string" && c !== "")) throw new RangeError(`userRead: the components of ${name} are ${JSON.stringify(v)}, not a list of names`);
  return [...(v as string[])];
}

/** The targets a 1.0.0 spectrum case holds (h5-schema.md `/spectra`: ON_NODES and ON_DOMAIN only). */
const SPECTRUM_TARGETS = ["nodes", "domain"] as const;

/**
 * ADR 0130: the renderer's view of a `.user.h5` (`user:read`'s index and A6
 * staleness): the User Combinations and Spectra families of the phase bar, the
 * User Results picker group, and the modal table with each row's verdict. A
 * target or basis this app does not know raises, so the caller refuses the
 * answer, never shows it.
 */
export function userInfoOf(index: UserIndexIn, staleness: UserStalenessIn): UserInfo {
  // Own keys only: a user result may be named `constructor`.
  const markOf = (rec: Record<string, StaleMark>, key: string): StaleMark => (Object.hasOwn(rec, key) ? rec[key]! : NO_VERDICT);
  const zoneMark = (z: string): StaleMark => markOf(staleness.zones, z);
  const known = USER_ZONES as readonly string[];
  const warnings = [...index.warnings];
  for (const z of index.zones) if (!known.includes(z)) warnings.push(`${index.path.split(/[\\/]/).pop()} holds /${z}, a zone this app does not know; it is not read`);
  let combinations: UserInfo["combinations"] = null;
  let table: UserInfo["table"] = null;
  const c = index.combinations;
  if (!c.ok) combinations = { ok: false, reason: c.reason };
  else if (c.value.rows.length || c.value.modelRows.length || c.value.refused.length) {
    const stale = new Set(staleness.combinations);
    const item = (origin: "user" | "model") => (r: UserCombinationIn): UserCombination => ({ name: r.name, rule: r.rule, origin, terms: r.terms.length, stale: stale.has(r.name) });
    combinations = { ok: true, items: [...c.value.rows.map(item("user")), ...c.value.modelRows.map(item("model"))], refused: refusalsOf(c.value.refused) };
    // VU7b-2: the rows as the evaluator takes them, copied so the state owns them.
    const row = (r: UserCombinationIn): UserCombinationIn => ({ ...r, terms: r.terms.map((t) => ({ ...t })) });
    table = { rows: c.value.rows.map(row), modelRows: c.value.modelRows.map(row), refused: c.value.refused.map((r) => ({ ...r })) };
  }
  let results: UserInfo["results"] = null;
  const r = index.results;
  if (!r.ok) results = { ok: false, reason: r.reason };
  else if (r.value.entries.length || r.value.refused.length) {
    results = {
      ok: true,
      items: r.value.entries.map((x) => ({
        name: x.name,
        target: oneOf(x.target, USER_RESULT_TARGETS, `the target of ${x.name}`),
        basis: oneOf(x.basis, USER_RESULT_BASES, `the basis of ${x.name}`),
        perState: x.perState,
        linear: x.linear,
        components: componentsOf(x.components, x.name),
        stale: markOf(staleness.results, x.name),
      })),
      refused: refusalsOf(r.value.refused),
    };
  }
  const m = index.modes;
  let modes: UserInfo["modes"] = null;
  if (m !== null && !m.ok) modes = { ok: false, reason: m.reason, stale: zoneMark("modes") };
  else if (m !== null) {
    // One verdict per row, in table order; a count that disagrees is no verdict at all.
    const verdicts = staleness.modes.length === m.value.rows.length ? staleness.modes : null;
    modes = { ok: true, components: [...m.value.components], rows: m.value.rows.map((r, i) => ({ ...r, verdict: verdicts?.[i] ?? NO_VERDICT })), stale: zoneMark("modes") };
  }
  const sp = index.spectra;
  let spectra: UserInfo["spectra"] = null;
  if (sp !== null && !sp.ok) spectra = { ok: false, reason: sp.reason, stale: zoneMark("spectra") };
  else if (sp !== null) {
    const items = sp.value.cases.map((c) => {
      const d = c.modes.map((x) => x.damping);
      const same = d.length > 0 && d.every((x) => Object.is(x, d[0]));
      return {
        name: c.name,
        direction: c.direction,
        scale: c.scale,
        modes: c.modes.map((x) => x.mode),
        damping: same ? d[0]! : null,
        results: c.results.map((q) => `${oneOf(q.target, SPECTRUM_TARGETS, `the target of ${c.name} ${q.label}`)}/${q.label}`),
        nodal: c.results.filter((q) => q.target === "nodes").map((q) => ({ label: q.label, components: [...q.components] })),
      };
    });
    spectra = { ok: true, items, refused: refusalsOf(sp.value.refused), stale: zoneMark("spectra") };
  }
  return { warnings, combinations, table, results, spectra, modes, entities: entitiesOf(index.entities ?? null, zoneMark("entities")) };
}

/** VU8: `/entities` as the renderer draws it: the standing rows, each with its `ENTITY_ROW`, and the refused ones. */
function entitiesOf(e: NonNullable<UserIndexIn["entities"]> | null, stale: StaleMark): UserInfo["entities"] {
  if (e === null) return null;
  if (!e.ok) return { ok: false, reason: e.reason, stale };
  const refused = e.value.refused.map((r) => ({ kind: oneOf(r.kind, USER_ENTITY_KINDS, `the kind of the refused entity ${r.name}`), row: r.row, name: r.name, reason: r.reason, detail: r.detail }));
  /** The rows of `kind` that stand, each with its position. */
  const standing = <T>(kind: (typeof USER_ENTITY_KINDS)[number], rows: readonly T[]) =>
    rows.map((x, row) => ({ x, row })).filter(({ row }) => !refused.some((r) => r.kind === kind && r.row === row));
  return {
    ok: true,
    cuts: standing("cut", e.value.cuts).map(({ x: c, row }) => ({
      row,
      name: c.name,
      side: oneOf(c.side, ["positive", "negative"] as const, `the side of the cut ${c.name}`),
      point: c.point,
      normal: c.normal,
      reference: c.reference,
      elementIds: [...c.elementIds],
      polygon: [...c.polygon],
    })),
    storeys: standing("storey", e.value.storeys).map(({ x: s, row }) => ({ row, name: s.name, level: s.level, reference: s.reference, nodeIds: [...s.nodeIds] })),
    pointSets: standing("point_set", e.value.pointSets).map(({ x: p, row }) => ({ row, name: p.name, xyz: [...p.xyz], hostElement: [...p.hostElement] })),
    refused,
    stale,
  };
}

/** The suffixes main/pairing.ts `classify` knows, longest first; a part file `<stem>.part-<N>.ladruno` is its stem's results. */
const SET_SUFFIX = /^(.*?)(?:\.part-\d+\.ladruno|\.geometry\.h5|\.results\.h5|\.user\.h5|\.ladruno|\.h5)$/i;

/**
 * ADR 0130 D2: the `.user.h5` of a set, `<stem>.user.h5` beside its model (or
 * its results, with no model): main/pairing.ts `candidates`, which the
 * renderer cannot import (it reads node:fs); test/state/user.test.ts holds the
 * two together. A path the app does not open raises.
 */
export function userPathOf(set: { model: string | null; results: string | null }): string | null {
  const from = set.model ?? set.results;
  if (from === null) return null;
  const m = SET_SUFFIX.exec(from);
  if (!m || m[1] === "") throw new Error(`userPathOf: ${from} is not a <stem>.h5, <stem>.results.h5 or <stem>.ladruno`);
  return `${m[1]}.user.h5`;
}

/** reader/userFile.ts `readUserFile`'s refusal of a path with no file: the set has no `.user.h5`, which is not a failure. */
export const NO_USER_FILE = (path: string): string => `no such file: ${path}`;

/** The event a `user:read` answer for `path` becomes: no file there is `userClosed`; any other answer is `userRead`. */
export function userEventOf(path: string, read: UserReadIn): Event {
  return !read.ok && read.error === NO_USER_FILE(path) ? { type: "userClosed" } : { type: "userRead", path, read };
}

/**
 * Whether the `.user.h5` may be read now: after the model's first read has
 * landed (ready or failed), so the first frame never waits on it (ADR 0130
 * VU3a: "the first frame reads none of it"). A set with no model reads at once.
 */
export function userReadable(s: State, model: string | null): boolean {
  if (model === null) return true;
  const a = s.artifacts.model;
  return a !== null && a.path === model && a.status !== "opened";
}

/** Another `.user.h5` state. VU7b-2 and VU16: a combination's and a user result's value read from the file before are dropped. */
function withUser(u: UserSlice, user: State["user"]): UserSlice {
  return { ...u, user, combination: null, userResult: null };
}

/** The user-file slice after `e`: the same object when nothing changed. */
export function reduceUser(u: UserSlice, e: UserEvent, inputs: UserInputs): UserSlice {
  switch (e.type) {
    case "userOpened": {
      if (typeof e.path !== "string" || e.path === "") throw new RangeError(`userOpened: ${JSON.stringify(e.path)} is not a path`);
      // A re-read of the same file is pending too: a verdict read before the change (to it, the model or the results) is not shown as current.
      if (u.user?.path === e.path && u.user.status === "reading") return u;
      // VU7b-2: a combination's value read before the change is not shown as current.
      return withUser(u, { status: "reading", path: e.path });
    }
    case "userRead": {
      if (!u.user || u.user.path !== e.path) throw new Error(`userRead: ${e.path} is not the open .user.h5`);
      if (!e.read.ok) return withUser(u, { status: "failed", path: e.path, error: e.read.error });
      if (e.read.user.path !== e.path) throw new Error(`userRead: the answer is for ${e.read.user.path}, not ${e.path}`);
      return withUser(u, { status: "ready", path: e.path, info: userInfoOf(e.read.user, e.read.staleness) });
    }
    case "userClosed":
      return u.user === null ? u : withUser(u, null);
    case "combinationRead": {
      const r = e.read;
      if (r === null || typeof r !== "object" || typeof r.key !== "string" || !["loading", "ready", "refused", "failed"].includes(r.status)) {
        throw new RangeError(`combinationRead: ${JSON.stringify(r)} is not a combination read`);
      }
      // Only for the combination and quantity shown now: a late answer for another is dropped.
      if (inputs.combinationKey() !== r.key) return u;
      return { ...u, combination: r };
    }
    case "userResultRead": {
      const r = e.read;
      if (r === null || typeof r !== "object" || typeof r.key !== "string" || !["loading", "ready", "refused", "failed"].includes(r.status)) {
        throw new RangeError(`userResultRead: ${JSON.stringify(r)} is not a user result read`);
      }
      if (r.status === "refused" && r.reason !== "refused" && r.reason !== "stale") throw new RangeError(`userResultRead: refusal reason ${JSON.stringify(r.reason)} is not refused or stale`);
      // Only for the user result and component picked now: a late answer for another is dropped.
      if (inputs.userResultKey() !== r.key) return u;
      return { ...u, userResult: r };
    }
    default: {
      const unknown: never = e;
      throw new Error(`reduceUser: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
