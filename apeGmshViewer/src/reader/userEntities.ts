// The `/entities` zone of a `<stem>.user.h5`: section cuts, storeys and point
// sets (ADR 0130 D8; h5-schema.md "`/entities`"). The app's twin of
// src/apeGmsh/results/user/_entities.py: reader/userFile.ts reads the columns
// and the offsets, and this module judges the rows and binds an `ON_ENTITIES`
// user result to them.
//
// A bad row refuses alone and keeps its position, so the rows after it keep
// their `ENTITY_ROW` (`refused`; `entityRow` refuses it by name). Every element
// id the zone holds (a cut's `elementIds`, a point's `hostElement`) is an
// OpenSees tag: `@element_index_space` is "opensees", and any other space
// refuses the zone (userFile.ts).

import { pyFloat } from "./rf1.ts";
import { pyList, pyStr } from "./userModes.ts";
import { nameProblem } from "./userRules.ts";

/** `ENTITY_KIND` of an `ON_ENTITIES` result, and the table it names (Python's `KINDS`). */
export const ENTITY_TABLES = { cut: "cuts", storey: "storeys", point_set: "point_sets" } as const;
export type EntityKind = keyof typeof ENTITY_TABLES;
export const ENTITY_KINDS = Object.keys(ENTITY_TABLES) as EntityKind[];
/** The space every element id in the zone is in (Python's `ELEMENT_INDEX_SPACE`). */
export const ELEMENT_INDEX_SPACE = "opensees";
export const CUT_SIDES = ["positive", "negative"] as const;
/** How far a stored normal may be from unit length (Python's `_UNIT_TOL`). */
const UNIT_TOL = 1e-9;

export type Vec3 = [number, number, number];

/** One row of `/entities/cuts`: the plane, the kept side, the OpenSees tags it filters to, its polygon (`[]` for none), and the point its moments are about. */
export interface CutEntity {
  name: string;
  side: string;
  point: Vec3;
  normal: Vec3;
  reference: Vec3;
  elementIds: number[];
  polygon: Vec3[];
}
/** One row of `/entities/storeys`: a level, its nodes, its reference point (the centre of mass). */
export interface Storey {
  name: string;
  level: number;
  reference: Vec3;
  nodeIds: number[];
}
/** One row of `/entities/point_sets`: points, each with a host element (an OpenSees tag) or -1. */
export interface PointSet {
  name: string;
  xyz: Vec3[];
  hostElement: number[];
}
/** A row that does not stand, and why (Python's `EntityRefusal`). */
export interface EntityRefusal {
  kind: EntityKind;
  row: number;
  name: string;
  reason: "empty-name" | "duplicate" | "bad-param";
  detail: string;
}
/** The `/entities` zone. A row's position in its table is its `ENTITY_ROW`. */
export interface Entities {
  cuts: CutEntity[];
  storeys: Storey[];
  pointSets: PointSet[];
  refused: EntityRefusal[];
}

/** A file with no `/entities`: empty tables (Python's `Entities()`). */
export const noEntities = (): Entities => ({ cuts: [], storeys: [], pointSets: [], refused: [] });

/** The rows an `ENTITY_KIND` names. */
export function table(e: Entities, kind: EntityKind): readonly { name: string }[] {
  return kind === "cut" ? e.cuts : kind === "storey" ? e.storeys : e.pointSets;
}

/**
 * Every coordinate of every point is finite. A loop, never a spread into a
 * call: a legal zone can hold a point set or a polygon of 10^5 points, past
 * the engine's argument limit (#2749 review).
 */
function finite(...groups: readonly (readonly (readonly number[] | number)[])[]): boolean {
  for (const g of groups) {
    for (const p of g) {
      if (typeof p === "number") {
        if (!Number.isFinite(p)) return false;
      } else {
        for (const x of p) if (!Number.isFinite(x)) return false;
      }
    }
  }
  return true;
}
const list = (v: readonly number[]) => `[${v.map(pyFloat).join(", ")}]`;
/** numpy's `norm` of a 3-vector: `sqrt(dot(v, v))`. */
const norm = (v: Vec3) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

function cutProblem(c: CutEntity): string | null {
  const name = pyStr(c.name);
  if (!(CUT_SIDES as readonly string[]).includes(c.side)) return `cut ${name}: side ${pyStr(c.side)} is not one of ('positive', 'negative')`;
  if (!finite(c.point, c.normal, c.reference, c.polygon)) return `cut ${name}: a coordinate is not finite`;
  if (Math.abs(norm(c.normal) - 1) > UNIT_TOL) return `cut ${name}: normal ${list(c.normal)} is not of unit length`;
  if (c.elementIds.length === 0) return `cut ${name} filters to no element`;
  if (c.polygon.length > 0 && c.polygon.length < 3) return `cut ${name}: a bounding polygon has at least three vertices`;
  return null;
}

function storeyProblem(s: Storey): string | null {
  if (!finite([s.level], s.reference)) return `storey ${pyStr(s.name)}: the level or reference is not finite`;
  if (s.nodeIds.length === 0) return `storey ${pyStr(s.name)} has no node`;
  return null;
}

function pointSetProblem(p: PointSet): string | null {
  const name = pyStr(p.name);
  if (!finite(p.xyz)) return `point set ${name}: a coordinate is not finite`;
  if (p.xyz.length === 0) return `point set ${name} has no point`;
  if (p.hostElement.length !== p.xyz.length) return `point set ${name}: one host_element per point`;
  if (p.hostElement.some((h) => h < -1)) return `point set ${name}: a host_element is below -1 (-1 = none)`;
  return null;
}

const RULES: readonly [EntityKind, (row: never) => string | null][] = [
  ["cut", cutProblem],
  ["storey", storeyProblem],
  ["point_set", pointSetProblem],
];

/**
 * Every row of `e` that does not stand, and why (Python's `row_refusals`): the
 * name rule of A2, names unique within a table (each such row refuses), then
 * the kind's own rule.
 */
export function rowRefusals(e: Omit<Entities, "refused">): EntityRefusal[] {
  const out: EntityRefusal[] = [];
  for (const [kind, rule] of RULES) {
    const rows = table(e as Entities, kind);
    const counts = new Map<string, number>();
    for (const r of rows) counts.set(r.name, (counts.get(r.name) ?? 0) + 1);
    rows.forEach((r, i) => {
      const problem = nameProblem(r.name);
      if (problem !== null) {
        out.push({ kind, row: i, name: r.name, reason: "empty-name", detail: `name ${pyStr(r.name)}: ${problem}` });
      } else if (counts.get(r.name)! > 1) {
        out.push({ kind, row: i, name: r.name, reason: "duplicate", detail: `${counts.get(r.name)} ${ENTITY_TABLES[kind]} rows are named ${pyStr(r.name)}` });
      } else {
        const bad = rule(r as never);
        if (bad !== null) out.push({ kind, row: i, name: r.name, reason: "bad-param", detail: bad });
      }
    });
  }
  return out;
}

/** Whether `kind` is an `ENTITY_KIND`. */
export const isEntityKind = (kind: string): kind is EntityKind => Object.hasOwn(ENTITY_TABLES, kind);

/**
 * Why the `ON_ENTITIES` result `name` names a row `e` does not hold, or a
 * refused one; null when every row it names stands (Python's `binding_problem`).
 */
export function bindingProblem(e: Entities, name: string, kinds: readonly string[], rows: readonly number[]): string | null {
  for (let i = 0; i < Math.min(kinds.length, rows.length); i++) {
    const kind = kinds[i]!;
    const row = rows[i]!;
    if (!isEntityKind(kind)) return `user result ${pyStr(name)}: entity kind ${pyStr(kind)} is not one of ${pyList([...ENTITY_KINDS].sort())}`;
    const held = table(e, kind).length;
    if (!(row >= 0 && row < held)) return `user result ${pyStr(name)} names ${kind} row ${row}, and /entities/${ENTITY_TABLES[kind]} holds ${held}`;
    const ref = e.refused.find((r) => r.kind === kind && r.row === row);
    if (ref !== undefined) return `user result ${pyStr(name)} names ${kind} row ${row}, which is refused: ${ref.reason}`;
  }
  return null;
}

/** The row an `(ENTITY_KIND, ENTITY_ROW)` pair names; raises on a row the zone does not hold or refuses (Python's `Entities.row`). */
export function entityRow(e: Entities, kind: EntityKind, row: number): CutEntity | Storey | PointSet {
  const rows = table(e, kind);
  if (!(row >= 0 && row < rows.length)) throw new RangeError(`/entities/${ENTITY_TABLES[kind]} holds ${rows.length} rows, not row ${row}`);
  const ref = e.refused.find((r) => r.kind === kind && r.row === row);
  if (ref !== undefined) throw new RangeError(`/entities/${ENTITY_TABLES[kind]} row ${row} (${pyStr(ref.name)}) is refused: ${ref.reason} (${ref.detail})`);
  return rows[row] as CutEntity | Storey | PointSet;
}
