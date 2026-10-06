// Colour by structural role (#1308 round 3, item 2): column, wall, beam,
// concrete, void, in the office colours. The role comes from the file
// only. Today no apeGmsh writer records one (architecture/h5-schema.md has
// no structural-role attribute; the only `role` columns are rebar bars and
// the HOLD pattern), so the mode reports that the file carries none and
// draws every element as unassigned. Nothing here guesses a role from a
// group's name, an element's type or its shape.
//
// The seam: a `role` attribute on an element-side physical group
// (`/physical_groups/element_side/<name>@role`, one of ROLES). When the
// writer gains it and the reader exposes it (chain K), the loader adds a
// `role` field to the group's declaration and this module lights up.

import { DARK, DARK_ROLE, hexToRgb, ROLES, type RGB, type Role } from "../theme/tokens.ts";
import type { Decl, DeclPath, LegendEntry, State } from "./types.ts";

/** The legend row of elements with no role in the file. */
export const UNASSIGNED_ROW = "(no structural role in the file)";
export const UNASSIGNED_COLOUR: RGB = hexToRgb(DARK.unassigned);

/** The role a group declaration carries, or null; a value outside ROLES is an error, never a guess. */
export function roleOfGroup(d: Decl): Role | null {
  const f = d.fields.find((x) => x.label === "role");
  if (!f) return null;
  if (!(ROLES as readonly string[]).includes(f.value)) throw new Error(`${d.h5}@role = ${JSON.stringify(f.value)} is not a structural role (${ROLES.join(", ")})`);
  return f.value as Role;
}

/** The role of an element: the one role its groups carry; two different roles are an error. */
export function roleOfElement(s: State, path: DeclPath): Role | null {
  const d = s.decls[path];
  if (!d?.element) return null;
  let role: Role | null = null;
  for (const g of d.element.groups) {
    const gd = s.decls[g];
    const r = gd ? roleOfGroup(gd) : null;
    if (r === null) continue;
    if (role !== null && role !== r) throw new Error(`${path}: its groups carry two roles, ${role} and ${r}`);
    role = r;
  }
  return role;
}

/**
 * Colour by role for the drawn elements: one colour index per element of
 * `mesh.elements`, the legend rows in ROLES order (present roles only, then
 * the unassigned row), and whether the file carries any role at all.
 */
export function roleColouring(s: State): { byElement: Int32Array; legend: LegendEntry[]; fileHasRoles: boolean } | null {
  const info = s.mesh;
  if (!info) return null;
  const present = new Set<Role>();
  const roles: (Role | null)[] = info.elements.map((p) => {
    const r = roleOfElement(s, p);
    if (r) present.add(r);
    return r;
  });
  const rows: Role[] = ROLES.filter((r) => present.has(r));
  const legend: LegendEntry[] = rows.map((r) => ({
    decl: null,
    name: r,
    color: hexToRgb(DARK_ROLE[r]),
    elements: roles.filter((x) => x === r).length,
    cue: null,
    slot: null,
  }));
  const unassigned = roles.filter((x) => x === null).length;
  if (unassigned) legend.push({ decl: null, name: UNASSIGNED_ROW, color: UNASSIGNED_COLOUR, elements: unassigned, cue: null, slot: null });
  const index = new Map<string, number>(legend.map((e, i) => [e.name, i]));
  const byElement = Int32Array.from(roles, (r) => index.get(r ?? UNASSIGNED_ROW)!);
  return { byElement, legend, fileHasRoles: present.size > 0 };
}
