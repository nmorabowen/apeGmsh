import assert from "node:assert/strict";
import { test } from "node:test";
import type { OpsObject, SectionProfile } from "../src/model/types.ts";
import { sectionProfileOf } from "../src/reader/read.ts";
import { triangulateSectionProfile } from "../src/state/frameExtrusion.ts";

function fiber(rows: unknown[][]): OpsObject {
  return {
    family: "section",
    path: "/opensees/sections/Fiber_1",
    groupName: "Fiber_1",
    type: "Fiber",
    tag: 1,
    params: [],
    attrs: {},
    tables: {
      patches: {
        path: "/opensees/sections/Fiber_1/patches",
        columns: ["kind", "material_ref", "ny", "nz", "coords"],
        rows: rows.map((c) => [c[0], "/opensees/materials/Steel01_1", 8, 2, c[1]]),
      },
    } as OpsObject["tables"],
  };
}

function area(loop: readonly (readonly [number, number])[]): number {
  let a2 = 0;
  for (let i = 0; i < loop.length; i++) {
    const [y0, z0] = loop[i]!;
    const [y1, z1] = loop[(i + 1) % loop.length]!;
    a2 += y0 * z1 - y1 * z0;
  }
  return 0.5 * a2;
}

function extent(loop: readonly (readonly [number, number])[]): number[] {
  const ys = loop.map((p) => p[0]);
  const zs = loop.map((p) => p[1]);
  return [Math.min(...ys), Math.max(...ys), Math.min(...zs), Math.max(...zs)];
}

/** HSS 101.6 x 101.6 x 5.92 mm, corner radii 11.84 (outer) / 5.92 (inner). */
function hss(): unknown[][] {
  const h = 50.8, t = 5.92, ro = 11.84, ri = ro - t;
  const c = h - ro;
  return [
    ["rect", [h - t, -c, h, c]], // +y wall
    ["rect", [-h, -c, -h + t, c]], // -y wall
    ["rect", [-c, h - t, c, h]], // +z wall
    ["rect", [-c, -h, c, -h + t]], // -z wall
    ["circ", [c, c, ri, ro, 0, 90]],
    ["circ", [-c, c, ri, ro, 90, 180]],
    ["circ", [-c, -c, ri, ro, 180, 270]],
    ["circ", [c, -c, ri, ro, 270, 360]],
  ];
}

test("an HSS of rect walls and quarter-ring circ corners gives one rounded-square outer loop and one rounded-square hole", () => {
  const prof = sectionProfileOf(fiber(hss())) as SectionProfile;
  assert.ok(prof, "mixed rect + circ patches draw a profile");
  assert.equal(prof.source, "patches");
  assert.equal(prof.axisymmetric, undefined);
  assert.equal(prof.loops.length, 2);
  const [outer, hole] = prof.loops;
  assert.ok(area(outer!) > 0 && area(hole!) < 0, "outer CCW, hole CW");
  const near = (got: number[], want: number[]) => got.forEach((g, i) => assert.ok(Math.abs(g - want[i]!) < 1e-9, `${got} vs ${want}`));
  near(extent(outer!), [-50.8, 50.8, -50.8, 50.8]);
  near(extent(hole!), [-44.88, 44.88, -44.88, 44.88]);
  // Rounded, not square: no vertex at the square's corners, and arc points on both radii
  const h = 50.8, c = h - 11.84;
  assert.ok(!outer!.some(([y, z]) => Math.abs(Math.abs(y) - h) < 1e-6 && Math.abs(Math.abs(z) - h) < 1e-6));
  for (const [loop, r] of [[outer!, 11.84], [hole!, 5.92]] as const) {
    const onArc = loop.filter(([y, z]) => Math.abs(y) > c + 1e-9 && Math.abs(z) > c + 1e-9);
    assert.ok(onArc.length >= 4 * 3, "every corner is an arc");
    for (const [y, z] of onArc) assert.ok(Math.abs(Math.hypot(Math.abs(y) - c, Math.abs(z) - c) - r) < 1e-9);
  }
  // Steel area: four walls + four quarter rings, ~ 2210 mm^2 (polygonal arcs a touch smaller)
  const exact = 4 * (2 * c) * 5.92 + Math.PI * (11.84 ** 2 - 5.92 ** 2);
  const net = area(outer!) + area(hole!);
  assert.ok(net < exact && net > 0.995 * exact, `${net} vs ${exact}`);
  // And it triangulates into a cap with that same area (what the extrusion draws)
  const tri = triangulateSectionProfile(prof);
  const capArea = tri.triangles.reduce((acc, [a, b, d]) => acc + Math.abs(area([tri.vertices[a]!, tri.vertices[b]!, tri.vertices[d]!])), 0);
  assert.ok(Math.abs(capArea - net) < 1e-6 * net, `${capArea} vs ${net}`);
});

test("rect + circ mixes that cannot be unioned exactly give no profile; disjoint or open ones still draw", () => {
  const rows = hss();
  // a missing corner leaves the ring open: still one loop, so it draws (a C-shape)
  const open = sectionProfileOf(fiber(rows.slice(0, 7)));
  assert.equal(open?.loops.length, 1);
  // a corner that overlaps a wall
  const h = 50.8, c = h - 11.84;
  assert.equal(sectionProfileOf(fiber([...rows.slice(0, 7), ["circ", [c, -c + 1, 5.92, 11.84, 270, 360]]])), undefined);
  // a duplicated patch
  assert.equal(sectionProfileOf(fiber([...rows, rows[0]!])), undefined);
  // a full circle beside a rect
  assert.equal(sectionProfileOf(fiber([["circ", [0, 0, 0, 0.2]], ["rect", [1, -0.1, 2, 0.1]]])), undefined);
  // a sector overlapping a rect
  assert.equal(sectionProfileOf(fiber([["circ", [0, 0, 0, 0.2, 0, 90]], ["rect", [-0.5, -0.1, 0.5, 0.1]]])), undefined);
  // a sector inside a rect (no edge crossing: caught by the coverage check)
  assert.equal(sectionProfileOf(fiber([["circ", [0, 0, 0, 0.05, 0, 90]], ["rect", [-0.5, -0.5, 0.5, 0.5]]])), undefined);
  // disjoint rect and sector: two outer loops
  assert.equal(sectionProfileOf(fiber([["circ", [0, 0, 0, 0.2, 0, 90]], ["rect", [1, -0.1, 2, 0.1]]]))?.loops.length, 2);
});
