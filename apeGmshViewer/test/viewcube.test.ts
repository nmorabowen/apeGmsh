// ViewCube pure geometry and heading math: 26 surface zones (6 faces, 12 edges,
// 8 corners), heading mapping, cardinal polar yaw snapping, and 2D orthographic
// projection with back-face culling; plus universal panel collapse helper.

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { orientationOf } from "../src/renderer/navigation.ts";
import {
  headingForZone, homeHeading, projectCube, snapCardinalYaw,
  VIEWCUBE_ZONES, zoneById, zoneIdOf,
} from "../src/renderer/viewcube.ts";
import { DEFAULT_PANEL_IDS, toggleCardCollapse } from "../src/ui/collapse.ts";

const close = (a: number, b: number, tol: number, what: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);

test("VIEWCUBE_ZONES: 26 unique surface zones (6 faces, 12 edges, 8 corners)", () => {
  assert.equal(VIEWCUBE_ZONES.length, 26);
  assert.equal(VIEWCUBE_ZONES.filter((z) => z.kind === "face").length, 6);
  assert.equal(VIEWCUBE_ZONES.filter((z) => z.kind === "edge").length, 12);
  assert.equal(VIEWCUBE_ZONES.filter((z) => z.kind === "corner").length, 8);
  const ids = new Set(VIEWCUBE_ZONES.map((z) => z.id));
  assert.equal(ids.size, 26);
  for (const z of VIEWCUBE_ZONES) {
    assert.equal(zoneIdOf(z.sign), z.id);
    assert.equal(zoneById(z.id), z);
  }
});

test("headingForZone: every face, edge, and corner looks inward along -sign", () => {
  for (const zone of VIEWCUBE_ZONES) {
    const h = headingForZone(zone, 0.3);
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(orientationOf(h));
    const expected = new THREE.Vector3(-zone.sign[0], -zone.sign[1], -zone.sign[2]).normalize();
    close(forward.x, expected.x, 1e-12, `${zone.id} forward.x`);
    close(forward.y, expected.y, 1e-12, `${zone.id} forward.y`);
    close(forward.z, expected.z, 1e-12, `${zone.id} forward.z`);
  }
});

test("TOP and BOTTOM faces snap yaw to the nearest 90-degree cardinal step", () => {
  const top = zoneById("top")!;
  const bottom = zoneById("bottom")!;
  const hTop = headingForZone(top, 0.2);
  close(hTop.tilt, 0, 1e-12, "TOP tilt");
  close(hTop.yaw, 0, 1e-12, "TOP yaw snaps near 0");

  const hTopEast = headingForZone(top, Math.PI / 2 - 0.15);
  close(hTopEast.yaw, Math.PI / 2, 1e-12, "TOP yaw snaps near +90 deg");

  const hBot = headingForZone(bottom, -Math.PI / 2 + 0.1);
  close(hBot.tilt, Math.PI, 1e-12, "BOTTOM tilt");
  close(hBot.yaw, -Math.PI / 2, 1e-12, "BOTTOM yaw snaps near -90 deg");

  close(snapCardinalYaw(0.79), Math.PI / 2, 1e-12, "snapCardinalYaw > 45 deg");
  close(snapCardinalYaw(0.78), 0, 1e-12, "snapCardinalYaw < 45 deg");
});

test("projectCube: back-face culling leaves 1..3 visible faces, each with 9 patches (1 face + 4 edges + 4 corners)", () => {
  // Isometric home view sees 3 faces (TOP, FRONT, RIGHT).
  const iso = projectCube(homeHeading());
  assert.equal(iso.length, 3);
  const faceIds = new Set(iso.map((f) => f.face.id));
  assert.deepEqual(faceIds, new Set(["top", "front", "right"]));
  for (const face of iso) {
    assert.equal(face.patches.length, 9);
    assert.equal(face.patches.filter((p) => p.kind === "face").length, 1);
    assert.equal(face.patches.filter((p) => p.kind === "edge").length, 4);
    assert.equal(face.patches.filter((p) => p.kind === "corner").length, 4);
    for (const p of face.patches) {
      assert.equal(p.points.length, 4);
    }
  }

  // Pure TOP view sees only 1 face ("top").
  const plan = projectCube({ yaw: 0, tilt: 0 });
  assert.equal(plan.length, 1);
  assert.equal(plan[0]!.face.id, "top");
});

test("toggleCardCollapse: toggles dataset.collapsed on any panel container", () => {
  assert.deepEqual(DEFAULT_PANEL_IDS, ["planes", "inspector", "sources", "legend"]);
  const fakePanel = { dataset: {} as Record<string, string | undefined> } as unknown as HTMLElement;
  assert.equal(toggleCardCollapse(fakePanel), true);
  assert.equal(fakePanel.dataset["collapsed"], "true");
  assert.equal(toggleCardCollapse(fakePanel), false);
  assert.equal(fakePanel.dataset["collapsed"], undefined);
});
