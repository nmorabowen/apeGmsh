// VA6d (#2766, G10): derived-scalar element contours (`von_mises_stress`,
// `von_mises_strain`) against Python, through the app's own path: the catalog
// (state/contour.ts `elementContourFields`), the effects' tensor read and
// per-Gauss-point computation (state/tensor.ts `derivedGroupValues`), the
// history scan, and, for a spec, the one compiler and the manifest.
//
// The oracle is v1's own default read, `stage.elements.gauss.get(component=name)`
// (`_composites._compute_derived`, `plane="auto"`: a 2-D element's zz recovered
// per element by `_plane_recovery.inject_out_of_plane` from its plane type
// and nu, 0 where the model does not classify it; the strain's engineering
// shear halved), reduced as the manifest oracle reduces a stored component
// (tests/viewers/test_agent_manifest_oracle.py `_element_case`: the element
// mean for `gauss: "cell"`, `extrapolate_gauss_slab_to_nodes` for `"node"`).
// `elem_mixed` carries an /opensees zone (quad and tri31, PlaneStrain,
// ElasticIsotropic nu = 0.2), so its sigma_zz is recovered;
// `ladruno_quad2d.h5` and `extrap_plane.h5` carry none, so v1 classifies none
// of their elements (it warns) and zz stays 0 there. Each set is read in a
// process of its own: `_plane_recovery._CACHE` is keyed on the FEM's
// snapshot id, which `extrap_plane` shares with `elem_mixed`, so a process that
// read `elem_mixed` first hands `extrap_plane` its plane map (a v1 cache
// fault, outside this slice). The script that printed these numbers is in the
// PR body (#2766); the fixtures are committed and unchanged.

import assert from "node:assert/strict";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { compileShot } from "../../src/spec/compile.ts";
import { describeSet } from "../../src/spec/describe.ts";
import { manifestEntry } from "../../src/spec/manifest.ts";
import { sameView, viewShotOf } from "../../src/spec/snapshot.ts";
import { validateSpec, type SpecShot } from "../../src/spec/spec.ts";
import { viewVerdict } from "../../src/spec/verdict.ts";
import { elementContourFields, gaussFields } from "../../src/state/contour.ts";
import { contourFieldsOf, contourView } from "../../src/state/selectors.ts";
import type { Event } from "../../src/state/types.ts";
import { opened, until } from "./harness.ts";

const fx = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const close = (a: number, b: number, what: string) => assert.ok(Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b)), `${what}: ${a}, Python ${b}`);

type Opened = Awaited<ReturnType<typeof opened>>;
const open: Opened[] = [];
after(() => {
  for (const h of open) {
    h.effects.dispose();
    h.host.close();
  }
});
async function openSet(model: string, results: string): Promise<Opened> {
  const h = await opened(fx(model), fx(results));
  open.push(h);
  return h;
}

// ---- ladruno_quad2d.ladruno: six plane-stress FourNodeQuad, 2 x 2 Gauss points, stage_0 of 2 steps ----

/** Python's derived value at each Gauss point of step 1, by element, in the natural order (-,-), (+,-), (+,+), (-,+). */
const QUAD_NATURAL = [
  [-1, -1],
  [1, -1],
  [1, 1],
  [-1, 1],
].map(([a, b]) => [a! * 0.5773502691896258, b! * 0.5773502691896258]);
const QUAD_STEP1: Record<"von_mises_stress" | "von_mises_strain", Record<number, number[]>> = {
  von_mises_stress: {
    1: [5.954662513942316, 5.10021352896747, 1.5443980351220203, 3.172849392076833],
    2: [3.9341330882645384, 3.1508976117084795, 1.3929173629106022, 2.9220492829781883],
    3: [2.652801997070772, 2.4194239501856134, 1.6604633165925808, 1.800139740446825],
    4: [3.172849392076834, 1.5443980351220206, 5.100213528967469, 5.954662513942316],
    5: [2.922049282978189, 1.3929173629106022, 3.1508976117084804, 3.934133088264537],
    6: [1.8001397404468227, 1.6604633165925797, 2.419423950185611, 2.652801997070771],
  },
  von_mises_strain: {
    1: [0.00443654613295378, 0.003755136933842203, 0.0012512164895117136, 0.0025896768121821635],
    2: [0.0030924328108610506, 0.002272858224463798, 0.0011254197185736174, 0.0024343484103592334],
    3: [0.0021756692427707205, 0.0020161860182505297, 0.0013254678489811476, 0.0015001000736771398],
    4: [0.002589676812182164, 0.0012512164895117138, 0.0037551369338422033, 0.00443654613295378],
    5: [0.002434348410359234, 0.0011254197185736174, 0.002272858224463799, 0.003092432810861049],
    6: [0.0015001000736771383, 0.0013254678489811468, 0.002016186018250528, 0.0021756692427707196],
  },
};
/** The whole-history range of the element means (both steps), Python. */
const QUAD_HISTORY = {
  von_mises_stress: { min: 1.066603625536972, max: 3.9430308675271597 },
  von_mises_strain: { min: 0.0008771778979599407, max: 0.0030081440921224654 },
} as const;

test("the catalog offers each complete tensor's von Mises after the stored components; the stored catalog is unchanged", async () => {
  const h = await openSet("ladruno_quad2d.h5", "ladruno_quad2d.ladruno");
  const gauss = h.store.get().results!.stages[0]!.gauss;
  const fields = elementContourFields(gauss);
  assert.deepEqual(
    fields.map((f) => [f.name, f.components.map((c) => c.key), f.derivedRefused]),
    [
      ["stress", ["stress/sigma11", "stress/sigma22", "stress/sigma12", "von_mises_stress"], []],
      ["strain", ["strain/eta11", "strain/eta22", "strain/eta12", "von_mises_strain"], []],
    ],
  );
  const vm = fields[0]!.components[3]!;
  assert.deepEqual([vm.label, vm.datasets, vm.derived?.name, vm.derived?.tensor.family], ["von_mises_stress", ["stress/sigma11", "stress/sigma22", "stress/sigma12"], "von_mises_stress", "stress"]);
  // gaussFields is the stored set main reads: a derived name is never asked of main.
  assert.deepEqual(gaussFields(gauss).map((f) => f.components.length), [3, 3]);
});

for (const name of ["von_mises_stress", "von_mises_strain"] as const) {
  test(`oracle: ${name} on ladruno_quad2d equals v1's default read at every Gauss point (no /opensees zone: zz 0), and its history range`, async () => {
    const h = await openSet("ladruno_quad2d.h5", "ladruno_quad2d.ladruno");
    const field = name === "von_mises_stress" ? "stress" : "strain";
    for (const e of [
      { type: "setColourBy", by: "contour" },
      { type: "setPhase", at: { kind: "results", stage: 0, step: 1 } },
      { type: "setContourField", field, component: name, source: "element" },
      { type: "setContourRange", range: "history" },
    ] as Event[]) {
      h.store.dispatch(e);
    }
    await until(() => h.effects.currentStep()?.element === name && h.effects.currentStep()?.step === 1, `${name} at step 1`);
    const s = h.store.get();
    const v = contourView(s);
    assert.equal(v.state, "element", v.state === "element" ? "" : JSON.stringify(v));
    if (v.state !== "element") return;
    assert.equal(v.component.key, name);
    // The picker lists what the still draws (one catalog).
    assert.ok(contourFieldsOf(s)!.elements.find((f) => f.name === field)!.components.some((c) => c.key === name));
    const step = h.effects.currentStep()!;
    let checked = 0;
    for (const gd of v.groups) {
      const g = s.results!.stages[0]!.gauss.find((x) => x.id === gd.id)!;
      const eids = h.blobs.get(g.elementIds!) as Float64Array;
      const vals = step.gauss[g.id]!;
      assert.equal(vals.length, g.nElements * g.nGauss);
      for (let e = 0; e < g.nElements; e++) {
        for (let k = 0; k < g.nGauss; k++) {
          const nat = g.naturalCoords![k]!;
          const j = QUAD_NATURAL.findIndex((p) => Math.abs(p[0]! - nat[0]!) < 1e-9 && Math.abs(p[1]! - nat[1]!) < 1e-9);
          assert.ok(j >= 0, `natural ${nat}`);
          close(vals[e * g.nGauss + k]!, QUAD_STEP1[name][eids[e]!]![j]!, `${name} element ${eids[e]} point ${k}`);
          checked++;
        }
      }
    }
    assert.equal(checked, 24);
    await until(() => h.store.get().contour.scan?.key === v.keys.scan && h.store.get().contour.scan?.status !== "running", `${name}'s history scan`);
    const scan = h.store.get().contour.scan!;
    assert.equal(scan.status, "done", scan.error ?? "");
    close(scan.min!, QUAD_HISTORY[name].min, `${name} history min`);
    close(scan.max!, QUAD_HISTORY[name].max, `${name} history max`);
    // Main's scan was never asked for a derived name (it reads stored components only).
    assert.ok(h.calls.scans.every((x) => !x.components.includes(name)), JSON.stringify(h.calls.scans));
  });
}

// ---- a spec, end to end: compile -> events -> effects -> manifest, against Python ----

function shotOf(shot: Record<string, unknown>): SpecShot {
  const v = validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, shots: [{ name: "s", ...shot }] });
  if (!v.ok) throw new Error(`${v.key}: ${v.reason}`);
  return v.spec.shots[0]!;
}

/**
 * v1, per step: the reduced values' finite min and max and what holds them (element for `cell`, node for `node`),
 * and the out-of-plane statement the manifest must make (v1 warns "15 of 15 element(s) could not be classified" on
 * `extrap_plane`).
 */
const SPEC_CASES = [
  {
    name: "elem_mixed, gauss cell (element mean); PlaneStrain nu 0.2, so sigma_zz = nu (sxx + syy)",
    set: "elem_mixed",
    gauss: "cell",
    topology: "gauss_cell",
    steps: [
      { min: [4.976444513907494, 2], max: [30.63111163506803, 16] },
      { min: [6.297618597533516, 2], max: [31.93649949509182, 16] },
      { min: [7.623975340988453, 2], max: [33.24432282360403, 16] },
    ],
    plane: { elements: 15, recorded: 0, recovered: 15, zero_exact: 0, unclassified: 0, no_nu: 0 },
    warning: null,
  },
  {
    name: "extrap_plane, gauss node (derived at the Gauss points, then extrapolated and averaged); no /opensees zone, so zz = 0",
    set: "extrap_plane",
    gauss: "node",
    topology: "gauss_node",
    steps: [
      { min: [5.591334441198571, 13], max: [48.19958506045462, 5] },
      { min: [11.182668882397142, 13], max: [96.39917012090925, 5] },
      { min: [16.774003323595714, 13], max: [144.59875518136386, 5] },
    ],
    plane: { elements: 15, recorded: 0, recovered: 0, zero_exact: 0, unclassified: 15, no_nu: 0 },
    warning: "Out-of-plane stress_zz recovery is incomplete: 15 of 15 element(s) could not be classified as plane stress / plane strain. stress_zz is left at 0 there.",
  },
] as const;

for (const c of SPEC_CASES) {
  test(`spec oracle: von_mises_stress on ${c.name} equals Python at step 2 and over the history`, async () => {
    const h = await openSet(`${c.set}.h5`, `${c.set}.results.h5`);
    const shot = shotOf({ stage: "load", step: 2, contour: { field: "stress", component: "von_mises_stress", gauss: c.gauss } });
    const d = describeSet(h.store.get(), []);
    // --describe offers it, saying what it is computed from.
    const described = d.stages[0]!.fields.find((f) => f.name === "stress" && f.source === "element")!.components.find((x) => x.key === "von_mises_stress");
    assert.match(
      described?.derived ?? "",
      /von Mises equivalent of the stress tensor \(stress_xx, stress_yy, stress_xy\) at each Gauss point; stress_zz, not recorded, is recovered per element from its plane type and Poisson's ratio .*left at 0 for an element the model does not classify/,
    );
    const compiled = compileShot(shot, d);
    assert.ok(compiled.ok, compiled.ok ? "" : compiled.reason);
    if (!compiled.ok) return;
    for (const e of compiled.plan.events) h.store.dispatch(e);
    const stepHere = () => {
      const at = h.store.get().phase.at;
      const step = h.effects.currentStep();
      return at?.kind === "results" && step?.stage === at.stage && step.step === at.step;
    };
    await until(() => viewVerdict(h.store.get(), shot, compiled.plan, { contour: stepHere(), deform: true }).kind === "shown", "the derived shot to be shown");
    const entry = manifestEntry(h.store.get(), {
      shot,
      png: "s.png",
      image: [1600, 1000],
      camera: { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1], fov: 30 },
      layout: null,
      blobs: h.blobs,
      step: h.effects.currentStep(),
    });
    const k = entry.contour;
    assert.ok(k?.drawn, JSON.stringify(k));
    if (!k.drawn) return;
    assert.deepEqual([k.component, k.label, k.source, k.topology], ["von_mises_stress", "von_mises_stress", "element", c.topology]);
    // F1: what the out-of-plane normal is, by element; never silent on a zz left at 0 the model does not vouch for.
    assert.deepEqual(k.derived, { from: ["stress_xx", "stress_yy", "stress_xy"], plane: c.plane, warning: c.warning });
    const want = c.steps[2]!;
    const holder = (p: { node: number | null; element: number | null }) => (c.gauss === "cell" ? p.element : p.node);
    close(k.extrema.recorded!.min.value, want.min[0], "min");
    close(k.extrema.recorded!.max.value, want.max[0], "max");
    assert.deepEqual([holder(k.extrema.recorded!.min), holder(k.extrema.recorded!.max)], [want.min[1], want.max[1]]);
    assert.equal(k.range.mode, "history");
    assert.equal(k.range.history_scan.status, "done");
    close(k.range.min!, Math.min(...c.steps.map((x) => x.min[0])), "history min");
    close(k.range.max!, Math.max(...c.steps.map((x) => x.max[0])), "history max");
    // INV-2/INV-3: the view read back from the window's state names the derived scalar and compiles to the same view.
    const back = viewShotOf(h.store.get(), { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1] });
    assert.deepEqual(back.outside, []);
    assert.equal(back.shot.contour?.component, "von_mises_stress");
    assert.ok(sameView(back.shot, shot, describeSet(h.store.get(), [])), "the read-back view draws what the shot drew");
  });
}

// ---- a model re-read that changes the plane map draws, measures and scans again (#2921 review) ----

test("a model re-read with another plane type re-fetches the derived step and re-scans its history; never the old plane's values", async () => {
  const h = await openSet("elem_mixed.h5", "elem_mixed.results.h5");
  const shot = shotOf({ stage: "load", step: 2, contour: { field: "stress", component: "von_mises_stress", gauss: "cell" } });
  const compiled = compileShot(shot, describeSet(h.store.get(), []));
  assert.ok(compiled.ok);
  if (!compiled.ok) return;
  for (const e of compiled.plan.events) h.store.dispatch(e);
  const shown = () => {
    const s = h.store.get();
    const at = s.phase.at;
    const step = h.effects.currentStep();
    const here = at?.kind === "results" && step?.stage === at.stage && step.step === at.step;
    return viewVerdict(s, shot, compiled.plan, { contour: here, deform: true }).kind === "shown";
  };
  const entry = () => {
    const k = manifestEntry(h.store.get(), { shot, png: "s.png", image: [1600, 1000], camera: { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1], fov: 30 }, layout: null, blobs: h.blobs, step: h.effects.currentStep() }).contour;
    assert.ok(k?.drawn);
    return k as Extract<typeof k, { drawn: true }>;
  };
  await until(shown, "the plane-strain shot");
  close(entry().extrema.recorded!.max.value, 33.24432282360403, "plane strain max");
  // The same model path read again, its quads and triangles now PlaneStress (v1: sigma_zz = 0, max 35.957).
  const bridge = (h.effects as unknown as { bridge: { openModel: (p: string) => Promise<{ ok: boolean; model?: { opensees?: { elementMeta: { args: (number | string)[][] }[] } } }> } }).bridge;
  const read = bridge.openModel.bind(bridge);
  bridge.openModel = async (p: string) => {
    const r = await read(p);
    for (const m of r.model?.opensees?.elementMeta ?? []) for (const a of m.args) a[1] = "PlaneStress";
    return r;
  };
  assert.ok(await h.effects.open(fx("elem_mixed.h5")));
  await until(() => shown() && Math.abs(entry().extrema.recorded!.max.value - 35.95657102672612) < 1e-9, "the plane-stress values after the re-read");
  const k = entry();
  close(k.extrema.recorded!.min.value, 8.023403267940607, "plane stress min");
  assert.deepEqual(k.derived?.plane, { elements: 15, recorded: 0, recovered: 0, zero_exact: 15, unclassified: 0, no_nu: 0 });
  // The history is scanned again for the new plane map, never the plane-strain range reused.
  assert.equal(k.range.history_scan.status, "done");
  close(k.range.min!, 5.326818938165629, "plane stress history min");
  close(k.range.max!, 35.95657102672612, "plane stress history max");
});

// ---- refusals: by name, with why; never a value from partial components ----

test("a derived name on a field without a complete tensor is refused by name with why (elem_solid records only stress_xx and stress_yy)", async () => {
  const h = await openSet("elem_solid.h5", "elem_solid.results.h5");
  const s = h.store.get();
  const f = elementContourFields(s.results!.stages[0]!.gauss).find((x) => x.name === "stress")!;
  assert.deepEqual(f.components.map((c) => c.key), ["stress_xx", "stress_yy"]);
  assert.deepEqual(f.derivedRefused?.map((r) => r.name), ["von_mises_stress"]);
  // The picker never offers it, and the reducer refuses it.
  assert.ok(!contourFieldsOf({ ...s, phase: { ...s.phase, at: { kind: "results", stage: 0, step: 0 } } })!.elements.some((x) => x.components.some((c) => c.key === "von_mises_stress")));
  assert.throws(() => h.store.dispatch({ type: "setContourField", field: "stress", component: "von_mises_stress", source: "element" }), /is not a element field component/);
  const d = describeSet(s, []);
  assert.deepEqual(d.stages[0]!.fields.find((x) => x.name === "stress")!.refusedComponents, [
    { key: "von_mises_stress", reason: "group hex_G8: no xy stress component, so its tensor is not known" },
  ]);
  const c = compileShot(shotOf({ stage: "load", step: 1, contour: { field: "stress", component: "von_mises_stress" } }), d);
  assert.deepEqual(c, { ok: false, reason: 'element field "stress" of stage "load" cannot give "von_mises_stress": group hex_G8: no xy stress component, so its tensor is not known' });
});
