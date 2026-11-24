// The material tester's stills (`npm run capture`, ADR 0135 Amendment 1), only on a
// model that declares a uniaxial material: `<out>.mt-<n>-<what>.png`; and, on a model
// that declares an nD material (S5), `<out>.mt-nd-<what>.png`.
//
// The runs are real: the capture window's preload reaches main's runner, so the
// curves come from the installed fork. Without one, each still shows the run's
// refusal, which is the screen a machine without the fork gets.

import { materialTestSlotsOf } from "../../state/materialTest.ts";
import { materialTestFormFor } from "../../state/materialTestForm.ts";
import type { MaterialTestForm } from "../../state/types.ts";
import { painted, until, type HarnessContext } from "./context.ts";

/** A protocol that shows each first material's behaviour; others take the form's default. */
const SHOWN: Readonly<Record<string, Partial<MaterialTestForm>>> = {
  Steel02: { kind: "cyclic", amplitudes: "0.005, 0.01, 0.02", stepsPerRamp: "100" },
  LadrunoUniaxialJ2: { kind: "cyclic", amplitudes: "0.005, 0.01, 0.02", stepsPerRamp: "100" },
  LadrunoRebarBuckling: { kind: "cyclic", amplitudes: "0.005, 0.01, 0.02", stepsPerRamp: "100" },
  Concrete02: { kind: "cyclic", amplitudes: "0.001, 0.002, 0.004", stepsPerRamp: "100" },
  LadrunoBondSlip: { kind: "monotonic", peak: "6", steps: "300" },
  LadrunoCohesiveHinge: { kind: "cyclic", amplitudes: "0.02, 0.05", stepsPerRamp: "100" },
};

/**
 * With the materials window open:
 * (1) `mt-1-form`: the first uniaxial material's form, before a run;
 * (2..) `mt-<n>-<handle>`: each uniaxial material (at most 8) run and drawn;
 * then `mt-<n>-refused`: a field that does not parse, Run disabled;
 * then, on the first Steel02 (or the first material), three runs overlaid
 * (`mt-<n>-overlay`) with the second plot on the history, the tangent
 * (`mt-<n>-tangent`) and the work done (`mt-<n>-energy`); then the declared
 * curve against an override of fy and b, with the changed values to copy
 * (`mt-<n>-override`); then stress control: Concrete02 to -29 and to -35 (stopped
 * past its peak) and a J2 stress cycle (`mt-<n>-stress-*`).
 * The window, form and run are put back afterwards.
 */
/** A still of the open tester, scrolled into view. */
const stillOf = (ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>) => async (name: string) => {
  await settle(150);
  document.querySelector("#windows .mt")?.scrollIntoView({ block: "center" });
  ctx.viewport.renderNow();
  await painted();
  await painted();
  await ctx.bridge.captureStill(name);
};

/** Set the form, press its Run, and wait for the answer. */
const runOf = (ctx: HarnessContext) => async (form: MaterialTestForm) => {
  const { store } = ctx;
  store.dispatch({ type: "materialTestFormSet", form });
  const id = store.get().materialTest.nextId;
  document.querySelector<HTMLButtonElement>(`#windows .mt[data-material-test="${CSS.escape(form.decl)}"] .mt-run-button`)?.click();
  await until(`capture: the material tester run of ${form.decl} to answer`, () => {
    const r = store.get().materialTest.runs.find((x) => x.id === id);
    return r !== undefined && r.status !== "running";
  }, 15000);
};

export async function captureMaterialTest(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>) {
  const { store } = ctx;
  const materials = Object.values(store.get().decls).filter((d) => d.kind === "uniaxialMaterial").slice(0, 8);
  if (materials.length === 0) return;
  const still = stillOf(ctx, settle);
  const run = runOf(ctx);
  store.dispatch({ type: "openWindow", window: "materials" });
  const first = materials[0]!;
  store.dispatch({ type: "materialTestFormSet", form: { ...materialTestFormFor(first.path), ...SHOWN[first.type] } });
  await still("mt-1-form");
  let n = 2;
  for (const d of materials) {
    store.dispatch({ type: "materialTestCleared" });
    await run({ ...materialTestFormFor(d.path), ...SHOWN[d.type] });
    await still(`mt-${n++}-${d.path.split("/").pop()!.replace(/[^A-Za-z0-9_-]/g, "_")}`);
  }
  store.dispatch({ type: "materialTestCleared" });
  const steel = materials.find((d) => d.type === "Steel02") ?? first;
  const base = materialTestFormFor(steel.path);
  await run({ ...base, kind: "monotonic", peak: "0.03", steps: "300" });
  await run({ ...base, kind: "cyclic", amplitudes: "0.01", stepsPerRamp: "100" });
  await run({ ...base, ...SHOWN[steel.type] });
  await still(`mt-${n++}-overlay`);
  store.dispatch({ type: "materialTestFormSet", form: { ...store.get().materialTest.form!, second: "tangent" } });
  await still(`mt-${n++}-tangent`);
  store.dispatch({ type: "materialTestFormSet", form: { ...store.get().materialTest.form!, second: "energy" } });
  await still(`mt-${n++}-energy`);
  // A3: the declared curve against two overrides (Steel02: fy and b), the changed values shown to copy.
  store.dispatch({ type: "materialTestCleared" });
  const cyc = { ...base, ...SHOWN[steel.type] };
  await run(cyc);
  await run({ ...cyc, overrides: steel.type === "Steel02" ? { "0": "600", "2": "0.03" } : {} });
  await still(`mt-${n++}-override`);
  // S3: stress control. Concrete02 reaches -29 and stops before -35 (past its peak); J2 follows a stress cycle.
  for (const [type, targets, tag] of [["Concrete02", "-29", "stress-reached"], ["Concrete02", "-35", "stress-stopped"], ["LadrunoUniaxialJ2", "450, -450, 0", "stress-cycle"]] as const) {
    const d = materials.find((m) => m.type === type);
    if (!d) continue;
    store.dispatch({ type: "materialTestCleared" });
    await run({ ...materialTestFormFor(d.path), kind: "stress", targets, stepsPerRamp: "10" });
    await still(`mt-${n++}-${tag}`);
  }
  store.dispatch({ type: "materialTestFormSet", form: { ...materialTestFormFor(first.path), kind: "monotonic", peak: "0.01x" } });
  await still(`mt-${n}-refused`);
  store.dispatch({ type: "materialTestCleared" });
  store.dispatch({ type: "materialTestFormSet", form: null });
  store.dispatch({ type: "closeWindow", window: "materials" });
  await settle();
}

/**
 * S5, with the materials window open, on the first nD material other than
 * ElasticIsotropic (else the first): `mt-nd-form` (the nD form, before a run);
 * `mt-nd-uniaxial` (uniaxial compression past the peak); `mt-nd-confined` (the
 * same, then triaxial at p = 5, overlaid); `mt-nd-cyclic` (a cyclic uniaxial
 * stress in plane stress); `mt-nd-shear` (simple shear, tangent plot);
 * `mt-nd-regularization` (with -autoRegularization: the 1-unit softening against
 * the slot at 1); and `mt-nd-refused` (triaxial asked of the plane-stress view).
 */
export async function captureMaterialTestNd(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>) {
  const { store } = ctx;
  const nds = Object.values(store.get().decls).filter((d) => d.kind === "nDMaterial");
  const d = nds.find((x) => x.type !== "ElasticIsotropic") ?? nds[0];
  if (!d) return;
  const still = stillOf(ctx, settle);
  const run = runOf(ctx);
  store.dispatch({ type: "openWindow", window: "materials" });
  const base: MaterialTestForm = { ...materialTestFormFor(d.path), kind: "monotonic", peak: "-0.004", steps: "200" };
  store.dispatch({ type: "materialTestFormSet", form: base });
  await still("mt-nd-form");
  store.dispatch({ type: "materialTestCleared" });
  await run(base);
  await still("mt-nd-uniaxial");
  await run({ ...base, mode: "triaxial", confinement: "5" });
  await still("mt-nd-confined");
  store.dispatch({ type: "materialTestCleared" });
  await run({ ...base, kind: "cyclic", amplitudes: "0.0001, 0.0005, 0.002", stepsPerRamp: "50", view: "PlaneStress" });
  await still("mt-nd-cyclic");
  store.dispatch({ type: "materialTestCleared" });
  await run({ ...base, peak: "0.002", mode: "simpleShear", second: "tangent" });
  await still("mt-nd-shear");
  // -autoRegularization with no element: the softening of a 1-unit element, against the slot at 1 (the curve as declared)
  const reg = materialTestSlotsOf(store.get(), d.path).find((x) => x.label.startsWith("-autoRegularization"));
  if (reg) {
    store.dispatch({ type: "materialTestCleared" });
    const tension: MaterialTestForm = { ...base, kind: "monotonic", peak: "0.001", steps: "100" };
    await run(tension);
    await run({ ...tension, overrides: { [String(reg.slot)]: "1" } });
    await still("mt-nd-regularization");
  }
  store.dispatch({ type: "materialTestCleared" });
  store.dispatch({ type: "materialTestFormSet", form: { ...base, mode: "triaxial", confinement: "5", view: "PlaneStress" } });
  await still("mt-nd-refused");
  store.dispatch({ type: "materialTestCleared" });
  store.dispatch({ type: "materialTestFormSet", form: null });
  store.dispatch({ type: "closeWindow", window: "materials" });
  await settle();
}
