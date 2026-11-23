// The material tester's stills (`npm run capture`, ADR 0135 Amendment 1), only on a
// model that declares a uniaxial material: `<out>.mt-<n>-<what>.png`.
//
// The runs are real: the capture window's preload reaches main's runner, so the
// curves come from the installed fork. Without one, each still shows the run's
// refusal, which is the screen a machine without the fork gets.

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
export async function captureMaterialTest(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>) {
  const { store, viewport, bridge } = ctx;
  const materials = Object.values(store.get().decls).filter((d) => d.kind === "uniaxialMaterial").slice(0, 8);
  if (materials.length === 0) return;
  const still = async (name: string) => {
    await settle(150);
    document.querySelector("#windows .mt")?.scrollIntoView({ block: "center" });
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  store.dispatch({ type: "openWindow", window: "materials" });
  const first = materials[0]!;
  store.dispatch({ type: "materialTestFormSet", form: { ...materialTestFormFor(first.path), ...SHOWN[first.type] } });
  await still("mt-1-form");
  /** Set the form, press its Run, and wait for the answer. */
  const run = async (form: MaterialTestForm) => {
    store.dispatch({ type: "materialTestFormSet", form });
    const id = store.get().materialTest.nextId;
    document.querySelector<HTMLButtonElement>(`#windows .mt[data-material-test="${CSS.escape(form.decl)}"] .mt-run-button`)?.click();
    await until(() => {
      const r = store.get().materialTest.runs.find((x) => x.id === id);
      return r !== undefined && r.status !== "running";
    }, 15000);
  };
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
