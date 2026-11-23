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
 * then `mt-<n>-refused`: a field that does not parse, Run disabled.
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
  let n = 2;
  for (const d of materials) {
    const form = { ...materialTestFormFor(d.path), ...SHOWN[d.type] };
    store.dispatch({ type: "materialTestFormSet", form });
    const id = store.get().materialTest.nextId;
    document.querySelector<HTMLButtonElement>(`#windows .mt[data-material-test="${CSS.escape(d.path)}"] .mt-bar .window-test`)?.click();
    await until(() => {
      const run = store.get().materialTest.run;
      return run !== null && run.id === id && run.status !== "running";
    }, 15000);
    await still(`mt-${n++}-${d.path.split("/").pop()!.replace(/[^A-Za-z0-9_-]/g, "_")}`);
  }
  store.dispatch({ type: "materialTestFormSet", form: { ...materialTestFormFor(first.path), kind: "monotonic", peak: "0.01x" } });
  await still(`mt-${n}-refused`);
  store.dispatch({ type: "materialTestCleared" });
  store.dispatch({ type: "materialTestFormSet", form: null });
  store.dispatch({ type: "closeWindow", window: "materials" });
  await settle();
}
