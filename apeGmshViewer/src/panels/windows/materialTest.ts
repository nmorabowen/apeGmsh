// The material tester's interim surface (ADR 0135 Amendment 1): under a uniaxial
// row of the materials window, a form for the strain protocol, a Run button, the
// run's state and the response curve. The fork computes; this only dispatches
// events and draws the state (ADR 0112 D6), so it moves to ADR 0137's tool host
// as it is.
//
// The window is rebuilt on every state change, so what is typed lives in the
// store (`materialTestFormSet`), and each field carries a focus key the frame
// puts the caret back on.

import type { Event, State } from "../../state/store.ts";
import { materialTestAxesOf, materialTestFormFor, materialTestProtocolOf, materialTestStepsOf } from "../../state/selectors.ts";
import type { DeclPath, MaterialTestForm, MaterialTestRun } from "../../state/types.ts";
import { el } from "../../ui/dom.ts";
import { fmt, xyPlot } from "../../ui/xyPlot.ts";

const KINDS: readonly (readonly [MaterialTestForm["kind"], string])[] = [
  ["monotonic", "monotonic"],
  ["cyclic", "cyclic"],
  ["values", "values"],
];

type Field = "peak" | "steps" | "amplitudes" | "stepsPerRamp" | "values";

/** Keep a click inside the tester from selecting the row under it. */
const contained = <T extends HTMLElement>(e: T): T => {
  e.addEventListener("click", (ev) => ev.stopPropagation());
  return e;
};

function input(f: MaterialTestForm, field: Field, label: string, title: string, act: (e: Event) => void, area = false): HTMLElement {
  const wrap = el("label", "mt-field");
  const box = contained(el(area ? "textarea" : "input", "mt-input") as HTMLInputElement);
  if (!area) box.type = "text";
  box.value = f[field];
  box.title = title;
  box.spellcheck = false;
  box.dataset["focusKey"] = `mt-${field}`;
  box.addEventListener("input", () => act({ type: "materialTestFormSet", form: { ...f, [field]: box.value } }));
  wrap.append(el("span", "mt-label", label), box);
  return wrap;
}

function runEl(run: MaterialTestRun, xName: string, yName: string): HTMLElement {
  const box = el("div", "mt-run");
  if (run.status === "running") {
    box.append(el("div", "window-note", "running on the fork…"));
    return box;
  }
  if (run.status === "failed") {
    box.append(el("div", "mt-error", run.error ?? "the run failed"));
    if (run.build) box.append(el("div", "window-row-decl", `fork build ${run.build.slice(0, 8)}`));
    return box;
  }
  const p = run.points!;
  let peak = 0;
  for (const y of p.stress) if (Math.abs(y) > Math.abs(peak)) peak = y;
  box.append(
    xyPlot({ xs: p.strain, ys: p.stress, xName, yName, title: `${yName} against ${xName}, ${run.type}` }),
    el("div", "window-row-decl", `fork build ${(run.build ?? "?").slice(0, 8)} · ${materialTestStepsOf(run.protocol).toLocaleString()} steps · peak ${yName} ${fmt(peak)}`),
  );
  return box;
}

/** The Test button of a uniaxial row's head: opens its form, or closes the one open there. */
export function testButton(s: State, path: DeclPath, act: (e: Event) => void): HTMLElement {
  const open = s.materialTest.form?.decl === path;
  const b = contained(el("button", "window-test" + (open ? " on" : ""), "Test"));
  b.title = open ? "close the material tester" : "run this material on the installed Ladruno fork and plot its response";
  b.addEventListener("click", () => act({ type: "materialTestFormSet", form: open ? null : materialTestFormFor(path) }));
  return b;
}

/** The tester under the row of `path`, or null when its form is not open. */
export function testerEl(s: State, path: DeclPath, type: string, act: (e: Event) => void): HTMLElement | null {
  const f = s.materialTest.form;
  if (!f || f.decl !== path) return null;
  const [xName, yName] = materialTestAxesOf(type);
  const box = contained(el("div", "mt"));
  box.dataset["materialTest"] = path;

  const kind = contained(el("select", "mt-kind") as HTMLSelectElement);
  kind.title = "the imposed history, from zero";
  for (const [v, label] of KINDS) {
    const o = el("option", undefined, label) as HTMLOptionElement;
    o.value = v;
    o.selected = v === f.kind;
    kind.append(o);
  }
  kind.addEventListener("change", () => act({ type: "materialTestFormSet", form: { ...f, kind: kind.value as MaterialTestForm["kind"] } }));
  const fields = el("div", "mt-fields");
  fields.append(kind);
  if (f.kind === "monotonic") {
    fields.append(input(f, "peak", `peak ${xName}`, `the ${xName} the ramp ends at`, act), input(f, "steps", "steps", "steps from zero to the peak", act));
  } else if (f.kind === "cyclic") {
    fields.append(
      input(f, "amplitudes", `amplitudes (${xName})`, "each amplitude once: 0 to +a to -a to 0; separate them with commas or spaces", act),
      input(f, "stepsPerRamp", "steps per ramp", "steps per quarter cycle", act),
    );
  } else {
    fields.append(input(f, "values", `${xName} values`, `the ${xName} of each step after zero; separate them with commas, spaces or new lines`, act, true));
  }
  box.append(fields);

  const asked = materialTestProtocolOf(f, xName);
  const bar = el("div", "mt-bar");
  const run = contained(el("button", "window-test", "Run") as HTMLButtonElement);
  run.disabled = "error" in asked;
  run.title = "error" in asked ? asked.error : "run the fork's uniaxial tester; a run still going is replaced";
  if ("protocol" in asked) run.addEventListener("click", () => act({ type: "materialTestRequested", decl: path, protocol: asked.protocol }));
  bar.append(run, el("span", "error" in asked ? "mt-error" : "window-row-decl", "error" in asked ? asked.error : `${materialTestStepsOf(asked.protocol).toLocaleString()} steps · values in the model's units`));
  box.append(bar);

  const r = s.materialTest.run;
  if (r && r.decl === path) box.append(runEl(r, xName, yName));
  return box;
}
