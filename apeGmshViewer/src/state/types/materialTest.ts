// The material tester slice (ADR 0135): the runs of the fork's uniaxial tester on
// declared materials, kept for overlay (D7, Amendment 1 slice A2). One slice of the store's types; src/state/types.ts
// re-exports it. The curves live in the BlobStore; the state holds their refs.

import type { BlobRef, DeclPath } from "./model.ts";

/** The imposed strain history (ADR 0135 D6). Every path starts from zero. */
export type MaterialTestProtocol =
  | { kind: "monotonic"; peak: number; steps: number }
  /** each amplitude once: 0 to +a to -a to 0, `stepsPerRamp` steps per quarter cycle */
  | { kind: "cyclic"; amplitudes: number[]; stepsPerRamp: number }
  | { kind: "values"; values: number[] };

/** One `uniaxialMaterial` command, as `/opensees/materials` stores it. */
export interface MaterialTestLine {
  type: string;
  tag: number;
  args: (number | string)[];
}

/** What the page asks main to run. Main validates all of it again (ADR 0135 D3). */
export interface MaterialTestJob {
  /** the material under test; its line is in `materials` */
  target: number;
  /** dependencies first, the target last */
  materials: MaterialTestLine[];
  protocol: MaterialTestProtocol;
}

/** A numeric argument slot of the material under test, replaced in one test only (D6, D8). */
export interface MaterialTestOverride {
  /** the argument's index in the material's command, after the tag */
  slot: number;
  /** the slot's label (`params_names`, or flag and position) */
  label: string;
  /** the declared value, from the file */
  declared: number;
  value: number;
}

/** One numeric argument slot of a material's command, as the form lists it (D6). */
export interface MaterialTestSlot {
  slot: number;
  label: string;
  /** the declared value */
  value: number;
  /** why the slot cannot be overridden (it holds a dependency's tag); null when it can */
  locked: string | null;
}

export interface MaterialTestRun {
  id: number;
  decl: DeclPath;
  /** the OpenSees type of the material under test */
  type: string;
  protocol: MaterialTestProtocol;
  /** the slots this run replaced; [] for the declared material */
  overrides: readonly MaterialTestOverride[];
  status: "running" | "ready" | "failed";
  /** why there is no curve; null unless failed */
  error: string | null;
  /** `ladrunoBuild` of the executable that ran it (ADR 0135 D1) */
  build: string | null;
  /** the three arrays, of one length; null unless ready */
  curves: { strain: BlobRef; stress: BlobRef; tangent: BlobRef } | null;
  /** the stress-strain curve thinned for drawing (every reversal kept); null unless ready. The full arrays stay in `curves` (ADR 0137 D6: the reduced values a tool draws) */
  points: MaterialTestPoints | null;
}

/** A run thinned for drawing: the same steps of every array, all of one length. */
export interface MaterialTestPoints {
  /** the step of each point (0 is the starting zero) */
  step: readonly number[];
  strain: readonly number[];
  stress: readonly number[];
  tangent: readonly number[];
  /** the work done, the running trapezoidal integral of stress over strain, computed in the app on the full run (D7) */
  energy: readonly number[];
}

/** The tester's second plot, against the step (D7). */
export type MaterialTestSecond = "history" | "tangent" | "energy";

/**
 * The tester's form in the materials window (ADR 0135 Amendment 1), as typed: the
 * fields are strings, so a half-typed number survives the window's redraw. One form
 * is open at a time, under the row of `decl`.
 */
export interface MaterialTestForm {
  decl: DeclPath;
  kind: MaterialTestProtocol["kind"];
  peak: string;
  steps: string;
  /** comma- or space-separated */
  amplitudes: string;
  stepsPerRamp: string;
  /** comma-, space- or newline-separated */
  values: string;
  /** what the second plot shows */
  second: MaterialTestSecond;
  /** typed override per slot index; a slot absent, blank or equal to its declared value is not overridden */
  overrides: Readonly<Record<string, string>>;
}

/** The most runs kept at once, over all materials; the oldest leaves first. */
export const MAX_MATERIAL_TEST_RUNS = 6;

export interface MaterialTestState {
  /** oldest first; at most one is running, and it is the last */
  runs: readonly MaterialTestRun[];
  /** the open form; null when none is */
  form: MaterialTestForm | null;
  /** the next run's id: never reused, so a late answer cannot match a newer run */
  nextId: number;
  /** the run whose CSV the effect is saving; null when none */
  saving: number | null;
  /** what the last save said ("saved to …", or why not); null when there is nothing to say */
  notice: string | null;
}
