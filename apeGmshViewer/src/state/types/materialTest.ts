// The material tester slice (ADR 0135): one run of the fork's uniaxial tester on
// one declared material. One slice of the store's types; src/state/types.ts
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

export interface MaterialTestRun {
  id: number;
  decl: DeclPath;
  /** the OpenSees type of the material under test */
  type: string;
  protocol: MaterialTestProtocol;
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

/** A curve to draw: strain and stress, of one length. */
export interface MaterialTestPoints {
  strain: readonly number[];
  stress: readonly number[];
}

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
}

export interface MaterialTestState {
  run: MaterialTestRun | null;
  /** the open form; null when none is */
  form: MaterialTestForm | null;
  /** the next run's id: never reused, so a late answer cannot match a newer run */
  nextId: number;
}
