// Which stage the `.ladruno` read row times (#2669). Pure, so the tests run it
// without Electron.

/** The translational displacement components, under the .ladruno and the native reader's names. */
export const DISPLACEMENT = /^(DISPLACEMENT\/U[xyz]|displacement_[xyz])$/;

interface StageLike {
  id: string;
  kind: string;
  nodes?: { components: readonly string[] } | null;
}

/** The first non-mode stage with nodal displacements, or why the file has no read row (an eigen-only run has none). */
export function readRowStage<S extends StageLike>(path: string, stages: readonly S[]): { stage: S } | { skip: string } {
  const stage = stages.find((st) => st.kind !== "mode" && st.nodes?.components.some((c) => DISPLACEMENT.test(c)));
  if (stage) return { stage };
  const modes = stages.filter((st) => st.kind === "mode").length;
  const why = stages.length === 0 ? "it has no stages" : modes === stages.length ? `its ${modes} stages are all mode shapes` : "no stage records nodal displacements";
  return { skip: `${path}: no read row, because ${why}` };
}
