// The model-definition window registry (ADR 0112 D2, D6, D7): one entry per
// `WindowKind`, each a title and a pure view in its own file. The mapped type
// makes a kind without a view a compile error; test/zones/definitions.test.ts
// checks the keys against `WINDOW_KINDS` at run time. A V3 slice that owns a
// window replaces only its `<kind>.ts`.

import type { WindowKind } from "../../state/types.ts";
import { view as analysis } from "./analysis.ts";
import { view as materials } from "./materials.ts";
import { view as patterns } from "./patterns.ts";
import { view as recorders } from "./recorders.ts";
import { view as search } from "./search.ts";
import { view as stages } from "./stages.ts";
import type { WindowView } from "./view.ts";

export interface WindowEntry {
  title: string;
  view: WindowView;
}

export const WINDOWS: { readonly [K in WindowKind]: WindowEntry } = {
  stages: { title: "Stages", view: stages },
  analysis: { title: "Analysis", view: analysis },
  patterns: { title: "Patterns & series", view: patterns },
  recorders: { title: "Recorders", view: recorders },
  materials: { title: "Materials & sections", view: materials },
  search: { title: "Search", view: search },
};
