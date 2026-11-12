// `State["visibility"]` (types/state.ts), named in a file of its own so the
// offers slice's `ViewMemo` can name it without importing `State`: offers.ts
// importing state.ts, which imports offers.ts, is a cycle the lint refuses.
// Not re-exported by src/state/types.ts: `State["visibility"]` stays its public name.

import type { ColourBy, DeclPath } from "./model.ts";

export type Visibility = {
  hidden: DeclPath[];
  edges: boolean;
  opacity: number;
  colourBy: ColourBy;
  extrudeFrames: boolean;
  showOffsets: boolean;
};
