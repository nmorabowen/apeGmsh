// The inspector's dataset formatter. The inputs are the tables the reader
// builds (readTable): a plain array gets columns `value` or `[j]`, a compound
// one its field names. The expected strings are the file's values: San Ramon's
// 3-D transform stores per_element_vecxz = [[1.0, 0.0, 0.0]] (float64, (1, 3));
// a 2-D transform stores shape (1, 0) by design (no vecxz). The old formatter
// printed "1 rows ([0], [1], [2])" and "0 rows ()" for these.

import assert from "node:assert/strict";
import { test } from "node:test";
import { formatTable } from "../src/chain/resolve.ts";

const path = "/opensees/transforms/Linear_1/per_element_vecxz";

test("a (1, 3) vecxz prints its values, not its column indices", () => {
  assert.equal(formatTable({ path, columns: ["[0]", "[1]", "[2]"], rows: [[1.0, 0.0, 0.0]] }), "[1, 0, 0]");
});

test("a (1, 0) vecxz (2-D transform) is empty, not an error", () => {
  // readTable returns no columns and no rows for a zero-width dataset.
  assert.equal(formatTable({ path, columns: [], rows: [] }), "empty (no values)");
});

test("a 1-D dataset prints as one list; long arrays are cut with a count", () => {
  assert.equal(formatTable({ path, columns: ["value"], rows: [[7]] }), "[7]");
  assert.equal(formatTable({ path, columns: ["value"], rows: [[3], [4]] }), "[3, 4]");
  const rows = Array.from({ length: 10 }, (_, i) => [i, 0, 1]);
  assert.equal(
    formatTable({ path, columns: ["[0]", "[1]", "[2]"], rows }),
    "[0, 0, 1]; [1, 0, 1]; [2, 0, 1]; [3, 0, 1]; [4, 0, 1]; [5, 0, 1]; [6, 0, 1]; [7, 0, 1] ... (+2 rows)",
  );
});

test("a compound table keeps its row count and column names", () => {
  assert.equal(
    formatTable({ path: "/opensees/sections/Fiber_1/patches", columns: ["material_ref", "nfy"], rows: [["/a", 4], ["/b", 2]] }),
    "2 rows (material_ref, nfy)",
  );
});
