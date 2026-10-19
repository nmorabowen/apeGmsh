// The node side of the .ladruno pairing oracle (ADR 0122 D10 row 4; #2129).
//
//   node scripts/pair_oracle.ts <model.h5> <run.ladruno>
//
// It opens the two files the way the app does and prints one JSON line: the
// model's /meta/lineage (main/zones.ts `openLineage`), the results index as
// main answers it to the renderer (main/results.ts `ResultsHost.open`, which
// reads a `.ladruno` with reader/ladruno.ts), and the verdict of the real
// state/results.ts `pairResults` on them. The model side is projected the way
// state/selectors.ts `resultsPairing` projects the store (the opensees zone's
// version and its `/opensees/decls@program_decl` marker).
//
// tests/test_app_ladruno_pairing_oracle.py spawns it on files apeGmsh wrote
// and stamped. It imports nothing from src/apeGmsh (ADR 0112 D4): the files
// are the only thing the two sides share. Any failure to read either file is
// an error exit with the message, never a verdict.

import * as h5wasm from "h5wasm/node";
import { ResultsHost } from "../src/main/results.ts";
import { openLineage } from "../src/main/zones.ts";
import { openModel } from "../src/reader/node.ts";
import type { ResultsH5 } from "../src/reader/results.ts";
import { pairResults } from "../src/state/results.ts";

const [modelPath, resultsPath, ...extra] = process.argv.slice(2);
if (!modelPath || !resultsPath || extra.length > 0) {
  process.stderr.write("usage: node scripts/pair_oracle.ts <model.h5> <run.ladruno>\n");
  process.exit(2);
}

await h5wasm.ready;
const lineage = await openLineage(modelPath);
if (!lineage.ok) throw new Error(`${modelPath}: /meta/lineage: ${lineage.error}`);
const model = await openModel(modelPath);
const ops = model.opensees;
// selectors.ts `resultsPairing`: no decls group in an opensees zone reads as "absent".
const programDecl = ops ? (ops.definitions.decls ? (ops.definitions.decls.programDecl ?? "absent") : "absent") : null;

const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);
const answer = await host.open(resultsPath);
host.close();
if (!answer.ok) throw new Error(`${resultsPath}: ${answer.error}`);
const meta = answer.meta;

const pairing = pairResults({ lineage: lineage.lineage, openseesVersion: ops ? ops.version : null, programDecl }, meta);
process.stdout.write(
  JSON.stringify({
    pairing,
    model: { lineage: lineage.lineage, openseesVersion: ops ? ops.version : null, programDecl },
    results: { sourceType: meta.sourceType, lineage: meta.lineage, lineageReason: meta.lineageReason ?? null },
  }) + "\n",
);
