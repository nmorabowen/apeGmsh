// Element-syntax decoders, one case per form in signatures.ts. The expected
// slots are the upstream OpenSees argument orders, cited per test, not values
// read back from the app. Plus the reader's args_str merge on a real file.

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as h5wasm from "h5wasm/node";
import { resolveChain } from "../src/chain/resolve.ts";
import type { ModelFile, OpsObject, Param } from "../src/model/types.ts";
import { readModel, type H5Module } from "../src/reader/read.ts";

const DIRS: Record<OpsObject["family"], string> = {
  uniaxialMaterial: "/opensees/materials/uniaxial",
  nDMaterial: "/opensees/materials/nd",
  section: "/opensees/sections",
  geomTransf: "/opensees/transforms",
  beamIntegration: "/opensees/beam_integration",
};

function obj(family: OpsObject["family"], type: string, tag: number, params: Param[] = []): OpsObject {
  const groupName = `${type}_${tag}`;
  return { family, path: `${DIRS[family]}/${groupName}`, groupName, type, tag, params, attrs: {}, tables: {} };
}

function model(objects: OpsObject[], type: string, args: Param[]): ModelFile {
  return {
    path: "synthetic.h5",
    sizeBytes: 0,
    neutralVersion: "2.33.0",
    meta: {},
    nodeIds: new Float64Array([1, 2]),
    nodeCoords: new Float64Array([0, 0, 0, 1, 0, 0]),
    blocks: [{ alias: "line2", code: 1, dim: 1, npe: 2, ids: new Float64Array([1]), connectivity: new Float64Array([1, 2]) }],
    physicalGroups: [],
    labels: [],
    opensees: {
      version: "2.21.0",
      objects,
      elementMeta: [{
        type,
        path: `/opensees/element_meta/${type}`,
        ids: new Float64Array([7]),
        femEids: new Float64Array([1]),
        args: [args],
        inlineConnectivity: null,
      }],
      names: [],
    },
    warnings: [],
    readMs: 0,
  };
}

/** [label, args slot, target name] for each first-level link. */
function links(type: string, args: Param[], objects: OpsObject[]) {
  const c = resolveChain(model(objects, type, args), { kind: "fem", blockIndex: 0, row: 0 });
  return {
    problems: c.problems,
    links: c.root.children.map((n) => [n.via!.label, /\[(\d+)\]$/.exec(n.via!.source)![1], n.name]),
  };
}

const nd = (tag: number) => obj("nDMaterial", "ElasticIsotropic", tag);
const sec = (tag: number) => obj("section", "ElasticMembranePlateSection", tag);
const uni = (tag: number) => obj("uniaxialMaterial", "Elastic", tag);
const beamObjs = () => [
  obj("geomTransf", "Linear", 1),
  obj("beamIntegration", "Lobatto", 1, [1, 5]),
  obj("section", "Elastic", 1),
];

test("SSPquad: matTag is slot 0 ('matTag type thickness', SSPquad.cpp)", () => {
  const r = links("SSPquad", [3, "PlaneStrain", 0.5], [nd(3), nd(7)]);
  assert.deepEqual(r.problems, []);
  assert.deepEqual(r.links, [["matTag", "0", "ElasticIsotropic_3"]]);
});

test("quad: matTag is slot 2 ('thick type matTag', FourNodeQuad.cpp)", () => {
  assert.deepEqual(links("quad", [0.5, "PlaneStrain", 7], [nd(3), nd(7)]).links, [["matTag", "2", "ElasticIsotropic_7"]]);
});

test("old-style beam 'numIntgrPts secTag transfTag' is refused, not mis-decoded", () => {
  const r = links("forceBeamColumn", [5, 1, 1], beamObjs());
  assert.deepEqual(r.links, []);
  assert.equal(r.problems.length, 1);
  assert.match(r.problems[0]!, /3 leading numeric args; only the 2-tag form/);
});

test("new-style beam 'transfTag integrationTag' with trailing flags decodes", () => {
  const r = links("dispBeamColumn", [1, 1, "-mass", 2.5], beamObjs());
  assert.deepEqual(r.problems, []);
  assert.deepEqual(r.links, [["transfTag", "0", "Linear_1"], ["integrationTag", "1", "Lobatto_1"]]);
});

test("elasticBeamColumn: section, 2-D and 3-D forms; other counts refused", () => {
  const objs = [obj("geomTransf", "Linear", 4), sec(9)];
  assert.deepEqual(links("elasticBeamColumn", [9, 4], objs).links, [
    ["secTag", "0", "ElasticMembranePlateSection_9"],
    ["transfTag", "1", "Linear_4"],
  ]);
  assert.deepEqual(links("elasticBeamColumn", [1, 2e11, 1e-4, 4], objs).links, [["transfTag", "3", "Linear_4"]]);
  assert.deepEqual(links("elasticBeamColumn", [1, 2e11, 8e10, 1e-4, 1e-4, 1e-4, 4], objs).links, [["transfTag", "6", "Linear_4"]]);
  assert.match(links("elasticBeamColumn", [1, 2, 3], objs).problems[0]!, /3 leading numeric args match no elasticBeamColumn form/);
});

test("Truss / CorotTruss: 'A matTag' and '-section secTag'", () => {
  assert.deepEqual(links("CorotTruss", [2.8e-4, 1], [uni(1)]).links, [["matTag", "1", "Elastic_1"]]);
  assert.deepEqual(links("Truss", ["-section", 9], [sec(9)]).links, [["secTag", "1", "ElasticMembranePlateSection_9"]]);
});

test("solids and shells: matTag / secTag in slot 0", () => {
  assert.deepEqual(links("LadrunoBrick", [3, "-formulation", "bbar"], [nd(3)]).links, [["matTag", "0", "ElasticIsotropic_3"]]);
  assert.deepEqual(links("FourNodeTetrahedron", [3], [nd(3)]).links, [["matTag", "0", "ElasticIsotropic_3"]]);
  assert.deepEqual(links("ASDShellQ4", [9, "-corotational"], [sec(9)]).links, [["secTag", "0", "ElasticMembranePlateSection_9"]]);
});

test("reader: args_str fills the string slots of args; trailing NaN padding is dropped", async () => {
  await h5wasm.ready;
  const path = join(mkdtempSync(join(tmpdir(), "agv-dec-")), "argsstr.h5");
  const f = new h5wasm.File(path, "w");
  const m = f.create_group("meta");
  m.create_attribute("neutral_schema_version", "2.33.0");
  m.create_attribute("opensees_schema_version", "2.21.0");
  const nodes = f.create_group("nodes");
  nodes.create_dataset({ name: "ids", data: new BigInt64Array([1n]) });
  nodes.create_dataset({ name: "coords", data: new Float64Array([0, 0, 0]), shape: [1, 3] });
  const em = f.create_group("opensees").create_group("element_meta").create_group("LadrunoBrick");
  em.create_dataset({ name: "ids", data: new BigInt64Array([2n, 3n]) });
  em.create_dataset({ name: "fem_eids", data: new BigInt64Array([10n, 11n]) });
  em.create_dataset({ name: "args", data: new Float64Array([1, NaN, NaN, 4, NaN, NaN]), shape: [2, 3] });
  em.create_dataset({ name: "args_str", data: ["", "-formulation", "bbar", "", "", ""], shape: [2, 3] });
  f.close();
  const r = readModel(h5wasm as unknown as H5Module, path, 0);
  const row = r.opensees!.elementMeta[0]!;
  assert.deepEqual(row.args, [[1, "-formulation", "bbar"], [4]]);
  assert.deepEqual(Array.from(row.femEids), [10, 11]);
});
