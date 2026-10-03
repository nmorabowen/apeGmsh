// OpenSees command syntax: which positional argument of an element or a
// beamIntegration is a tag of which object family.
//
// This is the one place where the app INTERPRETS the file instead of
// reading it. model.h5 stores an element's OpenSees arguments verbatim
// (`/opensees/element_meta/{type}/args`, after the node tags), and a
// beamIntegration's arguments verbatim (`params`), with cross-references
// as bare tags. Which slot is the geomTransf tag is OpenSees syntax, from
// the OpenSees command manual, not apeGmsh code. Every link resolved
// through this table is marked `interpreted` in the inspector and in
// MEASUREMENTS.md. A type that is not in a table is reported as
// unresolved, by name; it is never guessed.

import type { OpsFamily, Param } from "../model/types.ts";

export interface LinkSlot {
  /** index into the args (elements) or params (integrations) */
  slot: number;
  family: OpsFamily;
  /** the argument's name in the OpenSees manual */
  label: string;
}

/** Slots for one row, or a reason the row cannot be decoded. */
export type Decoded = { slots: LinkSlot[]; syntax: string } | { reason: string };

type Decoder = (args: Param[]) => Decoded;

const isNum = (p: Param | undefined): p is number => typeof p === "number" && !Number.isNaN(p);

/** Count of leading numeric arguments (before the first string flag). */
function leadingNumbers(args: Param[]): number {
  let n = 0;
  while (n < args.length && isNum(args[n])) n++;
  return n;
}

// New-style syntax only: `transfTag integrationTag` then flags. The old style
// `numIntgrPts secTag transfTag` also starts with numbers, so it is told
// apart by its count of leading numeric args (3) and refused, never decoded.
const beamWithIntegration: Decoder = (args) => {
  const n = leadingNumbers(args);
  return n === 2
    ? {
        syntax: "transfTag integrationTag",
        slots: [
          { slot: 0, family: "geomTransf", label: "transfTag" },
          { slot: 1, family: "beamIntegration", label: "integrationTag" },
        ],
      }
    : { reason: `${n} leading numeric args; only the 2-tag form 'transfTag integrationTag' is decoded (old-style syntax is not)` };
};

const elasticBeam: Decoder = (args) => {
  const n = leadingNumbers(args);
  if (n === 2) {
    return {
      syntax: "secTag transfTag",
      slots: [
        { slot: 0, family: "section", label: "secTag" },
        { slot: 1, family: "geomTransf", label: "transfTag" },
      ],
    };
  }
  if (n === 4) {
    return { syntax: "A E Iz transfTag (2-D)", slots: [{ slot: 3, family: "geomTransf", label: "transfTag" }] };
  }
  if (n === 7) {
    return {
      syntax: "A E G J Iy Iz transfTag (3-D)",
      slots: [{ slot: 6, family: "geomTransf", label: "transfTag" }],
    };
  }
  return { reason: `${n} leading numeric args match no elasticBeamColumn form (2, 4 or 7)` };
};

const truss: Decoder = (args) => {
  if (args[0] === "-section" && isNum(args[1])) {
    return { syntax: "-section secTag", slots: [{ slot: 1, family: "section", label: "secTag" }] };
  }
  return isNum(args[0]) && isNum(args[1])
    ? { syntax: "A matTag", slots: [{ slot: 1, family: "uniaxialMaterial", label: "matTag" }] }
    : { reason: "args are neither 'A matTag' nor '-section secTag'" };
};

const solidMat: Decoder = (args) =>
  isNum(args[0])
    ? { syntax: "matTag", slots: [{ slot: 0, family: "nDMaterial", label: "matTag" }] }
    : { reason: "first arg is not a numeric matTag" };

const shellSec: Decoder = (args) =>
  isNum(args[0])
    ? { syntax: "secTag", slots: [{ slot: 0, family: "section", label: "secTag" }] }
    : { reason: "first arg is not a numeric secTag" };

// quad: `thick type matTag ...` (FourNodeQuad.cpp).
const quadMat: Decoder = (args) =>
  isNum(args[0]) && typeof args[1] === "string" && isNum(args[2])
    ? { syntax: "thick type matTag", slots: [{ slot: 2, family: "nDMaterial", label: "matTag" }] }
    : { reason: "args are not 'thick type matTag'" };

// SSPquad: `matTag type thickness ...` (SSPquad.cpp).
const sspQuadMat: Decoder = (args) =>
  isNum(args[0]) && typeof args[1] === "string" && isNum(args[2])
    ? { syntax: "matTag type thickness", slots: [{ slot: 0, family: "nDMaterial", label: "matTag" }] }
    : { reason: "args are not 'matTag type thickness'" };

export const ELEMENT_SYNTAX: Readonly<Record<string, Decoder>> = {
  dispBeamColumn: beamWithIntegration,
  forceBeamColumn: beamWithIntegration,
  elasticBeamColumn: elasticBeam,
  Truss: truss,
  CorotTruss: truss,
  stdBrick: solidMat,
  bbarBrick: solidMat,
  SSPbrick: solidMat,
  LadrunoBrick: solidMat,
  FourNodeTetrahedron: solidMat,
  ShellMITC4: shellSec,
  ShellDKGQ: shellSec,
  ShellNLDKGQ: shellSec,
  ASDShellQ4: shellSec,
  quad: quadMat,
  SSPquad: sspQuadMat,
};

const secThenN: Decoder = (params) =>
  isNum(params[0])
    ? { syntax: "secTag N", slots: [{ slot: 0, family: "section", label: "secTag" }] }
    : { reason: "first param is not a numeric secTag" };

export const INTEGRATION_SYNTAX: Readonly<Record<string, Decoder>> = {
  Legendre: secThenN,
  Lobatto: secThenN,
  Radau: secThenN,
  NewtonCotes: secThenN,
  Trapezoidal: secThenN,
  CompositeSimpson: secThenN,
};

/** Section types whose materials are referenced by HDF5 path (read, not interpreted). */
export const SECTIONS_WITH_MATERIAL_REFS: ReadonlySet<string> = new Set(["Fiber"]);

/** Section types that reference no material at all (parameters only). */
export const SECTIONS_WITHOUT_MATERIALS: ReadonlySet<string> = new Set([
  "Elastic",
  "ElasticMembranePlateSection",
  "ElasticShear",
]);
