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

// LadrunoQuad (Ladruno fork): `matTag` then flags only, `<-formulation f>
// <-type t> <-geom g> <-thick t> <-rho r> <-body b1 b2> <-pressure p>`
// (SRC/element/ladrunoPlane/OPS_LadrunoQuad.cpp, the usage comment and the
// first OPS_GetIntInput: the one positional argument is the nDMaterial tag).
// The same shape as a solid's, so it shares `solidMat`.

// SSPquad: `matTag type thickness ...` (SSPquad.cpp).
const sspQuadMat: Decoder = (args) =>
  isNum(args[0]) && typeof args[1] === "string" && isNum(args[2])
    ? { syntax: "matTag type thickness", slots: [{ slot: 0, family: "nDMaterial", label: "matTag" }] }
    : { reason: "args are not 'matTag type thickness'" };

const IMK_MAT_FLAGS_1: ReadonlySet<string> = new Set([
  "-matZ",
  "-matY",
  "-matZi",
  "-matZj",
  "-matYi",
  "-matYj",
  "-hingeI",
  "-hingeJ",
  "-shearHinge",
  "-shearHingeY",
  "-shearHingeZ",
  "-hingeBiaxial",
  "-hingeBiaxialI",
  "-hingeBiaxialJ",
]);

const imkBeam: Decoder = (args) => {
  const n = leadingNumbers(args);
  const collectFlagSlots = (baseSlots: { slot: number; family: "geomTransf" | "uniaxialMaterial"; label: string }[]) => {
    const slots = [...baseSlots];
    for (let i = n; i < args.length; i++) {
      const tok = args[i];
      if (typeof tok === "string" && IMK_MAT_FLAGS_1.has(tok) && isNum(args[i + 1])) {
        slots.push({ slot: i + 1, family: "uniaxialMaterial", label: tok.slice(1) });
        i += 1;
      } else if (tok === "-hingeBiaxialIJ" && isNum(args[i + 1]) && isNum(args[i + 2])) {
        slots.push(
          { slot: i + 1, family: "uniaxialMaterial", label: "hingeBiaxialI" },
          { slot: i + 2, family: "uniaxialMaterial", label: "hingeBiaxialJ" },
        );
        i += 2;
      }
    }
    return slots;
  };
  if (n === 4 || n === 6) {
    const base: { slot: number; family: "geomTransf" | "uniaxialMaterial"; label: string }[] = [
      { slot: 3, family: "geomTransf", label: "transfTag" },
    ];
    if (n === 6) {
      base.push(
        { slot: 4, family: "uniaxialMaterial", label: "matTagI" },
        { slot: 5, family: "uniaxialMaterial", label: "matTagJ" },
      );
    }
    return {
      syntax: n === 6 ? "A E Iz transfTag matTagI matTagJ (2-D)" : "A E Iz transfTag (2-D)",
      slots: collectFlagSlots(base),
    };
  }
  if (n === 7 || n === 9) {
    const base: { slot: number; family: "geomTransf" | "uniaxialMaterial"; label: string }[] = [
      { slot: 6, family: "geomTransf", label: "transfTag" },
    ];
    if (n === 9) {
      base.push(
        { slot: 7, family: "uniaxialMaterial", label: "matTagI" },
        { slot: 8, family: "uniaxialMaterial", label: "matTagJ" },
      );
    }
    return {
      syntax: n === 9 ? "A E G Jx Iy Iz transfTag matTagI matTagJ (3-D)" : "A E G Jx Iy Iz transfTag (3-D)",
      slots: collectFlagSlots(base),
    };
  }
  return { reason: `${n} leading numeric args match no LadrunoIMKBeam form (4, 6, 7, or 9)` };
};

const timoshenkoBeam: Decoder = (args) => {
  const n = leadingNumbers(args);
  if (n === 6) {
    return { syntax: "E G A Iz Avy transfTag (2-D)", slots: [{ slot: 5, family: "geomTransf", label: "transfTag" }] };
  }
  if (n === 9) {
    return {
      syntax: "E G A Jx Iy Iz Avy Avz transfTag (3-D)",
      slots: [{ slot: 8, family: "geomTransf", label: "transfTag" }],
    };
  }
  return { reason: `${n} leading numeric args match no ElasticTimoshenkoBeam form (6 or 9)` };
};

const modElasticBeam2d: Decoder = (args) => {
  const n = leadingNumbers(args);
  if (n === 7) {
    return {
      syntax: "A E Iz K11 K33 K44 transfTag (2-D)",
      slots: [{ slot: 6, family: "geomTransf", label: "transfTag" }],
    };
  }
  return { reason: `${n} leading numeric args match no ModElasticBeam2d form (7)` };
};

export const ELEMENT_SYNTAX: Readonly<Record<string, Decoder>> = {
  dispBeamColumn: beamWithIntegration,
  forceBeamColumn: beamWithIntegration,
  LadrunoDispBeamColumn: beamWithIntegration,
  LadrunoForceBeamColumn: beamWithIntegration,
  elasticBeamColumn: elasticBeam,
  LadrunoIMKBeam: imkBeam,
  ElasticTimoshenkoBeam: timoshenkoBeam,
  elasticTimoshenkoBeam: timoshenkoBeam,
  ModElasticBeam2d: modElasticBeam2d,
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
  LadrunoQuad: solidMat,
  SSPquad: sspQuadMat,
};

const secThenN: Decoder = (params) =>
  isNum(params[0])
    ? { syntax: "secTag N", slots: [{ slot: 0, family: "section", label: "secTag" }] }
    : { reason: "first param is not a numeric secTag" };

const hingeIntegration: Decoder = (params) =>
  params.length === 5 && isNum(params[0]) && isNum(params[2]) && isNum(params[4])
    ? {
        syntax: "secITag lpI secJTag lpJ secETag",
        slots: [
          { slot: 0, family: "section", label: "secITag" },
          { slot: 2, family: "section", label: "secJTag" },
          { slot: 4, family: "section", label: "secETag" },
        ],
      }
    : { reason: `expected 5 params 'secITag lpI secJTag lpJ secETag', got ${params.length}` };

export const INTEGRATION_SYNTAX: Readonly<Record<string, Decoder>> = {
  Legendre: secThenN,
  Lobatto: secThenN,
  Radau: secThenN,
  NewtonCotes: secThenN,
  Trapezoidal: secThenN,
  CompositeSimpson: secThenN,
  HingeMidpoint: hingeIntegration,
  HingeRadau: hingeIntegration,
  HingeRadauTwo: hingeIntegration,
  HingeEndpoint: hingeIntegration,
};

export const SECTION_SYNTAX: Readonly<Record<string, Decoder>> = {
  Aggregator: (params) => {
    const slots: LinkSlot[] = [];
    let i = 0;
    while (i < params.length) {
      if (params[i] === "-section") {
        if (!isNum(params[i + 1])) return { reason: "Aggregator -section flag is missing a numeric secTag" };
        slots.push({ slot: i + 1, family: "section", label: "secTag" });
        i += 2;
      } else if (isNum(params[i]) && typeof params[i + 1] === "string") {
        slots.push({ slot: i, family: "uniaxialMaterial", label: `matTag(${params[i + 1]})` });
        i += 2;
      } else {
        return { reason: `Aggregator params at [${i}] match neither 'matTag dof' nor '-section secTag'` };
      }
    }
    return slots.length > 0
      ? { syntax: "(matTag dof)... [-section secTag]", slots }
      : { reason: "Aggregator has no slots" };
  },
  InsertionPoint: (params) =>
    params.length >= 1 && isNum(params[0])
      ? {
          syntax: "secTag -offset dy dz [-bbox yMin yMax zMin zMax]",
          slots: [{ slot: 0, family: "section", label: "secTag" }],
        }
      : { reason: "InsertionPoint first param is not a numeric secTag" },
  WFSection2d: (params) =>
    params.length === 7 && isNum(params[0])
      ? { syntax: "matTag d tw bf tf Nfw Nff", slots: [{ slot: 0, family: "uniaxialMaterial", label: "matTag" }] }
      : { reason: `WFSection2d expects 7 params, got ${params.length}` },
  RCSection2d: (params) =>
    params.length === 12 && isNum(params[0]) && isNum(params[1]) && isNum(params[2])
      ? {
          syntax: "coreTag coverTag steelTag d b cover Atop Abot Aside Nfcore Nfcover Nfs",
          slots: [
            { slot: 0, family: "uniaxialMaterial", label: "coreTag" },
            { slot: 1, family: "uniaxialMaterial", label: "coverTag" },
            { slot: 2, family: "uniaxialMaterial", label: "steelTag" },
          ],
        }
      : { reason: `RCSection2d expects 12 params, got ${params.length}` },
};

/** Section types whose materials are referenced by HDF5 path (read, not interpreted). */
export const SECTIONS_WITH_MATERIAL_REFS: ReadonlySet<string> = new Set(["Fiber", "FiberThermal", "NDFiber"]);

/** Section types that reference no material at all (parameters only). */
export const SECTIONS_WITHOUT_MATERIALS: ReadonlySet<string> = new Set([
  "Elastic",
  "ElasticMembranePlateSection",
  "ElasticShear",
  "ElasticTube",
  "ElasticTimoshenko",
]);

