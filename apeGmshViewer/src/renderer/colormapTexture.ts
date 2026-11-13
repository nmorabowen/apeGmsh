// The contour's colormap as a 1-D texture, sampled per fragment (VF-a, #2939;
// #2924 item 1). A vertex colour attribute makes the GPU interpolate RGB
// inside each primitive, so a coarse element shows hues off the ramp: under
// jet, blue at one corner and red at the opposite one blend through purple,
// which jet never shows. Here a vertex carries where its value falls on the
// ramp (a texture coordinate, `rampU`); the GPU interpolates that scalar and
// each fragment looks its colour up in the ramp, so every pixel is a ramp
// colour (state/stillLayers' scale bar already samples per pixel).
//
// - The texture is RAMP_TEXELS ramp samples (state/contour.ts `sampleCmap`
//   at i / (RAMP_TEXELS - 1), linearised: three.js shades in linear and
//   writes sRGB) plus one no-data texel (`NO_DATA_RGB`), filtered NEAREST
//   with no mipmaps: a fragment shows exactly one texel, never a blend of the
//   ramp with the no-data grey, and never a colour between two samples.
// - A vertex coordinate sits on a texel's centre: t in [0, 1] at ramp texel
//   round(t * (RAMP_TEXELS - 1)), no data on the last texel. A primitive with
//   a no-data vertex is no data whole (`primitiveColours`' rule), so its
//   three coordinates are equal and it shows the grey everywhere.

import * as THREE from "three";
import { NO_DATA_RGB, rampT, sampleCmap } from "../state/contour.ts";
import type { ContourCmap } from "../state/types/contour.ts";

/** sRGB (0..1) to linear, per channel (the sRGB transfer function); renderer/contour.ts re-exports it. */
export const toLinear = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

/** Ramp samples in the texture (the last texel after them is no data): 1 / 2046 of the range apart, below a screen's 8-bit step. */
export const RAMP_TEXELS = 2047;
/** The texture's width: the ramp samples, then the no-data texel. */
export const TEXTURE_WIDTH = RAMP_TEXELS + 1;

/** The texture coordinate of ramp position `t` (clamped to [0, 1]): the centre of its nearest ramp texel. */
export function rampU(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return (Math.round(x * (RAMP_TEXELS - 1)) + 0.5) / TEXTURE_WIDTH;
}

/** The texture coordinate of no data: the centre of the last texel. */
export const NO_DATA_U = (RAMP_TEXELS + 0.5) / TEXTURE_WIDTH;

/**
 * Per-vertex ramp coordinates of the drawn primitives (`map`: original
 * primitive indices, `perPrim` vertices each), by state/contour.ts
 * `primitiveColours`' rule: a primitive with any no-data vertex, or every
 * primitive when the range has no finite end, is no data whole. Returns the
 * coordinates and how many drawn primitives are no data.
 */
export function primitiveRamp(
  values: Float64Array,
  map: Int32Array,
  perPrim: number,
  clim: { min: number | null; max: number | null },
): { u: Float32Array; noData: number } {
  const u = new Float32Array(map.length * perPrim);
  let noData = 0;
  const ok = clim.min !== null && clim.max !== null;
  for (let d = 0; d < map.length; d++) {
    const o = map[d]! * perPrim;
    let missing = !ok;
    for (let v = 0; v < perPrim && !missing; v++) if (Number.isNaN(values[o + v]!)) missing = true;
    if (missing) noData++;
    for (let v = 0; v < perPrim; v++) u[d * perPrim + v] = missing ? NO_DATA_U : rampU(rampT(values[o + v]!, clim.min!, clim.max!));
  }
  return { u, noData };
}

/** `n` vertices, every one no data. */
export function noDataRamp(n: number): Float32Array {
  return new Float32Array(n).fill(NO_DATA_U);
}

/** The texture's texels (RGBA, linear, 0..1): the ramp samples, then the no-data colour. */
export function rampTexels(cmap: ContourCmap): Float32Array {
  const data = new Float32Array(TEXTURE_WIDTH * 4);
  for (let i = 0; i < RAMP_TEXELS; i++) {
    const c = sampleCmap(cmap, i / (RAMP_TEXELS - 1));
    data.set([toLinear(c[0]), toLinear(c[1]), toLinear(c[2]), 1], 4 * i);
  }
  data.set([toLinear(NO_DATA_RGB[0]), toLinear(NO_DATA_RGB[1]), toLinear(NO_DATA_RGB[2]), 1], 4 * RAMP_TEXELS);
  return data;
}

/**
 * The linear colour a fragment at texture coordinate `u` shows: what the
 * GPU's NEAREST lookup returns (texel floor(u * width), clamped to the edge).
 * The tests read the drawn colours through it.
 */
export function shownAt(texture: THREE.DataTexture, u: number): [number, number, number] {
  const data = texture.image.data as Float32Array;
  const w = texture.image.width;
  const i = Math.min(w - 1, Math.max(0, Math.floor(u * w)));
  return [data[4 * i]!, data[4 * i + 1]!, data[4 * i + 2]!];
}

const textures = new Map<ContourCmap, THREE.DataTexture>();

/**
 * The colormap's texture, one per colormap for the process (a few kB each,
 * shared by every surface that draws the contour: faces, frame fill, caps).
 * Nothing disposes it: three.js uploads it once per renderer.
 */
export function colormapTexture(cmap: ContourCmap): THREE.DataTexture {
  const held = textures.get(cmap);
  if (held) return held;
  const t = new THREE.DataTexture(rampTexels(cmap), TEXTURE_WIDTH, 1, THREE.RGBAFormat, THREE.FloatType);
  t.name = `colormap ${cmap}`;
  t.colorSpace = THREE.NoColorSpace; // the texels are linear already
  t.magFilter = THREE.NearestFilter;
  t.minFilter = THREE.NearestFilter;
  t.generateMipmaps = false;
  t.wrapS = THREE.ClampToEdgeWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  textures.set(cmap, t);
  return t;
}

/** A `uv` attribute from per-vertex ramp coordinates (v at the texture's one row). */
export function rampUv(u: Float32Array): Float32Array {
  const uv = new Float32Array(2 * u.length);
  for (let i = 0; i < u.length; i++) {
    uv[2 * i] = u[i]!;
    uv[2 * i + 1] = 0.5;
  }
  return uv;
}
