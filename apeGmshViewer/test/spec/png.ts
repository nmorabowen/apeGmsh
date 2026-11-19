// A minimal PNG reader for the render checks: the IHDR size and colour type,
// and the pixels of an 8-bit RGB or RGBA image (every filter type). Enough
// for the stills this app writes (Chromium's encoder and Electron's
// `toPNG`); anything else raises, naming what it met.

import { inflateSync } from "node:zlib";

export interface Png {
  width: number;
  height: number;
  /** 2 = RGB, 6 = RGBA */
  colorType: number;
  /** 4 bytes per pixel, row-major, alpha 255 for an RGB file */
  rgba: Uint8Array;
}

const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

export function readPng(buf: Uint8Array): Png {
  if (!SIGNATURE.every((b, i) => buf[i] === b)) throw new Error("not a PNG file");
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 8;
  let width = 0, height = 0, colorType = -1;
  const idat: Uint8Array[] = [];
  while (pos < buf.length) {
    const len = view.getUint32(pos);
    const type = String.fromCharCode(...buf.subarray(pos + 4, pos + 8));
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = view.getUint32(pos + 8);
      height = view.getUint32(pos + 12);
      const depth = data[8]!;
      colorType = data[9]!;
      if (depth !== 8 || (colorType !== 2 && colorType !== 6) || data[12] !== 0) {
        throw new Error(`PNG: bit depth ${depth}, colour type ${colorType}, interlace ${data[12]} is not read here`);
      }
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    pos += 12 + len;
  }
  const bpp = colorType === 6 ? 4 : 3;
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * bpp;
  const px = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)]!;
    const src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const row = px.subarray(y * stride, (y + 1) * stride);
    const up = y > 0 ? px.subarray((y - 1) * stride, y * stride) : null;
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? row[i - bpp]! : 0;
      const b = up ? up[i]! : 0;
      const c = up && i >= bpp ? up[i - bpp]! : 0;
      let v = src[i]!;
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      } else if (f !== 0) throw new Error(`PNG: filter type ${f}`);
      row[i] = v & 255;
    }
  }
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0, j = 0; i < width * height; i++, j += bpp) {
    rgba[4 * i] = px[j]!;
    rgba[4 * i + 1] = px[j + 1]!;
    rgba[4 * i + 2] = px[j + 2]!;
    rgba[4 * i + 3] = bpp === 4 ? px[j + 3]! : 255;
  }
  return { width, height, colorType, rgba };
}

/** Pixels whose RGB differ (alpha ignored, as an RGB diff), and their bounding box; inside `box` only when given. */
export function rgbDiff(a: Png, b: Png, box?: [number, number, number, number]): { count: number; bbox: [number, number, number, number] | null } {
  if (a.width !== b.width || a.height !== b.height) throw new Error(`sizes differ: ${a.width}x${a.height} vs ${b.width}x${b.height}`);
  let count = 0;
  let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
  const [bx0, by0, bx1, by1] = box ?? [0, 0, a.width, a.height];
  for (let y = by0; y < by1; y++) {
    for (let x = bx0; x < bx1; x++) {
      const i = 4 * (y * a.width + x);
      if (a.rgba[i] !== b.rgba[i] || a.rgba[i + 1] !== b.rgba[i + 1] || a.rgba[i + 2] !== b.rgba[i + 2]) {
        count++;
        x0 = Math.min(x0, x), y0 = Math.min(y0, y), x1 = Math.max(x1, x), y1 = Math.max(y1, y);
      }
    }
  }
  return { count, bbox: count ? [x0, y0, x1 + 1, y1 + 1] : null };
}
