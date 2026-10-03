// Open a model.h5 from disk in a Node or Electron-main process.

import { statSync } from "node:fs";
import * as h5wasm from "h5wasm/node";
import type { ModelFile } from "../model/types.ts";
import { readModel, type H5Module } from "./read.ts";

export async function openModel(path: string): Promise<ModelFile> {
  await h5wasm.ready;
  const size = statSync(path).size;
  return readModel(h5wasm as unknown as H5Module, path, size);
}
