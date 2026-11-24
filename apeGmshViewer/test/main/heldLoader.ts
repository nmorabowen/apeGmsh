// Stands in for src/main/loader/client.ts when test/main/openOrderDriver.ts loads main.ts: the
// real loader, whose `model` read waits, after the loader answered, until the test releases
// that path (`hold` / `release`). Two opens can then finish in the order the test picks.

import { Loader as RealLoader } from "../../src/main/loader/client.ts";

const gates = new Map<string, { wait: Promise<void>; open: () => void }>();

/** Hold every later `model(path)` answer until `release(path)`. */
export function hold(path: string): void {
  let open!: () => void;
  const wait = new Promise<void>((r) => (open = r));
  gates.set(path, { wait, open });
}

export function release(path: string): void {
  gates.get(path)?.open();
  gates.delete(path);
}

export class Loader extends RealLoader {
  override async model(path: string): ReturnType<RealLoader["model"]> {
    const gate = gates.get(path);
    try {
      return await super.model(path);
    } finally {
      if (gate) await gate.wait;
    }
  }
}
