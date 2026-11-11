// The combination authoring stills of the capture harness (VU9b, ADR 0130 A4; D7 for #2693).
//
// Only a set whose `<stem>.user.h5` main read takes them, so no earlier
// fixture's capture changes unless it has one. The harness works the panel as
// a human does: it clicks the phase bar's `+ Combination` (`combine-empty.png`:
// the panel open, the file and the generation in force), types a srss of two
// increments of the set's first `/spectra` case into the fields
// (`combine-valid.png`: the readers' rules and VU4's evaluator let it stand,
// with the value it evaluates to), switches the rule to directional with no
// fraction (`combine-refused.png`: the readers refuse it, `bad-param`, and Save
// is off), switches back and clicks Save (`combine-save-refused.png`: a capture
// window is no human window, so main's door refuses the save, read-only, and
// nothing is written). `combine-held.png` is the typed refusal main answers when
// another program holds the file (combination/author.ts `combinationHeldRefusal`),
// which this mode cannot produce on disk. `combine-listed.png` shows the file's
// last User Combination from the phase bar: on a set the app's writer saved
// into, the row it added; on one it reverted, the rows of the generation back
// in force. Their door behaviour is proven by test/main/combinationsSave.test.ts.

import { combinationHeldRefusal } from "../../combination/author.ts";
import type { HarnessContext } from "./context.ts";

export async function captureCombine(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>): Promise<void> {
  const { store, viewport, bridge, combine } = ctx;
  const u = store.get().user;
  if (u?.status !== "ready" || !combine) return;
  const shot = async (name: string) => {
    await settle();
    viewport.renderNow();
    await bridge.captureStill(name);
  };
  // The last User Combination of the file, shown from the phase bar (as its picker shows one).
  const last = u.info.table?.rows.at(-1)?.name ?? null;
  const entry = document.querySelector<HTMLButtonElement>('#phase [data-role="new-combination"]');
  if (!entry) throw new Error("capture: the phase bar has no + Combination with a user file read");
  entry.click();
  await settle(300);
  await shot("combine-empty");
  const cases = u.info.spectra?.ok ? u.info.spectra.items : [];
  if (cases.length) {
    const q = <T extends Element>(sel: string): T => {
      const e = document.querySelector<T>(`#combine ${sel}`);
      if (!e) throw new Error(`capture: the combine panel has no ${sel}`);
      return e;
    };
    const type = (sel: string, value: string, ev: "input" | "change" = "input") => {
      const e = q<HTMLInputElement | HTMLSelectElement>(sel);
      e.value = value;
      e.dispatchEvent(new Event(ev));
    };
    const tone = async (want: string) => {
      for (let t = 0; t < 10000; t += 50) {
        if (q<HTMLElement>('[data-role="combine-check"]').dataset["tone"] === want) return;
        await settle(50);
      }
      throw new Error(`capture: the draft's check never read ${want}: ${q<HTMLElement>('[data-role="combine-check"]').textContent}`);
    };
    const name = cases[0]!.name;
    type('[data-role="combine-name"]', "SRSS Ex modes 1-2 (capture)");
    type('[data-role="combine-rule"]', "srss", "change");
    type('[data-term="0"] [data-role="term-kind"]', "spectrum", "change");
    type('[data-term="0"] [data-role="term-source"]', name);
    type('[data-term="0"] [data-role="term-mode"]', "1");
    q<HTMLButtonElement>('[data-role="combine-add-term"]').click();
    type('[data-term="1"] [data-role="term-source"]', name);
    type('[data-term="1"] [data-role="term-mode"]', "2");
    await tone("ok");
    await shot("combine-valid");
    type('[data-role="combine-rule"]', "directional", "change");
    await tone("refused");
    await shot("combine-refused");
    type('[data-role="combine-rule"]', "srss", "change");
    await tone("ok");
    const save = q<HTMLButtonElement>('[data-role="combine-save"]');
    if (save.disabled) throw new Error(`capture: Save is off on a standing draft: ${save.title}`);
    save.click();
    for (let t = 0; t < 10000 && !document.querySelector('#combine [data-role="combine-notice"]'); t += 50) await settle(50);
    await shot("combine-save-refused");
    combine.receive(combinationHeldRefusal(u.path));
    await shot("combine-held");
  }
  if (last !== null && store.get().mesh) {
    store.dispatch({ type: "setPhase", at: { kind: "combination", name: last } });
    if (store.get().visibility.colourBy !== "contour") store.dispatch({ type: "setColourBy", by: "contour" });
    await settle(800);
    await shot("combine-listed");
  }
}
