// The workspace stills of the capture harness (VU9a, ADR 0132; D7 for #2803).
//
// Only a set with a `<stem>.workspace.h5` beside its model takes them, so no
// earlier fixture's capture changes. `<out>.workspace-sets.png` is the sets
// panel as main read the file: each set's verdict with its reason, or why the
// file is unreadable or refused. On a file this app may write, the harness
// then names a selection the way a human does: three elements clicked through
// the viewport's own pick (`pickAt`, as a mouse click runs it), the name typed
// into the panel's field and its Save clicked (`workspace-named.png`, then
// `workspace-save-refused.png`: the capture mode is no human window, so main's
// door refuses the save, read-only, and nothing is written). The last two
// stills show the panel and the banner for two answers a save in the human's
// window can get, which this mode cannot produce on disk: the typed refusal of
// a target another program holds (`heldRefusal`, the answer main/workspace.ts
// returns, `workspace-held.png`) and the banner naming the file an unreadable
// workspace was moved aside to (`workspace-aside.png`). Their door behaviour
// is proven by test/main/workspaceSave.test.ts.

import { heldRefusal } from "../../state/workspace.ts";
import type { HarnessContext } from "./context.ts";

export async function captureWorkspace(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>): Promise<void> {
  const { store, effects, viewport, bridge } = ctx;
  await effects.readWorkspace();
  const read = store.get().workspace.read;
  if (read === null || read.status === "none" || read.status === "absent") return;
  const shot = async (name: string) => {
    await settle();
    viewport.renderNow();
    await bridge.captureStill(name);
  };
  await shot("workspace-sets");
  if (read.status !== "ready" || read.readOnly !== null) return;
  // A human's selection: three drawn elements, each clicked at its projected midpoint, Shift held. In Nodes the
  // viewport highlights each clicked cell; the node markers are turned on, and the panel lists the node tags a save
  // takes (the node of each cell nearest the click), as the effects computed them.
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setNodes", on: true });
  const drawn = store.get().mesh?.elements ?? [];
  for (const decl of [drawn[0], drawn[Math.floor(drawn.length / 2)], drawn[drawn.length - 1]]) {
    const pt = decl === undefined ? null : viewport.screenPointOf(decl);
    const pick = pt ? viewport.pickAt(pt.x, pt.y) : null;
    if (pick) store.dispatch({ type: "selectAdd", pick });
  }
  const input = document.querySelector<HTMLInputElement>('#sets input[data-role="set-name"]');
  if (!input) throw new Error("capture: the sets panel has no name field");
  input.value = "picked nodes";
  input.dispatchEvent(new Event("input"));
  // The effects compute the preview a task after the selection changes.
  await settle(50);
  const preview = store.get().workspace.preview;
  if (!preview || !("ids" in preview) || preview.kind !== "nodes") throw new Error(`capture: the selection previews no node tags: ${JSON.stringify(preview)}`);
  await shot("workspace-named");
  const save = [...document.querySelectorAll<HTMLButtonElement>("#sets .sets-form button")].find((b) => !b.disabled);
  if (!save) throw new Error("capture: the sets panel's Save is disabled with three elements selected and a name typed");
  save.click();
  for (let t = 0; t < 10000 && store.get().workspace.saving !== null; t += 50) await settle(50);
  await shot("workspace-save-refused");
  // The typed refusal of a held target, as main answers it.
  store.dispatch({ type: "workspaceSave", edit: { op: "create", name: "picked nodes" } });
  store.dispatch({ type: "workspaceSaved", seq: store.get().workspace.seq, answer: heldRefusal(read.file) });
  await shot("workspace-held");
  // The banner after a save that moved an unreadable file aside.
  store.dispatch({ type: "workspaceSave", edit: { op: "create", name: "picked nodes" } });
  store.dispatch({ type: "workspaceSaved", seq: store.get().workspace.seq, answer: { ok: true, read, aside: `${read.file}.unreadable-20261111T090807Z` } });
  // The banner opened, as a click on it opens it, so the still shows the whole line and its Close.
  const summary = document.querySelector<HTMLButtonElement>("#banner .banner-summary");
  summary?.click();
  await shot("workspace-aside");
  if (summary) document.querySelector<HTMLButtonElement>("#banner .banner-summary")?.click();
  store.dispatch({ type: "dismissWorkspaceAside" });
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setNodes", on: false });
  // A set switch: another model opens, and the last set's workspace leaves the panel at once (nothing is saved against it).
  const model = store.get().artifacts.model?.path ?? "";
  store.dispatch({ type: "fileOpened", artifact: "model", path: model.replace(/(\.h5)$/i, "-other$1") });
  await shot("workspace-switch");
}
