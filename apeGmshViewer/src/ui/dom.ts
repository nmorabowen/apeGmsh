// DOM helpers the panels share. Panels import this, store.ts and
// selectors.ts; nothing else of the app (the panel-import lint rule).

export const el = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

export const byId = (id: string): HTMLElement => {
  const e = document.getElementById(id);
  if (!e) throw new Error(`#${id} is missing from index.html`);
  return e;
};

/** Add a listener; the returned function removes it. */
export function listen<K extends keyof HTMLElementEventMap>(
  target: HTMLElement,
  type: K,
  fn: (e: HTMLElementEventMap[K]) => void,
): () => void {
  target.addEventListener(type, fn);
  return () => target.removeEventListener(type, fn);
}

/** ADR 0138 D6: what a click or drag on the model does to the active kind's selection. */
export type SelectMode = "replace" | "add" | "remove";

/** ADR 0138 D6: no modifier replaces, Shift adds, Ctrl (Cmd) removes. */
export function selectModeOf(e?: { readonly shiftKey?: boolean; readonly ctrlKey?: boolean; readonly metaKey?: boolean }): SelectMode {
  if (e?.ctrlKey || e?.metaKey) return "remove";
  return e?.shiftKey ? "add" : "replace";
}

/** Whether a click or pointer event's modifiers toggle in the selection (Shift, Ctrl or Cmd/Meta): the list windows' rows. */
export function isSelectAdd(e?: { readonly shiftKey?: boolean; readonly ctrlKey?: boolean; readonly metaKey?: boolean }): boolean {
  return Boolean(e && (e.shiftKey || e.ctrlKey || e.metaKey));
}
