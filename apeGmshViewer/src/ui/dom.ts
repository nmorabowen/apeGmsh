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
