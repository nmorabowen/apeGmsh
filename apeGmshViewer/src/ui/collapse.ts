// Universal collapsible panel chrome:
// 1. Card accordion: clicking the top `<h3>` header of any `<aside>` panel
//    (`#planes`, `#inspector`, `#sources`, `#legend`) toggles
//    `data-collapsed="true"` on the `<aside>` container itself, so panel
//    re-renders (`panel.replaceChildren(...)`) never lose the collapsed state
//    and individual panel modules need no collapse code.
// 2. Right-rail drawer toggle (`#rail-toggle`): collapses or expands the entire
//    right rail (`#rail`) to the right edge of the window so the 3D viewport
//    can use the full screen width.

import { el, listen } from "./dom.ts";

export const DEFAULT_PANEL_IDS = ["planes", "inspector", "sources", "legend"] as const;

/** Toggle `data-collapsed="true"` on a panel `<aside>`. Returns the new collapsed state. */
export function toggleCardCollapse(panel: HTMLElement): boolean {
  const next = panel.dataset["collapsed"] !== "true";
  if (next) panel.dataset["collapsed"] = "true";
  else delete panel.dataset["collapsed"];
  return next;
}

/**
 * True when a click target inside a panel's first `<h3>` should toggle the
 * card rather than activate an inline control (`pin`, `show all`, etc.).
 */
export function isHeadingToggleTarget(target: Element | null, panel: HTMLElement): boolean {
  if (!target) return false;
  const h3 = target.closest("h3");
  if (!h3 || h3.parentElement !== panel || panel.firstElementChild !== h3) return false;
  const interactive = target.closest("button, input, select, a, label");
  return interactive === null || !h3.contains(interactive);
}

export interface CollapsibleChromeOptions {
  readonly panelIds?: readonly string[];
  readonly visible?: () => boolean;
  readonly onRailToggle?: (collapsed: boolean) => void;
}

export interface CollapsibleChromeHandle {
  readonly syncVisibility: () => void;
  readonly isRailCollapsed: () => boolean;
  readonly setRailCollapsed: (collapsed: boolean) => void;
  readonly dispose: () => void;
}

/**
 * Wire delegated `<h3>` collapse on every panel `<aside>` and mount the
 * `#rail-toggle` drawer tab for the right-hand rail.
 */
export function mountCollapsibleChrome(opts: CollapsibleChromeOptions = {}): CollapsibleChromeHandle {
  const ids = opts.panelIds ?? DEFAULT_PANEL_IDS;
  const unlisten: (() => void)[] = [];

  for (const id of ids) {
    const panel = document.getElementById(id);
    if (!panel) continue;
    unlisten.push(
      listen(panel, "click", (e: MouseEvent) => {
        const target = e.target instanceof Element ? e.target : null;
        if (isHeadingToggleTarget(target, panel)) toggleCardCollapse(panel);
      }),
    );
  }

  let railCollapsed = false;
  const railBtn = el("button", "rail-toggle");
  railBtn.id = "rail-toggle";
  railBtn.type = "button";
  railBtn.hidden = true;
  document.body.appendChild(railBtn);

  const syncRailBtn = () => {
    railBtn.textContent = railCollapsed ? "\u2039" : "\u203a";
    railBtn.title = railCollapsed ? "Expand right panels (])" : "Collapse right panels to the side (])";
    railBtn.setAttribute("aria-expanded", String(!railCollapsed));
    if (railCollapsed) document.body.dataset["railCollapsed"] = "true";
    else delete document.body.dataset["railCollapsed"];
  };

  const setRailCollapsed = (collapsed: boolean) => {
    if (railCollapsed === collapsed) return;
    railCollapsed = collapsed;
    syncRailBtn();
    opts.onRailToggle?.(railCollapsed);
  };

  const syncVisibility = () => {
    const show = opts.visible ? opts.visible() : true;
    railBtn.hidden = !show;
  };

  syncRailBtn();
  syncVisibility();

  unlisten.push(listen(railBtn, "click", () => setRailCollapsed(!railCollapsed)));
  unlisten.push(
    listen(window as unknown as HTMLElement, "keydown", (e: KeyboardEvent) => {
      if (e.key !== "]" || e.ctrlKey || e.altKey || e.metaKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (railBtn.hidden) return;
      e.preventDefault();
      setRailCollapsed(!railCollapsed);
    }),
  );

  return {
    syncVisibility,
    isRailCollapsed: () => railCollapsed,
    setRailCollapsed,
    dispose: () => {
      for (const u of unlisten) u();
      delete document.body.dataset["railCollapsed"];
      railBtn.remove();
    },
  };
}
