// Spotlight (`Ctrl+K` / `Cmd+K`) command and search palette (ADR 0112 D6, D7):
// one input that searches and executes viewer commands, drill-down parameter
// pickers, boolean declaration queries, and batch selection/visibility actions.
// Renders strictly from `spotlightViewOf(state)` and dispatches store events.

import { spotlightViewOf, type SpotlightItem } from "../state/selectors.ts";
import type { Event, State, Store } from "../state/store.ts";
import { el, listen } from "../ui/dom.ts";

export function mountSpotlight(store: Store): () => void {
  let root = document.getElementById("spotlight");
  const created = root === null;
  if (!root) {
    root = el("div");
    root.id = "spotlight";
    document.body.append(root);
  }
  root.hidden = true;

  const card = el("div", "spotlight-card");
  const top = el("div", "spotlight-top");
  const modeBadge = el("button", "spotlight-mode-badge");
  modeBadge.title = "return to root search (Esc)";
  const chipsBox = el("div", "spotlight-chips");
  const input = el("input", "spotlight-input");
  input.type = "text";
  input.autocomplete = "off";
  input.spellcheck = false;
  const helpBtn = el("button", "spotlight-help", "? Help");
  helpBtn.title = "How to use Spotlight (examples & keyboard shortcuts)";
  const closeBtn = el("button", "spotlight-close", "Esc");
  closeBtn.title = "close Spotlight (Esc)";
  top.append(modeBadge, chipsBox, input, helpBtn, closeBtn);

  const summaryBar = el("div", "spotlight-summary");
  const list = el("div", "spotlight-list");
  const footer = el("div", "spotlight-footer");
  footer.append(
    el("span", "spotlight-hint", "↵ run / select"),
    el("span", "spotlight-hint", "⇧↵ actions (hide / isolate)"),
    el("span", "spotlight-hint", "⇥ Tab: pin filter chip"),
    el("span", "spotlight-hint", "⌥↵ live toggle"),
    el("span", "spotlight-hint", "? help & examples"),
  );

  card.append(top, summaryBar, list, footer);
  root.replaceChildren(card);

  let activeIndex = 0;
  let currentItems: readonly SpotlightItem[] = [];
  let rowElements: HTMLElement[] = [];

  const dispatchAll = (events: readonly Event[], keepOpen?: boolean) => {
    for (const ev of events) store.dispatch(ev);
    if (!keepOpen && store.get().spotlight.open) {
      store.dispatch({ type: "closeSpotlight" });
    } else if (keepOpen && typeof input.focus === "function") {
      input.focus();
    }
  };

  const executeItem = (item: SpotlightItem, mods: { shift?: boolean; alt?: boolean; ctrl?: boolean } = {}) => {
    if (mods.shift && item.shiftEvents?.length) {
      dispatchAll(item.shiftEvents, true);
      return;
    }
    if (mods.alt && item.altEvents?.length) {
      dispatchAll(item.altEvents, true);
      return;
    }
    if (mods.ctrl && item.ctrlEvents?.length) {
      dispatchAll(item.ctrlEvents, false);
      return;
    }
    if (item.events.length === 0) return;
    dispatchAll(item.events, item.keepOpen);
  };

  const syncActiveRow = () => {
    rowElements.forEach((r, i) => {
      const on = i === activeIndex;
      r.classList.toggle("active", on);
      if (on && typeof r.scrollIntoView === "function") {
        r.scrollIntoView({ block: "nearest" });
      }
    });
  };

  const unlisten: (() => void)[] = [];

  const onKeyDown = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "k") {
      e.preventDefault();
      const s = store.get();
      store.dispatch(s.spotlight.open ? { type: "closeSpotlight" } : { type: "openSpotlight" });
      return;
    }
    const s = store.get();
    if (!s.spotlight.open) return;

    if (e.key === "Escape") {
      e.preventDefault();
      if (s.spotlight.mode !== "root") {
        store.dispatch({ type: "setSpotlightMode", mode: "root" });
      } else {
        store.dispatch({ type: "closeSpotlight" });
      }
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (currentItems.length > 0) {
        activeIndex = (activeIndex + 1) % currentItems.length;
        syncActiveRow();
      }
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (currentItems.length > 0) {
        activeIndex = (activeIndex - 1 + currentItems.length) % currentItems.length;
        syncActiveRow();
      }
      return;
    }
    if (e.key === "Backspace" && input.value === "") {
      if (s.spotlight.mode !== "root") {
        e.preventDefault();
        store.dispatch({ type: "setSpotlightMode", mode: "root" });
        return;
      }
      const last = s.spotlight.clauses.at(-1);
      if (last) {
        e.preventDefault();
        store.dispatch({ type: "removeSpotlightClause", id: last.id });
        return;
      }
    }
    if (e.key === "Tab") {
      // Always prevent default so Tab never moves focus out of the Spotlight input.
      e.preventDefault();
      const activeItem = currentItems[activeIndex];
      const targetItem =
        activeItem?.tabClauses?.length || activeItem?.tabClause
          ? activeItem
          : currentItems.find((it) => (it.tabClauses && it.tabClauses.length > 0) || it.tabClause !== undefined);
      const clausesToPin = targetItem?.tabClauses ?? (targetItem?.tabClause ? [targetItem.tabClause] : undefined);
      if (clausesToPin && clausesToPin.length > 0) {
        for (const clause of clausesToPin) {
          store.dispatch({ type: "addSpotlightClause", clause });
        }
        store.dispatch({ type: "setSpotlightQuery", query: "" });
      }
      if (typeof input.focus === "function") input.focus();
      return;
    }
    if (e.key === "Enter") {
      const item = currentItems[activeIndex];
      if (!item) return;
      e.preventDefault();
      executeItem(item, { shift: e.shiftKey, alt: e.altKey, ctrl: e.ctrlKey || e.metaKey });
    }
  };
  window.addEventListener("keydown", onKeyDown);

  unlisten.push(
    () => window.removeEventListener("keydown", onKeyDown),
    listen(root, "click", (e) => {
      if (e.target === root) store.dispatch({ type: "closeSpotlight" });
    }),
    listen(helpBtn, "click", () => {
      const cur = store.get().spotlight.mode;
      store.dispatch({ type: "setSpotlightMode", mode: cur === "help" ? "root" : "help" });
      if (typeof input.focus === "function") input.focus();
    }),
    listen(closeBtn, "click", () => store.dispatch({ type: "closeSpotlight" })),
    listen(modeBadge, "click", () => store.dispatch({ type: "setSpotlightMode", mode: "root" })),
    listen(input, "input", () => {
      activeIndex = 0;
      store.dispatch({ type: "setSpotlightQuery", query: input.value });
    }),
  );

  const render = (s: State, prev: State | null) => {
    const wasOpen = prev?.spotlight.open ?? false;
    root.hidden = !s.spotlight.open;
    if (!s.spotlight.open) return;

    if (
      prev &&
      wasOpen &&
      s.spotlight === prev.spotlight &&
      s.decls === prev.decls &&
      s.visibility === prev.visibility &&
      s.selection === prev.selection &&
      s.inspector === prev.inspector &&
      s.phase === prev.phase &&
      s.stage === prev.stage &&
      s.sequence === prev.sequence &&
      s.results === prev.results &&
      s.contour === prev.contour &&
      s.deform === prev.deform &&
      s.clip === prev.clip &&
      s.windows === prev.windows
    ) {
      return;
    }

    if (!wasOpen || s.spotlight.mode !== prev?.spotlight.mode || s.spotlight.clauses !== prev?.spotlight.clauses) {
      activeIndex = 0;
    }

    const view = spotlightViewOf(s);
    currentItems = view.items;
    if (activeIndex >= currentItems.length) activeIndex = Math.max(0, currentItems.length - 1);

    input.placeholder = view.placeholder;
    if (input.value !== s.spotlight.query) input.value = s.spotlight.query;

    modeBadge.hidden = view.mode === "root";
    modeBadge.textContent = view.mode === "root" ? "" : `← ${view.modeTitle}`;
    helpBtn.textContent = view.mode === "help" ? "← Back" : "? Help";

    // Filter chips
    const chipEls = view.clauses.map((c) => {
      const chip = el("span", "spotlight-chip");
      chip.append(el("span", "spotlight-chip-label", c.label));
      const rm = el("button", "spotlight-chip-rm", "×");
      rm.title = "remove filter chip";
      rm.addEventListener("click", () => store.dispatch({ type: "removeSpotlightClause", id: c.id }));
      chip.append(rm);
      return chip;
    });
    chipsBox.hidden = chipEls.length === 0;
    chipsBox.replaceChildren(...chipEls);

    summaryBar.hidden = view.summary === null;
    summaryBar.textContent = view.summary ?? "";

    // Items list
    rowElements = [];
    const listChildren: HTMLElement[] = [];
    let lastSection = "";
    view.items.forEach((item, idx) => {
      if (item.section !== lastSection) {
        lastSection = item.section;
        listChildren.push(el("div", "spotlight-section", item.section));
      }
      const row = el("div", "spotlight-row" + (idx === activeIndex ? " active" : ""));
      row.dataset["itemId"] = item.id;
      if (item.decl) row.dataset["decl"] = item.decl;

      const main = el("div", "spotlight-row-main");
      main.append(el("span", "spotlight-row-title", item.title));
      const rightMeta = el("span", "spotlight-row-meta");
      if (item.badge) rightMeta.append(el("span", "spotlight-row-badge", item.badge));
      const keyHints: string[] = [];
      if (item.tabClause || item.tabClauses?.length) keyHints.push("⇥ Tab: chip");
      if (item.shiftEvents?.length) keyHints.push(item.id.startsWith("decl:") ? "⇧↵ +select" : "⇧↵ actions");
      if (keyHints.length > 0) rightMeta.append(el("span", "spotlight-row-keys", keyHints.join(" · ")));
      main.append(rightMeta);
      row.append(main);
      if (item.subtitle) row.append(el("div", "spotlight-row-sub", item.subtitle));

      row.addEventListener("mouseenter", () => {
        activeIndex = idx;
        syncActiveRow();
      });
      row.addEventListener("click", (e) => {
        const me = e as MouseEvent;
        activeIndex = idx;
        executeItem(item, { shift: me.shiftKey, alt: me.altKey, ctrl: me.ctrlKey || me.metaKey });
      });

      rowElements.push(row);
      listChildren.push(row);
    });

    if (view.items.length === 0) {
      listChildren.push(
        el(
          "div",
          "spotlight-empty",
          "No matches — try partial names (e.g. beam_V50x70), combine with 'and'/'or', or click '? Help' above",
        ),
      );
    }
    list.replaceChildren(...listChildren);

    if (!wasOpen && typeof input.focus === "function") {
      input.focus();
      if (typeof input.select === "function") input.select();
    }
  };

  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
    if (created) root.remove();
  };
}
