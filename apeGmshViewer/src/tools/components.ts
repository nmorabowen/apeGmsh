// The nodal-component picker the tools share (Step readout's range, Node history's chart): the
// stage's components from `ToolContext`, the choice kept while the stage offers it.

/** The component a new stage starts on: the first x displacement, else the first component. */
export function defaultComponent(components: readonly string[]): string | null {
  return components.find((c) => /(^|[_/])(displacement_x|Ux)$/i.test(c)) ?? components[0] ?? null;
}

/** A `<select>` of a stage's nodal components; `sync` fills it for a stage and returns the choice. */
export function componentPicker(doc: Document, title: string, onPick: (c: string) => void) {
  const select = doc.createElement("select");
  select.title = title;
  let chosen: string | null = null;
  select.addEventListener("change", () => {
    chosen = select.value;
    onPick(chosen);
  });
  return {
    select,
    /** Fill the options for `comps` and return the chosen component (the default when the stage lacks the last one). */
    sync(comps: readonly string[]): string | null {
      if (chosen === null || !comps.includes(chosen)) chosen = defaultComponent(comps);
      if (select.options.length !== comps.length || [...select.options].some((o, i) => o.value !== comps[i])) {
        select.replaceChildren(
          ...comps.map((name) => {
            const o = doc.createElement("option");
            o.value = name;
            o.textContent = name;
            return o;
          }),
        );
      }
      if (chosen !== null) select.value = chosen;
      return chosen;
    },
  };
}
