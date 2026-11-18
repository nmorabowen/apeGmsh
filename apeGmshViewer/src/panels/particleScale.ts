// The MPM particles' colour scale in the viewport (`#particle-scale`; ADR 0134 D5,
// M2-S4 #2983, the owner's D7 ruling on #2992). It shows whenever the particles are
// drawn coloured by a field, and is the contour's and line diagram's own scale
// (ui/scaleChrome.ts: `fillScale` draws it, `mountScaleChrome` drags it and remembers
// its place), filled from `particleScaleOf` (state/results.ts): the field and
// component in the header's Units, the colormap the points are drawn with, the values,
// and, when some particles have no value (NaN), the no-value chip the contour card
// shows, with how many.
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors` and `ui/`.

import { NO_DATA_RGB, particleScaleOf, rampCss, type ParticleScale } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el } from "../ui/dom.ts";
import { fillScale, mountScaleChrome, type ScaleChrome } from "../ui/scaleChrome.ts";

const css = (rgb: readonly number[]): string => `rgb(${rgb.map((c) => Math.round(c * 255)).join(",")})`;

/** Draw `v` into the scale: the shared title, bar, values and range line, then the no-value row when there is one. */
export function fillParticleScale(scale: HTMLElement, v: ParticleScale, unlisten: (() => void)[], onOrient: () => void): void {
  fillScale(scale, { title: v.title, tooltip: v.tooltip, ramp: rampCss(v.cmap), ticks: v.ticks, range: v.range }, unlisten, onOrient);
  if (v.noValue === null) return;
  const row = el("div", "legend-row particle-novalue");
  const chip = el("span", "chip");
  chip.style.backgroundColor = css(NO_DATA_RGB);
  const text = el("span", "contour-nodata", v.noValue);
  text.title = "a particle whose material does not answer the field (a STATE_* value it does not hold) is NaN: never zero, never a ramp colour, drawn in this grey";
  row.append(chip, text);
  scale.append(row);
}

export function mountParticleScale(store: Store): () => void {
  const scale = byId("particle-scale");
  const unlisten: (() => void)[] = [];
  // undefined until the first render, which always draws (or hides) the scale.
  let drawn: string | null | undefined;
  const render = (s: State, force = false): void => {
    const v = particleScaleOf(s);
    const key = v === null ? null : JSON.stringify(v);
    if (!force && key === drawn) return;
    drawn = key;
    for (const u of unlisten.splice(0)) u();
    if (v === null) {
      scale.hidden = true;
      scale.replaceChildren();
      return;
    }
    fillParticleScale(scale, v, unlisten, () => {
      render(store.get(), true);
      chrome?.clamp();
      chrome?.save();
    });
    scale.hidden = false;
    chrome?.clamp();
  };
  // Only a real window has a place to drag in; the Node tests mount this panel without one.
  const chrome: ScaleChrome | null = typeof window === "undefined" ? null : mountScaleChrome(scale, "apegmsh.viewer.particleScale");
  render(store.get());
  const unsubscribe = store.subscribe((s) => render(s));
  return () => {
    unsubscribe();
    chrome?.dispose();
    for (const u of unlisten) u();
  };
}
