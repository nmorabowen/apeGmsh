// The code `npm run profile` puts into the page, as functions serialised with
// toString(): no import, no closure. The viewer's source is not touched; the
// store is reached through a one-shot debugger breakpoint in Store.dispatch.

export interface Probe {
  store: { get(): unknown; dispatch(e: unknown): void; listeners: Listener[] };
  /** the raw listeners seen so far; a subscriber's id is its index here */
  seen: Listener[];
  reset(): void;
  read(): { subscribers: { id: number; calls: number; totalMs: number; maxMs: number; samples: number[] }[]; events: Record<string, number>; dispatchMs: number[]; intervals: number[]; longTasks: number[] };
}

type Listener = (s: unknown, prev: unknown) => void;

/**
 * Runs with `this` = the Store (evaluateOnCallFrame). Wraps each listener for
 * the length of one dispatch only, so `subscribe` and `unsubscribe` still see
 * the raw functions outside a dispatch, and records each one's time. A nested
 * dispatch's cost also counts inside the subscriber that made it, as it does
 * for the user.
 */
export function installProbe(this: Probe["store"]): void {
  const store = this;
  const seen: Listener[] = [];
  const stats = new Map<number, { calls: number; totalMs: number; maxMs: number; samples: number[] }>();
  let events: Record<string, number> = {};
  let dispatchMs: number[] = [];
  let intervals: number[] = [];
  let longTasks: number[] = [];
  const raw = new WeakMap<Listener, Listener>();
  const idOf = (l: Listener) => {
    let i = seen.indexOf(l);
    if (i < 0) i = seen.push(l) - 1;
    return i;
  };
  const wrap = (l: Listener): Listener => {
    if (raw.has(l)) return l;
    const id = idOf(l);
    const timed: Listener = (s, prev) => {
      const t = performance.now();
      try {
        l(s, prev);
      } finally {
        const ms = performance.now() - t;
        let st = stats.get(id);
        if (!st) stats.set(id, (st = { calls: 0, totalMs: 0, maxMs: 0, samples: [] }));
        st.calls++;
        st.totalMs += ms;
        st.maxMs = Math.max(st.maxMs, ms);
        st.samples.push(ms);
      }
    };
    raw.set(timed, l);
    return timed;
  };
  const dispatch = store.dispatch;
  let depth = 0;
  store.dispatch = function (e: unknown) {
    const type = (e as { type?: string }).type ?? "?";
    events[type] = (events[type] ?? 0) + 1;
    const outer = depth === 0;
    depth++;
    if (outer) store.listeners = store.listeners.map(wrap);
    const t = performance.now();
    try {
      dispatch.call(store, e);
    } finally {
      depth--;
      if (outer) {
        dispatchMs.push(performance.now() - t);
        store.listeners = store.listeners.map((l) => raw.get(l) ?? l);
      }
    }
  };
  // Frames and long tasks, for the whole life of the page; `reset` starts a scenario's window.
  let last = performance.now();
  const tick = (now: number) => {
    intervals.push(now - last);
    last = now;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) longTasks.push(e.duration);
  }).observe({ entryTypes: ["longtask"] });
  (globalThis as unknown as { __agvProbe: Probe }).__agvProbe = {
    store,
    seen,
    reset() {
      stats.clear();
      events = {};
      dispatchMs = [];
      intervals = [];
      longTasks = [];
      last = performance.now();
    },
    read() {
      return {
        subscribers: [...stats].map(([id, s]) => ({ id, ...s })),
        events,
        dispatchMs,
        intervals,
        longTasks,
      };
    },
  };
}
