import json, os, sys
from collections import defaultdict, deque
HERE = os.path.dirname(os.path.abspath(__file__))
g = json.load(open(os.path.join(HERE, "graph.json")))
t = json.load(open(os.path.join(HERE, "testrefs.json")))
modules = g["modules"]
edges = g["edges"]

def is_public(m):
    return not any(p.startswith("_") for p in m.split(".")[1:])
mains = [m for m in modules if m.endswith("__main__")]

def closure(roots, kinds=("eager", "lazy", "string"), drop_scopes=(), drop_modules=()):
    adj = defaultdict(set)
    for (s, scope, k, d, ln) in edges:
        if k not in kinds:
            continue
        if any(s == dm and (scope == sp or scope.startswith(sp + ".")) for dm, sp in drop_scopes):
            continue
        adj[s].add(d)
    seen = set(r for r in roots if r not in drop_modules)
    q = deque(seen)
    while q:
        m = q.popleft()
        for d in adj[m]:
            if d in drop_modules or d in seen:
                continue
            seen.add(d); q.append(d)
    return seen

def loc(ms):
    return sum(modules[m]["loc"] for m in ms)

test_importers = defaultdict(set)
for tf, ms in list(t["refs"].items()) + list(t["str_refs"].items()):
    for m in ms:
        test_importers[m].add(tf)

LEG_MODS = ["apeGmsh.viewers.results_viewer", "apeGmsh.viewers.web_viewer", "apeGmsh.viewers.animation"]
R = "apeGmsh.results.Results"
VR = "apeGmsh.viewers.render"
LEG_SCOPES = [(R, "Results.export_animation"), (R, "Results.render"), (R, "Results.render_pack"),
              (R, "Results.show_web"), (R, "Results.serve_web")]
VR_SCOPES = [(VR, s) for s in ("render_results", "render_pack", "_pack_primary_component", "_apply_deform",
                                "_read_deform_field", "_has_static_reactions", "_extrema_node", "_try_history",
                                "_ensure_stage", "_resolve_step", "_require_deform_field")]

variant = sys.argv[1] if len(sys.argv) > 1 else "public"
roots = (["apeGmsh"] + mains) if variant == "strict" else (["apeGmsh"] + mains + [m for m in modules if is_public(m)])
roots_no_leg = [r for r in roots if r not in LEG_MODS]
full = closure(roots)
cut = closure(roots_no_leg, drop_scopes=LEG_SCOPES + VR_SCOPES, drop_modules=LEG_MODS)
only = sorted(full - cut)
print(f"[{variant}] reachable only via legacy (results_viewer/web_viewer/animation + Results.render*/show_web/export_animation):")
print("  modules:", len(only), " LOC:", loc(only))
by_pkg = defaultdict(lambda: [0, 0])
for m in only:
    pk = ".".join(m.split(".")[:3])
    by_pkg[pk][0] += 1; by_pkg[pk][1] += modules[m]["loc"]
for k, v in sorted(by_pkg.items(), key=lambda kv: -kv[1][1]):
    print(f"   {k:45s} n={v[0]:3d} loc={v[1]:6d}")
print()
for m in only:
    print(f"  {modules[m]['loc']:6d} {m}  tests={len(test_importers[m])}")
tf = set()
for m in only:
    tf |= test_importers[m]
print("\n test files importing any legacy-only module:", len(tf))
json.dump({"only": only, "tests": sorted(tf)}, open(os.path.join(HERE, f"legacy_{variant}.json"), "w"))
