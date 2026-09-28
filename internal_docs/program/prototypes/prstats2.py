import re, collections, datetime as dt, random
SP = r"C:/Users/nmora/AppData/Local/Temp/claude/C--Users-nmora-Github-apeGmsh--claude-worktrees-apegmsh-strains-assessment-c5f7dd/8e66ae63-de64-40a9-90ff-defa6b33bbfd/scratchpad/panel/P8"
commits = []; cur = None
for line in open(SP + "/log_names.txt", encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("@@@"):
        sha, date, subj = line[3:].split("\t", 2); cur = [sha, dt.date.fromisoformat(date), subj, []]; commits.append(cur)
    elif line.strip() and cur is not None: cur[3].append(line.strip())
pr_re = re.compile(r"\(#(\d+)\)\s*$"); merge_re = re.compile(r"^Merge pull request #(\d+)")
prs = [(int((pr_re.search(s) or merge_re.match(s)).group(1)), sha, d, s, files) for sha, d, s, files in commits if (pr_re.search(s) or merge_re.match(s))]
today = dt.date(2026, 9, 28)
w90 = [p for p in prs if (today - p[2]).days <= 90]
empty = [p for p in w90 if not p[4]]
print("PRs 90d:", len(w90), "with empty file list (merge commits):", len(empty))
w90f = [p for p in w90 if p[4]]
strict = lambda s: bool(re.match(r"^(fix|hotfix|bugfix)(\(|:|!|\b)", s, re.I))
loose = lambda s: bool(re.search(r"\bfix(es|ed)?\b", s, re.I))
print("strict fix:", sum(strict(p[3]) for p in w90f), "loose fix:", sum(loose(p[3]) for p in w90f), "of", len(w90f))
for lo, hi, lab in [(dt.date(2026,7,1), dt.date(2026,7,31), "Jul"), (dt.date(2026,8,1), dt.date(2026,8,31), "Aug"), (dt.date(2026,9,1), dt.date(2026,9,30), "Sep")]:
    ps = [p for p in w90f if lo <= p[2] <= hi]
    print(f"  {lab}: strict {sum(strict(p[3]) for p in ps)}/{len(ps)}  loose {sum(loose(p[3]) for p in ps)}/{len(ps)}  CHANGELOG {sum('CHANGELOG.md' in p[4] for p in ps)}/{len(ps)}")
src = lambda fs: set(f for f in fs if f.startswith("src/"))
def prox(target, pool, days=7):
    hits = 0; n = 0
    for num, sha, d, s, files in target:
        fs = src(files)
        if not fs: continue
        n += 1
        if any(d2 < d and (d - d2).days <= days and (fs & src(f2)) for n2, s2, d2, ss2, f2 in pool if s2 != sha): hits += 1
    return hits, n
fx = [p for p in w90f if loose(p[3])]; nonfx = [p for p in prs if not loose(p[3])]
print("loose-fix PRs touching a src file a NON-fix PR touched <=7d before:", prox(fx, nonfx))
print("CONTROL: non-fix PRs touching a src file ANY other PR touched <=7d before:", prox([p for p in w90f if not loose(p[3])], prs))
print("CONTROL: fix PRs touching a src file ANY other PR touched <=7d before:", prox(fx, prs))
idx = ("CHANGELOG.md", "architecture/decisions/README.md", "_api_index.json", ".claude/skills/apegmsh-helper", "skills/apegmsh/")
n = sum(1 for p in w90f if any(any(i in f for i in idx) for f in p[4]))
print(f"PRs 90d (with files) touching index/generated/skill file: {n}/{len(w90f)} ({100*n/len(w90f):.0f}%)")
print("PRs 90d touching apesees.py:", sum('src/apeGmsh/opensees/apesees.py' in p[4] for p in w90f), " build.py:", sum('src/apeGmsh/opensees/_internal/build.py' in p[4] for p in w90f))
# >2000-line hubs touched
hubs = ["opensees/apesees.py","_internal/build.py","viewers/results_viewer.py","mesh/_femdata_h5_io.py","emitter/h5.py","material/nd.py","core/ConstraintsComposite.py","mesh/_compose.py","core/_model_geometry.py","opensees/_response_catalog.py","capture/_domain.py","mesh/FEMData.py","_diagram_settings_tab.py","emitter/h5_reader.py","element/solid.py"]
print("PRs 90d touching ANY >2000-line hub:", sum(any(any(h in f for h in hubs) for f in p[4]) for p in w90f), "/", len(w90f))
import sys
for key in ["#784)", "#614)", "#793)", "#1132)", "#1140)", "#834)"]:
    for p in prs:
        if p[3].endswith(key):
            vf = [f for f in p[4] if f.startswith("src/apeGmsh/viewers")]
            sys.stdout.buffer.write(f"{key} {p[2]} total files {len(p[4])} src {len(src(p[4]))} viewers {len(vf)} | {p[3][:80]}\n".encode("utf-8"))
            if key == "#784)": sys.stdout.buffer.write(("   " + "\n   ".join(sorted(src(p[4]))) + "\n").encode("utf-8"))
