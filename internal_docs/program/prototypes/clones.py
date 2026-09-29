"""Line-window clone detector (exact after normalization).

Normalization: strip, drop comment-only lines and trailing comments, drop blank lines,
drop trivial lines (closing brackets, 'else:', 'try:', 'pass', 'return', 'continue',
'break', 'raise', import lines). Docstring lines are KEPT (copied docs count) but tracked.
Window W normalized lines; windows hashed; clone regions = maximal runs of matching windows
between file pairs. Reports dup lines per file pair and per package pair.
usage: clones.py ROOTDIR W [min_region]
"""
import os, sys, io, tokenize, hashlib, re, json
from collections import defaultdict

ROOT = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MINR = int(sys.argv[3]) if len(sys.argv) > 3 else W
REPO = r"C:\Users\nmora\Github\apeGmsh\.claude\worktrees\apegmsh-strains-assessment-c5f7dd"
TRIVIAL = re.compile(r"^([\)\]\}],?|else:|try:|finally:|pass|return|continue|break|raise|\"\"\"|'''|r\"\"\"|\)\s*->.*:|\):|\],|\),|\}\)|\]\)|\)\))$")

def norm_lines(path):
    src = open(path, "rb").read().decode("utf-8", "replace")
    raw = src.splitlines()
    # remove comments via tokenize
    comment_cols = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                comment_cols[tok.start[0]] = tok.start[1]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    out = []
    for i, line in enumerate(raw, 1):
        if i in comment_cols:
            line = line[: comment_cols[i]]
        s = line.strip()
        if not s:
            continue
        if s.startswith("import ") or s.startswith("from ") and " import " in s:
            continue
        if TRIVIAL.match(s):
            continue
        s = re.sub(r"\s+", " ", s)
        out.append((i, s))
    return out

files = []
for root, dirs, fs in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in fs:
        if f.endswith(".py"):
            files.append(os.path.join(root, f))
data = {}
for p in files:
    rel = os.path.relpath(p, REPO).replace(os.sep, "/")
    data[rel] = norm_lines(p)

index = defaultdict(list)  # hash -> [(file, start_idx)]
for f, lines in data.items():
    for k in range(len(lines) - W + 1):
        h = hashlib.md5("\n".join(l for _, l in lines[k:k + W]).encode()).hexdigest()
        index[h].append((f, k))

# mark duplicated normalized-line indices per file (lines that belong to any window seen elsewhere)
dup_idx = defaultdict(set)
pair_lines = defaultdict(set)  # (fa, fb) -> set of line numbers in fa
for h, occ in index.items():
    if len(occ) < 2:
        continue
    for (f, k) in occ:
        others = {g for (g, kk) in occ if (g, kk) != (f, k)}
        for j in range(k, k + W):
            dup_idx[f].add(j)
        for g in others:
            for j in range(k, k + W):
                pair_lines[(f, g)].add(data[f][j][0])

tot_norm = sum(len(v) for v in data.values())
tot_dup = sum(len(v) for v in dup_idx.values())
print(f"files={len(data)} normalized_lines={tot_norm} lines_in_clone_windows={tot_dup} ({100*tot_dup/max(tot_norm,1):.1f}%)")
# per file
per_file = sorted(((len(v), f) for f, v in dup_idx.items()), reverse=True)
print("\nTop files by cloned normalized lines:")
for n, f in per_file[:40]:
    print(f"  {n:5d}/{len(data[f]):5d} {f}")
# pairs (unordered, cross-file)
seen = set()
pairs = []
for (a, b), ls in pair_lines.items():
    if a == b:
        continue
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    la = len(pair_lines[(a, b)]); lb = len(pair_lines.get((b, a), ()))
    pairs.append((min(la, lb), la, lb, a, b))
pairs.sort(reverse=True)
print("\nTop cross-file clone pairs (dup normalized lines in A / in B):")
for m, la, lb, a, b in pairs[:60]:
    if m < MINR:
        break
    print(f"  {la:5d} {lb:5d}  {a}  <->  {b}")
# within-file
print("\nTop within-file clones:")
wf = sorted(((len(ls), a) for (a, b), ls in pair_lines.items() if a == b), reverse=True)
for n, a in wf[:25]:
    print(f"  {n:5d} {a}")
json.dump({"pairs": pairs[:500], "per_file": per_file[:500]}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "clones_%s_%d.json" % (os.path.basename(ROOT.rstrip('/\\')), W)), "w"))
