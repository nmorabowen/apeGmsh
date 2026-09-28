#!/usr/bin/env python
"""nav: AST-only code navigation for apeGmsh maintainers (P6 prototype).

Stdlib only. Never imports apeGmsh, so it reads the checkout it is pointed at
(no editable-install trap) and costs no gmsh/Qt start-up. Line ranges are
computed on demand from the working tree and cached per file (mtime+size);
nothing is committed, so nothing can go stale or collide in a merge.

Commands (every one prints a bounded, grep-shaped answer):
  map  FILE            classes/functions with line ranges + first doc line
  pkg  PACKAGE         modules: lines, fan-in (eager/lazy), first doc line, hub flag
  where NAME           definitions of NAME (class/def/method) with ranges
  at   FILE:LINE       the enclosing symbol chain of a line
  refs NAME [--kind call,attr,name,str,import]
                       code references (comments/docstrings excluded),
                       grouped by file and enclosing function
  h5   PATH-FRAGMENT   code users of an HDF5 path, classified
                       write / read / probe / use (one hop through constants)
  impl PROTO.METHOD    classes implementing METHOD of Protocol PROTO
  family BASE | --names a,b,c
                       touch-point recipe: every enumeration of the family
                       (literal tables, isinstance dispatch, per-member
                       methods, doc/data files) with coverage and gaps
"""
from __future__ import annotations

import argparse
import ast
import builtins
import marshal
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CACHE_VERSION = 6
SCAN_DIRS = ("src/apeGmsh", "tests", "scripts")
TEXT_GLOBS = ("docs/**/*.md", "skills/apegmsh/**/*.md",
              "src/apeGmsh/**/*.md", "src/apeGmsh/**/*.json")
HUB_LINES = 2000
_STR_RE = re.compile(r"^[A-Za-z_/][\w/.\-{}]{1,119}$")
_WRITE_CALLS = {"create_group", "create_dataset", "require_group",
                "require_dataset", "create", "copy", "move"}
_READ_CALLS = {"get", "visit", "visititems", "keys", "values", "items"}
_STOP = set(dir(builtins)) | {"self", "cls", "np", "Any", "Optional", "annotations"}

# tuple layouts (marshal-friendly):
# sym  = (qual, kind, start, end, doc1, bases, methods)
# ref  = (token, line, kind, scope, rw)
# enum = (line, target, kind, names)
# imp  = (target, line, lazy, type_checking)
S_QUAL, S_KIND, S_START, S_END, S_DOC, S_BASES, S_METHODS = range(7)
R_TOK, R_LINE, R_KIND, R_SCOPE, R_RW = range(5)


def _first_line(doc):
    if not doc:
        return ""
    for ln in doc.strip().splitlines():
        ln = ln.strip()
        if ln:
            return ln[:110]
    return ""


def _term(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _str_of(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str)
                       else "{}" for v in node.values)
    return None


def _dotted(rel):
    p = rel[:-3] if rel.endswith(".py") else rel
    p = p.replace("/", ".")
    if p.startswith("src."):
        p = p[4:]
    if p.endswith(".__init__"):
        p = p[: -len(".__init__")]
    return p


def _is_doc(stmt):
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


# -------------------------------------------------------------------- parse
def parse_file(root, rel):
    """Return a marshal-friendly dict for one module (single AST pass)."""
    p = Path(root) / rel
    src = p.read_text(encoding="utf-8-sig", errors="replace")
    lines = src.splitlines()
    mod = {"rel": rel, "dotted": _dotted(rel), "nlines": len(lines), "doc1": "",
           "syms": [], "banners": [], "refs": [], "imports": [], "enums": [],
           "consts": {}}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except SyntaxError:
        return mod
    mod["doc1"] = _first_line(ast.get_docstring(tree))
    syms, refs, imps, enums, consts = (mod["syms"], mod["refs"], mod["imports"],
                                       mod["enums"], mod["consts"])
    dotted = mod["dotted"]
    is_init = rel.endswith("__init__.py")

    def resolve(module, level):
        if level == 0:
            return module or ""
        base = dotted.split(".")
        if not is_init:
            base = base[:-1]
        if level > 1:
            base = base[: len(base) - (level - 1)]
        return ".".join(base + ([module] if module else []))

    def rw_of(chain):
        # chain: ancestors nearest-first, starting with the string node itself
        cur = chain[0]
        i = 1
        while i < len(chain) and isinstance(chain[i], (ast.BinOp, ast.JoinedStr,
                                                       ast.FormattedValue, ast.Tuple)):
            cur = chain[i]
            i += 1
        par = chain[i] if i < len(chain) else None
        if isinstance(par, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn))
                                                for o in par.ops):
            return "probe"
        if isinstance(par, ast.Subscript) and par.slice is cur:
            return "write" if isinstance(par.ctx, (ast.Store, ast.Del)) else "read"
        if isinstance(par, ast.Call):
            fn = _term(par.func) or ""
            if fn in _WRITE_CALLS:
                return "write"
            if fn in _READ_CALLS:
                return "read"
        if isinstance(par, (ast.Assign, ast.AnnAssign)):
            return "const"
        return "use"

    def literal(node, elts, chain, scope):
        names = set()
        for e in elts:
            if e is None:
                continue
            t = _term(e) if isinstance(e, (ast.Name, ast.Attribute)) else _str_of(e)
            if t:
                names.add(t)
        if len(names) >= 2:
            par = None
            for anc in chain[1:]:
                if isinstance(anc, ast.Call):
                    continue
                par = anc
                break
            target = scope
            if isinstance(par, ast.Assign) and par.targets:
                target = ast.unparse(par.targets[0])[:60]
            elif isinstance(par, ast.AnnAssign):
                target = ast.unparse(par.target)[:60]
            enums.append((node.lineno, target, "literal", tuple(sorted(names))))

    def visit(node, chain, stack, fn_depth, tc):
        scope = ".".join(stack) if stack else "<module>"
        t = type(node)
        if t in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef):
            is_cls = t is ast.ClassDef
            qual = ".".join(stack + [node.name])
            if is_cls:
                kind = "class" if not stack else "nested"
                bases = tuple(filter(None, (_term(b) for b in node.bases)))
                methods = tuple(n.name for n in node.body
                                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
            else:
                kind = "method" if (stack and fn_depth == 0) else ("def" if not stack else "nested")
                bases, methods = (), ()
            syms.append((qual, kind, node.lineno, node.end_lineno or node.lineno,
                         _first_line(ast.get_docstring(node)), bases, methods))
            nchain = [node] + chain
            for d in node.decorator_list:
                visit(d, [d] + nchain, stack, fn_depth, tc)
            if is_cls:
                for b in node.bases:
                    visit(b, [b] + nchain, stack, fn_depth, tc)
            else:
                a = node.args
                for arg in (a.posonlyargs + a.args + a.kwonlyargs):
                    if arg.arg not in _STOP:
                        refs.append((arg.arg, node.lineno, "param", qual, ""))
                visit(node.args, [node.args] + nchain, stack, fn_depth, tc)
                if node.returns is not None:
                    visit(node.returns, [node.returns] + nchain, stack, fn_depth, tc)
            body = node.body[1:] if node.body and _is_doc(node.body[0]) else node.body
            nd = fn_depth if is_cls else fn_depth + 1
            for s in body:
                visit(s, [s] + nchain, stack + [node.name], nd, tc)
            return
        if t is ast.If:
            test = node.test
            is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
            visit(test, [test] + chain, stack, fn_depth, tc)
            for s in node.body:
                visit(s, [s] + chain, stack, fn_depth, tc or is_tc)
            for s in node.orelse:
                visit(s, [s] + chain, stack, fn_depth, tc)
            return
        if t is ast.Import:
            for a in node.names:
                imps.append((a.name, node.lineno, fn_depth > 0, tc))
                refs.append((a.name.split(".")[-1], node.lineno, "import", scope, ""))
            return
        if t is ast.ImportFrom:
            tgt = resolve(node.module, node.level)
            for a in node.names:
                imps.append((f"{tgt}.{a.name}" if a.name != "*" else tgt,
                             node.lineno, fn_depth > 0, tc))
                refs.append((a.name, node.lineno, "import", scope, ""))
            return
        if t is ast.Call:
            f = node.func
            name = _term(f)
            if name:
                refs.append((name, node.lineno, "call", scope, ""))
            if isinstance(f, ast.Name) and f.id in ("isinstance", "issubclass") \
                    and len(node.args) == 2:
                a = node.args[1]
                for e in (a.elts if isinstance(a, ast.Tuple) else [a]):
                    n = _term(e)
                    if n:
                        enums.append((node.lineno, scope, "isinstance", (n,)))
            # visit func's children but not re-record func itself
            if isinstance(f, ast.Attribute):
                visit(f.value, [f.value, f] + [node] + chain, stack, fn_depth, tc)
            elif not isinstance(f, ast.Name):
                visit(f, [f, node] + chain, stack, fn_depth, tc)
            for a in node.args:
                visit(a, [a, node] + chain, stack, fn_depth, tc)
            for k in node.keywords:
                if k.arg:
                    refs.append((k.arg, node.lineno, "kw", scope, ""))
                visit(k.value, [k.value, k, node] + chain, stack, fn_depth, tc)
            return
        if t is ast.Attribute:
            refs.append((node.attr, node.lineno, "attr", scope, ""))
            visit(node.value, [node.value] + chain, stack, fn_depth, tc)
            return
        if t is ast.Name:
            if node.id not in _STOP:
                refs.append((node.id, node.lineno, "name", scope, ""))
            return
        if t is ast.Constant:
            v = node.value
            if isinstance(v, str) and _STR_RE.match(v):
                refs.append((v, node.lineno, "str", scope, rw_of(chain)))
            return
        if t is ast.JoinedStr:
            s = _str_of(node)
            if s and ("/" in s or "_" in s) and len(s) < 160:
                refs.append((s, node.lineno, "str", scope, rw_of(chain)))
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    visit(v.value, [v.value, v] + chain, stack, fn_depth, tc)
            return
        if t in (ast.List, ast.Tuple, ast.Set):
            literal(node, node.elts, chain, scope)
        elif t is ast.Dict:
            literal(node, node.keys, chain, scope)
        elif t is ast.Assign and not stack and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            s = _str_of(node.value)
            if s is not None:
                consts[node.targets[0].id] = s
        for ch in ast.iter_child_nodes(node):
            visit(ch, [ch] + chain, stack, fn_depth, tc)

    body = tree.body[1:] if tree.body and _is_doc(tree.body[0]) else tree.body
    sys.setrecursionlimit(10000)
    for s in body:
        visit(s, [s, tree], [], 0, False)
    # module-level banners:  # ---- / # Title / # ----
    for i in range(1, len(lines) - 1):
        a, b = lines[i - 1], lines[i]
        if a.startswith(("# ---", "# ===")) and b.startswith("# ") \
                and not b.startswith(("# --", "# ==")):
            mod["banners"].append((i + 1, b.strip("# ").strip()[:90]))
    return mod


def _parse_many(args):
    root, rels = args
    return [parse_file(root, r) for r in rels]


# -------------------------------------------------------------------- index
def repo_root(start):
    try:
        out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return Path(out)
    except Exception:
        return start


def cache_path(root):
    # PROTOTYPE: keep the cache next to this script (the panel's scratch dir),
    # so a read-only review never writes into the repo. The production tool
    # would use `git rev-parse --git-dir` (per worktree, never committed).
    return Path(__file__).resolve().parent / "nav_cache.marshal"


def load_index(root, cache=None, verbose=False, jobs=None):
    cache = cache or cache_path(root)
    t0 = time.perf_counter()
    old = {}
    if cache.is_file():
        try:
            blob = marshal.loads(cache.read_bytes())
            if blob.get("v") == CACHE_VERSION and blob.get("root") == str(root):
                old = blob["mods"]
        except Exception:
            old = {}
    t_load = time.perf_counter() - t0
    mods, keys, todo = {}, {}, []
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = Path(full).relative_to(root).as_posix()
                st = os.stat(full)
                key = (st.st_mtime_ns, st.st_size)
                keys[rel] = key
                hit = old.get(rel)
                if hit and tuple(hit[0]) == key:
                    mods[rel] = hit[1]
                else:
                    todo.append(rel)
    t1 = time.perf_counter()
    if todo:
        if len(todo) > 40:
            n = jobs or max(1, (os.cpu_count() or 2) - 1)
            chunks = [todo[i::n * 4] for i in range(n * 4)]
            with ProcessPoolExecutor(max_workers=n) as ex:
                for res in ex.map(_parse_many, [(str(root), c) for c in chunks if c]):
                    for m in res:
                        mods[m["rel"]] = m
        else:
            for r in todo:
                mods[r] = parse_file(root, r)
        blob = {"v": CACHE_VERSION, "root": str(root),
                "mods": {r: (keys[r], m) for r, m in mods.items()}}
        try:
            cache.write_bytes(marshal.dumps(blob))
        except OSError:
            pass
    if verbose:
        print(f"[index] {len(mods)} files; {len(todo)} parsed in "
              f"{time.perf_counter() - t1:.2f}s; cache load {t_load:.2f}s; "
              f"stat {t1 - t0 - t_load:.2f}s", file=sys.stderr)
    return mods


def _src(mods):
    return {r: m for r, m in mods.items() if r.startswith("src/")}


def _short(rel):
    return rel.replace("src/apeGmsh/", "")


def _find_file(mods, frag):
    frag = frag.replace("\\", "/")
    hits = [m for r, m in mods.items() if r == frag or r.endswith("/" + frag)]
    if not hits:
        sys.exit(f"no file matches {frag!r}")
    if len(hits) > 1:
        src_hits = [h for h in hits if h["rel"].startswith("src/")]
        if len(src_hits) == 1:
            return src_hits[0]
        sys.exit("ambiguous: " + ", ".join(h["rel"] for h in hits[:8]))
    return hits[0]


# ----------------------------------------------------------------- commands
def cmd_map(mods, args):
    m = _find_file(mods, args.file)
    hub = "  HUB" if m["nlines"] > HUB_LINES else ""
    print(f"{m['rel']}  ({m['nlines']} lines{hub})  {m['doc1'][:80]}")
    items = [(b[0], 0, b[1]) for b in m["banners"]]
    for s in m["syms"]:
        if s[S_QUAL].count(".") > args.depth:
            continue
        if s[S_KIND] == "nested" and not args.nested:
            continue
        items.append((s[S_START], 1, s))
    items.sort(key=lambda x: (x[0], x[1]))
    for ln, is_sym, s in items:
        if not is_sym:
            print(f"  == {s}  (L{ln})")
            continue
        size = s[S_END] - s[S_START] + 1
        ind = "  " * (s[S_QUAL].count(".") + 1)
        name = s[S_QUAL].rsplit(".", 1)[-1]
        tag = "class " if s[S_KIND] == "class" else ""
        big = " !" if size >= 300 else ""
        doc = f"  -- {s[S_DOC][:args.width]}" if (s[S_DOC] and args.width) else ""
        print(f"{ind}{tag}{name} {s[S_START]}-{s[S_END]}{big}{doc}")


def _fan_in(mods):
    dotted = {m["dotted"] for m in _src(mods).values()}
    eager, lazy = Counter(), Counter()
    for m in _src(mods).values():
        seen = {}
        for tgt, _ln, lz, _tc in m["imports"]:
            parts = tgt.split(".")
            for k in range(len(parts), 0, -1):
                cand = ".".join(parts[:k])
                if cand in dotted:
                    if cand != m["dotted"]:
                        seen[cand] = seen.get(cand, True) and lz
                    break
        for cand, lz in seen.items():
            (lazy if lz else eager)[cand] += 1
    return eager, lazy


def cmd_pkg(mods, args):
    pref = "src/apeGmsh/" + args.package.strip("/").replace(".", "/")
    eager, lazy = _fan_in(mods)
    rows = [m for r, m in sorted(mods.items())
            if r.startswith(pref + "/") or r == pref + ".py"]
    if not args.deep:
        rows = [m for m in rows if "/" not in m["rel"][len(pref) + 1:]] or rows
    if not rows:
        sys.exit(f"no modules under {pref}")
    tot = sum(m["nlines"] for m in rows)
    print(f"{_short(pref)}: {len(rows)} modules, {tot} lines. in=importing src modules "
          f"(eager+lazy), HUB > {HUB_LINES} lines")
    for m in rows:
        rel = m["rel"][len(pref) + 1:]
        hub = "HUB" if m["nlines"] > HUB_LINES else ""
        e, lz = eager.get(m["dotted"], 0), lazy.get(m["dotted"], 0)
        doc = m["doc1"][:args.width] if args.width else ""
        print(f"  {rel:<38} {m['nlines']:>6} {hub:3} in={e}+{lz:<3} {doc}")


def cmd_where(mods, args):
    n = 0
    for r, m in sorted(mods.items()):
        if not args.tests and not r.startswith("src/"):
            continue
        for s in m["syms"]:
            if s[S_QUAL] == args.name or s[S_QUAL].endswith("." + args.name):
                print(f"{_short(r)}:{s[S_START]}-{s[S_END]}  {s[S_KIND]} {s[S_QUAL]}"
                      f"  -- {s[S_DOC][:80]}")
                n += 1
    if not n:
        print(f"no definition of {args.name!r}")


def _enclosing(m, line):
    return sorted((s for s in m["syms"] if s[S_START] <= line <= s[S_END]),
                  key=lambda s: s[S_START])


def cmd_at(mods, args):
    for loc in args.loc:
        f, _, ln = loc.rpartition(":")
        m = _find_file(mods, f)
        line = int(ln)
        ban = [b for b in m["banners"] if b[0] <= line]
        chain = _enclosing(m, line)
        head = f"{_short(m['rel'])}:{line}"
        sec = f"  [section: {ban[-1][1]}]" if ban else ""
        if not chain:
            print(f"{head}  <module level>{sec}")
            continue
        print(f"{head}  in {chain[-1][S_QUAL]} ({chain[-1][S_START]}-{chain[-1][S_END]}){sec}"
              f"  -- {chain[-1][S_DOC][:70]}")


def cmd_refs(mods, args):
    kinds = set(args.kind.split(",")) if args.kind else None
    by_file = defaultdict(lambda: defaultdict(list))
    for r, m in mods.items():
        if not args.tests and not r.startswith("src/"):
            continue
        if args.within and args.within not in r:
            continue
        for ref in m["refs"]:
            if ref[R_TOK] != args.name:
                continue
            if kinds and ref[R_KIND] not in kinds:
                continue
            by_file[r][ref[R_SCOPE]].append(ref[R_LINE])
    if not by_file:
        print(f"no code references to {args.name!r}")
        return
    total = sum(len(v) for d in by_file.values() for v in d.values())
    print(f"{args.name}: {total} refs in {len(by_file)} files, "
          f"by enclosing scope (comments/docstrings excluded)")
    files = sorted(by_file, key=lambda k: -sum(len(v) for v in by_file[k].values()))
    for r in files[: args.files]:
        scopes = by_file[r]
        print(f"  {_short(r)}")
        for sc, lines in sorted(scopes.items(), key=lambda kv: min(kv[1]))[: args.limit]:
            ls = ",".join(str(x) for x in sorted(set(lines))[:6])
            print(f"      {sc}  L{ls}")
        if len(scopes) > args.limit:
            print(f"      ... +{len(scopes) - args.limit} more scopes")
    if len(files) > args.files:
        print(f"  ... +{len(files) - args.files} more files: "
              + ", ".join(_short(f) for f in files[args.files: args.files + 8]))


def cmd_deps(mods, args):
    """Importers of a module (eager / lazy / TYPE_CHECKING), grouped by package."""
    target = args.module.replace("/", ".").removesuffix(".py")
    if not target.startswith("apeGmsh"):
        target = "apeGmsh." + target
    rows = defaultdict(list)
    for r, m in mods.items():
        if not args.tests and not r.startswith("src/"):
            continue
        for tgt, ln, lazy, tc in m["imports"]:
            if tgt == target or tgt.startswith(target + "."):
                mode = "tc" if tc else ("lazy" if lazy else "eager")
                rows[r].append((ln, mode, tgt[len(target) + 1:] or "*"))
    if not rows:
        print(f"no importers of {target}")
        return
    pk = Counter()
    for r, lst in rows.items():
        pk[(_short(r).split("/")[0], min(x[1] for x in lst))] += 1
    print(f"{target}: imported by {len(rows)} files  "
          + ", ".join(f"{p}:{md}={c}" for (p, md), c in sorted(pk.items())))
    for r in sorted(rows)[: args.limit]:
        lst = rows[r]
        names = sorted({x[2] for x in lst})
        modes = "/".join(sorted({x[1] for x in lst}))
        print(f"  {modes:<10} {_short(r)}:{lst[0][0]}  {', '.join(names)[:80]}")
    if len(rows) > args.limit:
        print(f"  ... +{len(rows) - args.limit} more")


def cmd_up(mods, args):
    """Caller chain: who calls NAME, who calls them, ... (src only)."""
    calls = defaultdict(set)   # callee short name -> {(file, scope)}
    for r, m in _src(mods).items():
        for ref in m["refs"]:
            if ref[R_KIND] == "call":
                calls[ref[R_TOK]].add((r, ref[R_SCOPE], ref[R_LINE]))
    seen = set()

    def walk(name, depth, ind):
        sites = sorted(calls.get(name, ()))
        if not sites:
            return
        for r, sc, ln in sites[: args.limit]:
            key = (r, sc)
            mark = " (seen)" if key in seen else ""
            print(f"{'  ' * ind}<- {sc}  {_short(r)}:{ln}{mark}")
            if key in seen or sc == "<module>" or depth <= 1:
                continue
            seen.add(key)
            walk(sc.rsplit(".", 1)[-1], depth - 1, ind + 1)
        if len(sites) > args.limit:
            print(f"{'  ' * ind}   ... +{len(sites) - args.limit} more call sites")

    print(args.name)
    walk(args.name, args.depth, 1)


def cmd_h5(mods, args):
    frag = args.fragment.strip("/")
    leaf = frag.split("/")[-1]
    const_names = defaultdict(set)
    for r, m in _src(mods).items():
        for k, v in m["consts"].items():
            if frag in v.strip("/"):
                const_names[k].add(r)
    rows = defaultdict(list)
    for r, m in _src(mods).items():
        for ref in m["refs"]:
            tok, kind = ref[R_TOK], ref[R_KIND]
            if kind == "str":
                s = tok.strip("/")
                if frag in s or s == leaf or s.startswith(leaf + "/"):
                    rows[(r, ref[R_SCOPE])].append((ref[R_LINE], ref[R_RW] or "use"))
            elif kind in ("name", "attr") and tok in const_names:
                rows[(r, ref[R_SCOPE])].append((ref[R_LINE], "via-const"))
    if not rows:
        print(f"no code uses of {frag!r}")
        return
    order = {"write": 0, "read": 1, "probe": 2, "via-const": 3, "use": 4, "const": 5}
    print(f"'{frag}': {len(rows)} code sites (comments/docstrings excluded; "
          f"write=create/assign, read=subscript/get, probe='in' test)")
    agg = {k: Counter(rw for _l, rw in v) for k, v in rows.items()}
    for (r, sc), c in sorted(agg.items(), key=lambda kv: (min(order[k] for k in kv[1]), kv[0])):
        tag = "/".join(k if v == 1 else f"{k}x{v}"
                       for k, v in sorted(c.items(), key=lambda kv: order[kv[0]]))
        first = min(ln for ln, _ in rows[(r, sc)])
        print(f"  {tag:<16} {_short(r)}:{first}  {sc}")
    if const_names:
        print("  path constants: " + ", ".join(
            f"{k} ({', '.join(_short(x) for x in sorted(v))})" for k, v in sorted(const_names.items())))
    # one hop: public accessor methods that read the path, and who calls them
    acc = sorted({sc for (r, sc), c in agg.items()
                  if "." in sc and not sc.rsplit(".", 1)[-1].startswith("_")
                  and (c.get("read") or c.get("use"))})
    if acc:
        callers = defaultdict(set)
        ndefs = Counter(s[S_QUAL].rsplit(".", 1)[-1] for m in _src(mods).values()
                        for s in m["syms"] if s[S_KIND] == "method")
        # name-based resolution: skip accessor names defined on many classes
        names = {a.rsplit(".", 1)[-1] for a in acc if ndefs[a.rsplit(".", 1)[-1]] <= 2}
        for r, m in _src(mods).items():
            for ref in m["refs"]:
                if ref[R_KIND] == "call" and ref[R_TOK] in names:
                    callers[ref[R_TOK]].add(f"{_short(r)}:{ref[R_LINE]} {ref[R_SCOPE]}")
        print("  indirect readers (callers of the accessors above):")
        for nm in sorted(names):
            cs = sorted(callers.get(nm, ()))
            if cs:
                print(f"    .{nm}() <- " + "; ".join(cs[:5]) + (f"; +{len(cs) - 5}" if len(cs) > 5 else ""))


def cmd_impl(mods, args):
    proto, _, meth = args.target.partition(".")
    pdefs = [(r, s) for r, m in _src(mods).items() for s in m["syms"]
             if s[S_KIND] == "class" and s[S_QUAL] == proto]
    if not pdefs:
        sys.exit(f"no class {proto}")
    pr, ps = pdefs[0]
    pm = {n for n in ps[S_METHODS] if not n.startswith("_")}
    print(f"{proto} ({_short(pr)}:{ps[S_START]}) declares {len(pm)} public methods;"
          f" structural implementers (>= {args.min_cover:.0%} of them):")
    for r, m in sorted(_src(mods).items()):
        for s in m["syms"]:
            if s[S_KIND] not in ("class", "nested") or (r == pr and s[S_QUAL] == proto):
                continue
            cov = len(pm & set(s[S_METHODS]))
            if cov / max(1, len(pm)) < args.min_cover and proto not in s[S_BASES]:
                continue
            ms = [x for x in m["syms"] if x[S_QUAL] == f"{s[S_QUAL]}.{meth}"]
            loc = (f"{_short(r)}:{ms[0][S_START]}-{ms[0][S_END]}" if ms
                   else f"{_short(r)}:{s[S_START]}")
            print(f"  {'IMPL' if ms else 'MISSING':7} {s[S_QUAL]:<22} {cov}/{len(pm)}  {loc}")


def _family_members(mods, base):
    children, where = defaultdict(set), {}
    for r, m in _src(mods).items():
        for s in m["syms"]:
            if s[S_KIND] in ("class", "nested"):
                short = s[S_QUAL].rsplit(".", 1)[-1]
                where.setdefault(short, r)
                for b in s[S_BASES]:
                    children[b].add(short)
    out, st = {}, [base]
    while st:
        b = st.pop()
        for c in children.get(b, ()):
            if c not in out:
                out[c] = where.get(c, "?")
                st.append(c)
    return {k: v for k, v in out.items() if not k.startswith("_")}


def cmd_family(mods, args, root):
    if args.names:
        fam = {n: "" for n in args.names.split(",")}
        label = "names"
    else:
        fam = _family_members(mods, args.base)
        label = f"public subclasses of {args.base}"
    members = set(fam)
    n = len(members)
    if n < 2:
        sys.exit(f"family too small: {sorted(members)}")
    print(f"family: {n} {label}")
    if not args.names:
        byfile = Counter(fam.values())
        print("  defined in: " + ", ".join(f"{_short(k)} ({v})" for k, v in byfile.most_common(6)))
    sites = []
    perfile = {}
    for r, m in mods.items():
        named = {ref[R_TOK] for ref in m["refs"] if ref[R_TOK] in members}
        if len(named) * 2 >= n:
            perfile[r] = named
        if not args.tests and not r.startswith("src/"):
            continue
        disp, dline = defaultdict(set), {}
        for line, target, kind, names in m["enums"]:
            hit = set(names) & members
            if kind == "literal" and len(hit) >= 2:
                sites.append(("table", r, line, target, hit))
            elif kind == "isinstance" and hit:
                disp[target] |= hit
                dline.setdefault(target, line)
        for sc, hit in disp.items():
            if len(hit) >= 2:
                sites.append(("dispatch", r, dline[sc], sc, hit))
        for s in m["syms"]:
            if s[S_KIND] in ("class", "nested"):
                hit = set(s[S_METHODS]) & members
                if len(hit) >= 2:
                    sites.append(("methods", r, s[S_START], s[S_QUAL], hit))
        # hand enumeration: one function/class body naming several members
        # (attribute, keyword, string, name or parameter), e.g. a writer that
        # handles contacts, contact_planes and interfaces one after another
        coref, cline = defaultdict(set), {}
        for ref in m["refs"]:
            if ref[R_KIND] != "import" and ref[R_TOK] in members:
                sc = ref[R_SCOPE]
                coref[sc].add(ref[R_TOK])
                cline[sc] = min(cline.get(sc, ref[R_LINE]), ref[R_LINE])
        need = max(2, -(-n * 3 // 10)) if not args.names else 2
        for sc, hit in coref.items():
            if len(hit) >= need and sc != "<module>":
                sites.append(("scope", r, cline[sc], sc, hit))
    best = {}
    for kind, r, ln, tgt, hit in sites:
        key = (kind, r, tgt)
        if key not in best or len(hit) > len(best[key][4]):
            best[key] = (kind, r, ln, tgt, hit)
    # a scope already reported as a table/dispatch/methods site is not repeated
    typed = {(r, t) for (k, r, t) in best if k != "scope"}
    best = {k: v for k, v in best.items() if k[0] != "scope" or (k[1], k[2]) not in typed}
    sites = sorted(best.values(), key=lambda x: (-len(x[4]), x[1], x[2]))
    print(f"\ncode enumerations ({len(sites)}); coverage = members named / {n}; "
          f"'missing' shown when coverage >= {args.gap:.0%}")
    for kind, r, ln, tgt, hit in sites[: args.limit]:
        miss = members - hit
        gap = ""
        if len(hit) / n >= args.gap and miss:
            gap = "  missing: " + ", ".join(sorted(miss)[:5]) + (
                f" +{len(miss) - 5}" if len(miss) > 5 else "")
        print(f"  {len(hit):>3}/{n:<3} {kind:8} {_short(r)}:{ln}  {tgt}{gap}")
    if len(sites) > args.limit:
        print(f"  ... +{len(sites) - args.limit} more (--limit)")
    if perfile:
        print("\nsource/test files naming >= 50% of the family (any code reference):")
        for r, named in sorted(perfile.items(), key=lambda kv: -len(kv[1]))[:10]:
            miss = members - named
            gap = ("  missing: " + ", ".join(sorted(miss)[:4]) +
                   (f" +{len(miss) - 4}" if len(miss) > 4 else "")) if miss else ""
            print(f"  {len(named):>3}/{n:<3} {_short(r)}{gap}")
    if not args.nodocs:
        print("\ndoc/data files naming >= 25% of the family:")
        pats = {k: re.compile(r"\b" + re.escape(k) + r"\b") for k in members}
        rows = []
        for g in TEXT_GLOBS:
            for p in root.glob(g):
                try:
                    t = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                hit = {k for k, pt in pats.items() if pt.search(t)}
                if len(hit) >= max(2, n // 4):
                    rows.append((len(hit), p.relative_to(root).as_posix(), members - hit))
        for cov, rel, miss in sorted(rows, reverse=True)[:10]:
            gap = ""
            if cov / n >= args.gap and miss:
                gap = "  missing: " + ", ".join(sorted(miss)[:4]) + (
                    f" +{len(miss) - 4}" if len(miss) > 4 else "")
            print(f"  {cov:>3}/{n:<3} {rel}{gap}")


def main(argv=None):
    try:  # Windows consoles default to cp1252; source text is UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(prog="nav", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("map"); p.add_argument("file"); p.add_argument("--depth", type=int, default=1)
    p.add_argument("--nested", action="store_true"); p.add_argument("--width", type=int, default=60)
    p = sub.add_parser("pkg"); p.add_argument("package"); p.add_argument("--width", type=int, default=60)
    p.add_argument("--deep", action="store_true")
    p = sub.add_parser("where"); p.add_argument("name"); p.add_argument("--tests", action="store_true")
    p = sub.add_parser("at"); p.add_argument("loc", nargs="+")
    p = sub.add_parser("refs"); p.add_argument("name"); p.add_argument("--kind")
    p.add_argument("--tests", action="store_true"); p.add_argument("--within")
    p.add_argument("--limit", type=int, default=8); p.add_argument("--files", type=int, default=15)
    p = sub.add_parser("h5"); p.add_argument("fragment")
    p = sub.add_parser("up"); p.add_argument("name"); p.add_argument("--depth", type=int, default=3)
    p.add_argument("--limit", type=int, default=6)
    p = sub.add_parser("deps"); p.add_argument("module"); p.add_argument("--tests", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("impl"); p.add_argument("target")
    p.add_argument("--min-cover", type=float, default=0.5)
    p = sub.add_parser("family"); p.add_argument("base", nargs="?")
    p.add_argument("--names"); p.add_argument("--tests", action="store_true")
    p.add_argument("--limit", type=int, default=25); p.add_argument("--gap", type=float, default=0.3)
    p.add_argument("--nodocs", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else repo_root(Path.cwd())
    mods = load_index(root, Path(args.cache) if args.cache else None, args.verbose, args.jobs)
    {"map": cmd_map, "pkg": cmd_pkg, "where": cmd_where, "at": cmd_at, "refs": cmd_refs,
     "h5": cmd_h5, "impl": cmd_impl, "up": cmd_up, "deps": cmd_deps}.get(args.cmd, lambda m, a: cmd_family(m, a, root))(mods, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
