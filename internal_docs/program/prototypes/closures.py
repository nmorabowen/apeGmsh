"""List nested functions (closures) inside a given method, with spans and
nonlocal usage, to design a split of a giant closure-wired method."""
import ast
import sys
from pathlib import Path

REPO = Path(r"C:\Users\nmora\Github\apeGmsh\.claude\worktrees\apegmsh-strains-assessment-c5f7dd")


def main(rel, cls, meth):
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == cls:
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == meth:
                    print(f"{cls}.{meth}: {m.lineno}-{m.end_lineno} ({m.end_lineno-m.lineno+1})")
                    # direct statements of the method body: classify
                    nested = []
                    assigns = 0
                    for st in m.body:
                        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            nonloc = [x for s in ast.walk(st) if isinstance(s, ast.Nonlocal) for x in s.names]
                            nested.append((st.lineno, st.end_lineno, st.name, nonloc))
                        elif isinstance(st, ast.ClassDef):
                            nested.append((st.lineno, st.end_lineno, "[class] " + st.name, []))
                    # also nested defs deeper (inside ifs/with/try)
                    deep = [s for s in ast.walk(m) if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s is not m]
                    print(f"  top-level nested defs: {len(nested)}; all nested defs (any depth): {len(deep)}")
                    tot = 0
                    for a, b, name, nl in nested:
                        tot += b - a + 1
                        print(f"   {a:5}-{b:5} {b-a+1:4} {name} {'nonlocal='+','.join(nl) if nl else ''}")
                    print(f"  lines inside top-level nested defs: {tot}; remaining straight-line body: {m.end_lineno-m.lineno+1-tot}")
                    # names of locals assigned at top level of the method (the closure "state")
                    loc = set()
                    for st in m.body:
                        for t in ast.walk(st):
                            if isinstance(t, ast.Assign):
                                for tg in t.targets:
                                    if isinstance(tg, ast.Name):
                                        loc.add(tg.id)
                        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                    print(f"  assigned local names (approx): {len(loc)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
