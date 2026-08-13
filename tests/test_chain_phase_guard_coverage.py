"""Structural invariant: every kernel-backed mutation stays guarded.

The behavioural guards are tested in ``test_from_h5_kernel_guard.py``
(kernel reads) and ``test_phase_3b_2d.py`` (the freeze). Those check
the surfaces that exist *today*; this file checks the ones added
*tomorrow*, by sweeping the composites for methods that reach gmsh
without reaching a guard.

Why it exists: the guard coverage was built up over several passes,
and each pass found surfaces the previous one missed — a mutation
sitting on a "queries" composite, one sibling of twelve that never
called ``_guard``, four exporters that wrote an empty model to disk
rather than failing. Every one of those was invisible until something
swept for it. Without this test the next one is invisible too.

Method: AST with transitive closure, so a guard reached through a
helper (``self._guard``, ``_resolve_dt``, ``Model._register``) counts.
An earlier runtime probe was abandoned because it had to guess
argument counts and produced six false positives.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import apeGmsh

SRC = Path(apeGmsh.__file__).parent

# label -> (module relative to the package, class name or None for all)
COMPOSITES: dict[str, tuple[str, str | None]] = {
    "g.model.<geometry>":  ("core/_model_geometry.py", None),
    "g.model.boolean":     ("core/_model_boolean.py", None),
    "g.model.transforms":  ("core/_model_transforms.py", None),
    "g.model.io":          ("core/_model_io.py", None),
    "g.model.queries":     ("core/_model_queries.py", None),
    "g.mesh.generation":   ("mesh/_mesh_generation.py", None),
    "g.mesh.editing":      ("mesh/_mesh_editing.py", None),
    "g.mesh.sizing":       ("mesh/_mesh_sizing.py", None),
    "g.mesh.structured":   ("mesh/_mesh_structured.py", None),
    "g.mesh.recipe":       ("mesh/_mesh_recipe.py", None),
    "g.mesh.partitioning": ("mesh/_mesh_partitioning.py", None),
    "g.physical":          ("mesh/PhysicalGroups.py", "PhysicalGroups"),
    "g.labels":            ("core/Labels.py", "Labels"),
    "g.parts":             ("core/_parts_registry.py", "PartsRegistry"),
    "g.sections":          ("sections/_builder.py", None),
    "g.rebar":             ("core/RebarComposite.py", "RebarComposite"),
}

FREEZE_GUARDS = {"chain_phase_guard", "_check_chain_phase",
                 "_require_unfrozen", "_guard"}
KERNEL_GUARDS = {"raise_if_no_live_kernel", "_require_kernel",
                 "raise_if_from_h5_session"}
# Guarded helpers defined on another object, matched by attribute name.
EXTERNAL_FREEZE = {"_register", "_register_instance", "_resolve_dt"}

# gmsh APIs that change the model. `synchronize` is deliberately absent:
# it pushes OCC into the model but changes no content, so a read that
# syncs first is a read.
MUTATING_PREFIXES = (
    "add", "set", "remove", "create", "classify", "generate", "refine",
    "optimize", "recombine", "partition", "renumber", "clear", "reverse",
    "relocate", "split", "heal", "import", "translate", "rotate",
    "mirror", "dilate", "copy", "fragment", "cut", "fuse", "intersect",
    "extrude", "revolve", "field",
)
WRITE_APIS = {"write"}

# Reads that legitimately reach gmsh with no kernel guard. Each is a
# recorded decision, not an oversight: they fail with a raw gmsh error
# on a from_h5 session but cannot corrupt state or emit a bad artifact.
# Adding to this list should be a conscious review step — that is the
# point of the test.
KNOWN_UNGUARDED_READS = {
    ("g.mesh.recipe", "check"),
    ("g.parts", "build_face_map"),
    ("g.rebar", "resolve"),
    ("g.sections", "plot_faces"),
}


def _attr_root(node: ast.Attribute) -> str | None:
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return cur.id if isinstance(cur, ast.Name) else None


def _called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _self_call(node: ast.Call) -> str | None:
    """``self.foo(...)`` / ``self._mesh.foo(...)`` -> ``'foo'``."""
    f = node.func
    if not isinstance(f, ast.Attribute):
        return None
    v = f.value
    if isinstance(v, ast.Name) and v.id == "self":
        return f.attr
    if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) \
            and v.value.id == "self":
        return f.attr
    return None


def _gmsh_apis(fn: ast.FunctionDef) -> set[str]:
    """Leaf names of every ``gmsh.*.foo(...)`` call in ``fn``."""
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if _attr_root(n.func) == "gmsh":
                chain: list[str] = []
                cur: ast.expr = n.func
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                out.add(chain[0])
    return out


def _scan(path: Path, only_class: str | None) -> dict:
    """Per-class method facts, with guard/gmsh reachability closed over
    intra-class calls."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if only_class and cls.name != only_class:
            continue
        methods = {
            n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)
        }
        if not methods:
            continue
        freeze, kernel, apis, sibs = {}, {}, {}, {}
        for name, fn in methods.items():
            f = k = False
            calls: set[str] = set()
            for n in ast.walk(fn):
                if isinstance(n, ast.Call):
                    cn = _called_name(n)
                    if cn in FREEZE_GUARDS or cn in EXTERNAL_FREEZE:
                        f = True
                    if cn in KERNEL_GUARDS:
                        k = True
                    s = _self_call(n)
                    if s and s in methods:
                        calls.add(s)
            freeze[name], kernel[name] = f, k
            apis[name], sibs[name] = _gmsh_apis(fn), calls

        for _ in range(len(methods) + 2):        # fixed point
            changed = False
            for name in methods:
                for s in sibs[name]:
                    if freeze[s] and not freeze[name]:
                        freeze[name] = True; changed = True
                    if kernel[s] and not kernel[name]:
                        kernel[name] = True; changed = True
                    if not apis[s] <= apis[name]:
                        apis[name] |= apis[s]; changed = True
            if not changed:
                break
        result[cls.name] = (methods, freeze, kernel, apis)
    return result


def _classify():
    """Yield ``(verdict, label, method, lineno, rel_path)`` for every
    public gmsh-touching method that reaches no freeze guard."""
    for label, (rel, cls_filter) in COMPOSITES.items():
        path = SRC / rel
        assert path.exists(), f"composite module moved or renamed: {rel}"
        for _cls, (methods, freeze, kernel, apis) in _scan(
            path, cls_filter,
        ).items():
            for name in sorted(methods):
                if name.startswith("_") or not apis[name] or freeze[name]:
                    continue
                mutating = {
                    a for a in apis[name]
                    if a.lower().startswith(MUTATING_PREFIXES)
                }
                if apis[name] & WRITE_APIS:
                    verdict = "WRITES_OUT"
                elif mutating:
                    verdict = "MUTATES"
                else:
                    verdict = "READS"
                yield (verdict, label, name, kernel[name],
                       methods[name].lineno, rel, sorted(mutating)[:3])


def test_no_unguarded_mutations() -> None:
    """Every gmsh-mutating method reaches the chain-phase freeze guard.

    A mutation desyncs the FEMData broker from gmsh on ANY chain-phase
    session, so the freeze guard is required — a kernel guard is not
    enough, because it fires only for ``from_h5`` sessions and a live
    post-extraction session would sail straight through.
    """
    bad = [r for r in _classify() if r[0] == "MUTATES"]
    if bad:
        lines = [
            f"  {label}.{name}  ({rel}:{line})\n"
            f"      mutates gmsh via: {', '.join(apis)}"
            for _v, label, name, _k, line, rel, apis in bad
        ]
        pytest.fail(
            f"{len(bad)} gmsh-mutating method(s) reach no freeze guard:\n"
            + "\n".join(lines)
            + "\n\nAdd the freeze guard at the method's entry point:\n"
              "    from apeGmsh.core._compose_errors import chain_phase_guard\n"
              "    chain_phase_guard(<session>, 'g.x.y()')\n"
              "on composites with a _guard()/_require_unfrozen() helper, "
              "call that instead."
        )


def test_no_unguarded_exporters() -> None:
    """Every method that writes gmsh out reaches a kernel guard.

    Unguarded, these do not fail on a kernel-less session — they write
    whatever empty model gmsh holds, producing a plausible-looking file
    with nothing in it, which is worse than an error because the output
    looks trustworthy.
    """
    bad = [r for r in _classify() if r[0] == "WRITES_OUT" and not r[3]]
    if bad:
        lines = [
            f"  {label}.{name}  ({rel}:{line})"
            for _v, label, name, _k, line, rel, _a in bad
        ]
        pytest.fail(
            f"{len(bad)} exporter(s) reach no kernel guard:\n"
            + "\n".join(lines)
            + "\n\nGuard before the write so no file is emitted:\n"
              "    raise_if_no_live_kernel(<session>, 'g.x.save_y()')"
        )


def test_no_new_unguarded_reads() -> None:
    """Kernel-touching reads carry a kernel guard, modulo a recorded
    allowlist.

    Lower severity than the two above — an unguarded read fails with a
    raw gmsh error or answers from an unrelated model, but cannot
    corrupt state. The allowlist keeps the four known cases visible and
    forces a decision when a new one appears.
    """
    found = {
        (label, name)
        for v, label, name, kernel, _l, _r, _a in _classify()
        if v == "READS" and not kernel
    }
    new = found - KNOWN_UNGUARDED_READS
    stale = KNOWN_UNGUARDED_READS - found
    assert not new, (
        f"new unguarded kernel read(s): {sorted(new)}\n"
        "Either guard with raise_if_no_live_kernel(), or add to "
        "KNOWN_UNGUARDED_READS with a note on why it is acceptable."
    )
    assert not stale, (
        f"KNOWN_UNGUARDED_READS is stale — now guarded: {sorted(stale)}\n"
        "Remove them from the allowlist."
    )


def test_composite_modules_all_resolve() -> None:
    """The COMPOSITES map points at real modules.

    Without this a renamed module would silently drop a composite from
    the sweep, and the tests above would pass by not looking.
    """
    missing = [rel for rel, _c in COMPOSITES.values()
               if not (SRC / rel).exists()]
    assert not missing, f"COMPOSITES entries no longer exist: {missing}"


def test_sweep_actually_covers_methods() -> None:
    """Guard against a silently empty sweep.

    If the AST walk broke (a refactor to nested classes, say) every
    test above would pass vacuously. Pin a floor on what it sees.
    """
    seen = list(_classify())
    total = 0
    for label, (rel, cls_filter) in COMPOSITES.items():
        for _c, (methods, _f, _k, _a) in _scan(SRC / rel, cls_filter).items():
            total += sum(1 for m in methods if not m.startswith("_"))
    assert total > 150, f"sweep saw only {total} public methods — walk broken?"
    assert len(seen) >= len(KNOWN_UNGUARDED_READS)
