"""Every ``ops.*`` wrapper must mirror the defaults of what it delegates to.

The typed namespaces under ``_internal/ns/`` re-state each primitive's
defaults in their own signature and then pass **all** of them through
explicitly.  So a default changed on the primitive but not on the wrapper
is DEAD on the public surface: the wrapper keeps handing over the old
value, and the primitive's default is never reached.  Nothing compared the
two until this file.

That is not hypothetical.  On 2026-09-05 ``LadrunoSANISAND.tan_type`` moved
``0 -> 2`` in ``material/nd.py`` and had **zero** effect through
``ops.nDMaterial.LadrunoSANISAND(...)``; it was caught only because the
committed ``studio/_api_index.json`` (which harvests the WRAPPER signature)
refused to change after a rebuild.

**The comparison is against what the wrapper actually calls**, not against
``dataclasses.fields()``.  Several wrappers delegate to an alternate
constructor or a builder — ``ops.uniaxialMaterial.ConfinedConcrete1D`` goes
through ``ASDConcrete1D.from_mander``, whose ``auto_regularize`` default is
deliberately ``False`` where the field is ``True`` (the Mander envelope must
not be crack-band-rescaled).  Comparing to the field would report that as a
bug; comparing to ``from_mander`` reports it correctly as a match.

Three shapes are exempt, each for a stated reason rather than a broad skip:

* **Adapters.** A parameter whose wrapper annotation differs from the
  target's is converted, not mirrored — ``material: NDMaterial | str``
  resolving a name, ``G: float | None`` resolved from the mesh,
  ``dict | None`` normalised into the primitive's tuple form.  There is no
  shared default to compare, so it is skipped by that rule alone.
* **Non-constructors.** Namespace verbs that build nothing
  (``ops.profiler.start()``).  Enumerated exactly in
  :data:`_NOT_CONSTRUCTORS`; a new one must be added there deliberately.
* **Asymmetric requiredness.** A wrapper may make a parameter required that
  the target defaults (``ZeroLength.mat_dirs``, whose primitive defaults
  ``()`` and then rejects empty in ``__post_init__``), or supply a default
  the target requires.  Neither can produce the silent-wrong-value bug this
  file exists to catch — the wrapper's own value is what runs, and it is
  visible in the signature — so they are counted, not asserted on.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil
import textwrap

import pytest

import apeGmsh.opensees._internal.ns as nspkg

_EMPTY = inspect.Parameter.empty

#: Namespace methods that construct no primitive — verbs, or a re-export
#: that delegates to its base.  Exact: the test fails if this drifts, so a
#: constructor that stops resolving cannot silently vanish from coverage.
_NOT_CONSTRUCTORS: frozenset[str] = frozenset({
    "damping._StageDampingNS.modal",      # overrides, delegates to super()
    "profiler._ProfilerNS.start",
    "profiler._ProfilerNS.stop",
    "profiler._ProfilerNS.reset",
    "profiler._ProfilerNS.report",
    "profiler._ProfilerNS.memory",
    "recorder._RecorderNS.declare",       # registers a prebuilt declaration
    "strategy._StrategyNS.profile",       # returns a named preset
})

#: Parameters whose wrapper default INTENTIONALLY differs from its target's,
#: keyed ``module._ClassNS.method.param`` -> the reason.  Empty today: every
#: one of the compared pairs agrees.  A genuine divergence belongs here WITH
#: its reason, next to a comment at the wrapper — never fixed by loosening
#: the comparison.
_ALLOWED_DIVERGENCES: dict[str, str] = {}

#: Floor on the number of (wrapper, target) parameter pairs actually
#: compared.  Without it a regression in the resolver below would make this
#: file vacuously green: zero pairs compared, zero mismatches, all passing.
#: 527 pairs compare today across 190 methods.
_MIN_PAIRS_COMPARED = 500


def _norm(annotation: object) -> str | None:
    """Annotation text, quote- and whitespace-insensitive.

    ``from __future__ import annotations`` makes every annotation a string,
    and a source-level quoted ``"tuple[Bar, ...]"`` keeps its quotes while
    the target's does not — stripping them is what keeps such a pair in the
    comparison instead of misreading it as an adapter.
    """
    if annotation is _EMPTY:
        return None
    return " ".join(str(annotation).split()).replace('"', "").replace("'", "")


def _resolve(node: ast.expr, mod: object, cls: type) -> object | None:
    if isinstance(node, ast.Name):
        obj = vars(mod).get(node.id)
        return obj if callable(obj) else None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "self":
            return getattr(cls, node.attr, None)
        sub = getattr(vars(mod).get(node.value.id), node.attr, None)
        return sub if callable(sub) else None
    return None


def _delegate(meth: object, mod: object, cls: type) -> object | None:
    """The callable this wrapper hands its keywords to.

    Three shapes, tried in order: the argument of
    ``self._bridge._register(<CALL>, ...)``; a bare construction anywhere in
    the body (``prim = Uniform(...)``, ``return _Ladder(...)``,
    ``_build_mohr_coulomb_soil(...)``); or a shared builder on the namespace
    itself (``self._build_rc(...)``).
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(meth)))
    except (OSError, SyntaxError):     # pragma: no cover - source is on disk
        return None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_register"
                and node.args and isinstance(node.args[0], ast.Call)):
            found = _resolve(node.args[0].func, mod, cls)
            if found is not None:
                return found
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        found = _resolve(node.func, mod, cls)
        if found is None:
            continue
        if dataclasses.is_dataclass(found):
            return found
        if getattr(found, "__name__", "").startswith(("_build", "_Ladder")):
            return found
    return None


def _wrappers():
    """Every public method on every ``_*NS`` namespace class."""
    for info in pkgutil.iter_modules(nspkg.__path__):
        mod = importlib.import_module(f"{nspkg.__name__}.{info.name}")
        for cname, cls in vars(mod).items():
            if not (inspect.isclass(cls)
                    and cname.startswith("_") and cname.endswith("NS")
                    and cls.__module__ == mod.__name__):
                continue
            for mname, meth in vars(cls).items():
                if mname.startswith("_") or not inspect.isfunction(meth):
                    continue
                yield f"{info.name}.{cname}.{mname}", mod, cls, meth


def _audit() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"mismatch": [], "unresolved": [],
                                 "compared": []}
    for where, mod, cls, meth in _wrappers():
        target = _delegate(meth, mod, cls)
        if target is None:
            out["unresolved"].append(where)
            continue
        tsig = inspect.signature(target)
        for pname, wp in inspect.signature(meth).parameters.items():
            tp = tsig.parameters.get(pname)
            if pname == "self" or tp is None:
                continue
            if _norm(wp.annotation) != _norm(tp.annotation):
                continue                                   # adapter
            if repr(tp.default).startswith("<dataclasses._HAS_DEFAULT"):
                continue                                   # default_factory
            if wp.default is _EMPTY or tp.default is _EMPTY:
                continue                                   # asymmetric
            key = f"{where}.{pname}"
            out["compared"].append(key)
            if (wp.default != tp.default
                    or type(wp.default) is not type(tp.default)):
                if key in _ALLOWED_DIVERGENCES:
                    continue
                out["mismatch"].append(
                    f"{key}: wrapper={wp.default!r} "
                    f"{type(wp.default).__name__} != target={tp.default!r} "
                    f"{type(tp.default).__name__}"
                )
    return out


@pytest.fixture(scope="module")
def audit() -> dict[str, list[str]]:
    return _audit()


def test_every_wrapper_default_matches_its_target(audit) -> None:
    assert not audit["mismatch"], (
        "these ops.* wrappers re-state a default that differs from the "
        "constructor they delegate to, so the constructor's value never "
        "reaches a user:\n  " + "\n  ".join(audit["mismatch"])
    )


def test_only_known_verbs_construct_nothing(audit) -> None:
    """Coverage guard: an unresolved constructor is an untested one."""
    assert set(audit["unresolved"]) == set(_NOT_CONSTRUCTORS), (
        "the set of namespace methods with no resolvable constructor "
        "changed.\n  newly unresolved (now untested): "
        f"{sorted(set(audit['unresolved']) - _NOT_CONSTRUCTORS)}\n"
        "  no longer unresolved (drop from _NOT_CONSTRUCTORS): "
        f"{sorted(_NOT_CONSTRUCTORS - set(audit['unresolved']))}"
    )


def test_the_audit_actually_compared_something(audit) -> None:
    """A resolver regression must fail, not pass vacuously."""
    assert len(audit["compared"]) >= _MIN_PAIRS_COMPARED, (
        f"only {len(audit['compared'])} wrapper/target parameter pairs were "
        f"compared, below the {_MIN_PAIRS_COMPARED} floor — the resolver or "
        "the adapter rule has regressed and this file is no longer testing "
        "what it claims."
    )


def test_the_2026_09_05_regression_is_in_scope(audit) -> None:
    """The bug that motivated this file, pinned as a live example.

    ``tan_type`` is mirrored (same annotation on both sides), so it lands in
    the compared set rather than being skipped as an adapter.
    """
    from apeGmsh.opensees._internal.ns.nd import _NDMaterialNS
    from apeGmsh.opensees.material.nd import LadrunoSANISAND

    assert "nd._NDMaterialNS.LadrunoSANISAND.tan_type" in set(
        audit["compared"]
    )
    wrapper = inspect.signature(_NDMaterialNS.LadrunoSANISAND)
    field = LadrunoSANISAND.__dataclass_fields__["tan_type"]
    assert wrapper.parameters["tan_type"].default == field.default == 2
