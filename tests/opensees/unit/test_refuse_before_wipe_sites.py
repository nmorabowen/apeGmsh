"""#2449: every in-process path that wipes the live domain refuses a
fork-only chain on stock BEFORE the wipe.

#2440 put ``refuse_fork_chain_before_wipe`` ahead of the wipe for
``apeSees.analyze`` only.  ``ops.run()``, ``ops.eigen()`` and the procedure
drivers (``_modal`` / ``_explicit`` / ``_frf``) still constructed
``LiveOpsEmitter(wipe=True)`` first, so #2441's gate refused at the chain
command after the user's domain was gone.

Two layers:

* behaviour: for each site and each fork-only chain kind, a node the user
  put in a stand-in stock domain still answers ``nodeCoord`` after the
  refusal;
* source scan: every ``LiveOpsEmitter(...)`` construction in ``src/`` that
  can wipe (``wipe=True``, the default, or a variable) is preceded in its
  function by the refusal, or is one of the named exemptions below.  A new
  site fails here until it is guarded and has a behaviour row in
  :data:`SITES`.

#2533 adds the fork-only *verbs*.  ``complex_eigen``,
``modal_response_history``, ``response_spectrum_analysis``, the FRF sweeps,
``critical_time_step`` and ``analyze_explicit`` passed the chain refusal
with a stock chain, wiped, and only then failed at the fork command.  Each
now calls ``refuse_fork_verb_before_wipe`` (the ``has_fork`` verdict
``eigen_feast`` already read) before the wipe, and a staged ``ops.run()``
refuses before it too:

* behaviour: each fork-only verb, with a stock chain, refuses by name on a
  stand-in stock domain and on real stock openseespy, and the user's node
  survives; on a stand-in fork it reaches the wipe (the control);
* source scan: a wiping site that touches a fork-only ``LiveOpsEmitter``
  method (derived from ``emitter/live.py``) calls the verb refusal before
  the wipe, and has its rows in :data:`FORK_VERB_SITES`.
"""
from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.opensees.h5.test_h5_solve_stamp import (
    _TRANSIENT,
    _StockOps,
    _chain_frame,
)

SRC = Path(__file__).resolve().parents[3] / "src"

#: The fork-only chain kinds of the card: two handlers, an integrator and
#: a system.
FORK_KINDS = [
    ("constraints", "LadrunoProjection"),
    ("constraints", "LadrunoContact"),
    ("integrator", "ExplicitBathe"),
    ("system", "Pardiso"),
]

USER_NODE = 9999
USER_XYZ = [1.5, -2.0, 0.25]


class _StockDomain(_StockOps):
    """``_StockOps`` with a node table: ``wipe`` empties it, ``nodeCoord``
    answers from it, as openseespy's domain does."""

    def __init__(self) -> None:
        super().__init__()
        self.nodes: dict[int, list[float]] = {}

    def node(self, tag: int, *xyz: float) -> int:
        self.calls.append(("node", tag, *xyz))
        self.nodes[int(tag)] = [float(v) for v in xyz]
        return 0

    def wipe(self) -> int:
        self.calls.append(("wipe",))
        self.nodes.clear()
        return 0

    def nodeCoord(self, tag: int) -> list[float]:
        if int(tag) not in self.nodes:
            raise RuntimeError(f"nodeCoord: node {tag} not in the domain")
        return list(self.nodes[int(tag)])


def _modal_history(ops: Any) -> Any:
    return ops.modal_response_history(
        dt=0.01, n_steps=1, num_modes=1, base_accel=ops.timeSeries.Linear(),
        direction=1, damp=0.05)


#: One driver per guarded construction site, keyed ``<module>:<qualname>``
#: as :func:`_wiping_sites` names it.
SITES: dict[str, Callable[[Any, bool], Any]] = {
    "apeGmsh/opensees/apesees.py:apeSees.analyze":
        lambda o, tr: o.analyze(steps=1, dt=0.01 if tr else None),
    "apeGmsh/opensees/apesees.py:apeSees.run": lambda o, tr: o.run(),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.eigen":
        lambda o, tr: o.eigen(1),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.modal_properties":
        lambda o, tr: o.modal_properties(1),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.eigen_feast":
        lambda o, tr: o.eigen_feast(0.0, 10.0),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.complex_eigen":
        lambda o, tr: o.complex_eigen(1),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.modal_response_history":
        lambda o, tr: _modal_history(o),
    "apeGmsh/opensees/procedures/_modal.py:"
    "_ModalMixin.response_spectrum_analysis":
        lambda o, tr: o.response_spectrum_analysis(
            1, periods=[0.0, 1.0], accels=[1.0, 1.0], combine="SRSS",
            num_modes=1),
    "apeGmsh/opensees/procedures/_explicit.py:_ExplicitMixin.critical_time_step":
        lambda o, tr: o.critical_time_step(),
    "apeGmsh/opensees/procedures/_explicit.py:_ExplicitMixin.analyze_explicit":
        lambda o, tr: o.analyze_explicit(duration=0.1),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.residual_modes":
        lambda o, tr: o.residual_modes([1], num_modes=1),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.ritz_vectors":
        lambda o, tr: o.ritz_vectors(1, dirs=[1]),
    "apeGmsh/opensees/procedures/_gravity_modal.py:"
    "_GravityModalMixin._run_gravity_modal":
        lambda o, tr: o.modal_after_gravity(1),
    "apeGmsh/opensees/procedures/_frf.py:_FrfMixin._run_modal_sweep":
        lambda o, tr: o.frequency_response(
            f_min=0.0, f_max=10.0, n_freq=2, node=2, dof=1, num_modes=1,
            base_accel_dir=1, damp=0.05),
}

#: #2533: the wiping sites of the fork-only verbs, each with the public
#: verbs (the names their refusal gives, after ``apeSees.``) that reach it.
FORK_VERB_SITES: dict[str, tuple[str, ...]] = {
    "apeGmsh/opensees/apesees.py:apeSees.analyze":
        ("analyze(profile=...)",),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.eigen_feast":
        ("eigen_feast",),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.complex_eigen":
        ("complex_eigen",),
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.modal_response_history":
        ("modal_response_history",),
    "apeGmsh/opensees/procedures/_modal.py:"
    "_ModalMixin.response_spectrum_analysis":
        ("response_spectrum_analysis",),
    "apeGmsh/opensees/procedures/_explicit.py:_ExplicitMixin.critical_time_step":
        ("critical_time_step",),
    "apeGmsh/opensees/procedures/_explicit.py:_ExplicitMixin.analyze_explicit":
        ("analyze_explicit",),
    "apeGmsh/opensees/procedures/_frf.py:_FrfMixin._run_modal_sweep":
        ("frequency_response", "steady_state_dynamics", "random_response"),
}

#: The verb whose refusal a :data:`SITES` row meets first at a fork-verb
#: site, where it comes before the chain's: the row drives the site's first
#: verb.  ``analyze`` refuses only under ``profile=``, which its row omits.
_SITE_VERB = {
    site: verbs[0] for site, verbs in FORK_VERB_SITES.items()
    if site != "apeGmsh/opensees/apesees.py:apeSees.analyze"}


def _sweep(ops: Any, verb: str) -> Any:
    kw: dict[str, Any] = dict(
        f_min=0.1, f_max=10.0, n_freq=2, node=2, dof=1, num_modes=1,
        base_accel_dir=1, damp=0.05)
    if verb == "random_response":
        kw["input_psd"] = ops.timeSeries.Constant()
    return getattr(ops, verb)(**kw)


#: One driver per fork-only public verb, and the stock chain row it runs
#: with: a chain the #2449 refusal passes, so only the verb refusal stands
#: between the call and the wipe.
FORK_VERBS: dict[str, tuple[Callable[[Any], Any], tuple[str, str]]] = {
    "analyze(profile=...)": (lambda o: o.analyze(steps=1, profile="p.txt"),
                             ("integrator", "LoadControl")),
    "eigen_feast": (lambda o: o.eigen_feast(0.0, 10.0),
                    ("integrator", "LoadControl")),
    "complex_eigen": (lambda o: o.complex_eigen(1),
                      ("integrator", "LoadControl")),
    "modal_response_history": (_modal_history,
                               ("integrator", "LoadControl")),
    "response_spectrum_analysis": (
        lambda o: o.response_spectrum_analysis(
            1, periods=[0.0, 1.0], accels=[1.0, 1.0], combine="SRSS",
            num_modes=1),
        ("integrator", "LoadControl")),
    "frequency_response": (lambda o: _sweep(o, "frequency_response"),
                           ("integrator", "LoadControl")),
    "steady_state_dynamics": (lambda o: _sweep(o, "steady_state_dynamics"),
                              ("integrator", "LoadControl")),
    "random_response": (lambda o: _sweep(o, "random_response"),
                        ("integrator", "LoadControl")),
    "critical_time_step": (lambda o: o.critical_time_step(),
                           ("integrator", "CentralDifference")),
    "analyze_explicit": (lambda o: o.analyze_explicit(duration=0.1),
                         ("integrator", "CentralDifference")),
}


def _verb_refusal(verb: str) -> str:
    return (re.escape(f"apeSees.{verb}")
            + " requires the Ladruno fork build of OpenSees")


#: Sites behind ``require_rsa_feature``: on the stand-in stock domain that
#: gate refuses first (the domain survives either way), so their rows pass
#: it to reach the chain refusal this slice adds.
RSA_GATED = frozenset({
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.residual_modes",
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.ritz_vectors",
})

#: Rows a site refuses first by its own rule, before the chain refusal and
#: before the wipe, with that rule's message: the gravity preload needs a
#: Static LoadControl, so a Transient ExplicitBathe frame never gets as far
#: as the chain.  The domain must survive these too.
OWN_RULE_FIRST: dict[tuple[str, str], str] = {
    ("apeGmsh/opensees/procedures/_gravity_modal.py:"
     "_GravityModalMixin._run_gravity_modal", "ExplicitBathe"):
        "the gravity preload needs ops.analysis.Static",
}

#: Wiping sites that do not call ``refuse_fork_chain_before_wipe``, each
#: with the call that must precede the construction instead (``None``: no
#: call is required) and why.
EXEMPT: dict[str, tuple[str | None, str]] = {
    "apeGmsh/opensees/opensees_model.py:OpenSeesModel._build_live": (
        "refuse_live_requires",
        "an archive replay: it refuses the stamp's ``fork`` requirement "
        "(ADR 0114 D6) before the wipe; no declared primitives to read"),
    "apeGmsh/interop/strut_tie.py:strut_tie_overlays": (
        None,
        "a library-internal bridge with a fixed stock chain (Plain / RCM / "
        "BandGeneral / LoadControl): no fork-only type can reach it"),
    "apeGmsh/interop/strut_tie.py:strut_tie_pushover": (
        None,
        "a library-internal bridge with a fixed stock chain (Plain / RCM / "
        "UmfPack / DisplacementControl): no fork-only type can reach it"),
}

REFUSAL = "refuse_fork_chain_before_wipe"


def _site_params() -> list[Any]:
    return [pytest.param(s, f, r, id=f"{s.split(':')[1]}-{f}-{r}")
            for s in sorted(SITES) for f, r in FORK_KINDS]


@pytest.mark.parametrize("site, family, row", _site_params())
def test_a_refused_site_keeps_the_users_node(
    monkeypatch: pytest.MonkeyPatch, site: str, family: str, row: str,
) -> None:
    import apeGmsh.opensees._target as target_mod
    import apeGmsh.opensees.emitter.live as live_mod

    stock = _StockDomain()
    stock.node(USER_NODE, *USER_XYZ)
    stock.calls.clear()
    monkeypatch.setattr(live_mod, "_get_ops", lambda: stock)
    if site in RSA_GATED:
        monkeypatch.setattr(target_mod, "require_rsa_feature", lambda f: None)
    match = OWN_RULE_FIRST.get(
        (site, row), "requires the Ladruno fork|fork-only")
    if site in _SITE_VERB:
        match = _verb_refusal(_SITE_VERB[site])
    ops = _chain_frame(family, row, staged=False)
    transient = family == "integrator" and row in _TRANSIENT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match=match):
            SITES[site](ops, transient)
    assert stock.nodes.get(USER_NODE) == USER_XYZ, stock.calls[:5]
    assert ("wipe",) not in stock.calls, stock.calls[:5]


def _stand_in(monkeypatch: pytest.MonkeyPatch, domain: "_StockDomain") -> None:
    import apeGmsh.opensees.emitter.live as live_mod

    domain.node(USER_NODE, *USER_XYZ)
    domain.calls.clear()
    monkeypatch.setattr(live_mod, "_get_ops", lambda: domain)


@pytest.mark.parametrize("verb", sorted(FORK_VERBS))
def test_a_fork_verb_refuses_by_name_with_a_stock_chain(
    monkeypatch: pytest.MonkeyPatch, verb: str,
) -> None:
    """#2533: the chain is stock, so only the verb's own refusal can keep
    the user's node; before the fix each verb wiped, then failed."""
    stock = _StockDomain()
    _stand_in(monkeypatch, stock)
    drive, (family, row) = FORK_VERBS[verb]
    ops = _chain_frame(family, row, staged=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match=_verb_refusal(verb)):
            drive(ops)
    assert stock.nodes.get(USER_NODE) == USER_XYZ, stock.calls[:5]
    assert ("wipe",) not in stock.calls, stock.calls[:5]


class _ForkDomain(_StockDomain):
    """A stand-in fork: ``ladrunoBuild`` answers a sha."""

    def ladrunoBuild(self) -> str:
        return "f" * 40


@pytest.mark.parametrize("verb", sorted(FORK_VERBS))
def test_a_fork_verb_still_reaches_the_wipe_on_the_fork(
    monkeypatch: pytest.MonkeyPatch, verb: str,
) -> None:
    """The control: on a fork the refusal stays silent and the verb wipes
    and emits.  The stand-in answers 0 to every fork command, so what
    follows the emit may fail; only the wipe is asserted."""
    fork = _ForkDomain()
    _stand_in(monkeypatch, fork)
    drive, (family, row) = FORK_VERBS[verb]
    ops = _chain_frame(family, row, staged=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            drive(ops)
        except Exception as exc:  # the stand-in's 0 is no fork answer
            assert "requires the Ladruno fork" not in str(exc), exc
    assert fork.calls[0] == ("wipe",)
    assert USER_NODE not in fork.nodes


def test_a_staged_run_refuses_before_the_wipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#2533: ``ops.run()`` on a staged model refused at ``stage_open``,
    after the wipe; ``ops.eigen()`` refused the same model up front."""
    stock = _StockDomain()
    _stand_in(monkeypatch, stock)
    ops = _chain_frame("integrator", "LoadControl", staged=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError,
                           match=r"apeSees\.run: live execution does not "
                                 "support staged models"):
            ops.run()
    assert stock.nodes.get(USER_NODE) == USER_XYZ, stock.calls[:5]
    assert ("wipe",) not in stock.calls, stock.calls[:5]


def test_a_stock_chain_still_runs_through_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: no fork-only type, so ``ops.run()`` wipes and emits."""
    import apeGmsh.opensees.emitter.live as live_mod

    stock = _StockDomain()
    stock.node(USER_NODE, *USER_XYZ)
    stock.calls.clear()
    monkeypatch.setattr(live_mod, "_get_ops", lambda: stock)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _chain_frame("integrator", "LoadControl", staged=False).run()
    assert stock.calls[0] == ("wipe",)
    assert USER_NODE not in stock.nodes
    assert any(c[0] == "integrator" for c in stock.calls)


@pytest.mark.live
@pytest.mark.parametrize("site", [
    "apeGmsh/opensees/apesees.py:apeSees.run",
    "apeGmsh/opensees/procedures/_modal.py:_ModalMixin.eigen",
    "apeGmsh/opensees/procedures/_explicit.py:_ExplicitMixin.critical_time_step",
    "apeGmsh/opensees/procedures/_frf.py:_FrfMixin._run_modal_sweep",
    "apeGmsh/opensees/procedures/_gravity_modal.py:"
    "_GravityModalMixin._run_gravity_modal",
], ids=lambda s: s.split(":")[1])
def test_a_refused_site_keeps_the_users_node_on_real_stock(site: str) -> None:
    """The same contract in the real stock domain: a node the user built
    with the bridge's own module still answers ``nodeCoord`` after the
    refusal of ``LadrunoProjection``."""
    from apeGmsh.opensees.emitter.live import get_backend_info, get_ops

    try:
        kind = get_backend_info().kind
    except ImportError:
        pytest.skip("no in-process OpenSees backend")
    if kind == "fork":
        pytest.skip("the fork runs a fork-only chain; nothing is refused")
    live = get_ops()
    live.wipe()
    live.model("basic", "-ndm", 3, "-ndf", 6)
    live.node(USER_NODE, *USER_XYZ)
    try:
        ops = _chain_frame("constraints", "LadrunoProjection", staged=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError,
                               match="requires the Ladruno fork|fork-only"):
                SITES[site](ops, False)
        assert list(live.nodeCoord(USER_NODE)) == USER_XYZ
    finally:
        live.wipe()


@pytest.mark.live
@pytest.mark.parametrize("verb", sorted(FORK_VERBS))
def test_a_fork_verb_keeps_the_users_node_on_real_stock(verb: str) -> None:
    """#2533 in the real stock domain: with a stock chain, each fork-only
    verb refuses by name and the user's node still answers ``nodeCoord``."""
    from apeGmsh.opensees.emitter.live import get_backend_info, get_ops

    try:
        kind = get_backend_info().kind
    except ImportError:
        pytest.skip("no in-process OpenSees backend")
    if kind == "fork":
        pytest.skip("the fork runs a fork-only verb; nothing is refused")
    live = get_ops()
    live.wipe()
    live.model("basic", "-ndm", 3, "-ndf", 6)
    live.node(USER_NODE, *USER_XYZ)
    try:
        drive, (family, row) = FORK_VERBS[verb]
        ops = _chain_frame(family, row, staged=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match=_verb_refusal(verb)):
                drive(ops)
        assert list(live.nodeCoord(USER_NODE)) == USER_XYZ
    finally:
        live.wipe()


# -- source scan ------------------------------------------------------------


def _can_wipe(call: ast.Call) -> bool:
    """``wipe=`` absent (the default is ``True``) or anything but a literal
    ``False``."""
    for kw in call.keywords:
        if kw.arg == "wipe":
            return not (isinstance(kw.value, ast.Constant)
                        and kw.value.value is False)
        if kw.arg is None:  # ``**kwargs`` may carry wipe=True
            return True
    return True  # ``wipe`` is keyword-only, so it was not passed


def _is_emitter_ctor(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return ((isinstance(f, ast.Name) and f.id == "LiveOpsEmitter")
            or (isinstance(f, ast.Attribute) and f.attr == "LiveOpsEmitter"))


def _called_names(fn: ast.AST, before: int) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and node.lineno < before:
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _wiping_functions() -> dict[str, tuple[ast.AST, int]]:
    """``<module>:<qualname>`` of each function that constructs a wiping
    ``LiveOpsEmitter``, mapped to its node and the line of the first such
    construction."""
    sites: dict[str, tuple[ast.AST, int]] = {}
    for path in sorted((SRC / "apeGmsh").rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)

        def visit(node: ast.AST, scope: list[str]) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                    inner = scope + [child.name]
                    if not isinstance(child, ast.ClassDef):
                        ctors = [n for n in ast.walk(child)
                                 if _is_emitter_ctor(n) and _can_wipe(n)]
                        if ctors:
                            first = min(c.lineno for c in ctors)
                            key = f"{rel}:{'.'.join(inner)}"
                            sites[key] = (child, first)
                            continue
                    visit(child, inner)
                elif _is_emitter_ctor(child) and _can_wipe(child):
                    raise AssertionError(
                        f"{rel}:{child.lineno}: a wiping LiveOpsEmitter "
                        "outside any function; this scan cannot place it")
                else:
                    visit(child, scope)

        visit(tree, [])
    return sites


def _wiping_sites() -> dict[str, set[str]]:
    """Each wiping site mapped to the names it calls before its first
    wiping construction."""
    return {key: _called_names(fn, first)
            for key, (fn, first) in _wiping_functions().items()}


VERB_REFUSAL = "refuse_fork_verb_before_wipe"

#: Fork-only ``LiveOpsEmitter`` methods that no ``*_FORK_REQUIRED`` gate in
#: ``emitter/live.py`` marks, and why.
UNGATED_FORK_METHODS = {
    "eigen_feast": "stock has ``ops.eigen``; ``-feast`` is a parse error",
    "critical_time_step": "a bare ``ops.criticalTimeStep()``: "
                          "AttributeError on stock",
}


def _fork_emitter_methods() -> set[str]:
    """The ``LiveOpsEmitter`` methods that need the fork: each that names a
    ``*_FORK_REQUIRED`` message, each that calls one of those on ``self``,
    and :data:`UNGATED_FORK_METHODS`."""
    path = SRC / "apeGmsh/opensees/emitter/live.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "LiveOpsEmitter")
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    assert set(UNGATED_FORK_METHODS) <= set(methods)
    fork = {name for name, f in methods.items() if any(
        isinstance(n, ast.Name) and n.id.endswith("_FORK_REQUIRED")
        for n in ast.walk(f))} | set(UNGATED_FORK_METHODS)
    while True:
        more = {name for name, f in methods.items() if name not in fork
                and any(isinstance(n, ast.Attribute) and n.attr in fork
                        and isinstance(n.value, ast.Name)
                        and n.value.id == "self" for n in ast.walk(f))}
        if not more:
            return fork
        fork |= more


def _touches_fork_method(fn: ast.AST, fork: set[str]) -> bool:
    """``fn`` reads a fork-only emitter method (``x.complex_eigen``, not
    the bridge's own ``self.<verb>``), or dispatches by a computed name
    (``getattr(emitter, command)``), which may reach one."""
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute) and node.attr in fork
                and not (isinstance(node.value, ast.Name)
                         and node.value.id == "self")):
            return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and not isinstance(node.args[1], ast.Constant)):
            return True
    return False


def _fork_verb_sites() -> dict[str, set[str]]:
    fork = _fork_emitter_methods()
    return {key: _called_names(fn, first)
            for key, (fn, first) in _wiping_functions().items()
            if _touches_fork_method(fn, fork)}


def test_every_fork_verb_refuses_before_the_wipe() -> None:
    """#2533: a wiping site that runs a fork-only emitter method calls the
    verb refusal before the wipe, and has its rows here."""
    sites = _fork_verb_sites()
    unguarded = sorted(k for k, called in sites.items()
                       if VERB_REFUSAL not in called)
    assert not unguarded, (
        "a fork-only verb wipes with no "
        f"{VERB_REFUSAL}(...) before it: {unguarded}")
    assert set(sites) == set(FORK_VERB_SITES)
    assert {v for vs in FORK_VERB_SITES.values() for v in vs} == set(
        FORK_VERBS)


def test_the_fork_method_scan_sees_each_form() -> None:
    """Self-test: the derived set holds the gated, the transitively gated
    and the ungated fork methods, and no stock one; the site test sees an
    attribute and a computed ``getattr``, not ``self.<verb>``."""
    fork = _fork_emitter_methods()
    assert {"complex_eigen", "modal_response_history",
            "response_spectrum_analysis", "frequency_response",
            "steady_state_dynamics", "random_response", "eigen_feast",
            "critical_time_step", "ladruno_contact_force"} <= fork
    assert not {"eigen", "modal_properties", "analyze", "node"} & fork

    def fn(src: str) -> ast.AST:
        return ast.parse(src)

    assert _touches_fork_method(fn("le.complex_eigen(1)"), fork)
    assert _touches_fork_method(fn("r = getattr(le, command)"), fork)
    assert not _touches_fork_method(fn("getattr(le, 'eigen')"), fork)
    assert not _touches_fork_method(fn("self.complex_eigen(1)"), fork)
    assert not _touches_fork_method(fn("le.eigen(1)"), fork)


def test_every_wiping_site_refuses_before_the_wipe() -> None:
    sites = _wiping_sites()
    unguarded = []
    for key, called in sorted(sites.items()):
        if key in EXEMPT:
            need, _why = EXEMPT[key]
            if need is not None and need not in called:
                unguarded.append(f"{key} (exempt, but no {need} first)")
        elif REFUSAL not in called:
            unguarded.append(key)
    assert not unguarded, (
        "LiveOpsEmitter(wipe=True) constructed with no "
        f"{REFUSAL}(bm) before it: {unguarded}")


def test_the_site_tables_match_the_source() -> None:
    """Every guarded site has a behaviour row, and no row or exemption
    names a site that is gone."""
    sites = set(_wiping_sites())
    assert sites - set(EXEMPT) == set(SITES)
    assert set(EXEMPT) <= sites, set(EXEMPT) - sites
    assert {s for s, _r in OWN_RULE_FIRST} <= set(SITES)


def test_the_scan_sees_each_wipe_form() -> None:
    """Self-test: the default, ``True`` and a variable can wipe; a literal
    ``False`` cannot."""
    def ctor(src: str) -> ast.Call:
        node = ast.parse(src, mode="eval").body
        assert isinstance(node, ast.Call)
        return node

    assert _can_wipe(ctor("LiveOpsEmitter()"))
    assert _can_wipe(ctor("LiveOpsEmitter(wipe=True)"))
    assert _can_wipe(ctor("LiveOpsEmitter(wipe=wipe)"))
    assert _can_wipe(ctor("live.LiveOpsEmitter(**kw)"))
    assert not _can_wipe(ctor("LiveOpsEmitter(wipe=False)"))
    assert _is_emitter_ctor(ctor("live.LiveOpsEmitter()"))
