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
"""
from __future__ import annotations

import ast
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
    "apeGmsh/opensees/procedures/_frf.py:_FrfMixin._run_modal_sweep":
        lambda o, tr: o.frequency_response(
            f_min=0.0, f_max=10.0, n_freq=2, node=2, dof=1, num_modes=1,
            base_accel_dir=1, damp=0.05),
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


@pytest.mark.parametrize("family, row", FORK_KINDS,
                         ids=[f"{f}-{r}" for f, r in FORK_KINDS])
@pytest.mark.parametrize("site", sorted(SITES), ids=lambda s: s.split(":")[1])
def test_a_refused_site_keeps_the_users_node(
    monkeypatch: pytest.MonkeyPatch, site: str, family: str, row: str,
) -> None:
    import apeGmsh.opensees.emitter.live as live_mod

    stock = _StockDomain()
    stock.node(USER_NODE, *USER_XYZ)
    stock.calls.clear()
    monkeypatch.setattr(live_mod, "_get_ops", lambda: stock)
    ops = _chain_frame(family, row, staged=False)
    transient = family == "integrator" and row in _TRANSIENT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError,
                           match="requires the Ladruno fork|fork-only"):
            SITES[site](ops, transient)
    assert stock.nodeCoord(USER_NODE) == USER_XYZ
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


def _wiping_sites() -> dict[str, set[str]]:
    """``<module>:<qualname>`` of each function that constructs a wiping
    ``LiveOpsEmitter``, mapped to the names it calls before the first such
    construction."""
    sites: dict[str, set[str]] = {}
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
                            sites[key] = _called_names(child, first)
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
