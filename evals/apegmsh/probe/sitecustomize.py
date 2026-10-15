"""Grading probe: logs what apeGmsh and OpenSees did in every Python process of a grade.

The skill-eval graders put this directory on ``PYTHONPATH`` and name a log file in
``APEGMSH_EVAL_PROBE_LOG``; every Python process the graded script starts, a deck
run's child included, then imports this module at startup. It wraps:

* the OpenSees module (``openseespy.opensees``, or the fork's ``opensees``) when it
  is imported. Each ``model`` call opens a new model record (a bridge that re-emits
  wipes and rebuilds, so totals would double count); each record holds its counts of
  ``node`` / ``element`` / ``eigen`` calls and every ``analyze`` return code. Each
  ``recorder`` call's arguments are kept per process;
* ``apeGmsh.opensees.apesees.apeSees.__init__``: how many bridges were built, and
  the node count of the FEMData each was built from.

At exit the process appends one JSON line to the log. Without the variable this
module does nothing. A wrapping failure is logged as ``probe_errors``, which the
graders treat as a failed grade, never a silent skip.
"""
from __future__ import annotations

import atexit
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys

_LOG = os.environ.get("APEGMSH_EVAL_PROBE_LOG")
_OPS_MODULES = {"opensees", "openseespy.opensees"}
_BRIDGE_MODULE = "apeGmsh.opensees.apesees"
_MAX_ARGS = 16          # recorder arguments kept per call (node lists can be long)

_state: dict = {
    "pid": os.getpid(),
    "argv": list(sys.argv),
    "cwd": os.getcwd(),
    "ops_modules": [],
    "models": [],
    "recorders": [],
    "apesees": [],
    "probe_errors": [],
}


def _current_model() -> dict:
    if not _state["models"]:
        # OpenSees calls before any model(): kept, so the grader sees them, never matched
        _state["models"].append({"opened_by_model": False, "node": 0, "element": 0,
                                 "eigen": 0, "analyze": []})
    return _state["models"][-1]


def _recording(name: str, original):
    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        if name == "model":
            _state["models"].append({"opened_by_model": True, "node": 0, "element": 0,
                                     "eigen": 0, "analyze": []})
        elif name == "analyze":
            _current_model()["analyze"].append(result)
        elif name == "recorder":
            _state["recorders"].append([str(a) for a in args[:_MAX_ARGS]])
        else:
            _current_model()[name] += 1
        return result

    wrapper.__apegmsh_eval_probe__ = True
    return wrapper


def _wrap_ops(module) -> None:
    _state["ops_modules"].append(module.__name__)
    for name in ("model", "node", "element", "eigen", "analyze", "recorder"):
        original = getattr(module, name, None)
        if original is None:
            _state["probe_errors"].append(f"{module.__name__} has no {name}")
        elif not getattr(original, "__apegmsh_eval_probe__", False):
            # A module that re-exports the other one's functions holds them wrapped
            # already; wrapping twice would count every call twice.
            setattr(module, name, _recording(name, original))


def _wrap_bridge(module) -> None:
    cls = getattr(module, "apeSees", None)
    if cls is None:
        _state["probe_errors"].append(f"{module.__name__} has no apeSees")
        return
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        fem = args[0] if args else kwargs.get("fem")
        try:
            n_nodes = len(fem.nodes)
        except Exception as exc:  # noqa: BLE001 - recorded, the grade fails on it
            _state["probe_errors"].append(f"apeSees fem has no node count: {exc!r}")
            n_nodes = None
        _state["apesees"].append({"fem_nodes": n_nodes})

    cls.__init__ = __init__


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, patch) -> None:
        self.inner, self.patch = inner, patch

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module) -> None:
        self.inner.exec_module(module)
        try:
            self.patch(module)
        except Exception as exc:  # noqa: BLE001 - recorded, the grade fails on it
            _state["probe_errors"].append(f"patching {module.__name__}: {exc!r}")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in _OPS_MODULES:
            patch = _wrap_ops
        elif fullname == _BRIDGE_MODULE:
            patch = _wrap_bridge
        else:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, patch)
        return spec


def _write() -> None:
    with open(_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_state, default=repr) + "\n")


def _chain_next_sitecustomize() -> None:
    """Run any other sitecustomize this one shadows, so the environment is unchanged."""
    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or ".") != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


if _LOG:
    sys.meta_path.insert(0, _Finder())
    atexit.register(_write)
_chain_next_sitecustomize()
