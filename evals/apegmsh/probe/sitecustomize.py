"""Grading probe: logs what apeGmsh and OpenSees did in every Python process of a grade.

The skill-eval graders put this directory on ``PYTHONPATH`` and pass a log path and a
per-grade random key in ``APEGMSH_EVAL_PROBE_LOG`` / ``APEGMSH_EVAL_PROBE_KEY``. Every
Python process of the graded run imports this module at startup. It wraps:

* the OpenSees module (``openseespy.opensees``, or the fork's ``opensees``) when it
  is imported. Each ``model`` call opens a new model record (a bridge that re-emits
  wipes and rebuilds, so totals would double count); each record holds its counts of
  ``node`` / ``element`` / ``eigen`` calls and every ``analyze`` return code. Each
  ``recorder`` call's arguments are kept per process;
* ``apeGmsh.opensees.apesees.apeSees.__init__``: how many bridges were built, and
  the node count of the FEMData each was built from;
* ``subprocess.Popen``, to hand the log path and the key to child processes (a deck
  run is one), since the parent no longer has them in its environment.

Tamper resistance. Before the graded script's own code runs, both variables are
popped from ``os.environ``, and the key, the log path and the state live only in
closures inside ``_install``: ``import sitecustomize`` from the script finds no
attribute holding them. At exit each process appends one line holding its
record and an HMAC-SHA256 of it under the key; the grader drops nothing silently, and
a line whose MAC fails fails the grade. A script that appends a record, or edits one,
therefore fails. Code that walks interpreter internals for the key is out of scope.
A child started with ``-I``, ``-E`` or ``-S`` is not handed the key (it would not run
this module, so its own code could read it), and logs nothing.

Without the variables this module does nothing beyond running any sitecustomize it
shadows. A wrapping failure is logged as ``probe_errors``, which fails the grade.
"""
from __future__ import annotations

import os
import sys

_LOG_ENV = "APEGMSH_EVAL_PROBE_LOG"
_KEY_ENV = "APEGMSH_EVAL_PROBE_KEY"


def _install(log: str, key: bytes) -> None:
    import atexit
    import hashlib
    import hmac
    import importlib.abc
    import importlib.machinery
    import json
    import subprocess

    ops_modules = {"opensees", "openseespy.opensees"}
    bridge_module = "apeGmsh.opensees.apesees"
    max_args = 16           # recorder arguments kept per call (node lists can be long)
    no_site_flags = {"-I", "-E", "-S"}
    state: dict = {
        "pid": os.getpid(), "argv": list(sys.argv), "cwd": os.getcwd(),
        "ops_modules": [], "models": [], "recorders": [], "apesees": [], "probe_errors": [],
    }

    def current_model() -> dict:
        if not state["models"]:
            # OpenSees calls before any model(): kept, so the grader sees them, never matched
            state["models"].append({"opened_by_model": False, "node": 0, "element": 0,
                                    "eigen": 0, "analyze": []})
        return state["models"][-1]

    def recording(name: str, original):
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            if name == "model":
                state["models"].append({"opened_by_model": True, "node": 0, "element": 0,
                                        "eigen": 0, "analyze": []})
            elif name == "analyze":
                current_model()["analyze"].append(result)
            elif name == "recorder":
                state["recorders"].append([str(a) for a in args[:max_args]])
            else:
                current_model()[name] += 1
            return result

        wrapper.__apegmsh_eval_probe__ = True
        return wrapper

    def wrap_ops(module) -> None:
        state["ops_modules"].append(module.__name__)
        for name in ("model", "node", "element", "eigen", "analyze", "recorder"):
            original = getattr(module, name, None)
            if original is None:
                state["probe_errors"].append(f"{module.__name__} has no {name}")
            elif not getattr(original, "__apegmsh_eval_probe__", False):
                # A module that re-exports the other one's functions holds them wrapped
                # already; wrapping twice would count every call twice.
                setattr(module, name, recording(name, original))

    def wrap_bridge(module) -> None:
        cls = getattr(module, "apeSees", None)
        if cls is None:
            state["probe_errors"].append(f"{module.__name__} has no apeSees")
            return
        original = cls.__init__

        def __init__(self, *args, **kwargs):
            original(self, *args, **kwargs)
            fem = args[0] if args else kwargs.get("fem")
            try:
                n_nodes = len(fem.nodes)
            except Exception as exc:  # noqa: BLE001 - recorded, the grade fails on it
                state["probe_errors"].append(f"apeSees fem has no node count: {exc!r}")
                n_nodes = None
            state["apesees"].append({"fem_nodes": n_nodes})

        cls.__init__ = __init__

    class Loader(importlib.abc.Loader):
        def __init__(self, inner, patch) -> None:
            self.inner, self.patch = inner, patch

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module) -> None:
            self.inner.exec_module(module)
            try:
                self.patch(module)
            except Exception as exc:  # noqa: BLE001 - recorded, the grade fails on it
                state["probe_errors"].append(f"patching {module.__name__}: {exc!r}")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname in ops_modules:
                patch = wrap_ops
            elif fullname == bridge_module:
                patch = wrap_bridge
            else:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            spec.loader = Loader(spec.loader, patch)
            return spec

    popen_init = subprocess.Popen.__init__

    def popen(self, args, *rest, **kwargs):
        argv = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
        if not no_site_flags & {str(a) for a in argv}:
            env = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
            env[_LOG_ENV], env[_KEY_ENV] = log, key.hex()
            kwargs["env"] = env
        popen_init(self, args, *rest, **kwargs)

    def write() -> None:
        payload = json.dumps(state, default=repr, sort_keys=True)
        mac = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"record": payload, "mac": mac}) + "\n")

    subprocess.Popen.__init__ = popen
    sys.meta_path.insert(0, Finder())
    atexit.register(write)


def _chain_next_sitecustomize() -> None:
    """Run any other sitecustomize this one shadows, so the environment is unchanged."""
    import importlib.machinery
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or ".") != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


_log = os.environ.pop(_LOG_ENV, None)
_key = os.environ.pop(_KEY_ENV, None)
if _log and _key:
    _install(_log, bytes.fromhex(_key))
del _log, _key
_chain_next_sitecustomize()
