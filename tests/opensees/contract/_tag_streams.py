"""The ``(kind, tag)`` stream one ``BuiltModel`` emit drives (K1-3, #1361).

Shared by the tag-law pins:

* ``tests/opensees/contract/test_tag_streams.py`` compares the
  ``(verb, tag)`` multiset across the Recording, Tcl, Py and H5 emits of
  one ``BuiltModel``;
* ``tests/opensees/subprocess/test_tag_determinism.py`` compares the
  stream built in-process against the one a fresh interpreter builds.

A stream is read by a *tap*: a subclass of the concrete emitter whose
every Protocol verb records ``(kind, tag)`` and then defers to the real
method, so the bridge sees the class it would see anyway (its
``isinstance`` and ``supports_partitions`` branches are unchanged).
Only the outermost verb is recorded, so an emitter that calls its own
verbs internally does not add rows.

Which argument is the tag is derived from the ``Emitter`` Protocol's
signatures, not from a hand table: the first parameter named in
:data:`TAG_PARAMS`. When that parameter is the second one, the first is
the type token (``element('quad', 7, ...)``) and joins the kind.
"""
from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.verbs import VERBS

#: Protocol parameter names that carry an object's tag.
TAG_PARAMS: tuple[str, ...] = ("tag", "ele_tag", "pid")

Stream = list[tuple[str, int]]


def tag_positions() -> dict[str, int]:
    """Protocol verb -> index of its tag argument (verbs with no tag omitted)."""
    out: dict[str, int] = {}
    for verb, row in VERBS.items():
        if row.via != "protocol":
            continue
        params = [
            p for p in inspect.signature(getattr(Emitter, verb)).parameters.values()
            if p.name != "self"
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        for i, p in enumerate(params):
            if p.name in TAG_PARAMS:
                out[verb] = i
                break
    return out


_POSITIONS = tag_positions()


def _project(verb: str, args: tuple[Any, ...]) -> tuple[str, int] | None:
    i = _POSITIONS.get(verb)
    if i is None:
        return None
    tag = args[i]
    if isinstance(tag, bool) or not isinstance(tag, (int, np.integer)):
        raise TypeError(
            f"{verb}: argument {i} should be a tag, got {tag!r}; the "
            "Protocol signature and the call disagree"
        )
    kind = f"{verb}:{args[0]}" if i == 1 else verb
    return kind, int(tag)


def tapped(cls: type) -> type:
    """A subclass of ``cls`` whose Protocol verbs append to ``self.tap``."""
    ns: dict[str, Any] = {}
    for verb, row in VERBS.items():
        if row.via != "protocol":
            continue
        orig = getattr(cls, verb, None)
        if not callable(orig):
            raise AttributeError(f"{cls.__name__} lacks Protocol verb {verb!r}")

        def make(verb: str, orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                if self._tap_depth == 0:
                    row = _project(verb, args)
                    if row is not None:
                        self.tap.append(row)
                self._tap_depth += 1
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    self._tap_depth -= 1
            wrapper.__name__ = verb
            return wrapper

        ns[verb] = make(verb, orig)
    ns["tap"] = None
    ns["_tap_depth"] = 0
    return type(f"Tapped{cls.__name__}", (cls,), ns)


def emit_stream(
    bm: Any, emitter_cls: type, *, split: bool = False,
    **emitter_kwargs: Any,
) -> Stream:
    """Emit the ``BuiltModel`` ``bm`` through a tapped ``emitter_cls``."""
    emitter = tapped(emitter_cls)(**emitter_kwargs)
    emitter.tap = []
    with warnings.catch_warnings():
        # Advisory nudges (auto numberer/system on a partitioned mesh) are
        # not what these pins are about.
        warnings.simplefilter("ignore")
        bm.emit(emitter, split=split)
    stream: Stream = emitter.tap
    return stream


def tag_stream(
    ops: apeSees, emitter_cls: type, *, split: bool = False,
    **emitter_kwargs: Any,
) -> Stream:
    """Build ``ops`` and emit it through a tapped ``emitter_cls``."""
    return emit_stream(ops.build(), emitter_cls, split=split, **emitter_kwargs)


# ---------------------------------------------------------------------------
# The models the pins drive
# ---------------------------------------------------------------------------


#: Golden-corpus modes the streams cover: the flat, partitioned and staged
#: emit paths (``per_rank`` is the partitioned path sliced by the Tcl writer).
GOLDEN_MODES: tuple[str, ...] = (
    "flat", "partitioned", "staged", "staged_partitioned",
)


def models() -> dict[str, Callable[[], apeSees]]:
    """Name -> a fresh ``apeSees`` recipe, covering every emit path but split.

    The golden grid's ``recording`` cells carry node, element and MPCO
    recorders (region tags), and the arch fixture fans out one
    ``geomTransf`` per element. The H5 suites' fixtures add parameter
    tags: a flat initial stress, a staged one, and a staged absorbing flip.
    """
    from tests.opensees.golden import builder as golden
    from tests.opensees.h5.test_h5_initial_stress import _build_frame
    from tests.opensees.h5.test_h5_stages_reader import (
        _real_kitchen_sink_bridge,
        _real_two_stage_bridge,
    )

    out: dict[str, Callable[[], apeSees]] = {}
    for fixture in golden.FIXTURES:
        for mode in GOLDEN_MODES:
            if golden.applicability(fixture, mode, "recording") is not None:
                continue
            out[f"{fixture}/{mode}"] = (
                lambda f=fixture, m=mode: golden.build_model(f, m, "recording")
            )
    out["initial_stress_frame/flat"] = (
        lambda: _build_frame(with_initial_stress=True))
    out["two_stage_initial_stress/staged"] = _real_two_stage_bridge
    out["kitchen_sink_absorbing/staged"] = _real_kitchen_sink_bridge
    return out


def split_model() -> apeSees:
    """The golden two-module frame, emitted with ``split=True`` (Tcl/Py only)."""
    from tests.opensees.golden import builder as golden
    return golden.build_model("two_module_frame", "flat", "recording")


def all_streams() -> dict[str, list[list[object]]]:
    """Every model's Recording stream plus the split Tcl stream, JSON-ready."""
    from apeGmsh.opensees.emitter.recording import RecordingEmitter
    from apeGmsh.opensees.emitter.tcl import TclEmitter

    out: dict[str, list[list[object]]] = {}
    for name, recipe in models().items():
        out[name] = [list(r) for r in tag_stream(recipe(), RecordingEmitter)]
    out["two_module_frame/split"] = [
        list(r) for r in tag_stream(split_model(), TclEmitter, split=True)
    ]
    return out
