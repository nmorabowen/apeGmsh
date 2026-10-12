"""Warning attribution survives the S2-a split of ``_internal/build.py``.

``_stacklevel_outside_package`` names the first caller outside the
``apeGmsh`` package by counting ``dirname`` hops from its own ``__file__``,
so it must sit at ``_internal/`` depth (it does: ``_internal/_stacklevel.py``).
A warning raised from ``build/ndf.py`` and one from
``build/validators/loads.py`` must therefore point at THIS file, on the line
of the call that triggered it.
"""
from __future__ import annotations

import inspect
import os
import warnings
from typing import cast

from apeGmsh.opensees import (
    DetachedDiaphragmMasterWarning,
    UnconsumedModelDefinitionWarning,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees._internal.build.ndf import (
    validate_diaphragm_master_stiffness,
)

from tests.opensees.unit.test_diaphragm_master_stiffness import (
    _column_specs,
    _frame_with_master,
)
from tests.opensees.unit.test_unconsumed_model_definition import _model, _ops


FIX = {1: 6, 2: 6, 3: 6, 4: 6}


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _line() -> int:
    """Line number of the caller's current line."""
    return inspect.currentframe().f_back.f_lineno  # type: ignore[union-attr]


def test_ndf_warning_points_at_the_caller() -> None:
    fem = cast("object", _frame_with_master())
    specs = _column_specs(fem)  # type: ignore[arg-type]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_diaphragm_master_stiffness(fem, specs, 3, 6, FIX); call_line = _line()  # noqa: E702,E501
    hits = [w for w in caught
            if issubclass(w.category, DetachedDiaphragmMasterWarning)]
    assert len(hits) == 1
    assert _norm(hits[0].filename) == _norm(__file__)
    assert hits[0].lineno == call_line


def test_validators_loads_warning_points_at_the_caller() -> None:
    ops = _ops(_model())
    built = ops.build()
    rec = RecordingEmitter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        built.emit(rec); call_line = _line()  # noqa: E702
    hits = [w for w in caught
            if issubclass(w.category, UnconsumedModelDefinitionWarning)]
    assert len(hits) == 1
    assert _norm(hits[0].filename) == _norm(__file__)
    assert hits[0].lineno == call_line


def test_apegmsh_frames_in_between_are_skipped() -> None:
    """A caller inside ``apeGmsh`` but outside ``opensees`` (a synthetic
    frame filed under ``apeGmsh/core``) is skipped, so the warning still
    lands on this file. This is what a wrong ``dirname`` hop count in the
    helper breaks: a count that stops short of the package root treats
    that frame as the first outside caller."""
    import apeGmsh

    pkg = os.path.dirname(os.path.abspath(apeGmsh.__file__))
    fake = os.path.join(pkg, "core", "_fake_caller.py")
    ns: dict = {}
    exec(compile("def run(fn, *a):\n    return fn(*a)\n", fake, "exec"), ns)
    fem = cast("object", _frame_with_master())
    specs = _column_specs(fem)  # type: ignore[arg-type]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ns["run"](validate_diaphragm_master_stiffness, fem, specs, 3, 6, FIX); call_line = _line()  # noqa: E702,E501
    hits = [w for w in caught
            if issubclass(w.category, DetachedDiaphragmMasterWarning)]
    assert len(hits) == 1
    assert _norm(hits[0].filename) == _norm(__file__)
    assert hits[0].lineno == call_line
