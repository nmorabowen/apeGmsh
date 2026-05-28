"""Contract: every concrete emitter implements the full Emitter Protocol.

The :class:`apeGmsh.opensees.emitter.base.Emitter` Protocol is the
frozen interface every backend (Tcl / openseespy / H5 / live /
recording) must satisfy. It is deliberately **not**
``@runtime_checkable`` and it is widened often (ADR 0022 / 0024 /
0025 / 0027, stress-control, stage-bracketing, ``domain_change``).

Python does not error when a concrete emitter forgets a newly-added
Protocol method: the omission surfaces only at call time, and the
fixture-driven parity sweep (``parity/test_emitter_parity_sweep.py``,
``parity/test_h5_parity.py``) catches it only if a fixture happens to
exercise that exact primitive on that backend. This test closes the
gap structurally — it derives the method set from the Protocol body
itself (so the expected surface can't drift out of sync) and asserts
every backend implements all of it.

The pre-existing checks this complements:
  * ``h5/test_h5_emitter.py::test_h5emitter_protocol_conformance`` —
    a static-typing affordance (``e: Emitter = H5Emitter()``), not a
    runtime structural check; covers H5 only.
  * ``unit/test_emitter_protocol.py`` — a hand-curated
    ``REPRESENTATIVE_METHODS`` list, RecordingEmitter only, explicitly
    not the full surface.
"""
from __future__ import annotations

import inspect

import pytest

from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter


def _protocol_methods() -> list[str]:
    """Public method names declared on the Emitter Protocol body.

    Derived from ``vars(Emitter)`` so the expected surface tracks the
    Protocol automatically — a widening that adds a method extends the
    parametrization with no edit here.
    """
    return sorted(
        name
        for name, val in vars(Emitter).items()
        if not name.startswith("_") and inspect.isfunction(val)
    )


PROTOCOL_METHODS = _protocol_methods()

# All five concrete backends. Checked at the CLASS level (no
# instantiation) so this test needs neither a live openseespy domain
# nor an open HDF5 file.
EMITTERS = [
    TclEmitter,
    PyEmitter,
    H5Emitter,
    LiveOpsEmitter,
    RecordingEmitter,
]


def test_protocol_surface_is_nonempty() -> None:
    """Guard against the introspection silently yielding ``[]`` — which
    would make the conformance parametrization pass vacuously."""
    assert len(PROTOCOL_METHODS) >= 25, (
        f"Emitter Protocol surface looks too small ({PROTOCOL_METHODS}); "
        f"the introspection in _protocol_methods() probably broke."
    )


@pytest.mark.parametrize(
    "emitter_cls", EMITTERS, ids=lambda c: c.__name__,
)
@pytest.mark.parametrize("method_name", PROTOCOL_METHODS)
def test_emitter_implements_protocol_method(
    emitter_cls: type, method_name: str,
) -> None:
    """Each backend must define a callable for every Protocol method.

    A failure here means a Protocol widening landed without wiring this
    backend — the deck/archive it produces would silently drop that
    command (or raise ``AttributeError`` at call time).
    """
    attr = getattr(emitter_cls, method_name, None)
    assert callable(attr), (
        f"{emitter_cls.__name__} is missing Protocol method "
        f"{method_name!r}; a Protocol widening forgot this backend."
    )
