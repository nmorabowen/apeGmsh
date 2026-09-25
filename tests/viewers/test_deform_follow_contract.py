"""Deform-follow contract guard — every rendering diagram implements
``sync_substrate_points``.

The contract (see :meth:`Diagram.sync_substrate_points`): post-ADR-0042
every diagram emits backend-owned dataset COPIES — extracted submeshes
included — so the base no-op pins a layer to the reference
configuration while the substrate warps. The viewer's DEFORM pump fans
out through this hook and through NOTHING else (the old
``_sync_layer_grids`` actor walk was dead code and is deleted).

This guard is the structural half of the enforcement (the behavioural
half is ``test_diagram_deform_follow.py``'s shift/reset contract): it
walks every ``Diagram`` subclass in ``apeGmsh.viewers.diagrams`` — the
package a diagram must live in to be registered — so a NEW diagram
that forgets the hook fails here at test time, not silently in a
user's deformed view. Mirrors the ADR 0056 state-contract guard
pattern (``test_viewer_state_contract.py``).

The contract is unconditional: every discovered diagram must override
the hook — no exemption mechanism exists (ADR 0058 S4 retired the last
exempt diagram, ``DeformedShapeDiagram``).
"""
from __future__ import annotations

import inspect

import apeGmsh.viewers.diagrams as diagrams_pkg
from apeGmsh.viewers.diagrams._base import Diagram


def _all_diagram_classes() -> list[type]:
    """Every Diagram subclass defined under ``apeGmsh.viewers``.

    Union of the package's public exports and the live
    ``__subclasses__`` graph (filtered to apeGmsh modules so
    test-local dummy subclasses imported by other test files can't
    leak in).
    """
    found: dict[str, type] = {}
    for name in dir(diagrams_pkg):
        obj = getattr(diagrams_pkg, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, Diagram)
            and obj is not Diagram
        ):
            found[obj.__name__] = obj
    stack: list[type] = [Diagram]
    while stack:
        for sub in stack.pop().__subclasses__():
            if not sub.__module__.startswith("apeGmsh.viewers"):
                continue
            if sub.__name__ not in found:
                found[sub.__name__] = sub
            stack.append(sub)
    return sorted(found.values(), key=lambda c: c.__name__)


def test_discovery_sees_the_diagram_family() -> None:
    # Self-check: if discovery breaks, the contract test would pass
    # vacuously. Pin a handful of kinds that must always be found.
    names = {c.__name__ for c in _all_diagram_classes()}
    assert {
        "ContourDiagram",
        "FiberSectionDiagram",
        "LayerStackDiagram",
        "LineForceDiagram",
        "SpringForceDiagram",
    } <= names


def test_every_rendering_diagram_overrides_sync_substrate_points() -> None:
    missing: list[str] = []
    for cls in _all_diagram_classes():
        if cls.sync_substrate_points is Diagram.sync_substrate_points:
            missing.append(cls.__name__)
    assert not missing, (
        f"{missing} inherit(s) the base no-op sync_substrate_points. "
        "Post-ADR-0042 backend datasets are COPIES: without an override "
        "the layer stays pinned at the reference configuration while "
        "the substrate deforms (the regression PR #620 fixed). "
        "Implement sync_substrate_points — re-sample cached "
        "vtkOriginalPointIds rows for substrate-extracted submeshes, "
        "or recompute owned geometry — and add a shift/reset case to "
        "tests/viewers/test_diagram_deform_follow.py."
    )


def test_every_rendering_diagram_overrides_set_visible() -> None:
    # The base ``Diagram.set_visible`` walks ``self._actors``, which no
    # diagram has populated since the ADR 0042 R-B migration — the dead
    # list behind #593 (composition gate a silent no-op) and #620. A kind
    # that inherits it has a show/hide toggle, and a gate
    # (``apply_effective_visibility`` routes through it), that do
    # nothing. G-ACTORS in test_viewer_state_contract.py exempts
    # ``self._actors`` (the base owns the field), so this completeness
    # check is what closes that hole. All kinds override it today.
    missing = [
        cls.__name__ for cls in _all_diagram_classes()
        if cls.set_visible is Diagram.set_visible
    ]
    assert not missing, (
        f"{missing} inherit(s) the base set_visible, which walks the dead "
        "_actors list — show/hide and the composition gate would be silent "
        "no-ops (the #593 bug class). Override set_visible to toggle the "
        "diagram's backend layer handles."
    )
