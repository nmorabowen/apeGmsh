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

A diagram may be exempted ONLY if it renders no point geometry of its
own or is itself the deformation source — say why in ``_EXEMPT``.
"""
from __future__ import annotations

import inspect

import apeGmsh.viewers.diagrams as diagrams_pkg
from apeGmsh.viewers.diagrams._base import Diagram

# Class name -> why the base no-op is correct for it. Keep this list
# SHORT and justified; "I forgot" is exactly what the guard catches.
_EXEMPT: dict[str, str] = {
    # Legacy layer-kind that warps its OWN copy of the substrate from
    # the displacement field every step — it IS a deformation renderer;
    # following the globally-deformed substrate would double-warp it.
    "DeformedShapeDiagram": "renders its own warp from the displacement field",
}


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
        if cls.__name__ in _EXEMPT:
            continue
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
        "tests/viewers/test_diagram_deform_follow.py. Exempt a class "
        "in _EXEMPT only if it renders no point geometry of its own, "
        "with the reason."
    )


def test_exempt_list_is_live() -> None:
    # Exemptions must name real classes — a rename would otherwise
    # leave a stale entry silently exempting nothing.
    names = {c.__name__ for c in _all_diagram_classes()}
    stale = set(_EXEMPT) - names
    assert not stale, f"_EXEMPT names unknown diagram classes: {sorted(stale)}"
