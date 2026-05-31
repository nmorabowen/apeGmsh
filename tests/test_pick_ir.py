"""ADR 0047 R-D.1 — pick IR value types + ``PickBackend`` Protocol.

Pins the pick-side IR (``PickMode`` / ``PickModifiers`` / ``PickRequest``
/ ``PickHit`` / ``BoxGesture``) and the structural ``PickBackend``
Protocol probe. The keystone refinement over the ADR draft — a
``PickHit`` carries a plain ``prop_id`` integer (the domain registry key),
never a VTK object — is asserted here.

Fully headless — no Qt / VTK. (The INV-2 no-vtk/pyvista guard lives in
``test_scene_ir_pure.py``, which globs the whole package and so already
covers ``_pick.py``.)
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.scene_ir import (
    BoxGesture,
    PickBackend,
    PickHit,
    PickMode,
    PickModifiers,
    PickRequest,
)


# ── PickMode ────────────────────────────────────────────────────────

def test_pickmode_is_str_enum() -> None:
    # Subclassing str lets the results controller's bare "node"/"element"
    # /"gp" strings compare equal during the R-D.2 migration.
    assert PickMode.NODE == "node"
    assert PickMode.ELEMENT == "element"
    assert PickMode.GP == "gp"
    assert PickMode.FIBER == "fiber"


def test_pickmode_members_complete() -> None:
    assert {m.value for m in PickMode} == {"node", "element", "gp", "fiber"}


# ── PickModifiers ───────────────────────────────────────────────────

def test_modifiers_default_all_false() -> None:
    m = PickModifiers()
    assert m.ctrl is False and m.alt is False


def test_modifiers_frozen_and_hashable() -> None:
    m = PickModifiers(ctrl=True, alt=True)
    assert hash(m) == hash(PickModifiers(ctrl=True, alt=True))
    with pytest.raises((AttributeError, TypeError)):
        m.ctrl = False  # type: ignore[misc]


# ── PickRequest ─────────────────────────────────────────────────────

def test_request_defaults_to_node_mode_no_modifiers() -> None:
    r = PickRequest(10, 20)
    assert r.x == 10 and r.y == 20
    assert r.mode is PickMode.NODE
    assert r.modifiers == PickModifiers()


def test_request_coerces_bare_string_mode() -> None:
    # The results controller speaks "element"; the front door normalizes.
    r = PickRequest(0, 0, mode="element")  # type: ignore[arg-type]
    assert r.mode is PickMode.ELEMENT


def test_request_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        PickRequest(0, 0, mode="bogus")  # type: ignore[arg-type]


def test_request_coerces_coords_to_int() -> None:
    r = PickRequest(10.7, 20.2)  # type: ignore[arg-type]
    assert r.x == 10 and r.y == 20
    assert isinstance(r.x, int) and isinstance(r.y, int)


def test_request_rejects_bad_modifiers_type() -> None:
    with pytest.raises(TypeError, match="PickModifiers"):
        PickRequest(0, 0, modifiers=(True, False))  # type: ignore[arg-type]


# ── PickHit ─────────────────────────────────────────────────────────

def test_hit_carries_prop_id_as_plain_int() -> None:
    # The keystone: the domain registry key crosses the seam as an int,
    # never a VTK actor. Both EntityRegistry and the results inventory
    # key on exactly this id().
    sentinel = object()
    hit = PickHit(world=(1.0, 2.0, 3.0), cell_id=4, prop_id=id(sentinel))
    assert hit.prop_id == id(sentinel)
    assert isinstance(hit.prop_id, int)
    assert hit.cell_id == 4
    assert hit.world == (1.0, 2.0, 3.0)


def test_hit_frozen() -> None:
    hit = PickHit(world=(0.0, 0.0, 0.0))
    assert hit.cell_id is None and hit.prop_id is None
    with pytest.raises((AttributeError, TypeError)):
        hit.cell_id = 1  # type: ignore[misc]


# ── BoxGesture ──────────────────────────────────────────────────────

def test_box_gesture_carries_crossing_and_modifiers() -> None:
    g = BoxGesture(
        box=(5, 5, 50, 60), crossing=True, modifiers=PickModifiers(ctrl=True)
    )
    assert g.box == (5, 5, 50, 60)
    assert g.crossing is True
    assert g.modifiers.ctrl is True


# ── PickBackend Protocol (structural) ───────────────────────────────

class _ConformingBackend:
    """A minimal duck-typed backend with the right method surface."""

    def resolve_pick(self, request):  # noqa: ANN001
        return PickHit(world=(0.0, 0.0, 0.0), cell_id=0, prop_id=1)

    def project_points(self, world):  # noqa: ANN001
        return np.asarray(world)[:, :2]

    def frustum_planes(self, box):  # noqa: ANN001
        return None

    def install(self, *, on_pick, on_hover=None, on_box=None):  # noqa: ANN001
        pass

    def uninstall(self):
        pass


class _ViewOnlyBackend:
    """No pick surface — must NOT satisfy the Protocol."""

    def render(self) -> None:
        pass


def test_conforming_backend_satisfies_protocol() -> None:
    assert isinstance(_ConformingBackend(), PickBackend)


def test_view_only_backend_does_not_satisfy_protocol() -> None:
    assert not isinstance(_ViewOnlyBackend(), PickBackend)


def test_resolve_pick_round_trips_through_protocol() -> None:
    backend: PickBackend = _ConformingBackend()
    hit = backend.resolve_pick(PickRequest(3, 4, mode=PickMode.ELEMENT))
    assert hit is not None
    assert hit.prop_id == 1 and hit.cell_id == 0
