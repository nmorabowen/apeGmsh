"""ADR 0098 A7 (R1) — the pane's section-plane gizmos.

The named ``tested_by`` for ``install_clip_gizmo_interactor`` in
``test_session_capability_parity.py``: this file is what makes that
registry entry's "ported" claim mean something.

Headless throughout. :class:`PaneClipBinding` needs a backend with the
scene-IR layer verbs and a reconciler-shaped object with a ``view``; it
does not need Qt, a window, or a real interactor. The two traps it
exists to avoid are both asserted here rather than described:

* layers added through the reconciler's ``LedgerBackend`` are swept by
  ``_teardown`` on the next realize;
* the legend binding re-installs its priority-12 observers on every
  realize, which would leave the gizmo in front of the colour bar.
"""
from __future__ import annotations

import pytest

from apeGmsh.results.session._views import MeshView
from apeGmsh.viewers.session._clip_bind import PaneClipBinding
from apeGmsh.viewers.session._clip_control import ViewClipController

BOUNDS = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)


class FakeBackend:
    """Scene-IR layer verbs plus a render counter."""

    def __init__(self) -> None:
        self.layers: "dict[int, object]" = {}
        self._next = 0
        self.renders = 0
        self.ray: tuple = ((5.0, 5.0, 50.0), (0.0, 0.0, -1.0))

    def add_layer(self, layer):
        self._next += 1
        self.layers[self._next] = layer
        return self._next

    def update_layer(self, handle, layer):
        self.layers[handle] = layer

    def remove_layer(self, handle):
        self.layers.pop(handle, None)

    def render(self):
        self.renders += 1

    def display_to_world_ray(self, x, y):
        return self.ray


class FakeLedger:
    """The reconciler's ledger: forwards, and records what went through
    IT so ``_teardown`` can sweep exactly that."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.layer_handles: list = []

    @property
    def inner(self):
        return self._inner

    def add_layer(self, layer):
        handle = self._inner.add_layer(layer)
        self.layer_handles.append(handle)
        return handle

    def teardown(self) -> None:
        """What ``SessionReconciler._teardown`` does, in miniature."""
        for handle in list(self.layer_handles):
            self._inner.remove_layer(handle)
        self.layer_handles.clear()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class FakeReconciler:
    def __init__(self, view) -> None:
        self.view = view


class FakeRealized:
    def __init__(self, bounds=BOUNDS) -> None:
        self.reference_bounds = bounds


def _view(**clip):
    view = MeshView(pane_id="mesh-1")
    view.add_clip((1.0, 0.0, 0.0), offset=5.0, **clip)
    return view


def _bound(view, backend=None, bounds=BOUNDS):
    backend = backend or FakeBackend()
    binding = PaneClipBinding(backend, FakeReconciler(view))
    binding.refresh(FakeRealized(bounds))
    return backend, binding


# ---------------------------------------------------------------------
# The gizmo exists on the session path (criterion 1)
# ---------------------------------------------------------------------

def test_a_pane_with_a_visible_clip_draws_a_gizmo():
    """A7 criterion 1 — this is the whole of R1's defect, inverted.

    Before the binding, the session emitted no gizmo actors at all, so
    there was nothing for an interactor to grab.
    """
    backend, binding = _bound(_view())
    assert binding.gizmos is not None
    assert set(binding.gizmos.geometries()) == {"clip-1"}
    assert len(backend.layers) == 2, "expected a quad layer and an arrow layer"


def test_the_gizmo_layers_are_clip_exempt_and_unpickable():
    """A gizmo sliced by its own plane is useless, and a pickable quad
    covering the cross-section would eat every pick near it."""
    backend, _binding = _bound(_view())
    for layer in backend.layers.values():
        assert layer.clip_exempt is True, f"{layer.layer_id} is not clip_exempt"
        assert layer.pickable is False, f"{layer.layer_id} is pickable"
        assert layer.layer_id.startswith("clip_gizmo:")


def test_the_controller_is_the_five_member_adapter():
    _backend, binding = _bound(_view())
    assert isinstance(binding.controller, ViewClipController)


def test_a_view_with_no_clips_draws_nothing():
    backend = FakeBackend()
    binding = PaneClipBinding(backend, FakeReconciler(MeshView(pane_id="m")))
    binding.refresh(FakeRealized())
    assert backend.layers == {}


def test_no_bounds_means_no_gizmo_rather_than_a_guessed_box():
    """A realize that could not resolve reference bounds must not get a
    grab handle placed somewhere the model is not."""
    backend = FakeBackend()
    binding = PaneClipBinding(backend, FakeReconciler(_view()))
    binding.refresh(FakeRealized(bounds=None))
    assert backend.layers == {}
    assert binding.gizmos is None


def test_no_realization_yet_means_no_gizmo():
    backend = FakeBackend()
    binding = PaneClipBinding(backend, FakeReconciler(_view()))
    binding.refresh(None)
    assert backend.layers == {}


# ---------------------------------------------------------------------
# Surviving realize (criterion 4) — the LedgerBackend trap
# ---------------------------------------------------------------------

def test_the_gizmo_survives_a_realize_teardown():
    """A7 criterion 4, and the trap that is only visible from the
    reconciler's side.

    Realize emits through a ``LedgerBackend`` and ``_teardown`` removes
    EVERY handle that ledger recorded. A gizmo added through it would
    vanish on the next slot edit while the plane stayed — so the
    binding must build on ``ledger.inner``.
    """
    inner = FakeBackend()
    ledger = FakeLedger(inner)
    binding = PaneClipBinding(ledger, FakeReconciler(_view()))
    binding.refresh(FakeRealized())
    assert len(inner.layers) == 2

    ledger.teardown()          # what every full realize does first

    assert len(inner.layers) == 2, (
        "the gizmo went through the ledger and was swept by _teardown"
    )
    assert ledger.layer_handles == [], "the ledger recorded gizmo handles"


def test_the_renderer_is_the_same_object_across_flushes():
    """The renderer is a reconciler with an in-place fast path, so it
    must be LONG-LIVED — rebuilding it per flush would churn the actors
    a drag depends on re-using."""
    _backend, binding = _bound(_view())
    first = binding.gizmos
    for _ in range(3):
        binding.refresh(FakeRealized())
    assert binding.gizmos is first


def test_changed_reference_bounds_rebuild_the_renderer():
    """The bbox is baked in at construction, so a scope flip that
    changes the pane's extent must replace the renderer rather than
    leave a quad sized to geometry the pane no longer draws."""
    _backend, binding = _bound(_view())
    first = binding.gizmos
    # Read the arrow BEFORE the rebuild: replacing the renderer clears
    # the old one, so its geometry is gone by the time we could ask.
    big = first.geometries()["clip-1"].arrow_len

    binding.refresh(FakeRealized(bounds=(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)))

    assert binding.gizmos is not first, "the bbox change did not rebuild"
    assert first.geometries() == {}, "the replaced renderer kept its layers"
    assert binding.gizmos.geometries()["clip-1"].arrow_len < big, (
        "the quad is still sized to the old, larger extent"
    )


def test_retargeting_the_pane_drops_the_old_view_planes():
    """``set_pane`` points the reconciler at another view; the gizmos
    must follow rather than keep cutting with the old one's planes."""
    view_a, view_b = _view(), MeshView(pane_id="mesh-2")
    view_b.add_clip((0.0, 1.0, 0.0), offset=1.0)
    backend = FakeBackend()
    reconciler = FakeReconciler(view_a)
    binding = PaneClipBinding(backend, reconciler)
    binding.refresh(FakeRealized())
    assert binding.controller.view is view_a

    reconciler.view = view_b
    binding.refresh(FakeRealized())
    assert binding.controller.view is view_b
    assert set(binding.gizmos.geometries()) == {"clip-1"}


# ---------------------------------------------------------------------
# The view drives the picture
# ---------------------------------------------------------------------

def test_a_clip_edit_moves_the_gizmo_on_the_next_refresh():
    _backend, binding = _bound(_view())
    view = binding.controller.view
    before = binding.gizmos.geometries()["clip-1"].centre

    view.set_clip("clip-1", offset=8.0)
    binding.refresh(FakeRealized())
    after = binding.gizmos.geometries()["clip-1"].centre

    assert before[0] == pytest.approx(5.0)
    assert after[0] == pytest.approx(8.0)


def test_the_eye_hides_one_gizmo_without_touching_the_cut():
    _backend, binding = _bound(_view())
    view = binding.controller.view
    view.set_clip("clip-1", gizmo_visible=False)
    binding.refresh(FakeRealized())
    assert binding.gizmos.geometries() == {}
    assert view.clips[0].active is True, "hiding the gizmo unset the cut"


@pytest.mark.parametrize("n_planes", [1, 2, 3])
def test_show_gizmos_derives_exactly_over_every_flag_combination(n_planes):
    """A7 criterion 10 — the derivation is exact, not approximate.

    For every combination of per-plane flags the drawn set must equal
    ``{P : P.gizmo_visible}``, which is what makes deriving
    ``show_gizmos`` equivalent to the old viewer-wide flag rather than
    merely cheaper.
    """
    import itertools

    view = MeshView(pane_id="mesh-1")
    axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for i in range(n_planes):
        view.add_clip(axes[i], offset=float(i + 1))
    _backend, binding = _bound(view)

    for flags in itertools.product([True, False], repeat=n_planes):
        for clip, visible in zip(view.clips, flags):
            view.set_clip(clip.plane_id, gizmo_visible=visible)
        binding.refresh(FakeRealized())
        expected = {c.plane_id for c in view.clips if c.gizmo_visible}
        assert set(binding.gizmos.geometries()) == expected, (
            f"flags {flags} drew the wrong set"
        )


# ---------------------------------------------------------------------
# Legend precedence (criterion 8) and lifecycle (criterion 11)
# ---------------------------------------------------------------------

class _RecordingInteractor:
    """Stands in for the real one; records install/uninstall order."""

    order: list = []

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.installed = False

    def install(self) -> bool:
        self.installed = True
        _RecordingInteractor.order.append(self.tag)
        return True

    def uninstall(self) -> None:
        self.installed = False
        if self.tag in _RecordingInteractor.order:
            _RecordingInteractor.order.remove(self.tag)


def test_reseat_puts_the_gizmo_back_behind_the_legend(monkeypatch):
    """A7 criterion 8 — the defect that would only appear AFTER the
    first slot edit.

    VTK invokes same-priority observers in insertion order, and ADR
    0083 relies on it: the legend is installed first so a bar
    overlapping a gizmo wins the press. ``_realize_legends`` builds a
    new controller every realize, so the legend binding re-installs —
    landing it BEHIND the gizmo and silently inverting the rule.
    """
    _RecordingInteractor.order = ["legend"]        # installed first, at boot
    fake = _RecordingInteractor("gizmo")
    monkeypatch.setattr(
        "apeGmsh.viewers.core._clip_gizmo_interactor"
        ".install_clip_gizmo_interactor",
        lambda backend, controller, gizmos: (fake.install(), fake)[1],
    )
    _backend, binding = _bound(_view())
    assert _RecordingInteractor.order == ["legend", "gizmo"]

    # A full realize: the legend binding re-installs its observers.
    _RecordingInteractor.order.remove("legend")
    _RecordingInteractor.order.append("legend")
    assert _RecordingInteractor.order == ["gizmo", "legend"], (
        "precondition — the realize should have inverted the order"
    )

    binding.reseat()
    assert _RecordingInteractor.order == ["legend", "gizmo"], (
        "reseat did not put the gizmo back behind the colour bar"
    )


def test_reseat_is_safe_with_no_interactor(monkeypatch):
    """A headless backend installs none; re-seating must not raise."""
    monkeypatch.setattr(
        "apeGmsh.viewers.core._clip_gizmo_interactor"
        ".install_clip_gizmo_interactor",
        lambda backend, controller, gizmos: None,
    )
    _backend, binding = _bound(_view())
    assert binding.interactor is None
    binding.reseat()


def test_dispose_removes_every_layer_and_is_idempotent():
    """A7 criterion 11 — the layers live in a context the pane is about
    to release, and the observers on an interactor it is about to
    close."""
    backend, binding = _bound(_view())
    assert len(backend.layers) == 2

    binding.dispose()
    assert backend.layers == {}, "gizmo layers outlived the pane"
    assert binding.gizmos is None
    assert binding.interactor is None

    binding.dispose()          # must not raise


def test_a_headless_backend_gets_layers_but_no_gesture():
    """The offscreen row of A7's state table: the picture is harmless
    and the gesture is simply absent — not an error."""
    class NoInteractor(FakeBackend):
        display_to_world_ray = None

    backend = NoInteractor()
    _b, binding = _bound(_view(), backend=backend)
    assert len(backend.layers) == 2
    assert binding.interactor is None
