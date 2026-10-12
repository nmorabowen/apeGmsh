"""Compose v1.1-B slice B — results.viewer ColorMode controller path.

This file is the **architectural-decision lock** for slice B.b.

Survey finding (2026-05-27)
---------------------------
``results.viewer`` does NOT have a per-element ColorMode dropdown.
The mesh.viewer's ColorMode mechanism — ``DisplayTab`` + ``COLOR_MODES``
combo box + ``ColorModeController`` + idle-fn swap — is **mesh-side
only**.  ``ResultsViewer`` was designed diagram-centrically: a flat
substrate (theme-colored via ``palette.substrate_color``) with
contour / deformed-shape / vector-glyph diagrams overlaid on top.
Per-element coloring goes through the **diagram layers**, not a
ColorMode dropdown.

Key evidence (current ``main`` at PR #384):

* ``src/apeGmsh/viewers/core/color_mode_controller.py`` docstring
  begins with *"Drives **mesh-viewer** color modes."* — single
  consumer by design.
* ``ColorModeController`` is referenced only from
  ``src/apeGmsh/viewers/mesh_viewer.py`` and
  ``src/apeGmsh/viewers/core/color_mode_controller.py`` (self).
* ``COLOR_MODES`` lives in
  ``src/apeGmsh/viewers/ui/mesh_tabs.py`` and is consumed only by
  ``DisplayTab`` (also mesh-only — see #373 / #374 / #376).
* ``src/apeGmsh/viewers/results_viewer.py`` contains **zero**
  ``ColorMode`` / ``COLOR_MODES`` / ``_color_mgr`` / ``set_idle_fn``
  / ``_on_color_mode`` / ``_module_idle`` references.
* ``src/apeGmsh/viewers/ui/_results_window.py`` (the QMainWindow
  shell for ``ResultsViewer``) likewise has no ColorMode wiring; its
  only ``color`` reference is the Color-Map editor dock for contour
  colormaps (a diagram-styling tool, not a per-element coloring
  dropdown).

What v1.1-B slice B.b ships
---------------------------
**No source changes.**  Slice B.b is a documentation-and-regression
PR that:

1. Records the architectural decision: results.viewer's coloring
   model is diagram-overlay-on-substrate, not per-element ColorMode.
2. Locks the decision against silent drift via three regression
   guards (one per file).  If someone in the future ports the
   mesh-side ColorMode dropdown into results.viewer (e.g. to color
   the substrate by module before adding a contour layer), the new
   wiring will trip these tests and force them to:

   * delete the relevant guard (intentional decision), AND
   * mirror the 3F.2b / 3F.2d test surface on the new controller
     path so the Module / Module: Root / Module: Leaf modes have
     coverage on the results side.

Why this is the right answer for slice B.b
------------------------------------------
The kickoff memory (``project_compose_v1_1_b_results_viewer_kickoff``)
called out that the survey output dictates whether slice B.b is

  * **Path A** — mirror tests onto a shared controller,
  * **Path B** — fork ``_module_idle`` / Root / Leaf onto a separate
    results-side controller, or
  * **Path C** — no controller present in results.viewer; ship a
    documentation note.

The survey shows results.viewer has no ColorMode controller at all
— **Path C**.  Adding one without a user-driven need would be
speculative scope creep (CLAUDE.md §1 "Don't assume" + §2
"Simplicity first").  The data layer ``view.elements.module_for(eid)``
shipped in B.a (PR #384) remains available on the results-side
``ViewerData`` if a future diagram (e.g. ``ColorByModule``) wants to
consume it — but that's a diagram-layer feature, not a slice B.b
deliverable.

References
----------
* PR #373 — 3F.2b: ``_module_idle`` callback on mesh-side controller
* PR #376 — 3F.2d: ``_module_idle_by_root`` / ``_module_idle_by_leaf``
* PR #384 — 3F.2a results-side data-layer lock (parallel data layer)
* ``feedback_viewer_no_gpu`` — no headless rendering; verify
  structurally.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# No-openseespy CI gate per kickoff memo and ``feedback_viewer_no_gpu``.
# The asserted invariants only need viewer-source-text inspection, but
# the module imports ``apeGmsh._core`` which in turn pulls
# ``openseespy.opensees`` on this Python.
pytest.importorskip("openseespy.opensees")


# Tokens proving the mesh-side ColorMode mechanism is NOT wired into
# results.viewer or its window shell.  All tokens are searched as
# literal substrings (no regex) so the regression bites cleanly on
# any direct port of the mesh-side dropdown.
_COLOR_MODE_TOKENS = (
    "ColorModeController",   # the controller class
    "COLOR_MODES",           # the dropdown list
    "_color_mode_ctrl",      # mesh_viewer.py's attr name
    "_on_color_mode",        # mesh_viewer.py's callback
    "_module_idle",          # 3F.2b/d callbacks
    "set_idle_fn",           # ColorManager handoff
    "DisplayTab",            # mesh_tabs.py UI host
)


def _viewers_source(filename: str) -> str:
    """Read a source file under apeGmsh.viewers and return its text.

    Locates the file via the installed ``apeGmsh.viewers`` package so
    the editable-install-points-at-main caveat from
    ``reference_editable_install_main`` is honoured — we lock what
    the running interpreter sees, not what the worktree happens to
    contain.
    """
    import apeGmsh.viewers as viewers_pkg

    pkg_root = Path(viewers_pkg.__file__).parent
    target = pkg_root / filename
    assert target.exists(), (
        f"results-viewer source file missing: {target}. "
        "Slice B.b regression guard cannot run."
    )
    return target.read_text(encoding="utf-8", errors="ignore")


# =====================================================================
# Architectural lock — results.viewer has no per-element ColorMode
# =====================================================================


class TestResultsViewerHasNoColorModeController:
    """The results window must stay free of the mesh-side ColorMode
    dropdown wiring (``results_viewer.py`` itself was deleted at X3-d,
    #1734).  If someone wants to add per-element
    ColorMode (e.g. ``Module`` coloring on the substrate) they MUST
    update this test alongside the new controller so the test
    surface for ``_module_idle`` / Root / Leaf lands together — same
    rigour the mesh-side has in
    ``tests/test_phase_3f_2b_callback.py`` and
    ``tests/test_phase_3f_2d_root_leaf.py``."""

    @pytest.mark.parametrize("token", _COLOR_MODE_TOKENS)
    def test_results_window_source_has_no_color_mode_token(
        self, token: str,
    ) -> None:
        """No mesh-side ColorMode token appears in ``ui/_results_window.py``
        — the QMainWindow shell for ``ResultsViewer``."""
        src = _viewers_source("ui/_results_window.py")
        assert token not in src, (
            f"Found mesh-side ColorMode token {token!r} in "
            f"ui/_results_window.py.  If you're surfacing the "
            f"ColorMode dropdown in the results window, update this "
            f"test AND mirror the 3F.2b / 3F.2d coverage onto the "
            f"new wiring."
        )


# =====================================================================
# Cross-check — the mesh-side controller stays bound to mesh.viewer
# =====================================================================


class TestColorModeControllerIsMeshSideOnly:
    """The ``ColorModeController`` class advertises itself as
    mesh-side via its module docstring.  If the controller is ever
    generalized to cover results.viewer, the docstring should be
    updated alongside — at which point this guard reminds the
    author to also add results-side tests."""

    def test_controller_docstring_says_mesh_viewer(self) -> None:
        """Module docstring describes ColorModeController as
        mesh-viewer-only.  Verifies the design intent stays
        documented."""
        from apeGmsh.viewers.core import color_mode_controller as cmc

        doc = cmc.__doc__ or ""
        assert "mesh-viewer" in doc.lower() or "mesh viewer" in doc.lower(), (
            "ColorModeController module docstring no longer mentions "
            "mesh-viewer.  If the controller now also serves "
            "results.viewer, update tests/test_v1_1_b_b_results_controller.py "
            "to cover the new path."
        )

    def test_controller_is_imported_only_by_mesh_viewer(self) -> None:
        """``ColorModeController`` is referenced only from
        ``mesh_viewer.py`` (and the controller module itself).
        Scans the installed ``apeGmsh.viewers`` tree.  If any other
        file references it — including ``results_viewer.py`` — the
        single-consumer invariant has been broken and slice B.b's
        Path-C decision needs to be revisited."""
        import apeGmsh.viewers as viewers_pkg

        pkg_root = Path(viewers_pkg.__file__).parent

        offenders: list[str] = []
        for py_path in pkg_root.rglob("*.py"):
            # Skip the controller's own source — naturally references itself.
            if py_path.name == "color_mode_controller.py":
                continue
            try:
                src = py_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "ColorModeController" in src:
                offenders.append(str(py_path.relative_to(pkg_root)))

        expected = {"mesh_viewer.py"}
        unexpected = set(offenders) - expected
        assert not unexpected, (
            "ColorModeController is referenced from unexpected viewer "
            f"file(s): {sorted(unexpected)}.  Either the controller "
            "is being generalized (update this test + add results-"
            "side coverage) or the reference is accidental."
        )


# =====================================================================
# Forward-pointer — the v1.1-B data layer remains consumable
# =====================================================================


def test_v1_1_b_a_data_layer_is_still_results_side_callable() -> None:
    """Smoke check that the data-layer surface PR #384 locked is
    still importable through the public viewers API — the bridge a
    future results-side ColorByModule diagram (or controller, if
    that's the route taken) would consume.  If this import breaks,
    a downstream port of mesh-side ColorMode would also break."""
    from apeGmsh.viewers.data import ViewerData  # noqa: F401
