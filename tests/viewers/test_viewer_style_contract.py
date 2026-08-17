"""ADR 0087 D3 — viewer visual-design-system style guards.

Machine-enforces the enforceable slice of
[ADR 0087](../../src/apeGmsh/opensees/architecture/decisions/0087-viewer-visual-design-system.md):
no literal colors outside ``theme.py`` (G-HEX), no shouted header
labels (G-SHOUT), no dangling QSS selectors (G-QSS-SYNC), and a
per-file ``setStyleSheet`` call-site ratchet (G-INLINE). Sibling of
``test_viewer_state_contract.py`` and reusing its shape: AST/text walk
over ``viewers/ui/**``, per-file allowlisted budgets, **two-way
ratchet** — over budget fails; under budget fails with "ratchet the
allowlist down"; a file not listed fails on its FIRST violation.
Raising a budget requires citing ADR 0087 in the allowlist comment.

Budgets below were MEASURED at guard-landing time (GUI polish phase 1,
2026-08-04), not guessed. Phase-1 itself took G-SHOUT to zero; G-HEX
and G-INLINE enter with the measured debt and burn down from here.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

VIEWERS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "apeGmsh"
    / "viewers"
)
UI_DIR = VIEWERS_DIR / "ui"

# ── G-HEX allowlist — literal colors (hex anywhere in string
#    constants; named colors inside setStyleSheet / QColor) per file.
#    Every entry is measured phase-1 debt; the target outside
#    theme_editor_dialog.py is a burn-down to 0 (ADR 0087 G-HEX).
_HEX_ALLOW: dict[str, int] = {
    # "color: gray" hint styling — migrate to an EmptyHint objectName.
    "ui/_clip_plane_panel.py": 1,
    "ui/_clip_planes_panel.py": 2,
    "ui/_diagram_settings_tab.py": 1,
    # Neutral mid-gray fallback when the Qt palette isn't queryable.
    "ui/_eye_icon_delegate.py": 1,
    # "color: gray" hint styling — same burn-down as the clip panels.
    "ui/_measure_panel.py": 1,
    # Legacy Catppuccin-hardcoded row tints (pre-Palette roles).
    "ui/_selection_tree.py": 6,
    # Matplotlib line-color default (plot content, not chrome).
    "ui/_thickness_panel.py": 1,
    "ui/_time_history.py": 1,
    # Per-DOF constraint color table — needs a palette-role home.
    "ui/constraints_tab.py": 14,
    # Per-pattern load color table — same burn-down as constraints_tab.
    "ui/loads_tab.py": 7,
    # Reset-button warning tint — should be palette.error.
    "ui/preferences.py": 1,
    # "color: gray" hint styling.
    "ui/preferences_dialog.py": 1,
    # Legitimately manipulates hex text (swatches, defaults) — enters
    # with a measured budget per ADR 0087 G-HEX.
    "ui/theme_editor_dialog.py": 6,
    # Corner-triad axis colors (x/y/z convention) + "gray" fallback.
    "ui/viewer_window.py": 3,
}

# ── G-SHOUT — zero budget: phase 1 retired every letter-spaced
#    all-caps header (ADR 0087 INV-1 / G-SHOUT target 0).
_SHOUT_ALLOW: dict[str, int] = {}

# Acronyms that legitimately render all-caps (kept HERE, not as
# inline pragmas — ADR 0087 G-SHOUT).
_SHOUT_ACRONYMS = frozenset({"FEM", "HUD", "ID", "IDS", "X Y Z"})

# ── G-INLINE — setStyleSheet call sites per file. theme.py is the QSS
#    factory; the ONE designated sheet writer is the window host
#    (``ViewerWindow._apply_palette``). Everything else is tolerated
#    legacy that ratchets down as widgets migrate to objectName blocks
#    (ADR 0087 INV-6 / G-INLINE).
_INLINE_ALLOW: dict[str, int] = {
    "ui/_bg_toggle_gear.py": 1,
    "ui/_clip_plane_panel.py": 1,
    "ui/_clip_planes_panel.py": 2,
    "ui/_diagram_settings_tab.py": 1,
    # Section-header weight styling (Deformation / Threshold / Scope /
    # Display) — migrate to a shared SectionHeader objectName block.
    "ui/_geometry_settings_panel.py": 4,
    "ui/_measure_panel.py": 1,
    "ui/_parts_tree.py": 2,
    "ui/_selection_tree.py": 1,
    "ui/_session_panel.py": 1,
    "ui/constraints_tab.py": 3,
    "ui/loads_tab.py": 3,
    "ui/mass_tab.py": 3,
    "ui/preferences.py": 2,
    "ui/preferences_dialog.py": 1,
    "ui/theme_editor_dialog.py": 2,
    # The designated sheet writer (_apply_palette) — durable.
    "ui/viewer_window.py": 1,
}

_HEX_RE = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b")
_NAMED_COLORS = frozenset(
    {"white", "black", "red", "gray", "grey", "transparent"},
)
_SHOUT_RE = re.compile(r"^[A-Z][A-Z0-9 /]{3,}$")


def _ui_files() -> list[Path]:
    files = sorted(p for p in UI_DIR.rglob("*.py") if p.is_file())
    assert files, f"No files under {UI_DIR} — path constants are wrong."
    return files


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _string_constants(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            yield n


def _hex_hits(tree: ast.AST) -> list[tuple[int, str]]:
    """Hex literals in ANY string constant, plus named-color strings
    scoped to ``setStyleSheet`` / ``QColor`` call sites (the scoping
    avoids false positives on prose — ADR 0087 G-HEX)."""
    hits: list[tuple[int, str]] = []
    for n in _string_constants(tree):
        for m in _HEX_RE.findall(n.value):
            hits.append((n.lineno, m))
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name = _call_name(n)
        if name == "setStyleSheet":
            for arg in n.args:
                for c in _string_constants(arg):
                    for word in re.findall(r"[A-Za-z]+", c.value):
                        if word.lower() in _NAMED_COLORS:
                            hits.append((c.lineno, word))
        elif name == "QColor":
            for arg in n.args:
                if (isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.lower() in _NAMED_COLORS):
                    hits.append((arg.lineno, arg.value))
    return hits


def _shout_hits(tree: ast.AST) -> list[tuple[int, str]]:
    """All-caps strings passed to ``QLabel(...)`` or ``.setText(...)``
    — the cheapest mechanical proxy for INV-1 (ADR 0087 G-SHOUT)."""
    hits: list[tuple[int, str]] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        if _call_name(n) not in ("QLabel", "setText"):
            continue
        for arg in n.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                v = arg.value.strip()
                if _SHOUT_RE.match(v) and v not in _SHOUT_ACRONYMS:
                    hits.append((arg.lineno, v))
    return hits


def _inline_hits(tree: ast.AST) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "setStyleSheet"):
            hits.append((n.lineno, "setStyleSheet"))
    return hits


def _check(guard: str, allow: dict[str, int], collect, *,
           skip_theme: bool) -> None:
    failures: list[str] = []
    for path in _ui_files():
        if skip_theme and path.name == "theme.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = collect(tree)
        rel = path.relative_to(VIEWERS_DIR).as_posix()
        budget = allow.get(rel, 0)
        if len(hits) > budget:
            detail = ", ".join(f"line {ln}: {what}" for ln, what in hits)
            failures.append(
                f"  {rel}: {len(hits)} violation(s) (allowlisted: {budget})"
                f" — {detail}"
            )
        elif len(hits) < budget:
            failures.append(
                f"  {rel}: allowlist says {budget} but only {len(hits)} "
                f"remain — ratchet the {guard} allowlist down (ADR 0087)."
            )
    if failures:
        raise AssertionError(
            f"{guard} (ADR 0087 D3) violated — style goes through the "
            "Palette roles + objectName QSS blocks in theme.py, never "
            "literals in panel code:\n" + "\n".join(failures)
        )


def test_g_hex_no_literal_colors_outside_theme() -> None:
    _check("G-HEX", _HEX_ALLOW, _hex_hits, skip_theme=True)


def test_g_shout_no_all_caps_labels() -> None:
    _check("G-SHOUT", _SHOUT_ALLOW, _shout_hits, skip_theme=False)


def test_g_inline_stylesheet_call_site_ratchet() -> None:
    _check("G-INLINE", _INLINE_ALLOW, _inline_hits, skip_theme=True)


def test_g_qss_sync_no_dangling_selectors() -> None:
    """Every ``#ObjectName`` selector rendered by ``build_stylesheet``
    must be set by this codebase (ADR 0087 G-QSS-SYNC). One direction
    only — the reverse is unenforceable because objectNames also serve
    layout persistence and tests."""
    from apeGmsh.viewers.ui.theme import PALETTE_DARK, build_stylesheet

    qss = build_stylesheet(PALETTE_DARK)
    # Strip declaration bodies so palette hex values (#1e1e2e, ...)
    # never masquerade as selector tokens.
    selector_text = re.sub(r"\{[^{}]*\}", " ", qss)
    selectors = set(re.findall(r"#([A-Za-z_]\w*)", selector_text))
    assert selectors, "No #ObjectName selectors found — extraction broke?"

    # ``viewers/session/**`` joined the scan with ADR 0098 Amendment 1:
    # the pane host's styled widgets (SessionPaneFrame, SessionPaneHeader,
    # ...) live there, and a guard that cannot see their setObjectName
    # calls would report every one of their QSS blocks as dangling. This
    # WIDENS the guard — nothing was removed from its reach.
    scan_files = (
        list(_ui_files())
        + sorted(p for p in VIEWERS_DIR.glob("*.py") if p.is_file())
        + sorted(
            p for p in (VIEWERS_DIR / "session").rglob("*.py") if p.is_file()
        )
    )
    object_names: set[str] = set()
    for path in scan_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call)
                    and _call_name(n) == "setObjectName"
                    and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)):
                object_names.add(n.args[0].value)

    dangling = sorted(selectors - object_names)
    assert not dangling, (
        "G-QSS-SYNC (ADR 0087 D3): QSS selectors with no matching "
        f"setObjectName in viewers/ui/** or viewers/*.py: {dangling}. "
        "Dead QSS is how the sheet rots — set the objectName or drop "
        "the block."
    )
