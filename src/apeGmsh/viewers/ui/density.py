"""Viewer density — compact / comfortable token sets (B++ §5).

Density affects type size, row height, and padding only — it does *not*
change the grid template (260 / 1fr / 380 stays fixed). Per the spec
mock (``RV_DENSITY`` in ``tokens.js``):

* compact     — rowH 22, padX 8,  padY 4, gap 4, fs 11.5, fsHd 11
* comfortable — rowH 28, padX 12, padY 6, gap 6, fs 12.5, fsHd 12

A singleton :data:`DENSITY` mirrors :data:`THEME`'s observable pattern.
``ResultsWindow`` rebuilds the QSS through ``build_stylesheet`` whenever
the density changes; observers are called with the new
:class:`DensityTokens`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


DensityName = Literal["compact", "comfortable"]


@dataclass(frozen=True)
class DensityTokens:
    name: DensityName
    row_h: int       # base row height for rows / list items
    pad_x: int       # horizontal padding inside cards / rows
    pad_y: int       # vertical padding
    gap: int         # spacing between siblings
    fs_body: float   # primary text size
    fs_head: float   # section header / label text size


DENSITY_COMPACT = DensityTokens(
    name="compact",
    row_h=22, pad_x=8, pad_y=4, gap=4,
    fs_body=11.5, fs_head=11.0,
)
DENSITY_COMFORTABLE = DensityTokens(
    name="comfortable",
    row_h=28, pad_x=12, pad_y=6, gap=6,
    fs_body=12.5, fs_head=12.0,
)
DENSITIES: dict[str, DensityTokens] = {
    "compact": DENSITY_COMPACT,
    "comfortable": DENSITY_COMFORTABLE,
}


# ======================================================================
# Manager (observable singleton)
# ======================================================================

class DensityManager:
    """Global current density + observer list.

    Observers receive the new :class:`DensityTokens` whenever
    :meth:`set_density` flips the active value.
    """

    _settings_org = "apeGmsh"
    _settings_app = "viewer"
    _settings_key = "density"

    def __init__(self) -> None:
        self._current: DensityTokens = (
            self._load_saved() or DENSITY_COMFORTABLE
        )
        self._observers: list[Callable[[DensityTokens], None]] = []

    @property
    def current(self) -> DensityTokens:
        return self._current

    def set_density(self, name: str) -> None:
        key = name.lower()
        if key not in DENSITIES:
            raise ValueError(f"Unknown density: {name!r}")
        new = DENSITIES[key]
        if new is self._current:
            return
        self._current = new
        self._save(new)
        for cb in list(self._observers):
            try:
                cb(new)
            except Exception:
                import logging
                logging.getLogger("apeGmsh.viewer.density").exception(
                    "density observer failed: %r", cb,
                )

    def toggle(self) -> None:
        """Flip between compact and comfortable."""
        self.set_density(
            "compact" if self._current.name == "comfortable"
            else "comfortable"
        )

    def subscribe(
        self, cb: Callable[[DensityTokens], None],
    ) -> Callable[[], None]:
        """Register observer. Returns an unsubscribe callable."""
        self._observers.append(cb)

        def _unsub() -> None:
            try:
                self._observers.remove(cb)
            except ValueError:
                pass
        return _unsub

    @classmethod
    def _load_saved(cls) -> "DensityTokens | None":
        try:
            from qtpy.QtCore import QSettings
        except Exception:
            return None
        try:
            s = QSettings(cls._settings_org, cls._settings_app)
            name = s.value(cls._settings_key, "comfortable")
            return DENSITIES.get(str(name).lower())
        except Exception:
            return None

    @classmethod
    def _save(cls, density: DensityTokens) -> None:
        try:
            from qtpy.QtCore import QSettings
        except Exception:
            return
        try:
            QSettings(cls._settings_org, cls._settings_app).setValue(
                cls._settings_key, density.name,
            )
        except Exception:
            pass


DENSITY = DensityManager()
