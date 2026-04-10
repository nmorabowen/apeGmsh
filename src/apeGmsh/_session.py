"""
_SessionBase — shared base for objects that own a Gmsh session.
================================================================

Composite Parent Contract
-------------------------
Composites may access the following on ``self._parent``:

* ``_parent._verbose: bool``   — logging verbosity flag
* ``_parent.name: str``        — session / model name
* ``_parent.is_active: bool``  — property, True when Gmsh session is open
* ``_parent.model``            — the Model composite (Selection uses ``model._registry``)
* ``_parent.physical``         — PhysicalGroups composite (Selection uses ``physical.add()``)

Subclasses MUST define ``_COMPOSITES`` as a class-level tuple of
``(attr_name, relative_module, class_name, is_optional)`` entries.
"""

from __future__ import annotations

import importlib
from typing import ClassVar

import gmsh

from ._optional import MissingOptionalDependency


class _SessionBase:
    """Base class for objects that own a Gmsh session and parent composites."""

    _COMPOSITES: ClassVar[tuple[tuple[str, str, str, bool], ...]] = ()

    def __init__(self, name: str, *, verbose: bool = False) -> None:
        self.name: str = name
        self._verbose: bool = verbose
        self._active: bool = False
        # Pre-declare composite slots as None
        for attr_name, _, _, _ in self._COMPOSITES:
            setattr(self, attr_name, None)

    # ------------------------------------------------------------------
    # Composite parent interface
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True when the wrapped Gmsh session is open."""
        return self._active

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def begin(self, *, verbose: bool | None = None) -> "_SessionBase":
        """Open a Gmsh session, create composites.

        Parameters
        ----------
        verbose : bool or None
            Override the verbosity set in ``__init__``.  ``None`` keeps
            the current value.

        Returns ``self`` for chaining.
        """
        if self._active:
            raise RuntimeError(
                f"{type(self).__name__} '{self.name}' session is already open."
            )
        if verbose is not None:
            self._verbose = verbose
        gmsh.initialize()
        gmsh.model.add(self.name)
        if self._verbose:
            print(f"Gmsh version: {gmsh.__version__}")
        self._create_composites()
        self._active = True
        return self

    def end(self) -> None:
        """Close the Gmsh session."""
        if self._active:
            gmsh.finalize()
            self._active = False

    # Context-manager support
    def __enter__(self) -> "_SessionBase":
        return self.begin()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end()
        return False

    # ------------------------------------------------------------------
    # Composite creation
    # ------------------------------------------------------------------

    def _create_composites(self) -> None:
        """Instantiate each composite declared in ``_COMPOSITES``."""
        for attr_name, module_path, class_name, is_optional in self._COMPOSITES:
            try:
                mod = importlib.import_module(module_path, package=__package__)
                cls = getattr(mod, class_name)
                setattr(self, attr_name, cls(self))
            except ImportError as exc:
                if is_optional:
                    setattr(
                        self,
                        attr_name,
                        MissingOptionalDependency(
                            f"{class_name} support",
                            class_name.lower(),
                            extra=attr_name,
                            cause=exc,
                        ),
                    )
                else:
                    raise

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "active" if self._active else "closed"
        return f"{type(self).__name__}('{self.name}', {status})"
