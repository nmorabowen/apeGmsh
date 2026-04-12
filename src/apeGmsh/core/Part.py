"""
Part — Pure geometry builder.  Meshing lives in the Assembly.
==============================================================

Mirrors the Abaqus Part concept:

* **Part**  =  geometry only (points, curves, surfaces, volumes).
  No meshing, no physical groups, no mesh settings.
* **Assembly**  =  imports Parts, positions them, fragments,
  assigns physical groups, controls mesh, generates mesh,
  extracts FEM data, resolves constraints.

A Part is created in its own isolated Gmsh session, geometry is
built with the full ``apeGmsh`` API, then either saved to **STEP**
explicitly or auto-persisted to an OS tempfile on exit so it can
flow straight into ``assembly.parts.add(part)``.

Typical usage
-------------
::

    from apeGmsh import Part, apeGmsh

    # ── Build a plate (auto-persisted on __exit__) ─────────────
    plate = Part("plate")
    with plate:
        p1 = plate.model.geometry.add_point(0, 0, 0)
        # ... build geometry ...

    # ── Reuse it in an apeGmsh session ────────────────────────
    g = apeGmsh(model_name="bridge")
    g.begin()
    g.parts.add(plate, translate=(0, 0, 0))
    g.parts.add(plate, translate=(5000, 0, 0), label="plate_2")
    g.parts.fragment_all()
    g.mesh.generation.generate(dim=2)
    g.end()

Auto-persist
~~~~~~~~~~~~
If ``save()`` is never called inside the ``with`` block, the Part
auto-persists its geometry to an OS tempfile on exit so it can be
consumed directly by ``assembly.parts.add(part)``.  The tempfile
is deleted when the Part object is garbage-collected or when you
call ``part.cleanup()``.  Call ``save('my.step')`` explicitly
when you want a named, persistent CAD artifact under your control
— **the library will never delete a file you named**.

Set ``auto_persist=False`` on construction to opt out.

Why STEP?
~~~~~~~~~
STEP preserves the full parametric OCC geometry (exact NURBS,
topology, tolerances).  After import, apeGmsh can:

* Apply boolean operations (fragment, fuse, cut)
* Set transfinite meshing / recombine / mesh fields
* Generate and re-generate the mesh with any settings

IGES is also supported for legacy compatibility, but STEP is
preferred (better fidelity, modern spec, avoids IGES surface
junction gaps).  Auto-persist always writes STEP.

Design notes
~~~~~~~~~~~~
* Each Part owns its own Gmsh session.  ``begin()`` initialises
  Gmsh and creates only the composites it needs (Model, Inspect).
  ``end()`` finalises Gmsh — and, when ``auto_persist=True``,
  writes the geometry to an OS tempfile first.
* The Part remembers which file it was last saved to, so the
  Assembly can locate it automatically.
* Parts carry **metadata** in ``properties`` (thickness, material
  name, section type, etc.) that the Assembly passes through to
  the solver model.
* ``save()`` with no arguments writes ``"{name}.step"`` and
  transfers ownership of the output to the caller — if the Part
  had previously auto-persisted a tempfile, that tempfile is
  cleaned up before the explicit save runs.
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
import weakref
from pathlib import Path
from typing import Any, TYPE_CHECKING

import gmsh

from .._session import _SessionBase
from ._part_anchors import collect_anchors, write_sidecar

if TYPE_CHECKING:
    from .Labels import Labels
    from .Model import Model
    from ..mesh.PhysicalGroups import PhysicalGroups
    from ..viz.Inspect import Inspect
    from ..viz.Plot import Plot


class Part(_SessionBase):
    """
    An isolated geometry unit — no meshing, no physical groups.

    Parameters
    ----------
    name : str
        Descriptive name (also used as the Gmsh model name).
    auto_persist : bool, default True
        When True, the Part writes its geometry to an OS tempfile
        on ``end()`` if ``save()`` was not called explicitly.  The
        tempfile is reclaimed via ``weakref.finalize`` when the
        Part is garbage-collected, or eagerly via ``cleanup()``.
        Set to False to opt out — in that case ``parts.add(part)``
        will raise ``FileNotFoundError`` unless you called ``save()``
        by hand.
    """

    # Supported CAD export extensions
    _VALID_EXT = {'.step', '.stp', '.iges', '.igs'}

    # Module paths are rooted at the ``apeGmsh`` top-level package —
    # see :meth:`_SessionBase._create_composites` which uses
    # ``package=__package__`` from ``apeGmsh._session``, so relative
    # imports here must start with ``.core.*`` / ``.viz.*`` rather
    # than being rooted at ``apeGmsh.core``.
    _COMPOSITES = (
        ("model",    ".core.Model",          "Model",          False),
        ("labels",   ".core.Labels",         "Labels",         False),
        ("physical", ".mesh.PhysicalGroups", "PhysicalGroups", False),
        ("inspect",  ".viz.Inspect",         "Inspect",        False),
        ("plot",     ".viz.Plot",            "Plot",            True),
    )

    # -- Static type declarations for composites --
    model: Model
    labels: "Labels"
    physical: "PhysicalGroups"
    inspect: Inspect
    plot: Plot

    def __init__(self, name: str, *, auto_persist: bool = True) -> None:
        super().__init__(name=name, verbose=False)
        self.file_path: Path | None = None       # set by save() or auto-persist
        self.properties: dict[str, Any] = {}     # user metadata
        # When a geometry method is called with ``label="name"``,
        # ``Model._register`` auto-creates a physical group so the
        # label travels through the STEP sidecar into the Assembly.
        self._auto_pg_from_label = True

        # Auto-persist bookkeeping.  ``_owns_file`` is the
        # authorisation bit for deletion — it is True only when we
        # wrote the file ourselves into a temp directory, never
        # when the user called save() with an explicit path.
        self._auto_persist: bool = auto_persist
        self._owns_file: bool = False
        self._temp_dir: Path | None = None
        self._finalizer: weakref.finalize | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin(self, *, verbose: bool | None = None) -> "Part":
        """Open the Part's Gmsh session.

        If the Part is being reused — a previous ``with part:`` block
        auto-persisted a tempfile and this call re-enters — the stale
        tempfile is cleaned up before the new session starts so the
        next ``end()`` can auto-persist fresh geometry.
        """
        if self._owns_file:
            self.cleanup()
            self.file_path = None
        return super().begin(verbose=verbose)  # type: ignore[return-value]

    def end(self) -> None:
        """Close the Part's Gmsh session.

        When ``auto_persist=True`` and the user did not call
        ``save()`` inside the session, the geometry is written to
        an OS tempfile **before** Gmsh is finalised so the Part can
        flow straight into ``assembly.parts.add(part)``.

        Exceptions raised by auto-persist itself are caught and
        emitted as a warning rather than masking any exception the
        user's build code may have raised.  Gmsh finalisation
        always runs.
        """
        try:
            if (
                self._active
                and self._auto_persist
                and self.file_path is None
                and gmsh.model.getEntities()
            ):
                self._auto_persist_to_temp()
        except Exception as exc:
            warnings.warn(
                f"Part {self.name!r}: auto-persist failed ({exc!r}); "
                f"the Part will not be auto-importable via "
                f"parts.add(). Call part.save('...') explicitly to "
                f"recover.",
                stacklevel=2,
            )
        finally:
            super().end()

    def _auto_persist_to_temp(self) -> None:
        """Write the current geometry to an OS tempfile and record
        ownership so ``cleanup()`` can reclaim it later.

        The sidecar (``{name}.step.apegmsh.json``) carrying the
        label->COM anchor map is written as a follow-up step,
        unconditionally — auto-persist always writes anchors when
        available because the user has no other opportunity to do
        so.
        """
        self._temp_dir = Path(
            tempfile.mkdtemp(prefix=f"apeGmsh_part_{self.name}_")
        )
        target = self._temp_dir / f"{self.name}.step"
        self.save(target, _internal_autopersist=True, write_anchors=True)
        self._owns_file = True
        self._register_finalizer()

    def _write_anchors(self, target: Path) -> None:
        """Write the sidecar file for a CAD target if the Part has
        any named physical groups.

        Idempotent and crash-safe: failure to write the sidecar is
        warned, not raised, because a broken sidecar must never
        break the CAD export itself.
        """
        try:
            anchors = collect_anchors(gmsh)
            if not anchors:
                return
            write_sidecar(target, anchors, part_name=self.name)
        except Exception as exc:
            warnings.warn(
                f"Part {self.name!r}: could not write anchor sidecar "
                f"for {target} ({exc!r}). Physical group rebinding "
                f"will not be available for this Part.",
                stacklevel=2,
            )

    def _register_finalizer(self) -> None:
        """Install a ``weakref.finalize`` that removes the temp dir
        when this Part is garbage-collected.

        The finalizer closes over the temp-dir path, not over
        ``self``, so it does not create a reference cycle that
        would keep the Part alive.
        """
        temp_dir = self._temp_dir

        def _cleanup(path: Path | None) -> None:
            if path is not None and path.exists():
                shutil.rmtree(path, ignore_errors=True)

        self._finalizer = weakref.finalize(self, _cleanup, temp_dir)

    def cleanup(self) -> None:
        """Delete any auto-persisted tempfile now, without waiting
        for garbage collection.

        Safe to call multiple times.  Safe to call on a Part whose
        ``file_path`` was set by explicit ``save()`` — the
        ``_owns_file`` guard means the user's file is never
        touched.  After ``cleanup()``, ``has_file`` returns False
        and the Part can be re-built via a new ``with`` block.
        """
        # Snapshot ownership BEFORE resetting it so the
        # post-finalizer file_path reset only runs when we
        # genuinely owned the file.
        was_owned = self._owns_file

        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()
        self._finalizer = None
        self._owns_file = False
        self._temp_dir = None

        if was_owned:
            self.file_path = None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(
        self,
        file_path: str | Path | None = None,
        *,
        fmt: str | None = None,
        write_anchors: bool = True,
        _internal_autopersist: bool = False,
    ) -> Path:
        """
        Export the Part geometry to a CAD file.

        Calling ``save()`` with a user-supplied path **transfers
        ownership of the output file to the caller** — any
        tempfile previously created by auto-persist is cleaned up
        immediately, and the library will never delete the new
        output.

        Parameters
        ----------
        file_path : str, Path, or None
            Destination path.  If ``None``, defaults to
            ``"{name}.step"``.  The extension determines the format
            unless *fmt* overrides it.
        fmt : str, optional
            Force format: ``"step"`` or ``"iges"``.
        write_anchors : bool, default True
            Write a JSON sidecar (``{file_path}.apegmsh.json``)
            carrying the label -> center-of-mass map for every
            user-named entity in the Part.  This is what lets
            ``assembly.parts.add(part)`` expose the instance's
            labels via ``inst.by_label('name')``.  The sidecar is
            silently omitted when the Part has no user-named
            entities, so there is no cost for small throwaway
            Parts.  Pass ``write_anchors=False`` to suppress
            unconditionally — useful when publishing a CAD file
            to third-party tools that shouldn't see apeGmsh
            metadata.

        Returns
        -------
        Path
            Resolved path of the written file.
        """
        if not self._active:
            raise RuntimeError("Part session is not active — call begin() first.")

        # Explicit save by the user: hand off ownership.  The
        # internal auto-persist path sets ``_internal_autopersist``
        # so this branch is skipped — otherwise auto-persist would
        # cleanup() mid-write and zero out the temp directory we're
        # about to create the file in.
        if not _internal_autopersist and self._owns_file:
            self.cleanup()

        # Default: save as STEP using the Part name
        if file_path is None:
            file_path = Path(f"{self.name}.step")

        file_path = Path(file_path)

        # Override extension if fmt is given
        if fmt is not None:
            fmt = fmt.lower().strip(".")
            ext_map = {"step": ".step", "stp": ".step",
                       "iges": ".iges", "igs": ".iges"}
            ext = ext_map.get(fmt)
            if ext is None:
                raise ValueError(f"Unknown format '{fmt}'. Use 'step' or 'iges'.")
            file_path = file_path.with_suffix(ext)

        if file_path.suffix.lower() not in self._VALID_EXT:
            raise ValueError(
                f"Extension '{file_path.suffix}' is not a supported CAD format. "
                f"Use one of {self._VALID_EXT}."
            )

        # Sync OCC kernel before export
        gmsh.model.occ.synchronize()
        gmsh.write(str(file_path))
        self.file_path = file_path.resolve()

        # Write the label->COM sidecar so Assembly.parts.add(part)
        # can expose this Part's user-named entities via
        # ``inst.by_label(...)``.  Failures here are warned, not
        # raised — the CAD write itself already succeeded.
        if write_anchors:
            self._write_anchors(self.file_path)

        return self.file_path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def has_file(self) -> bool:
        """True if the Part has been saved to disk."""
        return self.file_path is not None and self.file_path.exists()

    def __repr__(self) -> str:
        status = "active" if self._active else "closed"
        saved  = f", file={self.file_path.name}" if self.file_path else ""
        owned  = " [temp]" if self._owns_file else ""
        return f"Part('{self.name}', {status}{saved}{owned})"
