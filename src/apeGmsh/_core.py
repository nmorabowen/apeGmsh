from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._session import ArtifactTargetUnavailable, _SessionBase

if TYPE_CHECKING:
    from .viz.Inspect import Inspect
    from .core.Model import Model
    from .core.Labels import Labels
    from .core.ConstraintsComposite import ConstraintsComposite
    from .core.ReinforcementsComposite import ReinforcementsComposite
    from .core.EmbedmentsComposite import EmbedmentsComposite
    from .core.RebarComposite import RebarComposite
    from .core.LoadsComposite import LoadsComposite
    from .core.DisplacementsComposite import DisplacementsComposite
    from .core.MassesComposite import MassesComposite
    from .core.DecoupledNodesComposite import DecoupledNodesComposite
    from .core._parts_registry import PartsRegistry
    from .sections._builder import SectionsBuilder
    from .mesh.Mesh import Mesh
    from .mesh.MshLoader import MshLoader
    from .mesh.PhysicalGroups import PhysicalGroups
    from .mesh.MeshSelectionSet import MeshSelectionSet
    from .mesh.Partition import Partition  # noqa: F401 (backward compat)
    from .mesh.View import View
    from .mesh._compose import Compose, ComposedModule
    from .mesh.FEMData import FEMData
    from .viz.Plot import Plot


#: Environment variable that overrides the conventional artifact directory.
ARTIFACT_DIR_ENV = "APEGMSH_ARTIFACT_DIR"


def default_artifact_dir() -> Path:
    """The directory a session writes its artifacts to when ``save_to`` is None.

    ADR 0112 D1: the artifacts live "at a conventional path next to the
    script".  Resolution order:

    1. ``$APEGMSH_ARTIFACT_DIR`` when set and non-empty (the test suite
       points it at a temporary directory so nothing lands in the repo);
    2. the directory of the ``__main__`` script, when Python is running
       one (``python model.py`` writes beside ``model.py``);
    3. the current working directory (a REPL or a notebook).

    Resolved at call time, not at construction, so a ``chdir`` before
    ``end()`` is honoured the way a relative ``save_to`` would be.
    """
    env = os.environ.get(ARTIFACT_DIR_ENV, "")
    if env:
        return Path(env)
    main = sys.modules.get("__main__")
    try:
        main_file = main.__file__ if main is not None else None
    except AttributeError:  # a REPL or notebook __main__ has no file
        main_file = None
    if main_file:
        return Path(main_file).resolve().parent
    return Path.cwd()


class apeGmsh(_SessionBase):
    """Standalone single-model Gmsh session with all composites.

    Parameters
    ----------
    model_name : str or None
        Name passed to ``gmsh.model.add()`` and the stem of the
        session's ``model.h5`` (``<dir>/<model_name>.h5``).  ``None``
        (the default) takes the stem of the script Python is running;
        with no real script (a notebook, ``-c``, stdin) the session has
        no name and writes nothing automatically (one warning).
    verbose : bool
        If True, composites print diagnostic messages.
    save_to : str, Path or None
        Where ``end()`` writes ``model.h5`` instead of the conventional
        path; a directory means ``<dir>/<model_name>.h5``.
    overwrite : bool
        ``False`` refuses to replace an existing target.
    """

    _COMPOSITES = (
        ("inspect",         ".viz.Inspect",                "Inspect",               False),
        ("model",           ".core.Model",                 "Model",                 False),
        ("labels",          ".core.Labels",                "Labels",                False),
        ("sections",        ".sections._builder",          "SectionsBuilder",       False),
        ("parts",           ".core._parts_registry",       "PartsRegistry",         False),
        ("constraints",     ".core.ConstraintsComposite",  "ConstraintsComposite",  False),
        ("reinforce",       ".core.ReinforcementsComposite", "ReinforcementsComposite", False),
        ("embed",           ".core.EmbedmentsComposite",   "EmbedmentsComposite",   False),
        ("rebar",           ".core.RebarComposite",        "RebarComposite",        False),
        ("loads",           ".core.LoadsComposite",        "LoadsComposite",        False),
        ("displacements",   ".core.DisplacementsComposite", "DisplacementsComposite", False),
        ("masses",          ".core.MassesComposite",       "MassesComposite",       False),
        ("decoupled_nodes", ".core.DecoupledNodesComposite", "DecoupledNodesComposite", False),
        ("mesh",            ".mesh.Mesh",                  "Mesh",                  False),
        ("loader",          ".mesh.MshLoader",             "MshLoader",             False),
        ("physical",        ".mesh.PhysicalGroups",        "PhysicalGroups",        False),
        ("mesh_selection",  ".mesh.MeshSelectionSet",      "MeshSelectionSet",      False),
        # ("partition",    ".mesh.Partition",             "Partition",             False),
        # ^ Removed: consolidated into g.mesh.partitioning
        ("view",            ".mesh.View",                  "View",                  False),
        # ("opensees", ".solvers.OpenSees", "OpenSees", False)
        # ^ Removed in PR γ of the Phase-8 bridge teardown.  The
        #   OpenSees deck is now constructed explicitly via
        #   ``apeGmsh.opensees.apeSees(fem)``, where ``fem`` is the
        #   FEMData snapshot from ``g.mesh.queries.get_fem_data(...)``.
        ("plot",            ".viz.Plot",                   "Plot",                  True),
    )

    # -- Static type declarations for composites (created at runtime by begin()) --
    inspect: Inspect
    model: Model
    labels: Labels
    sections: SectionsBuilder
    parts: PartsRegistry
    constraints: ConstraintsComposite
    reinforce: ReinforcementsComposite
    embed: EmbedmentsComposite
    rebar: RebarComposite
    loads: LoadsComposite
    displacements: DisplacementsComposite
    masses: MassesComposite
    decoupled_nodes: DecoupledNodesComposite
    mesh: Mesh
    loader: MshLoader
    physical: PhysicalGroups
    mesh_selection: MeshSelectionSet
    # partition: Partition  # removed — use g.mesh.partitioning
    view: View
    # opensees: removed in PR γ — use apeGmsh.opensees.apeSees(fem)
    plot: Plot

    def __init__(
        self,
        *,
        model_name: str | None = None,
        verbose: bool = False,
        save_to: str | Path | None = None,
        overwrite: bool = True,
        _artifacts: bool = True,
    ) -> None:
        # ADR 0112 D1: the default name is the ``__main__`` script's
        # stem, so ``python frame.py`` leaves ``frame.h5`` beside it.
        # With no real script (a notebook, ``-c``, stdin, a launcher) the
        # session has no name: ``end()`` writes nothing automatically and
        # warns once, and the snapshot's ``model_name`` is ``""``.  An
        # explicit ``model_name`` always wins; an empty one is refused.
        if model_name is None:
            from ._artifact_policy import main_script

            script = main_script()
            name = script.stem if script is not None else ""
        else:
            name = str(model_name)
            if not name:
                raise ValueError(
                    "apeGmsh(model_name=''): the model name must be a "
                    "non-empty string; leave it out to take the script's stem"
                )
        super().__init__(name=name, verbose=verbose)
        # ADR 0112 D1: this session *is* the model; ``end()`` writes
        # ``model.h5`` and its geometry sibling unconditionally.  The
        # private ``_artifacts=False`` is for library-internal sessions
        # only (section mesh workers, solver cross-checks, the demo
        # builder); it is not a user-facing opt-out.
        self._writes_artifacts = bool(_artifacts)
        # Labels (Tier 1 naming) are auto-created from label= kwargs
        # on geometry methods in both Part and Assembly sessions.
        self._auto_pg_from_label = True
        # Artifact path override (ADR 0112 D1).  ``end()`` always writes
        # the neutral-zone HDF5 and its geometry sibling before
        # finalizing gmsh: at ``save_to`` when given, else at the
        # conventional path ``default_artifact_dir() / <model_name>.h5``
        # (see :meth:`_resolve_save_target`).  Manual ``g.save()`` with
        # no argument still requires ``save_to``.
        self._save_to: Path | None = Path(save_to) if save_to else None
        self._overwrite: bool = overwrite
        # ── FEMData cache (Phase 3B.2b-prep / ADR 0038) ──────────
        # The session caches the most recent ``get_fem_data()`` result
        # so repeat calls return the same broker object identity (and
        # downstream consumers — the chain-phase shims — have a
        # single canonical snapshot to update
        # via ``FEMData.with_*`` transforms).  Every declaration
        # (``_DeclarationsMixin._declare`` — ``g.constraints.X``,
        # ``g.loads.X``, ``g.reinforce`` and the rest) bumps
        # ``_fem_counter``; the cached snapshot is fresh iff
        # ``_fem_counter == _fem_counter_at_build``.  The first
        # extraction stamps ``_fem_counter_at_build``; any mutation
        # afterwards invalidates the cache and the next
        # ``get_fem_data()`` re-extracts from gmsh + the def lists.
        self._fem: "FEMData | None" = None
        self._fem_counter: int = 0
        self._fem_counter_at_build: int | None = None
        # ``_fem_from_h5`` flags sessions built via
        # :meth:`apeGmsh.from_h5`: those have no gmsh state, so the
        # cache-stale path must re-use ``_fem`` as the chain head
        # rather than re-extracting from absent gmsh.
        self._fem_from_h5: bool = False

    # ------------------------------------------------------------------
    # Chain-phase constructor — Phase 3B.2c / ADR 0038
    # ------------------------------------------------------------------

    @classmethod
    def from_h5(
        cls,
        path: "str | Path",
        *,
        model_name: str | None = None,
        verbose: bool = False,
    ) -> "apeGmsh":
        """Construct a session in chain phase directly from a saved FEMData.

        Skips the gmsh build phase entirely: the loaded FEMData becomes
        the session's chain head and **there is no gmsh kernel behind
        this session at all**.  ``model.h5`` persists the FEMData
        snapshot (nodes, elements, physical groups, labels) — not the
        geometry kernel — so anything that would read or mutate BRep /
        mesh state raises :class:`~.core._compose_errors.ChainPhaseError`
        naming the H5-safe alternative.

        Composition is :class:`apeGmsh.assembly.Assembly` (ADR 0117):
        instance saved files, tie them, ``bridge()``; ``from_h5`` then
        opens the assembly archive like any ``model.h5``.

        What works
        ----------
        * ``g.mesh.queries.get_fem_data()`` — the chain head, and the
          surface every refusal below points back at.
        * The compose readers ``compose_inspect(...)`` /
          ``compose_list()`` / ``compose_tree()``, and :meth:`save`.
        * The chain-phase authoring shims, routed through
          ``FEMData.with_*``: ``g.constraints.bc`` / ``tie`` /
          ``embedded`` / ``tied_contact`` / ``equalDOF`` /
          ``rigid_link`` / ``rigid_diaphragm``, plus point
          ``g.loads.X`` / ``g.masses.X``.
        * Kernel-free helpers: ``g.model.queries.plane`` /
          ``registry``, ``g.view.list_views`` / ``count``,
          ``g.plot.show`` / ``savefig`` / ``clear`` / ``figsize`` /
          ``use_axes``.
        * ``repr()`` of any composite.  The two kernel-backed reprs
          (``g.physical``, ``g.labels``) report
          ``"no live gmsh kernel — from_h5 session"`` rather than
          raising, so debuggers and logging stay usable.

        Refused — no live kernel to read
        --------------------------------
        These need the gmsh model and raise on a ``from_h5`` session
        specifically (a live session still has a kernel, so they stay
        legal there).  Each message names the broker counterpart.

        =========================  ==================================
        Surface                    Guarded members
        =========================  ==================================
        ``g.inspect``              ``get_geometry_info``,
                                   ``get_mesh_info``, ``print_summary``
        ``g.physical``             ``get_all``, ``get_entities``,
                                   ``entities``,
                                   ``get_groups_for_entity``,
                                   ``get_name``, ``get_tag``,
                                   ``summary``, ``get_nodes``
        ``g.labels``               ``entities``, ``get_all``,
                                   ``summary``, ``has``,
                                   ``reverse_map``,
                                   ``labels_for_entity``
        ``g.mesh.queries``         ``get_nodes``, ``get_elements``,
                                   ``get_element_properties``,
                                   ``get_element_qualities``,
                                   ``quality_report``
        ``g.model.queries``        ``bounding_box``,
                                   ``center_of_mass``, ``mass``,
                                   ``boundary``, ``boundary_curves``,
                                   ``boundary_points``,
                                   ``adjacencies``,
                                   ``entities_in_bounding_box``
        ``g.mesh.partitioning``    ``n_partitions``, ``summary``,
                                   ``entity_table``, ``save``
        ``g.model.io``             ``save_step``, ``save_iges``,
                                   ``save_dxf``, ``save_msh`` — the
                                   exporters only; the importers are
                                   frozen instead (below)
        ``g.model.<geometry>``     ``find_stale_metadata``, and
                                   ``validate_pre_mesh`` through it
        ``g.mesh.recipe``          ``check``
        ``g.parts``                ``build_face_map``
        ``g.rebar``                ``resolve``
        ``g.sections``             ``plot_faces``
        ``g.view``                 ``add_element_scalar`` /
                                   ``add_element_vector`` /
                                   ``add_node_scalar`` /
                                   ``add_node_vector``
        ``g.plot``                 ``geometry``, ``mesh``, ``quality``,
                                   ``label_entities``, ``label_nodes``,
                                   ``label_elements``,
                                   ``physical_groups``,
                                   ``physical_groups_mesh``
        =========================  ==================================

        Counterparts: ``fem.inspect`` for summaries, ``fem.physical``
        (:class:`~.mesh._group_set.PhysicalGroupSet`) for physical
        groups, ``fem.nodes.labels`` / ``fem.elements.labels``
        (:class:`~.mesh._group_set.LabelSet`) for labels,
        ``fem.nodes`` / ``fem.elements`` / ``fem.info`` for mesh data,
        and ``results.inspect`` for post-processing — where
        ``fem = g.mesh.queries.get_fem_data()``.  BRep geometry has no
        counterpart: derive it from mesh coordinates or rebuild the
        geometry in a live session.

        Refused — model frozen
        ----------------------
        Mutations are refused on **any** chain-phase session, not just
        this one: once a FEMData snapshot exists the broker is
        canonical, and mutating gmsh would silently desync the two.
        Listed by composite — each guards its mutating operations at a
        shared chokepoint, so the coverage is per-composite rather than
        the per-method enumeration given for the reads above.

        * Geometry — ``g.model.<geometry>`` (via ``Model._register``,
          plus ``add_wire``, which creates OCC geometry but is
          deliberately not registered),
          ``g.model.boolean``, ``g.model.transforms``,
          ``g.model.io.heal_shapes`` / ``load_msh`` / ``load_geo``,
          and ``g.model.queries.remove`` / ``remove_duplicates`` /
          ``make_conformal`` (mutations despite the composite name).
        * Mesh — ``g.mesh.generation``, ``g.mesh.editing``,
          ``g.mesh.sizing``, ``g.mesh.structured``, ``g.mesh.recipe``,
          and ``g.mesh.partitioning`` (its mutating ops ``partition`` /
          ``partition_explicit`` / ``unpartition`` / ``renumber``; the
          composite's four readers take the kernel guard instead, and
          are listed in the read table above).
        * Naming — ``g.physical.add`` / ``set_name`` / ``remove`` /
          ``remove_name`` / ``remove_all``, and ``g.labels.add`` /
          ``remove`` / ``rename`` / ``promote_to_physical``.
        * Assembly — ``g.parts`` instance registration,
          ``g.sections`` builds, ``g.rebar.place``.

        Refused — resolves from live geometry
        -------------------------------------
        ``g.constraints.contact`` / ``contact_plane`` / ``interface``,
        ``g.embed``, ``g.reinforce`` and ``g.decouple_node`` record
        definitions that are resolved against live gmsh at extraction.
        A ``from_h5`` session never re-extracts, so the definition
        would be stored and silently never applied — declare these in
        the source part session before saving; the resolved records
        round-trip through ``model.h5`` and survive ``g.compose``.

        Parameters
        ----------
        path : str or Path
            Path to a ``model.h5`` written by :meth:`save` /
            :meth:`FEMData.to_h5`.
        model_name : str or None
            Session name (used by :meth:`save` for ``/meta/model_name``).
            Defaults to the source file's stem.
        verbose : bool, default False
            Verbose-mode flag forwarded to the constructor.

        Raises
        ------
        ~.core._compose_errors.ChainPhaseError
            From any surface listed above.  The message names the
            offending call and the alternative that answers it.
        """
        from .mesh.FEMData import FEMData

        p = Path(path)
        loaded_fem = FEMData.from_h5(str(p))
        name = model_name if model_name is not None else p.stem
        instance = cls(model_name=name, verbose=verbose)
        instance._fem = loaded_fem
        instance._fem_from_h5 = True
        # Mark the cache fresh so the first ``get_fem_data()`` returns
        # the loaded chain head without an extraction attempt.
        instance._mark_fem_fresh()
        # Instantiate the session composites so chain-phase APIs that
        # touch ``g.mesh.queries.get_fem_data()`` / ``g.compose`` /
        # ``g.save`` work without ``begin()`` ever running.  No gmsh
        # state is created here — composite constructors only require
        # the parent session.  Every gmsh-backed sub-API is guarded
        # (kernel reads via ``raise_if_no_live_kernel``, mutations via
        # the chain-phase freeze guard); the docstring above lists the
        # surfaces and their H5-safe counterparts.
        instance._create_composites()
        return instance

    # ------------------------------------------------------------------
    # FEMData cache + dirty-bit (Phase 3B.2b-prep / ADR 0038)
    # ------------------------------------------------------------------

    def _bump_fem_counter(self) -> None:
        """Mark the FEMData cache dirty.

        Called by
        :class:`~.core._declarations._DeclarationsMixin` after every
        declaration (``_declare``) and every ``clear()``
        (``_clear_declarations``) on the composites that record
        intent.  Next ``get_fem_data()`` will re-extract from gmsh +
        the updated def lists instead of returning the stale cached
        snapshot.
        """
        self._fem_counter += 1

    def _fem_is_fresh(self) -> bool:
        """True iff the cached :attr:`_fem` matches the current counter.

        Used by ``g.mesh.queries.get_fem_data`` to decide between
        returning the cached snapshot (identity-stable across repeat
        calls) and re-extracting from gmsh.
        """
        return (
            self._fem is not None
            and self._fem_counter_at_build is not None
            and self._fem_counter == self._fem_counter_at_build
        )

    def _mark_fem_fresh(self) -> None:
        """Snapshot the counter so :meth:`_fem_is_fresh` reports True
        until the next mutation.

        Called by ``g.mesh.queries.get_fem_data`` immediately after a
        successful extraction populates :attr:`_fem`.
        """
        self._fem_counter_at_build = self._fem_counter

    # ------------------------------------------------------------------
    # Decoupled nodes (ADR 0049)
    # ------------------------------------------------------------------

    def decouple_node(
        self,
        *,
        coords: "tuple[float, float, float] | None" = None,
        point: "str | None" = None,
        label: "str | None" = None,
    ) -> Any:
        """Declare a decoupled node — an auxiliary node that is **not**
        a Gmsh mesh vertex (spring/dashpot ground, ``rigidDiaphragm``
        master, control node, load/mass anchor).

        Exactly one of ``coords=(x, y, z)`` or ``point="label"`` locates
        it; ``point=`` is snapshotted to coordinates at mesh-extraction
        time.  ``label`` is an optional friendly name.

        The node is appended to ``fem.nodes`` at extraction with a
        deterministic tag above every mesh node (dedup-immune by
        construction) and ``provenance == "decoupled"``.  It carries
        **no** ``ndf`` — DOF count is a bridge concern (``ops.ndf``).

        Returns the :class:`~apeGmsh._kernel.defs.decoupled.DecoupledNodeDef`
        handle; its ``tag`` is populated after
        ``g.mesh.queries.get_fem_data(...)``.
        """
        return self.decoupled_nodes.add(
            coords=coords, point=point, label=label,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _resolve_save_target(self, path: "str | Path | None") -> Path:
        """Normalize a save destination to a concrete ``.h5`` file path.

        ``path`` wins; else ``save_to``; else (ADR 0112 D1, the
        unconditional write) the conventional path
        ``default_artifact_dir() / <model_name>.h5``.

        A directory target (an existing directory, or a path with no
        suffix) means "drop the model file in here" — it is resolved to
        ``<dir>/<model_name>.h5``.  Passing a directory straight to h5py
        truncate-opens it as a file and fails with a cryptic OS-level
        ``PermissionError`` on Windows; this gives both :meth:`save` and
        the :meth:`end` autosave a usable file path instead.

        Raises :class:`~apeGmsh._session.ArtifactTargetUnavailable` when
        the path needs the session's name and it has none (P2, #1307: no
        ``model_name`` and no script file); ``end()`` turns that into one
        warning and writes nothing.
        """
        if path is not None:
            target = Path(path)
        elif self._save_to is not None:
            target = self._save_to
        else:
            target = default_artifact_dir()
        if target.is_dir() or target.suffix == "":
            if not self.name:
                raise ArtifactTargetUnavailable(
                    f"no model name: the session has no model_name and "
                    f"Python is not running a script file (a notebook, -c, "
                    f"stdin, or a console-script launcher such as pytest or "
                    f"jupyter), so there is no conventional model.h5 path "
                    f"under {target}; nothing is written automatically. "
                    f"Pass model_name= or save_to=<file>."
                )
            target = target / f"{self.name}.h5"
        return target

    def save(self, path: str | Path | None = None) -> Path:
        """Write the neutral-zone ``model.h5`` for this session.

        Persists what the session knows about the model: nodes,
        elements, physical groups, labels, constraints, loads, masses.
        Downstream solver enrichment (e.g. ``apeSees(fem).h5(p)``) is
        a separate user-driven action and not invoked here.

        Parameters
        ----------
        path : str, Path, or None
            Destination file.  ``None`` (default) uses the ``save_to``
            given to the constructor.  Raises if neither is set.

        Returns the resolved path.
        """
        if path is None and self._save_to is None:
            raise RuntimeError(
                "g.save() requires a path — either pass one explicitly "
                "or construct the session with save_to=<path>."
            )
        target = self._resolve_save_target(path)
        if target.exists() and not self._overwrite:
            raise FileExistsError(
                f"{target} already exists and overwrite=False."
            )
        self._do_save(target)
        return target

    def _snapshot_to_save(self) -> "FEMData":
        """The broker snapshot a save writes.

        Chain-phase sessions (built via :meth:`from_h5`) save the
        cached ``_fem`` directly — they have no gmsh state to
        re-extract from.
        """
        if (
            getattr(self, "_fem_from_h5", False)
            and getattr(self, "_fem", None) is not None
        ):
            return self._fem
        return self.mesh.queries.get_fem_data()

    def _do_save(self, path: Path, fem: "FEMData | None" = None) -> None:
        """Write the broker snapshot (``fem``, else :meth:`_snapshot_to_save`)
        to ``path``."""
        from . import __version__ as _ver

        if fem is None:
            fem = self._snapshot_to_save()
        fem.to_h5(
            str(path),
            model_name=self.name,
            apegmsh_version=_ver,
        )

    # ------------------------------------------------------------------
    # Compose facade — ADR 0038
    # ------------------------------------------------------------------

    def compose_inspect(self, path: "str | Path") -> dict:
        """Read a module's H5 header without composing it.

        See :meth:`apeGmsh.mesh._compose.Compose.compose_inspect` for
        the returned dict shape.
        """
        return self._compose_facade().compose_inspect(path)

    def compose_list(self) -> "tuple[ComposedModule, ...]":
        """Composed modules currently on this session.

        See :meth:`apeGmsh.mesh._compose.Compose.compose_list`.
        """
        return self._compose_facade().compose_list()

    def compose_tree(self) -> "tuple":
        """Derived nested-compose tree view of this session's modules.

        See :meth:`apeGmsh.mesh._compose.Compose.compose_tree`.
        """
        return self._compose_facade().compose_tree()

    def _compose_facade(self) -> "Compose":
        """Lazy-instantiate the single per-session :class:`Compose` facade.

        Compose is a session-level facade rather than a ``_COMPOSITES``
        entry so the three public methods (``compose`` /
        ``compose_inspect`` / ``compose_list``) read naturally on the
        session.  The lazy pattern keeps unused sessions free of the
        facade's import cost.
        """
        cached: "Compose | None" = getattr(self, "_compose", None)
        if cached is not None:
            return cached
        from .mesh._compose import Compose
        facade = Compose(self)
        self._compose = facade
        return facade
