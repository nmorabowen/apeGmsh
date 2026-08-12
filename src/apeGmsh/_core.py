from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._session import _SessionBase

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


class apeGmsh(_SessionBase):
    """Standalone single-model Gmsh session with all composites.

    Parameters
    ----------
    model_name : str
        Name passed to ``gmsh.model.add()``.
    verbose : bool
        If True, composites print diagnostic messages.
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
        model_name: str = "ModelName",
        verbose: bool = False,
        save_to: str | Path | None = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__(name=model_name, verbose=verbose)
        # Labels (Tier 1 naming) are auto-created from label= kwargs
        # on geometry methods in both Part and Assembly sessions.
        self._auto_pg_from_label = True
        # Autosave configuration. ``save_to=None`` disables autosave;
        # otherwise ``end()`` writes the neutral-zone HDF5 to this path
        # before finalizing gmsh.  Manual ``g.save()`` uses the same path.
        self._save_to: Path | None = Path(save_to) if save_to else None
        self._overwrite: bool = overwrite
        # ── FEMData cache (Phase 3B.2b-prep / ADR 0038) ──────────
        # The session caches the most recent ``get_fem_data()`` result
        # so repeat calls return the same broker object identity (and
        # downstream consumers — chain-phase shims, future
        # ``g.compose()`` — have a single canonical snapshot to update
        # via ``FEMData.with_*`` transforms).  Every broker mutation
        # (``g.constraints.X`` / ``g.loads.X`` / ``g.masses.X``) bumps
        # ``_fem_counter``; the cached snapshot is fresh iff
        # ``_fem_counter == _fem_counter_at_build``.  The first
        # extraction stamps ``_fem_counter_at_build``; any mutation
        # afterwards invalidates the cache and the next
        # ``get_fem_data()`` re-extracts from gmsh + the def lists.
        self._fem: "FEMData | None" = None
        self._fem_counter: int = 0
        self._fem_counter_at_build: int | None = None
        # ── Compose state (Phase 3B.2c / ADR 0038) ────────────────
        # ``_compose_bundles`` holds every ``_RewrittenBundle`` produced
        # by a ``g.compose(...)`` call on this session in compose-call
        # order.  When a broker mutation invalidates the cache, the
        # next ``get_fem_data()`` re-extracts from gmsh + def lists and
        # then re-applies every stored bundle on top — so the composed
        # modules survive any subsequent ``g.constraints.X`` / etc.
        # mutation.
        #
        # ``_fem_from_h5`` flags sessions built via
        # :meth:`apeGmsh.from_h5`: those have no gmsh state, so the
        # cache-stale path must re-use ``_fem`` as the chain head
        # rather than re-extracting from absent gmsh.  3B.2c chooses
        # this scoped-flag approach rather than generalising
        # ``get_fem_data()`` over a missing-gmsh case because the
        # alternative — making ``from_gmsh`` tolerate absent gmsh —
        # would bleed compose-only concerns into every extraction
        # caller.  3B.2d's resolver refactor takes the cleaner cut.
        self._compose_bundles: tuple = ()
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

        Useful for cross-session composition workflows::

            # Day 1
            with apeGmsh(model_name="host", save_to="host.h5") as g:
                ...

            # Day 2
            g = apeGmsh.from_h5("host.h5")
            g.compose("module_a.h5", label="A")
            g.compose("module_b.h5", label="B")
            g.save("final.h5")

        What works
        ----------
        * ``g.mesh.queries.get_fem_data()`` — the chain head, and the
          surface every refusal below points back at.
        * ``g.compose(...)`` / ``compose_inspect(...)`` /
          ``compose_list()`` and :meth:`save`.
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

        * Geometry — ``g.model.<geometry>`` (via ``Model._register``),
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

        Called by every broker mutation on
        :class:`~.core.ConstraintsComposite.ConstraintsComposite` /
        :class:`~.core.LoadsComposite.LoadsComposite` /
        :class:`~.core.MassesComposite.MassesComposite` after the
        underlying def-list is appended.  Next ``get_fem_data()`` will
        re-extract from gmsh + the updated def lists instead of
        returning the stale cached snapshot.
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

        A directory target (an existing directory, or a path with no
        suffix) means "drop the model file in here" — it is resolved to
        ``<dir>/<model_name>.h5``.  Passing a directory straight to h5py
        truncate-opens it as a file and fails with a cryptic OS-level
        ``PermissionError`` on Windows; this gives both :meth:`save` and
        the :meth:`end` autosave a usable file path instead.
        """
        target = Path(path) if path is not None else self._save_to
        if target.is_dir() or target.suffix == "":
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

    def _do_save(self, path: Path) -> None:
        """Extract the broker snapshot and write it to ``path``.

        Chain-phase sessions (built via :meth:`from_h5`) save the
        cached ``_fem`` directly — they have no gmsh state to
        re-extract from.
        """
        from . import __version__ as _ver

        if (
            getattr(self, "_fem_from_h5", False)
            and getattr(self, "_fem", None) is not None
        ):
            fem = self._fem
        else:
            fem = self.mesh.queries.get_fem_data()
        fem.to_h5(
            str(path),
            model_name=self.name,
            apegmsh_version=_ver,
        )

    # ------------------------------------------------------------------
    # Compose facade — ADR 0038
    # ------------------------------------------------------------------

    def compose(
        self,
        source: "str | Path",
        *,
        label: str,
        **kwargs: Any,
    ) -> "ComposedModule":
        """Merge a previously-saved apeGmsh model into this session.

        See :meth:`apeGmsh.mesh._compose.Compose.compose` for the full
        signature, validation contract, and exception types.  Phase
        3B.1 scaffolds the facade — the merge engine itself lands in
        Phase 3B.2.
        """
        return self._compose_facade().compose(source, label=label, **kwargs)

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
