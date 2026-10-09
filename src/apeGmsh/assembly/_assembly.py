"""``Assembly`` v2 — instances of model files, tied by label (ADR 0117).

::

    from apeGmsh.assembly import Assembly

    asm = Assembly("stack")
    asm.instance("pier_1", "pier.h5")
    asm.instance("pier_2", "pier.h5", translate=(0.0, 0.0, 10.0))
    asm.tie("pier_1.top", "pier_2.bot", enforce="equation", dofs=[1, 2, 3])
    ops = asm.bridge(ndm=3, ndf=3)       # one forward apeSees
    ops.fix(pg="pier_1.bot", dofs=(1, 1, 1))
    ...

Every instance is namespaced; there is no host. An instance's model
content (mesh, groups, materials, sections, element specs) travels with it
under ``{instance}.``; analysis content (fixes, masses, patterns, recorders,
stages, analysis) is declared on the returned bridge (ADR 0117 D4).

``instance`` and ``tie`` only record, after validating; ``bridge`` merges the
FEM side through the existing compose engine (``FEMData.compose``), resolves
each tie in chain phase, and rehydrates each instance's ``/opensees`` zone.
No tag is allocated here: the bridge plans every tag at build (ADR 0114 D4).

``h5`` writes the bridge's ``model.h5`` plus the ``/assembly`` zone (ADR 0117
D5, ``_h5.py``); ``from_h5`` re-lists an archive's instances and ties.

The v1 declarations (``add`` / ``couple`` / ``materialize``) are inherited
unchanged until AS5 deletes them; one assembly uses one API or the other
(the v1 build of a v2 assembly finds no parts and raises).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from apeGmsh._internal.provenance import ProvenanceStore

from ._h5 import (
    TIE_PARAMS,
    InstanceRow,
    TieRow,
    read_assembly_zone,
    validate_rows,
    write_assembly_zone,
)
from ._couplings import (
    NODE_PORTS,
    canonical_params,
    coupling_definition,
)
from ._instances import (
    Coupling,
    Instance,
    RefNode,
    Tie,
    check_dense_ranks,
    check_label,
    check_partition_rank,
    check_point,
    check_rotate,
    check_translate,
    check_unranked_source,
    merged_port,
    split_port,
)
from ._v1 import Assembly as _AssemblyV1
from ._v1 import AssemblyError

if TYPE_CHECKING:
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.opensees_model import OpenSeesModel

__all__ = ["Assembly"]


@dataclass(frozen=True)
class _Bridged:
    """What ``bridge()`` built, kept for ``h5()``."""

    ops: "apeSees"
    fem: "FEMData"
    #: The declarations the bridge was built from; ``h5()`` refuses if
    #: they changed since.
    instances: tuple[Instance, ...]
    nodes: tuple[RefNode, ...]
    ties: tuple["Tie | Coupling", ...]
    #: ``model_hash`` of each instance's source, by instance label.
    opensees_hash: dict[str, str]


class Assembly(_AssemblyV1):
    """Instances of saved ``model.h5`` files joined by assembly-level ties.

    See the module docstring. ``name`` is the assembly's own name.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._instances: list[Instance] = []
        self._nodes: list[RefNode] = []
        self._ties: "list[Tie | Coupling]" = []
        # ADR 0117 D5: records ``assembly/instances/<label>`` and
        # ``assembly/ties/<name|#k>`` at the user's declaring line.
        self._provenance = ProvenanceStore()
        self._bridged: "_Bridged | None" = None
        #: The archive ``from_h5`` read, or ``None`` for a declared assembly.
        self._archive: "Path | None" = None

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def instances(self) -> tuple[Instance, ...]:
        """Declared instances, in declaration order."""
        return tuple(self._instances)

    @property
    def nodes(self) -> tuple[RefNode, ...]:
        """Declared assembly-owned reference nodes, in declaration order."""
        return tuple(self._nodes)

    @property
    def ties(self) -> "tuple[Tie | Coupling, ...]":
        """Declared assembly-level ties and couplings, in declaration order."""
        return tuple(self._ties)

    # ------------------------------------------------------------------
    # v2 declarations
    # ------------------------------------------------------------------

    def instance(
        self,
        label: str,
        source: "str | Path",
        *,
        translate: Sequence[float] = (0.0, 0.0, 0.0),
        rotate: "tuple[Sequence[float], float] | None" = None,
        partition_rank: "int | None" = None,
    ) -> "Assembly":
        """Place a saved ``model.h5`` under ``label``. Returns ``self``.

        Every name the file owns becomes ``{label}.{name}``. ``rotate`` is
        ``((ax, ay, az), theta_radians)`` about the origin, applied before
        ``translate``. The same file may be instanced any number of times.

        ``partition_rank`` is the OpenSeesMP rank (``getPID``) that owns the
        instance (ADR 0038 Layer 2). With no rank anywhere the assembly is
        serial and ``tcl()`` writes an unpartitioned deck. With ranks, every
        instance carries one, one per rank, and ``bridge()`` requires them
        to run ``0 .. n-1``; the reference nodes (``node()``) live on
        rank 0 and are ghosted where a coupling needs them.

        Raises :class:`AssemblyError`, before recording anything, for a
        label that is empty, contains ``.``, ``/`` or whitespace, starts or
        ends with ``_``, or is already declared; for a missing file or one
        whose composed modules carry a partition rank (a ranked assembly
        archive); for a zero rotation axis; and for a ``partition_rank``
        that is not an ``int >= 0``, that another instance holds, or that
        is given on one instance and not another.
        """
        self._refuse_mixed("instance")
        self._check_new_name(label, what="instance label")
        path = Path(source)
        if not path.is_file():
            raise AssemblyError(f"instance {label!r}: no file at {str(path)!r}.")
        check_unranked_source(path, label)
        placed = Instance(
            label=label,
            source=path,
            translate=check_translate(translate),
            rotate=check_rotate(rotate),
            partition_rank=check_partition_rank(
                partition_rank, label, self._instances),
        )
        # Every check above runs first: a refused call records nothing.
        self._provenance.capture(
            "assembly", "instances", label, on_existing="raise")
        self._instances.append(placed)
        return self

    def tie(
        self,
        master: str,
        slave: str,
        *,
        enforce: str = "penalty",
        method: str = "collocation",
        dofs: "Sequence[int] | None" = None,
        tolerance: float = 1.0,
        name: "str | None" = None,
    ) -> "Assembly":
        """Tie two instance ports, ``"{instance}.{pg|label}"``. Returns ``self``.

        Resolved by ``bridge()`` exactly as ``g.constraints.tie`` resolves
        in a chain-phase session (ADR 0068, 0086): ``master`` is the master
        surface, ``slave`` the projected side. ``enforce="equation"`` is
        exact and needs the Lagrange handler and an unsymmetric system.

        Raises :class:`AssemblyError` for a port that names no declared
        instance (a bare port names an assembly object, and this assembly
        declares none), a ``name`` that contains ``.`` or repeats, or tie
        options ``TieDef`` refuses (an unknown ``enforce``, ``method="mortar"``
        without ``enforce="equation"``).
        """
        self._refuse_mixed("tie")
        labels = [i.label for i in self._instances]
        m_name = merged_port(master, labels)
        s_name = merged_port(slave, labels)
        if name is not None:
            self._check_new_name(name, what="tie name")
        from apeGmsh._kernel.defs.constraints import TieDef

        dofs_t = tuple(int(d) for d in dofs) if dofs is not None else None
        try:
            definition = TieDef(
                master_label=m_name, slave_label=s_name,
                dofs=list(dofs_t) if dofs_t is not None else None,
                tolerance=float(tolerance), enforce=enforce, method=method,
                name=name,
            )
        except ValueError as exc:
            raise AssemblyError(f"tie({master!r}, {slave!r}): {exc}") from exc
        self._provenance.capture("assembly", "ties", name, on_existing="raise")
        self._ties.append(Tie(
            master=master,
            slave=slave,
            enforce=enforce,
            method=method,
            dofs=dofs_t,
            tolerance=float(tolerance),
            name=name,
            definition=definition,
        ))
        return self

    def node(self, name: str, coords: Sequence[float]) -> "Assembly":
        """Declare an assembly-owned reference node. Returns ``self``.

        ``name`` is a bare name (no ``.``) in the assembly's own namespace,
        shared with instance labels and tie names. ``bridge()`` adds the
        node to the merged FEM as an element-less decoupled node labelled
        ``name`` (``fem.nodes.select(label=name)``), at FEM id ``k`` for
        the ``k``-th declared node; its ndf is the bridge's ``ndf`` unless
        ``ops.ndf(tag, ndf=K)`` states it. A node is a port of
        ``equal_dof``, ``rigid_link`` and ``rigid_diaphragm``, and the
        ``reference`` of ``couple``.

        Raises :class:`AssemblyError` for a name that is invalid or already
        declared (an instance, node, tie or coupling), and for non-finite
        coordinates.
        """
        self._refuse_mixed("node")
        self._check_new_name(name, what="node name")
        xyz = check_point(coords, what="coords")
        self._provenance.capture("assembly", "ties", name, on_existing="raise")
        self._nodes.append(RefNode(name=name, coords=xyz))
        return self

    def equal_dof(
        self,
        master: str,
        slave: str,
        *,
        dofs: "Sequence[int] | None" = None,
        tolerance: float = 1e-6,
        name: "str | None" = None,
    ) -> "Assembly":
        """Tie the co-located nodes of two ports with ``equalDOF``. Returns ``self``.

        Each port is ``"{instance}.{pg|label}"`` or a reference node. Every
        slave node within ``tolerance`` of a master node shares ``dofs``
        (``None``: DOFs 1-6, which the bridge refuses on a node with
        fewer) with it. Exact on matching meshes; the non-matching case is
        ``tie``.

        Raises :class:`AssemblyError` here for a bad port, name or option,
        and from ``bridge()`` when no pair is co-located (ADR 0117 INV-7).
        """
        return self._couple_ports(
            "equal_dof", master, slave, name,
            {"dofs": _list_or_none(dofs), "tolerance": tolerance})

    def rigid_link(
        self,
        master: str,
        slave: str,
        *,
        link_type: str = "beam",
        master_point: "Sequence[float] | None" = None,
        name: "str | None" = None,
    ) -> "Assembly":
        """Link every slave node rigidly to one master node. Returns ``self``.

        The master is the ``master`` node nearest ``master_point`` (default:
        the master set's centroid), so a reference node is the natural
        master. ``link_type="beam"`` ties translations and rotations
        (``rigidLink beam``), ``"rod"`` translations only.

        Raises :class:`AssemblyError` here for a bad port, name or option,
        and from ``bridge()`` when the slave set holds no node but the
        master (ADR 0117 INV-7).
        """
        return self._couple_ports(
            "rigid_link", master, slave, name,
            {"link_type": link_type,
             "master_point": _point_or_none(master_point, "master_point")})

    def rigid_diaphragm(
        self,
        master: str,
        slave: str,
        *,
        master_point: "Sequence[float] | None" = None,
        plane_normal: Sequence[float] = (0.0, 0.0, 1.0),
        constrained_dofs: Sequence[int] = (1, 2, 6),
        plane_tolerance: float = 1.0,
        name: "str | None" = None,
    ) -> "Assembly":
        """Make the in-plane motion of two ports one rigid body. Returns ``self``.

        Every node of ``master`` and ``slave`` within ``plane_tolerance`` of
        the plane through ``master_point`` with normal ``plane_normal``
        follows the master (the plane node nearest ``master_point``) in
        that plane (``rigidDiaphragm``; the master needs ndf 6 in 3-D).
        ``master_point`` defaults to the coordinates of ``master`` when it
        is a reference node and is required otherwise.

        Raises :class:`AssemblyError` here for a bad port, name or option,
        and from ``bridge()`` when no node lies in the plane (INV-7).
        """
        if master_point is None:
            self._refuse_mixed("rigid_diaphragm")
            split_port(master, [i.label for i in self._instances],
                       [n.name for n in self._nodes])
            ref = self._node_coords().get(master)
            if ref is None:
                raise AssemblyError(
                    f"rigid_diaphragm({master!r}, {slave!r}): master_point= is "
                    f"required when the master is an instance port; it "
                    f"defaults to a reference node's coordinates."
                )
            master_point = ref
        return self._couple_ports(
            "rigid_diaphragm", master, slave, name,
            {"master_point": list(check_point(master_point, what="master_point")),
             "plane_normal": list(check_point(plane_normal, what="plane_normal")),
             "constrained_dofs": _list_or_none(constrained_dofs),
             "plane_tolerance": plane_tolerance})

    def embedded(
        self,
        host: str,
        embedded: str,
        *,
        tolerance: float = 0.0,
        stiffness: "float | str" = "auto",
        name: "str | None" = None,
    ) -> "Assembly":
        """Embed the nodes of one instance port in the elements of another.

        Each node of ``embedded`` that is not a corner of a ``host`` element
        is tied to the host sub-element containing it
        (``ASDEmbeddedNodeElement``). ``tolerance`` is the barycentric
        excess allowed; ``stiffness="auto"`` takes the penalty from the host
        material at build. Both ports are instance ports. Returns ``self``.

        Raises :class:`AssemblyError` here for a bad port, name or option,
        and from ``bridge()`` when every embedded node is a host corner or
        a node lies outside the host (INV-7).
        """
        return self._couple_ports(
            "embedded", host, embedded, name,
            {"tolerance": tolerance, "stiffness": stiffness})

    # The v2 form renames v1's first argument (part_a -> target); AS5
    # deletes v1, and this ignore with it.
    def couple(  # type: ignore[override]
        self,
        target: str,
        part_b: "str | None" = None,
        *,
        kind: str,
        reference: "str | None" = None,
        dofs: "Sequence[int] | None" = None,
        weighting: str = "uniform",
        name: "str | None" = None,
        ports: "Sequence[str] | None" = None,
        tolerance: "float | None" = None,
        **options: Any,
    ) -> "Assembly":
        """Couple an instance port to a reference node. Returns ``self``.

        ``kind="kinematic"`` is RBE2: the ``reference`` node drives
        ``target`` rigidly (``dofs``: the slave DOFs tied, ``None`` for
        all). ``kind="distributing"`` is RBE3: a load on ``reference`` is
        spread over ``target`` with ``weighting="uniform"`` or ``"area"``,
        adding no stiffness. Both emit Ladruno-fork elements
        (``LadrunoKinematicCoupling`` / ``LadrunoDistributingCoupling``);
        stock OpenSees refuses them at the element line. The reference
        needs the rotational DOFs (ndf 6 in 3-D).

        Any other ``kind`` is the v1 ``couple(part_a, part_b, kind=,
        ports=)`` of an ``add``-declared assembly, kept until AS5 deletes
        it. ``contact`` and ``interface`` are not assembly couplings (ADR
        0117 D3); they stay inside an instance.

        Raises :class:`AssemblyError` here for a bad port, name, option or
        kind, and from ``bridge()`` when the target resolves no node
        (INV-7).
        """
        v2_kind = _V2_COUPLE_KINDS.get(kind)
        if v2_kind is None:
            if self._instances or self._nodes or self._ties:
                raise AssemblyError(
                    f"couple(kind={kind!r}): an instance-declared assembly "
                    f"couples with kind 'kinematic' (RBE2) or 'distributing' "
                    f"(RBE3); use tie, equal_dof, rigid_link, rigid_diaphragm "
                    f"or embedded for the others. contact and interface are "
                    f"not assembly couplings (ADR 0117 D3)."
                )
            if part_b is None or ports is None:
                raise AssemblyError(
                    f"couple(kind={kind!r}): the v1 form is couple(part_a, "
                    f"part_b, kind=, ports=); the instance form takes kind "
                    f"'kinematic' or 'distributing'."
                )
            if reference is not None or weighting != "uniform":
                # Options of the instance form: v1 forwards unknown
                # keywords to g.constraints, so refuse rather than drop.
                raise AssemblyError(
                    f"couple(kind={kind!r}): reference= and weighting= are "
                    f"options of kind 'kinematic' / 'distributing' on an "
                    f"instance-declared assembly, not of the v1 form."
                )
            super().couple(
                target, part_b, kind=kind, ports=ports, dofs=dofs,
                tolerance=tolerance, name=name, **options)
            return self
        extra = {k: v for k, v in (("part_b", part_b), ("ports", ports),
                                   ("tolerance", tolerance)) if v is not None}
        extra.update(options)
        if extra:
            raise AssemblyError(
                f"couple(kind={kind!r}): unexpected options {sorted(extra)}; "
                f"the instance form is couple(target, kind=, reference=, "
                f"dofs= | weighting=, name=)."
            )
        if reference is None:
            raise AssemblyError(
                f"couple({target!r}, kind={kind!r}): reference= (a reference "
                f"node declared with Assembly.node) is required."
            )
        if v2_kind == "kinematic_coupling":
            if weighting != "uniform":
                raise AssemblyError(
                    f"couple({target!r}, kind='kinematic'): weighting= is an "
                    f"option of kind='distributing'.")
            params: dict[str, Any] = {"dofs": _list_or_none(dofs)}
        else:
            if dofs is not None:
                raise AssemblyError(
                    f"couple({target!r}, kind='distributing'): dofs= is an "
                    f"option of kind='kinematic'.")
            params = {"weighting": weighting}
        return self._couple_ports(v2_kind, reference, target, name, params)

    # ------------------------------------------------------------------
    # Declaration helpers
    # ------------------------------------------------------------------

    def _node_coords(self) -> dict[str, tuple[float, float, float]]:
        return {n.name: n.coords for n in self._nodes}

    def _check_new_name(self, name: object, *, what: str) -> str:
        """Refuse an invalid name or one the assembly already owns.

        Instance labels, reference nodes and tie / coupling names share one
        namespace (ADR 0117 INV-1): a name clashes with any of them.
        """
        check_label(name, what=what)
        owned = ([i.label for i in self._instances]
                 + [n.name for n in self._nodes]
                 + [t.name for t in self._ties if t.name is not None])
        if name in owned:
            raise AssemblyError(
                f"{what} {name!r} is already declared (instance labels, "
                f"reference nodes and tie names share one namespace)."
            )
        return str(name)

    def _couple_ports(
        self,
        kind: str,
        master: str,
        slave: str,
        name: "str | None",
        params: dict[str, Any],
    ) -> "Assembly":
        """Validate one coupling completely, then record it."""
        self._refuse_mixed(kind)
        labels = [i.label for i in self._instances]
        nodes = [n.name for n in self._nodes]
        master_ok, slave_ok = NODE_PORTS[kind]
        m_name = merged_port(master, labels, nodes if master_ok else ())
        s_name = merged_port(slave, labels, nodes if slave_ok else ())
        if name is not None:
            self._check_new_name(name, what=f"{kind} name")
        definition = coupling_definition(
            kind, m_name, s_name, params, name, self._node_coords())
        # Every check above runs first: a refused call records nothing.
        self._provenance.capture("assembly", "ties", name, on_existing="raise")
        self._ties.append(Coupling(
            kind=kind, master=master, slave=slave,
            params=canonical_params(params), name=name,
            definition=definition,
        ))
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def bridge(
        self, ndm: int, ndf: int, *,
        element_tags: Literal["sequential", "fem"] = "fem",
    ) -> "apeSees":
        """Build one forward :class:`~apeGmsh.opensees.apeSees` (ADR 0117 D1).

        The bridge holds the flat FEM of every instance plus the resolved
        ties, ``model(ndm, ndf)``, and each instance's rehydrated materials,
        sections and element specs. Declare fixes, loads, recorders and the
        analysis on it. ``element_tags="fem"`` (the default) keeps every
        element's relocated FEM id as its tag.

        An assembly whose instances carry no ``partition_rank`` is serial:
        its ``tcl()`` deck is unpartitioned. A ranked one is partitioned one
        rank per instance, and each cross-instance constraint is written on
        every rank that owns one of its nodes (ADR 0027).

        Each instance carries its model content: mesh, groups, labels,
        intra-instance constraints and the ``/rebar_elements`` stream, plus
        its uniaxial ``Elastic`` and nD ``ElasticIsotropic`` materials,
        ``Elastic`` and ``ElasticMembranePlateSection`` sections,
        transforms, uniform beam integrations, element-attached dampings and
        element specs. Its analysis content (fixes, masses, patterns,
        recorders, stages, the analysis) stays behind (ADR 0117 D4).

        Raises :class:`AssemblyError` before anything is merged or
        registered if no instance is declared, the instances' ranks leave a
        rank with no instance, a source file now composes ranked modules,
        or a source attaches a damping by region (global or in a stage).
        Raises it while merging if a tie or coupling resolves to no record.
        Raises it while rehydrating if an instance was built with another
        ``ndm`` or ``ndf``, or carries model content that is not rehydrated:
        another material, section, transform, integration or element type,
        or element rows whose args vary inside a physical group (the
        per-row selector, #1542).
        """
        self._refuse_mixed("bridge")
        if self._archive is not None:
            raise AssemblyError(
                f"Assembly({self.name!r}).bridge(): this assembly was read from "
                f"{str(self._archive)!r}, which re-lists its instances; the "
                f"archive itself is the model (OpenSeesModel.from_h5). Instance "
                f"files are never re-fetched."
            )
        if not self._instances:
            raise AssemblyError(f"Assembly({self.name!r}).bridge(): no instances.")
        try:
            check_dense_ranks(self._instances)
        except AssemblyError as exc:
            raise AssemblyError(f"Assembly({self.name!r}).bridge(): {exc}") from exc
        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees.opensees_model import OpenSeesModel

        from ._rehydrate import rehydrate

        fem = self._merged_fem()
        ranks = list(range(len(self._instances)))
        if (self._instances[0].partition_rank is not None
                and fem.nodes.partitions != ranks):
            raise AssemblyError(
                f"Assembly({self.name!r}).bridge(): the merged FEM has ranks "
                f"{fem.nodes.partitions}, not one per instance {ranks}. An "
                f"instance whose source is itself an assembly archive cannot "
                f"carry partition_rank: its nested instances take ranks of "
                f"their own."
            )
        # AS3 hook: the assembly's declaration provenance rides on the merged
        # FEM, so apeSees.h5 writes it to /provenance beside the bridge's.
        if fem.provenance is not None:
            raise AssemblyError(
                f"Assembly({self.name!r}).bridge(): the merged FEM already "
                f"carries provenance; merging it with the assembly's records "
                f"is not implemented."
            )
        fem.provenance = self._provenance.snapshot()
        ops = apeSees(fem, element_tags=element_tags)
        ops.model(ndm=ndm, ndf=ndf)
        # Read each source once. The key carries the resolved path as well
        # as the FEM hash: two files with one mesh but different materials
        # share a FEM hash.
        models: dict[tuple[str, str], "OpenSeesModel"] = {}
        opensees_hash: dict[str, str] = {}
        for inst in self._instances:
            key = (
                str(inst.source.resolve()),
                fem.composed_from[inst.label].source_fem_hash,
            )
            if key not in models:
                models[key] = OpenSeesModel.from_h5(inst.source)
            model = models[key]
            if model.ndm != ndm or model.ndf != ndf:
                raise AssemblyError(
                    f"instance {inst.label!r} was built with ndm={model.ndm}, "
                    f"ndf={model.ndf}; the assembly bridge has ndm={ndm}, "
                    f"ndf={ndf}."
                )
            rehydrate(ops, inst.label, model)
            opensees_hash[inst.label] = model.lineage.model_hash or ""
        self._bridged = _Bridged(
            ops=ops, fem=fem, instances=tuple(self._instances),
            nodes=tuple(self._nodes), ties=tuple(self._ties),
            opensees_hash=opensees_hash,
        )
        return ops

    # ------------------------------------------------------------------
    # Persistence (ADR 0117 D5)
    # ------------------------------------------------------------------

    def h5(self, path: "str | Path", *, model_name: "str | None" = None) -> None:
        """Write the last ``bridge()``'s ``model.h5`` plus ``/assembly``.

        The file is exactly what ``ops.h5(path)`` writes, so every reader
        opens it unchanged, with one more root zone that re-lists the
        instances and ties and ``/meta@assembly_schema_version``. Declare
        fixes, loads and the analysis on the bridge first, as for
        ``ops.h5``.

        Raises :class:`AssemblyError`, before writing anything, if
        ``bridge()`` was not called, if an instance or tie was declared
        after it, or if the assembly was read with :meth:`from_h5`.
        """
        if self._archive is not None:
            raise AssemblyError(
                f"Assembly({self.name!r}).h5(): this assembly was read from "
                f"{str(self._archive)!r}; it re-lists, it does not rebuild."
            )
        b = self._bridged
        if b is None:
            raise AssemblyError(
                f"Assembly({self.name!r}).h5(): call bridge() first; the "
                f"archive is the bridge's model.h5 plus /assembly."
            )
        if (b.instances != tuple(self._instances)
                or b.nodes != tuple(self._nodes)
                or b.ties != tuple(self._ties)):
            raise AssemblyError(
                f"Assembly({self.name!r}).h5(): instances, nodes or ties were "
                f"declared after bridge(); call bridge() again."
            )
        instances = _instance_rows(b)
        ties = _tie_rows(b)
        # Refuse a bad row before ops.h5 overwrites ``path``.
        validate_rows(self.name, instances, ties)
        b.ops.h5(str(path), model_name=model_name)
        write_assembly_zone(path, self.name, instances, ties)

    @classmethod
    def from_h5(cls, path: "str | Path") -> "Assembly":
        """Re-list the instances and ties an assembly archive declares.

        Reads ``/assembly`` only: the flat zones stay authoritative, and
        instance files are never re-fetched, so ``instances`` and ``ties``
        equal the declared ones even when the sources have moved. The
        result cannot ``bridge()`` or ``h5()``; open the archive itself
        with ``OpenSeesModel.from_h5`` to rebuild the model.

        Raises :class:`AssemblyError` for a file without ``/assembly`` or
        with a row the declaring verbs would refuse, and ``MalformedH5Error``
        for a rotation row with a zero axis and a nonzero angle (only the
        all-zero row means "not rotated").
        """
        from apeGmsh._kernel.defs.constraints import TieDef
        from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

        zone = read_assembly_zone(path)
        asm = cls(zone.name)
        for r in zone.instances:
            check_label(r.label, what="instance label")
            ax, ay, az, theta = r.rotate
            if (ax, ay, az, theta) == (0.0, 0.0, 0.0, 0.0):
                rotate = None
            elif (ax, ay, az) == (0.0, 0.0, 0.0):
                raise MalformedH5Error(
                    f"{path}: /assembly instance {r.label!r} has rotate "
                    f"{r.rotate!r}: a zero axis with a nonzero angle. Only "
                    f"the all-zero row means 'not rotated'."
                )
            else:
                rotate = ((ax, ay, az), theta)
            asm._instances.append(Instance(
                label=r.label,
                source=Path(r.source_path),
                translate=check_translate(r.translate),
                rotate=check_rotate(rotate),
                partition_rank=check_partition_rank(
                    None if r.partition_rank == -1 else r.partition_rank,
                    r.label, asm._instances),
            ))
        labels = [i.label for i in asm._instances]
        # Reference nodes first: a coupling row may name any of them.
        for t in zone.ties:
            if t.kind != "node":
                continue
            params = json.loads(t.params)
            if set(params) != TIE_PARAMS["node"] or t.master or t.slave:
                raise AssemblyError(
                    f"{path}: /assembly node row {t.name!r} must carry params "
                    f"{sorted(TIE_PARAMS['node'])} and empty ports."
                )
            asm._check_new_name(t.name, what="node name")
            asm._nodes.append(RefNode(
                name=t.name, coords=check_point(params["coords"], what="coords")))
        nodes = [n.name for n in asm._nodes]
        for t in zone.ties:
            if t.kind == "node":
                continue
            params = json.loads(t.params)
            name = t.name or None
            if name is not None:
                asm._check_new_name(name, what=f"{t.kind} name")
            if t.kind != "tie":
                master_ok, slave_ok = NODE_PORTS[t.kind]
                m_name = merged_port(t.master, labels, nodes if master_ok else ())
                s_name = merged_port(t.slave, labels, nodes if slave_ok else ())
                try:
                    definition = coupling_definition(
                        t.kind, m_name, s_name, params, name,
                        asm._node_coords())
                except AssemblyError as exc:
                    raise AssemblyError(f"{path}: /assembly row: {exc}") from exc
                asm._ties.append(Coupling(
                    kind=t.kind, master=t.master, slave=t.slave,
                    params=canonical_params(params), name=name,
                    definition=definition,
                ))
                continue
            if set(params) != TIE_PARAMS["tie"]:
                raise AssemblyError(
                    f"{path}: /assembly tie {t.name!r} params carry "
                    f"{sorted(params)}, expected {sorted(TIE_PARAMS['tie'])}."
                )
            m_name = merged_port(t.master, labels)
            s_name = merged_port(t.slave, labels)
            dofs = (tuple(int(d) for d in params["dofs"])
                    if params["dofs"] is not None else None)
            try:
                definition = TieDef(
                    master_label=m_name, slave_label=s_name,
                    dofs=list(dofs) if dofs is not None else None,
                    tolerance=float(params["tolerance"]),
                    enforce=params["enforce"], method=params["method"],
                    name=name,
                )
            except ValueError as exc:
                raise AssemblyError(f"{path}: /assembly tie {t.name!r}: {exc}") from exc
            asm._ties.append(Tie(
                master=t.master, slave=t.slave, enforce=params["enforce"],
                method=params["method"], dofs=dofs,
                tolerance=float(params["tolerance"]), name=name,
                definition=definition,
            ))
        asm._archive = Path(path)
        return asm

    def _merged_fem(self) -> "FEMData":
        """Every instance composed onto an empty broker, then every tie."""
        from apeGmsh._kernel.resolvers._chain_phase_router import route_def_to_fem

        from ._rehydrate import refuse_region_dampings

        # Refused before anything is merged or registered: a region
        # attach is not carried, and the model reader cannot see it. The
        # unranked-source check of instance() runs again, because the file
        # at a source path can change between instance() and bridge().
        for inst in self._instances:
            check_unranked_source(inst.source, inst.label)
            refuse_region_dampings(inst.label, inst.source)
        fem = _base_fem(self._nodes)
        for inst in self._instances:
            fem = fem.compose(
                inst.source,
                label=inst.label,
                translate=inst.translate,
                rotate=inst.compose_rotate(),
                partition_rank=inst.partition_rank,
            )
        for t in self._ties:
            verb = "tie" if isinstance(t, Tie) else t.kind
            what = f"{verb}({t.master!r}, {t.slave!r})"
            before = _constraint_count(fem)
            try:
                routed = route_def_to_fem(fem, t.definition)
            except KeyError as exc:
                raise AssemblyError(
                    f"{what}: a port is not a physical group or label of its "
                    f"instance. {exc}"
                ) from exc
            except ValueError as exc:
                # The resolvers refuse a tie that attaches no slave node,
                # an RBE3 with no independent and an embedded node outside
                # every host element.
                raise AssemblyError(f"{what} resolved to no record: {exc}") from exc
            # A group record with no slave constrains nothing: a
            # diaphragm whose master is in the plane but whose ports put
            # no other node there routes to one record and the deck line
            # ``rigidDiaphragm 3 1``.
            if (routed is None or _constraint_count(routed) <= before
                    or _slaveless(fem, routed)):
                hint = (f"Check that the two surfaces meet within "
                        f"tolerance={t.tolerance}." if isinstance(t, Tie)
                        else _EMPTY_HINT[t.kind])
                raise AssemblyError(
                    f"{what} resolved to no record: the ports exist but "
                    f"couple nothing. {hint}"
                )
            fem = routed
        return fem

    # ------------------------------------------------------------------
    # v1 coexistence (deleted in AS5)
    # ------------------------------------------------------------------

    def _refuse_mixed(self, verb: str) -> None:
        if self._parts or self._couples:
            raise AssemblyError(
                f"{verb}(): this assembly already uses the v1 add/couple API; "
                f"declare it with instance/tie/bridge only."
            )

    def add(
        self,
        label: str,
        source: str,
        *,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: "tuple[float, float, float, float] | None" = None,
        anchor: "str | None" = None,
    ) -> "Assembly":
        if self._instances or self._nodes or self._ties:
            raise AssemblyError(
                "add(): this assembly already uses instance/tie; the v1 "
                "add/couple/materialize API cannot be mixed in."
            )
        super().add(
            label, source, translate=translate, rotate=rotate, anchor=anchor)
        return self

    add.__doc__ = _AssemblyV1.add.__doc__


def _base_fem(ref_nodes: Sequence[RefNode] = ()) -> "FEMData":
    """The merge's start: the assembly's reference nodes and nothing else.

    With no reference node it is the empty broker. Reference node ``k``
    (1-based, declaration order) is FEM id ``k``, decoupled (ADR 0049) and
    labelled with its name. The ids sit below the first instance window
    (the merge engine rounds the host's top id up to the next million), so
    declaring a node does not move any instance.
    """
    import numpy as np

    from apeGmsh.mesh import FEMData, LabelSet, MeshInfo, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import (
        PROVENANCE_DECOUPLED,
        ElementComposite,
        NodeComposite,
    )

    n = len(ref_nodes)
    ids = np.arange(1, n + 1, dtype=np.int64)
    coords = np.array([r.coords for r in ref_nodes],
                      dtype=np.float64).reshape(n, 3)
    labels = LabelSet({
        (0, int(nid)): {
            "name": r.name,
            "node_ids": np.array([nid], dtype=np.int64),
            "node_coords": coords[k:k + 1],
        }
        for k, (nid, r) in enumerate(zip(ids, ref_nodes))
    })
    nodes = NodeComposite(
        node_ids=ids,
        node_coords=coords,
        physical=PhysicalGroupSet({}),
        labels=labels,
        provenance=(np.full(n, PROVENANCE_DECOUPLED, dtype=np.int8)
                    if n else None),
    )
    elements = ElementComposite(
        groups={}, physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    return FEMData(nodes=nodes, elements=elements, info=MeshInfo(n, 0, 0))


#: Per coupling kind, why a coupling between existing ports can resolve
#: to no record (ADR 0117 INV-7).
_EMPTY_HINT: dict[str, str] = {
    "equal_dof": "No slave node lies within tolerance of a master node.",
    "rigid_link": "The slave set holds no node but the master.",
    "rigid_diaphragm": "No node of either port but the master lies within "
                       "plane_tolerance of the plane.",
    "embedded": "Every embedded node is a corner of a host element.",
    "kinematic_coupling": "The target holds no node but the reference.",
    "distributing_coupling": "The target holds no node but the reference.",
}

#: ``couple(kind=)`` of an instance-declared assembly → the row kind.
_V2_COUPLE_KINDS: dict[str, str] = {
    "kinematic": "kinematic_coupling",
    "distributing": "distributing_coupling",
}


def _list_or_none(values: "Sequence[Any] | None") -> "list[Any] | None":
    """A JSON-ready list of ``values`` (``None`` kept); checked later."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise AssemblyError(f"expected a sequence of DOFs, got {values!r}.")
    return list(values)


def _point_or_none(value: object, what: str) -> "list[float] | None":
    return None if value is None else list(check_point(value, what=what))


def _instance_rows(b: _Bridged) -> list[InstanceRow]:
    """One ``/assembly/instances`` row per instance of the bridge.

    ``fem_id_base`` / ``fem_id_span`` are the relocated FEM-id window the
    instance's nodes and elements occupy (ADR 0117 D2): the merge engine
    maps the source's smallest id to the window's base.
    """
    import numpy as np

    fem = b.fem
    node_lbl = fem.nodes.module_label
    elem_lbl = fem.elements.module_label_by_id()
    if node_lbl is None or elem_lbl is None:
        raise AssemblyError(
            "h5(): the merged FEM carries no per-row instance labels; the "
            "FEM-id window of an instance cannot be recorded."
        )
    from apeGmsh.mesh._compose import _split_joined_label

    # A row's label is the instance's own, or, for an instance whose
    # source is itself an assembly archive, the joined ``X/A`` the merge
    # engine grafts (ADR 0038 nested composition): its root names the
    # instance. ``""`` marks an assembly-owned reference node.
    roots: dict[str, str] = {}

    def root(lbl: object) -> str:
        key = str(lbl)
        if key not in roots:
            parts = _split_joined_label(key)
            roots[key] = parts[0] if parts else ""
        return roots[key]

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    node_root = np.array([root(lbl) for lbl in node_lbl], dtype=object)
    rows: list[InstanceRow] = []
    for inst in b.instances:
        ids = [int(i) for i in node_ids[node_root == inst.label]]
        ids += [int(e) for e, lbl in elem_lbl.items() if root(lbl) == inst.label]
        if not ids:
            raise AssemblyError(
                f"h5(): instance {inst.label!r} owns no node or element of "
                f"the merged FEM."
            )
        rec = fem.composed_from[inst.label]
        model_hash = b.opensees_hash[inst.label]
        if not model_hash:
            raise AssemblyError(
                f"h5(): the source of instance {inst.label!r} carries no "
                f"/meta/lineage model_hash."
            )
        if inst.rotate is None:
            rotate = (0.0, 0.0, 0.0, 0.0)
        else:
            (ax, ay, az), theta = inst.rotate
            rotate = (ax, ay, az, theta)
        rows.append(InstanceRow(
            label=inst.label,
            source_path=inst.source.as_posix(),
            source_fem_hash=str(rec.source_fem_hash),
            source_opensees_hash=model_hash,
            translate=inst.translate,
            rotate=rotate,
            fem_id_base=min(ids),
            fem_id_span=max(ids) - min(ids) + 1,
            partition_rank=(
                -1 if rec.partition_rank is None else int(rec.partition_rank)),
        ))
    return rows


def _tie_rows(b: _Bridged) -> list[TieRow]:
    """One ``/assembly/ties`` row per tie of the bridge.

    ``n_records`` re-resolves the tie against the merged FEM. Resolution
    reads only coordinates and groups, which ties do not change, so it
    yields the records ``bridge()`` added for that tie; the FEM is not
    modified (``route_def_to_fem`` returns a new one).
    """
    from apeGmsh._kernel.resolvers._chain_phase_router import route_def_to_fem

    fem = b.fem
    before = _constraint_count(fem)
    # Reference nodes first (kind ``node``, empty ports, one node each):
    # ``from_h5`` reads them before the couplings that name them.
    rows: list[TieRow] = [
        TieRow(
            name=n.name, kind="node", master="", slave="",
            params=canonical_params({"coords": list(n.coords)}),
            n_records=1,
        )
        for n in b.nodes
    ]
    for t in b.ties:
        kind = "tie" if isinstance(t, Tie) else t.kind
        with warnings.catch_warnings():
            # bridge() already showed this resolution's warnings.
            warnings.simplefilter("ignore")
            routed = route_def_to_fem(fem, t.definition)
        if routed is None:
            raise AssemblyError(
                f"h5(): {kind}({t.master!r}, {t.slave!r}) no longer resolves "
                f"in chain phase."
            )
        if isinstance(t, Tie):
            params = canonical_params({
                "dofs": list(t.dofs) if t.dofs is not None else None,
                "enforce": t.enforce,
                "method": t.method,
                "tolerance": t.tolerance,
            })
        else:
            params = t.params
        rows.append(TieRow(
            name=t.name or "",
            kind=kind,
            master=t.master,
            slave=t.slave,
            params=params,
            n_records=_constraint_count(routed) - before,
        ))
    return rows


def _constraint_count(fem: "FEMData") -> int:
    """Node-side plus element-side constraint records on a broker."""
    return len(tuple(fem.nodes.constraints)) + len(tuple(fem.elements.constraints))


def _slaveless(before: "FEMData", after: "FEMData") -> bool:
    """True when routing appended a node-group record with no slave.

    Records are appended (``with_constraint``), so the new ones are the
    tail of ``after``'s node-side set.
    """
    from apeGmsh._kernel.records._constraints import NodeGroupRecord

    n = len(tuple(before.nodes.constraints))
    return any(
        isinstance(rec, NodeGroupRecord) and not rec.slave_nodes
        for rec in tuple(after.nodes.constraints)[n:]
    )
