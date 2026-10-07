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
from typing import TYPE_CHECKING, Literal, Sequence

from apeGmsh._internal.provenance import ProvenanceStore

from ._h5 import (
    InstanceRow,
    TieRow,
    read_assembly_zone,
    validate_rows,
    write_assembly_zone,
)
from ._instances import (
    Instance,
    Tie,
    check_label,
    check_rotate,
    check_translate,
    split_port,
)
from ._v1 import Assembly as _AssemblyV1
from ._v1 import AssemblyError

if TYPE_CHECKING:
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.opensees_model import OpenSeesModel

__all__ = ["Assembly", "AssemblyRankWarning"]


class AssemblyRankWarning(UserWarning):
    """``bridge()`` built a partitioned FEM whose rank 0 is empty (AS4, #1530)."""


@dataclass(frozen=True)
class _Bridged:
    """What ``bridge()`` built, kept for ``h5()``."""

    ops: "apeSees"
    fem: "FEMData"
    #: The declarations the bridge was built from; ``h5()`` refuses if
    #: they changed since.
    instances: tuple[Instance, ...]
    ties: tuple[Tie, ...]
    #: ``model_hash`` of each instance's source, by instance label.
    opensees_hash: dict[str, str]


class Assembly(_AssemblyV1):
    """Instances of saved ``model.h5`` files joined by assembly-level ties.

    See the module docstring. ``name`` is the assembly's own name.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._instances: list[Instance] = []
        self._ties: list[Tie] = []
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
    def ties(self) -> tuple[Tie, ...]:
        """Declared assembly-level ties, in declaration order."""
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
    ) -> "Assembly":
        """Place a saved ``model.h5`` under ``label``. Returns ``self``.

        Every name the file owns becomes ``{label}.{name}``. ``rotate`` is
        ``((ax, ay, az), theta_radians)`` about the origin, applied before
        ``translate``. The same file may be instanced any number of times.

        Raises :class:`AssemblyError`, before recording anything, for a
        label that is empty, contains ``.``, ``/`` or whitespace, starts or
        ends with ``_``, or is already declared; for a missing file; and for
        a zero rotation axis.
        """
        self._refuse_mixed("instance")
        check_label(label, what="instance label")
        if any(i.label == label for i in self._instances):
            raise AssemblyError(f"instance label {label!r} is already declared.")
        path = Path(source)
        if not path.is_file():
            raise AssemblyError(f"instance {label!r}: no file at {str(path)!r}.")
        placed = Instance(
            label=label,
            source=path,
            translate=check_translate(translate),
            rotate=check_rotate(rotate),
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
        split_port(master, labels)
        split_port(slave, labels)
        if name is not None:
            check_label(name, what="tie name")
            if any(t.name == name for t in self._ties):
                raise AssemblyError(f"tie name {name!r} is already declared.")
        from apeGmsh._kernel.defs.constraints import TieDef

        dofs_t = tuple(int(d) for d in dofs) if dofs is not None else None
        try:
            definition = TieDef(
                master_label=master, slave_label=slave,
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

        Raises :class:`AssemblyError` if no instance is declared, an
        instance was built with another ``ndm`` or ``ndf``, a tie resolves to
        no record, or an instance carries model content AS1 cannot rehydrate.
        Warns :class:`AssemblyRankWarning` when the merged FEM is partitioned
        (always, until AS4): rank 0 is empty, so use ``tcl(flat=True)`` for
        the serial deck.
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
        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees.opensees_model import OpenSeesModel

        from ._rehydrate import rehydrate

        fem = self._merged_fem()
        # AS3 hook: the assembly's declaration provenance rides on the merged
        # FEM, so apeSees.h5 writes it to /provenance beside the bridge's.
        if fem.provenance is not None:
            raise AssemblyError(
                f"Assembly({self.name!r}).bridge(): the merged FEM already "
                f"carries provenance; merging it with the assembly's records "
                f"is not implemented."
            )
        fem.provenance = self._provenance.snapshot()
        if len(fem.partitions) > 1:
            warnings.warn(
                f"Assembly({self.name!r}).bridge(): the merged FEM is "
                f"partitioned one rank per instance with an empty rank 0 "
                f"(no host; AS4, #1530), so the default tcl() writes a "
                f"partitioned deck. tcl(flat=True) is the serial deck; live "
                f"runs are serial.",
                AssemblyRankWarning,
                stacklevel=2,
            )
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
            ties=tuple(self._ties), opensees_hash=opensees_hash,
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
        if b.instances != tuple(self._instances) or b.ties != tuple(self._ties):
            raise AssemblyError(
                f"Assembly({self.name!r}).h5(): instances or ties were "
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
            ))
        labels = [i.label for i in asm._instances]
        for t in zone.ties:
            params = json.loads(t.params)
            if set(params) != _TIE_PARAMS:
                raise AssemblyError(
                    f"{path}: /assembly tie {t.name!r} params carry "
                    f"{sorted(params)}, expected {sorted(_TIE_PARAMS)}."
                )
            split_port(t.master, labels)
            split_port(t.slave, labels)
            name = t.name or None
            if name is not None:
                check_label(name, what="tie name")
            dofs = (tuple(int(d) for d in params["dofs"])
                    if params["dofs"] is not None else None)
            try:
                definition = TieDef(
                    master_label=t.master, slave_label=t.slave,
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
        # attach is not carried, and the model reader cannot see it.
        for inst in self._instances:
            refuse_region_dampings(inst.label, inst.source)
        fem = _empty_fem()
        for inst in self._instances:
            fem = fem.compose(
                inst.source,
                label=inst.label,
                translate=inst.translate,
                rotate=inst.compose_rotate(),
            )
        for t in self._ties:
            before = _constraint_count(fem)
            try:
                routed = route_def_to_fem(fem, t.definition)
            except KeyError as exc:
                raise AssemblyError(
                    f"tie({t.master!r}, {t.slave!r}): a port is not a physical "
                    f"group or label of its instance. {exc}"
                ) from exc
            except ValueError as exc:
                # The resolver refuses a tie that attaches no slave node.
                raise AssemblyError(
                    f"tie({t.master!r}, {t.slave!r}) resolved to no record: "
                    f"{exc}"
                ) from exc
            if routed is None or _constraint_count(routed) <= before:
                raise AssemblyError(
                    f"tie({t.master!r}, {t.slave!r}) resolved to no record: "
                    f"the ports exist but tie nothing. Check that the two "
                    f"surfaces meet within tolerance={t.tolerance}."
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
        if self._instances or self._ties:
            raise AssemblyError(
                "add(): this assembly already uses instance/tie; the v1 "
                "add/couple/materialize API cannot be mixed in."
            )
        super().add(
            label, source, translate=translate, rotate=rotate, anchor=anchor)
        return self

    add.__doc__ = _AssemblyV1.add.__doc__


def _empty_fem() -> "FEMData":
    """A broker with no nodes, elements or groups: the merge's start."""
    import numpy as np

    from apeGmsh.mesh import FEMData, LabelSet, MeshInfo, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import ElementComposite, NodeComposite

    nodes = NodeComposite(
        node_ids=np.empty(0, dtype=np.int64),
        node_coords=np.empty((0, 3), dtype=np.float64),
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={}, physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    return FEMData(nodes=nodes, elements=elements, info=MeshInfo(0, 0, 0))


#: The keys of a tie row's ``params`` JSON.
_TIE_PARAMS = frozenset({"dofs", "enforce", "method", "tolerance"})


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
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    rows: list[InstanceRow] = []
    for inst in b.instances:
        ids = [int(i) for i in node_ids[node_lbl == inst.label]]
        ids += [int(e) for e, lbl in elem_lbl.items() if lbl == inst.label]
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
    rows: list[TieRow] = []
    for t in b.ties:
        with warnings.catch_warnings():
            # bridge() already showed this resolution's warnings.
            warnings.simplefilter("ignore")
            routed = route_def_to_fem(fem, t.definition)
        if routed is None:
            raise AssemblyError(
                f"h5(): tie({t.master!r}, {t.slave!r}) no longer resolves "
                f"in chain phase."
            )
        params = {
            "dofs": list(t.dofs) if t.dofs is not None else None,
            "enforce": t.enforce,
            "method": t.method,
            "tolerance": t.tolerance,
        }
        rows.append(TieRow(
            name=t.name or "",
            kind="tie",
            master=t.master,
            slave=t.slave,
            params=json.dumps(params, sort_keys=True, separators=(",", ":")),
            n_records=_constraint_count(routed) - before,
        ))
    return rows


def _constraint_count(fem: "FEMData") -> int:
    """Node-side plus element-side constraint records on a broker."""
    return len(tuple(fem.nodes.constraints)) + len(tuple(fem.elements.constraints))
