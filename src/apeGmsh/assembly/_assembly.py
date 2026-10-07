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

The v1 declarations (``add`` / ``couple`` / ``materialize``) are inherited
unchanged until AS5 deletes them; one assembly uses one API or the other
(the v1 build of a v2 assembly finds no parts and raises).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

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


class Assembly(_AssemblyV1):
    """Instances of saved ``model.h5`` files joined by assembly-level ties.

    See the module docstring. ``name`` is the assembly's own name.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._instances: list[Instance] = []
        self._ties: list[Tie] = []

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
        self._instances.append(Instance(
            label=label,
            source=path,
            translate=check_translate(translate),
            rotate=check_rotate(rotate),
        ))
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
        if not self._instances:
            raise AssemblyError(f"Assembly({self.name!r}).bridge(): no instances.")
        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees.opensees_model import OpenSeesModel

        from ._rehydrate import rehydrate

        fem = self._merged_fem()
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
        return ops

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


def _constraint_count(fem: "FEMData") -> int:
    """Node-side plus element-side constraint records on a broker."""
    return len(tuple(fem.nodes.constraints)) + len(tuple(fem.elements.constraints))
