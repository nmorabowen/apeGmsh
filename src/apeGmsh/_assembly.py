"""``Assembly`` — declarative spatial model-coupling graph (ADR 0043 slice 1.4).

The :class:`Assembly` is the user-facing "Workbench" object the
connectivity-graph initiative is building toward: declare several saved
apeGmsh models (``model.h5``) as **parts**, join them with interface
**couples** (constraints), then :meth:`~Assembly.materialize` the whole
graph into one composed session in a single pass.

::

    from apeGmsh import Assembly

    asm = Assembly("cerro_lindo")
    asm.add("pier", "pier.h5")                 # first add = host
    asm.add("soil", "soil.h5", translate=(0.0, 0.0, -5.0))
    asm.couple(
        "soil", "pier", kind="tied_contact",
        ports=("top", "base"), dofs=[1, 2, 3],
    )
    g = asm.materialize()                      # -> composed apeGmsh session
    g.save("cerro_lindo.h5")                   # or apeSees(g._fem).tcl(...)

Design (ADR 0043 mode A; settled via the slice-1.4 red/blue pass):

* **Thin wrapper.** ``add`` resolves to :meth:`apeGmsh.from_h5` (first
  part = host) + :meth:`apeGmsh.compose` (each later part, namespaced
  under its label); ``couple`` resolves to ``g.constraints.<kind>(...)``
  routed through the chain-phase resolver (ADR 0041). No new emit / merge
  machinery — the pipeline was verified end-to-end before this landed.
* **Host asymmetry is handled.** ``g.compose`` leaves the host's physical
  groups *un-namespaced* and prefixes every later part's groups with its
  label. So a ``port`` on the host part resolves to the bare PG name; a
  ``port`` on a composed part resolves to ``"{label}.{pg}"``. The caller
  always writes bare per-part PG names; :meth:`materialize` does the
  namespacing.
* **Fail loud.** The chain-phase router silently no-ops a constraint whose
  target name does not resolve (it swallows ``KeyError`` for build-phase
  back-compat). :meth:`materialize` detects a couple that produced no
  constraint record and raises :class:`AssemblyError` — a declared
  interface that ties nothing is a setup bug, never a silent empty deck.
* **Declarative / lazy.** ``add`` and ``couple`` only record; all I/O and
  resolution happen in :meth:`materialize`.

Out of scope for slice 1.4 (deferred to 1.5): ``Assembly.emit`` (the
split-deck export — use ``apeSees(g._fem).tcl(split=True)`` on the
materialised session) and ``Assembly.graph`` (inspection). ``couple``
supports ``kind="equal_dof"`` and ``kind="tied_contact"``; ``embedded``
needs host-volume geometry the bare-PG ports model does not express and
is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh


#: Couple kinds supported in slice 1.4 (chain-phase-routed, node/face
#: interface constraints). ``embedded`` / ``rigid_link`` are deferred.
_COUPLE_KINDS: tuple[str, ...] = ("equal_dof", "tied_contact")


class AssemblyError(Exception):
    """Raised for invalid :class:`Assembly` declarations or materialisation
    failures (ADR 0043 slice 1.4)."""


@dataclass(frozen=True)
class _Part:
    """One declared part node — a saved model.h5 + its placement."""

    label: str
    source: str
    translate: tuple[float, float, float]
    rotate: "tuple[float, float, float, float] | None"
    anchor: "str | None"


@dataclass(frozen=True)
class _Couple:
    """One declared A-edge — an interface constraint between two parts."""

    part_a: str
    part_b: str
    kind: str
    ports: tuple[str, str]
    dofs: "tuple[int, ...] | None"
    tolerance: "float | None"
    name: "str | None"
    options: dict = field(default_factory=dict)


class Assembly:
    """A declarative spatial assembly of composed parts + interface couples.

    See the module docstring for the full contract and an example.
    """

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise AssemblyError("Assembly(name=) requires a non-empty string.")
        self.name = name
        self._parts: list[_Part] = []
        self._couples: list[_Couple] = []

    # ------------------------------------------------------------------
    # Declaration
    # ------------------------------------------------------------------

    def add(
        self,
        label: str,
        source: str,
        *,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: "tuple[float, float, float, float] | None" = None,
        anchor: "str | None" = None,
    ) -> "Assembly":
        """Register a part (a saved ``model.h5``). Returns ``self`` (chainable).

        The **first** ``add`` is the host: its model is loaded via
        :meth:`apeGmsh.from_h5` and its physical groups stay
        un-namespaced. Every later ``add`` is composed onto the host under
        its ``label`` (so its PG ``"top"`` becomes ``"{label}.top"``),
        with the given rigid-body placement.

        ``translate`` / ``rotate`` / ``anchor`` mirror
        :meth:`apeGmsh.compose` (``anchor`` is mutually exclusive with a
        non-zero ``translate``); they are ignored for the host part.
        """
        if not isinstance(label, str) or not label:
            raise AssemblyError("add(label=) must be a non-empty string.")
        if any(p.label == label for p in self._parts):
            raise AssemblyError(
                f"add(label={label!r}): duplicate part label; each part "
                f"must have a unique label."
            )
        self._parts.append(
            _Part(
                label=label,
                source=str(source),
                translate=tuple(float(t) for t in translate),  # type: ignore[arg-type]
                rotate=rotate,
                anchor=anchor,
            )
        )
        return self

    def couple(
        self,
        part_a: str,
        part_b: str,
        *,
        kind: str,
        ports: Sequence[str],
        dofs: "Sequence[int] | None" = None,
        tolerance: "float | None" = None,
        name: "str | None" = None,
        **options: Any,
    ) -> "Assembly":
        """Declare an interface constraint between two parts. Chainable.

        Parameters
        ----------
        part_a, part_b : str
            Part labels (must have been ``add``-ed). ``part_a``'s port is
            the master side; ``part_b``'s is the slave.
        kind : {"equal_dof", "tied_contact"}
            The constraint kind; resolves to ``g.constraints.<kind>``.
        ports : (str, str)
            **Bare** physical-group names, one per part: ``ports[0]`` on
            ``part_a``, ``ports[1]`` on ``part_b``. :meth:`materialize`
            namespaces them (host → bare, composed → ``"{label}.{pg}"``).
        dofs : sequence of int, optional
            DOFs to tie (e.g. ``[1, 2, 3]``); forwarded to the constraint.
        tolerance : float, optional
            Co-location / projection tolerance; forwarded when given.
        name : str, optional
            Optional constraint name; forwarded when given.
        **options
            Extra kind-specific keyword args forwarded verbatim to the
            constraint method (e.g. ``stiffness=`` for ``tied_contact``).
        """
        if kind not in _COUPLE_KINDS:
            raise AssemblyError(
                f"couple(kind={kind!r}): unsupported kind; slice 1.4 "
                f"supports {list(_COUPLE_KINDS)} (embedded/rigid_link "
                f"deferred)."
            )
        ports_t = tuple(ports)
        if len(ports_t) != 2 or not all(
            isinstance(p, str) and p for p in ports_t
        ):
            raise AssemblyError(
                f"couple(ports={ports!r}): expected a 2-tuple of non-empty "
                f"PG-name strings (one per part)."
            )
        self._couples.append(
            _Couple(
                part_a=str(part_a),
                part_b=str(part_b),
                kind=kind,
                ports=(ports_t[0], ports_t[1]),
                dofs=tuple(int(d) for d in dofs) if dofs is not None else None,
                tolerance=tolerance,
                name=name,
                options=dict(options),
            )
        )
        return self

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def materialize(self) -> "apeGmsh":
        """Run the graph: compose every part + apply every couple.

        Returns the composed :class:`apeGmsh` session (its ``_fem`` is the
        merged broker; emit via ``apeSees(g._fem).tcl(...)`` or persist via
        ``g.save(...)``).

        Raises
        ------
        AssemblyError
            If no parts were added, a couple references an unknown part,
            or a couple resolved to **zero** constraint records (a declared
            interface that ties nothing — the chain-phase router could not
            resolve a port name).
        """
        if not self._parts:
            raise AssemblyError(
                f"Assembly({self.name!r}).materialize(): no parts added."
            )
        from apeGmsh._core import apeGmsh

        host = self._parts[0]
        labels = {p.label for p in self._parts}

        # Host part — loaded directly into chain phase; PGs stay bare.
        g = apeGmsh.from_h5(host.source)

        # Remaining parts — composed under their label.
        for part in self._parts[1:]:
            g.compose(
                part.source,
                label=part.label,
                translate=part.translate,
                rotate=part.rotate,
                anchor=part.anchor,
            )

        host_label = host.label
        for c in self._couples:
            for side in (c.part_a, c.part_b):
                if side not in labels:
                    raise AssemblyError(
                        f"couple references unknown part {side!r}; declared "
                        f"parts are {sorted(labels)}."
                    )
            master = self._resolve_port(c.part_a, c.ports[0], host_label)
            slave = self._resolve_port(c.part_b, c.ports[1], host_label)

            method = getattr(g.constraints, c.kind)
            kwargs: dict[str, Any] = dict(c.options)
            if c.dofs is not None:
                kwargs["dofs"] = list(c.dofs)
            if c.tolerance is not None:
                kwargs["tolerance"] = c.tolerance
            if c.name is not None:
                kwargs["name"] = c.name

            before = _constraint_count(g._fem)
            # Two fail-loud layers collapse into one AssemblyError: the
            # constraint composite raises KeyError eagerly for a port name
            # that resolves to no label/PG; and a name that DOES resolve
            # but ties nothing (empty interface, or the chain-phase router
            # swallowing a deeper resolution failure) leaves the broker
            # unchanged — both are setup bugs for a declared couple.
            try:
                method(master, slave, **kwargs)
            except KeyError as exc:
                raise AssemblyError(
                    f"couple({c.part_a!r}, {c.part_b!r}, kind={c.kind!r}, "
                    f"ports={c.ports!r}): port resolved to master={master!r} "
                    f"/ slave={slave!r}, which is not a physical group on "
                    f"the composed model. {exc.args[0] if exc.args else exc}"
                ) from exc
            after = _constraint_count(g._fem)
            if after <= before:
                raise AssemblyError(
                    f"couple({c.part_a!r}, {c.part_b!r}, kind={c.kind!r}, "
                    f"ports={c.ports!r}) produced no constraint: the "
                    f"interface resolved to master={master!r} / "
                    f"slave={slave!r} and tied nothing. Check the port PG "
                    f"names exist on each part and the interface nodes/faces "
                    f"actually meet."
                )
        return g

    @staticmethod
    def _resolve_port(part_label: str, pg: str, host_label: str) -> str:
        """Bare PG name for the host part; ``"{label}.{pg}"`` for composed."""
        if part_label == host_label:
            return pg
        return f"{part_label}.{pg}"


def _constraint_count(fem: Any) -> int:
    """Total node-side + element-side constraint records on a broker."""
    return len(tuple(fem.nodes.constraints)) + len(
        tuple(fem.elements.constraints)
    )
