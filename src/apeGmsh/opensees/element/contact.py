"""`contactSurface` + `contact` — emit-grammar builders (Ladruno fork).

The fork contact subsystem is a pair of commands plus a constraint handler:

    contactSurface tag (-slave | -master nps | -slave-segments nps) nodeTag…
    contact tag masterTag slaveTag [kn kt mu | auto] [-outward ox oy oz]
            [-mortar -epsN auto|v -mu -epsT -cohesion -tauMax -augTol -maxAug
             -ngp -tie]
    constraints LadrunoContact

A contact interaction is two named meshed faces: the master is always a
**faceted** surface (`-master`, flat node connectivity with stride `nps`); the
slave is a node set (NTS, `-slave`) or a faceted surface (mortar,
`-slave-segments`). These builders are the single source of truth for the two
grammars — the `g.constraints.contact` generator + emit path call them.

Fork-only: `contactSurface`/`contact`/`contactPlane` are unavailable on stock
openseespy and bite only at run time. The explicit-only `-soft`/`-visc`, the
solver-coupled `-consistanttan`/`-geomtan`, the broad-phase `-cell` modifiers,
and the mortar-only edge-edge fallback (`-edgeedge` + the `-edge*` knobs,
ADR-57 E2–E7) are supported (ADR 0073), as is the rigid-plane `contactPlane`
command (`contact_plane_args`).
"""
from __future__ import annotations

from collections.abc import Sequence


__all__ = ["contact_surface_args", "contact_args", "contact_plane_args"]


def contact_surface_args(
    kind: str,
    node_tags: Sequence[int],
    nps: int = 0,
) -> list[int | float | str]:
    """Args **after** the surface tag for one `contactSurface` call.

    ``kind`` is ``"master"`` / ``"slave"`` / ``"slave-segments"``. The faceted
    forms (master / slave-segments) take ``nps`` (per-facet node count, 3=tri,
    4=quad) followed by the flat node connectivity; ``slave`` takes the node
    tags directly.
    """
    nodes = [int(n) for n in node_tags]
    if not nodes:
        raise ValueError("contactSurface: no node tags given")
    if kind == "slave":
        return ["-slave", *nodes]
    if kind == "master":
        _check_nps(nps, len(nodes))
        return ["-master", int(nps), *nodes]
    if kind == "slave-segments":
        _check_nps(nps, len(nodes))
        return ["-slave-segments", int(nps), *nodes]
    raise ValueError(
        f"contactSurface: kind must be 'master', 'slave' or 'slave-segments', "
        f"got {kind!r}")


def _check_nps(nps: int, n_nodes: int) -> None:
    if nps not in (3, 4):
        raise ValueError(
            f"contactSurface faceted surface: nps (nodes-per-facet) must be "
            f"3 (tri) or 4 (quad), got {nps} — higher-order surfaces must be "
            f"dropped to corner facets before emit")
    if n_nodes % nps != 0:
        raise ValueError(
            f"contactSurface faceted surface: {n_nodes} node tags is not a "
            f"multiple of nps={nps} (flat facet connectivity)")


def contact_args(
    master_tag: int,
    slave_tag: int,
    formulation: str,
    *,
    kn: float | str | None = None,
    kt: float | None = None,
    mu: float | None = None,
    eps_n: float | str | None = None,
    eps_t: float | str | None = None,
    cohesion: float | None = None,
    tau_max: float | None = None,
    aug_tol: float | None = None,
    max_aug: int | None = None,
    ngp: int | None = None,
    tie: bool = False,
    soft: float | bool | None = None,
    visc: float | None = None,
    consistent_tan: bool = False,
    geom_tan: bool = False,
    cell: float | None = None,
    edge_edge: bool = False,
    edge_kn: float | str | None = None,
    edge_band: float | None = None,
    edge_mu: float | None = None,
    edge_kt: float | None = None,
    edge_cohesion: float | None = None,
    edge_tau_max: float | None = None,
    edge_consistent_tan: bool = False,
    edge_soft: float | bool | None = None,
    edge_alm: bool = False,
    edge_aug_tol: float | None = None,
    outward: Sequence[float] | None = None,
) -> list[int | float | str]:
    """Args **after** the contact tag for one `contact` call.

    Returns ``[masterTag, slaveTag, <grammar>]`` — pass as
    ``emitter.contact(tag, *args)``.

    NTS emits ``kn [kt mu]`` (or ``auto``); the fork parser reads either 1 or 3
    numbers, so friction emits all three (kt/mu default 0.0). Mortar emits
    ``-mortar -epsN …`` + the friction-cone / augmentation flags.

    The extension modifiers (``-soft``/``-visc``/``-consistanttan``/
    ``-geomtan``) are parsed by the fork's order-independent option loop, so
    they emit after the formulation block (and before ``-outward``).
    ``soft=True`` emits a bare ``-soft`` (fork default SOFSCL 0.10); a float
    emits ``-soft SOFSCL``. ``geom_tan`` is NTS-only (the def enforces it).
    ``cell`` (``-cell frac``) is the broad-phase cell-size scale (both lanes).

    The edge-edge fallback (``edge_edge`` + the ``edge_*`` knobs, ADR-57 E2–E7)
    is mortar-only (the def enforces it). When ``edge_edge`` is set it emits
    ``-edgeedge`` followed by the requested edge knobs (``-edgeKn auto|<v>`` /
    ``-edgeBand`` / ``-edgeMu`` / ``-edgeKt`` / ``-edgeCohesion`` /
    ``-edgeTauMax`` / ``-edgeConsistentTan`` / ``-edgeSoft [SOFSCL]`` /
    ``-edgeAlm`` / ``-edgeAugTol``); ``edge_soft=True`` emits a bare
    ``-edgeSoft``. The edge knobs are dropped when ``edge_edge`` is False.
    """
    args: list[int | float | str] = [int(master_tag), int(slave_tag)]

    if formulation == "nts":
        if kn == "auto":
            # The ``auto`` path peeks-and-unreads a trailing flag safely, so
            # kt/mu only need emitting when friction is requested.
            args.append("auto")
            if kt is not None or mu is not None:
                args.append(float(kt) if kt is not None else 0.0)
                args.append(float(mu) if mu is not None else 0.0)
        elif kn is not None:
            # Numeric kn: ALWAYS emit the full ``kn kt mu`` triple (kt/mu
            # default 0.0 ⇒ frictionless). The fork's numeric kn-slot reader
            # (OPS_LadrunoContact) sizes its double read as
            # ``m = (remaining >= 3) ? 3 : 1`` counting ALL trailing tokens —
            # flags included. A bare numeric ``kn`` followed by ANY trailing
            # token (``-outward`` or an extension flag like ``-soft``/``-visc``)
            # makes it read that token as a double and abort the whole
            # ``contact`` command. Padding the triple is semantically identical
            # (kt=mu=0 is frictionless) and immune to which trailing tokens
            # follow — so we never have to keep this guard in sync with the set
            # of trailing options.
            args.append(float(kn))
            args.append(float(kt) if kt is not None else 0.0)
            args.append(float(mu) if mu is not None else 0.0)
    elif formulation == "mortar":
        args.append("-mortar")
        if eps_n is not None:
            args += ["-epsN", "auto" if eps_n == "auto" else float(eps_n)]
        if mu is not None:
            args += ["-mu", float(mu)]
        if eps_t is not None:
            args += ["-epsT", "auto" if eps_t == "auto" else float(eps_t)]
        if cohesion is not None:
            args += ["-cohesion", float(cohesion)]
        if tau_max is not None:
            args += ["-tauMax", float(tau_max)]
        if aug_tol is not None:
            args += ["-augTol", float(aug_tol)]
        if max_aug is not None:
            args += ["-maxAug", int(max_aug)]
        if ngp is not None:
            args += ["-ngp", int(ngp)]
        if tie:
            args.append("-tie")
    else:
        raise ValueError(
            f"contact: formulation must be 'nts' or 'mortar', got "
            f"{formulation!r}")

    # Extension modifiers — the fork's option loop reads them in either lane
    # order-independently (geom_tan is NTS-only, enforced on the def). A bare
    # `-soft` (soft=True) takes the fork's default SOFSCL (0.10); a numeric soft
    # emits `-soft SOFSCL`. The fork peeks-and-unreads the token after `-soft`,
    # so a following flag (-visc/-outward/…) is safe.
    if soft is not None and soft is not False:
        if soft is True:
            args.append("-soft")
        else:
            args += ["-soft", float(soft)]
    if visc is not None:
        args += ["-visc", float(visc)]
    if consistent_tan:
        args.append("-consistanttan")
    if geom_tan:
        args.append("-geomtan")
    # -cell <frac>: broad-phase spatial-hash cell scale (both lanes). The fork
    # reads exactly one double after it, so a following flag (-outward/…) is safe.
    if cell is not None:
        args += ["-cell", float(cell)]

    # Edge-edge fallback (ADR-57 E2–E7) — mortar-only (the def enforces it).
    # The fork's order-independent option loop reads these in any order; we emit
    # `-edgeedge` first (the convention) then the edge knobs. `-edgeSoft`
    # peeks-and-unreads its SOFSCL, so a following flag (-outward/…) is safe.
    if edge_edge:
        args.append("-edgeedge")
        if edge_kn is not None:
            args += ["-edgeKn", "auto" if edge_kn == "auto" else float(edge_kn)]
        if edge_band is not None:
            args += ["-edgeBand", float(edge_band)]
        if edge_mu is not None:
            args += ["-edgeMu", float(edge_mu)]
        if edge_kt is not None:
            args += ["-edgeKt", float(edge_kt)]
        if edge_cohesion is not None:
            args += ["-edgeCohesion", float(edge_cohesion)]
        if edge_tau_max is not None:
            args += ["-edgeTauMax", float(edge_tau_max)]
        if edge_consistent_tan:
            args.append("-edgeConsistentTan")
        if edge_soft is not None and edge_soft is not False:
            if edge_soft is True:
                args.append("-edgeSoft")
            else:
                args += ["-edgeSoft", float(edge_soft)]
        if edge_alm:
            args.append("-edgeAlm")
        if edge_aug_tol is not None:
            args += ["-edgeAugTol", float(edge_aug_tol)]

    if outward is not None:
        if len(outward) != 3:
            raise ValueError(
                f"contact -outward: need (ox, oy, oz), got {outward!r}")
        args += ["-outward", *(float(x) for x in outward)]

    return args


def contact_plane_args(
    slave_tag: int,
    normal: Sequence[float],
    point: Sequence[float],
    kn: float,
    *,
    visc: float | None = None,
    soft: float | bool | None = None,
) -> list[int | float | str]:
    """Args **after** the contactPlane tag for one `contactPlane` call.

    Returns ``[slaveSurfTag, nx, ny, nz, px, py, pz, kn, <flags>]`` — pass as
    ``emitter.contact_plane(tag, *args)``. The fork grammar is::

        contactPlane tag slaveSurfTag nx ny nz px py pz kn [-visc μ] [-soft S]

    A meshed slave surface (the `slaveSurfTag` contactSurface) contacts a fixed
    rigid plane (`normal` + `point`) with normal penalty `kn` — frictionless, no
    master. `kn` is a plain value (no ``"auto"``). The optional `-visc` /
    `-soft` modifiers mirror the `contact` extension knobs (`-soft` peeks-and-
    unreads its SOFSCL, so it is safe before / after `-visc`).
    """
    n = tuple(float(x) for x in normal)
    p = tuple(float(x) for x in point)
    if len(n) != 3 or len(p) != 3:
        raise ValueError(
            f"contactPlane: normal/point must be 3-vectors, got "
            f"normal={normal!r}, point={point!r}")
    args: list[int | float | str] = [
        int(slave_tag), n[0], n[1], n[2], p[0], p[1], p[2], float(kn),
    ]
    if visc is not None:
        args += ["-visc", float(visc)]
    if soft is not None and soft is not False:
        if soft is True:
            args.append("-soft")
        else:
            args += ["-soft", float(soft)]
    return args
