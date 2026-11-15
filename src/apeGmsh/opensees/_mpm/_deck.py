"""The particle set a deck's ``mpm`` commands describe (ADR 0134 D1, D2, D6).

The bridge's emit and the replay from ``model.h5`` both reach the particle set
through the same thing: the ``mpm`` command rows, as the argument tuples after
the ``mpm`` token (``("grid", 7, "-origin", 0.0, 0.0, ...)``,
``("body", "soil", "-pg", "Soil", "-mat", 1, "-rho", 1800.0, "-ppc", 2)``).
:func:`plan_from_commands` reads the grid, the bodies and the units from those
rows; :func:`prepare_deck_particle_set` seeds the plan on a ``FEMData`` and runs
every refusal of the writer, writing nothing. One reader of the rows means the
bridge's deck and its replay seed the same particles (D2).

Charter P3/P9: stdlib, numpy and this package only.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from ._particle_set import (
    ParticleSetError,
    ParticleSetOverwriteRefused,
    PreparedParticleSet,
    Provenance,
    Units,
    check_overwrite,
    particle_set_path,
    preflight_particle_set,
    py_particles_line,
    tcl_particles_line,
)
from ._seeding import MPMBody, MPMGrid, seed_lattice

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apeGmsh.mesh.FEMData import FEMData

__all__ = [
    "MPMDeckError",
    "MPMDeckPlan",
    "PLANE_STRAIN_THICKNESS",
    "StaleParticleSetWarning",
    "plan_from_commands",
    "prepare_deck_particle_set",
    "remove_stale_particle_set",
    "resolve_units",
    "stale_particle_set",
    "write_deck_then_remove",
]

#: The fork's plane-strain convention (guide, "A minimal run": ``ndm 2 = plane
#: strain (thickness 1)``). The engine adopts a loaded set's thickness, so the
#: set states 1 and the deck's 2-D masses are per unit thickness.
PLANE_STRAIN_THICKNESS = 1.0

Arg = Any  # int | float | str, as the command channel carries them


class MPMDeckError(ValueError):
    """``mpm`` command rows that do not describe one particle set."""


@dataclass(frozen=True)
class MPMDeckPlan:
    """What the ``mpm`` rows say about the particle set."""

    grid: MPMGrid
    bodies: tuple[MPMBody, ...]
    material_tags: dict[str, int]
    ppc: int
    units: Units | None


def _flag_values(args: Sequence[Arg], flag: str, sub: str) -> list[Arg]:
    """The values after ``flag`` up to the next ``-flag`` token."""
    try:
        i = list(args).index(flag)
    except ValueError:
        raise MPMDeckError(f"mpm {sub}: no {flag} in {tuple(args)!r}") from None
    out: list[Arg] = []
    for a in args[i + 1:]:
        if isinstance(a, str) and a.startswith("-"):
            break
        out.append(a)
    if not out:
        raise MPMDeckError(f"mpm {sub}: {flag} has no value in {tuple(args)!r}")
    return out


def _one(args: Sequence[Arg], flag: str, sub: str) -> Arg:
    vals = _flag_values(args, flag, sub)
    if len(vals) != 1:
        raise MPMDeckError(f"mpm {sub}: {flag} takes one value, got {vals!r}")
    return vals[0]


def _grid(args: Sequence[Arg]) -> MPMGrid:
    origin = tuple(float(v) for v in _flag_values(args, "-origin", "grid"))
    h = float(_one(args, "-h", "grid"))
    cells = tuple(int(v) for v in _flag_values(args, "-cells", "grid"))
    basis = str(_one(args, "-basis", "grid")) if "-basis" in args else "gimp"
    return MPMGrid(origin=origin, cell_size=(h,) * len(origin), num_cells=cells,
                   basis=basis)


def plan_from_commands(commands: Sequence[Sequence[Arg]]) -> MPMDeckPlan:
    """Read the particle set from the ``mpm`` rows (the args after ``mpm``).

    Refused, by name: no grid or two grids, no body, a repeated body name,
    bodies that disagree on ``-ppc``, two ``units`` rows, and a ``particles``
    row (the ``-file`` line is never a row, D1).
    """
    grids: list[MPMGrid] = []
    bodies: list[MPMBody] = []
    tags: dict[str, int] = {}
    ppcs: dict[str, int] = {}
    units: list[Units] = []
    for row in commands:
        args = list(row)
        if not args or not isinstance(args[0], str):
            raise MPMDeckError(f"an mpm row needs a sub-command, got {tuple(row)!r}")
        sub = args[0]
        if sub == "grid":
            grids.append(_grid(args))
        elif sub == "body":
            if len(args) < 2 or not isinstance(args[1], str):
                raise MPMDeckError(f"mpm body: no name in {tuple(row)!r}")
            name = args[1]
            if name in tags:
                raise MPMDeckError(f"mpm body {name!r} is declared twice")
            bodies.append(MPMBody(name=name, pg=str(_one(args, "-pg", "body")),
                                  density=float(_one(args, "-rho", "body"))))
            tags[name] = int(_one(args, "-mat", "body"))
            ppcs[name] = int(_one(args, "-ppc", "body"))
        elif sub == "units":
            if len(args) < 4:
                raise MPMDeckError(f"mpm units needs L F T, got {tuple(row)!r}")
            system = str(_one(args, "-system", "units")) if "-system" in args else None
            units.append(Units(length=str(args[1]), force=str(args[2]),
                               time=str(args[3]), system=system))
        elif sub == "particles":
            raise MPMDeckError("an mpm particles row: the particle set's -file "
                               "line is written by each deck emitter for its own "
                               "target, never stored as a row (ADR 0134 D1)")
    if len(grids) != 1:
        raise MPMDeckError(f"the mpm rows declare {len(grids)} grids; a particle "
                           "set needs exactly one")
    if not bodies:
        raise MPMDeckError("the mpm rows declare no body; a particle set needs one")
    if len(set(ppcs.values())) != 1:
        raise MPMDeckError(f"the mpm bodies disagree on -ppc ({ppcs}); one particle "
                           "set has one lattice")
    if len(units) > 1:
        raise MPMDeckError("the mpm rows declare units twice")
    return MPMDeckPlan(grid=grids[0], bodies=tuple(bodies), material_tags=tags,
                       ppc=next(iter(ppcs.values())),
                       units=units[0] if units else None)


def resolve_units(declared: Units | None, model: Units | None) -> Units:
    """The particle set's units: the ``mpm units`` row, else the model's.

    Refused, by name: neither (Contract 1 requires unit symbols, and the deck
    has no other source), or both with different symbols (the fork refuses a
    ``mpm units`` that differs from a loaded set's).
    """
    if declared is None and model is None:
        raise MPMDeckError(
            "the particle set needs unit symbols (Contract 1 UNIT_LENGTH, "
            "UNIT_FORCE, UNIT_TIME): declare ops.mpm.units(L, F, T) or "
            "construct the bridge with units=")
    if declared is None:
        assert model is not None
        return model
    if model is not None and (declared.length, declared.force, declared.time) != (
            model.length, model.force, model.time):
        raise MPMDeckError(
            f"ops.mpm.units({declared.length!r}, {declared.force!r}, "
            f"{declared.time!r}) differs from the model's units "
            f"({model.length!r}, {model.force!r}, {model.time!r})")
    return declared


def prepare_deck_particle_set(
    fem: "FEMData",
    plan: MPMDeckPlan,
    deck_path: str | os.PathLike[str],
    *,
    units: Units,
    provenance: Provenance,
) -> PreparedParticleSet:
    """Seed ``plan`` on ``fem`` and run every refusal of the writer for the
    set beside ``deck_path`` (``<deck stem>.particles.h5``). Writes nothing:
    the caller writes the set and its deck together (D1)."""
    thickness = PLANE_STRAIN_THICKNESS if plan.grid.ndm == 2 else None
    seeded = seed_lattice(fem, plan.grid, plan.bodies, n=plan.ppc,
                          thickness=thickness)
    return preflight_particle_set(
        particle_set_path(deck_path), seeded, material_tags=plan.material_tags,
        units=units, provenance=provenance)


# ---------------------------------------------------------------------------
# a deck with no mpm members (ADR 0134 D1: never pair a new deck with a stale set)
# ---------------------------------------------------------------------------

class StaleParticleSetWarning(UserWarning):
    """A deck with no ``mpm`` members is emitted beside a ``<stem>.particles.h5``
    that it keeps: a file no apeGmsh emit wrote, or an apeGmsh set the deck it
    overwrites does not read."""


def stale_particle_set(deck_path: str | os.PathLike[str]) -> Path | None:
    """Decide, before a deck with no ``mpm`` members is written over
    ``deck_path``, whether the ``<deck stem>.particles.h5`` beside it goes
    with the deck it replaces (ADR 0134 D1: never pair a new deck with a
    stale set). Removes nothing.

    Returns the set's path only when both hold: the set carries this
    contract's apeGmsh provenance (the :func:`check_overwrite` rule,
    ``FORMAT="ladruno-mpm-particles"`` and ``GENERATOR="apeGmsh"``), and the
    deck being overwritten exists and its own ``mpm particles -file`` line
    names that set. The caller removes it once the new deck is written and
    in place (:func:`write_deck_then_remove`, :func:`remove_stale_particle_set`).

    Returns ``None`` when nothing is there. Any other file there is kept,
    with a :class:`StaleParticleSetWarning` naming it: a foreign file, and
    an apeGmsh set that the overwritten deck does not read (no deck there,
    or one that reads another set or none), which a sibling deck of the
    same stem (``case.tcl`` beside ``case.py``) may read.
    """
    try:
        path = particle_set_path(deck_path)
    except ParticleSetError:
        return None
    if not os.path.lexists(path):
        return None
    try:
        check_overwrite(path)
    except ParticleSetOverwriteRefused as exc:
        _warn_kept(deck_path, path,
                   f"it is not a particle set an apeGmsh emit wrote ({exc})")
        return None
    if not _deck_reads_set(deck_path, path):
        _warn_kept(deck_path, path,
                   "it is an apeGmsh particle set, but the deck being "
                   "overwritten does not read it (no deck there, or its mpm "
                   "particles -file line names no such set), so another deck "
                   "of this stem may")
        return None
    return path


def _deck_reads_set(deck_path: str | os.PathLike[str], set_path: Path) -> bool:
    """Whether the deck file at ``deck_path`` holds the ``mpm particles
    -file`` line (Tcl or Python form) that names ``set_path``."""
    deck = Path(deck_path)
    if not deck.is_file():
        return False
    lines = {tcl_particles_line(set_path), py_particles_line(set_path)}
    with open(deck, encoding="utf-8", errors="replace") as f:
        return any(line.strip() in lines for line in f)


def _warn_kept(deck_path: str | os.PathLike[str], path: Path, why: str) -> None:
    warnings.warn(
        f"the deck {str(deck_path)!r} declares no mpm members, and "
        f"{str(path)!r} beside it is kept: {why}. A viewer pairing "
        "'<deck stem>.particles.h5' with this deck would read it (ADR 0134 "
        "D1); move or delete it.",
        StaleParticleSetWarning, stacklevel=4)


def write_deck_then_remove(
    deck_path: str | os.PathLike[str], write: Callable[[TextIO], Any],
    stale: Path | None,
) -> None:
    """Write the deck with ``write(f)``, then remove ``stale``.

    ``stale is None``: the deck is written in place, as before. Otherwise
    the deck goes to a sibling temporary file and replaces ``deck_path``
    only once it is whole, and the set is removed only after that, so a
    failed write leaves the old deck and its set as they were.
    """
    if stale is None:
        with open(deck_path, "w", encoding="utf-8") as f:
            write(f)
        return
    deck = Path(deck_path)
    tmp = deck.with_name(deck.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            write(f)
        os.replace(tmp, deck)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    remove_stale_particle_set(stale)


def remove_stale_particle_set(path: Path | None) -> None:
    """Remove the apeGmsh set :func:`stale_particle_set` chose (``None``:
    nothing), once the deck that replaced its reader is in place."""
    if path is not None:
        os.remove(path)
