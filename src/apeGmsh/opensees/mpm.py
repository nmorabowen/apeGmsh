"""The ``mpm`` primitive family: the fork's MPM engine as bridge declarations.

ADR 0134 (D1, D2, D6-D8). The Ladruno fork drives a Material Point Method run
with one ``mpm`` command and its sub-commands (fork guide
``Ladruno_implementation/232_ladruno_mpm_guide.md``, parser
``SRC/domain/pattern/ladrunoMPM/OPS_LadrunoMPM.cpp``). Each member here is
one of those sub-commands, named after it, and declared through
``ops.mpm.<member>(...)``:

========================  ===================================================
member                    fork line
========================  ===================================================
:class:`Grid`             ``mpm grid <tag> -origin <x0..> -h <h> -cells <n..>
                          [-basis gimp|linear]``
:class:`Body`             no line of its own: seeded into the deck's particle
                          set (``mpm particles -file``), see below
:class:`Fix`              ``mpm fix <face> <components>``
:class:`Wall`             ``mpm wall <face> <mu>``
:class:`Gravity`          ``mpm gravity <g..>``
:class:`TimeSeries`       ``mpm timeSeries <tsTag>``
:class:`Damping`          ``mpm damping local <alpha>``
:class:`Flip`             ``mpm flip <alpha_F>``
:class:`MassCutoff`       ``mpm massCutoff <c_m>``
:class:`Safety`           ``mpm safety <s>``
:class:`Units`            ``mpm units <L> <F> <T> [-system <id>]``
:class:`Output`           ``mpm output <path> [-every N | -dt dt] [-fields ..]
                          [-compress k] [-flush s]``
:class:`Analyze`          ``mpm analyze <n> <dt>``
========================  ===================================================

Every member emits through ``command("mpm", ...)``, so each declaration is a
``/opensees/commands`` row of ``model.h5`` (ADR 0134 D2, ADR 0114). A
:class:`Body` row (``mpm body <name> -pg <pg> -mat <tag> -rho <rho> -ppc <n>``)
is apeGmsh's, not the fork's: the deck emitters write it as a comment and
write, after the grid, the ``mpm particles -file`` line of the particle set
``apeSees.tcl(path)`` / ``apeSees.py(path)`` seed beside the deck (D1). The
set's ``MATERIAL_TAG`` is the body material's allocated ``nDMaterial`` tag
(D6); users never type a tag.

The grid's tag is the fork's load-pattern tag (``mpm grid <tag>`` creates
load pattern ``<tag>``). It is allocated in the ``mpm`` kind and written
above every ``pattern`` tag and every other member's ``mpm`` tag of the
model (:func:`host_tag`); ``apeSees.tag_for(grid)`` answers that tag.

MPM is deck-only (D1, D7): the live route refuses any model that declares a
member, before the domain is touched.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._internal.tag_resolution import resolve_tag
from ._internal.types import MPM, NDMaterial, Primitive
from ._internal.types import TimeSeries as _TimeSeriesPrimitive

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData
    from ._mpm import PreparedParticleSet
    from ._mpm import Units as _ParticleUnits
    from .apesees import apeSees
    from .emitter.base import Emitter


__all__ = [
    "MPM",
    "ALL_MPM",
    "Analyze",
    "Body",
    "Damping",
    "Fix",
    "Flip",
    "Gravity",
    "Grid",
    "MassCutoff",
    "Output",
    "Safety",
    "TimeSeries",
    "Units",
    "Wall",
    "MPMError",
    "OUTPUT_FIELDS",
]

#: Grid faces, per spatial dimension, in the fork's ``faceIndex`` order.
FACES: dict[int, tuple[str, ...]] = {
    2: ("xmin", "xmax", "ymin", "ymax"),
    3: ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
}
BASES = ("gimp", "linear")
#: The particle fields ``mpm output -fields`` admits in v1 (fork guide,
#: "Commands"); no ``STATE_*`` token is admitted.
OUTPUT_FIELDS = ("POSITION", "VELOCITY", "STRESS", "STRAIN", "VOLUME",
                 "GIMP_HALF_LENGTH", "DEFORMATION_GRADIENT")


class MPMError(ValueError):
    """An ``mpm`` declaration, or a set of them, the bridge refuses."""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise MPMError(f"{name} must be a number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise MPMError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise MPMError(f"{name} must be finite, got {value!r}")
    return out


def _int(name: str, value: int, *, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise MPMError(f"{name} must be an integer >= {low}, got {value!r}")
    return int(value)


def _label(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MPMError(f"{name} must be a non-empty string, got {value!r}")
    if any(ch in value for ch in "{}\n\r"):
        raise MPMError(f"{name} {value!r} carries a brace or a line break, "
                       "which an emitted deck cannot quote")
    return value


def _word(name: str, value: str) -> str:
    """A token the deck writes bare into an ``mpm`` line: one word.

    The fork's parser reads one Tcl word per argument, so a value with
    whitespace would be split: the engine keeps the first word and drops
    the rest, while the particle set records the whole string.
    """
    _label(name, value)
    if any(ch.isspace() for ch in value):
        raise MPMError(f"{name} {value!r} must be one word (no whitespace): "
                       "the deck writes it bare into an mpm line, where the "
                       "fork's parser would split it")
    return value


def _face(value: str) -> str:
    if value not in FACES[3]:
        raise MPMError(f"face must be one of {FACES[3]}, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# members
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Grid(MPM):
    """``mpm grid <tag> -origin <x0..> -h <h> -cells <n..> [-basis gimp|linear]``.

    The background grid, one per model (the fork's v1: one per Domain). Its
    spatial dimension is ``len(origin)``, 2 (plane strain, thickness 1) or 3.
    ``basis=None`` emits no ``-basis`` and keeps the fork's default, uGIMP.
    """

    origin: tuple[float, ...]
    h: float
    cells: tuple[int, ...]
    basis: str | None = None

    def __post_init__(self) -> None:
        origin = tuple(_finite("Grid.origin", v) for v in self.origin)
        if len(origin) not in (2, 3):
            raise MPMError(f"Grid.origin needs 2 or 3 coordinates, got {origin}")
        cells = tuple(_int("Grid.cells", c, low=1) for c in self.cells)
        if len(cells) != len(origin):
            raise MPMError(f"Grid.cells needs {len(origin)} counts (one per axis "
                           f"of origin), got {cells}")
        h = _finite("Grid.h", self.h)
        if h <= 0.0:
            raise MPMError(f"Grid.h must be > 0, got {self.h!r}")
        if self.basis is not None and self.basis not in BASES:
            raise MPMError(f"Grid.basis must be one of {BASES} or None, got "
                           f"{self.basis!r}")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "h", h)

    @property
    def ndm(self) -> int:
        return len(self.origin)

    def args(self, tag: int) -> tuple[int | float | str, ...]:
        out: list[int | float | str] = [
            "grid", int(tag), "-origin", *self.origin, "-h", self.h,
            "-cells", *self.cells]
        if self.basis is not None:
            out += ["-basis", self.basis]
        return tuple(out)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        emitter.command("mpm", *self.args(tag))

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Body(MPM):
    """One MPM body: the elements of physical group ``pg``, lattice-seeded.

    ``ppc`` particles per cell per axis are placed at the grid's sub-cell
    centres and kept where they fall inside an element of ``pg`` (ADR 0134
    D2). Each particle carries ``density`` and the allocated tag of
    ``material`` (D6). ``name`` is the body's name in the particle set and
    in the results (``results.particles.get(body=...)``).

    The archive row is ``mpm body <name> -pg <pg> -mat <tag> -rho <rho>
    -ppc <n>``; a deck carries it as a comment, beside the particle set's
    ``mpm particles -file`` line, because the fork's own ``mpm body``
    seeds boxes, discs and balls only.
    """

    name: str
    pg: str
    material: NDMaterial
    density: float
    ppc: int

    def __post_init__(self) -> None:
        _label("Body.name", self.name)
        _label("Body.pg", self.pg)
        if not isinstance(self.material, NDMaterial):
            raise MPMError(f"Body.material must be an nDMaterial, got "
                           f"{type(self.material).__name__}")
        rho = _finite("Body.density", self.density)
        if rho <= 0.0:
            raise MPMError(f"Body.density must be > 0, got {self.density!r}")
        object.__setattr__(self, "density", rho)
        object.__setattr__(self, "ppc", _int("Body.ppc", self.ppc, low=1))

    def args(self, material_tag: int) -> tuple[int | float | str, ...]:
        return ("body", self.name, "-pg", self.pg, "-mat", int(material_tag),
                "-rho", self.density, "-ppc", self.ppc)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args(resolve_tag(emitter, self.material)))

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)


@dataclass(frozen=True, kw_only=True, slots=True)
class Fix(MPM):
    """``mpm fix <face> <components>``: zero the listed grid-velocity
    components on a grid face (``"y"`` on ``ymin`` is a roller)."""

    face: str
    components: str

    def __post_init__(self) -> None:
        _face(self.face)
        c = self.components
        if (not isinstance(c, str) or not c or any(ch not in "xyz" for ch in c)
                or len(set(c)) != len(c)):
            raise MPMError(f"Fix.components must be a non-empty string over "
                           f"x, y, z without repeats, got {c!r}")

    def args(self) -> tuple[int | float | str, ...]:
        return ("fix", self.face, self.components)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Wall(MPM):
    """``mpm wall <face> <mu>``: a stationary rigid wall with Coulomb
    friction ``mu >= 0`` on a grid face."""

    face: str
    mu: float

    def __post_init__(self) -> None:
        _face(self.face)
        mu = _finite("Wall.mu", self.mu)
        if mu < 0.0:
            raise MPMError(f"Wall.mu must be >= 0, got {self.mu!r}")
        object.__setattr__(self, "mu", mu)

    def args(self) -> tuple[int | float | str, ...]:
        return ("wall", self.face, self.mu)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Gravity(MPM):
    """``mpm gravity <g..>``: the body force per unit mass, signed, one
    component per grid axis."""

    g: tuple[float, ...]

    def __post_init__(self) -> None:
        g = tuple(_finite("Gravity.g", v) for v in self.g)
        if len(g) not in (2, 3):
            raise MPMError(f"Gravity.g needs 2 or 3 components, got {g}")
        object.__setattr__(self, "g", g)

    def args(self) -> tuple[int | float | str, ...]:
        return ("gravity", *self.g)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeSeries(MPM):
    """``mpm timeSeries <tsTag>``: the factor applied to gravity (default 1)."""

    series: _TimeSeriesPrimitive

    def __post_init__(self) -> None:
        if not isinstance(self.series, _TimeSeriesPrimitive):
            raise MPMError(f"TimeSeries.series must be a time series, got "
                           f"{type(self.series).__name__}")

    def args(self, series_tag: int) -> tuple[int | float | str, ...]:
        return ("timeSeries", int(series_tag))

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args(resolve_tag(emitter, self.series)))

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.series,)


@dataclass(frozen=True, kw_only=True, slots=True)
class Damping(MPM):
    """``mpm damping local <alpha>``: local non-viscous damping,
    ``0 <= alpha < 1``, for quasi-static runs."""

    alpha: float

    def __post_init__(self) -> None:
        a = _finite("Damping.alpha", self.alpha)
        if not 0.0 <= a < 1.0:
            raise MPMError(f"Damping.alpha must be in [0, 1), got {self.alpha!r}")
        object.__setattr__(self, "alpha", a)

    def args(self) -> tuple[int | float | str, ...]:
        return ("damping", "local", self.alpha)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


def _unit_interval(cls: str, field: str, value: float, *, low_open: bool) -> float:
    v = _finite(f"{cls}.{field}", value)
    ok = (0.0 < v <= 1.0) if low_open else (0.0 <= v <= 1.0)
    if not ok:
        span = "(0, 1]" if low_open else "[0, 1]"
        raise MPMError(f"{cls}.{field} must be in {span}, got {value!r}")
    return v


@dataclass(frozen=True, kw_only=True, slots=True)
class Flip(MPM):
    """``mpm flip <alpha_F>``: the FLIP fraction, 1 pure FLIP (the fork's
    default), 0 pure PIC."""

    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _unit_interval(
            "Flip", "alpha", self.alpha, low_open=False))

    def args(self) -> tuple[int | float | str, ...]:
        return ("flip", self.alpha)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class MassCutoff(MPM):
    """``mpm massCutoff <c_m>``: a grid node is active iff its mass exceeds
    ``c_m * max m_p`` (fork default 1e-10)."""

    c: float

    def __post_init__(self) -> None:
        c = _finite("MassCutoff.c", self.c)
        if c < 0.0:
            raise MPMError(f"MassCutoff.c must be >= 0, got {self.c!r}")
        object.__setattr__(self, "c", c)

    def args(self) -> tuple[int | float | str, ...]:
        return ("massCutoff", self.c)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Safety(MPM):
    """``mpm safety <s>``: a step is refused iff ``dt > s * dt_c``,
    ``0 < s <= 1`` (fork default 0.5)."""

    s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", _unit_interval(
            "Safety", "s", self.s, low_open=True))

    def args(self) -> tuple[int | float | str, ...]:
        return ("safety", self.s)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Units(MPM):
    """``mpm units <L> <F> <T> [-system <id>]``: the unit symbols of the
    particle set and of the ``.ladruno`` group.

    The particle set carries them too (Contract 1), and the fork refuses a
    ``mpm units`` that differs from a loaded set's, so the deck emitters
    seed the set with these. Without this member the set takes the
    bridge's ``units=``; with neither, the deck is refused.
    """

    length: str
    force: str
    time: str
    system: str | None = None

    def __post_init__(self) -> None:
        for field in ("length", "force", "time"):
            _word(f"Units.{field}", getattr(self, field))
        if self.system is not None:
            _word("Units.system", self.system)

    def args(self) -> tuple[int | float | str, ...]:
        out: list[int | float | str] = ["units", self.length, self.force, self.time]
        if self.system is not None:
            out += ["-system", self.system]
        return tuple(out)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Output(MPM):
    """``mpm output <path> [-every N | -dt dt] [-fields F..] [-compress k]
    [-flush s]``: the particle results, a ``.ladruno`` file of the fork's
    own (Contract 2). The fork opens ``path`` as given (relative to the
    working directory of the run). ``fields=None`` keeps the fork's default
    (POSITION VELOCITY STRESS VOLUME); POSITION is always written.
    """

    path: str
    every: int | None = None
    dt: float | None = None
    fields: tuple[str, ...] | None = None
    compress: int | None = None
    flush: float | None = None

    def __post_init__(self) -> None:
        _label("Output.path", self.path)
        if self.every is not None and self.dt is not None:
            raise MPMError("Output: every= and dt= are exclusive (the fork "
                           "keeps one output cadence)")
        if self.every is not None:
            _int("Output.every", self.every, low=1)
        if self.dt is not None:
            dt = _finite("Output.dt", self.dt)
            if dt <= 0.0:
                raise MPMError(f"Output.dt must be > 0, got {self.dt!r}")
            object.__setattr__(self, "dt", dt)
        if self.fields is not None:
            fields = tuple(self.fields)
            bad = [f for f in fields if f not in OUTPUT_FIELDS]
            if not fields or bad:
                raise MPMError(f"Output.fields must be a non-empty subset of "
                               f"{OUTPUT_FIELDS} (no STATE_* token in v1), got "
                               f"{self.fields!r}")
            object.__setattr__(self, "fields", fields)
        if self.compress is not None:
            k = _int("Output.compress", self.compress, low=0)
            if k > 9:
                raise MPMError(f"Output.compress must be in 0..9, got {k}")
        if self.flush is not None:
            object.__setattr__(self, "flush", _finite("Output.flush", self.flush))

    def args(self) -> tuple[int | float | str, ...]:
        out: list[int | float | str] = ["output", self.path]
        if self.every is not None:
            out += ["-every", self.every]
        if self.dt is not None:
            out += ["-dt", self.dt]
        if self.fields is not None:
            out += ["-fields", *self.fields]
        if self.compress is not None:
            out += ["-compress", self.compress]
        if self.flush is not None:
            out += ["-flush", self.flush]
        return tuple(out)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class Analyze(MPM):
    """``mpm analyze <n> <dt>``: ``n`` explicit steps of size ``dt``."""

    steps: int
    dt: float

    def __post_init__(self) -> None:
        _int("Analyze.steps", self.steps, low=0)
        dt = _finite("Analyze.dt", self.dt)
        if dt <= 0.0:
            raise MPMError(f"Analyze.dt must be > 0, got {self.dt!r}")
        object.__setattr__(self, "dt", dt)

    def args(self) -> tuple[int | float | str, ...]:
        return ("analyze", self.steps, self.dt)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.command("mpm", *self.args())

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


#: The family's contract list (``tests/families.py``).
ALL_MPM: tuple[type[MPM], ...] = (
    Grid, Body, Fix, Wall, Gravity, TimeSeries, Damping, Flip, MassCutoff,
    Safety, Units, Output, Analyze,
)


# ---------------------------------------------------------------------------
# the block a model's members emit as (ADR 0134 D1, D8)
# ---------------------------------------------------------------------------

def mpm_block(members: Sequence[MPM], *, ndm: int) -> tuple[MPM, ...]:
    """The members in their deck order, after the model-wide refusals.

    The grid first (the fork reads every other sub-command against it), then
    the bodies (the deck's particle set follows the grid), then the rest in
    declaration order. Refused, by name: no grid or two grids, no body, a
    grid whose dimension is not the model's ``ndm``, bodies that disagree on
    ``ppc`` (one particle set has one lattice), a repeated body name, a
    z face or a gravity of the wrong length for the grid, two ``Units``.
    """
    grids = [m for m in members if isinstance(m, Grid)]
    bodies = [m for m in members if isinstance(m, Body)]
    rest = [m for m in members if not isinstance(m, (Grid, Body))]
    for m in members:
        if not isinstance(m, ALL_MPM):
            raise MPMError(f"{type(m).__name__} is not a member of the mpm family "
                           f"({', '.join(c.__name__ for c in ALL_MPM)})")
    if len(grids) != 1:
        raise MPMError(f"an MPM model declares exactly one ops.mpm.grid (the "
                       f"fork's v1: one per Domain); this one declares "
                       f"{len(grids)}")
    if not bodies:
        raise MPMError("an MPM model declares at least one ops.mpm.body; "
                       "this one has a grid and no body, so its particle "
                       "set would be empty")
    grid = grids[0]
    if grid.ndm != ndm:
        raise MPMError(f"the mpm grid is {grid.ndm}-D but ops.model declares "
                       f"ndm={ndm}")
    ppcs = sorted({b.ppc for b in bodies})
    if len(ppcs) != 1:
        raise MPMError(f"every mpm body must share one ppc (one particle set "
                       f"has one lattice), got {ppcs} for "
                       f"{[b.name for b in bodies]}")
    names = [b.name for b in bodies]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise MPMError(f"mpm body names must be unique; repeated: {dup}")
    if sum(isinstance(m, Units) for m in rest) > 1:
        raise MPMError("an MPM model declares ops.mpm.units at most once")
    for m in rest:
        if isinstance(m, (Fix, Wall)) and m.face not in FACES[grid.ndm]:
            raise MPMError(f"mpm {type(m).__name__.lower()} on face {m.face!r}: "
                           f"a {grid.ndm}-D grid has faces {FACES[grid.ndm]}")
        if isinstance(m, Fix) and any(ord(ch) - ord("x") >= grid.ndm
                                      for ch in m.components):
            raise MPMError(f"mpm fix components {m.components!r}: a "
                           f"{grid.ndm}-D grid has components "
                           f"{'xyz'[:grid.ndm]!r}")
        if isinstance(m, Gravity) and len(m.g) != grid.ndm:
            raise MPMError(f"mpm gravity {m.g}: a {grid.ndm}-D grid takes "
                           f"{grid.ndm} components")
    return (grid, *bodies, *rest)


def host_tag(
    pattern_tags: "Iterable[int | None]", member_tags: "Iterable[int | None]",
) -> int:
    """The tag the deck writes for ``mpm grid`` (ADR 0134): one above every
    ``pattern`` tag and every other ``mpm`` member's tag.

    ``mpm grid <tag>`` creates load pattern ``<tag>`` (fork guide,
    "Commands"), and the allocator keeps one counter per kind, so the grid's
    own ``mpm`` tag would meet ``pattern`` 1 in any deck that has a pattern.
    Every load pattern a deck writes is a registered ``Pattern`` (the tag
    plan mints none), so a tag above all of theirs is never a pattern's. It
    is also above the other members' ``mpm`` tags, because the grid's row in
    the archive's declaration table is ``("mpm", <this tag>)``, beside
    theirs, and a ``(kind, tag)`` has one declaration. ``apeSees.tag_for``,
    the declaration table and the emit all read the grid's tag from here,
    so a ``pattern`` or ``mpm`` member declared later moves it.
    """
    taken = [int(t) for t in (*pattern_tags, *member_tags) if t is not None]
    return max(taken, default=0) + 1


def prepare_particle_set(
    fem: "FEMData",
    commands: Sequence[Sequence[int | float | str]],
    deck_path: str,
    *,
    model_units: str | None,
) -> "PreparedParticleSet":
    """Seed the particle set the ``mpm`` rows describe, for the deck at
    ``deck_path``, and run every refusal; write nothing (ADR 0134 D1, D2).

    ``commands`` are the rows after the ``mpm`` token, as the bridge emits
    them or ``model.h5`` stores them. The set's units are the ``mpm units``
    row's, else ``model_units`` (an apeGmsh unit-system id). Its provenance
    is the in-memory lineage of ``fem``: its ``session_id`` and
    ``snapshot_id`` (the ``fem_hash``), never a ``model.h5`` on disk, which
    may not exist yet or may hold another run's lineage.
    """
    from apeGmsh import __version__

    from ._mpm import (
        Provenance,
        plan_from_commands,
        prepare_deck_particle_set,
        resolve_units,
    )

    plan = plan_from_commands(commands)
    units = resolve_units(plan.units, _units_of_system(model_units))
    fem_hash = str(fem.snapshot_id)
    provenance = Provenance(
        session_id=str(fem.session_id),
        apegmsh_version=str(__version__),
        apegmsh_commit="unknown",
        fem_hash=fem_hash or None,
        model_hash=None,
    )
    return prepare_deck_particle_set(fem, plan, deck_path, units=units,
                                     provenance=provenance)


def _units_of_system(system: str | None) -> "_ParticleUnits | None":
    """The Contract-1 unit symbols of an apeGmsh unit-system id."""
    if system is None:
        return None
    from apeGmsh.units import unit_spec

    from ._mpm import Units as _Units

    spec = unit_spec(system)
    return _Units(length=spec.length_symbol, force=spec.force_symbol,
                  time=spec.time_symbol, system=str(spec.id))


# ---------------------------------------------------------------------------
# ops.mpm
# ---------------------------------------------------------------------------

class _MPMNS:
    """``ops.mpm.<member>(...)``: declare one ``mpm`` sub-command.

    Every argument is checked before the member is registered, so a refused
    call leaves no primitive, no tag and no declaration behind.
    """

    __slots__ = ("_bridge",)

    def __init__(self, bridge: "apeSees") -> None:
        self._bridge = bridge

    def _add(self, prim: MPM, name: str | None = None) -> MPM:
        return self._bridge._register(prim, name=name, alias=False)

    def grid(
        self, *, origin: Sequence[float], h: float, cells: Sequence[int],
        basis: str | None = None,
    ) -> Grid:
        """``mpm grid``: the background grid (one per model)."""
        prim = Grid(origin=tuple(origin), h=h, cells=tuple(cells), basis=basis)
        if any(isinstance(p, Grid) for p in self._bridge._primitives):
            raise MPMError("apeSees.mpm.grid: this model already has an mpm "
                           "grid (the fork's v1: one per Domain)")
        self._add(prim)
        return prim

    def body(
        self, pg: str, *, material: "NDMaterial | str", density: float,
        ppc: int, name: str | None = None,
    ) -> Body:
        """An MPM body: the elements of physical group ``pg``, lattice-seeded
        at ``ppc`` particles per cell per axis, with ``density`` and the
        nDMaterial ``material`` (a handle or its registered name). ``name``
        defaults to ``pg``."""
        mat = self._bridge._resolve(material, base=NDMaterial)
        body_name = pg if name is None else name
        prim = Body(name=body_name, pg=pg, material=mat, density=density,
                    ppc=ppc)
        try:
            self._bridge.fem.elements.select(pg=pg)
        except KeyError as exc:
            raise MPMError(f"apeSees.mpm.body: the model has no physical group "
                           f"{pg!r} ({exc})") from None
        for p in self._bridge._primitives:
            if isinstance(p, Body) and p.name == body_name:
                raise MPMError(f"apeSees.mpm.body: a body named {body_name!r} "
                               "is already declared")
        self._add(prim, name=body_name)
        return prim

    def fix(self, face: str, components: str) -> Fix:
        """``mpm fix``: zero grid-velocity ``components`` on ``face``."""
        prim = Fix(face=face, components=components)
        self._add(prim)
        return prim

    def wall(self, face: str, mu: float) -> Wall:
        """``mpm wall``: a stationary Coulomb wall on ``face``."""
        prim = Wall(face=face, mu=mu)
        self._add(prim)
        return prim

    def gravity(self, *g: float) -> Gravity:
        """``mpm gravity``: the body force per unit mass."""
        prim = Gravity(g=tuple(g))
        self._add(prim)
        return prim

    def timeSeries(self, series: "_TimeSeriesPrimitive | str") -> TimeSeries:
        """``mpm timeSeries``: the factor on gravity (a series or its name)."""
        ts = self._bridge._resolve(series, base=_TimeSeriesPrimitive)
        prim = TimeSeries(series=ts)
        self._add(prim)
        return prim

    def damping(self, alpha: float) -> Damping:
        """``mpm damping local``: local non-viscous damping."""
        prim = Damping(alpha=alpha)
        self._add(prim)
        return prim

    def flip(self, alpha: float) -> Flip:
        """``mpm flip``: the FLIP fraction."""
        prim = Flip(alpha=alpha)
        self._add(prim)
        return prim

    def massCutoff(self, c: float) -> MassCutoff:
        """``mpm massCutoff``: the active-node mass cutoff."""
        prim = MassCutoff(c=c)
        self._add(prim)
        return prim

    def safety(self, s: float) -> Safety:
        """``mpm safety``: the time-step safety factor."""
        prim = Safety(s=s)
        self._add(prim)
        return prim

    def units(
        self, length: str, force: str, time: str, *, system: str | None = None,
    ) -> Units:
        """``mpm units``: the unit symbols of the set and the results."""
        prim = Units(length=length, force=force, time=time, system=system)
        if any(isinstance(p, Units) for p in self._bridge._primitives):
            raise MPMError("apeSees.mpm.units: this model already declares "
                           "mpm units")
        self._add(prim)
        return prim

    def output(
        self, path: str, *, every: int | None = None, dt: float | None = None,
        fields: Sequence[str] | None = None, compress: int | None = None,
        flush: float | None = None,
    ) -> Output:
        """``mpm output``: the particle results file (``.ladruno``)."""
        prim = Output(path=path, every=every, dt=dt,
                      fields=None if fields is None else tuple(fields),
                      compress=compress, flush=flush)
        self._add(prim)
        return prim

    def analyze(self, steps: int, dt: float) -> Analyze:
        """``mpm analyze``: ``steps`` explicit steps of size ``dt``."""
        prim = Analyze(steps=steps, dt=dt)
        self._add(prim)
        return prim
