"""
Stage 1 — Reinforcement-cage Definitions (pre-mesh, geometry-level intent).

These dataclasses describe *what* reinforcement a user wants — bars,
stirrups, hooks, a whole cage — at the geometry level, as **pure,
frozen, serialisable data**. They carry no node tags, no OpenSees
handles, and no live gmsh entities (Option-B layering, mirroring
:class:`~apeGmsh._kernel.defs.constraints.ReinforceDef`). The L2
``g.rebar`` composite (:class:`~apeGmsh.core.RebarComposite`) generates
geometry from them and routes coupling; the L1 spec itself stays inert
and round-trips through ``to_dict`` / ``from_dict``.

Lengths are **unitless** here — a bare ``float`` is in model units, and a
``"<k>db"`` string (e.g. ``"12db"``) is resolved against the bar
diameter by a :class:`DetailingStandard` at *bind* time, never at
construction. This keeps the spec serialisable and standard-agnostic.

See ADR 0066 (reinforcement-cage authoring) for the design rationale and
the L1/L2/L3 split. The detailing standards + bar catalogue live in
:mod:`apeGmsh.rebar.detailing`; geometry generation + coupling live in
:mod:`apeGmsh.core.RebarComposite`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

Vec3 = tuple[float, float, float]

# Current on-disk schema version for Cage.to_dict (bump on a breaking
# change; from_dict accepts this version and the un-tagged legacy form).
CAGE_SCHEMA = "apeGmsh.rebar.cage"
CAGE_SCHEMA_VERSION = 1

# A "<k>db" length token: a *positive* number immediately followed by
# "db", e.g. "12db", "4db", "2.5db". Resolved to k * bar_diameter at bind
# time. _parse_db_token is the single source of truth shared with the
# detailing resolver so acceptance and resolution can never drift.
_DB_TOKEN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*db\s*$", re.IGNORECASE)

# Sentinel for "store the bend/corner radius as metadata, emit a sharp
# polyline corner" — the default fidelity (ADR 0066 §5). Path uses
# corner_radius == METADATA and Hook uses true_arc is False to express
# the same "sharp polyline, radius is metadata" intent.
METADATA = "metadata"

# Bend-plane selector tokens accepted by Hook.turn (besides an explicit
# 3-vector). Resolution happens at bind time in the L2 composite.
_TURN_TOKENS = frozenset(
    {"centroid", "in", "out", "up", "down",
     "+x", "-x", "+y", "-y", "+z", "-z"}
)


def _parse_db_token(value: Any) -> float | None:
    """Return ``k > 0`` for a ``"<k>db"`` token, else ``None``.

    The one place the token grammar lives; both :func:`_is_db_token`
    (L1 acceptance) and the detailing ``resolve_length`` (bind-time
    resolution) go through it, so a ``"0db"`` can never be accepted at
    construction and then explode at resolve.
    """
    if not isinstance(value, str):
        return None
    m = _DB_TOKEN.match(value)
    if m is None:
        return None
    k = float(m.group(1))
    return k if k > 0.0 else None


def _is_db_token(value: Any) -> bool:
    """True iff *value* is a valid (positive) ``"<k>db"`` length string."""
    return _parse_db_token(value) is not None


def _validate_length(value: float | str, *, field_name: str, owner: str,
                     allow_metadata: bool = False) -> None:
    """Fail-loud check for a length field: a positive finite float, a
    positive ``"<k>db"`` token, or (optionally) the ``"metadata"``
    sentinel."""
    if allow_metadata and value == METADATA:
        return
    if isinstance(value, str):
        if _DB_TOKEN.match(value):
            if _parse_db_token(value) is None:
                raise ValueError(
                    f'{owner}: {field_name}={value!r} must use a positive '
                    f'multiple, e.g. "12db" (a zero multiple is not allowed).'
                )
            return
        extra = ' or "metadata"' if allow_metadata else ""
        raise ValueError(
            f'{owner}: {field_name}={value!r} is not a valid length. '
            f'Use a positive number (model units) or a "<k>db" string '
            f'such as "12db"{extra}.'
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{owner}: {field_name} must be a number or a \"<k>db\" "
            f"string, got {type(value).__name__}."
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"{owner}: {field_name} must be finite, got {value}."
        )
    if value <= 0.0:
        raise ValueError(
            f"{owner}: {field_name} must be > 0, got {value}."
        )


def _validate_db(value: float | str, *, owner: str) -> None:
    """A bar size: a positive finite float (model-unit diameter) or a
    catalogue designation string (e.g. ``"#8"``, ``"20mm"``). The
    designation is resolved by a :class:`DetailingStandard` at bind time,
    so here we only reject the empty/non-finite/non-positive cases."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{owner}: db designation is empty.")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{owner}: db must be a positive number or a designation "
            f"string (e.g. \"#8\"), got {type(value).__name__}."
        )
    if not math.isfinite(float(value)):
        raise ValueError(f"{owner}: db must be finite, got {value}.")
    if value <= 0.0:
        raise ValueError(f"{owner}: db must be > 0, got {value}.")


def _bundle_count_for_pattern(pattern: str) -> int | None:
    """The bar count an explicit cluster pattern implies, or ``None`` for
    ``"auto"`` (any count)."""
    return {"line": 2, "triangle": 3, "square": 4}.get(pattern)


def _validate_bundle(bundle: Any, pattern: Any, db: float | str, *,
                     owner: str) -> None:
    """Fail-loud check for a bundle count + pattern (ACI 318-19 §25.6.1.1):
    1–4 bars, the pattern is known and (if explicit) matches the count, and
    a #14/#18 bar is limited to 2 per bundle."""
    if not isinstance(bundle, int) or isinstance(bundle, bool) or bundle < 1:
        raise ValueError(f"{owner}: bundle must be an int ≥ 1, got {bundle!r}.")
    if bundle > 4:
        raise ValueError(
            f"{owner}: bundle must be ≤ 4 bars (ACI 318-19 §25.6.1.1), "
            f"got {bundle}.")
    if not isinstance(pattern, str) or pattern not in _BUNDLE_PATTERNS:
        raise ValueError(
            f"{owner}: bundle_pattern must be one of {sorted(_BUNDLE_PATTERNS)}, "
            f"got {pattern!r}.")
    want = _bundle_count_for_pattern(pattern)
    if want is not None and want != bundle:
        raise ValueError(
            f"{owner}: bundle_pattern={pattern!r} implies {want} bars but "
            f"bundle={bundle}; use bundle_pattern='auto' or match the count.")
    if (bundle > 2 and isinstance(db, str)
            and db.replace(" ", "") in _BUNDLE_MAX2):
        raise ValueError(
            f"{owner}: a {db} bar is limited to 2 per bundle "
            f"(ACI 318-19 §25.6.1.1), got bundle={bundle}.")


def _norm_vec3(v: Any) -> Vec3 | None:
    """Return a float-normalised (x, y, z) tuple, or None if *v* is not a
    well-formed 3-vector."""
    if (isinstance(v, tuple) and len(v) == 3
            and all(isinstance(c, (int, float)) and not isinstance(c, bool)
                    for c in v)):
        return tuple(float(c) for c in v)
    return None


# ── Hook ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Hook:
    """A bend detail at a **free end** of a bar/stirrup path (ADR 0066 §3).

    The incoming tangent comes from the parent path — it is *not* stored
    here. The bend *plane* is fixed at bind time from :attr:`turn`
    (default ``"centroid"`` ⇒ toward the host/cage centroid).

    ``tail=None`` and ``bend_radius=None`` defer to the
    :class:`DetailingStandard` (``hook_tail`` / ``min_bend_diameter``) at
    bind time — the ACI multiples live in *one* place, the standard, not
    duplicated on every hook. An explicit number or ``"<k>db"`` token
    overrides. ``true_arc=False`` (default) emits a sharp polyline corner
    with the radius carried as metadata; ``true_arc=True`` builds a real
    fillet (L2 concern).
    """

    angle: float                                  # 90 | 135 | 180 (deg)
    tail: float | str | None = None               # None ⇒ from standard
    bend_radius: float | str | None = None        # None ⇒ from standard
    turn: str | Vec3 = "centroid"                 # bend-plane selector
    true_arc: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.angle, (int, float)) or isinstance(self.angle, bool):
            raise ValueError(f"Hook: angle must be a number, got {self.angle!r}.")
        if not math.isfinite(float(self.angle)):
            raise ValueError(f"Hook: angle must be finite, got {self.angle}.")
        if not (0.0 < float(self.angle) <= 180.0):
            raise ValueError(
                f"Hook: angle must be in (0, 180] degrees, got {self.angle}."
            )
        if self.tail is not None:
            _validate_length(self.tail, field_name="tail", owner="Hook")
        if self.bend_radius is not None:
            _validate_length(self.bend_radius, field_name="bend_radius",
                             owner="Hook")
        self._validate_turn()

    def _validate_turn(self) -> None:
        turn = self.turn
        if isinstance(turn, str):
            if turn.lower() not in _TURN_TOKENS:
                raise ValueError(
                    f"Hook: turn={turn!r} is not a known token "
                    f"({sorted(_TURN_TOKENS)}) or a 3-vector."
                )
            return
        vec = _norm_vec3(turn)
        if vec is None:
            raise ValueError(
                f"Hook: turn must be a token string or an (x, y, z) vector, "
                f"got {turn!r}."
            )
        if all(c == 0.0 for c in vec):
            raise ValueError("Hook: turn vector must be non-zero.")
        object.__setattr__(self, "turn", vec)

    # Code-aware factories. They set only the ACI bend ANGLE and leave
    # tail/bend_radius=None so the DetailingStandard fills them at bind
    # time (the nominal 12db/6db/etc. live in ONE place — the standard).
    @classmethod
    def standard_90(cls, *, turn: str | Vec3 = "centroid",
                    true_arc: bool = False, name: str | None = None) -> "Hook":
        return cls(angle=90.0, turn=turn, true_arc=true_arc, name=name)

    @classmethod
    def standard_135(cls, *, turn: str | Vec3 = "centroid",
                     true_arc: bool = False, name: str | None = None) -> "Hook":
        return cls(angle=135.0, turn=turn, true_arc=true_arc, name=name)

    @classmethod
    def standard_180(cls, *, turn: str | Vec3 = "centroid",
                     true_arc: bool = False, name: str | None = None) -> "Hook":
        return cls(angle=180.0, turn=turn, true_arc=true_arc, name=name)

    # Ergonomic alias: a 135° hook. "Seismic-ness" (the 3 in tail floor)
    # is a property of the ACI318_seismic standard + the seismic_hoop
    # kind at resolve time, NOT of the hook itself.
    @classmethod
    def seismic_135(cls, *, turn: str | Vec3 = "centroid",
                    true_arc: bool = False, name: str | None = None) -> "Hook":
        return cls.standard_135(turn=turn, true_arc=true_arc, name=name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle": float(self.angle),
            "tail": self.tail,
            "bend_radius": self.bend_radius,
            "turn": list(self.turn) if isinstance(self.turn, tuple) else self.turn,
            "true_arc": bool(self.true_arc),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Hook":
        turn = d.get("turn", "centroid")
        if isinstance(turn, list):
            turn = tuple(turn)
        return cls(
            angle=d["angle"], tail=d.get("tail"), bend_radius=d.get("bend_radius"),
            turn=turn, true_arc=d.get("true_arc", False), name=d.get("name"),
        )


# ── Path ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Path:
    """An ordered open polyline of 3D control points, with a bend radius
    applied at every interior vertex (ADR 0066 §3).

    ``corner_radius="metadata"`` (default) emits sharp polyline corners
    and stores the radius as metadata; a number or ``"<k>db"`` token sets
    a true fillet radius consumed by the L2 builder under ``true_arc``.
    Consecutive points must differ (no zero-length segments); a closed
    loop's first==last (non-consecutive) is allowed.
    """

    points: tuple[Vec3, ...]
    corner_radius: float | str = METADATA

    def __post_init__(self) -> None:
        pts = tuple(tuple(float(c) for c in p) for p in self.points)
        for p in pts:
            if len(p) != 3:
                raise ValueError(
                    f"Path: every point must be (x, y, z), got {p!r}."
                )
            if not all(math.isfinite(c) for c in p):
                raise ValueError(f"Path: point has non-finite coordinate: {p!r}.")
        if len(pts) < 2:
            raise ValueError(
                f"Path: need at least 2 points, got {len(pts)}."
            )
        for a, b in zip(pts, pts[1:]):
            if a == b:
                raise ValueError(
                    f"Path: consecutive points coincide (zero-length "
                    f"segment) at {a!r}."
                )
        object.__setattr__(self, "points", pts)
        _validate_length(self.corner_radius, field_name="corner_radius",
                         owner="Path", allow_metadata=True)

    @property
    def is_metadata_bend(self) -> bool:
        return self.corner_radius == METADATA

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [list(p) for p in self.points],
            "corner_radius": self.corner_radius,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Path":
        return cls(
            points=tuple(tuple(p) for p in d["points"]),
            corner_radius=d.get("corner_radius", METADATA),
        )


# ── Bar ──────────────────────────────────────────────────────────────

_ELEMENT_KINDS = frozenset({"truss", "beam"})


@dataclass(frozen=True)
class Bar:
    """A single reinforcing bar: a :class:`Path` centerline + diameter +
    material (by name) + optional end hooks (ADR 0066 §3, §7).

    ``element="truss"`` (default, realised as ``CorotTruss``) is
    axial-only; ``element="beam"`` is opt-in dowel action and is gated on
    the ADR-0010 Phase-4 orientation fan-out for curved bars (enforced at
    bridge time, not here).
    """

    path: Path
    db: float | str
    material: str
    role: str = "longitudinal"
    element: str = "truss"
    start_hook: Hook | None = None
    end_hook: Hook | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ValueError(
                f"Bar: path must be a Path, got {type(self.path).__name__}."
            )
        _validate_db(self.db, owner="Bar")
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("Bar: material must be a non-empty name string.")
        if self.element not in _ELEMENT_KINDS:
            raise ValueError(
                f"Bar: element must be one of {sorted(_ELEMENT_KINDS)}, "
                f"got {self.element!r}."
            )
        for h, slot in ((self.start_hook, "start_hook"),
                        (self.end_hook, "end_hook")):
            if h is not None and not isinstance(h, Hook):
                raise ValueError(
                    f"Bar: {slot} must be a Hook or None, "
                    f"got {type(h).__name__}."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "db": self.db,
            "material": self.material,
            "role": self.role,
            "element": self.element,
            "start_hook": self.start_hook.to_dict() if self.start_hook else None,
            "end_hook": self.end_hook.to_dict() if self.end_hook else None,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Bar":
        sh, eh = d.get("start_hook"), d.get("end_hook")
        return cls(
            path=Path.from_dict(d["path"]),
            db=d["db"], material=d["material"],
            role=d.get("role", "longitudinal"),
            element=d.get("element", "truss"),
            start_hook=Hook.from_dict(sh) if sh else None,
            end_hook=Hook.from_dict(eh) if eh else None,
            name=d.get("name"),
        )


# ── Stirrup ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stirrup:
    """A tie / hoop: an **open** corner :class:`Path` that returns near
    its origin with a hooked closure (ADR 0066 §3). A real tie is *not* a
    closed loop — the two hooked tails overlap at one corner; the L2
    builder realises the closure-overlap geometry.
    """

    path: Path
    db: float | str
    material: str
    role: str = "tie"
    closure_hook: Hook = field(default_factory=Hook.seismic_135)
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ValueError(
                f"Stirrup: path must be a Path, got {type(self.path).__name__}."
            )
        _validate_db(self.db, owner="Stirrup")
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("Stirrup: material must be a non-empty name string.")
        if not isinstance(self.closure_hook, Hook):
            raise ValueError(
                f"Stirrup: closure_hook must be a Hook, "
                f"got {type(self.closure_hook).__name__}."
            )

    @classmethod
    def rect(cls, bx: float, by: float, cover: float, *, db: float | str,
             material: str, z: float = 0.0, plane: str = "xy",
             origin: tuple[float, float] = (0.0, 0.0),
             db_value: float | None = None,
             closure_hook: Hook | None = None, role: str = "tie",
             name: str | None = None) -> "Stirrup":
        """A rectangular tie. ``bx``/``by`` are the two in-plane section
        dimensions; ``cover`` is clear cover to the *outside* of the tie bar
        (centerline inset by ``cover + db/2``). ``plane`` places the ring:
        ``"xy"`` (column tie, ring in x-y at height ``z``), ``"yz"`` (beam
        stirrup, ring in y-z at station ``z`` along x), or ``"xz"``.
        ``origin`` offsets the in-plane corner. ``db_value`` (model-unit
        diameter) sizes the inset when ``db`` is a designation string.
        """
        if plane not in ("xy", "yz", "xz"):
            raise ValueError(
                f"Stirrup.rect: plane must be 'xy'|'yz'|'xz', got {plane!r}.")
        d = db_value if db_value is not None else (
            db if isinstance(db, (int, float)) and not isinstance(db, bool) else None)
        if d is None:
            raise ValueError(
                "Stirrup.rect: pass db_value=<diameter> when db is a "
                "designation string, so the cover→centerline inset is known."
            )
        for v, nm in ((bx, "bx"), (by, "by"), (cover, "cover"), (z, "z"),
                      (d, "db")):
            if not isinstance(v, (int, float)) or isinstance(v, bool) \
                    or not math.isfinite(float(v)):
                raise ValueError(f"Stirrup.rect: {nm} must be a finite number, got {v!r}.")
        for v, nm in ((bx, "bx"), (by, "by"), (d, "db")):
            if v <= 0.0:
                raise ValueError(f"Stirrup.rect: {nm} must be > 0, got {v}.")
        if cover < 0.0:
            raise ValueError(f"Stirrup.rect: cover must be >= 0, got {cover}.")
        inset = cover + d / 2.0
        if 2.0 * inset >= min(bx, by):
            raise ValueError(
                f"Stirrup.rect: cover+db/2={inset} too large for section "
                f"{bx}x{by} (degenerate tie polygon)."
            )
        ou, ov = origin
        u0, v0 = ou + inset, ov + inset
        u1, v1 = ou + bx - inset, ov + by - inset

        def _xyz(u: float, v: float) -> Vec3:
            if plane == "xy":
                return (u, v, z)
            if plane == "yz":
                return (z, u, v)
            return (u, z, v)                     # xz

        # open corner polyline that returns to the start corner (the
        # closure overlap + seam stagger is an L2 geometry concern)
        corners = [_xyz(u0, v0), _xyz(u1, v0), _xyz(u1, v1),
                   _xyz(u0, v1), _xyz(u0, v0)]
        return cls(
            path=Path(points=tuple(corners)),
            db=db, material=material, role=role,
            closure_hook=closure_hook or Hook.seismic_135(), name=name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "db": self.db,
            "material": self.material,
            "role": self.role,
            "closure_hook": self.closure_hook.to_dict(),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Stirrup":
        ch = d.get("closure_hook")
        return cls(
            path=Path.from_dict(d["path"]),
            db=d["db"], material=d["material"], role=d.get("role", "tie"),
            closure_hook=Hook.from_dict(ch) if ch else Hook.seismic_135(),
            name=d.get("name"),
        )


# ── Cage ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Cage:
    """A reinforcement cage: bars + stirrups, the serialisable source of
    truth that feeds both binding surfaces (inline + composed-Part).

    The optional ``standard`` is a :class:`DetailingStandard` reference;
    it is *not* serialised (the L2 composite re-binds it at ``place``
    time), so ``to_dict`` carries only geometry+intent, tagged with a
    schema version so the composed-Part / H5 path (ADR §6.3) can evolve
    it forward-compatibly.
    """

    bars: tuple[Bar, ...] = ()
    stirrups: tuple[Stirrup, ...] = ()
    standard: Any = None              # DetailingStandard | None (not serialised)

    def __post_init__(self) -> None:
        bars = tuple(self.bars)
        stirrups = tuple(self.stirrups)
        for b in bars:
            if not isinstance(b, Bar):
                raise ValueError(
                    f"Cage: bars must contain Bar instances, "
                    f"got {type(b).__name__}."
                )
        for s in stirrups:
            if not isinstance(s, Stirrup):
                raise ValueError(
                    f"Cage: stirrups must contain Stirrup instances, "
                    f"got {type(s).__name__}."
                )
        if not bars and not stirrups:
            raise ValueError("Cage: empty — needs at least one bar or stirrup.")
        object.__setattr__(self, "bars", bars)
        object.__setattr__(self, "stirrups", stirrups)

    def to_dict(self) -> dict[str, Any]:
        """Geometry+intent only (no standard, no OpenSees handles),
        tagged with the schema name+version."""
        return {
            "__schema__": CAGE_SCHEMA,
            "version": CAGE_SCHEMA_VERSION,
            "bars": [b.to_dict() for b in self.bars],
            "stirrups": [s.to_dict() for s in self.stirrups],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Cage":
        ver = d.get("version", CAGE_SCHEMA_VERSION)
        if ver != CAGE_SCHEMA_VERSION:
            raise ValueError(
                f"Cage.from_dict: unsupported schema version {ver!r} "
                f"(this build reads version {CAGE_SCHEMA_VERSION})."
            )
        return cls(
            bars=tuple(Bar.from_dict(b) for b in d.get("bars", [])),
            stirrups=tuple(Stirrup.from_dict(s) for s in d.get("stirrups", [])),
        )


# ── standardized-member layout inputs + fluent builder (pure data) ───

# Bundled-bar layouts (ACI 318-19 §25.6). A bundle is 2–4 parallel bars
# in contact acting as a unit; the geometry layer realises it as that many
# individual offset bar lines (each a distinct truss/embedded member). The
# token names a standard cluster shape; "auto" picks one from the count.
_BUNDLE_PATTERNS = frozenset({"auto", "line", "triangle", "square"})
# Designations that ACI §25.6.1.1 caps at 2 bars per bundle (#14, #18).
_BUNDLE_MAX2 = frozenset({"#14", "#18"})


@dataclass(frozen=True)
class BarLayout:
    """Longitudinal-bar layout for a standardized member. ``n_x``/``n_y``
    are bars along each face (corners shared); for a beam line use ``n_x``
    as the count (``n_y`` is ignored by ``beam()``).

    ``bundle`` (default 1) replaces each single bar position with a contact
    bundle of that many parallel bars (ACI 318-19 §25.6 — 2, 3 or 4 bars).
    ``bundle_pattern`` chooses the cluster shape: ``"auto"`` (default —
    ``line`` for 2, ``triangle`` for 3, ``square`` for 4), or an explicit
    ``"line"`` / ``"triangle"`` / ``"square"`` matching the count. The outer
    bars sit on the nominal cover line and the cluster stacks inward toward
    the section interior (so no bar is shallower than the single-bar
    position); for strict corner cover, inset the layout for the bundle's
    equivalent diameter ``√n·d_b``.
    """
    n_x: int
    n_y: int = 2
    db: float | str = "#8"
    material: str = "rebar"
    bundle: int = 1
    bundle_pattern: str = "auto"

    def __post_init__(self) -> None:
        for v, nm in ((self.n_x, "n_x"), (self.n_y, "n_y")):
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"BarLayout: {nm} must be an int ≥ 1, got {v!r}.")
        _validate_db(self.db, owner="BarLayout")
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("BarLayout: material must be a non-empty name.")
        _validate_bundle(self.bundle, self.bundle_pattern, self.db,
                         owner="BarLayout")


@dataclass(frozen=True)
class TieLayout:
    """Transverse-reinforcement layout. ``spacing`` is the regular tie
    spacing; ``hinge_spacing`` (denser) applies within ``hinge_length`` of
    each member end (ACI seismic confinement zones). Provide both or
    neither hinge field."""
    db: float | str
    spacing: float
    material: str = "rebar"
    hinge_spacing: float | None = None
    hinge_length: float | None = None
    db_value: float | None = None
    hook: Hook | None = None

    def __post_init__(self) -> None:
        _validate_db(self.db, owner="TieLayout")
        if (not isinstance(self.spacing, (int, float)) or isinstance(self.spacing, bool)
                or not math.isfinite(float(self.spacing)) or self.spacing <= 0):
            raise ValueError(f"TieLayout: spacing must be > 0, got {self.spacing!r}.")
        if (self.hinge_spacing is None) != (self.hinge_length is None):
            raise ValueError(
                "TieLayout: hinge_spacing and hinge_length must be set "
                "together (both or neither).")
        for v, nm in ((self.hinge_spacing, "hinge_spacing"),
                      (self.hinge_length, "hinge_length")):
            if v is not None and (not isinstance(v, (int, float))
                                  or isinstance(v, bool) or v <= 0):
                raise ValueError(f"TieLayout: {nm} must be > 0, got {v!r}.")
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("TieLayout: material must be a non-empty name.")


class BarBuilder:
    """L3 fluent sugar — chains into an L1 :class:`Bar`. Holds no gmsh
    state; an abandoned builder emits nothing."""

    def __init__(self, *, db, material, role: str = "longitudinal",
                 element: str = "truss") -> None:
        self._db, self._material = db, material
        self._role, self._element = role, element
        self._points: tuple | None = None
        self._start: Hook | None = None
        self._end: Hook | None = None
        self._corner = METADATA
        self._name: str | None = None

    def through(self, points) -> "BarBuilder":
        self._points = tuple(points)
        return self

    def hook_start(self, hook: Hook) -> "BarBuilder":
        self._start = hook
        return self

    def hook_end(self, hook: Hook) -> "BarBuilder":
        self._end = hook
        return self

    def corner_radius(self, radius) -> "BarBuilder":
        self._corner = radius
        return self

    def role(self, role: str) -> "BarBuilder":
        self._role = role
        return self

    def element(self, element: str) -> "BarBuilder":
        self._element = element
        return self

    def build(self) -> "Bar":
        if self._points is None:
            raise ValueError("BarBuilder: call .through(points) before build().")
        return Bar(path=Path(self._points, corner_radius=self._corner),
                   db=self._db, material=self._material, role=self._role,
                   element=self._element, start_hook=self._start,
                   end_hook=self._end, name=self._name)

    def as_(self, name: str) -> "Bar":
        """Terminal: name the bar and return the built :class:`Bar`."""
        self._name = name
        return self.build()


__all__ = [
    "Vec3", "METADATA", "CAGE_SCHEMA", "CAGE_SCHEMA_VERSION",
    "Hook", "Path", "Bar", "Stirrup", "Cage",
    "BarLayout", "TieLayout", "BarBuilder",
]
